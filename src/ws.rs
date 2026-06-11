use std::sync::Arc;

use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use crate::hub::{BroadcastMsg, Hub};
use crate::joypad::{self, Joypad};
use crate::EmulatorCmd;

pub async fn serve(
    hub: Arc<Hub>,
    joypad: Arc<Joypad>,
    cmd_tx: mpsc::UnboundedSender<EmulatorCmd>,
    addr: &str,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    log::info!("ws: listening on ws://{addr}");

    loop {
        let (stream, peer) = listener.accept().await?;
        let hub = hub.clone();
        let joypad = joypad.clone();
        let cmd_tx = cmd_tx.clone();
        log::info!("ws: client connected from {peer}");
        tokio::spawn(async move {
            if let Err(e) = handle_client(hub, joypad, cmd_tx, stream).await {
                log::warn!("ws: client {peer} error: {e}");
            }
            log::info!("ws: client disconnected {peer}");
        });
    }
}

async fn handle_client(
    hub: Arc<Hub>,
    joypad: Arc<Joypad>,
    cmd_tx: mpsc::UnboundedSender<EmulatorCmd>,
    stream: tokio::net::TcpStream,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (ws_write, mut ws_read) = ws_stream.split();

    let mut rx = hub.subscribe();
    let write_task = tokio::spawn(async move {
        use futures_util::SinkExt;

        let mut ws_write = ws_write;
        while let Ok(msg) = rx.recv().await {
            let ws_msg = match &msg {
                BroadcastMsg::Frame(data) => Message::Binary({
                    let mut buf = vec![0x00];
                    buf.extend_from_slice(data);
                    buf
                }),
                BroadcastMsg::Text(text) => Message::Text(text.clone().into()),
            };
            if ws_write.send(ws_msg).await.is_err() {
                break;
            }
        }
    });

    while let Some(msg) = ws_read.next().await {
        let msg = msg?;
        match msg {
            Message::Text(text) => {
                handle_text_message(&joypad, &cmd_tx, &text);
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    joypad.release_all();
    write_task.abort();
    Ok(())
}

fn handle_text_message(joypad: &Joypad, cmd_tx: &mpsc::UnboundedSender<EmulatorCmd>, text: &str) {
    if text.contains("\"key\"") {
        let is_press = text.contains("\"pressed\":true") || text.contains("\"pressed\": true");
        if let Some(key_start) = text.find("\"key\":\"") {
            let val_start = key_start + 7;
            if let Some(val_end) = text[val_start..].find('"') {
                let key = &text[val_start..val_start + val_end];
                if let Some(bits) = joypad::key_to_bits(key) {
                    if is_press { joypad.press(bits); } else { joypad.release(bits); }
                }
            }
        }
    } else if text.contains("\"takeover\"") {
        let takeover = text.contains("\"takeover\":true") || text.contains("\"takeover\": true");
        if takeover { joypad.release_all(); }
    } else if text.contains("\"save_state\"") {
        if let Some(path) = extract_string_field(text, "\"save_state\"") {
            log::info!("save state requested: {path}");
            let _ = cmd_tx.send(EmulatorCmd::SaveState(path));
        }
    } else if text.contains("\"load_state\"") {
        if let Some(path) = extract_string_field(text, "\"load_state\"") {
            log::info!("load state requested: {path}");
            let _ = cmd_tx.send(EmulatorCmd::LoadState(path));
        }
    }
}

fn extract_string_field(text: &str, _field: &str) -> Option<String> {
    let colon = text.find(':')?;
    let after_colon = &text[colon + 1..];
    let val_start = after_colon.find('"')?;
    let value_start = val_start + 1;
    let val_end = after_colon[value_start..].find('"')?;
    Some(after_colon[value_start..value_start + val_end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joypad::JOYPAD_A;
    use tokio::sync::mpsc;

    fn test_joypad() -> Arc<Joypad> {
        Arc::new(Joypad::new())
    }

    fn dummy_cmd_tx() -> mpsc::UnboundedSender<EmulatorCmd> {
        let (tx, _rx) = mpsc::unbounded_channel();
        tx
    }

    #[test]
    fn test_handle_key_press() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        handle_text_message(&j, &tx, r#"{"key":"k","pressed":true}"#);
        assert_eq!(j.state(), JOYPAD_A);
    }

    #[test]
    fn test_handle_key_release() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        handle_text_message(&j, &tx, r#"{"key":"k","pressed":true}"#);
        handle_text_message(&j, &tx, r#"{"key":"k","pressed":false}"#);
        assert!(j.is_idle());
    }

    #[test]
    fn test_handle_invalid_key() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        handle_text_message(&j, &tx, r#"{"key":"z","pressed":true}"#);
        assert!(j.is_idle());
    }

    #[test]
    fn test_handle_takeover_clears_joypad() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        j.press(JOYPAD_A);
        handle_text_message(&j, &tx, r#"{"takeover":true}"#);
        assert!(j.is_idle());
    }

    #[test]
    fn test_extract_string_field() {
        let result = extract_string_field(r#"{"save_state":"path/to/save.sav"}"#, "save_state");
        assert_eq!(result.as_deref(), Some("path/to/save.sav"));
    }

    #[test]
    fn test_extract_string_field_empty() {
        let result = extract_string_field(r#"{}"#, "missing");
        assert!(result.is_none());
    }

    #[test]
    fn test_handle_save_state_sends_command() {
        let j = test_joypad();
        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_text_message(&j, &tx, r#"{"save_state":"test.sav"}"#);
        let cmd = rx.try_recv().unwrap();
        match cmd {
            EmulatorCmd::SaveState(p) => assert_eq!(p, "test.sav"),
            _ => panic!("expected SaveState"),
        }
    }

    #[test]
    fn test_handle_load_state_sends_command() {
        let j = test_joypad();
        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_text_message(&j, &tx, r#"{"load_state":"test.sav"}"#);
        let cmd = rx.try_recv().unwrap();
        match cmd {
            EmulatorCmd::LoadState(p) => assert_eq!(p, "test.sav"),
            _ => panic!("expected LoadState"),
        }
    }

    #[test]
    fn test_handle_text_trash() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        handle_text_message(&j, &tx, "not json at all {{{{{{");
        assert!(j.is_idle());
    }

    #[test]
    fn test_handle_text_empty() {
        let j = test_joypad();
        let tx = dummy_cmd_tx();
        handle_text_message(&j, &tx, "");
        assert!(j.is_idle());
    }
}
