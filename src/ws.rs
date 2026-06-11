use std::sync::Arc;

use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use crate::hub::{BroadcastMsg, Hub};
use crate::joypad::{self, Joypad};

/// Start the WebSocket server on the given address.
pub async fn serve(
    hub: Arc<Hub>,
    joypad: Arc<Joypad>,
    addr: &str,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    log::info!("ws: listening on ws://{addr}");

    loop {
        let (stream, peer) = listener.accept().await?;
        let hub = hub.clone();
        let joypad = joypad.clone();
        log::info!("ws: client connected from {peer}");
        tokio::spawn(async move {
            if let Err(e) = handle_client(hub, joypad, stream).await {
                log::warn!("ws: client {peer} error: {e}");
            }
            log::info!("ws: client disconnected {peer}");
        });
    }
}

async fn handle_client(
    hub: Arc<Hub>,
    joypad: Arc<Joypad>,
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

    // Read loop: handle incoming messages
    while let Some(msg) = ws_read.next().await {
        let msg = msg?;
        match msg {
            Message::Text(text) => {
                handle_text_message(&joypad, &text);
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Client disconnected — release all buttons
    joypad.release_all();
    write_task.abort();
    Ok(())
}

fn handle_text_message(joypad: &Joypad, text: &str) {
    // Quick parse — skip full deserialization overhead for simple key events
    if text.contains("\"key\"") {
        // Keyboard event: {"key":"w","pressed":true}
        let is_press = text.contains("\"pressed\":true") || text.contains("\"pressed\": true");
        // Extract key name
        if let Some(key_start) = text.find("\"key\":\"") {
            let val_start = key_start + 7;
            if let Some(val_end) = text[val_start..].find('"') {
                let key = &text[val_start..val_start + val_end];
                if let Some(bits) = joypad::key_to_bits(key) {
                    if is_press {
                        joypad.press(bits);
                    } else {
                        joypad.release(bits);
                    }
                }
            }
        }
    } else if text.contains("\"takeover\"") {
        // Takeover toggle: {"takeover":true} or {"takeover":false}
        let takeover = text.contains("\"takeover\":true") || text.contains("\"takeover\": true");
        log::info!("takeover: {}", takeover);
        if takeover {
            joypad.release_all();
        }
    } else if text.contains("\"save_state\"") {
        // Save state: {"save_state":"path"}
        if let Some(path) = extract_string_field(text, "\"save_state\"") {
            log::info!("save state requested: {path}");
        }
    } else if text.contains("\"load_state\"") {
        // Load state: {"load_state":"path"}
        if let Some(path) = extract_string_field(text, "\"load_state\"") {
            log::info!("load state requested: {path}");
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
