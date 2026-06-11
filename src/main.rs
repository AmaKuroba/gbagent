use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::sync::watch;

use boytacean::pad::PadKey;

mod emulator;
mod http;
mod hub;
mod joypad;
mod ws;
mod config;
mod reward;
mod agent;

#[derive(Parser)]
#[command(name = "rust-gbagent", about = "Game Boy RL agent")]
struct Cli {
    /// Path to Game Boy ROM file
    #[arg(short = 'r', long = "rom")]
    rom: String,

    /// Start training immediately
    #[arg(long = "train")]
    train: bool,

    /// Path to game config YAML (RAM scanner addresses)
    #[arg(long = "config")]
    config: Option<String>,

    /// Resume training from checkpoint
    #[arg(long = "resume")]
    resume: Option<String>,

    /// Load pre-trained ViT encoder weights
    #[arg(long = "pretrained")]
    pretrained: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Core shared state
    let hub = hub::Hub::new(256).shared();
    let joypad = Arc::new(joypad::Joypad::new());
    let (train_tx, mut train_rx) = watch::channel(cli.train);

    // Read ROM and init emulator
    let rom_data = std::fs::read(&cli.rom)
        .map_err(|e| anyhow::anyhow!("failed to read ROM '{}': {}", cli.rom, e))?;
    let mut gb = emulator::Emulator::new(&rom_data)?;
    gb.boot();
    log::info!("emulator initialized: {} ({} bytes)", gb.description(), rom_data.len());

    // Shared state for HTTP server
    let http_state = Arc::new(http::HttpState {
        hub: hub.clone(),
        train_tx,
    });

    // Start HTTP server on :8765
    let server_http_state = http_state;
    tokio::spawn(async move {
        if let Err(e) = http::serve(server_http_state, "0.0.0.0:8765").await {
            log::error!("http server error: {e}");
        }
    });

    // Start WebSocket server on :8766
    let ws_hub = hub.clone();
    let ws_joypad = joypad.clone();
    tokio::spawn(async move {
        if let Err(e) = ws::serve(ws_hub, ws_joypad, "0.0.0.0:8766").await {
            log::error!("ws server error: {e}");
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    log::info!("server ready: http://localhost:8765, ws://localhost:8766");

    // Map joypad bits to boytacean PadKey variants
    fn joypad_bits_to_padkeys(bits: u32) -> Vec<PadKey> {
        let mut keys = Vec::with_capacity(4);
        if bits & joypad::JOYPAD_RIGHT != 0 { keys.push(PadKey::Right); }
        if bits & joypad::JOYPAD_LEFT != 0 { keys.push(PadKey::Left); }
        if bits & joypad::JOYPAD_UP != 0 { keys.push(PadKey::Up); }
        if bits & joypad::JOYPAD_DOWN != 0 { keys.push(PadKey::Down); }
        if bits & joypad::JOYPAD_A != 0 { keys.push(PadKey::A); }
        if bits & joypad::JOYPAD_B != 0 { keys.push(PadKey::B); }
        if bits & joypad::JOYPAD_SELECT != 0 { keys.push(PadKey::Select); }
        if bits & joypad::JOYPAD_START != 0 { keys.push(PadKey::Start); }
        keys
    }

    // Main loop: 60fps emulation + frame relay
    let frame_interval = std::time::Duration::from_micros(16_667);
    let mut _training = cli.train;
    let mut prev_joypad: u32 = 0;

    loop {
        let frame_start = std::time::Instant::now();

        // Sync joypad state to emulator
        let current = joypad.state();
        if current != prev_joypad {
            if current == 0 {
                gb.release_all();
            } else {
                gb.release_all();
                for key in joypad_bits_to_padkeys(current) {
                    gb.press(key);
                }
            }
            prev_joypad = current;
        }

        // Advance emulator by one frame
        gb.next_frame();

        // Encode screen to PNG and broadcast
        let rgba = gb.screen_rgba();
        if let Ok(png_data) = encode_png_rgba(&rgba, emulator::SCREEN_W, emulator::SCREEN_H) {
            hub.broadcast_frame(png_data);
        }

        // Check training state changes from HTTP
        if train_rx.has_changed().unwrap_or(false) {
            let getting_training = *train_rx.borrow();
            if getting_training != _training {
                _training = getting_training;
                log::info!("training: {}", if _training { "on" } else { "off" });
            }
        }

        // Maintain ~60fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            tokio::time::sleep(frame_interval - elapsed).await;
        }
    }
}

/// Encode RGBA pixel data to PNG bytes.
fn encode_png_rgba(rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    let mut buf = std::io::Cursor::new(Vec::new());
    let encoder = PngEncoder::new(&mut buf);
    encoder.write_image(rgba, width as u32, height as u32, image::ExtendedColorType::Rgba8)?;
    Ok(buf.into_inner())
}
