use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::sync::watch;

use boytacean::pad::PadKey;

mod agent;
mod config;
mod emulator;
mod http;
mod hub;
mod joypad;
mod recorder;
mod reward;
mod ws;

use agent::{PPOAgent, PPOConfig, AgentStats, Transition};
use reward::RewardSystem;

#[allow(unused_mut, dead_code)]

#[derive(Parser)]
#[command(name = "rust-gbagent", about = "Game Boy RL agent")]
struct Cli {
    #[arg(short = 'r', long = "rom")]
    rom: String,
    #[arg(long = "train")]
    train: bool,
    #[arg(long = "config")]
    config: Option<String>,
    #[arg(long = "resume")]
    resume: Option<String>,
    #[arg(long = "pretrained")]
    pretrained: Option<String>,
    #[arg(long = "record-dir", default_value = ".")]
    record_dir: String,
    #[arg(long = "frame-skip", default_value = "4")]
    frame_skip: usize,
    #[arg(long = "frame-stack", default_value = "4")]
    frame_stack: usize,
}

impl Cli {
    fn rollout_len(&self) -> usize { 128 }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Core shared state
    let hub = hub::Hub::new(256).shared();
    let joypad = Arc::new(joypad::Joypad::new());
    let (train_tx, mut train_rx) = watch::channel(cli.train);
    let rec = Arc::new(recorder::Recorder::new());

    // Read ROM
    let rom_data = std::fs::read(&cli.rom)
        .map_err(|e| anyhow::anyhow!("failed to read ROM '{}': {}", cli.rom, e))?;
    let mut gb = emulator::Emulator::new(&rom_data)?;
    gb.boot();
    log::info!("emulator: {}", gb.description());

    // Reward system
    let game_config = cli.config.as_ref().and_then(|p| config::load_game_config(p).ok());
    let mut reward_sys = RewardSystem::new(game_config);

    // PPO agent
    let device = candle_core::Device::Cpu;
    let ppo_cfg = PPOConfig {
        total_steps: 6_000_000,
        rollout_len: cli.rollout_len(),
        ..Default::default()
    };
    let mut agent = PPOAgent::new(ppo_cfg, &device)?;
    if let Some(ref path) = cli.resume {
        agent.load(path)?;
    }
    if let Some(ref path) = cli.pretrained {
        // Load pretrained encoder weights
        let _vars = std::collections::HashMap::<String, candle_core::Tensor>::new();
        // Placeholder: in production, load from safetensors
        log::info!("pretrained path: {path} (loading not yet implemented)");
    }

    // HTTP state
    let http_state = Arc::new(http::HttpState {
        hub: hub.clone(),
        train_tx,
        rec: rec.clone(),
    });

    // Start HTTP
    let s = http_state.clone();
    tokio::spawn(async move { if let Err(e) = http::serve(s, "0.0.0.0:8765").await { log::error!("http: {e}"); } });

    // Start WS
    let ws_hub = hub.clone();
    let ws_joypad = joypad.clone();
    tokio::spawn(async move { if let Err(e) = ws::serve(ws_hub, ws_joypad, "0.0.0.0:8766").await { log::error!("ws: {e}"); } });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    log::info!("ready: http://localhost:8765  ws://localhost:8766");

    // Training state
    let frame_interval = std::time::Duration::from_micros(16_667);
    let mut training = cli.train;
    let mut prev_joypad: u32 = 0;
    let mut frame_buffer: Vec<Vec<u8>> = Vec::with_capacity(cli.frame_stack);
    let mut episode_reward = 0.0;
    let mut episode_step = 0;
    let mut episode_num = 0;
    let mut steps_since_update = 0;
    let mut frame_count = 0u64;
    let mut last_metrics_time = std::time::Instant::now();
    let mut last_dpad: u8 = 0;
    let mut last_btn: u8 = 0;

    // Initialize frame stack
    for _ in 0..cli.frame_stack {
        gb.next_frame();
        frame_buffer.push(gb.screen_rgba().to_vec());
    }

    loop {
        let frame_start = std::time::Instant::now();

        // Sync joypad
        let current = joypad.state();
        if current != prev_joypad {
            if current == 0 { gb.release_all(); }
            else { gb.release_all(); for k in joypad_bits(current) { gb.press(k); } }
            prev_joypad = current;
        }
        let (dpad, btn) = joypad::decode_joypad(current);

        // Store last action for recorder
        if dpad != 0 || btn != 0 {
            last_dpad = dpad;
            last_btn = btn;
        }

        // Advance emulator
        gb.next_frame();
        frame_count += 1;

        // Update frame stack
        let current_frame = gb.screen_rgba().to_vec();
        frame_buffer.push(current_frame.clone());
        if frame_buffer.len() > cli.frame_stack {
            frame_buffer.remove(0);
        }

        // Training loop
        if training && current == 0 {
            // Build observation from frame stack
            let obs: Vec<u8> = frame_buffer.iter().flat_map(|f| f[..4*144*160/4].to_vec()).collect();
            // Actually: f[..4*144*160/4] takes only first 4 bytes per pixel? No. f is RGBA, we want grayscale.
            // For now, use full RGBA (4 channels per pixel, frame_stack copies = obs)
            let obs_full: Vec<u8> = frame_buffer.iter().flat_map(|f| f.iter().copied()).collect();

            // Agent acts
            if let Ok((da, ba, d_lp, b_lp, val)) = agent.act(&obs_full, true) {
                let action_dpad = da as u8;
                let action_btn = ba as u8;

                // Execute action with frame_skip
                let mut action_reward = 0.0;
                for _ in 0..cli.frame_skip {
                    // Apply action for one frame
                    gb.release_all();
                    for k in action_bits(action_dpad, action_btn) { gb.press(k); }
                    gb.next_frame();
                    // Screen novelty reward
                    let f = gb.screen_rgba().to_vec();
                    let step_reward = reward_sys.compute(&mut gb, &f);
                    action_reward += step_reward;
                }

                episode_reward += action_reward;
                episode_step += 1;
                steps_since_update += 1;

                // Store transition
                agent.buffer.push(Transition {
                    obs: obs_full,
                    dpad_action: da,
                    btn_action: ba,
                    dpad_log_prob: d_lp,
                    btn_log_prob: b_lp,
                    reward: action_reward,
                    done: false,
                    value: val,
                });

                // PPO update when buffer is full
                if agent.buffer.len() >= agent.config.rollout_len {
                    if let Ok(stats) = agent.update() {
                        let elapsed = last_metrics_time.elapsed();
                        let sps = steps_since_update as f64 / elapsed.as_secs_f64().max(0.001);
                        log_info_training(&hub, &mut last_metrics_time, frame_count,
                            &stats, episode_reward, episode_step, episode_num, sps,
                            &rec, &mut last_dpad, &mut last_btn, &current_frame);
                        // Broadcast metrics to dashboard
                        let metrics = serde_json::json!({
                            "reward_total": episode_reward,
                            "loss": stats.policy_loss,
                            "epsilon": 0.0,
                            "sps": sps,
                        });
                        hub.broadcast_text(metrics.to_string());
                        steps_since_update = 0;
                    }
                }

                // Episode termination (placeholder: check for game over later)
                if episode_step >= 3000 {
                    log::info!("episode {episode_num} done: reward={episode_reward:.2}, steps={episode_step}");
                    // Reset emulator
                    gb.reset();
                    gb.boot();
                    // Reload ROM after reset
                    gb = emulator::Emulator::new(&rom_data)?;
                    gb.boot();
                    reward_sys.reset();
                    agent.network.reset_hidden();
                    episode_reward = 0.0;
                    episode_step = 0;
                    episode_num += 1;
                    // Rebuild frame stack
                    frame_buffer.clear();
                    for _ in 0..cli.frame_stack {
                        gb.next_frame();
                        frame_buffer.push(gb.screen_rgba().to_vec());
                    }
                }
            }
        } else if training && current != 0 {
            // Human is playing during training — clear agent hidden state
            agent.network.reset_hidden();
        }

        // Encode and broadcast screen
        if let Ok(png_data) = encode_png(&current_frame) {
            hub.broadcast_frame(png_data.clone());

            // Record if active
            if rec.status().active {
                rec.record_frame(&png_data, last_dpad, last_btn, episode_reward, 0.0, 0.0);
            }
        }

        // Check training state from HTTP
        if train_rx.has_changed().unwrap_or(false) {
            let new_val = *train_rx.borrow();
            if new_val != training {
                training = new_val;
                if training {
                    log::info!("training started");
                } else {
                    // Save checkpoint on stop
                    let _ = agent.save("checkpoints/final.pt");
                    log::info!("training stopped, checkpoint saved");
                }
            }
        }

        // ~60fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            tokio::time::sleep(frame_interval - elapsed).await;
        }
    }
}

fn joypad_bits(bits: u32) -> Vec<PadKey> {
    let mut v = Vec::with_capacity(4);
    if bits & joypad::JOYPAD_RIGHT != 0 { v.push(PadKey::Right); }
    if bits & joypad::JOYPAD_LEFT != 0 { v.push(PadKey::Left); }
    if bits & joypad::JOYPAD_UP != 0 { v.push(PadKey::Up); }
    if bits & joypad::JOYPAD_DOWN != 0 { v.push(PadKey::Down); }
    if bits & joypad::JOYPAD_A != 0 { v.push(PadKey::A); }
    if bits & joypad::JOYPAD_B != 0 { v.push(PadKey::B); }
    if bits & joypad::JOYPAD_SELECT != 0 { v.push(PadKey::Select); }
    if bits & joypad::JOYPAD_START != 0 { v.push(PadKey::Start); }
    v
}

fn action_bits(dpad: u8, btn: u8) -> Vec<PadKey> {
    let mut v = Vec::with_capacity(2);
    match dpad { 1 => v.push(PadKey::Up), 2 => v.push(PadKey::Down), 3 => v.push(PadKey::Left), 4 => v.push(PadKey::Right), _ => {} }
    match btn { 1 => v.push(PadKey::A), 2 => v.push(PadKey::B), 3 => v.push(PadKey::Start), 4 => v.push(PadKey::Select), _ => {} }
    v
}

fn encode_png(data: &[u8]) -> Result<Vec<u8>> {
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    let mut buf = std::io::Cursor::new(Vec::new());
    let encoder = PngEncoder::new(&mut buf);
    encoder.write_image(data, 160, 144, image::ExtendedColorType::Rgba8)?;
    Ok(buf.into_inner())
}

fn log_info_training(
    hub: &hub::Hub,
    last: &mut std::time::Instant,
    frames: u64,
    stats: &AgentStats,
    ep_reward: f64,
    ep_step: usize,
    ep_num: usize,
    sps: f64,
    rec: &Arc<recorder::Recorder>,
    dpad: &mut u8,
    btn: &mut u8,
    frame: &[u8],
) {
    let elapsed = last.elapsed();
    log::info!(
        "[{frames:>7}] ep {ep_num:>4} | return {ep_reward:>+7.2} | len {ep_step:>4} | "
    );
}
