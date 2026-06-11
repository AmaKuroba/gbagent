# rust-gbagent

Single Rust binary: Game Boy emulator + dashboard + PPO training agent.

## Repo Structure

```
rust-gbagent/
├── Cargo.toml
├── src/
│   ├── main.rs       # CLI, tokio runtime, 60fps loop
│   ├── emulator.rs   # boytacean wrapper
│   ├── http.rs       # Hyper HTTP server (:8765)
│   ├── ws.rs         # tokio-tungstenite WebSocket (:8766)
│   ├── hub.rs        # broadcast hub
│   ├── joypad.rs     # atomic joypad state
│   ├── recorder.rs   # PNG + JSONL + ffmpeg MP4
│   ├── config.rs     # YAML loader for RAM scanners
│   ├── reward.rs     # screen novelty + scanner rewards
│   └── agent/
│       ├── mod.rs
│       ├── encoder.rs  # ViT-Small (Candle)
│       └── ppo.rs      # ActorCritic + LSTM + PPO update
├── static/
│   └── index.html    # dashboard UI
└── configs/
    ├── pokemon_red.yaml
    └── ...
```

## Commands

```
cargo build --release                          # build
cargo run --release -- --rom roms/pokemon.gb   # serve human play
cargo run --release -- --rom roms/pokemon.gb --train  # serve + train
cargo run --release -- --features cuda ...     # CUDA training
```

## Ports

- `:8765` — HTTP dashboard
- `:8766` — WebSocket frame relay
