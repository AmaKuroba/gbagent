# GBAGent — Game Boy RL Agent

A vision-transformer-based PPO agent trained on Game Boy ROMs via [Stable-Retro](https://github.com/openai/retro).
Exports to TFLite (FP32/FP16/INT8) and ONNX for deployment on mobile, web, or edge devices.

## Quick start

```bash
# Install with base dependencies
pip install -e .

# Run with defaults
python train.py

# See resolved config
python train.py --show-config

# Override via CLI
python train.py --game PokemonRed-GB --total-timesteps 5_000_000
```

## Export & deploy

```bash
# Export a trained model to TFLite
python -m gbagent export checkpoints/model.keras export/model.tflite

# INT8 quantized (requires representative data)
python -m gbagent export model.keras export/model_int8.tflite --quantize int8 --repr-data samples.npy

# Export to ONNX
pip install -e ".[onnx]"
python -m gbagent export model.keras export/model.onnx --format onnx

# Run headless inference (no training/dashboard)
python -m gbagent run --model export/model.tflite --game PokemonRed-GB --steps 5000

# Show model info
python -m gbagent info checkpoints/model.keras
```

See **[DEPLOY.md](DEPLOY.md)** for full deployment documentation.

## Project structure

```
.
├── pyproject.toml         # Project metadata & dependencies
├── train.py               # Training entry point
├── configs/
│   └── default.yaml       # Default hyper-parameters
├── gbagent/
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point (export, run, info)
│   ├── action.py          # Action sampling & combining
│   ├── agent.py           # ViT + ActorCritic Keras model
│   ├── buffer.py          # GAE rollout buffer
│   ├── config.py          # YAML config loader (dataclass-backed)
│   ├── env.py             # GBAGEnv — Stable-Retro wrapper
│   ├── export.py          # TFLite / ONNX model export
│   ├── inference.py       # Headless inference engine
│   ├── ppo.py             # PPO loss & update step
│   ├── recorder.py        # Frame recording + ffmpeg export
│   ├── reward.py          # Shaped rewards & RAM scanners
│   └── server/            # Dashboard HTTP + WebSocket
│       ├── frames.py
│       ├── http.py
│       └── hub.py
├── DEPLOY.md              # Deployment guide (export, mobile, web, edge)
└── README.md
```

## Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 1 | **Scaffold** — project structure, env wrapper, config loader | ✅ |
| 2 | **Agent** — ViT encoder + ActorCritic Keras model | ✅ |
| 3 | **PPO training loop** — GAE, surrogate loss, update step | ✅ |
| 4 | **Reward system** — screen novelty + RAM scanners | ✅ |
| 5 | **Dashboard** — HTTP server + WebSocket + HTML UI | ✅ |
| 6 | **Game configs** — Pokemon Red + template | ✅ |
| 7 | **Recording** — PNG + JSONL + ffmpeg export | ✅ |
| 8 | **Export & deploy** — TFLite / ONNX inference | ✅ |
