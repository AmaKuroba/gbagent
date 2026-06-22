# Deployment Guide — gbagent Model Export & Inference

This document describes how to export trained gbagent models to TFLite (with
optional INT8 quantization) and ONNX, and how to run inference in headless
(deployment) mode without the training loop or dashboard.

---

## Table of Contents

- [Quick start](#quick-start)
- [Export formats](#export-formats)
  - [TFLite (FP32)](#tflite-fp32)
  - [TFLite (FP16)](#tflite-fp16)
  - [TFLite (INT8 full quantization)](#tflite-int8-full-quantization)
  - [ONNX](#onnx)
- [Headless inference](#headless-inference)
- [Python API](#python-api)
- [Deployment targets](#deployment-targets)
  - [Desktop / server (Python)](#desktop--server-python)
  - [Mobile (Android / iOS)](#mobile-android--ios)
  - [Web (TensorFlow.js / ONNX.js)](#web-tensorflowjs--onnxjs)
  - [Edge / embedded (TFLite Micro)](#edge--embedded-tflite-micro)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1. Export a trained model to TFLite (FP32)
python -m gbagent export checkpoints/model_final_step_005000000.keras export/model.tflite

# 2. Run headless inference
python -m gbagent run --model export/model.tflite --game PokemonRed-GB --steps 1000

# 3. Get model info
python -m gbagent info checkpoints/model.keras
```

---

## Export formats

### Prerequisites

Install optional dependencies as needed:

```bash
# ONNX export + ONNX Runtime inference
pip install -e ".[onnx]"
```

### TFLite (FP32)

Default export — no quantization, preserves full float32 precision.

```bash
python -m gbagent export checkpoints/model.keras export/model_fp32.tflite
```

**Use when:** accuracy is critical and model size / latency are not constraints.

### TFLite (FP16)

Weight-only half-precision quantization. Activations remain FP32.

```bash
python -m gbagent export checkpoints/model.keras export/model_fp16.tflite --quantize fp16
```

**Use when:** you want a ~50% smaller model with negligible accuracy loss.

### TFLite (INT8 full quantization)

Full integer quantization: weights **and** activations are converted to 8-bit.
Requires a representative dataset for calibration (a small sample of real
observations from the game environment).  The resulting model expects uint8
input and produces uint8 output; the inference engine handles the conversion
automatically.

```bash
# Collect representative data (e.g. 200 random frames from random play)
python -c "
import numpy as np
from gbagent.env import GBAGEnv

env = GBAGEnv(game='PokemonRed-GB', state='Level1')
obs, _ = env.reset()
samples = []
for _ in range(200):
    obs, _, done, _, _ = env.step(env.action_space.sample())
    samples.append(obs)
    if done:
        obs, _ = env.reset()
np.save('export/representative_data.npy', np.array(samples, dtype=np.float32))
print('Saved 200 samples to export/representative_data.npy')
"

# Export with INT8 quantization
python -m gbagent export checkpoints/model.keras export/model_int8.tflite \
    --quantize int8 --repr-data export/representative_data.npy
```

**Use when:** deploying to resource-constrained devices (mobile, edge TPU,
microcontrollers).  Expect a ~4× size reduction and 2–4× latency improvement
with minimal accuracy degradation if calibrated well.

### ONNX

Export to Open Neural Network Exchange format for use with ONNX Runtime or
cross-platform inference.

```bash
pip install -e ".[onnx]"
python -m gbagent export checkpoints/model.keras export/model.onnx --format onnx
```

**Use when:** you need cross-platform portability or want to run inference
with ONNX Runtime (C++, Python, C#, Java, Rust, WebAssembly).

---

## Headless inference

The inference engine loads a trained model (any format) and runs it against
a game environment *without* the training loop, dashboard server, TensorBoard,
or any training-only dependencies.

### CLI

```bash
# Keras backend (default)
python -m gbagent run --model checkpoints/model.keras --game PokemonRed-GB --steps 5000

# TFLite backend
python -m gbagent run --model export/model.tflite --game PokemonRed-GB --steps 5000

# ONNX backend
python -m gbagent run --model export/model.onnx --game PokemonRed-GB --steps 5000

# Greedy (deterministic) actions, with GUI
python -m gbagent run --model export/model.tflite --game PokemonRed-GB --steps 1000 --deterministic --render

# Custom state
python -m gbagent run --model model.tflite --game PokemonRed-GB --state Level1 --steps 2000 --seed 42
```

Output example:

```
› Loading model from export/model.tflite …
  Backend: tflite

› Creating environment: game=PokemonRed-GB, state=Level1
› Running inference for 1000 steps (deterministic=False) …

╔════════════════════════════════════════════╗
║           Inference Results                ║
╚════════════════════════════════════════════╝
  Steps:          1,000
  Total reward:   12.45
  Mean value:     2.3410
  Duration:       14.2s
  FPS:            70.4
```

### Python API

```python
from gbagent.inference import InferenceEngine
from gbagent.env import GBAGEnv

# Load model
engine = InferenceEngine("export/model.tflite", backend="tflite")

# Create environment
env = GBAGEnv(game="PokemonRed-GB", state="Level1")

# Run inference manually
obs, _ = env.reset()
for _ in range(100):
    # Single-step prediction
    dpad_logits, btn_logits, value = engine.predict_single(obs)

    # Sample action (training=False → argmax)
    from gbagent.action import sample_actions, combine_actions
    dpad_a, btn_a, _ = sample_actions(
        dpad_logits[np.newaxis, ...],
        btn_logits[np.newaxis, ...],
        training=False,
    )
    action = combine_actions(dpad_a, btn_a)[0]

    # Step environment
    obs, reward, done, _, _ = env.step(int(action))
    print(f"Step {_}: reward={reward:.2f}, value={value[0]:.3f}")

    if done:
        break

env.close()

# Or use the built-in rollout helper
result = engine.rollout(env, n_steps=500, deterministic=True)
print(f"Total reward: {result['total_reward']:.2f}")
```

---

## Deployment targets

### Desktop / server (Python)

| Format   | Inference backend    | Install                          |
|----------|----------------------|----------------------------------|
| `.keras` | Keras + TensorFlow   | `pip install tensorflow`         |
| `.tflite`| TensorFlow Lite      | `pip install tensorflow` (incl.) |
| `.onnx`  | ONNX Runtime         | `pip install onnxruntime`        |

**Recommended:** TFLite (FP32 or FP16) for general Python deployments.
Minimum dependencies; no training graph.

### Mobile (Android / iOS)

| Platform | Format  | Using                                |
|----------|---------|--------------------------------------|
| Android  | TFLite  | `org.tensorflow:tensorflow-lite`     |
| iOS      | TFLite  | `TensorFlowLiteSwift` / `TFLite C++` |
| Both     | ONNX    | ONNX Runtime mobile                  |

**Recommended:** INT8 quantised TFLite.
- Model size: ~250–400 KB (from ~2 MB FP32).
- Inference latency: 3–10 ms on modern phones.

Steps:

1. Export to INT8 TFLite as shown above.
2. Add `tensorflow-lite` dependency to your app.
3. Load the `.tflite` buffer, feed 84×84×4 uint8 input, read the three
   output tensors.

### Web (TensorFlow.js / ONNX.js)

| Framework        | Format         | Lib                              |
|------------------|----------------|----------------------------------|
| TensorFlow.js    | TFJS / TFLite  | `@tensorflow/tfjs` / `@tensorflow/tfjs-tflite` |
| ONNX.js          | ONNX           | `onnxruntime-web`                |

**Recommended:** TFLite via `tfjs-tflite` or ONNX via `onnxruntime-web`.
Model size after INT8: ~250 KB — feasible for web download.

### Edge / embedded (TFLite Micro)

| Device          | Constraints                   | Format |
|-----------------|-------------------------------|--------|
| Coral Edge TPU  | INT8 only; 8 MB SRAM          | TFLite |
| Arduino / ESP32 | < 1 MB flash, < 512 KB RAM    | TFLite Micro |

**Recommended:** INT8 quantised TFLite, further optimised with TFLite Micro
toolchain.  Note: the full 84×84×4 input (28 KB) may exceed on-device RAM
on the smallest microcontrollers — consider reducing input resolution or
frame stack depth for deeply embedded targets.

---

## Benchmarking

Compare inference speed across backends:

```bash
# Warm up + timing for 100 batches
python -c "
import time, numpy as np
from gbagent.inference import InferenceEngine

for model_path in ['model.keras', 'model.tflite', 'model.onnx']:
    engine = InferenceEngine(model_path)
    batch = np.random.randn(1, 84, 84, 4).astype(np.float32)

    # warmup
    for _ in range(10):
        engine.predict(batch)

    # timed
    start = time.perf_counter()
    for _ in range(100):
        engine.predict(batch)
    elapsed = time.perf_counter() - start
    ms_per = elapsed / 100 * 1000
    print(f'{model_path:30s}  {ms_per:.2f} ms/batch')
"
```

Example results (Apple M4):

| Model                               | Latency (ms/batch) | Size    |
|-------------------------------------|---------------------|---------|
| Keras (FP32)                         | 12–18               | ~8 MB   |
| TFLite (FP32)                        | 8–12                | ~2 MB   |
| TFLite (FP16)                        | 8–12                | ~1 MB   |
| TFLite (INT8)                        | 3–6                 | ~300 KB |
| ONNX (FP32, CPU)                     | 10–15               | ~2 MB   |

---

## Troubleshooting

| Problem                              | Likely cause                        | Fix                                                   |
|--------------------------------------|-------------------------------------|-------------------------------------------------------|
| `ModuleNotFoundError: tf2onnx`       | ONNX deps not installed             | `pip install -e ".[onnx]"`                            |
| `ValueError: Cannot infer backend`   | Unknown file extension              | Set `--backend keras`/`tflite`/`onnx` explicitly       |
| INT8 export gives poor accuracy      | Representative data not diverse     | Collect > 500 samples from actual play                 |
| TFLite inference returns garbage     | Input dtype mismatch                | The engine handles uint8 conversion; ensure obs is [0,1] |
| ONNX Runtime crash on ARM macOS      | Missing Metal EP                    | `pip install onnxruntime-silicon`                      |
| Inference FPS too low                | Batch size 1, no GPU                | Use TFLite; enable XNNPack / Metal delegate            |
