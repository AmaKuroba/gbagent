# GBAgent — Design Document

## Overview

Train reinforcement learning agents to play Game Boy / GBA games using custom PPO, a ViT encoder, and the Stable-Retro emulator backend. Everything is handwritten — no Stable-Baselines3, no off-the-shelf RL libraries.

## Hardware

| Role | Machine | GPU | Notes |
|------|---------|-----|-------|
| Dev | localhost (M1 Mac) | Apple M1 GPU | Code, debug, small-scale tests |
| Training | freighter.local | NVIDIA GPU | Heavy training runs, long sessions |

Code is developed on the M1 Mac and synced/run on freighter.local for training with CUDA acceleration.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Emulator | Stable-Retro (Farama) | Gymnasium-native, save/load states, 1000+ games, multi-platform |
| ML framework | Keras 3 | Backend-agnostic (TF/JAX/PyTorch), clean API, TensorBoard integration, TFLite export |
| Backend | TensorFlow (dev) / JAX (training) | TF on M1 for dev, JAX on freighter.local for CUDA GPU training |
| Visualization | TensorBoard + custom web dashboard | TensorBoard for metrics, web UI for live gameplay + controls |
| Config | YAML | Human-readable game-specific reward definitions |
| Recording | PNG + JSONL → optional ffmpeg MP4 | Frame-by-frame capture for dataset creation |

## Project Structure

```
gbagent/
├── pyproject.toml           # Dependencies, metadata
├── DESIGN.md                # This file
├── justfile                 # Build/run commands
│
├── gbagent/
│   ├── __init__.py
│   ├── env.py               # Stable-Retro environment wrapper
│   ├── config.py            # YAML config loading
│   ├── reward.py            # Reward system (screen novelty + RAM scanners)
│   ├── joypad.py            # Joypad state (shared between WS/HTTP/env)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── encoder.py       # ViT-Small vision encoder
│   │   ├── network.py       # ActorCritic: ViT → LSTM → policy/value heads
│   │   ├── ppo.py           # PPO algorithm (GAE, surrogate loss, update)
│   │   └── buffer.py        # Rollout buffer
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   ├── http.py          # Hypercorn/FastAPI or aiohttp HTTP dashboard
│   │   ├── ws.py            # WebSocket frame relay + joypad input
│   │   └── hub.py           # Broadcast hub (fan-out frames + metrics)
│   │
│   ├── recorder.py          # PNG + JSONL recording, ffmpeg encoding
│   │
│   └── train.py             # Main training loop / entry point
│
├── configs/
│   ├── pokemon_red.yaml     # RAM scanners for Pokemon Red
│   ├── pokemon_emerald.yaml # RAM scanners for Pokemon Emerald
│   └── _template.yaml       # Template for adding new games
│
├── static/
│   └── index.html           # Dashboard HTML + JS
│
├── recordings/              # Gameplay recordings (gitignored)
├── checkpoints/             # Model weights (gitignored)
└── logs/                    # TensorBoard logs (gitignored)
```

---

## 1. Environment Layer (`gbagent/env.py`)

### Stable-Retro Wrapper

Stable-Retro provides `retro.make(game=..., state=...)` returning a Gymnasium `Env`. We wrap it with:

- **Frame stack** — last N frames as observation (N=4 default)
- **Preprocessing** — convert RGB→grayscale, downsample if needed, normalize to [0,1]
- **Action space** — Stable-Retro exposes a `MultiBinaryDiscrete` action space per game. We discretize to 9 actions:
  - D-pad: Up, Down, Left, Right, None (5)
  - Buttons: A, B, Start, Select, None (5)
  - Combined: 5×5 = 25 possible action combos
  - *(Can be expanded for GBA with L/R shoulder buttons)*
- **Frame skip** — repeat action N frames (N=4 default)
- **Save/load state** — stable_retro supports this natively via `env.reset(state=...)` and manual savestate files

### Interface

```python
class GBAgentEnv:
    def __init__(self, rom_path, game_name, config):
        self.env = retro.make(game=game_name, state=retro.State.DEFAULT)

    def reset(self):
        # Reset emulator, clear frame stack, return observation

    def step(self, dpad_action, btn_action):
        # Apply action with frame_skip
        # Accumulate reward from RewardSystem
        # Return (observation, reward, done, info)

    def render(self):
        # Return current frame as RGBA numpy array

    def close(self):
        self.env.close()

    # Save/load state helpers
    def save_state(self, path)
    def load_state(self, path)
```

---

## 2. Agent Architecture

### Overview

```
[Screen] → ViT-Encoder → [Embedding] → LSTM → [Hidden]
                                                ├─ DPadHead → [5-way softmax]
                                                ├─ BtnHead  → [5-way softmax]
                                                └─ ValueHead → [scalar]
```

### ViT Encoder (`gbagent/agent/encoder.py`)

**ViT-Small** — matches the Rust version's architecture:

| Parameter | Value |
|-----------|-------|
| Input | 4×144×160 (4 stacked grayscale frames) |
| Patch size | 16×16 |
| Grid | 9×10 = 90 patches |
| Embed dim | 384 |
| Layers | 6 |
| Heads | 6 |
| MLP ratio | 4.0 |
| Parameters | ~22M |

Implementation in Keras:

```python
class ViTEncoder(keras.Model):
    def __init__(self, config):
        self.patch_embed = keras.Sequential([
            layers.Conv2D(embed_dim, patch_size, stride=patch_size),
            layers.Reshape((num_patches, embed_dim)),
        ])
        self.pos_embed = self.add_weight(shape=(1, num_patches, embed_dim))
        self.blocks = [TransformerBlock(...) for _ in range(num_layers)]
        self.norm = layers.LayerNormalization()

    def call(self, x):
        # x: (B, H, W, C) → patches + pos_embed → transformer → mean pool
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(axis=1)  # (B, embed_dim)
```

### ActorCritic (`gbagent/agent/network.py`)

```python
class ActorCritic(keras.Model):
    def __init__(self, embed_dim=384, lstm_units=256):
        self.encoder = ViTEncoder(...)
        self.lstm = layers.LSTM(lstm_units, return_state=True)
        self.dpad_head = layers.Dense(5, activation=None)   # logits
        self.btn_head = layers.Dense(5, activation=None)    # logits
        self.value_head = layers.Dense(1, activation=None)  # scalar

    def call(self, obs, states=None, training=False):
        # obs: (B, H, W, C) → encoded → LSTM → heads
        encoded = self.encoder(obs)
        lstm_out, h, c = self.lstm(encoded, initial_state=states)
        dpad_logits = self.dpad_head(lstm_out)
        btn_logits = self.btn_head(lstm_out)
        value = self.value_head(lstm_out)
        return dpad_logits, btn_logits, value, (h, c)
```

### Action Sampling

```python
def sample_action(dpad_logits, btn_logits, training=True):
    if training:
        dpad = tf.squeeze(tf.random.categorical(dpad_logits, 1))
        btn = tf.squeeze(tf.random.categorical(btn_logits, 1))
    else:
        dpad = tf.argmax(dpad_logits, axis=-1)
        btn = tf.argmax(btn_logits, axis=-1)
    # Compute log probs for PPO
    dpad_log_prob = ...  # cross_entropy with sampled action
    btn_log_prob = ...
    return dpad, btn, dpad_log_prob, btn_log_prob
```

---

## 3. PPO Algorithm (`gbagent/agent/ppo.py`)

### Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| γ (gamma) | 0.99 | Discount factor |
| λ (GAE lambda) | 0.95 | GAE trace decay |
| ε (clip) | 0.2 | PPO clipping range |
| c_entropy | 0.01 | Entropy bonus coefficient |
| c_value | 0.5 | Value loss coefficient |
| lr | 3e-4 | AdamW learning rate |
| Rollout length | 128 | Steps before each PPO update |
| PPO epochs | 4 | Epochs per batch |
| Batch size | 32 | Minibatch size |
| Total steps | 6M+ | Total environment steps |
| Max grad norm | 0.5 | Gradient clipping |

### GAE (Generalized Advantage Estimation)

```python
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
```

### PPO Update

```python
def update(self, rollout):
    # Normalize advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    for _ in range(ppo_epochs):
        for batch in minibatches(rollout, batch_size):
            with tf.GradientTape() as tape:
                # Forward pass
                dpad_logits, btn_logits, values, _ = self.network(batch.obs)

                # Ratio = exp(new_log_prob - old_log_prob)
                ratio = tf.exp(new_log_prob - old_log_prob)

                # Clipped surrogate loss
                surr1 = ratio * advs
                surr2 = tf.clip_by_value(ratio, 1-eps, 1+eps) * advs
                policy_loss = -tf.minimum(surr1, surr2)

                # Value loss (MSE)
                value_loss = tf.reduce_mean((values - returns)**2)

                # Entropy bonus
                entropy = -tf.reduce_mean(
                    dpad_probs * tf.math.log(dpad_probs + 1e-10) +
                    btn_probs * tf.math.log(btn_probs + 1e-10)
                )

                loss = policy_loss + c_value * value_loss - c_entropy * entropy

            grads = tape.gradient(loss, self.network.trainable_variables)
            grads = tf.clip_by_global_norm(grads, max_grad_norm)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
```

---

## 4. Reward System (`gbagent/reward.py`)

Three reward sources, summed:

### 4a. Screen Novelty

Pixel-level difference between consecutive frames. Encourages exploration.

```
avg_diff = mean(|frame[t] - frame[t-1]|)  # per pixel, per channel
novelty = clamp(avg_diff / 255.0, 0, 1) * 0.5
```

### 4b. Stale Penalty

If the screen hasn't changed meaningfully for >60 frames, apply a ramping penalty to discourage getting stuck.

```
stale_penalty = -0.05 * min(stale_frames / 60.0, 10.0)
```

### 4c. RAM Scanners — *The main reward signal*

YAML-defined memory scanners that watch specific RAM addresses and fire rewards when values change. This is the key mechanism for game-specific reward shaping.

```yaml
# configs/pokemon_red.yaml
game: "PokemonRed-GB"
scanners:
  # Reward for HP changes (battles, damage, healing)
  - name: "player_hp"
    ram_addr: 0xD163
    type: "discrete"      # reward_on_change on any value change
    reward_on_change: 0.2

  # Reward for XP gain
  - name: "player_xp"
    ram_addr: 0xD179
    type: "delta"          # reward_per_unit × amount of change
    reward_per_unit: 0.01

  # Reward for money
  - name: "money"
    ram_addr: 0xD347
    length: 2
    type: "delta_multi"    # multi-byte delta
    reward_per_unit: 0.005

  # Penalty for blacking out (fainting)
  - name: "blackout"
    ram_addr: 0xD057
    type: "discrete"
    reward_on_change: -5.0  # negative reward for game over state
```

Scanner types:

| Type | Behavior |
|------|----------|
| `discrete` | Any byte change → fixed reward |
| `delta` | Byte change scaled by magnitude |
| `multi_byte` | Any byte change in range → fixed reward |
| `delta_multi` | Multi-byte change scaled by magnitude |

---

## 5. Training Loop (`gbagent/train.py`)

High-level flow:

```
for total_step in range(6_000_000):
    obs = env.reset()

    # Rollout phase
    for step in range(rollout_len):
        dpad, btn, lp_d, lp_b, value = agent.act(obs, training=True)
        next_obs, reward, done, _ = env.step(dpad, btn)
        buffer.add(obs, dpad, btn, lp_d, lp_b, reward, value, done)

        if done or step == max_episode_len:
            obs = env.reset()

    # Learning phase
    stats = agent.update(buffer)
    log_metrics(stats)
```

Episode reset uses Stable-Retro's savestate feature:
1. First reset: load the game's initial state
2. After that: reload from a clean savestate snapshot

---

## 6. Server & Dashboard

### Architecture

```
┌─────────┐     WebSocket(:8766)     ┌──────────┐
│ Browser │ ←──── frames + metrics ──→ │  Python  │
│ (HTML)  │ ──── joypad input ─────→ │  Server  │ ←→ Emulator
└─────────┘     HTTP(:8765)          └──────────┘
```

### HTTP Server

- Framework: **aiohttp** (lighter than FastAPI for this use case)
- Routes:
  - `GET /` → serve `static/index.html`
  - `POST /train/start` → begin training
  - `POST /train/stop` → stop training, save checkpoint
  - `POST /record/start` → start recording
  - `POST /record/stop` → stop recording (optional MP4 encode)
  - `GET /record/status` → current recording state
  - `POST /state/save` → save emulator state
  - `POST /state/load` → load emulator state

### WebSocket

- Binary message (0x00 + PNG data) → live frame
- Text message (JSON) → metrics from training
- Incoming text → joypad press/release commands, takeover, save/load

### Dashboard (`static/index.html`)

Dark cyberpunk theme (same vibe as the original Rust version):

- **Left:** Live Game Boy screen (PNG stream via WebSocket)
- **Right panel:**
  - Virtual joypad (touch/click friendly)
  - Train/Stop button
  - Connection status
  - Live training stats (reward, loss, SPS, episode length)
  - Mini reward chart (rolling graph)
  - Event log (scrolling text panel)

### TensorBoard

Logged automatically by Keras during training:

- `loss/total` — combined PPO loss
- `loss/policy` — policy gradient loss
- `loss/value` — value function MSE
- `loss/entropy` — entropy bonus
- `metrics/reward` — episode return
- `metrics/episode_length` — steps per episode
- `metrics/sps` — steps per second
- `lr` — learning rate
- `grad_norm` — gradient norm

---

## 7. Recording (`gbagent/recorder.py`)

Same format as the Rust version:

```
recordings/
└── YYYY-MM-DD_HH-MM-SS/
    ├── data.jsonl       # Per-frame metadata
    └── frames/
        ├── 000001.png
        ├── 000002.png
        └── ...
```

`data.jsonl` schema:

```json
{
  "frame": "000001.png",
  "dpad": 3,
  "btn": 0,
  "reward": 0.05,
  "loss": 0.12,
  "epsilon": 0.01,
  "t_ms": 1703275200000
}
```

On stop, optionally combine frames into MP4 via ffmpeg.

---

## 8. Configuration

### Game Config (`configs/*.yaml`)

```yaml
game: "PokemonRed-GB"           # Stable-Retro game name
state: "Level1.state"           # Optional: start from a specific savestate
reward_clip: [-10, 10]          # Clamp total reward per step
max_episode_steps: 3000         # Max steps before forced reset

scanners:
  - name: "player_hp"
    ram_addr: 0xD163
    type: "discrete"
    reward_on_change: 0.2
```

### CLI Arguments

```bash
# Train on Pokemon Red
python -m gbagent.train --rom roms/PokemonRed.gb --config configs/pokemon_red.yaml --train

# Just serve dashboard (human play)
python -m gbagent.train --rom roms/PokemonRed.gb

# Resume from checkpoint
python -m gbagent.train --rom roms/PokemonRed.gb --resume checkpoints/model_100k.weights.h5

# Record gameplay
python -m gbagent.train --rom roms/PokemonRed.gb --record
```

---

## 9. Model Export & Deployment

### Checkpointing

- Keras `.weights.h5` format during training (lightweight, just weights)
- Full `.keras` model periodically for inference deployment

### Export Paths

```python
# TFLite (for lightweight/edge inference)
converter = tf.lite.TFLiteConverter.from_keras_model(agent.network)
tflite_model = converter.convert()
with open("gbagent.tflite", "wb") as f:
    f.write(tflite_model)

# ONNX (for cross-framework portability)
# Requires tf2onnx
```

### Deployment Target

The agent model (ViT + LSTM) at inference is ~22M params. With INT8 quantization via TFLite, can run on:
- Modern phones
- Raspberry Pi 4/5
- Even the same Mac it was trained on for headless rollout

---

## 10. Comparison with Rust Version

| Aspect | Rust (dead) | Python (new) |
|--------|------------|--------------|
| Emulator | boytacean (custom) | Stable-Retro (mature, maintained) |
| ML | Candle (Rust ML) | Keras 3 (TF/JAX/PyTorch) |
| Training | Custom PPO | Custom PPO (same algorithm) |
| Arch | ViT-Small + LSTM | ViT-Small + LSTM (same arch) |
| Dashboard | HTTP + WS | HTTP + WS (similar) |
| Recording | PNG + JSONL | PNG + JSONL (same format) |
| Config | YAML | YAML (same format) |
| Difficulty | Writing everything + Rust ML immaturity | Python ecosystem maturity |

---

## Milestones

1. **Scaffold** — Project structure, env wrapper, config loading
2. **Agent** — ViT encoder + ActorCritic Keras model
3. **Training loop** — PPO rollout, GAE, update step
4. **Dashboard** — HTTP server, WebSocket, HTML UI
5. **Reward system** — Screen novelty + RAM scanners
6. **Game-specific tuning** — Work through Pokemon Red config
7. **Recording** — PNG + JSONL, ffmpeg export
8. **Export & deploy** — TFLite/ONNX, headless inference
9. **Iterate** — Tune hyperparameters, add more games
