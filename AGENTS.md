# Anchored Summary

## GBAGent — Game Boy RL Agent

### State file system

- **`State` enum** (`gbagent/state.py:13`) — `NONE`, `DEFAULT`, `CHECKPOINT` replace bare string literals
- **Auto-creation** (`gbagent/state.py:158`) — `resolve_state_file` auto-creates empty marked metadata when file missing
- **Fallback fix** (`train.py:240`) — uses `State.NONE` not `State.DEFAULT` to avoid crash when `metadata.json` absent
- **Gym wrappers** (`gbagent/env.py`) — `DumpInfoWrapper` optionally writes state info to `metadata.json`; `SaveStateWrapper` persists `metadata.json` and auto-saves on reset

### ROM directory support

- **`rom_dir` field** (`gbagent/config.py:57`) — optional path for custom ROM directories
- **`init_retro`** (`gbagent/env.py:112`) — registers `rom_dir` as a data directory via `stable_retro.data.add_data_dir` if set

### TF → JAX migration

- **`ppo.py`** — removed TF imports/globals, uses `K.GradientTape`, removed `max_grad_norm` param
- **`train.py`** — removed `keras.optimizers` + `global_clipnorm`, removed `max_grad_norm` from `ppo_update_step` call, rewrote TensorBoard helpers (`_create_tb_writer`, `_write_episode_summary`, `_write_final_summary`) using standalone `tensorboard.summary`
- **`action.py`** — already used `keras.ops`, no changes needed
- **`pyproject.toml`** — replaced `tensorflow>=2.16` with `jax>=0.4`
- **`Makefile`** — default backend `jax`, removed `-k "not keras"` filter
- **`tests/test_action.py`** — removed all `@pytest.mark.keras`, removed `pytest.importorskip("tensorflow")`, removed `import tensorflow as tf`, switched to `keras.ops.convert_to_tensor` and `float(...)` for assertions
- **`pyproject.toml`** — removed `keras` pytest marker
- **`.github/workflows/ci.yml`** — created: macOS + Linux, ruff lint + pytest with `KERAS_BACKEND=jax`

### Intent inference convention change

- **`train.py`** — switched from `add_action_integration()` (global state mutation) to explicit `inttype=...` parameter on `InferenceAgent`

### What still references TF

- `gbagent/export.py` — `import tensorflow as tf` inside TFLite/ONNX export functions (optional, guarded)
- `gbagent/inference.py` — `import tensorflow as tf` inside TF-inference function (optional, guarded)
