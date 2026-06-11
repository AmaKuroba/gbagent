# AGENTS.md

## Repo Structure

Two packages in a monorepo, no shared dependencies:

- **`emulator/`** — Go. Game Boy emulator core + JSON-RPC server.
- **`retro-driver/`** — Python (uv). RL agent (PPO+ViT) that connects to the emulator via WebSocket.

Entry points: `emulator/cmd/gbagent/main.go`, `emulator/cmd/viewer/main.go`, `retro-driver/retro_driver/train.py`

Three separate processes:
1. **gbagent** (emulator) — runs Game Boy at 60fps, serves JSON-RPC WebSocket on port 8767
2. **viewer** (dashboard) — serves web UI on port 8765, metrics on port 8766, connects to emulator as client
3. **train** (Python driver) — connects to emulator for game control, viewer for metrics

## Commands

### Go (from `emulator/`)

```
just build          # build emulator binary to bin/gbagent
just build-viewer   # build viewer binary to bin/viewer
just check          # build both + vet + lint + fast tests
just ci             # same as check
just test           # all tests (downloads test ROMs on first run)
just test-fast      # skip slow Mooneye tests
just test-gb        # emulator core tests only
just lint           # golangci-lint
just fmt            # go fmt
```

### Python (from `retro-driver/`)

```
just test-retro     # pytest tests/ -v
just lint-retro     # ruff check (source only — pre-existing warnings in tests/)
just fmt-retro      # ruff format
just train          # train agent against running gbagent
```

Or directly: `uv run ruff check retro_driver/`, `uv run pytest tests/ -v`

### Combined

```
just watch-train <rom> [checkpoint]  # starts emulator + viewer + train, cleans up on exit
just tensorboard                     # http://localhost:6006
just killall                         # kill leftover processes
```

## Verification Order

Go: `go vet` → `golangci-lint` → `go test`
Python: `ruff check` → `pytest`

All `just` commands run from the repo root — each recipe sets its own working directory.

## Key Quirks

- `just test` downloads test ROMs via `scripts/fetch-testdata.sh` on first run (requires network).
- Ruff config lives in `pyproject.toml` — source code (`retro_driver/`) is clean; test files have pre-existing warnings.
- `golangci-lint` config in `emulator/.golangci.yml` enables `errorlint`, `gocritic`, `prealloc` beyond defaults.
- Python requires 3.12+. Managed with `uv sync` (not pip).
- Game configs in `retro-driver/configs/` — YAML with RAM addresses verified against pret/pokered `.sym` file.
- WebSocket server (`CheckOrigin: true`) is intentional — app must work on LAN viewed by multiple devices.
- Emulator is pure: no web UI, no training awareness, no takeover logic. Just input/output.
- Takeover and action tracking live in the viewer/dashboard, not the emulator.
