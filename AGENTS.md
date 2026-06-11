# AGENTS.md

## Repo Structure

Two packages in a monorepo, no shared dependencies:

- **`emulator/`** — Go. Headless Game Boy emulator with MCP/JSON-RPC interface.
- **`retro-driver/`** — Python (uv). RL agent (DQN) that connects to the emulator via WebSocket.

Entry points: `emulator/cmd/gbagent/main.go`, `retro-driver/retro_driver/train.py`

## Commands

### Go (from `emulator/`)

```
just build          # build binary to bin/gbagent
just check          # build + vet + lint + fast tests (preferred pre-commit)
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
just watch-train <rom> [checkpoint]  # starts gbagent + train, cleans up on exit
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
