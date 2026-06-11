# === Build ===

[working-directory: 'emulator']
build:
    mkdir -p ../bin
    go build -o ../bin/gbagent ./cmd/gbagent
    go build -o ../bin/viewer ./cmd/viewer

# === Test ===

# Download test ROMs (Blargg, dmg_acid2, Mooneye)
[working-directory: 'emulator']
testdata:
    ./scripts/fetch-testdata.sh

[working-directory: 'emulator']
test: testdata
    go test ./... -count=1

[working-directory: 'emulator']
test-fast: testdata
    go test ./... -count=1 -skip 'Mooneye'

[working-directory: 'retro-driver']
test-retro:
    uv run python -m pytest tests/ -v

# === Lint / Format ===

[working-directory: 'emulator']
lint:
    golangci-lint run

[working-directory: 'emulator']
vet:
    go vet ./...

[working-directory: 'emulator']
fmt:
    go fmt ./...

[working-directory: 'retro-driver']
fmt-retro:
    uv run ruff format

# === Pre-commit ===

check: build vet lint test-fast

# === Run ===

[working-directory: 'emulator']
serve rom:
    go run ./cmd/gbagent serve --rom {{rom}}

[working-directory: 'retro-driver']
train args='':
    uv run python -m retro_driver.train {{args}}

# Start emulator + viewer + training in one go
# checkpoint: optional --load-state path (pre-saved state to skip intro)
watch rom checkpoint='' args='':
    #!/usr/bin/env bash
    set -euo pipefail
    cd emulator
    GB_LOAD=
    if [ -n "{{checkpoint}}" ]; then GB_LOAD="--load-state {{checkpoint}}"; fi
    go run ./cmd/gbagent serve --rom {{rom}} --jsonrpc-port 8767 $GB_LOAD & EMU_PID=$!
    go run ./cmd/viewer --emulator-url ws://localhost:8767/ws --port 8765 --metrics-port 8766 & VIEWER_PID=$!
    sleep 3
    cd ../retro-driver
    uv run python -m retro_driver.train --viewer-url ws://localhost:8766/metrics {{args}} || true
    kill $EMU_PID $VIEWER_PID 2>/dev/null || true
    wait $EMU_PID $VIEWER_PID 2>/dev/null || true

[working-directory: 'retro-driver']
tensorboard:
    uv run python -m tensorboard.main --logdir logs --bind_all

killall:
    pkill -9 -f "bin/gbagent" 2>/dev/null || true
    pkill -9 -f "bin/viewer" 2>/dev/null || true
    pkill -9 -f "tensorboard" 2>/dev/null || true
    pkill -9 -f "retro_driver" 2>/dev/null || true
    pkill -9 -f "uv run python" 2>/dev/null || true

default:
    just --list
