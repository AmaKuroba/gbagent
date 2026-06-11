# === Go (emulator/) ===

[working-directory: 'emulator']
build:
    mkdir -p ../bin
    go build -o ../bin/gbagent ./cmd/gbagent

[working-directory: 'emulator']
build-viewer:
    mkdir -p ../bin
    go build -o ../bin/viewer ./cmd/viewer

[working-directory: 'emulator']
vet:
    go vet ./...

[working-directory: 'emulator']
lint:
    golangci-lint run

[working-directory: 'emulator']
fmt:
    go fmt ./...

[working-directory: 'emulator']
tidy:
    go mod tidy

[working-directory: 'emulator']
clean:
    go clean -cache
    rm -rf ../bin

[working-directory: 'emulator']
bench: testdata
    go test ./... -bench=. -benchmem

[working-directory: 'emulator']
test: testdata
    go test ./... -count=1

[working-directory: 'emulator']
test-fast: testdata
    go test ./... -count=1 -skip 'Mooneye'

[working-directory: 'emulator']
test-gb: testdata
    go test ./internal/gb/... -count=1

[working-directory: 'emulator']
test-gb-fast: testdata
    go test ./internal/gb/... -count=1 -skip 'Mooneye'

[working-directory: 'emulator']
test-cpu: testdata
    go test ./internal/gb/... -run TestCPU -count=1

[working-directory: 'emulator']
test-ppu: testdata
    go test ./internal/gb/... -run TestPPU -count=1

[working-directory: 'emulator']
test-apu: testdata
    go test ./internal/gb/... -run TestAPU -count=1

# Download test ROMs (Blargg, dmg_acid2, Mooneye)
[working-directory: 'emulator']
testdata:
    ./scripts/fetch-testdata.sh

# Start emulator (JSON-RPC server only)
[working-directory: 'emulator']
serve rom:
    go run ./cmd/gbagent serve --rom {{rom}}

# Start dashboard viewer (connects to emulator)
[working-directory: 'emulator']
viewer rom:
    go run ./cmd/viewer --emulator-url ws://localhost:8767/ws

# Quick check: build + vet + lint + fast tests
check: build build-viewer vet lint test-fast

# CI pipeline: build + vet + lint + fast tests only
ci: build build-viewer vet lint test-fast

# === Retro-driver (Python) ===

[working-directory: 'retro-driver']
setup-retro:
    uv sync

[working-directory: 'retro-driver']
test-retro:
    uv run python -m pytest tests/ -v

[working-directory: 'retro-driver']
lint-retro:
    uv run ruff check

[working-directory: 'retro-driver']
fmt-retro:
    uv run ruff format

# Train: connect to running gbagent, run training
# Example: just train '--resume models/final.pt'
[working-directory: 'retro-driver']
train args='':
    uv run python -m retro_driver.train {{args}}

# Start emulator + viewer + training in one go
# checkpoint: optional --load-state path (pre-saved state to skip intro)
watch-train rom checkpoint='' args='':
    #!/usr/bin/env bash
    set -euo pipefail
    cd emulator
    GB_LOAD=
    if [ -n "{{checkpoint}}" ]; then GB_LOAD="--load-state {{checkpoint}}"; fi
    go run ./cmd/gbagent serve --rom {{rom}} --jsonrpc-port 8767 $GB_LOAD & EMU_PID=$!
    go run ./cmd/viewer --emulator-url ws://localhost:8767/ws --port 8765 --metrics-port 8766 & VIEWER_PID=$!
    sleep 2
    cd ../retro-driver
    uv run python -m retro_driver.train --viewer-url ws://localhost:8766/metrics {{args}} || true
    kill $EMU_PID $VIEWER_PID 2>/dev/null || true
    wait $EMU_PID $VIEWER_PID 2>/dev/null || true

# Start TensorBoard to monitor training (http://localhost:6006)
[working-directory: 'retro-driver']
tensorboard:
    uv run python -m tensorboard.main --logdir logs --bind_all

# Kill any leftover processes
killall:
    pkill -9 -f "bin/gbagent" 2>/dev/null || true
    pkill -9 -f "bin/viewer" 2>/dev/null || true
    pkill -9 -f "tensorboard" 2>/dev/null || true
    pkill -9 -f "retro_driver" 2>/dev/null || true
    pkill -9 -f "uv run python" 2>/dev/null || true

default:
    just --list
