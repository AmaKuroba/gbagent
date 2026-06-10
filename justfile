set working-directory := 'emulator'

# === Go (emulator/) ===

build:
    mkdir -p ../bin
    go build -o ../bin/gbagent ./cmd/gbagent

vet:
    go vet ./...

lint:
    golangci-lint run

fmt:
    go fmt ./...

tidy:
    go mod tidy

clean:
    go clean -cache
    rm -rf ../bin

bench: testdata
    go test ./... -bench=. -benchmem

test: testdata
    go test ./... -count=1

test-fast: testdata
    go test ./... -count=1 -skip 'Mooneye'

test-gb: testdata
    go test ./internal/gb/... -count=1

test-gb-fast: testdata
    go test ./internal/gb/... -count=1 -skip 'Mooneye'

test-cpu: testdata
    go test ./internal/gb/... -run TestCPU -count=1

test-ppu: testdata
    go test ./internal/gb/... -run TestPPU -count=1

test-apu: testdata
    go test ./internal/gb/... -run TestAPU -count=1

# Download test ROMs (Blargg, dmg_acid2, Mooneye)
testdata:
    ./scripts/fetch-testdata.sh

serve rom:
    go run ./cmd/gbagent serve --rom {{rom}}

serve-with-ws rom port='8765' jsonrpc_port='8767':
    go run ./cmd/gbagent serve --rom {{rom}} --port {{port}} --jsonrpc-port {{jsonrpc_port}}

# Quick check: build + vet + lint + fast tests
check: build vet lint test-fast

# CI pipeline: build + vet + lint + fast tests only
ci: build vet lint test-fast

# === Retro-driver (Python) ===

retro := "cd ../retro-driver &&"

setup-retro:
    {{retro}} uv sync

test-retro:
    {{retro}} uv run python -m pytest tests/ -v

lint-retro:
    {{retro}} uv run ruff check

fmt-retro:
    {{retro}} uv run ruff format

# Train: connect to running gbagent, run training
# Example: just train '--resume models/final.pt'
train args='':
    {{retro}} uv run python -m retro_driver.train {{args}}

# Like train but starts gbagent + opens dashboard at http://localhost:8765
watch-train rom args='':
    ../bin/gbagent serve --rom {{rom}} --jsonrpc-port 8767 & GBAGENT_PID=$! && sleep 2 && {{retro}} uv run python -m retro_driver.train --ram-scanner configs/pokemon_red.yaml {{args}}; kill $GBAGENT_PID 2>/dev/null; wait $GBAGENT_PID 2>/dev/null || true

# Start TensorBoard to monitor training (http://localhost:6006)
tensorboard:
    {{retro}} uv run python -m tensorboard.main --logdir logs --bind_all

# Kill any leftover gbagent / tensorboard / training processes
killall:
    pkill -9 -f "bin/gbagent" 2>/dev/null || true
    pkill -9 -f "tensorboard" 2>/dev/null || true
    pkill -9 -f "retro_driver" 2>/dev/null || true
    pkill -9 -f "uv run python" 2>/dev/null || true

default:
    just --list
