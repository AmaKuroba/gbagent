.PHONY: test test-roms lint bench clean

test:
	go test ./... -v -count=1

test-short:
	go test ./... -count=1

test-cpu:
	go test ./internal/gb/... -run TestCPU -v -count=1

test-ppu:
	go test ./internal/gb/... -run TestPPU -v -count=1

test-apu:
	go test ./internal/gb/... -run TestAPU -v -count=1

test-roms:
	go test ./... -run TestROM -v -count=1

lint:
	go vet ./...
	staticcheck ./...

bench:
	go test ./... -bench=. -benchmem

clean:
	go clean -cache
