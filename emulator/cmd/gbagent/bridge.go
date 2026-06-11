package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/AmaKuroba/gbagent/internal/gb"
)

// mcpCmd is dispatched to the command processor for race-free execution.
type mcpCmd struct {
	fn     func() any
	result chan any
}

// bridgeSnapshot is an atomic snapshot of the emulator state, updated by the
// frame loop every tick and read by RPC handlers concurrently.
type bridgeSnapshot struct {
	mu         sync.RWMutex
	screen     [160][144]byte
	cpuState   gb.CPUState
	ppuState   gb.PPUState
	timerState gb.TimerState
	frameCount uint64
}

func (s *bridgeSnapshot) read() (screen [160][144]byte, cpu gb.CPUState, ppu gb.PPUState, timer gb.TimerState, frameCount uint64) {
	s.mu.RLock()
	screen = s.screen
	cpu = s.cpuState
	ppu = s.ppuState
	timer = s.timerState
	frameCount = s.frameCount
	s.mu.RUnlock()
	return
}

func (s *bridgeSnapshot) write(ppu *gb.PPUCore, cpu *gb.Core, timer *gb.Timer) {
	s.mu.Lock()
	s.screen = ppu.GetScreen()
	s.cpuState = cpu.GetState()
	s.ppuState = ppu.GetState()
	s.timerState = timer.GetState()
	s.frameCount++
	s.mu.Unlock()
}

// mcpBridge adapts the live emulator for JSON-RPC handlers.
// The frame loop owns the emulator state; RPC read handlers use the snapshot;
// RPC write handlers use the latch; state-modifying handlers (save/load/reset)
// queue via the cmds channel and are processed by a background goroutine.
type mcpBridge struct {
	mmu     *gb.MemoryBus
	cpu     *gb.Core
	ppu     *gb.PPUCore
	timer   *gb.Timer
	apu     *gb.APU
	cart    gb.Cartridge
	romPath string

	cmds  chan mcpCmd
	snap  bridgeSnapshot
	donec chan struct{}

	// Agent button latch: the RPC agent writes bits via press_button /
	// release_button; the frame loop reads them every tick and injects
	// into the MMU. Mutex is fine — writes are rare, reads are fast.
	latchMu sync.Mutex
	latched byte
}

func (b *mcpBridge) SetLatchedBits(bits byte) {
	b.latchMu.Lock()
	b.latched |= bits
	b.latchMu.Unlock()
}

func (b *mcpBridge) ClearLatchedBits(bits byte) {
	b.latchMu.Lock()
	b.latched &^= bits
	b.latchMu.Unlock()
}

func (b *mcpBridge) ClearAllLatched() {
	b.latchMu.Lock()
	b.latched = 0
	b.latchMu.Unlock()
}

func (b *mcpBridge) ReadLatched() byte {
	b.latchMu.Lock()
	bits := b.latched
	b.latchMu.Unlock()
	return bits
}

func newBridge(mmu *gb.MemoryBus, cpu *gb.Core, ppu *gb.PPUCore, timer *gb.Timer, apu *gb.APU, cart gb.Cartridge, romPath string) *mcpBridge {
	return &mcpBridge{
		mmu:     mmu,
		cpu:     cpu,
		ppu:     ppu,
		timer:   timer,
		apu:     apu,
		cart:    cart,
		romPath: romPath,
		cmds:    make(chan mcpCmd, 64),
		donec:   make(chan struct{}),
	}
}

// SaveState persists the full emulator state to a gob file.
// Thread-safe: runs on the background processor goroutine.
func (b *mcpBridge) SaveState(path string) error {
	result := b.exec(func() any {
		state := b.mmu.DumpEmulatorState()
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		if err := gob.NewEncoder(f).Encode(state); err != nil {
			f.Close() //nolint: errcheck
			return err
		}
		return f.Close()
	})
	if err, ok := result.(error); ok {
		return err
	}
	return nil
}

// LoadState restores the full emulator state from a gob file.
// Thread-safe: runs on the background processor goroutine.
func (b *mcpBridge) LoadState(path string) error {
	result := b.exec(func() any {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close() //nolint: errcheck

		var state gb.EmulatorState
		if err := gob.NewDecoder(f).Decode(&state); err != nil {
			return err
		}

		b.mmu.LoadEmulatorState(state)
		return nil
	})
	if err, ok := result.(error); ok {
		return err
	}
	return nil
}

// exec sends a command to the background processor and waits for the result.
// Returns nil if the command cannot be queued within 5 seconds (prevents deadlock).
func (b *mcpBridge) exec(fn func() any) any {
	result := make(chan any, 1)
	cmd := mcpCmd{fn: fn, result: result}
	select {
	case b.cmds <- cmd:
		return <-result
	case <-time.After(5 * time.Second):
		log.Printf("bridge: exec timeout — command queue full")
		return nil
	}
}

// startProcessor launches a goroutine that processes exec commands.
// It runs commands one at a time between frame ticks so the frame loop
// is never blocked.
func (b *mcpBridge) startProcessor() {
	go func() {
		for {
			select {
			case cmd := <-b.cmds:
				cmd.result <- cmd.fn()
			case <-b.donec:
				return
			}
		}
	}()
}

func (b *mcpBridge) stopProcessor() {
	close(b.donec)
}

// runFrame advances the emulator by one frame (70224 T-cycles),
// injects the agent's latched input, and updates the snapshot.
func (b *mcpBridge) runFrame() {
	b.mmu.SetJoypadButtons(b.ReadLatched())

	target := b.cpu.Cycles + 70224
	for b.cpu.Cycles < target {
		if _, err := b.cpu.Step(); err != nil {
			return
		}
	}

	b.snap.write(b.ppu, b.cpu, b.timer)
}

// loadSavedState loads a gob-encoded EmulatorState directly into the bridge's
// emulator components. Used by --load-state to skip the intro on boot.
func loadSavedState(bridge *mcpBridge, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open state file: %w", err)
	}
	defer f.Close() //nolint: errcheck

	var state gb.EmulatorState
	if err := gob.NewDecoder(f).Decode(&state); err != nil {
		return fmt.Errorf("decode state: %w", err)
	}

	bridge.mmu.LoadEmulatorState(state)
	return nil
}
