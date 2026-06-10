package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"

	"github.com/AmaKuroba/gbagent/dashboard"
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

	hub   *dashboard.Hub
	cmds  chan mcpCmd
	snap  bridgeSnapshot
	donec chan struct{}

	// Agent button latch: the RPC agent writes bits via press_button /
	// release_button; the frame loop reads them every tick and injects
	// into the MMU. Mutex is fine — writes are rare, reads are fast.
	latchMu sync.Mutex
	latched byte

	// Dashboard joypad state, read by frame loop each tick.
	joypadState func() byte

	// Takeover mode: human plays via dashboard, agent inputs are ignored.
	takeover atomic.Bool

	// Last human action (for training to read during takeover).
	lastActMu   sync.Mutex
	lastActDpad byte
	lastActBtn  byte
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

func (b *mcpBridge) SetTakeover(v bool) {
	b.takeover.Store(v)
	if v {
		b.ClearAllLatched()
	}
}

func (b *mcpBridge) IsTakeover() bool {
	return b.takeover.Load()
}

func (b *mcpBridge) GetLastAction() (dpad, btn byte) {
	b.lastActMu.Lock()
	dpad = b.lastActDpad
	btn = b.lastActBtn
	b.lastActMu.Unlock()
	return
}

// decodeJoypad extracts dpad index (0-4) and btn index (0-4) from a joypad byte.
func decodeJoypad(bits byte) (dpad, btn byte) {
	switch {
	case bits&btnBits["RIGHT"] != 0:
		dpad = 4
	case bits&btnBits["LEFT"] != 0:
		dpad = 3
	case bits&btnBits["UP"] != 0:
		dpad = 1
	case bits&btnBits["DOWN"] != 0:
		dpad = 2
	}
	switch {
	case bits&btnBits["A"] != 0:
		btn = 1
	case bits&btnBits["B"] != 0:
		btn = 2
	case bits&btnBits["START"] != 0:
		btn = 3
	case bits&btnBits["SELECT"] != 0:
		btn = 4
	}
	return
}

// broadcastAction sends an action event to the dashboard's WebSocket clients.
func (b *mcpBridge) broadcastAction(tool, args string) {
	if b.hub == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"action": tool, "args": args})
	b.hub.BroadcastText(data)
}

func newBridge(mmu *gb.MemoryBus, cpu *gb.Core, ppu *gb.PPUCore, timer *gb.Timer, apu *gb.APU, cart gb.Cartridge, romPath string, hub *dashboard.Hub, joypadState func() byte) *mcpBridge {
	return &mcpBridge{
		mmu:          mmu,
		cpu:          cpu,
		ppu:          ppu,
		timer:        timer,
		apu:          apu,
		cart:         cart,
		romPath:      romPath,
		hub:          hub,
		cmds:         make(chan mcpCmd, 64),
		donec:        make(chan struct{}),
		joypadState:  joypadState,
	}
}

// SaveState persists the full emulator state to a gob file.
// Thread-safe: runs on the background processor goroutine.
func (b *mcpBridge) SaveState(path string) error {
	b.broadcastAction("save_state", path)
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
	b.broadcastAction("load_state", path)
	result := b.exec(func() any {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

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
func (b *mcpBridge) exec(fn func() any) any {
	result := make(chan any, 1)
	b.cmds <- mcpCmd{fn: fn, result: result}
	return <-result
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
// injects joypad state (agent or human depending on takeover),
// and updates the snapshot.
func (b *mcpBridge) runFrame() {
	if b.takeover.Load() {
		// Human controls via dashboard exclusively.
		var bits byte
		if b.joypadState != nil {
			bits = b.joypadState()
		}
		b.mmu.SetJoypadButtons(bits)
		b.lastActMu.Lock()
		b.lastActDpad, b.lastActBtn = decodeJoypad(bits)
		b.lastActMu.Unlock()
	} else {
		bits := b.ReadLatched()
		if b.joypadState != nil {
			bits |= b.joypadState()
		}
		b.mmu.SetJoypadButtons(bits)
	}

	target := b.cpu.Cycles + 70224
	for b.cpu.Cycles < target {
		if _, err := b.cpu.Step(); err != nil {
			return
		}
	}

	b.snap.write(b.ppu, b.cpu, b.timer)
}

var btnBits = map[string]byte{
	"A":      1 << 4,
	"B":      1 << 5,
	"SELECT": 1 << 6,
	"START":  1 << 7,
	"RIGHT":  1 << 0,
	"LEFT":   1 << 1,
	"UP":     1 << 2,
	"DOWN":   1 << 3,
}

// loadSavedState loads a gob-encoded EmulatorState directly into the bridge's
// emulator components. Used by --load-state to skip the intro on boot.
func loadSavedState(bridge *mcpBridge, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open state file: %w", err)
	}
	defer f.Close()

	var state gb.EmulatorState
	if err := gob.NewDecoder(f).Decode(&state); err != nil {
		return fmt.Errorf("decode state: %w", err)
	}

	bridge.mmu.LoadEmulatorState(state)
	return nil
}
