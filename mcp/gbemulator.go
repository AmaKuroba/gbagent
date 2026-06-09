package mcp

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/AmaKuroba/gbagent/internal/gb"
)

// Button bit positions for the joypad byte.
const (
	btnRight  = 1 << 0
	btnLeft   = 1 << 1
	btnUp     = 1 << 2
	btnDown   = 1 << 3
	btnA      = 1 << 4
	btnB      = 1 << 5
	btnSelect = 1 << 6
	btnStart  = 1 << 7
)

// buttonMap maps uppercase button names to their bit position.
var buttonMap = map[string]byte{
	"A":      btnA,
	"B":      btnB,
	"START":  btnStart,
	"SELECT": btnSelect,
	"UP":     btnUp,
	"DOWN":   btnDown,
	"LEFT":   btnLeft,
	"RIGHT":  btnRight,
}

// GBEmulator wraps the internal/gb components into an EmulatorHandle.
type GBEmulator struct {
	mmu   *gb.MemoryBus
	cpu   *gb.Core
	ppu   *gb.PPUCore
	timer *gb.Timer
	apu   *gb.APU

	romPath string
}

// NewGBEmulator creates a new emulator, loads the ROM at romPath,
// and wires together all internal components (CPU, PPU, Timer, Joypad).
func NewGBEmulator(romPath string) (*GBEmulator, error) {
	// Read the ROM file.
	data, err := os.ReadFile(romPath)
	if err != nil {
		return nil, fmt.Errorf("read ROM: %w", err)
	}

	// Create the memory bus with no cartridge (LoadROM will create one).
	mmu := gb.NewMMU(nil)
	mmu.LoadROM(data)

	// Create components in dependency order.
	ppu := gb.NewPPU(mmu)
	timer := gb.NewTimer(mmu)
	joypad := gb.NewJoypad(mmu)
	apu := gb.NewAPU(mmu)
	cpu := gb.NewCore(mmu)
	mmu.SetCPU(cpu)

	// Attach peripherals to the memory bus.
	mmu.SetPPU(ppu)
	mmu.SetTimer(timer)
	mmu.SetJoypad(joypad)
	mmu.SetAPU(apu)

	return &GBEmulator{
		mmu:     mmu,
		cpu:     cpu,
		ppu:     ppu,
		timer:   timer,
		apu:     apu,
		romPath: romPath,
	}, nil
}

// runFrame runs one full frame (70224 T-cycles) of the emulator.
// Device stepping (PPU, timer, APU, DMA, serial) is handled internally
// by M-cycle stepping in cpu.Step().
func (e *GBEmulator) runFrame() {
	target := e.cpu.Cycles + 70224
	for e.cpu.Cycles < target {
		e.cpu.Step()
	}
}

// GetScreen returns the current PPU framebuffer.
func (e *GBEmulator) GetScreen() [160][144]byte {
	e.runFrame()
	return e.ppu.GetScreen()
}

// PressButton presses a named button for a brief moment (16 T-cycles).
func (e *GBEmulator) PressButton(button string) error {
	bit, ok := buttonMap[button]
	if !ok {
		return fmt.Errorf("unknown button: %s", button)
	}

	// Set the button.
	e.mmu.SetJoypadButtons(bit)

	// Run a few cycles so the game detects the button press.
	for i := 0; i < 4; i++ {
		e.cpu.Step()
	}

	// Release the button.
	e.mmu.SetJoypadButtons(0)

	// Run a few more cycles after release for debounce.
	for i := 0; i < 4; i++ {
		e.cpu.Step()
	}

	return nil
}

// ReadRAM reads a byte from Game Boy memory.
func (e *GBEmulator) ReadRAM(addr uint16) byte {
	return e.mmu.Read(addr)
}

// WriteRAM writes a byte to Game Boy memory.
func (e *GBEmulator) WriteRAM(addr uint16, val byte) {
	e.mmu.Write(addr, val)
}

// GetState returns a snapshot of the full emulator state.
func (e *GBEmulator) GetState() *EmulatorState {
	// Run a frame to advance the PPU to a known state.
	e.runFrame()

	cpuState := e.cpu.GetState()
	ppuState := e.ppu.GetState()
	timerState := e.timer.GetState()

	// Build cartridge info.
	var cartTitle, cartType string
	if e.mmu.Cartridge != nil {
		cartTitle = e.mmu.Cartridge.GetTitle()
		cartType = fmt.Sprintf("0x%02X", byte(e.mmu.Cartridge.GetType()))
	}

	return &EmulatorState{
		CPU: CPUState{
			AF:      cpuState.AF,
			BC:      cpuState.BC,
			DE:      cpuState.DE,
			HL:      cpuState.HL,
			SP:      cpuState.SP,
			PC:      cpuState.PC,
			IME:     cpuState.IME,
			Halted:  cpuState.Halted,
			Stopped: cpuState.Stopped,
			Cycles:  cpuState.Cycles,
		},
		PPU: PPUState{
			Mode:       ppuState.Mode,
			LY:         ppuState.LY,
			LCDC:       ppuState.LCDC,
			STAT:       ppuState.STAT,
			FrameCount: ppuState.FrameCount,
		},
		Cart: CartState{
			Title: cartTitle,
			Type:  cartType,
		},
		Timer: timerStateToMCP(timerState),
		APU: APUState{
			NR50: e.apu.ReadRegister(0xFF24),
			NR51: e.apu.ReadRegister(0xFF25),
			NR52: e.apu.ReadRegister(0xFF26),
		},
	}
}

// timerStateToMCP converts a gb.TimerState to an mcp.TimerState.
func timerStateToMCP(ts gb.TimerState) TimerState {
	var freq string
	switch ts.TAC & 0x03 {
	case 0:
		freq = "4096 Hz"
	case 1:
		freq = "262144 Hz"
	case 2:
		freq = "65536 Hz"
	case 3:
		freq = "16384 Hz"
	}
	enabled := ts.TAC&0x04 != 0

	return TimerState{
		DIV:    ts.DIV,
		TIMA:   ts.TIMA,
		TMA:    ts.TMA,
		TAC:    ts.TAC,
		Freq:   freq,
		Enable: enabled,
	}
}

// SaveState serialises the full emulator state to a file using gob encoding.
func (e *GBEmulator) SaveState(path string) error {
	// Build a serialisable snapshot.
	state := saveState{
		CPU:   e.cpu.GetState(),
		PPU:   e.ppu.GetState(),
		Timer: e.timer.GetState(),
		// Save cartridge RAM for battery-backed saves.
		BatteryRAM: e.saveBatteryRAM(),
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create state file: %w", err)
	}
	defer f.Close()

	if err := gob.NewEncoder(f).Encode(state); err != nil {
		return fmt.Errorf("encode state: %w", err)
	}
	return nil
}

// LoadState deserialises the emulator state from a file using gob decoding.
func (e *GBEmulator) LoadState(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open state file: %w", err)
	}
	defer f.Close()

	var state saveState
	if err := gob.NewDecoder(f).Decode(&state); err != nil {
		return fmt.Errorf("decode state: %w", err)
	}

	// Restore battery RAM if available.
	e.loadBatteryRAM(state.BatteryRAM)

	return nil
}

// saveState is a serialisable snapshot used by SaveState/LoadState.
type saveState struct {
	CPU        gb.CPUState
	PPU        gb.PPUState
	Timer      gb.TimerState
	BatteryRAM []byte
}

// saveBatteryRAM returns cartridge battery-backed RAM if present.
func (e *GBEmulator) saveBatteryRAM() []byte {
	if e.mmu.Cartridge != nil && e.mmu.Cartridge.HasBattery() {
		return e.mmu.Cartridge.SaveRAM()
	}
	return nil
}

// loadBatteryRAM restores cartridge battery-backed RAM.
func (e *GBEmulator) loadBatteryRAM(data []byte) {
	if len(data) > 0 && e.mmu.Cartridge != nil {
		e.mmu.Cartridge.LoadRAM(data)
	}
}
