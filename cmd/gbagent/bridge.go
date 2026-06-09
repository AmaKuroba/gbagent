package main

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/AmaKuroba/gbagent/internal/gb"
	"github.com/AmaKuroba/gbagent/mcp"
)

// mcpCmd is dispatched to the emulation goroutine for race-free execution.
type mcpCmd struct {
	fn     func() any
	result chan any
}

// mcpBridge adapts the live emulator instance to the mcp.EmulatorHandle
// interface. All methods send commands over a channel and are processed
// by the emulation goroutine between frames — no locks needed.
type mcpBridge struct {
	mmu     *gb.MemoryBus
	cpu     *gb.Core
	ppu     *gb.PPUCore
	timer   *gb.Timer
	apu     *gb.APU
	cart    gb.Cartridge
	romPath string

	cmds chan mcpCmd
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
	}
}

// exec sends a command to the emulation goroutine and waits for the result.
// Must NOT be called from the emulation goroutine.
func (b *mcpBridge) exec(fn func() any) any {
	result := make(chan any, 1)
	b.cmds <- mcpCmd{fn: fn, result: result}
	return <-result
}

// processPending drains all queued commands on the emulation goroutine.
// Must be called FROM the emulation goroutine only.
func (b *mcpBridge) processPending() {
	for {
		select {
		case cmd := <-b.cmds:
			cmd.result <- cmd.fn()
		default:
			return
		}
	}
}

// runFrame advances the emulator by one frame (70224 T-cycles).
// Device stepping (PPU, timer, APU, DMA, serial) is handled internally
// by M-cycle stepping in cpu.Step().
func (b *mcpBridge) runFrame() {
	target := b.cpu.Cycles + 70224
	for b.cpu.Cycles < target {
		b.cpu.Step()
	}
}

// --- EmulatorHandle implementation ---

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

func (b *mcpBridge) GetScreen() [160][144]byte {
	result := b.exec(func() any {
		b.runFrame()
		return b.ppu.GetScreen()
	})
	return result.([160][144]byte)
}

func (b *mcpBridge) PressButton(button string) error {
	bit, ok := btnBits[button]
	if !ok {
		return fmt.Errorf("unknown button: %s", button)
	}

	err := b.exec(func() any {
		b.mmu.SetJoypadButtons(bit)
		for i := 0; i < 4; i++ {
			b.cpu.Step()
		}
		b.mmu.SetJoypadButtons(0)
		for i := 0; i < 4; i++ {
			b.cpu.Step()
		}
		return nil
	})
	if err != nil {
		return err.(error)
	}
	return nil
}

func (b *mcpBridge) ReadRAM(addr uint16) byte {
	result := b.exec(func() any {
		return b.mmu.Read(addr)
	})
	return result.(byte)
}

func (b *mcpBridge) WriteRAM(addr uint16, val byte) {
	b.exec(func() any {
		b.mmu.Write(addr, val)
		return nil
	})
}

func (b *mcpBridge) GetState() *mcp.EmulatorState {
	result := b.exec(func() any {
		b.runFrame()

		cs := b.cpu.GetState()
		ps := b.ppu.GetState()
		ts := b.timer.GetState()

		var cartTitle, cartType string
		if b.cart != nil {
			cartTitle = b.cart.GetTitle()
			cartType = fmt.Sprintf("0x%02X", b.cart.GetType())
		}

		return &mcp.EmulatorState{
			CPU: mcp.CPUState{
				AF: cs.AF, BC: cs.BC, DE: cs.DE, HL: cs.HL,
				SP: cs.SP, PC: cs.PC, IME: cs.IME,
				Halted: cs.Halted, Stopped: cs.Stopped, Cycles: cs.Cycles,
			},
			PPU: mcp.PPUState{
				Mode: ps.Mode, LY: ps.LY, LCDC: ps.LCDC,
				STAT: ps.STAT, FrameCount: ps.FrameCount,
			},
			Cart: mcp.CartState{Title: cartTitle, Type: cartType},
			Timer: mcp.TimerState{
				DIV: ts.DIV, TIMA: ts.TIMA, TMA: ts.TMA, TAC: ts.TAC,
				Freq: timerFreqStr(ts.TAC), Enable: ts.TAC&0x04 != 0,
			},
			APU: mcp.APUState{
				NR50: b.apu.ReadRegister(0xFF24),
				NR51: b.apu.ReadRegister(0xFF25),
				NR52: b.apu.ReadRegister(0xFF26),
			},
		}
	})
	return result.(*mcp.EmulatorState)
}

func (b *mcpBridge) SaveState(path string) error {
	err := b.exec(func() any {
		state := saveFile{
			CPU:   b.cpu.GetState(),
			PPU:   b.ppu.GetState(),
			Timer: b.timer.GetState(),
		}
		if b.cart != nil && b.cart.HasBattery() {
			if ram := b.cart.SaveRAM(); ram != nil {
				state.BatteryRAM = ram
			}
		}
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		defer f.Close()
		return gob.NewEncoder(f).Encode(state)
	})
	if err != nil {
		return err.(error)
	}
	return nil
}

func (b *mcpBridge) LoadState(path string) error {
	err := b.exec(func() any {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		var state saveFile
		if err := gob.NewDecoder(f).Decode(&state); err != nil {
			return err
		}

		if len(state.BatteryRAM) > 0 && b.cart != nil && b.cart.HasBattery() {
			b.cart.LoadRAM(state.BatteryRAM)
		}
		return nil
	})
	if err != nil {
		return err.(error)
	}
	return nil
}

// saveFile is a gob-serialisable snapshot for SaveState/LoadState.
type saveFile struct {
	CPU        gb.CPUState
	PPU        gb.PPUState
	Timer      gb.TimerState
	BatteryRAM []byte
}

func timerFreqStr(tac byte) string {
	switch tac & 0x03 {
	case 0:
		return "4096 Hz"
	case 1:
		return "262144 Hz"
	case 2:
		return "65536 Hz"
	case 3:
		return "16384 Hz"
	}
	return "?"
}

// init registers the saveFile type with gob so encoding/decoding works.
func init() {
	gob.Register(saveFile{})
}
