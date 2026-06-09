package mcp

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testEmulator is a minimal stub for testing.
type testEmulator struct{}

func (e *testEmulator) GetScreen() [160][144]byte { return [160][144]byte{} }

func (e *testEmulator) PressButton(button string) error { return nil }

func (e *testEmulator) ReadRAM(addr uint16) byte {
	return byte(addr & 0xFF)
}

func (e *testEmulator) WriteRAM(addr uint16, val byte) {
}

func (e *testEmulator) GetState() *EmulatorState {
	return &EmulatorState{
		CPU: CPUState{
			AF: 0x01B0, BC: 0x0013, DE: 0x00D8, HL: 0x014D,
			SP: 0xFFFE, PC: 0x0100,
			IME: false, Halted: false, Cycles: 42,
		},
		PPU: PPUState{
			Mode: 2, LY: 0x00, LCDC: 0x91, STAT: 0x82, FrameCount: 0,
		},
		Cart: CartState{
			Title: "TEST",
			Type:  "ROMOnly",
		},
		Timer: TimerState{
			DIV: 0x00, TIMA: 0x00, TMA: 0x00, TAC: 0xF8,
			Freq: "4096 Hz", Enable: false,
		},
	}
}

func (e *testEmulator) SaveState(path string) error { return nil }

func (e *testEmulator) LoadState(path string) error { return nil }

func TestGetState_ReturnsJSON(t *testing.T) {
	srv := NewServer(&testEmulator{})

	req := mcp.CallToolRequest{}
	req.Params.Name = "get_state"

	result, err := srv.handleGetState(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Result is structured, response should contain JSON with "af".
	text, ok := result.Content[0].(mcp.TextContent)
	require.True(t, ok, "expected TextContent")
	require.Contains(t, text.Text, "af")

	var state EmulatorState
	err = json.Unmarshal([]byte(text.Text), &state)
	require.NoError(t, err, "response should be valid JSON")

	// Verify CPU registers.
	assert.Equal(t, uint16(0x01B0), state.CPU.AF)
	assert.Equal(t, uint16(0x0100), state.CPU.PC)
	assert.Equal(t, uint16(0xFFFE), state.CPU.SP)
	assert.False(t, state.CPU.IME)

	// Verify PPU state.
	assert.Equal(t, 2, state.PPU.Mode)
	assert.Equal(t, byte(0x91), state.PPU.LCDC)
	assert.Equal(t, byte(0x82), state.PPU.STAT)

	// Verify cart state.
	assert.Equal(t, "TEST", state.Cart.Title)
	assert.Equal(t, "ROMOnly", state.Cart.Type)

	// Verify timer state.
	assert.Equal(t, byte(0xF8), state.Timer.TAC)
	assert.False(t, state.Timer.Enable)
}

func TestGetState_AllFieldsPresent(t *testing.T) {
	srv := NewServer(&testEmulator{})

	req := mcp.CallToolRequest{}
	req.Params.Name = "get_state"

	result, err := srv.handleGetState(context.Background(), req)
	require.NoError(t, err)

	text, ok := result.Content[0].(mcp.TextContent)
	require.True(t, ok)

	var raw map[string]json.RawMessage
	err = json.Unmarshal([]byte(text.Text), &raw)
	require.NoError(t, err)

	// Top-level sections.
	assert.Contains(t, raw, "cpu")
	assert.Contains(t, raw, "ppu")
	assert.Contains(t, raw, "cart")
	assert.Contains(t, raw, "timer")

	// CPU sub-fields.
	var cpu map[string]json.RawMessage
	json.Unmarshal(raw["cpu"], &cpu)
	assert.Contains(t, cpu, "af")
	assert.Contains(t, cpu, "bc")
	assert.Contains(t, cpu, "de")
	assert.Contains(t, cpu, "hl")
	assert.Contains(t, cpu, "sp")
	assert.Contains(t, cpu, "pc")
	assert.Contains(t, cpu, "ime")
	assert.Contains(t, cpu, "halted")
	assert.Contains(t, cpu, "stopped")
	assert.Contains(t, cpu, "cycles")

	// PPU sub-fields.
	var ppu map[string]json.RawMessage
	json.Unmarshal(raw["ppu"], &ppu)
	assert.Contains(t, ppu, "mode")
	assert.Contains(t, ppu, "ly")
	assert.Contains(t, ppu, "lcdc")
	assert.Contains(t, ppu, "stat")
	assert.Contains(t, ppu, "frame_count")

	// Cart sub-fields.
	var cart map[string]json.RawMessage
	json.Unmarshal(raw["cart"], &cart)
	assert.Contains(t, cart, "title")
	assert.Contains(t, cart, "type")

	// Timer sub-fields.
	var timer map[string]json.RawMessage
	json.Unmarshal(raw["timer"], &timer)
	assert.Contains(t, timer, "div")
	assert.Contains(t, timer, "tima")
	assert.Contains(t, timer, "tma")
	assert.Contains(t, timer, "tac")
	assert.Contains(t, timer, "freq")
	assert.Contains(t, timer, "enable")
}
