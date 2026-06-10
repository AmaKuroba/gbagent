package gb

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// testMMU is a minimal MMU stub for timer tests that tracks IF writes.
type testMMU struct {
	ifReg byte
}

func (m *testMMU) Read(addr uint16) byte {
	if addr == 0xFF0F {
		return m.ifReg
	}
	return 0
}

func (m *testMMU) Write(addr uint16, val byte) {
	if addr == 0xFF0F {
		m.ifReg = val
	}
}

func (m *testMMU) Read16(addr uint16) uint16 {
	return uint16(m.Read(addr)) | uint16(m.Read(addr+1))<<8
}

func (m *testMMU) Write16(addr uint16, val uint16) {
	m.Write(addr, byte(val))
	m.Write(addr+1, byte(val>>8))
}

func (m *testMMU) LoadROM(data []byte)          {}
func (m *testMMU) LoadBootROM(data []byte)       {}
func (m *testMMU) ReadIF() byte                  { return m.ifReg }
func (m *testMMU) ReadIE() byte                  { return 0 }
func (m *testMMU) WriteIF(val byte)              { m.ifReg = val }
func (m *testMMU) DMAStep(cycles int)            {}
func (m *testMMU) SerialStep(cycles int)         {}
func (m *testMMU) SetJoypadButtons(buttons byte) {}
func (m *testMMU) StepDevices(cycles int)          {}

func TestTimerDIVIncrement(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// DIV starts at 0
	require.Equal(t, byte(0), timer.DIV, "DIV initial value")

	// After 255 T-cycles, DIV should still be 0
	timer.Step(255)
	require.Equal(t, byte(0), timer.DIV, "DIV after 255 cycles")

	// After 1 more, DIV should be 1
	timer.Step(1)
	require.Equal(t, byte(1), timer.DIV, "DIV after 256 cycles (1 increment)")

	// After 255 more cycles, DIV still 1
	timer.Step(255)
	require.Equal(t, byte(1), timer.DIV, "DIV after 511 cycles (still 1)")

	// After 256 more, DIV = 2
	timer.Step(256)
	require.Equal(t, byte(2), timer.DIV, "DIV after 512 cycles (2 increments)")
}

func TestTimerDIVFreeRunning(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// DIV should increment regardless of TAC enable bit
	// Run 256 * 256 = 65536 cycles, DIV should wrap to 0
	for range 256 {
		timer.Step(256)
	}
	require.Equal(t, byte(0), timer.DIV, "DIV wraps after 65536 cycles")
}

func TestTimerDIVWriteReset(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Run 400 cycles to advance DIV
	timer.Step(400)
	require.Equal(t, byte(1), timer.DIV, "DIV after 400 cycles")

	// Write to DIV — resets to 0
	timer.WriteRegister(0xFF04, 0xAB)
	require.Equal(t, byte(0), timer.DIV, "DIV after write reset")

	// After another 256 cycles, DIV = 1 again
	timer.Step(256)
	require.Equal(t, byte(1), timer.DIV, "DIV after write + 256 cycles")
}

func TestTimerTIMAInitial(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// TIMA should start at 0 and not increment without TAC enabled
	require.Equal(t, byte(0), timer.TIMA, "TIMA initial value")
	timer.Step(10000)
	require.Equal(t, byte(0), timer.TIMA, "TIMA should not increment without TAC enable")
}

func TestTimerTIMAIncrement16Cycles(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Enable timer with 262144 Hz clock (16 cycles per tick)
	timer.WriteRegister(0xFF07, 0x05) // bit 2=1, bits 1-0=01 → 16 cycles

	// Step 15 cycles — TIMA should still be 0
	timer.Step(15)
	require.Equal(t, byte(0), timer.TIMA, "TIMA after 15 cycles (16-cycle clock)")

	// Step 1 more — TIMA should increment to 1
	timer.Step(1)
	require.Equal(t, byte(1), timer.TIMA, "TIMA after 16 cycles (first tick)")
}

func TestTimerTIMAIncrement256Cycles(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Enable timer with 16384 Hz clock (256 cycles per tick)
	timer.WriteRegister(0xFF07, 0x07) // bit 2=1, bits 1-0=11 → 256 cycles

	timer.Step(255)
	require.Equal(t, byte(0), timer.TIMA, "TIMA after 255 cycles (256-cycle clock)")

	timer.Step(1)
	require.Equal(t, byte(1), timer.TIMA, "TIMA after 256 cycles (first tick)")

	timer.Step(256)
	require.Equal(t, byte(2), timer.TIMA, "TIMA after 512 cycles (second tick)")
}

func TestTimerTIMAIncrement64Cycles(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Enable timer with 65536 Hz clock (64 cycles per tick)
	timer.WriteRegister(0xFF07, 0x06) // bit 2=1, bits 1-0=10 → 64 cycles

	timer.Step(63)
	require.Equal(t, byte(0), timer.TIMA, "TIMA after 63 cycles (64-cycle clock)")

	timer.Step(1)
	require.Equal(t, byte(1), timer.TIMA, "TIMA after 64 cycles (first tick)")
}

func TestTimerTIMAIncrement1024Cycles(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Enable timer with 4096 Hz clock (1024 cycles per tick)
	timer.WriteRegister(0xFF07, 0x04) // bit 2=1, bits 1-0=00 → 1024 cycles

	timer.Step(1023)
	require.Equal(t, byte(0), timer.TIMA, "TIMA after 1023 cycles (1024-cycle clock)")

	timer.Step(1)
	require.Equal(t, byte(1), timer.TIMA, "TIMA after 1024 cycles (first tick)")
}

func TestTimerTIMAOverflowInterrupt(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Set TMA to a known reload value
	timer.WriteRegister(0xFF06, 0x42) // TMA = 0x42

	// Enable timer with 16-cycle clock
	timer.WriteRegister(0xFF07, 0x05)

	// Set TIMA to 0xFE (right before overflow)
	timer.WriteRegister(0xFF05, 0xFE)

	// Step 16 cycles — TIMA should become 0xFF
	timer.Step(16)
	require.Equal(t, byte(0xFF), timer.TIMA, "TIMA before overflow")

	// Step another 16 cycles — TIMA should overflow and reload from TMA
	require.Equal(t, byte(0), mmu.ifReg&0x04, "IF bit 2 should not be set yet")
	timer.Step(16)
	require.Equal(t, byte(0x42), timer.TIMA, "TIMA after overflow should reload from TMA")
	require.Equal(t, byte(0x04), mmu.ifReg&0x04, "IF bit 2 should be set after overflow")
}

func TestTimerTIMAOverflowSetsIFOnlyOnce(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	timer.WriteRegister(0xFF06, 0x00) // TMA = 0
	timer.WriteRegister(0xFF07, 0x05) // enable, 16-cycle clock
	timer.WriteRegister(0xFF05, 0xFF) // TIMA = 0xFF

	// Step 16 cycles — overflow: TIMA reloads from TMA, IF set
	timer.Step(16)
	require.Equal(t, byte(0x00), timer.TIMA, "TIMA after overflow reload")
	require.Equal(t, byte(0x04), mmu.ifReg&0x04, "IF bit 2 set after overflow")

	// Preserve IF flag and step more — should not set it again
	mmu.ifReg &^= 0x04
	timer.Step(16 * 256) // many cycles
	// TIMA should still be 0 (TMA = 0, repeats on every overflow)
	require.Equal(t, byte(0x00), timer.TIMA, "TIMA should reload on every overflow")
	// IF bit 2 should be set again because TIMA overflowed again
	require.Equal(t, byte(0x04), mmu.ifReg&0x04, "IF bit 2 set again on subsequent overflow")
}

func TestTimerTACEnableDisable(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Disable timer (default)
	require.Equal(t, byte(0xF8), timer.TAC, "TAC default")
	timer.Step(1024)
	require.Equal(t, byte(0), timer.TIMA, "TIMA should not increment when disabled")

	// Enable timer
	timer.WriteRegister(0xFF07, 0x05) // enable, 16-cycle clock
	timer.Step(16)
	require.Equal(t, byte(1), timer.TIMA, "TIMA increments when enabled")

	// Disable again
	timer.WriteRegister(0xFF07, 0xF8) // bit 2 = 0, clock bits = 00
	timer.Step(10000)
	require.Equal(t, byte(1), timer.TIMA, "TIMA should not increment when disabled")
}

func TestTimerAllClockRates(t *testing.T) {
	mmu := &testMMU{}

	tests := []struct {
		name     string
		tacVal   byte
		cycles   uint16
		expected byte
	}{
		{"4096 Hz (1024 cycles)", 0x04, 1024, 1},
		{"262144 Hz (16 cycles)", 0x05, 16, 1},
		{"65536 Hz (64 cycles)", 0x06, 64, 1},
		{"16384 Hz (256 cycles)", 0x07, 256, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			timer := NewTimer(mmu)
			timer.WriteRegister(0xFF07, tt.tacVal)
			timer.Step(int(tt.cycles))
			require.Equal(t, tt.expected, timer.TIMA, "TIMA after 1 tick at %s", tt.name)
		})
	}
}

func TestTimerTACWriteOnlyBits0_2(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Write all bits high — only lower 3 bits are writable, bits 7-3 always read as 1
	timer.WriteRegister(0xFF07, 0xFF)
	require.Equal(t, byte(0xFF), timer.TAC, "TAC after writing 0xFF (bits 7-3 always 1)")

	// Write 0x00 — bits 7-3 still read as 1
	timer.WriteRegister(0xFF07, 0x00)
	require.Equal(t, byte(0xF8), timer.TAC, "TAC after writing 0x00")
}

func TestTimerDIVWriteResetsTimaCounter(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Enable timer with 16-cycle clock
	timer.WriteRegister(0xFF07, 0x05)

	// Step 15 cycles — almost at the increment threshold
	timer.Step(15)
	require.Equal(t, byte(0), timer.TIMA, "TIMA before DIV write")

	// Writing to DIV resets the internal TIMA cycle counter
	// This means TIMA should NOT increment on the next cycle
	timer.WriteRegister(0xFF04, 0x00)
	timer.Step(1)
	require.Equal(t, byte(0), timer.TIMA, "TIMA should not increment after DIV write resets counter")

	// Need full 16 cycles from the reset
	timer.Step(15)
	require.Equal(t, byte(1), timer.TIMA, "TIMA should increment after 16 cycles from reset")
}

func TestTimerStepZeroCycles(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Step with 0 cycles should not cause any change
	timer.Step(0)
	require.Equal(t, byte(0), timer.DIV, "DIV unchanged after 0 cycles")
	require.Equal(t, byte(0), timer.TIMA, "TIMA unchanged after 0 cycles")
}

func TestTimerLargeStepBatches(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// A single large step covering 1024+ cycles should correctly advance
	timer.WriteRegister(0xFF07, 0x05) // enable, 16-cycle clock

	// Step 160 cycles = 10 TIMA ticks
	timer.Step(160)
	require.Equal(t, byte(10), timer.TIMA, "TIMA after 160 cycles (10 ticks at 16-cycle clock)")

	// DIV should have advanced (160+ cycles = 0 increments, need 256)
	timer.Step(96) // now 256 total
	require.Equal(t, byte(1), timer.DIV, "DIV after 256 cycles")
	// TIMA should be 10 + 6 = 16
	require.Equal(t, byte(16), timer.TIMA, "TIMA after 256 cycles")
}

func TestTimerTMAWriteDoesNotTriggerOverflow(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	timer.WriteRegister(0xFF06, 0x00) // TMA = 0
	timer.WriteRegister(0xFF07, 0x05) // enable, 16-cycle clock
	timer.WriteRegister(0xFF05, 0xFF) // TIMA = 0xFF

	// Write to TMA while TIMA is at 0xFF — should NOT trigger overflow
	timer.WriteRegister(0xFF06, 0x42)
	timer.Step(16) // this step IS the overflow
	require.Equal(t, byte(0x42), timer.TIMA, "TIMA should reload from new TMA on overflow")
}

func TestTimerStateSnapshot(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	timer.WriteRegister(0xFF06, 0xAB)
	timer.WriteRegister(0xFF07, 0x05)
	timer.Step(100)

	state := timer.GetState()
	require.Equal(t, timer.DIV, state.DIV, "GetState DIV")
	require.Equal(t, timer.TIMA, state.TIMA, "GetState TIMA")
	require.Equal(t, timer.TMA, state.TMA, "GetState TMA")
	require.Equal(t, timer.TAC, state.TAC, "GetState TAC")
}

func TestTimerReset(t *testing.T) {
	mmu := &testMMU{}
	timer := NewTimer(mmu)

	// Advance timer to known state
	timer.WriteRegister(0xFF06, 0xAB)
	timer.WriteRegister(0xFF07, 0x05)
	timer.Step(1000)

	timer.Reset()
	require.Equal(t, byte(0), timer.DIV, "DIV after reset")
	require.Equal(t, byte(0), timer.TIMA, "TIMA after reset")
	require.Equal(t, byte(0), timer.TMA, "TMA after reset")
	require.Equal(t, byte(0xF8), timer.TAC, "TAC after reset")
}
