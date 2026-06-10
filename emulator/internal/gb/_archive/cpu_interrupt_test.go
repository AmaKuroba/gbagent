package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestCore creates a Core with a fresh MemoryBus for testing.
func newTestCore() *Core {
	bus := NewMemoryBus()
	return NewCore(bus)
}

// writeCode places instruction bytes into memory at the given address.
func writeCode(mmu *MemoryBus, addr uint16, data ...byte) {
	for i, b := range data {
		mmu.Write(addr+uint16(i), b)
	}
}

func stepN(t *testing.T, c *Core, n int) {
	for i := 0; i < n; i++ {
		_, err := c.Step()
		require.NoError(t, err)
	}
}

// ---------------------------------------------------------------------------
// IME / IE / IF register tests
// ---------------------------------------------------------------------------

func TestIMEStartsCleared(t *testing.T) {
	c := newTestCore()
	state := c.GetState()
	assert.False(t, state.IME, "IME should start cleared after reset")
}

func TestEI_SetsIMEDelayed(t *testing.T) {
	c := newTestCore()
	// EI (0xFB) followed by NOP (0x00)
	writeCode(c.MMU, 0x0100, 0xFB, 0x00)
	c.State.PC = 0x0100

	// Step 1: EI — IME should NOT yet be set
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.State.IME, "IME should NOT be set during EI itself")

	// Step 2: NOP — IME should now be set (EI takes 1 instruction delay)
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.State.IME, "IME should be set after the instruction following EI")
}

func TestDI_ClearsIMEImmediately(t *testing.T) {
	c := newTestCore()
	// Set IME enabled
	c.State.IME = true
	c.State.IMEEnablePending = false

	// DI (0xF3) clears IME immediately
	writeCode(c.MMU, 0x0100, 0xF3)
	c.State.PC = 0x0100

	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.State.IME, "DI should clear IME immediately during its execution")
	assert.False(t, c.State.IMEEnablePending, "DI should clear any pending EI enable")
}

func TestIF_IE_RegisterReadWrite(t *testing.T) {
	c := newTestCore()

	// Write IF via MMU
	c.MMU.Write(0xFF0F, 0x01) // VBlank flag
	assert.Equal(t, byte(0x01), c.MMU.Read(0xFF0F), "IF should be readable")

	// Write IE via MMU
	c.MMU.Write(0xFFFF, 0x01)
	assert.Equal(t, byte(0x01), c.MMU.Read(0xFFFF), "IE should be readable")

	// Clear IF
	c.MMU.Write(0xFF0F, 0x00)
	assert.Equal(t, byte(0x00), c.MMU.Read(0xFF0F), "IF should be writable")
}

// ---------------------------------------------------------------------------
// VBlank interrupt servicing
// ---------------------------------------------------------------------------

func TestVBlankInterrupt_ServicesWhenEnabled(t *testing.T) {
	c := newTestCore()
	// Set up: IE = 0x01 (VBlank enabled), IF = 0x01 (VBlank requested)
	// EI + NOP so IME becomes active
	// Then execute NOPs — the interrupt should fire on the next instruction
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending

	// EI (0xFB), NOP (0x00), NOP (0x00)
	writeCode(c.MMU, 0x0100, 0xFB, 0x00, 0x00)
	c.State.PC = 0x0100
	c.State.SP = 0xFFFE

	// Step 1: EI
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.State.IME)

	// Step 2: NOP — enables IME
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.State.IME)

	// Step 3: NOP — interrupt should fire (IME=1, IE & IF != 0)
	pcBefore := c.State.PC
	spBefore := c.State.SP
	_, err = c.Step()
	require.NoError(t, err)

	// Should have jumped to VBlank vector 0x40
	assert.Equal(t, uint16(0x0040), c.State.PC, "Should jump to VBlank vector")
	// IME should be cleared
	assert.False(t, c.State.IME, "IME should be cleared during interrupt servicing")
	// SP should have decremented by 2 (PC pushed)
	assert.Equal(t, spBefore-2, c.State.SP, "SP should decrement by 2")
	// IF bit 0 should be cleared
	assert.Equal(t, byte(0x00), c.MMU.Read(0xFF0F)&0x01, "VBlank IF flag should be cleared")
	// PC should be pushed onto stack
	pushedPC := c.MMU.Read16(c.State.SP)
	assert.Equal(t, pcBefore, pushedPC, "PC should be pushed onto stack")
}

func TestVBlankInterrupt_DoesNotFireWhenIMECleared(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending
	c.State.IME = false

	writeCode(c.MMU, 0x0100, 0x00, 0x00) // NOP, NOP
	c.State.PC = 0x0100

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint16(0x0101), c.State.PC, "PC should advance normally, no interrupt")
	assert.True(t, c.MMU.Read(0xFF0F)&0x01 == 0x01, "IF flag should remain set")
}

func TestVBlankInterrupt_DoesNotFireWhenIFCleared(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x00) // IF: nothing pending
	c.State.IME = true

	writeCode(c.MMU, 0x0100, 0x00) // NOP
	c.State.PC = 0x0100

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint16(0x0101), c.State.PC, "PC should advance normally, no interrupt")
}

func TestVBlankInterrupt_HighestPriority(t *testing.T) {
	c := newTestCore()
	// All interrupts pending but VBlank (bit 0) has highest priority
	c.MMU.Write(0xFFFF, 0x1F) // All 5 interrupts enabled
	c.MMU.Write(0xFF0F, 0x1F) // All 5 interrupts pending
	c.State.IME = true
	c.State.SP = 0xFFFE

	writeCode(c.MMU, 0x0100, 0x00) // NOP
	c.State.PC = 0x0100

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint16(0x0040), c.State.PC, "Should jump to VBlank vector (highest priority)")
}

// ---------------------------------------------------------------------------
// HALT test
// ---------------------------------------------------------------------------

func TestHALT_HaltsCPU(t *testing.T) {
	c := newTestCore()
	// HALT (0x76) should stop execution
	writeCode(c.MMU, 0x0100, 0x76, 0x00, 0x00)
	c.State.PC = 0x0100

	// Step HALT
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.State.Halted, "CPU should be halted after HALT instruction")

	// Next step should not advance PC while halted
	stateBefore := c.GetState()
	_, err = c.Step()
	require.NoError(t, err)
	assert.Equal(t, stateBefore.PC, c.State.PC, "PC should not advance while halted")
	assert.Equal(t, stateBefore.Cycles+4, c.State.Cycles, "Cycles should still increment while halted")
}

func TestHALT_WakesOnInterrupt(t *testing.T) {
	c := newTestCore()
	// HALT with interrupts enabled — will wake on VBlank
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.State.IME = true
	c.State.SP = 0xFFFE

	writeCode(c.MMU, 0x0100, 0x76) // HALT
	c.State.PC = 0x0100

	// Step: HALT
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.State.Halted, "CPU should be halted")

	// Set IF
	c.MMU.Write(0xFF0F, 0x01) // VBlank pending

	// Next step should wake from halt AND service the interrupt
	_, err = c.Step()
	require.NoError(t, err)
	assert.False(t, c.State.Halted, "CPU should wake from halt")
	assert.Equal(t, uint16(0x0040), c.State.PC, "Should jump to VBlank vector")
	assert.False(t, c.State.IME, "IME should be cleared during servicing")
}

// ---------------------------------------------------------------------------
// HALT bug test
// ---------------------------------------------------------------------------

func TestHALT_BugWithIMEClearedAndInterruptPending(t *testing.T) {
	c := newTestCore()
	// HALT bug: when HALT is executed with IME=0 and an interrupt is already pending,
	// the CPU advances PC past HALT but then executes the following byte as NOP
	// (without advancing PC for that byte).
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank ALREADY pending
	c.State.IME = false

	// Place HALT (0x76) followed by a recognizable opcode
	// The byte AFTER HALT will be executed TWICE (once as NOP-ish, once normally)
	writeCode(c.MMU, 0x0100, 0x76, 0x00) // HALT, NOP
	c.State.PC = 0x0100

	// Step 1: HALT with IME=0 and interrupt pending — HALT BUG
	_, err := c.Step()
	require.NoError(t, err)

	// HALT bug: the instruction following HALT is executed (PC goes past HALT
	// but the following byte is eaten as NOP without advancing PC for it).
	// After HALT, interrupts remain pending but not serviced since IME=0.
	// The CPU is NOT truly halted but continues.

	// After the bug, PC should have advanced past HALT but the next byte
	// was consumed as a NOP (PC = 0x0102)
	assert.Equal(t, uint16(0x0102), c.State.PC,
		"HALT bug: PC should advance past HALT, then the next byte is consumed as NOP")
	assert.False(t, c.State.Halted, "HALT bug: CPU should not be halted")
}
