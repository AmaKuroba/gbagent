package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test memory base — we write code to WRAM (0xC000+) because
// writes to ROM region (0x0000-0x7FFF) are dropped with nil cartridge.
const testAddr = uint16(0xC000)

// newTestCore creates a Core with a fresh MemoryBus for testing.
func newTestCore() *Core {
	bus := NewMMU(nil)
	return NewCore(bus)
}

// writeCode places instruction bytes into memory at the given address.
func writeCode(mmu MMU, addr uint16, data ...byte) {
	for i, b := range data {
		mmu.Write(addr+uint16(i), b)
	}
}

// ---------------------------------------------------------------------------
// IME / IF / IE register tests
// ---------------------------------------------------------------------------

func TestIMEStartsCleared(t *testing.T) {
	c := newTestCore()
	state := c.GetState()
	assert.False(t, state.IME, "IME should start cleared after reset")
}

func TestEI_SetsIMEDelayed(t *testing.T) {
	c := newTestCore()
	// EI (0xFB) followed by NOP (0x00)
	writeCode(c.MMU, testAddr, 0xFB, 0x00)
	c.PC = testAddr

	// Step 1: EI — IME should NOT yet be set
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.IME, "IME should NOT be set during EI itself")

	// Step 2: NOP — IME should now be set (EI takes 1 instruction delay)
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.IME, "IME should be set after the instruction following EI")
}

func TestDI_ClearsIMEImmediately(t *testing.T) {
	c := newTestCore()
	// Set IME enabled
	c.IME = true
	c.IMEScheduled = 0

	// DI (0xF3) clears IME immediately
	writeCode(c.MMU, testAddr, 0xF3)
	c.PC = testAddr

	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.IME, "DI should clear IME immediately during its execution")
	assert.Equal(t, 0, c.IMEScheduled, "DI should clear any pending EI enable")
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
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending

	// EI (0xFB), NOP (0x00), NOP (0x00)
	writeCode(c.MMU, testAddr, 0xFB, 0x00, 0x00)
	c.PC = testAddr
	c.SP = 0xFFFE

	// Step 1: EI
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.IME)

	// Step 2: NOP — enables IME after this instruction
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.IME)

	// Step 3: interrupt should fire (IME=1, IE & IF != 0)
	pcBefore := c.PC
	spBefore := c.SP
	_, err = c.Step()
	require.NoError(t, err)

	// Should have jumped to VBlank vector 0x40
	assert.Equal(t, uint16(0x0040), c.PC, "Should jump to VBlank vector")
	// IME should be cleared
	assert.False(t, c.IME, "IME should be cleared during interrupt servicing")
	// SP should have decremented by 2 (PC pushed)
	assert.Equal(t, spBefore-2, c.SP, "SP should decrement by 2")
	// IF bit 0 should be cleared
	assert.Equal(t, byte(0x00), c.MMU.Read(0xFF0F)&0x01, "VBlank IF flag should be cleared")
	// PC before interrupt should be pushed onto stack
	pushedPC := c.MMU.Read16(c.SP)
	assert.Equal(t, pcBefore, pushedPC, "PC should be pushed onto stack")
}

func TestVBlankInterrupt_DoesNotFireWhenIMECleared(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending
	c.IME = false

	writeCode(c.MMU, testAddr, 0x00, 0x00) // NOP, NOP
	c.PC = testAddr

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+1, c.PC, "PC should advance normally, no interrupt")
	assert.Equal(t, byte(0x01), c.MMU.Read(0xFF0F)&0x01, "IF flag should remain set")
}

func TestVBlankInterrupt_DoesNotFireWhenIFCleared(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x00) // IF: nothing pending
	c.IME = true

	writeCode(c.MMU, testAddr, 0x00) // NOP
	c.PC = testAddr

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+1, c.PC, "PC should advance normally, no interrupt")
}

func TestVBlankInterrupt_HighestPriority(t *testing.T) {
	c := newTestCore()
	// All interrupts pending but VBlank (bit 0) has highest priority
	c.MMU.Write(0xFFFF, 0x1F) // All 5 interrupts enabled
	c.MMU.Write(0xFF0F, 0x1F) // All 5 interrupts pending
	c.IME = true
	c.SP = 0xFFFE

	writeCode(c.MMU, testAddr, 0x00) // NOP
	c.PC = testAddr

	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint16(0x0040), c.PC, "Should jump to VBlank vector (highest priority)")
}

// ---------------------------------------------------------------------------
// HALT tests
// ---------------------------------------------------------------------------

func TestHALT_HaltsCPU(t *testing.T) {
	c := newTestCore()
	// HALT (0x76) should stop execution
	writeCode(c.MMU, testAddr, 0x76, 0x00, 0x00)
	c.PC = testAddr

	// Step 1: HALT
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Halted, "CPU should be halted after HALT instruction")

	// Next step should not advance PC while halted
	stateBefore := c.GetState()
	_, err = c.Step()
	require.NoError(t, err)
	assert.Equal(t, stateBefore.PC, c.PC, "PC should not advance while halted")
	assert.Equal(t, stateBefore.Cycles+4, c.Cycles, "Cycles should still increment while halted")
}

func TestHALT_WakesOnInterrupt(t *testing.T) {
	c := newTestCore()
	// HALT with interrupts enabled — will wake and service on VBlank
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.IME = true
	c.SP = 0xFFFE

	writeCode(c.MMU, testAddr, 0x76) // HALT
	c.PC = testAddr
	c.MMU.Write(0xFF0F, 0x00) // clear IF initially

	// Step 1: HALT — no interrupt pending, should halt
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Halted, "CPU should be halted")

	// Set IF (simulate VBlank arriving)
	c.MMU.Write(0xFF0F, 0x01)

	// Step 2: should wake from halt AND service the interrupt
	_, err = c.Step()
	require.NoError(t, err)
	assert.False(t, c.Halted, "CPU should wake from halt")
	assert.Equal(t, uint16(0x0040), c.PC, "Should jump to VBlank vector")
	assert.False(t, c.IME, "IME should be cleared during servicing")
}

func TestHALT_WakesOnInterrupt_ButDoesNotService(t *testing.T) {
	c := newTestCore()
	// HALT with IME=0 — wakes on interrupt but does NOT service it
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.IME = false
	c.SP = 0xFFFE

	writeCode(c.MMU, testAddr, 0x76, 0x00) // HALT, NOP
	c.PC = testAddr
	c.MMU.Write(0xFF0F, 0x00) // clear IF initially

	// Step 1: HALT — no interrupt pending yet
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Halted, "CPU should be halted")

	// Set IF (simulate VBlank arriving during halt)
	c.MMU.Write(0xFF0F, 0x01)

	// Step 2: IME=0, serveInterrupt doesn't fire, but Halted check
	// sees pending interrupt and wakes CPU (Halted=false, no instruction executed)
	_, err = c.Step()
	require.NoError(t, err)
	assert.False(t, c.Halted, "CPU should wake from halt when interrupt pending even with IME=0")
	assert.Equal(t, testAddr+1, c.PC, "PC stays at NOP after HALT while waking")

	// Step 3: execute the NOP after HALT
	_, err = c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+2, c.PC,
		"PC should advance after executing the instruction following HALT")
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
	c.IME = false

	writeCode(c.MMU, testAddr, 0x76, 0x00) // HALT, NOP
	c.PC = testAddr

	// Step 1: HALT with IME=0 and interrupt pending — HALT BUG
	_, err := c.Step()
	require.NoError(t, err)

	// After the bug, PC should have advanced past HALT only.
	// The HALT bug sets HaltBug=true which suppresses PC increment on the
	// *next* instruction fetch (Step 2), not during HALT itself (Step 1).
	assert.Equal(t, testAddr+1, c.PC,
		"HALT bug: PC should advance past HALT only; the next byte's PC increment is suppressed in Step 2")
	assert.False(t, c.Halted, "HALT bug: CPU should not be halted")
}

// ---------------------------------------------------------------------------
// STOP tests
// ---------------------------------------------------------------------------

func TestSTOP_StopsCPU(t *testing.T) {
	c := newTestCore()
	// STOP (0x10) followed by ignored byte (0x00), then NOP (0x00)
	writeCode(c.MMU, testAddr, 0x10, 0x00, 0x00)
	c.PC = testAddr

	// Step 1: STOP — enters stopped state
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Stopped, "CPU should be stopped after STOP instruction")
	assert.Equal(t, testAddr+2, c.PC, "PC should advance past STOP and its ignored byte")

	// Step 2: should not execute any instruction while stopped
	stateBefore := c.GetState()
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.Stopped, "CPU should remain stopped")
	assert.Equal(t, stateBefore.PC, c.PC, "PC should not advance while stopped")
	assert.Equal(t, stateBefore.Cycles+4, c.Cycles, "Cycles should still increment while stopped")
}

func TestSTOP_WakesOnInterrupt(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled

	// STOP (0x10) + ignored byte (0x00) + NOP (0x00)
	writeCode(c.MMU, testAddr, 0x10, 0x00, 0x00)
	c.PC = testAddr
	c.MMU.Write(0xFF0F, 0x00) // clear IF initially

	// Step 1: STOP — no interrupt pending, enters stopped state
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Stopped, "CPU should be stopped")

	// Set IF (simulate VBlank arriving while stopped)
	c.MMU.Write(0xFF0F, 0x01)

	// Step 2: should wake from STOP (IF&IE != 0, regardless of IME)
	_, err = c.Step()
	require.NoError(t, err)
	assert.False(t, c.Stopped, "CPU should wake from stopped state")
	assert.Equal(t, testAddr+2, c.PC, "PC should point to NOP after STOP")

	// Step 3: execute the NOP following STOP
	_, err = c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+3, c.PC, "PC should advance after executing NOP")
}

func TestSTOP_NoInterruptPending_RemainsStopped(t *testing.T) {
	c := newTestCore()
	// IE enabled but no IF bits set
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x00) // IF: nothing pending

	writeCode(c.MMU, testAddr, 0x10, 0x00)
	c.PC = testAddr

	// Step 1: STOP
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Stopped, "CPU should be stopped")

	// Step 2: still stopped (no enabled interrupt pending)
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.Stopped, "CPU should remain stopped when no interrupt pending")
}

// ---------------------------------------------------------------------------
// HALT tests (existing — for reference, these verify the companion task's coverage)
// ---------------------------------------------------------------------------
// TestHALT_WakesOnInterrupt_ButDoesNotService already verifies that HALT
// wakes on an enabled interrupt even when IME=0, which is the HALT edge case
// described in the companion task.
// TestHALT_NormalWithIMEClearedNoInterrupt already verifies that HALT
// remains halted when no interrupt is pending.

func TestHALT_NormalWithIMEClearedNoInterrupt(t *testing.T) {
	c := newTestCore()
	// Normal HALT with IME=0 and no pending interrupt
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x00) // IF: nothing pending
	c.IME = false

	writeCode(c.MMU, testAddr, 0x76, 0x00) // HALT, NOP
	c.PC = testAddr

	// Step 1: HALT — should halt normally since no interrupt pending
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.Halted, "CPU should be halted normally")
	assert.Equal(t, testAddr+1, c.PC, "PC should be just past HALT")

	// Step 2: Still halted
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.Halted, "CPU should remain halted")
}

func TestHALT_IMESetAndInterruptPending_ServicedFirst(t *testing.T) {
	c := newTestCore()
	// When IME=1 and interrupt pending, serveInterrupt fires at the start of Step()
	// before any instruction fetch
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending
	c.IME = true
	c.SP = 0xFFFE

	writeCode(c.MMU, testAddr, 0x76, 0x00) // HALT, NOP
	c.PC = testAddr

	// serveInterrupt fires before HALT is even fetched
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint16(0x0040), c.PC,
		"Interrupt should fire before HALT when IME=1 and interrupt pending")
	assert.False(t, c.Halted, "CPU should not be halted (interrupt serviced first)")
}

// ---------------------------------------------------------------------------
// EI+DI interaction
// ---------------------------------------------------------------------------

func TestEI_DI_Sequence_DisablesInterrupt(t *testing.T) {
	c := newTestCore()
	c.MMU.Write(0xFFFF, 0x01) // IE: VBlank enabled
	c.MMU.Write(0xFF0F, 0x01) // IF: VBlank pending
	c.SP = 0xFFFE

	// EI (0xFB) followed by DI (0xF3) — DI should cancel the delayed EI
	writeCode(c.MMU, testAddr, 0xFB, 0xF3, 0x00)
	c.PC = testAddr

	// Step 1: EI — sets IMEScheduled to 2, IME stays false
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.IME)

	// Step 2: DI — clears IME and IMEScheduled
	_, err = c.Step()
	require.NoError(t, err)
	assert.False(t, c.IME, "DI should prevent EI from enabling IME")
	assert.Equal(t, 0, c.IMEScheduled, "DI should clear pending enable")

	// Step 3: NOP — no interrupt should fire (IME is still false)
	_, err = c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+3, c.PC,
		"PC should advance normally, no interrupt")
}
