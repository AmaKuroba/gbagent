package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// joypadTestMMU is a minimal MMU stub for joypad tests that tracks IF writes.
type joypadTestMMU struct {
	ifReg byte
}

func (m *joypadTestMMU) Read(addr uint16) byte {
	if addr == 0xFF0F {
		return m.ifReg
	}
	return 0
}

func (m *joypadTestMMU) Write(addr uint16, val byte) {
	if addr == 0xFF0F {
		m.ifReg = val
	}
}

func (m *joypadTestMMU) Read16(addr uint16) uint16 {
	return uint16(m.Read(addr)) | uint16(m.Read(addr+1))<<8
}

func (m *joypadTestMMU) Write16(addr uint16, val uint16) {
	m.Write(addr, byte(val))
	m.Write(addr+1, byte(val>>8))
}

func (m *joypadTestMMU) LoadROM(data []byte)           {}
func (m *joypadTestMMU) LoadBootROM(data []byte)       {}
func (m *joypadTestMMU) ReadIF() byte                  { return m.ifReg }
func (m *joypadTestMMU) ReadIE() byte                  { return 0 }
func (m *joypadTestMMU) WriteIF(val byte)              { m.ifReg = val }
func (m *joypadTestMMU) DMAStep(cycles int)            {}
func (m *joypadTestMMU) SerialStep(cycles int)         {}
func (m *joypadTestMMU) SetJoypadButtons(buttons byte) {}
func (m *joypadTestMMU) StepDevices(cycles int)        {}

// ---------------------------------------------------------------------------
// Joypad interrupt tests
// ---------------------------------------------------------------------------

func TestJoypadInterrupt_NoInterruptOnFirstRead(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// First read should not trigger an interrupt
	joypad.ReadRegister()
	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should not be set on first read")
}

func TestJoypadInterrupt_NoInterruptWithoutButtonPress(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column

	// Read once to establish baseline
	joypad.ReadRegister()
	mmu.ifReg = 0x00 // clear IF if any

	// Read again with no buttons pressed — no interrupt
	joypad.ReadRegister()
	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should not be set when no button is pressed")

	// Simulate holding a button between reads (stays pressed)
	joypad.SetButtons(0x01) // Right pressed
	joypad.ReadRegister()   // first read with button pressed → establishes baseline
	mmu.ifReg = 0x00        // clear IF

	joypad.ReadRegister() // second read, still pressed — no transition
	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should not be set when button stays pressed")
}

func TestJoypadInterrupt_ButtonPressTriggersInterrupt(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column

	// Establish baseline (no buttons pressed)
	joypad.ReadRegister()
	mmu.ifReg = 0x00

	// Press Right button (bit 0)
	joypad.SetButtons(0x01)
	result := joypad.ReadRegister()

	// IF bit 4 should be set
	assert.Equal(t, byte(0x10), mmu.ifReg&0x10, "IF bit 4 should be set on button press")
	// Lower nibble should show Right as pressed (active low)
	assert.Equal(t, byte(0x0E), result&0x0F, "Right button should read as pressed (bit 0=0)")
}

func TestJoypadInterrupt_MultipleButtonsTriggerOnce(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column

	// Establish baseline
	joypad.ReadRegister()
	mmu.ifReg = 0x00

	// Press two buttons at once
	joypad.SetButtons(0x03) // Right | Left
	joypad.ReadRegister()

	// IF bit 4 should be set
	assert.Equal(t, byte(0x10), mmu.ifReg&0x10, "IF bit 4 should be set when multiple buttons pressed")
}

func TestJoypadInterrupt_ButtonReleaseDoesNotTrigger(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column

	// Press button
	joypad.SetButtons(0x01)
	joypad.ReadRegister() // triggers interrupt
	mmu.ifReg = 0x00      // clear IF

	// Release button (0→1 transition should NOT trigger interrupt)
	joypad.SetButtons(0x00)
	joypad.ReadRegister()

	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should NOT be set on button release (0→1)")
}

func TestJoypadInterrupt_InterruptPreservesOtherIFBits(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column

	// Pre-set IF with VBlank bit
	mmu.ifReg = 0x01 // VBlank pending

	// Establish baseline read (no buttons)
	joypad.ReadRegister()

	// Press a button
	joypad.SetButtons(0x02) // Left
	joypad.ReadRegister()

	// IF should have both VBlank (bit 0) and Joypad (bit 4)
	assert.Equal(t, byte(0x11), mmu.ifReg, "IF should preserve existing bits when setting joypad interrupt")
}

func TestJoypadInterrupt_BothColumns(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select both columns
	joypad.WriteRegister(0x00) // both P14 and P15 selected

	// Establish baseline
	joypad.ReadRegister()
	mmu.ifReg = 0x00

	// Press A button (bit 4) — appears in lower nibble when P15 is selected
	joypad.SetButtons(0x10) // A
	result := joypad.ReadRegister()

	// IF bit 4 should be set
	assert.Equal(t, byte(0x10), mmu.ifReg&0x10, "IF bit 4 should be set on action button press")
	// Lower nibble should have bit 0 clear (A maps to bit 0 of lower nibble in action column)
	assert.Equal(t, byte(0x0E), result&0x0F, "A button should read as pressed")
}

func TestJoypadInterrupt_ColumnChangeGeneratesNewBaseline(t *testing.T) {
	// When the game switches columns, the new column's state becomes the baseline
	// and a repeated read of the same row won't re-trigger.
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Select action column (P15=0)
	joypad.WriteRegister(0x10) // P14=1, P15=0 → action column
	joypad.ReadRegister()      // establish baseline (no action buttons pressed)
	mmu.ifReg = 0x00

	// Press A (bit 4) — this is in the action column
	joypad.SetButtons(0x10)
	joypad.ReadRegister()
	assert.Equal(t, byte(0x10), mmu.ifReg&0x10, "IF bit 4 should be set on press in action column")
	mmu.ifReg = 0x00

	// Read again with same column — no transition, same state
	joypad.ReadRegister()
	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should not repeat on same state")

	// Now switch to direction column (P14=0, P15=1)
	joypad.SetButtons(0x00)    // release all
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column
	joypad.ReadRegister()      // establish baseline for direction column
	mmu.ifReg = 0x00

	// Press Right button
	joypad.SetButtons(0x01)
	joypad.ReadRegister()
	assert.Equal(t, byte(0x10), mmu.ifReg&0x10, "IF bit 4 should be set on direction column press")
}

// ---------------------------------------------------------------------------
// Integration test: joypad interrupt flows through CPU
// ---------------------------------------------------------------------------

func TestJoypadInterrupt_CPUServicesWhenEnabled(t *testing.T) {
	c := newTestCore()
	// Create a new joypad with the MMU and attach it
	joypad := NewJoypad(c.MMU)
	c.MMU.(*MemoryBus).SetJoypad(joypad)

	// Enable joypad interrupt in IE
	c.MMU.Write(0xFFFF, 0x10) // IE: Joypad enabled
	c.IME = true
	c.SP = 0xFFFE

	// Set up code at test address
	writeCode(c.MMU, testAddr, 0xFB, 0x00) // EI, NOP
	c.PC = testAddr

	// Step 1: EI
	_, err := c.Step()
	require.NoError(t, err)

	// Step 2: NOP — IME becomes active
	_, err = c.Step()
	require.NoError(t, err)
	assert.True(t, c.IME, "IME should be set after EI+NOP")

	// Select direction column
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column
	joypad.SetButtons(0x01)    // press Right
	joypad.ReadRegister()      // triggers IF bit 4

	assert.Equal(t, byte(0x10), c.MMU.Read(0xFF0F)&0x10, "IF bit 4 should be set")

	// Next step should service the joypad interrupt
	pcBefore := c.PC
	spBefore := c.SP
	_, err = c.Step()
	require.NoError(t, err)

	// Should have jumped to joypad vector 0x60
	assert.Equal(t, uint16(0x0060), c.PC, "Should jump to joypad interrupt vector 0x60")
	assert.False(t, c.IME, "IME should be cleared during interrupt servicing")
	assert.Equal(t, spBefore-2, c.SP, "SP should decrement by 2")
	assert.Equal(t, byte(0x00), c.MMU.Read(0xFF0F)&0x10, "IF bit 4 should be cleared after servicing")

	// PC before interrupt should be pushed onto stack
	pushedPC := c.MMU.Read16(c.SP)
	assert.Equal(t, pcBefore, pushedPC, "PC should be pushed onto stack")
}

func TestJoypadInterrupt_DoesNotFireWhenDisabledInIE(t *testing.T) {
	c := newTestCore()
	joypad := NewJoypad(c.MMU)
	c.MMU.(*MemoryBus).SetJoypad(joypad)

	// IE has no joypad interrupt enabled
	c.MMU.Write(0xFFFF, 0x01) // IE: only VBlank enabled
	c.IME = true
	c.SP = 0xFFFE

	writeCode(c.MMU, testAddr, 0x00) // NOP
	c.PC = testAddr

	// Trigger joypad interrupt in IF
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column
	joypad.SetButtons(0x01)
	joypad.ReadRegister()

	assert.Equal(t, byte(0x10), c.MMU.Read(0xFF0F)&0x10, "IF bit 4 should be set")

	// Step should not service joypad interrupt (IE bit 4 not set)
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, testAddr+1, c.PC, "PC should advance normally, no joypad interrupt")
	assert.True(t, c.IME, "IME should remain set")
}

func TestJoypadInterrupt_ReadDoesNotTriggerOnColumnSwitchWithoutPress(t *testing.T) {
	// Switching columns changes the lower nibble, but if no button is newly pressed,
	// the interrupt should not fire.
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	// Direction column read — no buttons pressed
	joypad.WriteRegister(0x20) // P14=0, P15=1 → direction column
	joypad.ReadRegister()
	mmu.ifReg = 0x00

	// Switch to action column — no buttons pressed, lower nibble is 0x0F in both
	joypad.WriteRegister(0x10) // P14=1, P15=0 → action column
	joypad.ReadRegister()

	// Should NOT trigger interrupt — no 1→0 transition, both read as 0x0F
	assert.Equal(t, byte(0x00), mmu.ifReg&0x10, "IF bit 4 should not be set on column switch without press")
}
