package gb

// Joypad manages the Game Boy joypad / P1 register at 0xFF00.
//
// Writing to the register sets the column-select bits (P10=P4, P11=P5).
// Reading returns the currently selected button states (active low) OR'd
// with the stored column-select bits.  Unused bits 6-7 always read as 1.
//
// Joypad interrupt (INT 0, IF bit 4): when ReadRegister detects that a bit
// in the lower nibble transitioned from 1 to 0 (button was pressed since the
// last read), it sets IF bit 4 to request a joypad interrupt.
type Joypad struct {
	// Internal button state (active high: 1 = pressed).
	// Bit 0: Right   Bit 1: Left    Bit 2: Up    Bit 3: Down
	// Bit 4: A       Bit 5: B       Bit 6: Select Bit 7: Start
	buttons byte

	// Stored column-select from the last write to 0xFF00 (bits 4-5).
	columnSelect byte

	// MMU reference for requesting the joypad interrupt (IF bit 4).
	mmu MMU

	// Value returned by the last ReadRegister, used to detect 1→0
	// transitions in the lower nibble (button press).
	lastRead byte
}

// NewJoypad creates a new Joypad with both columns deselected
// (neither column readable, lower nibble reads as 0x0F).
func NewJoypad(mmu MMU) *Joypad {
	return &Joypad{
		columnSelect: 0xFF,
		mmu:          mmu,
		lastRead:     0xFF, // all bits 1 → no transition on first read
	}
}

// SetButtons stores the current active-high button state.
// Called once per frame from the main loop.
func (j *Joypad) SetButtons(buttons byte) {
	j.buttons = buttons
}

// ReadRegister returns the value of the P1 register (0xFF00).
// It also detects button presses (1→0 transitions in the lower nibble)
// and requests a joypad interrupt (IF bit 4) when one occurs.
func (j *Joypad) ReadRegister() byte {
	// Start with the written column-select bits (upper nibble).
	result := j.columnSelect

	// Lower nibble: start with all ones (nothing pressed).
	var lowNibble byte = 0x0F

	// P14 (bit 4 = 0) → Direction column selected: bits 0-3 (Right/Left/Up/Down)
	if j.columnSelect&0x10 == 0 {
		dir := j.buttons & 0x0F
		lowNibble &^= dir
	}

	// P15 (bit 5 = 0) → Button column selected: bits 4-7 (A/B/Select/Start)
	if j.columnSelect&0x20 == 0 {
		action := (j.buttons >> 4) & 0x0F
		lowNibble &^= action
	}

	result |= lowNibble
	result |= 0xC0 // bits 6-7 always 1

	// Detect joypad interrupt: any bit in the lower nibble that was 1 in the
	// previous read and is now 0 indicates a button press (falling edge).
	if j.mmu != nil {
		prevLow := j.lastRead & 0x0F
		currLow := result & 0x0F
		falling := prevLow & ^currLow // bits that went from 1→0
		if falling != 0 {
			// Set IF bit 4 (joypad interrupt)
			ifReg := j.mmu.ReadIF()
			j.mmu.WriteIF(ifReg | 0x10)
		}
	}

	j.lastRead = result
	return result
}

// WriteRegister writes a value to the P1 register (0xFF00).
// Only bits 4-5 are stored; the lower nibble is ignored.
func (j *Joypad) WriteRegister(val byte) {
	j.columnSelect = val & 0xF0
}
