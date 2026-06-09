package gb

// Timer implements the Game Boy timer and divider hardware.
// Registers (all 8-bit):
//   0xFF04 — DIV:  Incremented at 16384 Hz (every 256 T-cycles).
//                  Writing any value resets it to $00.
//   0xFF05 — TIMA: Timer counter. Incremented at a rate controlled by TAC.
//                  On overflow ($FF→$00) reloads from TMA and sets IF bit 2.
//   0xFF06 — TMA:  Timer modulo. Value loaded into TIMA on overflow.
//   0xFF07 — TAC:  Timer control. Bit 2 = enable, bits 1-0 = clock select.
//
// Clock selection (TAC bits 1-0):
//   00: 4096 Hz   (every 1024 T-cycles)
//   01: 262144 Hz (every 16 T-cycles)
//   10: 65536 Hz  (every 64 T-cycles)
//   11: 16384 Hz  (every 256 T-cycles)
type Timer struct {
	// IO registers
	DIV  byte // 0xFF04
	TIMA byte // 0xFF05
	TMA  byte // 0xFF06
	TAC  byte // 0xFF07

	// Internal 16-bit cycle counters.
	// divCycles counts toward DIV increments (free-running, every 256 cycles).
	// timaCycles counts toward TIMA increments (only when TAC bit 2 is set).
	divCycles  uint16
	timaCycles uint16

	// mmu for requesting interrupts (set on TIMA overflow)
	mmu MMU
}

// NewTimer creates a new Timer with default register values.
func NewTimer(mmu MMU) *Timer {
	// Power-on defaults per Pan Docs: TIMA=0x00, TMA=0x00, TAC=0xF8 (bits 7-3 always 1).
	return &Timer{
		TAC: 0xF8,
		mmu: mmu,
	}
}

// Step advances the timer by the given number of T-cycles.
// Must be called after every CPU instruction to keep DIV and TIMA in sync.
func (t *Timer) Step(cycles int) {
	if cycles <= 0 {
		return
	}

	// --- DIV: free-running, always increments ---
	t.divCycles += uint16(cycles)
	for t.divCycles >= 256 {
		t.divCycles -= 256
		t.DIV++ // wraps naturally
	}

	// --- TIMA: only when TAC bit 2 (enable) is set ---
	if t.TAC&0x04 == 0 {
		return
	}

	threshold := t.timaThreshold()
	t.timaCycles += uint16(cycles)

	for t.timaCycles >= threshold {
		t.timaCycles -= threshold
		t.advanceTIMA()
	}
}

// timaThreshold returns the number of T-cycles between TIMA increments
// based on TAC bits 1-0.
func (t *Timer) timaThreshold() uint16 {
	switch t.TAC & 0x03 {
	case 0:
		return 1024 // 4096 Hz
	case 1:
		return 16 // 262144 Hz
	case 2:
		return 64 // 65536 Hz
	case 3:
		return 256 // 16384 Hz
	default:
		return 1024
	}
}

// advanceTIMA increments TIMA and handles overflow (reload from TMA + interrupt).
func (t *Timer) advanceTIMA() {
	t.TIMA++
	if t.TIMA == 0 { // overflow from $FF → $00
		t.TIMA = t.TMA
		// Request timer interrupt: IF bit 2 (timer interrupt)
		if t.mmu != nil {
			t.mmu.WriteIF(t.mmu.ReadIF() | 0x04)
		}
	}
}

// ReadRegister returns the value of a timer IO register at the given address.
// Only 0xFF04-0xFF07 are handled; other addresses should not reach here.
func (t *Timer) ReadRegister(addr uint16) byte {
	switch addr {
	case 0xFF04:
		return t.DIV
	case 0xFF05:
		return t.TIMA
	case 0xFF06:
		return t.TMA
	case 0xFF07:
		// TAC: bits 7-3 always read as 1 (per hardware spec)
		return t.TAC | 0xF8
	default:
		return 0xFF
	}
}

// WriteRegister writes a value to a timer IO register at the given address.
// Only 0xFF04-0xFF07 are handled.
func (t *Timer) WriteRegister(addr uint16, val byte) {
	switch addr {
	case 0xFF04:
		// Writing any value resets DIV (and the internal divider) to $00
		t.DIV = 0
		t.divCycles = 0
		// Writing to DIV also resets the internal cycle counter for TIMA
		// (the same 16-bit internal divider feeds both DIV and TIMA timing)
		t.timaCycles = 0
	case 0xFF05:
		t.TIMA = val
	case 0xFF06:
		t.TMA = val
	case 0xFF07:
		// TAC: only bits 2 and 1-0 are writable; bits 7-3 always read as 1
		t.TAC = (val & 0x07) | 0xF8
		// Writing to TAC resets the internal TIMA cycle counter
		t.timaCycles = 0
	}
}

// Reset sets the timer to power-on state.
func (t *Timer) Reset() {
	t.DIV = 0
	t.TIMA = 0
	t.TMA = 0
	t.TAC = 0xF8
	t.divCycles = 0
	t.timaCycles = 0
}

// GetState returns a snapshot of the timer registers for debugging/display.
func (t *Timer) GetState() TimerState {
	return TimerState{
		DIV:  t.DIV,
		TIMA: t.TIMA,
		TMA:  t.TMA,
		TAC:  t.TAC,
	}
}

// TimerState is a debug snapshot of the timer hardware.
type TimerState struct {
	DIV   byte
	TIMA  byte
	TMA   byte
	TAC   byte
	Freq  string
	Enable bool
}
