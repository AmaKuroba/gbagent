package gb

// Core implements the CPU interface for the LR35902 (Sharp SM83) processor.
type Core struct {
	// 16-bit register pairs
	AF, BC, DE, HL, SP, PC uint16

	// Interrupt state
	IME              bool // Interrupt Master Enable
	IMEEnablePending bool // EI sets this, IME becomes active after next instruction
	Halted           bool
	Stopped          bool

	// Cycle counter
	Cycles uint64

	// Memory bus (interface)
	MMU MMU
}

// Compile-time check: *Core implements CPU
var _ CPU = (*Core)(nil)

// Flag bit masks for the F register (lower byte of AF).
const (
	flagZ = 0x80 // Zero flag  (bit 7)
	flagN = 0x40 // Subtract flag (bit 6)
	flagH = 0x20 // Half-carry flag (bit 5)
	flagC = 0x10 // Carry flag (bit 4)
)

// NewCore creates a new CPU core with initialized register values.
func NewCore(mmu MMU) *Core {
	return &Core{
		PC:  0x0100, // Default entry point after boot ROM
		SP:  0xFFFE,
		MMU: mmu,
	}
}

// NewCoreCPU is an alias for NewCore (compatibility with ALU tests).
func NewCoreCPU(mmu MMU) *Core { return NewCore(mmu) }

// NewCPU is an alias for NewCore (compatibility with CB tests).
func NewCPU(mmu MMU) *Core { return NewCore(mmu) }

// --- 8-bit register accessors ---

func (c *Core) A() byte     { return byte(c.AF >> 8) }
func (c *Core) F() byte     { return byte(c.AF) }
func (c *Core) B() byte     { return byte(c.BC >> 8) }
func (c *Core) C() byte     { return byte(c.BC) }
func (c *Core) D() byte     { return byte(c.DE >> 8) }
func (c *Core) E() byte     { return byte(c.DE) }
func (c *Core) H() byte     { return byte(c.HL >> 8) }
func (c *Core) L() byte     { return byte(c.HL) }

// Register accessors for ALU test compatibility (getX / setX pattern)
func (c *Core) getA() byte   { return c.A() }
func (c *Core) getB() byte   { return c.B() }
func (c *Core) getC() byte   { return c.C() }
func (c *Core) getD() byte   { return c.D() }
func (c *Core) getE() byte   { return c.E() }
func (c *Core) getH() byte   { return c.H() }
func (c *Core) getL() byte   { return c.L() }
func (c *Core) getF() byte   { return c.F() }

func (c *Core) setA(v byte)   { c.AF = (uint16(v) << 8) | uint16(c.F()) }
func (c *Core) setB(v byte)   { c.BC = (uint16(v) << 8) | uint16(c.C()) }
func (c *Core) setC(v byte)   { c.BC = (uint16(c.B()) << 8) | uint16(v) }
func (c *Core) setD(v byte)   { c.DE = (uint16(v) << 8) | uint16(c.E()) }
func (c *Core) setE(v byte)   { c.DE = (uint16(c.D()) << 8) | uint16(v) }
func (c *Core) setH(v byte)   { c.HL = (uint16(v) << 8) | uint16(c.L()) }
func (c *Core) setL(v byte)   { c.HL = (uint16(c.H()) << 8) | uint16(v) }

// --- 16-bit register helpers ---

func (c *Core) BC() uint16 { return c.BC }
func (c *Core) DE() uint16 { return c.DE }
func (c *Core) HL() uint16 { return c.HL }

func (c *Core) SetBC(v uint16) { c.BC = v }
func (c *Core) SetDE(v uint16) { c.DE = v }
func (c *Core) SetHL(v uint16) { c.HL = v }

// --- Flag accessors (for test compatibility) ---

func (c *Core) flagZ() bool  { return c.AF&flagZ != 0 }
func (c *Core) flagN() bool  { return c.AF&flagN != 0 }
func (c *Core) flagH() bool  { return c.AF&flagH != 0 }
func (c *Core) flagC() bool  { return c.AF&flagC != 0 }

// getZ/getN/getH/getC — public aliases for flag tests (CB test compatibility)
func (c *Core) getZ() bool { return c.flagZ() }
func (c *Core) getN() bool { return c.flagN() }
func (c *Core) getH() bool { return c.flagH() }
func (c *Core) getC() bool { return c.flagC() }

func (c *Core) setFlagZ(v bool) { c.setFlagBit(7, v) }
func (c *Core) setFlagN(v bool) { c.setFlagBit(6, v) }
func (c *Core) setFlagH(v bool) { c.setFlagBit(5, v) }
func (c *Core) setFlagC(v bool) { c.setFlagBit(4, v) }

func (c *Core) setFlagBit(bit uint, v bool) {
	if v {
		c.AF |= 1 << bit
	} else {
		c.AF &^= 1 << bit
	}
}

func (c *Core) setZNHC(z, n, h, cf bool) {
	c.setFlagZ(z)
	c.setFlagN(n)
	c.setFlagH(h)
	c.setFlagC(cf)
}

// Flags returns the current flag states (for test assertions).
func (c *Core) Flags() struct{ Z, N, H, C bool } {
	return struct{ Z, N, H, C bool }{c.flagZ(), c.flagN(), c.flagH(), c.flagC()}
}

// StepCycles returns the cycle count (for tests).
func (c *Core) StepCycles() int { return int(c.Cycles) }

// SetZF / SetCF — for test convenience.
func (c *Core) SetZF(v bool) { c.setFlagZ(v) }
func (c *Core) SetCF(v bool) { c.setFlagC(v) }

// GetA / GetF — for test convenience.
func (c *Core) GetA() byte { return c.A() }
func (c *Core) GetF() byte { return c.F() }

// --- Memory access helpers ---

// Read the next instruction byte and advance PC.
func (c *Core) fetch8() byte {
	val := c.MMU.Read(c.PC)
	c.PC++
	return val
}

// Read a 16-bit immediate value (little-endian) and advance PC by 2.
func (c *Core) fetch16() uint16 {
	lo := c.MMU.Read(c.PC)
	hi := c.MMU.Read(c.PC + 1)
	c.PC += 2
	return uint16(lo) | uint16(hi)<<8
}

// Push and pop from stack
func (c *Core) push16(val uint16) {
	c.SP -= 2
	c.MMU.Write16(c.SP, val)
}

func (c *Core) pop16() uint16 {
	val := c.MMU.Read16(c.SP)
	c.SP += 2
	return val
}

// --- ALU helpers (called directly by ALU tests) ---

// add performs A += val and sets flags accordingly.
func (c *Core) add(val byte) {
	a := uint16(c.A())
	result := a + uint16(val)
	c.setA(byte(result))
	c.setFlagZ(byte(result) == 0)
	c.setFlagN(false)
	c.setFlagH((a&0x0F)+(uint16(val)&0x0F) > 0x0F)
	c.setFlagC(result > 0xFF)
}

// addHL performs A += (HL) (memory-indirect ADD).
func (c *Core) addHL() {
	c.add(c.MMU.Read(c.HL))
}

// sub performs A -= val and sets flags accordingly.
func (c *Core) sub(val byte) {
	a := uint16(c.A())
	result := a - uint16(val)
	c.setA(byte(result))
	c.setFlagZ(byte(result) == 0)
	c.setFlagN(true)
	c.setFlagH((a&0x0F)-(uint16(val)&0x0F) > 0x0F || (a&0x0F) < (uint16(val)&0x0F))
	c.setFlagC(result > 0xFF) // borrow: result > 0xFF when a < val as uint16
}

// subHL performs A -= (HL).
func (c *Core) subHL() {
	c.sub(c.MMU.Read(c.HL))
}

// and performs A &= val and sets flags accordingly.
func (c *Core) and(val byte) {
	result := c.A() & val
	c.setA(result)
	c.setFlagZ(result == 0)
	c.setFlagN(false)
	c.setFlagH(true) // H is always set for AND
	c.setFlagC(false)
}

// andHL performs A &= (HL).
func (c *Core) andHL() {
	c.and(c.MMU.Read(c.HL))
}

// or performs A |= val and sets flags accordingly.
func (c *Core) or(val byte) {
	result := c.A() | val
	c.setA(result)
	c.setFlagZ(result == 0)
	c.setFlagN(false)
	c.setFlagH(false)
	c.setFlagC(false)
}

// orHL performs A |= (HL).
func (c *Core) orHL() {
	c.or(c.MMU.Read(c.HL))
}

// xor performs A ^= val and sets flags accordingly.
func (c *Core) xor(val byte) {
	result := c.A() ^ val
	c.setA(result)
	c.setFlagZ(result == 0)
	c.setFlagN(false)
	c.setFlagH(false)
	c.setFlagC(false)
}

// xorHL performs A ^= (HL).
func (c *Core) xorHL() {
	c.xor(c.MMU.Read(c.HL))
}

// cp performs A - val without storing, sets flags as if SUB was executed.
func (c *Core) cp(val byte) {
	a := uint16(c.A())
	result := a - uint16(val)
	c.setFlagZ(byte(result) == 0)
	c.setFlagN(true)
	c.setFlagH((a&0x0F)-(uint16(val)&0x0F) > 0x0F || (a&0x0F) < (uint16(val)&0x0F))
	c.setFlagC(result > 0xFF)
}

// cpHL performs CP with (HL).
func (c *Core) cpHL() {
	c.cp(c.MMU.Read(c.HL))
}

// inc performs val + 1, returns (result, zero, half-carry).
func (c *Core) inc(val byte) (byte, bool, bool) {
	result := val + 1
	z := result == 0
	h := (val&0x0F)+(1&0x0F) > 0x0F
	return result, z, h
}

// dec performs val - 1, returns (result, zero, half-borrow).
func (c *Core) dec(val byte) (byte, bool, bool) {
	result := val - 1
	z := result == 0
	h := (val & 0x0F) == 0 // half-borrow if lower nibble was 0
	return result, z, h
}

// --- Step and interrupt handling ---

// Step executes one CPU instruction and returns the number of cycles consumed.
func (c *Core) Step() (int, error) {
	// Check for interrupt servicing before each instruction
	if c.serveInterrupt() {
		return 20, nil
	}

	// Handle halted state
	if c.Halted {
		c.Cycles += 4
		// Check if an interrupt should wake us
		if c.IME && (c.MMU.Read(0xFF0F)&c.MMU.Read(0xFFFF)) != 0 {
			c.Halted = false
		}
		return 4, nil
	}

	if c.Stopped {
		c.Cycles += 4
		return 4, nil
	}

	// Fetch opcode and execute
	opcode := c.fetch8()
	// Fetch opcode and execute
	opcode := c.fetch8()
	_ = opcode
	// TODO: dispatch via mainHandler when cpu_core_ops.go is rebuilt
	c.Cycles += 4
	return 4, nil
}

// serveInterrupt checks if an interrupt should fire and services it.
func (c *Core) serveInterrupt() bool {
	if !c.IME {
		return false
	}

	requested := c.MMU.Read(0xFF0F)
	enabled := c.MMU.Read(0xFFFF)
	trigger := requested & enabled
	if trigger == 0 {
		return false
	}

	// Find highest-priority interrupt (bit 0 = VBlank = highest)
	var vector uint16
	var bit byte
	switch {
	case trigger&0x01 != 0:
		vector = 0x0040 // VBlank
		bit = 0x01
	case trigger&0x02 != 0:
		vector = 0x0048 // LCD STAT
		bit = 0x02
	case trigger&0x04 != 0:
		vector = 0x0050 // Timer
		bit = 0x04
	case trigger&0x08 != 0:
		vector = 0x0058 // Serial
		bit = 0x08
	case trigger&0x10 != 0:
		vector = 0x0060 // Joypad
		bit = 0x10
	}

	// Service the interrupt
	c.IME = false
	c.IMEEnablePending = false

	// Push current PC onto stack
	c.push16(c.PC)

	// Jump to vector
	c.PC = vector

	// Clear the interrupt flag bit
	c.MMU.Write(0xFF0F, requested & ^bit)

	// Wake from HALT if needed
	c.Halted = false

	c.Cycles += 20
	return true
}

// Reset resets the CPU to its initial state.
func (c *Core) Reset() {
	c.AF = 0x01B0
	c.BC = 0x0013
	c.DE = 0x00D8
	c.HL = 0x014D
	c.SP = 0xFFFE
	c.PC = 0x0100
	c.IME = false
	c.IMEEnablePending = false
	c.Halted = false
	c.Stopped = false
	c.Cycles = 0
}

// GetState returns a snapshot of the current CPU state.
func (c *Core) GetState() CPUState {
	return CPUState{
		AF:      c.AF,
		BC:      c.BC,
		DE:      c.DE,
		HL:      c.HL,
		SP:      c.SP,
		PC:      c.PC,
		IME:     c.IME,
		Halted:  c.Halted,
		Stopped: c.Stopped,
		Cycles:  c.Cycles,
	}
}
