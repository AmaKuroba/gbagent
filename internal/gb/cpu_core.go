package gb

// Core implements the CPU interface for the LR35902 (Sharp SM83) processor.
type Core struct {
	// 16-bit register pairs
	AF, BC, DE, HL, SP, PC uint16

	// Interrupt state
	IME          bool // Interrupt Master Enable
	IMEScheduled int  // EI schedule: 2=just set, 1=enable after this instr, 0=active
	Halted       bool
	Stopped      bool
	HaltBug      bool // HALT bug: prevent PC increment on next opcode fetch

	// Cycle counter
	Cycles uint64

	// Memory bus (interface)
	MMU MMU
}

var _ CPU = (*Core)(nil)

const (
	flagZ = 0x80
	flagN = 0x40
	flagH = 0x20
	flagC = 0x10
)

func NewCore(mmu MMU) *Core {
	return &Core{PC: 0x0100, SP: 0xFFFE, MMU: mmu}
}

func (c *Core) A() byte   { return byte(c.AF >> 8) }
func (c *Core) F() byte   { return byte(c.AF) }
func (c *Core) B() byte   { return byte(c.BC >> 8) }
func (c *Core) C() byte   { return byte(c.BC) }
func (c *Core) D() byte   { return byte(c.DE >> 8) }
func (c *Core) E() byte   { return byte(c.DE) }
func (c *Core) H() byte   { return byte(c.HL >> 8) }
func (c *Core) L() byte   { return byte(c.HL) }

func (c *Core) setA(v byte) { c.AF = uint16(v)<<8 | uint16(c.F()) }
func (c *Core) setB(v byte) { c.BC = uint16(v)<<8 | uint16(c.C()) }
func (c *Core) setC(v byte) { c.BC = uint16(c.B())<<8 | uint16(v) }
func (c *Core) setD(v byte) { c.DE = uint16(v)<<8 | uint16(c.E()) }
func (c *Core) setE(v byte) { c.DE = uint16(c.D())<<8 | uint16(v) }
func (c *Core) setH(v byte) { c.HL = uint16(v)<<8 | uint16(c.L()) }
func (c *Core) setL(v byte) { c.HL = uint16(c.H())<<8 | uint16(v) }

func (c *Core) flagZ() bool { return c.AF&flagZ != 0 }
func (c *Core) flagN() bool { return c.AF&flagN != 0 }
func (c *Core) flagH() bool { return c.AF&flagH != 0 }
func (c *Core) flagC() bool { return c.AF&flagC != 0 }

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
	c.setFlagZ(z); c.setFlagN(n); c.setFlagH(h); c.setFlagC(cf)
}

func (c *Core) fetch8() byte {
	v := c.MMU.Read(c.PC)
	c.PC++
	return v
}

func (c *Core) fetch16() uint16 {
	lo := c.MMU.Read(c.PC)
	hi := c.MMU.Read(c.PC + 1)
	c.PC += 2
	return uint16(lo) | uint16(hi)<<8
}

func (c *Core) push16(val uint16) {
	c.SP -= 2
	c.MMU.Write16(c.SP, val)
}

func (c *Core) pop16() uint16 {
	v := c.MMU.Read16(c.SP)
	c.SP += 2
	return v
}

func (c *Core) readReg8(idx int) byte {
	switch idx {
	case 0: return c.B()
	case 1: return c.C()
	case 2: return c.D()
	case 3: return c.E()
	case 4: return c.H()
	case 5: return c.L()
	case 6: return c.MMU.Read(c.HL)
	case 7: return c.A()
	}
	return 0
}

func (c *Core) writeReg8(idx int, val byte) {
	switch idx {
	case 0: c.setB(val)
	case 1: c.setC(val)
	case 2: c.setD(val)
	case 3: c.setE(val)
	case 4: c.setH(val)
	case 5: c.setL(val)
	case 6: c.MMU.Write(c.HL, val)
	case 7: c.setA(val)
	}
}

func boolToByte(b bool) byte { if b { return 1 }; return 0 }

func (c *Core) addA(val byte) {
	a := uint16(c.A())
	r := a + uint16(val)
	c.setA(byte(r))
	c.setFlagZ(byte(r) == 0)
	c.setFlagN(false)
	c.setFlagH((a&0x0F)+(uint16(val)&0x0F) > 0x0F)
	c.setFlagC(r > 0xFF)
}

func (c *Core) adcA(val byte) {
	cy := boolToByte(c.flagC())
	a := uint16(c.A())
	r := a + uint16(val) + uint16(cy)
	c.setA(byte(r))
	c.setFlagZ(byte(r) == 0)
	c.setFlagN(false)
	c.setFlagH((a&0x0F)+(uint16(val)&0x0F)+uint16(cy) > 0x0F)
	c.setFlagC(r > 0xFF)
}

func (c *Core) subA(val byte) {
	a := uint16(c.A())
	r := a - uint16(val)
	c.setA(byte(r))
	c.setFlagZ(byte(r) == 0)
	c.setFlagN(true)
	c.setFlagH((a & 0x0F) < (uint16(val) & 0x0F))
	c.setFlagC(r > 0xFF)
}

func (c *Core) sbcA(val byte) {
	cy := boolToByte(c.flagC())
	a := uint16(c.A())
	r := a - uint16(val) - uint16(cy)
	c.setA(byte(r))
	c.setFlagZ(byte(r) == 0)
	c.setFlagN(true)
	c.setFlagH((a & 0x0F) < (uint16(val)&0x0F)+uint16(cy))
	c.setFlagC(r > 0xFF)
}

func (c *Core) andA(val byte) {
	r := c.A() & val
	c.setA(r)
	c.setFlagZ(r == 0); c.setFlagN(false); c.setFlagH(true); c.setFlagC(false)
}

func (c *Core) orA(val byte) {
	r := c.A() | val
	c.setA(r)
	c.setFlagZ(r == 0); c.setFlagN(false); c.setFlagH(false); c.setFlagC(false)
}

func (c *Core) xorA(val byte) {
	r := c.A() ^ val
	c.setA(r)
	c.setFlagZ(r == 0); c.setFlagN(false); c.setFlagH(false); c.setFlagC(false)
}

func (c *Core) cpA(val byte) {
	a := uint16(c.A())
	r := a - uint16(val)
	c.setFlagZ(byte(r) == 0)
	c.setFlagN(true)
	c.setFlagH((a & 0x0F) < (uint16(val) & 0x0F))
	c.setFlagC(r > 0xFF)
}

func (c *Core) inc8(val byte) byte {
	r := val + 1
	c.setFlagZ(r == 0)
	c.setFlagN(false)
	c.setFlagH((val&0x0F)+1 > 0x0F)
	return r
}

func (c *Core) dec8(val byte) byte {
	r := val - 1
	c.setFlagZ(r == 0)
	c.setFlagN(true)
	c.setFlagH((val & 0x0F) == 0)
	return r
}

func (c *Core) addHL(val uint16) {
	r := uint32(c.HL) + uint32(val)
	c.setFlagN(false)
	c.setFlagH((c.HL&0x0FFF)+(val&0x0FFF) > 0x0FFF)
	c.setFlagC(r > 0xFFFF)
	c.HL = uint16(r)
}

func (c *Core) Step() (int, error) {
	if c.serveInterrupt() {
		return 20, nil
	}
	if c.Halted {
		c.Cycles += 4
		if (c.MMU.Read(0xFF0F)&c.MMU.Read(0xFFFF)) != 0 {
			c.Halted = false
		}
		return 4, nil
	}
	if c.Stopped {
		c.Cycles += 4
		return 4, nil
	}
	opcode := c.fetch8()
	// HALT bug: the instruction after HALT is executed a second time
	// because the opcode fetch's PC increment was suppressed.
	// Execute the handler now but restore PC so the next Step()
	// re-fetches the same opcode.
	if c.HaltBug {
		c.HaltBug = false
		c.PC-- // undo fetch8 increment
		savedPC := c.PC
		h := mainHandler[opcode]
		if h == nil {
			c.Cycles += 4
			return 4, nil
		}
		cycles, err := h(c)
		c.PC = savedPC // restore so next fetch re-reads same opcode
		c.Cycles += uint64(cycles)

		if c.IMEScheduled == 2 {
			c.IMEScheduled = 1
		} else if c.IMEScheduled == 1 {
			c.IME = true
			c.IMEScheduled = 0
		}
		return cycles, err
	}
	h := mainHandler[opcode]
	if h == nil {
		c.Cycles += 4
		return 4, nil
	}
	cycles, err := h(c)
	c.Cycles += uint64(cycles)

	// Handle EI delay (after instruction execution, not during EI's own step)
	// IMEScheduled=2 after EI, becomes 1 at end of EI's step,
	// becomes 0 at end of next instruction, then IME becomes true.
	if c.IMEScheduled == 2 {
		c.IMEScheduled = 1
	} else if c.IMEScheduled == 1 {
		c.IME = true
		c.IMEScheduled = 0
	}

	return cycles, err
}

func (c *Core) serveInterrupt() bool {
	if !c.IME {
		return false
	}
	req := c.MMU.Read(0xFF0F)
	en := c.MMU.Read(0xFFFF)
	t := req & en
	if t == 0 {
		return false
	}
	var vector uint16
	var bit byte
	switch {
	case t&0x01 != 0: vector = 0x0040; bit = 0x01
	case t&0x02 != 0: vector = 0x0048; bit = 0x02
	case t&0x04 != 0: vector = 0x0050; bit = 0x04
	case t&0x08 != 0: vector = 0x0058; bit = 0x08
	case t&0x10 != 0: vector = 0x0060; bit = 0x10
	}
	c.IME = false
	c.IMEScheduled = 0
	c.push16(c.PC)
	c.PC = vector
	c.MMU.Write(0xFF0F, req & ^bit)
	c.Halted = false
	c.Cycles += 20
	return true
}

func (c *Core) Reset() {
	c.AF = 0x01B0; c.BC = 0x0013; c.DE = 0x00D8
	c.HL = 0x014D; c.SP = 0xFFFE; c.PC = 0x0100
	c.IME = false; c.IMEScheduled = 0
	c.Halted = false; c.Stopped = false; c.HaltBug = false; c.Cycles = 0
}

func (c *Core) GetState() CPUState {
	return CPUState{
		AF: c.AF, BC: c.BC, DE: c.DE, HL: c.HL,
		SP: c.SP, PC: c.PC, IME: c.IME,
		Halted: c.Halted, Stopped: c.Stopped, Cycles: c.Cycles,
	}
}

// NewCoreCPU is an alias for NewCore (ALU test compatibility).
func NewCoreCPU(mmu MMU) *Core { return NewCore(mmu) }

// NewCPU is an alias for NewCore (CB test compatibility).
func NewCPU(mmu MMU) *Core { return NewCore(mmu) }

// 16-bit register accessors (test compatibility).
func (c *Core) GetBC() uint16 { return c.BC }
func (c *Core) GetDE() uint16 { return c.DE }
func (c *Core) GetHL() uint16 { return c.HL }
func (c *Core) SetBC(v uint16) { c.BC = v }
func (c *Core) SetDE(v uint16) { c.DE = v }
func (c *Core) SetHL(v uint16) { c.HL = v }

// Lowercase getters for ALU test compatibility.
func (c *Core) getA() byte { return c.A() }
func (c *Core) getB() byte { return c.B() }
func (c *Core) getC() byte { return c.C() }
func (c *Core) getD() byte { return c.D() }
func (c *Core) getE() byte { return c.E() }
func (c *Core) getF() byte { return c.F() }
func (c *Core) getH() byte { return c.H() }
func (c *Core) getL() byte { return c.L() }

// ALU wrappers without the 'A' suffix (test compatibility).
func (c *Core) add(val byte)   { c.addA(val) }
func (c *Core) sub(val byte)   { c.subA(val) }
func (c *Core) and(val byte)   { c.andA(val) }
func (c *Core) or(val byte)    { c.orA(val) }
func (c *Core) xor(val byte)   { c.xorA(val) }
func (c *Core) cp(val byte)    { c.cpA(val) }

// inc returns (result, zero, half-carry) for ALU test compatibility.
func (c *Core) inc(val byte) (byte, bool, bool) {
	r := c.inc8(val)
	return r, c.flagZ(), c.flagH()
}

// dec returns (result, zero, half-borrow) for ALU test compatibility.
func (c *Core) dec(val byte) (byte, bool, bool) {
	r := c.dec8(val)
	return r, c.flagZ(), c.flagH()
}

// ALU (HL) memory-indirect wrappers (test compatibility).
func (c *Core) addAHL()  { c.addA(c.MMU.Read(c.HL)) }
func (c *Core) subAHL()  { c.subA(c.MMU.Read(c.HL)) }
func (c *Core) andAHL()  { c.andA(c.MMU.Read(c.HL)) }
func (c *Core) orHL()   { c.orA(c.MMU.Read(c.HL)) }
func (c *Core) xorHL()  { c.xorA(c.MMU.Read(c.HL)) }
func (c *Core) cpHL()   { c.cpA(c.MMU.Read(c.HL)) }

// StepCycles returns the cycle count (for tests).
func (c *Core) StepCycles() int { return int(c.Cycles) }

// Flags returns the current flag states (for test assertions).
func (c *Core) Flags() struct{ Z, N, H, C bool } {
	return struct{ Z, N, H, C bool }{c.flagZ(), c.flagN(), c.flagH(), c.flagC()}
}
