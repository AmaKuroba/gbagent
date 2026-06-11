package gb

// Core implements the CPU interface for the LR35902 (Sharp SM83) processor.
type Core struct {
	*CPUState
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
	return &Core{
		CPUState: &CPUState{PC: 0x0100, SP: 0xFFFE},
		MMU:      mmu,
	}
}

func (c *Core) A() byte { return byte(c.AF >> 8) }
func (c *Core) F() byte { return byte(c.AF) }
func (c *Core) B() byte { return byte(c.BC >> 8) }
func (c *Core) C() byte { return byte(c.BC) }
func (c *Core) D() byte { return byte(c.DE >> 8) }
func (c *Core) E() byte { return byte(c.DE) }
func (c *Core) H() byte { return byte(c.HL >> 8) }
func (c *Core) L() byte { return byte(c.HL) }

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

func (c *Core) fetch8() byte {
	v := c.MMU.Read(c.PC)
	c.PC++
	return v
}

// push16 writes val to the stack and decrements SP by 2.
// Does NOT call stepDevices — callers handle M-cycle stepping.
func (c *Core) push16(val uint16) {
	c.SP -= 2
	c.MMU.Write16(c.SP, val)
}

func (c *Core) readReg8(idx int) byte {
	switch idx {
	case 0:
		return c.B()
	case 1:
		return c.C()
	case 2:
		return c.D()
	case 3:
		return c.E()
	case 4:
		return c.H()
	case 5:
		return c.L()
	case 6:
		return c.MMU.Read(c.HL)
	case 7:
		return c.A()
	}
	return 0
}

func (c *Core) writeReg8(idx int, val byte) {
	switch idx {
	case 0:
		c.setB(val)
	case 1:
		c.setC(val)
	case 2:
		c.setD(val)
	case 3:
		c.setE(val)
	case 4:
		c.setH(val)
	case 5:
		c.setL(val)
	case 6:
		c.MMU.Write(c.HL, val)
	case 7:
		c.setA(val)
	}
}

func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

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
	c.setFlagZ(r == 0)
	c.setFlagN(false)
	c.setFlagH(true)
	c.setFlagC(false)
}

func (c *Core) orA(val byte) {
	r := c.A() | val
	c.setA(r)
	c.setFlagZ(r == 0)
	c.setFlagN(false)
	c.setFlagH(false)
	c.setFlagC(false)
}

func (c *Core) xorA(val byte) {
	r := c.A() ^ val
	c.setA(r)
	c.setFlagZ(r == 0)
	c.setFlagN(false)
	c.setFlagH(false)
	c.setFlagC(false)
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

// stepDevices advances all devices by the given T-cycles and increments the
// CPU cycle counter. Called after each M-cycle (4 T-cycles) so that PPU, timer,
// APU, DMA, and serial are interleaved correctly with instruction execution.
func (c *Core) stepDevices(cycles int) {
	c.Cycles += uint64(cycles)
	c.MMU.StepDevices(cycles)
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
		c.stepDevices(20)
		return 20, nil
	}
	if c.Halted {
		c.stepDevices(4)
		if (c.MMU.Read(0xFF0F) & c.MMU.Read(0xFFFF)) != 0 {
			c.Halted = false
		}
		return 4, nil
	}
	if c.Stopped {
		c.stepDevices(4)
		if (c.MMU.Read(0xFF0F) & c.MMU.Read(0xFFFF)) != 0 {
			c.Stopped = false
		}
		return 4, nil
	}
	opcode := c.fetch8()

	// HALT bug: the opcode fetch's PC increment was suppressed, causing the
	// instruction after HALT to be fetched and executed a second time.
	// Fetch the byte, undo the increment, execute, then restore PC.
	if c.HaltBug {
		c.HaltBug = false
		c.PC-- // undo fetch8 increment
		savedPC := c.PC
		c.stepDevices(4) // M1: opcode fetch cycle
		h := mainHandler[opcode]
		if h == nil {
			return 4, nil
		}
		cycles, err := h(c)
		c.PC = savedPC // restore so next fetch re-reads same opcode

		switch c.IMEScheduled {
		case 2:
			c.IMEScheduled = 1
		case 1:
			c.IME = true
			c.IMEScheduled = 0
		}
		return cycles, err
	}

	c.stepDevices(4) // M1: opcode fetch cycle (4 T-cycles)
	h := mainHandler[opcode]
	if h == nil {
		return 4, nil
	}
	cycles, err := h(c)

	// Handle EI delay (after instruction execution, not during EI's own step)
	switch c.IMEScheduled {
	case 2:
		c.IMEScheduled = 1
	case 1:
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
	case t&0x01 != 0:
		vector = 0x0040
		bit = 0x01
	case t&0x02 != 0:
		vector = 0x0048
		bit = 0x02
	case t&0x04 != 0:
		vector = 0x0050
		bit = 0x04
	case t&0x08 != 0:
		vector = 0x0058
		bit = 0x08
	case t&0x10 != 0:
		vector = 0x0060
		bit = 0x10
	}
	c.IME = false
	c.IMEScheduled = 0
	c.push16(c.PC)
	c.PC = vector
	c.MMU.Write(0xFF0F, req & ^bit)
	c.Halted = false
	return true
}

func (c *Core) Reset() {
	c.AF = 0x01B0
	c.BC = 0x0013
	c.DE = 0x00D8
	c.HL = 0x014D
	c.SP = 0xFFFE
	c.PC = 0x0100
	c.IME = false
	c.IMEScheduled = 0
	c.Halted = false
	c.Stopped = false
	c.HaltBug = false
	c.Cycles = 0
}

func (c *Core) GetState() CPUState {
	return *c.CPUState
}

func (c *Core) SetState(s CPUState) {
	*c.CPUState = s
}

// NewCoreCPU is an alias for NewCore (ALU test compatibility).
func NewCoreCPU(mmu MMU) *Core { return NewCore(mmu) }

// NewCPU is an alias for NewCore (CB test compatibility).
func NewCPU(mmu MMU) *Core { return NewCore(mmu) }

// 16-bit register accessors (test compatibility).
func (c *Core) GetBC() uint16  { return c.BC }
func (c *Core) GetDE() uint16  { return c.DE }
func (c *Core) GetHL() uint16  { return c.HL }
func (c *Core) SetBC(v uint16) { c.BC = v }
func (c *Core) SetDE(v uint16) { c.DE = v }
func (c *Core) SetHL(v uint16) { c.HL = v }

// StepCycles returns the cycle count (for tests).
func (c *Core) StepCycles() int { return int(c.Cycles) }
