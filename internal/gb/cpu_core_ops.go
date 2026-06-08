package gb

// mainHandler maps each opcode to its execution handler.
// nil entries are undefined opcodes; Step() treats them as NOP.
var mainHandler [256]func(c *Core) (int, error)

func init() { initMainHandlers() }

func initMainHandlers() {
	// ===== NOP (0x00) =====
	mainHandler[0x00] = func(c *Core) (int, error) { return 4, nil }

	// ===== 8-bit LD r8,r8 (0x40-0x7F) =====
	// Register indices: 0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A
	for dst := 0; dst < 8; dst++ {
		for src := 0; src < 8; src++ {
			op := 0x40 | dst<<3 | src
			if dst == 6 && src == 6 {
				continue // 0x76 = HALT
			}
			d, s := dst, src
			mainHandler[op] = func(c *Core) (int, error) {
				c.writeReg8(d, c.readReg8(s))
				if d == 6 || s == 6 {
					return 8, nil
				}
				return 4, nil
			}
		}
	}

	// HALT (0x76)
	mainHandler[0x76] = func(c *Core) (int, error) {
		c.Halted = true
		if !c.IME && (c.MMU.Read(0xFF0F)&c.MMU.Read(0xFFFF)) != 0 {
			c.Halted = false
			c.PC++ // HALT bug: skip next byte
		}
		return 4, nil
	}

	// LD r8,d8
	for _, e := range [][2]int{{0x06, 0}, {0x0E, 1}, {0x16, 2}, {0x1E, 3}, {0x26, 4}, {0x2E, 5}, {0x36, 6}, {0x3E, 7}} {
		op, r := e[0], e[1]
		mainHandler[op] = func(c *Core) (int, error) {
			val := c.fetch8()
			c.writeReg8(r, val)
			if r == 6 {
				return 12, nil
			}
			return 8, nil
		}
	}

	// ===== 16-bit LD =====
	mainHandler[0x01] = func(c *Core) (int, error) { c.BC = c.fetch16(); return 12, nil }
	mainHandler[0x11] = func(c *Core) (int, error) { c.DE = c.fetch16(); return 12, nil }
	mainHandler[0x21] = func(c *Core) (int, error) { c.HL = c.fetch16(); return 12, nil }
	mainHandler[0x31] = func(c *Core) (int, error) { c.SP = c.fetch16(); return 12, nil }

	// LD (a16),SP
	mainHandler[0x08] = func(c *Core) (int, error) {
		addr := c.fetch16()
		c.MMU.Write16(addr, c.SP)
		return 20, nil
	}

	// LD (BC),A / LD (DE),A
	mainHandler[0x02] = func(c *Core) (int, error) { c.MMU.Write(c.BC, c.A()); return 8, nil }
	mainHandler[0x12] = func(c *Core) (int, error) { c.MMU.Write(c.DE, c.A()); return 8, nil }

	// LD A,(BC) / LD A,(DE)
	mainHandler[0x0A] = func(c *Core) (int, error) { c.setA(c.MMU.Read(c.BC)); return 8, nil }
	mainHandler[0x1A] = func(c *Core) (int, error) { c.setA(c.MMU.Read(c.DE)); return 8, nil }

	// LDI (HL),A / LDI A,(HL)
	mainHandler[0x22] = func(c *Core) (int, error) { c.MMU.Write(c.HL, c.A()); c.HL++; return 8, nil }
	mainHandler[0x2A] = func(c *Core) (int, error) { c.setA(c.MMU.Read(c.HL)); c.HL++; return 8, nil }

	// LDD (HL),A / LDD A,(HL)
	mainHandler[0x32] = func(c *Core) (int, error) { c.MMU.Write(c.HL, c.A()); c.HL--; return 8, nil }
	mainHandler[0x3A] = func(c *Core) (int, error) { c.setA(c.MMU.Read(c.HL)); c.HL--; return 8, nil }

	// LDH (a8),A (0xE0)
	mainHandler[0xE0] = func(c *Core) (int, error) {
		c.MMU.Write(0xFF00|uint16(c.fetch8()), c.A())
		return 12, nil
	}

	// LDH A,(a8) (0xF0)
	mainHandler[0xF0] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(0xFF00 | uint16(c.fetch8())))
		return 12, nil
	}

	// LDH (C),A (0xE2)
	mainHandler[0xE2] = func(c *Core) (int, error) {
		c.MMU.Write(0xFF00|uint16(c.C()), c.A())
		return 8, nil
	}

	// LDH A,(C) (0xF2)
	mainHandler[0xF2] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(0xFF00 | uint16(c.C())))
		return 8, nil
	}

	// LD (a16),A (0xEA)
	mainHandler[0xEA] = func(c *Core) (int, error) {
		c.MMU.Write(c.fetch16(), c.A())
		return 16, nil
	}

	// LD A,(a16) (0xFA)
	mainHandler[0xFA] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.fetch16()))
		return 16, nil
	}

	// LD SP,HL (0xF9)
	mainHandler[0xF9] = func(c *Core) (int, error) { c.SP = c.HL; return 8, nil }

	// LD HL,SP+r8 (0xF8)
	mainHandler[0xF8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		result := uint32(int32(c.SP) + int32(e))
		c.HL = uint16(result)
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(e)&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(e)&0xFF) > 0xFF)
		return 12, nil
	}

	// ===== PUSH / POP =====
	for _, e := range [][4]int{
		{0xC5, 0, 0, 0}, // PUSH BC — placeholder
		{0xD5, 0, 0, 0}, // PUSH DE
		{0xE5, 0, 0, 0}, // PUSH HL
		{0xF5, 0, 0, 0}, // PUSH AF
	} {
		op := e[0]
		switch op {
		case 0xC5:
			mainHandler[0xC5] = func(c *Core) (int, error) { c.push16(c.BC); return 16, nil }
		case 0xD5:
			mainHandler[0xD5] = func(c *Core) (int, error) { c.push16(c.DE); return 16, nil }
		case 0xE5:
			mainHandler[0xE5] = func(c *Core) (int, error) { c.push16(c.HL); return 16, nil }
		case 0xF5:
			mainHandler[0xF5] = func(c *Core) (int, error) { c.push16(c.AF); return 16, nil }
		}
	}

	for _, e := range [][4]int{
		{0xC1, 0, 0, 0}, // POP BC
		{0xD1, 0, 0, 0}, // POP DE
		{0xE1, 0, 0, 0}, // POP HL
		{0xF1, 0, 0, 0}, // POP AF
	} {
		op := e[0]
		switch op {
		case 0xC1:
			mainHandler[0xC1] = func(c *Core) (int, error) { c.BC = c.pop16(); return 12, nil }
		case 0xD1:
			mainHandler[0xD1] = func(c *Core) (int, error) { c.DE = c.pop16(); return 12, nil }
		case 0xE1:
			mainHandler[0xE1] = func(c *Core) (int, error) { c.HL = c.pop16(); return 12, nil }
		case 0xF1:
			mainHandler[0xF1] = func(c *Core) (int, error) {
				c.AF = c.pop16() & 0xFFF0
				return 12, nil
			}
		}
	}

	// ===== 8-bit ALU =====
	for i := 0; i < 8; i++ {
		op, r := 0x80+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.addA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xC6] = func(c *Core) (int, error) { c.addA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0x88+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.adcA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xCE] = func(c *Core) (int, error) { c.adcA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0x90+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.subA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xD6] = func(c *Core) (int, error) { c.subA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0x98+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.sbcA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xDE] = func(c *Core) (int, error) { c.sbcA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0xA0+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.andA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xE6] = func(c *Core) (int, error) { c.andA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0xB0+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.orA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xF6] = func(c *Core) (int, error) { c.orA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0xA8+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.xorA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xEE] = func(c *Core) (int, error) { c.xorA(c.fetch8()); return 8, nil }

	for i := 0; i < 8; i++ {
		op, r := 0xB8+i, i
		mainHandler[op] = func(c *Core) (int, error) {
			c.cpA(c.readReg8(r))
			if r == 6 {
				return 8, nil
			}
			return 4, nil
		}
	}
	mainHandler[0xFE] = func(c *Core) (int, error) { c.cpA(c.fetch8()); return 8, nil }

	// INC r8
	for _, e := range [][2]int{{0x04, 0}, {0x0C, 1}, {0x14, 2}, {0x1C, 3}, {0x24, 4}, {0x2C, 5}, {0x34, 6}, {0x3C, 7}} {
		op, r := e[0], e[1]
		mainHandler[op] = func(c *Core) (int, error) {
			c.writeReg8(r, c.inc8(c.readReg8(r)))
			if r == 6 {
				return 12, nil
			}
			return 4, nil
		}
	}

	// DEC r8
	for _, e := range [][2]int{{0x05, 0}, {0x0D, 1}, {0x15, 2}, {0x1D, 3}, {0x25, 4}, {0x2D, 5}, {0x35, 6}, {0x3D, 7}} {
		op, r := e[0], e[1]
		mainHandler[op] = func(c *Core) (int, error) {
			c.writeReg8(r, c.dec8(c.readReg8(r)))
			if r == 6 {
				return 12, nil
			}
			return 4, nil
		}
	}

	// ===== 16-bit ALU =====
	mainHandler[0x09] = func(c *Core) (int, error) { c.addHL(c.BC); return 8, nil }
	mainHandler[0x19] = func(c *Core) (int, error) { c.addHL(c.DE); return 8, nil }
	mainHandler[0x29] = func(c *Core) (int, error) { c.addHL(c.HL); return 8, nil }
	mainHandler[0x39] = func(c *Core) (int, error) { c.addHL(c.SP); return 8, nil }

	mainHandler[0x03] = func(c *Core) (int, error) { c.BC++; return 8, nil }
	mainHandler[0x13] = func(c *Core) (int, error) { c.DE++; return 8, nil }
	mainHandler[0x23] = func(c *Core) (int, error) { c.HL++; return 8, nil }
	mainHandler[0x33] = func(c *Core) (int, error) { c.SP++; return 8, nil }

	mainHandler[0x0B] = func(c *Core) (int, error) { c.BC--; return 8, nil }
	mainHandler[0x1B] = func(c *Core) (int, error) { c.DE--; return 8, nil }
	mainHandler[0x2B] = func(c *Core) (int, error) { c.HL--; return 8, nil }
	mainHandler[0x3B] = func(c *Core) (int, error) { c.SP--; return 8, nil }

	// ADD SP,r8 (0xE8)
	mainHandler[0xE8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		result := uint32(int32(c.SP) + int32(e))
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(e)&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(e)&0xFF) > 0xFF)
		c.SP = uint16(result)
		return 16, nil
	}

	// ===== Control flow =====
	mainHandler[0xC3] = func(c *Core) (int, error) { c.PC = c.fetch16(); return 16, nil }
	mainHandler[0xE9] = func(c *Core) (int, error) { c.PC = c.HL; return 4, nil }

	// JP cc,a16
	mainHandler[0xC2] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagZ() {
			c.PC = addr
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xCA] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagZ() {
			c.PC = addr
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xD2] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagC() {
			c.PC = addr
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xDA] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagC() {
			c.PC = addr
			return 16, nil
		}
		return 12, nil
	}

	// JR r8 (0x18)
	mainHandler[0x18] = func(c *Core) (int, error) {
		c.PC = uint16(int32(c.PC) + int32(int8(c.fetch8())))
		return 12, nil
	}

	// JR cc,r8
	mainHandler[0x20] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if !c.flagZ() {
			c.PC = uint16(int32(c.PC) + int32(e))
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x28] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if c.flagZ() {
			c.PC = uint16(int32(c.PC) + int32(e))
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x30] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if !c.flagC() {
			c.PC = uint16(int32(c.PC) + int32(e))
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x38] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if c.flagC() {
			c.PC = uint16(int32(c.PC) + int32(e))
			return 12, nil
		}
		return 8, nil
	}

	// CALL a16 (0xCD)
	mainHandler[0xCD] = func(c *Core) (int, error) {
		addr := c.fetch16()
		c.push16(c.PC)
		c.PC = addr
		return 24, nil
	}

	// CALL cc,a16
	mainHandler[0xC4] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagZ() {
			c.push16(c.PC)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xCC] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagZ() {
			c.push16(c.PC)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xD4] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagC() {
			c.push16(c.PC)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xDC] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagC() {
			c.push16(c.PC)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}

	// RET (0xC9)
	mainHandler[0xC9] = func(c *Core) (int, error) { c.PC = c.pop16(); return 16, nil }

	// RET cc
	mainHandler[0xC0] = func(c *Core) (int, error) {
		if !c.flagZ() {
			c.PC = c.pop16()
			return 20, nil
		}
		return 8, nil
	}
	mainHandler[0xC8] = func(c *Core) (int, error) {
		if c.flagZ() {
			c.PC = c.pop16()
			return 20, nil
		}
		return 8, nil
	}
	mainHandler[0xD0] = func(c *Core) (int, error) {
		if !c.flagC() {
			c.PC = c.pop16()
			return 20, nil
		}
		return 8, nil
	}
	mainHandler[0xD8] = func(c *Core) (int, error) {
		if c.flagC() {
			c.PC = c.pop16()
			return 20, nil
		}
		return 8, nil
	}

	// RETI (0xD9)
	mainHandler[0xD9] = func(c *Core) (int, error) {
		c.PC = c.pop16()
		c.IME = true
		return 16, nil
	}

	// RST n
	mainHandler[0xC7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x00; return 16, nil }
	mainHandler[0xCF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x08; return 16, nil }
	mainHandler[0xD7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x10; return 16, nil }
	mainHandler[0xDF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x18; return 16, nil }
	mainHandler[0xE7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x20; return 16, nil }
	mainHandler[0xEF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x28; return 16, nil }
	mainHandler[0xF7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x30; return 16, nil }
	mainHandler[0xFF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x38; return 16, nil }

	// ===== Miscellaneous =====
	mainHandler[0x07] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x80 != 0)
		c.setA((a << 1) | (a >> 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	mainHandler[0x0F] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x01 != 0)
		c.setA((a >> 1) | (a << 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	mainHandler[0x17] = func(c *Core) (int, error) {
		a := c.A()
		oldC := c.flagC()
		c.setFlagC(a&0x80 != 0)
		c.setA((a << 1) | boolToByte(oldC))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	mainHandler[0x1F] = func(c *Core) (int, error) {
		a := c.A()
		oldC := c.flagC()
		c.setFlagC(a&0x01 != 0)
		c.setA((a >> 1) | (boolToByte(oldC) << 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}

	// DAA (0x27)
	mainHandler[0x27] = func(c *Core) (int, error) {
		a := c.A()
		var corr byte
		if !c.flagN() {
			if c.flagH() || (a&0x0F) > 0x09 {
				corr |= 0x06
			}
			if c.flagC() || a > 0x99 {
				corr |= 0x60
				c.setFlagC(true)
			}
		} else {
			if c.flagH() {
				corr |= 0x06
			}
			if c.flagC() {
				corr |= 0x60
			}
		}
		if c.flagN() {
			a -= corr
		} else {
			a += corr
		}
		c.setA(a)
		c.setFlagZ(a == 0)
		c.setFlagH(false)
		return 4, nil
	}

	mainHandler[0x2F] = func(c *Core) (int, error) {
		c.setA(^c.A())
		c.setFlagN(true); c.setFlagH(true)
		return 4, nil
	}
	mainHandler[0x37] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false); c.setFlagC(true)
		return 4, nil
	}
	mainHandler[0x3F] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false); c.setFlagC(!c.flagC())
		return 4, nil
	}
	mainHandler[0xF3] = func(c *Core) (int, error) {
		c.IME = false
		c.IMEScheduled = 0
		return 4, nil
	}
	mainHandler[0xFB] = func(c *Core) (int, error) {
		c.IMEScheduled = 2
		return 4, nil
	}
	mainHandler[0x10] = func(c *Core) (int, error) {
		c.fetch8() // consume immediate byte
		c.Stopped = true
		return 4, nil
	}

	// PREFIX CB (0xCB) — dispatch to cbHandler table
	mainHandler[0xCB] = func(c *Core) (int, error) {
		sub := c.fetch8()
		h := cbHandler[sub]
		if h == nil {
			return 4, nil // undefined CB op → NOP
		}
		return h(c)
	}

	// Undefined opcodes remain nil — Step() treats them as NOP.
}
