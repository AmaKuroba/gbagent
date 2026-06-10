package gb

// mainHandler maps each opcode to its execution handler.
var mainHandler [256]func(c *Core) (int, error)

func init() {
	initMainHandlers()
}

func initMainHandlers() {
	// ===== 8-bit LD r8,r8 =====
	// Opcodes 0x40-0x7F: LD dst,src where dst and src are encoded as dst<<3|src
	// Register indices: 0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A
	for dst := 0; dst < 8; dst++ {
		for src := 0; src < 8; src++ {
			op := 0x40 | dst<<3 | src
			if dst == 6 && src == 6 {
				// 0x76 = HALT, handled separately
				continue
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
	mainHandler[0x76] = haltHandler

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

	// LDH (a8),A
	mainHandler[0xE0] = func(c *Core) (int, error) {
		c.MMU.Write(0xFF00|uint16(c.fetch8()), c.A())
		return 12, nil
	}
	// LDH A,(a8)
	mainHandler[0xF0] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(0xFF00 | uint16(c.fetch8())))
		return 12, nil
	}
	// LDH (C),A
	mainHandler[0xE2] = func(c *Core) (int, error) {
		c.MMU.Write(0xFF00|uint16(c.C()), c.A())
		return 8, nil
	}
	// LDH A,(C)
	mainHandler[0xF2] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(0xFF00 | uint16(c.C())))
		return 8, nil
	}
	// LD (a16),A
	mainHandler[0xEA] = func(c *Core) (int, error) {
		c.MMU.Write(c.fetch16(), c.A())
		return 16, nil
	}
	// LD A,(a16)
	mainHandler[0xFA] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.fetch16()))
		return 16, nil
	}
	// LD SP,HL
	mainHandler[0xF9] = func(c *Core) (int, error) { c.SP = c.HL; return 8, nil }
	// LD HL,SP+r8
	mainHandler[0xF8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		result := uint32(int32(c.SP) + int32(e))
		c.HL = uint16(result)
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(uint8(e))&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(uint8(e))&0xFF) > 0xFF)
		return 12, nil
	}

	// ===== PUSH =====
	mainHandler[0xC5] = func(c *Core) (int, error) { c.push16(c.BC); return 16, nil }
	mainHandler[0xD5] = func(c *Core) (int, error) { c.push16(c.DE); return 16, nil }
	mainHandler[0xE5] = func(c *Core) (int, error) { c.push16(c.HL); return 16, nil }
	mainHandler[0xF5] = func(c *Core) (int, error) { c.push16(c.AF); return 16, nil }

	// ===== POP =====
	mainHandler[0xC1] = func(c *Core) (int, error) { c.BC = c.pop16(); return 12, nil }
	mainHandler[0xD1] = func(c *Core) (int, error) { c.DE = c.pop16(); return 12, nil }
	mainHandler[0xE1] = func(c *Core) (int, error) { c.HL = c.pop16(); return 12, nil }
	mainHandler[0xF1] = func(c *Core) (int, error) {
		c.AF = c.pop16() & 0xFFF0 // Preserve flags low nibble
		return 12, nil
	}

	// ===== 8-bit ALU =====
	setALUGroup(0x80, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.addA(v) }) // ADD A,B
	setALUGroup(0x81, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x82, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x83, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x84, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x85, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x86, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.addA(v) })
	setALUGroup(0x87, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.addA(v) })

	setALUGroup(0x88, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.adcA(v) }) // ADC A,B
	setALUGroup(0x89, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8A, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8B, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8C, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8D, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8E, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.adcA(v) })
	setALUGroup(0x8F, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.adcA(v) })

	setALUGroup(0x90, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.subA(v) }) // SUB B
	setALUGroup(0x91, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x92, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x93, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x94, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x95, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x96, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.subA(v) })
	setALUGroup(0x97, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.subA(v) })

	setALUGroup(0x98, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.sbcA(v) }) // SBC A,B
	setALUGroup(0x99, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9A, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9B, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9C, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9D, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9E, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.sbcA(v) })
	setALUGroup(0x9F, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.sbcA(v) })

	setALUGroup(0xA0, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.andA(v) }) // AND B
	setALUGroup(0xA1, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA2, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA3, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA4, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA5, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA6, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.andA(v) })
	setALUGroup(0xA7, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.andA(v) })

	setALUGroup(0xA8, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.xorA(v) }) // XOR B
	setALUGroup(0xA9, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAA, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAB, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAC, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAD, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAE, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.xorA(v) })
	setALUGroup(0xAF, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.xorA(v) })

	setALUGroup(0xB0, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.orA(v) }) // OR B
	setALUGroup(0xB1, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB2, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB3, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB4, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB5, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB6, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.orA(v) })
	setALUGroup(0xB7, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.orA(v) })

	setALUGroup(0xB8, func(c *Core) byte { return c.readReg8(0) }, func(c *Core, v byte) { c.cpA(v) }) // CP B
	setALUGroup(0xB9, func(c *Core) byte { return c.readReg8(1) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBA, func(c *Core) byte { return c.readReg8(2) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBB, func(c *Core) byte { return c.readReg8(3) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBC, func(c *Core) byte { return c.readReg8(4) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBD, func(c *Core) byte { return c.readReg8(5) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBE, func(c *Core) byte { return c.readReg8(6) }, func(c *Core, v byte) { c.cpA(v) })
	setALUGroup(0xBF, func(c *Core) byte { return c.readReg8(7) }, func(c *Core, v byte) { c.cpA(v) })

	// ADD A,d8
	mainHandler[0xC6] = func(c *Core) (int, error) { c.addA(c.fetch8()); return 8, nil }
	// ADC A,d8
	mainHandler[0xCE] = func(c *Core) (int, error) { c.adcA(c.fetch8()); return 8, nil }
	// SUB d8
	mainHandler[0xD6] = func(c *Core) (int, error) { c.subA(c.fetch8()); return 8, nil }
	// SBC A,d8
	mainHandler[0xDE] = func(c *Core) (int, error) { c.sbcA(c.fetch8()); return 8, nil }
	// AND d8
	mainHandler[0xE6] = func(c *Core) (int, error) { c.andA(c.fetch8()); return 8, nil }
	// OR d8
	mainHandler[0xF6] = func(c *Core) (int, error) { c.orA(c.fetch8()); return 8, nil }
	// XOR d8
	mainHandler[0xEE] = func(c *Core) (int, error) { c.xorA(c.fetch8()); return 8, nil }
	// CP d8
	mainHandler[0xFE] = func(c *Core) (int, error) { c.cpA(c.fetch8()); return 8, nil }

	// ===== INC r8 =====
	mainHandler[0x04] = func(c *Core) (int, error) { c.setB(c.inc8(c.B())); return 4, nil }
	mainHandler[0x0C] = func(c *Core) (int, error) { c.setC(c.inc8(c.C())); return 4, nil }
	mainHandler[0x14] = func(c *Core) (int, error) { c.setD(c.inc8(c.D())); return 4, nil }
	mainHandler[0x1C] = func(c *Core) (int, error) { c.setE(c.inc8(c.E())); return 4, nil }
	mainHandler[0x24] = func(c *Core) (int, error) { c.setH(c.inc8(c.H())); return 4, nil }
	mainHandler[0x2C] = func(c *Core) (int, error) { c.setL(c.inc8(c.L())); return 4, nil }
	mainHandler[0x34] = func(c *Core) (int, error) {
		c.MMU.Write(c.HL, c.inc8(c.MMU.Read(c.HL)))
		return 12, nil
	}
	mainHandler[0x3C] = func(c *Core) (int, error) { c.setA(c.inc8(c.A())); return 4, nil }

	// ===== DEC r8 =====
	mainHandler[0x05] = func(c *Core) (int, error) { c.setB(c.dec8(c.B())); return 4, nil }
	mainHandler[0x0D] = func(c *Core) (int, error) { c.setC(c.dec8(c.C())); return 4, nil }
	mainHandler[0x15] = func(c *Core) (int, error) { c.setD(c.dec8(c.D())); return 4, nil }
	mainHandler[0x1D] = func(c *Core) (int, error) { c.setE(c.dec8(c.E())); return 4, nil }
	mainHandler[0x25] = func(c *Core) (int, error) { c.setH(c.dec8(c.H())); return 4, nil }
	mainHandler[0x2D] = func(c *Core) (int, error) { c.setL(c.dec8(c.L())); return 4, nil }
	mainHandler[0x35] = func(c *Core) (int, error) {
		c.MMU.Write(c.HL, c.dec8(c.MMU.Read(c.HL)))
		return 12, nil
	}
	mainHandler[0x3D] = func(c *Core) (int, error) { c.setA(c.dec8(c.A())); return 4, nil }

	// ===== 16-bit INC/DEC =====
	mainHandler[0x03] = func(c *Core) (int, error) { c.BC++; return 8, nil }
	mainHandler[0x13] = func(c *Core) (int, error) { c.DE++; return 8, nil }
	mainHandler[0x23] = func(c *Core) (int, error) { c.HL++; return 8, nil }
	mainHandler[0x33] = func(c *Core) (int, error) { c.SP++; return 8, nil }

	mainHandler[0x0B] = func(c *Core) (int, error) { c.BC--; return 8, nil }
	mainHandler[0x1B] = func(c *Core) (int, error) { c.DE--; return 8, nil }
	mainHandler[0x2B] = func(c *Core) (int, error) { c.HL--; return 8, nil }
	mainHandler[0x3B] = func(c *Core) (int, error) { c.SP--; return 8, nil }

	// ===== ADD HL,r16 =====
	mainHandler[0x09] = func(c *Core) (int, error) { c.addHL(c.BC); return 8, nil }
	mainHandler[0x19] = func(c *Core) (int, error) { c.addHL(c.DE); return 8, nil }
	mainHandler[0x29] = func(c *Core) (int, error) { c.addHL(c.HL); return 8, nil }
	mainHandler[0x39] = func(c *Core) (int, error) { c.addHL(c.SP); return 8, nil }

	// ADD SP,r8
	mainHandler[0xE8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		result := uint32(int32(c.SP) + int32(e))
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(uint8(e))&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(uint8(e))&0xFF) > 0xFF)
		c.SP = uint16(result)
		return 16, nil
	}

	// ===== Control flow =====

	// JP a16
	mainHandler[0xC3] = func(c *Core) (int, error) { c.PC = c.fetch16(); return 16, nil }
	// JP (HL)
	mainHandler[0xE9] = func(c *Core) (int, error) { c.PC = c.HL; return 4, nil }

	// JP cc,a16
	mainHandler[0xC2] = func(c *Core) (int, error) { addr := c.fetch16(); if !c.flagZ() { c.PC = addr; return 16 }; return 12, nil }
	mainHandler[0xCA] = func(c *Core) (int, error) { addr := c.fetch16(); if c.flagZ() { c.PC = addr; return 16 }; return 12, nil }
	mainHandler[0xD2] = func(c *Core) (int, error) { addr := c.fetch16(); if !c.flagC() { c.PC = addr; return 16 }; return 12, nil }
	mainHandler[0xDA] = func(c *Core) (int, error) { addr := c.fetch16(); if c.flagC() { c.PC = addr; return 16 }; return 12, nil }

	// JR r8
	mainHandler[0x18] = func(c *Core) (int, error) {
		c.PC = uint16(int32(c.PC) + int32(int8(c.fetch8())))
		return 12, nil
	}
	// JR cc,r8
	mainHandler[0x20] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if !c.flagZ() { c.PC = uint16(int32(c.PC) + int32(e)); return 12 }
		return 8, nil
	}
	mainHandler[0x28] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if c.flagZ() { c.PC = uint16(int32(c.PC) + int32(e)); return 12 }
		return 8, nil
	}
	mainHandler[0x30] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if !c.flagC() { c.PC = uint16(int32(c.PC) + int32(e)); return 12 }
		return 8, nil
	}
	mainHandler[0x38] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		if c.flagC() { c.PC = uint16(int32(c.PC) + int32(e)); return 12 }
		return 8, nil
	}

	// CALL a16
	mainHandler[0xCD] = func(c *Core) (int, error) {
		addr := c.fetch16()
		c.push16(c.PC)
		c.PC = addr
		return 24, nil
	}
	// CALL cc,a16
	mainHandler[0xC4] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagZ() { c.push16(c.PC); c.PC = addr; return 24 }
		return 12, nil
	}
	mainHandler[0xCC] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagZ() { c.push16(c.PC); c.PC = addr; return 24 }
		return 12, nil
	}
	mainHandler[0xD4] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if !c.flagC() { c.push16(c.PC); c.PC = addr; return 24 }
		return 12, nil
	}
	mainHandler[0xDC] = func(c *Core) (int, error) {
		addr := c.fetch16()
		if c.flagC() { c.push16(c.PC); c.PC = addr; return 24 }
		return 12, nil
	}

	// RET
	mainHandler[0xC9] = func(c *Core) (int, error) { c.PC = c.pop16(); return 16, nil }
	// RET cc
	mainHandler[0xC0] = func(c *Core) (int, error) {
		if !c.flagZ() { c.PC = c.pop16(); return 20 }
		return 8, nil
	}
	mainHandler[0xC8] = func(c *Core) (int, error) {
		if c.flagZ() { c.PC = c.pop16(); return 20 }
		return 8, nil
	}
	mainHandler[0xD0] = func(c *Core) (int, error) {
		if !c.flagC() { c.PC = c.pop16(); return 20 }
		return 8, nil
	}
	mainHandler[0xD8] = func(c *Core) (int, error) {
		if c.flagC() { c.PC = c.pop16(); return 20 }
		return 8, nil
	}
	// RETI
	mainHandler[0xD9] = func(c *Core) (int, error) {
		c.PC = c.pop16()
		c.IME = true
		return 16, nil
	}

	// RST
	mainHandler[0xC7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x00; return 16, nil }
	mainHandler[0xCF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x08; return 16, nil }
	mainHandler[0xD7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x10; return 16, nil }
	mainHandler[0xDF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x18; return 16, nil }
	mainHandler[0xE7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x20; return 16, nil }
	mainHandler[0xEF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x28; return 16, nil }
	mainHandler[0xF7] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x30; return 16, nil }
	mainHandler[0xFF] = func(c *Core) (int, error) { c.push16(c.PC); c.PC = 0x38; return 16, nil }

	// ===== Rotates & misc =====

	// NOP
	mainHandler[0x00] = func(c *Core) (int, error) { return 4, nil }

	// RLCA
	mainHandler[0x07] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x80 != 0)
		a = (a << 1) | (a >> 7)
		c.setA(a)
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RRCA
	mainHandler[0x0F] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x01 != 0)
		a = (a >> 1) | (a << 7)
		c.setA(a)
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RLA
	mainHandler[0x17] = func(c *Core) (int, error) {
		a, oldC := c.A(), c.flagC()
		c.setFlagC(a&0x80 != 0)
		a = (a << 1) | boolToByte(oldC)
		c.setA(a)
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RRA
	mainHandler[0x1F] = func(c *Core) (int, error) {
		a, oldC := c.A(), c.flagC()
		c.setFlagC(a&0x01 != 0)
		a = (a >> 1) | (boolToByte(oldC) << 7)
		c.setA(a)
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}

	// DAA
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

	// CPL
	mainHandler[0x2F] = func(c *Core) (int, error) {
		c.setA(^c.A())
		c.setFlagN(true); c.setFlagH(true)
		return 4, nil
	}
	// SCF
	mainHandler[0x37] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false); c.setFlagC(true)
		return 4, nil
	}
	// CCF
	mainHandler[0x3F] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false)
		c.setFlagC(!c.flagC())
		return 4, nil
	}

	// DI
	mainHandler[0xF3] = func(c *Core) (int, error) {
		c.IME = false
		c.IMEEnablePending = false
		return 4, nil
	}
	// EI
	mainHandler[0xFB] = func(c *Core) (int, error) {
		c.IMEEnablePending = true
		return 4, nil
	}

	// STOP
	mainHandler[0x10] = func(c *Core) (int, error) {
		c.fetch8() // consume immediate byte (usually 0x00)
		c.Stopped = true
		return 4, nil
	}

	// PREFIX CB
	mainHandler[0xCB] = func(c *Core) (int, error) {
		op := c.fetch8()
		return cbHandler[op](c)
	}

	// Undefined opcodes remain nil — Core.Step handles them as NOP
}

// setALUGroup sets up a handler for ALU operations on registers.
func setALUGroup(op int, getVal func(c *Core) byte, aluFn func(c *Core, v byte)) {
	mainHandler[op] = func(c *Core) (int, error) {
		val := getVal(c)
		aluFn(c, val)
		// Check if this is a (HL) variant: opcodes ending in 6
		if op&0x07 == 6 {
			return 8, nil
		}
		return 4, nil
	}
}

func haltHandler(c *Core) (int, error) {
	c.Halted = true
	// HALT bug: if IME=0 and an interrupt is pending, skip next byte
	if !c.IME && (c.MMU.ReadIF()&c.MMU.ReadIE()) != 0 {
		c.Halted = false
		c.PC++ // skip next instruction byte
	}
	return 4, nil
}
