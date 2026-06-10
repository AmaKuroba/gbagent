package gb

// mainHandler maps each opcode to its execution handler.
// nil entries are undefined opcodes; Step() treats them as NOP.
var mainHandler [256]func(c *Core) (int, error)

func init() { initMainHandlers() }

// All handlers follow the M-cycle convention:
//   - M1 (opcode fetch) is handled by Step() which calls c.stepDevices(4)
//   - Handlers are responsible for calling c.stepDevices(4) after each
//     additional M-cycle (M2, M3, ...)
//   - Handlers return the total instruction cost in T-cycles
//   - Memory writes are immediate (not deferred)

func initMainHandlers() {
	initNOP()
	initLDr8r8()
	initLD16()
	initPUSHPOP()
	initALU()
	initINCDEC()
	initALU16()
	initControlFlow()
	initMisc()
}

// ===== NOP (0x00) =====
func initNOP() {
	mainHandler[0x00] = func(c *Core) (int, error) { return 4, nil }
}

// ===== 8-bit LD r8,r8 (0x40-0x7F) =====
func initLDr8r8() {
	// Register indices: 0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A
	for dst := range 8 {
		for src := range 8 {
			op := 0x40 | dst<<3 | src
			if dst == 6 && src == 6 {
				continue // 0x76 = HALT
			}
			d, s := dst, src
			switch {
			case d == 6:
				// LD (HL), r8: 8 cycles, M2 write
				mainHandler[op] = func(c *Core) (int, error) {
					val := c.readReg8(s)
					c.MMU.Write(c.HL, val)
					c.stepDevices(4)
					return 8, nil
				}
			case s == 6:
				// LD r8, (HL): 8 cycles, M2 read
				mainHandler[op] = func(c *Core) (int, error) {
					c.writeReg8(d, c.MMU.Read(c.HL))
					c.stepDevices(4)
					return 8, nil
				}
			default:
				// LD r8, r8: 4 cycles, internal
				mainHandler[op] = func(c *Core) (int, error) {
					c.writeReg8(d, c.readReg8(s))
					return 4, nil
				}
			}
		}
	}

	// HALT (0x76)
	mainHandler[0x76] = func(c *Core) (int, error) {
		c.Halted = true
		if !c.IME && (c.MMU.Read(0xFF0F)&c.MMU.Read(0xFFFF)) != 0 {
			c.Halted = false
			c.HaltBug = true
		}
		return 4, nil
	}

	// LD r8,d8
	for _, e := range [][2]int{{0x06, 0}, {0x0E, 1}, {0x16, 2}, {0x1E, 3}, {0x26, 4}, {0x2E, 5}, {0x36, 6}, {0x3E, 7}} {
		op, r := e[0], e[1]
		if r == 6 {
			// LD (HL), d8: 12 cycles, M2 read imm, M3 write (HL)
			mainHandler[op] = func(c *Core) (int, error) {
				val := c.fetch8()
				c.stepDevices(4)
				c.MMU.Write(c.HL, val)
				c.stepDevices(4)
				return 12, nil
			}
		} else {
			// LD r8, d8: 8 cycles, M2 read imm
			mainHandler[op] = func(c *Core) (int, error) {
				val := c.fetch8()
				c.stepDevices(4)
				c.writeReg8(r, val)
				return 8, nil
			}
		}
	}
}

// ===== 16-bit LD =====
func initLD16() {
	// LD r16, d16: 12 cycles, M2-M3 read imm
	mainHandler[0x01] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.BC = uint16(lo) | uint16(hi)<<8
		return 12, nil
	}
	mainHandler[0x11] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.DE = uint16(lo) | uint16(hi)<<8
		return 12, nil
	}
	mainHandler[0x21] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.HL = uint16(lo) | uint16(hi)<<8
		return 12, nil
	}
	mainHandler[0x31] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.SP = uint16(lo) | uint16(hi)<<8
		return 12, nil
	}

	// LD (a16),SP: 20 cycles, M2-M3 addr, M4-M5 writes
	mainHandler[0x08] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		addr := uint16(lo) | uint16(hi)<<8
		c.MMU.Write(addr, byte(c.SP))
		c.stepDevices(4)
		c.MMU.Write(addr+1, byte(c.SP>>8))
		c.stepDevices(4)
		return 20, nil
	}

	// LD (BC),A / LD (DE),A: 8 cycles, M2 write
	mainHandler[0x02] = func(c *Core) (int, error) {
		c.MMU.Write(c.BC, c.A()); c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0x12] = func(c *Core) (int, error) {
		c.MMU.Write(c.DE, c.A()); c.stepDevices(4)
		return 8, nil
	}

	// LD A,(BC) / LD A,(DE): 8 cycles, M2 read
	mainHandler[0x0A] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.BC)); c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0x1A] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.DE)); c.stepDevices(4)
		return 8, nil
	}

	// LDI (HL),A / LDD (HL),A: 8 cycles, M2 write + inc/dec
	mainHandler[0x22] = func(c *Core) (int, error) {
		c.MMU.Write(c.HL, c.A()); c.HL++; c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0x32] = func(c *Core) (int, error) {
		c.MMU.Write(c.HL, c.A()); c.HL--; c.stepDevices(4)
		return 8, nil
	}

	// LDI A,(HL) / LDD A,(HL): 8 cycles, M2 read + inc/dec
	mainHandler[0x2A] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.HL)); c.HL++; c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0x3A] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(c.HL)); c.HL--; c.stepDevices(4)
		return 8, nil
	}

	// LDH (a8),A: 12 cycles, M2 read imm, M3 write
	mainHandler[0xE0] = func(c *Core) (int, error) {
		off := c.fetch8(); c.stepDevices(4)
		c.MMU.Write(0xFF00|uint16(off), c.A()); c.stepDevices(4)
		return 12, nil
	}

	// LDH A,(a8): 12 cycles, M2 read imm, M3 read
	mainHandler[0xF0] = func(c *Core) (int, error) {
		off := c.fetch8(); c.stepDevices(4)
		c.setA(c.MMU.Read(0xFF00 | uint16(off))); c.stepDevices(4)
		return 12, nil
	}

	// LDH (C),A: 8 cycles, M2 write
	mainHandler[0xE2] = func(c *Core) (int, error) {
		c.MMU.Write(0xFF00|uint16(c.C()), c.A()); c.stepDevices(4)
		return 8, nil
	}

	// LDH A,(C): 8 cycles, M2 read
	mainHandler[0xF2] = func(c *Core) (int, error) {
		c.setA(c.MMU.Read(0xFF00 | uint16(c.C()))); c.stepDevices(4)
		return 8, nil
	}

	// LD (a16),A: 16 cycles, M2-M3 addr, M4 write
	mainHandler[0xEA] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.MMU.Write(uint16(lo)|uint16(hi)<<8, c.A()); c.stepDevices(4)
		return 16, nil
	}

	// LD A,(a16): 16 cycles, M2-M3 addr, M4 read
	mainHandler[0xFA] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.setA(c.MMU.Read(uint16(lo) | uint16(hi)<<8)); c.stepDevices(4)
		return 16, nil
	}

	// LD SP,HL (0xF9): 8 cycles, M2 internal
	mainHandler[0xF9] = func(c *Core) (int, error) {
		c.SP = c.HL; c.stepDevices(4)
		return 8, nil
	}

	// LD HL,SP+r8 (0xF8): 12 cycles, M2 read imm, M3 ALU
	mainHandler[0xF8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		c.stepDevices(4)
		result := uint32(int32(c.SP) + int32(e))
		c.HL = uint16(result)
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(e)&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(e)&0xFF) > 0xFF)
		c.stepDevices(4)
		return 12, nil
	}
}

// ===== PUSH/POP =====
func initPUSHPOP() {
	// PUSH (16 cycles): M2 SP-=2, M3 write low, M4 write high
	for _, e := range [][2]int{{0xC5, 0}, {0xD5, 1}, {0xE5, 2}, {0xF5, 3}} {
		op, r := e[0], e[1]
		switch r {
		case 0:
			mainHandler[op] = func(c *Core) (int, error) {
				c.SP -= 2; c.stepDevices(4)
				c.MMU.Write(c.SP, byte(c.BC)); c.stepDevices(4)
				c.MMU.Write(c.SP+1, byte(c.BC>>8)); c.stepDevices(4)
				return 16, nil
			}
		case 1:
			mainHandler[op] = func(c *Core) (int, error) {
				c.SP -= 2; c.stepDevices(4)
				c.MMU.Write(c.SP, byte(c.DE)); c.stepDevices(4)
				c.MMU.Write(c.SP+1, byte(c.DE>>8)); c.stepDevices(4)
				return 16, nil
			}
		case 2:
			mainHandler[op] = func(c *Core) (int, error) {
				c.SP -= 2; c.stepDevices(4)
				c.MMU.Write(c.SP, byte(c.HL)); c.stepDevices(4)
				c.MMU.Write(c.SP+1, byte(c.HL>>8)); c.stepDevices(4)
				return 16, nil
			}
		case 3:
			mainHandler[op] = func(c *Core) (int, error) {
				c.SP -= 2; c.stepDevices(4)
				c.MMU.Write(c.SP, byte(c.AF)); c.stepDevices(4)
				c.MMU.Write(c.SP+1, byte(c.AF>>8)); c.stepDevices(4)
				return 16, nil
			}
		}
	}

	// POP (12 cycles): M2 read low, M3 read high
	for _, e := range [][2]int{{0xC1, 0}, {0xD1, 1}, {0xE1, 2}, {0xF1, 3}} {
		op, r := e[0], e[1]
		switch r {
		case 0:
			mainHandler[op] = func(c *Core) (int, error) {
				lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				c.BC = uint16(lo) | uint16(hi)<<8
				return 12, nil
			}
		case 1:
			mainHandler[op] = func(c *Core) (int, error) {
				lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				c.DE = uint16(lo) | uint16(hi)<<8
				return 12, nil
			}
		case 2:
			mainHandler[op] = func(c *Core) (int, error) {
				lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				c.HL = uint16(lo) | uint16(hi)<<8
				return 12, nil
			}
		case 3:
			mainHandler[op] = func(c *Core) (int, error) {
				lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
				c.AF = (uint16(lo) | uint16(hi)<<8) & 0xFFF0
				return 12, nil
			}
		}
	}
}
// ===== 8-bit ALU =====
func initALU() {
	for i := range 8 {
		op, r := 0x80+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.addA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.addA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xC6] = func(c *Core) (int, error) {
		c.addA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// ADC A,r8
	for i := range 8 {
		op, r := 0x88+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.adcA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.adcA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xCE] = func(c *Core) (int, error) {
		c.adcA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// SUB A,r8
	for i := range 8 {
		op, r := 0x90+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.subA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.subA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xD6] = func(c *Core) (int, error) {
		c.subA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// SBC A,r8
	for i := range 8 {
		op, r := 0x98+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.sbcA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.sbcA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xDE] = func(c *Core) (int, error) {
		c.sbcA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// AND A,r8
	for i := range 8 {
		op, r := 0xA0+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.andA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.andA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xE6] = func(c *Core) (int, error) {
		c.andA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// OR A,r8
	for i := range 8 {
		op, r := 0xB0+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.orA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.orA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xF6] = func(c *Core) (int, error) {
		c.orA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// XOR A,r8
	for i := range 8 {
		op, r := 0xA8+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.xorA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.xorA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xEE] = func(c *Core) (int, error) {
		c.xorA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}

	// CP A,r8
	for i := range 8 {
		op, r := 0xB8+i, i
		if r == 6 {
			mainHandler[op] = func(c *Core) (int, error) {
				c.cpA(c.MMU.Read(c.HL)); c.stepDevices(4)
				return 8, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.cpA(c.readReg8(r)); return 4, nil
			}
		}
	}
	mainHandler[0xFE] = func(c *Core) (int, error) {
		c.cpA(c.fetch8()); c.stepDevices(4)
		return 8, nil
	}
}

func initINCDEC() {
	// INC r8
	for _, e := range [][2]int{{0x04, 0}, {0x0C, 1}, {0x14, 2}, {0x1C, 3}, {0x24, 4}, {0x2C, 5}, {0x34, 6}, {0x3C, 7}} {
		op, r := e[0], e[1]
		if r == 6 {
			// INC (HL): 12 cycles, M2 read, M3 write
			mainHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL); c.stepDevices(4)
				c.MMU.Write(c.HL, c.inc8(val)); c.stepDevices(4)
				return 12, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.writeReg8(r, c.inc8(c.readReg8(r)))
				return 4, nil
			}
		}
	}

	// DEC r8
	for _, e := range [][2]int{{0x05, 0}, {0x0D, 1}, {0x15, 2}, {0x1D, 3}, {0x25, 4}, {0x2D, 5}, {0x35, 6}, {0x3D, 7}} {
		op, r := e[0], e[1]
		if r == 6 {
			// DEC (HL): 12 cycles, M2 read, M3 write
			mainHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL); c.stepDevices(4)
				c.MMU.Write(c.HL, c.dec8(val)); c.stepDevices(4)
				return 12, nil
			}
		} else {
			mainHandler[op] = func(c *Core) (int, error) {
				c.writeReg8(r, c.dec8(c.readReg8(r)))
				return 4, nil
			}
		}
	}
}

func initALU16() {
	// ===== 16-bit ALU =====
	// ADD HL,r16: 8 cycles, M2 internal ALU
	mainHandler[0x09] = func(c *Core) (int, error) { c.addHL(c.BC); c.stepDevices(4); return 8, nil }
	mainHandler[0x19] = func(c *Core) (int, error) { c.addHL(c.DE); c.stepDevices(4); return 8, nil }
	mainHandler[0x29] = func(c *Core) (int, error) { c.addHL(c.HL); c.stepDevices(4); return 8, nil }
	mainHandler[0x39] = func(c *Core) (int, error) { c.addHL(c.SP); c.stepDevices(4); return 8, nil }

	// INC r16: 8 cycles, M2 internal
	mainHandler[0x03] = func(c *Core) (int, error) { c.BC++; c.stepDevices(4); return 8, nil }
	mainHandler[0x13] = func(c *Core) (int, error) { c.DE++; c.stepDevices(4); return 8, nil }
	mainHandler[0x23] = func(c *Core) (int, error) { c.HL++; c.stepDevices(4); return 8, nil }
	mainHandler[0x33] = func(c *Core) (int, error) { c.SP++; c.stepDevices(4); return 8, nil }

	// DEC r16: 8 cycles, M2 internal
	mainHandler[0x0B] = func(c *Core) (int, error) { c.BC--; c.stepDevices(4); return 8, nil }
	mainHandler[0x1B] = func(c *Core) (int, error) { c.DE--; c.stepDevices(4); return 8, nil }
	mainHandler[0x2B] = func(c *Core) (int, error) { c.HL--; c.stepDevices(4); return 8, nil }
	mainHandler[0x3B] = func(c *Core) (int, error) { c.SP--; c.stepDevices(4); return 8, nil }

	// ADD SP,r8 (0xE8): 16 cycles, M2 read imm, M3-M4 internal
	mainHandler[0xE8] = func(c *Core) (int, error) {
		e := int8(c.fetch8())
		c.stepDevices(4)
		result := uint32(int32(c.SP) + int32(e))
		c.setFlagZ(false)
		c.setFlagN(false)
		c.setFlagH((c.SP&0x0F)+(uint16(e)&0x0F) > 0x0F)
		c.setFlagC((c.SP&0xFF)+(uint16(e)&0xFF) > 0xFF)
		c.SP = uint16(result)
		c.stepDevices(4)
		c.stepDevices(4)
		return 16, nil
	}
}

func initControlFlow() {
	// ===== Control flow =====

	// JP a16 (0xC3): 16 cycles, M2-M3 addr, M4 set PC
	mainHandler[0xC3] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		c.PC = uint16(lo) | uint16(hi)<<8
		c.stepDevices(4)
		return 16, nil
	}

	// JP HL (0xE9): 4 cycles, internal
	mainHandler[0xE9] = func(c *Core) (int, error) { c.PC = c.HL; return 4, nil }

	// JP cc,a16
	mainHandler[0xC2] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if !c.flagZ() {
			c.PC = uint16(lo) | uint16(hi)<<8
			c.stepDevices(4)
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xCA] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if c.flagZ() {
			c.PC = uint16(lo) | uint16(hi)<<8
			c.stepDevices(4)
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xD2] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if !c.flagC() {
			c.PC = uint16(lo) | uint16(hi)<<8
			c.stepDevices(4)
			return 16, nil
		}
		return 12, nil
	}
	mainHandler[0xDA] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if c.flagC() {
			c.PC = uint16(lo) | uint16(hi)<<8
			c.stepDevices(4)
			return 16, nil
		}
		return 12, nil
	}

	// JR r8 (0x18): 12 cycles, M2 read imm, M3 add
	mainHandler[0x18] = func(c *Core) (int, error) {
		off := int8(c.fetch8()); c.stepDevices(4)
		c.PC = uint16(int32(c.PC) + int32(off)); c.stepDevices(4)
		return 12, nil
	}

	// JR cc,r8
	mainHandler[0x20] = func(c *Core) (int, error) {
		off := int8(c.fetch8()); c.stepDevices(4)
		if !c.flagZ() {
			c.PC = uint16(int32(c.PC) + int32(off)); c.stepDevices(4)
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x28] = func(c *Core) (int, error) {
		off := int8(c.fetch8()); c.stepDevices(4)
		if c.flagZ() {
			c.PC = uint16(int32(c.PC) + int32(off)); c.stepDevices(4)
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x30] = func(c *Core) (int, error) {
		off := int8(c.fetch8()); c.stepDevices(4)
		if !c.flagC() {
			c.PC = uint16(int32(c.PC) + int32(off)); c.stepDevices(4)
			return 12, nil
		}
		return 8, nil
	}
	mainHandler[0x38] = func(c *Core) (int, error) {
		off := int8(c.fetch8()); c.stepDevices(4)
		if c.flagC() {
			c.PC = uint16(int32(c.PC) + int32(off)); c.stepDevices(4)
			return 12, nil
		}
		return 8, nil
	}

	// CALL a16 (0xCD): 24 cycles, M2-M3 addr, M4 SP-=2, M5 write low, M6 write high
	mainHandler[0xCD] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		addr := uint16(lo) | uint16(hi)<<8
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = addr
		return 24, nil
	}

	// CALL cc,a16
	mainHandler[0xC4] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if !c.flagZ() {
			addr := uint16(lo) | uint16(hi)<<8
			c.SP -= 2; c.stepDevices(4)
			c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
			c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xCC] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if c.flagZ() {
			addr := uint16(lo) | uint16(hi)<<8
			c.SP -= 2; c.stepDevices(4)
			c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
			c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xD4] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if !c.flagC() {
			addr := uint16(lo) | uint16(hi)<<8
			c.SP -= 2; c.stepDevices(4)
			c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
			c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}
	mainHandler[0xDC] = func(c *Core) (int, error) {
		lo := c.fetch8(); c.stepDevices(4)
		hi := c.fetch8(); c.stepDevices(4)
		if c.flagC() {
			addr := uint16(lo) | uint16(hi)<<8
			c.SP -= 2; c.stepDevices(4)
			c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
			c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
			c.PC = addr
			return 24, nil
		}
		return 12, nil
	}

	// RET (0xC9): 16 cycles, M2 read low, M3 read high, M4 set PC
	mainHandler[0xC9] = func(c *Core) (int, error) {
		lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
		hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
		c.PC = uint16(lo) | uint16(hi)<<8; c.stepDevices(4)
		return 16, nil
	}

	// RET cc
	mainHandler[0xC0] = func(c *Core) (int, error) {
		if !c.flagZ() {
			c.stepDevices(4) // M2: condition met, prepare
			lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			c.PC = uint16(lo) | uint16(hi)<<8; c.stepDevices(4)
			return 20, nil
		}
		c.stepDevices(4) // M2: condition not met
		return 8, nil
	}
	mainHandler[0xC8] = func(c *Core) (int, error) {
		if c.flagZ() {
			c.stepDevices(4)
			lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			c.PC = uint16(lo) | uint16(hi)<<8; c.stepDevices(4)
			return 20, nil
		}
		c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0xD0] = func(c *Core) (int, error) {
		if !c.flagC() {
			c.stepDevices(4)
			lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			c.PC = uint16(lo) | uint16(hi)<<8; c.stepDevices(4)
			return 20, nil
		}
		c.stepDevices(4)
		return 8, nil
	}
	mainHandler[0xD8] = func(c *Core) (int, error) {
		if c.flagC() {
			c.stepDevices(4)
			lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
			c.PC = uint16(lo) | uint16(hi)<<8; c.stepDevices(4)
			return 20, nil
		}
		c.stepDevices(4)
		return 8, nil
	}

	// RETI (0xD9): 16 cycles, same as RET but re-enables IME
	mainHandler[0xD9] = func(c *Core) (int, error) {
		lo := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
		hi := c.MMU.Read(c.SP); c.SP++; c.stepDevices(4)
		c.PC = uint16(lo) | uint16(hi)<<8
		c.IME = true
		c.stepDevices(4)
		return 16, nil
	}

	// RST n: 16 cycles, same structure as PUSH
	mainHandler[0xC7] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x00
		return 16, nil
	}
	mainHandler[0xCF] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x08
		return 16, nil
	}
	mainHandler[0xD7] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x10
		return 16, nil
	}
	mainHandler[0xDF] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x18
		return 16, nil
	}
	mainHandler[0xE7] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x20
		return 16, nil
	}
	mainHandler[0xEF] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x28
		return 16, nil
	}
	mainHandler[0xF7] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x30
		return 16, nil
	}
	mainHandler[0xFF] = func(c *Core) (int, error) {
		c.SP -= 2; c.stepDevices(4)
		c.MMU.Write(c.SP, byte(c.PC)); c.stepDevices(4)
		c.MMU.Write(c.SP+1, byte(c.PC>>8)); c.stepDevices(4)
		c.PC = 0x38
		return 16, nil
	}
}

func initMisc() {
	// ===== Miscellaneous =====

	// RLCA (0x07): 4 cycles
	mainHandler[0x07] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x80 != 0)
		c.setA((a << 1) | (a >> 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RRCA (0x0F): 4 cycles
	mainHandler[0x0F] = func(c *Core) (int, error) {
		a := c.A()
		c.setFlagC(a&0x01 != 0)
		c.setA((a >> 1) | (a << 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RLA (0x17): 4 cycles
	mainHandler[0x17] = func(c *Core) (int, error) {
		a := c.A()
		oldC := c.flagC()
		c.setFlagC(a&0x80 != 0)
		c.setA((a << 1) | boolToByte(oldC))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}
	// RRA (0x1F): 4 cycles
	mainHandler[0x1F] = func(c *Core) (int, error) {
		a := c.A()
		oldC := c.flagC()
		c.setFlagC(a&0x01 != 0)
		c.setA((a >> 1) | (boolToByte(oldC) << 7))
		c.setFlagZ(false); c.setFlagN(false); c.setFlagH(false)
		return 4, nil
	}

	// DAA (0x27): 4 cycles
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

	// CPL (0x2F): 4 cycles
	mainHandler[0x2F] = func(c *Core) (int, error) {
		c.setA(^c.A())
		c.setFlagN(true); c.setFlagH(true)
		return 4, nil
	}
	// SCF (0x37): 4 cycles
	mainHandler[0x37] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false); c.setFlagC(true)
		return 4, nil
	}
	// CCF (0x3F): 4 cycles
	mainHandler[0x3F] = func(c *Core) (int, error) {
		c.setFlagN(false); c.setFlagH(false); c.setFlagC(!c.flagC())
		return 4, nil
	}
	// DI (0xF3): 4 cycles
	mainHandler[0xF3] = func(c *Core) (int, error) {
		c.IME = false
		c.IMEScheduled = 0
		return 4, nil
	}
	// EI (0xFB): 4 cycles
	mainHandler[0xFB] = func(c *Core) (int, error) {
		c.IMEScheduled = 2
		return 4, nil
	}

	// STOP (0x10): 8 cycles, M2 read imm (ignored)
	mainHandler[0x10] = func(c *Core) (int, error) {
		c.fetch8() // consume immediate byte
		c.Stopped = true
		c.stepDevices(4)
		return 8, nil
	}

	// PREFIX CB (0xCB): M2 fetch sub-opcode, then dispatch to cbHandler
	mainHandler[0xCB] = func(c *Core) (int, error) {
		sub := c.fetch8()
		c.stepDevices(4)
		h := cbHandler[sub]
		if h == nil {
			return 8, nil
		}
		return h(c)
	}

	// Undefined opcodes remain nil — Step() treats them as NOP.
}
