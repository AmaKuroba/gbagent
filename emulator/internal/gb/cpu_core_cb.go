package gb

// cbHandler maps each CB-prefixed sub-opcode to its execution handler.
// The CB prefix (0xCB) has already been fetched by mainHandler[0xCB].
// The main handler also fetches the sub-opcode (M2).
// These handlers are responsible for M3+ steps and return the total
// instruction cost (including M1-M2, which is 8 for register ops
// or more for (HL) ops).
var cbHandler [256]func(c *Core) (int, error)

func init() { initCBHandlers() }

func initCBHandlers() {
	// ===== Rotates and Shifts (0x00-0x3F) =====
	for op := 0; op < 64; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		opType := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			// (HL) operations: 16 cycles total (M1-M4)
			// M1: fetch 0xCB (Step), M2: fetch sub-opcode (mainHandler)
			// M3: read from (HL), M4: write result to (HL)
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				c.stepDevices(4) // M3 done

				var result byte
				var carry bool

				switch opType {
				case 0: // RLC
					carry = (val & 0x80) != 0
					result = (val << 1) | (val >> 7)
				case 1: // RRC
					carry = (val & 0x01) != 0
					result = (val >> 1) | (val << 7)
				case 2: // RL
					oldC := c.flagC()
					carry = (val & 0x80) != 0
					result = (val << 1) | boolToByte(oldC)
				case 3: // RR
					oldC := c.flagC()
					carry = (val & 0x01) != 0
					result = (val >> 1) | (boolToByte(oldC) << 7)
				case 4: // SLA
					carry = (val & 0x80) != 0
					result = val << 1
				case 5: // SRA
					carry = (val & 0x01) != 0
					result = (val >> 1) | (val & 0x80)
				case 6: // SWAP
					carry = false
					result = (val << 4) | (val >> 4)
				case 7: // SRL
					carry = (val & 0x01) != 0
					result = val >> 1
				}

				c.MMU.Write(c.HL, result)
				c.stepDevices(4) // M4 done

				c.setFlagZ(result == 0)
				c.setFlagN(false)
				c.setFlagH(false)
				c.setFlagC(carry)
				return 16, nil
			}
		} else {
			// Register operations: 8 cycles total (M1-M2 only)
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)

				var result byte
				var carry bool

				switch opType {
				case 0: // RLC
					carry = (val & 0x80) != 0
					result = (val << 1) | (val >> 7)
				case 1: // RRC
					carry = (val & 0x01) != 0
					result = (val >> 1) | (val << 7)
				case 2: // RL
					oldC := c.flagC()
					carry = (val & 0x80) != 0
					result = (val << 1) | boolToByte(oldC)
				case 3: // RR
					oldC := c.flagC()
					carry = (val & 0x01) != 0
					result = (val >> 1) | (boolToByte(oldC) << 7)
				case 4: // SLA
					carry = (val & 0x80) != 0
					result = val << 1
				case 5: // SRA
					carry = (val & 0x01) != 0
					result = (val >> 1) | (val & 0x80)
				case 6: // SWAP
					carry = false
					result = (val << 4) | (val >> 4)
				case 7: // SRL
					carry = (val & 0x01) != 0
					result = val >> 1
				}

				c.writeReg8(reg, result)
				c.setFlagZ(result == 0)
				c.setFlagN(false)
				c.setFlagH(false)
				c.setFlagC(carry)
				return 8, nil
			}
		}
	}

	// ===== BIT b,r (0x40-0x7F) =====
	// Z=!bit, N=0, H=1, C unchanged
	for op := 0x40; op <= 0x7F; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			// BIT (HL): 12 cycles, M3 read only
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				c.stepDevices(4) // M3 done
				bitSet := (val>>bit)&0x01 != 0
				c.setFlagZ(!bitSet)
				c.setFlagN(false)
				c.setFlagH(true)
				return 12, nil
			}
		} else {
			// BIT r8: 8 cycles, internal
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)
				bitSet := (val>>bit)&0x01 != 0
				c.setFlagZ(!bitSet)
				c.setFlagN(false)
				c.setFlagH(true)
				return 8, nil
			}
		}
	}

	// ===== RES b,r (0x80-0xBF) =====
	for op := 0x80; op <= 0xBF; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			// RES (HL): 16 cycles, M3 read, M4 write
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				c.stepDevices(4) // M3 done
				val &^= 1 << bit
				c.MMU.Write(c.HL, val)
				c.stepDevices(4) // M4 done
				return 16, nil
			}
		} else {
			// RES r8: 8 cycles, internal
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)
				val &^= 1 << bit
				c.writeReg8(reg, val)
				return 8, nil
			}
		}
	}

	// ===== SET b,r (0xC0-0xFF) =====
	for op := 0xC0; op <= 0xFF; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			// SET (HL): 16 cycles, M3 read, M4 write
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				c.stepDevices(4) // M3 done
				val |= 1 << bit
				c.MMU.Write(c.HL, val)
				c.stepDevices(4) // M4 done
				return 16, nil
			}
		} else {
			// SET r8: 8 cycles, internal
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)
				val |= 1 << bit
				c.writeReg8(reg, val)
				return 8, nil
			}
		}
	}
}
