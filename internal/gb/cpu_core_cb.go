package gb

// cbHandler maps each CB-prefixed sub-opcode to its execution handler.
// The CB prefix (0xCB) has already been fetched by Step() when these run.
// These handlers return the *total* cycle count for the CB instruction
// (including the 4 cycles for the CB prefix fetch itself).
var cbHandler [256]func(c *Core) (int, error)

func init() { initCBHandlers() }

func initCBHandlers() {
	// ===== Rotates and Shifts (0x00-0x3F) =====
	// Bits 5-3 select the operation type; bits 2-0 select the register.
	// Register index: 0=B,1=C,2=D,3=E,4=H,5=L,6=(HL),7=A
	for op := 0; op < 64; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		opType := (op >> 3) & 0x07 // 0=RLC,1=RRC,2=RL,3=RR,4=SLA,5=SRA,6=SWAP,7=SRL
		isHL := reg == 6

		cbHandler[op] = func(c *Core) (int, error) {
			val := c.readReg8(reg)

			var result byte
			var carry bool

			switch opType {
			case 0: // RLC — rotate left circular
				carry = (val & 0x80) != 0
				result = (val << 1) | (val >> 7)
			case 1: // RRC — rotate right circular
				carry = (val & 0x01) != 0
				result = (val >> 1) | (val << 7)
			case 2: // RL — rotate left through carry
				oldC := c.flagC()
				carry = (val & 0x80) != 0
				result = (val << 1) | boolToByte(oldC)
			case 3: // RR — rotate right through carry
				oldC := c.flagC()
				carry = (val & 0x01) != 0
				result = (result >> 1) | (boolToByte(oldC) << 7) // result not yet set, use 0
				result = (val >> 1) | (boolToByte(oldC) << 7)
			case 4: // SLA — shift left arithmetic
				carry = (val & 0x80) != 0
				result = val << 1
			case 5: // SRA — shift right arithmetic (MSB preserved)
				carry = (val & 0x01) != 0
				result = (val >> 1) | (val & 0x80)
			case 6: // SWAP — swap nibbles
				carry = false
				result = (val << 4) | (val >> 4)
			case 7: // SRL — shift right logical
				carry = (val & 0x01) != 0
				result = val >> 1
			}

			c.writeReg8(reg, result)
			c.setFlagZ(result == 0)
			c.setFlagN(false)
			c.setFlagH(false)
			c.setFlagC(carry)

			if isHL {
				c.schedWrite(c.HL, result)
				return 16, nil
			}
			c.writeReg8(reg, result)
			return 8, nil
		}
	}

	// ===== BIT b,r (0x40-0x7F) =====
	// Z=!bit, N=0, H=1, C unchanged
	for op := 0x40; op <= 0x7F; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		cbHandler[op] = func(c *Core) (int, error) {
			val := c.readReg8(reg)
			bitSet := (val>>bit)&0x01 != 0
			c.setFlagZ(!bitSet)
			c.setFlagN(false)
			c.setFlagH(true)
			// C flag preserved

			if isHL {
				return 12, nil
			}
			return 8, nil
		}
	}

	// ===== RES b,r (0x80-0xBF) — clear bit, write deferred for (HL) =====
	for op := 0x80; op <= 0xBF; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				val &^= 1 << bit
				c.schedWrite(c.HL, val)
				return 16, nil
			}
		} else {
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)
				val &^= 1 << bit
				c.writeReg8(reg, val)
				return 8, nil
			}
		}
	}

	// ===== SET b,r (0xC0-0xFF) — set bit, write deferred for (HL) =====
	for op := 0xC0; op <= 0xFF; op++ {
		op := byte(op)
		reg := int(op & 0x07)
		bit := (op >> 3) & 0x07
		isHL := reg == 6

		if isHL {
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.MMU.Read(c.HL)
				val |= 1 << bit
				c.schedWrite(c.HL, val)
				return 16, nil
			}
		} else {
			cbHandler[op] = func(c *Core) (int, error) {
				val := c.readReg8(reg)
				val |= 1 << bit
				c.writeReg8(reg, val)
				return 8, nil
			}
		}
	}
}
