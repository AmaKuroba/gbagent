package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mkCBOp places CB prefix + sub-opcode at PC and returns a fresh Core ready to Step.
func mkCBOp(cbOp byte, setup func(*Core)) *Core {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB) // CB prefix
	bus.Write(0x0101, cbOp) // CB sub-opcode
	c := NewCore(bus)
	c.PC = 0x0100
	if setup != nil {
		setup(c)
	}
	return c
}

// checkFlags asserts Z,N,H,C match expected.
func checkFlags(t *testing.T, c *Core, z, n, h, cf bool) {
	t.Helper()
	assert.Equal(t, z, c.flagZ(), "Z flag")
	assert.Equal(t, n, c.flagN(), "N flag")
	assert.Equal(t, h, c.flagH(), "H flag")
	assert.Equal(t, cf, c.flagC(), "C flag")
}

// ============================================================================
// RLC — rotate left circular (old bit 7 → carry, bit 0 ← old bit 7)
// ============================================================================

func TestCBCmd_RLC_B(t *testing.T) {
	c := mkCBOp(0x00, func(c *Core) { c.setB(0x85) }) // 1000_0101
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 8, cycles)
	assert.Equal(t, byte(0x0B), c.B(), "RLC B: 0x85 -> 0x0B")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_RLC_B_zero(t *testing.T) {
	c := mkCBOp(0x00, func(c *Core) { c.setB(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.B())
	checkFlags(t, c, true, false, false, false)
}

func TestCBCmd_RLC_C(t *testing.T) {
	c := mkCBOp(0x01, func(c *Core) { c.setC(0x01) }) // 0000_0001
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x02), c.C(), "RLC C: 0x01 -> 0x02")
	checkFlags(t, c, false, false, false, false)
}

func TestCBCmd_RLC_HL(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x06) // RLC (HL)
	bus.Write(0xC000, 0x85)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 16, cycles, "(HL) RLC takes 16 cycles")
	assert.Equal(t, byte(0x0B), bus.Read(0xC000), "RLC (HL): 0x85 -> 0x0B")
	checkFlags(t, c, false, false, false, true)
}

// ============================================================================
// RRC — rotate right circular (old bit 0 → carry, bit 7 ← old bit 0)
// ============================================================================

func TestCBCmd_RRC_C(t *testing.T) {
	c := mkCBOp(0x09, func(c *Core) { c.setC(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x80), c.C(), "RRC C: 0x01 -> 0x80")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_RRC_A(t *testing.T) {
	c := mkCBOp(0x0F, func(c *Core) { c.setA(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x80), c.A(), "RRC A: 0x01 -> 0x80")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_RRC_A_zero(t *testing.T) {
	c := mkCBOp(0x0F, func(c *Core) { c.setA(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.A())
	checkFlags(t, c, true, false, false, false)
}

// ============================================================================
// RL — rotate left through carry (old bit 7 → carry, bit 0 ← old carry)
// ============================================================================

func TestCBCmd_RL_B(t *testing.T) {
	c := mkCBOp(0x10, func(c *Core) { c.setB(0x85); c.setFlagC(false) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x0A), c.B(), "RL B: 0x85 C=0 -> 0x0A")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_RL_B_with_carry(t *testing.T) {
	c := mkCBOp(0x10, func(c *Core) { c.setB(0x05); c.setFlagC(true) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x0B), c.B(), "RL B: 0x05 C=1 -> 0x0B")
	checkFlags(t, c, false, false, false, false)
}

func TestCBCmd_RL_A_zero(t *testing.T) {
	c := mkCBOp(0x17, func(c *Core) { c.setA(0x80); c.setFlagC(false) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.A(), "RL A: 0x80 C=0 -> 0x00")
	checkFlags(t, c, true, false, false, true)
}

// ============================================================================
// RR — rotate right through carry (old bit 0 → carry, bit 7 ← old carry)
// ============================================================================

func TestCBCmd_RR_B(t *testing.T) {
	c := mkCBOp(0x18, func(c *Core) { c.setB(0x85); c.setFlagC(false) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x42), c.B(), "RR B: 0x85 C=0 -> 0x42")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_RR_B_with_carry(t *testing.T) {
	c := mkCBOp(0x18, func(c *Core) { c.setB(0x04); c.setFlagC(true) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x82), c.B(), "RR B: 0x04 C=1 -> 0x82")
	checkFlags(t, c, false, false, false, false)
}

// ============================================================================
// SLA — shift left arithmetic (bit 7 → carry, LSB = 0)
// ============================================================================

func TestCBCmd_SLA_B(t *testing.T) {
	c := mkCBOp(0x20, func(c *Core) { c.setB(0x85) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x0A), c.B(), "SLA B: 0x85 -> 0x0A")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_SLA_B_zero(t *testing.T) {
	c := mkCBOp(0x20, func(c *Core) { c.setB(0x80) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.B())
	checkFlags(t, c, true, false, false, true)
}

// ============================================================================
// SRA — shift right arithmetic (bit 0 → carry, MSB preserved)
// ============================================================================

func TestCBCmd_SRA_B(t *testing.T) {
	c := mkCBOp(0x28, func(c *Core) { c.setB(0x85) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0xC2), c.B(), "SRA B: 0x85 -> 0xC2 (sign extended)")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_SRA_L(t *testing.T) {
	c := mkCBOp(0x2D, func(c *Core) { c.setL(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.L(), "SRA L: 0x01 -> 0x00")
	checkFlags(t, c, true, false, false, true)
}

// ============================================================================
// SWAP — swap upper and lower nibbles
// ============================================================================

func TestCBCmd_SWAP_B(t *testing.T) {
	c := mkCBOp(0x30, func(c *Core) { c.setB(0xAB) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0xBA), c.B(), "SWAP B: 0xAB -> 0xBA")
	checkFlags(t, c, false, false, false, false)
}

func TestCBCmd_SWAP_H(t *testing.T) {
	c := mkCBOp(0x34, func(c *Core) { c.setH(0xF0) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x0F), c.H(), "SWAP H: 0xF0 -> 0x0F")
	checkFlags(t, c, false, false, false, false)
}

func TestCBCmd_SWAP_zero(t *testing.T) {
	c := mkCBOp(0x37, func(c *Core) { c.setA(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.A())
	checkFlags(t, c, true, false, false, false)
}

func TestCBCmd_SWAP_HL(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x36) // SWAP (HL)
	bus.Write(0xC000, 0xAB)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 16, cycles, "(HL) SWAP takes 16 cycles")
	assert.Equal(t, byte(0xBA), bus.Read(0xC000), "SWAP (HL): 0xAB -> 0xBA")
}

// ============================================================================
// SRL — shift right logical (bit 0 → carry, MSB = 0)
// ============================================================================

func TestCBCmd_SRL_B(t *testing.T) {
	c := mkCBOp(0x38, func(c *Core) { c.setB(0x85) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x42), c.B(), "SRL B: 0x85 -> 0x42")
	checkFlags(t, c, false, false, false, true)
}

func TestCBCmd_SRL_D(t *testing.T) {
	c := mkCBOp(0x3A, func(c *Core) { c.setD(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.D())
	checkFlags(t, c, true, false, false, true)
}

// ============================================================================
// BIT b,r — test bit, Z=!bit, N=0, H=1 (C preserved)
// ============================================================================

func TestCBCmd_BIT_0_B_set(t *testing.T) {
	c := mkCBOp(0x40, func(c *Core) { c.setB(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x01), c.B(), "BIT 0,B: B unchanged")
	assert.Equal(t, false, c.flagZ(), "BIT 0,B: Z=0 (bit set)")
	assert.Equal(t, false, c.flagN(), "BIT 0,B: N=0")
	assert.Equal(t, true, c.flagH(), "BIT 0,B: H=1")
}

func TestCBCmd_BIT_0_B_clear(t *testing.T) {
	c := mkCBOp(0x40, func(c *Core) { c.setB(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, true, c.flagZ(), "BIT 0,B: Z=1 (bit clear)")
	assert.Equal(t, false, c.flagN(), "BIT 0,B: N=0")
	assert.Equal(t, true, c.flagH(), "BIT 0,B: H=1")
}

func TestCBCmd_BIT_3_H_set(t *testing.T) {
	c := mkCBOp(0x5C, func(c *Core) { c.setH(0x08) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.flagZ(), "BIT 3,H: Z=0 (bit set)")
	assert.True(t, c.flagH(), "BIT 3,H: H=1")
	assert.False(t, c.flagN(), "BIT 3,H: N=0")
}

func TestCBCmd_BIT_3_H_clear(t *testing.T) {
	c := mkCBOp(0x5C, func(c *Core) { c.setH(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.flagZ(), "BIT 3,H: Z=1 (bit clear)")
}

func TestCBCmd_BIT_7_B(t *testing.T) {
	c := mkCBOp(0x78, func(c *Core) { c.setB(0x80) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.False(t, c.flagZ(), "BIT 7,B: Z=0 (bit set)")
	assert.True(t, c.flagH(), "BIT 7,B: H=1")
	assert.False(t, c.flagN(), "BIT 7,B: N=0")
}

func TestCBCmd_BIT_7_A(t *testing.T) {
	c := mkCBOp(0x7F, func(c *Core) { c.setA(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.True(t, c.flagZ(), "BIT 7,A: Z=1 (bit clear)")
}

func TestCBCmd_BIT_HL(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x46) // BIT 0,(HL)
	bus.Write(0xC000, 0x01)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 12, cycles, "BIT (HL) takes 12 cycles")
	assert.False(t, c.flagZ(), "BIT 0,(HL): Z=0 (bit set)")
	assert.True(t, c.flagH(), "BIT 0,(HL): H=1")
	assert.False(t, c.flagN(), "BIT 0,(HL): N=0")
}

// ============================================================================
// SET b,r — set bit (no flags affected)
// ============================================================================

func TestCBCmd_SET_0_B(t *testing.T) {
	c := mkCBOp(0xC0, func(c *Core) { c.setB(0x00); c.AF = 0x00F0 }) // all flags set
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x01), c.B(), "SET 0,B: 0x00 -> 0x01")
	// SET preserves all flags
	assert.True(t, c.flagZ(), "SET: Z preserved")
	assert.True(t, c.flagN(), "SET: N preserved")
	assert.True(t, c.flagH(), "SET: H preserved")
	assert.True(t, c.flagC(), "SET: C preserved")
}

func TestCBCmd_SET_0_B_already_set(t *testing.T) {
	c := mkCBOp(0xC0, func(c *Core) { c.setB(0x01) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x01), c.B())
}

func TestCBCmd_SET_7_B(t *testing.T) {
	c := mkCBOp(0xF8, func(c *Core) { c.setB(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x80), c.B(), "SET 7,B: 0x00 -> 0x80")
}

func TestCBCmd_SET_3_E(t *testing.T) {
	c := mkCBOp(0xDB, func(c *Core) { c.setE(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x08), c.E(), "SET 3,E: 0x00 -> 0x08")
}

func TestCBCmd_SET_HL(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0xC6) // SET 0,(HL)
	bus.Write(0xC000, 0x00)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 16, cycles, "SET (HL) takes 16 cycles")
	assert.Equal(t, byte(0x01), bus.Read(0xC000), "SET 0,(HL): 0x00 -> 0x01")
}

// ============================================================================
// RES b,r — clear bit (no flags affected)
// ============================================================================

func TestCBCmd_RES_0_B(t *testing.T) {
	c := mkCBOp(0x80, func(c *Core) { c.setB(0x01); c.AF = 0x00F0 }) // all flags set
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.B(), "RES 0,B: 0x01 -> 0x00")
	// RES preserves all flags
	assert.True(t, c.flagZ(), "RES: Z preserved")
	assert.True(t, c.flagN(), "RES: N preserved")
	assert.True(t, c.flagH(), "RES: H preserved")
	assert.True(t, c.flagC(), "RES: C preserved")
}

func TestCBCmd_RES_0_B_already_clear(t *testing.T) {
	c := mkCBOp(0x80, func(c *Core) { c.setB(0x00) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x00), c.B())
}

func TestCBCmd_RES_7_B(t *testing.T) {
	c := mkCBOp(0xB8, func(c *Core) { c.setB(0xFF) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0x7F), c.B(), "RES 7,B: 0xFF -> 0x7F")
}

func TestCBCmd_RES_5_H(t *testing.T) {
	c := mkCBOp(0xAC, func(c *Core) { c.setH(0xFF) })
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, byte(0xDF), c.H(), "RES 5,H: 0xFF -> 0xDF")
}

func TestCBCmd_RES_HL(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x86) // RES 0,(HL)
	bus.Write(0xC000, 0x01)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	cycles, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, 16, cycles, "RES (HL) takes 16 cycles")
	assert.Equal(t, byte(0x00), bus.Read(0xC000), "RES 0,(HL): 0x01 -> 0x00")
}

// ============================================================================
// Cycle count tests
// ============================================================================

func TestCBCmd_CyclesRegister(t *testing.T) {
	c := mkCBOp(0x00, nil) // RLC B
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint64(8), c.Cycles, "register CB op: 8 cycles total")
}

func TestCBCmd_CyclesHLIndirect_rotate(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x06) // RLC (HL)
	bus.Write(0xC000, 0x00)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint64(16), c.Cycles, "(HL) rotate CB op: 16 cycles total")
}

func TestCBCmd_CyclesHLIndirect_bit(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x46) // BIT 0,(HL) — 12 cycles
	bus.Write(0xC000, 0x00)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint64(12), c.Cycles, "BIT (HL): 12 cycles total")
}

func TestCBCmd_CyclesHLIndirect_res_set(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x0100, 0xCB)
	bus.Write(0x0101, 0x86) // RES 0,(HL) — 16 cycles
	bus.Write(0xC000, 0x01)
	c := NewCore(bus)
	c.PC = 0x0100
	c.HL = 0xC000
	_, err := c.Step()
	require.NoError(t, err)
	assert.Equal(t, uint64(16), c.Cycles, "RES (HL): 16 cycles total")
}
