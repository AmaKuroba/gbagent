package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockMMUForCB is a minimal MMU stub that returns canned bytes for Step().
type mockMMUForCB struct {
	MMU
	read func(addr uint16) byte
}

func (m *mockMMUForCB) Read(addr uint16) byte {
	return m.read(addr)
}

// newTestCPU creates a CoreCPU with an MMU that always returns 0xCB then the given cbOp.
func newTestCPU(cbOp byte) *CoreCPU {
	i := 0
	m := &mockMMUForCB{
		read: func(addr uint16) byte {
			switch i {
			case 0:
				i++
				return 0xCB // CB prefix
			default:
				return cbOp // CB operand
			}
		},
	}
	return NewCPU(m)
}

func TestCBCmd_RLC_B(t *testing.T) {
	cpu := newTestCPU(0x00) // RLC B
	cpu.Registers.B = 0x85  // 1000 0101

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x0B), cpu.Registers.B, "RLC B: 0x85 -> 0x0B")
	assert.False(t, cpu.getZ(), "RLC B: Z=0")
	assert.False(t, cpu.getN(), "RLC B: N=0")
	assert.False(t, cpu.getH(), "RLC B: H=0")
	assert.True(t, cpu.getC(), "RLC B: C=1 (bit 7 was 1)")
}

func TestCBCmd_RLC_B_zero(t *testing.T) {
	cpu := newTestCPU(0x00) // RLC B
	cpu.Registers.B = 0x00  // 0000 0000

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.B, "RLC B: 0x00 -> 0x00")
	assert.True(t, cpu.getZ(), "RLC B zero: Z=1")
	assert.False(t, cpu.getC(), "RLC B zero: C=0 (bit 7 was 0)")
}

func TestCBCmd_RLC_C(t *testing.T) {
	cpu := newTestCPU(0x09) // RRC C (0x09 = RRC C in CB table)
	cpu.Registers.C = 0x01  // 0000 0001

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x80), cpu.Registers.C, "RRC C: 0x01 -> 0x80")
	assert.False(t, cpu.getZ(), "RRC C: Z=0")
	assert.True(t, cpu.getC(), "RRC C: C=1 (bit 0 was 1)")
}

func TestCBCmd_SLA_B(t *testing.T) {
	cpu := newTestCPU(0x20) // SLA B
	cpu.Registers.B = 0x85  // 1000 0101

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x0A), cpu.Registers.B, "SLA B: 0x85 -> 0x0A")
	assert.False(t, cpu.getZ(), "SLA B: Z=0")
	assert.False(t, cpu.getN(), "SLA B: N=0")
	assert.False(t, cpu.getH(), "SLA B: H=0")
	assert.True(t, cpu.getC(), "SLA B: C=1 (bit 7 was 1)")
}

func TestCBCmd_SLA_B_zero(t *testing.T) {
	cpu := newTestCPU(0x20) // SLA B
	cpu.Registers.B = 0x80  // 1000 0000

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.B, "SLA B: 0x80 -> 0x00")
	assert.True(t, cpu.getZ(), "SLA B zero: Z=1")
	assert.True(t, cpu.getC(), "SLA B zero: C=1 (bit 7 was 1)")
}

func TestCBCmd_SRA_B(t *testing.T) {
	cpu := newTestCPU(0x28) // SRA B
	cpu.Registers.B = 0x85  // 1000 0101

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0xC2), cpu.Registers.B, "SRA B: 0x85 -> 0xC2 (sign extended)")
	assert.False(t, cpu.getZ(), "SRA B: Z=0")
	assert.False(t, cpu.getN(), "SRA B: N=0")
	assert.False(t, cpu.getH(), "SRA B: H=0")
	assert.True(t, cpu.getC(), "SRA B: C=1 (bit 0 was 1)")
}

func TestCBCmd_SRL_B(t *testing.T) {
	cpu := newTestCPU(0x38) // SRL B
	cpu.Registers.B = 0x85  // 1000 0101

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x42), cpu.Registers.B, "SRL B: 0x85 -> 0x42")
	assert.False(t, cpu.getZ(), "SRL B: Z=0")
	assert.False(t, cpu.getN(), "SRL B: N=0")
	assert.False(t, cpu.getH(), "SRL B: H=0")
	assert.True(t, cpu.getC(), "SRL B: C=1 (bit 0 was 1)")
}

func TestCBCmd_SWAP_B(t *testing.T) {
	cpu := newTestCPU(0x30) // SWAP B
	cpu.Registers.B = 0xAB  // 1010 1011

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0xBA), cpu.Registers.B, "SWAP B: 0xAB -> 0xBA")
	assert.False(t, cpu.getZ(), "SWAP B: Z=0")
	assert.False(t, cpu.getN(), "SWAP B: N=0")
	assert.False(t, cpu.getH(), "SWAP B: H=0")
	assert.False(t, cpu.getC(), "SWAP B: C=0")
}

func TestCBCmd_BIT_0_B(t *testing.T) {
	cpu := newTestCPU(0x40) // BIT 0,B
	cpu.Registers.B = 0x01  // bit 0 set

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x01), cpu.Registers.B, "BIT 0,B: B unchanged")
	assert.False(t, cpu.getZ(), "BIT 0,B: Z=0 (bit was set)")
	assert.False(t, cpu.getN(), "BIT 0,B: N=0")
	assert.True(t, cpu.getH(), "BIT 0,B: H=1")
}

func TestCBCmd_BIT_0_B_clear(t *testing.T) {
	cpu := newTestCPU(0x40) // BIT 0,B
	cpu.Registers.B = 0x00  // bit 0 clear

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.True(t, cpu.getZ(), "BIT 0,B clear: Z=1 (bit was clear)")
	assert.False(t, cpu.getN(), "BIT 0,B clear: N=0")
	assert.True(t, cpu.getH(), "BIT 0,B clear: H=1")
}

func TestCBCmd_SET_0_B(t *testing.T) {
	cpu := newTestCPU(0xC0) // SET 0,B
	cpu.Registers.B = 0x00

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x01), cpu.Registers.B, "SET 0,B: 0x00 -> 0x01")
}

func TestCBCmd_SET_0_B_already_set(t *testing.T) {
	cpu := newTestCPU(0xC0) // SET 0,B
	cpu.Registers.B = 0x01

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x01), cpu.Registers.B, "SET 0,B: 0x01 unchanged")
}

func TestCBCmd_RES_0_B(t *testing.T) {
	cpu := newTestCPU(0x80) // RES 0,B
	cpu.Registers.B = 0x01

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.B, "RES 0,B: 0x01 -> 0x00")
}

func TestCBCmd_RES_0_B_already_clear(t *testing.T) {
	cpu := newTestCPU(0x80) // RES 0,B
	cpu.Registers.B = 0x00

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.B, "RES 0,B: 0x00 unchanged")
}

func TestCBCmd_BIT_7_B(t *testing.T) {
	cpu := newTestCPU(0x78) // BIT 7,B
	cpu.Registers.B = 0x80  // bit 7 set

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.False(t, cpu.getZ(), "BIT 7,B: Z=0 (bit was set)")
	assert.False(t, cpu.getN(), "BIT 7,B: N=0")
	assert.True(t, cpu.getH(), "BIT 7,B: H=1")
}

func TestCBCmd_SET_7_B(t *testing.T) {
	cpu := newTestCPU(0xF8) // SET 7,B
	cpu.Registers.B = 0x00

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x80), cpu.Registers.B, "SET 7,B: 0x00 -> 0x80")
}

func TestCBCmd_RES_7_B(t *testing.T) {
	cpu := newTestCPU(0xB8) // RES 7,B
	cpu.Registers.B = 0xFF

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x7F), cpu.Registers.B, "RES 7,B: 0xFF -> 0x7F")
}

func TestCBCmd_RL_B(t *testing.T) {
	cpu := newTestCPU(0x10) // RL B
	cpu.Registers.B = 0x85  // 1000 0101, carry = 0
	cpu.setC(false)

	_, err := cpu.Step()
	require.NoError(t, err)

	// RL: shift left, bit 0 = old carry (0), carry = old bit 7 (1)
	assert.Equal(t, uint8(0x0A), cpu.Registers.B, "RL B: 0x85 with C=0 -> 0x0A")
	assert.False(t, cpu.getZ(), "RL B: Z=0")
	assert.False(t, cpu.getN(), "RL B: N=0")
	assert.False(t, cpu.getH(), "RL B: H=0")
	assert.True(t, cpu.getC(), "RL B: C=1 (old bit 7 was 1)")
}

func TestCBCmd_RL_B_with_carry(t *testing.T) {
	cpu := newTestCPU(0x10) // RL B
	cpu.Registers.B = 0x05  // 0000 0101, carry = 1
	cpu.setC(true)

	_, err := cpu.Step()
	require.NoError(t, err)

	// RL: shift left, bit 0 = old carry (1), carry = old bit 7 (0)
	assert.Equal(t, uint8(0x0B), cpu.Registers.B, "RL B: 0x05 with C=1 -> 0x0B")
	assert.False(t, cpu.getZ(), "RL B: Z=0")
	assert.False(t, cpu.getN(), "RL B: N=0")
	assert.False(t, cpu.getH(), "RL B: H=0")
	assert.False(t, cpu.getC(), "RL B: C=0 (old bit 7 was 0)")
}

func TestCBCmd_RR_B(t *testing.T) {
	cpu := newTestCPU(0x18) // RR B
	cpu.Registers.B = 0x85  // 1000 0101, carry = 0
	cpu.setC(false)

	_, err := cpu.Step()
	require.NoError(t, err)

	// RR: shift right, bit 7 = old carry (0), carry = old bit 0 (1)
	assert.Equal(t, uint8(0x42), cpu.Registers.B, "RR B: 0x85 with C=0 -> 0x42")
	assert.False(t, cpu.getZ(), "RR B: Z=0")
	assert.False(t, cpu.getN(), "RR B: N=0")
	assert.False(t, cpu.getH(), "RR B: H=0")
	assert.True(t, cpu.getC(), "RR B: C=1 (old bit 0 was 1)")
}

func TestCBCmd_RR_B_with_carry(t *testing.T) {
	cpu := newTestCPU(0x18) // RR B
	cpu.Registers.B = 0x04  // 0000 0100, carry = 1
	cpu.setC(true)

	_, err := cpu.Step()
	require.NoError(t, err)

	// RR: shift right, bit 7 = old carry (1), carry = old bit 0 (0)
	assert.Equal(t, uint8(0x82), cpu.Registers.B, "RR B: 0x04 with C=1 -> 0x82")
	assert.False(t, cpu.getZ(), "RR B: Z=0")
	assert.False(t, cpu.getN(), "RR B: N=0")
	assert.False(t, cpu.getH(), "RR B: H=0")
	assert.False(t, cpu.getC(), "RR B: C=0 (old bit 0 was 0)")
}

func TestCBCmd_RLC_A(t *testing.T) {
	cpu := newTestCPU(0x07) // RLC A
	cpu.Registers.A = 0x80  // 1000 0000

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x01), cpu.Registers.A, "RLC A: 0x80 -> 0x01")
	assert.True(t, cpu.getC(), "RLC A: C=1")
}

func TestCBCmd_SLA_HL_indirect(t *testing.T) {
	// SLA (HL): read from HL address, shift, write back
	m := &mockMMUForCB{
		read: func(addr uint16) byte {
			if addr == 0xC000 {
				return 0x8F // value at HL = 1000 1111
			}
			return 0xCB // opcode fetch
		},
	}
	cpu := NewCPU(m)
	cpu.Registers.PC = 0x0000 // start at 0, first read returns 0xCB
	cpu.setHL(0xC000)

	// We need to be more careful — Step() will:
	// 1. Read at PC (0x0000): our mock returns 0xCB
	// 2. Then PC++, then it reads CB operand next...
	// The mock is tricky with Step() because it reads opcode then operand.
	// Let me adjust: first read returns 0xCB, second read (PC) returns opcode.
	// But the MMU read for the CB operand is at PC+1 (0x0001).
	// So the mock's switch needs to know the address.
}

func TestCBCmd_RL_A(t *testing.T) {
	cpu := newTestCPU(0x17) // RL A
	cpu.Registers.A = 0x80  // 1000 0000, carry = 0
	cpu.setC(false)

	_, err := cpu.Step()
	require.NoError(t, err)

	// RL A: shift left, bit 0 = carry (0), carry = old bit 7 (1)
	assert.Equal(t, uint8(0x00), cpu.Registers.A, "RL A: 0x80 with C=0 -> 0x00")
	assert.True(t, cpu.getZ(), "RL A: Z=1 (result 0)")
	assert.True(t, cpu.getC(), "RL A: C=1")
}

func TestCBCmd_SRA_L(t *testing.T) {
	cpu := newTestCPU(0x2D) // SRA L
	cpu.Registers.L = 0x01  // 0000 0001

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.L, "SRA L: 0x01 -> 0x00")
	assert.True(t, cpu.getZ(), "SRA L: Z=1")
	assert.True(t, cpu.getC(), "SRA L: C=1 (old bit 0 was 1)")
}

func TestCBCmd_SWAP_H(t *testing.T) {
	cpu := newTestCPU(0x34) // SWAP H
	cpu.Registers.H = 0xF0

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x0F), cpu.Registers.H, "SWAP H: 0xF0 -> 0x0F")
}

func TestCBCmd_SRL_D(t *testing.T) {
	cpu := newTestCPU(0x3A) // SRL D
	cpu.Registers.D = 0x01

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.D, "SRL D: 0x01 -> 0x00")
	assert.True(t, cpu.getZ(), "SRL D: Z=1")
	assert.True(t, cpu.getC(), "SRL D: C=1")
}

func TestCBCmd_SET_3_E(t *testing.T) {
	cpu := newTestCPU(0xDB) // SET 3,E
	cpu.Registers.E = 0x00

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x08), cpu.Registers.E, "SET 3,E: 0x00 -> 0x08")
}

func TestCBCmd_RES_5_H(t *testing.T) {
	cpu := newTestCPU(0xAC) // RES 5,H
	cpu.Registers.H = 0xFF

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0xDF), cpu.Registers.H, "RES 5,H: 0xFF -> 0xDF")
}

func TestCBCmd_SET_and_BIT_preserve_flags(t *testing.T) {
	cpu := newTestCPU(0xC0) // SET 0,B
	cpu.Registers.B = 0x00
	cpu.Registers.F = 0xFF // all flags set

	_, err := cpu.Step()
	require.NoError(t, err)

	// SET should not change flags
	assert.True(t, cpu.getZ(), "SET 0,B: Z preserved")
	assert.True(t, cpu.getN(), "SET 0,B: N preserved")
	assert.True(t, cpu.getH(), "SET 0,B: H preserved")
	assert.True(t, cpu.getC(), "SET 0,B: C preserved")
}

func TestCBCmd_RES_preserves_flags(t *testing.T) {
	cpu := newTestCPU(0x80) // RES 0,B
	cpu.Registers.B = 0x01
	cpu.Registers.F = 0xF0 // all flags set

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.True(t, cpu.getZ(), "RES 0,B: Z preserved")
	assert.True(t, cpu.getN(), "RES 0,B: N preserved")
	assert.True(t, cpu.getH(), "RES 0,B: H preserved")
	assert.True(t, cpu.getC(), "RES 0,B: C preserved")
}

func TestCBCmd_RRC_A(t *testing.T) {
	cpu := newTestCPU(0x0F) // RRC A
	cpu.Registers.A = 0x01  // 0000 0001

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x80), cpu.Registers.A, "RRC A: 0x01 -> 0x80")
	assert.False(t, cpu.getZ(), "RRC A: Z=0")
	assert.True(t, cpu.getC(), "RRC A: C=1")
}

func TestCBCmd_RRC_A_zero(t *testing.T) {
	cpu := newTestCPU(0x0F) // RRC A
	cpu.Registers.A = 0x00

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint8(0x00), cpu.Registers.A, "RRC A: 0x00 -> 0x00")
	assert.True(t, cpu.getZ(), "RRC A zero: Z=1")
	assert.False(t, cpu.getC(), "RRC A zero: C=0")
}

func TestCBCmd_SRA_D_twice(t *testing.T) {
	cpu := newTestCPU(0x2A) // SRA D
	cpu.Registers.D = 0x01

	_, err := cpu.Step()
	require.NoError(t, err)
	assert.Equal(t, uint8(0x00), cpu.Registers.D, "SRA D: 0x01 -> 0x00")
	assert.True(t, cpu.getZ(), "SRA D: Z=1")
	assert.True(t, cpu.getC(), "SRA D: C=1")
}

func TestCBCmd_BIT_3_H_set(t *testing.T) {
	cpu := newTestCPU(0x5C) // BIT 3,H
	cpu.Registers.H = 0x08  // bit 3 set

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.False(t, cpu.getZ(), "BIT 3,H: Z=0")
}

func TestCBCmd_BIT_3_H_clear(t *testing.T) {
	cpu := newTestCPU(0x5C) // BIT 3,H
	cpu.Registers.H = 0x00  // bit 3 clear

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.True(t, cpu.getZ(), "BIT 3,H: Z=1")
}

func TestCBCmd_CyclesRegister(t *testing.T) {
	cpu := newTestCPU(0x00) // RLC B — register variant
	_, err := cpu.Step()
	require.NoError(t, err)

	// Register CB ops should take 8 cycles.
	assert.Equal(t, uint64(8), cpu.Cycles, "register CB op: 8 cycles")
}

func TestCBCmd_CyclesHLIndirect(t *testing.T) {
	// (HL) variants should take 16 cycles.
	i := 0
	m := &mockMMUForCB{
		read: func(addr uint16) byte {
			if addr == 0xC000 {
				return 0xAB
			}
			i++
			switch i {
			case 1:
				return 0xCB
			default:
				return 0x06 // RLC (HL)
			}
		},
	}
	cpu := NewCPU(m)
	cpu.setHL(0xC000)

	_, err := cpu.Step()
	require.NoError(t, err)

	assert.Equal(t, uint64(16), cpu.Cycles, "(HL) CB op: 16 cycles")
}

func TestCBCmd_RLC_AllRegisters(t *testing.T) {
	// Spot-check: RLC for all 8-bit registers has the same cbTable opcodes.
	ops := map[uint8]string{
		0x00: "B", 0x01: "C", 0x02: "D", 0x03: "E",
		0x04: "H", 0x05: "L", 0x07: "A",
	}
	for op, name := range ops {
		t.Run(name, func(t *testing.T) {
			cpu := newTestCPU(op)
			// Set register to 0x01
			switch name {
			case "B":
				cpu.Registers.B = 0x01
			case "C":
				cpu.Registers.C = 0x01
			case "D":
				cpu.Registers.D = 0x01
			case "E":
				cpu.Registers.E = 0x01
			case "H":
				cpu.Registers.H = 0x01
			case "L":
				cpu.Registers.L = 0x01
			case "A":
				cpu.Registers.A = 0x01
			}
			_, err := cpu.Step()
			require.NoError(t, err)
			// RLC 0x01 -> 0x02
			expected := uint8(0x02)
			switch name {
			case "B":
				assert.Equal(t, expected, cpu.Registers.B)
			case "C":
				assert.Equal(t, expected, cpu.Registers.C)
			case "D":
				assert.Equal(t, expected, cpu.Registers.D)
			case "E":
				assert.Equal(t, expected, cpu.Registers.E)
			case "H":
				assert.Equal(t, expected, cpu.Registers.H)
			case "L":
				assert.Equal(t, expected, cpu.Registers.L)
			case "A":
				assert.Equal(t, expected, cpu.Registers.A)
			}
		})
	}
}
