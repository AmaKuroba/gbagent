package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// testMMU is a minimal MMU stub for ALU tests.
type testMMU struct {
	data [0x10000]byte
}

func (m *testMMU) Read(addr uint16) byte       { return m.data[addr] }
func (m *testMMU) Write(addr uint16, val byte)  { m.data[addr] = val }
func (m *testMMU) Read16(addr uint16) uint16    { return uint16(m.data[addr]) | uint16(m.data[addr+1])<<8 }
func (m *testMMU) Write16(addr uint16, val uint16) { m.data[addr] = byte(val); m.data[addr+1] = byte(val >> 8) }
func (m *testMMU) LoadROM(data []byte)           {}
func (m *testMMU) LoadBootROM(data []byte)       {}

func TestALU_ADD_AB(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x12)
	cpu.setB(0x34)

	cpu.add(cpu.getB())

	assert.Equal(t, byte(0x46), cpu.getA(), "A = 0x12 + 0x34")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0 (non-zero result)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0 for ADD")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 (no half-carry from bit 3)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 (no carry from bit 7)")
}

func TestALU_ADD_AHL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0x3A)
	cpu.HL = 0xC000
	mmu.Write(0xC000, 0x8E)

	cpu.addHL()

	assert.Equal(t, byte(0xC8), cpu.getA(), "A = 0x3A + 0x8E")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0 for ADD")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 (0xA + 0xE = 0x18, bit 4 carry but bit 3 half-carry happens at overflow from bit 3)")
	assert.Equal(t, flagC, cpu.getF()&flagC, "C must be set (0x3A + 0x8E = 0xC8, no carry from bit 7)")
}

func TestALU_ADD_HalfCarry(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x0F)
	cpu.setB(0x01)

	cpu.add(cpu.getB())

	assert.Equal(t, byte(0x10), cpu.getA())
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be set (0xF + 0x1 = 0x10, half-carry from bit 3)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0")
}

func TestALU_ADD_Carry(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0xFF)
	cpu.setB(0x01)

	cpu.add(cpu.getB())

	assert.Equal(t, byte(0x00), cpu.getA(), "A = 0xFF + 0x01 = 0x00 (wrapped)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set (result = 0)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be set (0xF + 0x1 = 0x10)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagC, "C must be set (0xFF + 0x01 = 0x100, carry from bit 7)")
}

// ---------------------------------------------------------------------------

func TestALU_SUB_Zero(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x42)
	cpu.setB(0x42)

	cpu.sub(cpu.getB())

	assert.Equal(t, byte(0x00), cpu.getA(), "A = 0x42 - 0x42 = 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set (result = 0)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagN, "N must be set for SUB")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 (no half-borrow)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 (A >= val)")
}

func TestALU_SUB_Borrow(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x10)
	cpu.setB(0x20)

	cpu.sub(cpu.getB())

	assert.Equal(t, byte(0xF0), cpu.getA(), "A = 0x10 - 0x20 = 0xF0 (borrow)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagN, "N must be set for SUB")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 (no half-borrow)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagC, "C must be set (A < val)")
}

func TestALU_SUB_HalfBorrow(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x10)
	cpu.setC(0x01)

	cpu.sub(cpu.getC())

	assert.Equal(t, byte(0x0F), cpu.getA(), "A = 0x10 - 0x01 = 0x0F")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagN, "N must be set")
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be set (0x0 < 0x1, half-borrow from bit 4)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 (0x10 >= 0x01)")
}

// ---------------------------------------------------------------------------

func TestALU_AND(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0xFF)
	cpu.setB(0x0F)

	cpu.and(cpu.getB())

	assert.Equal(t, byte(0x0F), cpu.getA(), "A = 0xFF & 0x0F")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0 for AND")
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be 1 for AND")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 for AND")
}

func TestALU_AND_Zero(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0xF0)
	cpu.setB(0x0F)

	cpu.and(cpu.getB())

	assert.Equal(t, byte(0x00), cpu.getA(), "A = 0xF0 & 0x0F = 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be 1 for AND")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 for AND")
}

// ---------------------------------------------------------------------------

func TestALU_OR(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x0F)
	cpu.setB(0xF0)

	cpu.or(cpu.getB())

	assert.Equal(t, byte(0xFF), cpu.getA(), "A = 0x0F | 0xF0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0 for OR")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 for OR")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 for OR")
}

func TestALU_OR_Zero(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x00)
	cpu.setB(0x00)

	cpu.or(cpu.getB())

	assert.Equal(t, byte(0x00), cpu.getA())
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set")
}

// ---------------------------------------------------------------------------

func TestALU_XOR(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0xFF)
	cpu.setB(0x0F)

	cpu.xor(cpu.getB())

	assert.Equal(t, byte(0xF0), cpu.getA(), "A = 0xFF ^ 0x0F")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagN, "N must be 0 for XOR")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 for XOR")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0 for XOR")
}

func TestALU_XOR_Zero(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0xAB)
	cpu.setB(0xAB)

	cpu.xor(cpu.getB())

	assert.Equal(t, byte(0x00), cpu.getA(), "A ^ A = 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set")
}

// ---------------------------------------------------------------------------

func TestALU_CP_Equal(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x42)
	cpu.setB(0x42)

	cpu.cp(cpu.getB())

	assert.Equal(t, byte(0x42), cpu.getA(), "A must NOT change for CP")
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set (A == val)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagN, "N must be set for CP")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0")
}

func TestALU_CP_Less(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x05)
	cpu.setC(0x10)

	cpu.cp(cpu.getC())

	assert.Equal(t, byte(0x05), cpu.getA(), "A must NOT change")
	assert.Equal(t, byte(0x00), cpu.getF()&flagZ, "Z must be 0")
	assert.NotEqual(t, byte(0), cpu.getF()&flagN, "N must be 1")
	assert.Equal(t, byte(0x00), cpu.getF()&flagH, "H must be 0 (0x5 & 0xF = 0x5 >= 0x0 & 0xF = 0x0, no half-borrow)")
	assert.NotEqual(t, byte(0), cpu.getF()&flagC, "C must be set (A < val)")
}

func TestALU_CP_HalfBorrow(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setA(0x20)
	cpu.setD(0x01)

	cpu.cp(cpu.getD())

	assert.Equal(t, byte(0x20), cpu.getA(), "A must NOT change")
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be set (0x0 < 0x1, half-borrow from bit 4)")
	assert.Equal(t, byte(0x00), cpu.getF()&flagC, "C must be 0")
}

// ---------------------------------------------------------------------------

func TestALU_INC_Basic(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setB(0x05)

	result, z, h := cpu.inc(cpu.getB())

	assert.Equal(t, byte(0x06), result)
	assert.Equal(t, false, z, "Z must be false (result != 0)")
	assert.Equal(t, false, h, "H must be false (no half-carry)")
}

func TestALU_INC_Overflow(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setB(0xFF)

	result, z, h := cpu.inc(cpu.getB())

	assert.Equal(t, byte(0x00), result, "0xFF + 1 = 0x00")
	assert.Equal(t, true, z, "Z must be set (result == 0)")
	assert.Equal(t, true, h, "H must be set (0xF + 1 = 0x10, half-carry)")
}

func TestALU_INC_HalfCarry(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setC(0x0F)

	result, z, h := cpu.inc(cpu.getC())

	assert.Equal(t, byte(0x10), result)
	assert.Equal(t, false, z)
	assert.Equal(t, true, h, "H must be set (0xF + 1)")
}

// ---------------------------------------------------------------------------

func TestALU_DEC_Basic(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setB(0x05)

	result, z, h := cpu.dec(cpu.getB())

	assert.Equal(t, byte(0x04), result)
	assert.Equal(t, false, z, "Z must be false")
	assert.Equal(t, false, h, "H must be false (no half-borrow)")
}

func TestALU_DEC_Underflow(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setB(0x00)

	result, z, h := cpu.dec(cpu.getB())

	assert.Equal(t, byte(0xFF), result, "0x00 - 1 = 0xFF")
	assert.Equal(t, false, z, "Z must be false (result != 0)")
	assert.Equal(t, true, h, "H must be set (0x00 & 0xF == 0, half-borrow)")
}

func TestALU_DEC_Zero(t *testing.T) {
	cpu := NewCoreCPU(&testMMU{})
	cpu.setC(0x01)

	result, z, h := cpu.dec(cpu.getC())

	assert.Equal(t, byte(0x00), result)
	assert.Equal(t, true, z, "Z must be set (result == 0)")
	assert.Equal(t, false, h, "H must be false (0x01 & 0xF > 0, no half-borrow)")
}

// ---------------------------------------------------------------------------
// (HL) memory-indirect variants
// ---------------------------------------------------------------------------

func TestALU_AND_HL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0xFF)
	cpu.HL = 0xD000
	mmu.Write(0xD000, 0x0F)

	cpu.andHL()

	assert.Equal(t, byte(0x0F), cpu.getA())
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be 1 for AND")
}

func TestALU_OR_HL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0x0F)
	cpu.HL = 0xD000
	mmu.Write(0xD000, 0xF0)

	cpu.orHL()

	assert.Equal(t, byte(0xFF), cpu.getA())
}

func TestALU_XOR_HL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0xFF)
	cpu.HL = 0xD000
	mmu.Write(0xD000, 0xFF)

	cpu.xorHL()

	assert.Equal(t, byte(0x00), cpu.getA())
	assert.NotEqual(t, byte(0), cpu.getF()&flagZ, "Z must be set (0xFF ^ 0xFF = 0)")
}

func TestALU_SUB_HL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0x10)
	cpu.HL = 0xD000
	mmu.Write(0xD000, 0x01)

	cpu.subHL()

	assert.Equal(t, byte(0x0F), cpu.getA())
	assert.NotEqual(t, byte(0), cpu.getF()&flagH, "H must be set (half-borrow)")
}

func TestALU_CP_HL(t *testing.T) {
	mmu := &testMMU{}
	cpu := NewCoreCPU(mmu)
	cpu.setA(0x10)
	cpu.HL = 0xD000
	mmu.Write(0xD000, 0x20)

	cpu.cpHL()

	assert.Equal(t, byte(0x10), cpu.getA(), "A must NOT change")
	assert.NotEqual(t, byte(0), cpu.getF()&flagC, "C must be set (A < val)")
}
