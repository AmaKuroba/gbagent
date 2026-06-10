package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewMMU(t *testing.T) {
	mmu := NewMMU(nil)
	assert.NotNil(t, mmu, "NewMMU should return a non-nil instance")
}

func TestMMU_ReadWriteVRAM(t *testing.T) {
	mmu := NewMMU(nil)

	// Write a value to the start of VRAM
	mmu.Write(0x8000, 0xAB)
	val := mmu.Read(0x8000)
	assert.Equal(t, byte(0xAB), val, "should read back written value at VRAM start")

	// Write and read back at the end of VRAM
	mmu.Write(0x9FFF, 0xCD)
	val = mmu.Read(0x9FFF)
	assert.Equal(t, byte(0xCD), val, "should read back written value at VRAM end")

	// VRAM is 8KB: 0x8000-0x9FFF
	assert.Equal(t, uint16(0x2000), uint16(0x9FFF-0x8000+1), "VRAM should be 8KB")
}

func TestMMU_ReadWriteWRAM(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to the start of WRAM
	mmu.Write(0xC000, 0x12)
	val := mmu.Read(0xC000)
	assert.Equal(t, byte(0x12), val, "should read back written value at WRAM start")

	// Write to the end of WRAM
	mmu.Write(0xDFFF, 0x34)
	val = mmu.Read(0xDFFF)
	assert.Equal(t, byte(0x34), val, "should read back written value at WRAM end")

	// Verify different WRAM locations are independent
	mmu.Write(0xC100, 0x56)
	mmu.Write(0xC200, 0x78)
	assert.Equal(t, byte(0x56), mmu.Read(0xC100), "WRAM location 0xC100 should be independent")
	assert.Equal(t, byte(0x78), mmu.Read(0xC200), "WRAM location 0xC200 should be independent")
}

func TestMMU_EchoRAMMirror(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to WRAM at 0xC000
	mmu.Write(0xC000, 0xAB)

	// Read back from the echo area at 0xE000 (0xE000 - 0x2000 = 0xC000)
	val := mmu.Read(0xE000)
	assert.Equal(t, byte(0xAB), val, "echo area 0xE000 should mirror 0xC000")

	// Write to WRAM at 0xD000
	mmu.Write(0xD000, 0xCD)

	// Read back from echo area at 0xF000 (0xF000 - 0x2000 = 0xD000)
	val = mmu.Read(0xF000)
	assert.Equal(t, byte(0xCD), val, "echo area 0xF000 should mirror 0xD000")

	// Write to echo area and verify it reflects in WRAM
	// Note: on real hardware, writes to echo area also go to WRAM
	mmu.Write(0xE001, 0xEF)
	val = mmu.Read(0xC001)
	assert.Equal(t, byte(0xEF), val, "writing to echo 0xE001 should appear in WRAM 0xC001")
}

func TestMMU_OAMBounds(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to OAM
	mmu.Write(0xFE00, 0xAA)
	val := mmu.Read(0xFE00)
	assert.Equal(t, byte(0xAA), val, "should read back written value at OAM start")

	// Write to last byte of OAM
	mmu.Write(0xFE9F, 0xBB)
	val = mmu.Read(0xFE9F)
	assert.Equal(t, byte(0xBB), val, "should read back at OAM end")

	// OAM is 160 bytes (0xA0): 0xFE00-0xFE9F
	assert.Equal(t, uint16(0xA0), uint16(0xFE9F-0xFE00+1), "OAM should be 160 bytes")

	// Unused area 0xFEA0-0xFEFF should read back something consistent
	unusedVal := mmu.Read(0xFEA0)
	// Should not panic or return anything anomalous
	assert.Equal(t, byte(0x00), unusedVal, "unused area should return 0x00 on read")
}

func TestMMU_HRAM(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to HRAM start
	mmu.Write(0xFF80, 0xDE)
	val := mmu.Read(0xFF80)
	assert.Equal(t, byte(0xDE), val, "should read back written value at HRAM start")

	// Write to HRAM end
	mmu.Write(0xFFFE, 0xAD)
	val = mmu.Read(0xFFFE)
	assert.Equal(t, byte(0xAD), val, "should read back written value at HRAM end")

	// HRAM is 127 bytes: 0xFF80-0xFFFE
	assert.Equal(t, 127, 0xFFFE-0xFF80+1, "HRAM should be 127 bytes")
}

func TestMMU_IORegisters(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to IO register area
	mmu.Write(0xFF00, 0x01)
	val := mmu.Read(0xFF00)
	assert.Equal(t, byte(0x01), val, "should read back written value at IO register 0xFF00")

	// Write to last IO register
	mmu.Write(0xFF7F, 0x02)
	val = mmu.Read(0xFF7F)
	assert.Equal(t, byte(0x02), val, "should read back written value at IO register 0xFF7F")
}

func TestMMU_IERegister(t *testing.T) {
	mmu := NewMMU(nil)

	// Write to IE register
	mmu.Write(0xFFFF, 0x1F)
	val := mmu.Read(0xFFFF)
	assert.Equal(t, byte(0x1F), val, "should read back written IE register value")

	// IE register should hold its value across writes
	mmu.Write(0xFFFF, 0x00)
	val = mmu.Read(0xFFFF)
	assert.Equal(t, byte(0x00), val, "IE register should update on write")
}

func TestMMU_Read16(t *testing.T) {
	mmu := NewMMU(nil)

	// Write two bytes to consecutive VRAM locations
	mmu.Write(0x8000, 0x34)
	mmu.Write(0x8001, 0x12)

	// Read16 should return little-endian (low byte first)
	val := mmu.Read16(0x8000)
	assert.Equal(t, uint16(0x1234), val, "Read16 should return little-endian value")
}

func TestMMU_Write16(t *testing.T) {
	mmu := NewMMU(nil)

	// Write16 should store little-endian
	mmu.Write16(0x8000, 0x1234)

	val := mmu.Read(0x8000)
	assert.Equal(t, byte(0x34), val, "Write16 low byte should be at addr")
	val = mmu.Read(0x8001)
	assert.Equal(t, byte(0x12), val, "Write16 high byte should be at addr+1")
}

func TestMMU_DefaultValues(t *testing.T) {
	mmu := NewMMU(nil)

	// Default values for writable regions should be 0
	assert.Equal(t, byte(0x00), mmu.Read(0x8000), "VRAM should init to 0")
	assert.Equal(t, byte(0x00), mmu.Read(0xC000), "WRAM should init to 0")
	assert.Equal(t, byte(0x00), mmu.Read(0xFE00), "OAM should init to 0")
	assert.Equal(t, byte(0x00), mmu.Read(0xFF80), "HRAM should init to 0")
	assert.Equal(t, byte(0x00), mmu.Read(0xFFFF), "IE should init to 0")

	// ROM area should not be accessible without loaded ROM
	assert.Equal(t, byte(0xFF), mmu.Read(0x0000), "ROM without data should return 0xFF")
}
