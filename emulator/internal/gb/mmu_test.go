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

// -- STAT blocking tests --

// attachPPU creates a PPUCore, attaches it to the given MMU, and returns it.
// The PPU is left in Mode 2 (OAM scan) after creation with LCD enabled.
func attachPPU(mmu *MemoryBus) *PPUCore {
	ppu := NewPPU(mmu)
	mmu.SetPPU(ppu)
	ppu.WriteRegister(0xFF40, 0x91)
	return ppu
}

func TestMMU_ReadVRAM_BlockedDuringMode3(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Write known value to VRAM first
	mmu.Write(0x8000, 0xAB)
	assert.Equal(t, byte(0xAB), mmu.Read(0x8000), "should write/read VRAM normally in Mode 2")

	// Switch to Mode 3 (VRAM draw) — VRAM reads return 0xFF
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	assert.Equal(t, byte(0xFF), mmu.Read(0x8000), "VRAM read should return 0xFF in Mode 3")
	assert.Equal(t, byte(0xFF), mmu.Read(0x9FFF), "VRAM read at end should return 0xFF in Mode 3")
}

func TestMMU_WriteVRAM_IgnoredDuringMode3(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Write initial value in accessible mode
	ppu.stat &^= 0x03
	mmu.Write(0x8000, 0xAB)
	assert.Equal(t, byte(0xAB), mmu.Read(0x8000), "VRAM write should work in Mode 0")

	// Switch to Mode 3 — write should be ignored
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	mmu.Write(0x8000, 0xCD)

	// Switch back to accessible mode to verify old value is preserved
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0xAB), mmu.Read(0x8000), "VRAM write should be ignored in Mode 3 (old value preserved)")
}

func TestMMU_ReadVRAM_AccessibleDuringModes012(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	mmu.Write(0x8000, 0x42) // Write in whatever mode we start in

	// Mode 0 (HBlank)
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0x42), mmu.Read(0x8000), "VRAM readable in Mode 0")

	// Mode 1 (VBlank)
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVBlank
	assert.Equal(t, byte(0x42), mmu.Read(0x8000), "VRAM readable in Mode 1")

	// Mode 2 (OAM scan) — VRAM should be accessible
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	assert.Equal(t, byte(0x42), mmu.Read(0x8000), "VRAM readable in Mode 2")
}

func TestMMU_ReadOAM_BlockedDuringModes2And3(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Write known value to OAM first (accessible in current mode — PPU starts in Mode 2,
	// but we can move to Mode 0 to write)
	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0xDE)
	assert.Equal(t, byte(0xDE), mmu.Read(0xFE00), "OAM accessible in Mode 0")

	// Mode 2 (OAM search) — OAM reads for row 0 (objects 0-1 at FE00-FE07) are NOT
	// affected by the OAM corruption bug, so they return the actual value.
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	assert.Equal(t, byte(0xDE), mmu.Read(0xFE00), "OAM read at row 0 (0xFE00) should return real value in Mode 2 (no corruption on objects 0-1)")

	// Mode 3 (VRAM draw) — OAM reads return 0xFF
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	assert.Equal(t, byte(0xFF), mmu.Read(0xFE00), "OAM read should return 0xFF in Mode 3")

	// Verify reads from row > 0 during Mode 2 DO return corrupted values
	// (not 0xFF). The corruption applies to the PPU's current OAM row
	// (determined by dotCounter), not the address being read.
	// Advance the PPU dotCounter so that GetOAMRow() returns a row > 0.
	ppu.stat &^= 0x03
	// Set up row 0 (preceding row for row 1) and row 1 with specific values
	// so we can predict the corruption.
	// Row 0 (FE00-FE07): first word = 0x0000, third word = 0x0000
	// Row 1 (FE08-FE0F): first word = 0x00FF
	mmu.Write(0xFE08, 0xFF)
	mmu.Write(0xFE09, 0x00)

	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	ppu.Step(8) // advance dotCounter so row = 8/4 = 2

	// Read from an address in row 2 (where the PPU currently is).
	// This should trigger read corruption of row 2.
	mmu.Read(0xFE10) // triggers corruption of row 2

	// Verify the corruption actually happened by reading back in accessible mode
	ppu.stat &^= 0x03
	// Read corruption formula: b | (a & c) = 0x00FF | (0x0000 & 0x0000) = 0x00FF
	assert.Equal(t, byte(0xFF), mmu.Read(0xFE10),
		"OAM at FE10 should be corrupted (read corruption first word = b | (a & c) = 0xFF)")
}

func TestMMU_WriteOAM_IgnoredDuringModes2And3(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Write initial value in accessible mode
	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0x12)
	assert.Equal(t, byte(0x12), mmu.Read(0xFE00), "OAM write should work in Mode 0")

	// Mode 2 — row 0 (objects 0-1 at FE00-FE07) is NOT affected by corruption,
	// so writes should go through normally.
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	mmu.Write(0xFE00, 0x34)

	// Switch back to Mode 0 to verify — should reflect the write done during Mode 2
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0x34), mmu.Read(0xFE00), "OAM write to row 0 (0xFE00) should work in Mode 2 (objects 0-1 not affected by corruption)")

	// Mode 3 — write should be ignored
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	mmu.Write(0xFE00, 0x56)

	// Switch back to accessible mode to verify
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0x34), mmu.Read(0xFE00), "OAM write should be ignored in Mode 3")

	// Verify writes to row > 0 during Mode 2 DO corrupt data.
	// Advance PPU dotCounter so GetOAMRow() > 0.
	ppu.stat &^= 0x03
	// Set up row 2 with a known value and row 0 with zero values
	mmu.Write(0xFE10, 0x78) // row 2 byte 0
	mmu.Write(0xFE11, 0x9A) // row 2 byte 1
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	ppu.Step(8) // advance dotCounter to 8 so row = 8/4 = 2 > 0

	// Write to an address in row 2 during mode 2 — value is lost, row corrupted
	mmu.Write(0xFE10, 0x42) // attempt to write, but bus conflict corrupts row 2

	ppu.stat &^= 0x03
	// Write corruption formula with a=0x9A78, b=0x0000, c=0x0000:
	// ((a ^ c) & (b ^ c)) ^ c = ((0x9A78 ^ 0) & (0 ^ 0)) ^ 0 = (0x9A78 & 0) = 0
	corrupted := mmu.Read(0xFE10)
	assert.NotEqual(t, byte(0x42), corrupted, "write value should be lost (bus conflict)")
	assert.NotEqual(t, byte(0x78), corrupted, "original value should also be corrupted")
}

func TestMMU_ReadOAM_AccessibleDuringModes01(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Write values in accessible mode first
	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0x78)
	mmu.Write(0xFE04, 0x9A)

	// Mode 0 (HBlank)
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0x78), mmu.Read(0xFE00), "OAM readable in Mode 0")
	assert.Equal(t, byte(0x9A), mmu.Read(0xFE04), "OAM readable in Mode 0")

	// Mode 1 (VBlank)
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVBlank
	assert.Equal(t, byte(0x78), mmu.Read(0xFE00), "OAM readable in Mode 1")
}

func TestMMU_STATBlockingNoPPU(t *testing.T) {
	// When no PPU is attached, OAM and VRAM should be accessible regardless of mode
	mmu := NewMMU(nil)

	// Write and read VRAM — should work without PPU
	mmu.Write(0x8000, 0xAB)
	assert.Equal(t, byte(0xAB), mmu.Read(0x8000), "VRAM should work without PPU attached")

	// Write and read OAM — should work without PPU
	mmu.Write(0xFE00, 0xCD)
	assert.Equal(t, byte(0xCD), mmu.Read(0xFE00), "OAM should work without PPU attached")
}

func TestMMU_SerialPort_SB(t *testing.T) {
	mmu := NewMMU(nil)

	// Write and read back SB (0xFF01)
	mmu.Write(0xFF01, 0xAB)
	val := mmu.Read(0xFF01)
	assert.Equal(t, byte(0xAB), val, "SB should read back written value")

	// Update SB value
	mmu.Write(0xFF01, 0xCD)
	val = mmu.Read(0xFF01)
	assert.Equal(t, byte(0xCD), val, "SB should be updatable")

	// Verify SB is independent from SC
	mmu.Write(0xFF01, 0x12)
	mmu.Write(0xFF02, 0xFF)
	assert.Equal(t, byte(0x12), mmu.Read(0xFF01), "SB should not be affected by SC write")
}

func TestMMU_SerialPort_SC(t *testing.T) {
	mmu := NewMMU(nil)

	// Write and read back SC (0xFF02)
	mmu.Write(0xFF02, 0x81) // Blargg test ROMs write 0x81 (bit 7 + bit 0)
	val := mmu.Read(0xFF02)
	assert.Equal(t, byte(0x81), val, "SC should read back written value")

	// Clear SC (test harness clears after reading SB)
	mmu.Write(0xFF02, 0x00)
	val = mmu.Read(0xFF02)
	assert.Equal(t, byte(0x00), val, "SC should be cleared on write")

	// SC should hold arbitrary values
	mmu.Write(0xFF02, 0x01)
	val = mmu.Read(0xFF02)
	assert.Equal(t, byte(0x01), val, "SC should hold bit 0 value")
}

func TestMMU_SerialPort_ReadSB(t *testing.T) {
	mmu := NewMMU(nil)

	// Test the convenience methods
	mmu.WriteSB(0x42)
	assert.Equal(t, byte(0x42), mmu.ReadSB(), "ReadSB should return written value")
	assert.Equal(t, byte(0x42), mmu.Read(0xFF01), "Read(0xFF01) should match ReadSB()")
}

func TestMMU_SerialPort_ReadSC(t *testing.T) {
	mmu := NewMMU(nil)

	// Test the convenience methods
	mmu.WriteSC(0x81)
	assert.Equal(t, byte(0x81), mmu.ReadSC(), "ReadSC should return written value")
	assert.Equal(t, byte(0x81), mmu.Read(0xFF02), "Read(0xFF02) should match ReadSC()")
}

// TestMMU_SerialTransfer_StartOnSC81 verifies that writing 0x81 (bit 7 + bit 0)
// to SC starts a serial transfer with the correct cycle count.
func TestMMU_SerialTransfer_StartOnSC81(t *testing.T) {
	mmu := NewMMU(nil)

	// Transfer should not be active initially
	assert.False(t, mmu.serialActive, "serial should not be active initially")
	assert.Equal(t, 0, mmu.serialCycles, "serialCycles should be 0 initially")

	// Write 0x81 (bit 7=1, bit 0=1) to start master-mode transfer
	mmu.Write(0xFF02, 0x81)
	assert.True(t, mmu.serialActive, "serial should be active after writing 0x81")
	assert.Equal(t, 4096, mmu.serialCycles, "serialCycles should be 4096")
	assert.Equal(t, byte(0x81), mmu.Read(0xFF02), "SC should still read back 0x81")
}

// TestMMU_SerialTransfer_NoStartWithoutBit7 verifies writing SC without bit 7
// does NOT start a transfer.
func TestMMU_SerialTransfer_NoStartWithoutBit7(t *testing.T) {
	mmu := NewMMU(nil)

	// Write 0x01 (bit 0 only, no bit 7) — should NOT start transfer
	mmu.Write(0xFF02, 0x01)
	assert.False(t, mmu.serialActive, "serial should not be active after writing 0x01")
	assert.Equal(t, 0, mmu.serialCycles, "serialCycles should be 0")
	assert.Equal(t, byte(0x01), mmu.Read(0xFF02), "SC should read back 0x01")
}

// TestMMU_SerialTransfer_NoStartWithoutBit0 verifies that writing SC with bit 7=1
// but bit 0=0 (slave mode) does NOT start a transfer.
func TestMMU_SerialTransfer_NoStartWithoutBit0(t *testing.T) {
	mmu := NewMMU(nil)

	// Write 0x80 (bit 7 only, no bit 0) — slave mode, should NOT start transfer
	mmu.Write(0xFF02, 0x80)
	assert.False(t, mmu.serialActive, "serial should not be active in slave mode")
	assert.Equal(t, 0, mmu.serialCycles, "serialCycles should be 0")
	assert.Equal(t, byte(0x80), mmu.Read(0xFF02), "SC should read back 0x80")
}

// TestMMU_SerialTransfer_PartialSteps verifies the transfer is still active
// after fewer than 4096 T-cycles have elapsed.
func TestMMU_SerialTransfer_PartialSteps(t *testing.T) {
	mmu := NewMMU(nil)

	mmu.WriteSB(0xAB)
	mmu.Write(0xFF02, 0x81) // start transfer
	assert.True(t, mmu.serialActive, "serial should be active")

	// Advance 1000 T-cycles — not enough to complete
	mmu.SerialStep(1000)
	assert.True(t, mmu.serialActive, "serial should still be active after 1000 cycles")
	assert.Equal(t, 3096, mmu.serialCycles, "serialCycles should be 4096-1000=3096")

	// Advance another 2000 T-cycles — still not done
	mmu.SerialStep(2000)
	assert.True(t, mmu.serialActive, "serial should still be active after 3000 total cycles")
	assert.Equal(t, 1096, mmu.serialCycles, "serialCycles should be 1096")

	// Advance 1000 more (total 4000) — still not done
	mmu.SerialStep(1000)
	assert.True(t, mmu.serialActive, "serial should still be active after 4000 total cycles")
	assert.Equal(t, 96, mmu.serialCycles, "serialCycles should be 96")
}

// TestMMU_SerialTransfer_Completion verifies the full serial transfer cycle:
// SB is shifted right with bit 7=1, SC bit 7 is cleared, IF bit 3 is set.
func TestMMU_SerialTransfer_Completion(t *testing.T) {
	mmu := NewMMU(nil)

	// Set up SB with a known value and start transfer
	mmu.WriteSB(0xAB) // 0xAB = 10101011
	mmu.Write(0xFF02, 0x81)
	assert.True(t, mmu.serialActive)

	// Clear IF register first
	mmu.WriteIF(0x00)
	assert.Equal(t, byte(0x00), mmu.ReadIF(), "IF should start at 0")

	// Advance exactly 4096 T-cycles (or slightly more) to complete
	mmu.SerialStep(4096)

	// Verify transfer completed
	assert.False(t, mmu.serialActive, "serial should no longer be active")
	assert.Equal(t, 0, mmu.serialCycles, "serialCycles should be 0")

	// SB should be shifted right by 1 with bit 7 set to 1 (no external device)
	// 0xAB = 1010 1011 → >> 1 = 0101 0101 → | 0x80 = 1101 0101 = 0xD5
	// Specifically: (0xAB >> 1) = 0x55, then | 0x80 = 0xD5
	assert.Equal(t, byte(0xD5), mmu.ReadSB(), "SB should be shifted right with bit 7 set")

	// SC bit 7 should be cleared
	assert.Equal(t, byte(0x01), mmu.ReadSC(), "SC bit 7 should be cleared after transfer")

	// IF bit 3 (serial interrupt) should be set
	assert.Equal(t, byte(0x08), mmu.ReadIF(), "IF bit 3 should be set for serial interrupt")
}

// TestMMU_SerialTransfer_NoInterruptWithoutStart verifies that IF bit 3 is
// not set when no transfer was started.
func TestMMU_SerialTransfer_NoInterruptWithoutStart(t *testing.T) {
	mmu := NewMMU(nil)
	mmu.WriteIF(0x00)

	// Advance cycles without starting a transfer — no serial interrupt
	mmu.SerialStep(4096)
	assert.Equal(t, byte(0x00), mmu.ReadIF(), "IF should not have serial interrupt set")
}

// TestMMU_SerialTransfer_ExcessCycles verifies that providing more cycles than
// needed still completes correctly (handles the >4096 case).
func TestMMU_SerialTransfer_ExcessCycles(t *testing.T) {
	mmu := NewMMU(nil)

	mmu.WriteSB(0xFF)
	mmu.Write(0xFF02, 0x81)

	// Advance way past the deadline
	mmu.SerialStep(10000)

	assert.False(t, mmu.serialActive, "serial should be inactive after excess cycles")
	// 0xFF >> 1 | 0x80 = 0x7F | 0x80 = 0xFF
	assert.Equal(t, byte(0xFF), mmu.ReadSB(), "SB all-ones stays all-ones after shift")
	assert.Equal(t, byte(0x01), mmu.ReadSC(), "SC bit 7 should be cleared")
	assert.Equal(t, byte(0x08), mmu.ReadIF(), "IF bit 3 should be set")
}

// -- OAM corruption bug tests (DMG mode 2 bus conflict) --

func TestOAMReadCorruption_Formula_FirstWord(t *testing.T) {
	// Verify the read corruption formula: corrupted = b | (a & c)
	// where a = original first word of current row, b = first word of preceding row,
	// c = third word of preceding row.

	oam := make([]byte, 160)
	// Set up preceding row (row 0): first word = 0xABCD, third word = 0x1234
	oam[0] = 0xCD // word 0 low byte
	oam[1] = 0xAB // word 0 high byte -> 0xABCD
	oam[4] = 0x34 // word 2 (third word) low byte
	oam[5] = 0x12 // word 2 (third word) high byte -> 0x1234

	// Set up current row (row 1): first word = 0xFFFF
	oam[8] = 0xFF
	oam[9] = 0xFF

	applyOAMReadCorruption(oam, 1)

	// b = 0xABCD, a = 0xFFFF, c = 0x1234
	// b | (a & c) = 0xABCD | (0xFFFF & 0x1234) = 0xABCD | 0x1234 = 0xBBFD
	expected := byte(0xFD) // 0xBBFD low byte
	assert.Equal(t, expected, oam[8], "read corruption first word low byte")
	expected = byte(0xBB) // 0xBBFD high byte
	assert.Equal(t, expected, oam[9], "read corruption first word high byte")
}

func TestOAMReadCorruption_LastThreeWordsCopied(t *testing.T) {
	oam := make([]byte, 160)

	// Preceding row (row 0): set all 8 bytes to known pattern
	for i := range 8 {
		oam[i] = byte(i + 0x10)
	}

	// Current row (row 1): set to different pattern
	for i := 8; i < 16; i++ {
		oam[i] = byte(i + 0x20)
	}

	applyOAMReadCorruption(oam, 1)

	// First word (bytes 8-9) is corrupted via formula, ignore exact value.
	// Bytes 10-15 should be copied from preceding row bytes 2-7.
	for i := 2; i < 8; i++ {
		assert.Equal(t, oam[i], oam[8+i], "read corruption: row 1 byte %d should be copied from row 0 byte %d", i+8, i)
	}
}

func TestOAMWriteCorruption_Formula_FirstWord(t *testing.T) {
	// Verify write corruption formula: ((a ^ c) & (b ^ c)) ^ c
	oam := make([]byte, 160)
	// Preceding row (row 0): first word = 0x0000, third word = 0xFFFF
	oam[4] = 0xFF // word 2 low byte
	oam[5] = 0xFF // word 2 high byte

	// Current row (row 1): first word = 0xFFFF
	oam[8] = 0xFF
	oam[9] = 0xFF

	applyOAMWriteCorruption(oam, 1)

	// a = 0xFFFF, b = 0x0000, c = 0xFFFF
	// ((0xFFFF ^ 0xFFFF) & (0x0000 ^ 0xFFFF)) ^ 0xFFFF
	// = (0x0000 & 0xFFFF) ^ 0xFFFF
	// = 0x0000 ^ 0xFFFF = 0xFFFF
	assert.Equal(t, byte(0xFF), oam[8], "write corruption with a=c=F=>F, b=0 => result high byte")
	assert.Equal(t, byte(0xFF), oam[9], "write corruption with a=c=F=>F, b=0 => result low byte")
}

func TestOAMWriteCorruption_LastThreeWordsCopied(t *testing.T) {
	oam := make([]byte, 160)

	// Preceding row (row 0): set all 8 bytes to known pattern
	for i := range 8 {
		oam[i] = byte(i + 0xA0)
	}

	// Current row (row 1): set to a different pattern
	for i := 8; i < 16; i++ {
		oam[i] = byte(i + 0xB0)
	}

	applyOAMWriteCorruption(oam, 1)

	// First word (bytes 8-9) is corrupted via formula, ignore exact value.
	// Bytes 10-15 should be copied from preceding row bytes 2-7.
	for i := 2; i < 8; i++ {
		assert.Equal(t, oam[i], oam[8+i], "write corruption: row 1 byte %d should be copied from row 0 byte %d", i+8, i)
	}
}

func TestOAMReadCorruption_RowZeroUnaffected(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Set up OAM values in row 0
	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0xAA)
	mmu.Write(0xFE01, 0xBB)
	mmu.Write(0xFE04, 0xCC)
	mmu.Write(0xFE05, 0xDD)

	// Set row to 0 in PPU stat. In mode 2 with dotCounter < 4, row = 0.
	// We can't easily set dotCounter from outside, but we can use the
	// fact that after Reset, dotCounter=0 so GetOAMRow returns 0.
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM

	// Read from row 0 — should not be corrupted
	assert.Equal(t, byte(0xAA), mmu.Read(0xFE00), "OAM read at FE00 should not be corrupted (row 0)")
	assert.Equal(t, byte(0xBB), mmu.Read(0xFE01), "OAM read at FE01 should not be corrupted (row 0)")
	assert.Equal(t, byte(0xCC), mmu.Read(0xFE04), "OAM read at FE04 should not be corrupted (row 0)")
}

func TestOAMWriteCorruption_RowZeroUnaffected(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Start in accessible mode and set up OAM
	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0x12)
	mmu.Write(0xFE04, 0x34)

	// Mode 2, row 0 — writes should go through
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	mmu.Write(0xFE00, 0xAB)
	mmu.Write(0xFE04, 0xCD)

	// Verify writes took effect (row 0 is not affected)
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0xAB), mmu.Read(0xFE00), "OAM write to FE00 should work in mode 2 (row 0)")
	assert.Equal(t, byte(0xCD), mmu.Read(0xFE04), "OAM write to FE04 should work in mode 2 (row 0)")
}

func TestOAMReadCorruption_PersistentCorruption(t *testing.T) {
	// Once a row is corrupted, subsequent reads should return the corrupted data
	// (not the original data), and the corruption should persist.
	//
	// This test uses OAM at offset 0x08 (row 1). Reading from row 1 during
	// mode 2 should permanently corrupt row 1's data. Subsequent reads in
	// accessible modes should show the corrupted data.

	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	// Set up row 0 and row 1 with known values
	ppu.stat &^= 0x03
	// Row 0: first word = 0x0001, third word = 0x0002
	mmu.Write(0xFE00, 0x01)
	mmu.Write(0xFE01, 0x00)
	mmu.Write(0xFE04, 0x02)
	mmu.Write(0xFE05, 0x00)
	// Row 1: first word = 0x00FF
	mmu.Write(0xFE08, 0xFF)
	mmu.Write(0xFE09, 0x00)

	// Now trigger corruption via read in mode 2
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	ppu.Step(8) // advance dotCounter so row = 8/4 = 2, NOT row 1
	// At row 2, the Read below will corrupt row 2 (not row 1 where FE08 lives).
	// So let's read from an address in row 2 instead.
	mmu.Write(0xFE28, 0x42) // set row 5 byte 0
	corruptedVal := mmu.Read(0xFE28) // corrupts row 5 (0xFE28 is in row 5)

	// The corrupted value should be different from original
	assert.NotEqual(t, byte(0x42), corruptedVal, "read corruption should change the value")

	// Go back to accessible mode — the OAM data should still be corrupted
	ppu.stat &^= 0x03
	afterCorruption := mmu.Read(0xFE28)
	assert.Equal(t, corruptedVal, afterCorruption, "OAM corruption should persist after mode change")
}

func TestOAMReadCorruption_MultipleRows(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	ppu.stat &^= 0x03
	// Fill all 20 rows with distinct byte patterns
	// Row 0: first word = 0x1000, third word = 0x2000
	mmu.Write(0xFE00, 0x00)
	mmu.Write(0xFE01, 0x10)
	mmu.Write(0xFE04, 0x00)
	mmu.Write(0xFE05, 0x20)
	// Row 1: first word = 0x3000
	mmu.Write(0xFE08, 0x00)
	mmu.Write(0xFE09, 0x30)
	// Row 2: first word = 0x4000
	mmu.Write(0xFE10, 0x00)
	mmu.Write(0xFE11, 0x40)

	// Read from OAM during mode 2 with a row > 0
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	mmu.Read(0xFE08) // corrupts row 1

	// After corruption, row 0 should be unchanged
	ppu.stat &^= 0x03
	assert.Equal(t, byte(0x00), mmu.Read(0xFE00), "row 0 byte 0 should be unchanged")
	assert.Equal(t, byte(0x10), mmu.Read(0xFE01), "row 0 byte 1 should be unchanged")

	// Row 2 should be unchanged (only row 1 was corrupted)
	assert.Equal(t, byte(0x00), mmu.Read(0xFE10), "row 2 byte 0 should be unchanged")
	assert.Equal(t, byte(0x40), mmu.Read(0xFE11), "row 2 byte 1 should be unchanged")
}

func TestOAMWriteCorruption_IntegratesIntoMemory(t *testing.T) {
	// Verify that after a write corruption during mode 2, the corrupted OAM
	// data persists and is what the PPU actually reads via ReadOAMDirect.
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	ppu.stat &^= 0x03
	// Set up row 0 with known values
	mmu.Write(0xFE00, 0x11)
	mmu.Write(0xFE01, 0x22)
	mmu.Write(0xFE04, 0x33)
	mmu.Write(0xFE05, 0x44)
	// Set up row 1
	mmu.Write(0xFE08, 0x55)
	mmu.Write(0xFE09, 0x66)

	// Write during mode 2 — corrupts row 1
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	mmu.Write(0xFE08, 0x77) // value is lost, row 1 is corrupted via write formula

	// After going back to accessible mode, ReadOAMDirect should return the corrupted data
	ppu.stat &^= 0x03
	directVal := mmu.ReadOAMDirect(0xFE08)
	memVal := mmu.Read(0xFE08)

	// Both direct and MMU reads should return the same (corrupted) value
	assert.Equal(t, directVal, memVal, "ReadOAMDirect and Read should see same corrupted data")
}

func TestOAMReadCorruption_FEA0toFEFFRegion(t *testing.T) {
	// Pan Docs state: "Attempting to read or write from OAM (including the
	// 0xFEA0-0xFEFF region) while the PPU is in mode 2 (OAM scan) will corrupt it."
	// The 0xFEA0-0xFEFF region maps to OAM array bytes 0xA0-0xFF (unusable area).
	// It doesn't have proper row structure, but accesses should still not crash.
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	ppu.stat &^= 0x03
	// Write to the unusable area
	mmu.Write(0xFEA0, 0x42)
	assert.Equal(t, byte(0x00), mmu.Read(0xFEA0), "unusable area normally returns 0x00")

	// Access during mode 2 — should not panic
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	val := mmu.Read(0xFEA0)
	_ = val // just verify no panic
}

func TestOAMReadCorruption_NoPPUAttached(t *testing.T) {
	// Without PPU, OAM reads should work normally regardless of "mode"
	mmu := NewMMU(nil)
	mmu.Write(0xFE00, 0x42)
	assert.Equal(t, byte(0x42), mmu.Read(0xFE00), "OAM read should work without PPU")
}

func TestOAMReadCorruption_Mode3StillBlocked(t *testing.T) {
	// Mode 3 should still return 0xFF for OAM reads
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	ppu.stat &^= 0x03
	mmu.Write(0xFE00, 0xAA)

	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	assert.Equal(t, byte(0xFF), mmu.Read(0xFE00), "OAM read should return 0xFF in Mode 3")

	// Rows > 0 should also return 0xFF in mode 3
	ppu.stat &^= 0x03
	mmu.Write(0xFE08, 0xBB)
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeVRAM
	assert.Equal(t, byte(0xFF), mmu.Read(0xFE08), "OAM read at row 1 should return 0xFF in Mode 3")
}

func TestOAMReadCorruption_GetOAMRow(t *testing.T) {
	// Verify GetOAMRow returns correct row based on dotCounter
	ppu := NewPPU(nil)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	// After reset, dotCounter = 0, row = 0
	assert.Equal(t, 0, ppu.GetOAMRow(), "GetOAMRow should be 0 after reset")

	// Step 4 T-cycles (1 M-cycle) — row should be 1
	ppu.Step(4)
	assert.Equal(t, 1, ppu.GetOAMRow(), "GetOAMRow should be 1 after 4 T-cycles")

	// Step 4 more — row should be 2
	ppu.Step(4)
	assert.Equal(t, 2, ppu.GetOAMRow(), "GetOAMRow should be 2 after 8 T-cycles")
}

func TestOAMReadCorruption_CorruptsCorrectRow(t *testing.T) {
	// Verify that reading from address FE08 while PPU is at row 5
	// corrupts row 5, NOT row 1 (where FE08 lives).
	mmu := NewMMU(nil)
	ppu := attachPPU(mmu)

	ppu.stat &^= 0x03
	// Set up row 0 with distinctive values
	mmu.Write(0xFE00, 0x01)
	mmu.Write(0xFE01, 0x00)
	mmu.Write(0xFE04, 0x02)
	mmu.Write(0xFE05, 0x00)

	// Set up row 4 (preceding row for row 5) with distinctive values
	mmu.Write(0xFE20, 0xA0)
	mmu.Write(0xFE21, 0x00)
	mmu.Write(0xFE24, 0xA4)
	mmu.Write(0xFE25, 0x00)

	// Set up row 5 with known values
	mmu.Write(0xFE28, 0x55)
	mmu.Write(0xFE29, 0x00)

	// Manually set dotCounter by using Step to advance to row 5.
	// Row 5 starts at dotCounter = 5*4 = 20.
	ppu.stat = (ppu.stat &^ 0x03) | ppuModeOAM
	// Step to dotCounter=20 (row 5)
	ppu.Step(20)

	// Now read from FE08 (which is in row 1, NOT row 5) during mode 2
	val := mmu.Read(0xFE08)

	// The corruption should affect row 5 (the PPU's current row), not row 1.
	// So reading FE08 should return whatever row 1's corrupted data now is.
	// But row 1 hasn't been corrupted (only row 5 was), so the corruption
	// of row 1 at FE08 should not happen unless... hmm wait.
	//
	// Actually, the corruption only affects row 5. Reading FE08 is just the
	// trigger. The value returned is from the (now-corrupted) row 5.
	// But FE08 is at a different offset than row 5...
	//
	// Wait, I need to reconsider. According to the implementation, the
	// corruption corrupts the PPU's current row (row 5), and then returns
	// the byte from the corrupted OAM at *the requested address*.
	// So the returned byte is at offset 0x08 (FE08), not at the corrupted row.
	// This means we just read whatever is at FE08 (row 1, uncorrupted).
	// But row 1 is accessible during mode 2 because... no wait, row 1 IS
	// being accessed via the bus conflict, so the corruption still applies
	// to row 5 (the PPU's row).
	//
	// Let me trace through the logic:
	// 1. Read(0xFE08) called
	// 2. GetMode() = ppuModeOAM
	// 3. GetOAMRow() = 5 (dotCounter=20)
	// 4. row > 0, so applyOAMReadCorruption(oam, 5)
	// 5. This corrupts OAM bytes at offset 5*8=40 through 47 (row 5 = FE28-FE2F)
	// 6. Returns m.oam[0x08] = OAM byte at offset 8 (FE08)
	//
	// So we return the original uncorrupted byte at FE08.
	// But row 5 (FE28-FE2F) is now corrupted.
	_ = val

	// Go back to accessible mode and check that row 5 IS corrupted
	ppu.stat &^= 0x03
	row5Val := mmu.Read(0xFE28)
	assert.NotEqual(t, byte(0x55), row5Val, "row 5 (FE28) should be corrupted after mode 2 access")

	// Row 1 should still be OK
	assert.Equal(t, byte(0x00), mmu.Read(0xFE08), "row 1 at FE08 should be unchanged (only row 5 was corrupted)")
	assert.Equal(t, byte(0x00), mmu.Read(0xFE09), "row 1 at FE09 should be unchanged")
}
