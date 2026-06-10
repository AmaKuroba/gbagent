package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestPPUVBlankSetsIF checks that the PPU triggers VBlank interrupt (IF bit 0)
// when LY becomes 144.
func TestPPUVBlankSetsIF(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	// Run through all visible scanlines (144 * 456 = 65664 cycles).
	// After this, LY should be 144.
	ppu.Step(visibleScanlines * scanlineCycles)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x01 != 0,
		"VBlank interrupt (IF bit 0) should be set after entering VBlank (LY=144), got IF=%02x", ifVal)
}

// TestPPUVBlankNoFalseTrigger checks VBlank is NOT set mid-frame or at frame wrap.
func TestPPUVBlankNoFalseTrigger(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	// Clear any spurious IF bits
	mmu.Write(0xFF0F, 0x00)

	// Run partial frame, check VBlank not set
	ppu.Step(50 * scanlineCycles) // LY = 50
	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x01 == 0,
		"VBlank should NOT be set mid-frame at LY=50, got IF=%02x", ifVal)

	// Run through VBlank and frame wrap
	ppu.Step(totalScanlines * scanlineCycles) // completes frame, starts new one

	// VBlank should be set at the transition from LY=143 to LY=144
	// After wrap, LY=0, but IF bit 0 should have been set during VBlank
	mmu.Write(0xFF0F, 0x00) // clear it for the next frame

	// Run another full frame
	ppu.Step(visibleScanlines * scanlineCycles)
	ifVal = mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x01 != 0,
		"VBlank should be set every frame when LY=144, got IF=%02x", ifVal)
}

// TestPPUSTATInterrupt_LYCCoincidence checks that LYC=LY rising edge
// triggers a STAT interrupt if STAT bit 6 is set.
func TestPPUSTATInterrupt_LYCCoincidence(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	mmu.Write(0xFF0F, 0x00) // clear IF

	// Set LYC=3, with STAT bit 6 (LYC=LY STAT IE) enabled
	ppu.WriteRegister(0xFF41, statBitIntLYC) // enable LYC STAT interrupt
	ppu.WriteRegister(0xFF45, 3)             // LYC = 3

	// Run scanlines 0, 1, 2 — LYC not hit yet
	ppu.Step(3 * scanlineCycles) // now at start of scanline 3

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT interrupt (IF bit 1) should fire when LYC=LY=3, got IF=%02x", ifVal)
}

// TestPPUSTATInterrupt_OAM checks that entering OAM search triggers STAT
// interrupt if STAT bit 5 is set.
func TestPPUSTATInterrupt_OAM(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	mmu.Write(0xFF0F, 0x00) // clear IF

	// Enable OAM STAT interrupt (bit 5)
	ppu.WriteRegister(0xFF41, statBitIntMode2)

	// Step one full scanline — OAM entry happens at dot 0 of each scanline
	ppu.Step(scanlineCycles)

	// STAT interrupt should have fired at OAM entry of scanline 1
	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT interrupt should fire on OAM entry when bit 5 set, got IF=%02x", ifVal)
}

// TestPPUSTATInterrupt_HBlank checks that entering HBlank triggers STAT
// interrupt if STAT bit 3 is set.
func TestPPUSTATInterrupt_HBlank(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	mmu.Write(0xFF0F, 0x00) // clear IF

	// Enable HBlank STAT interrupt (bit 3)
	ppu.WriteRegister(0xFF41, statBitIntMode0)

	// Step through OAM + VRAM (80 + 172 = 252 cycles) to reach HBlank
	ppu.Step(oamSearchCycles + vramDrawCycles)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT interrupt should fire on HBlank entry when bit 3 set, got IF=%02x", ifVal)
}

// TestPPUFullFrameInterrupts runs a full frame and verifies VBlank fires exactly once.
func TestPPUFullFrameInterrupts(t *testing.T) {
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	mmu.Write(0xFF0F, 0x00) // clear IF

	// Run one full frame
	vblankCount := 0
	statCount := 0

	for i := 0; i < frameCycles; i += 17 {
		ppu.Step(17)
		ifVal := mmu.Read(0xFF0F)
		if ifVal&0x01 != 0 {
			vblankCount++
			mmu.Write(0xFF0F, ifVal & ^byte(0x01)) // acknowledge VBlank
		}
		if ifVal&0x02 != 0 {
			statCount++
			mmu.Write(0xFF0F, ifVal & ^byte(0x02)) // acknowledge STAT
		}
	}

	t.Logf("VBlank interrupts: %d, STAT interrupts: %d", vblankCount, statCount)
	assert.Equal(t, 1, vblankCount, "VBlank should fire exactly once per frame")

	// With no STAT enable bits set (statBitIntMode0/1/2/LYC all 0),
	// no STAT interrupt should fire.
	assert.Equal(t, 0, statCount,
		"STAT with no enables should not fire any interrupts")
}
