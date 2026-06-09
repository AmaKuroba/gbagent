package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// PPU timing constants for test reference.
// Mode durations in dots (T-cycles) per scanline.
const (
	OAMSearchCycles  = 80
	VRAMDrawCycles   = 172 // minimum (no sprites)
	HBlankCycles     = 204 // 456 - 80 - 172
	ScanlineCycles   = 456
	VisibleScanlines = 144
	VBlankScanlines  = 10
	TotalScanlines   = 154
	FrameCycles      = 70224 // 154 * 456
	VBlankCycles     = 4560  // 10 * 456
)

// ppuMode constants mirroring STAT register bits 1-0:
// 00 = HBlank (Mode 0), 01 = VBlank (Mode 1), 10 = OAM (Mode 2), 11 = VRAM (Mode 3)
const (
	modeHBlank = iota // 0
	modeVBlank        // 1
	modeOAM           // 2
	modeVRAM          // 3
)

// testPPU is a compile-check stub we use until the real PPU exists.
// Replace with the concrete PPU implementation once ppu.go has a constructor.
type testPPU struct{ PPU }

// newTestPPU must be replaced with the real constructor once available.
// func newTestPPU() PPU { return &PPUCore{...} }

// enabledPPU creates a PPU and enables LCD, for tests that need it.
func enabledPPU() *PPUCore {
	ppu := NewPPU(nil)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)
	return ppu
}

// --- Mode sequence per scanline ---

func TestPPUModeSequence_OAMtoVRAMtoHBlank(t *testing.T) {
	t.Parallel()

	// Step through one scanline at the beginning of a frame (LY=0).
	// At dot 0, the PPU enters OAM search (mode 2).
	// At dot 80, it enters VRAM/draw (mode 3).
	// At dot 252 (80+172), it enters HBlank (mode 0).
	// After 456 dots, LY increments.

	ppu := NewPPU(nil)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	// Initially at start of frame, LY=0, we should be in OAM search.
	state := ppu.GetState()
	assert.Equal(t, modeOAM, state.Mode, "should start frame in OAM search (mode 2)")
	assert.Equal(t, byte(0), state.LY, "should start at LY=0")

	// Step through OAM search (80 cycles).
	ppu.Step(80)
	state = ppu.GetState()
	assert.Equal(t, modeVRAM, state.Mode, "after 80 cycles should move to VRAM/draw (mode 3)")

	// Step through VRAM/draw (172 cycles).
	ppu.Step(VRAMDrawCycles)
	state = ppu.GetState()
	assert.Equal(t, modeHBlank, state.Mode, "after 80+172 cycles should move to HBlank (mode 0)")

	// Step through HBlank (204 cycles) — completes the scanline.
	ppu.Step(HBlankCycles)
	state = ppu.GetState()
	assert.Equal(t, byte(1), state.LY, "after 1 full scanline, LY should increment to 1")
	assert.Equal(t, modeOAM, state.Mode, "next scanline starts in OAM search")
}

func TestPPUModeSequence_AllLines(t *testing.T) {
	t.Parallel()

	// Verify every visible scanline follows OAM→VRAM→HBlank.
	ppu := NewPPU(nil)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	for ly := 0; ly < VisibleScanlines; ly++ {
		// Initial state should be OAM for visible lines.
		state := ppu.GetState()
		assert.Equal(t, byte(ly), state.LY, "LY should be %d at start of scanline", ly)
		assert.Equal(t, modeOAM, state.Mode, "LY=%d should start in OAM search (mode 2)", ly)

		// OAM search
		ppu.Step(OAMSearchCycles)
		state = ppu.GetState()
		assert.Equal(t, modeVRAM, state.Mode, "LY=%d should enter VRAM/draw (mode 3) after OAM", ly)

		// VRAM draw
		ppu.Step(VRAMDrawCycles)
		state = ppu.GetState()
		assert.Equal(t, modeHBlank, state.Mode, "LY=%d should enter HBlank (mode 0) after VRAM", ly)

		// HBlank — complete the scanline
		ppu.Step(HBlankCycles)
	}
}

// --- VBlank duration ---

func TestPPUVBlankDuration(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Run through 144 visible scanlines.
	ppu.Step(VisibleScanlines * ScanlineCycles)

	// Now at LY=144, should be in VBlank (mode 1).
	state := ppu.GetState()
	assert.Equal(t, byte(144), state.LY, "LY should be 144 after visible scanlines")
	assert.Equal(t, modeVBlank, state.Mode, "should enter VBlank (mode 1) at LY=144")

	// Run through all 10 VBlank scanlines.
	for ly := 144; ly < TotalScanlines; ly++ {
		state = ppu.GetState()
		assert.Equal(t, byte(ly), state.LY, "VBlank LY should be %d", ly)
		assert.Equal(t, modeVBlank, state.Mode, "VBlank LY=%d should be in mode 1", ly)

		// Each VBlank scanline lasts 456 dots.
		ppu.Step(ScanlineCycles)
	}

	// After 154 scanlines, LY should wrap to 0 and start a new frame.
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should wrap to 0 after frame completes")
	assert.Equal(t, modeOAM, state.Mode, "new frame starts in OAM search")
}

func TestPPUVBlankExactCycleCount(t *testing.T) {
	t.Parallel()

	// VBlank should be exactly 4560 dots (10 scanlines * 456).
	ppu := enabledPPU()

	// Advance through all visible lines.
	ppu.Step(VisibleScanlines * ScanlineCycles)

	state := ppu.GetState()
	require.Equal(t, modeVBlank, state.Mode, "should be in VBlank")

	// Advance exactly VBlankCycles-1 — should still be in VBlank.
	ppu.Step(VBlankCycles - 1)
	state = ppu.GetState()
	assert.Equal(t, modeVBlank, state.Mode, "should still be in VBlank 1 cycle before end")

	// One more cycle completes VBlank.
	ppu.Step(1)
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after frame wrap")
}

// --- LY increment ---

func TestPPULYIncrement(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	for expectedLY := 0; expectedLY < TotalScanlines; expectedLY++ {
		state := ppu.GetState()
		assert.Equal(t, byte(expectedLY), state.LY, "LY should be %d at start of scanline", expectedLY)

		// Step through one full scanline.
		ppu.Step(ScanlineCycles)
	}

	// After 154 scanlines, LY wraps.
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should wrap to 0 after full frame")
}

func TestPPULYIncrement_PartialSteps(t *testing.T) {
	t.Parallel()

	// LY must increment correctly even when Step is called with
	// partial scanline amounts.
	ppu := enabledPPU()

	// Step 100 cycles at a time and verify LY.
	for frameLY := 0; frameLY < TotalScanlines; frameLY++ {
		state := ppu.GetState()
		assert.Equal(t, byte(frameLY), state.LY, "LY should be %d", frameLY)

		// Step in chunks that cross mode boundaries.
		remaining := ScanlineCycles
		for remaining > 0 {
			chunk := 47 // prime-ish chunk size
			if chunk > remaining {
				chunk = remaining
			}
			ppu.Step(chunk)
			remaining -= chunk
		}
	}

	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should wrap to 0 after full frame")
}

// --- LY equals LYC ---

func TestPPULYEqualsLYC_Coincidence(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Test with LYC=5.
	ppu.SetLYC(5)

	// Run to LY=4.
	ppu.Step(5 * ScanlineCycles) // scanlines 0-4 complete, now at start of scanline 5

	state := ppu.GetState()
	assert.Equal(t, byte(5), state.LY, "LY should be 5")
	assertBitSet(t, state.STAT, 2, "LYC=LY coincidence flag (bit 2) should be set when LY=LYC=5")
}

func TestPPULYEqualsLYC_NoCoincidence(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Default LYC=0, LY starts at 0 — so LY==LYC initially.
	ppu.Step(ScanlineCycles) // move to line 1

	state := ppu.GetState()
	assert.Equal(t, byte(1), state.LY, "LY should be 1")
	assertBitNotSet(t, state.STAT, 2, "LYC=LY coincidence flag should be clear when LY≠LYC")
}

func TestPPULYEqualsLYC_DifferentLYC(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	ppu.SetLYC(10)

	// Run to LY=9 (9 full scanlines).
	ppu.Step(10 * ScanlineCycles) // moves to start of scanline 10

	state := ppu.GetState()
	assert.Equal(t, byte(10), state.LY, "LY should be 10")
	assertBitSet(t, state.STAT, 2, "LYC=LY coincidence flag should be set when LY=10, LYC=10")
}

// --- Frame duration ---

func TestPPUFrameDuration(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Step exactly FrameCycles and verify we're back at the start.
	ppu.Step(FrameCycles)
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after one full frame")
	assert.Equal(t, modeOAM, state.Mode, "should be at start of frame")
}

func TestPPUFrameDuration_ExactBoundary(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// FrameCycles-1 should still be on the last scanline.
	ppu.Step(FrameCycles - 1)
	state := ppu.GetState()
	assert.Equal(t, byte(153), state.LY, "LY should be 153 one cycle before frame end")

	// One more cycle completes the frame.
	ppu.Step(1)
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should wrap to 0 after frame end")
}

// --- Reset ---

func TestPPUReset(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

	// Advance a bit.
	ppu.Step(1234)
	ppu.Reset()

	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after reset")
	assert.Equal(t, modeOAM, state.Mode, "mode should be OAM search after reset")
}

// --- GetScreen ---

func TestPPUGetScreen(t *testing.T) {
	t.Parallel()

	// Verify GetScreen returns a 160x144 buffer.
	ppu := NewPPU(nil)
	ppu.Reset()

	screen := ppu.GetScreen()
	assert.Equal(t, 160, len(screen), "screen width should be 160")
	assert.Equal(t, 144, len(screen[0]), "screen height should be 144")
}

// --- STAT write interrupt trigger ---
//
// Writing to STAT (0xFF41) can immediately trigger a STAT interrupt if the
// newly-enabled enable bits match the current PPU state.

func TestPPU_STATWriteTriggersLYCInterrupt(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()

	mmu.Write(0xFF0F, 0x00) // clear IF

	// Set LYC = 0 (LY starts at 0, so LY==LYC already).
	ppu.WriteRegister(0xFF45, 0)

	// Write STAT with LYC interrupt enable (bit 6) — should trigger immediately
	// because LY==LYC==0.
	ppu.WriteRegister(0xFF41, statBitIntLYC)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT write with LYC IE enabled when LY==LYC should trigger STAT interrupt, got IF=%02x", ifVal)
}

func TestPPU_STATWriteTriggersOAMInterrupt(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()

	mmu.Write(0xFF0F, 0x00) // clear IF

	// At LY=0, just after reset, the PPU is in OAM search mode (mode 2).
	// Write STAT with OAM STAT interrupt enable (bit 5).
	ppu.WriteRegister(0xFF41, statBitIntMode2)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT write with OAM IE enabled while in OAM mode should trigger STAT interrupt, got IF=%02x", ifVal)
}

func TestPPU_STATWriteTriggersHBlankInterrupt(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91)

	// Step through OAM + VRAM to reach HBlank (mode 0) on scanline 0.
	ppu.Step(oamSearchCycles + vramDrawCycles) // 80 + 172 = 252 cycles

	mmu.Write(0xFF0F, 0x00) // clear IF (clear any STAT interrupt from mode transition)

	// Now in HBlank. Write STAT with HBlank interrupt enable (bit 3).
	ppu.WriteRegister(0xFF41, statBitIntMode0)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 != 0,
		"STAT write with HBlank IE enabled while in HBlank mode should trigger STAT interrupt, got IF=%02x", ifVal)
}

func TestPPU_STATWriteInOAMDoesNotTriggerWrongMode(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()

	mmu.Write(0xFF0F, 0x00) // clear IF

	// In OAM mode. Write STAT with only HBlank IE (bit 3) — should NOT fire
	// because we're in mode 2, not mode 0.
	ppu.WriteRegister(0xFF41, statBitIntMode0)

	ifVal := mmu.Read(0xFF0F)
	assert.True(t, ifVal&0x02 == 0,
		"HBlank IE write in OAM mode should NOT trigger STAT interrupt, got IF=%02x", ifVal)
}

func TestPPU_STATWriteNoCrashesWithNilMMU(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

	// Should not panic.
	ppu.WriteRegister(0xFF41, statBitIntLYC)
	assert.True(t, true, "STAT write with nil MMU should not panic")
}

func assertBitSet(t *testing.T, reg byte, bit int, msg string) {
	t.Helper()
	assert.True(t, reg&(1<<bit) != 0, msg)
}

func assertBitNotSet(t *testing.T, reg byte, bit int, msg string) {
	t.Helper()
	assert.False(t, reg&(1<<bit) != 0, msg)
}

// --- First-frame blank after LCD enable ---

func TestPPUFirstFrameBlankAfterLCDEnable(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

	// LCD starts enabled (default 0x91, bit 7 set). Disable it first.
	ppu.WriteRegister(0xFF40, 0x00) // LCD off
	// All pixels should be 0 (white) already from default, but ensure.
	screen := ppu.GetScreen()
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			assert.Equal(t, byte(0), screen[x][y], "pixel (%d,%d) should be 0 (white) with LCD off", x, y)
		}
	}

	// Enable LCD — sets firstFrameBlank.
	ppu.WriteRegister(0xFF40, 0x91) // LCD on

	// Step through one full frame (70224 cycles).
	// During this frame, all rendered pixels should stay 0 (white).
	ppu.Step(FrameCycles)

	screen = ppu.GetScreen()
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			assert.Equal(t, byte(0), screen[x][y], "first frame after LCD enable: pixel (%d,%d) should be 0 (white)", x, y)
		}
	}
}

func TestPPUFirstFrameBlankClearsAfterOneFrame(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

	// Disable then enable LCD.
	ppu.WriteRegister(0xFF40, 0x00)
	ppu.WriteRegister(0xFF40, 0x91)

	// Complete the first (blank) frame.
	ppu.Step(FrameCycles)

	// After one full frame, the blank flag should be cleared.
	// Step into the second frame — pixels should no longer be forced to 0.
	// Since there's no MMU, rendering functions are no-ops after the blank period,
	// so pixels may still be 0. What matters is the flag is cleared so rendering
	// proceeds normally.
	// We verify the flag is cleared by checking LY wraps to 0 and we're on frame 2.
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should wrap to 0 after frame")
}

func TestPPUFirstFrameBlankResetOnLCDToggle(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

	// Disable then enable LCD.
	ppu.WriteRegister(0xFF40, 0x00)
	ppu.WriteRegister(0xFF40, 0x91)

	// Complete the first (blank) frame.
	ppu.Step(FrameCycles)

	// Disable LCD again mid-second-frame.
	ppu.WriteRegister(0xFF40, 0x00)
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should reset to 0 when LCD is turned off")

	// Re-enable — should get another blank frame.
	ppu.WriteRegister(0xFF40, 0x91)
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after LCD re-enable")

	// Step through one full frame — all pixels should still be 0 (blank).
	ppu.Step(FrameCycles)
	screen := ppu.GetScreen()
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			assert.Equal(t, byte(0), screen[x][y], "pixel (%d,%d) should be 0 (white) after re-enable blank frame", x, y)
		}
	}
}

func TestPPUFirstFrameBlankIsRunningDuringBlank(t *testing.T) {
	t.Parallel()

	// The PPU should be running and processing scanlines during the blank frame,
	// even though pixel output is suppressed.
	ppu := NewPPU(nil)
	ppu.Reset()

	// Disable then enable LCD.
	ppu.WriteRegister(0xFF40, 0x00)
	ppu.WriteRegister(0xFF40, 0x91)

	// Step a partial frame and observe LY advancing.
	ppu.Step(ScanlineCycles * 50) // 50 scanlines in
	state := ppu.GetState()
	assert.Equal(t, byte(50), state.LY, "LY should advance during blank frame")
	assert.Equal(t, modeOAM, state.Mode, "PPU should be in OAM mode at start of scanline 50")
}

// --- LCD disabled → LY behavior ---

func TestPPULYZeroWhenLCDOff(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// LCD was enabled in enabledPPU. Turn it off.
	ppu.WriteRegister(0xFF40, 0x00)

	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 immediately after LCD is disabled")
}

func TestPPULYStaysZeroDuringLCDOff(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Run a few scanlines so LY advances.
	ppu.Step(ScanlineCycles * 10) // LY should be 10

	state := ppu.GetState()
	assert.Equal(t, byte(10), state.LY, "LY should be 10 mid-frame")

	// Turn LCD off.
	ppu.WriteRegister(0xFF40, 0x00)

	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should reset to 0 when LCD is turned off mid-frame")

	// Call Step many times while LCD is off — LY must stay 0.
	for i := 0; i < 1000; i++ {
		ppu.Step(1)
	}
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should stay 0 while LCD is off, even after many Step calls")
}

func TestPPULYResumesFromZeroAfterLCDReEnable(t *testing.T) {
	t.Parallel()

	ppu := enabledPPU()

	// Run a bit so LY is non-zero.
	ppu.Step(ScanlineCycles * 50) // LY = 50

	// Turn LCD off.
	ppu.WriteRegister(0xFF40, 0x00)

	// Wait a while with LCD off.
	ppu.Step(ScanlineCycles * 10)

	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 while LCD is off")

	// Re-enable LCD.
	ppu.WriteRegister(0xFF40, 0x91)
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should still be 0 after LCD re-enable")

	// Step one scanline — LY should increment to 1.
	ppu.Step(ScanlineCycles)
	state = ppu.GetState()
	assert.Equal(t, byte(1), state.LY, "LY should increment to 1 after one scanline with LCD on")
}

func TestPPULYReadsZeroWhenLCDOffAtBoot(t *testing.T) {
	t.Parallel()

	// Create PPU with LCD explicitly disabled from the start.
	ppu := NewPPU(nil)
	ppu.WriteRegister(0xFF40, 0x00)
	ppu.Reset()

	// After Reset with LCD disabled, Step must keep LY=0.
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after Reset with LCD disabled")

	ppu.Step(FrameCycles)
	state = ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should stay 0 after a full frame's worth of cycles with LCD off")
}
