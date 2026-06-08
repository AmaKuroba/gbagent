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

	ppu := NewPPU(nil)
	ppu.Reset()

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
	ppu := NewPPU(nil)
	ppu.Reset()

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

	ppu := NewPPU(nil)
	ppu.Reset()

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
	ppu := NewPPU(nil)
	ppu.Reset()

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

	ppu := NewPPU(nil)
	ppu.Reset()

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

	ppu := NewPPU(nil)
	ppu.Reset()

	// Default LYC=0, LY starts at 0 — so LY==LYC initially.
	ppu.Step(ScanlineCycles) // move to line 1

	state := ppu.GetState()
	assert.Equal(t, byte(1), state.LY, "LY should be 1")
	assertBitNotSet(t, state.STAT, 2, "LYC=LY coincidence flag should be clear when LY≠LYC")
}

func TestPPULYEqualsLYC_DifferentLYC(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

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

	ppu := NewPPU(nil)
	ppu.Reset()

	// Step exactly FrameCycles and verify we're back at the start.
	ppu.Step(FrameCycles)
	state := ppu.GetState()
	assert.Equal(t, byte(0), state.LY, "LY should be 0 after one full frame")
	assert.Equal(t, modeOAM, state.Mode, "should be at start of frame")
}

func TestPPUFrameDuration_ExactBoundary(t *testing.T) {
	t.Parallel()

	ppu := NewPPU(nil)
	ppu.Reset()

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

// --- Helper assertions ---

func assertBitSet(t *testing.T, reg byte, bit int, msg string) {
	t.Helper()
	assert.True(t, reg&(1<<bit) != 0, msg)
}

func assertBitNotSet(t *testing.T, reg byte, bit int, msg string) {
	t.Helper()
	assert.False(t, reg&(1<<bit) != 0, msg)
}
