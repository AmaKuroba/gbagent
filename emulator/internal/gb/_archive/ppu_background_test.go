package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupPPUWithVRAM creates a PPU + MemoryBus with a specific 8x8 tile
// written to VRAM at the given tile index, and the tile map entry set.
//
// The tile is an 8x8 pixel block where each pixel's 2-bit value is
// determined by a repeating pattern across the tile.
func setupPPUWithVRAM(t *testing.T) (*PPUCore, *MemoryBus) {
	t.Helper()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()

	// Write a checkerboard tile to VRAM tile index 0 at 0x8000.
	// Tile data format: 16 bytes per tile, 2 bytes per row.
	// Byte 0 = low bits, Byte 1 = high bits.
	// Pixel value = (high_bit << 1) | low_bit
	//
	// Row 0: pixels alternation 0,1,0,1,0,1,0,1
	//   low byte:  0b01010101 = 0x55
	//   high byte: 0b00000000 = 0x00
	//
	// This means even columns are shade 0 (white), odd columns are shade 1 (light).
	for row := 0; row < 8; row++ {
		tileAddr := uint16(0x8000 + row*2) // 16 bytes per tile, 2 per row
		if row%2 == 0 {
			// Even row: pattern 0,1,0,1,0,1,0,1
			mmu.Write(tileAddr, 0x55)   // low byte
			mmu.Write(tileAddr+1, 0x00) // high byte
		} else {
			// Odd row: pattern 1,0,1,0,1,0,1,0
			mmu.Write(tileAddr, 0xAA)   // low byte
			mmu.Write(tileAddr+1, 0x00) // high byte
		}
	}

	// Set tile map at 0x9800 (default BG tile map area) entry 0 to tile index 0.
	// Map is 32x32 entries, each 1 byte = tile index.
	mmu.Write(0x9800, 0x00) // tile index 0 at map position (0,0)

	return ppu, mmu
}

// TestBackgroundTileDataRead verifies the PPU can read tile data from VRAM
// and decode 8x8 tiles correctly.
func TestBackgroundTileDataRead(t *testing.T) {
	ppu, mmu := setupPPUWithVRAM(t)

	// Read tile 0, row 0 from VRAM.
	// Each tile is 16 bytes at 0x8000 + tile_index * 16.
	tileAddr := uint16(0x8000) // tile index 0, row 0
	lo := mmu.Read(tileAddr)
	hi := mmu.Read(tileAddr + 1)

	// For our test pattern: even rows have 0x55 low byte (0,1,0,1,0,1,0,1 pattern)
	assert.Equal(t, byte(0x55), lo, "tile 0 row 0 low byte should be 0x55")
	assert.Equal(t, byte(0x00), hi, "tile 0 row 0 high byte should be 0x00")

	// Verify pixel decoding: for pixel column 0: lo bit 0 = 1, hi bit 0 = 0 → pixel=1
	pixel0 := (hi&0x01)<<1 | (lo & 0x01)
	assert.Equal(t, byte(1), pixel0, "pixel 0 should be 1 (light)")

	// Pixel column 1: lo bit 1 = 0, hi bit 1 = 0 → pixel=0
	pixel1 := (hi&0x02)<<0 | (lo&0x02)>>1
	_ = pixel1

	// Step through a full frame to let the PPU render.
	ppu.Step(FrameCycles)

	// Screen should now have something in it.
	screen := ppu.GetScreen()
	require.NotNil(t, screen, "screen should not be nil after rendering a frame")

	// At LY=0, SCY=0, SCX=0, tile 0 should appear at the top-left of the screen.
	// Pixel (0,0) corresponds to tile map entry (0,0), tile 0, pixel (0,0).
	// Our tile: row 0 even → lo=0x55, hi=0x00 → pixel(0,0) = 1 (shade 1 / light)
	assert.Equal(t, byte(1), screen[0][0], "top-left pixel should be shade 1 (tile row 0, pixel pattern)")
}

// TestBackgroundRender verifies a known tile renders correct pixels.
func TestBackgroundRender(t *testing.T) {
	ppu, mmu := setupPPUWithVRAM(t)

	// Write tile 0 to all 32x32 tile map positions so we see a full grid.
	for i := uint16(0); i < 32*32; i++ {
		mmu.Write(0x9800+i, 0x00)
	}

	// Step through one frame.
	ppu.Step(FrameCycles)

	screen := ppu.GetScreen()

	// The tile at map (0,0) appears at screen position (0,0) when SCY=0, SCX=0.
	// Our checkerboard tile: row 0 has pattern 0,1,0,1,0,1,0,1.
	// So screen[0][0] = shade 1 (the 0th column of the tile, which is pixel value 1).
	// screen[1][0] = shade 0 (column 1 of tile row 0 = pixel value 0).
	assert.Equal(t, byte(1), screen[0][0], "pixel (0,0) should be shade 1")
	assert.Equal(t, byte(0), screen[1][0], "pixel (1,0) should be shade 0")

	// Tile at map (1,0) starts at screen column 8.
	// That's tile index 0 again, so same pattern.
	// screen[8][0] should be same as screen[0][0] (tile repeats).
	assert.Equal(t, screen[0][0], screen[8][0], "pixel (8,0) should match pixel (0,0) since tile repeats")

	// Row 1 of the tile has pattern 1,0,1,0,1,0,1,0 (odd row).
	// So screen[0][1] = shade 0 (column 0 = lo bit 0 of the second byte pair).
	// For row 1: lo=0xAA, hi=0x00 → pixel = 0 (even though the lo bit pattern is 1,0,1,0...)
	// Wait: 0xAA = 0b10101010, bit 0 = 0, so pixel = 0 (white).
	assert.Equal(t, byte(0), screen[0][1], "pixel (0,1) should be shade 0 (tile row 1, col 0 = lo bit 0 = 0)")
}

// TestBackgroundRenderKnownTileMap verifies that the tile map addressing works.
// LCDC.3 = 0 → tile map at 0x9800, LCDC.3 = 1 → tile map at 0x9C00.
func TestBackgroundRenderKnownTileMap(t *testing.T) {
	mmu := NewMMU(nil)

	// Write a solid tile to VRAM: all pixels = shade 3 (black).
	// Low byte = 0xFF, High byte = 0xFF → pixel = (1<<1)|1 = 3
	for addr := uint16(0x8000); addr < 0x8010; addr++ {
		mmu.Write(addr, 0xFF) // all 16 bytes of tile 0
	}

	// Set tile map at 0x9C00 (LCDC.3 = 1) entry 0 to tile index 0.
	mmu.Write(0x9C00, 0x00)

	ppu := NewPPU(mmu)
	ppu.Reset()

	// Set LCDC bit 3 to use 0x9C00 tile map area.
	ppu.lcdc |= 0x08

	// Step through a frame.
	ppu.Step(FrameCycles)

	screen := ppu.GetScreen()

	// With solid black tile at map (0,0), pixel (0,0) should be shade 3.
	assert.Equal(t, byte(3), screen[0][0], "top-left pixel should be shade 3 (black tile)")
}

// TestSCYSCXScroll verifies the background scroll registers shift the view.
func TestSCYSCXScroll(t *testing.T) {
	mmu := NewMMU(nil)

	// Write a distinctive tile: row 0 = all shade 3, rest = all shade 0.
	// Row 0: low=0xFF, high=0xFF → pixel=3 everywhere
	mmu.Write(0x8000, 0xFF)
	mmu.Write(0x8001, 0xFF)
	// Rows 1-7: all zeros
	for addr := uint16(0x8002); addr < 0x8010; addr++ {
		mmu.Write(addr, 0x00)
	}

	// Write a different tile at index 1 for map position (1,0).
	// All shade 1.
	mmu.Write(0x8010, 0x55)
	mmu.Write(0x8011, 0xAA)

	// Set tile map: position (0,0) = tile 0, position (1,0) = tile 1
	mmu.Write(0x9800, 0x00) // tile 0
	mmu.Write(0x9801, 0x01) // tile 1

	ppu := NewPPU(mmu)
	ppu.Reset()

	// Set SCY=0, SCX=0
	ppu.scy = 0
	ppu.scx = 0

	// Step through a frame.
	ppu.Step(FrameCycles)
	screenZero := ppu.GetScreen()

	// At (0,0) with no scroll, we should see tile 0, row 0, pixel 0 = shade 3.
	assert.Equal(t, byte(3), screenZero[0][0], "without scroll, pixel (0,0) should be shade 3")

	// Now scroll right by 8 pixels — this shifts tile 1 into view at position 0.
	// Actually, SCX=8 shifts the background LEFT by 8, so the visible area starts at
	// tile map column 1. Tile 1 starts at screen position 0.
	mmu.Write(0xFF43, 8) // SCX = 8 via MMU (triggers ppu.scx update)

	ppu = NewPPU(mmu)
	ppu.Reset()
	ppu.scx = 8
	ppu.Step(FrameCycles)
	screenScrolled := ppu.GetScreen()

	// With SCX=8, the first pixel is from tile map entry (1,0).
	// Tile 1 has all shade 1 pixels.
	// But with the complex SCX wrap, pixel (0,0) = tile index 1, pixel col 0 = shade 1.
	// Wait, actually SCX=8 means we skip 8 pixels of the first tile.
	// Tile 0 (map pos 0) has 8 pixels at columns 0-7. After skipping 8 (SCX=8),
	// tile 0 is entirely skipped, and we start at map position (1,0).
	// So screen pixel 0 should be tile 1's pixel 0.
	assert.Equal(t, byte(1), screenScrolled[0][0], "with SCX=8, visible pixel 0 should be from tile 1 → shade 1")
	_ = screenScrolled

	// Reset and scroll down by 8 pixels (SCY=8).
	// This means LY=0 shows tile row 1 instead of tile row 0.
	ppu = NewPPU(mmu)
	ppu.Reset()
	ppu.scy = 8
	ppu.Step(FrameCycles)
	screenScrolledY := ppu.GetScreen()

	// With SCY=8, tile 0 row 1 is at screen row 0.
	// Tile 0 row 1 has all shade 0 pixels.
	assert.Equal(t, byte(0), screenScrolledY[0][0], "with SCY=8, visible pixel (0,0) should be tile 0 row 1 → shade 0")
}

// TestWindowLayer verifies the window layer renders on top of the background.
func TestWindowLayer(t *testing.T) {
	mmu := NewMMU(nil)

	// Fill background tiles with shade 1 (light) so window stands out.
	// Write a distinctive "W" pattern tile at tile index 1 for testing.
	// All pixels shade 3 for the window.
	for addr := uint16(0x8000); addr < 0x8010; addr++ {
		mmu.Write(addr, 0xFF) // tile 0: solid shade 3
	}
	// Tile 1: all shade 1
	for addr := uint16(0x8010); addr < 0x8020; addr++ {
		mmu.Write(addr, 0x55) // low byte = 0x55 → pixel col = 1
	}
	for addr := uint16(0x8020); addr < 0x8030; addr++ {
		mmu.Write(addr, 0x00) // tile 2: all shade 0
	}

	// Background tile map: all tile 1 (shade 1)
	for i := uint16(0); i < 32*32; i++ {
		mmu.Write(0x9800+i, 0x01)
	}

	// Window tile map (at 0x9C00, LCDC.6 = 1): tile 0 (solid shade 3) at top-left
	for i := uint16(0); i < 32*32; i++ {
		mmu.Write(0x9C00+i, 0x02) // tile 2: all shade 0 — fill window map with 0
	}
	// Place tile 0 (solid shade 3) at window map position (0,0)
	mmu.Write(0x9C00, 0x00)

	ppu := NewPPU(mmu)
	ppu.Reset()

	// Enable window: LCDC bit 5 = 1
	ppu.lcdc |= 0x20
	// Set window tile map to 0x9C00: LCDC bit 6 = 1
	ppu.lcdc |= 0x40

	// Set window position: WX=7, WY=0
	// (WX is stored as written minus 7 in hardware)
	ppu.wx = 7
	ppu.wy = 0

	// Step through a frame.
	ppu.Step(FrameCycles)

	screen := ppu.GetScreen()

	// Window with tile 0 (shade 3) at position (0,0) window coords maps to
	// screen position (WX-7, WY) = (0, 0). But we set WX=7, so screen_x = 7-7 = 0.
	// So screen[0][0] should be from the window, tile 0 = shade 3.
	assert.Equal(t, byte(3), screen[0][0], "window pixel (0,0) should be shade 3 (tile 0)")
}
