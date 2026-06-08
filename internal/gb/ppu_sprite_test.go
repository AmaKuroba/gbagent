package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Helper: write sprite data to OAM.
// Sprite Y = oam[0] - 16, Sprite X = oam[1] - 8.
func writeSpriteOAM(mmu *MemoryBus, index int, y, x, tile byte, attrs byte) {
	base := uint16(0xFE00 + index*4)
	mmu.Write(base, y)
	mmu.Write(base+1, x)
	mmu.Write(base+2, tile)
	mmu.Write(base+3, attrs)
}

// --- Test: Single sprite rendering ---

func TestSpriteRender(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	// LCDC: enable + BG + OBJ
	ppu.WriteRegister(0xFF40, 0x93)

	// Place sprite at screen (10,10) with tile 1, shade 3
	writeSpriteOAM(mmu, 0, 26, 18, 1, 0) // Y=10, X=10

	// Tile 1 = solid shade 3
	writeTileToVRAM(mmu, 1, 3)
	// BG tile 0 = solid shade 0
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	assert.Equal(t, byte(3), s[10][10], "sprite top-left")
	assert.Equal(t, byte(3), s[17][10], "sprite right edge")
	assert.Equal(t, byte(3), s[10][17], "sprite bottom edge")

	// Pixels outside sprite bounds should be BG (shade 0)
	assert.Equal(t, byte(0), s[9][10], "left of sprite")
	assert.Equal(t, byte(0), s[18][10], "right of sprite")
	assert.Equal(t, byte(0), s[10][18], "below sprite")
}

// --- Test: Sprite disabled via LCDC bit 1 ---

func TestSpriteDisabled(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	// LCDC: enable + BG, but NOT OBJ (bit 1 = 0)
	ppu.WriteRegister(0xFF40, 0x91)

	writeSpriteOAM(mmu, 0, 26, 18, 1, 0) // Y=10, X=10
	writeTileToVRAM(mmu, 1, 3)
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	assert.Equal(t, byte(0), s[10][10], "sprite invisible when OBJ disabled")
}

// --- Test: Sprite priority (lower X = higher priority) ---

func TestSpritePriority(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	// Two overlapping sprites on LY=10:
	// Sprite 0 at X=10 (higher priority), tile 1 = shade 3
	// Sprite 1 at X=12 (lower priority), tile 2 = shade 1
	writeSpriteOAM(mmu, 0, 26, 18, 1, 0) // Y=10, X=10
	writeSpriteOAM(mmu, 1, 26, 20, 2, 0) // Y=10, X=12

	writeTileToVRAM(mmu, 1, 3) // shade 3
	writeTileToVRAM(mmu, 2, 1) // shade 1
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	// At (10,10): only sprite 0 covers this
	assert.Equal(t, byte(3), s[10][10], "sprite 0 alone")

	// At (12,10): both sprites cover; sprite 0 has lower X → wins
	assert.Equal(t, byte(3), s[12][10], "sprite 0 wins at overlap")

	// At (18,10): only sprite 1 covers (sprite 0 ends at 17)
	assert.Equal(t, byte(1), s[18][10], "sprite 1 alone")
}

// --- Test: Sprite X flip, Y flip, and both ---

func TestSpriteFlip(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	// Build tile 1: row 0 has pattern (pixel 0=shade1, pixel 1=shade2, pixel 2=shade3, rest=0)
	// Row 1: pixel 0=shade1. Rows 2-7: all 0.
	t1 := uint16(0x8000 + 1*16)
	// Row 0: lo bits 0,2=1 → pixels 0,2 have lo=1; hi bits 1,2=1 → pixels 1,2 have hi=1
	// pixel 0 = 01 = 1, pixel 1 = 10 = 2, pixel 2 = 11 = 3, rest = 0
	mmu.Write(t1, 0x05)
	mmu.Write(t1+1, 0x06)
	// Row 1: pixel 0=1
	mmu.Write(t1+2, 0x01)
	mmu.Write(t1+3, 0x00)
	// Rows 2-7: all 0

	// Sprite 0: no flip at (10,10)
	writeSpriteOAM(mmu, 0, 26, 18, 1, 0)
	// Sprite 1: X flip at (30,10)
	writeSpriteOAM(mmu, 1, 26, 38, 1, 0x20)
	// Sprite 2: Y flip at (10,30)
	writeSpriteOAM(mmu, 2, 46, 18, 1, 0x40)
	// Sprite 3: both flips at (30,30)
	writeSpriteOAM(mmu, 3, 46, 38, 1, 0x60)

	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	// No flip: (10,10) = sprite pixel 0=1
	assert.Equal(t, byte(1), s[10][10], "no flip: (10,10)")
	assert.Equal(t, byte(2), s[11][10], "no flip: (11,10)")
	assert.Equal(t, byte(3), s[12][10], "no flip: (12,10)")

	// X flip at (30,10): pixel order reversed
	// screen X=30 → spritePixelX=0 → tile pixel 7 → shade 0
	assert.Equal(t, byte(0), s[30][10], "x flip: left → tile pixel 7")
	// screen X=37 → spritePixelX=7 → tile pixel 0 → shade 1
	assert.Equal(t, byte(1), s[37][10], "x flip: right → tile pixel 0")

	// Y flip at (10,30): row order reversed
	// screen Y=30 → spriteRow=0 → tile row 7 → shade 0
	assert.Equal(t, byte(0), s[10][30], "y flip: top → tile row 7")
	// screen Y=31 → spriteRow=1 → tile row 6 → shade 0
	assert.Equal(t, byte(0), s[10][31], "y flip: row 1 → tile row 6")

	// Both flips at (30,30): row+pixel reversed
	// screen (30,30) → row 0, pixel 0 → tile row 7, pixel 7 → shade 0
	assert.Equal(t, byte(0), s[30][30], "both: (30,30)")

	// screen (37,37) → row 7, pixel 7 → tile row 0, pixel 0 → shade 1
	// but row 1 of tile is also shade 1 for pixel 0, so...
	// Actually spriteRow = 37-30 = 7, Y-flip → tileRow = 7-7 = 0. tile pixel 0 = shade 1. ✓
	assert.Equal(t, byte(1), s[37][37], "both: (37,37)")
}

// --- Test: OBP0 and OBP1 palette mapping ---

func TestSpritePaletteOBP0(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	// Sprite using OBP0 (bit 4 clear), tile 1 = shade 3
	writeSpriteOAM(mmu, 0, 26, 18, 1, 0) // Y=10, X=10, attr=0 → OBP0
	writeTileToVRAM(mmu, 1, 3)
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	// Set OBP0: pixel value 3 → shade 1 (0x40 = 01 00 00 00)
	ppu.WriteRegister(0xFF48, 0x40) // pixel3→1, pixel2→0, pixel1→0, pixel0→0

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	assert.Equal(t, byte(1), s[10][10], "OBP0: shade 3 pixel → shade 1")
}

func TestSpritePaletteOBP1(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	// Sprite using OBP1 (bit 4 set)
	writeSpriteOAM(mmu, 0, 26, 18, 1, 0x10) // Y=10, X=10, attr=0x10 → OBP1
	writeTileToVRAM(mmu, 1, 3)
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	// Set OBP1 to map all pixels → shade 0
	ppu.WriteRegister(0xFF49, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	assert.Equal(t, byte(0), s[10][10], "OBP1: shade 3 pixel → shade 0")
}

// --- Test: 8x16 sprite mode ---

func TestSpriteSize8x16(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	// LCDC: enable + BG + OBJ + OBJ size (bit 2)
	ppu.WriteRegister(0xFF40, 0x93|0x04)

	// 8x16 sprite: tile index has bit 0 forced to 0.
	// Tile 2 = upper half, tile 3 = lower half.
	// Using tile index 3 → tile & 0xFE = 2 = upper, 3 = lower
	writeSpriteOAM(mmu, 0, 26, 18, 3, 0) // Y=10, X=10

	writeTileToVRAM(mmu, 2, 3) // upper half: shade 3
	writeTileToVRAM(mmu, 3, 1) // lower half: shade 1
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	// Upper tile (rows 0-7 of sprite): shade 3
	assert.Equal(t, byte(3), s[10][10], "8x16: upper half")
	assert.Equal(t, byte(3), s[10][17], "8x16: upper half bottom")

	// Lower tile (rows 8-15 of sprite): tile 3 (shade 1)
	assert.Equal(t, byte(1), s[10][18], "8x16: lower half top")
	assert.Equal(t, byte(1), s[10][25], "8x16: lower half bottom")

	// Below sprite: BG (shade 0)
	assert.Equal(t, byte(0), s[10][26], "8x16: below sprite")
}

// --- Test: Sprite vs background priority (bit 7) ---

func TestSpriteVsBackgroundPriority(t *testing.T) {
	t.Parallel()

	// Case 1: Priority bit set, BG pixel = 0 → sprite shows
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	writeSpriteOAM(mmu, 0, 26, 18, 1, 0x80) // Y=10, X=10, behind BG
	writeTileToVRAM(mmu, 1, 3) // sprite tile = shade 3
	writeTileToVRAM(mmu, 0, 0) // BG tile 0 = shade 0
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	assert.Equal(t, byte(3), s[10][10], "prio bit set, BG=0 → sprite shows")

	// Case 2: Priority bit set, BG pixel != 0 → BG shows
	mmu2 := NewMMU(nil)
	ppu2 := NewPPU(mmu2)
	ppu2.Reset()
	ppu2.WriteRegister(0xFF40, 0x93)

	writeTileToVRAM(mmu2, 0, 1) // BG tile 0 = shade 1
	fillTileMap(mmu2, 0x9800, 0x00)
	writeSpriteOAM(mmu2, 0, 26, 18, 1, 0x80) // Y=10, X=10, behind BG
	writeTileToVRAM(mmu2, 1, 3) // sprite tile = shade 3

	ppu2.Step(FrameCycles)
	s2 := ppu2.GetScreen()

	assert.Equal(t, byte(1), s2[10][10], "prio bit set, BG≠0 → BG shows")

	// Case 3: Priority bit clear, BG pixel != 0 → sprite shows
	mmu3 := NewMMU(nil)
	ppu3 := NewPPU(mmu3)
	ppu3.Reset()
	ppu3.WriteRegister(0xFF40, 0x93)

	writeTileToVRAM(mmu3, 0, 1) // BG tile 0 = shade 1
	fillTileMap(mmu3, 0x9800, 0x00)
	writeSpriteOAM(mmu3, 0, 26, 18, 1, 0x00) // Y=10, X=10, above BG
	writeTileToVRAM(mmu3, 1, 3) // sprite tile = shade 3

	ppu3.Step(FrameCycles)
	s3 := ppu3.GetScreen()

	assert.Equal(t, byte(3), s3[10][10], "prio bit clear, BG≠0 → sprite shows")
}

// --- Test: Sprite max 10 per scanline ---

func TestSpriteMax10PerLine(t *testing.T) {
	t.Parallel()

	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x93)

	// Place 12 sprites on LY=10, at different X positions
	for i := 0; i < 12; i++ {
		x := byte(10 + i*12)
		writeSpriteOAM(mmu, i, 26, x+8, 1, 0) // Y=10, X=x
	}
	writeTileToVRAM(mmu, 1, 3)
	writeTileToVRAM(mmu, 0, 0)
	fillTileMap(mmu, 0x9800, 0x00)

	ppu.Step(FrameCycles)
	s := ppu.GetScreen()

	// Sprites 0-9 should be visible (first 10)
	assert.Equal(t, byte(3), s[10][10], "sprite 0 visible")
	assert.Equal(t, byte(3), s[118][10], "sprite 9 visible")

	// Sprite 10 should not be visible (past 10 limit)
	assert.Equal(t, byte(0), s[130][10], "sprite 10 not visible (past limit)")
}
