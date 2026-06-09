package gb

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeTileToVRAM(mmu *MemoryBus, tileIndex int, shade byte) {
	lo := byte(0x00)
	hi := byte(0x00)
	switch shade {
	case 1:
		lo = 0xFF
	case 2:
		hi = 0xFF
	case 3:
		lo = 0xFF
		hi = 0xFF
	}
	base := uint16(0x8000 + tileIndex*16)
	for row := 0; row < 8; row++ {
		mmu.Write(base+uint16(row*2), lo)
		mmu.Write(base+uint16(row*2+1), hi)
	}
}

func writeCheckerboardTile(mmu *MemoryBus, tileIndex int) {
	base := uint16(0x8000 + tileIndex*16)
	for row := 0; row < 8; row++ {
		var lo byte
		if row%2 == 0 {
			lo = 0x55
		} else {
			lo = 0xAA
		}
		mmu.Write(base+uint16(row*2), lo)
		mmu.Write(base+uint16(row*2+1), 0x00)
	}
}

func fillTileMap(mmu *MemoryBus, mapBase uint16, tileIndex byte) {
	for i := uint16(0); i < 32*32; i++ {
		mmu.Write(mapBase+i, tileIndex)
	}
}

func setupBasicBg(t *testing.T) (*PPUCore, *MemoryBus) {
	t.Helper()
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	writeCheckerboardTile(mmu, 0)
	fillTileMap(mmu, 0x9800, 0x00)
	return ppu, mmu
}

func TestTileDataRead(t *testing.T) {
	t.Parallel()
	ppu, mmu := setupBasicBg(t)
	lo := mmu.Read(0x8000)
	hi := mmu.Read(0x8001)
	assert.Equal(t, byte(0x55), lo)
	assert.Equal(t, byte(0x00), hi)
	pixel0 := (hi&0x01)<<1 | (lo & 0x01)
	assert.Equal(t, byte(1), pixel0)
	ppu.Step(FrameCycles)
	screen := ppu.GetScreen()
	require.NotNil(t, screen)
	assert.Equal(t, 160, len(screen))
	assert.Equal(t, 144, len(screen[0]))
}

func TestBackgroundRender(t *testing.T) {
	t.Parallel()
	ppu, _ := setupBasicBg(t)
	ppu.Step(FrameCycles)
	s := ppu.GetScreen()
	assert.Equal(t, byte(1), s[0][0], "p(0,0)")
	assert.Equal(t, byte(0), s[1][0], "p(1,0)")
	assert.Equal(t, s[0][0], s[8][0], "p(8,0)=p(0,0)")
	assert.Equal(t, byte(0), s[0][1], "p(0,1)")
	assert.Equal(t, byte(1), s[1][1], "p(1,1)")
}

func TestBackgroundRenderSolidTile(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	ppu := NewPPU(mmu)
	ppu.Reset()
	writeTileToVRAM(mmu, 0, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	ppu.Step(FrameCycles)
	s := ppu.GetScreen()
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			assert.Equal(t, byte(3), s[x][y], "(%d,%d)", x, y)
		}
	}
}

func TestBackgroundRenderTileMapAddressing(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	writeTileToVRAM(mmu, 1, 1)
	fillTileMap(mmu, 0x9800, 0x00)
	fillTileMap(mmu, 0x9C00, 0x01)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(3), ppu.GetScreen()[0][0], "map 0x9800")
	mmu2 := NewMMU(nil)
	writeTileToVRAM(mmu2, 0, 3)
	writeTileToVRAM(mmu2, 1, 1)
	fillTileMap(mmu2, 0x9800, 0x00)
	fillTileMap(mmu2, 0x9C00, 0x01)
	ppu2 := NewPPU(mmu2)
	ppu2.Reset()
	ppu2.WriteRegister(0xFF40, 0x91|0x08)
	ppu2.Step(FrameCycles)
	assert.Equal(t, byte(1), ppu2.GetScreen()[0][0], "map 0x9C00")
}

func TestBackgroundRenderTileDataAddressing(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	writeTileToVRAM(mmu, 128, 1)
	fillTileMap(mmu, 0x9800, 0x00)
	ppu := NewPPU(mmu)
	ppu.Reset()
	// Signed mode: LCDC bit 4 = 0. Default 0x91 has bit 4 set, so clear it.
	ppu.WriteRegister(0xFF40, 0x81) // LCD enable + BG enable, bit 4 = 0
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(1), ppu.GetScreen()[0][0], "signed mode")
	mmu2 := NewMMU(nil)
	writeTileToVRAM(mmu2, 0, 3)
	writeTileToVRAM(mmu2, 128, 1)
	fillTileMap(mmu2, 0x9800, 0x00)
	ppu2 := NewPPU(mmu2)
	ppu2.Reset()
	// Default 0x91 already has bit 4 set = unsigned mode.
	ppu2.Step(FrameCycles)
	assert.Equal(t, byte(3), ppu2.GetScreen()[0][0], "unsigned mode")
}

func TestSCYScroll(t *testing.T) {
	t.Parallel()
	makePPU := func() *PPUCore {
		mmu := NewMMU(nil)
		base := uint16(0x8000)
		mmu.Write(base, 0xFF)
		mmu.Write(base+1, 0xFF)
		for addr := base + 2; addr < base+16; addr++ {
			mmu.Write(addr, 0x00)
		}
		fillTileMap(mmu, 0x9800, 0x00)
		ppu := NewPPU(mmu)
		ppu.Reset()
		return ppu
	}
	ppu := makePPU()
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(3), ppu.GetScreen()[0][0], "no scroll")
	ppu2 := makePPU()
	ppu2.WriteRegister(0xFF42, 1) // SCY = 1: shift by 1 pixel, tile row 1 is now at screen row 0
	ppu2.Step(FrameCycles)
	assert.Equal(t, byte(0), ppu2.GetScreen()[0][0], "SCY=1")
}

func TestSCXScroll(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	writeTileToVRAM(mmu, 1, 1)
	mmu.Write(0x9800, 0x00)
	mmu.Write(0x9801, 0x01)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(3), ppu.GetScreen()[0][0], "no SCX col0")
	assert.Equal(t, byte(1), ppu.GetScreen()[8][0], "no SCX col8")
	mmu2 := NewMMU(nil)
	writeTileToVRAM(mmu2, 0, 3)
	writeTileToVRAM(mmu2, 1, 1)
	mmu2.Write(0x9800, 0x00)
	mmu2.Write(0x9801, 0x01)
	ppu2 := NewPPU(mmu2)
	ppu2.Reset()
	ppu2.WriteRegister(0xFF43, 8)
	ppu2.Step(FrameCycles)
	assert.Equal(t, byte(1), ppu2.GetScreen()[0][0], "SCX=8")
}

func TestSCYSCXCombined(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	writeTileToVRAM(mmu, 1, 1)
	writeTileToVRAM(mmu, 2, 2)
	writeTileToVRAM(mmu, 3, 0)
	mmu.Write(0x9800, 0x00)
	mmu.Write(0x9801, 0x01)
	mmu.Write(0x9820, 0x02)
	mmu.Write(0x9821, 0x03)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF42, 8)
	ppu.WriteRegister(0xFF43, 8)
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(0), ppu.GetScreen()[0][0], "SCY=8,SCX=8")
}

func TestWindowLayer(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 1)
	writeTileToVRAM(mmu, 1, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	fillTileMap(mmu, 0x9C00, 0x00)
	mmu.Write(0x9C00, 0x01)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91|0x20|0x40)
	ppu.WriteRegister(0xFF4A, 0)
	ppu.WriteRegister(0xFF4B, 7)
	ppu.Step(FrameCycles)
	s := ppu.GetScreen()
	assert.Equal(t, byte(3), s[0][0], "win(0,0)")
	assert.Equal(t, byte(1), s[8][0], "win(8,0)")
}

func TestWindowLayerOff(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 1)
	writeTileToVRAM(mmu, 1, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	mmu.Write(0x9C00, 0x01)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x91|0x40)
	ppu.WriteRegister(0xFF4A, 0)
	ppu.WriteRegister(0xFF4B, 7)
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(1), ppu.GetScreen()[0][0], "win off")
}

func TestBackgroundPalette(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(3), ppu.GetScreen()[0][0], "default BGP")
	mmu2 := NewMMU(nil)
	writeTileToVRAM(mmu2, 0, 3)
	fillTileMap(mmu2, 0x9800, 0x00)
	ppu2 := NewPPU(mmu2)
	ppu2.Reset()
	ppu2.WriteRegister(0xFF47, 0x00)
	ppu2.Step(FrameCycles)
	assert.Equal(t, byte(0), ppu2.GetScreen()[0][0], "BGP=0")
}

func TestWindowDisabledByLCDC0(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 1)
	writeTileToVRAM(mmu, 1, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	fillTileMap(mmu, 0x9C00, 0x00)
	mmu.Write(0x9C00, 0x01)
	ppu := NewPPU(mmu)
	ppu.Reset()
	// Set LCDC to: bit 7=1 (LCD enable), bit 6=1 (win tile map 0x9C00), bit 5=1 (win enable)
	// bit 0=0 (BG/Win disabled), bit 4=1 for unsigned tile addressing so our test tiles resolve.
	// Value: 0x80 | 0x40 | 0x20 | 0x10 = 0xF0
	ppu.WriteRegister(0xFF40, 0xF0)
	ppu.WriteRegister(0xFF4A, 0)  // WY = 0
	ppu.WriteRegister(0xFF4B, 7)  // WX = 7
	ppu.Step(FrameCycles)
	s := ppu.GetScreen()
	// When LCDC bit 0 is 0, window should not be drawn even though bit 5 is set.
	// Background rendering also fills with 0 when bit 0 is 0.
	assert.Equal(t, byte(0), s[0][0], "win disabled by LCDC.0 = 0 at win origin")
	assert.Equal(t, byte(0), s[8][0], "win disabled by LCDC.0 = 0 inside window area")
}

func TestBackgroundDisabled(t *testing.T) {
	t.Parallel()
	mmu := NewMMU(nil)
	writeTileToVRAM(mmu, 0, 3)
	fillTileMap(mmu, 0x9800, 0x00)
	ppu := NewPPU(mmu)
	ppu.Reset()
	ppu.WriteRegister(0xFF40, 0x90)
	ppu.Step(FrameCycles)
	assert.Equal(t, byte(0), ppu.GetScreen()[0][0], "BG disabled")
}
