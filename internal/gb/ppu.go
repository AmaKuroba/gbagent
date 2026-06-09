package gb

// PPU represents the Picture Processing Unit (160x144 LCD).
type PPU interface {
	Step(cycles int)
	Reset()
	GetScreen() [160][144]byte
	GetState() PPUState
}

// PPUState is a snapshot of PPU registers and timing.
type PPUState struct {
	Mode       int
	LY         byte
	LCDC       byte
	STAT       byte
	FrameCount int
}

// PPU timing constants.
const (
	oamSearchCycles  = 80
	vramDrawCycles   = 172
	hBlankCycles     = 204
	scanlineCycles   = 456
	visibleScanlines = 144
	vBlankScanlines  = 10
	totalScanlines   = 154
	frameCycles      = 70224
)

// PPU modes.
const (
	ppuModeHBlank = iota
	ppuModeVBlank
	ppuModeOAM
	ppuModeVRAM
)

// STAT register bits.
const (
	statBitPPUMode0 = 1 << 0
	statBitPPUMode1 = 1 << 1
	statBitLYCoinc  = 1 << 2
	statBitIntMode0 = 1 << 3
	statBitIntMode1 = 1 << 4
	statBitIntMode2 = 1 << 5
	statBitIntLYC   = 1 << 6
)

// LCDC bit masks.
const (
	lcdcBitBGEnable   = 1 << 0
	lcdcBitOBJEnable  = 1 << 1
	lcdcBitOBJSize    = 1 << 2
	lcdcBitBGTileMap  = 1 << 3
	lcdcBitBGTileData = 1 << 4
	lcdcBitWinEnable  = 1 << 5
	lcdcBitWinTileMap = 1 << 6
	lcdcBitLCDEnable  = 1 << 7
)

// Sprite attribute flags.
const (
	spriteAttrPriority  = 1 << 7 // 0=above BG, 1=behind BG (if BG pixel != 0)
	spriteAttrYFlip     = 1 << 6
	spriteAttrXFlip     = 1 << 5
	spriteAttrPaletteDMG = 1 << 4 // 0=OBP0, 1=OBP1
)

// spriteEntry holds one OAM sprite's metadata for the current scanline.
type spriteEntry struct {
	x     byte // Raw X value (sprite X = x - 8)
	y     byte // Raw Y value (sprite Y = y - 16)
	tile  byte // Tile index
	attrs byte // Attributes
}

// PPUCore is the concrete PPU timing state machine with background, window & sprite rendering.
type PPUCore struct {
	mmu  MMU

	dotCounter int
	ly         byte
	lyc        byte
	stat       byte
	lcdc       byte
	frameCtr   int

	scy    byte
	scx    byte
	bgp    byte
	obp0   byte
	obp1   byte
	wx     byte
	wy     byte

	isRunning  bool
	screen     [160][144]byte
	scanlineRendered bool
	mode2End   int
	mode3End   int

	// Sprite / OAM state (per scanline)
	oamScanned bool
	oamSprites []spriteEntry

	// firstFrameBlank is set when LCD is enabled (LCDC bit 7 rising edge).
	// While set, the screen remains blank (all white) for one full frame.
	firstFrameBlank bool
}

var _ PPU = (*PPUCore)(nil)

// NewPPU creates a new PPU instance.
func NewPPU(mmu MMU) *PPUCore {
	ppu := &PPUCore{
		mmu:  mmu,
		lcdc: 0x00,
		bgp:  0xE4,
		obp0: 0xE4,
		obp1: 0xE4,
	}
	ppu.Reset()
	return ppu
}

func (p *PPUCore) Reset() {
	p.dotCounter = 0
	p.ly = 0
	p.stat &^= 0x07
	p.stat |= ppuModeOAM << 0
	p.isRunning = true
	p.frameCtr = 0
	p.scanlineRendered = false
	p.mode2End = oamSearchCycles
	p.mode3End = oamSearchCycles + vramDrawCycles
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			p.screen[x][y] = 0
		}
	}
}

func (p *PPUCore) Step(cycles int) {
	// When LCD is disabled (LCDC bit 7 = 0), LY must be 0 and must not increment.
	if p.lcdc&lcdcBitLCDEnable == 0 {
		p.ly = 0
		return
	}
	if cycles <= 0 {
		return
	}
	for cycles > 0 {
		remaining := scanlineCycles - p.dotCounter
		step := cycles
		if step > remaining {
			step = remaining
		}
		prevDot := p.dotCounter

		p.dotCounter += step
		cycles -= step
		p.updateMode()

		// Detect OAM search crossing to scan sprites for this scanline.
		crossedOAM := prevDot < p.mode2End && p.dotCounter > 0
		if p.ly < visibleScanlines && !p.oamScanned && crossedOAM {
			if p.mmu != nil {
				p.scanOAM()
			}
			p.oamScanned = true
		}

		// Detect if we crossed through the VRAM drawing period [mode2End, mode3End).
		crossedVRAM := prevDot < p.mode3End && p.dotCounter > p.mode2End
		if p.ly < visibleScanlines && !p.scanlineRendered && crossedVRAM {
			if p.mmu != nil {
				if p.firstFrameBlank {
					// First frame after LCD enable — render blank (all white).
					for x := 0; x < 160; x++ {
						p.screen[x][p.ly] = 0
					}
				} else {
					p.renderBackgroundScanline()
					p.renderWindowScanline()
					if p.lcdc&lcdcBitOBJEnable != 0 {
						p.renderSprites()
					}
				}
			}
			p.scanlineRendered = true
		}
		if p.dotCounter >= scanlineCycles {
			p.advanceScanline()
		}
	}
}

func (p *PPUCore) advanceScanline() {
	p.dotCounter = 0
	p.ly++
	p.scanlineRendered = false
	p.oamScanned = false
	if p.ly >= totalScanlines {
		p.ly = 0
		p.frameCtr++
		// End first-frame blank period after one complete frame.
		p.firstFrameBlank = false
	}
	p.mode2End = oamSearchCycles
	p.mode3End = oamSearchCycles + vramDrawCycles
	p.updateMode()

	// VBlank interrupt: always fires when entering VBlank (LY becomes 144).
	if p.ly == 144 && p.mmu != nil {
		p.mmu.WriteIF(p.mmu.ReadIF() | 0x01)
	}
}

func (p *PPUCore) updateMode() int {
	oldMode := p.stat & 0x03
	oldLYCoinc := p.stat&statBitLYCoinc != 0

	var mode int
	if p.ly < visibleScanlines {
		if p.dotCounter < p.mode2End {
			mode = ppuModeOAM
		} else if p.dotCounter < p.mode3End {
			mode = ppuModeVRAM
		} else {
			mode = ppuModeHBlank
		}
	} else {
		mode = ppuModeVBlank
	}
	p.stat &^= 0x03
	p.stat |= byte(mode) & 0x03

	newLYCoinc := p.ly == p.lyc
	if newLYCoinc {
		p.stat |= statBitLYCoinc
	} else {
		p.stat &^= statBitLYCoinc
	}

	// Trigger STAT interrupt (IF bit 1) on LYC=LY rising edge.
	if !oldLYCoinc && newLYCoinc && p.stat&statBitIntLYC != 0 {
		if p.mmu != nil {
			p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
		}
	}

	// Trigger STAT interrupt on mode transitions (except VRAM mode entry, which
	// does not generate a STAT interrupt on DMG).
	newMode := byte(mode)
	if oldMode != newMode {
		switch mode {
		case ppuModeHBlank:
			if p.stat&statBitIntMode0 != 0 && p.mmu != nil {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
		case ppuModeVBlank:
			if p.stat&statBitIntMode1 != 0 && p.mmu != nil {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
		case ppuModeOAM:
			if p.stat&statBitIntMode2 != 0 && p.mmu != nil {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
		}
	}

	return mode
}

func (p *PPUCore) tileMapBase() uint16 {
	if p.lcdc&lcdcBitBGTileMap != 0 {
		return 0x9C00
	}
	return 0x9800
}

func (p *PPUCore) tileDataAddress(tileIndex byte) uint16 {
	if p.lcdc&lcdcBitBGTileData != 0 {
		return 0x8000 + uint16(tileIndex)*16
	}
	return 0x9000 + uint16(int8(tileIndex))*16
}

func (p *PPUCore) decodeTileRow(tileAddr uint16, row int) [8]byte {
	bus := p.mmu.(*MemoryBus)
	lo := bus.ReadVRAMDirect(tileAddr + uint16(row*2))
	hi := bus.ReadVRAMDirect(tileAddr + uint16(row*2+1))
	var pixels [8]byte
	for i := 0; i < 8; i++ {
		lowBit := (lo >> (7 - i)) & 1
		highBit := (hi >> (7 - i)) & 1
		pixels[i] = (highBit << 1) | lowBit
	}
	return pixels
}

func (p *PPUCore) applyPalette(pixel byte, palette byte) byte {
	shift := pixel * 2
	return (palette >> shift) & 0x03
}

func (p *PPUCore) renderBackgroundScanline() {
	if p.lcdc&lcdcBitBGEnable == 0 {
		for x := 0; x < 160; x++ {
			p.screen[x][p.ly] = 0
		}
		return
	}
	y := int(p.ly)
	scrollY := int(p.scy)
	scrollX := int(p.scx)
	mapBase := p.tileMapBase()
	for x := 0; x < 160; x++ {
		tileMapX := (x + scrollX) / 8
		tileMapY := (y + scrollY) / 8
		tileMapAddr := mapBase + uint16((tileMapY%32)*32+(tileMapX%32))
		bus := p.mmu.(*MemoryBus)
		tileIndex := bus.ReadVRAMDirect(tileMapAddr)
		tileAddr := p.tileDataAddress(tileIndex)
		tileRow := (y + scrollY) % 8
		pixels := p.decodeTileRow(tileAddr, tileRow)
		tileCol := (x + scrollX) % 8
		p.screen[x][y] = p.applyPalette(pixels[tileCol], p.bgp)
	}
}

func (p *PPUCore) windowTileMapBase() uint16 {
	if p.lcdc&lcdcBitWinTileMap != 0 {
		return 0x9C00
	}
	return 0x9800
}

func (p *PPUCore) renderWindowScanline() {
	// When LCDC bit 0 (BG/Window enable) is 0, both BG and Window are disabled
	// regardless of bit 5 (Window enable).
	if p.lcdc&lcdcBitBGEnable == 0 {
		return
	}
	if p.lcdc&lcdcBitWinEnable == 0 {
		return
	}
	y := int(p.ly)
	winY := int(p.wy)
	winX := int(p.wx) - 7
	if y >= visibleScanlines || y < winY {
		return
	}
	winMapBase := p.windowTileMapBase()
	windowTileY := (y - winY) / 8
	windowRowInTile := (y - winY) % 8
	for x := 0; x < 160; x++ {
		if x < winX {
			continue
		}
		winPixelX := x - winX
		windowTileX := winPixelX / 8
		windowColInTile := winPixelX % 8
		tileMapAddr := winMapBase + uint16((windowTileY%32)*32+(windowTileX%32))
		bus := p.mmu.(*MemoryBus)
		tileIndex := bus.ReadVRAMDirect(tileMapAddr)
		tileAddr := p.tileDataAddress(tileIndex)
		pixels := p.decodeTileRow(tileAddr, windowRowInTile)
		p.screen[x][y] = p.applyPalette(pixels[windowColInTile], p.bgp)
	}
}

// scanOAM scans the OAM table (0xFE00-0xFE9F) and selects up to 10 sprites
// that intersect the current scanline, sorted by X (lower X = higher priority).
func (p *PPUCore) scanOAM() {
	p.oamSprites = p.oamSprites[:0]

	spriteHeight := 8
	if p.lcdc&lcdcBitOBJSize != 0 {
		spriteHeight = 16
	}

	ly := int(p.ly)
	for i := 0; i < 40 && len(p.oamSprites) < 10; i++ {
		base := uint16(0xFE00 + i*4)
		// Use ReadOAMDirect to bypass MMU mode checking — the PPU's own OAM
		// scan reads must not trigger the OAM corruption/bug (which only
		// applies to CPU-initiated accesses during mode 2).
		mmu := p.mmu.(*MemoryBus)
		y := mmu.ReadOAMDirect(base)
		x := mmu.ReadOAMDirect(base + 1)

		spriteY := int(y) - 16

		if ly >= spriteY && ly < spriteY+spriteHeight {
			p.oamSprites = append(p.oamSprites, spriteEntry{
				x:     x,
				y:     y,
				tile:  mmu.ReadOAMDirect(base + 2),
				attrs: mmu.ReadOAMDirect(base + 3),
			})
		}
	}

	// Sort by X ascending (lower X = higher priority).
	// Stable sort preserves OAM order for equal X (first in OAM = higher priority).
	for i := 1; i < len(p.oamSprites); i++ {
		for j := i; j > 0 && p.oamSprites[j].x < p.oamSprites[j-1].x; j-- {
			p.oamSprites[j], p.oamSprites[j-1] = p.oamSprites[j-1], p.oamSprites[j]
		}
	}
}

// renderSprites composites selected sprites onto the current scanline.
// Called during mode 3 (VRAM draw) after background & window rendering.
func (p *PPUCore) renderSprites() {
	spriteHeight := 8
	if p.lcdc&lcdcBitOBJSize != 0 {
		spriteHeight = 16
	}

	ly := int(p.ly)

	for i := len(p.oamSprites) - 1; i >= 0; i-- {
		s := p.oamSprites[i]
		spriteX := int(s.x) - 8
		spriteY := int(s.y) - 16

		// Iterate over screen X pixels this sprite covers.
		for screenX := spriteX; screenX < spriteX+8 && screenX < 160; screenX++ {
			if screenX < 0 {
				continue
			}

			spritePixelX := screenX - spriteX // 0-7 within sprite

			// Apply X flip.
			if s.attrs&spriteAttrXFlip != 0 {
				spritePixelX = 7 - spritePixelX
			}

			// Calculate row within the sprite (0 to spriteHeight-1).
			spriteRow := ly - spriteY

			// Apply Y flip.
			if s.attrs&spriteAttrYFlip != 0 {
				spriteRow = (spriteHeight - 1) - spriteRow
			}

			// Determine tile index and row within that tile.
			tileIndex := s.tile
			rowInTile := spriteRow
			if spriteHeight == 16 {
				tileIndex &^= 0x01 // Force bit 0 to 0 (even tile index)
				if spriteRow >= 8 {
					tileIndex |= 0x01 // Lower half uses tile+1
					rowInTile = spriteRow - 8
				}
			}

			// Fetch pixel value from tile data using direct VRAM read.
			bus := p.mmu.(*MemoryBus)
			tileAddr := uint16(0x8000) + uint16(tileIndex)*16 + uint16(rowInTile)*2
			lo := bus.ReadVRAMDirect(tileAddr)
			hi := bus.ReadVRAMDirect(tileAddr + 1)
			pixel := ((hi>>(7-uint(spritePixelX)))&1)<<1 | ((lo >> (7 - uint(spritePixelX))) & 1)

			if pixel == 0 {
				continue // Transparent pixel
			}

			// Apply palette (OBP0 or OBP1).
			var palette byte
			if s.attrs&spriteAttrPaletteDMG != 0 {
				palette = p.obp1
			} else {
				palette = p.obp0
			}
			mappedPixel := p.applyPalette(pixel, palette)

			// Priority check: if bit 7 set AND BG pixel is non-zero, BG wins.
			bgPixel := p.screen[screenX][ly]
			if s.attrs&spriteAttrPriority != 0 && bgPixel != 0 {
				continue
			}

			p.screen[screenX][ly] = mappedPixel
		}
	}
}

// GetMode returns the current PPU mode (0-3) as defined by the ppuMode* constants.
// This is extracted from the STAT register bits 0-1 for direct use by the MMU
// to enforce OAM/VRAM access blocking.
func (p *PPUCore) GetMode() int {
	return int(p.stat & 0x03)
}

// GetOAMRow returns the OAM row index (0-19) that the PPU is currently
// scanning during mode 2 (OAM search). During mode 2, the PPU reads one
// OAM row every M-cycle (4 T-cycles). Row 0 covers objects 0-1 at
// 0xFE00-0xFE07; row 19 covers objects 38-39 at 0xFE98-0xFE9F.
// Only meaningful during ppuModeOAM; returns a stale value otherwise.
func (p *PPUCore) GetOAMRow() int {
	return p.dotCounter / 4
}

func (p *PPUCore) GetState() PPUState {
	mode := p.GetMode()
	return PPUState{
		Mode:       mode,
		LY:         p.ly,
		LCDC:       p.lcdc,
		STAT:       p.stat,
		FrameCount: p.frameCtr,
	}
}

func (p *PPUCore) GetScreen() [160][144]byte {
	return p.screen
}

func (p *PPUCore) SetLYC(val byte) {
	p.lyc = val
	p.updateMode()
}

func (p *PPUCore) ReadRegister(addr uint16) byte {
	switch addr {
	case 0xFF40:
		return p.lcdc
	case 0xFF41:
		return p.stat | 0x80
	case 0xFF42:
		return p.scy
	case 0xFF43:
		return p.scx
	case 0xFF44:
		return p.ly
	case 0xFF45:
		return p.lyc
	case 0xFF47:
		return p.bgp
	case 0xFF48:
		return p.obp0
	case 0xFF49:
		return p.obp1
	case 0xFF4A:
		return p.wy
	case 0xFF4B:
		return p.wx
	default:
		return 0xFF
	}
}

func (p *PPUCore) WriteRegister(addr uint16, val byte) {
	switch addr {
	case 0xFF40:
		oldEnable := (p.lcdc & lcdcBitLCDEnable) != 0
		newEnable := (val & lcdcBitLCDEnable) != 0
		p.lcdc = val
		p.isRunning = newEnable
		if !p.isRunning {
			p.ly = 0
			p.dotCounter = 0
			p.stat &^= 0x03
			p.scanlineRendered = false
			p.firstFrameBlank = false
		} else if !oldEnable && newEnable {
			// LCD just turned on — first frame must be blank.
			p.firstFrameBlank = true
			// Clear screen to white (palette index 0).
			for x := 0; x < 160; x++ {
				for y := 0; y < 144; y++ {
					p.screen[x][y] = 0
				}
			}
		}
	case 0xFF41:
		p.stat = (p.stat & 0x07) | (val & 0x78)
		// Writing to STAT can immediately trigger a STAT interrupt if the newly
		// enabled interrupt conditions match the current PPU state (LY==LYC for
		// bit 6, or the current mode matches the mode-specific enable bits 3-5).
		// This is documented DMG/LR35902 behavior.
		if p.mmu != nil {
			if p.stat&statBitIntLYC != 0 && p.ly == p.lyc {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
			if p.stat&statBitIntMode2 != 0 && p.GetMode() == ppuModeOAM {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
			if p.stat&statBitIntMode1 != 0 && p.GetMode() == ppuModeVBlank {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
			if p.stat&statBitIntMode0 != 0 && p.GetMode() == ppuModeHBlank {
				p.mmu.WriteIF(p.mmu.ReadIF() | 0x02)
			}
		}
	case 0xFF42:
		p.scy = val
	case 0xFF43:
		p.scx = val
	case 0xFF45:
		p.lyc = val
		p.updateMode()
	case 0xFF47:
		p.bgp = val
	case 0xFF48:
		p.obp0 = val
	case 0xFF49:
		p.obp1 = val
	case 0xFF4A:
		p.wy = val
	case 0xFF4B:
		p.wx = val
	}
}
