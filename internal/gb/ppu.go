package gb

// PPU represents the Picture Processing Unit (160x144 LCD).
type PPU interface {
	Step(cycles int)
	Reset()
	GetScreen() [160][144]byte // returns pixel data (4 shades indexed 0-3)
	GetState() PPUState
}

// PPUState is a snapshot of PPU registers and timing.
type PPUState struct {
	Mode       int  // 0=HBlank, 1=VBlank, 2=OAM, 3=VRAM
	LY         byte // current scanline
	LCDC       byte // LCD control register
	STAT       byte // LCD status register
	FrameCount int
}

// PPU timing constants (in T-cycles/dots).
const (
	oamSearchCycles  = 80
	vramDrawCycles   = 172 // minimum (no sprites)
	hBlankCycles     = 204 // 456 - 80 - 172
	scanlineCycles   = 456
	visibleScanlines = 144
	vBlankScanlines  = 10
	totalScanlines   = 154
	frameCycles      = 70224 // 154 * 456
)

// PPU modes (STAT register bits 1-0).
const (
	ppuModeHBlank = iota // 0
	ppuModeVBlank        // 1
	ppuModeOAM           // 2
	ppuModeVRAM          // 3
)

// STAT register bit positions.
const (
	statBitPPUMode0 = 1 << 0
	statBitPPUMode1 = 1 << 1
	statBitLYCoinc  = 1 << 2 // LYC == LY flag
	statBitIntMode0 = 1 << 3 // HBlank STAT interrupt select
	statBitIntMode1 = 1 << 4 // VBlank STAT interrupt select
	statBitIntMode2 = 1 << 5 // OAM STAT interrupt select
	statBitIntLYC   = 1 << 6 // LYC STAT interrupt select
)

// PPUCore is the concrete PPU timing state machine.
type PPUCore struct {
	// Timing
	dotCounter int // cycles consumed within the current scanline (0-455)

	// Registers
	ly        byte   // LY register (0xFF44) - current scanline
	lyc       byte   // LYC register (0xFF45) - LY compare
	stat      byte   // STAT register (0xFF41) - LCD status
	lcdc      byte   // LCDC register (0xFF40) - LCD control
	frameCtr  int    // frame counter (increments each frame)
	isRunning bool   // LCD is enabled (LCDC.7)

	// Screen buffer (160x144 pixels, 4 shades indexed 0-3)
	screen [160][144]byte

	// Mode transition cycle boundaries for the current scanline.
	// Precomputed for efficiency.
	mode2End int // end of OAM search (mode 2 → mode 3)
	mode3End int // end of VRAM draw (mode 3 → mode 0/hblank or vblank)
}

// NewPPU creates a new PPU instance.
func NewPPU() *PPUCore {
	ppu := &PPUCore{
		lcdc: 0x91, // default: LCD enabled, BG enabled, sprites 8x8, BG tile map 9800, etc.
	}
	ppu.Reset()
	return ppu
}

// Reset sets the PPU to its initial state.
func (p *PPUCore) Reset() {
	p.dotCounter = 0
	p.ly = 0
	p.stat &^= 0x07 // clear mode bits and LYC flag
	p.stat |= ppuModeOAM << 0 // mode = OAM search (2)
	p.isRunning = true
	p.frameCtr = 0

	// Recompute mode boundaries.
	p.mode2End = oamSearchCycles
	p.mode3End = oamSearchCycles + vramDrawCycles

	// Clear screen buffer.
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			p.screen[x][y] = 0
		}
	}
}

// Step advances the PPU by the given number of cycles.
func (p *PPUCore) Step(cycles int) {
	if !p.isRunning || cycles <= 0 {
		return
	}

	for cycles > 0 {
		// How many cycles until the next event in the current scanline?
		remaining := scanlineCycles - p.dotCounter
		step := cycles
		if step > remaining {
			step = remaining
		}

		p.dotCounter += step
		cycles -= step

		// Update mode if we crossed a boundary within this step.
		p.updateMode()

		// If we completed the scanline, advance to the next one.
		if p.dotCounter >= scanlineCycles {
			p.advanceScanline()
		}
	}
}

// advanceScanline moves to the next scanline (LY + 1) or wraps.
func (p *PPUCore) advanceScanline() {
	p.dotCounter = 0
	p.ly++

	if p.ly >= totalScanlines {
		p.ly = 0
		p.frameCtr++
	}

	// Recompute mode boundaries for VBlank vs visible scanlines.
	p.mode2End = oamSearchCycles
	p.mode3End = oamSearchCycles + vramDrawCycles

	// Update mode for the start of the new scanline.
	p.updateMode()
}

// updateMode sets the current PPU mode based on dotCounter and LY.
func (p *PPUCore) updateMode() {
	var mode int

	if p.ly < visibleScanlines {
		// Visible scanline: OAM → VRAM → HBlank
		if p.dotCounter < p.mode2End {
			mode = ppuModeOAM
		} else if p.dotCounter < p.mode3End {
			mode = ppuModeVRAM
		} else {
			mode = ppuModeHBlank
		}
	} else {
		// VBlank scanline: always mode 1
		mode = ppuModeVBlank
	}

	// Update STAT mode bits (bits 1-0).
	p.stat &^= 0x03 // clear mode bits
	p.stat |= byte(mode) & 0x03

	// Update LYC == LY coincidence flag (bit 2).
	if p.ly == p.lyc {
		p.stat |= statBitLYCoinc
	} else {
		p.stat &^= statBitLYCoinc
	}
}

// GetState returns a snapshot of current PPU state.
func (p *PPUCore) GetState() PPUState {
	mode := int(p.stat & 0x03)
	return PPUState{
		Mode:       mode,
		LY:         p.ly,
		LCDC:       p.lcdc,
		STAT:       p.stat,
		FrameCount: p.frameCtr,
	}
}

// GetScreen returns the current screen buffer.
func (p *PPUCore) GetScreen() [160][144]byte {
	return p.screen
}

// SetLYC sets the LYC compare register value.
// Exported for test access; will be wired through MMU later.
func (p *PPUCore) SetLYC(val byte) {
	p.lyc = val
	// Re-evaluate coincidence.
	p.updateMode()
}

// ReadRegister reads an LCD-related register by address.
// Used by the MMU. Returns 0xFF for unmapped addresses.
func (p *PPUCore) ReadRegister(addr uint16) byte {
	switch addr {
	case 0xFF40:
		return p.lcdc
	case 0xFF41:
		return p.stat | 0x80 // bit 7 always set per spec
	case 0xFF42:
		return 0 // SCY (scroll Y) - not implemented yet
	case 0xFF43:
		return 0 // SCX (scroll X) - not implemented yet
	case 0xFF44:
		return p.ly
	case 0xFF45:
		return p.lyc
	case 0xFF46:
		return 0 // DMA - not implemented yet
	case 0xFF47:
		return 0 // BGP - not implemented yet
	case 0xFF48:
		return 0 // OBP0 - not implemented yet
	case 0xFF49:
		return 0 // OBP1 - not implemented yet
	case 0xFF4A:
		return 0 // WY - not implemented yet
	case 0xFF4B:
		return 0 // WX - not implemented yet
	default:
		return 0xFF
	}
}

// WriteRegister writes an LCD-related register by address.
// Used by the MMU.
func (p *PPUCore) WriteRegister(addr uint16, val byte) {
	switch addr {
	case 0xFF40:
		p.lcdc = val
		// Bit 7 enables/disables the LCD.
		p.isRunning = (val & 0x80) != 0
		if !p.isRunning {
			p.ly = 0
			p.dotCounter = 0
			p.stat &^= 0x03
		}
	case 0xFF41:
		// Only writable bits are 6-3 (interrupt selects).
		// Bits 2-0 are read-only.
		p.stat = (p.stat & 0x07) | (val & 0x78)
	case 0xFF42:
	// SCY - not implemented
	case 0xFF43:
	// SCX - not implemented
	case 0xFF45:
		p.lyc = val
		p.updateMode()
	case 0xFF46:
	// OAM DMA - not implemented
	case 0xFF47:
	// BGP - not implemented
	case 0xFF48:
	// OBP0 - not implemented
	case 0xFF49:
	// OBP1 - not implemented
	case 0xFF4A:
	// WY - not implemented
	case 0xFF4B:
	// WX - not implemented
	}
}
