package gb

// MemoryBus implements the MMU interface with the full Game Boy memory map.
type MemoryBus struct {
	// Cartridge is the attached ROM cartridge.
	Cartridge Cartridge

	// Internal RAM regions
	vram  [0x2000]byte // 0x8000-0x9FFF
	wram  [0x2000]byte // 0xC000-0xDFFF
	oam   [0xA0]byte   // 0xFE00-0xFE9F
	hram  [0x7F]byte   // 0xFF80-0xFFFE
	io    [0x80]byte   // 0xFF00-0xFF7F IO registers

	// IE register at 0xFFFF
	ie byte

	// Serial port registers
	sb byte // 0xFF01 — Serial transfer data
	sc byte // 0xFF02 — Serial control

	// PPU reference for register routing
	ppu *PPUCore

	// Timer reference for register routing
	timer *Timer

	// Joypad reference for 0xFF00 register routing
	joypad *Joypad

	// APU reference for audio register routing (0xFF10-0xFF26, 0xFF30-0xFF3F)
	apu *APU

	// Boot ROM data and state
	bootROM        [256]byte
	bootROMEnabled bool

	// OAM DMA state (0xFF46)
	dmaActive    bool
	dmaSource    uint16 // source address base (val << 8)
	dmaRemaining int    // T-cycles remaining

	// Serial transfer state (0xFF01-0xFF02)
	serialCycles int  // T-cycles remaining in current serial transfer
	serialActive bool // true while serial transfer is in progress
}

// NewMMU creates a new MemoryBus with the given cartridge.
func NewMMU(cartridge Cartridge) *MemoryBus {
	return &MemoryBus{
		Cartridge: cartridge,
	}
}

// SetPPU attaches a PPU to the memory bus for register read/write routing.
func (m *MemoryBus) SetPPU(ppu *PPUCore) {
	m.ppu = ppu
}

// SetTimer attaches a Timer to the memory bus for register read/write routing.
func (m *MemoryBus) SetTimer(timer *Timer) {
	m.timer = timer
}

// SetJoypad attaches a Joypad handler to the memory bus for 0xFF00 read routing.
func (m *MemoryBus) SetJoypad(j *Joypad) {
	m.joypad = j
}

// SetAPU attaches an APU to the memory bus for register read/write routing
// at 0xFF10-0xFF26 and 0xFF30-0xFF3F.
func (m *MemoryBus) SetAPU(apu *APU) {
	m.apu = apu
}

// SetJoypadButtons sets the active-high button state on the joypad handler.
func (m *MemoryBus) SetJoypadButtons(buttons byte) {
	if m.joypad != nil {
		m.joypad.SetButtons(buttons)
	}
}

// Read reads a byte from the given memory address.
// During OAM DMA, only HRAM (0xFF80-0xFFFE) is accessible.
// All other accesses return 0xFF until DMA completes.
func (m *MemoryBus) Read(addr uint16) byte {
	// During OAM DMA, only HRAM is accessible; other reads return 0xFF.
	if m.dmaActive && (addr < 0xFF80 || addr > 0xFFFE) {
		return 0xFF
	}

	switch {
	case addr <= 0x7FFF:
		// Boot ROM overrides cartridge ROM at 0x0000-0x00FF when enabled
		if addr <= 0x00FF && m.bootROMEnabled {
			return m.bootROM[addr]
		}
		// ROM area
		if m.Cartridge != nil {
			return m.Cartridge.Read(addr)
		}
		return 0xFF // open bus returns 0xFF
	case addr >= 0x8000 && addr <= 0x9FFF:
		// VRAM is inaccessible during Mode 3 (VRAM draw / pixel transfer)
		if m.ppu != nil && m.ppu.GetMode() == ppuModeVRAM {
			return 0xFF
		}
		return m.vram[addr-0x8000]
	case addr >= 0xA000 && addr <= 0xBFFF:
		if m.Cartridge != nil {
			return m.Cartridge.Read(addr)
		}
		return 0xFF
	case addr >= 0xC000 && addr <= 0xDFFF:
		return m.wram[addr-0xC000]
	case addr >= 0xE000 && addr <= 0xFDFF:
		// Echo RAM: mirrors 0xC000-0xDDFF
		return m.wram[(addr-0xE000)&0x1FFF]
	case addr >= 0xFE00 && addr <= 0xFE9F:
		// OAM corruption bug during Mode 2 (OAM search) — DMG bus conflict.
		// When the CPU reads OAM while the PPU is scanning it in mode 2, the
		// OAM data becomes corrupted via the read corruption formula.
		// Row 0 (objects 0-1 at FE00-FE07) is NOT affected.
		if m.ppu != nil && m.ppu.GetMode() == ppuModeOAM {
			row := m.ppu.GetOAMRow()
			if row > 0 {
				applyOAMReadCorruption(m.oam[:], row)
				return m.oam[addr-0xFE00]
			}
			// Row 0 — no corruption, fall through to normal read.
		} else if m.ppu != nil && m.ppu.GetMode() == ppuModeVRAM {
			// OAM is inaccessible during Mode 3 (VRAM draw)
			return 0xFF
		}
		return m.oam[addr-0xFE00]
	case addr >= 0xFEA0 && addr <= 0xFEFF:
		// Unusable area, returns 0x00
		return 0x00
	case addr >= 0xFF00 && addr <= 0xFF7F:
		// Joypad register (0xFF00)
		if addr == 0xFF00 && m.joypad != nil {
			return m.joypad.ReadRegister()
		}
		// Serial registers (0xFF01-0xFF02)
		switch addr {
		case 0xFF01:
			return m.sb
		case 0xFF02:
			return m.sc
		}
		// Timer registers (0xFF04-0xFF07) are routed to the Timer
		if addr >= 0xFF04 && addr <= 0xFF07 && m.timer != nil {
			return m.timer.ReadRegister(addr)
		}
		// APU registers (0xFF10-0xFF26) and wave RAM (0xFF30-0xFF3F) are routed
		// to the APU
		if ((addr >= 0xFF10 && addr <= 0xFF26) || (addr >= 0xFF30 && addr <= 0xFF3F)) && m.apu != nil {
			return m.apu.ReadRegister(addr)
		}
		// LCD registers (0xFF40-0xFF4B) are routed to the PPU
		if addr >= 0xFF40 && addr <= 0xFF4B && m.ppu != nil {
			return m.ppu.ReadRegister(addr)
		}
		// Return 0xFF for unmapped IO registers (open bus behavior on DMG).
		// KEY1 (0xFF4D) is not mapped on DMG — games read it via the
		// speed-switch routine and expect 0xFF (bit 7 set) to avoid
		// entering the double-speed toggle (which uses STOP).
		v := m.io[addr-0xFF00]
		if v != 0 {
			return v
		}
		// For registers that were never written (still zero), check if
		// they are known implemented registers. Unmapped IO addresses
		// return 0xFF on DMG (open bus).
		switch addr {
		case 0xFF01, 0xFF02, 0xFF04, 0xFF05, 0xFF06, 0xFF07,
			0xFF0F,
			0xFF40, 0xFF41, 0xFF42, 0xFF43, 0xFF44, 0xFF45,
			0xFF46, 0xFF47, 0xFF48, 0xFF49, 0xFF4A, 0xFF4B:
			return 0x00
		default:
			return 0xFF // open bus — includes 0xFF4D (KEY1 on DMG)
		}
	case addr >= 0xFF80 && addr <= 0xFFFE:
		return m.hram[addr-0xFF80]
	case addr == 0xFFFF:
		return m.ie
	default:
		return 0xFF
	}
}

// Write writes a byte to the given memory address.
// During OAM DMA, only HRAM (0xFF80-0xFFFE) is accessible.
func (m *MemoryBus) Write(addr uint16, val byte) {
	// During OAM DMA, only HRAM is accessible; other writes are ignored.
	if m.dmaActive && (addr < 0xFF80 || addr > 0xFFFE) {
		return
	}

	switch {
	case addr <= 0x7FFF:
		if m.Cartridge != nil {
			m.Cartridge.Write(addr, val)
		}
	case addr >= 0x8000 && addr <= 0x9FFF:
		// VRAM writes are blocked during Mode 3 (VRAM draw / pixel transfer)
		if m.ppu != nil && m.ppu.GetMode() == ppuModeVRAM {
			return
		}
		m.vram[addr-0x8000] = val
	case addr >= 0xA000 && addr <= 0xBFFF:
		if m.Cartridge != nil {
			m.Cartridge.Write(addr, val)
		}
	case addr >= 0xC000 && addr <= 0xDFFF:
		m.wram[addr-0xC000] = val
	case addr >= 0xE000 && addr <= 0xFDFF:
		// Echo RAM: mirrors to 0xC000-0xDDFF
		m.wram[(addr-0xE000)&0x1FFF] = val
	case addr >= 0xFE00 && addr <= 0xFE9F:
		if m.ppu != nil && m.ppu.GetMode() == ppuModeOAM {
			// OAM write corruption during Mode 2 (DMG bus conflict bug).
			// The write value is lost in the bus conflict, and the OAM data
			// in the PPU's current row is corrupted via the write formula.
			// Row 0 (objects 0-1 at FE00-FE07) is NOT affected.
			row := m.ppu.GetOAMRow()
			if row > 0 {
				applyOAMWriteCorruption(m.oam[:], row)
				return
			}
			// Row 0 — no corruption, fall through to normal write.
		} else if m.ppu != nil && m.ppu.GetMode() == ppuModeVRAM {
			// OAM writes are blocked during Mode 3 (VRAM draw)
			return
		}
		m.oam[addr-0xFE00] = val
	case addr >= 0xFEA0 && addr <= 0xFEFF:
		// Unusable area, ignored
	case addr == 0xFF46:
		// OAM DMA trigger: copy 160 bytes from (val << 8) to OAM (0xFE00-0xFE9F)
		// DMA takes 160 T-cycles (1 byte/cycle). During DMA only HRAM is accessible.
		m.startDMA(val)
	case addr >= 0xFF00 && addr <= 0xFF7F:
		// 0xFF00 is the JOYP (P1) register — route writes to Joypad
		if addr == 0xFF00 && m.joypad != nil {
			m.joypad.WriteRegister(val)
			return
		}
		// Serial registers (0xFF01-0xFF02)
		switch addr {
		case 0xFF01:
			m.sb = val
			return
		case 0xFF02:
			m.sc = val
			// Start serial transfer when bit 7=1 and bit 0=1 (master mode)
			if val&0x81 == 0x81 {
				m.serialActive = true
				m.serialCycles = 4096
			}
			return
		}
		// 0xFF50 is the BOOT register — writing bit 0 disables boot ROM
		if addr == 0xFF50 && val&0x01 != 0 {
			m.bootROMEnabled = false
		}
		// Timer registers (0xFF04-0xFF07) are routed to the Timer
		if addr >= 0xFF04 && addr <= 0xFF07 && m.timer != nil {
			m.timer.WriteRegister(addr, val)
			return
		}
		// APU registers (0xFF10-0xFF26) and wave RAM (0xFF30-0xFF3F) are routed
		// to the APU
		if ((addr >= 0xFF10 && addr <= 0xFF26) || (addr >= 0xFF30 && addr <= 0xFF3F)) && m.apu != nil {
			m.apu.WriteRegister(addr, val)
			return
		}
		// LCD registers (0xFF40-0xFF4B) are routed to the PPU
		if addr >= 0xFF40 && addr <= 0xFF4B && m.ppu != nil {
			m.ppu.WriteRegister(addr, val)
			return
		}
		m.io[addr-0xFF00] = val
	case addr >= 0xFF80 && addr <= 0xFFFE:
		m.hram[addr-0xFF80] = val
	case addr == 0xFFFF:
		m.ie = val
	}
}

// startDMA initiates an OAM DMA transfer from the given source register.
// The source address is (val << 8), and 160 bytes are copied to OAM (0xFE00-0xFE9F).
// DMA takes 160 T-cycles (1 byte/cycle). During this period only HRAM is accessible.
func (m *MemoryBus) startDMA(val byte) {
	src := uint16(val) << 8
	m.dmaSource = src
	m.dmaActive = true
	m.dmaRemaining = 160
}

// DMAStep advances the OAM DMA transfer by the given number of T-cycles.
func (m *MemoryBus) DMAStep(cycles int) {
	if !m.dmaActive {
		return
	}

	for cycles > 0 && m.dmaRemaining > 0 {
		// Read source byte directly (not through m.Read, which restricts
		// access during DMA). The DMA controller has its own bus master
		// port and can access arbitrary memory during OAM DMA.
		srcAddr := m.dmaSource + uint16(160-m.dmaRemaining)
		val := m.dmaRead(srcAddr)
		m.oam[160-m.dmaRemaining] = val
		m.dmaRemaining--
		cycles--
	}

	if m.dmaRemaining <= 0 {
		m.dmaActive = false
	}
}

// dmaRead reads a byte during OAM DMA, bypassing the dmaActive restriction.
// The DMA controller has its own bus master that can access any address.
func (m *MemoryBus) dmaRead(addr uint16) byte {
	if m.Cartridge != nil && addr <= 0x7FFF {
		return m.Cartridge.Read(addr)
	}
	if addr >= 0x8000 && addr <= 0x9FFF {
		return m.vram[addr-0x8000]
	}
	if addr >= 0xA000 && addr <= 0xBFFF && m.Cartridge != nil {
		return m.Cartridge.Read(addr)
	}
	if addr >= 0xC000 && addr <= 0xDFFF {
		return m.wram[addr-0xC000]
	}
	if addr >= 0xE000 && addr <= 0xFDFF {
		return m.wram[(addr-0xE000)&0x1FFF]
	}
	return 0xFF
}

// — OAM corruption helpers (DMG PPU mode 2 bus conflict) —

// oamRowOffset returns the OAM array offset for the first byte of the given row (0-19).
// Row 0 = FE00-FE07, row 1 = FE08-FE0F, ..., row 19 = FE98-FE9F.
func oamRowOffset(row int) int {
	return row * 8
}

// applyOAMReadCorruption applies the OAM read corruption formula to OAM row R
// (R > 0, rows 1-19). Row 0 (objects 0-1 at FE00-FE07) is NOT corrupted.
//
// Read corruption first-word formula: b | (a & c)
//   a = original value of the first 16-bit word in row R
//   b = first 16-bit word in the preceding row (R-1)
//   c = third 16-bit word in the preceding row (R-1)
//
// The remaining three 16-bit words in row R are overwritten with the
// corresponding words from the preceding row (R-1).
func applyOAMReadCorruption(oam []byte, row int) {
	off := oamRowOffset(row)
	prevOff := oamRowOffset(row - 1)

	// Read 16-bit words (little-endian)
	a := uint16(oam[off]) | uint16(oam[off+1])<<8
	b := uint16(oam[prevOff]) | uint16(oam[prevOff+1])<<8
	c := uint16(oam[prevOff+4]) | uint16(oam[prevOff+5])<<8

	// Corrupted first word: b | (a & c)
	corrupted := b | (a & c)
	oam[off] = byte(corrupted)
	oam[off+1] = byte(corrupted >> 8)

	// Copy last three words from preceding row (bytes 2-7)
	copy(oam[off+2:off+8], oam[prevOff+2:prevOff+8])
}

// applyOAMWriteCorruption applies the OAM write corruption formula to OAM row R
// (R > 0, rows 1-19). Row 0 (objects 0-1 at FE00-FE07) is NOT corrupted.
//
// Write corruption first-word formula: ((a ^ c) & (b ^ c)) ^ c
//   a = original value of the first 16-bit word in row R
//   b = first 16-bit word in the preceding row (R-1)
//   c = third 16-bit word in the preceding row (R-1)
//
// The remaining three 16-bit words in row R are overwritten with the
// corresponding words from the preceding row (R-1).
func applyOAMWriteCorruption(oam []byte, row int) {
	off := oamRowOffset(row)
	prevOff := oamRowOffset(row - 1)

	a := uint16(oam[off]) | uint16(oam[off+1])<<8
	b := uint16(oam[prevOff]) | uint16(oam[prevOff+1])<<8
	c := uint16(oam[prevOff+4]) | uint16(oam[prevOff+5])<<8

	corrupted := ((a ^ c) & (b ^ c)) ^ c
	oam[off] = byte(corrupted)
	oam[off+1] = byte(corrupted >> 8)

	copy(oam[off+2:off+8], oam[prevOff+2:prevOff+8])
}

// ReadOAMDirect reads a byte from OAM bypassing mode/blocking checks.
// Used by the PPU's scanOAM to read sprite data without triggering
// the CPU OAM corruption or blocking logic.
func (m *MemoryBus) ReadOAMDirect(addr uint16) byte {
	return m.oam[addr-0xFE00]
}

// WriteOAMDirect writes a byte to OAM bypassing mode/blocking checks.
// Used by the PPU when it needs direct OAM access without triggering
// corruption or blocking logic.
func (m *MemoryBus) WriteOAMDirect(addr uint16, val byte) {
	m.oam[addr-0xFE00] = val
}

// SerialStep advances the serial transfer by the given number of T-cycles.
// In master mode, a transfer takes 4096 T-cycles to shift 8 bits serially.
// On completion: SB is shifted right by 1 (bit 0 transmitted), bit 7 is set
// to 1 (no external device = SI pull-up), SC bit 7 is cleared, and a serial
// interrupt (IF bit 3) is requested.
func (m *MemoryBus) SerialStep(cycles int) {
	if !m.serialActive {
		return
	}
	m.serialCycles -= cycles
	if m.serialCycles > 0 {
		return
	}

	// Transfer complete
	// Shift SB right by 1, shift in 1 (no external device = SI pulled high)
	m.sb = (m.sb >> 1) | 0x80
	// Clear transfer active flag (SC bit 7)
	m.sc &^= 0x80
	// Request serial interrupt (IF bit 3)
	m.io[0x0F] |= 0x08
	m.serialActive = false
}

// ReadIF returns the IF (interrupt flag) register value at 0xFF0F.
func (m *MemoryBus) ReadIF() byte {
	return m.io[0x0F]
}

// WriteIF sets the IF (interrupt flag) register value at 0xFF0F.
func (m *MemoryBus) WriteIF(val byte) {
	m.io[0x0F] = val
}

// ReadIE returns the IE (interrupt enable) register value at 0xFFFF.
func (m *MemoryBus) ReadIE() byte {
	return m.ie
}

// WriteIE sets the IE (interrupt enable) register value at 0xFFFF.
func (m *MemoryBus) WriteIE(val byte) {
	m.ie = val
}

// ReadSB returns the serial transfer data register (0xFF01).
func (m *MemoryBus) ReadSB() byte {
	return m.sb
}

// WriteSB sets the serial transfer data register (0xFF01).
func (m *MemoryBus) WriteSB(val byte) {
	m.sb = val
}

// ReadSC returns the serial transfer control register (0xFF02).
func (m *MemoryBus) ReadSC() byte {
	return m.sc
}

// WriteSC sets the serial transfer control register (0xFF02).
func (m *MemoryBus) WriteSC(val byte) {
	m.sc = val
}

// Read16 reads two bytes (little-endian) starting at addr.
func (m *MemoryBus) Read16(addr uint16) uint16 {
	return uint16(m.Read(addr)) | uint16(m.Read(addr+1))<<8
}

// Write16 writes two bytes (little-endian) starting at addr.
func (m *MemoryBus) Write16(addr uint16, val uint16) {
	m.Write(addr, byte(val))
	m.Write(addr+1, byte(val>>8))
}

// LoadROM loads cartridge ROM data.
func (m *MemoryBus) LoadROM(data []byte) {
	if m.Cartridge == nil {
		// If no cartridge was provided, create one based on the header
		m.Cartridge = NewCartridge(data)
		return
	}
	// Otherwise the cartridge handles it
}

// LoadBootROM loads boot ROM data and enables boot ROM mapping at 0x0000-0x00FF.
func (m *MemoryBus) LoadBootROM(data []byte) {
	if len(data) > 256 {
		data = data[:256]
	}
	copy(m.bootROM[:], data)
	m.bootROMEnabled = true
}

// romOnlyCartridge is a minimal cartridge that maps ROM at 0x0000-0x7FFF.
type romOnlyCartridge struct {
	data []byte
}

func (c *romOnlyCartridge) Read(addr uint16) byte {
	if int(addr) < len(c.data) {
		return c.data[addr]
	}
	return 0xFF
}

func (c *romOnlyCartridge) Write(addr uint16, val byte) {
	// ROM only, no writes
}

func (c *romOnlyCartridge) GetTitle() string {
	if len(c.data) < 0x150 {
		return ""
	}
	var title []byte
	for i := 0x134; i < 0x143 && i < len(c.data); i++ {
		if c.data[i] == 0 {
			break
		}
		title = append(title, c.data[i])
	}
	return string(title)
}

func (c *romOnlyCartridge) GetType() CartridgeType {
	if len(c.data) < 0x148 {
		return CartridgeROMOnly
	}
	return CartridgeType(c.data[0x147])
}

func (c *romOnlyCartridge) HasBattery() bool {
	return false
}

func (c *romOnlyCartridge) SaveRAM() []byte {
	return nil
}

func (c *romOnlyCartridge) LoadRAM(data []byte) {
}

func (c *romOnlyCartridge) TickRTC(seconds int64) {
	// ROM-only cartridge has no RTC
}
