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

	// PPU reference for register routing
	ppu *PPUCore
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

// Read reads a byte from the given memory address.
func (m *MemoryBus) Read(addr uint16) byte {
	switch {
	case addr <= 0x7FFF:
		// ROM area
		if m.Cartridge != nil {
			return m.Cartridge.Read(addr)
		}
		return 0xFF // open bus returns 0xFF
	case addr >= 0x8000 && addr <= 0x9FFF:
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
		return m.oam[addr-0xFE00]
	case addr >= 0xFEA0 && addr <= 0xFEFF:
		// Unusable area, returns 0x00
		return 0x00
	case addr >= 0xFF00 && addr <= 0xFF7F:
		// LCD registers (0xFF40-0xFF4B) are routed to the PPU
		if addr >= 0xFF40 && addr <= 0xFF4B && m.ppu != nil {
			return m.ppu.ReadRegister(addr)
		}
		return m.io[addr-0xFF00]
	case addr >= 0xFF80 && addr <= 0xFFFE:
		return m.hram[addr-0xFF80]
	case addr == 0xFFFF:
		return m.ie
	default:
		return 0xFF
	}
}

// Write writes a byte to the given memory address.
func (m *MemoryBus) Write(addr uint16, val byte) {
	switch {
	case addr <= 0x7FFF:
		if m.Cartridge != nil {
			m.Cartridge.Write(addr, val)
		}
	case addr >= 0x8000 && addr <= 0x9FFF:
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
		m.oam[addr-0xFE00] = val
	case addr >= 0xFEA0 && addr <= 0xFEFF:
		// Unusable area, ignored
	case addr >= 0xFF00 && addr <= 0xFF7F:
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
		// If no cartridge was provided, create a ROM-only one
		m.Cartridge = &romOnlyCartridge{data: data}
		return
	}
	// Otherwise the cartridge handles it
}

// LoadBootROM loads boot ROM data (not yet implemented).
func (m *MemoryBus) LoadBootROM(data []byte) {
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
