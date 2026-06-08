package gb

// Cartridge represents the ROM cartridge and MBC (Memory Bank Controller).
type Cartridge interface {
	Read(addr uint16) byte
	Write(addr uint16, val byte)
	GetTitle() string
	GetType() CartridgeType
	HasBattery() bool
	SaveRAM() []byte
	LoadRAM(data []byte)
}

// CartridgeType identifies the MBC type from the cartridge header byte at 0x0147.
type CartridgeType byte

const (
	CartridgeROMOnly          CartridgeType = 0x00
	CartridgeMBC1             CartridgeType = 0x01
	CartridgeMBC1RAM          CartridgeType = 0x02
	CartridgeMBC1RAMBattery   CartridgeType = 0x03
	CartridgeMBC3             CartridgeType = 0x11
	CartridgeMBC3RAMBattery   CartridgeType = 0x13
	CartridgeMBC5             CartridgeType = 0x1A
	CartridgeMBC5RAM          CartridgeType = 0x1B
	CartridgeMBC5RAMBattery   CartridgeType = 0x1E
)

// NewCartridge creates the appropriate cartridge type based on the ROM header.
func NewCartridge(romData []byte) Cartridge {
	if len(romData) < 0x150 {
		return &romOnlyCartridge{data: romData}
	}

	cType := CartridgeType(romData[0x147])
	switch cType {
	case CartridgeMBC1, CartridgeMBC1RAM, CartridgeMBC1RAMBattery:
		return newMBC1(romData, cType)
	default:
		return &romOnlyCartridge{data: romData}
	}
}

// romBanksFromCode decodes the ROM size byte at 0x0148 into the number of 16KB banks.
//
//	$00=32K($02=8, $03=16, $04=32, $05=64, $06=128
func romBanksFromCode(code byte) int {
	switch {
	case code <= 0x06:
		return 2 << code
	case code == 0x52:
		return 72
	case code == 0x53:
		return 80
	case code == 0x54:
		return 96
	default:
		return 2
	}
}

// ramBanksFromCode decodes the RAM size byte at 0x0149 into the number of 8KB banks.
//
//	$00=0, $02=8K(1bank), $03=32K(4banks), $04=128K(16banks), $05=64K(8banks)
func ramBanksFromCode(code byte) int {
	switch code {
	case 0x00:
		return 0
	case 0x01:
		return 0 // unused (some PD ROMs misuse this)
	case 0x02:
		return 1 // 8 KiB
	case 0x03:
		return 4 // 32 KiB
	case 0x04:
		return 16 // 128 KiB
	case 0x05:
		return 8 // 64 KiB
	default:
		return 0
	}
}

// ---------------------------------------------------------------------------
// MBC1 implementation
// ---------------------------------------------------------------------------

// mbc1Cartridge implements MBC1 memory banking.
// Supports up to 2MB ROM and/or 32KB RAM.
type mbc1Cartridge struct {
	data       []byte   // Full ROM contents
	ram        [][]byte  // RAM banks (each 8KB)
	ramEnabled bool
	romBank    byte // 5-bit ROM bank register (write to 2000-3FFF)
	ramBankReg byte // 2-bit register (write to 4000-5FFF)
	mode       byte // Banking mode (write to 6000-7FFF): 0=simple, 1=advanced
	romBanks   int  // Number of 16KB ROM banks
	ramBanks   int  // Number of 8KB RAM banks
	hasBattery bool
}

func newMBC1(romData []byte, cType CartridgeType) *mbc1Cartridge {
	romSizeCode := romData[0x148]
	ramSizeCode := romData[0x149]

	nBanks := romBanksFromCode(romSizeCode)
	nRAM := ramBanksFromCode(ramSizeCode)

	ram := make([][]byte, nRAM)
	for i := 0; i < nRAM; i++ {
		ram[i] = make([]byte, 0x2000) // 8KB per bank
	}

	hasBattery := (cType == CartridgeMBC1RAMBattery)

	return &mbc1Cartridge{
		data:       romData,
		ram:        ram,
		romBanks:   nBanks,
		ramBanks:   nRAM,
		hasBattery: hasBattery,
	}
}

// getROMBank4000 returns the effective ROM bank number for the 4000-7FFF region.
func (c *mbc1Cartridge) getROMBank4000() int {
	lower := int(c.romBank)
	if lower == 0 {
		lower = 1
	}

	if c.romBanks > 32 {
		// Large ROM (>512KB): 2-bit register extends the upper bits
		return ((int(c.ramBankReg) << 5) | lower) % c.romBanks
	}

	// Small ROM: just mask the lower bits to available banks
	mask := c.romBanks - 1
	return lower & mask
}

// getROMBank0000 returns the effective ROM bank number for the 0000-3FFF region.
func (c *mbc1Cartridge) getROMBank0000() int {
	if c.mode == 0 || c.romBanks <= 32 {
		return 0
	}
	// Mode 1 with large ROM: bank is determined by the 2-bit register
	bank := int(c.ramBankReg) << 5
	if bank >= c.romBanks {
		bank = 0
	}
	return bank
}

// getRAMBank returns the effective RAM bank number for the A000-BFFF region.
func (c *mbc1Cartridge) getRAMBank() int {
	if c.mode == 0 || c.ramBanks <= 1 {
		return 0
	}
	bank := int(c.ramBankReg) % c.ramBanks
	if bank >= len(c.ram) {
		return 0
	}
	return bank
}

// Read reads from the cartridge address space.
func (c *mbc1Cartridge) Read(addr uint16) byte {
	switch {
	case addr <= 0x3FFF:
		bank := c.getROMBank0000()
		offset := bank*0x4000 + int(addr)
		if offset < len(c.data) {
			return c.data[offset]
		}
		return 0xFF

	case addr <= 0x7FFF:
		bank := c.getROMBank4000()
		offset := bank*0x4000 + int(addr-0x4000)
		if offset < len(c.data) {
			return c.data[offset]
		}
		return 0xFF

	case addr >= 0xA000 && addr <= 0xBFFF:
		if !c.ramEnabled || c.ramBanks == 0 {
			return 0xFF // open bus
		}
		bank := c.getRAMBank()
		return c.ram[bank][addr-0xA000]

	default:
		return 0xFF
	}
}

// Write writes to the cartridge address space.
func (c *mbc1Cartridge) Write(addr uint16, val byte) {
	switch {
	case addr <= 0x1FFF:
		// RAM enable: writing 0x0A to low nibble enables
		c.ramEnabled = (val & 0x0F) == 0x0A

	case addr >= 0x2000 && addr <= 0x3FFF:
		// ROM bank number (lower 5 bits)
		c.romBank = val & 0x1F

	case addr >= 0x4000 && addr <= 0x5FFF:
		// RAM bank number OR upper bits of ROM bank number
		c.ramBankReg = val & 0x03

	case addr >= 0x6000 && addr <= 0x7FFF:
		// Banking mode select
		c.mode = val & 0x01

	case addr >= 0xA000 && addr <= 0xBFFF:
		if c.ramEnabled && c.ramBanks > 0 {
			bank := c.getRAMBank()
			c.ram[bank][addr-0xA000] = val
		}
	}
}

func (c *mbc1Cartridge) GetTitle() string {
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

func (c *mbc1Cartridge) GetType() CartridgeType {
	if len(c.data) < 0x148 {
		return CartridgeROMOnly
	}
	return CartridgeType(c.data[0x147])
}

func (c *mbc1Cartridge) HasBattery() bool {
	return c.hasBattery
}

// SaveRAM returns a serialised copy of all RAM banks for battery-backed saving.
func (c *mbc1Cartridge) SaveRAM() []byte {
	if len(c.ram) == 0 {
		return nil
	}
	total := len(c.ram) * 0x2000
	dst := make([]byte, total)
	for i, bank := range c.ram {
		copy(dst[i*0x2000:(i+1)*0x2000], bank[:])
	}
	return dst
}

// LoadRAM restores RAM data previously saved via SaveRAM.
func (c *mbc1Cartridge) LoadRAM(data []byte) {
	if len(data) == 0 || len(c.ram) == 0 {
		return
	}
	for i := 0; i < len(c.ram) && i*0x2000 < len(data); i++ {
		n := copy(c.ram[i][:], data[i*0x2000:])
		if n < 0x2000 {
			break
		}
	}
}
