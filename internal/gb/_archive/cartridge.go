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
	CartridgeROMOnly  CartridgeType = 0x00
	CartridgeMBC1     CartridgeType = 0x01
	CartridgeMBC1RAM  CartridgeType = 0x02
	CartridgeMBC3     CartridgeType = 0x13
	CartridgeMBC5     CartridgeType = 0x1A
	CartridgeMBC5RAM  CartridgeType = 0x1B
)
