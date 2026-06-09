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
	// TickRTC advances the real-time clock by the given number of seconds.
	// Only MBC3 timer cartridges implement this; others are no-ops.
	TickRTC(seconds int64)
}

// CartridgeType identifies the MBC type from the cartridge header byte at 0x0147.
type CartridgeType byte

const (
	CartridgeROMOnly          CartridgeType = 0x00
	CartridgeMBC1             CartridgeType = 0x01
	CartridgeMBC1RAM          CartridgeType = 0x02
	CartridgeMBC1RAMBattery   CartridgeType = 0x03
	CartridgeMBC3TimerBattery      CartridgeType = 0x0F // MBC3 + Timer + Battery
	CartridgeMBC3TimerRAMBattery   CartridgeType = 0x10 // MBC3 + Timer + RAM + Battery
	CartridgeMBC3                  CartridgeType = 0x11 // MBC3
	CartridgeMBC3RAM               CartridgeType = 0x12 // MBC3 + RAM
	CartridgeMBC3RAMBattery        CartridgeType = 0x13 // MBC3 + RAM + Battery
	CartridgeMBC5             CartridgeType = 0x19
	CartridgeMBC5RAM          CartridgeType = 0x1A
	CartridgeMBC5RAMBattery   CartridgeType = 0x1B
	CartridgeMBC5Rumble       CartridgeType = 0x1C
	CartridgeMBC5RumbleRAM          CartridgeType = 0x1D
	CartridgeMBC5RumbleRAMBattery CartridgeType = 0x1E
	CartridgeMBC2              CartridgeType = 0x05
	CartridgeMBC2RAMBattery    CartridgeType = 0x06
)

// IsMBC3 returns true if the cartridge type uses the MBC3 banking scheme.
func (c CartridgeType) IsMBC3() bool {
	return c >= 0x0F && c <= 0x13
}

// IsMBC5 returns true if the cartridge type uses the MBC5 banking scheme.
func (c CartridgeType) IsMBC5() bool {
	switch c {
	case CartridgeMBC5, CartridgeMBC5RAM, CartridgeMBC5RAMBattery,
		CartridgeMBC5Rumble, CartridgeMBC5RumbleRAM, CartridgeMBC5RumbleRAMBattery:
		return true
	}
	return false
}

// HasRumble returns true if the cartridge type includes a rumble motor.
func (c CartridgeType) HasRumble() bool {
	return c == CartridgeMBC5Rumble || c == CartridgeMBC5RumbleRAM || c == CartridgeMBC5RumbleRAMBattery
}

// NewCartridge creates the appropriate cartridge type based on the ROM header.
func NewCartridge(romData []byte) Cartridge {
	if len(romData) < 0x150 {
		return &romOnlyCartridge{data: romData}
	}

	cType := CartridgeType(romData[0x147])
	switch {
	case cType == CartridgeMBC1, cType == CartridgeMBC1RAM, cType == CartridgeMBC1RAMBattery:
		return newMBC1(romData, cType)
	case cType == CartridgeMBC2, cType == CartridgeMBC2RAMBattery:
		return newMBC2(romData, cType)
	case cType.IsMBC3():
		return newMBC3(romData, cType)
	case cType.IsMBC5():
		return newMBC5(romData, cType)
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
	ram        [][]byte // RAM banks (each 8KB)
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

// No-op TickRTC: MBC1 has no RTC.
func (c *mbc1Cartridge) TickRTC(seconds int64) {}

// ---------------------------------------------------------------------------
// MBC2 implementation
// ---------------------------------------------------------------------------

// mbc2Cartridge implements MBC2 memory banking.
//
// MBC2 has a unique design: ROM bank select and RAM enable are both controlled
// through the 0x0000-0x3FFF range, distinguished by bit 8 of the address.
// Built-in 512×4-bit RAM (no external RAM chip). Supports up to 256KB ROM
// (16 banks).
type mbc2Cartridge struct {
	data       []byte    // Full ROM contents
	ram        [0x200]byte // Internal 512×4-bit RAM (lower nibble only)
	ramEnabled bool
	romBank    byte // 4-bit ROM bank register (0→1 mapped)
	romBanks   int  // Number of 16KB ROM banks
	hasBattery bool
}

func newMBC2(romData []byte, cType CartridgeType) *mbc2Cartridge {
	romSizeCode := romData[0x148]
	nBanks := romBanksFromCode(romSizeCode)
	if nBanks > 16 {
		nBanks = 16
	}
	hasBattery := (cType == CartridgeMBC2RAMBattery)

	return &mbc2Cartridge{
		data:       romData,
		romBanks:   nBanks,
		hasBattery: hasBattery,
		romBank:    1, // Default to bank 1 on startup
	}
}

// getROMBank returns the effective ROM bank number for the 0x4000-0x7FFF region.
func (c *mbc2Cartridge) getROMBank() int {
	bank := int(c.romBank & 0x0F)
	if bank == 0 {
		bank = 1
	}
	return bank % c.romBanks
}

// Read reads from the MBC2 cartridge address space.
func (c *mbc2Cartridge) Read(addr uint16) byte {
	switch {
	case addr <= 0x3FFF:
		if int(addr) < len(c.data) {
			return c.data[addr]
		}
		return 0xFF

	case addr <= 0x7FFF:
		bank := c.getROMBank()
		offset := bank*0x4000 + int(addr-0x4000)
		if offset < len(c.data) {
			return c.data[offset]
		}
		return 0xFF

	case addr >= 0xA000 && addr <= 0xBFFF:
		if !c.ramEnabled {
			return 0xFF
		}
		// Internal 512×4-bit RAM; bottom 9 address bits for linear mapping
		// and echo handling. Upper nibble returns 0xF.
		ramOffset := (addr - 0xA000) & 0x01FF
		return 0xF0 | (c.ram[ramOffset] & 0x0F)

	default:
		return 0xFF
	}
}

// Write writes to the MBC2 cartridge address space.
func (c *mbc2Cartridge) Write(addr uint16, val byte) {
	switch {
	case addr <= 0x3FFF:
		if addr&0x0100 == 0 {
			// Bit 8 clear → RAM enable: low nibble == 0x0A enables
			c.ramEnabled = (val & 0x0F) == 0x0A
		} else {
			// Bit 8 set → ROM bank select: lower 4 bits, 0→1
			bank := int(val & 0x0F)
			if bank == 0 {
				bank = 1
			}
			c.romBank = byte(bank % c.romBanks)
		}

	case addr >= 0xA000 && addr <= 0xBFFF:
		if c.ramEnabled {
			// Internal 512×4-bit RAM; bottom 9 address bits, lower nibble only
			ramOffset := (addr - 0xA000) & 0x01FF
			c.ram[ramOffset] = val & 0x0F
		}
	}
}

func (c *mbc2Cartridge) GetTitle() string {
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

func (c *mbc2Cartridge) GetType() CartridgeType {
	if len(c.data) < 0x148 {
		return CartridgeROMOnly
	}
	return CartridgeType(c.data[0x147])
}

func (c *mbc2Cartridge) HasBattery() bool {
	return c.hasBattery
}

// SaveRAM returns a serialised copy of internal RAM for battery-backed saving.
func (c *mbc2Cartridge) SaveRAM() []byte {
	if !c.hasBattery {
		return nil
	}
	dst := make([]byte, 0x200)
	copy(dst, c.ram[:])
	return dst
}

// LoadRAM restores internal RAM data previously saved via SaveRAM.
func (c *mbc2Cartridge) LoadRAM(data []byte) {
	if len(data) == 0 {
		return
	}
	n := copy(c.ram[:], data[:])
	_ = n
}

// No-op TickRTC: MBC2 has no RTC.
func (c *mbc2Cartridge) TickRTC(seconds int64) {}

// ---------------------------------------------------------------------------
// MBC3 implementation
// ---------------------------------------------------------------------------

// mbc3Cartridge implements MBC3 memory banking.
//
// MBC3 supports up to 2MB ROM (128 banks) and/or 32KB RAM (4 banks),
// plus an optional real-time clock (RTC) with 5 registers.
// ROM bank select uses a 7-bit register at 0x2000-0x3FFF (0 maps to bank 1).
// RAM bank / RTC register select at 0x4000-0x5FFF:
//
//	0x00-0x03 select RAM banks 0-3
//	0x08-0x0C select RTC registers (S, M, H, DL, DH)
//
// RTC latch: write 0x00 then 0x01 to 0x6000-0x7FFF to latch current time.
// RTC tick: call TickRTC(seconds) to advance time (halted when DH bit 6 set).
// Day counter is 9-bit (DL + DH bit 0); carry (DH bit 7) set on overflow.
//
// Used by Pokémon G/S/C, Pokémon Yellow (J), and many other late-DMG/GBC games.
type mbc3Cartridge struct {
	data       []byte    // Full ROM contents
	romBanks   int       // Number of 16KB ROM banks
	ram        [][]byte  // RAM banks (each 8KB)
	ramBanks   int       // Number of 8KB RAM banks
	ramEnabled bool
	romBank    byte      // 7-bit ROM bank register (0 → bank 1)
	ramBank    byte      // RAM bank select (0x00-0x03) or RTC register select (0x08-0x0C)
	rtcRegs    [5]byte   // RTC registers: S, M, H, DL, DH
	rtcLatched [5]byte   // Latched RTC register snapshot
	latchStep  int       // Latch sequence step (0=idle, 1=first write, 2=latched)
	rtcClock   int64     // RTC tick counter (in seconds)
	hasBattery bool
	hasTimer   bool      // true for types 0x0F and 0x10 (with RTC)
}

func newMBC3(romData []byte, cType CartridgeType) *mbc3Cartridge {
	romSizeCode := romData[0x148]
	ramSizeCode := romData[0x149]

	nBanks := romBanksFromCode(romSizeCode)
	nRAM := ramBanksFromCode(ramSizeCode)

	ram := make([][]byte, nRAM)
	for i := 0; i < nRAM; i++ {
		ram[i] = make([]byte, 0x2000) // 8KB per bank
	}

	hasBattery := (cType == CartridgeMBC3TimerBattery ||
		cType == CartridgeMBC3TimerRAMBattery ||
		cType == CartridgeMBC3RAMBattery)

	hasTimer := cType == CartridgeMBC3TimerBattery ||
		cType == CartridgeMBC3TimerRAMBattery

	return &mbc3Cartridge{
		data:       romData,
		ram:        ram,
		romBanks:   nBanks,
		ramBanks:   nRAM,
		hasBattery: hasBattery,
		hasTimer:   hasTimer,
	}
}

// getROMBank returns the effective ROM bank number for the 0x4000-0x7FFF region.
// MBC3 uses a 7-bit register; 0 maps to bank 1.
func (c *mbc3Cartridge) getROMBank() int {
	bank := int(c.romBank & 0x7F)
	if bank == 0 {
		bank = 1
	}
	return bank % c.romBanks
}

// hasRAMSelected returns true if the current ramBank value selects RAM (0x00-0x03).
func (c *mbc3Cartridge) hasRAMSelected() bool {
	return c.ramBank <= 0x03 && c.ramBanks > 0
}

// Read reads from the MBC3 cartridge address space.
func (c *mbc3Cartridge) Read(addr uint16) byte {
	switch {
	case addr <= 0x3FFF:
		// Bank 0 is always at 0x0000-0x3FFF
		if int(addr) < len(c.data) {
			return c.data[addr]
		}
		return 0xFF

	case addr <= 0x7FFF:
		bank := c.getROMBank()
		offset := bank*0x4000 + int(addr-0x4000)
		if offset < len(c.data) {
			return c.data[offset]
		}
		return 0xFF

	case addr >= 0xA000 && addr <= 0xBFFF:
		if !c.ramEnabled {
			return 0xFF
		}
		if c.hasRAMSelected() {
			bank := int(c.ramBank) % c.ramBanks
			if bank >= len(c.ram) {
				return 0xFF
			}
			return c.ram[bank][addr-0xA000]
		}
		// RTC register read (ramBank 0x08-0x0C)
		if c.hasTimer && c.ramBank >= 0x08 && c.ramBank <= 0x0C {
			reg := int(c.ramBank - 0x08)
			return c.rtcLatched[reg]
		}
		return 0xFF

	default:
		return 0xFF
	}
}

// Write writes to the MBC3 cartridge address space.
func (c *mbc3Cartridge) Write(addr uint16, val byte) {
	switch {
	case addr <= 0x1FFF:
		// RAM enable: writing 0x0A to low nibble enables
		c.ramEnabled = (val & 0x0F) == 0x0A

	case addr >= 0x2000 && addr <= 0x3FFF:
		// ROM bank number (7-bit, 0 maps to bank 1)
		c.romBank = val & 0x7F

	case addr >= 0x4000 && addr <= 0x5FFF:
		// RAM bank number (0x00-0x03) or RTC register select (0x08-0x0C)
		c.ramBank = val

	case addr >= 0x6000 && addr <= 0x7FFF:
		// RTC latch control
		if !c.hasTimer {
			return
		}
		if val == 0x00 {
			c.latchStep = 1
		} else if val == 0x01 && c.latchStep == 1 {
			c.latchTime()
			c.latchStep = 2
		} else {
			c.latchStep = 0
		}

	case addr >= 0xA000 && addr <= 0xBFFF:
		if !c.ramEnabled {
			return
		}
		if c.hasRAMSelected() {
			bank := int(c.ramBank) % c.ramBanks
			if bank < len(c.ram) {
				c.ram[bank][addr-0xA000] = val
			}
		} else if c.hasTimer && c.ramBank >= 0x08 && c.ramBank <= 0x0C {
			// RTC register write
			reg := int(c.ramBank - 0x08)
			if reg == 4 {
				// DH: writing clears carry (bit 7), sets day MSB (bit 0) and halt (bit 6)
				c.rtcRegs[4] = val & 0x41
			} else {
				c.rtcRegs[reg] = val
			}
			c.syncClockFromRegs()
		}
	}
}

// latchTime snapshots the current RTC register values into the latched buffer.
// This is triggered by the latch sequence (write 0x00 then 0x01 to 0x6000-0x7FFF).
func (c *mbc3Cartridge) latchTime() {
	c.rtcLatched = c.rtcRegs
}

// syncClockFromRegs recomputes rtcClock from the RTC register values.
// Called after writing to RTC registers.
func (c *mbc3Cartridge) syncClockFromRegs() {
	days := int(c.rtcRegs[3]) | (int(c.rtcRegs[4]&0x01) << 8)
	c.rtcClock = int64(days)*86400 +
		int64(c.rtcRegs[2])*3600 +
		int64(c.rtcRegs[1])*60 +
		int64(c.rtcRegs[0])
}

// TickRTC advances the RTC by the given number of seconds.
// If the RTC halt bit (DH bit 6) is set, the tick is ignored.
// Detects day counter overflow and sets the carry bit (DH bit 7).
func (c *mbc3Cartridge) TickRTC(seconds int64) {
	if !c.hasTimer || c.rtcRegs[4]&0x40 != 0 {
		return // halted or no timer hardware
	}

	// Track 9-bit day counter before advancing for carry detection
	oldDays9 := (c.rtcClock / 86400) & 0x1FF

	c.rtcClock += seconds

	// Detect 9-bit day counter overflow (wrapped past 512 days)
	newDays9 := (c.rtcClock / 86400) & 0x1FF
	if oldDays9 > newDays9 {
		c.rtcRegs[4] |= 0x80 // set carry
	}

	// Update registers from clock
	c.rtcRegs[0] = byte(c.rtcClock % 60)       // S
	c.rtcRegs[1] = byte((c.rtcClock / 60) % 60) // M
	c.rtcRegs[2] = byte((c.rtcClock / 3600) % 24) // H
	totalDays := c.rtcClock / 86400
	c.rtcRegs[3] = byte(totalDays & 0xFF)          // DL
	c.rtcRegs[4] = (c.rtcRegs[4] & 0xC0) | byte((totalDays>>8)&0x01) // DH
}

func (c *mbc3Cartridge) GetTitle() string {
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

func (c *mbc3Cartridge) GetType() CartridgeType {
	if len(c.data) < 0x148 {
		return CartridgeROMOnly
	}
	return CartridgeType(c.data[0x147])
}

func (c *mbc3Cartridge) HasBattery() bool {
	return c.hasBattery
}

// SaveRAM returns a serialised copy of all RAM banks and RTC state
// for battery-backed saving. RTC state (5 register bytes + 8 clock bytes)
// is appended after the RAM data when hasTimer is true.
func (c *mbc3Cartridge) SaveRAM() []byte {
	ramSize := len(c.ram) * 0x2000
	rtcSize := 0
	if c.hasTimer {
		rtcSize = 13 // 5 regs + 8 int64 clock
	}
	if ramSize == 0 && rtcSize == 0 {
		return nil
	}
	dst := make([]byte, ramSize+rtcSize)
	for i, bank := range c.ram {
		copy(dst[i*0x2000:(i+1)*0x2000], bank[:])
	}
	if c.hasTimer {
		copy(dst[ramSize:], c.rtcRegs[:])               // 5 bytes
		dst[ramSize+5] = byte(c.rtcClock)                // int64 little-endian
		dst[ramSize+6] = byte(c.rtcClock >> 8)
		dst[ramSize+7] = byte(c.rtcClock >> 16)
		dst[ramSize+8] = byte(c.rtcClock >> 24)
		dst[ramSize+9] = byte(c.rtcClock >> 32)
		dst[ramSize+10] = byte(c.rtcClock >> 40)
		dst[ramSize+11] = byte(c.rtcClock >> 48)
		dst[ramSize+12] = byte(c.rtcClock >> 56)
	}
	return dst
}

// LoadRAM restores RAM data and RTC state previously saved via SaveRAM.
func (c *mbc3Cartridge) LoadRAM(data []byte) {
	if len(data) == 0 {
		return
	}
	ramSize := len(c.ram) * 0x2000

	// Load RAM banks
	for i := 0; i < len(c.ram) && i*0x2000 < len(data); i++ {
		n := copy(c.ram[i][:], data[i*0x2000:])
		if n < 0x2000 {
			break
		}
	}

	// Load RTC state if present (13 bytes appended after RAM)
	if c.hasTimer && len(data) >= ramSize+13 {
		copy(c.rtcRegs[:], data[ramSize:ramSize+5])
		c.rtcClock = int64(data[ramSize+5]) |
			int64(data[ramSize+6])<<8 |
			int64(data[ramSize+7])<<16 |
			int64(data[ramSize+8])<<24 |
			int64(data[ramSize+9])<<32 |
			int64(data[ramSize+10])<<40 |
			int64(data[ramSize+11])<<48 |
			int64(data[ramSize+12])<<56
	}
}

// ---------------------------------------------------------------------------
// MBC5 implementation
// ---------------------------------------------------------------------------

// mbc5Cartridge implements MBC5 memory banking.
//
// MBC5 has a 9-bit ROM bank register (lower 8 bits at 0x2000-0x2FFF,
// bit 8 at 0x3000-0x3FFF) and a 4-bit RAM bank register at 0x4000-0x5FFF.
// On rumble carts, bit 3 of the RAM bank register controls the rumble motor
// and bits 0-2 select the RAM bank.
//
// Supports up to 8MB ROM (512 banks) and/or 128KB RAM (16 banks).
// Used by Pokémon R/B/Y, Pokémon G/S/C, and most late-DMG games.
type mbc5Cartridge struct {
	data        []byte   // Full ROM contents
	ram         [][]byte // RAM banks (each 8KB)
	ramEnabled  bool
	romBankLow  byte // Lower 8 bits of ROM bank (write to 2000-2FFF)
	romBankHigh byte // Bit 8 of ROM bank (write to 3000-3FFF, only bit 0 used)
	ramBankReg  byte // RAM bank register (write to 4000-5FFF, lower 4 bits)
	romBanks    int  // Number of 16KB ROM banks
	ramBanks    int  // Number of 8KB RAM banks
	hasBattery  bool
	hasRumble   bool
}

func newMBC5(romData []byte, cType CartridgeType) *mbc5Cartridge {
	romSizeCode := romData[0x148]
	ramSizeCode := romData[0x149]

	nBanks := romBanksFromCode(romSizeCode)
	nRAM := ramBanksFromCode(ramSizeCode)

	ram := make([][]byte, nRAM)
	for i := 0; i < nRAM; i++ {
		ram[i] = make([]byte, 0x2000) // 8KB per bank
	}

	hasBattery := (cType == CartridgeMBC5RAMBattery || cType == CartridgeMBC5RumbleRAMBattery)
	hasRumble := cType.HasRumble()

	return &mbc5Cartridge{
		data:       romData,
		ram:        ram,
		romBanks:   nBanks,
		ramBanks:   nRAM,
		hasBattery: hasBattery,
		hasRumble:  hasRumble,
	}
}

// getROMBank returns the effective ROM bank number for the 0x4000-0x7FFF region.
func (c *mbc5Cartridge) getROMBank() int {
	bank := (int(c.romBankHigh&0x01) << 8) | int(c.romBankLow)
	if bank == 0 {
		bank = 1
	}
	return bank % c.romBanks
}

// getRAMBank returns the effective RAM bank number for the 0xA000-0xBFFF region.
func (c *mbc5Cartridge) getRAMBank() int {
	bank := int(c.ramBankReg)
	if c.hasRumble {
		// Rumble carts: bits 0-2 select RAM bank, bit 3 controls rumble
		bank = int(c.ramBankReg & 0x07)
	} else {
		// Non-rumble: lower 4 bits select RAM bank
		bank = int(c.ramBankReg & 0x0F)
	}
	if c.ramBanks <= 1 {
		return 0
	}
	return bank % c.ramBanks
}

// Read reads from the MBC5 cartridge address space.
func (c *mbc5Cartridge) Read(addr uint16) byte {
	switch {
	case addr <= 0x3FFF:
		// Bank 0 is always at 0x0000-0x3FFF
		if int(addr) < len(c.data) {
			return c.data[addr]
		}
		return 0xFF

	case addr <= 0x7FFF:
		bank := c.getROMBank()
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
		if bank >= len(c.ram) {
			return 0xFF
		}
		return c.ram[bank][addr-0xA000]

	default:
		return 0xFF
	}
}

// Write writes to the MBC5 cartridge address space.
func (c *mbc5Cartridge) Write(addr uint16, val byte) {
	switch {
	case addr <= 0x1FFF:
		// RAM enable: writing 0x0A to low nibble enables
		c.ramEnabled = (val & 0x0F) == 0x0A

	case addr >= 0x2000 && addr <= 0x2FFF:
		// ROM bank lower 8 bits
		c.romBankLow = val

	case addr >= 0x3000 && addr <= 0x3FFF:
		// ROM bank bit 8 (bit 9 in the ROM bank number)
		c.romBankHigh = val & 0x01

	case addr >= 0x4000 && addr <= 0x5FFF:
		// RAM bank select (and rumble control for rumble carts)
		c.ramBankReg = val

	case addr >= 0xA000 && addr <= 0xBFFF:
		if c.ramEnabled && c.ramBanks > 0 {
			bank := c.getRAMBank()
			if bank < len(c.ram) {
				c.ram[bank][addr-0xA000] = val
			}
		}
	}
}

func (c *mbc5Cartridge) GetTitle() string {
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

func (c *mbc5Cartridge) GetType() CartridgeType {
	if len(c.data) < 0x148 {
		return CartridgeROMOnly
	}
	return CartridgeType(c.data[0x147])
}

func (c *mbc5Cartridge) HasBattery() bool {
	return c.hasBattery
}

// SaveRAM returns a serialised copy of all RAM banks for battery-backed saving.
func (c *mbc5Cartridge) SaveRAM() []byte {
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
func (c *mbc5Cartridge) LoadRAM(data []byte) {
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

// No-op TickRTC: MBC5 has no RTC.
func (c *mbc5Cartridge) TickRTC(seconds int64) {}
