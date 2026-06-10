package gb

// CPUState is a snapshot of CPU registers and flags.
type CPUState struct {
	AF, BC, DE, HL, SP, PC uint16
	IME                     bool
	Halted                  bool
	Stopped                 bool
	Cycles                  uint64
}

// PPUState is a snapshot of PPU registers and timing (basic, for dashboard).
type PPUState struct {
	Mode       int
	LY         byte
	LCDC       byte
	STAT       byte
	FrameCount int
}

// PPUFullState captures all PPU internal state for save-state serialisation.
type PPUFullState struct {
	Mode       int
	LY         byte
	LYC        byte
	STAT       byte
	LCDC       byte
	FrameCount int

	DotCounter int
	SCY        byte
	SCX        byte
	BGP        byte
	OBP0       byte
	OBP1       byte
	WX         byte
	WY         byte

	IsRunning          bool
	ScanlineRendered   bool
	Mode2End           int
	Mode3End           int
	OAMScanned         bool
	FirstFrameBlank    bool

	// Framebuffer
	Screen [160][144]byte
}

// MBCState captures memory-bank controller state for full save/restore.
type MBCState struct {
	RamEnabled  bool
	RomBankLow  byte   // Lower 8 bits of ROM bank (MBC1/2/3/5)
	RomBankHi   byte   // Bit 8 of ROM bank (MBC5 only; upper bank bits for MBC1/3)
	RamBankReg  byte   // RAM bank register (MBC1/3/5)
	Mode        byte   // Banking mode (MBC1 only: 0=simple, 1=advanced)
	HasRTC      bool   // MBC3 with RTC
	RTCRegs     [5]byte
	RTCLatched  [5]byte
	RTCLatchStep byte
	RTCClock    int64
	MBCType     byte   // 0=ROMonly, 1=MBC1, 2=MBC2, 3=MBC3, 5=MBC5
}

// APUFullState captures all APU registers and internal state.
type APUFullState struct {
	Regs        [23]byte
	WaveRAM     [16]byte
	Ch1On       bool
	Ch2On       bool
	Ch3On       bool
	Ch4On       bool
	CycAccum    int
	SeqStep     int
	LengthCnt   [4]int
	HPIntoL     int16
	HPOutL      int
	HPIntoR     int16
	HPOutR      int
}

// TimerFullState captures timer state including internal cycle counters.
type TimerFullState struct {
	DIV        byte
	TIMA       byte
	TMA        byte
	TAC        byte
	DivCycles  uint64
	TimaCycles uint64
}

// DMAState captures OAM DMA transfer state.
type DMAState struct {
	Active    bool
	Source    uint16
	Remaining int
}

// FullState is a complete emulator snapshot for save/load.
// It captures everything needed to perfectly restore emulation.
type FullState struct {
	CPU        CPUState
	PPU        PPUFullState
	Timer      TimerFullState
	APU        APUFullState

	// All writable memory regions
	WRAM [0x2000]byte // 0xC000-0xDFFF (8KB)
	VRAM [0x2000]byte // 0x8000-0x9FFF (8KB)
	OAM  [0xA0]byte   // 0xFE00-0xFE9F (160 bytes)
	HRAM [0x7F]byte   // 0xFF80-0xFFFE (127 bytes)
	IO   [0x80]byte   // 0xFF00-0xFF7F (128 bytes)
	IE   byte         // 0xFFFF

	// MMU internal state
	BootROMEnabled bool
	BootROM        [256]byte
	DMA            DMAState
	SerialActive   bool
	SerialCycles   int
	SB             byte
	SC             byte

	// Cartridge / MBC
	MBC        MBCState
	BatteryRAM []byte
}
