package gb

// CPUState is the canonical CPU state containing all registers, flags,
// interrupt state, and cycle counter. Embedded in Core and EmulatorState.
type CPUState struct {
	AF, BC, DE, HL, SP, PC uint16
	IME          bool
	IMEScheduled int  // EI schedule: 2=just set, 1=enable after this instr, 0=active
	Halted       bool
	Stopped      bool
	HaltBug      bool // HALT bug: prevent PC increment on next opcode fetch
	Cycles       uint64
}

// PPUState is the canonical PPU state with all registers, timing, and framebuffer.
type PPUState struct {
	Mode       int
	LY         byte
	LYC        byte
	Stat       byte  // STAT register
	LCDC       byte
	FrameCtr   int   // frame counter

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

// APUState captures all APU registers, channel state, and internal state.
type APUState struct {
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

	// CH1 pulse channel
	Pulse1Freq    int
	Pulse1Accum   int
	Pulse1DutyPos int
	Pulse1Duty    int
	Pulse1Vol     int
	Pulse1EnvTimer  int
	Pulse1EnvPeriod int
	Pulse1EnvDir    int

	// CH2 pulse channel
	Pulse2Freq    int
	Pulse2Accum   int
	Pulse2DutyPos int
	Pulse2Duty    int
	Pulse2Vol     int
	Pulse2EnvTimer  int
	Pulse2EnvPeriod int
	Pulse2EnvDir    int

	// CH1 sweep unit
	SweepShadowFreq   int
	SweepTimer        int
	SweepEnabled      bool
	SweepPeriod       int
	SweepNegate       bool
	SweepShift        int

	// CH4 noise channel
	NoiseLFSR     uint16
	NoiseAccum    int
	NoiseVol      int
	NoiseEnvTimer  int
	NoiseEnvPeriod int
	NoiseEnvDir    int

	// CH3 wave channel
	WaveFreq      int
	WaveAccum     int
	WaveSamplePos int
	WaveOutLevel  int
}

// TimerState is the canonical timer state including internal cycle counters.
type TimerState struct {
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
	PPU        PPUState
	Timer      TimerState
	APU        APUState

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
