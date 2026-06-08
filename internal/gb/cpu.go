package gb

// CPU represents the LR35902 (Sharp SM83) processor core.
type CPU interface {
	Step() (cycles int, err error)
	Reset()
	GetState() CPUState
}

// CPUState is a snapshot of the CPU registers and flags.
type CPUState struct {
	AF, BC, DE, HL, SP, PC uint16
	IME                     bool
	Halted                  bool
	Stopped                 bool
	Cycles                  uint64
}
