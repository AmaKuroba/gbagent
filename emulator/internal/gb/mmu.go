package gb

// MMU represents the memory management unit and address space.
type MMU interface {
	Read(addr uint16) byte
	Write(addr uint16, val byte)
	Read16(addr uint16) uint16
	Write16(addr uint16, val uint16)
	LoadROM(data []byte)
	LoadBootROM(data []byte)
	ReadIF() byte
	ReadIE() byte
	WriteIF(val byte)
	SetJoypadButtons(buttons byte)
	DMAStep(cycles int)
	SerialStep(cycles int)
	StepDevices(cycles int)
}
