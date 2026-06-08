package gb

// Emulator is the top-level facade that ties all components together.
type Emulator struct {
	CPU       CPU
	MMU       MMU
	PPU       PPU
	APU       APU
	Cartridge Cartridge
	Booted    bool
}

// StepFrame executes one complete frame (70224 cycles).
func (e *Emulator) StepFrame() error {
	for i := 0; i < 70224; i++ {
		cycles, err := e.CPU.Step()
		if err != nil {
			return err
		}
		e.PPU.Step(cycles)
		e.APU.Step(cycles)
	}
	return nil
}
