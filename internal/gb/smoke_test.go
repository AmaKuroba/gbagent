package gb

// stepFrame runs one full frame (70224 T-cycles) of the CPU.
// It accumulates cycles until exactly 70224 have elapsed.
func stepFrame(cpu *Core) error {
	target := cpu.Cycles + FrameCycles
	for cpu.Cycles < target {
		if _, err := cpu.Step(); err != nil {
			return err
		}
	}
	return nil
}
