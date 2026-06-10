package gb

// APU represents the Audio Processing Unit (4 channels).
type APU interface {
	Step(cycles int)
	Reset()
	GetSamples() []float32 // stereo samples at 44100 Hz
	GetState() APUState
}

// APUState is a snapshot of APU registers and channel state.
type APUState struct {
	Enabled bool
}
