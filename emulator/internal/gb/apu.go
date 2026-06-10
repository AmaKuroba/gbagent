package gb

// APU implements the Game Boy Audio Processing Unit register file with
// memory-mapped I/O at 0xFF10-0xFF26 and wave RAM at 0xFF30-0xFF3F.
// Includes the frame sequencer (512 Hz), length counters (256 Hz),
// sweep timing (128 Hz), envelope timing (64 Hz), and CH1/CH2 pulse
// channel synthesis.
//
// Register map (NR = Notational Register):
//
//	0xFF10 — NR10  — Channel 1 sweep
//	0xFF11 — NR11  — Channel 1 length / wave duty
//	0xFF12 — NR12  — Channel 1 envelope
//	0xFF13 — NR13  — Channel 1 frequency low
//	0xFF14 — NR14  — Channel 1 frequency high + control
//	0xFF16 — NR21  — Channel 2 length / wave duty
//	0xFF17 — NR22  — Channel 2 envelope
//	0xFF18 — NR23  — Channel 2 frequency low
//	0xFF19 — NR24  — Channel 2 frequency high + control
//	0xFF1A — NR30  — Channel 3 DAC on/off
//	0xFF1B — NR31  — Channel 3 length
//	0xFF1C — NR32  — Channel 3 output level
//	0xFF1D — NR33  — Channel 3 frequency low
//	0xFF1E — NR34  — Channel 3 frequency high + control
//	0xFF20 — NR41  — Channel 4 length
//	0xFF21 — NR42  — Channel 4 envelope
//	0xFF22 — NR43  — Channel 4 polynomial / frequency
//	0xFF23 — NR44  — Channel 4 control
//	0xFF24 — NR50  — Master volume / VIN panning
//	0xFF25 — NR51  — Channel output select (panning)
//	0xFF26 — NR52  — Sound on/off control
//	0xFF30-0xFF3F  — Channel 3 wave pattern RAM (16 bytes)
//
// Register index constants (addr - 0xFF10):
const (
	regNR10 = 0x00 // 0xFF10
	regNR11 = 0x01 // 0xFF11
	regNR12 = 0x02 // 0xFF12
	regNR13 = 0x03 // 0xFF13
	regNR14 = 0x04 // 0xFF14
	// 0xFF15 is a gap (index 0x05)
	regNR21 = 0x06 // 0xFF16
	regNR22 = 0x07 // 0xFF17
	regNR23 = 0x08 // 0xFF18
	regNR24 = 0x09 // 0xFF19
	regNR30 = 0x0A // 0xFF1A
	regNR31 = 0x0B // 0xFF1B
	regNR32 = 0x0C // 0xFF1C
	regNR33 = 0x0D // 0xFF1D
	regNR34 = 0x0E // 0xFF1E
	// 0xFF1F is a gap (index 0x0F)
	regNR41 = 0x10 // 0xFF20
	regNR42 = 0x11 // 0xFF21
	regNR43 = 0x12 // 0xFF22
	regNR44 = 0x13 // 0xFF23
	regNR50 = 0x14 // 0xFF24
	regNR51 = 0x15 // 0xFF25
	regNR52 = 0x16 // 0xFF26
)

// Frame sequencer step interval in T-cycles.
// The sequencer advances every 8192 T-cycles (512 Hz),
// triggered by the falling edge of DIV bit 4.
const sequencerInterval = 8192

// Duty waveform tables for pulse channels (CH1, CH2).
// dutyTable[duty][position] returns 0 or 1.
var dutyTable = [4][8]byte{
	{0, 0, 0, 0, 0, 0, 0, 1}, // 0: 12.5%
	{1, 0, 0, 0, 0, 0, 0, 1}, // 1: 25%
	{1, 1, 1, 1, 0, 0, 0, 0}, // 2: 50%
	{0, 1, 1, 1, 1, 1, 1, 0}, // 3: 75%
}

// pulseChannel implements a Game Boy square wave (pulse) channel
// with frequency generation and volume envelope.
// Used by Channel 1 (with sweep) and Channel 2 (without sweep).
type pulseChannel struct {
	// Frequency generator state
	frequency int // 11-bit raw frequency (0-2047)
	freqAccum int // T-cycle accumulator for frequency period counter
	dutyPos   int // current position in duty waveform (0-7)
	duty      int // duty cycle index (0-3), cached from NRx1 bits 7-6

	// Volume envelope state
	volume    int // current volume (0-15)
	envTimer  int // ticks remaining at 64 Hz until next volume change
	envPeriod int // envelope period from NRx2 bits 2-0 (0=disabled)
	envDir    int // envelope direction: 0=decrease, 1=increase
}

// period returns the frequency period in T-cycles for this channel.
// This is the number of T-cycles between each duty position advance.
// One full 8-step duty cycle takes period() * 8 T-cycles.
func (c *pulseChannel) period() int {
	return (2048 - c.frequency) * 4
}

// getOutput returns the current digital sample (0-15) for this channel.
// Returns 0 when volume is 0 (DAC off) or the current duty position is low.
func (c *pulseChannel) getOutput() int {
	if c.volume <= 0 || c.duty < 0 || c.duty > 3 {
		return 0
	}
	if dutyTable[c.duty][c.dutyPos] == 0 {
		return 0
	}
	return c.volume
}

// clockFrequency advances the frequency generator by `cycles` T-cycles.
// Advances the duty position whenever the frequency period is reached.
func (c *pulseChannel) clockFrequency(cycles int) {
	p := c.period()
	if p <= 0 {
		return
	}
	c.freqAccum += cycles
	for c.freqAccum >= p {
		c.freqAccum -= p
		c.dutyPos = (c.dutyPos + 1) & 7
	}
}

// triggerEnvelope initialises the volume envelope state on trigger.
// volume is set to initial volume from NRx2 bits 7-4.
func (c *pulseChannel) triggerEnvelope(nrx2 byte) {
	c.volume = int(nrx2 >> 4)
	c.envDir = int(nrx2 >> 3 & 1)
	c.envPeriod = int(nrx2 & 7)
	c.envTimer = c.envPeriod
}

// clockEnvelope advances the volume envelope by one 64 Hz tick.
func (c *pulseChannel) clockEnvelope() {
	if c.envPeriod == 0 {
		return // envelope disabled
	}
	c.envTimer--
	if c.envTimer > 0 {
		return
	}
	c.envTimer = c.envPeriod
	if c.envDir == 1 { // increase
		if c.volume < 15 {
			c.volume++
		}
	} else { // decrease
		if c.volume > 0 {
			c.volume--
		}
	}
}

// sweepUnit implements frequency sweep for Channel 1.
// The sweep modifies the channel frequency at 128 Hz intervals
// (sequencer steps 2 and 6) based on NR10 register configuration.
type sweepUnit struct {
	shadowFreq    int  // shadow frequency register (captured on trigger)
	sweepTimer    int  // ticks remaining (at 128 Hz) until next sweep
	sweepEnabled  bool // sweep is active
	sweepPeriod   int  // NR10 bits 6-4: sweep time (0-7, 0 is treated as 8)
	sweepNegate   bool // NR10 bit 3: negate (true = subtraction)
	sweepShift    int  // NR10 bits 2-0: shift value (0-7)
}

// noiseChannel implements the Game Boy noise channel (CH4).
// Uses a 15-bit LFSR in Galois configuration with XOR at bits 0 and 1.
type noiseChannel struct {
	lfsr  uint16 // 15-bit LFSR value
	accum int    // T-cycle accumulator for LFSR clock divider

	// Volume envelope (shared pattern with pulseChannel).
	volume    int // current volume (0-15)
	envTimer  int // ticks remaining at 64 Hz until next volume change
	envPeriod int // envelope period from NR42 bits 2-0 (0=disabled)
	envDir    int // envelope direction: 0=decrease, 1=increase
}

// noiseDividers maps NR43 divisor code (bits 0-2) to the divider value.
var noiseDividers = [8]int{8, 16, 32, 48, 64, 80, 96, 112}

// divisorPeriod returns T-cycles between LFSR steps: divider * 2^shift.
func (c *noiseChannel) divisorPeriod(nr43 byte) int {
	return noiseDividers[nr43&0x07] << ((nr43 >> 4) & 0x0F)
}

// stepLFSR performs one step of the 15-bit LFSR (Galois configuration).
func (c *noiseChannel) stepLFSR(nr43 byte) {
	feedback := (c.lfsr & 1) ^ ((c.lfsr >> 1) & 1)
	c.lfsr >>= 1
	c.lfsr |= feedback << 14
	c.lfsr &= 0x7FFF
	if nr43&0x08 != 0 { // 7-bit mode: inject feedback at bit 6
		c.lfsr &^= 1 << 6
		c.lfsr |= feedback << 6
	}
}

// clockLFSR advances the LFSR by the given number of T-cycles.
func (c *noiseChannel) clockLFSR(cycles int, nr43 byte) {
	period := c.divisorPeriod(nr43)
	if period <= 0 {
		return
	}
	c.accum += cycles
	for c.accum >= period {
		c.accum -= period
		c.stepLFSR(nr43)
	}
}

// triggerLFSR resets the LFSR to initial state (all 1s).
func (c *noiseChannel) triggerLFSR() {
	c.lfsr = 0x7FFF
	c.accum = 0
}

// triggerEnvelope initialises envelope from NR42 on trigger.
func (c *noiseChannel) triggerEnvelope(nr42 byte) {
	c.volume = int(nr42 >> 4)
	c.envDir = int(nr42 >> 3 & 1)
	c.envPeriod = int(nr42 & 7)
	c.envTimer = c.envPeriod
}

// clockEnvelope advances the volume envelope by one 64 Hz tick.
func (c *noiseChannel) clockEnvelope() {
	if c.envPeriod == 0 {
		return
	}
	c.envTimer--
	if c.envTimer > 0 {
		return
	}
	c.envTimer = c.envPeriod
	if c.envDir == 1 {
		if c.volume < 15 {
			c.volume++
		}
	} else {
		if c.volume > 0 {
			c.volume--
		}
	}
}

// getOutput returns the current sample (0-15): volume when LFSR bit 0 is 1.
func (c *noiseChannel) getOutput() int {
	if c.volume <= 0 || c.lfsr&1 == 0 {
		return 0
	}
	return c.volume
}

// waveChannel implements the Game Boy wave channel (CH3).
// Uses wave pattern RAM for sample values with configurable output level.
type waveChannel struct {
	frequency   int  // 11-bit raw frequency (0-2047)
	freqAccum   int  // T-cycle accumulator for frequency period counter
	samplePos   int  // current position in the 32-sample wave pattern (0-31)
	outputLevel int  // cached from NR32 bits 5-6: 0=100%, 1=50%, 2=25%, 3=0%
}

// period returns the frequency period in T-cycles for the wave channel.
// Period = (2048 - frequency) T-cycles per sample advance.
// One full 32-sample cycle = period * 32 T-cycles.
func (c *waveChannel) period() int {
	return (2048 - c.frequency) * 2
}

// clockFrequency advances the wave channel frequency generator by `cycles` T-cycles.
func (c *waveChannel) clockFrequency(cycles int) {
	p := c.period()
	if p <= 0 {
		return
	}
	c.freqAccum += cycles
	for c.freqAccum >= p {
		c.freqAccum -= p
		c.samplePos = (c.samplePos + 1) & 31
	}
}

// getSample returns the current 4-bit sample value from wave RAM,
// scaled by the output level. Returns 0-15.
func (c *waveChannel) getSample(waveRAM [16]byte) int {
	// Each wave RAM byte holds two 4-bit samples: high nibble at even pos, low at odd pos.
	byteIdx := c.samplePos >> 1
	isHigh := c.samplePos&1 == 0
	var sample int
	if isHigh {
		sample = int(waveRAM[byteIdx] >> 4)
	} else {
		sample = int(waveRAM[byteIdx] & 0x0F)
	}
	// Apply output level shift from NR32 bits 5-6.
	switch c.outputLevel {
	case 0: // 100%
		return sample
	case 1: // 50%
		return sample >> 1
	case 2: // 25%
		return sample >> 2
	default: // 3: 0% (mute)
		return 0
	}
}


type APU struct {
	// mmu for requesting interrupts (if needed by future channel logic)
	mmu MMU

	// Registers stored in a flat array indexed by (addr - 0xFF10).
	// Covers 0xFF10-0xFF26 inclusive (23 entries). The gaps at 0xFF15
	// and 0xFF1F are part of the array but unused by real hardware.
	regs [23]byte

	// Channel 3 wave pattern RAM (0xFF30-0xFF3F).
	waveRAM [16]byte

	// Channel active status, bits 0-3 (read via NR52 bits 0-3).
	ch1On bool
	ch2On bool
	ch3On bool
	ch4On bool

	// Pulse channel state for CH1 and CH2.
	ch1 *pulseChannel
	ch2 *pulseChannel

	// CH1 frequency sweep unit.
	sweep sweepUnit

	// Noise channel (CH4) state.
	ch4 *noiseChannel

	// Wave channel (CH3) state.
	ch3 *waveChannel

	// Frame sequencer state.
	cyclAccum int  // T-cycles accumulated since last sequencer tick
	seqStep   int  // current frame sequencer step (0-7)

	// Length counters for channels 1-4.
	lengthCnt [4]int

	// Audio sample buffer for mixer output.
	// Samples stored as 16-bit signed PCM (mono interleaved as left/right pairs).
	audioBuf    []int16 // accumulated audio samples for the current frame
	sampleAccum int     // T-cycle accumulator for sample rate generation
	hpLIn       int16   // high-pass filter: x[n-1] (left)
	hpLOut      int     // high-pass filter: y[n-1] (left)
	hpRIn       int16   // high-pass filter: x[n-1] (right)
	hpROut      int     // high-pass filter: y[n-1] (right)
}

// NewAPU creates a new APU instance with default power-on state.
func NewAPU(mmu MMU) *APU {
	return &APU{
		mmu:      mmu,
		regs:     [23]byte{regNR52: 0x70}, // APU off, bits 4-6 = 1
		seqStep:  0,
		ch1:      &pulseChannel{},
		ch2:      &pulseChannel{},
		ch3:      &waveChannel{},
		ch4:      &noiseChannel{},
		audioBuf: make([]int16, 0, 2048),
	}
}

// regIndex returns the register array index for an address in 0xFF10-0xFF26,
// or -1 if the address is outside this range.
func regIndex(addr uint16) int {
	if addr >= 0xFF10 && addr <= 0xFF26 {
		return int(addr - 0xFF10)
	}
	return -1
}

// ReadRegister returns the value of an APU register at the given address.
func (a *APU) ReadRegister(addr uint16) byte {
	if idx := regIndex(addr); idx >= 0 {
		v := a.regs[idx]
		if addr == 0xFF26 { // NR52
			v &^= 0x3F // clear bits 0-5; bits 4-6 overwritten below
			v |= 0x70  // bits 4-6 always 1
			if a.ch1On {
				v |= 0x01
			}
			if a.ch2On {
				v |= 0x02
			}
			if a.ch3On {
				v |= 0x04
			}
			if a.ch4On {
				v |= 0x08
			}
			// bit 7 is stored (APU on/off)
		}
		return v
	}
	if addr >= 0xFF30 && addr <= 0xFF3F {
		// Wave RAM gating: reads return 0xFF when CH3 is active.
		if a.ch3Active() {
			return 0xFF
		}
		return a.waveRAM[addr-0xFF30]
	}
	return 0xFF
}

// WriteRegister writes a value to an APU register at the given address.
func (a *APU) WriteRegister(addr uint16, val byte) {
	idx := regIndex(addr)

	// NR52 gating
	if idx >= 0 && addr != 0xFF26 {
		if a.regs[regNR52]&0x80 == 0 {
			return
		}
	}

	if idx >= 0 {
		if addr == 0xFF26 {
			wasOn := a.regs[regNR52]&0x80 != 0
			nowOn := val&0x80 != 0
			if wasOn && !nowOn {
				a.resetRegisters()
			} else if !wasOn && nowOn {
				a.cyclAccum = 0
				a.seqStep = 0
			}
			a.regs[idx] = (val & 0x80) | 0x70
			return
		}

		// Write register value first so trigger handlers see latest state.
		a.regs[idx] = val

		switch idx {
		case regNR30:
			// NR30 bit 7 controls DAC enable. Writing here doesn't re-trigger.
			// If DAC is turned off, channel is disabled.
			if val&0x80 == 0 {
				a.ch3On = false
			}
		case regNR32:
			a.cacheCh3OutputLevel()
		case regNR14:
			a.updateCh1Frequency()
			if val&0x80 != 0 {
				a.triggerCh1()
			}
		case regNR24:
			a.updateCh2Frequency()
			if val&0x80 != 0 {
				a.triggerCh2()
			}
		case regNR33:
			a.updateCh3Frequency()
		case regNR34:
			a.updateCh3Frequency()
			if val&0x80 != 0 {
				a.reloadLengthCounter(2)
				a.ch3On = true
				// DAC enable check: NR30 bit 7 must be set
				if a.regs[regNR30]&0x80 == 0 {
					a.ch3On = false
				}
				// First-sample skip: set pos=0 then immediately advance to pos=1
				a.ch3.samplePos = 1
				a.ch3.freqAccum = 0
			}
		case regNR44:
			if val&0x80 != 0 {
				a.triggerCh4()
			}
		case regNR13:
			a.updateCh1Frequency()
		case regNR23:
			a.updateCh2Frequency()
		}
		return
	}

	if addr >= 0xFF30 && addr <= 0xFF3F {
		// Wave RAM gating: writes ignored when CH3 is active.
		if !a.ch3Active() {
			a.waveRAM[addr-0xFF30] = val
		}
	}
}

// updateCh1Frequency refreshes the cached frequency from NR13 and NR14 bits 0-2.
func (a *APU) updateCh1Frequency() {
	a.ch1.frequency = int(a.regs[regNR13]) | (int(a.regs[regNR14]&7) << 8)
}

// updateCh2Frequency refreshes the cached frequency from NR23 and NR24 bits 0-2.
func (a *APU) updateCh2Frequency() {
	a.ch2.frequency = int(a.regs[regNR23]) | (int(a.regs[regNR24]&7) << 8)
}

// updateCh3Frequency refreshes the cached frequency from NR33 and NR34 bits 0-2.
func (a *APU) updateCh3Frequency() {
	a.ch3.frequency = int(a.regs[regNR33]) | (int(a.regs[regNR34]&7) << 8)
}

// cacheCh3OutputLevel caches the output level from NR32 bits 5-6.
func (a *APU) cacheCh3OutputLevel() {
	a.ch3.outputLevel = int(a.regs[regNR32] >> 5 & 3)
}

// ch3Active returns true when CH3 is active (DAC on AND channel on).
// CH3 DAC is controlled by NR30 bit 7.
func (a *APU) ch3Active() bool {
	return a.regs[regNR30]&0x80 != 0 && a.ch3On
}

// triggerCh1 handles Channel 1 trigger (NR14 bit 7).
func (a *APU) triggerCh1() {
	a.reloadLengthCounter(0)
	a.ch1On = true

	a.ch1.duty = int(a.regs[regNR11] >> 6 & 3)
	a.ch1.frequency = int(a.regs[regNR13]) | (int(a.regs[regNR14]&7) << 8)
	a.ch1.dutyPos = 0
	a.ch1.freqAccum = 0
	a.ch1.triggerEnvelope(a.regs[regNR12])

	// DAC off check: NR12 bits 7-3 all zero → DAC off.
	if a.regs[regNR12]&0xF8 == 0 {
		a.ch1On = false
	}

	// Sweep init
	nr10 := a.regs[regNR10]
	a.sweep.sweepPeriod = int(nr10 >> 4 & 7)
	a.sweep.sweepNegate = nr10&0x08 != 0
	a.sweep.sweepShift = int(nr10 & 7)
	a.sweep.shadowFreq = a.ch1.frequency
	a.sweep.sweepTimer = a.sweep.sweepPeriod
	if a.sweep.sweepTimer == 0 {
		a.sweep.sweepTimer = 8
	}
	a.sweep.sweepEnabled = a.sweep.sweepPeriod != 0 || a.sweep.sweepShift != 0

	// Immediate overflow check on trigger.
	if a.sweep.sweepEnabled && a.sweep.sweepShift != 0 {
		offset := a.sweep.shadowFreq >> a.sweep.sweepShift
		var newFreq int
		if a.sweep.sweepNegate {
			newFreq = a.sweep.shadowFreq - offset
		} else {
			newFreq = a.sweep.shadowFreq + offset
		}
		if newFreq < 0 || newFreq > 2047 {
			a.ch1On = false
		}
	}
}

// triggerCh2 handles Channel 2 trigger (NR24 bit 7).
func (a *APU) triggerCh2() {
	a.reloadLengthCounter(1)
	a.ch2On = true

	a.ch2.duty = int(a.regs[regNR21] >> 6 & 3)
	a.ch2.frequency = int(a.regs[regNR23]) | (int(a.regs[regNR24]&7) << 8)
	a.ch2.dutyPos = 0
	a.ch2.freqAccum = 0
	a.ch2.triggerEnvelope(a.regs[regNR22])

	if a.regs[regNR22]&0xF8 == 0 {
		a.ch2On = false
	}
}

// triggerCh4 handles Channel 4 trigger (NR44 bit 7).
func (a *APU) triggerCh4() {
	a.reloadLengthCounter(3)
	a.ch4On = true
	a.ch4.triggerLFSR()
	a.ch4.triggerEnvelope(a.regs[regNR42])
	if a.regs[regNR42]&0xF8 == 0 {
		a.ch4On = false
	}
}

// resetRegisters clears all registers and state to power-off configuration.
func (a *APU) resetRegisters() {
	a.regs = [23]byte{regNR52: 0x70}
	a.waveRAM = [16]byte{}
	a.ch1On = false
	a.ch2On = false
	a.ch3On = false
	a.ch4On = false
	a.cyclAccum = 0
	a.seqStep = 0
	a.lengthCnt = [4]int{}
	*a.ch1 = pulseChannel{}
	*a.ch2 = pulseChannel{}
	a.sweep = sweepUnit{}
	*a.ch3 = waveChannel{}
	*a.ch4 = noiseChannel{}
	a.audioBuf = a.audioBuf[:0]
	a.sampleAccum = 0
	a.hpLIn = 0
	a.hpROut = 0
	a.hpRIn = 0
	a.hpLOut = 0
}

// reloadLengthCounter reloads the length counter for the given channel.
func (a *APU) reloadLengthCounter(ch int) {
	switch ch {
	case 0:
		a.lengthCnt[0] = 64 - int(a.regs[regNR11]&0x3F)
	case 1:
		a.lengthCnt[1] = 64 - int(a.regs[regNR21]&0x3F)
	case 2:
		a.lengthCnt[2] = 256 - int(a.regs[regNR31])
	case 3:
		a.lengthCnt[3] = 64 - int(a.regs[regNR41]&0x3F)
	}
}

// Step advances the APU by the given number of T-cycles.
func (a *APU) Step(cycles int) {
	if cycles <= 0 || a.isOff() {
		return
	}

	// Clock frequency generators for active pulse channels.
	if a.ch1On {
		a.ch1.clockFrequency(cycles)
	}
	if a.ch2On {
		a.ch2.clockFrequency(cycles)
	}

	// Clock CH3 wave channel frequency generator.
	if a.ch3On {
		a.ch3.clockFrequency(cycles)
	}

	a.cyclAccum += cycles
	for a.cyclAccum >= sequencerInterval {
		a.cyclAccum -= sequencerInterval
		a.clockSequencer()
	}

	// CH4 (Noise) LFSR clocking. Runs as long as APU is on.
	a.ch4.clockLFSR(cycles, a.regs[regNR43])

	// Accumulate audio samples.
	a.fillAudioBuffer(cycles)
}

// isOff returns true when the APU is powered off (NR52 bit 7 = 0).
func (a *APU) isOff() bool {
	return a.regs[regNR52]&0x80 == 0
}

// clockSequencer fires events for the current frame sequencer step,
// then advances to the next step.
func (a *APU) clockSequencer() {
	step := a.seqStep

	if step&1 == 0 {
		a.clockLengthCounters()
	}
	if step == 2 || step == 6 {
		a.clockSweep()
	}
	if step == 7 {
		a.clockEnvelopes()
	}

	a.seqStep = (step + 1) & 7
}

// clockLengthCounters decrements the length counters for all four channels.
func (a *APU) clockLengthCounters() {
	if a.regs[regNR14]&0x40 != 0 && a.lengthCnt[0] > 0 {
		a.lengthCnt[0]--
		if a.lengthCnt[0] <= 0 {
			a.ch1On = false
		}
	}
	if a.regs[regNR24]&0x40 != 0 && a.lengthCnt[1] > 0 {
		a.lengthCnt[1]--
		if a.lengthCnt[1] <= 0 {
			a.ch2On = false
		}
	}
	if a.regs[regNR34]&0x40 != 0 && a.lengthCnt[2] > 0 {
		a.lengthCnt[2]--
		if a.lengthCnt[2] <= 0 {
			a.ch3On = false
		}
	}
	if a.regs[regNR44]&0x40 != 0 && a.lengthCnt[3] > 0 {
		a.lengthCnt[3]--
		if a.lengthCnt[3] <= 0 {
			a.ch4On = false
		}
	}
}

// clockSweep fires the sweep timing event for channel 1 (128 Hz).
func (a *APU) clockSweep() {
	if !a.sweep.sweepEnabled {
		return
	}
	a.sweep.sweepTimer--
	if a.sweep.sweepTimer > 0 {
		return
	}

	a.sweep.sweepTimer = a.sweep.sweepPeriod
	if a.sweep.sweepTimer == 0 {
		a.sweep.sweepTimer = 8
	}

	if a.sweep.sweepShift == 0 {
		return
	}

	offset := a.sweep.shadowFreq >> a.sweep.sweepShift
	var newFreq int
	if a.sweep.sweepNegate {
		newFreq = a.sweep.shadowFreq - offset
	} else {
		newFreq = a.sweep.shadowFreq + offset
	}

	if newFreq < 0 || newFreq > 2047 {
		a.ch1On = false
		return
	}

	a.sweep.shadowFreq = newFreq
	a.regs[regNR13] = byte(newFreq & 0xFF)
	a.regs[regNR14] = (a.regs[regNR14] & 0xF8) | byte((newFreq>>8)&7)
	a.ch1.frequency = newFreq

	// Second overflow check
	if a.sweep.sweepShift != 0 {
		offset2 := a.sweep.shadowFreq >> a.sweep.sweepShift
		var checkFreq int
		if a.sweep.sweepNegate {
			checkFreq = a.sweep.shadowFreq - offset2
		} else {
			checkFreq = a.sweep.shadowFreq + offset2
		}
		if checkFreq < 0 || checkFreq > 2047 {
			a.ch1On = false
		}
	}
}

// clockEnvelopes fires the volume envelope timing event for CH1, CH2, and CH4 (64 Hz).
func (a *APU) clockEnvelopes() {
	a.ch1.clockEnvelope()
	a.ch2.clockEnvelope()
	a.ch4.clockEnvelope()
}

// ---------------------------------------------------------------------------
// Sample output
// ---------------------------------------------------------------------------

// GetSampleCh1 returns the current audio sample for channel 1 (0-15).
func (a *APU) GetSampleCh1() int {
	if !a.ch1On {
		return 0
	}
	return a.ch1.getOutput()
}

// GetSampleCh2 returns the current audio sample for channel 2 (0-15).
func (a *APU) GetSampleCh2() int {
	if !a.ch2On {
		return 0
	}
	return a.ch2.getOutput()
}

// GetSampleCh3 returns the current audio sample for channel 3 (0-15).
// Output is from the wave channel's current wave RAM position, scaled by output level.
func (a *APU) GetSampleCh3() int {
	if !a.ch3Active() {
		return 0
	}
	return a.ch3.getSample(a.waveRAM)
}

// GetSampleCh4 returns the current audio sample for channel 4 (0-15).
func (a *APU) GetSampleCh4() int {
	if !a.ch4On {
		return 0
	}
	return a.ch4.getOutput()
}

// ---------------------------------------------------------------------------
// Mixer: combines all channels with NR50 master volume and NR51 panning.
// ---------------------------------------------------------------------------

// getMixedSample returns the mixed left and right 16-bit PCM samples
// for the current state. Applies a DC-blocking high-pass filter.
func (a *APU) getMixedSample() (int16, int16) {
	nr50 := a.regs[regNR50]
	nr51 := a.regs[regNR51]
	leftVol := int(nr50>>4&7) + 1  // 1-8
	rightVol := int(nr50&7) + 1    // 1-8

	// Get raw channel samples (0-15 each).
	ch1S := a.GetSampleCh1()
	ch2S := a.GetSampleCh2()
	ch3S := a.GetSampleCh3()
	ch4S := a.GetSampleCh4()

	// Sum per-side based on NR51 panning.
	leftSum := 0
	if nr51&0x01 != 0 {
		leftSum += ch1S
	}
	if nr51&0x02 != 0 {
		leftSum += ch2S
	}
	if nr51&0x04 != 0 {
		leftSum += ch3S
	}
	if nr51&0x08 != 0 {
		leftSum += ch4S
	}

	rightSum := 0
	if nr51&0x10 != 0 {
		rightSum += ch1S
	}
	if nr51&0x20 != 0 {
		rightSum += ch2S
	}
	if nr51&0x40 != 0 {
		rightSum += ch3S
	}
	if nr51&0x80 != 0 {
		rightSum += ch4S
	}

	// Scale by master volume.
	// Channel sum max = 4 * 15 = 60. Then scale by vol/8.
	// Final range: 0 to 60, typically centered around 0 for output.
	leftOut := leftSum * leftVol / 8
	rightOut := rightSum * rightVol / 8

	// Convert to 16-bit signed: map [0..60] to [-32768..32767].
	// 60 → ~32767, 0 → ~-32768, 30 → 0
	ls := int32(leftOut) * 1092 - 32768
	rs := int32(rightOut) * 1092 - 32768

	// Clamp to 16-bit range.
	if ls > 32767 {
		ls = 32767
	}
	if ls < -32768 {
		ls = -32768
	}
	if rs > 32767 {
		rs = 32767
	}
	if rs < -32768 {
		rs = -32768
	}

	// DC-blocking high-pass filter (first-order IIR).
	// y[n] = x[n] - x[n-1] + 0.999 * y[n-1]
	// Using integer math: y[n] = x[n] - x[n-1] + (y[n-1] * 999) / 1000
	fl := int16(ls)
	fr := int16(rs)
	// y[n] = x[n] - x[n-1] + 0.999 * y[n-1]
	hl := int(fl) - int(a.hpLIn) + (a.hpLOut*999)/1000
	hr := int(fr) - int(a.hpRIn) + (a.hpROut*999)/1000

	// Clamp after filter.
	if hl > 32767 {
		hl = 32767
	}
	if hl < -32768 {
		hl = -32768
	}
	if hr > 32767 {
		hr = 32767
	}
	if hr < -32768 {
		hr = -32768
	}

	a.hpLIn = fl   // x[n-1] for next frame
	a.hpLOut = hl  // y[n-1] for next frame
	a.hpRIn = fr
	a.hpROut = hr

	return int16(hl), int16(hr)
}

// ---------------------------------------------------------------------------
// Audio sample rate generation
// ---------------------------------------------------------------------------

// samplePeriod stores the number of T-cycles between audio samples.
// Target: ~44.1 kHz from 4194304 Hz T-cycles.
// 4194304 / 44100 ≈ 95.1, so we use 95 T-cycles per sample.
const samplePeriod = 95

// fillAudioBuffer advances the sample accumulator and generates audio samples
// for the given number of T-cycles.
func (a *APU) fillAudioBuffer(cycles int) {
	a.sampleAccum += cycles
	for a.sampleAccum >= samplePeriod {
		a.sampleAccum -= samplePeriod
		l, r := a.getMixedSample()
		a.audioBuf = append(a.audioBuf, l, r)
	}
}

// GetAudioBuffer returns the accumulated audio samples and clears the buffer.
// Returns stereo interleaved 16-bit PCM (L,R,L,R,...).
func (a *APU) GetAudioBuffer() []int16 {
	buf := a.audioBuf
	a.audioBuf = make([]int16, 0, 2048)
	return buf
}
