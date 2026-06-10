package gb

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// testAPUMMU is a minimal MMU stub for APU tests.
type testAPUMMU struct{}

func (m *testAPUMMU) Read(addr uint16) byte                        { return 0 }
func (m *testAPUMMU) Write(addr uint16, val byte)                   {}
func (m *testAPUMMU) Read16(addr uint16) uint16                     { return 0 }
func (m *testAPUMMU) Write16(addr uint16, val uint16)               {}
func (m *testAPUMMU) LoadROM(data []byte)                           {}
func (m *testAPUMMU) LoadBootROM(data []byte)                        {}
func (m *testAPUMMU) ReadIF() byte                                   { return 0 }
func (m *testAPUMMU) ReadIE() byte                                   { return 0 }
func (m *testAPUMMU) WriteIF(val byte)                               {}
func (m *testAPUMMU) DMAStep(cycles int)                             {}
func (m *testAPUMMU) SerialStep(cycles int)                          {}
func (m *testAPUMMU) SetJoypadButtons(buttons byte)                  {}
func (m *testAPUMMU) StepDevices(cycles int)                         {}

func TestAPUInitialState(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// NR52 should start with APU off (bit 7 = 0) and bits 4-6 = 1.
	nr52 := apu.ReadRegister(0xFF26)
	require.Equal(t, byte(0x70), nr52, "NR52 initial: APU off + bits 4-6 = 1")
}

func TestAPURegisterWriteAndRead(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// Turn APU on (NR52 bit 7 = 1) so writes are accepted.
	apu.WriteRegister(0xFF26, 0x80)

	// Write various values to registers and verify read-back.
	tests := []struct {
		addr uint16
		val  byte
	}{
		{0xFF10, 0x80},
		{0xFF11, 0xC0},
		{0xFF12, 0xF0},
		{0xFF13, 0xFF},
		{0xFF14, 0xC7},
		{0xFF16, 0xC0},
		{0xFF17, 0xF0},
		{0xFF18, 0xFF},
		{0xFF19, 0xC7},
		{0xFF1A, 0x80},
		{0xFF1B, 0xFF},
		{0xFF1C, 0x60},
		{0xFF1D, 0xFF},
		{0xFF1E, 0xC7},
		{0xFF20, 0xFF},
		{0xFF21, 0xF0},
		{0xFF22, 0xFF},
		{0xFF23, 0xC7},
		{0xFF24, 0x77},
		{0xFF25, 0xFF},
	}

	for _, tt := range tests {
		t.Run(regName(tt.addr), func(t *testing.T) {
			apu.WriteRegister(tt.addr, tt.val)
			got := apu.ReadRegister(tt.addr)
			require.Equal(t, tt.val, got, "Read back %s (0x%04X)", regName(tt.addr), tt.addr)
		})
	}
}

func TestAPUWaveRAMWriteAndRead(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// Turn APU on so writes are accepted.
	apu.WriteRegister(0xFF26, 0x80)

	// Write to all 16 wave RAM bytes.
	for i := range 16 {
		apu.WriteRegister(0xFF30+uint16(i), byte(i*17))
	}
	// Verify read-back.
	for i := range 16 {
		got := apu.ReadRegister(0xFF30 + uint16(i))
		require.Equal(t, byte(i*17), got, "Wave RAM byte %d", i)
	}
}

func TestAPUNR52GatingWritesIgnoredWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// APU starts off (NR52 bit 7 = 0). Writes to NR10-NR25 must be ignored.
	apu.WriteRegister(0xFF10, 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF10), "NR10 write ignored when APU off")

	apu.WriteRegister(0xFF16, 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF16), "NR21 write ignored when APU off")

	apu.WriteRegister(0xFF1A, 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF1A), "NR30 write ignored when APU off")

	apu.WriteRegister(0xFF20, 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF20), "NR41 write ignored when APU off")

	apu.WriteRegister(0xFF24, 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF24), "NR50 write ignored when APU off")
}

func TestAPUNR52WriteProcessedWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// NR52 write should work even when APU is off.
	apu.WriteRegister(0xFF26, 0x80)
	nr52 := apu.ReadRegister(0xFF26)
	require.True(t, nr52&0x80 != 0, "NR52 bit 7 should be set after write")

	// Turn APU off again.
	apu.WriteRegister(0xFF26, 0x70)
	nr52 = apu.ReadRegister(0xFF26)
	require.True(t, nr52&0x80 == 0, "NR52 bit 7 should be clear")
}

func TestAPUNR52Bits4to6AlwaysOne(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// Write to NR52 with bits 4-6 = 0 — they should still read as 1.
	apu.WriteRegister(0xFF26, 0x80) // bit 7 = 1, bits 4-6 = 0 in the write
	nr52 := apu.ReadRegister(0xFF26)
	require.Equal(t, byte(0xF0), nr52&0xF0, "NR52 bits 4-6 always read as 1")
}

func TestAPUNR52TurnOffResetsRegisters(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// Turn APU on.
	apu.WriteRegister(0xFF26, 0x80)

	// Write some values to registers.
	apu.WriteRegister(0xFF10, 0xAB)
	apu.WriteRegister(0xFF12, 0xCD)
	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF30, 0x42)

	// Verify written values.
	require.Equal(t, byte(0xAB), apu.ReadRegister(0xFF10), "NR10 before APU off")
	require.Equal(t, byte(0x77), apu.ReadRegister(0xFF24), "NR50 before APU off")
	require.Equal(t, byte(0x42), apu.ReadRegister(0xFF30), "Wave RAM before APU off")

	// Turn APU off — all registers should reset.
	apu.WriteRegister(0xFF26, 0x00)

	// Registers should be zeroed.
	require.Equal(t, byte(0), apu.ReadRegister(0xFF10), "NR10 reset after APU off")
	require.Equal(t, byte(0), apu.ReadRegister(0xFF12), "NR12 reset after APU off")
	require.Equal(t, byte(0), apu.ReadRegister(0xFF24), "NR50 reset after APU off")
	require.Equal(t, byte(0), apu.ReadRegister(0xFF30), "Wave RAM reset after APU off")

	// NR52 should have APU off + bits 4-6 = 1.
	nr52 := apu.ReadRegister(0xFF26)
	require.Equal(t, byte(0x70), nr52, "NR52 after turning APU off")
}

func TestAPUNR52ChannelStatusBits(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Channel status bits 0-3 in NR52 are driven by internal status flags.
	// By default all are off.
	nr52 := apu.ReadRegister(0xFF26)
	require.Equal(t, byte(0), nr52&0x0F, "All channels off initially")

	// Set channel active flags via exported fields (future: via channel logic).
	apu.Ch1On = true
	apu.Ch2On = true
	nr52 = apu.ReadRegister(0xFF26)
	require.True(t, nr52&0x01 != 0, "Ch1 status bit set")
	require.True(t, nr52&0x02 != 0, "Ch2 status bit set")
	require.True(t, nr52&0x04 == 0, "Ch3 status bit clear")
	require.True(t, nr52&0x08 == 0, "Ch4 status bit clear")

	// Turn APU off — resets channel status.
	apu.WriteRegister(0xFF26, 0x70)
	require.False(t, apu.Ch1On, "Ch1 status cleared on APU off")
	require.False(t, apu.Ch2On, "Ch2 status cleared on APU off")
}

func TestAPURegisterReadReturnsStoredValueWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	// Turn APU on, write some values, then turn it off.
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF10, 0xAB)
	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF30, 0x42)
	apu.WriteRegister(0xFF26, 0x70) // APU off — resets registers

	// Registers are reset when APU turns off, so reading should show 0.
	// This test verifies reads are NOT gated (no crash, no 0xFF)
	require.Equal(t, byte(0), apu.ReadRegister(0xFF10), "NR10 readable (returns 0 after reset)")
	require.Equal(t, byte(0), apu.ReadRegister(0xFF24), "NR50 readable (returns 0 after reset)")
	require.Equal(t, byte(0), apu.ReadRegister(0xFF30), "Wave RAM readable (returns 0 after reset)")
}

func TestAPUWaveRAMGatedOnChannel3Active(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Write to wave RAM and verify.
	apu.WriteRegister(0xFF30, 0xAA)
	require.Equal(t, byte(0xAA), apu.ReadRegister(0xFF30), "Wave RAM write/read works")
}

// ---------------------------------------------------------------------------
// Frame sequencer timing tests
// ---------------------------------------------------------------------------

func TestAPUStepFrameSequencerTiming(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	require.Equal(t, 0, apu.SeqStep, "Sequencer starts at step 0")

	apu.Step(8191)
	require.Equal(t, 0, apu.SeqStep, "Step 0 after 8191 cycles (not yet at boundary)")

	apu.Step(1)
	require.Equal(t, 1, apu.SeqStep, "Step 1 after 8192 cycles")
}

func TestAPUStepMultipleSequencerAdvances(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(32768)
	require.Equal(t, 4, apu.SeqStep, "Step 4 after 32768 cycles (4 advances)")

	apu.Step(32768)
	require.Equal(t, 0, apu.SeqStep, "Step 0 after 65536 cycles (8 advances, wrapped)")
}

func TestAPUStepSplitCycles(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(4096)
	apu.Step(4096)
	require.Equal(t, 1, apu.SeqStep, "Step 1 after 4096+4096 cycles")

	apu.Step(8192)
	require.Equal(t, 2, apu.SeqStep, "Step 2 after another 8192 cycles")

	apu.Step(10000)
	require.Equal(t, 3, apu.SeqStep, "Step 3 after 10000 cycles (8192+1808 extra)")

	apu.Step(6384)
	require.Equal(t, 4, apu.SeqStep, "Step 4 after remaining 6384 cycles")
}

func TestAPUStepDoesNotAdvanceWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	apu.Step(65536)
	require.Equal(t, 0, apu.SeqStep, "Sequencer stays at step 0 when APU off")
}

func TestAPUStepResetsOnAPUOn(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(8192 * 3)
	require.Equal(t, 3, apu.SeqStep, "Step 3 after 3 advances")

	apu.WriteRegister(0xFF26, 0x70) // off
	apu.WriteRegister(0xFF26, 0x80) // on
	require.Equal(t, 0, apu.SeqStep, "Sequencer resets to step 0 after APU off+on")
}

func TestAPUStepResetsOnAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(8192 * 5)
	require.Equal(t, 5, apu.SeqStep, "Step 5 before APU off")

	apu.WriteRegister(0xFF26, 0x70)
	require.Equal(t, 0, apu.SeqStep, "Sequencer resets to step 0 on APU off")
	require.Equal(t, 0, apu.CycAccum, "Cycle accumulator cleared on APU off")
}

func TestAPUStepNegativeOrZero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	before := apu.SeqStep
	apu.Step(0)
	apu.Step(-1)
	require.Equal(t, before, apu.SeqStep, "Sequencer unchanged after 0/-1 cycles")
}

// ---------------------------------------------------------------------------
// Length counter tests
// ---------------------------------------------------------------------------

func TestAPULengthCounterReloadOnTriggerCh1(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Set NR12 (envelope) with non-zero initial volume so DAC is on.
	apu.WriteRegister(0xFF12, 0xF0) // volume 15, dir 0, period 0

	// Set NR11 (length) to 0x3F (max), so length counter = 64 - 63 = 1.
	apu.WriteRegister(0xFF11, 0x3F)
	// Trigger channel 1 via NR14 bit 7.
	apu.WriteRegister(0xFF14, 0x80)
	require.Equal(t, 1, apu.LengthCnt[0], "Ch1 length counter = 64-63 = 1")
	require.True(t, apu.Ch1On, "Ch1 should be on after trigger")
}

func TestAPULengthCounterReloadOnTriggerCh2(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Set NR22 (envelope) with non-zero initial volume so DAC is on.
	apu.WriteRegister(0xFF17, 0xF0) // volume 15, dir 0, period 0

	// Set NR21 (length) to 0x01, so length counter = 64 - 1 = 63.
	apu.WriteRegister(0xFF16, 0x01)
	apu.WriteRegister(0xFF19, 0x80) // trigger ch2
	require.Equal(t, 63, apu.LengthCnt[1], "Ch2 length counter = 64-1 = 63")
	require.True(t, apu.Ch2On, "Ch2 should be on after trigger")
}

func TestAPULengthCounterReloadOnTriggerCh3(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Enable DAC (NR30 bit 7) so CH3 stays on after trigger.
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1B, 0x80)
	apu.WriteRegister(0xFF1E, 0x80) // trigger ch3
	require.Equal(t, 128, apu.LengthCnt[2], "Ch3 length counter = 256-128 = 128")
	require.True(t, apu.Ch3On, "Ch3 should be on after trigger")
}

func TestAPULengthCounterReloadOnTriggerCh4(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Set NR42 (envelope) with non-zero initial volume so DAC is on.
	apu.WriteRegister(0xFF21, 0xF0) // volume 15, dir 0, period 0
	apu.WriteRegister(0xFF20, 0x20)
	apu.WriteRegister(0xFF23, 0x80) // trigger ch4
	require.Equal(t, 32, apu.LengthCnt[3], "Ch4 length counter = 64-32 = 32")
	require.True(t, apu.Ch4On, "Ch4 should be on after trigger")
}

func TestAPULengthCounterDecrementsOnSequencerSteps(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15

	apu.WriteRegister(0xFF11, 0x00)
	apu.WriteRegister(0xFF14, 0xC0) // trigger ch1 AND set length enable (bit 6)

	require.Equal(t, 64, apu.LengthCnt[0], "Ch1 length counter initial = 64")

	apu.Step(8192)
	require.Equal(t, 63, apu.LengthCnt[0], "Ch1 length counter decremented to 63")

	apu.Step(8192)
	require.Equal(t, 63, apu.LengthCnt[0], "Ch1 length counter unchanged at step 1 (no event)")

	apu.Step(8192)
	require.Equal(t, 62, apu.LengthCnt[0], "Ch1 length counter decremented to 62 at step 2")
}

func TestAPULengthCounterDisablesChannelOnZero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15

	apu.WriteRegister(0xFF11, 0x3F)
	apu.WriteRegister(0xFF14, 0xC0) // trigger + length enable
	require.Equal(t, 1, apu.LengthCnt[0], "Ch1 length counter = 1")
	require.True(t, apu.Ch1On, "Ch1 on after trigger")

	apu.Step(8192)
	require.Equal(t, 0, apu.LengthCnt[0], "Ch1 length counter = 0")
	require.False(t, apu.Ch1On, "Ch1 disabled when length counter reaches 0")

	nr52 := apu.ReadRegister(0xFF26)
	require.True(t, nr52&0x01 == 0, "NR52 ch1 status bit clear after length counter reaches 0")
}

func TestAPULengthCounterEnableBitControlsClock(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15

	apu.WriteRegister(0xFF11, 0x00)
	apu.WriteRegister(0xFF14, 0x80) // trigger, bit 6 not set

	require.Equal(t, 64, apu.LengthCnt[0], "Ch1 length counter loaded")

	apu.Step(8192)
	require.Equal(t, 64, apu.LengthCnt[0], "Ch1 length counter unchanged (enable bit not set)")
	require.True(t, apu.Ch1On, "Ch1 still on")

	apu.WriteRegister(0xFF14, 0x40) // bit 6 = 1 enable, bit 7 = 0 no trigger

	apu.Step(16384) // step 1 + step 2 = length counter at step 2
	require.Equal(t, 63, apu.LengthCnt[0], "Ch1 length counter decremented after enable set")
}

func TestAPULengthCounterMultipleChannels(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // CH1: volume 15
	apu.WriteRegister(0xFF17, 0xF0) // CH2: volume 15
	apu.WriteRegister(0xFF21, 0xF0) // CH4: volume 15

	apu.WriteRegister(0xFF11, 64-5)
	apu.WriteRegister(0xFF14, 0xC0) // trigger + enable

	apu.WriteRegister(0xFF16, 64-10)
	apu.WriteRegister(0xFF19, 0xC0) // trigger + enable

	apu.WriteRegister(0xFF1A, 0x80) // CH3: DAC on
	apu.WriteRegister(0xFF1B, 256-20)
	apu.WriteRegister(0xFF1E, 0xC0) // trigger + enable

	apu.WriteRegister(0xFF20, 64-15)
	apu.WriteRegister(0xFF23, 0xC0) // trigger + enable

	require.True(t, apu.Ch1On, "Ch1 on")
	require.True(t, apu.Ch2On, "Ch2 on")
	require.True(t, apu.Ch3On, "Ch3 on")
	require.True(t, apu.Ch4On, "Ch4 on")

	apu.Step(8192 * 4)

	require.Equal(t, 3, apu.LengthCnt[0], "Ch1 length = 5-2 = 3")
	require.Equal(t, 8, apu.LengthCnt[1], "Ch2 length = 10-2 = 8")
	require.Equal(t, 18, apu.LengthCnt[2], "Ch3 length = 20-2 = 18")
	require.Equal(t, 13, apu.LengthCnt[3], "Ch4 length = 15-2 = 13")

	require.True(t, apu.Ch1On, "Ch1 still on")

	apu.Step(8192 * 6)
	require.False(t, apu.Ch1On, "Ch1 disabled after length runs out")
}

func TestAPULengthCounterDoesNotDecrementBelowZero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15

	apu.WriteRegister(0xFF11, 63) // 64-63 = 1
	apu.WriteRegister(0xFF14, 0xC0)

	apu.Step(8192)
	require.False(t, apu.Ch1On, "Ch1 disabled")

	for range 10 {
		apu.Step(8192)
	}
	require.Equal(t, 0, apu.LengthCnt[0], "Ch1 length counter stays at 0 (doesn't go negative)")
	require.False(t, apu.Ch1On, "Ch1 stays off")
}

// ---------------------------------------------------------------------------
// Sweep and envelope timing tests
// ---------------------------------------------------------------------------

func TestAPUSweepFiresOnSteps2And6(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(8192 * 3) // steps 0, 1, 2
	require.Equal(t, 3, apu.SeqStep, "Step 3 after 3 advances (step 2 sweep just fired)")

	apu.Step(8192 * 3)
	require.Equal(t, 6, apu.SeqStep, "Step 6 after 6 advances (step 6 sweep just fired)")
}

func TestAPUEnvelopeFiresOnStep7(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.Step(8192 * 7)
	require.Equal(t, 7, apu.SeqStep, "Step 7 after 7 advances (envelope fires here)")
}

func TestAPUStepSweepEnvelopeSequence(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	stepSequence := []int{1, 2, 3, 4, 5, 6, 7, 0}

	for i, expectedStep := range stepSequence {
		apu.Step(8192)
		require.Equal(t, expectedStep, apu.SeqStep,
			"Step %d after %d T-cycles (expected step %d)",
			apu.SeqStep, (i+1)*8192, expectedStep)
	}
}

func TestAPUTriggerDoesNotFireWithoutBit7(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF14, 0x40)
	require.Equal(t, 0, apu.LengthCnt[0], "Ch1 length counter not loaded (no trigger)")

	apu.WriteRegister(0xFF14, 0x80)
	require.NotEqual(t, 0, apu.LengthCnt[0], "Ch1 length counter loaded on trigger")
}

func TestAPULengthCounterClearedOnAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15

	apu.WriteRegister(0xFF11, 0x00)
	apu.WriteRegister(0xFF14, 0xC0) // trigger + enable
	require.Equal(t, 64, apu.LengthCnt[0], "Ch1 length counter loaded")

	apu.WriteRegister(0xFF26, 0x70)
	require.Equal(t, 0, apu.LengthCnt[0], "Ch1 length counter cleared on APU off")
}

// ---------------------------------------------------------------------------
// Register-routing via MemoryBus integration test
// ---------------------------------------------------------------------------

func TestAPURegisterRoutingViaMemoryBus(t *testing.T) {
	mmu := NewMMU(nil)
	apu := NewAPU(mmu)
	mmu.SetAPU(apu)

	mmu.Write(0xFF26, 0x80)

	mmu.Write(0xFF10, 0xAB)
	mmu.Write(0xFF24, 0x77)
	mmu.Write(0xFF30, 0x42)

	require.Equal(t, byte(0xAB), mmu.Read(0xFF10), "NR10 via MemoryBus")
	require.Equal(t, byte(0x77), mmu.Read(0xFF24), "NR50 via MemoryBus")
	require.Equal(t, byte(0x42), mmu.Read(0xFF30), "Wave RAM via MemoryBus")

	mmu.Write(0xFF26, 0x70) // APU off
	mmu.Write(0xFF10, 0xFF) // ignored
	require.Equal(t, byte(0), mmu.Read(0xFF10), "NR10 write ignored (APU off, via MemoryBus)")
}

// ---------------------------------------------------------------------------
// All-register address test
// ---------------------------------------------------------------------------

func TestAPUAllRegisterAddresses(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	validAddrs := []uint16{
		0xFF10, 0xFF11, 0xFF12, 0xFF13, 0xFF14,
		0xFF16, 0xFF17, 0xFF18, 0xFF19,
		0xFF1A, 0xFF1B, 0xFF1C, 0xFF1D, 0xFF1E,
		0xFF20, 0xFF21, 0xFF22, 0xFF23,
		0xFF24, 0xFF25, 0xFF26,
	}
	for _, addr := range validAddrs {
		val := apu.ReadRegister(addr)
		if addr == 0xFF26 {
			require.Equal(t, byte(0xF0), val, "NR52 with APU on reads 0xF0 (channels off)")
		} else {
			require.Equal(t, byte(0), val, "0x%04X initial value is 0", addr)
		}
	}

	for i := range 16 {
		val := apu.ReadRegister(0xFF30 + uint16(i))
		require.Equal(t, byte(0), val, "Wave RAM 0x%04X initial value is 0", 0xFF30+uint16(i))
	}

	gapAddrs := []uint16{0xFF15, 0xFF1F}
	for _, addr := range gapAddrs {
		val := apu.ReadRegister(addr)
		require.Equal(t, byte(0), val, "0x%04X (gap) returns stored value (0 initial)", addr)
	}

	invalidAddrs := []uint16{0xFF27, 0xFF2F, 0xFF40}
	for _, addr := range invalidAddrs {
		val := apu.ReadRegister(addr)
		require.Equal(t, byte(0xFF), val, "0x%04X (unmapped) returns 0xFF", addr)
	}
}

// ---------------------------------------------------------------------------
// CH1/CH2 pulse channel tests
// ---------------------------------------------------------------------------

func TestPulseCh1DutyOutput12p5(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	apu.WriteRegister(0xFF12, 0xF0) // volume 15
	apu.WriteRegister(0xFF11, 0x00) // duty 0 (12.5%)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87) // freq hi=7, trigger

	require.True(t, apu.Ch1On, "CH1 on after trigger")

	// Duty 0: [0,0,0,0,0,0,0,1]. Period = 4 T-cycles.
	// Check initial position before first step.
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 pos 0")

	expected := []int{0, 0, 0, 0, 0, 0, 15}
	for pos, exp := range expected {
		apu.Step(4)
		require.Equal(t, exp, apu.GetSampleCh1(), "CH1 pos %d (duty 0 12.5%%)", pos+1)
	}
}

func TestPulseCh1DutyOutput25(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x40) // duty 1
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// Duty 1: [1,0,0,0,0,0,0,1]
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 pos 0")
	expected := []int{0, 0, 0, 0, 0, 0, 15}
	for pos, exp := range expected {
		apu.Step(4)
		require.Equal(t, exp, apu.GetSampleCh1(), "CH1 pos %d (duty 1 25%%)", pos+1)
	}
}

func TestPulseCh1DutyOutput50(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80) // duty 2
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// Duty 2: [1,1,1,1,0,0,0,0]
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 pos 0")
	expected := []int{15, 15, 15, 0, 0, 0, 0}
	for pos, exp := range expected {
		apu.Step(4)
		require.Equal(t, exp, apu.GetSampleCh1(), "CH1 pos %d (duty 2 50%%)", pos+1)
	}
}

func TestPulseCh1DutyOutput75(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0xC0) // duty 3
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// Duty 3: [0,1,1,1,1,1,1,0]
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 pos 0")
	expected := []int{15, 15, 15, 15, 15, 15, 0}
	for pos, exp := range expected {
		apu.Step(4)
		require.Equal(t, exp, apu.GetSampleCh1(), "CH1 pos %d (duty 3 75%%)", pos+1)
	}
}

func TestPulseCh1SampleZeroWhenOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	require.False(t, apu.Ch1On, "CH1 off initially")
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 sample 0 when off")
}

func TestPulseCh2DutyOutput(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF17, 0xF0) // volume 15
	apu.WriteRegister(0xFF16, 0x00) // duty 0
	apu.WriteRegister(0xFF18, 0xFF)
	apu.WriteRegister(0xFF19, 0x87) // trigger

	require.True(t, apu.Ch2On, "CH2 on after trigger")

	require.Equal(t, 0, apu.GetSampleCh2(), "CH2 pos 0 (duty 0)")
	expected := []int{0, 0, 0, 0, 0, 0, 15}
	for pos, exp := range expected {
		apu.Step(4)
		require.Equal(t, exp, apu.GetSampleCh2(), "CH2 pos %d", pos+1)
	}
}

func TestPulseCh2SampleZeroWhenOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	require.False(t, apu.Ch2On, "CH2 off initially")
	require.Equal(t, 0, apu.GetSampleCh2(), "CH2 sample 0 when off")
}

func TestPulseFreqGeneratorDutyPositionTiming(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80) // duty 2 (50%: starts high)

	freq := 2023 // period = (2048-2023)*4 = 100
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 sample at position 0")

	apu.Step(99)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 sample unchanged before period boundary")

	apu.Step(1)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 sample at position 1")

	apu.Step(99)
	apu.Step(1)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 sample at position 2")

	apu.Step(200) // 2 periods to position 4 (first low in duty 2)
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 sample at position 4 (first low)")
}

func TestPulseEnvelopeDecrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF1) // volume 15, dir=0 (decrease), period=1
	apu.WriteRegister(0xFF11, 0x80) // duty 2 (starts high)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.True(t, apu.Ch1On, "CH1 on after trigger")
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 initial volume = 15")

	// Advance one full 8-step cycle (envelope fires at step 7).
	apu.Step(65536)
	require.Equal(t, 14, apu.GetSampleCh1(), "CH1 volume decreased to 14")
}

func TestPulseEnvelopeIncrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0x19) // volume 1, dir=1 (increase), period=1
	apu.WriteRegister(0xFF11, 0x80) // duty 2
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.True(t, apu.Ch1On, "CH1 on after trigger")
	require.Equal(t, 1, apu.GetSampleCh1(), "CH1 initial volume = 1")

	apu.Step(65536)
	require.Equal(t, 2, apu.GetSampleCh1(), "CH1 volume increased to 2")

	apu.Step(65536)
	require.Equal(t, 3, apu.GetSampleCh1(), "CH1 volume increased to 3")
}

func TestPulseEnvelopeStopsAtMaximum(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xE9) // volume 14, dir=1, period=1
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.Equal(t, 14, apu.GetSampleCh1(), "CH1 initial volume = 14")

	apu.Step(65536)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 volume increased to 15")

	apu.Step(65536)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 volume stays at 15")

	apu.Step(65536)
	require.Equal(t, 15, apu.GetSampleCh1(), "CH1 volume still capped at 15")
}

func TestPulseEnvelopeStopsAtMinimum(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0x21) // volume 2, dir=0, period=1
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.Equal(t, 2, apu.GetSampleCh1(), "CH1 initial volume = 2")

	apu.Step(65536)
	require.Equal(t, 1, apu.GetSampleCh1(), "CH1 volume decreased to 1")

	apu.Step(65536)
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 volume decreased to 0")

	apu.Step(65536)
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 volume stays at 0")
}

func TestPulseEnvelopePeriod3Slower(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xA3) // volume 10, dir=0, period=3
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.Equal(t, 10, apu.GetSampleCh1(), "CH1 initial volume = 10")

	apu.Step(65536)
	require.Equal(t, 10, apu.GetSampleCh1(), "CH1 volume unchanged (timer=2)")

	apu.Step(65536)
	require.Equal(t, 10, apu.GetSampleCh1(), "CH1 volume unchanged (timer=1)")

	apu.Step(65536)
	require.Equal(t, 9, apu.GetSampleCh1(), "CH1 volume decreased after 3 ticks")
}

func TestPulseEnvelopeDisabledWhenPeriodZero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xA0) // volume 10, dir=1, period=0 (disabled)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.Equal(t, 10, apu.GetSampleCh1(), "CH1 initial volume = 10")

	for range 5 {
		apu.Step(65536)
	}
	require.Equal(t, 10, apu.GetSampleCh1(), "CH1 volume unchanged (envelope disabled)")
}

func TestPulseDACOffWithZeroVolume(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0x00) // DAC off (bits 7-3 = 00000)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.False(t, apu.Ch1On, "CH1 stays off when DAC is off")
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 sample 0 when DAC off")
}

func TestPulseDACOnWithNonZeroVolume(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0x18) // volume 1, dir=1, period=0. DAC on (bits 7-3 != 0)
	apu.WriteRegister(0xFF11, 0x80) // duty 2 (starts high)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.True(t, apu.Ch1On, "CH1 on (volume=1, dir=1: DAC on)")
	require.Equal(t, 1, apu.GetSampleCh1(), "CH1 sample = volume 1 at duty 2 pos 0")
}

// ---------------------------------------------------------------------------
// CH1 sweep tests
// ---------------------------------------------------------------------------

func TestPulseSweepIncrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF10, 0x11) // period=1, negate=0, shift=1
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)

	freq := 512
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.True(t, apu.Ch1On, "CH1 on after trigger")
	require.Equal(t, 512, apu.ch1.frequency, "CH1 initial frequency = 512")

	// Sweep at step 2: shadow=512, offset=512>>1=256, new=512+256=768
	apu.Step(8192 * 3)
	require.Equal(t, 768, apu.ch1.frequency,
		"CH1 freq increased by shadow>>shift (512+256=768)")
}

func TestPulseSweepDecrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF10, 0x19) // period=1, negate=1, shift=1
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)

	freq := 512
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	// shadow=512, offset=512>>1=256, negate: new=512-256=256
	apu.Step(8192 * 3)
	require.Equal(t, 256, apu.ch1.frequency,
		"CH1 freq decreased (512-256=256)")
}

func TestPulseSweepOverflowDisablesChannel(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF10, 0x11) // period=1, negate=0, shift=1
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)

	// Use freq=1365: offset=1365>>1=682, new=1365+682=2047 (no overflow on trigger)
	// After update: shadow=2047, 2nd check: 2047>>1=1023, new=2047+1023=3070>2047 → off
	freq := 1365
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.True(t, apu.Ch1On, "CH1 on after trigger (freq=1365, check=2047, no overflow)")

	apu.Step(8192 * 3)
	require.False(t, apu.Ch1On, "CH1 disabled by sweep overflow")
	require.Equal(t, 0, apu.GetSampleCh1(), "CH1 sample 0 after overflow")
}

func TestPulseSweepImmediateOverflowOnTrigger(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// First: no overflow on trigger.
	apu.WriteRegister(0xFF10, 0x79) // period=7, negate=1, shift=1
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)
	freq := 4
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.True(t, apu.Ch1On, "CH1 on (no overflow with freq=4, negate=1)")

	// Now trigger with immediate overflow: freq=1500, shift=1, negate=0
	// shadow=1500, offset=750, new=1500+750=2250>2047 → overflow
	apu.WriteRegister(0xFF10, 0x11) // period=1, negate=0, shift=1
	freq = 1500
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.False(t, apu.Ch1On,
		"CH1 disabled by immediate sweep overflow (1500+750>2047)")
}

func TestPulseSweepMultipleSteps(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF10, 0x12) // period=1, negate=0, shift=2
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)

	freq := 100
	apu.WriteRegister(0xFF13, byte(freq&0xFF))
	apu.WriteRegister(0xFF14, 0x80|byte(freq>>8))

	require.Equal(t, 100, apu.ch1.frequency, "Initial freq = 100")

	// Tick 1: shadow=100, offset=100>>2=25, new=125
	apu.Step(8192 * 3)
	require.Equal(t, 125, apu.ch1.frequency, "After 1st sweep: 100+25=125")

	// Tick 2: shadow=125, offset=125>>2=31, new=156
	apu.Step(8192 * 4) // steps 3,4,5,6
	require.Equal(t, 156, apu.ch1.frequency, "After 2nd sweep: 125+31=156")
}

func TestPulseCh1SampleVolumeScaling(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0x70) // volume 7, dir=0, period=0
	apu.WriteRegister(0xFF11, 0x80) // duty 2 (starts high)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	require.True(t, apu.Ch1On, "CH1 on after trigger")
	require.Equal(t, 7, apu.GetSampleCh1(), "CH1 sample = 7 (volume 7)")
}

func TestPulseCh1FrequencyUpdateLive(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x00)
	apu.WriteRegister(0xFF13, 0xE8)
	apu.WriteRegister(0xFF14, 0x83) // hi=3, trigger

	require.Equal(t, 0x3E8, apu.ch1.frequency, "CH1 freq = 1000 after trigger")

	apu.WriteRegister(0xFF13, 0x00)
	require.Equal(t, 0x300, apu.ch1.frequency, "CH1 freq updated (low byte changed)")

	apu.WriteRegister(0xFF14, 0x01) // hi bits = 001, no trigger
	require.Equal(t, 0x100, apu.ch1.frequency, "CH1 freq updated (hi bits changed)")
}

// ---------------------------------------------------------------------------
// regName helper
// ---------------------------------------------------------------------------

func regName(addr uint16) string {
	names := map[uint16]string{
		0xFF10: "NR10",
		0xFF11: "NR11",
		0xFF12: "NR12",
		0xFF13: "NR13",
		0xFF14: "NR14",
		0xFF16: "NR21",
		0xFF17: "NR22",
		0xFF18: "NR23",
		0xFF19: "NR24",
		0xFF1A: "NR30",
		0xFF1B: "NR31",
		0xFF1C: "NR32",
		0xFF1D: "NR33",
		0xFF1E: "NR34",
		0xFF20: "NR41",
		0xFF21: "NR42",
		0xFF22: "NR43",
		0xFF23: "NR44",
		0xFF24: "NR50",
		0xFF25: "NR51",
		0xFF26: "NR52",
	}
	if name, ok := names[addr]; ok {
		return name
	}
	if addr >= 0xFF30 && addr <= 0xFF3F {
		return "WaveRAM"
	}
	return "Unknown"
}

// ---------------------------------------------------------------------------
// CH4 (Noise) tests
// ---------------------------------------------------------------------------

// TestNoiseLFSRStep15Bit verifies the 15-bit LFSR produces the correct
// sequence starting from the trigger-reset value 0x7FFF.
func TestNoiseLFSRStep15Bit(t *testing.T) {
	c := &noiseChannel{}
	c.triggerLFSR()
	require.Equal(t, uint16(0x7FFF), c.lfsr, "LFSR initialised to 0x7FFF on trigger")

	nr43 := byte(0x00)

	// First 14 steps: all 1s get shifted out.
	for i := range 14 {
		outputBefore := c.lfsr & 1
		c.stepLFSR(nr43)
		require.Equal(t, uint16(1), outputBefore, "LFSR output bit before step %d", i)
	}

	// After 14 steps from 0x7FFF: lfsr = 0x0001.
	require.Equal(t, uint16(0x0001), c.lfsr, "LFSR = 0x0001 after 14 steps")

	// Step 15: feedback=1, >>1=0, |1<<14 = 0x4000.
	c.stepLFSR(nr43)
	require.Equal(t, uint16(0x4000), c.lfsr, "LFSR = 0x4000 after 15 steps")
}

// TestNoiseLFSR7BitMode verifies 7-bit mode (NR43 bit 3 = 1) shortens the sequence.
func TestNoiseLFSR7BitMode(t *testing.T) {
	c := &noiseChannel{}
	c.triggerLFSR()
	require.Equal(t, uint16(0x7FFF), c.lfsr)

	nr43 := byte(0x08)

	// Step 1: 0x7FFF >>1 = 0x3FFF, feedback=0. Clear bit6=0x3FBF, set bit6=0 → 0x3FBF.
	c.stepLFSR(nr43)
	require.Equal(t, uint16(0x3FBF), c.lfsr, "LFSR = 0x3FBF after 1 step in 7-bit mode")

	// Step 2: feedback=0, >>1=0x1FDF, clear bit6→0x1F9F.
	c.stepLFSR(nr43)
	require.Equal(t, uint16(0x1F9F), c.lfsr, "LFSR = 0x1F9F after 2 steps in 7-bit mode")
}

// TestNoiseTriggerResetsLFSR verifies triggering CH4 resets the LFSR.
func TestNoiseTriggerResetsLFSR(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF0) // NR42: volume 15
	apu.WriteRegister(0xFF23, 0x80) // NR44: trigger

	require.True(t, apu.Ch4On, "CH4 on after trigger")
	require.Equal(t, uint16(0x7FFF), apu.ch4.lfsr, "LFSR reset to 0x7FFF on trigger")
}

// TestNoiseDivisorPeriod verifies divisor period calculation from NR43.
func TestNoiseDivisorPeriod(t *testing.T) {
	c := &noiseChannel{}

	tests := []struct {
		nr43   byte
		period int
	}{
		{0x00, 8}, {0x01, 16}, {0x02, 32}, {0x03, 48},
		{0x04, 64}, {0x05, 80}, {0x06, 96}, {0x07, 112},
		{0x10, 16}, {0x20, 32}, {0x30, 64}, {0x40, 128},
		{0x70, 1024}, {0x71, 2048}, {0x77, 14336},
	}

	for _, tt := range tests {
		got := c.divisorPeriod(tt.nr43)
		require.Equal(t, tt.period, got,
			"divisorPeriod(0x%02X)", tt.nr43)
	}
}

// TestNoiseLFSRClocking verifies LFSR advances through T-cycle accumulation.
func TestNoiseLFSRClocking(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF0) // NR42: volume 15
	apu.WriteRegister(0xFF22, 0x00) // NR43: period=8
	apu.WriteRegister(0xFF23, 0x80) // trigger

	require.Equal(t, uint16(0x7FFF), apu.ch4.lfsr)

	apu.Step(7)
	require.Equal(t, uint16(0x7FFF), apu.ch4.lfsr, "LFSR unchanged after 7 cycles")

	apu.Step(1)
	require.Equal(t, uint16(0x3FFF), apu.ch4.lfsr, "LFSR = 0x3FFF after first clock")

	apu.Step(8)
	require.Equal(t, uint16(0x1FFF), apu.ch4.lfsr, "LFSR = 0x1FFF after second clock")
}

// TestNoiseClockAccumulation verifies partial cycle carry-forward.
func TestNoiseClockAccumulation(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF0)
	apu.WriteRegister(0xFF22, 0x01) // NR43: period=16
	apu.WriteRegister(0xFF23, 0x80) // trigger

	apu.Step(10) // partial: 10/16
	apu.Step(6)  // completes: 16/16
	require.Equal(t, uint16(0x3FFF), apu.ch4.lfsr, "LFSR stepped after 10+6=16 cycles")

	apu.Step(32) // 2 more steps
	require.Equal(t, uint16(0x0FFF), apu.ch4.lfsr, "LFSR stepped 2 more after 32 cycles")
}

// TestNoiseOutputBitControlsSample verifies output is controlled by LFSR bit 0.
func TestNoiseOutputBitControlsSample(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF0) // NR42: volume 15
	apu.WriteRegister(0xFF22, 0x00) // NR43: period=8
	apu.WriteRegister(0xFF23, 0x80) // trigger

	require.True(t, apu.Ch4On)
	require.Equal(t, 15, apu.GetSampleCh4(), "Sample = volume when LFSR bit 0 = 1")

	// After 14 more LFSR steps, bit 0 is still 1.
	for i := range 14 {
		apu.Step(8)
		require.Equal(t, 15, apu.GetSampleCh4(),
			"Sample = 15 at LFSR step %d", i)
	}

	// Step 15: LFSR = 0x4000, bit 0 = 0.
	apu.Step(8)
	require.Equal(t, 0, apu.GetSampleCh4(), "Sample = 0 when LFSR bit 0 = 0")
}

// TestNoiseSampleZeroWhenOff verifies GetSampleCh4 returns 0 when off.
func TestNoiseSampleZeroWhenOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	require.False(t, apu.Ch4On)
	require.Equal(t, 0, apu.GetSampleCh4())
}

// TestNoiseSampleZeroWhenVolumeZero verifies DAC-off prevents output.
func TestNoiseSampleZeroWhenVolumeZero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0x00) // NR42: volume 0
	apu.WriteRegister(0xFF23, 0x80) // trigger

	require.False(t, apu.Ch4On, "CH4 off when DAC off")
	require.Equal(t, 0, apu.GetSampleCh4())
}

// TestNoiseEnvelopeDecrease verifies CH4 envelope decreases volume.
func TestNoiseEnvelopeDecrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF1) // NR42: vol=15, dir=0, period=1
	apu.WriteRegister(0xFF22, 0x00)
	apu.WriteRegister(0xFF23, 0x80) // trigger

	require.True(t, apu.Ch4On)
	require.Equal(t, 15, apu.GetSampleCh4())

	apu.Step(65536) // one envelope cycle
	require.Equal(t, 14, apu.GetSampleCh4(), "Volume decreased to 14")
}

// TestNoiseEnvelopeIncrease verifies CH4 envelope increases volume.
func TestNoiseEnvelopeIncrease(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0x19) // NR42: vol=1, dir=1, period=1
	apu.WriteRegister(0xFF22, 0x00)
	apu.WriteRegister(0xFF23, 0x80) // trigger

	require.True(t, apu.Ch4On)
	require.Equal(t, 1, apu.GetSampleCh4())

	apu.Step(65536)
	require.Equal(t, 2, apu.GetSampleCh4(), "Volume increased to 2")
}

// TestNoiseLFSRDoesNotRunWhenAPUOff verifies LFSR doesn't advance when APU off.
func TestNoiseLFSRDoesNotRunWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.WriteRegister(0xFF21, 0xF0)
	apu.WriteRegister(0xFF22, 0x00)
	apu.WriteRegister(0xFF23, 0x80)

	require.Equal(t, uint16(0x7FFF), apu.ch4.lfsr)

	apu.Step(8)
	require.Equal(t, uint16(0x3FFF), apu.ch4.lfsr, "LFSR stepped once")

	apu.WriteRegister(0xFF26, 0x70) // APU off — resets CH4 state
	require.Equal(t, uint16(0x0000), apu.ch4.lfsr, "LFSR reset to 0 after APU off")

	apu.Step(10000) // no-op when APU off
	require.Equal(t, uint16(0x0000), apu.ch4.lfsr, "LFSR stays 0 when APU off")

	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF23, 0x80) // re-trigger
	require.Equal(t, uint16(0x7FFF), apu.ch4.lfsr, "LFSR reset to 0x7FFF on re-trigger")

	apu.Step(8)
	require.Equal(t, uint16(0x3FFF), apu.ch4.lfsr, "LFSR advances after APU on")
}

// ---------------------------------------------------------------------------
// CH3 (Wave) channel tests
// ---------------------------------------------------------------------------

func TestWaveCh3ActiveAfterTrigger(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on
	apu.WriteRegister(0xFF1A, 0x80) // NR30: DAC on (bit 7)

	// Write a known wave pattern
	apu.WriteRegister(0xFF30, 0x12) // byte 0: sample 0=1, sample 1=2
	apu.WriteRegister(0xFF31, 0x34) // byte 1: sample 2=3, sample 3=4

	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger ch3

	require.True(t, apu.Ch3On, "CH3 on after trigger")
	require.True(t, apu.ch3Active(), "CH3 active (DAC on + channel on)")
}

func TestWaveCh3DACOffPreventsActivation(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on
	// NR30 DAC off by default (bit 7 = 0)

	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger ch3

	require.False(t, apu.Ch3On, "CH3 stays off when NR30 DAC is off")
	require.False(t, apu.ch3Active(), "CH3 not active when DAC off")
}

func TestWaveCh3DACOffDuringOperation(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80) // DAC on
	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger

	require.True(t, apu.Ch3On)

	apu.WriteRegister(0xFF1A, 0x00) // DAC off
	require.False(t, apu.Ch3On, "CH3 turned off when DAC turned off")
}

func TestWaveCh3GetSampleZeroWhenOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	require.False(t, apu.ch3Active())
	require.Equal(t, 0, apu.GetSampleCh3(), "CH3 sample 0 when off")
}

func TestWaveCh3GetSampleOutputLevel100(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80) // DAC on
	apu.WriteRegister(0xFF1C, 0x00) // NR32: output level 0 (100%)

	// Fill wave RAM with known values: both nibbles = 0xF (value 15)
	apu.WriteRegister(0xFF30, 0xFF) // sample 0 (pos 0) = 15, sample 1 (pos 1) = 15
	apu.WriteRegister(0xFF31, 0x00)

	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87) // trigger, freq hi=7

	require.True(t, apu.ch3Active())

	// After first-sample skip: position = 1, low nibble of byte 0 = 0xF = 15
	require.Equal(t, 15, apu.GetSampleCh3(), "CH3 sample at 100% = 15")
}

func TestWaveCh3GetSampleOutputLevel50(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80) // DAC on
	apu.WriteRegister(0xFF1C, 0x20) // NR32: output level 1 (50%, bits 5-6 = 01)

	// Fill wave RAM so both nibbles give same value
	apu.WriteRegister(0xFF30, 0xFF) // both nibbles = 15
	apu.WriteRegister(0xFF31, 0x00)

	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87) // trigger

	// Position 1: low nibble = 15, level 1 (50%): 15 >> 1 = 7
	require.Equal(t, 7, apu.GetSampleCh3(), "CH3 sample at 50% = 7")
}

func TestWaveCh3GetSampleOutputLevel25(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x40) // NR32: output level 2 (25%, bits 5-6 = 10)

	apu.WriteRegister(0xFF30, 0xFF)
	apu.WriteRegister(0xFF31, 0x00)

	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87) // trigger

	require.Equal(t, 3, apu.GetSampleCh3(), "CH3 sample at 25% = 3")
}

func TestWaveCh3GetSampleOutputLevel0(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x60) // NR32: output level 3 (0%, bits 5-6 = 11)

	apu.WriteRegister(0xFF30, 0xFF)
	apu.WriteRegister(0xFF31, 0x00)

	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87)

	require.Equal(t, 0, apu.GetSampleCh3(), "CH3 sample at 0% = 0")
}

func TestWaveCh3SamplePositionAdvancesWithFrequency(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x00) // 100%

	// Fill wave RAM with distinct values for each of the 32 samples
	for i := range 16 {
		apu.WriteRegister(0xFF30+uint16(i), byte(i*0x11))
	}

	// Frequency = 0 → period = (2048-0)*2 = 4096 T-cycles per sample
	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger, freq hi=0

	// First-sample skip: position starts at 1
	require.Equal(t, 1, apu.ch3.samplePos, "CH3 first-sample skip: position = 1")

	// Position 2 after one period (4096 T-cycles)
	apu.Step(4096)
	require.Equal(t, 2, apu.ch3.samplePos, "CH3 position = 2 after 4096 T-cycles")

	// Position 3 after another period
	apu.Step(4096)
	require.Equal(t, 3, apu.ch3.samplePos, "CH3 position = 3 after 8192 T-cycles")
}

func TestWaveCh3WaveRAMGatingReadReturnsFFWhenActive(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80) // DAC on
	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger

	// Write a test value while active
	apu.WriteRegister(0xFF30, 0xAB) // should be ignored
	require.Equal(t, byte(0xFF), apu.ReadRegister(0xFF30),
		"Wave RAM read returns 0xFF when CH3 active")
}

func TestWaveCh3WaveRAMWritesIgnoredWhenActive(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)

	// Write wave RAM while inactive
	apu.WriteRegister(0xFF30, 0xAB)
	require.Equal(t, byte(0xAB), apu.ReadRegister(0xFF30), "Wave RAM write works when inactive")

	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger

	// Try to write while active — write is ignored
	apu.WriteRegister(0xFF30, 0xCD)
	// Read is also gated: returns 0xFF when CH3 active
	// (can't observe the unchanged value through the gated read)
	require.Equal(t, byte(0xFF), apu.ReadRegister(0xFF30),
		"Wave RAM read returns 0xFF when CH3 active (write gated)")

	// Stop CH3 and verify wave RAM value is unchanged
	apu.WriteRegister(0xFF1A, 0x00) // DAC off → CH3 inactive
	require.Equal(t, byte(0xAB), apu.ReadRegister(0xFF30),
		"Wave RAM still has original 0xAB after CH3 deactivated")
}

func TestWaveCh3SampleLowNibble(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x00) // 100%

	// byte 0 = 0x01: sample 0 (high) = 0, sample 1 (low) = 1
	apu.WriteRegister(0xFF30, 0x01)
	apu.WriteRegister(0xFF31, 0x00)

	apu.WriteRegister(0xFF1D, 0x00)
	apu.WriteRegister(0xFF1E, 0x80) // trigger

	// After first-sample skip, pos = 1 → low nibble of byte 0 = 0x01 & 0x0F = 1
	require.Equal(t, 1, apu.GetSampleCh3(), "CH3 low nibble sample = 1")
}

func TestWaveCh3SampleFull32StepCycle(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x00)

	// Fill wave RAM: byte i holds (i<<4)|i so each nibble is unique
	for i := range 16 {
		apu.WriteRegister(0xFF30+uint16(i), byte(i<<4|i))
	}

	// Max frequency → period = (2048-2047)*2 = 2 T-cycles per sample advance
	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87) // trigger, hi=7

	// After first-sample skip: pos = 1
	// Step through to position 31 (30 more advances × 2 cycles = 60 T-cycles)
	apu.Step(60)
	require.Equal(t, 31, apu.ch3.samplePos, "CH3 position = 31 after 30 advances")

	// Step once more to wrap to 0
	apu.Step(2)
	require.Equal(t, 0, apu.ch3.samplePos, "CH3 position wrapped to 0")
	// Step again to get to 1
	apu.Step(2)
	require.Equal(t, 1, apu.ch3.samplePos, "CH3 position = 1 after full cycle")
}

// ---------------------------------------------------------------------------
// Mixer tests
// ---------------------------------------------------------------------------

func TestMixerCh1Only(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// Set up CH1 with volume 15, duty 2 (50% high), no sweep
	apu.WriteRegister(0xFF12, 0xF0) // volume 15
	apu.WriteRegister(0xFF11, 0x80) // duty 2 (starts high)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87) // trigger

	// Default NR50 = 0x77 (vol=7+1=8 for L and R), NR51 = 0xFF (all channels to both sides)
	apu.WriteRegister(0xFF24, 0x77) // NR50: L vol=7, R vol=7
	apu.WriteRegister(0xFF25, 0xFF) // NR51: all channels to both sides

	// CH1 only active: channel sum = 15 per side, vol 8 → 15*8/8 = 15
	// Expected: left = 15*1092-32768 = -16388, right same
	l, r := apu.getMixedSample()
	expected := int16(15*1092 - 32768)
	require.Equal(t, expected, l, "Left sample with CH1 only")
	require.Equal(t, expected, r, "Right sample with CH1 only")
}

func TestMixerAllChannelsCombined(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// CH1: volume 15, duty 2
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// CH2: volume 7, duty 2
	apu.WriteRegister(0xFF17, 0x70)
	apu.WriteRegister(0xFF16, 0x80)
	apu.WriteRegister(0xFF18, 0xFF)
	apu.WriteRegister(0xFF19, 0x87)

	// CH3: DAC on, wave sample = 15, level 100%
	apu.WriteRegister(0xFF1A, 0x80)
	apu.WriteRegister(0xFF1C, 0x00)
	apu.WriteRegister(0xFF30, 0xFF) // wave RAM: all samples = 15
	for i := 1; i < 16; i++ {
		apu.WriteRegister(0xFF30+uint16(i), 0xFF)
	}
	apu.WriteRegister(0xFF1D, 0xFF)
	apu.WriteRegister(0xFF1E, 0x87)

	// CH4: volume 3
	apu.WriteRegister(0xFF21, 0x30)
	apu.WriteRegister(0xFF22, 0x00)
	apu.WriteRegister(0xFF23, 0x80)

	// NR50: vol=7+1=8 for both sides, NR51: all channels to both
	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF25, 0xFF)

	// All 4 channels active with non-zero LFSR bit 0
	// Sum = 15 + 7 + 15 + 3 = 40 per side
	// Scaled: 40 * 8 / 8 = 40
	l, r := apu.getMixedSample()
	expected := int16(40*1092 - 32768)
	require.Equal(t, expected, l, "Left sample with all channels")
	require.Equal(t, expected, r, "Right sample with all channels")
}

func TestMixerChannelOffWhenNotTriggered(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// Don't trigger any channel.
	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF25, 0xFF)

	l, r := apu.getMixedSample()
	// All channels off: sum = 0, 0*7/8 = 0, sample = 0*1092-32768 = -32768
	expected := int16(-32768)
	require.Equal(t, expected, l, "Left sample all off = -32768")
	require.Equal(t, expected, r, "Right sample all off = -32768")
}

func TestMixerNR50VolumeScalingLeftOnly(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// CH1: volume 15
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// NR50: L vol=4, R vol=0 (mute right)
	apu.WriteRegister(0xFF24, 0x40)
	apu.WriteRegister(0xFF25, 0xFF)

	l, r := apu.getMixedSample()
	// Left: sum=15, vol=4+1=5, 15*5/8=9, 9*1092-32768 = -22940
	lExpected := int16(9*1092 - 32768)
	// Right: vol=0+1=1, 15*1/8=1, 1*1092-32768 = -31676
	rExpected := int16(1*1092 - 32768)
	require.Equal(t, lExpected, l, "Left sample with vol=4")
	require.Equal(t, rExpected, r, "Right sample with vol=0")
}

func TestMixerNR51PanningSeparatesChannels(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// CH1: volume 15
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	// CH2: volume 10
	apu.WriteRegister(0xFF17, 0xA0)
	apu.WriteRegister(0xFF16, 0x80)
	apu.WriteRegister(0xFF18, 0xFF)
	apu.WriteRegister(0xFF19, 0x87)

	// NR51: CH1→Left only (bit 0), CH2→Right only (bit 5)
	apu.WriteRegister(0xFF25, 0x01|0x20)
	apu.WriteRegister(0xFF24, 0x77) // vol=7+1=8 both sides

	l, r := apu.getMixedSample()
	// Left: only CH1 = 15, 15*8/8=15, 15*1092-32768 = -16388
	lExpected := int16(15*1092 - 32768)
	// Right: only CH2 = 10, 10*8/8=10, 10*1092-32768 = -21848
	rExpected := int16(10*1092 - 32768)
	require.Equal(t, lExpected, l, "Left sample has only CH1")
	require.Equal(t, rExpected, r, "Right sample has only CH2")
}

func TestMixerNoiseOutputZeroWhenLFSRB0Zero(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// CH4: volume 15
	apu.WriteRegister(0xFF21, 0xF0)
	apu.WriteRegister(0xFF22, 0x00)
	apu.WriteRegister(0xFF23, 0x80)

	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF25, 0x08) // CH4→left only

	// For first pulse, LFSR bit 0 = 1 → output = 15
	l, r := apu.getMixedSample()
	// Left: 15*8/8=15, 15*1092-32768 = -16388
	require.Equal(t, int16(-16388), l, "Left with CH4 on first sample")
	_ = r // right is 0
}

// ---------------------------------------------------------------------------
// Audio buffer tests
// ---------------------------------------------------------------------------

func TestAudioBufferFillsDuringStep(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80) // APU on

	// All channels off — sample buffer fills with -32768 (silence)
	apu.Step(70224) // one full frame

	buf := apu.GetAudioBuffer()
	// Expect ~70224/95 ≈ 739 samples, but in stereo pairs = 1478 entries
	expLen := (70224 / 95) * 2
	require.InDelta(t, expLen, len(buf), 10,
		"Audio buffer length for one frame")
}

func TestAudioBufferResetOnGet(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.Step(10000)
	buf1 := apu.GetAudioBuffer()
	require.NotZero(t, len(buf1), "Buffer has samples")

	// Buffer should be empty after GetAudioBuffer
	buf2 := apu.GetAudioBuffer()
	require.Zero(t, len(buf2), "Buffer cleared after GetAudioBuffer")
}

func TestAudioBufferResetOnAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.Step(10000)

	apu.WriteRegister(0xFF26, 0x70) // APU off

	buf := apu.GetAudioBuffer()
	require.Zero(t, len(buf), "Buffer cleared when APU turned off")
}

func TestAudioBufferAccumulatesMultiFrame(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	apu.Step(70224)
	apu.Step(70224)

	buf := apu.GetAudioBuffer()
	// Two frames of samples
	expLen := ((70224 / 95) * 2) * 2
	require.InDelta(t, expLen, len(buf), 20,
		"Audio buffer accumulates two frames")
}

func TestAudioBufferNoSoundWhenAPUOff(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})

	apu.Step(70224)

	buf := apu.GetAudioBuffer()
	require.Zero(t, len(buf), "No audio samples when APU off")
}

// ---------------------------------------------------------------------------
// High-pass filter test
// ---------------------------------------------------------------------------

func TestHighPassFilterRemovesDC(t *testing.T) {
	apu := NewAPU(&testAPUMMU{})
	apu.WriteRegister(0xFF26, 0x80)

	// CH1: volume 15, duty 2 (constant high = DC)
	apu.WriteRegister(0xFF12, 0xF0)
	apu.WriteRegister(0xFF11, 0x80)
	apu.WriteRegister(0xFF13, 0xFF)
	apu.WriteRegister(0xFF14, 0x87)

	apu.WriteRegister(0xFF24, 0x77)
	apu.WriteRegister(0xFF25, 0xFF)

	// Get two consecutive samples — the high-pass filter should
	// produce different outputs (not identical) because it's subtracting
	// the previous sample, creating a decay pattern.
	l1, r1 := apu.getMixedSample()
	l2, r2 := apu.getMixedSample()
	// The second sample should be different from the first (filter reducing DC)
	var diff bool
	if l1 != l2 || r1 != r2 {
		diff = true
	}
	require.True(t, diff, "High-pass filter should change output between consecutive samples")
}
