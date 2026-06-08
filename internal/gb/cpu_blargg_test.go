package gb

import (
	"bytes"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

const vblankCycles = 70224

// blarggSerialOutput runs a Blargg test ROM and captures its serial output.
// It creates a minimal Game Boy setup (CPU + MMU + ROM), runs until the
// test reports "Passed" or "Failed", and reads serial data from the SB
// register (0xFF01). LCD timing (VBlank + LY) and timer hardware are
// emulated minimally to keep the test framework running.
func blarggSerialOutput(t *testing.T, romPath string, timeout time.Duration) string {
	data, err := os.ReadFile(romPath)
	require.NoError(t, err, "failed to read ROM: %s", romPath)

	mmu := NewMMU(nil)
	mmu.LoadROM(data)
	cpu := NewCore(mmu)
	cpu.Reset()

	var serialBuf bytes.Buffer
	done := make(chan string, 1)

	go func() {
		var lastVBlank uint64
		var lastTimerTick uint64

		for {
			// --- LCD timing ---
			vOff := cpu.Cycles - lastVBlank
			if vOff >= vblankCycles {
				mmu.Write(0xFF44, 144)
				mmu.Write(0xFF0F, mmu.Read(0xFF0F)|0x01)
				lastVBlank = cpu.Cycles
			} else if vOff < 4560 {
				ly := vOff / 456
				if ly > 153 {
					ly = 153
				}
				mmu.Write(0xFF44, byte(ly))
			} else {
				ly := (vOff - 4560) / 456
				if ly > 143 {
					ly = 143
				}
				mmu.Write(0xFF44, byte(ly))
			}

			// --- Timer ---
			tac := mmu.Read(0xFF07)
			if tac&0x04 != 0 {
				div := uint64([]uint16{1024, 16, 64, 256}[tac&0x03])
				ticks := cpu.Cycles / div
				if ticks > lastTimerTick {
					tima := mmu.Read(0xFF05)
					for ticks > lastTimerTick {
						if tima == 0xFF {
							tima = mmu.Read(0xFF06)
							mmu.Write(0xFF0F, mmu.Read(0xFF0F)|0x04)
						} else {
							tima++
						}
						lastTimerTick++
					}
					mmu.Write(0xFF05, tima)
				}
			}

			if _, err := cpu.Step(); err != nil {
				done <- serialBuf.String()
				return
			}

			// --- Serial output ---
			if sc := mmu.Read(0xFF02); sc&0x80 != 0 {
				serialBuf.WriteByte(mmu.Read(0xFF01))
				mmu.Write(0xFF02, 0x00)

				out := serialBuf.String()
				if strings.Contains(out, "Passed") || strings.Contains(out, "Failed") {
					done <- out
					return
				}
			}
		}
	}()

	select {
	case result := <-done:
		return result
	case <-time.After(timeout):
		t.Logf("timeout after %s (%d bytes)\n%s", timeout, serialBuf.Len(), serialBuf.String())
		return serialBuf.String()
	}
}

// blarggPass checks that the Blargg test output contains "Passed".
func blarggPass(t *testing.T, output string) {
	t.Logf("Serial output:\n%s", output)
	if strings.Contains(output, "Failed") {
		for _, line := range strings.Split(output, "\n") {
			if strings.Contains(line, "Failed") {
				t.Errorf("Blargg test FAILED: %s", line)
			}
		}
		t.FailNow()
	}
	if !strings.Contains(output, "Passed") {
		t.Errorf("Blargg test did not report Passed — output may be empty or truncated")
	}
}

// ---------------------------------------------------------------------------

func TestBlargg_cpu_instrs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Blargg test in short mode")
	}
	output := blarggSerialOutput(t, "/home/ama/projects/gbagent/testdata/roms/cpu_instrs.gb", 30*time.Second)
	blarggPass(t, output)
}

func TestBlargg_instr_timing(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Blargg test in short mode")
	}
	output := blarggSerialOutput(t, "/home/ama/projects/gbagent/testdata/roms/instr_timing.gb", 30*time.Second)
	blarggPass(t, output)
}

func TestBlargg_halt_bug(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Blargg test in short mode")
	}
	output := blarggSerialOutput(t, "/home/ama/projects/gbagent/testdata/roms/halt_bug.gb", 30*time.Second)
	blarggPass(t, output)
}
