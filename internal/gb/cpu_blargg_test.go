package gb

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

const vblankCycles = 70224

// projectRoot walks up from the given dir to find the module root (has go.mod).
func projectRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			panic("go.mod not found")
		}
		dir = parent
	}
}

// testRomPath returns an absolute path to a ROM in testdata/roms/.
func testRomPath(name string) string {
	return filepath.Join(projectRoot(), "testdata", "roms", name)
}

// scanWRAM scans WRAM for "Passed" or "Failed" strings and returns
// the full output buffer content.
func scanWRAM(mmu MMU) string {
	var out string
	found := false
	accum := ""
	for addr := uint16(0xC000); addr < 0xE000; addr++ {
		b := mmu.Read(addr)
		if b == 0x00 || b == 0xFF {
			if len(accum) > 0 {
				out += accum + "\n"
				if strings.Contains(accum, "Passed") || strings.Contains(accum, "Failed") {
					found = true
				}
				accum = ""
			}
			if found {
				break
			}
			continue
		}
		accum += string(b)
	}
	if len(accum) > 0 {
		out += accum + "\n"
	}
	if found {
		return out
	}
	return ""
}

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
			// One frame = 70224 cycles = 154 scanlines * 456 cycles.
			// VBlank period: scanlines 144-153 (10 lines).
			// Visible period: scanlines 0-143 (144 lines).
			// LY wraps after 153.
			vOff := cpu.Cycles - lastVBlank
			if vOff >= vblankCycles {
				// Fire VBlank interrupt and reset frame counter
				mmu.Write(0xFF0F, mmu.Read(0xFF0F)|0x01)
				lastVBlank = cpu.Cycles
				vOff = 0
			}
			scanlines := vOff / 456
			var ly byte
			if scanlines < 10 {
				ly = byte(144 + scanlines) // VBlank: LY 144-153
			} else {
				ly = byte(scanlines - 10) // Visible: LY 0-143
				if ly > 143 {
					ly = 143
				}
			}
			mmu.Write(0xFF44, ly)

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
	output := blarggSerialOutput(t, testRomPath("cpu_instrs.gb"), 30*time.Second)
	blarggPass(t, output)
}

func TestBlargg_instr_timing(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Blargg test in short mode")
	}
	output := blarggSerialOutput(t, testRomPath("instr_timing.gb"), 30*time.Second)
	blarggPass(t, output)
}

func TestBlargg_halt_bug(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Blargg test in short mode")
	}
	// halt_bug.gb uses VRAM/WRAM output, not serial.
	// Use WRAM-based test framework for this ROM.
	output := blarggWRAMOutput(t, testRomPath("halt_bug.gb"), 30*time.Second)
	blarggPass(t, output)
}

// blarggWRAMOutput runs a display-based Blargg test ROM and captures its
// output by scanning WRAM for result strings.
func blarggWRAMOutput(t *testing.T, romPath string, timeout time.Duration) string {
	data, err := os.ReadFile(romPath)
	if err != nil {
		t.Fatalf("failed to read ROM: %s: %v", romPath, err)
	}

	mmu := NewMMU(nil)
	mmu.LoadROM(data)
	cpu := NewCore(mmu)
	cpu.Reset()

	done := make(chan string, 1)
	var lastVBlank uint64

	go func() {
		for {
			// --- LCD timing with correct LY ---
			vOff := cpu.Cycles - lastVBlank
			if vOff >= vblankCycles {
				mmu.Write(0xFF0F, mmu.Read(0xFF0F)|0x01)
				lastVBlank = cpu.Cycles
				vOff = 0
			}
			scanlines := vOff / 456
			var ly byte
			if scanlines < 10 {
				ly = byte(144 + scanlines)
			} else {
				ly = byte(scanlines - 10)
				if ly > 143 {
					ly = 143
				}
			}
			mmu.Write(0xFF44, ly)

			if _, err := cpu.Step(); err != nil {
				done <- scanWRAM(mmu)
				return
			}

			if result := scanWRAM(mmu); result != "" {
				done <- result
				return
			}
		}
	}()

	select {
	case result := <-done:
		return result
	case <-time.After(timeout):
		wram := scanWRAM(mmu)
		t.Logf("timeout after %s", timeout)
		t.Logf("WRAM output: %s", wram)
		return wram
	}
}
