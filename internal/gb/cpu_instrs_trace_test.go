package gb

import (
	"bytes"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCpuInstrs_BreakpointCpuFast(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	data, err := os.ReadFile(testRomPath("cpu_instrs.gb"))
	require.NoError(t, err)

	mmu := NewMMU(nil)
	mmu.LoadROM(data)
	cpu := NewCore(mmu)
	cpu.Reset()

	var lastVBlank uint64
	var lastTimerTick uint64
	var serialBuf bytes.Buffer
	hitCount := 0

	for i := 0; i < 50000000; i++ {
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

		pcBefore := cpu.PC

		_, err := cpu.Step()
		if err != nil {
			break
		}

		if sc := mmu.Read(0xFF02); sc&0x80 != 0 {
			serialBuf.WriteByte(mmu.Read(0xFF01))
			mmu.Write(0xFF02, 0x00)
		}

		// Breakpoint: LD A,($D800) at cpu_fast entry (0xC2E7)
		if pcBefore == 0xC2E7 || pcBefore == 0xC2E1 || pcBefore == 0xC2F1 {
			hitCount++
			// Read KEY1 ($FF4D) via MMU to see what it returns
			key1Val := mmu.Read(0xFF4D)
			// Dump stack to see who called us
			t.Logf("[%d] BP#%d at 0x%04X, cycle=%d, SP=0x%04X",
				hitCount, hitCount, pcBefore, cpu.Cycles, cpu.SP)
			t.Logf("  AF=0x%04X BC=0x%04X DE=0x%04X HL=0x%04X", cpu.AF, cpu.BC, cpu.DE, cpu.HL)
			t.Logf("  $D800 (gb_id)=0x%02X KEY1(FF4D)=0x%02X", mmu.Read(0xD800), key1Val)
			t.Logf("  Serial: %q", serialBuf.String())
			retAddr := mmu.Read16(cpu.SP)
			t.Logf("  Return address: 0x%04X", retAddr)
		}

		if cpu.Stopped {
			t.Logf("STOP at cycle %d", cpu.Cycles)
			break
		}
	}

	t.Logf("Final serial: %q", serialBuf.String())
}
