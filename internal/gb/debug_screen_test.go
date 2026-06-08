package gb

import (
	"os"
	"testing"
)

func TestDebugScreenAfterFrames(t *testing.T) {
	if testing.Short() {
		t.Skip("debug test")
	}
	romData, err := os.ReadFile("/home/ama/projects/gbagent/testdata/roms/pokemon_red.gb")
	if err != nil {
		t.Fatal(err)
	}
	mmu := NewMMU(nil)
	mmu.LoadROM(romData)
	mmu.LoadBootROM(DMGBootROMData[:])
	ppu := NewPPU(mmu)
	mmu.SetPPU(ppu)
	cpu := NewCore(mmu)
	cpu.Reset()
	cpu.PC = 0x0000

	// Check frames at different intervals
	checkFrames := []int{1, 5, 10, 30, 60, 100, 200, 300, 400, 500, 600}
	checkIdx := 0

	for frame := 1; frame <= 600; frame++ {
		for remaining := frameCycles; remaining > 0; {
			cycles, err := cpu.Step()
			if err != nil {
				t.Fatalf("CPU error at frame %d: %v", frame, err)
			}
			ppu.Step(cycles)
			remaining -= cycles
		}

		if checkIdx < len(checkFrames) && frame == checkFrames[checkIdx] {
			checkIdx++
			screen := ppu.GetScreen()
			nonZero := 0
			var firstNonZero [2]int
			for x := 0; x < 160 && nonZero < 2; x++ {
				for y := 0; y < 144 && nonZero < 2; y++ {
					if screen[x][y] != 0 {
						firstNonZero[nonZero] = x*1000 + y
						nonZero++
					}
				}
			}
			// Count total non-zero
			nonZero = 0
			for x := 0; x < 160; x++ {
				for y := 0; y < 144; y++ {
					if screen[x][y] != 0 {
						nonZero++
					}
				}
			}
			state := ppu.GetState()
			t.Logf("Frame %3d: non-zero=%5d/%5d mode=%d LY=%3d LCDC=0x%02x cycles=%d frameCtr=%d",
				frame, nonZero, 160*144, state.Mode, state.LY, state.LCDC, cpu.Cycles, state.FrameCount)

			// Check if boot ROM has finished
			if frame == 3 || frame == 5 {
				t.Logf("  PC=0x%04x A=0x%02x B=0x%02x bootEnabled=%v",
					cpu.PC, cpu.A(), cpu.B(), mmu.Read(0xFF50)&0x01 != 0)

				// Read BGP and one VRAM tile
				t.Logf("  BGP=0x%02x VRAM[0x8010]=0x%02x VRAM[0x9800]=0x%02x",
					mmu.Read(0xFF47), mmu.Read(0x8010), mmu.Read(0x9800))
			}
		}
	}
}
