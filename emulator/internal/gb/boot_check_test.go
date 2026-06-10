package gb

import (
	"os"
	"testing"
)

func TestLYVBlankExits(t *testing.T) {
	romPath := "/Users/ama/roms/gb/pokemon_red.gb"
	romData, err := os.ReadFile(romPath)
	if err != nil {
		t.Fatalf("read rom: %v", err)
	}

	cart := NewCartridge(romData)
	mmu := NewMMU(cart)
	ppu := NewPPU(mmu)
	mmu.SetPPU(ppu)
	timer := NewTimer(mmu)
	mmu.SetTimer(timer)
	apu := NewAPU(mmu)
	mmu.SetAPU(apu)
	joypad := NewJoypad(mmu)
	mmu.SetJoypad(joypad)

	mmu.LoadBootROM(DMGBootROMData[:])
	cpu := NewCPU(mmu)
	mmu.SetCPU(cpu)
	cpu.PC = 0x0000

	// Run for many frames, track if PC ever leaves the VBlank loop
	// VBlank loop addresses: 0x0064-0x0068

	inLoop := 0
	exitedLoop := false
	maxIter := uint64(70224 * 120)

	for cycles := uint64(0); cycles < maxIter && !exitedLoop; cycles++ {
		lastPC := cpu.PC
		if _, err := cpu.Step(); err != nil {
			t.Fatalf("cpu.Step failed: %v", err)
		}

		// Check if PC moves past the VBlank wait loop.
		// The loop is at 0x0064-0x0068: LDH A,($44) / CP $90 / JR NZ.
		// When JR NZ falls through (not taken), PC lands at 0x006A.
		if lastPC >= 0x0064 && lastPC <= 0x0068 {
			inLoop++
			if cpu.PC >= 0x006A {
				exitedLoop = true
				t.Logf("Exited VBlank loop: PC 0x%04X -> 0x%04X at cycle %d (loop count=%d)",
					lastPC, cpu.PC, cpu.Cycles, inLoop)
			}
		}
	}

	if !exitedLoop {
		t.Errorf("NEVER exited VBlank loop after %d iterations (max %d)", inLoop, maxIter)
	}

	// Now check if boot ROM progresses
	ps := ppu.GetState()
	t.Logf("After exit: PC=0x%04X LY=%d Frame=%d", cpu.PC, ps.LY, ps.FrameCtr)
	t.Logf("CPU: AF=0x%04X BC=0x%04X DE=0x%04X HL=0x%04X SP=0x%04X",
		cpu.AF, cpu.BC, cpu.DE, cpu.HL, cpu.SP)
	bootRO := mmu.Read(0xFF50)
	t.Logf("BootROM enabled: %d", bootRO)
}
