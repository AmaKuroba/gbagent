//go:build ignore
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/AmaKuroba/gbagent/internal/gb"
)

func testRomPath(name string) string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "testdata", "roms", name)
}

func main() {
	romPath := testRomPath("halt_bug.gb")
	data, err := os.ReadFile(romPath)
	if err != nil {
		panic(err)
	}

	mmu := gb.NewMMU(nil)
	mmu.LoadROM(data)
	cpu := gb.NewCore(mmu)
	cpu.Reset()

	fmt.Printf("Starting PC=0x%04X\n", cpu.GetState().PC)

	for i := 0; i < 500000; i++ {
		// LCD timing - VBlank
		cycles := cpu.StepCycles()
		if cycles%70224 < 4 {
			mmu.Write(0xFF0F, mmu.Read(0xFF0F)|0x01)
		}

		if _, err := cpu.Step(); err != nil {
			fmt.Printf("Error after %d steps: %v\n", i, err)
			break
		}

		// Serial output
		if sc := mmu.Read(0xFF02); sc&0x80 != 0 {
			b := mmu.Read(0xFF01)
			fmt.Printf("SERIAL[%d]: %c (0x%02X)\n", i, b, b)
			mmu.Write(0xFF02, 0x00)
		}

		state := cpu.GetState()
		if state.PC <= 0x0100 && i > 100 {
			fmt.Printf("Step %d: PC=0x%04X (possible reset/loop)\n", i, state.PC)
			if i > 10000 {
				break
			}
		}
	}

	state := cpu.GetState()
	fmt.Printf("Final: PC=0x%04X, Halted=%v, Cycles=%d\n", state.PC, state.Halted, state.Cycles)
}
