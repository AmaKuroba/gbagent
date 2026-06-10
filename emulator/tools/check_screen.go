//go:build ignore

package main

import (
	"image"
	"image/color"
	"image/png"
	"os"

	"github.com/AmaKuroba/gbagent/internal/gb"
)

func main() {
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}

	mmu := gb.NewMMU(nil)
	mmu.LoadROM(data)
	ppu := gb.NewPPU(mmu)
	timer := gb.NewTimer(mmu)
	joypad := gb.NewJoypad(mmu)
	cpu := gb.NewCore(mmu)

	mmu.SetPPU(ppu)
	mmu.SetTimer(timer)
	mmu.SetJoypad(joypad)

	frameCycles := 70224
	for frame := 0; frame < 600; frame++ {
		target := cpu.Cycles + uint64(frameCycles)
		for cpu.Cycles < target {
			cycles, _ := cpu.Step()
			ppu.Step(cycles)
			timer.Step(cycles)
			mmu.DMAStep(cycles)
		}
	}

	fb := ppu.GetScreen()

	// Count unique colors used
	colorMap := map[byte]int{}
	for y := 0; y < 144; y++ {
		for x := 0; x < 160; x++ {
			colorMap[fb[x][y]]++
		}
	}

	println("Frame count:", ppu.GetState().FrameCtr)
	println("LY:", ppu.GetState().LY)
	println("Mode:", ppu.GetState().Mode)
	println("Unique colors on screen:", len(colorMap))
	for c, n := range colorMap {
		println("  shade", c, ":", n, "pixels")
	}

	// Save as PNG
	img := image.NewPaletted(image.Rect(0, 0, 160, 144), color.Palette{
		color.RGBA{0xFF, 0xFF, 0xFF, 0xFF},
		color.RGBA{0xAA, 0xAA, 0xAA, 0xFF},
		color.RGBA{0x55, 0x55, 0x55, 0xFF},
		color.RGBA{0x00, 0x00, 0x00, 0xFF},
	})
	for y := 0; y < 144; y++ {
		for x := 0; x < 160; x++ {
			img.SetColorIndex(x, y, fb[x][y])
		}
	}

	f, _ := os.Create("/tmp/gbagent_screen.png")
	png.Encode(f, img)
	f.Close()
	println("Saved /tmp/gbagent_screen.png")
}
