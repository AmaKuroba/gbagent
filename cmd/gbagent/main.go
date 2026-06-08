package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/AmaKuroba/gbagent/dashboard"
	"github.com/AmaKuroba/gbagent/internal/gb"
)

// DMG classic green palette (4 shades).
var dmgPalette = [4]color.RGBA{
	{0x9B, 0xBC, 0x0F, 0xFF}, // 0 — lightest
	{0x8B, 0xAC, 0x0F, 0xFF}, // 1
	{0x30, 0x62, 0x30, 0xFF}, // 2
	{0x0F, 0x38, 0x0F, 0xFF}, // 3 — darkest
}

// encodeFrame converts a PPU framebuffer (palette indices 0-3) to raw PNG bytes.
func encodeFrame(fb [160][144]byte) []byte {
	img := image.NewRGBA(image.Rect(0, 0, 160, 144))
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			idx := fb[x][y] & 0x03
			img.Set(x, y, dmgPalette[idx])
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		log.Printf("encode frame: %v", err)
		return nil
	}
	return buf.Bytes()
}

func runServe(romPath string, port int) {
	// Load ROM
	romData, err := os.ReadFile(romPath)
	if err != nil {
		log.Fatalf("failed to read ROM: %v", err)
	}

	// Initialize emulator
	cart := gb.NewCartridge(romData)
	mmu := gb.NewMMU(cart)
	ppu := gb.NewPPU(mmu)
	mmu.SetPPU(ppu)
	cpu := gb.NewCPU(mmu)

	// Set initial CPU state (skip boot ROM)
	cpu.Reset()

	// Start dashboard hub and server
	hub := dashboard.NewHub()
	go hub.Run()

	srv := dashboard.NewServer(hub, fmt.Sprintf(":%d", port))
	go func() {
		log.Printf("gbagent dashboard: http://localhost:%d", port)
		if err := srv.ListenAndServe(); err != nil {
			log.Fatalf("server error: %v", err)
		}
	}()

	// Emulation loop at ~30fps
	frameTicker := time.NewTicker(time.Second / 30)
	defer frameTicker.Stop()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	log.Printf("gbagent: emulation running (%s)", cart.GetTitle())
	for {
		select {
		case <-sigCh:
			log.Println("gbagent: shutting down")
			return
		case <-frameTicker.C:
			// Step one full frame (70224 cycles)
			for i := 0; i < 70224; i++ {
				cycles, err := cpu.Step()
				if err != nil {
					log.Printf("cpu error: %v", err)
					return
				}
				ppu.Step(cycles)
			}

			// Broadcast frame as PNG binary
			fb := ppu.GetScreen()
			if pngData := encodeFrame(fb); pngData != nil {
				hub.BroadcastBinary(pngData)
			}

			// Broadcast game state as JSON text
			state := cpu.GetState()
			ppuState := ppu.GetState()
			stateJSON := fmt.Sprintf(
				`{"pc":%d,"af":%d,"bc":%d,"de":%d,"hl":%d,"sp":%d,"ime":%t,"frame":%d}`,
				state.PC, state.AF, state.BC, state.DE, state.HL, state.SP,
				state.IME, ppuState.FrameCount,
			)
			hub.BroadcastText([]byte(stateJSON))
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: gbagent <command> [flags]\n\nCommands:\n  serve   Start the dashboard server with emulation\n\n")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
		romPath := serveCmd.String("rom", "", "Path to Game Boy ROM file (required)")
		port := serveCmd.Int("port", 8765, "Dashboard HTTP server port")
		serveCmd.Parse(os.Args[2:])

		if *romPath == "" {
			fmt.Fprintf(os.Stderr, "Usage: gbagent serve --rom <rom.gb> [--port 8765]\n")
			os.Exit(1)
		}
		runServe(*romPath, *port)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %q\n\nUsage: gbagent <command> [flags]\n\nCommands:\n  serve   Start the dashboard server with emulation\n\n", os.Args[1])
		os.Exit(1)
	}
}
