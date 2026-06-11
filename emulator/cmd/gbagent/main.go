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
	"path/filepath"
	"time"

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
	for x := range 160 {
		for y := range 144 {
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

// savPath derives the battery save file path from a ROM path.
func savPath(romPath string) string {
	return romPath[:len(romPath)-len(filepath.Ext(romPath))] + ".sav"
}

// loadRAM reads a .sav file from disk and restores cartridge RAM.
func loadRAM(romPath string, cart gb.Cartridge) {
	if !cart.HasBattery() {
		return
	}
	path := savPath(romPath)
	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("load battery RAM: error reading %s: %v", path, err)
		}
		return
	}
	cart.LoadRAM(data)
	log.Printf("loaded battery-backed RAM from %s (%d bytes)", path, len(data))
}

func createBridge(romPath string) *mcpBridge {
	romData, err := os.ReadFile(romPath)
	if err != nil {
		log.Fatalf("failed to read ROM: %v", err)
	}

	cart := gb.NewCartridge(romData)
	loadRAM(romPath, cart)

	mmu := gb.NewMMU(cart)
	ppu := gb.NewPPU(mmu)
	mmu.SetPPU(ppu)
	timer := gb.NewTimer(mmu)
	mmu.SetTimer(timer)
	apu := gb.NewAPU(mmu)
	mmu.SetAPU(apu)
	cpu := gb.NewCPU(mmu)
	mmu.SetCPU(cpu)

	joypad := gb.NewJoypad(mmu)
	mmu.SetJoypad(joypad)

	cpu.Reset()
	mmu.LoadBootROM(gb.DMGBootROMData[:])
	cpu.PC = 0x0000

	return newBridge(mmu, cpu, ppu, timer, apu, cart, romPath)
}

func runServe(romPath string, jsonrpcPort int, loadState string) {
	bridge := createBridge(romPath)

	if loadState != "" {
		if err := loadSavedState(bridge, loadState); err != nil {
			log.Fatalf("--load-state: %v", err)
		}
		log.Printf("loaded start state from %s", loadState)
	}

	bridge.startProcessor()

	if jsonrpcPort > 0 {
		go runJSONRPCWebSocket(bridge, jsonrpcPort, loadState)
	}

	if loadState != "" {
		time.Sleep(time.Second / 60)
	}

	frameTicker := time.NewTicker(time.Second / 60)
	defer frameTicker.Stop()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	log.Printf("gbagent: emulation running (%s)", bridge.cart.GetTitle())
	var frameCount int
	for {
		select {
		case <-sigCh:
			bridge.stopProcessor()
			log.Println("gbagent: shutting down")
			return
		case <-frameTicker.C:
			bridge.runFrame()

			// Advance MBC3 RTC once per second (every 60 frames).
			frameCount++
			if frameCount%60 == 0 {
				bridge.cart.TickRTC(1)
			}
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: gbagent <command> [flags]\n\nCommands:\n  serve     Start the emulation server\n\n")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
		romPath := serveCmd.String("rom", "", "Path to Game Boy ROM file (required)")
		jsonrpcPort := serveCmd.Int("jsonrpc-port", 8767, "JSON-RPC WebSocket port (0 = disable)")
		loadState := serveCmd.String("load-state", "", "Path to a pre-saved state to load after boot (skips intro)")
		serveCmd.Parse(os.Args[2:]) //nolint: errcheck

		if *romPath == "" {
			fmt.Fprintf(os.Stderr, "Usage: gbagent serve --rom <rom.gb> [--jsonrpc-port 8767] [--load-state <state.sav>]\n")
			os.Exit(1)
		}
		runServe(*romPath, *jsonrpcPort, *loadState)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %q\n\nUsage: gbagent <command> [flags]\n\nCommands:\n  serve     Start the emulation server\n\n", os.Args[1])
		os.Exit(1)
	}
}
