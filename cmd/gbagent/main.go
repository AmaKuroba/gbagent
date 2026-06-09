package main

import (
	"bytes"
	"encoding/binary"
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

	"github.com/AmaKuroba/gbagent/dashboard"
	"github.com/AmaKuroba/gbagent/internal/gb"
	"github.com/AmaKuroba/gbagent/mcp"
	"github.com/mark3labs/mcp-go/server"
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

// savPath derives the battery save file path from a ROM path.
// e.g. "roms/pokemon.gb" → "roms/pokemon.sav"
func savPath(romPath string) string {
	return romPath[:len(romPath)-len(filepath.Ext(romPath))] + ".sav"
}

// saveRAM writes battery-backed cartridge RAM to a .sav file.
func saveRAM(romPath string, cart gb.Cartridge, label string) {
	if !cart.HasBattery() {
		return
	}
	ramData := cart.SaveRAM()
	if ramData == nil {
		return
	}
	path := savPath(romPath)
	if err := os.WriteFile(path, ramData, 0644); err != nil {
		log.Printf("%s: failed to save battery RAM: %v", label, err)
	} else {
		log.Printf("%s: saved battery-backed RAM to %s", label, path)
	}
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

func runServe(romPath string, port int, mcpPort int) {
	// Load ROM
	romData, err := os.ReadFile(romPath)
	if err != nil {
		log.Fatalf("failed to read ROM: %v", err)
	}

	// Initialize emulator
	cart := gb.NewCartridge(romData)

	// Load battery-backed RAM from disk if it exists.
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

	// Create and attach the joypad register handler
	joypad := gb.NewJoypad(mmu)
	mmu.SetJoypad(joypad)

	// Set initial CPU registers (matching DMG post-boot-ROM state)
	cpu.Reset()
	// Load the DMG boot ROM data so the Nintendo logo copies to VRAM,
	// then the boot ROM jumps to the cartridge entry point at $0100.
	mmu.LoadBootROM(gb.DMGBootROMData[:])
	cpu.PC = 0x0000

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

	// Create MCP bridge and SSE server.
	bridge := newBridge(mmu, cpu, ppu, timer, apu, cart, romPath, hub)

	mcpServer := mcp.NewServer(bridge)
	sseServer := server.NewSSEServer(
		mcpServer.MCPServer(),
		server.WithBaseURL(fmt.Sprintf("http://localhost:%d", mcpPort)),
	)
	go func() {
		log.Printf("gbagent MCP server SSE: http://localhost:%d/sse", mcpPort)
		if err := sseServer.Start(fmt.Sprintf(":%d", mcpPort)); err != nil {
			log.Fatalf("MCP server error: %v", err)
		}
	}()

	// Emulation loop at ~30fps
	frameTicker := time.NewTicker(time.Second / 60)
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
			// Process pending MCP commands before this frame.
			bridge.processPending()

			// Push current joypad state into the emulator's MMU so the
			// game reads the pressed buttons via I/O register 0xFF00.
			mmu.SetJoypadButtons(srv.Joypad().State())

			// Step one full frame (70224 T-cycles).
			// Device stepping is handled internally by M-cycle stepping in cpu.Step().
			var cyclesThisFrame int
			for cyclesThisFrame < 70224 {
				cycles, err := cpu.Step()
				if err != nil {
					log.Printf("cpu error: %v", err)
					return
				}
				cyclesThisFrame += cycles
			}

			// Broadcast frame as PNG binary
			fb := ppu.GetScreen()
			if pngData := encodeFrame(fb); pngData != nil {
				hub.BroadcastBinary(append([]byte{0x00}, pngData...))
			}

			// Broadcast game state + joypad state as JSON text
			state := cpu.GetState()
			ppuState := ppu.GetState()
			joypadBits := srv.Joypad().State()
			stateJSON := fmt.Sprintf(
				`{"pc":%d,"af":%d,"bc":%d,"de":%d,"hl":%d,"sp":%d,"ime":%t,"frame":%d,"joypad":%d}`,
				state.PC, state.AF, state.BC, state.DE, state.HL, state.SP,
				state.IME, ppuState.FrameCount, joypadBits,
			)
			hub.BroadcastText([]byte(stateJSON))

		// Broadcast accumulated audio samples as a binary message.
		if audioBuf := apu.GetAudioBuffer(); len(audioBuf) > 0 {
			b := make([]byte, 2+len(audioBuf)*2) // 2-byte sample count + raw int16 LE
			binary.LittleEndian.PutUint16(b[:2], uint16(len(audioBuf)/2))
			for i, s := range audioBuf {
				binary.LittleEndian.PutUint16(b[2+i*2:], uint16(s))
			}
			hub.BroadcastBinary(append([]byte{0x01}, b...))
		}
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
		mcpPort := serveCmd.Int("mcp-port", 8766, "MCP SSE server port")
		serveCmd.Parse(os.Args[2:])

		if *romPath == "" {
			fmt.Fprintf(os.Stderr, "Usage: gbagent serve --rom <rom.gb> [--port 8765] [--mcp-port 8766]\n")
			os.Exit(1)
		}
		runServe(*romPath, *port, *mcpPort)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %q\n\nUsage: gbagent <command> [flags]\n\nCommands:\n  serve   Start the dashboard server with emulation\n\n", os.Args[1])
		os.Exit(1)
	}
}
