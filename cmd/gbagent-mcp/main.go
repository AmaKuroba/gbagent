package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/AmaKuroba/gbagent/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// stubEmulator provides a minimal EmulatorHandle for development.
// Replace with a real emulator bridge once the gb package is complete.
type stubEmulator struct{}

func (s *stubEmulator) GetScreen() [160][144]byte {
	// Return a checkerboard pattern so the screenshot is visually distinct.
	var fb [160][144]byte
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			shade := ((x / 8) + (y / 8)) % 4
			fb[x][y] = byte(shade)
		}
	}
	return fb
}

func main() {
	emu := &stubEmulator{}
	srv := mcp.NewServer(emu)

	// Run on stdio transport (Hermes MCP integration).
	stdioServer := server.NewStdioServer(srv.MCPServer())
	stdioServer.SetErrorLogger(log.New(os.Stderr, "[gbagent-mcp] ", log.LstdFlags))

	fmt.Fprintf(os.Stderr, "gbagent-mcp: starting stdio server\n")
	if err := stdioServer.Listen(context.Background(), os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "gbagent-mcp: error: %v\n", err)
		os.Exit(1)
	}
}
