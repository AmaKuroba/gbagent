package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/AmaKuroba/gbagent/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	romPath := flag.String("rom", "", "Path to Game Boy ROM file")
	flag.Parse()

	if *romPath == "" {
		fmt.Fprintf(os.Stderr, "error: --rom flag is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Create the real emulator with the given ROM.
	emu, err := mcp.NewGBEmulator(*romPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating emulator: %v\n", err)
		os.Exit(1)
	}

	srv := mcp.NewServer(emu)

	// Run on stdio transport (Hermes MCP integration).
	stdioServer := server.NewStdioServer(srv.MCPServer())
	stdioServer.SetErrorLogger(log.New(os.Stderr, "[gbagent-mcp] ", log.LstdFlags))

	fmt.Fprintf(os.Stderr, "gbagent-mcp: starting stdio server (rom=%s)\n", *romPath)
	if err := stdioServer.Listen(context.Background(), os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "gbagent-mcp: error: %v\n", err)
		os.Exit(1)
	}
}
