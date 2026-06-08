package mcp

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// EmulatorHandle is the interface the MCP server needs from the emulator.
// Concrete implementations bridge to the actual emulator package.
type EmulatorHandle interface {
	GetScreen() [160][144]byte
	// Future: PressButton, ReadRAM, WriteRAM, GetState, SaveState, LoadState
}

// Server wraps the mcp-go MCPServer with the gbagent emulator.
type Server struct {
	mcp     *server.MCPServer
	emulator EmulatorHandle
}

// NewServer creates a new MCP server with all gbagent tools registered.
func NewServer(emu EmulatorHandle) *Server {
	s := &Server{
		mcp: server.NewMCPServer(
			"gbagent-mcp",
			"0.1.0",
		),
		emulator: emu,
	}

	s.registerTools()
	return s
}

// MCPServer returns the underlying mcp-go server (for use with stdio/SSE transports).
func (s *Server) MCPServer() *server.MCPServer {
	return s.mcp
}

func (s *Server) registerTools() {
	// --- get_screenshot ---
	screenshotTool := mcp.NewTool("get_screenshot",
		mcp.WithDescription("Capture the current PPU framebuffer as a base64-encoded PNG"),
	)
	s.mcp.AddTool(screenshotTool, s.handleGetScreenshot)

	// --- press_button (stub) ---
	pressBtnTool := mcp.NewTool("press_button",
		mcp.WithDescription("Press a Game Boy button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT)"),
		mcp.WithString("button",
			mcp.Description("Button name: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(pressBtnTool, s.handlePressButton)

	// --- read_ram (stub) ---
	readRAMTool := mcp.NewTool("read_ram",
		mcp.WithDescription("Read a byte from Game Boy memory at the given address"),
		mcp.WithNumber("address",
			mcp.Description("Memory address (0x0000-0xFFFF)"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(readRAMTool, s.handleReadRAM)

	// --- write_ram (stub) ---
	writeRAMTool := mcp.NewTool("write_ram",
		mcp.WithDescription("Write a byte to Game Boy memory at the given address"),
		mcp.WithNumber("address",
			mcp.Description("Memory address (0x0000-0xFFFF)"),
			mcp.Required(),
		),
		mcp.WithNumber("value",
			mcp.Description("Byte value (0x00-0xFF)"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(writeRAMTool, s.handleWriteRAM)

	// --- get_state (stub) ---
	getStateTool := mcp.NewTool("get_state",
		mcp.WithDescription("Get a snapshot of the full emulator state (CPU, PPU, MMU)"),
	)
	s.mcp.AddTool(getStateTool, s.handleGetState)

	// --- save_state (stub) ---
	saveStateTool := mcp.NewTool("save_state",
		mcp.WithDescription("Save the full emulator state to a file"),
		mcp.WithString("path",
			mcp.Description("File path to save the state"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(saveStateTool, s.handleSaveState)

	// --- load_state (stub) ---
	loadStateTool := mcp.NewTool("load_state",
		mcp.WithDescription("Load a full emulator state from a file"),
		mcp.WithString("path",
			mcp.Description("File path to load the state from"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(loadStateTool, s.handleLoadState)
}

// --- Handler implementations ---

func (s *Server) handleGetScreenshot(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	fb := s.emulator.GetScreen()

	result, err := EncodeScreenshot(fb)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to encode screenshot: %v", err)), nil
	}

	return mcp.NewToolResultStructuredOnly(result), nil
}

func (s *Server) handlePressButton(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	button, _ := args["button"].(string)
	return mcp.NewToolResultText(fmt.Sprintf("stub: press_button(%s) — not yet implemented", button)), nil
}

func (s *Server) handleReadRAM(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	addr, _ := args["address"].(float64)
	return mcp.NewToolResultText(fmt.Sprintf("stub: read_ram(0x%04X) — not yet implemented", int(addr))), nil
}

func (s *Server) handleWriteRAM(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	addr, _ := args["address"].(float64)
	val, _ := args["value"].(float64)
	return mcp.NewToolResultText(fmt.Sprintf("stub: write_ram(0x%04X, 0x%02X) — not yet implemented", int(addr), int(val))), nil
}

func (s *Server) handleGetState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("stub: get_state() — not yet implemented"), nil
}

func (s *Server) handleSaveState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	path, _ := args["path"].(string)
	return mcp.NewToolResultText(fmt.Sprintf("stub: save_state(%s) — not yet implemented", path)), nil
}

func (s *Server) handleLoadState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	path, _ := args["path"].(string)
	return mcp.NewToolResultText(fmt.Sprintf("stub: load_state(%s) — not yet implemented", path)), nil
}
