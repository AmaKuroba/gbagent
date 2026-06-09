package mcp

import (
	"context"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// CPUState is a snapshot of the CPU registers and flags.
type CPUState struct {
	AF      uint16 `json:"af"`
	BC      uint16 `json:"bc"`
	DE      uint16 `json:"de"`
	HL      uint16 `json:"hl"`
	SP      uint16 `json:"sp"`
	PC      uint16 `json:"pc"`
	IME     bool   `json:"ime"`
	Halted  bool   `json:"halted"`
	Stopped bool   `json:"stopped"`
	Cycles  uint64 `json:"cycles"`
}

// PPUState is a snapshot of PPU registers and timing.
type PPUState struct {
	Mode       int    `json:"mode"`
	LY         byte   `json:"ly"`
	LCDC       byte   `json:"lcdc"`
	STAT       byte   `json:"stat"`
	FrameCount int    `json:"frame_count"`
	Screen     string `json:"screen,omitempty"` // base64 PNG (only filled when get_state's include_screen is true)
}

// TimerState is a snapshot of the timer hardware registers.
type TimerState struct {
	DIV    byte   `json:"div"`
	TIMA   byte   `json:"tima"`
	TMA    byte   `json:"tma"`
	TAC    byte   `json:"tac"`
	Freq   string `json:"freq"`
	Enable bool   `json:"enable"`
}

// APUState is a snapshot of the APU control registers.
type APUState struct {
	NR50 byte `json:"nr50"`
	NR51 byte `json:"nr51"`
	NR52 byte `json:"nr52"`
}

// EmulatorHandle is the interface the MCP server needs from the emulator.
// Concrete implementations bridge to the actual emulator package.
type EmulatorHandle interface {
	GetScreen() [160][144]byte
	PressButton(button string) error
	ReadRAM(addr uint16) byte
	WriteRAM(addr uint16, val byte)
	GetState() *EmulatorState
	SaveState(path string) error
	LoadState(path string) error
}

// EmulatorState holds the full emulator state snapshot.
type EmulatorState struct {
	CPU   CPUState   `json:"cpu"`
	PPU   PPUState   `json:"ppu"`
	Cart  CartState  `json:"cart"`
	Timer TimerState `json:"timer"`
	APU   APUState   `json:"apu"`
}

// CartState holds cartridge metadata.
type CartState struct {
	Title string `json:"title"`
	Type  string `json:"type"`
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

// validButtons contains the set of valid Game Boy button names.
var validButtons = map[string]struct{}{
	"A": {}, "B": {}, "START": {}, "SELECT": {},
	"UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {},
}

func (s *Server) registerTools() {
	// --- get_screenshot ---
	screenshotTool := mcp.NewTool("get_screenshot",
		mcp.WithDescription("Capture the current PPU framebuffer as a base64-encoded PNG"),
	)
	s.mcp.AddTool(screenshotTool, s.handleGetScreenshot)

	// --- press_button ---
	pressBtnTool := mcp.NewTool("press_button",
		mcp.WithDescription("Press a Game Boy button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT)"),
		mcp.WithString("button",
			mcp.Description("Button name: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT"),
			mcp.Required(),
		),
		mcp.WithNumber("hold_cycles",
			mcp.Description("Number of cycles to hold the button (default: 16)"),
		),
	)
	s.mcp.AddTool(pressBtnTool, s.handlePressButton)

	// --- read_ram ---
	readRAMTool := mcp.NewTool("read_ram",
		mcp.WithDescription("Read a byte from Game Boy memory at the given address"),
		mcp.WithNumber("address",
			mcp.Description("Memory address (0x0000-0xFFFF)"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(readRAMTool, s.handleReadRAM)

	// --- write_ram ---
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

	// --- get_state ---
	getStateTool := mcp.NewTool("get_state",
		mcp.WithDescription("Get a snapshot of the full emulator state (CPU, PPU, MMU)"),
		mcp.WithBoolean("include_screen",
			mcp.Description("Include the current screen as a base64 PNG in the response"),
		),
	)
	s.mcp.AddTool(getStateTool, s.handleGetState)

	// --- save_state ---
	saveStateTool := mcp.NewTool("save_state",
		mcp.WithDescription("Save the full emulator state to a file"),
		mcp.WithString("path",
			mcp.Description("File path to save the state"),
			mcp.Required(),
		),
	)
	s.mcp.AddTool(saveStateTool, s.handleSaveState)

	// --- load_state ---
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

	// Normalise and validate button name.
	upper := strings.ToUpper(button)
	if _, ok := validButtons[upper]; !ok {
		return mcp.NewToolResultText(fmt.Sprintf(
			"invalid button: %q (valid: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT)", button)), nil
	}

	err := s.emulator.PressButton(upper)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error pressing %s: %v", upper, err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("pressed %s", upper)), nil
}

func (s *Server) handleReadRAM(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	addr, _ := args["address"].(float64)

	emuAddr := uint16(addr)
	val := s.emulator.ReadRAM(emuAddr)
	return mcp.NewToolResultStructuredOnly(map[string]any{
		"address": emuAddr,
		"value":   val,
	}), nil
}

func (s *Server) handleWriteRAM(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	addr, _ := args["address"].(float64)
	val, _ := args["value"].(float64)

	s.emulator.WriteRAM(uint16(addr), byte(val))
	return mcp.NewToolResultText(fmt.Sprintf("wrote 0x%02X to 0x%04X", byte(val), uint16(addr))), nil
}

func (s *Server) handleGetState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	state := s.emulator.GetState()

	// Handle optional screenshot embedding.
	includeScreen, _ := req.GetArguments()["include_screen"].(bool)
	if includeScreen {
		fb := s.emulator.GetScreen()
		screenshot, err := EncodeScreenshot(fb)
		if err == nil {
			state.PPU.Screen = screenshot.Image
		}
	}

	return mcp.NewToolResultStructuredOnly(state), nil
}

func (s *Server) handleSaveState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	path, _ := args["path"].(string)

	if err := s.emulator.SaveState(path); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error saving state: %v", err)), nil
	}
	return mcp.NewToolResultText(fmt.Sprintf("state saved to %s", path)), nil
}

func (s *Server) handleLoadState(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	path, _ := args["path"].(string)

	if err := s.emulator.LoadState(path); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error loading state: %v", err)), nil
	}
	return mcp.NewToolResultText(fmt.Sprintf("state loaded from %s", path)), nil
}
