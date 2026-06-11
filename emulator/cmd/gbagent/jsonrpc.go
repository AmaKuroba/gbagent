package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/AmaKuroba/gbagent/internal/gb"
	"github.com/gorilla/websocket"
)

// --- JSON-RPC 2.0 types -----------------------------------

const jsonrpcVersion = "2.0"

type jsonrpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// --- Handler -----------------------------------------------

// jsonrpcHandler dispatches JSON-RPC methods against the bridge.
// All read handlers read from the bridge's atomic snapshot (non-blocking).
// All write handlers use the bridge's latch or queue via the exec channel.
type jsonrpcHandler struct {
	bridge *mcpBridge

	mu      sync.Mutex
	latched byte // bitmask of currently pressed buttons

	startStatePath string // set via --load-state; used by reset_state
}

func newJSONRPCHandler(bridge *mcpBridge, startStatePath string) *jsonrpcHandler {
	return &jsonrpcHandler{bridge: bridge, startStatePath: startStatePath}
}

// exec queues a function via the bridge's background processor.
func (h *jsonrpcHandler) exec(fn func() any) any {
	return h.bridge.exec(fn)
}

func (h *jsonrpcHandler) Handle(req jsonrpcRequest) jsonrpcResponse {
	resp := jsonrpcResponse{JSONRPC: jsonrpcVersion, ID: req.ID}

	switch req.Method {
	case "get_screen":
		h.handleGetScreen(req, &resp)
	case "get_takeover":
		resp.Result = h.bridge.IsTakeover()
	case "get_last_action":
		dpad, btn := h.bridge.GetLastAction()
		resp.Result = map[string]any{"dpad": dpad, "btn": btn}
	case "set_takeover":
		var params struct {
			Value bool `json:"value"`
		}
		if err := json.Unmarshal(req.Params, &params); err == nil {
			h.bridge.SetTakeover(params.Value)
		}
		resp.Result = "ok"
	case "report_reward":
		h.handleReportReward(req, &resp)
	case "press_button":
		h.handlePressButton(req, &resp)
	case "release_button":
		h.handleReleaseButton(req, &resp)
	case "release_all":
		h.mu.Lock()
		h.latched = 0
		h.mu.Unlock()
		h.bridge.ClearAllLatched()
		h.bridge.broadcastAction("release_all", "")
		resp.Result = "ok"
	case "read_ram":
		h.handleReadRAM(req, &resp)
	case "read_ram_range":
		h.handleReadRAMRange(req, &resp)
	case "get_state":
		h.handleGetState(req, &resp)
	case "save_state":
		h.handleSaveState(req, &resp)
	case "load_state":
		h.handleLoadState(req, &resp)
	case "reset":
		h.reset()
		resp.Result = "ok"
	case "reset_state":
		h.handleResetState(req, &resp)
	default:
		resp.Error = &rpcError{Code: -32601, Message: fmt.Sprintf("method not found: %s", req.Method)}
	}

	return resp
}

func (h *jsonrpcHandler) reset() {
	h.mu.Lock()
	h.latched = 0
	h.mu.Unlock()
	h.bridge.ClearAllLatched()
	h.bridge.broadcastAction("reset", "")
	h.exec(func() any {
		h.bridge.cpu.Reset()
		h.bridge.ppu.Reset()
		h.bridge.mmu.LoadBootROM(gb.DMGBootROMData[:])
		h.bridge.cpu.PC = 0x0000
		return nil
	})
}

// handleResetState loads the saved start state, optionally at a custom path.
// If no path param is given, falls back to --load-state (set on startup).
func (h *jsonrpcHandler) handleResetState(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Path string `json:"path,omitempty"`
	}
	if req.Params != nil {
		json.Unmarshal(req.Params, &params) //nolint: errcheck
	}

	path := params.Path
	if path == "" {
		path = h.startStatePath
	}
	if path == "" {
		resp.Error = &rpcError{Code: -32602, Message: "no start state path (set --load-state or pass 'path' param)"}
		return
	}

	if err := h.bridge.LoadState(path); err != nil {
		resp.Error = &rpcError{Code: 1, Message: err.Error()}
		return
	}
	resp.Result = "ok"
}

func (h *jsonrpcHandler) handleGetScreen(req jsonrpcRequest, resp *jsonrpcResponse) {
	screen, _, _, _, _ := h.bridge.snap.read()
	s, err := encodeFrameB64(screen)
	if err != nil {
		resp.Error = &rpcError{Code: 1, Message: err.Error()}
	} else {
		resp.Result = s
	}
}

func (h *jsonrpcHandler) handlePressButton(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Button string `json:"button"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Button == "" {
		resp.Error = &rpcError{Code: -32602, Message: "missing 'button' field"}
		return
	}
	bit, ok := btnBits[params.Button]
	if !ok {
		resp.Error = &rpcError{Code: -32602, Message: fmt.Sprintf("unknown button: %s", params.Button)}
		return
	}
	h.mu.Lock()
	h.latched |= bit
	h.mu.Unlock()
	h.bridge.SetLatchedBits(bit)
	h.bridge.broadcastAction("press_button", params.Button)
	resp.Result = "ok"
}

func (h *jsonrpcHandler) handleReleaseButton(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Button string `json:"button"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Button == "" {
		resp.Error = &rpcError{Code: -32602, Message: "missing 'button' field"}
		return
	}
	bit, ok := btnBits[params.Button]
	if !ok {
		resp.Error = &rpcError{Code: -32602, Message: fmt.Sprintf("unknown button: %s", params.Button)}
		return
	}
	h.mu.Lock()
	h.latched &^= bit
	h.mu.Unlock()
	h.bridge.ClearLatchedBits(bit)
	h.bridge.broadcastAction("release_button", params.Button)
	resp.Result = "ok"
}

func (h *jsonrpcHandler) handleReadRAM(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Address int `json:"address"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		resp.Error = &rpcError{Code: -32602, Message: "missing 'address' field"}
		return
	}
	result := h.exec(func() any {
		return int(h.bridge.mmu.Read(uint16(params.Address)))
	})
	resp.Result = result
}

func (h *jsonrpcHandler) handleReadRAMRange(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Address int `json:"address"`
		Length  int `json:"length"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Length < 1 {
		resp.Error = &rpcError{Code: -32602, Message: "invalid params"}
		return
	}
	result := h.exec(func() any {
		vals := make([]int, params.Length)
		for i := range vals {
			vals[i] = int(h.bridge.mmu.Read(uint16(params.Address + i)))
		}
		return vals
	})
	resp.Result = result
}

func (h *jsonrpcHandler) handleGetState(req jsonrpcRequest, resp *jsonrpcResponse) {
	_, cs, ps, ts, fc := h.bridge.snap.read()
	resp.Result = map[string]any{
		"cpu": map[string]any{
			"af": cs.AF, "bc": cs.BC, "de": cs.DE, "hl": cs.HL,
			"sp": cs.SP, "pc": cs.PC, "ime": cs.IME,
			"halted": cs.Halted, "stopped": cs.Stopped, "cycles": cs.Cycles,
		},
		"ppu": map[string]any{
			"mode": ps.Mode, "ly": ps.LY, "lcdc": ps.LCDC,
			"stat": ps.Stat, "frame_count": fc,
		},
		"timer": map[string]any{
			"div": ts.DIV, "tima": ts.TIMA, "tma": ts.TMA, "tac": ts.TAC,
		},
	}
}

func (h *jsonrpcHandler) handleSaveState(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Path == "" {
		resp.Error = &rpcError{Code: -32602, Message: "missing 'path' field"}
		return
	}
	h.bridge.broadcastAction("save_state", params.Path)
	result := h.exec(func() any {
		state := h.bridge.mmu.DumpEmulatorState()
		if err := os.MkdirAll(filepath.Dir(params.Path), 0755); err != nil {
			return err
		}
		f, err := os.Create(params.Path)
		if err != nil {
			return err
		}
		err = gob.NewEncoder(f).Encode(state)
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
		return err
	})
	if err, ok := result.(error); ok {
		resp.Error = &rpcError{Code: 1, Message: err.Error()}
	} else {
		resp.Result = "ok"
	}
}

func (h *jsonrpcHandler) handleLoadState(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Path == "" {
		resp.Error = &rpcError{Code: -32602, Message: "missing 'path' field"}
		return
	}
	h.bridge.broadcastAction("load_state", params.Path)
	result := h.exec(func() any {
		f, err := os.Open(params.Path)
		if err != nil {
			return err
		}
		var state gb.EmulatorState
		err = gob.NewDecoder(f).Decode(&state)
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
		h.bridge.mmu.LoadEmulatorState(state)
		return nil
	})
	if err, ok := result.(error); ok {
		resp.Error = &rpcError{Code: 1, Message: err.Error()}
	} else {
		resp.Result = "ok"
	}
}

func (h *jsonrpcHandler) handleReportReward(req jsonrpcRequest, resp *jsonrpcResponse) {
	var params struct {
		Total     float64            `json:"total"`
		Breakdown map[string]float64 `json:"breakdown,omitempty"`
		Loss      float64            `json:"loss,omitempty"`
		Epsilon   float64            `json:"epsilon,omitempty"`
		Sps       float64            `json:"sps,omitempty"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		resp.Error = &rpcError{Code: -32602, Message: "invalid params"}
		return
	}
	data, _ := json.Marshal(map[string]any{
		"reward_total": params.Total,
		"breakdown":    params.Breakdown,
		"loss":         params.Loss,
		"epsilon":      params.Epsilon,
		"sps":          params.Sps,
	})
	if h.bridge.hub != nil {
		h.bridge.hub.BroadcastText(data)
	}
	resp.Result = "ok"
}

// --- WebSocket server --------------------------------------

// wsWriter is the interface for reading/writing text messages.
type wsWriter interface {
	Read(p []byte) (int, error)
	Write(p []byte) (int, error)
}

// wsConn adapts gorilla/websocket.Conn to wsWriter.
type wsConn struct {
	conn *websocket.Conn
	mu   sync.Mutex
	buf  bytes.Buffer
}

func (w *wsConn) Read(p []byte) (int, error) {
	if w.buf.Len() > 0 {
		return w.buf.Read(p)
	}
	_, msg, err := w.conn.ReadMessage()
	if err != nil {
		return 0, err
	}
	w.buf.Write(msg)
	w.buf.WriteByte('\n')
	return w.buf.Read(p)
}

func (w *wsConn) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(p), w.conn.WriteMessage(websocket.TextMessage, p)
}

// handleWSJSONRPC handles one WebSocket JSON-RPC session.
func handleWSJSONRPC(ws wsWriter, bridge *mcpBridge, startStatePath string) {
	log.Println("jsonrpc ws: client connected")
	defer log.Println("jsonrpc ws: client disconnected")

	handler := newJSONRPCHandler(bridge, startStatePath)
	reader := bufio.NewReader(ws)
	enc := json.NewEncoder(ws)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return
		}
		line = line[:len(line)-1]
		if line == "" {
			continue
		}
		var req jsonrpcRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			enc.Encode(jsonrpcResponse{ //nolint: errcheck
				JSONRPC: jsonrpcVersion, ID: nil,
				Error: &rpcError{Code: -32700, Message: "parse error"},
			})
			continue
		}
		resp := handler.Handle(req)
		if err := enc.Encode(resp); err != nil {
			log.Printf("jsonrpc ws: encode error: %v", err)
			return
		}
	}
}

// runJSONRPCWebSocket starts a WebSocket server for JSON-RPC on the given port.
func runJSONRPCWebSocket(bridge *mcpBridge, port int, startStatePath string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		upgrader := websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("jsonrpc ws: upgrade error: %v", err)
			return
		}
		defer conn.Close() //nolint: errcheck

		ws := &wsConn{conn: conn}
		handleWSJSONRPC(ws, bridge, startStatePath)
	})

	addr := fmt.Sprintf(":%d", port)
	log.Printf("gbagent JSON-RPC WebSocket: ws://localhost:%d/ws", port)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("jsonrpc ws server error: %v", err)
	}
}

// --- helpers -----------------------------------------------

func encodeFrameB64(fb [160][144]byte) (string, error) {
	pngData := encodeFrame(fb)
	if pngData == nil {
		return "", fmt.Errorf("failed to encode screenshot")
	}
	return base64.StdEncoding.EncodeToString(pngData), nil
}
