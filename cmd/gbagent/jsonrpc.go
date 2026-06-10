package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/AmaKuroba/gbagent/internal/gb"
	"github.com/gorilla/websocket"
)

// --- JSON-RPC 2.0 types -----------------------------------

type jsonrpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string     `json:"jsonrpc"`
	ID      any        `json:"id"`
	Result  any        `json:"result,omitempty"`
	Error   *rpcError  `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// --- Handler -----------------------------------------------

// jsonrpcHandler dispatches JSON-RPC methods against the bridge.
type jsonrpcHandler struct {
	bridge *mcpBridge

	mu      sync.Mutex
	latched byte // bitmask of currently pressed buttons
	direct  bool // true = call bridge directly (stdio), false = via exec channel (serve/WS)
}

func newJSONRPCHandler(bridge *mcpBridge, direct bool) *jsonrpcHandler {
	return &jsonrpcHandler{bridge: bridge, direct: direct}
}

// exec runs a function on the bridge, either directly or via the exec channel.
func (h *jsonrpcHandler) exec(fn func() any) any {
	if h.direct {
		return h.bridge.execNow(fn)
	}
	return h.bridge.exec(fn)
}

func (h *jsonrpcHandler) Handle(req jsonrpcRequest) jsonrpcResponse {
	resp := jsonrpcResponse{JSONRPC: "2.0", ID: req.ID}

	switch req.Method {
	case "screenshot":
		h.bridge.broadcastAction("screenshot", "")
		result := h.exec(func() any {
			h.bridge.runFrame()
			s, err := encodeFrameB64(h.bridge.ppu.GetScreen())
			if err != nil {
				return err
			}
			return s
		})
		if err, ok := result.(error); ok {
			resp.Error = &rpcError{Code: 1, Message: err.Error()}
		} else {
			resp.Result = result
		}

	case "step_frames":
		var sfParams struct {
			Count   int      `json:"count"`
			Buttons []string `json:"buttons,omitempty"`
		}
		if req.Params != nil {
			if err := json.Unmarshal(req.Params, &sfParams); err != nil {
				resp.Error = &rpcError{Code: -32602, Message: fmt.Sprintf("invalid params: %v", err)}
				return resp
			}
		}
		if sfParams.Count < 1 {
			sfParams.Count = 1
		}
		h.bridge.broadcastAction("step_frames", fmt.Sprintf("%d frames", sfParams.Count))

		result := h.exec(func() any {
			bits := h.latched
			for _, name := range sfParams.Buttons {
				if b, ok := btnBits[name]; ok {
					bits |= b
				}
			}
			h.bridge.mmu.SetJoypadButtons(bits)
			for i := 0; i < sfParams.Count; i++ {
				h.bridge.runFrame()
			}
			h.bridge.mmu.SetJoypadButtons(h.latched)
			s, err := encodeFrameB64(h.bridge.ppu.GetScreen())
			if err != nil {
				return err
			}
			return s
		})
		if err, ok := result.(error); ok {
			resp.Error = &rpcError{Code: 1, Message: err.Error()}
		} else {
			resp.Result = result
		}

	case "press_button":
		var params struct{ Button string `json:"button"` }
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Button == "" {
			resp.Error = &rpcError{Code: -32602, Message: "missing 'button' field"}
			return resp
		}
		bit, ok := btnBits[params.Button]
		if !ok {
			resp.Error = &rpcError{Code: -32602, Message: fmt.Sprintf("unknown button: %s", params.Button)}
			return resp
		}
		h.mu.Lock()
		h.latched |= bit
		h.mu.Unlock()
		h.bridge.broadcastAction("press_button", params.Button)
		resp.Result = "ok"

	case "release_button":
		var params struct{ Button string `json:"button"` }
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Button == "" {
			resp.Error = &rpcError{Code: -32602, Message: "missing 'button' field"}
			return resp
		}
		bit, ok := btnBits[params.Button]
		if !ok {
			resp.Error = &rpcError{Code: -32602, Message: fmt.Sprintf("unknown button: %s", params.Button)}
			return resp
		}
		h.mu.Lock()
		h.latched &^= bit
		h.mu.Unlock()
		h.bridge.broadcastAction("release_button", params.Button)
		resp.Result = "ok"

	case "release_all":
		h.mu.Lock()
		h.latched = 0
		h.mu.Unlock()
		h.bridge.broadcastAction("release_all", "")
		resp.Result = "ok"

	case "read_ram":
		var params struct{ Address int `json:"address"` }
		if err := json.Unmarshal(req.Params, &params); err != nil {
			resp.Error = &rpcError{Code: -32602, Message: "missing 'address' field"}
			return resp
		}
		result := h.exec(func() any {
			return int(h.bridge.mmu.Read(uint16(params.Address)))
		})
		resp.Result = result

	case "read_ram_range":
		var params struct {
			Address int `json:"address"`
			Length  int `json:"length"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Length < 1 {
			resp.Error = &rpcError{Code: -32602, Message: "invalid params"}
			return resp
		}
		result := h.exec(func() any {
			vals := make([]int, params.Length)
			for i := range vals {
				vals[i] = int(h.bridge.mmu.Read(uint16(params.Address + i)))
			}
			return vals
		})
		resp.Result = result

	case "write_ram":
		var params struct {
			Address int `json:"address"`
			Value   int `json:"value"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil {
			resp.Error = &rpcError{Code: -32602, Message: "invalid params"}
			return resp
		}
		h.exec(func() any {
			h.bridge.mmu.Write(uint16(params.Address), byte(params.Value))
			return nil
		})
		resp.Result = "ok"

	case "get_state":
		h.bridge.broadcastAction("get_state", "")
		result := h.exec(func() any {
			h.bridge.runFrame()
			cs := h.bridge.cpu.GetState()
			ps := h.bridge.ppu.GetState()
			ts := h.bridge.timer.GetState()
			return map[string]any{
				"cpu": map[string]any{
					"af": cs.AF, "bc": cs.BC, "de": cs.DE, "hl": cs.HL,
					"sp": cs.SP, "pc": cs.PC, "ime": cs.IME,
					"halted": cs.Halted, "stopped": cs.Stopped, "cycles": cs.Cycles,
				},
				"ppu": map[string]any{
					"mode": ps.Mode, "ly": ps.LY, "lcdc": ps.LCDC,
					"stat": ps.STAT, "frame_count": ps.FrameCount,
				},
				"timer": map[string]any{
					"div": ts.DIV, "tima": ts.TIMA, "tma": ts.TMA, "tac": ts.TAC,
				},
			}
		})
		resp.Result = result

	case "save_state":
		var params struct{ Path string `json:"path"` }
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Path == "" {
			resp.Error = &rpcError{Code: -32602, Message: "missing 'path' field"}
			return resp
		}
		err := h.bridge.SaveState(params.Path)
		if err != nil {
			resp.Error = &rpcError{Code: 1, Message: err.Error()}
		} else {
			resp.Result = "ok"
		}

	case "load_state":
		var params struct{ Path string `json:"path"` }
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Path == "" {
			resp.Error = &rpcError{Code: -32602, Message: "missing 'path' field"}
			return resp
		}
		err := h.bridge.LoadState(params.Path)
		if err != nil {
			resp.Error = &rpcError{Code: 1, Message: err.Error()}
		} else {
			resp.Result = "ok"
		}

	case "reset":
		h.mu.Lock()
		h.latched = 0
		h.mu.Unlock()
		h.bridge.broadcastAction("reset", "")
		h.exec(func() any {
			h.bridge.cpu.Reset()
			h.bridge.ppu.Reset()
			h.bridge.mmu.LoadBootROM(gb.DMGBootROMData[:])
			h.bridge.cpu.PC = 0x0000
			return nil
		})
		resp.Result = "ok"

	default:
		resp.Error = &rpcError{Code: -32601, Message: fmt.Sprintf("method not found: %s", req.Method)}
	}

	return resp
}

// --- stdio server ------------------------------------------

// runJSONRPCStdio reads JSON-RPC from stdin, writes responses to stdout.
func runJSONRPCStdio(bridge *mcpBridge) {
	handler := newJSONRPCHandler(bridge, true) // direct = true, single goroutine
	scanner := bufio.NewScanner(os.Stdin)
	enc := json.NewEncoder(os.Stdout)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		var req jsonrpcRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			log.Printf("jsonrpc: bad request: %v", err)
			resp := jsonrpcResponse{
				JSONRPC: "2.0", ID: nil,
				Error: &rpcError{Code: -32700, Message: "parse error"},
			}
			enc.Encode(resp)
			continue
		}
		resp := handler.Handle(req)
		if err := enc.Encode(resp); err != nil {
			log.Printf("jsonrpc: write error: %v", err)
			return
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("jsonrpc: stdin error: %v", err)
	}
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
func handleWSJSONRPC(ws wsWriter, bridge *mcpBridge) {
	log.Println("jsonrpc ws: client connected")
	defer log.Println("jsonrpc ws: client disconnected")

	handler := newJSONRPCHandler(bridge, false) // direct = false, use exec channel
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
			enc.Encode(jsonrpcResponse{
				JSONRPC: "2.0", ID: nil,
				Error: &rpcError{Code: -32700, Message: "parse error"},
			})
			continue
		}
		resp := handler.Handle(req)
		enc.Encode(resp)
	}
}

// runJSONRPCWebSocket starts a WebSocket server for JSON-RPC on the given port.
func runJSONRPCWebSocket(bridge *mcpBridge, port int) {
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
		defer conn.Close()

		ws := &wsConn{conn: conn}
		handleWSJSONRPC(ws, bridge)
	})

	addr := fmt.Sprintf(":%d", port)
	log.Printf("gbagent JSON-RPC WebSocket: ws://localhost:%d/ws", port)
	if err := http.ListenAndServe(addr, nil); err != nil {
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
