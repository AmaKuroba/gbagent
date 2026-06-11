package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/gorilla/websocket"
)

// EmulatorClient is a JSON-RPC 2.0 WebSocket client that connects to the
// emulator's JSON-RPC server. It provides typed methods for each RPC call.
type EmulatorClient struct {
	url string

	conn *websocket.Conn
	mu   sync.Mutex
	buf  bytes.Buffer

	reqID   int
	pending map[int]chan json.RawMessage
	lock    sync.Mutex

	done chan struct{}
}

// NewEmulatorClient creates a new client connected to the given WebSocket URL.
func NewEmulatorClient(url string) *EmulatorClient {
	return &EmulatorClient{
		url:     url,
		pending: make(map[int]chan json.RawMessage),
		done:    make(chan struct{}),
	}
}

// Connect establishes the WebSocket connection and starts the read loop.
func (c *EmulatorClient) Connect() error {
	conn, _, err := websocket.DefaultDialer.Dial(c.url, nil)
	if err != nil {
		return fmt.Errorf("emulator ws dial: %w", err)
	}
	c.conn = conn
	go c.readLoop()
	return nil
}

// Close shuts down the client.
func (c *EmulatorClient) Close() {
	close(c.done)
	if c.conn != nil {
		c.conn.Close() //nolint: errcheck
	}
}

func (c *EmulatorClient) readLoop() {
	defer close(c.done)
	for {
		_, msg, err := c.conn.ReadMessage()
		if err != nil {
			return
		}
		var resp struct {
			ID     int              `json:"id"`
			Result json.RawMessage  `json:"result"`
			Error  *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(msg, &resp); err != nil {
			continue
		}
		c.lock.Lock()
		ch, ok := c.pending[resp.ID]
		if ok {
			delete(c.pending, resp.ID)
		}
		c.lock.Unlock()
		if ok {
			ch <- resp.Result
		}
	}
}

func (c *EmulatorClient) call(method string, params any) (json.RawMessage, error) {
	if c.conn == nil {
		return nil, fmt.Errorf("not connected")
	}

	c.lock.Lock()
	c.reqID++
	id := c.reqID
	ch := make(chan json.RawMessage, 1)
	c.pending[id] = ch
	c.lock.Unlock()

	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		req["params"] = params
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	err = c.conn.WriteMessage(websocket.TextMessage, append(data, '\n'))
	c.mu.Unlock()
	if err != nil {
		return nil, err
	}

	result := <-ch
	return result, nil
}

// --- Typed RPC methods ---

// GetScreen returns the current frame as a base64-encoded PNG string.
func (c *EmulatorClient) GetScreen() (string, error) {
	result, err := c.call("get_screen", nil)
	if err != nil {
		return "", err
	}
	var s string
	if err := json.Unmarshal(result, &s); err != nil {
		return "", err
	}
	return s, nil
}

// GetState returns the emulator state (CPU registers, PPU mode, frame count, etc.).
func (c *EmulatorClient) GetState() (map[string]any, error) {
	result, err := c.call("get_state", nil)
	if err != nil {
		return nil, err
	}
	var state map[string]any
	if err := json.Unmarshal(result, &state); err != nil {
		return nil, err
	}
	return state, nil
}

// PressButton presses a Game Boy button by name (e.g. "A", "UP", "START").
func (c *EmulatorClient) PressButton(button string) error {
	_, err := c.call("press_button", map[string]string{"button": button})
	return err
}

// ReleaseButton releases a Game Boy button by name.
func (c *EmulatorClient) ReleaseButton(button string) error {
	_, err := c.call("release_button", map[string]string{"button": button})
	return err
}

// ReleaseAll releases all held buttons.
func (c *EmulatorClient) ReleaseAll() error {
	_, err := c.call("release_all", nil)
	return err
}

// SaveState saves the emulator state to the given path.
func (c *EmulatorClient) SaveState(path string) error {
	_, err := c.call("save_state", map[string]string{"path": path})
	return err
}

// LoadState loads the emulator state from the given path.
func (c *EmulatorClient) LoadState(path string) error {
	_, err := c.call("load_state", map[string]string{"path": path})
	return err
}

// Reset resets the emulator.
func (c *EmulatorClient) Reset() error {
	_, err := c.call("reset", nil)
	return err
}

// ResetState loads the saved start state.
func (c *EmulatorClient) ResetState(path string) error {
	params := map[string]string{}
	if path != "" {
		params["path"] = path
	}
	_, err := c.call("reset_state", params)
	return err
}

// ReadRAM reads one byte from Game Boy RAM.
func (c *EmulatorClient) ReadRAM(addr int) (int, error) {
	result, err := c.call("read_ram", map[string]int{"address": addr})
	if err != nil {
		return 0, err
	}
	var v int
	if err := json.Unmarshal(result, &v); err != nil {
		return 0, err
	}
	return v, nil
}


