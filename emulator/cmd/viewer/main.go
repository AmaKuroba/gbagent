package main

import (
	"embed"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed static/*
var staticFiles embed.FS

// DashboardServer serves the web dashboard UI and bridges the browser
// WebSocket with the emulator and training driver.
type DashboardServer struct {
	hub      *Hub
	addr     string
	mux      *http.ServeMux
	joypad   *Joypad
	emulator *EmulatorClient

	// Takeover state managed by the dashboard, not the emulator.
	takeoverMu  sync.RWMutex
	takeover    bool
	lastActMu   sync.Mutex
	lastActDpad byte
	lastActBtn  byte

	// Connection status reported to the frontend.
	statusMu sync.RWMutex
	status   dashboardStatus
}

type dashboardStatus struct {
	Emulator bool `json:"emulator"`
	Driver   bool `json:"driver"`
}

func NewDashboardServer(hub *Hub, addr string, joypad *Joypad, emulator *EmulatorClient) *DashboardServer {
	s := &DashboardServer{
		hub:      hub,
		addr:     addr,
		mux:      http.NewServeMux(),
		joypad:   joypad,
		emulator: emulator,
	}
	s.routes()
	return s
}

func (s *DashboardServer) routes() {
	subFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("viewer: failed to get static sub-filesystem: %v", err)
	}
	s.mux.Handle("/", http.FileServer(http.FS(subFS)))
	s.mux.HandleFunc("/ws", s.handleWS)
}

func (s *DashboardServer) ListenAndServe() error {
	return http.ListenAndServe(s.addr, s.mux)
}

// SetDriverConnected updates the driver connection status.
func (s *DashboardServer) SetDriverConnected(v bool) {
	s.statusMu.Lock()
	s.status.Driver = v
	s.statusMu.Unlock()
	s.broadcastStatus()
}

// SetEmulatorConnected updates the emulator connection status.
func (s *DashboardServer) SetEmulatorConnected(v bool) {
	s.statusMu.Lock()
	s.status.Emulator = v
	s.statusMu.Unlock()
	s.broadcastStatus()
}

func (s *DashboardServer) broadcastStatus() {
	s.statusMu.RLock()
	status := s.status
	s.statusMu.RUnlock()
	data, _ := json.Marshal(map[string]any{
		"type":     "status",
		"emulator": status.Emulator,
		"driver":   status.Driver,
	})
	s.hub.BroadcastText(data)
}

var dashboardUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

type inputMessage struct {
	Key     string `json:"key"`
	Pressed *bool  `json:"pressed,omitempty"`
}

func (s *DashboardServer) handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := dashboardUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("dashboard ws: upgrade error: %v", err)
		return
	}

	client := &Client{
		hub:  s.hub,
		send: make(chan Message, 4096),
	}
	s.hub.register <- client

	// Send current connection status immediately.
	s.statusMu.RLock()
	status := s.status
	s.statusMu.RUnlock()
	statusData, _ := json.Marshal(map[string]any{
		"type":     "status",
		"emulator": status.Emulator,
		"driver":   status.Driver,
	})
	select {
	case client.send <- Message{Type: 1, Data: statusData}:
	default:
	}

	// Write pump
	go func() {
		defer conn.Close() //nolint: errcheck
		for msg := range client.send {
			if err := conn.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				break
			}
			if err := conn.WriteMessage(msg.Type, msg.Data); err != nil {
				break
			}
		}
	}()

	// Read pump
	defer func() {
		s.hub.unregister <- client
		s.takeoverMu.Lock()
		wasTakeover := s.takeover
		s.takeover = false
		s.takeoverMu.Unlock()
		if wasTakeover {
			s.emulator.ReleaseAll() //nolint: errcheck
		}
	}()
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		// Keyboard input → forward to emulator
		var msg inputMessage
		if err := json.Unmarshal(message, &msg); err == nil && msg.Key != "" {
			if bits, ok := keyToButton[msg.Key]; ok {
				if msg.Pressed != nil && !*msg.Pressed {
					s.joypad.Release(bits)
					s.emulator.ReleaseButton(buttonName(bits)) //nolint: errcheck
				} else {
					s.joypad.Press(bits)
					s.emulator.PressButton(buttonName(bits)) //nolint: errcheck
				}
				s.lastActMu.Lock()
				s.lastActDpad, s.lastActBtn = decodeJoypad(bits)
				s.lastActMu.Unlock()
			}
		}

		// Takeover toggle
		var takeoverMsg struct {
			Takeover *bool `json:"takeover"`
		}
		if err := json.Unmarshal(message, &takeoverMsg); err == nil && takeoverMsg.Takeover != nil {
			s.takeoverMu.Lock()
			s.takeover = *takeoverMsg.Takeover
			s.takeoverMu.Unlock()
			if s.takeover {
				s.emulator.ReleaseAll() //nolint: errcheck
			}
		}

		// Save/load state
		var stateMsg struct {
			SaveState string `json:"save_state"`
			LoadState string `json:"load_state"`
		}
		if err := json.Unmarshal(message, &stateMsg); err == nil {
			if stateMsg.SaveState != "" {
				log.Printf("dashboard: saving state to %s", stateMsg.SaveState)
				if err := s.emulator.SaveState(stateMsg.SaveState); err != nil {
					log.Printf("dashboard: save_state error: %v", err)
				}
			}
			if stateMsg.LoadState != "" {
				log.Printf("dashboard: loading state from %s", stateMsg.LoadState)
				if err := s.emulator.LoadState(stateMsg.LoadState); err != nil {
					log.Printf("dashboard: load_state error: %v", err)
				}
			}
		}
	}
}

// decodeJoypad extracts dpad index (0-4) and btn index (0-4) from a joypad byte.
func decodeJoypad(bits byte) (dpad, btn byte) {
	switch {
	case bits&JoypadRight != 0:
		dpad = 4
	case bits&JoypadLeft != 0:
		dpad = 3
	case bits&JoypadUp != 0:
		dpad = 1
	case bits&JoypadDown != 0:
		dpad = 2
	}
	switch {
	case bits&JoypadA != 0:
		btn = 1
	case bits&JoypadB != 0:
		btn = 2
	case bits&JoypadStart != 0:
		btn = 3
	case bits&JoypadSelect != 0:
		btn = 4
	}
	return
}

// buttonName returns the button name string for a joypad bitmask.
func buttonName(bits byte) string {
	switch bits {
	case JoypadRight:
		return "RIGHT"
	case JoypadLeft:
		return "LEFT"
	case JoypadUp:
		return "UP"
	case JoypadDown:
		return "DOWN"
	case JoypadA:
		return "A"
	case JoypadB:
		return "B"
	case JoypadStart:
		return "START"
	case JoypadSelect:
		return "SELECT"
	default:
		return ""
	}
}

func main() {
	emulatorURL := flag.String("emulator-url", "ws://localhost:8767/ws", "WebSocket URL of the emulator")
	port := flag.Int("port", 8765, "Dashboard HTTP server port")
	metricsPort := flag.Int("metrics-port", 8766, "Metrics WebSocket port for the training driver")
	flag.Parse()

	hub := NewHub()
	go hub.Run()

	joypad := &Joypad{}

	// Connect to emulator (retry until ready)
	emulator := NewEmulatorClient(*emulatorURL)
	go func() {
		log.Printf("connecting to emulator at %s...", *emulatorURL)
		if err := emulator.ConnectWithRetry(30, 2*time.Second); err != nil {
			log.Fatalf("emulator: %v", err)
		}
		log.Printf("connected to emulator")
	}()

	// Dashboard server — starts immediately, works without emulator
	dashboard := NewDashboardServer(hub, fmt.Sprintf(":%d", *port), joypad, emulator)
	go func() {
		log.Printf("dashboard: http://localhost:%d", *port)
		if err := dashboard.ListenAndServe(); err != nil {
			log.Fatalf("dashboard server error: %v", err)
		}
	}()

	// Metrics server for the training driver
	metricsServer := NewMetricsServer(hub, fmt.Sprintf(":%d", *metricsPort), joypad)
	metricsServer.TakeoverFunc = func() bool {
		dashboard.takeoverMu.RLock()
		defer dashboard.takeoverMu.RUnlock()
		return dashboard.takeover
	}
	metricsServer.SetTakeoverFunc = func(v bool) {
		dashboard.takeoverMu.Lock()
		dashboard.takeover = v
		dashboard.takeoverMu.Unlock()
		if v {
			emulator.ReleaseAll() //nolint: errcheck
		}
	}
	metricsServer.GetLastActionFunc = func() (byte, byte) {
		dashboard.lastActMu.Lock()
		defer dashboard.lastActMu.Unlock()
		return dashboard.lastActDpad, dashboard.lastActBtn
	}
	go func() {
		log.Printf("metrics: ws://localhost:%d/metrics", *metricsPort)
		if err := metricsServer.ListenAndServe(); err != nil {
			log.Fatalf("metrics server error: %v", err)
		}
	}()

	// Wait for emulator connection before starting relay
	for !emulator.Connected() {
		time.Sleep(100 * time.Millisecond)
	}
	dashboard.SetEmulatorConnected(true)

	// Relay loop: poll emulator at 60fps, broadcast frames to browser
	frameTicker := time.NewTicker(time.Second / 60)
	defer frameTicker.Stop()

	for range frameTicker.C {
		screenB64, err := emulator.GetScreen()
		if err != nil {
			log.Printf("relay: get_screen error: %v", err)
			continue
		}

		pngData, err := base64.StdEncoding.DecodeString(screenB64)
		if err != nil {
			log.Printf("relay: base64 decode error: %v", err)
			continue
		}
		hub.BroadcastBinary(append([]byte{0x00}, pngData...))
	}
}
