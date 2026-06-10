package dashboard

import (
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed static/*
var staticFiles embed.FS

// Server wraps the HTTP server and WebSocket hub for the Game Boy dashboard.
type Server struct {
	hub   *Hub
	addr  string
	mux   *http.ServeMux
	input chan string
	joypad *Joypad

	// TakeoverFunc is called when the dashboard sends a takeover toggle.
	// Set by the caller (main.go) to bridge.SetTakeover.
	TakeoverFunc func(bool)

	// SaveStateFunc is called when the dashboard sends a save_state command.
	// Set by the caller (main.go) to bridge.SaveState.
	SaveStateFunc func(path string) error

	// LoadStateFunc is called when the dashboard sends a load_state command.
	// Set by the caller (main.go) to bridge.LoadState.
	LoadStateFunc func(path string) error
}

// NewServer creates a new dashboard server with the given hub and listen address.
func NewServer(hub *Hub, addr string) *Server {
	s := &Server{
		hub:    hub,
		addr:   addr,
		mux:    http.NewServeMux(),
		input:  make(chan string, 256),
		joypad: &Joypad{},
	}
	s.routes()
	return s
}

// Handler returns the HTTP handler (used for testing with httptest).
func (s *Server) Handler() http.Handler {
	return s.mux
}

// Joypad returns the server's joypad state tracker.
func (s *Server) Joypad() *Joypad {
	return s.joypad
}

// routes sets up HTTP routes for the dashboard.
func (s *Server) routes() {
	subFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("dashboard: failed to get static sub-filesystem: %v", err)
	}
	s.mux.Handle("/", http.FileServer(http.FS(subFS)))
	s.mux.HandleFunc("/ws", s.handleWS)
}

// ListenAndServe starts the HTTP server on the configured address.
func (s *Server) ListenAndServe() error {
	httpServer := &http.Server{
		Addr:    s.addr,
		Handler: s.mux,
	}
	return httpServer.ListenAndServe()
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

// inputMessage is the JSON structure sent by the frontend for keyboard events.
type inputMessage struct {
	Key     string `json:"key"`
	Pressed *bool  `json:"pressed,omitempty"`
}

// handleWS upgrades an HTTP connection to WebSocket and bridges
// the Hub's broadcast channel with the WebSocket connection.
// It also parses incoming keyboard input to update the joypad state.
func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("dashboard: ws upgrade error: %v", err)
		return
	}

	client := &Client{
		hub:  s.hub,
		send: make(chan Message, 4096),
	}
	s.hub.register <- client

	// Write pump: reads from the hub's client.send channel and writes to the
	// WebSocket connection. Exits when client.send is closed (on unregister).
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

	// Read pump: reads messages from the WebSocket connection (keyboard input),
	// updates joypad state, and forwards raw messages to the input channel.
	// Unregisters the client when the connection drops.
	defer func() {
		s.hub.unregister <- client
		if s.TakeoverFunc != nil {
			s.TakeoverFunc(false)
		}
	}()
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		// Forward raw input to the channel.
		select {
		case s.input <- string(message):
		default:
			// Input buffer full; drop.
		}

		// Parse keyboard input and update joypad state.
		var msg inputMessage
		if err := json.Unmarshal(message, &msg); err == nil && msg.Key != "" {
			if bits, ok := keyToButton[msg.Key]; ok {
				if msg.Pressed != nil && !*msg.Pressed {
					s.joypad.Release(bits)
				} else {
					s.joypad.Press(bits)
				}
			}
		}

		// Parse takeover toggle (use *bool so keyboard msgs don't reset it).
		var takeoverMsg struct {
			Takeover *bool `json:"takeover"`
		}
		if err := json.Unmarshal(message, &takeoverMsg); err == nil && takeoverMsg.Takeover != nil && s.TakeoverFunc != nil {
			s.TakeoverFunc(*takeoverMsg.Takeover)
		}

		// Check for save/load state commands from dashboard buttons.
		var stateMsg struct {
			SaveState string `json:"save_state"`
			LoadState string `json:"load_state"`
		}
		if err := json.Unmarshal(message, &stateMsg); err == nil {
			if stateMsg.SaveState != "" && s.SaveStateFunc != nil {
				log.Printf("dashboard: saving state to %s", stateMsg.SaveState)
				if err := s.SaveStateFunc(stateMsg.SaveState); err != nil {
					log.Printf("dashboard: save_state error: %v", err)
				}
			}
			if stateMsg.LoadState != "" && s.LoadStateFunc != nil {
				log.Printf("dashboard: loading state from %s", stateMsg.LoadState)
				if err := s.LoadStateFunc(stateMsg.LoadState); err != nil {
					log.Printf("dashboard: load_state error: %v", err)
				}
			}
		}
	}
}
