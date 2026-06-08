package dashboard

import (
	"embed"
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
}

// NewServer creates a new dashboard server with the given hub and listen address.
func NewServer(hub *Hub, addr string) *Server {
	s := &Server{
		hub:   hub,
		addr:  addr,
		mux:   http.NewServeMux(),
		input: make(chan string, 256),
	}
	s.routes()
	return s
}

// Handler returns the HTTP handler (used for testing with httptest).
func (s *Server) Handler() http.Handler {
	return s.mux
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

// handleWS upgrades an HTTP connection to WebSocket and bridges
// the Hub's broadcast channel with the WebSocket connection.
func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("dashboard: ws upgrade error: %v", err)
		return
	}

	client := &Client{
		hub:  s.hub,
		send: make(chan Message, 256),
	}
	s.hub.register <- client

	// Write pump: reads from the hub's client.send channel and writes to the
	// WebSocket connection. Exits when client.send is closed (on unregister).
	go func() {
		defer conn.Close()
		for msg := range client.send {
			if err := conn.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				break
			}
			if err := conn.WriteMessage(msg.Type, msg.Data); err != nil {
				break
			}
		}
	}()

	// Read pump: reads messages from the WebSocket connection (keyboard input)
	// and forwards them to the server's input channel. Unregisters the client
	// when the connection drops.
	defer func() {
		s.hub.unregister <- client
	}()
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}
		select {
		case s.input <- string(message):
		default:
			// Input buffer full; drop.
		}
	}
}
