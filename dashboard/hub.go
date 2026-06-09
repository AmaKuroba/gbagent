package dashboard

import (
	"sync"
)

// Message represents a message sent to/from a WebSocket client.
type Message struct {
	Type int
	Data []byte
}

// Client represents a connected WebSocket client managed by the Hub.
type Client struct {
	hub  *Hub
	send chan Message
}

// Hub manages connected clients and broadcasts messages to them.
type Hub struct {
	mu         sync.RWMutex
	clients    map[*Client]bool
	register   chan *Client
	unregister chan *Client
	stop       chan struct{}
	stopped    chan struct{}
}

// NewHub creates and returns a new Hub.
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		stop:       make(chan struct{}),
		stopped:    make(chan struct{}),
	}
}

// Run starts the hub's event loop, processing client registration and
// unregistration. Must be called as a goroutine. Exits when Stop is called.
func (h *Hub) Run() {
	defer close(h.stopped)
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
		case <-h.stop:
			// Unregister all connected clients.
			h.mu.Lock()
			for client := range h.clients {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
			return
		}
	}
}

// Stop signals the hub's Run loop to shut down and waits for it to finish.
func (h *Hub) Stop() {
	close(h.stop)
	<-h.stopped
}

// BroadcastBinary sends binary data to all connected clients.
// Drops messages for clients with a full send buffer.
func (h *Hub) BroadcastBinary(data []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for client := range h.clients {
		select {
		case client.send <- Message{Type: 2, Data: data}:
		default:
			// Client buffer full; skip.
		}
	}
}

// BroadcastText sends text data to all connected clients.
// Drops messages for clients with a full send buffer.
func (h *Hub) BroadcastText(data []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for client := range h.clients {
		select {
		case client.send <- Message{Type: 1, Data: data}:
		default:
			// Client buffer full; skip.
		}
	}
}
