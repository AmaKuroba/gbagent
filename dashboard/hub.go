package dashboard

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
	clients    map[*Client]bool
	register   chan *Client
	unregister chan *Client
}

// NewHub creates and returns a new Hub.
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

// Run starts the hub's event loop, processing client registration and
// unregistration. Must be called as a goroutine.
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.clients[client] = true
		case client := <-h.unregister:
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
		}
	}
}

// BroadcastBinary sends binary data to all connected clients.
// Drops messages for clients with a full send buffer.
func (h *Hub) BroadcastBinary(data []byte) {
	for client := range h.clients {
		select {
		case client.send <- Message{Type: 1, Data: data}:
		default:
			// Client buffer full; skip.
		}
	}
}

// BroadcastText sends text data to all connected clients.
// Drops messages for clients with a full send buffer.
func (h *Hub) BroadcastText(data []byte) {
	for client := range h.clients {
		select {
		case client.send <- Message{Type: 2, Data: data}:
		default:
			// Client buffer full; skip.
		}
	}
}
