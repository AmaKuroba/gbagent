package dashboard

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestHubBroadcast(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	client := &Client{
		hub:  hub,
		send: make(chan Message, 256),
	}
	hub.register <- client

	// Give hub time to process registration
	time.Sleep(10 * time.Millisecond)

	// Broadcast a binary frame (PNG header)
	frame := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	hub.BroadcastBinary(frame)

	select {
	case msg := <-client.send:
		assert.Equal(t, 1, msg.Type, "should be binary message type")
		assert.Equal(t, frame, msg.Data, "should receive the broadcast frame")
	case <-time.After(time.Second):
		t.Fatal("client did not receive message within timeout")
	}

	// Cleanup
	hub.unregister <- client
}

func TestHubMultipleClients(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	const numClients = 5
	clients := make([]*Client, numClients)
	for i := 0; i < numClients; i++ {
		client := &Client{
			hub:  hub,
			send: make(chan Message, 256),
		}
		hub.register <- client
		clients[i] = client
	}

	// Give hub time to process registrations
	time.Sleep(10 * time.Millisecond)

	// Broadcast text message (game state)
	state := []byte(`{"pc":0,"af":0}`)
	hub.BroadcastText(state)

	for i, client := range clients {
		select {
		case msg := <-client.send:
			assert.Equal(t, 2, msg.Type, "should be text message type")
			assert.Equal(t, state, msg.Data, "client %d should receive state", i)
		case <-time.After(time.Second):
			t.Fatalf("client %d did not receive message within timeout", i)
		}
	}

	// Cleanup
	for _, client := range clients {
		hub.unregister <- client
	}
}

func TestHubUnregister(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	client := &Client{
		hub:  hub,
		send: make(chan Message, 256),
	}
	hub.register <- client
	time.Sleep(10 * time.Millisecond)

	// Unregister client
	hub.unregister <- client
	time.Sleep(10 * time.Millisecond)

	// After unregister, send channel should be closed
	_, ok := <-client.send
	assert.False(t, ok, "unregistered client's send channel should be closed")

	// Broadcast after unregister — should not panic, client already removed
	hub.BroadcastBinary([]byte{0x01})
	// No assertion needed beyond no panic
}

func TestHubBroadcastToNoClients(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	// Broadcast with zero clients — should not panic
	hub.BroadcastBinary([]byte{0x01})
	hub.BroadcastText([]byte("hello"))
	// If we get here without panic, test passes
}
