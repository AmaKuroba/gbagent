package dashboard

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServerServesIndex(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	srv := NewServer(hub, ":0")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/")
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.True(t, strings.Contains(resp.Header.Get("Content-Type"), "text/html"))
}

func TestWebSocketUpgrade(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	srv := NewServer(hub, ":0")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// WebSocket dial
	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Connection established — hub should have this client
	require.NotNil(t, conn)
}

func TestWebSocketReceivesBinaryFrame(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	srv := NewServer(hub, ":0")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Give server time to register client
	time.Sleep(10 * time.Millisecond)

	// Broadcast a binary frame
	frame := []byte("fake-png-data")
	hub.BroadcastBinary(frame)

	// Client should receive it as binary message
	msgType, data, err := conn.ReadMessage()
	require.NoError(t, err)
	assert.Equal(t, websocket.BinaryMessage, msgType)
	assert.Equal(t, frame, data)
}

func TestWebSocketReceivesTextState(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	srv := NewServer(hub, ":0")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	time.Sleep(10 * time.Millisecond)

	// Broadcast a text message (game state)
	state := `{"pc":0,"af":0}`
	hub.BroadcastText([]byte(state))

	msgType, data, err := conn.ReadMessage()
	require.NoError(t, err)
	assert.Equal(t, websocket.TextMessage, msgType)
	assert.Equal(t, state, string(data))
}

func TestWebSocketInput(t *testing.T) {
	hub := NewHub()
	go hub.Run()

	// Channel to receive input from WS
	inputCh := make(chan string, 10)
	srv := &Server{
		hub:   hub,
		addr:  ":0",
		mux:   http.NewServeMux(),
		input: inputCh,
	}
	srv.routes()

	ts := httptest.NewServer(srv.mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	time.Sleep(10 * time.Millisecond)

	// Send keyboard input via WebSocket
	err = conn.WriteMessage(websocket.TextMessage, []byte(`{"key":"ArrowRight"}`))
	require.NoError(t, err)

	select {
	case input := <-inputCh:
		assert.Equal(t, `{"key":"ArrowRight"}`, input)
	case <-time.After(time.Second):
		t.Fatal("did not receive input from WS within timeout")
	}
}
