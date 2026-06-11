package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

// MetricsServer accepts WebSocket connections from the Python training driver
// and broadcasts training metrics to connected browser clients.
type MetricsServer struct {
	hub    *Hub
	addr   string
	mux    *http.ServeMux
	joypad *Joypad

	// TakeoverFunc is called when the driver queries takeover state.
	TakeoverFunc func() bool

	// SetTakeoverFunc is called when the driver sets takeover mode.
	SetTakeoverFunc func(bool)

	// GetLastActionFunc returns the last human action (dpad, btn).
	GetLastActionFunc func() (dpad, btn byte)
}

// NewMetricsServer creates a new metrics server.
func NewMetricsServer(hub *Hub, addr string, joypad *Joypad) *MetricsServer {
	s := &MetricsServer{
		hub:    hub,
		addr:   addr,
		mux:    http.NewServeMux(),
		joypad: joypad,
	}
	s.routes()
	return s
}

func (s *MetricsServer) routes() {
	s.mux.HandleFunc("/metrics", s.handleMetrics)
}

// ListenAndServe starts the HTTP server.
func (s *MetricsServer) ListenAndServe() error {
	return http.ListenAndServe(s.addr, s.mux)
}

var metricsUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

func (s *MetricsServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	conn, err := metricsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("metrics ws: upgrade error: %v", err)
		return
	}
	defer conn.Close() //nolint: errcheck

	log.Println("metrics ws: driver connected")
	defer log.Println("metrics ws: driver disconnected")

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			return
		}

		var msg struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params,omitempty"`
		}
		if err := json.Unmarshal(message, &msg); err != nil {
			continue
		}

		switch msg.Method {
		case "report_reward":
			s.handleReportReward(msg.Params)
			s.sendResponse(conn, msg.ID, "ok")
		case "get_takeover":
			result := s.TakeoverFunc != nil && s.TakeoverFunc()
			s.sendResponse(conn, msg.ID, result)
		case "set_takeover":
			var params struct {
				Value bool `json:"value"`
			}
			if err := json.Unmarshal(msg.Params, &params); err == nil && s.SetTakeoverFunc != nil {
				s.SetTakeoverFunc(params.Value)
			}
			s.sendResponse(conn, msg.ID, "ok")
		case "get_last_action":
			if s.GetLastActionFunc != nil {
				dpad, btn := s.GetLastActionFunc()
				s.sendResponse(conn, msg.ID, map[string]any{"dpad": dpad, "btn": btn})
			} else {
				s.sendResponse(conn, msg.ID, map[string]any{"dpad": 0, "btn": 0})
			}
		}
	}
}

func (s *MetricsServer) handleReportReward(params json.RawMessage) {
	var p struct {
		Total     float64            `json:"total"`
		Breakdown map[string]float64 `json:"breakdown,omitempty"`
		Loss      float64            `json:"loss,omitempty"`
		Epsilon   float64            `json:"epsilon,omitempty"`
		Sps       float64            `json:"sps,omitempty"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return
	}
	data, _ := json.Marshal(map[string]any{
		"reward_total": p.Total,
		"breakdown":    p.Breakdown,
		"loss":         p.Loss,
		"epsilon":      p.Epsilon,
		"sps":          p.Sps,
	})
	s.hub.BroadcastText(data)
}

func (s *MetricsServer) sendResponse(conn *websocket.Conn, id any, result any) {
	resp := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	}
	data, _ := json.Marshal(resp)
	conn.WriteMessage(websocket.TextMessage, data) //nolint: errcheck
}
