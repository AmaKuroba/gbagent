package main

import "sync/atomic"

// Joypad tracks the state of all 8 Game Boy buttons as active-high bits.
// Thread-safe via atomic operations so the dashboard relay loop can read
// the current state without blocking the WebSocket read pump.
type Joypad struct {
	state atomic.Uint32
}

// Joypad button bit constants (active-high internal state).
const (
	JoypadRight  = 0x01
	JoypadLeft   = 0x02
	JoypadUp     = 0x04
	JoypadDown   = 0x08
	JoypadA      = 0x10
	JoypadB      = 0x20
	JoypadSelect = 0x40
	JoypadStart  = 0x80
)

// Press sets the given button bits (pressed).
func (j *Joypad) Press(bits byte) {
	for {
		old := j.state.Load()
		if j.state.CompareAndSwap(old, old|uint32(bits)) {
			return
		}
	}
}

// Release clears the given button bits (released).
func (j *Joypad) Release(bits byte) {
	for {
		old := j.state.Load()
		if j.state.CompareAndSwap(old, old & ^uint32(bits)) {
			return
		}
	}
}

// State returns the current button state as an active-high byte.
// Bit layout: R(0) L(1) U(2) D(3) A(4) B(5) Select(6) Start(7).
func (j *Joypad) State() byte {
	return byte(j.state.Load())
}

// keyToButton maps keyboard key names (from the frontend) to Game Boy
// joypad button bitmasks.
var keyToButton = map[string]byte{
	"w": JoypadUp,
	"s": JoypadDown,
	"a": JoypadLeft,
	"d": JoypadRight,
	"k": JoypadA,
	"j": JoypadB,
	"h": JoypadStart,
	"g": JoypadSelect,
}
