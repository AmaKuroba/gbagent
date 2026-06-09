package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// busStub is a minimal memory bus for testing CPU execution.
type busStub struct {
	data [0x10000]byte
}

func (b *busStub) Read(addr uint16) byte               { return b.data[addr] }
func (b *busStub) Write(addr uint16, val byte)          { b.data[addr] = val }
func (b *busStub) Read16(addr uint16) uint16            { return uint16(b.data[addr]) | uint16(b.data[addr+1])<<8 }
func (b *busStub) Write16(addr uint16, val uint16)      { b.data[addr] = byte(val); b.data[addr+1] = byte(val >> 8) }
func (b *busStub) LoadROM(data []byte)                   {}
func (b *busStub) LoadBootROM(data []byte)               {}
func (b *busStub) ReadIF() byte                          { return 0 }
func (b *busStub) ReadIE() byte                          { return 0 }
func (b *busStub) WriteIF(val byte)                      {}
func (b *busStub) SetJoypadButtons(buttons byte)          {}
func (b *busStub) DMAStep(cycles int)                      {}
func (b *busStub) SerialStep(cycles int)                    {}
func (b *busStub) StepDevices(cycles int)                   {}

func NewBusStub() *busStub { return &busStub{} }

func TestExecuteNOP(t *testing.T) {
	c := NewCore(NewBusStub())
	pc := c.PC
	cycles, err := c.Step()
	assert.NoError(t, err)
	assert.Equal(t, 4, cycles)
	assert.Equal(t, pc+1, c.PC, "PC should advance by 1 (no operands)")
}

func TestExecuteLD_BC_d16(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x100, 0x01) // LD BC, d16
	bus.Write(0x101, 0x34) // low byte
	bus.Write(0x102, 0x12) // high byte
	c := NewCore(bus)
	c.PC = 0x100

	_, err := c.Step()
	assert.NoError(t, err)
	assert.Equal(t, uint16(0x1234), c.BC, "BC should be loaded with $1234")
	assert.Equal(t, uint16(0x103), c.PC, "PC should advance by 3 (1 opcode + 2 operands)")
}

func TestExecuteLD_A_d8(t *testing.T) {
	bus := NewBusStub()
	bus.Write(0x100, 0x3E) // LD A, d8
	bus.Write(0x101, 0x42) // value
	c := NewCore(bus)
	c.PC = 0x100

	_, err := c.Step()
	assert.NoError(t, err)
	assert.Equal(t, byte(0x42), c.A(), "A should be loaded with $42")
}
