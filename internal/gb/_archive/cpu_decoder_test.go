package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDecodeNOP(t *testing.T) {
	inst := Decode(0x00)
	assert.Equal(t, "NOP", inst.Mnemonic)
	assert.Equal(t, 4, inst.Cycles)
	assert.Equal(t, 0, inst.OperandBytes)
}

func TestDecodeAllOpcodes(t *testing.T) {
	for op := 0; op <= 0xFF; op++ {
		inst := Decode(byte(op))
		assert.NotNil(t, inst, "opcode 0x%02X should decode", op)
		assert.Greater(t, inst.Cycles, 0, "opcode 0x%02X should have non-zero cycles", op)
		assert.NotEmpty(t, inst.Mnemonic, "opcode 0x%02X should have a mnemonic", op)
	}
}

func TestDecodeCB(t *testing.T) {
	// CB prefix opcodes use a second table indexed by the byte after 0xCB
	inst := DecodeCB(0x00) // RLC B
	assert.Equal(t, "RLC B", inst.Mnemonic)
	assert.Equal(t, 8, inst.Cycles)
}

func TestDecodeCBAll(t *testing.T) {
	for op := 0; op <= 0xFF; op++ {
		inst := DecodeCB(byte(op))
		assert.NotNil(t, inst, "CB opcode 0x%02X should decode", op)
		assert.Greater(t, inst.Cycles, 0, "CB opcode 0x%02X should have non-zero cycles", op)
		assert.NotEmpty(t, inst.Mnemonic, "CB opcode 0x%02X should have a mnemonic", op)
	}
}

func TestDecodeKnownOpcodes(t *testing.T) {
	tests := []struct {
		opcode   byte
		expected string
		cycles   int
		operands int
	}{
		{0x00, "NOP", 4, 0},
		{0x01, "LD BC,d16", 12, 2},
		{0x02, "LD (BC),A", 8, 0},
		{0x06, "LD B,d8", 8, 1},
		{0x0E, "LD C,d8", 8, 1},
		{0x11, "LD DE,d16", 12, 2},
		{0x21, "LD HL,d16", 12, 2},
		{0x31, "LD SP,d16", 12, 2},
		{0x3E, "LD A,d8", 8, 1},
		{0x76, "HALT", 4, 0},
		{0xC3, "JP a16", 16, 2},
		{0xCD, "CALL a16", 24, 2},
		{0xC9, "RET", 16, 0},
		{0xE0, "LDH (a8),A", 12, 1},
		{0xF0, "LDH A,(a8)", 12, 1},
		{0xFF, "RST 38H", 16, 0},
	}
	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			inst := Decode(tt.opcode)
			assert.Equal(t, tt.expected, inst.Mnemonic)
			assert.Equal(t, tt.cycles, inst.Cycles)
			assert.Equal(t, tt.operands, inst.OperandBytes)
		})
	}
}
