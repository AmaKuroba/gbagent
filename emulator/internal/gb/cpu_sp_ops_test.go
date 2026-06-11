package gb

import (
	"testing"
)

// TestAddSPe8 verifies ADD SP, e8 (0xE8) flag behavior per Pan Docs:
//
//	Z=0, N=0, H=carry from bit 3 of low byte, C=carry from bit 7 of low byte
func TestAddSPe8(t *testing.T) {
	tests := []struct {
		name   string
		sp     uint16
		e8     byte // raw byte from ROM
		wantZ  bool
		wantN  bool
		wantH  bool
		wantC  bool
		wantSP uint16
	}{
		// Basic positive: SP=0x0000, e8=0x01
		{"pos_add_no_carry", 0x0000, 0x01, false, false, false, false, 0x0001},
		// SP=0x000F, e8=0x01 → H=1 (carry from bit 3)
		{"pos_add_half_carry", 0x000F, 0x01, false, false, true, false, 0x0010},
		// SP=0x00FF, e8=0x01 → full carry
		{"pos_add_full_carry", 0x00FF, 0x01, false, false, true, true, 0x0100},
		// Negative: e8=0xFF (-1)
		// SP=0x0010, e=-1: low byte 0x10+0xFF=0x10F, H from bit3: 0x0+0xF=0xF (no carry), C from bit7: carry
		{"neg_sub_one", 0x0010, 0xFF, false, false, false, true, 0x000F},
		// Negative: SP=0x0100, e8=0xFF → no carry flags (0x00+0xFF=0xFF)
		{"neg_no_carry", 0x0100, 0xFF, false, false, false, false, 0x00FF},
		// e8=0x80 (-128)
		{"neg_128", 0x0000, 0x80, false, false, false, false, 0xFF80},
		// SP=0x00F0, e8=0x10 → C but not H
		{"carry_no_half", 0x00F0, 0x10, false, false, false, true, 0x0100},
		// SP+0x7F positive max
		{"pos_max", 0x0000, 0x7F, false, false, false, false, 0x007F},
		// SP=0xFFFF, e8=0x01 → wrap around
		{"wrap_full", 0xFFFF, 0x01, false, false, true, true, 0x0000},
		// SP=0xFF80, e8=0x80 → SP+(-128) = 0xFF00, carry from low byte
		{"neg_128_from_ff80", 0xFF80, 0x80, false, false, false, true, 0xFF00},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bus := NewBusStub()
			bus.Write(0x0100, 0xE8)  // ADD SP, e8 opcode
			bus.Write(0x0101, tt.e8) // e8 operand
			c := NewCore(bus)
			c.SP = tt.sp
			c.PC = 0x0100

			cycles, err := c.Step()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cycles != 16 {
				t.Errorf("expected 16 cycles, got %d", cycles)
			}

			if c.SP != tt.wantSP {
				t.Errorf("SP = 0x%04X, want 0x%04X", c.SP, tt.wantSP)
			}
			if c.flagZ() != tt.wantZ {
				t.Errorf("Z = %v, want %v (flags=0x%02X)", c.flagZ(), tt.wantZ, c.F())
			}
			if c.flagN() != tt.wantN {
				t.Errorf("N = %v, want %v (flags=0x%02X)", c.flagN(), tt.wantN, c.F())
			}
			if c.flagH() != tt.wantH {
				t.Errorf("H = %v, want %v (flags=0x%02X)", c.flagH(), tt.wantH, c.F())
			}
			if c.flagC() != tt.wantC {
				t.Errorf("C = %v, want %v (flags=0x%02X)", c.flagC(), tt.wantC, c.F())
			}
		})
	}
}

// TestLDHLSPe8 verifies LD HL, SP+e8 (0xF8) flag behavior per Pan Docs:
//
//	Z=0, N=0, H=carry from bit 3 of low byte, C=carry from bit 7 of low byte
func TestLDHLSPe8(t *testing.T) {
	tests := []struct {
		name   string
		sp     uint16
		e8     byte
		wantZ  bool
		wantN  bool
		wantH  bool
		wantC  bool
		wantHL uint16
	}{
		{"pos_add_no_carry", 0x0000, 0x01, false, false, false, false, 0x0001},
		{"pos_add_half_carry", 0x000F, 0x01, false, false, true, false, 0x0010},
		{"pos_add_full_carry", 0x00FF, 0x01, false, false, true, true, 0x0100},
		{"neg_sub_one", 0x0010, 0xFF, false, false, false, true, 0x000F},
		{"neg_no_carry", 0x0100, 0xFF, false, false, false, false, 0x00FF},
		{"carry_no_half", 0x00F0, 0x10, false, false, false, true, 0x0100},
		{"neg_128", 0x0000, 0x80, false, false, false, false, 0xFF80},
		{"wrap_full", 0xFFFF, 0x01, false, false, true, true, 0x0000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bus := NewBusStub()
			bus.Write(0x0100, 0xF8)  // LD HL, SP+e8 opcode
			bus.Write(0x0101, tt.e8) // e8 operand
			c := NewCore(bus)
			c.SP = tt.sp
			c.PC = 0x0100

			cycles, err := c.Step()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cycles != 12 {
				t.Errorf("expected 12 cycles, got %d", cycles)
			}

			if c.HL != tt.wantHL {
				t.Errorf("HL = 0x%04X, want 0x%04X", c.HL, tt.wantHL)
			}
			if c.flagZ() != tt.wantZ {
				t.Errorf("Z = %v, want %v (flags=0x%02X)", c.flagZ(), tt.wantZ, c.F())
			}
			if c.flagN() != tt.wantN {
				t.Errorf("N = %v, want %v (flags=0x%02X)", c.flagN(), tt.wantN, c.F())
			}
			if c.flagH() != tt.wantH {
				t.Errorf("H = %v, want %v (flags=0x%02X)", c.flagH(), tt.wantH, c.F())
			}
			if c.flagC() != tt.wantC {
				t.Errorf("C = %v, want %v (flags=0x%02X)", c.flagC(), tt.wantC, c.F())
			}
		})
	}
}
