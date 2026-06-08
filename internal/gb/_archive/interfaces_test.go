package gb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// mockCPU is a compile-check stub for the CPU interface.
type mockCPU struct{ CPU }

// mockMMU is a compile-check stub for the MMU interface.
type mockMMU struct{ MMU }

// mockPPU is a compile-check stub for the PPU interface.
type mockPPU struct{ PPU }

// mockAPU is a compile-check stub for the APU interface.
type mockAPU struct{ APU }

// mockCartridge is a compile-check stub for the Cartridge interface.
type mockCartridge struct{ Cartridge }

func TestInterfacesCompile(t *testing.T) {
	e := &Emulator{
		CPU:       &mockCPU{},
		MMU:       &mockMMU{},
		PPU:       &mockPPU{},
		APU:       &mockAPU{},
		Cartridge: &mockCartridge{},
	}
	assert.NotNil(t, e, "Emulator should be constructable from interface stubs")
}
