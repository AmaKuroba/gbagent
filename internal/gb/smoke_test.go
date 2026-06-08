package gb

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// stepFrame runs one full frame (70224 cycles) of the CPU.
func stepFrame(cpu *Core) error {
	for i := 0; i < 70224; i++ {
		if _, err := cpu.Step(); err != nil {
			return err
		}
	}
	return nil
}

func TestPokemonSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping smoke test in short mode")
	}

	data, err := os.ReadFile("testdata/roms/pokemon_red.gb")
	require.NoError(t, err, "read ROM")

	mmu := NewMMU(nil)
	require.NotPanics(t, func() { mmu.LoadROM(data) })

	cpu := NewCore(mmu)

	startCycles := cpu.Cycles
	for i := 0; i < 600; i++ {
		err := stepFrame(cpu)
		require.NoError(t, err, "frame %d", i)
	}
	elapsed := cpu.Cycles - startCycles
	expected := uint64(600) * 70224
	t.Logf("%d frames completed, %d cycles, expected %d, Δ=%d", 600, elapsed, expected, elapsed-expected)
}
