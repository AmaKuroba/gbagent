package gb

import (
	"fmt"
	"testing"
)

func TestDebugJoypad(t *testing.T) {
	mmu := &joypadTestMMU{}
	joypad := NewJoypad(mmu)

	t.Logf("initial lastRead=%02x columnSelect=%02x buttons=%02x", joypad.lastRead, joypad.columnSelect, joypad.buttons)

	joypad.WriteRegister(0x10)
	t.Logf("after write: columnSelect=%02x", joypad.columnSelect)

	r1 := joypad.ReadRegister()
	t.Logf("first read: result=%02x lastRead=%02x ifReg=%02x", r1, joypad.lastRead, mmu.ifReg)

	mmu.ifReg = 0x00

	joypad.SetButtons(0x01)
	t.Logf("after SetButtons(0x01): buttons=%02x", joypad.buttons)

	r2 := joypad.ReadRegister()
	t.Logf("second read: result=%02x lastRead=%02x ifReg=%02x", r2, joypad.lastRead, mmu.ifReg)
	t.Logf("low nibble: %02x", r2&0x0F)
	
	fmt.Printf("DEBUG r2=%02x buttons=%02x columnSelect=%02x lastRead=%02x ifReg=%02x\n", r2, joypad.buttons, joypad.columnSelect, joypad.lastRead, mmu.ifReg)
}
