package gb

import (
	"crypto/sha256"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBootROM_DataIntegrity(t *testing.T) {
	assert.Len(t, DMGBootROMData, 256, "DMG boot ROM must be exactly 256 bytes")
	hash := sha256.Sum256(DMGBootROMData[:])
	got := fmt.Sprintf("%x", hash)
	assert.Equal(t, "cf053eccb4ccafff9e67339d4e78e98dce7d1ed59be819d2a1ba2232c6fce1c7", got)
}

func TestBootROM_StartsEnabledAfterLoad(t *testing.T) {
	mmu := NewMMU(nil)
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
	assert.Equal(t, DMGBootROMData[0x00FE], mmu.Read(0x00FE))
	assert.Equal(t, DMGBootROMData[0x00FF], mmu.Read(0x00FF))
}

func TestBootROM_NotAffectedWithoutLoad(t *testing.T) {
	mmu := NewMMU(nil)
	assert.Equal(t, byte(0xFF), mmu.Read(0x0000))
}

func TestBootROM_DisabledByFF50(t *testing.T) {
	mmu := NewMMU(nil)
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
	mmu.Write(0xFF50, 0x01)
	assert.Equal(t, byte(0xFF), mmu.Read(0x0000))
}

func TestBootROM_AfterDisableReadsCartridgeROM(t *testing.T) {
	cartData := make([]byte, 0x0100)
	for i := range cartData {
		cartData[i] = byte(0xA0 + i)
	}
	mmu := NewMMU(&romOnlyCartridge{data: cartData})
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
	mmu.Write(0xFF50, 0x01)
	assert.Equal(t, byte(0xA0), mmu.Read(0x0000))
}

func TestBootROM_OnlyCoversFirst256Bytes(t *testing.T) {
	cartData := make([]byte, 0x8000)
	for i := range cartData {
		cartData[i] = byte(i & 0xFF)
	}
	mmu := NewMMU(&romOnlyCartridge{data: cartData})
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
	assert.Equal(t, byte(0x00), mmu.Read(0x0100))
	assert.Equal(t, byte(0x01), mmu.Read(0x0101))
}

func TestBootROM_FF50OnlyDisablesBootROM(t *testing.T) {
	mmu := NewMMU(nil)
	mmu.LoadBootROM(DMGBootROMData[:])
	mmu.Write(0xFF50, 0x00)
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
	mmu.Write(0xFF50, 0x01)
	assert.NotEqual(t, DMGBootROMData[0], mmu.Read(0x0000))
}

func TestBootROM_LoadBootROMEnables(t *testing.T) {
	mmu := NewMMU(nil)
	assert.Equal(t, byte(0xFF), mmu.Read(0x0000))
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
}

func TestBootROM_ReloadBootROM(t *testing.T) {
	mmu := NewMMU(nil)
	mmu.LoadBootROM(DMGBootROMData[:])
	mmu.Write(0xFF50, 0x01)
	assert.Equal(t, byte(0xFF), mmu.Read(0x0000))
	mmu.LoadBootROM(DMGBootROMData[:])
	assert.Equal(t, DMGBootROMData[0], mmu.Read(0x0000))
}
