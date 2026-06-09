package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
)

var dmgPalette = [4]color.RGBA{
	{0x9B, 0xBC, 0x0F, 0xFF},
	{0x8B, 0xAC, 0x0F, 0xFF},
	{0x30, 0x62, 0x30, 0xFF},
	{0x0F, 0x38, 0x0F, 0xFF},
}

func main() {
	// Simulate an all-zero framebuffer (solid green)
	var fb [160][144]byte
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			fb[x][y] = byte(0) // all palette index 0
		}
	}

	// Same encoding as EncodeScreenshot
	img := image.NewRGBA(image.Rect(0, 0, 160, 144))
	for x := 0; x < 160; x++ {
		for y := 0; y < 144; y++ {
			idx := fb[x][y]
			if int(idx) >= len(dmgPalette) {
				idx = 0
			}
			img.Set(x, y, dmgPalette[idx])
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}

	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("PNG bytes: %d\n", buf.Len())
	fmt.Printf("Base64: %s\n", b64)
}
