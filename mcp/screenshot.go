package mcp

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
)

// DMG classic green palette (4 shades).
var dmgPalette = [4]color.RGBA{
	{0x9B, 0xBC, 0x0F, 0xFF}, // 0 — lightest
	{0x8B, 0xAC, 0x0F, 0xFF}, // 1
	{0x30, 0x62, 0x30, 0xFF}, // 2
	{0x0F, 0x38, 0x0F, 0xFF}, // 3 — darkest
}

// ScreenshotResult holds the encoded screenshot metadata.
type ScreenshotResult struct {
	Image  string `json:"image"`  // base64-encoded PNG
	Format string `json:"format"` // "png"
	Width  int    `json:"width"`  // 160
	Height int    `json:"height"` // 144
}

// EncodeScreenshot converts a [160][144]byte PPU framebuffer (palette indices 0-3)
// into a base64-encoded PNG image.
func EncodeScreenshot(fb [160][144]byte) (*ScreenshotResult, error) {
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
		return nil, err
	}

	return &ScreenshotResult{
		Image:  base64.StdEncoding.EncodeToString(buf.Bytes()),
		Format: "png",
		Width:  160,
		Height: 144,
	}, nil
}

// base64StdDecoder is a convenience reference for tests in this package.
var base64StdDecoder = base64.StdEncoding
