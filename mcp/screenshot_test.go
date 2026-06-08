package mcp

import (
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncodeScreenshot_DimensionsAndPixels(t *testing.T) {
	// Build a framebuffer with 4 horizontal bands, one per DMG shade.
	var fb [160][144]byte
	for y := 0; y < 144; y++ {
		shade := byte(y / 36) // 0, 1, 2, 3 — each band is 36 rows tall
		for x := 0; x < 160; x++ {
			fb[x][y] = shade
		}
	}

	result, err := EncodeScreenshot(fb)
	require.NoError(t, err)
	require.Equal(t, 160, result.Width)
	require.Equal(t, 144, result.Height)
	require.Equal(t, "png", result.Format)
	require.True(t, strings.HasPrefix(result.Image, "iVBOR"), "should be valid base64 PNG header")

	// Decode the PNG and verify pixels.
	pngData, err := base64StdDecoder.DecodeString(result.Image)
	require.NoError(t, err)

	img, err := png.Decode(strings.NewReader(string(pngData)))
	require.NoError(t, err)

	bounds := img.Bounds()
	assert.Equal(t, 160, bounds.Dx())
	assert.Equal(t, 144, bounds.Dy())

	// Check a pixel from each shade band using direct Pix access.
	rgbaImg, ok := img.(*image.RGBA)
	require.True(t, ok, "decoded image must be *image.RGBA")

	type check struct {
		x, y int
		want color.RGBA
	}
	checks := []check{
		{0, 0, dmgPalette[0]},   // shade 0
		{0, 40, dmgPalette[1]},  // shade 1 (row 36-71)
		{0, 76, dmgPalette[2]},  // shade 2 (row 72-107)
		{0, 112, dmgPalette[3]}, // shade 3 (row 108-143)
	}

	for _, c := range checks {
		off := rgbaImg.PixOffset(c.x, c.y)
		got := color.RGBA{
			R: rgbaImg.Pix[off+0],
			G: rgbaImg.Pix[off+1],
			B: rgbaImg.Pix[off+2],
			A: rgbaImg.Pix[off+3],
		}
		assert.Equal(t, c.want, got, "pixel at (%d,%d)", c.x, c.y)
	}
}
