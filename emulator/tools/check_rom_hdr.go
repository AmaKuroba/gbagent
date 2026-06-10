//go:build ignore
package main

import (
	"fmt"
	"os"
)

func main() {
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	fmt.Printf("Size: %d bytes\n", len(data))
	fmt.Printf("Cartridge type (0x147): 0x%02X\n", data[0x147])
	fmt.Printf("ROM size code (0x148): 0x%02X\n", data[0x148])
	fmt.Printf("RAM size code (0x149): 0x%02X\n", data[0x149])
	fmt.Printf("Title: %q\n", string(data[0x134:0x143]))
}
