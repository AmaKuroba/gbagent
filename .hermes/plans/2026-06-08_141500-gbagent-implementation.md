# GB Agent — Game Boy Emulator with MCP Interface

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a cycle-accurate Game Boy (DMG/CGB) emulator in Go with a full MCP server interface so AI agents can play, debug, and analyze Game Boy games programmatically.

**Architecture:** A Go module exposing a headless emulator core (CPU, PPU, APU, memory map, MBC, input, save states) wrapped by an MCP-Go server that defines tools for every operation — screenshot (base64 PNG), button input, RAM R/W, save/load, breakpoints. Strict TDD throughout: each component is driven by test ROMs (Blargg, Mooneye, dmg-acid2) and Go unit tests.

**Tech Stack:** Go (1.22+), `github.com/mark3labs/mcp-go` for the MCP server, standard library testing with `testify/assert` for assertions. Test ROMs sourced from Blargg/Mooneye and run via the emulator's own boot path.

**References (awesome-gbdev):**
- [Pan Docs](https://gbdev.github.io/pandocs/) — The single, most comprehensive technical reference
- [The Cycle-Accurate Game Boy Docs](https://github.com/AntonioND/giibiiadvance/blob/master/docs/TCAGBD.pdf) — by AntonioND
- [Complete Technical Reference](https://gekkio.fi/files/gb-docs/gbctr.pdf) — by Gekkio
- [gb-opcodes table](https://gbdev.github.io/gb-opcodes/optables/) — Full opcode reference
- [Blargg's test ROMs](http://gbdev.gg8.se/files/roms/blargg-gb-tests/) — CPU, APU, timing tests
- [Mooneye GB test ROMs](https://gekkio.fi/files/mooneye-gb/latest/) — Edge-case granular tests
- [dmg-acid2](https://github.com/mattcurrie/dmg-acid2) — Basic PPU rendering test
- [SameBoy](https://github.com/LIJI32/SameBoy) — Open-source reference emulator
- [Game Boy Architecture: A Practical Analysis](https://www.copetti.org/writings/consoles/game-boy/) — by Rodrigo Copetti

---

## Phase 0: Project Setup & Tooling

### Task 0.1: Initialize Go module and test infrastructure

**Objective:** Scaffold the Go module with test runner config, directory structure, and CI-ready tooling.

**Design Rationale:** Go modules are the standard for dependency management. Using `go mod init` creates the foundation. A `Makefile` (or justfile) with `test`, `test-roms`, `lint` targets keeps the dev loop fast. The directory layout splits the pure emulator (`internal/gb/`) from the MCP server (`mcp/`) so the emulator core has zero external dependencies.

**Files:**
- Create: `go.mod`
- Create: `Makefile`
- Create: `tools.go` (tool dependency tracking)

**Step 1: Initialize Go module**

```bash
cd ~/projects/gbagent
go mod init github.com/AmaKuroba/gbagent
```

**Step 2: Create Makefile**

```makefile
.PHONY: test test-roms lint bench

test:
	go test ./... -v -count=1

test-roms:
	go test ./internal/gb/... -run TestROM -v -count=1

lint:
	go vet ./...
	staticcheck ./...

bench:
	go test ./internal/gb/... -bench=. -benchmem

test-cpu:
	go test ./internal/gb/... -run TestCPURom -v -count=1

test-ppu:
	go test ./internal/gb/... -run TestPPURom -v -count=1

test-apu:
	go test ./internal/gb/... -run TestAPURom -v -count=1
```

**Step 3: Install dev dependencies**

```bash
go install honnef.co/go/tools/cmd/staticcheck@latest
go install github.com/rakyll/gotest@latest
```

**Step 4: Create test ROMs directory**

```bash
mkdir -p testdata/roms
# Download Blargg's CPU test ROMs
curl -L -o testdata/roms/cpu_instrs.zip https://github.com/retrio/gb-test-roms/raw/master/cpu_instrs/cpu_instrs.gb
# (ROMs are checked into testdata/ by convention; they are freely redistributable test binaries)
```

**Step 5: Verify**

Run: `go test ./...`
Expected: No tests exist yet, package compiles cleanly.

**Step 6: Commit**

```bash
git init
git add go.mod Makefile tools.go testdata/
git commit -m "chore: scaffold Go module with test infrastructure"
```

---

### Task 0.2: Define emulator core interfaces

**Objective:** Write the Go interfaces for the emulator components that all implementations must satisfy — then write a test that a minimal stub compiles. This is the TDD contract for every subsequent task.

**Design Rationale:** Interfaces first means every component has a defined shape before any implementation begins. The CPU, MMU, PPU, APU, and Cartridge interfaces each isolate one concern, making them independently testable and swappable. The `Emulator` facade bundles them for the MCP layer.

**Files:**
- Create: `internal/gb/cpu.go` — CPU interface
- Create: `internal/gb/mmu.go` — MMU (memory) interface
- Create: `internal/gb/ppu.go` — PPU interface
- Create: `internal/gb/apu.go` — APU interface
- Create: `internal/gb/cartridge.go` — Cartridge interface
- Create: `internal/gb/emulator.go` — Facade interface
- Test: `internal/gb/interfaces_test.go`

**Step 1 (RED): Write compilation test**

```go
package gb

import (
    "testing"
    "github.com/stretchr/testify/assert"
)

// Test that all interfaces compile with a minimal mock.
// This proves the interface shapes are valid.

type mockCPU struct {
    CPU
}
type mockMMU struct {
    MMU
}
type mockPPU struct {
    PPU
}
type mockAPU struct {
    APU
}
type mockCartridge struct {
    Cartridge
}

func TestInterfacesCompile(t *testing.T) {
    e := &Emulator{
        CPU:       &mockCPU{},
        MMU:       &mockMMU{},
        PPU:       &mockPPU{},
        APU:       &mockAPU{},
        Cartridge: &mockCartridge{},
    }
    assert.NotNil(t, e)
}
```

Run: `go test ./internal/gb/ -v`
Expected: FAIL — "undefined: CPU" (interface not yet defined)

**Step 2: Write interface definitions**

```go
// internal/gb/cpu.go
package gb

// CPU represents the LR35902 (Sharp SM83) processor core.
type CPU interface {
    Step() (cycles int, err error)
    Reset()
    GetState() CPUState
}

type CPUState struct {
    AF, BC, DE, HL, SP, PC uint16
    IME                     bool
    Halted                  bool
    Stopped                 bool
    Cycles                  uint64
}
```

```go
// internal/gb/mmu.go
package gb

// MMU represents the memory management unit and address space.
type MMU interface {
    Read(addr uint16) byte
    Write(addr uint16, val byte)
    Read16(addr uint16) uint16
    Write16(addr uint16, val uint16)
    LoadROM(data []byte)
    LoadBootROM(data []byte)
}
```

```go
// internal/gb/ppu.go
package gb

// PPU represents the Picture Processing Unit (160x144 LCD).
type PPU interface {
    Step(cycles int)
    Reset()
    GetScreen() [160][144]byte  // returns pixel data (4 shades)
    GetState() PPUState
}

type PPUState struct {
    Mode     int  // 0=HBlank, 1=VBlank, 2=OAM, 3=VRAM
    LY       byte // current scanline
    LCDC     byte // LCD control register
    STAT     byte // LCD status register
    FrameCount int
}
```

```go
// internal/gb/apu.go
package gb

// APU represents the Audio Processing Unit (4 channels).
type APU interface {
    Step(cycles int)
    Reset()
    GetSamples() []float32  // stereo samples at 44100 Hz
    GetState() APUState
}

type APUState struct {
    Enabled bool
    // channel-specific state
}
```

```go
// internal/gb/cartridge.go
package gb

// Cartridge represents the ROM cartridge and MBC.
type Cartridge interface {
    Read(addr uint16) byte
    Write(addr uint16, val byte)
    GetTitle() string
    GetType() CartridgeType
    HasBattery() bool
    SaveRAM() []byte
    LoadRAM(data []byte)
}

type CartridgeType byte

const (
    CartridgeROMOnly  CartridgeType = 0x00
    CartridgeMBC1     CartridgeType = 0x01
    CartridgeMBC3     CartridgeType = 0x13
    CartridgeMBC5     CartridgeType = 0x1A
    // ... others
)
```

```go
// internal/gb/emulator.go
package gb

// Emulator is the top-level facade that ties all components together.
type Emulator struct {
    CPU       CPU
    MMU       MMU
    PPU       PPU
    APU       APU
    Cartridge Cartridge
    FrameBuffer [160][144]byte
    Booted      bool
}

func (e *Emulator) StepFrame() error {
    for i := 0; i < 70224; i++ { // 154 lines * 456 cycles
        cycles, err := e.CPU.Step()
        if err != nil {
            return err
        }
        e.PPU.Step(cycles)
        e.APU.Step(cycles)
    }
    return nil
}
```

**Step 3 (GREEN): Verify compilation**

Run: `go test ./internal/gb/ -v`
Expected: PASS — TestInterfacesCompile passes

**Step 4: Commit**

```bash
git add internal/gb/
git commit -m "feat: define emulator core interfaces"
```

---

## Phase 1: CPU & Memory

### Task 1.1: CPU — Decoder and instruction table

**Objective:** Implement the LR35902 instruction decoder using a lookup-table approach (not a giant switch). Test: given a specific opcode byte, the decoder returns the correct instruction mnemonic and cycle count.

**Design Rationale:** A lookup table indexed by opcode byte (0x00-0xFF) is the standard approach for GB emulators. It maps each opcode to a struct with mnemonic, cycles, operand size, and handler function pointer. CB-prefixed instructions use a second table. This is cleaner and faster than switch statements, and makes it trivial to verify cycle counts match the reference.

**Files:**
- Create: `internal/gb/cpu_decoder.go`
- Create: `internal/gb/cpu_decoder_cb.go`
- Test: `internal/gb/cpu_decoder_test.go`

**Step 1 (RED): Write decoder test**

```go
func TestDecodeNOP(t *testing.T) {
    inst := Decode(0x00)
    assert.Equal(t, "NOP", inst.Mnemonic)
    assert.Equal(t, 4, inst.Cycles)
}

func TestDecodeAllOpcodes(t *testing.T) {
    for op := 0; op <= 0xFF; op++ {
        inst := Decode(byte(op))
        assert.NotNil(t, inst, "opcode 0x%02X should decode", op)
        assert.Greater(t, inst.Cycles, 0, "opcode 0x%02X should have cycles", op)
    }
}

func TestDecodeCB(t *testing.T) {
    inst := DecodeCB(0x00)  // RLC B
    assert.Equal(t, "RLC B", inst.Mnemonic)
    assert.Equal(t, 8, inst.Cycles)
}
```

**Step 2 (GREEN): Implement decoder**

```go
type Instruction struct {
    Mnemonic string
    Cycles   int
    Operands int  // bytes of operands after opcode
    Handler  func(c *Core) error
}

var mainTable [256]Instruction
var cbTable [256]Instruction

func init() {
    mainTable[0x00] = Instruction{Mnemonic: "NOP", Cycles: 4, Operands: 0}
    // ... all 256 opcodes
}

func Decode(op byte) Instruction { return mainTable[op] }
func DecodeCB(op byte) Instruction { return cbTable[op] }
```

**Step 3 (GREEN): Run tests**

Run: `go test ./internal/gb/ -run TestDecode -v`
Expected: PASS

**Step 4: Commit**

```bash
git add internal/gb/cpu_decoder*.go
git commit -m "feat: CPU instruction decoder with full opcode table"
```

---

### Task 1.2: CPU — Core execution engine

**Objective:** Implement the CPU core that executes instructions by fetching, decoding, and dispatching to handler functions. Test runs a simple sequence (NOP, LD, ADD) and verifies register state changes.

**Design Rationale:** The LR35902 is a hybrid of the Intel 8080 and Z80. It has 8-bit registers (A, B, C, D, E, F, H, L) paired into 16-bit (AF, BC, DE, HL), plus SP and PC. The F register is the flags register (Z, N, H, C bits only). The Step() method fetches one byte at PC, decodes it, executes the handler, and returns elapsed cycles.

**Files:**
- Create: `internal/gb/cpu_core.go`
- Create: `internal/gb/cpu_core_ops.go` — operation implementations
- Test: `internal/gb/cpu_core_test.go`

**Step 1 (RED): Write execution test**

```go
func TestExecuteNOP(t *testing.T) {
    c := NewCore(nil)
    pc := c.State.PC
    cycles, err := c.Step()
    assert.NoError(t, err)
    assert.Equal(t, 4, cycles)
    assert.Equal(t, pc+1, c.State.PC)  // PC advanced by 1
}

func TestExecuteLD_BC_d16(t *testing.T) {
    // LD BC, $1234 (opcode 0x01, operands 0x34, 0x12)
    bus := NewBus()
    bus.Write(0x100, 0x01)  // LD BC, d16
    bus.Write(0x101, 0x34)  // low byte
    bus.Write(0x102, 0x12)  // high byte
    c := NewCore(bus)
    c.State.PC = 0x100
    _, err := c.Step()
    assert.NoError(t, err)
    assert.Equal(t, uint16(0x1234), c.State.BC)
    assert.Equal(t, uint16(0x103), c.State.PC)
}
```

**Step 2 (GREEN): Implement minimal Core**

```go
type Core struct {
    State CPUState
    Bus   MemoryBus
}

func NewCore(bus MemoryBus) *Core {
    return &Core{
        State: CPUState{
            PC: 0x0100,  // default entry point
            SP: 0xFFFE,
        },
        Bus: bus,
    }
}

func (c *Core) Step() (int, error) {
    opcode := c.Bus.Read(c.State.PC)
    inst := Decode(opcode)
    c.State.PC++
    // Handle operand bytes
    switch inst.Operands {
    case 1:
        _ = c.Bus.Read(c.State.PC)  // skip operand
        c.State.PC++
    case 2:
        _ = c.Bus.Read(c.State.PC)   // low
        _ = c.Bus.Read(c.State.PC+1) // high
        c.State.PC += 2
    }
    return inst.Cycles, nil
}
```

**Step 3 (GREEN): Run tests**

Run: `go test ./internal/gb/ -run TestExecute -v`
Expected: PASS

**Step 4: Commit**

```bash
git add internal/gb/cpu_core*.go
git commit -m "feat: CPU core with fetch-decode-execute loop"
```

---

### Task 1.3: CPU — 8-bit ALU operations (ADD, SUB, AND, OR, XOR, CP, INC, DEC)

**Objective:** Implement the full set of 8-bit ALU instructions with correct flag updates. TDD each operation group. Blargg's CPU test ROMs are the ultimate validation.

**Design Rationale:** ALU operations are the largest category of instructions. Each affects the F register flags (Z=zero, N=subtract, H=half-carry, C=carry) according to specific rules. Half-carry is the most commonly bugged flag — it's carry from bit 3 (8-bit) or bit 11 (16-bit). The GBZ80 uses BCD-style half-carry, which differs from Z80 convention.

**Files:**
- Modify: `internal/gb/cpu_core_ops.go`
- Test: `internal/gb/cpu_alu_test.go`

**Step 1 (RED): Write ALU tests**

```go
func TestADD_A_B(t *testing.T) {
    // ADD A, B (opcode 0x80)
    bus := NewBus()
    bus.Write(0x100, 0x80)
    c := NewCore(bus)
    c.State.A = 0x0F
    c.State.B = 0x01
    c.State.PC = 0x100
    c.Step()
    assert.Equal(t, byte(0x10), c.State.A)
    assert.False(t, c.State.Flags.Z)
    assert.False(t, c.State.Flags.N)
    assert.True(t, c.State.Flags.H)   // half-carry from bit 3
    assert.False(t, c.State.Flags.C)
}

func TestADD_A_HalfCarry(t *testing.T) {
    bus := NewBus()
    bus.Write(0x100, 0x80)
    c := NewCore(bus)
    c.State.A = 0x0F
    c.State.B = 0x01
    c.State.PC = 0x100
    c.Step()
    assert.True(t, c.State.Flags.H)
}

func TestSUB_B_Zero(t *testing.T) {
    // SUB B when A == B
    bus := NewBus()
    bus.Write(0x100, 0x90)
    c := NewCore(bus)
    c.State.A = 0x05
    c.State.B = 0x05
    c.State.PC = 0x100
    c.Step()
    assert.Equal(t, byte(0x00), c.State.A)
    assert.True(t, c.State.Flags.Z)
    assert.True(t, c.State.Flags.N)
}
```

**Step 2-4: Implement, verify, commit**

---

### Task 1.4: CPU — 16-bit operations (LD, ADD HL, INC/DEC, PUSH/POP)

**Objective:** Implement all 16-bit register operations with correct timing and flag effects (ADD HL sets H and C based on 16-bit addition).

**Files:**
- Modify: `internal/gb/cpu_core_ops.go`
- Test: `internal/gb/cpu_16bit_test.go`

---

### Task 1.5: CPU — Control flow (JP, JR, CALL, RET, RST, conditional jumps)

**Objective:** Implement all jump, call, return, and restart instructions with correct cycle counts for taken vs. not-taken branches.

**Files:**
- Modify: `internal/gb/cpu_core_ops.go`
- Test: `internal/gb/cpu_control_test.go`

---

### Task 1.6: CPU — Rotates, shifts, bit operations (CB prefix)

**Objective:** Implement all CB-prefixed instructions: RLC, RRC, RL, RR, SLA, SRA, SWAP, SRL, BIT, SET, RES.

**Files:**
- Create: `internal/gb/cpu_core_cb.go`
- Test: `internal/gb/cpu_cb_test.go`

---

### Task 1.7: CPU — Interrupt handling (IME, IF, IE, EI, DI, HALT, STOP)

**Objective:** Implement interrupt master enable, interrupt flags, interrupt servicing (push PC, jump to vector), HALT state (with HALT bug), and STOP state.

**Design Rationale:** The HALT bug is a well-known edge case: when HALT is executed with IME=0 and an interrupt is already pending, the CPU executes the following byte as NOP then proceeds. This is a common source of emulation bugs.

**Files:**
- Modify: `internal/gb/cpu_core.go`
- Test: `internal/gb/cpu_interrupt_test.go`

---

### Task 1.8: Run Blargg's CPU test ROMs

**Objective:** Wire the emulator to load and run Blargg's CPU instruction test ROMs, capturing their serial output (printed at 0xFF01) to verify correctness.

**Design Rationale:** Blargg's test ROMs are THE standard for CPU validation. They print pass/fail to the serial port (0xFF01). The emulator needs a minimal boot path, MMU, and timer to run them, but no PPU is needed.

**Files:**
- Create: `internal/gb/test_rom.go` — reusable test ROM runner
- Create: `testdata/roms/cpu_instrs/` — Blargg CPU tests

```go
func RunTestROM(t *testing.T, path string) string {
    data, err := os.ReadFile(path)
    require.NoError(t, err)
    // Create minimal emulator
    emu := NewEmulator()
    emu.LoadROM(data)
    // Run until serial output contains "Passed" or "Failed"
    var output strings.Builder
    for i := 0; i < 10_000_000; i++ {
        emu.Step()
        if c := emu.MMU.Read(0xFF01); c != 0 {
            output.WriteByte(c)
            emu.MMU.Write(0xFF01, 0) // clear
            if strings.Contains(output.String(), "Passed") ||
               strings.Contains(output.String(), "Failed") {
                break
            }
        }
    }
    return output.String()
}

func TestBlargg_CPUInstrs(t *testing.T) {
    result := RunTestROM(t, "testdata/roms/cpu_instrs/cpu_instrs.gb")
    assert.Contains(t, result, "Passed")
}
```

---

## Phase 2: Memory Map & Boot

### Task 2.1: Memory map with MMU

**Objective:** Implement the Game Boy memory map: ROM (0x0000-0x7FFF), VRAM (0x8000-0x9FFF), SRAM (0xA000-0xBFFF), WRAM (0xC000-0xDFFF), Echo RAM (0xE000-0xFDFF), OAM (0xFE00-0xFE9F), HRAM (0xFF80-0xFFFE), IO registers (0xFF00-0xFF7F). Test every region for correct read/write behavior and mirroring.

**Files:**
- Create: `internal/gb/mmu.go` — MMU implementation
- Create: `internal/gb/memory_map.go` — constants and helpers
- Test: `internal/gb/mmu_test.go`

### Task 2.2: Boot ROM support

**Objective:** Implement the DMG boot ROM (256 bytes) that runs at power-on, scrolls the Nintendo logo, checks the cartridge header, and jumps to 0x0100. Boot ROM is mapped at 0x0000-0x00FF and disabled after the jump.

**Files:**
- Create: `internal/gb/bootrom.go`
- Test: `internal/gb/bootrom_test.go`

---

## Phase 3: PPU (Video)

### Task 3.1: PPU — LCD timing and mode cycles

**Objective:** Implement the PPU timing state machine: OAM search (mode 2, 80 cycles), VRAM read (mode 3, ~172 cycles), HBlank (mode 0, ~204 cycles), VBlank (mode 1, 4560 cycles). Total: 70224 cycles per frame. Test with dmg-acid2 test ROM.

**Design Rationale:** The PPU is the most timing-sensitive component. Mode 3 (VRAM read) varies in duration based on the number of visible sprites on the current scanline. Accurate timing is essential for games that use STAT interrupts for raster effects.

**Files:**
- Create: `internal/gb/ppu.go` — PPU implementation
- Test: `internal/gb/ppu_timing_test.go`

### Task 3.2: PPU — Tile and background rendering

**Objective:** Implement tile-based background rendering: tiles are 8x8 pixels, each pixel is 2 bits (4 shades), stored in tile data at 0x8000-0x97FF. Two addressing modes: 0x8000 (unsigned) and 0x8800 (signed). Tile map at 0x9800-0x9BFF or 0x9C00-0x9FFF.

**Files:**
- Modify: `internal/gb/ppu.go`
- Test: `internal/gb/ppu_background_test.go`

### Task 3.3: PPU — Window layer

**Objective:** Implement the window overlay layer (controlled by WX, WY registers). Window can be positioned anywhere on screen and uses a separate tile map.

**Files:**
- Modify: `internal/gb/ppu.go`
- Test: `internal/gb/ppu_window_test.go`

### Task 3.4: PPU — Sprite/OAM rendering

**Objective:** Implement the 40-sprite OAM table at FE00-FE9F. Each sprite: Y, X, tile index, attributes (priority, flip, palette). 8x8 and 8x16 modes. Priority/overlay rules: sprite priority vs. background, lower X = higher priority within sprites.

**Files:**
- Modify: `internal/gb/ppu.go`
- Test: `internal/gb/ppu_sprite_test.go`

### Task 3.5: PPU — LCD control and STAT interrupt

**Objective:** Implement LCDC register (0xFF40) controls: display enable, window enable, sprite enable, sprite size, BG tile map, BG tile data, BG window enable. STAT interrupts for mode 0/1/2 and LY=LYC coincidence.

**Files:**
- Modify: `internal/gb/ppu.go`
- Test: `internal/gb/ppu_stat_test.go`

### Task 3.6: PPU — DMA from ROM/RAM to OAM

**Objective:** Implement the OAM DMA transfer (triggered by writing to 0xFF46). Transfers 160 bytes from the specified source page to OAM during 160 bus cycles (~40 microseconds). During DMA, only HRAM is accessible to the CPU (bus contention).

**Files:**
- Modify: `internal/gb/mmu.go`
- Test: `internal/gb/ppu_dma_test.go`

### Task 3.7: Pass dmg-acid2 test ROM

**Objective:** Run dmg-acid2 which draws a specific pixel pattern — if ANY pixel is wrong, the test fails. Use the screenshot output to verify.

---

## Phase 4: APU (Audio)

### Task 4.1: APU — Channel 1 & 2 (pulse with sweep/envelope)

**Objective:** Implement the two pulse wave channels. Each has: duty cycle (12.5/25/50/75%), length counter, volume envelope (increasing/decreasing), frequency sweep (channel 1 only, can change frequency over time). Output 44100 Hz stereo samples.

**Files:**
- Create: `internal/gb/apu.go`
- Test: `internal/gb/apu_pulse_test.go`

### Task 4.2: APU — Channel 3 (wave)

**Objective:** Implement the wave channel: 32 4-bit samples at 0xFF30-0xFF3F, configurable volume shift, length counter. Can produce any waveform, commonly used for melodic instruments.

**Files:**
- Modify: `internal/gb/apu.go`
- Test: `internal/gb/apu_wave_test.go`

### Task 4.3: APU — Channel 4 (noise)

**Objective:** Implement the noise channel: 7-bit or 15-bit LFSR (linear-feedback shift register) for pseudo-random noise, with configurable clock divider and shift. Used for percussion and sound effects.

**Files:**
- Modify: `internal/gb/apu.go`
- Test: `internal/gb/apu_noise_test.go`

---

## Phase 5: Cartridge & MBC

### Task 5.1: ROM-only cartridge

**Objective:** Implement passthrough cartridge (no banking). ROM is mapped at 0x0000-0x7FFF. Test with Tetris ROM (no MBC, simple boot test).

**Files:**
- Create: `internal/gb/cartridge.go` — implementation
- Test: `internal/gb/cartridge_test.go`

### Task 5.2: MBC1 mapper

**Objective:** Implement MBC1: up to 2MB ROM (32 banks), 32KB RAM (4 banks), bank switching via registers at 0x2000-0x3FFF, RAM enable at 0x0000-0x1FFF, banking mode selection at 0x6000-0x7FFF.

**Files:**
- Modify: `internal/gb/cartridge.go`
- Test: `internal/gb/mbc1_test.go`

### Task 5.3: MBC3 mapper with RTC

**Objective:** Implement MBC3: up to 2MB ROM, 32KB RAM, RTC registers (seconds, minutes, hours, days, day carry, latch). RTC tick advances in real-time. Used by Pokemon Gold/Silver.

**Files:**
- Modify: `internal/gb/cartridge.go`
- Create: `internal/gb/rtc.go`
- Test: `internal/gb/mbc3_test.go`

---

## Phase 6: Input, Timer, Serial

### Task 6.1: Joypad input

**Objective:** Implement the joypad register (0xFF00). Bit 5 selects direction buttons, bit 4 selects action buttons. Bits 0-3 are active-low button states. Joypad interrupt on transition from high to low.

**Files:**
- Create: `internal/gb/input.go`
- Test: `internal/gb/input_test.go`

### Task 6.2: Timer

**Objective:** Implement DIV (0xFF04, always increments at 16384 Hz), TIMA (0xFF05, increment at rate selected by TAC), TMA (0xFF06, reload value), TAC (0xFF07, control). Timer interrupt when TIMA overflows.

**Files:**
- Create: `internal/gb/timer.go`
- Test: `internal/gb/timer_test.go`

### Task 6.3: Serial link (minimal)

**Objective:** Implement the serial registers (0xFF01-0xFF02) for test ROM output. Data written to 0xFF01 is transferred when 0xFF02 bit 7 is set. Used for test ROM serial output at 0xFF01.

**Files:**
- Create: `internal/gb/serial.go`
- Test: `internal/gb/serial_test.go`

---

## Phase 7: Save States

### Task 7.1: Full state serialization

**Objective:** Implement binary save/restore of full emulator state: CPU registers + MMU state + PPU state + APU state + Cartridge state + WRAM + HRAM + VRAM + OAM. Use Go's `encoding/binary` for compact binary format with magic + version + checksum.

**Files:**
- Create: `internal/gb/savestate.go`
- Test: `internal/gb/savestate_test.go`

### Task 7.2: Battery-backed SRAM persistence

**Objective:** Save cartridge SRAM to `.sav` files. Load on boot. Wire to MCP tools.

**Files:**
- Create: `internal/gb/sram.go`
- Test: `internal/gb/sram_test.go`

---

## Phase 8: MCP Interface

### Task 8.1: MCP server scaffold

**Objective:** Set up a Go MCP server using `github.com/mark3labs/mcp-go` that starts an mcp-gateway-compatible server. Test: server compiles, registers tools, accepts transport.

**Files:**
- Create: `cmd/gbagent-mcp/main.go`
- Create: `mcp/server.go`
- Create: `mcp/tools.go`

### Task 8.2: MCP tool — get_screenshot

**Objective:** Expose the framebuffer as a base64-encoded PNG via MCP tool. The tool returns `{ "image": "iVBOR...", "format": "png", "width": 160, "height": 144 }`. Reference: is the user's question about how to do visuals via MCP.

**Design Rationale:** Base64 PNG is the standard interchange format used by LLM providers for vision. The Game Boy's 160x144 4-color framebuffer compresses extremely well (~2-6 KB per frame). Raw pixel arrays would be ~23 KB (160*144) for 2-bit indexed or ~92 KB for RGBA. PNG is both smaller and universally understood.

Implementation: store the PPU pixel buffer as `[160][144]byte` where each byte is 0-3 (white, light, dark, black). Convert to RGBA PNG using Go's `image/png` + `image.RGBA` packages. Encode to base64.

```go
func (s *Server) Screenshot() (string, error) {
    pixels := s.emu.PPU.GetScreen()
    img := image.NewRGBA(image.Rect(0, 0, 160, 144))
    palette := [4]color.RGBA{
        {0x9B, 0xBC, 0x0F, 0xFF}, // white
        {0x8B, 0xAC, 0x0F, 0xFF}, // light
        {0x30, 0x62, 0x30, 0xFF}, // dark
        {0x0F, 0x38, 0x0F, 0xFF}, // black
    }
    for y := 0; y < 144; y++ {
        for x := 0; x < 160; x++ {
            img.SetRGBA(x, y, palette[pixels[y][x]])
        }
    }
    var buf bytes.Buffer
    png.Encode(&buf, img)
    return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}
```

**Files:**
- Create: `mcp/screenshot.go`
- Test: `mcp/screenshot_test.go`

### Task 8.3: MCP tool — press_button

**Objective:** Tool that accepts a button name ("a", "b", "start", "select", "up", "down", "left", "right") + hold duration, presses it, and ticks the emulator.

```json
// Request
{
  "button": "a",
  "frames": 10
}

// Response
{
  "cycles_executed": 10,
  "frame_count": 12345
}
```

### Task 8.4: MCP tool — read_ram / write_ram

**Objective:** Tools for reading and writing emulator memory. Read accepts an address range; write accepts address + value.

### Task 8.5: MCP tool — get_state

**Objective:** Tool that returns structured game state parsed from RAM, similar to pokemon-agent: player position, party, bag, battle, dialog, flags. Implement a generic memory reader that knows game-specific RAM layouts.

### Task 8.6: MCP tool — save_state / load_state

**Objective:** Tools that serialize/deserialize the full emulator state to/from the save file system. Save accepts a name; Load accepts a name.

### Task 8.7: MCP tool — breakpoint system

**Objective:** Tools for set_breakpoint (PC address), clear_breakpoint, list_breakpoints, continue (run until breakpoint hit), step (single instruction).

### Task 8.8: CLI entry point

**Objective:** Wire everything into a `gbagent-mcp` binary that starts the MCP server. Use `gbagent-mcp serve --rom path/to/rom.gb --port 8765`.

---

## Phase 9: Live Dashboard (WebSocket Frame Streaming)

**Objective:** Build a web dashboard that displays live Game Boy frames in the browser. The dashboard connects to the emulator via WebSocket and receives PNG-encoded frames in real-time, rendered on an HTML canvas.

**Design Rationale:** The dashboard is a separate HTTP server (same Go binary) that:
- Serves a static HTML page with embedded JS
- Exposes a WebSocket endpoint that pushes frame data
- The emulator ticks at its natural pace; each completed frame is broadcast to all connected WebSocket clients
- Frames are PNG-encoded (~2-6 KB) and sent as binary WebSocket messages
- The dashboard also shows game state (player name, HP, map, etc.) as JSON messages alongside frames

Architecture:
```
┌──────────────┐    WS binary (PNG)    ┌──────────────────┐
│  Emulator    │ ◄──────────────────►  │  Web Dashboard   │
│  (Go core)   │   WS JSON (state)     │  (HTML + Canvas) │
└──────────────┘                       └──────────────────┘
       │                                        │
       │ MCP tools                               │ Browser
       ▼                                        ▼
   AI Agent                                Human watching
```

### Task 9.1: Dashboard HTTP server + WebSocket hub

**Objective:** Create an HTTP server that serves static files and manages WebSocket connections. Use gorilla/websocket (or the newer stdlib `nhooyr.io/websocket`) for WebSocket support. The hub pattern is classic: one hub goroutine tracks all connections, a broadcast channel sends frames to every client.

**Files:**
- Create: `cmd/gbagent-serve/main.go` — combined emulator + dashboard binary
- Create: `dashboard/hub.go` — WebSocket hub (register, unregister, broadcast)
- Create: `dashboard/server.go` — HTTP handler registration
- Create: `dashboard/static/index.html` — dashboard page

**Step 1: Write WebSocket hub test**

```go
func TestHubBroadcast(t *testing.T) {
    hub := NewHub()
    go hub.Run()

    // Add a mock client
    client := NewMockClient()
    hub.Register <- client

    // Broadcast a frame
    frame := []byte{1, 2, 3, 4}
    hub.Broadcast <- frame

    // Client should receive it
    select {
    case msg := <-client.Receive:
        assert.Equal(t, frame, msg)
    case <-time.After(time.Second):
        t.Fatal("timed out waiting for broadcast")
    }
}
```

**Step 2: Implement hub**

```go
type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
}

func (h *Hub) Run() {
    for {
        select {
        case client := <-h.register:
            h.clients[client] = true
        case client := <-h.unregister:
            delete(h.clients, client)
            close(client.send)
        case message := <-h.broadcast:
            for client := range h.clients {
                select {
                case client.send <- message:
                default:
                    close(client.send)
                    delete(h.clients, client)
                }
            }
        }
    }
}
```

**Step 3: Dashboard HTML page**

The page auto-connects to `ws://{host}/ws` and renders frames on a `<canvas>`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>GB Agent Dashboard</title>
  <style>
    body { background: #111; color: #fff; font-family: monospace; display: flex; }
    #screen { border: 2px solid #333; image-rendering: pixelated; width: 480px; height: 432px; }
    #info { margin-left: 20px; }
  </style>
</head>
<body>
  <div>
    <canvas id="screen" width="160" height="144"></canvas>
    <p id="status">Connecting...</p>
  </div>
  <div id="info">
    <h3>Game State</h3>
    <pre id="state">Waiting...</pre>
  </div>

  <script>
    const canvas = document.getElementById('screen');
    const ctx = canvas.getContext('2d');
    const ws = new WebSocket('ws://' + location.host + '/ws');

    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Binary — PNG frame
        const blob = new Blob([event.data], { type: 'image/png' });
        const img = new Image();
        img.onload = () => {
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(img.src);
        };
        img.src = URL.createObjectURL(blob);
      } else {
        // Text — JSON state update
        document.getElementById('state').textContent = event.data;
      }
    };

    ws.onopen = () => document.getElementById('status').textContent = 'Connected';
    ws.onclose = () => document.getElementById('status').textContent = 'Disconnected';
  </script>
</body>
</html>
```

### Task 9.2: Frame broadcaster

**Objective:** After the PPU renders each frame (VBlank), encode it as PNG and push to the WebSocket hub. Control frame rate — the emulator can run faster than 60fps, so throttle to ~30fps for the dashboard.

**Files:**
- Create: `dashboard/broadcaster.go`

```go
type Broadcaster struct {
    hub    *Hub
    emu    *gb.Emulator
    ticker *time.Ticker
}

func (b *Broadcaster) Start() {
    for range b.ticker.C {
        pixels := b.emu.PPU.GetScreen()
        pngBytes := encodePNG(pixels)
        b.hub.Broadcast <- pngBytes

        // Also send structured state as text
        state := b.emu.GetGameState()
        jsonBytes, _ := json.Marshal(state)
        b.hub.Broadcast <- jsonBytes
    }
}
```

### Task 9.3: State overlay + keyboard input

**Objective:** Extend the dashboard to show structured game state (player name, position, HP, map, dialog, battle status) alongside the frame. Also allow keyboard input via the dashboard: arrow keys + Z/X/Enter → button presses on the emulator.

**Files:**
- Modify: `dashboard/static/index.html`
- Create: `dashboard/input.go` — WebSocket endpoint that accepts button commands

Keyboard mapping on the dashboard:

| Key | GB Button |
|-----|-----------|
| Arrow keys | D-Pad |
| Z | B |
| X | A |
| Enter | Start |
| Shift | Select |

### Task 9.4: Combined CLI entry point

**Objective:** The main binary starts both the MCP server AND the dashboard. Single port or separate ports. Two modes:
- `gbagent serve --rom game.gb` — emulator + dashboard (HTTP + WS) + MCP (stdio or HTTP)
- `gbagent mcp --rom game.gb` — MCP only (headless, no dashboard)

**Files:**
- Create: `cmd/gbagent/main.go`

```bash
# Full stack (dashboard on :8765, MCP on stdio or :8766)
gbagent serve --rom pokemon.gb --dashboard-port 8765

# Headless MCP only
gbagent mcp --rom pokemon.gb
```

---

## Phase 10: End-to-End Validation

### Task 10.1: Boot Tetris to title screen

**Objective:** Load Tetris (no MBC ROM), run until the title screen is rendered. Capture screenshot via the dashboard or MCP and verify the framebuffer shows recognizable content (not blank/black).

### Task 10.2: Run Blargg's full test suite

**Objective:** Run all Blargg's test ROMs through the emulator and verify pass: cpu_instrs, instr_timing, mem_timing, halt_bug, oam_bug, etc.

### Task 10.3: Run Mooneye test suite

**Objective:** Run Gekkio's Mooneye test suite for edge cases: MBC timings, STAT IRQ ordering, DMA conflict behavior, HALT/STOP timing, etc.

---

## Risks & Trade-offs

1. **SGB Enhanced ROMs** — Some ROMs (like the one we had with pokemon-agent) have different memory mappings. Only DMG/CGB standard will be supported initially.
2. **GBC color support** — Phase 2 addition. DMG-only first.
3. **APU not needed for gameplay** — Most games work without audio. APU is lower priority.
4. **Test ROMs are the ground truth** — Don't trust the plan over test ROM output. If Blargg says FAIL, the emulator is wrong, period.
5. **Cycle accuracy vs. playability** — Target cycle accuracy for official hardware compatibility. Tradeoffs documented if they arise.

## Open Questions

- Embed boot ROM binary in Go binary, or load from file? (File first, embed later)
- Should we support save states mid-frame, or only at frame boundaries? (Mid-frame for debug, frame boundaries for normal use)
- What's the MCP transport? stdio (for direct Hermes MCP integration) or HTTP+SSE? (stdio for Hermes, HTTP for external clients)
