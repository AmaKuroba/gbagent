# gbagent TODOs

A running list of known issues and what needs to be done.

## Critical Bugs

### 1. CPU/PPU interleaving — boot ROM stuck
- **What:** Boot ROM VBlank wait loop at 0x0064-0x0068 never exits. CPU reads LY via `LDH A, ($44)` but PPU steps *after* the CPU instruction, so LY is always one instruction behind — the loop never sees LY=0x90.
- **Why:** The CPU and PPU step sequentially (CPU completes an instruction, then PPU catches up) instead of interleaving on every T-cycle.
- **Symptoms:** No game boots, screen stays dark (boot ROM never disables itself via 0xFF50 write).
- **Fix needed:** Rewrite the main loop so PPU ticks on every T-cycle alongside the CPU, like real hardware.

### 2. PPU VRAM direct reads — partially fixed
- **What:** PPU reads tile data via `p.mmu.Read()` during mode 3, but MMU blocks VRAM reads during mode 3 returning `0xFF`. We added `ReadVRAMDirect()` to MemoryBus to bypass the check, but not all render paths in `ppu.go` use it yet.
- **Symptoms:** Garbled or solid-green screen.
- **Done:** `decodeTileRow`, `renderBackgroundScanline`, `renderWindowScanline`, sprite rendering.
- **Needed:** Audit every VRAM read in PPU rendering paths and switch to `ReadVRAMDirect()`.

### 3. Failing `mem_timing` Blargg test
- **What:** `TestBlargg_mem_timing` — sub-tests 01, 02, 03 all print `Failed`.
- **Root cause:** The CPU needs cycle-accurate memory access timing. Commit `748cbfa` added deferred writes, but **deferred reads are still missing**. Reads happen immediately instead of being delayed by the correct number of T-cycles.
- **Fix needed:** Implement deferred reads (reads should take cycles just like writes do), timing-sensitive instructions like `LDH (C), A` and `LDI` need correct T-cycle accounting.

### 4. APU / Sound — reported as "really broken"
- **What:** Audio output has severe issues. Likely caused by the same sequential stepping problem — the APU ticks based on CPU cycle counts that are wrong because the timing model is off.
- **Fix:** Might resolve automatically once CPU/PPU interleaving and cycle-accuracy are in place, but may need dedicated APU timing fixes too.
- **Related:** APU frame sequencer ticks at 512Hz derived from DIV timer — if DIV timing is off, audio breaks.

## Minor Issues

### 5. Test suite coverage
- **Failing:** `TestBlargg_mem_timing` (3 sub-tests fail — 01, 02, 03)
- **Status of others:** All other Blargg tests pass, Mooneye tests mostly pass, dmg-acid2 smoke test OK
- **Goal:** Get 100% on all test ROMs before calling accuracy "good"

### 6. No cycle-accurate DMG boot ROM replacement
- **Current boot ROM:** Hardcoded 256-byte DMG boot ROM. If the main loop uses a different stepping model, the boot ROM's timing assumptions (like waiting for specific frames) may need adjustment.

### 7. Dashboard audio sync
- Dashboard streams audio via WebSocket alongside framebuffer. If the timing loop is wrong, audio drifts out of sync with video.

### 8. Emulation speed
- At 30fps dashboard rendering, the emulator needs to run ~2 full frames of emulation per rendered frame. If timing is wrong, the speed multiplier may not be accurate.

## Architecture Ideas

### 9. Consider CGo core swap
- **Option A:** Fix the pure-Go timing model (the work above)
- **Option B:** Replace the core with [mGBA's libmgba](https://mgba.io) via CGo — battle-tested C API with `mCore.runFrame()`, `setKeys()`, `busRead8()`, save states
- **Option C:** Replace with [SameBoy](https://sameboy.github.io) library build via CGo — most accurate emulator available
- **Trade-off:** CGo adds complexity and build requirements; pure-Go is cleaner if the bugs are fixable

## Infrastructure

### 10. MCP server robustness
- Currently a simple SSE server with no reconnection logic
- No graceful shutdown when client disconnects
- Command timeout/error handling could be better

### 11. Dashboard improvements
- No mobile-friendly layout
- Missing audio volume control
- No keyboard shortcut guide overlay
- No ROM file picker (must pass via CLI)
