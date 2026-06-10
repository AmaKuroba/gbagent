# gbagent

**Game Boy emulator with an MCP interface.**

A headless Game Boy emulator that exposes its full control surface through the Model Context Protocol — screenshots, input, memory read/write, save states, and debugging — so AI agents can play and analyze games programmatically.

## Design

- **Emulator core** — Go implementation of the LR35902 (Sharp SM83) CPU, PPU, APU, timers, and cartridge MBCs
- **MCP interface** — tools for screenshots (base64 PNG), button input, RAM read/write, save/load states, breakpoints
- **Headless** — no display server required, runs anywhere with Go

## Status

🚧 Planning phase — see [implementation plan](.hermes/plans/2026-06-08_141500-gbagent-implementation.md) for the full TDD roadmap and [kanban board](https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban) for task tracking.

## TDD

This project uses strict **Test-Driven Development**:
1. Write a failing test first (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor while keeping tests green

Every component is verified against standard test ROMs (Blargg, Mooneye, dmg-acid2) and Go unit tests.

## Visuals via MCP

Screenshots are served as **base64-encoded PNG** through MCP tools. The Game Boy's 160×144, 4-color framebuffer compresses to ~2-6 KB per frame — orders of magnitude smaller than raw pixel arrays (92 KB for RGBA). The MCP tool `get_screenshot` returns:

```json
{ "image": "iVBORw0KGgo...", "format": "png", "width": 160, "height": 144 }
```

This is compatible with any LLM that supports vision. An ASCII art fallback (couple hundred bytes) is also available for fast text-based reads.

## References

- [Pan Docs](https://gbdev.github.io/pandocs/) — Game Boy hardware reference
- [gb-opcodes](https://gbdev.github.io/gb-opcodes/optables/) — Instruction set table
- [Blargg's test ROMs](http://gbdev.gg8.se/files/roms/blargg-gb-tests/) — CPU/APU validation
- [awesome-gbdev](https://github.com/gbdev/awesome-gbdev) — Curated resource list
