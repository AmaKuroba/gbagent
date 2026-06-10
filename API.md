# gbagent JSON-RPC API — Contract

Retro-driver and gbagent agree on this JSON-RPC 2.0 protocol over
**stdio** (local subprocess) or **WebSocket** (remote).

All requests follow the standard format:

```json
{"jsonrpc":"2.0", "id":1, "method":"<method>", "params":{...}}
```

Response:

```json
{"jsonrpc":"2.0", "id":1, "result":<value>}
{"jsonrpc":"2.0", "id":1, "error":{"code":-32601, "message":"..."}}
```

---

## Methods

### screenshot

Get the current screen. Advances one frame.

- Params: `{}`
- Result: `string` — base64-encoded PNG (160×144)

### step_frames

Advance N frames with optional buttons held. Returns final frame.

- Params:
  - `count: int` — number of frames (≥1)
  - `buttons?: string[]` — buttons to hold (optional, released after)
- Result: `string` — base64-encoded PNG of final frame

### press_button

Press and latch a button. Stays pressed until release_button or release_all.

- Params: `{ "button": "A" }`
- Result: `"ok"`

Valid buttons: `A`, `B`, `START`, `SELECT`, `UP`, `DOWN`, `LEFT`, `RIGHT`

### release_button

Release a previously latched button.

- Params: `{ "button": "A" }`
- Result: `"ok"`

### release_all

Release all latched buttons.

- Params: `{}`
- Result: `"ok"`

### read_ram

Read one byte from Game Boy memory.

- Params: `{ "address": 0xFF00 }`
- Result: `int` — byte value (0–255)

### read_ram_range

Read a range of memory bytes.

- Params: `{ "address": 0xD000, "length": 10 }`
- Result: `int[]` — array of byte values

### write_ram

Write one byte to Game Boy memory.

- Params: `{ "address": 0xD000, "value": 0x42 }`
- Result: `"ok"`

### get_state

Get full emulator state snapshot. Advances one frame.

- Params: `{}`
- Result: `{ cpu: {...}, ppu: {...}, timer: {...} }`

### save_state

Save emulator state to a file.

- Params: `{ "path": "/tmp/state.sav" }`
- Result: `"ok"`

### load_state

Load emulator state from a file.

- Params: `{ "path": "/tmp/state.sav" }`
- Result: `"ok"`

### reset

Soft reset the emulator (reloads boot ROM, resets CPU/PPU).

- Params: `{}`
- Result: `"ok"`

---

## Running

**Local (subprocess for training):**

```bash
gbagent jsonrpc --rom ~/roms/pokemon-red.gb
```

Reads JSON-RPC from stdin, writes to stdout. One request per line,
one response per line.

**Remote (WebSocket for distributed):**

```bash
gbagent serve --rom ~/roms/pokemon-red.gb --jsonrpc-port 8767
```

Connect to `ws://host:8767/ws`. One JSON object per text message.
