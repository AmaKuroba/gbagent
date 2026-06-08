#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/gbagent-mcp"
if [ ! -x "$BIN" ]; then
  echo "[gbagent-mcp-wrapper] building..." >&2
  cd "$DIR" && go build -o "$BIN" ./cmd/gbagent-mcp/
fi
exec "$BIN" --rom "$DIR/testdata/roms/pokemon_red.gb" "$@"
