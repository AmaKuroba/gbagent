#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROMS_DIR="$SCRIPT_DIR/../testdata/roms"
mkdir -p "$ROMS_DIR/dmg_sound"

echo "==> Downloading Blargg test ROMs..."

download_rom() {
  local url="$1"
  local dest="$2"
  if [ -f "$dest" ]; then
    echo "  SKIP $dest (exists)"
    return
  fi
  echo "  GET $dest"
  curl -sL -o "$dest" "$url"
}

BASE="https://raw.githubusercontent.com/retrio/gb-test-roms/master"

download_rom "$BASE/cpu_instrs/cpu_instrs.gb"       "$ROMS_DIR/cpu_instrs.gb"
download_rom "$BASE/instr_timing/instr_timing.gb"     "$ROMS_DIR/instr_timing.gb"
download_rom "$BASE/mem_timing/mem_timing.gb"         "$ROMS_DIR/mem_timing.gb"
download_rom "$BASE/mem_timing-2/mem_timing.gb"       "$ROMS_DIR/mem_timing_2.gb"
download_rom "$BASE/oam_bug/oam_bug.gb"               "$ROMS_DIR/oam_bug.gb"
download_rom "https://raw.githubusercontent.com/retrio/gb-test-roms/master/halt_bug.gb" \
  "$ROMS_DIR/halt_bug.gb"
download_rom "$BASE/dmg_sound/dmg_sound.gb"           "$ROMS_DIR/dmg_sound/dmg_sound.gb"

echo "==> Downloading dmg_acid2..."
download_rom "https://github.com/mattcurrie/dmg-acid2/releases/download/v1.0/dmg-acid2.gb" \
  "$ROMS_DIR/dmg_acid2.gb"

echo "==> Downloading Mooneye test suite..."
LATEST_MTS=$(curl -sL "https://gekkio.fi/files/mooneye-test-suite/" \
  | grep -o 'href="mts-[^"]*"' \
  | sed 's/href="//;s/\&#x2f.*//' \
  | sort \
  | tail -1)

if [ -z "$LATEST_MTS" ]; then
  echo "  FAILED to detect latest MTS version"
  echo "  Download manually from https://gekkio.fi/files/mooneye-test-suite/"
  echo "  and extract to: $ROMS_DIR/mooneye-test-suite/"
  exit 1
fi

MTS_DIR="$ROMS_DIR/mooneye-test-suite"
if [ -d "$MTS_DIR" ]; then
  echo "  SKIP $MTS_DIR (exists)"
else
  MTS_ZIP_URL="https://gekkio.fi/files/mooneye-test-suite/${LATEST_MTS}/${LATEST_MTS}.zip"
  echo "  GET ${LATEST_MTS}.zip"
  curl -sL -o /tmp/mts.zip "$MTS_ZIP_URL"
  mkdir -p /tmp/mts-extract
  unzip -o /tmp/mts.zip -d /tmp/mts-extract > /dev/null
  # Strip the versioned top-level directory
  mkdir -p "$MTS_DIR"
  mv /tmp/mts-extract/mts-*/* "$MTS_DIR"
  rm -rf /tmp/mts.zip /tmp/mts-extract
  echo "  Extracted to $MTS_DIR"
fi

echo ""
echo "==> Test data available at: $ROMS_DIR"
ls -1 "$ROMS_DIR"/*.gb 2>/dev/null
echo "  dmg_sound/..."
echo "  mooneye-test-suite/"
echo ""
echo "Run 'just test' to verify."
