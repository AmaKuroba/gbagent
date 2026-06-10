#!/usr/bin/env python3
"""One-shot save/load state for gbagent WebSocket JSON-RPC.

Usage:
    python scripts/save_state.py save checkpoints/pokemon_red_outskirts.sav
    python scripts/save_state.py load checkpoints/pokemon_red_outskirts.sav

Connects to the running gbagent JSON-RPC WebSocket, fires the
request once, and exits.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add retro-driver to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retro_driver.ws_client import GBWSClient


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <save|load> <path> [ws_url]")
        print(f"  Default WS URL: ws://localhost:8767/ws")
        sys.exit(1)

    command = sys.argv[1]
    state_path = sys.argv[2]
    ws_url = sys.argv[3] if len(sys.argv) > 3 else "ws://localhost:8767/ws"

    client = GBWSClient(ws_url)
    try:
        client.start()

        if command == "save":
            client.save_state(state_path)
            print(f"✅ Saved state to {state_path}")
        elif command == "load":
            client.load_state(state_path)
            print(f"✅ Loaded state from {state_path}")
        else:
            print(f"Unknown command: {command}  (use 'save' or 'load')")
            sys.exit(1)

        # Brief pause so the server processes it before we disconnect
        time.sleep(0.1)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
