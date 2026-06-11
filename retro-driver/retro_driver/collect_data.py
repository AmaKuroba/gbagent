"""Collect training data from save states using a random agent.

Usage:
    1. Manually play to key locations and save states via the dashboard
    2. Run this script to collect frames from each save state
    3. Use the collected dataset for MAE pre-training

Usage:
    uv run python -m retro_driver.collect_data --save-dir saves/
    uv run python -m retro_driver.collect_data --save-dir saves/ --frames-per-location 50000
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from retro_driver.ws_client import GBWSClient

# Default save states for Pokemon Red/Blue/Yellow
# Keys are human-readable names, values are paths relative to --save-dir
DEFAULT_SAVE_STATES = [
    # Phase 1: Starting Area
    "pallet_house_inside.sav",
    "pallet_house_outside.sav",
    "route1_grass.sav",
    "route1_tall_grass.sav",
    # Phase 2: Early Towns
    "viridian_pokemon_center.sav",
    "viridian_poke_mart.sav",
    "pewter_gym.sav",
    # Phase 3: Routes & Caves
    "viridian_forest.sav",
    "mt_moon_entrance.sav",
    "mt_moon_deep.sav",
    # Phase 4: Mid-Game
    "cerulean_gym.sav",
    "vermilion_gym.sav",
    # Phase 5: Battles
    "wild_battle.sav",
    "trainer_battle.sav",
    "gym_battle.sav",
]

# Game Boy buttons for random actions
BUTTONS = ["up", "down", "left", "right", "a", "b", "start", "select", ""]


def collect_from_save_state(
    client: GBWSClient,
    save_path: str,
    num_frames: int,
    frame_skip: int = 4,
) -> list[np.ndarray]:
    """Run random agent from a save state and collect frames.

    Args:
        client: Connected emulator client.
        save_path: Path to the save state file.
        num_frames: Number of frames to collect.
        frame_skip: Emulator frames per agent step.

    Returns:
        List of grayscale frames (144x160 uint8 numpy arrays).
    """
    frames: list[np.ndarray] = []

    # Load save state
    try:
        client.load_state(save_path)
    except Exception as e:
        print(f"  Warning: failed to load {save_path}: {e}")
        return frames

    # Let state settle
    client.wait_frames(10)

    # Collect frames with random actions
    for _ in range(num_frames):
        # Get current screen
        try:
            screen = client.get_screen()
            frames.append(np.array(screen))
        except Exception:
            continue

        # Random action
        action = random.choice(BUTTONS)
        if action:
            client.press_button(action)
        client.wait_frames(frame_skip)
        client.release_all()

    return frames


def scan_save_dir(save_dir: Path) -> list[str]:
    """Scan directory for .sav files and return sorted list of names."""
    sav_files = sorted(save_dir.glob("*.sav"))
    return [f.name for f in sav_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect training data from save states")
    parser.add_argument("--url", default="ws://localhost:8767/ws", help="Emulator WebSocket URL")
    parser.add_argument("--save-dir", default="saves", help="Directory containing .sav files")
    parser.add_argument("--output", default="pokemon_dataset.npz", help="Output dataset file")
    parser.add_argument("--frames-per-location", type=int, default=30_000,
                        help="Frames to collect per save state")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Emulator frames per agent step")
    parser.add_argument("--locations", nargs="*", default=None,
                        help="Specific save states to use (without .sav extension). "
                             "If not specified, uses all .sav files in save-dir.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        print(f"Error: save directory not found: {save_dir}")
        print("Create it and add .sav files, or specify --save-dir <path>")
        return

    # Discover save states
    if args.locations:
        save_states = [f"{loc}.sav" for loc in args.locations]
    else:
        save_states = scan_save_dir(save_dir)

    if not save_states:
        print(f"No .sav files found in {save_dir}")
        print("Manually play to key locations and save states via the dashboard.")
        return

    print(f"Found {len(save_states)} save states:")
    for s in save_states:
        print(f"  - {s}")
    print()

    # Connect to emulator
    print(f"Connecting to emulator at {args.url}...")
    client = GBWSClient(args.url)
    client.start()
    print("Connected.")
    print()

    # Collect data from each save state
    all_frames: list[np.ndarray] = []
    total_frames = len(save_states) * args.frames_per_location

    for i, save_name in enumerate(save_states, 1):
        save_path = str(save_dir / save_name)
        location = save_name.replace(".sav", "")

        print(f"[{i}/{len(save_states)}] Collecting from {location}...")
        start_time = time.time()

        frames = collect_from_save_state(
            client,
            save_path,
            args.frames_per_location,
            args.frame_skip,
        )

        elapsed = time.time() - start_time
        fps = len(frames) / elapsed if elapsed > 0 else 0
        all_frames.extend(frames)

        print(f"  Collected {len(frames)} frames ({fps:.1f} fps)")
        print(f"  Progress: {len(all_frames)}/{total_frames} total frames")
        print()

    # Save dataset
    print("Saving dataset...")
    dataset = np.array(all_frames)
    np.savez_compressed(args.output, frames=dataset)

    print("\nDone!")
    print(f"Total frames: {len(dataset)}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Saved to: {args.output}")

    # Print dataset stats
    file_size = Path(args.output).stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    client.stop()


if __name__ == "__main__":
    main()
