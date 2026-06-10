"""Playback/evaluation mode for trained retro-driver agents.

Usage:
    retro-play --rom path/to/rom.gb --checkpoint models/best-100000.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from retro_driver.dqn import DQNAgent, FrameStore
from retro_driver.env import GBEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="retro-driver playback")
    parser.add_argument("--rom", required=True, help="Path to Game Boy ROM file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--gbagent-bin", default="gbagent-mcp", help="Path to gbagent binary")
    parser.add_argument("--frame-skip", type=int, default=4, help="Emulator frames per step")
    parser.add_argument("--max-steps", type=int, default=10_000, help="Max steps to run")
    parser.add_argument("--render", action="store_true", help="Print frame info each step")
    args = parser.parse_args()

    if not Path(args.rom).exists():
        print(f"Error: ROM not found: {args.rom}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    frame_store = FrameStore()
    env = GBEnv(
        rom_path=args.rom,
        gbagent_bin=args.gbagent_bin,
        frame_skip=args.frame_skip,
        frame_store=frame_store,
    )

    agent = DQNAgent(frame_store=frame_store)
    agent.load(args.checkpoint)
    agent.online.eval()

    obs, _info = env.reset()
    agent.reset_episode()

    total_reward = 0.0
    steps = 0
    start_time = time.time()

    for _ in range(args.max_steps):
        dpad_a, btn_a = agent.act(obs, training=False)
        obs, reward, terminated, truncated, _info = env.step((dpad_a, btn_a))

        total_reward += reward
        steps += 1

        if args.render:
            print(
                f"step {steps:>5d} | reward {reward:>+7.2f} | total {total_reward:>+7.2f} "
                f"| dpad {dpad_a} btn {btn_a}"
            )

        if terminated or truncated:
            break

    elapsed = time.time() - start_time
    print(f"\nDone: {steps} steps in {elapsed:.1f}s ({steps / elapsed:.0f} sps)")
    print(f"Total reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
