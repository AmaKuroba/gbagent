"""retro-driver training loop.

Usage:
    uv run python -m retro_driver.train
    uv run python -m retro_driver.train --resume models/checkpoint.pt
    uv run python -m retro_driver.train --pretrained pretrained_vit.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retro_driver.env import GBEnv
from retro_driver.ppo import PPOAgent, PPOConfig
from retro_driver.reward import RewardConfig
from retro_driver.ws_client import ViewerClient


@dataclass
class TrainConfig:
    """Training configuration."""

    # Environment
    gbagent_url: str = "ws://localhost:8767/ws"
    viewer_url: str = "ws://localhost:8766/metrics"
    frame_stack: int = 4
    frame_skip: int = 4
    max_episode_steps: int = 3000

    # Training
    total_steps: int = 6_000_000
    eval_interval: int = 10_000
    eval_episodes: int = 3
    save_interval: int = 25_000
    log_interval: int = 1_000

    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"
    resume: str = ""
    pretrained: str = ""

    # Reward
    ram_scanner_config: str = ""
    use_rnd: bool = False


def train(config: TrainConfig) -> None:
    """Main PPO training loop."""
    save_dir = Path(config.save_dir)
    log_dir = Path(config.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build reward config with optional game-specific scanners
    reward_config = RewardConfig(use_rnd=config.use_rnd)
    if config.ram_scanner_config:
        if not Path(config.ram_scanner_config).exists():
            print(f"Warning: RAM scanner config not found: {config.ram_scanner_config}")
        else:
            from retro_driver.config import load_game_config

            game_cfg = load_game_config(config.ram_scanner_config)
            reward_config.scanners = game_cfg.scanners

    # Create environment
    env = GBEnv(
        gbagent_url=config.gbagent_url,
        frame_stack=config.frame_stack,
        frame_skip=config.frame_skip,
        reward_config=reward_config,
        max_steps=config.max_episode_steps,
    )

    # Create PPO agent
    ppo_cfg = PPOConfig(
        total_steps=config.total_steps,
        rollout_len=128,
        batch_size=32,
        ppo_epochs=4,
    )
    agent = PPOAgent(ppo_cfg)

    # Load pre-trained encoder if provided
    if config.pretrained and Path(config.pretrained).exists():
        import torch

        encoder_state = torch.load(config.pretrained, map_location=agent.network.encoder)
        agent.network.encoder.load_state_dict(encoder_state)
        print(f"Loaded pre-trained encoder from {config.pretrained}")

    # Resume from checkpoint
    start_step = 0
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            agent.load(str(resume_path))
            start_step = agent.step_count
            print(f"Resumed from {resume_path} at step {start_step}")

    # Viewer client
    viewer = None
    if config.viewer_url:
        try:
            viewer = ViewerClient(url=config.viewer_url)
            viewer.start()
            print(f"Connected to viewer at {config.viewer_url}")
        except Exception as e:
            print(f"Warning: could not connect to viewer: {e}")
            viewer = None

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
        has_tb = True
    except ImportError:
        has_tb = False
        print("TensorBoard not available, logging to stdout only")

    print(f"Device: {next(agent.network.parameters()).device}")
    print(f"Starting PPO training: {config.total_steps} steps")
    print(f"  Rollout length: {ppo_cfg.rollout_len}")
    print(f"  PPO epochs: {ppo_cfg.ppo_epochs}")
    print(f"  Learning rate: {ppo_cfg.learning_rate}")
    print(f"  Frame stack: {config.frame_stack}, Frame skip: {config.frame_skip}")
    print(f"  Save: every {config.save_interval} steps to {save_dir}")
    print(f"  Logs: {log_dir}")
    print()

    # Training loop
    obs, _ = env.reset()
    agent.network.reset_hidden()
    hidden = agent.network.get_hidden()
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    start_time = time.time()
    steps_per_sec = 0.0

    while agent.step_count < config.total_steps:
        # Collect rollout
        agent.buffer.clear()
        for _ in range(ppo_cfg.rollout_len):
            dpad_a, btn_a, dpad_lp, btn_lp, value, new_hidden = agent.act(
                obs, hidden, training=True
            )

            # Step environment
            next_obs, reward, terminated, truncated, _info = env.step((dpad_a, btn_a))
            done = terminated or truncated

            # Store in buffer
            agent.add_to_buffer(
                obs=obs,
                dpad_action=dpad_a,
                btn_action=btn_a,
                dpad_log_prob=dpad_lp,
                btn_log_prob=btn_lp,
                reward=reward,
                done=done,
                value=value,
            )

            obs = next_obs
            hidden = new_hidden
            episode_reward += reward
            episode_steps += 1

            # Episode end
            if done:
                elapsed = time.time() - start_time
                steps_per_sec = agent.step_count / elapsed if elapsed > 0 else 0
                print(
                    f"[{agent.step_count:>7d}/{config.total_steps}] "
                    f"ep {episode_num:>4d} | "
                    f"return {episode_reward:>+7.2f} | "
                    f"len {episode_steps:>4d} | "
                    f"{steps_per_sec:.0f} sps"
                )

                if has_tb:
                    writer.add_scalar("train/episode_return", episode_reward, episode_num)
                    writer.add_scalar("train/episode_length", episode_steps, episode_num)

                # Reset
                obs, _ = env.reset()
                agent.network.reset_hidden()
                hidden = agent.network.get_hidden()
                episode_reward = 0.0
                episode_steps = 0
                episode_num += 1

        # PPO update
        stats = agent.update()

        # Report to viewer
        if viewer and stats:
            try:
                viewer.report_reward(
                    total=episode_reward,
                    loss=stats.get("policy_loss", 0.0),
                    epsilon=0.0,
                    sps=steps_per_sec,
                )
            except Exception:
                pass

        # Log training stats
        if has_tb and stats:
            writer.add_scalar("train/policy_loss", stats["policy_loss"], agent.step_count)
            writer.add_scalar("train/value_loss", stats["value_loss"], agent.step_count)
            writer.add_scalar("train/entropy", stats["entropy_loss"], agent.step_count)

        # Periodic checkpoint
        if agent.step_count % config.save_interval == 0 and agent.step_count > 0:
            ckpt_path = save_dir / f"checkpoint-{agent.step_count}.pt"
            agent.save(str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = save_dir / "final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model: {final_path}")

    env.close()
    if viewer:
        viewer.stop()
    if has_tb:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="retro-driver PPO training")
    parser.add_argument("--gbagent-url", default="ws://localhost:8767/ws",
                        help="WebSocket URL of running gbagent")
    parser.add_argument("--viewer-url", default="ws://localhost:8766/metrics",
                        help="WebSocket URL of dashboard viewer for metrics")
    parser.add_argument("--total-steps", type=int, default=6_000_000,
                        help="Total training steps")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument("--pretrained", default="",
                        help="Path to pre-trained ViT encoder weights")
    parser.add_argument("--save-dir", default="models", help="Checkpoint save directory")
    parser.add_argument("--log-dir", default="logs", help="TensorBoard log directory")
    parser.add_argument("--ram-scanner", default="",
                        help="Path to game-specific RAM scanner YAML")
    parser.add_argument("--use-rnd", action="store_true",
                        help="Enable RND exploration bonus (unsupervised mode)")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Emulator frames per step")
    parser.add_argument("--max-episode-steps", type=int, default=3000,
                        help="Max steps per episode")
    parser.add_argument("--frame-stack", type=int, default=4,
                        help="Number of stacked frames in observation")
    parser.add_argument("--save-interval", type=int, default=25_000,
                        help="Checkpoint frequency")

    args = parser.parse_args()

    config = TrainConfig(
        gbagent_url=args.gbagent_url,
        viewer_url=args.viewer_url,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        total_steps=args.total_steps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        pretrained=args.pretrained,
        save_interval=args.save_interval,
        ram_scanner_config=args.ram_scanner,
        use_rnd=args.use_rnd,
    )

    train(config)


if __name__ == "__main__":
    main()
