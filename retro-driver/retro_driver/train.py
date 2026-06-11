"""retro-driver training loop.

Usage:
    uv run python -m retro_driver.train
    uv run python -m retro_driver.train --resume models/checkpoint.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from retro_driver.dqn import DQNAgent, DQNConfig, FrameStore, Transition
from retro_driver.env import GBEnv
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

    # DQN
    dqn: DQNConfig = field(default_factory=DQNConfig)

    # Training
    total_steps: int = 500_000
    eval_interval: int = 10_000
    eval_episodes: int = 3
    save_interval: int = 25_000
    log_interval: int = 1_000

    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"
    resume: str = ""

    # Reward
    ram_scanner_config: str = ""


def train(config: TrainConfig) -> None:
    """Main training loop."""
    save_dir = Path(config.save_dir)
    log_dir = Path(config.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build reward config with optional game-specific scanners
    reward_config = RewardConfig()
    game_cfg = None
    if config.ram_scanner_config:
        if not Path(config.ram_scanner_config).exists():
            print(f"Warning: RAM scanner config not found: {config.ram_scanner_config}")
        else:
            from retro_driver.config import load_game_config

            game_cfg = load_game_config(config.ram_scanner_config)
            reward_config.scanners = game_cfg.scanners

    # Shared frame store (individual frames, shared by env and agent)
    frame_store = FrameStore(max_frames=config.dqn.buffer_capacity + config.dqn.seq_len + 100)

    # Environment
    env = GBEnv(
        gbagent_url=config.gbagent_url,
        frame_stack=config.frame_stack,
        frame_skip=config.frame_skip,
        reward_config=reward_config,
        max_steps=config.max_episode_steps,
        frame_store=frame_store,
    )

    # Viewer client for metrics and takeover
    viewer: ViewerClient | None = None
    if config.viewer_url:
        try:
            viewer = ViewerClient(url=config.viewer_url)
            viewer.start()
            print(f"Connected to viewer at {config.viewer_url}")
        except Exception as e:
            print(f"Warning: could not connect to viewer: {e}")
            viewer = None

    # Agent (reuse the same frame store)
    agent = DQNAgent(config=config.dqn, frame_store=frame_store)
    start_step: int = 0

    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            agent.load(str(resume_path))
            # Infer step count from filename: {checkpoint,best,final}-{step}.pt
            import re

            m = re.search(r"(?:checkpoint|best|final)-?(\d+)", resume_path.stem)
            start_step = int(m.group(1)) if m else agent.step_count

            print(f"Resumed from {resume_path} at step {start_step}")

    # Demo loading
    demos_path = save_dir / "demos.pt"
    demos_loaded = 0
    if demos_path.exists():
        try:
            import torch
            demo_data = torch.load(demos_path, weights_only=False)
            for f_arr, d, b, r, term, trunc in demo_data:
                gid = frame_store.add(np.array(f_arr, dtype=np.uint8))
                agent.buffer.push(Transition(
                    frame_gid=gid, dpad_action=d, btn_action=b,
                    reward=r, terminated=term, truncated=trunc,
                ))
                demos_loaded += 1
            print(f"Loaded {demos_loaded} demo transitions from {demos_path}")
        except Exception as e:
            print(f"Warning: failed to load demos: {e}")

    step: int = start_step

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
        has_tb = True
    except ImportError:
        has_tb = False
        print("TensorBoard not available, logging to stdout only")

    # Eval stats accumulator
    best_mean_reward = float("-inf")

    print(f"Device: {agent.online.feature_extractor.conv[0].weight.device}")
    print(f"Starting training: {config.total_steps} steps")
    print(
        f"  Epsilon: {agent.config.epsilon_start} → {agent.config.epsilon_end} over {agent.config.epsilon_decay_steps} steps"
    )
    print(f"  Buffer: {agent.config.buffer_capacity}, Batch: {agent.config.batch_size}")
    print(f"  Frame stack: {config.frame_stack}, Frame skip: {config.frame_skip}")
    print(f"  Save: every {config.save_interval} steps to {save_dir}")
    print(f"  Logs: {log_dir}")
    print()

    obs, info = env.reset()
    agent.reset_episode()
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    episode_rewards: list[float] = []
    episode_lengths: list[float] = []
    start_time = time.time()
    terminated = False
    truncated = False
    train_stats = {"loss": 0.0, "q_dpad": 0.0, "q_btn": 0.0, "epsilon": agent.epsilon}

    # Demo tracking
    demo_frames: list[np.ndarray] = []
    demo_actions: list[tuple[int, int]] = []
    prev_takeover = False
    reward_window: list[float] = []
    last_report_time = time.time()

    while step < config.total_steps:
        step += 1
        takeover = viewer.get_takeover() if viewer else False

        # Takeover released → flush demos into replay buffer
        if prev_takeover and not takeover and demo_frames:
            n = len(demo_frames)
            for _i, (frame, (dpad_a, btn_a)) in enumerate(zip(demo_frames, demo_actions, strict=False)):
                if frame_store is not None:
                    gid = frame_store.add(frame)
                    start_gid = gid - (config.frame_stack - 1)
                else:
                    start_gid = 0
                agent.buffer.push(Transition(
                    frame_gid=start_gid, dpad_action=dpad_a, btn_action=btn_a,
                    reward=0.0, terminated=False, truncated=False,
                ))
            # Train on the demonstrations
            for _ in range(min(n, 64)):
                train_stats = agent.learn()
            print(f"  └─ Flushed {n} demo steps into buffer, {min(n, 64)} train steps")
            demo_frames.clear()
            demo_actions.clear()
        prev_takeover = takeover

        if takeover:
            # Human plays — training loop yields, just records
            time.sleep(0.016)
            img = env.client.get_screen()
            action = viewer.get_last_action() if viewer else {"dpad": 0, "btn": 0}
            frame = np.array(img, dtype=np.uint8)
            demo_frames.append(frame)
            demo_actions.append((int(action["dpad"]), int(action["btn"])))
            info = {"reward_breakdown": {}}
            continue

        # Normal agent training
        dpad_a, btn_a = agent.act(obs, training=True)
        action = (dpad_a, btn_a)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.buffer.push(
            Transition(
                frame_gid=info["frame_gid"],
                dpad_action=dpad_a,
                btn_action=btn_a,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        )

        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        reward_window.append(reward)

        train_stats = agent.learn()

        # Report reward every ~10s (20 steps at ~2 sps)
        if step % 20 == 0:
            try:
                now = time.time()
                sps = 20 / (now - last_report_time) if last_report_time else 0
                last_report_time = now
                if viewer:
                    viewer.report_reward(
                        total=sum(reward_window),
                        loss=train_stats["loss"],
                        epsilon=train_stats["epsilon"],
                        sps=sps,
                    )
                reward_window.clear()
            except Exception:
                pass

        # Periodic console + TensorBoard logging
        if step % max(1, config.log_interval // 10) == 0 and not takeover:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            print(
                f"[{step:>7d}/{config.total_steps}] "
                f"ep {episode_num:>4d} | "
                f"return {episode_reward:>+7.2f} | "
                f"eps {train_stats['epsilon']:.3f} | "
                f"loss {train_stats['loss']:.4f} | "
                f"{steps_per_sec:.0f} sps"
            )
            if has_tb:
                writer.add_scalar("train/loss", train_stats["loss"], step)
                writer.add_scalar("train/q_dpad", train_stats["q_dpad"], step)
                writer.add_scalar("train/q_btn", train_stats["q_btn"], step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, step)
                writer.add_scalar("train/epsilon", train_stats["epsilon"], step)

        # Episode end
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            # Log episode stats
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0

            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0

            print(
                f"[{step:>7d}/{config.total_steps}] "
                f"ep {episode_num:>4d} | "
                f"return {episode_reward:>+7.2f} | "
                f"len {episode_steps:>4d} | "
                f"avg100r {avg_reward:>+7.2f} | "
                f"eps {train_stats['epsilon']:.3f} | "
                f"loss {train_stats['loss']:.4f} | "
                f"{steps_per_sec:.0f} sps"
            )

            if has_tb:
                writer.add_scalar("train/episode_return", episode_reward, episode_num)
                writer.add_scalar("train/episode_length", episode_steps, episode_num)
                writer.add_scalar("train/avg100_return", avg_reward, episode_num)

            # Stats reset
            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1

            # Reset env and agent for new episode
            obs, _ = env.reset()
            agent.reset_episode()

        # Periodic evaluation (only during normal training)
        if not takeover and step % config.eval_interval == 0 and step > 0:
            eval_rewards = _evaluate(env, agent, config.eval_episodes)
            mean_reward = np.mean(eval_rewards)
            if has_tb:
                writer.add_scalar("eval/mean_return", mean_reward, step)
            print(
                f"  └─ EVAL [{step:>7d}]: mean_return={mean_reward:>+7.2f} over {config.eval_episodes} eps"
            )

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_path = save_dir / f"best-{step}.pt"
                agent.save(str(best_path))
                print(f"  └─ 🏆 New best model saved: {best_path}")

        # Periodic checkpoint
        if step % config.save_interval == 0 and step > 0:
            ckpt_path = save_dir / f"checkpoint-{step}.pt"
            agent.save(str(ckpt_path))
            print(f"  └─ 💾 Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = save_dir / "final.pt"
    agent.save(str(final_path))
    print(f"\nTraining complete. Final model: {final_path}")

    env.close()
    if viewer:
        viewer.stop()
    if has_tb:
        writer.close()


def _evaluate(env: GBEnv, agent: DQNAgent, episodes: int, checkpoint_path: str = "") -> list[float]:
    """Run evaluation episodes with epsilon=0."""
    rewards: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        agent.reset_episode()
        total = 0.0
        done = False
        while not done:
            dpad_a, btn_a = agent.act(obs, training=False)
            obs, reward, term, trunc, _ = env.step((dpad_a, btn_a))
            total += reward
            done = term or trunc
        rewards.append(total)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="retro-driver training loop")
    parser.add_argument("--gbagent-url", default="ws://localhost:8767/ws", help="WebSocket URL of running gbagent")
    parser.add_argument("--viewer-url", default="ws://localhost:8766/metrics", help="WebSocket URL of dashboard viewer for metrics")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument("--save-dir", default="models", help="Checkpoint save directory")
    parser.add_argument("--log-dir", default="logs", help="TensorBoard log directory")
    parser.add_argument("--ram-scanner", default="", help="Path to game-specific RAM scanner YAML")
    parser.add_argument("--frame-skip", type=int, default=4, help="Emulator frames per step")
    parser.add_argument(
        "--max-episode-steps", type=int, default=3000, help="Max steps per episode"
    )
    parser.add_argument("--eval-interval", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument("--save-interval", type=int, default=25_000, help="Checkpoint frequency")
    parser.add_argument("--buffer-capacity", type=int, default=10_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-decay", type=int, default=50_000, help="Epsilon decay steps")
    parser.add_argument(
        "--frame-stack", type=int, default=4, help="Number of stacked frames in observation"
    )
    parser.add_argument(
        "--train-start", type=int, default=2000, help="Steps before training begins"
    )
    parser.add_argument(
        "--seq-len", type=int, default=32, help="LSTM sequence length (1 to disable)"
    )
    # (--start-state removed — use gbagent's --load-state instead)

    args = parser.parse_args()

    dqn_cfg = DQNConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        epsilon_decay_steps=args.epsilon_decay,
        train_start=args.train_start,
        seq_len=args.seq_len,
    )

    train_cfg = TrainConfig(
        gbagent_url=args.gbagent_url,
        viewer_url=args.viewer_url,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        total_steps=args.total_steps,
        dqn=dqn_cfg,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        ram_scanner_config=args.ram_scanner,
    )

    train(train_cfg)


if __name__ == "__main__":
    main()
