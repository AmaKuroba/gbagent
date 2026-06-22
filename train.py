#!/usr/bin/env python3
"""Training entry point for gbagent — PPO with GAE, TensorBoard, checkpointing.

Usage
-----
    python train.py
    python train.py --config configs/pokemon.yaml
    python train.py --game PokemonRed-GB --total-timesteps 5_000_000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import keras
import numpy as np

# Ensure the package root is on sys.path when running from the repo root
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gbagent import __version__  # noqa: E402
from gbagent.config import Config, format_config, load_config  # noqa: E402

# Lazy-imported in main() to control startup order


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gbagent — Game Boy RL agent training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (section fields override defaults)",
    )
    # Environment overrides
    parser.add_argument("--game", type=str, default=None, help="ROM/game name")
    parser.add_argument("--state", type=str, default=None, help="Start state")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--rom-dir", type=str, default=None, help="Custom ROM directory")
    # Training overrides
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training steps")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    # Agent overrides
    parser.add_argument("--learning-rate", type=float, default=None)
    # Dashboard
    parser.add_argument(
        "--no-dashboard",
        action="store_false",
        dest="dashboard",
        default=True,
        help="Disable the Web dashboard server",
    )
    parser.add_argument(
        "--dashboard-host",
        type=str,
        default=None,
        help="Dashboard server bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Dashboard server port (default: 8765)",
    )
    # Performance
    parser.add_argument(
        "--dashboard-update-interval",
        type=int,
        default=None,
        help="Throttle dashboard frames to every N iterations (default: 10)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable dashboard + recorder, reduce logging for max SPS",
    )
    # Misc
    parser.add_argument("--show-config", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser


def override_config(cfg: Config, args: argparse.Namespace) -> None:
    """Apply CLI overrides on top of the loaded config."""
    mappings: list[tuple[str, str]] = [
        ("env", "game"),
        ("env", "state"),
        ("env", "frame_skip"),
        ("env", "frame_stack"),
        ("env", "rom_dir"),
        ("train", "total_timesteps"),
        ("train", "num_envs"),
        ("train", "seed"),
        ("train", "log_dir"),
        ("train", "checkpoint_dir"),
        ("train", "enable_dashboard"),
        ("train", "dashboard_host"),
        ("train", "dashboard_port"),
        ("agent", "learning_rate"),
        ("train", "dashboard_update_interval"),
    ]
    for section, field in mappings:
        val = getattr(args, field, None)
        if val is not None:
            dc = getattr(cfg, section)
            setattr(dc, field, val)

    # --no-dashboard sets args.dashboard, map to enable_dashboard
    dashboard_val = getattr(args, "dashboard", None)
    if dashboard_val is not None and not dashboard_val:
        cfg.train.enable_dashboard = False


# ===================================================================
# PPO training loop
# ===================================================================


def train(cfg: Config, args: argparse.Namespace | None = None) -> None:
    """Run the full PPO training loop.

    Stages per iteration
    --------------------
    1. Collect ``n_steps`` transitions from ``num_envs`` parallel envs.
    2. Compute GAE (λ) advantages.
    3. Run ``update_epochs`` of PPO mini-batch SGD.
    4. Log metrics to TensorBoard and console.
    5. Save model checkpoints periodically.
    """
    # Import here so --show-config / --version don't load heavy deps
    from keras import ops

    from gbagent.action import combine_actions, sample_actions
    from gbagent.agent import ActorCritic
    from gbagent.buffer import RolloutBuffer
    from gbagent.env import GBAGEnv
    from gbagent.ppo import ppo_update_step
    from gbagent.recorder import Recorder
    from gbagent.reward import RewardSystem
    from gbagent.server import DashboardServer, frame_to_png

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    np.random.seed(cfg.train.seed)

    # ------------------------------------------------------------------
    # Performance tuning flags
    # ------------------------------------------------------------------
    if getattr(args, 'fast', False):
        cfg.train.enable_dashboard = False
        cfg.recorder.enabled = False
        cfg.train.log_interval = 500
        cfg.train.save_interval = max(cfg.train.save_interval, 1_000_000)
        print("  ⚡ FAST mode: dashboard OFF, reduced logging, fewer checkpoints")

    dashboard_update_interval = getattr(cfg.train, "dashboard_update_interval", 10)
    if dashboard_update_interval is None:
        dashboard_update_interval = 10

    # ------------------------------------------------------------------
    # Parallel environments
    # ------------------------------------------------------------------
    # Detect GBA mode from config. If btn_size explicitly set to 8, enable
    # GBA mode; otherwise let the env auto-detect from available buttons.
    btn_size = getattr(cfg.agent, "btn_size", 6)
    gba_mode = getattr(cfg.env, "gba_mode", False) or (btn_size >= 8)
    if gba_mode and btn_size < 8:
        btn_size = 8

    # Pre-allocate observation arrays for batch collection
    # (avoids repeated np.array(list)) overhead
    def make_env(rank: int) -> GBAGEnv:
        env = GBAGEnv(
            game=cfg.env.game,
            state=cfg.env.state,
            frame_skip=cfg.env.frame_skip,
            frame_stack=cfg.env.frame_stack,
            gba_mode=gba_mode,
            rom_dir=cfg.env.rom_dir,
        )
        env.reset(seed=cfg.train.seed + rank)
        return env

    print(f"› Creating {cfg.train.num_envs} parallel environments …")
    envs = [make_env(i) for i in range(cfg.train.num_envs)]
    obs_shape = envs[0].observation_space.shape  # (84, 84, 4)
    assert obs_shape is not None and len(obs_shape) == 3

    # Reward systems (one per env — independent RAM/scanner state)
    reward_config = cfg.reward.to_dict()
    reward_systems = [RewardSystem(env, reward_config) for env in envs]

    # ------------------------------------------------------------------
    # Agent (ActorCritic)
    # ------------------------------------------------------------------
    print("› Building ActorCritic model …")
    cfg.agent.btn_size = btn_size  # ensure model uses correct head size
    model = ActorCritic(config=cfg.agent)

    # Build model with dummy batch to materialise weights
    dummy = np.zeros((1, *obs_shape), dtype=np.float32)
    model(dummy, training=False)
    print(f"  ✓ Parameter count: {model.count_params():,}")

    # ------------------------------------------------------------------
    # AdamW optimizer (with weight decay + global norm clipping)
    # ------------------------------------------------------------------
    optimizer = keras.optimizers.AdamW(
        learning_rate=cfg.agent.learning_rate,
        weight_decay=1e-4,
        global_clipnorm=cfg.agent.max_grad_norm,
    )

    # ------------------------------------------------------------------
    # Rollout buffer
    # ------------------------------------------------------------------
    buffer = RolloutBuffer(
        num_envs=cfg.train.num_envs,
        n_steps=cfg.train.n_steps,
        obs_shape=obs_shape,
    )

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    log_dir = Path(cfg.train.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = _create_tb_writer(str(log_dir))

    # ------------------------------------------------------------------
    # Dashboard server (background thread)
    # ------------------------------------------------------------------
    dashboard: DashboardServer | None = None
    if cfg.train.enable_dashboard:
        addr = f"{cfg.train.dashboard_host}:{cfg.train.dashboard_port}"
        print(f"› Starting dashboard server on {addr} …")
        dashboard = DashboardServer(
            host=cfg.train.dashboard_host,
            port=cfg.train.dashboard_port,
        )
        dashboard.start(config=cfg)
        print(f"  ✓ Dashboard at http://{addr}")

    # ------------------------------------------------------------------
    # Recorder (frame-by-frame PNG + JSONL)
    # ------------------------------------------------------------------
    recorder = Recorder(
        output_dir=cfg.recorder.output_dir,
        fps=60 // cfg.env.frame_skip,
    )
    if dashboard is not None:
        dashboard.recorder = recorder

    if cfg.recorder.enabled:
        session_name = recorder.start(start_step=0)
        print(f"  ✓ Recording started → {session_name}")

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    global_step = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_returns = deque(maxlen=100)  # trailing 100-episode mean

    # Collect initial state
    obs = np.array([env.reset()[0] for env in envs], dtype=np.float32)

    # Initialise reward systems after env reset
    for rs in reward_systems:
        rs.reset()

    # LSTM state — tuple of (h, c) each of shape (num_envs, hidden_dim) or None
    lstm_state = None

    print(f"\n{'=' * 58}")
    print("  Starting PPO training loop")
    print(f"  Total timesteps: {cfg.train.total_timesteps:,}")
    print(f"  Rollout length:  {cfg.train.n_steps} steps × {cfg.train.num_envs} envs")
    print(f"  Update epochs:   {cfg.agent.update_epochs}")
    print(f"  Minibatches:     {cfg.agent.num_minibatches}")
    print(f"  Batch size:      {cfg.train.batch_size}")
    print(f"{'=' * 58}\n")

    start_time = time.time()
    next_save_step = cfg.train.save_interval
    next_log_episode = cfg.train.log_interval

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    iter_counter = 0  # iteration counter for dashboard throttling

    # Pre-allocate observation buffer to avoid list→array overhead
    _obs_buf = np.empty((cfg.train.num_envs, *obs_shape), dtype=np.float32)

    while global_step < cfg.train.total_timesteps:
        iter_counter += 1

        # ──────────────────────────────────────────────────────────────
        # 1. Collect rollout
        # ──────────────────────────────────────────────────────────────
        buffer.clear()
        ep_info_buf: list[dict] = []

        _bonus_samples: list[float] = []  # shaped-bonus tracking

        # Pre-allocate arrays for batch env results
        _next_obs_buf = np.empty_like(_obs_buf)
        _rewards_buf = np.empty(cfg.train.num_envs, dtype=np.float32)
        _dones_buf = np.empty(cfg.train.num_envs, dtype=bool)

        for _step in range(cfg.train.n_steps):
            # Forward pass with LSTM state carry-over
            dpad_logits, btn_logits, values, (h, c) = model(
                obs, state=lstm_state, training=False,
            )
            dpad_action, btn_action, log_probs = sample_actions(
                ops.convert_to_numpy(dpad_logits),
                ops.convert_to_numpy(btn_logits),
                training=True,
            )
            values_np = ops.convert_to_numpy(values).squeeze(-1)  # (num_envs,)

            # Step environments — batch into pre-allocated arrays
            actions = combine_actions(dpad_action, btn_action, btn_size=btn_size)

            for env_idx, env in enumerate(envs):
                n_obs, r, term, trunc, info = env.step(int(actions[env_idx]))
                done = term or trunc

                # Shaped reward bonus
                rs = reward_systems[env_idx]
                bonus = rs.step(n_obs)
                r_modified = r + bonus
                _bonus_samples.append(bonus)

                # Write into pre-allocated arrays
                _next_obs_buf[env_idx] = n_obs
                _rewards_buf[env_idx] = r_modified
                _dones_buf[env_idx] = done

                # Reset reward system state and LSTM state when episode ends
                if done:
                    ep_info_buf.append(info)
                    rs.reset()

                # ── Record frame from the first environment ──────────
                if env_idx == 0 and recorder.is_recording:
                    try:
                        raw_frame = env.raw_env.render()
                        if raw_frame is not None:
                            frame_metadata = {
                                "step": global_step + _step,
                                "env_step": _step,
                                "action": int(actions[env_idx]),
                                "raw_reward": float(r),
                                "reward": float(r_modified),
                                "bonus": float(bonus),
                                "value": float(values_np[env_idx]),
                                "logprob": float(log_probs[env_idx]),
                                "done": bool(done),
                            }
                            recorder.record_frame(raw_frame, frame_metadata)
                    except Exception:
                        pass  # recording is best-effort

            # Store in buffer (use pre-allocated arrays directly)
            buffer.store(
                obs,
                dpad_action,
                btn_action,
                _rewards_buf,
                _dones_buf,
                log_probs,
                values_np,
            )

            # Swap observation buffers for next iteration
            obs, _obs_buf, _next_obs_buf = _next_obs_buf, _obs_buf, _next_obs_buf

            # Update LSTM state — zero out for terminated episodes
            h_np = ops.convert_to_numpy(h)
            c_np = ops.convert_to_numpy(c)
            dones_np = _dones_buf.astype(np.float32)
            h_np = h_np * (1.0 - dones_np[:, np.newaxis])
            c_np = c_np * (1.0 - dones_np[:, np.newaxis])
            lstm_state = (h_np, c_np)

        # ──────────────────────────────────────────────────────────────
        # 2. Compute GAE
        # ──────────────────────────────────────────────────────────────
        _, _, last_values, _ = model(obs, state=lstm_state, training=False)
        last_values_np = ops.convert_to_numpy(last_values).squeeze(-1)  # (num_envs,)

        buffer.compute_gae(
            last_value=last_values_np,
            gamma=cfg.agent.gamma,
            gae_lambda=cfg.agent.gae_lambda,
        )

        # ──────────────────────────────────────────────────────────────
        # 3. PPO updates
        # ──────────────────────────────────────────────────────────────
        # Compute mini-batch size from num_minibatches
        total_samples = cfg.train.n_steps * cfg.train.num_envs
        mini_batch_size = max(1, total_samples // cfg.agent.num_minibatches)

        pg_losses: list[float] = []
        vf_losses: list[float] = []
        ent_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        for _epoch in range(cfg.agent.update_epochs):
            for batch in buffer.get_batches(batch_size=mini_batch_size):
                (
                    obs_b,
                    dpad_b,
                    btn_b,
                    ret_b,
                    adv_b,
                    old_lp_b,
                ) = batch

                obs_t = ops.convert_to_tensor(obs_b)
                dpad_t = ops.convert_to_tensor(dpad_b)
                btn_t = ops.convert_to_tensor(btn_b)
                ret_t = ops.convert_to_tensor(ret_b)
                adv_t = ops.convert_to_tensor(adv_b)
                old_lp_t = ops.convert_to_tensor(old_lp_b)

                losses = ppo_update_step(
                    model=model,
                    optimizer=optimizer,
                    obs_batch=obs_t,
                    dpad_batch=dpad_t,
                    btn_batch=btn_t,
                    return_batch=ret_t,
                    adv_batch=adv_t,
                    old_lp_batch=old_lp_t,
                    clip_epsilon=cfg.agent.clip_epsilon,
                    value_coef=cfg.agent.value_coef,
                    entropy_coef=cfg.agent.entropy_coef,
                )

                pg_losses.append(losses["policy_loss"])
                vf_losses.append(losses["value_loss"])
                ent_losses.append(losses["entropy_loss"])
                approx_kls.append(losses["approx_kl"])
                clip_fracs.append(losses["clip_frac"])

        # ──────────────────────────────────────────────────────────────
        # 4. Logging
        # ──────────────────────────────────────────────────────────────
        global_step += cfg.train.n_steps * cfg.train.num_envs

        # Episode tracking (from done events during rollout)
        if ep_info_buf:
            for info in ep_info_buf:
                ep_ret = info.get("episode", {}).get("r", 0.0)
                ep_len = info.get("episode", {}).get("l", 0)
                if ep_ret != 0.0:
                    episode_rewards.append(ep_ret)
                    episode_lengths.append(ep_len)
                    episode_returns.append(ep_ret)

        # TensorBoard — write every iteration
        log_data = {
            "loss/policy": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "loss/value": float(np.mean(vf_losses)) if vf_losses else 0.0,
            "loss/entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
            "loss/total": float(
                np.mean(pg_losses)
                + cfg.agent.value_coef * np.mean(vf_losses)
                + cfg.agent.entropy_coef * np.mean(ent_losses)
            )
            if pg_losses
            else 0.0,
            "policy/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "policy/clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "policy/learning_rate": float(
                optimizer.learning_rate.numpy()
                if hasattr(optimizer.learning_rate, "numpy")
                else optimizer.learning_rate
            ),
            "train/global_step": global_step,
            "train/sps": global_step / (time.time() - start_time + 1e-8),
            "train/epochs_done": cfg.agent.update_epochs,
        }
        if _bonus_samples:
            log_data["reward/bonus_mean"] = float(np.mean(_bonus_samples))
            log_data["reward/bonus_max"] = float(np.max(_bonus_samples))
        if episode_returns:
            log_data["episode/return_mean"] = float(np.mean(episode_returns))
            log_data["episode/return_max"] = float(np.max(list(episode_returns)))
            log_data["episode/return_min"] = float(np.min(list(episode_returns)))
            log_data["episode/length_mean"] = float(
                np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
            )

        # Write via TensorBoard callback's internal writer
        _log_scalars(writer, log_data, global_step)

        # ──────────────────────────────────────────────────────────────
        # 5. Console progress
        # ──────────────────────────────────────────────────────────────
        if len(episode_rewards) >= next_log_episode or next_log_episode == cfg.train.log_interval:
            sps = global_step / (time.time() - start_time + 1e-8)
            elapsed = time.time() - start_time
            pct = 100.0 * global_step / max(cfg.train.total_timesteps, 1)

            avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            avg_pg = float(np.mean(pg_losses)) if pg_losses else 0.0
            avg_vf = float(np.mean(vf_losses)) if vf_losses else 0.0
            avg_ent = float(np.mean(ent_losses)) if ent_losses else 0.0
            avg_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
            avg_cf = float(np.mean(clip_fracs)) if clip_fracs else 0.0

            print(
                f"[{elapsed:>8.0f}s]  "
                f"step {global_step:>9,} ({pct:5.1f}%)  "
                f"SPS {sps:>6.0f}  "
                f"R {avg_return:6.2f}  "
                f"PG {avg_pg:.3f}  "
                f"VF {avg_vf:.3f}  "
                f"Ent {avg_ent:.3f}  "
                f"KL {avg_kl:.4f}  "
                f"CF {avg_cf:.3f}"
            )

            # Set next log point  (one more rollout)
            next_log_episode = len(episode_rewards) + 1

        # ──────────────────────────────────────────────────────────────
        # 4b. Dashboard broadcast (throttled)
        # ──────────────────────────────────────────────────────────────
        if dashboard is not None and dashboard.hub.num_clients > 0:
            import contextlib

            # Only broadcast frames/PNG every N iterations to save CPU
            if iter_counter % dashboard_update_interval == 0:
                with contextlib.suppress(Exception):
                    raw_frame = envs[0].raw_env.render()
                    if raw_frame is not None:
                        png_bytes = frame_to_png(raw_frame)
                        dashboard.broadcast_frame(png_bytes)

            # Always broadcast metrics (lightweight — just JSON)
            with contextlib.suppress(Exception):
                dashboard.broadcast_metrics(log_data)

        # ──────────────────────────────────────────────────────────────
        # 4c. Check stop_requested
        # ──────────────────────────────────────────────────────────────
        if dashboard is not None and dashboard.hub.stop_requested:
            print("\n  ⏹ Stop requested — saving checkpoint and exiting …")
            ckpt_name = f"model_stopped_step_{global_step:09d}.keras"
            ckpt_path = str(checkpoint_dir / ckpt_name)
            model.save(ckpt_path)
            print(f"  ✓ Saved → {ckpt_path}")
            dashboard.broadcast_event("train_stopped", msg="Training halted by user")
            dashboard.hub.stop_requested = False
            dashboard.hub.train_requested = False

            # Stop recording if active
            if recorder.is_recording:
                summary = recorder.stop(encode_mp4=cfg.recorder.encode_mp4_on_stop)
                _log_recorder_summary(summary)

            break

        # ──────────────────────────────────────────────────────────────
        # 5/6. Checkpointing (renumbered to avoid collision)
        # ──────────────────────────────────────────────────────────────
        # This is the original checkpointing block; epoch/episode numbering
        # has been preserved from the original code for clarity.
        # ──────────────────────────────────────────────────────────────
        if global_step >= next_save_step:
            ckpt_name = f"model_step_{global_step:09d}.keras"
            ckpt_path = str(checkpoint_dir / ckpt_name)
            model.save(ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}  ({global_step:,} steps)")
            next_save_step += cfg.train.save_interval

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    ckpt_name = f"model_final_step_{global_step:09d}.keras"
    ckpt_path = str(checkpoint_dir / ckpt_name)
    model.save(ckpt_path)
    total_time = time.time() - start_time
    print(f"\n✅ Training complete  ({total_time:.0f}s, {global_step:,} steps)")
    print(f"   Final checkpoint → {ckpt_path}")
    if episode_rewards:
        print(f"   Best 100-ep avg return: {np.max([np.mean(episode_returns)]):.2f}")

    # Clean-up — stop recording if still active
    if recorder.is_recording:
        summary = recorder.stop(encode_mp4=cfg.recorder.encode_mp4_on_stop)
        _log_recorder_summary(summary)

    for env in envs:
        env.close()

    if dashboard is not None:
        msg = f"Training finished at {global_step:,} steps"
        dashboard.broadcast_event("train_complete", msg=msg)
        dashboard.stop()

    _write_final_summary(writer, global_step)


# ===================================================================
# Recorder helpers
# ===================================================================


def _log_recorder_summary(summary: dict) -> None:
    """Print a compact recorder summary to the console."""
    if not summary:
        return
    frames = summary.get("frames", 0)
    duration = summary.get("duration_s", 0.0)
    encoded = summary.get("encoded", False)
    encoded_path = summary.get("encoded_path")
    parts = [f"  ✓ Recording finished — {frames} frames ({duration:.1f}s)"]
    if encoded:
        parts.append(f"       MP4 → {encoded_path}")
    print("\n".join(parts))


# ===================================================================
# TensorBoard helpers
# ===================================================================


def _create_tb_writer(log_dir: str):
    """Create a TensorBoard writer (backend-agnostic, standalone tensorboard pkg)."""
    try:
        from tensorboard.summary import Writer  # ty: ignore[unresolved-import]

        return Writer(log_dir)
    except Exception:
        return None


def _log_scalars(writer, data: dict[str, float], step: int) -> None:
    """Write scalar values to TensorBoard."""
    if writer is None:
        return
    try:
        for key, value in data.items():
            writer.add_scalar(key, value, step=step)
    except Exception:
        pass  # non-critical


def _write_final_summary(writer, step: int) -> None:
    """Write a final 'done' marker to TensorBoard."""
    if writer is None:
        return
    from contextlib import suppress
    with suppress(Exception):
        writer.add_text("training/status", "complete", step=step)


# ===================================================================
# Main entry point
# ===================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"gbagent v{__version__}")
        sys.exit(0)

    # Load config (YAML + CLI overrides)
    cfg = load_config(args.config)
    override_config(cfg, args)

    if args.show_config:
        print(format_config(cfg))
        sys.exit(0)

    print("╔══════════════════════════════════════════╗")
    print("║   GBAGent — Game Boy RL Agent Trainer   ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print(format_config(cfg))
    print()

    # Run training
    train(cfg)


if __name__ == "__main__":
    main()
