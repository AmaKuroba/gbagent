"""YAML-based configuration loader for gbagent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("gbagent.config")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    game: str = "PokemonRed-GB"
    state: str = "Level1"
    frame_skip: int = 4
    frame_stack: int = 4
    render_mode: str | None = None
    gba_mode: bool = False  # True → add L/R shoulder buttons
    rom_dir: str = "roms"   # custom ROM directory (relative to CWD)


@dataclass
class AgentConfig:
    """ViT + Actor-Critic hyper-parameters."""

    # ViT
    patch_size: int = 16
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Actor-Critic heads
    hidden_dim: int = 512

    btn_size: int = 6  # 6 (GB) or 8 (GBA auto-detected)

    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    num_envs: int = 8
    n_steps: int = 128  # steps per env per rollout
    batch_size: int = 256
    log_interval: int = 100  # episodes
    save_interval: int = 500_000  # timesteps
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42
    enable_dashboard: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_update_interval: int = 10  # throttle frame broadcasts to every N iterations


@dataclass
class RewardConfig:
    """Shaped-reward parameters.

    Fields are loaded from the ``reward:`` section of the YAML config.
    The ``ram_scanners`` field holds a list of scanner dicts — the config
    loader passes them through as-is.
    """

    screen_novelty_scale: float = 0.0
    stale_penalty: float = 0.0
    stale_threshold: int = 100
    stale_diff_threshold: float = 0.001
    ram_scanners: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return all fields as a plain dict (useful for RewardSystem)."""
        return {
            "screen_novelty_scale": self.screen_novelty_scale,
            "stale_penalty": self.stale_penalty,
            "stale_threshold": self.stale_threshold,
            "stale_diff_threshold": self.stale_diff_threshold,
            "ram_scanners": self.ram_scanners,
        }


@dataclass
class RecorderConfig:
    output_dir: str = "recordings"
    enabled: bool = False
    encode_mp4_on_stop: bool = True
    ffmpeg_crf: int = 23
    ffmpeg_preset: str = "medium"
    max_frames: int = 0  # 0 = unlimited


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, merging with defaults.

    Any field not present in the YAML file retains its default value.
    If *path* is ``None``, the loader looks for ``configs/default.yaml``
    relative to the project root (walked from the calling module), or
    falls back to pure defaults.
    """
    raw: dict[str, Any] = {}

    path = Path(path) if path is not None else _find_default_config()

    if path and path.is_file():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()

    for section in ("env", "agent", "train", "reward", "recorder"):
        section_cfg = raw.get(section, {}) or {}
        if not isinstance(section_cfg, dict):
            continue
        dc = getattr(cfg, section)
        valid_keys = dc.__dataclass_fields__
        for key, value in section_cfg.items():
            if key in valid_keys:
                setattr(dc, key, value)
            else:
                logger.warning("Unknown config field '%s.%s' — ignored", section, key)

    return cfg


def _find_default_config() -> Path | None:
    """Walk up from CWD or script dir looking for ``configs/default.yaml``."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        current = start
        for _ in range(8):  # max depth
            candidate = current / "configs" / "default.yaml"
            if candidate.is_file():
                return candidate.resolve()
            if (current.parent / "configs" / "default.yaml").is_file():
                return (current.parent / "configs" / "default.yaml").resolve()
            if current.parent == current:
                break
            current = current.parent
    return None


# ---------------------------------------------------------------------------
# Pretty-printer (for CLI / logs)
# ---------------------------------------------------------------------------


def format_config(cfg: Config) -> str:
    """Return a human-readable YAML-ish representation of the config."""
    lines = ["# GBAGent Configuration", "---"]
    for section_name in ("env", "agent", "train", "reward", "recorder"):
        lines.append(f"\n{section_name}:")
        dc = getattr(cfg, section_name)
        for key in dc.__dataclass_fields__:
            val = getattr(dc, key)
            # Print scanners as an indented YAML block when non-empty
            if key == "ram_scanners" and val:
                lines.append(f"  {key}:")
                for sc in val:
                    lines.append(f"    - # {sc.get('name', '')}")
                    for sk, sv in sc.items():
                        lines.append(f"      {sk}: {sv!r}")
            else:
                lines.append(f"  {key}: {val!r}")
    return "\n".join(lines)


def save_config(cfg: Config, path: str | Path) -> None:
    """Dump the current config to YAML (preserves section structure)."""
    raw: dict[str, dict[str, Any]] = {}
    for section_name in ("env", "agent", "train", "reward", "recorder"):
        dc = getattr(cfg, section_name)
        raw[section_name] = {k: getattr(dc, k) for k in dc.__dataclass_fields__}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)


# Re-export for convenience
__all__ = [
    "Config",
    "EnvConfig",
    "AgentConfig",
    "TrainConfig",
    "RewardConfig",
    "RecorderConfig",
    "load_config",
    "format_config",
    "save_config",
]
