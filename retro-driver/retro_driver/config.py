"""Configuration loading for retro-driver scanners and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ScannerConfig:
    """Configuration for a RAM scanner reward source."""

    name: str
    ram_addr: int
    type: str = "discrete"  # "discrete" | "delta" | "state_penalty" | "first_visit" | "stuck_penalty"
    reward_on_change: float = 1.0
    reward_per_unit: float = 0.1
    length: int = 1
    novelty_decay: bool = False
    target_value: int = 0  # used by state_penalty
    penalty_per_step: float = -0.1  # used by state_penalty / stuck_penalty
    grace_period: int = 300  # steps before stuck_penalty kicks in

    def __post_init__(self) -> None:
        self._last_val: int | None = None
        self._last_sum: int | None = None
        self._stuck_count: int = 0
        self._last_stuck_val: int | None = None
        self._seen_set: set[int] | None = None if self.type != "first_visit" else set()


@dataclass
class GameConfig:
    """Per-game configuration."""

    game: str
    scanners: list[ScannerConfig] = field(default_factory=list)


def load_game_config(path: str | Path) -> GameConfig:
    """Load a YAML game config file."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    scanners = [ScannerConfig(**s) for s in raw.get("scanners", [])]
    return GameConfig(game=raw.get("game", path.stem), scanners=scanners)
