"""Reward system for retro-driver.

Hybrid approach:
1. Screen novelty baseline — rewards meaningful pixel change
2. Per-game RAM scanners — config-driven progress signals
3. Anti-grind diminishing returns

All reward components are summed per step and clipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_driver.config import ScannerConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the reward system."""

    # Screen novelty
    novelty_scale: float = 0.2
    novelty_decay_per_stimulus: int = 30  # steps before same screen mode decays

    # Anti-grind
    anti_grind_after: int = 100  # consecutive steps on same tile before floor
    anti_grind_movement_bonus: float = 0.10  # reward for moving to a new tile

    # Per-game scanners
    scanners: list[ScannerConfig] = field(default_factory=list)

    # Clipping
    min_reward: float = -10.0
    max_reward: float = 10.0


class RewardSystem:
    """Computes reward from screen frames and optional RAM scanners."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Call at start of each episode."""
        self._prev_frame: np.ndarray | None = None
        self._novelty_buffer: list[float] = []  # rolling window of novelty values
        self._same_screen_steps = 0
        self._same_tile_steps = 0
        self._last_breakdown: dict[str, float] = {}
        self._last_map_id: int | None = None
        self._last_tile_id: int | None = None
        self._tile_visits: set[int] = set()

    def compute(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        mcp_client: Any,  # MCPClient duck-type
    ) -> float:
        """Compute reward for the current step.

        Args:
            frame: Current grayscale frame (H, W).
            prev_frame: Previous frame (H, W) or None.
            mcp_client: MCPClient instance for RAM reads.

        Returns:
            Total reward for this step, clipped.
        """
        total = 0.0
        breakdown: dict[str, float] = {}

        # 1. Screen novelty
        novelty = self._compute_novelty(frame, prev_frame)
        breakdown["novelty"] = novelty
        total += novelty

        # 2. Anti-grind screen-mode decay
        if novelty < 0.1:
            self._same_screen_steps += 1
        else:
            self._same_screen_steps = max(0, self._same_screen_steps - 1)

        decay_factor = max(
            0.0, 1.0 - self._same_screen_steps / self.config.novelty_decay_per_stimulus
        )
        if decay_factor < 0.5:
            penalty = -0.01 * (1.0 - decay_factor)
            breakdown["stale_penalty"] = penalty
            total += penalty

        # 3. RAM scanners
        ram_reward = self._scan_ram(mcp_client)
        breakdown["ram_scanners"] = ram_reward
        total += ram_reward

        # 4. Anti-grind tile movement
        tile_id = self._current_tile_id(frame)
        if tile_id is not None:
            movement = 0.0
            if tile_id not in self._tile_visits:
                self._tile_visits.add(tile_id)
                self._same_tile_steps = 0
                movement = self.config.anti_grind_movement_bonus
            else:
                self._same_tile_steps += 1

            if self._same_tile_steps >= self.config.anti_grind_after:
                # Floor the reward — minimal reward until agent moves
                movement = -0.1
                breakdown["grind_penalty"] = movement
                total += movement
            else:
                breakdown["movement"] = movement
                total += movement

        self._last_breakdown = breakdown

        # Clip
        return max(self.config.min_reward, min(self.config.max_reward, total))

    def last_breakdown(self) -> dict[str, float]:
        return dict(self._last_breakdown)

    # ── components ────────────────────────────────────────────

    def _compute_novelty(self, curr: np.ndarray, prev: np.ndarray | None) -> float:
        if prev is None or curr.shape != prev.shape:
            return 0.5  # first step bonus

        diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
        mean_diff = float(diff.mean()) / 255.0  # normalize to [0, 1]

        # Mean diff of ~5% pixel change gives ~0.5 reward
        scaled = mean_diff * 5.0 * self.config.novelty_scale

        # Clamp novelty contribution
        return max(-1.0, min(2.0, scaled))

    def _scan_ram(self, client: Any) -> float:
        """Run configured RAM scanners and sum rewards."""
        total = 0.0
        for scanner in self.config.scanners:
            try:
                if scanner.type == "state_penalty":
                    val = client.read_ram(scanner.ram_addr)
                    if val == scanner.target_value:
                        total += scanner.penalty_per_step
                elif scanner.type == "first_visit":
                    val = client.read_ram(scanner.ram_addr)
                    if scanner._seen_set is not None and val not in scanner._seen_set:
                        scanner._seen_set.add(val)
                        total += scanner.reward_on_change
                elif scanner.type == "stuck_penalty":
                    val = client.read_ram(scanner.ram_addr)
                    if scanner._last_stuck_val is not None and val == scanner._last_stuck_val:
                        scanner._stuck_count += 1
                        if scanner._stuck_count > scanner.grace_period:
                            total += scanner.penalty_per_step
                    else:
                        scanner._stuck_count = 0
                    scanner._last_stuck_val = val
                elif scanner.type == "discrete":
                    val = client.read_ram(scanner.ram_addr)
                    if val != scanner._last_val:  # type: ignore[attr-defined]
                        total += scanner.reward_on_change
                        scanner._last_val = val  # type: ignore[attr-defined]
                elif scanner.type == "delta":
                    vals = [
                        client.read_ram(addr)
                        for addr in range(
                            scanner.ram_addr, scanner.ram_addr + (scanner.length or 1)
                        )
                    ]
                    current = sum(v for v in vals if v is not None)
                    if scanner._last_sum is not None:  # type: ignore[attr-defined]
                        delta = current - scanner._last_sum
                        if delta > 0:
                            total += delta * scanner.reward_per_unit
                    scanner._last_sum = current  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("RAM scanner '%s' failed: %s", scanner.name, exc)
        return total

    def _current_tile_id(self, frame: np.ndarray) -> int | None:
        """Compute a coarse tile hash from the bottom portion of the frame.

        This is a lightweight heuristic — proper game-specific tile detection
        would come from a RAM scanner.
        """
        if frame.shape[0] < 16 or frame.shape[1] < 16:
            return None
        # Hash the center-bottom region where the player usually stands
        bottom = frame[-16:, 64:96]
        return int(bottom.mean()) // 4  # crude 0-63 range
