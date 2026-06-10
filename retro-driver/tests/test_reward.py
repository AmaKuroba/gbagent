from __future__ import annotations

import numpy as np
import pytest

from retro_driver.config import ScannerConfig
from retro_driver.reward import RewardConfig, RewardSystem


class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.novelty_scale == 0.2
        assert cfg.min_reward == -10.0
        assert cfg.max_reward == 10.0
        assert cfg.scanners == []

    def test_custom(self):
        cfg = RewardConfig(novelty_scale=0.5, min_reward=-5.0, max_reward=5.0)
        assert cfg.novelty_scale == 0.5
        assert cfg.min_reward == -5.0
        assert cfg.max_reward == 5.0


class TestRewardSystem:
    def test_reset(self):
        rs = RewardSystem(RewardConfig())
        rs.reset()
        assert rs._prev_frame is None
        assert rs._novelty_buffer == []

    def test_compute_first_step(self):
        rs = RewardSystem(RewardConfig())
        rs.reset()
        frame = np.zeros((144, 160), dtype=np.uint8)
        reward = rs.compute(frame, None, None)
        # First step gives novelty bonus
        assert isinstance(reward, float)

    def test_compute_identical_frames(self):
        rs = RewardSystem(RewardConfig())
        rs.reset()
        frame = np.zeros((144, 160), dtype=np.uint8)
        # Two identical frames should give near-zero novelty
        r1 = rs.compute(frame, None, None)
        r2 = rs.compute(frame, frame, None)
        # Second call: no pixel change, novelty ≈ 0
        assert r2 < r1

    def test_compute_different_frames(self):
        rs = RewardSystem(RewardConfig())
        rs.reset()
        frame_a = np.zeros((144, 160), dtype=np.uint8)
        frame_b = np.ones((144, 160), dtype=np.uint8) * 255
        rs.compute(frame_a, None, None)
        reward = rs.compute(frame_b, frame_a, None)
        # Large pixel change should give positive reward
        assert reward > 0

    def test_staleness_penalty(self):
        rs = RewardSystem(RewardConfig(novelty_decay_per_stimulus=5))
        rs.reset()
        frame = np.ones((144, 160), dtype=np.uint8) * 128
        rs.compute(frame, None, None)
        # Feed same frame repeatedly to trigger staleness
        for _ in range(10):
            rs.compute(frame, frame, None)
        breakdown = rs.last_breakdown()
        assert "stale_penalty" in breakdown
        assert breakdown["stale_penalty"] < 0

    def test_anti_grind_same_tile(self):
        rs = RewardSystem(RewardConfig(anti_grind_after=5, anti_grind_movement_bonus=0.05))
        rs.reset()
        frame = np.zeros((144, 160), dtype=np.uint8)
        rs.compute(frame, None, None)
        # Stay on same tile enough times to trigger penalty
        for _ in range(10):
            rs.compute(frame, frame, None)
        breakdown = rs.last_breakdown()
        assert breakdown.get("grind_penalty", 0) < 0

    def test_anti_grind_new_tile(self):
        rs = RewardSystem(RewardConfig(anti_grind_movement_bonus=0.05))
        rs.reset()
        frame = np.zeros((144, 160), dtype=np.uint8)
        rs.compute(frame, None, None)
        # Different tile (different bottom region)
        frame2 = np.copy(frame)
        frame2[-16:, 64:96] = 255
        rs.compute(frame2, frame, None)
        breakdown = rs.last_breakdown()
        assert breakdown.get("movement", 0) > 0

    def test_ram_scanner_discrete(self):
        class MockClient:
            def read_ram(self, addr):
                return 5

        scanner = ScannerConfig(name="test", ram_addr=0xD000, type="discrete", reward_on_change=2.0)
        cfg = RewardConfig(scanners=[scanner])
        rs = RewardSystem(cfg)
        rs.reset()
        client = MockClient()
        # First read triggers change
        reward = rs.compute(np.zeros((144, 160), dtype=np.uint8), None, client)
        assert reward != 0  # should include scanner reward
        # Second read: same value, no change
        reward2 = rs.compute(np.ones((144, 160), dtype=np.uint8), None, client)

    def test_ram_scanner_delta(self):
        class MockClient:
            def __init__(self):
                self._call_count = 0

            def read_ram(self, addr):
                self._call_count += 1
                return self._call_count

        scanner = ScannerConfig(
            name="delta_test", ram_addr=0xD000, type="delta", length=2, reward_per_unit=1.0
        )
        cfg = RewardConfig(scanners=[scanner])
        rs = RewardSystem(cfg)
        rs.reset()
        client = MockClient()
        reward = rs.compute(np.zeros((144, 160), dtype=np.uint8), None, client)
        assert isinstance(reward, float)

    def test_clipping(self):
        rs = RewardSystem(RewardConfig(min_reward=-1.0, max_reward=1.0))
        rs.reset()
        # Create massive pixel change that would overflow normal reward
        frame_a = np.zeros((144, 160), dtype=np.uint8)
        frame_b = np.ones((144, 160), dtype=np.uint8) * 255
        rs.compute(frame_a, None, None)
        reward = rs.compute(frame_b, frame_a, None)
        assert -1.0 <= reward <= 1.0

    def test_last_breakdown(self):
        rs = RewardSystem(RewardConfig())
        rs.reset()
        frame = np.zeros((144, 160), dtype=np.uint8)
        rs.compute(frame, None, None)
        bd = rs.last_breakdown()
        assert isinstance(bd, dict)
        assert "novelty" in bd
