from __future__ import annotations

import numpy as np
import pytest

from gbagent.reward import RewardSystem


class MockRAMEnv:
    """Minimal mock of a GBAGEnv for testing RewardSystem."""

    def __init__(self, ram: np.ndarray | None = None):
        self._ram = ram if ram is not None else np.zeros(0x10000, dtype=np.uint8)

    @property
    def raw_env(self):
        return self

    def get_ram(self):
        return self._ram


def make_reward_config(**overrides):
    config = {
        "screen_novelty_scale": 0.0,
        "stale_penalty": 0.0,
        "stale_threshold": 100,
        "stale_diff_threshold": 0.001,
        "ram_scanners": [],
    }
    config.update(overrides)
    return config


class TestScreenNovelty:
    def test_disabled(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(screen_novelty_scale=0.0))
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == 0.0

    def test_first_frame_no_bonus(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(screen_novelty_scale=0.1))
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == 0.0

    def test_novelty_bonus(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(screen_novelty_scale=1.0))
        frame1 = np.zeros((84, 84, 4), dtype=np.float32)
        rs.step(frame1)  # first call, no bonus
        frame2 = np.ones((84, 84, 4), dtype=np.float32)
        bonus = rs.step(frame2)
        # Mean diff of 1.0 between zeros and ones
        assert bonus == 1.0

    def test_no_novelty_same_frame(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(screen_novelty_scale=1.0))
        frame = np.random.randn(84, 84, 4).astype(np.float32)
        rs.step(frame)
        bonus = rs.step(frame.copy())
        assert bonus < 0.01  # near-zero (floating point)


class TestStalePenalty:
    def test_disabled(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(
            stale_penalty=0.0, stale_threshold=1, stale_diff_threshold=0.5,
        ))
        for _ in range(10):
            rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        # stale_penalty=0 → counter not updated (returns early)
        assert rs.stale_steps == 0

    def test_no_penalty_before_threshold(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(
            stale_penalty=-0.1, stale_threshold=50, stale_diff_threshold=0.5,
        ))
        for _ in range(50):
            bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
            assert bonus == 0.0, f"failed at iter {_}"

    def test_penalty_after_threshold(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(
            stale_penalty=-0.1, stale_threshold=10, stale_diff_threshold=0.5,
        ))
        # First call doesn't increment (no prev_frame). Need 12 calls
        # so stale_steps reaches 11 > 10.
        for _ in range(11):
            rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == -0.1

    def test_penalty_resets_on_motion(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(
            stale_penalty=-0.1, stale_threshold=10, stale_diff_threshold=0.5,
        ))
        for _ in range(10):
            rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        # Change the frame → resets stale counter
        rs.step(np.ones((84, 84, 4), dtype=np.float32))
        assert rs.stale_steps == 0


class TestRAMScanners:
    def test_no_scanners(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config())
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == 0.0

    def test_discrete_scanner_hit(self):
        env = MockRAMEnv()
        # Set the address before creating RewardSystem
        env._ram[0xD163] = 42
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xD163,
             "target_value": 42, "reward": 1.0},
        ]))
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == 1.0

    def test_discrete_scanner_no_hit(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 99
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xD163,
             "target_value": 42, "reward": 1.0},
        ]))
        bonus = rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert bonus == 0.0

    def test_discrete_scanner_once_per_transition(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 42
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xD163,
             "target_value": 42, "reward": 1.0},
        ]))
        assert rs.step(np.zeros((84, 84, 4))) == 1.0  # hit
        assert rs.step(np.zeros((84, 84, 4))) == 0.0  # already rewarded
        # Leave the target and come back
        env._ram[0xD163] = 0
        assert rs.step(np.zeros((84, 84, 4))) == 0.0
        env._ram[0xD163] = 42
        assert rs.step(np.zeros((84, 84, 4))) == 1.0  # re-rewarded

    def test_delta_scanner_positive(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 10
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "delta", "address": 0xD163,
             "delta_sign": "positive", "reward": 0.5},
        ]))
        rs.step(np.zeros((84, 84, 4)))  # first step (init prev_value)
        env._ram[0xD163] = 20
        bonus = rs.step(np.zeros((84, 84, 4)))  # +10 delta
        assert bonus == 0.5

    def test_delta_scanner_negative_not_rewarded(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 20
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "delta", "address": 0xD163,
             "delta_sign": "positive", "reward": 0.5},
        ]))
        rs.step(np.zeros((84, 84, 4)))
        env._ram[0xD163] = 10
        bonus = rs.step(np.zeros((84, 84, 4)))  # -10 delta, should not reward
        assert bonus == 0.0

    def test_delta_scanner_any(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 10
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "delta", "address": 0xD163,
             "delta_sign": "any", "reward": 0.5},
        ]))
        rs.step(np.zeros((84, 84, 4)))
        env._ram[0xD163] = 5
        bonus = rs.step(np.zeros((84, 84, 4)))  # -5 delta, any sign
        assert bonus == 0.5

    def test_multi_byte_scanner(self):
        env = MockRAMEnv()
        # Write 5 as little-endian 2-byte at address
        env._ram[0xD347] = 5
        env._ram[0xD348] = 0
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "multi_byte", "address": 0xD347,
             "n_bytes": 2, "byte_order": "little", "target_value": 5,
             "reward": 2.0},
        ]))
        bonus = rs.step(np.zeros((84, 84, 4)))
        assert bonus == 2.0

    def test_delta_multi_scanner(self):
        env = MockRAMEnv()
        # First step at value 10
        env._ram[0xD347] = 10
        env._ram[0xD348] = 0
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "delta_multi", "address": 0xD347,
             "n_bytes": 2, "byte_order": "little", "reward_per_unit": 0.1},
        ]))
        rs.step(np.zeros((84, 84, 4)))  # init

        # Change to 15 → delta = 5
        env._ram[0xD347] = 15
        bonus = rs.step(np.zeros((84, 84, 4)))
        assert bonus == pytest.approx(0.5)  # 5 * 0.1

        # Change to 15 again → delta = 0
        bonus = rs.step(np.zeros((84, 84, 4)))
        assert bonus == pytest.approx(0.0)

    def test_scanner_out_of_bounds(self):
        env = MockRAMEnv(np.zeros(256, dtype=np.uint8))  # tiny RAM
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xFFFF,
             "target_value": 0, "reward": 1.0},
        ]))
        bonus = rs.step(np.zeros((84, 84, 4)))
        assert bonus == 0.0

    def test_reset_clears_state(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 42
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xD163,
             "target_value": 42, "reward": 1.0},
        ]))
        rs.step(np.zeros((84, 84, 4)))
        rs.reset()
        # After reset, should reward again (state cleared)
        bonus = rs.step(np.zeros((84, 84, 4)))
        assert bonus == 1.0


class TestStaleProperty:
    def test_is_stale(self):
        rs = RewardSystem(MockRAMEnv(), make_reward_config(
            stale_penalty=-0.1, stale_threshold=10, stale_diff_threshold=0.5,
        ))
        assert not rs.is_stale
        for _ in range(12):
            rs.step(np.zeros((84, 84, 4), dtype=np.float32))
        assert rs.is_stale

    def test_scanner_states_property(self):
        env = MockRAMEnv()
        env._ram[0xD163] = 42
        rs = RewardSystem(env, make_reward_config(ram_scanners=[
            {"name": "test", "type": "discrete", "address": 0xD163,
             "target_value": 42, "reward": 1.0},
        ]))
        rs.step(np.zeros((84, 84, 4)))
        states = rs.scanner_states
        assert "test" in states
