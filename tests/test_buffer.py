from __future__ import annotations

import numpy as np

from gbagent.buffer import RolloutBuffer


def test_buffer_init():
    buf = RolloutBuffer(num_envs=2, n_steps=10, obs_shape=(84, 84, 4))
    assert buf.num_envs == 2
    assert buf.n_steps == 10
    assert buf.obs_shape == (84, 84, 4)
    assert buf._step == 0


def test_buffer_store():
    buf = RolloutBuffer(num_envs=2, n_steps=10, obs_shape=(84, 84, 4))
    obs = np.random.randn(2, 84, 84, 4).astype(np.float32)
    dpad = np.array([0, 1], dtype=np.int32)
    btn = np.array([2, 0], dtype=np.int32)
    reward = np.array([0.5, -0.1], dtype=np.float32)
    done = np.array([False, False], dtype=bool)
    log_prob = np.array([-0.5, -0.3], dtype=np.float32)
    value = np.array([0.1, 0.2], dtype=np.float32)

    buf.store(obs, dpad, btn, reward, done, log_prob, value)
    assert buf._step == 1
    assert np.allclose(buf.obs[0], obs)
    assert np.allclose(buf.rewards[0], reward)


def test_buffer_clear():
    buf = RolloutBuffer(num_envs=2, n_steps=10, obs_shape=(84, 84, 4))
    obs = np.random.randn(2, 84, 84, 4).astype(np.float32)
    buf.store(obs, np.array([0, 1]), np.array([0, 1]),
              np.array([0.5, 0.5]), np.array([False, False]),
              np.array([-0.5, -0.5]), np.array([0.1, 0.1]))
    assert buf._step == 1
    buf.clear()
    assert buf._step == 0
    assert np.allclose(buf.obs[0], 0.0)


class TestGAE:
    def test_gae_monotonic_reward(self):
        buf = RolloutBuffer(num_envs=1, n_steps=5, obs_shape=(84, 84, 4))
        for i in range(5):
            obs = np.random.randn(1, 84, 84, 4).astype(np.float32)
            buf.store(obs,
                      np.array([0]), np.array([0]),
                      np.array([1.0], dtype=np.float32),
                      np.array([False]),
                      np.array([-0.5], dtype=np.float32),
                      np.array([0.5], dtype=np.float32))

        last_value = np.array([0.0], dtype=np.float32)
        buf.compute_gae(last_value, gamma=0.99, gae_lambda=0.95)

        assert buf.advantages.shape == (5, 1)
        assert buf.returns.shape == (5, 1)
        # Advantages should be decreasing (earlier steps have higher uncertainty)
        assert buf.advantages[0, 0] > buf.advantages[-1, 0]

    def test_gae_with_terminal(self):
        buf = RolloutBuffer(num_envs=1, n_steps=4, obs_shape=(84, 84, 4))
        obs = np.random.randn(1, 84, 84, 4).astype(np.float32)
        for i in range(4):
            done = i == 2  # environment terminates at step 2
            buf.store(obs,
                      np.array([0]), np.array([0]),
                      np.array([1.0], dtype=np.float32),
                      np.array([done]),
                      np.array([-0.5], dtype=np.float32),
                      np.array([0.5], dtype=np.float32))

        last_value = np.array([0.0], dtype=np.float32)
        buf.compute_gae(last_value, gamma=0.99, gae_lambda=0.95)

        # Terminal at step 2 cuts off bootstrap to step 3, so step 2's
        # advantage is just its own TD error (no future bootstrapping).
        # Steps before the terminal (0, 1) include bootstrapped future reward.
        assert buf.advantages[0, 0] > buf.advantages[1, 0] > buf.advantages[2, 0]

    def test_gae_multiple_envs(self):
        buf = RolloutBuffer(num_envs=3, n_steps=10, obs_shape=(84, 84, 4))
        for i in range(10):
            obs = np.random.randn(3, 84, 84, 4).astype(np.float32)
            buf.store(obs,
                      np.array([0, 1, 2]), np.array([0, 0, 1]),
                      np.ones(3, dtype=np.float32),
                      np.zeros(3, dtype=bool),
                      np.ones(3, dtype=np.float32) * -0.5,
                      np.ones(3, dtype=np.float32) * 0.5)

        last_value = np.zeros(3, dtype=np.float32)
        buf.compute_gae(last_value, gamma=0.99, gae_lambda=0.95)

        assert buf.advantages.shape == (10, 3)
        assert buf.returns.shape == (10, 3)
        # All envs identical → advantages should be same
        assert np.allclose(buf.advantages[:, 0], buf.advantages[:, 1])

    def test_gae_zero_returns_for_zero_rewards(self):
        buf = RolloutBuffer(num_envs=1, n_steps=5, obs_shape=(84, 84, 4))
        for i in range(5):
            obs = np.random.randn(1, 84, 84, 4).astype(np.float32)
            buf.store(obs,
                      np.array([0]), np.array([0]),
                      np.array([0.0], dtype=np.float32),
                      np.array([False]),
                      np.array([0.0], dtype=np.float32),
                      np.array([0.0], dtype=np.float32))

        buf.compute_gae(np.array([0.0]), gamma=0.99, gae_lambda=0.95)
        assert np.allclose(buf.advantages, 0.0, atol=1e-7)
        assert np.allclose(buf.returns, 0.0, atol=1e-7)


class TestGetBatches:
    def test_batch_size(self):
        buf = RolloutBuffer(num_envs=2, n_steps=10, obs_shape=(4, 4, 1))
        for i in range(10):
            obs = np.random.randn(2, 4, 4, 1).astype(np.float32)
            buf.store(obs,
                      np.array([0, 1]), np.array([0, 0]),
                      np.ones(2, dtype=np.float32),
                      np.zeros(2, dtype=bool),
                      np.ones(2, dtype=np.float32) * -0.5,
                      np.ones(2, dtype=np.float32) * 0.5)

        buf.compute_gae(np.zeros(2), gamma=0.99, gae_lambda=0.95)
        batches = list(buf.get_batches(batch_size=4))
        assert len(batches) == 5  # 20 samples / 4 = 5 batches
        for obs_b, dpad_b, btn_b, ret_b, adv_b, lp_b in batches:
            assert obs_b.shape[0] == 4
            assert dpad_b.shape[0] == 4
            assert btn_b.shape[0] == 4
            assert ret_b.shape[0] == 4
            assert adv_b.shape[0] == 4
            assert lp_b.shape[0] == 4

    def test_advantage_normalization(self):
        buf = RolloutBuffer(num_envs=1, n_steps=20, obs_shape=(4, 4, 1))
        for i in range(20):
            obs = np.random.randn(1, 4, 4, 1).astype(np.float32)
            reward = np.array([float(i) / 10.0], dtype=np.float32)
            buf.store(obs,
                      np.array([0]), np.array([0]),
                      reward,
                      np.array([False]),
                      np.array([-0.5], dtype=np.float32),
                      np.array([0.5], dtype=np.float32))

        buf.compute_gae(np.array([0.0]), gamma=0.99, gae_lambda=0.95)
        batches = list(buf.get_batches(batch_size=20))
        adv = batches[0][4]
        assert abs(adv.mean()) < 1e-6  # should be normalized to ~0 mean
        assert abs(adv.std() - 1.0) < 0.1  # should have ~1 std

    def test_returns_equals_adv_plus_values(self):
        buf = RolloutBuffer(num_envs=1, n_steps=5, obs_shape=(4, 4, 1))
        for i in range(5):
            obs = np.random.randn(1, 4, 4, 1).astype(np.float32)
            buf.store(obs,
                      np.array([0]), np.array([0]),
                      np.array([0.5], dtype=np.float32),
                      np.array([False]),
                      np.array([-0.5], dtype=np.float32),
                      np.array([0.3], dtype=np.float32))

        buf.compute_gae(np.array([0.2]), gamma=0.99, gae_lambda=0.95)
        assert np.allclose(buf.returns, buf.advantages + buf.values)
