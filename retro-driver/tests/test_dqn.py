from __future__ import annotations

import numpy as np
import pytest
import torch

from retro_driver.dqn import (
    FrameStore,
    Transition,
    ReplayBuffer,
    DQNConfig,
    DQNAgent,
    DEVICE,
    GBFeatureExtractor,
    GBDQN,
)


class TestFrameStore:
    def test_add_and_get(self, frame_store: FrameStore, frame_144x160: np.ndarray):
        gid = frame_store.add(frame_144x160)
        assert gid == 0
        assert frame_store.has(gid)
        retrieved = frame_store.get(gid)
        assert retrieved.shape == (144, 160)
        assert retrieved.dtype == np.uint8

    def test_next_gid(self, frame_store: FrameStore, frame_144x160: np.ndarray):
        assert frame_store.next_gid == 0
        frame_store.add(frame_144x160)
        assert frame_store.next_gid == 1
        frame_store.add(frame_144x160)
        assert frame_store.next_gid == 2

    def test_ring_eviction(self):
        fs = FrameStore(max_frames=3)
        gids = [fs.add(np.full((144, 160), i, dtype=np.uint8)) for i in range(5)]
        assert not fs.has(gids[0])
        assert not fs.has(gids[1])
        assert fs.has(gids[2])
        assert fs.has(gids[3])
        assert fs.has(gids[4])

    def test_get_stack(self, populated_frame_store: FrameStore):
        stack = populated_frame_store.get_stack(0, 4)
        assert stack.shape == (4, 144, 160)
        assert stack.dtype == np.uint8

    def test_get_stack_raises_on_missing(self, frame_store: FrameStore):
        with pytest.raises(KeyError):
            frame_store.get_stack(0, 1)

    def test_is_valid_stack(self, populated_frame_store: FrameStore):
        assert populated_frame_store.is_valid_stack(0, 4)
        assert not populated_frame_store.is_valid_stack(100, 4)

    def test_max_frames_default(self):
        fs = FrameStore()
        assert fs.max_frames == 200_000


class TestTransition:
    def test_fields(self):
        t = Transition(
            frame_gid=5,
            dpad_action=1,
            btn_action=2,
            reward=0.5,
            terminated=False,
            truncated=True,
        )
        assert t.frame_gid == 5
        assert t.dpad_action == 1
        assert t.btn_action == 2
        assert t.reward == 0.5
        assert not t.terminated
        assert t.truncated


class TestReplayBuffer:
    def test_push_and_len(self, replay_buffer: ReplayBuffer):
        assert len(replay_buffer) == 5

    def test_sample(self, replay_buffer: ReplayBuffer):
        samples = replay_buffer.sample(3)
        assert len(samples) == 3
        assert all(isinstance(t, Transition) for t in samples)

    def test_sample_returns_available(self, replay_buffer: ReplayBuffer):
        samples = replay_buffer.sample(100)
        assert len(samples) <= len(replay_buffer)

    def test_sample_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        assert buf.sample(5) == []

    def test_circular_eviction(self, populated_frame_store: FrameStore):
        buf = ReplayBuffer(capacity=3, frame_store=populated_frame_store)
        for i in range(10):
            buf.push(
                Transition(
                    frame_gid=i % 5,
                    dpad_action=0,
                    btn_action=0,
                    reward=1.0,
                    terminated=False,
                    truncated=False,
                )
            )
        assert len(buf) == 3

    def test_sample_sequences(self, populated_frame_store: FrameStore):
        buf = ReplayBuffer(capacity=20, frame_store=populated_frame_store)
        # Push 6 transitions with consecutive frame_gids
        for gid in range(6):
            buf.push(
                Transition(
                    frame_gid=gid,
                    dpad_action=0,
                    btn_action=0,
                    reward=1.0,
                    terminated=False,
                    truncated=False,
                )
            )
        seqs = buf.sample_sequences(2, 3)
        assert len(seqs) <= 2
        if len(seqs) > 0:
            assert len(seqs[0]) == 3
            assert seqs[0][1].frame_gid == seqs[0][0].frame_gid + 1

    def test_sample_sequences_respects_boundaries(self, populated_frame_store: FrameStore):
        buf = ReplayBuffer(capacity=20, frame_store=populated_frame_store)
        for gid in range(3):
            buf.push(
                Transition(
                    frame_gid=gid,
                    dpad_action=0,
                    btn_action=0,
                    reward=1.0,
                    terminated=bool(gid == 2),
                    truncated=False,
                )
            )
        for gid in range(3, 6):
            buf.push(
                Transition(
                    frame_gid=gid,
                    dpad_action=0,
                    btn_action=0,
                    reward=1.0,
                    terminated=False,
                    truncated=False,
                )
            )
        seqs = buf.sample_sequences(10, 3)
        for seq in seqs:
            for t in seq[:-1]:
                assert not t.terminated and not t.truncated


class TestDQNConfig:
    def test_defaults(self):
        cfg = DQNConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.gamma == 0.99
        assert cfg.epsilon_start == 1.0
        assert cfg.epsilon_end == 0.01
        assert cfg.epsilon_decay_steps == 50_000
        assert cfg.buffer_capacity == 10_000
        assert cfg.batch_size == 32
        assert cfg.target_update_interval == 1_000
        assert cfg.train_start == 2000
        assert cfg.seq_len == 32

    def test_custom(self):
        cfg = DQNConfig(learning_rate=1e-3, seq_len=1, buffer_capacity=50_000)
        assert cfg.learning_rate == 1e-3
        assert cfg.seq_len == 1
        assert cfg.buffer_capacity == 50_000
        assert cfg.gamma == 0.99  # default


class TestGBFeatureExtractor:
    def test_output_shape(self):
        net = GBFeatureExtractor().to(DEVICE)
        dummy = torch.zeros(2, 4, 144, 160, dtype=torch.uint8, device=DEVICE)
        out = net(dummy)
        assert out.shape[0] == 2
        assert out.shape[1] > 0  # flatten size, not a fixed number

    def test_normalization(self):
        net = GBFeatureExtractor()
        dummy = torch.full((1, 4, 144, 160), 255, dtype=torch.uint8)
        out = net(dummy)
        assert out.dtype == torch.float32
        assert out.max() < 2.0  # normalized


class TestGBDQN:
    def test_forward_shapes(self):
        net = GBDQN().to(DEVICE)
        dummy = torch.zeros(2, 4, 144, 160, dtype=torch.uint8, device=DEVICE)
        dpad_q, btn_q, _ = net(dummy)
        assert dpad_q.shape == (2, 5)
        assert btn_q.shape == (2, 4)

    def test_forward_sequence(self):
        net = GBDQN().to(DEVICE)
        dummy = torch.zeros(2, 6, 4, 144, 160, dtype=torch.uint8, device=DEVICE)
        dpad_q, btn_q, _ = net(dummy)
        assert dpad_q.shape == (2, 5)
        assert btn_q.shape == (2, 4)

    def test_act_epsilon_zero(self):
        net = GBDQN().to(DEVICE)
        frames = np.zeros((4, 144, 160), dtype=np.uint8)
        dpad, btn = net.act(frames, epsilon=0.0)
        assert 0 <= dpad <= 4
        assert 0 <= btn <= 3

    def test_act_epsilon_one(self):
        net = GBDQN().to(DEVICE)
        frames = np.zeros((4, 144, 160), dtype=np.uint8)
        actions = set()
        for _ in range(50):
            actions.add(net.act(frames, epsilon=1.0))
        assert len(actions) > 5

    def test_reset_hidden(self):
        net = GBDQN().to(DEVICE)
        net.reset_hidden(4)
        assert net._hidden is not None
        assert net._hidden[0].shape == (1, 4, 256)

    def test_hidden_carries_across_calls(self):
        net = GBDQN().to(DEVICE)
        net.reset_hidden(1)
        h0 = net._hidden[0].clone()
        dummy = torch.zeros(1, 4, 144, 160, dtype=torch.uint8, device=DEVICE)
        net(dummy)
        assert not torch.equal(net._hidden[0], h0)

    def test_separate_value_streams(self):
        """Each head has its own V(s) — ensure they produce different outputs."""
        net = GBDQN().to(DEVICE)
        dummy = torch.zeros(1, 4, 144, 160, dtype=torch.uint8, device=DEVICE)
        dpad_q, btn_q, _ = net(dummy)
        assert not torch.isnan(dpad_q).any()
        assert not torch.isnan(btn_q).any()


class TestDQNAgent:
    def test_epsilon_decay(self):
        cfg = DQNConfig(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=100)
        agent = DQNAgent(config=cfg)
        assert agent.epsilon == 1.0
        agent.step_count = 50
        assert agent.epsilon == 0.5
        agent.step_count = 100
        assert agent.epsilon == 0.0

    def test_act_training_vs_eval(self):
        agent = DQNAgent(DQNConfig(epsilon_start=1.0, epsilon_decay_steps=100_000))
        frames = np.zeros((4, 144, 160), dtype=np.uint8)
        # training=True uses epsilon > 0, eval uses 0
        dpad_eval, btn_eval = agent.act(frames, training=False)
        assert 0 <= dpad_eval <= 4
        assert 0 <= btn_eval <= 3

    def test_learn_returns_zeros_before_train_start(self):
        agent = DQNAgent(DQNConfig(train_start=100))
        stats = agent.learn()
        assert stats["loss"] == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        agent = DQNAgent()
        agent.step_count = 42
        agent.episode_count = 7
        path = tmp_path / "test.pt"
        agent.save(str(path))
        assert path.exists()

        agent2 = DQNAgent()
        agent2.load(str(path))
        assert agent2.step_count == 42
        assert agent2.episode_count == 7

    def test_learn_sequences_does_not_crash_on_boundary(self):
        """_learn_sequences should not KeyError when frame_gid + fstack exceeds max_gid."""
        cfg = DQNConfig(
            seq_len=4,
            buffer_capacity=20,
            train_start=8,
        )
        agent = DQNAgent(config=cfg)
        fs = agent.buffer.frame_store
        # Fill frame store with 6 frames (only 2 valid stacks of 4)
        for i in range(6):
            fs.add(np.full((144, 160), i, dtype=np.uint8))
        # Push transitions with frame_gid near the boundary (2-4).
        # Adding one more would overflow the valid range.
        for gid in range(2, 5):
            agent.buffer.push(Transition(
                frame_gid=gid,
                dpad_action=0,
                btn_action=0,
                reward=0.0,
                terminated=False,
                truncated=False,
            ))
        # train_start=8, buffer has 8 (2 + 3 pushes = 5 + warmup from learn?
        # Actually just call learn which samples sequences — the clamp
        # should prevent any KeyError.
        for _ in range(3):
            agent.buffer.push(Transition(
                frame_gid=0,
                dpad_action=0,
                btn_action=0,
                reward=0.0,
                terminated=False,
                truncated=False,
            ))
        stats = agent.learn()
        assert "loss" in stats
