from __future__ import annotations

import numpy as np

from gbagent.action import (
    _DPAD_ONLY,
    _LOOKUP_GB,
    _LOOKUP_GBA,
    BTN_GB,
    BTN_GBA,
    _log_softmax,
    _softmax,
    combine_actions,
    get_btn_size,
    sample_actions,
)


def test_get_btn_size():
    assert get_btn_size(gba_mode=False) == BTN_GB
    assert get_btn_size(gba_mode=True) == BTN_GBA


def test_softmax():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    probs = _softmax(logits)
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(), 1.0)
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]


def test_log_softmax():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    lprobs = _log_softmax(logits)
    probs = _softmax(logits)
    assert np.allclose(lprobs, np.log(probs))


def test_softmax_numerical_stability():
    large = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float64)
    probs = _softmax(large)
    assert np.allclose(probs, 1.0 / 3.0)


class TestSampleActions:
    def test_sample_training(self):
        dpad_logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        btn_logits = np.array([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        dpad, btn, lp = sample_actions(dpad_logits, btn_logits, training=True)
        assert dpad.shape == (1,)
        assert btn.shape == (1,)
        assert lp.shape == (1,)
        # With high logits, argmax is nearly certain
        assert dpad[0] == 0
        assert btn[0] == 1

    def test_sample_deterministic(self):
        dpad_logits = np.array([[0.0, 10.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        btn_logits = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        dpad, btn, lp = sample_actions(dpad_logits, btn_logits, training=False)
        assert dpad[0] == 1  # UP
        assert btn[0] == 2  # A

    def test_sample_batch(self):
        dpad_logits = np.random.randn(4, 5).astype(np.float64)
        btn_logits = np.random.randn(4, 6).astype(np.float64)
        dpad, btn, lp = sample_actions(dpad_logits, btn_logits, training=True)
        assert dpad.shape == (4,)
        assert btn.shape == (4,)
        assert lp.shape == (4,)

    def test_keras_tensor_conversion(self):
        class FakeTensor:
            def numpy(self):
                return np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)

        dpad_logits = FakeTensor()
        btn_logits = np.random.randn(1, 6).astype(np.float64)
        dpad, btn, lp = sample_actions(dpad_logits, btn_logits, training=False)
        assert dpad.shape == (1,)
        assert btn.shape == (1,)


class TestCombineActions:
    def test_noop(self):
        result = combine_actions(
            np.array([0]), np.array([0]), btn_size=BTN_GB
        )
        assert result[0] == _LOOKUP_GB[(0, 0)]  # 0 = NOOP

    def test_dpad_only(self):
        for dpad_val, expected_action in _DPAD_ONLY.items():
            result = combine_actions(
                np.array([dpad_val]), np.array([0]), btn_size=BTN_GB
            )
            assert result[0] == expected_action

    def test_btn_only_gb(self):
        cases = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 9)]  # A+B
        for btn_val, expected in cases:
            result = combine_actions(
                np.array([0]), np.array([btn_val]), btn_size=BTN_GB
            )
            assert result[0] == expected

    def test_combined_dpad_priority(self):
        result = combine_actions(
            np.array([1]), np.array([2]), btn_size=BTN_GB
        )
        assert result[0] == _DPAD_ONLY[1]  # dpad wins

    def test_batch(self):
        dpad = np.array([0, 1, 2, 3, 4])
        btn = np.array([0, 0, 0, 0, 0])
        result = combine_actions(dpad, btn, btn_size=BTN_GB)
        assert len(result) == 5
        assert result[0] == _LOOKUP_GB[(0, 0)]
        assert result[1] == _LOOKUP_GB[(1, 0)]

    def test_gba_mode(self):
        result = combine_actions(
            np.array([0]), np.array([6]), btn_size=BTN_GBA
        )
        assert result[0] == _LOOKUP_GBA[(0, 6)]  # L

        result = combine_actions(
            np.array([0]), np.array([7]), btn_size=BTN_GBA
        )
        assert result[0] == _LOOKUP_GBA[(0, 7)]  # R


def test_compute_log_probs():
    from keras import ops

    from gbagent.action import compute_log_probs

    dpad_logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    btn_logits = np.array([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dpad_actions = np.array([0], dtype=np.int32)
    btn_actions = np.array([1], dtype=np.int32)

    lp = compute_log_probs(
        ops.convert_to_tensor(dpad_logits),
        ops.convert_to_tensor(btn_logits),
        ops.convert_to_tensor(dpad_actions),
        ops.convert_to_tensor(btn_actions),
    )
    assert lp.shape == (1,)
    assert float(lp[0]) < 0.0  # log prob should be negative


def test_policy_entropy():
    from keras import ops

    from gbagent.action import policy_entropy

    dpad_logits = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    btn_logits = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    ent = policy_entropy(dpad_logits, btn_logits)
    assert ent.shape == (1,)
    assert float(ent[0]) > 0.0  # entropy should be positive
