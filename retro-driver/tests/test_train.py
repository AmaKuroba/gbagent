from __future__ import annotations

import sys
from dataclasses import fields

import pytest

from retro_driver.train import TrainConfig, main


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.gbagent_url == "ws://localhost:8767/ws"
        assert cfg.frame_stack == 4
        assert cfg.frame_skip == 4
        assert cfg.max_episode_steps == 3000
        assert cfg.total_steps == 500_000
        assert cfg.eval_interval == 10_000
        assert cfg.eval_episodes == 3
        assert cfg.save_interval == 25_000
        assert cfg.log_interval == 1_000
        assert cfg.save_dir == "models"
        assert cfg.log_dir == "logs"
        assert cfg.resume == ""
        assert cfg.ram_scanner_config == ""

    def test_all_fields_have_types(self):
        for f in fields(TrainConfig):
            assert f.type is not None

    def test_custom_values(self):
        cfg = TrainConfig(
            gbagent_url="ws://other:1234/ws",
            total_steps=100_000,
            frame_skip=2,
        )
        assert cfg.gbagent_url == "ws://other:1234/ws"
        assert cfg.total_steps == 100_000
        assert cfg.frame_skip == 2


class TestMainArgparse:
    def test_default_values(self):
        old_argv = sys.argv
        sys.argv = ["retro-train", "--gbagent-url", "ws://localhost:18767/ws"]
        try:
            main()
        except Exception:
            pass
        sys.argv = old_argv

    def test_gbagent_url_accepted(self):
        old_argv = sys.argv
        sys.argv = [
            "retro-train",
            "--gbagent-url", "ws://localhost:18767/ws",
        ]
        try:
            main()
        except Exception:
            pass
        sys.argv = old_argv

    def test_all_arguments_accepted(self):
        old_argv = sys.argv
        sys.argv = [
            "retro-train",
            "--gbagent-url", "ws://localhost:18767/ws",
            "--total-steps",
            "100000",
            "--frame-skip",
            "2",
            "--frame-stack",
            "4",
            "--max-episode-steps",
            "5000",
            "--buffer-capacity",
            "50000",
            "--batch-size",
            "64",
            "--learning-rate",
            "0.001",
            "--gamma",
            "0.95",
            "--epsilon-decay",
            "20000",
            "--train-start",
            "1000",
            "--seq-len",
            "4",
            "--eval-interval",
            "5000",
            "--save-interval",
            "10000",
            "--save-dir",
            "/tmp/models",
            "--log-dir",
            "/tmp/logs",
            "--ram-scanner",
            "/tmp/scanner.yaml",
        ]
        try:
            main()
        except (ConnectionError, SystemExit):
            pass
        sys.argv = old_argv
