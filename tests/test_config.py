from __future__ import annotations

from pathlib import Path

import yaml

from gbagent.config import (
    AgentConfig,
    Config,
    EnvConfig,
    RecorderConfig,
    RewardConfig,
    TrainConfig,
    format_config,
    load_config,
    save_config,
)


class TestConfigDefaults:
    def test_env_config_defaults(self):
        cfg = EnvConfig()
        assert cfg.game == "PokemonRed-GB"
        assert cfg.state == "Level1"
        assert cfg.frame_skip == 4
        assert cfg.frame_stack == 4
        assert not cfg.gba_mode

    def test_agent_config_defaults(self):
        cfg = AgentConfig()
        assert cfg.embed_dim == 256
        assert cfg.num_layers == 6
        assert cfg.num_heads == 8
        assert cfg.learning_rate == 3e-4
        assert cfg.hidden_dim == 512

    def test_train_config_defaults(self):
        cfg = TrainConfig()
        assert cfg.total_timesteps == 10_000_000
        assert cfg.num_envs == 8
        assert cfg.n_steps == 128
        assert cfg.seed == 42

    def test_reward_config_defaults(self):
        cfg = RewardConfig()
        assert cfg.screen_novelty_scale == 0.0
        assert cfg.stale_penalty == 0.0
        assert cfg.ram_scanners == []

    def test_recorder_config_defaults(self):
        cfg = RecorderConfig()
        assert cfg.output_dir == "recordings"
        assert not cfg.enabled

    def test_top_level_config(self):
        cfg = Config()
        assert isinstance(cfg.env, EnvConfig)
        assert isinstance(cfg.agent, AgentConfig)
        assert isinstance(cfg.train, TrainConfig)
        assert isinstance(cfg.reward, RewardConfig)
        assert isinstance(cfg.recorder, RecorderConfig)


class TestLoadConfig:
    def test_load_nonexistent_file(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.env.game == "PokemonRed-GB"

    def test_load_with_overrides(self, tmp_path: Path):
        config_file = tmp_path / "test_config.yaml"
        data = {
            "env": {"game": "Tetris-GB", "frame_skip": 2},
            "agent": {"learning_rate": 0.001, "hidden_dim": 256},
            "train": {"num_envs": 4, "n_steps": 64},
        }
        with open(config_file, "w") as f:
            yaml.dump(data, f)

        cfg = load_config(config_file)
        assert cfg.env.game == "Tetris-GB"
        assert cfg.env.frame_skip == 2
        assert cfg.agent.learning_rate == 0.001
        assert cfg.agent.hidden_dim == 256
        assert cfg.train.num_envs == 4
        assert cfg.train.n_steps == 64
        # Defaults preserved
        assert cfg.env.frame_stack == 4

    def test_load_empty_section(self, tmp_path: Path):
        config_file = tmp_path / "empty_section.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"env": None, "train": {}}, f)

        # Should not crash on None section
        cfg = load_config(config_file)
        assert cfg.env.game == "PokemonRed-GB"

    def test_load_ram_scanners(self, tmp_path: Path):
        config_file = tmp_path / "scanners.yaml"
        scanners = [
            {"name": "hp", "type": "delta", "address": 0xD163, "reward": 0.1},
            {"name": "money", "type": "multi_byte", "address": 0xD347,
             "n_bytes": 2, "byte_order": "little", "target_value": 0, "reward": -0.5},
        ]
        data = {"reward": {"ram_scanners": scanners}}
        with open(config_file, "w") as f:
            yaml.dump(data, f)

        cfg = load_config(config_file)
        assert len(cfg.reward.ram_scanners) == 2
        assert cfg.reward.ram_scanners[0]["name"] == "hp"

    def test_to_dict(self):
        cfg = RewardConfig(screen_novelty_scale=0.5, ram_scanners=[{"name": "test"}])
        d = cfg.to_dict()
        assert d["screen_novelty_scale"] == 0.5
        assert d["ram_scanners"] == [{"name": "test"}]


class TestSaveConfig:
    def test_round_trip(self, tmp_path: Path):
        cfg = Config()
        cfg.env.game = "TestGame"
        cfg.agent.learning_rate = 0.0001

        save_path = tmp_path / "saved.yaml"
        save_config(cfg, save_path)
        assert save_path.is_file()

        loaded = load_config(save_path)
        assert loaded.env.game == "TestGame"
        assert loaded.agent.learning_rate == 0.0001

    def test_format_config(self):
        cfg = Config()
        text = format_config(cfg)
        assert "GBAGent Configuration" in text
        assert "env:" in text
        assert "PokemonRed-GB" in text
