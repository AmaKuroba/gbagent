from __future__ import annotations

from pathlib import Path

import pytest

from retro_driver.config import ScannerConfig, GameConfig, load_game_config


class TestScannerConfig:
    def test_defaults(self):
        sc = ScannerConfig(name="test", ram_addr=0xD000)
        assert sc.name == "test"
        assert sc.ram_addr == 0xD000
        assert sc.type == "discrete"
        assert sc.reward_on_change == 1.0
        assert sc.reward_per_unit == 0.1
        assert sc.length == 1
        assert not sc.novelty_decay

    def test_post_init_state(self):
        sc = ScannerConfig(name="test", ram_addr=0xD000)
        assert sc._last_val is None
        assert sc._last_sum is None

    def test_custom_values(self):
        sc = ScannerConfig(
            name="levels",
            ram_addr=0xD16B,
            type="delta",
            reward_on_change=0.5,
            reward_per_unit=5.0,
            length=4,
            novelty_decay=True,
        )
        assert sc.type == "delta"
        assert sc.reward_per_unit == 5.0
        assert sc.length == 4
        assert sc.novelty_decay


class TestGameConfig:
    def test_empty_scanners(self):
        gc = GameConfig(game="test", scanners=[])
        assert gc.game == "test"
        assert gc.scanners == []

    def test_with_scanners(self):
        sc = ScannerConfig(name="s1", ram_addr=0xD000)
        gc = GameConfig(game="test", scanners=[sc])
        assert len(gc.scanners) == 1


class TestLoadGameConfig:
    def test_load_valid_yaml(self, mock_yaml_config: Path):
        gc = load_game_config(mock_yaml_config)
        assert gc.game == "pokemon-red"
        assert len(gc.scanners) == 2
        assert gc.scanners[0].name == "map_change"
        assert gc.scanners[0].type == "discrete"
        assert gc.scanners[1].name == "party_levels"
        assert gc.scanners[1].type == "delta"
        assert gc.scanners[1].length == 4

    def test_load_empty_scanners(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("game: test\nscanners: []")
        gc = load_game_config(str(path))
        assert gc.game == "test"
        assert gc.scanners == []

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_game_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid")
        with pytest.raises(Exception):
            load_game_config(str(path))
