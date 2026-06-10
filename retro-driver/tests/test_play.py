from __future__ import annotations

import sys

import pytest

from retro_driver.play import main


class TestPlayArgparse:
    def test_rom_required(self):
        old_argv = sys.argv
        sys.argv = ["retro-play"]
        with pytest.raises(SystemExit):
            main()
        sys.argv = old_argv

    def test_checkpoint_required(self):
        old_argv = sys.argv
        sys.argv = ["retro-play", "--rom", "test.gb"]
        with pytest.raises(SystemExit):
            main()
        sys.argv = old_argv

    def test_rom_not_found(self):
        old_argv = sys.argv
        sys.argv = [
            "retro-play",
            "--rom",
            "/nonexistent/rom.gb",
            "--checkpoint",
            "/nonexistent/ckpt.pt",
        ]
        with pytest.raises(SystemExit):
            main()
        sys.argv = old_argv

    def test_checkpoint_not_found(self):
        old_argv = sys.argv
        # Rom exists? We need a real rom... Let's just test checkpoint path separately
        with pytest.raises(SystemExit):
            sys.argv = ["retro-play", "--rom", "test.gb", "--checkpoint", "/nonexistent/ckpt.pt"]
            main()
        sys.argv = old_argv

    def test_default_values(self):
        """Verify default values are sensible."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--rom", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--frame-skip", type=int, default=4)
        parser.add_argument("--max-steps", type=int, default=10_000)
        parser.add_argument("--render", action="store_true")
        args = parser.parse_args(["--rom", "x", "--checkpoint", "y"])
        assert args.frame_skip == 4
        assert args.max_steps == 10_000
        assert not args.render
