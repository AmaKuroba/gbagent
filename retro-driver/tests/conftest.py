from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def frame_144x160() -> np.ndarray:
    return np.zeros((144, 160), dtype=np.uint8)


@pytest.fixture
def random_frame() -> np.ndarray:
    return np.random.randint(0, 256, (144, 160), dtype=np.uint8)


@pytest.fixture
def test_image_bytes() -> bytes:
    img = Image.new("L", (160, 144), 128)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_yaml_config(tmp_path: Path) -> Path:
    path = tmp_path / "scanner.yaml"
    path.write_text("""\
game: "pokemon-red"
scanners:
  - name: "map_change"
    ram_addr: 54238
    type: "discrete"
    reward_on_change: 1.0
  - name: "party_levels"
    ram_addr: 53611
    type: "delta"
    length: 4
    reward_per_unit: 5.0
""")
    return path
