from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_visual_aux_checkpoint_index_help_is_read_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/chaoyang/ops/build_visual_aux_checkpoint_index_v71.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--source" in result.stdout
    assert "--output-root" in result.stdout
    assert list(tmp_path.iterdir()) == []
