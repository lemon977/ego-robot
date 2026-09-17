from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_help_never_runs_protection_or_deletion(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/chaoyang/ops/cleanup_current_only_v71.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    assert result.returncode == 0
    assert "--commit-delete" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_cleanup_commit_requires_prior_manifest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/chaoyang/ops/cleanup_current_only_v71.py"),
            "--output-root",
            str(tmp_path / "out"),
            "--commit-delete",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    assert result.returncode != 0
    assert "requires --pre-delete-manifest" in result.stderr
    assert not (tmp_path / "out").exists()
