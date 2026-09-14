from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import robot_renderer_eevee_fullchain as s7  # noqa: E402


def test_verified_shared_io_path_can_be_replaced_before_import(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reproduce hash-A/consume-B in the S7 shared-I/O bootstrap."""

    tools = tmp_path / "tools"
    tools.mkdir()
    source = Path("tools/immutable_artifact_io.py").resolve()
    target = tools / "immutable_artifact_io.py"
    shutil.copyfile(source, target)
    replacement = tools / "replacement.py"
    replacement.write_text("SENTINEL = 'replacement-consumed'\n", encoding="utf-8")

    original_hash = s7._bootstrap_sha256

    def hash_then_replace(path: Path) -> str:
        digest = original_hash(path)
        os.replace(replacement, path)
        return digest

    monkeypatch.setattr(s7, "_bootstrap_sha256", hash_then_replace)
    sys.modules.pop("s7_immutable_artifact_io", None)

    loaded = s7.load_strict_io(tmp_path)
    assert loaded.SENTINEL == "replacement-consumed"
