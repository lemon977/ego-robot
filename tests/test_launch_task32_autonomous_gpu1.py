from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "launch_task32_autonomous_gpu1.py"
SPEC = importlib.util.spec_from_file_location("task32_autonomous_launcher_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_runner_pin_is_live() -> None:
    record = MODULE.strict_io.read_bytes_nofollow(
        MODULE.RUNNER,
        expected_bytes=MODULE.RUNNER_BYTES,
        expected_sha256=MODULE.RUNNER_SHA256,
        allowed_root=MODULE.PROJECT,
    )
    assert record.bytes == MODULE.RUNNER_BYTES


@pytest.mark.parametrize("value", ["", "f" * 63, "g" * 64, "F" * 64])
def test_sha_parser_rejects_noncanonical(value: str) -> None:
    with pytest.raises(Exception):
        MODULE._sha(value)


def test_launch_rejects_existing_run_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "existing"
    run_root.mkdir()
    monkeypatch.setattr(MODULE, "RUN_ROOT", run_root)
    monkeypatch.setattr(MODULE, "_load_qa", lambda **_kwargs: {})
    monkeypatch.setattr(MODULE.strict_io, "read_bytes_nofollow", lambda *_a, **_k: object())
    with pytest.raises(MODULE.LaunchError, match="already exists"):
        MODULE.launch(qa_bytes=1, qa_sha256="0" * 64)


def test_release_requires_empty_process_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = {
        "status": "GPU_RELEASED_AFTER_SUCCESS",
        "child_exit_observed": True,
        "child_exit_code": 0,
        "gpu_processes_remaining": [123],
    }
    monkeypatch.setattr(MODULE, "RUN_ROOT", tmp_path / "new")
    monkeypatch.setattr(MODULE, "_load_qa", lambda **_kwargs: {})
    monkeypatch.setattr(MODULE.strict_io, "read_bytes_nofollow", lambda *_a, **_k: object())
    monkeypatch.setattr(MODULE, "_environment", lambda: {})
    monkeypatch.setattr(MODULE, "_probe_import_identity", lambda _env: "/official/model_builder.py")
    monkeypatch.setattr(MODULE, "_run_runner", lambda *_a, **_k: None)
    monkeypatch.setattr(MODULE.strict_io, "read_json_nofollow", lambda *_a, **_k: (release, object()))
    with pytest.raises(MODULE.LaunchError, match="zero-process"):
        MODULE.launch(qa_bytes=1, qa_sha256="0" * 64)
