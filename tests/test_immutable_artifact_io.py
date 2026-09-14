from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "immutable_artifact_io.py"
SPEC = importlib.util.spec_from_file_location("immutable_artifact_io_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_same_fd_read_and_expected_identity(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"frozen")
    digest = MODULE.sha256_bytes(b"frozen")
    record = MODULE.read_bytes_nofollow(
        path,
        expected_sha256=digest,
        expected_bytes=6,
        allowed_root=tmp_path,
    )
    assert record.payload == b"frozen"
    assert record.sha256 == digest


def test_intermediate_and_final_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "payload"
    payload.write_bytes(b"x")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        MODULE.read_bytes_nofollow(tmp_path / "alias" / "payload")
    (real / "link").symlink_to(payload)
    with pytest.raises(OSError):
        MODULE.read_bytes_nofollow(real / "link")


def test_sha_and_size_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"abc")
    with pytest.raises(MODULE.StrictIOError):
        MODULE.read_bytes_nofollow(path, expected_sha256="0" * 64)
    with pytest.raises(MODULE.StrictIOError):
        MODULE.read_bytes_nofollow(path, expected_bytes=4)


def test_write_new_is_no_overwrite_and_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    MODULE.write_new_json(path, {"status": "PASS"}, allowed_root=tmp_path)
    value, record = MODULE.read_json_nofollow(path, allowed_root=tmp_path)
    assert value == {"status": "PASS"}
    assert record.bytes == path.stat().st_size
    with pytest.raises(FileExistsError):
        MODULE.write_new_json(path, {"status": "OVERWRITE"}, allowed_root=tmp_path)


def test_direct_child_mkdir_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    child = MODULE.ensure_direct_child_dir(tmp_path, "run_v1")
    assert child.is_dir()
    with pytest.raises(MODULE.StrictIOError):
        MODULE.ensure_direct_child_dir(tmp_path, "../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        MODULE.ensure_direct_child_dir(tmp_path, "alias", exist_ok=True)


def test_append_and_atomic_mutable_state(tmp_path: Path) -> None:
    events = tmp_path / "EVENTS.jsonl"
    MODULE.append_jsonl(events, {"event": 1}, allowed_root=tmp_path)
    MODULE.append_jsonl(events, {"event": 2}, allowed_root=tmp_path)
    assert [json.loads(line)["event"] for line in events.read_text().splitlines()] == [1, 2]
    heartbeat = tmp_path / "HEARTBEAT.json"
    MODULE.atomic_replace_json(heartbeat, {"tick": 1}, allowed_root=tmp_path)
    MODULE.atomic_replace_json(heartbeat, {"tick": 2}, allowed_root=tmp_path)
    assert json.loads(heartbeat.read_text()) == {"tick": 2}


def test_exclusive_command_output_binds_inherited_fd(tmp_path: Path) -> None:
    output = tmp_path / "video.bin"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'video')",
        "{output_fd}",
    ]
    _, record = MODULE.run_command_exclusive_output(
        command,
        output,
        allowed_root=tmp_path,
    )
    assert output.read_bytes() == b"video"
    assert record["sha256"] == MODULE.sha256_bytes(b"video")
    with pytest.raises(FileExistsError):
        MODULE.run_command_exclusive_output(command, output, allowed_root=tmp_path)


def test_exclusive_command_does_not_follow_preexisting_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"sentinel")
    output = tmp_path / "video.bin"
    output.symlink_to(outside)
    command = [sys.executable, "-c", "open(__import__('sys').argv[1],'wb').write(b'x')", "{output_fd}"]
    with pytest.raises(FileExistsError):
        MODULE.run_command_exclusive_output(command, output, allowed_root=tmp_path)
    assert outside.read_bytes() == b"sentinel"


def test_streaming_exclusive_output_waits_and_publishes(tmp_path: Path) -> None:
    output = tmp_path / "stream.bin"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
        "{output_fd}",
    ]
    with MODULE.start_command_exclusive_output(command, output, allowed_root=tmp_path) as child:
        assert child.stdin is not None
        child.stdin.write(b"frame-1\nframe-2\n")
        result = child.wait_and_publish(timeout=10)
    assert result["child_exit_observed"] is True
    assert result["returncode"] == 0
    assert output.read_bytes() == b"frame-1\nframe-2\n"


def test_streaming_exclusive_output_aborts_without_publish(tmp_path: Path) -> None:
    output = tmp_path / "stream.bin"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'partial'); sys.exit(3)",
        "{output_fd}",
    ]
    with pytest.raises(MODULE.StrictIOError):
        with MODULE.start_command_exclusive_output(command, output, allowed_root=tmp_path) as child:
            child.wait_and_publish(timeout=10)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial"))
