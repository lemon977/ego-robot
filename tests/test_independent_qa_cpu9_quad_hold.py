from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_quad_review_video_t1.py"
SPEC = importlib.util.spec_from_file_location("cpu9_quad_independent_hold", RUNNER)
assert SPEC and SPEC.loader
quad = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quad
SPEC.loader.exec_module(quad)


def _write_executable(path: Path, body: str) -> bytes:
    payload = ("#!/bin/sh\nset -eu\n" + body).encode()
    path.write_bytes(payload)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return payload


def test_hold_replay_ffprobe_hash_a_path_executes_replacement_b(
    tmp_path: Path,
) -> None:
    """A verified executable is not the executable later consumed by _run_probe."""

    tool = tmp_path / "ffprobe"
    payload_a = _write_executable(tool, "printf 'verified-A\\n'\n")
    strict = quad.load_strict_io(PROJECT)
    verified = strict.read_bytes_nofollow(
        tool,
        expected_bytes=len(payload_a),
        expected_sha256=hashlib.sha256(payload_a).hexdigest(),
    )
    assert verified.payload == payload_a

    replacement = tmp_path / "replacement"
    _write_executable(
        replacement,
        "printf '%s\\n' '{\"streams\":[{\"codec_type\":\"video\","
        "\"width\":1,\"height\":1,\"avg_frame_rate\":\"30/1\","
        "\"nb_read_frames\":\"1\",\"sentinel\":\"replacement-B\"}]}'\n",
    )
    os.replace(replacement, tool)

    original = quad.FFPROBE
    descriptor = os.memfd_create("cpu9-ffprobe-path-replacement")
    try:
        quad.FFPROBE = tool
        probe = quad._run_probe(descriptor)
    finally:
        quad.FFPROBE = original
        os.close(descriptor)
    assert probe["sentinel"] == "replacement-B"


def test_hold_replay_current_upstream_a_holds_are_not_runtime_gates(
    tmp_path: Path,
) -> None:
    """The frozen S7/S8 A-HOLD map is descriptive, not checked by admission."""

    strict = quad.load_strict_io(PROJECT)
    freeze_path = (
        PROJECT
        / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/cpu9_quad_review_video/CPU_FREEZE_FINAL.json"
    )
    freeze, record = strict.read_json_nofollow(
        freeze_path,
        expected_bytes=8582,
        expected_sha256="52aa607db07f457aae5956cadcb10feed568f4bf571a75caa97d09dde958b263",
        allowed_root=PROJECT / "_run",
    )
    assert freeze["current_upstream_execution_holds"]["S7"].startswith("A_HOLD")
    assert freeze["current_upstream_execution_holds"]["S8"].startswith("A_HOLD")
    quad.validate_cpu_freeze(
        freeze,
        freeze_record=record,
        expected_runner_sha256=freeze["implementation"]["runner"]["sha256"],
    )

    author_test_path = PROJECT / "tests/test_quad_review_video_t1.py"
    author_spec = importlib.util.spec_from_file_location(
        "cpu9_quad_author_helpers_for_independent_hold", author_test_path
    )
    assert author_spec and author_spec.loader
    author_tests = importlib.util.module_from_spec(author_spec)
    sys.modules[author_spec.name] = author_tests
    author_spec.loader.exec_module(author_tests)
    synthetic_project = tmp_path / "project"
    (synthetic_project / "_run").mkdir(parents=True)
    manifest, _ = author_tests._build_inputs(synthetic_project)
    admissions = quad.validate_input_manifest(
        manifest,
        strict=strict,
        project_root=synthetic_project,
    )
    assert [item.role for item in admissions] == list(quad.LAYOUT)
    arm_qa_ref = manifest["panels"][3]["producer_independent_qa"]
    arm_qa = Path(arm_qa_ref["path"]).read_text()
    assert "INDEPENDENT_QA_S7" not in arm_qa
    assert "INDEPENDENT_QA_S8" not in arm_qa
    assert "supersed" not in arm_qa.lower()


def test_hold_replay_published_output_can_be_replaced_before_evidence_hash(
    tmp_path: Path,
) -> None:
    """The encoder fd closes before the published pathname is hashed."""

    strict = quad.load_strict_io(PROJECT)
    output = tmp_path / "output.bin"
    child_payload = b"child-output-A"
    replacement_payload = b"replacement-output-B"
    handle = strict.start_command_exclusive_output(
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'wb').write(b'child-output-A')",
            "{output_fd}",
        ],
        output,
        allowed_root=tmp_path,
    )
    assert isinstance(handle.process, subprocess.Popen)

    original_link = strict.os.link

    def link_then_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        os.unlink(destination, dir_fd=dst_dir_fd)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, replacement_payload)
        finally:
            os.close(descriptor)

    strict.os.link = link_then_replace
    try:
        terminal = handle.wait_and_publish(timeout=10)
    finally:
        strict.os.link = original_link
        if handle.process.poll() is None:
            handle.abort()

    assert child_payload != replacement_payload
    assert output.read_bytes() == replacement_payload
    assert terminal["sha256"] == hashlib.sha256(replacement_payload).hexdigest()
