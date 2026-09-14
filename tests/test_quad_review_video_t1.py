from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_quad_review_video_t1.py"
SPEC = importlib.util.spec_from_file_location("quad_review_video_t1_tested", RUNNER)
assert SPEC and SPEC.loader
quad = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quad
SPEC.loader.exec_module(quad)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    return _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _contract(session_id: str = "grap_a_cap_004") -> dict[str, object]:
    return dict(quad.SESSION_CONTRACTS[session_id])


def _fake_freeze(project: Path, strict: object) -> tuple[dict[str, object], object, str]:
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    freeze = {
        "schema_version": "cpu9-quad-review-video-cpu-freeze-v1",
        "status": "CPU_READY_WAITING_FOUR_A_CLEAN_SHA_BOUND_SINGLE_PANEL_VIDEOS_NO_MEDIA_EXECUTED",
        "implementation": {"runner": {"sha256": runner_sha}},
        "lineage_hardening": {
            "shared_io": {"sha256": quad.SHARED_IO_SHA256},
            "runner_entry": "SAME_O_NOFOLLOW_FD_VERIFIED_BYTES_COMPILE_EXEC",
        },
    }
    ref = _write_json(project / "_run" / "CPU_FREEZE.json", freeze)
    value, record = strict.read_json_nofollow(
        ref["path"],
        expected_bytes=ref["bytes"],
        expected_sha256=ref["sha256"],
        allowed_root=project / "_run",
    )
    return value, record, runner_sha


def _build_inputs(
    project: Path,
    *,
    session_id: str = "grap_a_cap_004",
    video_payloads: dict[str, bytes] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    contract = _contract(session_id)
    source = project.parent / "authoritative_raw" / session_id
    input_root = project / "_run" / "quad_inputs"
    audit_root = project / "audits" / "quad_inputs"
    payloads = video_payloads or {
        role: f"synthetic-{role}".encode() for role in quad.LAYOUT
    }
    panels: list[dict[str, object]] = []
    for role in quad.LAYOUT:
        video_path = (
            source / "raw.mp4"
            if role == "RAW"
            else input_root / role.lower() / f"{role.lower()}.mp4"
        )
        video_ref = _write(video_path, payloads[role])
        producer: dict[str, object] = {
            "schema_version": "test-panel-producer-v1",
            "status": "CANDIDATE_ACCEPTABLE",
            "session_id": session_id,
            "frame_count": contract["frame_count"],
            "fps": contract["fps"],
            "width": contract["resolution"][0],
            "height": contract["resolution"][1],
            "video": video_ref,
            "governance": dict(quad.GOVERNANCE),
        }
        if role == "ARM_KAIHAND":
            producer.update(
                {
                    "mount_provenance": "PROVISIONAL_MOUNT_VISUAL_ONLY",
                    "contact_infeasible": "UNMEASURED",
                }
            )
        producer_ref = _write_json(
            input_root / role.lower() / "PRODUCER_MANIFEST.json", producer
        )
        panel_qa = {
            "schema_version": "quad-panel-producer-independent-qa-v1",
            "producer": "INDEPENDENT_CPU_QA",
            "status": "PASS",
            "a_class_p0": False,
            "session_id": session_id,
            "role": role,
            "producer_manifest": producer_ref,
            "video": video_ref,
            "candidate_requires_human_review": True,
            "next_bucket_blocked": True,
            "advancement_authorized": False,
            "formal_consumer_allowed": False,
        }
        qa_ref = _write_json(audit_root / f"{role}_QA.json", panel_qa)
        panels.append(
            {
                "role": role,
                "representation": quad.REPRESENTATIONS[role],
                "video": video_ref,
                "producer_manifest": producer_ref,
                "producer_schema_version": "test-panel-producer-v1",
                "producer_independent_qa": qa_ref,
                "binding": {
                    "video_ref_pointer": "/video",
                    "status_pointer": "/status",
                    "admitted_status": "CANDIDATE_ACCEPTABLE",
                    "session_id_pointer": "/session_id",
                    "frame_count_pointer": "/frame_count",
                    "fps_pointer": "/fps",
                    "width_pointer": "/width",
                    "height_pointer": "/height",
                    "governance_pointer": None if role == "RAW" else "/governance",
                },
            }
        )
    manifest = {
        "schema_version": "quad-review-input-manifest-v1",
        "auth_tier": "T1_CANDIDATE_REVIEW_ONLY",
        "session_id": session_id,
        "frame_count": contract["frame_count"],
        "fps": contract["fps"],
        "source_resolution": contract["resolution"],
        "layout": list(quad.LAYOUT),
        "frame_order_policy": "ZERO_BASED_CONTIGUOUS_DECODE_ORDER_NO_FILL",
        "display_contract": {
            "panel_resolution": quad.DISPLAY_PANEL_RESOLUTION,
            "header_height": quad.HEADER_HEIGHT,
            "output_resolution": quad.OUTPUT_RESOLUTION,
            "resample_semantics": "DISPLAY_ONLY_AREA_RESAMPLE_NO_SOURCE_PIXEL_SEMANTICS",
            "crop_or_pad": "NONE",
        },
        "governance": dict(quad.GOVERNANCE),
        "panels": panels,
    }
    return manifest, panels


@pytest.fixture
def strict() -> object:
    return quad.load_strict_io(PROJECT)


def test_shared_io_is_compiled_from_same_verified_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"SENTINEL = 'verified-bytes'\n"
    expected = hashlib.sha256(payload).hexdigest()
    opened = io.BytesIO(payload)
    monkeypatch.setattr(quad.os, "open", lambda *_args, **_kwargs: 19)
    monkeypatch.setattr(quad.os, "fstat", lambda _fd: type("S", (), {"st_mode": 0o100444})())
    monkeypatch.setattr(quad.stat, "S_ISREG", lambda _mode: True)
    monkeypatch.setattr(quad.os, "read", lambda _fd, _n: opened.read(_n))
    monkeypatch.setattr(quad.os, "close", lambda _fd: None)
    module = quad._same_fd_module(
        Path("/path/replaced-after-open.py"),
        expected_bytes=len(payload),
        expected_sha256=expected,
        module_name="same_fd_counterexample",
    )
    assert module.SENTINEL == "verified-bytes"


def test_manifest_admits_four_exact_joined_a_clean_panels(tmp_path: Path, strict: object) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, _ = _build_inputs(project)
    admitted = quad.validate_input_manifest(manifest, strict=strict, project_root=project)
    assert [item.role for item in admitted] == list(quad.LAYOUT)


def test_panel_qa_must_bind_exact_producer_and_video(tmp_path: Path, strict: object) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, panels = _build_inputs(project)
    qa_path = Path(panels[1]["producer_independent_qa"]["path"])
    qa = json.loads(qa_path.read_text())
    qa["video"]["sha256"] = "0" * 64
    qa_ref = _write_json(qa_path.with_name("MASK_QA_TAMPERED.json"), qa)
    manifest["panels"][1]["producer_independent_qa"] = qa_ref
    with pytest.raises(quad.QuadReviewError, match="producer independent QA mismatch: video"):
        quad.validate_input_manifest(manifest, strict=strict, project_root=project)


def test_current_a_hold_cannot_be_laundered_as_panel_pass(tmp_path: Path, strict: object) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, panels = _build_inputs(project)
    qa_path = Path(panels[3]["producer_independent_qa"]["path"])
    qa = json.loads(qa_path.read_text())
    qa["status"] = "HOLD_A_P0_3_S8_EXECUTION_FORBIDDEN"
    qa["a_class_p0"] = True
    qa_ref = _write_json(qa_path.with_name("ARM_QA_A_HOLD.json"), qa)
    manifest["panels"][3]["producer_independent_qa"] = qa_ref
    with pytest.raises(quad.QuadReviewError, match="producer independent QA mismatch: status"):
        quad.validate_input_manifest(manifest, strict=strict, project_root=project)


def test_cross_panel_video_alias_is_rejected(tmp_path: Path, strict: object) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, _ = _build_inputs(project)
    manifest["panels"][2]["video"] = dict(manifest["panels"][1]["video"])
    with pytest.raises(quad.QuadReviewError, match="does not bind the admitted video"):
        quad.validate_input_manifest(manifest, strict=strict, project_root=project)


def test_hold_producer_status_is_rejected(tmp_path: Path, strict: object) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, panels = _build_inputs(project)
    producer_path = Path(panels[2]["producer_manifest"]["path"])
    producer = json.loads(producer_path.read_text())
    producer["status"] = "HOLD_VISUAL_INCOMPLETE"
    producer_ref = _write_json(producer_path.with_name("PRODUCER_HOLD.json"), producer)
    manifest["panels"][2]["producer_manifest"] = producer_ref
    manifest["panels"][2]["binding"]["admitted_status"] = "HOLD_VISUAL_INCOMPLETE"
    with pytest.raises(quad.QuadReviewError, match="producer status is not consumable"):
        quad.validate_input_manifest(manifest, strict=strict, project_root=project)


@pytest.mark.parametrize("name", ["labels", "future_blind_split", "grap_a_cap_025", "processed"])
def test_forbidden_partition_paths_fail_closed(
    tmp_path: Path, strict: object, name: str
) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    manifest, _ = _build_inputs(project)
    bad = project / "_run" / name / "mask.mp4"
    manifest["panels"][1]["video"] = _write(bad, b"bad")
    with pytest.raises(quad.QuadReviewError, match="forbidden partition"):
        quad.validate_input_manifest(manifest, strict=strict, project_root=project)


def test_input_qa_binds_exact_manifest_record(tmp_path: Path, strict: object) -> None:
    path = tmp_path / "MANIFEST.json"
    record = type(
        "R",
        (),
        {"path": str(path), "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
    )()
    freeze_record = type(
        "R",
        (),
        {"path": str(tmp_path / "FREEZE.json"), "bytes": 3, "sha256": "a" * 64},
    )()
    qa = {
        "schema_version": "quad-review-input-independent-qa-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "input_manifest": {
            "path": record.path,
            "bytes": record.bytes,
            "sha256": record.sha256,
        },
        "cpu_freeze": {
            "path": freeze_record.path,
            "bytes": freeze_record.bytes,
            "sha256": freeze_record.sha256,
        },
        "session_id": "grap_a_cap_004",
        "layout": list(quad.LAYOUT),
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
    }
    quad.validate_independent_qa(
        qa,
        manifest_record=record,
        cpu_freeze_record=freeze_record,
        session_id="grap_a_cap_004",
    )
    qa["input_manifest"]["sha256"] = "f" * 64
    with pytest.raises(quad.QuadReviewError, match="does not bind"):
        quad.validate_independent_qa(
            qa,
            manifest_record=record,
            cpu_freeze_record=freeze_record,
            session_id="grap_a_cap_004",
        )


def test_sealed_input_survives_source_path_replacement(tmp_path: Path, strict: object) -> None:
    source = tmp_path / "source.mp4"
    original = b"verified-A"
    source.write_bytes(original)
    record = strict.read_bytes_nofollow(
        source,
        expected_bytes=len(original),
        expected_sha256=hashlib.sha256(original).hexdigest(),
    )
    descriptor = quad.sealed_memfd(record.payload, "input-counterexample")
    try:
        source.write_bytes(b"replacement-B")
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.read(descriptor, 1024) == original
        with pytest.raises(OSError):
            os.write(descriptor, b"mutation")
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        assert seals & fcntl.F_SEAL_WRITE
    finally:
        os.close(descriptor)


def test_short_decoder_frame_fails_without_fill() -> None:
    with pytest.raises(quad.QuadReviewError, match="ended before"):
        quad._read_exact(io.BytesIO(b"abc"), 4, "MASK frame 7")


def test_encoder_output_is_fd_bound_and_header_shape_is_exact() -> None:
    command = quad.encoder_command(30)
    assert command.count("{output_fd}") == 1
    assert command[-1] == "{output_fd}"
    assert "-y" in command
    assert not any(item.endswith(".mp4") for item in command)
    header = quad._header(12, "grap_a_cap_004")
    assert header.shape == (quad.HEADER_HEIGHT, quad.OUTPUT_RESOLUTION[0], 3)


def _make_video(path: Path, color: tuple[int, int, int], frames: int) -> bytes:
    width, height = 64, 48
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = color
    payload = frame.tobytes() * frames
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(quad.FFMPEG),
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return path.read_bytes()


def test_synthetic_three_frame_end_to_end_no_gpu(
    tmp_path: Path, strict: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    run_root = project / "_run"
    run_root.mkdir(parents=True)
    session = "synthetic_dev"
    monkeypatch.setitem(
        quad.SESSION_CONTRACTS,
        session,
        {"frame_count": 3, "fps": 30, "resolution": [64, 48]},
    )
    media_root = tmp_path / "media"
    payloads = {
        role: _make_video(media_root / f"{role}.mp4", color, 3)
        for role, color in zip(
            quad.LAYOUT,
            ((10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)),
        )
    }
    manifest, _ = _build_inputs(project, session_id=session, video_payloads=payloads)
    cpu_freeze, cpu_freeze_record, runner_sha = _fake_freeze(project, strict)
    manifest_ref = _write_json(run_root / "INPUT_MANIFEST.json", manifest)
    manifest_value, manifest_record = strict.read_json_nofollow(
        manifest_ref["path"],
        expected_bytes=manifest_ref["bytes"],
        expected_sha256=manifest_ref["sha256"],
        allowed_root=run_root,
    )
    qa = {
        "schema_version": "quad-review-input-independent-qa-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "input_manifest": manifest_ref,
        "cpu_freeze": {
            "path": cpu_freeze_record.path,
            "bytes": cpu_freeze_record.bytes,
            "sha256": cpu_freeze_record.sha256,
        },
        "session_id": session,
        "layout": list(quad.LAYOUT),
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
    }
    qa_ref = _write_json(run_root / "INPUT_QA.json", qa)
    qa_value, qa_record = strict.read_json_nofollow(
        qa_ref["path"],
        expected_bytes=qa_ref["bytes"],
        expected_sha256=qa_ref["sha256"],
        allowed_root=run_root,
    )
    result = quad.execute(
        strict=strict,
        project_root=project,
        cpu_freeze_record=cpu_freeze_record,
        cpu_freeze=cpu_freeze,
        expected_runner_sha256=runner_sha,
        manifest_record=manifest_record,
        manifest=manifest_value,
        qa_record=qa_record,
        qa=qa_value,
        output_parent=run_root,
        run_name="quad_synthetic_t0",
    )
    assert result["status"] == "CANDIDATE_REVIEW_VIDEO_COMPLETE_NOT_BASELINE"
    assert result["video_probe"]["nb_read_frames"] == "3"
    assert result["frame_policy"] == "EXACT_LOCKSTEP_NO_FILL_NO_REPEAT_NO_INTERPOLATION"
    assert (run_root / "quad_synthetic_t0" / "RUN_MANIFEST.json").is_file()
