from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import draft_eligible68_split_v2 as producer
from utils import source_contract


def _ref(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _valid_sidecar(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame_names=np.asarray([f"{i:05d}" for i in range(frames)]),
        timestamps_ns=np.arange(frames, dtype=np.int64),
        wrist_9d=np.zeros((frames, 2, 9)),
        wrist_T_camera=np.zeros((frames, 2, 4, 4)),
        q=np.zeros((frames, 2, 22)),
        valid=np.ones((frames, 2), dtype=bool),
        confidence=np.ones((frames, 2)),
        grasp=np.zeros((frames, 2)),
        joint_names=np.zeros((2, 22), dtype="U1"),
        joint_lower=np.full((2, 22), -1.0),
        joint_upper=np.full((2, 22), 1.0),
        canonical_sha256=np.asarray("0" * 64),
        schema_version=np.asarray("humanego-robot-sidecar-v1"),
        retarget_confidence=np.ones((frames, 2)),
        failure_reason=np.zeros((frames, 2), dtype="U1"),
        hand_object_T=np.zeros((frames, 2, 4, 4)),
    )


def _synthetic_refs(tmp_path: Path) -> tuple[dict, dict, Path]:
    subject_path = tmp_path / "subject.json"
    qa_path = tmp_path / "qa.json"
    rows = []
    for order, sid in enumerate(source_contract.ELIGIBLE68_ORDER):
        frames = source_contract.ELIGIBLE68_FRAME_COUNTS[sid]
        session_root = tmp_path / "production" / sid
        adapter = session_root / "09_humanego_adapter"
        adapter.mkdir(parents=True)
        status = session_root / "status.json"
        status.write_bytes(b"status")
        sidecar = tmp_path / "sidecars" / "kai22" / sid / "sidecar.npz"
        _valid_sidecar(sidecar, frames)
        constituents = []
        for role, suffix in enumerate(producer.CONSTITUENT_ROLES):
            if suffix == "retarget_sidecar":
                path = sidecar
            elif suffix == "production_status":
                path = status
            else:
                path = session_root / f"constituent_{role}.json"
                path.write_bytes(f"{sid}-{role}".encode())
            record = _ref(path)
            record.update({"role": suffix})
            constituents.append(record)
        rows.append(
            {
                "eligible68_order": order,
                "session_id": sid,
                "frame_count": frames,
                "constituent_count": 10,
                "canonical_manifest_entry": {
                    "adapter": str(adapter),
                    "production_status_sha256": _ref(status)["sha256"],
                },
                "completeness": {"complete": True},
                "constituents": constituents,
            }
        )
    subject = {
        "schema_version": producer.SUBJECT_SCHEMA,
        "status": "COMPLETE_68_OF_68_CONSTITUENT_METADATA_BYTES_BOUND_NO_PIXEL_OR_EXECUTION_ADMISSION",
        "sessions": rows,
    }
    subject_path.write_text(json.dumps(subject), encoding="utf-8")
    subject_ref = _ref(subject_path)
    qa = {
        "status": "PASS_68_OF_68_SYNTHETIC",
        "subject": subject_ref,
        "access_boundary": {
            "labels_opened": 0,
            "blind_payloads_opened": 0,
            "processed_or_clean_opened": 0,
            "raw_pixels_opened": 0,
            "gpu_or_model_calls": 0,
        },
        "recomputed": {
            "session_rows": 68,
            "constituent_files_exact_rehashed": 680,
            "forbidden_identity_rows": 0,
            "forbidden_session_directory_entries_or_files_opened_by_qa": 0,
            "forbidden_session_path_constructions_by_qa": 0,
            "static_no_discovery_guard_findings": 0,
            "literal_allowlist_order_match": True,
        },
        "verdict": {
            "pixel_or_selector_execution_admission": False,
            "gpu_queueable_sessions": 0,
        },
    }
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    return subject_ref, _ref(qa_path), tmp_path / "_run" / "draft"


def test_synthetic_draft_is_private_and_final_sidecars_are_not_read(
    tmp_path, monkeypatch
):
    subject_ref, qa_ref, output = _synthetic_refs(tmp_path)
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    output = output.parent / "nested" / "existing" / output.name
    output.parent.mkdir(parents=True)
    calls = 0
    original = producer._verify_sidecar_bytes

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(producer, "_verify_sidecar_bytes", spy)
    result = producer.publish_draft(
        subject_ref=subject_ref, author_qa_ref=qa_ref, output_root=output
    )
    assert calls == 60
    assert result["final_test8_sidecar_payload_reads"] == 0
    assert result["formal_consumer_allowed"] is False
    assert output.stat().st_mode & 0o777 == 0o700
    assert [p.name for p in output.iterdir()] == [
        "DRAFT_SPLIT.json",
        "DRAFT_REPORT.json",
    ]
    assert (output / "DRAFT_SPLIT.json").stat().st_mode & 0o777 == 0o444
    assert (output / "DRAFT_REPORT.json").stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="versioned"):
        source_contract.validate_eligible68_split_phase1(
            output / "DRAFT_SPLIT.json", "kai22", project_root=tmp_path
        )


def test_final_test_reader_spy_is_never_called(tmp_path, monkeypatch):
    subject_ref, qa_ref, output = _synthetic_refs(tmp_path)
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    output.parent.mkdir()
    final_stat_calls = 0
    original = producer._stat_final_sidecar

    def spy(*args, **kwargs):
        nonlocal final_stat_calls
        final_stat_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(producer, "_stat_final_sidecar", spy)
    result = producer.publish_draft(
        subject_ref=subject_ref, author_qa_ref=qa_ref, output_root=output
    )
    assert final_stat_calls == 8
    assert result["final_test8_sidecar_payload_reads"] == 0


def test_hardlinked_cli_ref_is_rejected_before_payload_use(tmp_path):
    subject_ref, qa_ref, _ = _synthetic_refs(tmp_path)
    alias = tmp_path / "subject-alias.json"
    alias.hardlink_to(Path(subject_ref["path"]))
    alias_ref = _ref(alias)
    producer.PROJECT_ROOT = tmp_path
    with pytest.raises(ValueError, match="singly-linked"):
        producer._preflight(alias_ref, qa_ref)


def test_intermediate_output_symlink_cannot_escape_run(tmp_path, monkeypatch):
    subject_ref, qa_ref, _ = _synthetic_refs(tmp_path)
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    run_root = tmp_path / "_run"
    run_root.mkdir()
    outside = tmp_path / "outside" / "sub"
    outside.mkdir(parents=True)
    hop = run_root / "hop"
    hop.symlink_to(outside, target_is_directory=True)
    output = hop / "root"

    with pytest.raises(ValueError, match="ordinary held directory"):
        producer.publish_draft(
            subject_ref=subject_ref,
            author_qa_ref=qa_ref,
            output_root=output,
        )
    assert not (outside / "root").exists()
    assert not output.exists()


def test_output_ancestor_rename_and_symlink_drift_is_rejected(tmp_path, monkeypatch):
    subject_ref, qa_ref, _ = _synthetic_refs(tmp_path)
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    held = tmp_path / "_run" / "held"
    parent = held / "parent"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "_run" / "held-moved"
    output = parent / "root"
    original = producer._reverify_directory_chain
    drifted = False

    def rename_and_replace(chain):
        nonlocal drifted
        if not drifted:
            held.rename(moved)
            held.symlink_to(outside, target_is_directory=True)
            drifted = True
        return original(chain)

    monkeypatch.setattr(producer, "_reverify_directory_chain", rename_and_replace)
    with pytest.raises(ValueError, match="identity drift"):
        producer.publish_draft(
            subject_ref=subject_ref,
            author_qa_ref=qa_ref,
            output_root=output,
        )
    assert not (outside / "parent" / "root").exists()
    assert not (moved / "parent" / "root").exists()


def test_role_drift_and_foreign_output_are_rejected(tmp_path):
    subject_ref, qa_ref, output = _synthetic_refs(tmp_path)
    producer.PROJECT_ROOT = tmp_path
    payload = json.loads(Path(subject_ref["path"]).read_text())
    payload["sessions"][0]["constituents"][0]["role"] = "foreign"
    Path(subject_ref["path"]).write_text(json.dumps(payload), encoding="utf-8")
    subject_ref = _ref(Path(subject_ref["path"]))
    qa_payload = json.loads(Path(qa_ref["path"]).read_text())
    qa_payload["subject"] = subject_ref
    Path(qa_ref["path"]).write_text(json.dumps(qa_payload), encoding="utf-8")
    qa_ref = _ref(Path(qa_ref["path"]))
    output.parent.mkdir()
    with pytest.raises(ValueError, match="role"):
        producer.publish_draft(
            subject_ref=subject_ref, author_qa_ref=qa_ref, output_root=output
        )

    clean_subject_ref, clean_qa_ref, occupied = _synthetic_refs(tmp_path / "occupied")
    producer.PROJECT_ROOT = tmp_path / "occupied"
    occupied.parent.mkdir()
    occupied.mkdir(mode=0o700)
    marker = occupied / "foreign-marker"
    marker.write_bytes(b"must survive")
    with pytest.raises(FileExistsError):
        producer.publish_draft(
            subject_ref=clean_subject_ref,
            author_qa_ref=clean_qa_ref,
            output_root=occupied,
        )
    assert marker.read_bytes() == b"must survive"
