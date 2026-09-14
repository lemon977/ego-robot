from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import finalize_checkpoint_2000_control_plane as finalizer  # noqa: E402
import immutable_artifact_io as strict_io  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _paths(tmp_path: Path) -> finalizer.FinalizerPaths:
    project = tmp_path / "project"
    run = project / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829"
    checkpoints = run / "checkpoints"
    audits = checkpoints / "audits"
    guardian = project / "_run/guardian_v1"
    for directory in (checkpoints, audits, guardian, project / "tools"):
        directory.mkdir(parents=True, exist_ok=True)
    return finalizer.FinalizerPaths(
        project_root=project,
        run_root=run,
        checkpoint_root=checkpoints,
        audit_root=audits,
        supervisor_heartbeat=run / "HEARTBEAT.json",
        guardian_heartbeat=guardian / "HEARTBEAT.json",
        shared_io=project / "tools/immutable_artifact_io.py",
        utilization_tool=project / "tools/build_autonomous_checkpoint_utilization.py",
        windowed_redline_tool=project / "tools/build_autonomous_redline_increment_windowed.py",
        legacy_redline_tool=project / "tools/build_autonomous_redline_increment.py",
    )


def _supervisor(timestamp: str) -> dict[str, object]:
    return {
        "schema_version": "autonomous-20h-heartbeat-v1",
        "timestamp": timestamp,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "gpu_current_card": None,
        "cpu_current_cards": [],
        "counts": {"completed": 5, "degraded_a": 5, "suspended": 0},
    }


def _guardian(timestamp: str) -> dict[str, object]:
    return {
        "schema_version": "project-guardian-heartbeat-v1",
        "timestamp": timestamp,
        "mode": "READ_ONLY_MONITOR_NO_AUTOMATIC_RESTART",
        "policy": {
            "cross_human_gate": False,
            "modify_outputs": False,
            "restart_producer": False,
        },
        "suspicious_pre_review_processes": [],
    }


def test_cutoff_uses_supervisor_timestamp_and_accepts_fresh_older_guardian(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_json(paths.supervisor_heartbeat, _supervisor("2026-08-29T19:40:11+08:00"))
    _write_json(paths.guardian_heartbeat, _guardian("2026-08-29T19:40:00+08:00"))
    supervisor, guardian = finalizer._capture_heartbeats(
        paths, now=datetime.fromisoformat("2026-08-29T19:41:00+08:00")
    )
    assert supervisor.timestamp.isoformat() == "2026-08-29T19:40:11+08:00"
    assert (supervisor.timestamp - guardian.timestamp).total_seconds() == 11
    assert supervisor.successor_path.name.startswith("SUPERVISOR_HEARTBEAT_20260829T194011+0800_")
    assert guardian.successor_path.name.startswith("GUARDIAN_HEARTBEAT_20260829T194000+0800_")
    outputs = finalizer._output_paths(supervisor.timestamp, 1, paths)
    assert outputs["metrics"].name == "CHECKPOINT_2000_UTILIZATION_CUTOFF_1940_T0_V1.json"
    assert outputs["redline"].name == "RED_LINE_INCREMENT_1600_1940_T0_V1.json"


def test_stale_guardian_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_json(paths.supervisor_heartbeat, _supervisor("2026-08-29T19:40:11+08:00"))
    _write_json(paths.guardian_heartbeat, _guardian("2026-08-29T19:38:10+08:00"))
    with pytest.raises(finalizer.FinalizerError, match="stale guardian heartbeat"):
        finalizer._capture_heartbeats(
            paths, now=datetime.fromisoformat("2026-08-29T19:41:00+08:00")
        )


def test_guardian_later_than_supervisor_cutoff_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_json(paths.supervisor_heartbeat, _supervisor("2026-08-29T19:40:11+08:00"))
    _write_json(paths.guardian_heartbeat, _guardian("2026-08-29T19:40:12+08:00"))
    with pytest.raises(finalizer.FinalizerError, match="later than the supervisor cutoff"):
        finalizer._capture_heartbeats(
            paths, now=datetime.fromisoformat("2026-08-29T19:41:00+08:00")
        )


def test_path_replacement_is_detected_even_when_payload_is_identical(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.checkpoint_root / "fixed.json"
    target.write_bytes(b"{}\n")
    record = strict_io.read_bytes_nofollow(target, allowed_root=paths.project_root)
    replacement = paths.checkpoint_root / "replacement.json"
    replacement.write_bytes(record.payload)
    os.replace(replacement, target)
    with pytest.raises(finalizer.FinalizerError, match="inode replacement"):
        finalizer._revalidate(record, paths)


def test_alias_and_existing_output_are_rejected_before_publication(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    a = paths.checkpoint_root / "a.json"
    with pytest.raises(finalizer.FinalizerError, match="path alias"):
        finalizer._require_unique_paths({"a": a, "b": a.parent / "." / a.name})
    a.write_text("{}\n", encoding="utf-8")
    with pytest.raises(finalizer.FinalizerError, match="already exists or is an alias"):
        finalizer._require_absent(a)
    link = paths.checkpoint_root / "link.json"
    link.symlink_to(a)
    with pytest.raises(finalizer.FinalizerError, match="already exists or is an alias"):
        finalizer._require_absent(link)


def test_verdict_abbreviation_is_rejected_and_full_top_level_value_is_preserved() -> None:
    qa = {
        "overall_verdict": {
            "status": "PASS_FULL_EXACT_STATUS_WITH_ZERO_ADMISSION",
        },
        "summary": {"qa_verdict": "PASS"},
    }
    with pytest.raises(finalizer.FinalizerError, match="abbreviation or mismatch"):
        finalizer._qa_verdict(qa, "PASS", label="synthetic QA")
    assert finalizer._qa_verdict(
        qa, "PASS_FULL_EXACT_STATUS_WITH_ZERO_ADMISSION", label="synthetic QA"
    ) == {
        "field": "overall_verdict.status",
        "value": "PASS_FULL_EXACT_STATUS_WITH_ZERO_ADMISSION",
    }


def test_subject_reference_accepts_exact_expected_and_observed_schema_pairs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    subject = paths.checkpoint_root / "subject.json"
    subject.write_bytes(b"{}\n")
    record = strict_io.read_bytes_nofollow(subject, allowed_root=paths.project_root)
    expected_schema = {
        "subject": {
            "path": record.path,
            "expected_bytes": record.bytes,
            "expected_sha256": record.sha256,
        }
    }
    observed_schema = {
        "subject": {
            "path": record.path,
            "observed_bytes": record.bytes,
            "observed_sha256": record.sha256,
        }
    }
    assert finalizer._contains_exact_ref(expected_schema, record)
    assert finalizer._contains_exact_ref(observed_schema, record)
    mixed_pair = {
        "subject": {
            "path": record.path,
            "expected_bytes": record.bytes,
            "observed_sha256": record.sha256,
        }
    }
    assert not finalizer._contains_exact_ref(mixed_pair, record)


def test_dual_scope_does_not_conflate_named_false_with_canonical_absence() -> None:
    valid = {
        "named_ledger_scope": {
            "scope": "NAMED_LEDGER",
            "new_a_event": False,
        },
        "canonical_evidence_scope": {
            "scope": "CANONICAL_EVIDENCE",
            "retained_a_at_window_start": {
                "present": True,
                "owner_decision_group_ids": ["A1", "A2", "A3"],
            },
            "a_state_at_cutoff": {"present": True},
        },
    }
    finalizer._validate_dual_scope(valid)
    conflated = json.loads(json.dumps(valid))
    conflated["canonical_evidence_scope"]["retained_a_at_window_start"] = {
        "present": False,
        "owner_decision_group_ids": [],
    }
    with pytest.raises(finalizer.FinalizerError, match="erased retained canonical A"):
        finalizer._validate_dual_scope(conflated)
    same_label = json.loads(json.dumps(valid))
    same_label["canonical_evidence_scope"]["scope"] = "NAMED_LEDGER"
    with pytest.raises(finalizer.FinalizerError, match="scope labels are conflated"):
        finalizer._validate_dual_scope(same_label)


def test_content_addressed_heartbeat_successor_is_exact_and_o_excl(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_json(paths.supervisor_heartbeat, _supervisor("2026-08-29T19:40:11+08:00"))
    _write_json(paths.guardian_heartbeat, _guardian("2026-08-29T19:40:00+08:00"))
    supervisor, _ = finalizer._capture_heartbeats(
        paths, now=datetime.fromisoformat("2026-08-29T19:41:00+08:00")
    )
    finalizer._require_absent(supervisor.successor_path)
    published = finalizer._publish_heartbeat(supervisor, paths)
    assert published.payload == supervisor.record.payload
    assert published.sha256 == supervisor.record.sha256
    with pytest.raises(FileExistsError):
        finalizer._publish_heartbeat(supervisor, paths)


def test_relative_explicit_ref_and_noncanonical_byte_count_are_rejected() -> None:
    with pytest.raises(finalizer.FinalizerError, match="path must be absolute"):
        finalizer._parse_ref("x", ["relative.json", "1", "0" * 64])
    with pytest.raises(finalizer.FinalizerError, match="canonical non-negative decimal"):
        finalizer._parse_ref("x", ["/tmp/x", "01", "0" * 64])


def test_canonical_scope_retains_packet_a_and_gap_contradiction() -> None:
    packet = {
        "status": "OWNER_DECISIONS_REQUIRED",
        "decision_groups": [
            {"decision_id": "A1"},
            {"decision_id": "A2"},
            {"decision_id": "A3"},
        ],
        "checkpoint_1600_post_sync_binding": {
            "execution_gpu_pixel_queue_admission": 0,
            "canonical_nonledger_bq05_finding_id": "BQ05",
            "supervisor_successor_activation": "HOLD",
            "all_a_holds": "UNCHANGED",
        },
        "eligible68_future_mask_coverage_contract_hold_delta": {
            "new_a_class_p0_count": 2,
            "new_a_class_p0_ids": ["C1", "C2"],
            "new_p1_count": 2,
            "new_p1_ids": ["P1", "P2"],
        },
    }
    matrix = {"status": "NOT_READY", "readiness_invariants_after_correction": {}}
    gap = {
        "status": "CONTRADICTED",
        "requirement_count": 131,
        "leaf_state_counts": {"PROVEN": 3, "CONTRADICTED": 44, "INCOMPLETE": 84, "MISSING": 0},
        "admission_and_governance": {"execution_gpu_pixel_queue_admission": 0},
        "completion_conclusion": {
            "goal_complete": False,
            "evidence_state": "CONTRADICTED",
            "eta": "UNMEASURED",
        },
        "b_p1_closure": [],
    }
    scope = finalizer._canonical_scope(
        packet,
        matrix,
        gap,
        {"findings": {"subject_new_p0_count": 0}},
        {"gap_qa": {"field": "overall_verdict.status", "value": "PASS"}},
        {"packet": {"path": "/packet", "bytes": 1, "sha256": "0" * 64}},
    )
    assert scope["retained_a_at_window_start"]["present"] is True
    assert scope["retained_a_at_window_start"]["bq05_finding_id"] == "BQ05"
    assert scope["new_a_evidence_during_window"]["state"] == "ZERO_REPORTED_BY_CURRENT_GAP_QA"
    assert scope["gap_v2"]["completion_conclusion"]["goal_complete"] is False
    assert scope["source_refs"]["packet"]["path"] == "/packet"
    assert "BQ05" in scope["finding_ids"]


def test_source_contains_no_sleep_discovery_signal_subprocess_or_gpu_call() -> None:
    source = (PROJECT_ROOT / "tools/finalize_checkpoint_2000_control_plane.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "time.sleep(",
        "os.walk(",
        ".rglob(",
        "glob.glob(",
        "os.listdir(",
        "os.scandir(",
        "subprocess.",
        "os.kill(",
        "nvidia-smi",
    )
    assert [token for token in forbidden if token in source] == []
