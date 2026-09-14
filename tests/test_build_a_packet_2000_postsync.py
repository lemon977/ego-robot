from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import build_a_packet_2000_postsync as packet_builder  # noqa: E402
import immutable_artifact_io as strict_io  # noqa: E402


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: object) -> None:
    _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _record(path: Path, root: Path) -> strict_io.VerifiedBytes:
    return strict_io.read_bytes_nofollow(path, allowed_root=root)


def _node(record: strict_io.VerifiedBytes) -> dict[str, object]:
    return {"path": record.path, "bytes": record.bytes, "sha256": record.sha256}


def _qa(value: str, subjects: list[strict_io.VerifiedBytes]) -> dict[str, object]:
    return {
        "overall_verdict": {"verdict": value},
        "subjects": [_node(subject) for subject in subjects],
    }


def _fixture(tmp_path: Path) -> tuple[
    packet_builder.BuilderPaths,
    dict[str, packet_builder.ExplicitRef],
    dict[str, str],
    str,
]:
    project = tmp_path / "project"
    run = project / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829"
    checkpoints = run / "checkpoints"
    audits = checkpoints / "audits"
    paths = packet_builder.BuilderPaths(
        project_root=project,
        run_root=run,
        checkpoint_root=checkpoints,
        audit_root=audits,
        output=run / "A_CLASS_DECISION_QUEUE_2000_V1.json",
    )
    for directory in (run, checkpoints, audits, project / "task"):
        directory.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {
        "packet_1600_v3": run / "A_CLASS_DECISION_QUEUE_1600_V3.json",
        "packet_1600_v3_qa": audits / "INDEPENDENT_QA_A_CLASS_DECISION_QUEUE_1600_V3_T0_V1.json",
        "final_metrics": checkpoints / "CHECKPOINT_2000_UTILIZATION_CUTOFF_1940_T0_V1.json",
        "final_redline": checkpoints / "RED_LINE_INCREMENT_1600_1940_T0_V1.json",
        "final_pack": checkpoints / "CHECKPOINT_2000_EVIDENCE_PACK_T0_V1.json",
        "final_artifacts_qa": audits / "INDEPENDENT_QA_CHECKPOINT_2000_FINAL_ARTIFACTS_T0_V1.json",
        "pre2000_pointer": checkpoints / "OWNER_AND_TASK_DOC_PRE_2000_PAYLOAD_PRESERVATION_T0_V1.json",
        "document_sync_preflight_v2": checkpoints / "CHECKPOINT_2000_DOCUMENT_SYNC_PREFLIGHT_T0_V2.json",
        "document_sync_preflight_v2_qa": audits / "INDEPENDENT_QA_CHECKPOINT_2000_DOCUMENT_SYNC_PREFLIGHT_T0_V2.json",
        "document_sync_result_qa": audits / "INDEPENDENT_QA_CHECKPOINT_2000_DOCUMENT_SYNC_RESULT_T0_V1.json",
        "gap_v2": checkpoints / "FINAL_GOAL_COMPLETION_GAP_AUDIT_PRE_2000_T0_V2.json",
        "gap_v2_method_hold_qa": audits / "INDEPENDENT_QA_FINAL_GOAL_COMPLETION_GAP_AUDIT_PRE_2000_T0_V2.json",
        "gap_v2_clean_qa": audits / "INDEPENDENT_QA_FINAL_GOAL_COMPLETION_GAP_AUDIT_PRE_2000_T0_V2_CLEAN_REPLAY_V1.json",
        "matrix_v7": checkpoints / "DELIVERY_REQUIREMENT_EVIDENCE_MATRIX_T0_V7.json",
        "matrix_v7_qa": audits / "INDEPENDENT_QA_DELIVERY_REQUIREMENT_EVIDENCE_MATRIX_T0_V7.json",
        "summary_preflight": checkpoints / "OVERNIGHT_SUMMARY_20260829_PREFLIGHT_T0_V1.json",
        "summary_preflight_qa": audits / "INDEPENDENT_QA_OVERNIGHT_SUMMARY_20260829_PREFLIGHT_T0_V1.json",
        "finalizer_author_freeze": checkpoints / "CHECKPOINT_2000_CONTROL_PLANE_FINALIZER_AUTHOR_FREEZE_T0_V1.json",
        "finalizer_qa": audits / "INDEPENDENT_QA_CHECKPOINT_2000_CONTROL_PLANE_FINALIZER_T0_V1.json",
        "finalizer_scheduled": checkpoints / "CHECKPOINT_2000_FINALIZER_SCHEDULED_T0_V1.json",
        "finalizer_terminal": checkpoints / "CHECKPOINT_2000_FINALIZER_SCHEDULER_TERMINAL_T0_V1.json",
    }

    snapshot_records: dict[str, strict_io.VerifiedBytes] = {}
    old_live_records: dict[str, strict_io.VerifiedBytes] = {}
    for index, spec in enumerate(packet_builder.DOC_SPECS):
        old = f"# {spec['logical']}\npre-2000-{index}\n".encode()
        digest = hashlib.sha256(old).hexdigest()
        snapshot = checkpoints / f"{spec['snapshot_prefix']}{digest}.md"
        live = project / spec["live_relative"]
        _write(snapshot, old)
        _write(live, old)
        files[spec["snapshot_role"]] = snapshot
        files[spec["postlive_role"]] = live
        snapshot_records[spec["snapshot_role"]] = _record(snapshot, project)
        old_live_records[spec["postlive_role"]] = _record(live, project)

    stable_rows: list[dict[str, object]] = []
    for index in range(99):
        path = checkpoints / f"stable_{index:03d}.json"
        _write_json(path, {"stable": index})
        record = _record(path, project)
        stable_rows.append({"id": f"stable_{index:03d}", **_node(record)})
    stale_rows = []
    for spec in packet_builder.DOC_SPECS:
        snapshot = snapshot_records[spec["snapshot_role"]]
        live = old_live_records[spec["postlive_role"]]
        stale_rows.append(
            {
                "id": spec["stale_ref_id"],
                "path": live.path,
                "bytes": snapshot.bytes,
                "sha256": snapshot.sha256,
            }
        )
    base = {
        "schema_version": "autonomous-20h-a-class-owner-decision-queue-1600-v3-b-reporting-fix",
        "status": "OWNER_DECISIONS_REQUIRED",
        "producer": "fixture",
        "created_at": "2026-08-29T16:00:00+08:00",
        "scope": "fixture",
        "claim_limit": "fixture",
        "supersedes": {},
        "decision_groups": [
            {"decision_id": "A-DECISION-01", "owner_options": ["HOLD"]},
            {"decision_id": "A-DECISION-02", "owner_options": ["HOLD"]},
            {"decision_id": "A-DECISION-03", "owner_options": ["HOLD"]},
        ],
        "decision_response_contract": {"default_without_owner_response": "KEEP_EACH_A_CLASS_LINE_ON_HOLD"},
        "checkpoint_1600_post_sync_binding": {
            "supervisor_successor_activation": "HOLD_ROUND3_EXHAUSTED_BQ05_DEGRADED_TO_A",
            "execution_gpu_pixel_queue_admission": 0,
        },
        "eligible68_future_mask_coverage_contract_hold_delta": {
            "new_a_class_p0_count": 2,
            "new_a_class_p0_ids": ["C1", "C2"],
            "new_p1_count": 2,
            "new_p1_ids": ["P1", "P2"],
        },
        "supervisor_control_plane_round3_final_a_hold_delta": {"successor_activation": "HOLD"},
        "authority_binding": {"state": "HOLD"},
        "authority_rule": "owner",
        "canonical_references": stable_rows + stale_rows,
        "actions_performed": {"execution_admitted": False},
        "semantic_preservation_validation": {},
        "reference_validation": {},
    }
    _write_json(files["packet_1600_v3"], base)

    # Authorized append happens only after the immutable snapshots and V3
    # capture exist.
    for spec in packet_builder.DOC_SPECS:
        live = files[spec["postlive_role"]]
        _write(live, snapshot_records[spec["snapshot_role"]].payload + b"post-2000\n")

    documents = []
    for spec in packet_builder.DOC_SPECS:
        snapshot = snapshot_records[spec["snapshot_role"]]
        live = _record(files[spec["postlive_role"]], project)
        documents.append(
            {
                "logical_document": spec["logical"],
                "immutable_snapshot": _node(snapshot),
                "historical_same_path_capture": {
                    "path": live.path,
                    "bytes": snapshot.bytes,
                    "sha256": snapshot.sha256,
                    "status_after_authorized_2000_append": "PATH_STALE",
                },
            }
        )
    _write_json(
        files["pre2000_pointer"],
        {"historical_same_path_reference_status_after_append": "PATH_STALE", "documents": documents},
    )

    _write_json(
        files["matrix_v7"],
        {
            "readiness_invariants_after_correction": {
                "grap_a_cap_004_all_columns": "NOT_READY",
                "grap_a_cap_002_all_columns": "NOT_READY",
                "current_coverage_admitted_sessions": 0,
                "current_coverage_total_sessions": 68,
                "current_coverage_report": "NOT_PRODUCED",
                "E68I": False,
                "eta": "UNMEASURED",
                "command_admission": 0,
                "gpu_admission": 0,
                "pixel_or_selector_admission": 0,
            }
        },
    )
    _write_json(
        files["gap_v2"],
        {
            "status": "CONTRADICTED",
            "requirement_count": 131,
            "admission_and_governance": {"execution_gpu_pixel_queue_admission": 0},
            "completion_conclusion": {"goal_complete": False},
        },
    )
    _write_json(
        files["summary_preflight"],
        {"actions_performed": {"final_summary_created": False, "execution_gpu_pixel_queue_admission": 0}},
    )
    _write_json(files["finalizer_author_freeze"], {"status": "CPU_READY_NOT_EXECUTED"})
    _write_json(
        files["finalizer_scheduled"],
        {
            "status": "SCHEDULED_CPU_CONTROL_PLANE_ONLY_NOT_EXECUTED",
            "terminal_path": str(files["finalizer_terminal"]),
        },
    )
    _write_json(
        files["document_sync_preflight_v2"],
        {
            "status": "CPU11_FULL_20_MINUTE_ENVELOPE_READY_NOT_EXECUTED_ZERO_ADMISSION",
            "post_2000_packet_minimum_contract": {
                "required_successor_path": str(paths.output),
                "bind_exactly": [f"binding-{index}" for index in range(11)],
            },
        },
    )
    _write_json(files["final_metrics"], {"status": "FINAL_METRICS"})
    _write_json(files["final_redline"], {"status": "FINAL_REDLINE"})

    current = {role: _record(path, project) for role, path in files.items() if path.exists()}
    verdicts = {
        **packet_builder.STATIC_QA_VERDICTS,
        "final_artifacts_qa": "PASS_FINAL_ARTIFACTS_QA_ZERO_ADMISSION",
        "document_sync_result_qa": "PASS_DOCUMENT_SYNC_RESULT_ZERO_ADMISSION",
    }
    qa_subjects = {
        "packet_1600_v3_qa": [current["packet_1600_v3"]],
        "document_sync_preflight_v2_qa": [current["document_sync_preflight_v2"]],
        "gap_v2_method_hold_qa": [current["gap_v2"]],
        "gap_v2_clean_qa": [current["gap_v2"]],
        "matrix_v7_qa": [current["matrix_v7"]],
        "summary_preflight_qa": [current["summary_preflight"]],
        "finalizer_qa": [current["finalizer_author_freeze"]],
    }
    for role, subjects in qa_subjects.items():
        _write_json(files[role], _qa(verdicts[role], subjects))
        current[role] = _record(files[role], project)

    pack_refs = [
        current[role]
        for role in (
            "final_metrics",
            "final_redline",
            "packet_1600_v3",
            "packet_1600_v3_qa",
            "matrix_v7",
            "matrix_v7_qa",
            "gap_v2",
            "gap_v2_method_hold_qa",
            "gap_v2_clean_qa",
        )
    ]
    _write_json(
        files["final_pack"],
        {
            "exact_refs": [_node(record) for record in pack_refs],
            "admission": {"execution_gpu_pixel_queue_admission": 0},
            "completion": {"goal_complete": False},
        },
    )
    current["final_pack"] = _record(files["final_pack"], project)
    _write_json(
        files["final_artifacts_qa"],
        _qa(
            verdicts["final_artifacts_qa"],
            [current["final_metrics"], current["final_redline"], current["final_pack"]],
        ),
    )
    current["final_artifacts_qa"] = _record(files["final_artifacts_qa"], project)

    doc_subjects = [current["pre2000_pointer"]]
    for spec in packet_builder.DOC_SPECS:
        doc_subjects.extend(
            [
                _record(files[spec["snapshot_role"]], project),
                _record(files[spec["postlive_role"]], project),
            ]
        )
    _write_json(files["document_sync_result_qa"], _qa(verdicts["document_sync_result_qa"], doc_subjects))
    current["document_sync_result_qa"] = _record(files["document_sync_result_qa"], project)

    terminal_status = "SUCCEEDED_FINALIZED_CONTROL_PLANE_REPORTING_ONLY_ZERO_ADMISSION"
    _write_json(
        files["finalizer_terminal"],
        {
            "status": terminal_status,
            "execution_gpu_pixel_queue_admission": 0,
            "result": {
                role: _node(current[role]) for role in ("final_metrics", "final_redline", "final_pack")
            },
        },
    )
    current["finalizer_terminal"] = _record(files["finalizer_terminal"], project)

    refs = {
        role: packet_builder.ExplicitRef(role, files[role], current[role].bytes, current[role].sha256)
        for role in packet_builder.ROLE_ORDER
    }
    return paths, refs, verdicts, terminal_status


def test_success_rebinds_four_stale_paths_and_publishes_0440(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    result = packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert result["canonical_reference_count"] == 128
    assert result["execution_gpu_pixel_queue_admission"] == 0
    assert (paths.output.stat().st_mode & 0o777) == 0o440
    packet = json.loads(paths.output.read_text())
    rows = {row["id"]: row for row in packet["canonical_references"]}
    for spec in packet_builder.DOC_SPECS:
        assert rows[spec["stale_ref_id"]]["path"] == str(refs[spec["snapshot_role"]].path)
        assert rows[spec["stale_ref_id"]]["inherited_1600_v3_capture"]["state"].startswith("PATH_STALE")
        assert rows[spec["new_ref_id"]]["path"] == str(refs[spec["postlive_role"]].path)
    assert packet["checkpoint_2000_post_sync_binding"]["decision_group_count"] == 3
    assert packet["checkpoint_2000_post_sync_binding"]["execution_gpu_pixel_queue_admission"] == 0


def test_missing_role_and_path_alias_are_rejected() -> None:
    rows = [[role, f"/tmp/{role}", "1", "0" * 64] for role in packet_builder.ROLE_ORDER]
    with pytest.raises(packet_builder.PacketBuildError, match="missing explicit ref roles"):
        packet_builder._parse_reference_rows(rows[:-1])
    rows[-1][1] = rows[-2][1]
    with pytest.raises(packet_builder.PacketBuildError, match="path alias"):
        packet_builder._parse_reference_rows(rows)


def test_sha_mismatch_fails_before_publication(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    refs["matrix_v7"] = replace(refs["matrix_v7"], sha256="0" * 64)
    with pytest.raises(packet_builder.PacketBuildError, match="matrix_v7 exact same-FD read failed"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_postlive_must_be_strict_snapshot_prefix(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    role = "owner_postlive"
    _write(refs[role].path, b"not-the-snapshot\n")
    changed = _record(refs[role].path, paths.project_root)
    refs[role] = packet_builder.ExplicitRef(role, refs[role].path, changed.bytes, changed.sha256)
    with pytest.raises(packet_builder.PacketBuildError, match="not a strict append"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_qa_verdict_abbreviation_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    verdicts["final_artifacts_qa"] = "PASS"
    with pytest.raises(packet_builder.PacketBuildError, match="requires an exact full PASS_ QA verdict"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_dynamic_final_artifacts_qa_hold_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    role = "final_artifacts_qa"
    hold = "HOLD_P1_SYNTHETIC_FINAL_ARTIFACTS_QA"
    payload = json.loads(refs[role].path.read_text())
    payload["overall_verdict"]["verdict"] = hold
    _write_json(refs[role].path, payload)
    changed = _record(refs[role].path, paths.project_root)
    refs[role] = packet_builder.ExplicitRef(role, refs[role].path, changed.bytes, changed.sha256)
    verdicts[role] = hold
    with pytest.raises(packet_builder.PacketBuildError, match="requires an exact full PASS_ QA verdict"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_dynamic_document_sync_result_qa_hold_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    role = "document_sync_result_qa"
    hold = "HOLD_P1_SYNTHETIC_DOCUMENT_SYNC_QA"
    payload = json.loads(refs[role].path.read_text())
    payload["overall_verdict"]["verdict"] = hold
    _write_json(refs[role].path, payload)
    changed = _record(refs[role].path, paths.project_root)
    refs[role] = packet_builder.ExplicitRef(role, refs[role].path, changed.bytes, changed.sha256)
    verdicts[role] = hold
    with pytest.raises(packet_builder.PacketBuildError, match="requires an exact full PASS_ QA verdict"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_dynamic_final_artifacts_qa_mixed_pass_and_hold_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    role = "final_artifacts_qa"
    payload = json.loads(refs[role].path.read_text())
    payload["status"] = "HOLD_P1_CONFLICTING_TOP_LEVEL_DISPOSITION"
    _write_json(refs[role].path, payload)
    changed = _record(refs[role].path, paths.project_root)
    refs[role] = packet_builder.ExplicitRef(role, refs[role].path, changed.bytes, changed.sha256)
    with pytest.raises(packet_builder.PacketBuildError, match="conflicting top-level QA dispositions"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_dynamic_document_sync_qa_mixed_pass_and_hold_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    role = "document_sync_result_qa"
    payload = json.loads(refs[role].path.read_text())
    payload["status"] = "HOLD_P1_CONFLICTING_TOP_LEVEL_DISPOSITION"
    _write_json(refs[role].path, payload)
    changed = _record(refs[role].path, paths.project_root)
    refs[role] = packet_builder.ExplicitRef(role, refs[role].path, changed.bytes, changed.sha256)
    with pytest.raises(packet_builder.PacketBuildError, match="conflicting top-level QA dispositions"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_failed_finalizer_terminal_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, _ = _fixture(tmp_path)
    failed = "FAILED_SYNTHETIC_FINALIZER_ZERO_ADMISSION"
    payload = json.loads(refs["finalizer_terminal"].path.read_text())
    payload["status"] = failed
    _write_json(refs["finalizer_terminal"].path, payload)
    changed = _record(refs["finalizer_terminal"].path, paths.project_root)
    refs["finalizer_terminal"] = packet_builder.ExplicitRef(
        "finalizer_terminal", refs["finalizer_terminal"].path, changed.bytes, changed.sha256
    )
    with pytest.raises(packet_builder.PacketBuildError, match="not the exact successful terminal"):
        packet_builder.build(refs, verdicts, failed, paths=paths)
    assert not paths.output.exists()


def test_hardlink_alias_between_runtime_roles_is_rejected(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    metrics = refs["final_metrics"].path
    redline = refs["final_redline"].path
    redline.unlink()
    os.link(metrics, redline)
    changed = _record(redline, paths.project_root)
    refs["final_redline"] = packet_builder.ExplicitRef(
        "final_redline", redline, changed.bytes, changed.sha256
    )
    assert (os.stat(metrics).st_dev, os.stat(metrics).st_ino) == (
        os.stat(redline).st_dev,
        os.stat(redline).st_ino,
    )
    with pytest.raises(packet_builder.PacketBuildError, match="file identity alias"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert not paths.output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    paths, refs, verdicts, terminal = _fixture(tmp_path)
    first = packet_builder.build(refs, verdicts, terminal, paths=paths)
    before = paths.output.read_bytes()
    with pytest.raises(packet_builder.PacketBuildError, match="exclusive post-2000 publication failed"):
        packet_builder.build(refs, verdicts, terminal, paths=paths)
    assert paths.output.read_bytes() == before
    assert first["publication"]["sha256"] == hashlib.sha256(before).hexdigest()


def test_source_has_no_discovery_media_gpu_or_process_execution() -> None:
    source = (PROJECT_ROOT / "tools/build_a_packet_2000_postsync.py").read_text()
    forbidden = (
        "os.walk(",
        "os.listdir(",
        "os.scandir(",
        ".rglob(",
        "glob.glob(",
        "subprocess.",
        "os.kill(",
        "nvidia-smi",
        "cv2.",
        "ffmpeg",
    )
    assert [token for token in forbidden if token in source] == []
