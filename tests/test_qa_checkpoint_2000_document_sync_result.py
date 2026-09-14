from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import qa_checkpoint_2000_document_sync_result as subject  # noqa: E402


FINAL_QA_PASS = (
    "PASS_CHECKPOINT_2000_FINAL_CONTROL_PLANE_ARTIFACTS_EXACT_CUTOFF_"
    "DUAL_SCOPE_GOAL_CONTRADICTED_ZERO_ADMISSION"
)
WINDOW = {
    "start": "2026-08-29T16:00:00+08:00",
    "cutoff": "2026-08-29T19:55:00+08:00",
}
TARGET = "2026-08-29T20:00:00+08:00"
COUNTS = {"PROVEN": 3, "CONTRADICTED": 44, "INCOMPLETE": 84, "MISSING": 0}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, payload: bytes, *, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _write_json(path: Path, value: Any, *, mode: int = 0o640) -> None:
    _write(path, _json_bytes(value), mode=mode)


def _file_ref(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _explicit(role: str, path: Path) -> subject.ExplicitRef:
    row = _file_ref(path)
    return subject.ExplicitRef(role, path, row["bytes"], row["sha256"])


def _qa_expected(role: str) -> str:
    value = next(spec.qa_verdict for spec in subject.STATIC_SPECS if spec.role == role)
    assert value is not None
    return value


def _static_fixture(root: Path) -> tuple[subject.StaticSpec, ...]:
    base = root / "static"
    payloads: dict[str, Any] = {
        "document_sync_preflight_v2": {
            "status": "CPU11_FULL_20_MINUTE_ENVELOPE_READY_NOT_EXECUTED_ZERO_ADMISSION",
            "cpu11_single_wall_timer_contract": {
                "clock": "CLOCK_MONOTONIC",
                "budget_seconds": 1200,
                "start_before_step": "PRE_2000_FOUR_DOCUMENT_SNAPSHOT_CAPTURE",
            },
            "document_append_contract": {
                "owner_numbered_items_exact": 4,
                "historical_text_rewrite_forbidden": True,
                "self_hash_cycle_forbidden": True,
            },
        },
        "gap_v2": {
            "schema_version": "autonomous-20h-final-goal-completion-gap-audit-pre-2000-t0-v2",
            "status": "CONTRADICTED",
            "requirement_count": 131,
            "leaf_state_counts": COUNTS,
            "completion_conclusion": {"goal_complete": False},
            "execution_gpu_pixel_queue_admission": 0,
        },
        "matrix_v7": {
            "readiness_invariants_after_correction": {
                "grap_a_cap_002_all_columns": "NOT_READY",
                "grap_a_cap_004_all_columns": "NOT_READY",
                "future_coverage_retained_a_class_p0_count": 2,
                "future_coverage_retained_p1_count": 2,
                "command_admission": 0,
                "gpu_admission": 0,
                "pixel_or_selector_admission": 0,
            }
        },
        "packet_1600_v3": {
            "schema_version": "autonomous-20h-a-class-owner-decision-queue-1600-v3-b-reporting-fix",
            "status": "OWNER_DECISIONS_REQUIRED",
            "decision_groups": [{"id": 1}, {"id": 2}, {"id": 3}],
            "checkpoint_1600_post_sync_binding": {
                "supervisor_successor_activation": "HOLD_ROUND3_EXHAUSTED_BQ05_DEGRADED_TO_A",
                "execution_gpu_pixel_queue_admission": 0,
            },
            "eligible68_future_mask_coverage_contract_hold_delta": {
                "new_a_class_p0_count": 2,
                "new_p1_count": 2,
            },
        },
        "summary_preflight": {
            "actions_performed": {
                "final_summary_created": False,
                "execution_gpu_pixel_queue_admission": 0,
            },
            "slots": [
                {
                    "role": role,
                    "status": "MISSING_NOT_READY_AT_PREFLIGHT",
                    "absolute_path": "MISSING",
                    "bytes": None,
                    "sha256": None,
                }
                for role in sorted(subject.DELIVERY_SLOT_ROLES)
            ],
        },
    }
    subject_paths: dict[str, Path] = {}
    for role, payload in payloads.items():
        path = base / f"{role}.json"
        _write_json(path, payload)
        subject_paths[role] = path
    signed_owner = base / "signed_owner.md"
    shared_io = base / "immutable_artifact_io.py"
    _write(signed_owner, b"signed owner payload\n")
    _write(shared_io, b"# pinned synthetic shared IO\n")
    subject_paths["signed_owner"] = signed_owner
    subject_paths["shared_io"] = shared_io

    qa_subjects = {
        "document_sync_preflight_v2_qa": "document_sync_preflight_v2",
        "gap_v2_clean_qa": "gap_v2",
        "matrix_v7_qa": "matrix_v7",
        "packet_1600_v3_qa": "packet_1600_v3",
        "summary_preflight_qa": "summary_preflight",
    }
    for qa_role, source_role in qa_subjects.items():
        path = base / f"{qa_role}.json"
        _write_json(
            path,
            {
                "overall_verdict": {"verdict": _qa_expected(qa_role)},
                "subject": _file_ref(subject_paths[source_role]),
                "execution_gpu_pixel_queue_admission": 0,
            },
        )
        subject_paths[qa_role] = path

    specs: list[subject.StaticSpec] = []
    for role in (
        "document_sync_preflight_v2",
        "document_sync_preflight_v2_qa",
        "signed_owner",
        "gap_v2",
        "gap_v2_clean_qa",
        "matrix_v7",
        "matrix_v7_qa",
        "packet_1600_v3",
        "packet_1600_v3_qa",
        "summary_preflight",
        "summary_preflight_qa",
        "shared_io",
    ):
        path = subject_paths[role]
        ref = _file_ref(path)
        qa_subject = qa_subjects.get(role)
        specs.append(
            subject.StaticSpec(
                role,
                str(path.relative_to(root)),
                ref["bytes"],
                ref["sha256"],
                _qa_expected(role) if qa_subject else None,
                qa_subject,
            )
        )
    return tuple(specs)


def _machine_facts() -> dict[str, Any]:
    return {
        "window": {"start": WINDOW["start"], "cutoff": WINDOW["cutoff"], "target": TARGET},
        "delivery_slots": "10_OF_10_MISSING_NOT_READY",
        "gap": {
            "requirement_count": 131,
            "leaf_state_counts": COUNTS,
            "goal_complete": False,
        },
        "owner_a_decision_group_count": 3,
        "bq05": "HOLD_ROUND3_EXHAUSTED_BQ05_DEGRADED_TO_A",
        "future_coverage": {"a_class_p0_count": 2, "p1_count": 2},
        "scope_separation": {
            "named_ledger_new_a_class_event": False,
            "canonical_nonledger_a_class_evidence_present": True,
            "scopes_collapsed": False,
        },
        "execution_gpu_pixel_queue_admission": 0,
    }


@dataclass
class Case:
    paths: subject.BuilderPaths
    refs: dict[str, subject.ExplicitRef]
    static_specs: tuple[subject.StaticSpec, ...]
    timer_start: int
    observed_ns: int

    def refresh(self, role: str) -> None:
        self.refs[role] = _explicit(role, self.refs[role].path)

    def load_json(self, role: str) -> dict[str, Any]:
        return json.loads(self.refs[role].path.read_text())

    def save_json(self, role: str, value: Any) -> None:
        _write_json(self.refs[role].path, value, mode=stat_mode(self.refs[role].path))
        self.refresh(role)

    def run(self) -> dict[str, Any]:
        return subject.build(
            self.refs,
            FINAL_QA_PASS,
            self.timer_start,
            paths=self.paths,
            static_specs=self.static_specs,
            observed_monotonic_ns=self.observed_ns,
            require_loaded_shared_io_match=False,
        )


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@pytest.fixture
def case(tmp_path: Path) -> Case:
    root = tmp_path / "project"
    run = root / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829"
    checkpoints = run / "checkpoints"
    audits = checkpoints / "audits"
    (root / "task").mkdir(parents=True)
    audits.mkdir(parents=True)
    paths = subject.BuilderPaths(
        root,
        run,
        checkpoints,
        audits,
        audits / subject.OUTPUT_PATH.name,
    )
    static_specs = _static_fixture(root)

    identical_snapshot = b"# Immutable historical payload\n"
    runtime_paths: dict[str, Path] = {}
    for spec in subject.DOCUMENT_SPECS:
        digest = hashlib.sha256(identical_snapshot).hexdigest()
        path = checkpoints / f"{spec.snapshot_prefix}{digest}.md"
        _write(path, identical_snapshot, mode=0o444)
        runtime_paths[spec.snapshot_role] = path

    metrics_path = checkpoints / "CHECKPOINT_2000_UTILIZATION_CUTOFF_1955_T0_V1.json"
    redline_path = checkpoints / "RED_LINE_INCREMENT_1600_1955_T0_V1.json"
    pack_path = checkpoints / "CHECKPOINT_2000_EVIDENCE_PACK_T0_V1.json"
    final_qa_path = audits / "INDEPENDENT_QA_CHECKPOINT_2000_FINAL_CONTROL_PLANE_ARTIFACTS_T0_V1.json"
    _write_json(
        metrics_path,
        {
            "schema_version": "autonomous-20h-checkpoint-utilization-t0-v1",
            "requested_window": WINDOW,
            "checkpoint_target": TARGET,
            "execution_gpu_pixel_queue_admission": 0,
        },
    )
    retained_a = {
        "present": True,
        "owner_decision_group_ids": ["A-DECISION-01", "A-DECISION-02", "A-DECISION-03"],
        "bq05_finding_id": "BQ05",
        "bq05_state": "HOLD_ROUND3_EXHAUSTED_BQ05_DEGRADED_TO_A",
        "coverage_a_p0_count": 2,
        "coverage_a_p0_ids": ["A-COVERAGE-01", "A-COVERAGE-02"],
        "coverage_p1_count": 2,
        "coverage_p1_ids": ["P1-COVERAGE-01", "P1-COVERAGE-02"],
    }
    scope_separation = {
        "named_ledger_scope": {"new_a_class_event": False},
        "canonical_evidence_scope": {
            "retained_a_at_window_start": retained_a,
            "a_state_at_cutoff": retained_a,
            "delivery_matrix": {
                "readiness": {
                    "grap_a_cap_002_all_columns": "NOT_READY",
                    "grap_a_cap_004_all_columns": "NOT_READY",
                }
            },
        },
    }
    _write_json(
        redline_path,
        {
            "schema_version": "autonomous-20h-red-line-increment-t0-v1",
            "audit_window": WINDOW,
            "checkpoint_target": TARGET,
            "scope_separation": scope_separation,
            "execution_gpu_pixel_queue_admission": 0,
        },
    )
    _write_json(
        pack_path,
        {
            "schema_version": "checkpoint-2000-control-plane-evidence-pack-t0-v1",
            "status": (
                "CHECKPOINT_REPORTING_ONLY_GOAL_CONTRADICTED_"
                "OWNER_DECISIONS_REQUIRED_ZERO_ADMISSION"
            ),
            "requested_window": WINDOW,
            "checkpoint_target": TARGET,
            "exact_refs": {
                "final_metrics": _file_ref(metrics_path),
                "final_redline": _file_ref(redline_path),
            },
            "completion": {
                "goal_complete": False,
                "evidence_state": "CONTRADICTED",
                "eta": "UNMEASURED",
                "requirement_count": 131,
                "leaf_state_counts": COUNTS,
            },
            "scope_separation": scope_separation,
            "execution_gpu_pixel_queue_admission": 0,
        },
    )
    _write_json(
        final_qa_path,
        {
            "overall_verdict": {"verdict": FINAL_QA_PASS},
            "subject": {
                "final_metrics": _file_ref(metrics_path),
                "final_redline": _file_ref(redline_path),
                "final_pack": _file_ref(pack_path),
            },
            "terminal_subject": {
                "status": subject.TERMINAL_STATUS,
                "execution_gpu_pixel_queue_admission": 0,
                "result": {
                    "final_metrics": _file_ref(metrics_path),
                    "final_redline": _file_ref(redline_path),
                    "final_pack": _file_ref(pack_path),
                },
            },
            "execution_gpu_pixel_queue_admission": 0,
        },
    )
    runtime_paths.update(
        {
            "final_metrics": metrics_path,
            "final_redline": redline_path,
            "final_pack": pack_path,
            "final_artifacts_qa": final_qa_path,
        }
    )

    runtime_observed: dict[str, subject.Observed] = {}
    for role in ("final_metrics", "final_redline", "final_pack", "final_artifacts_qa"):
        path = runtime_paths[role]
        raw = path.read_bytes()
        info = path.stat()
        record = subject.strict_io.VerifiedBytes(
            str(path), raw, len(raw), hashlib.sha256(raw).hexdigest(), info.st_dev, info.st_ino
        )
        runtime_observed[role] = subject.Observed(record, info.st_mode & 0o777)
    facts = subject._fact_contract(_machine_facts(), runtime_observed, FINAL_QA_PASS)
    marker = subject.canonical_fact_marker(facts)
    suffixes = {
        "owner_authorization": (
            "\n### 6.5 20:00 checkpoint\n"
            "1. Final facts remain reporting-only.\n"
            "2. Ten delivery slots remain missing/not ready.\n"
            "3. Three owner A groups and BQ05 remain held.\n"
            "4. Execution, GPU, pixels, and queues remain at zero admission.\n"
            f"{marker}\n"
        ),
        "task05_pipeline_map": f"\n## 17. 20:00 checkpoint\nExact facts follow.\n{marker}\n",
        "task06_history": f"\n### 8A.16 20:00 checkpoint\nExact facts follow.\n{marker}\n",
        "task15_lr_diagnosis": f"\n### E-15-14 20:00 checkpoint\nExact facts follow.\n{marker}\n",
    }
    for spec in subject.DOCUMENT_SPECS:
        path = root / spec.live_relative
        _write(path, identical_snapshot + suffixes[spec.logical].encode())
        runtime_paths[spec.postlive_role] = path

    timer_start = 8_000_000_000_000
    pointer_path = checkpoints / subject.POINTER_NAME
    documents = []
    for spec in subject.DOCUMENT_SPECS:
        snapshot_ref = _file_ref(runtime_paths[spec.snapshot_role])
        live_path = runtime_paths[spec.postlive_role]
        documents.append(
            {
                "logical_document": spec.logical,
                "historical_same_path_capture": {
                    "path": str(live_path),
                    "bytes": snapshot_ref["bytes"],
                    "sha256": snapshot_ref["sha256"],
                    "status_after_authorized_2000_append": "PATH_STALE",
                },
                "immutable_snapshot": {
                    **snapshot_ref,
                    "mode": "0444",
                    "relation": "EXACT_FULL_PAYLOAD_COPY_BEFORE_2000_APPEND",
                },
            }
        )
    _write_json(
        pointer_path,
        {
            "schema_version": "OWNER_AND_TASK_DOC_PRE_2000_PAYLOAD_PRESERVATION_T0_V1",
            "created_at": "2026-08-29T19:40:00+08:00",
            "documents": documents,
            "cpu11_single_wall_timer": {
                "clock": "CLOCK_MONOTONIC",
                "budget_seconds": 1200,
                "start_before_step": "PRE_2000_FOUR_DOCUMENT_SNAPSHOT_CAPTURE",
                "start_monotonic_ns": timer_start,
                "timer_restart_forbidden": True,
            },
            "execution_effect": "NONE_CONTROL_PLANE_SNAPSHOT_ONLY",
            "retroactive_rewrite": False,
        },
        mode=0o444,
    )
    runtime_paths["pre2000_pointer"] = pointer_path
    refs = {role: _explicit(role, runtime_paths[role]) for role in subject.ROLE_ORDER}
    return Case(paths, refs, static_specs, timer_start, timer_start + 30_000_000_000)


def test_success_writes_one_exact_pass_artifact_mode_0440(case: Case) -> None:
    result = case.run()
    assert result["verdict"] == subject.PASS_VERDICT
    output = json.loads(case.paths.output.read_text())
    assert output["overall_verdict"] == {"verdict": subject.PASS_VERDICT}
    assert output["findings"]["new_a_p0_count"] == 0
    assert output["findings"]["new_p1_count"] == 0
    assert output["admission"]["execution_gpu_pixel_queue_admission"] == 0
    timer = output["cpu11_single_wall_timer"]
    assert "elapsed_monotonic_ns_at_qa_publication" not in timer
    assert timer["final_prepublication_gate"]["elapsed_monotonic_ns"] == 30_000_000_000
    assert result["reported_postpublication_elapsed_monotonic_ns"] == 30_000_000_000
    assert result["within_budget_at_postpublication_observation"] is True
    assert stat_mode(case.paths.output) == 0o440


def test_rejects_prefix_drift(case: Case) -> None:
    path = case.refs["owner_postlive"].path
    payload = path.read_bytes()
    _write(path, b"X" + payload[1:])
    case.refresh("owner_postlive")
    with pytest.raises(subject.ResultQAError, match="strict byte-prefix"):
        case.run()


def test_rejects_wrong_section(case: Case) -> None:
    path = case.refs["task05_postlive"].path
    _write(path, path.read_bytes().replace(b"## 17.", b"## 18."))
    case.refresh("task05_postlive")
    with pytest.raises(subject.ResultQAError, match="task05.*exactly once"):
        case.run()


def test_rejects_owner_item_count(case: Case) -> None:
    path = case.refs["owner_postlive"].path
    _write(path, path.read_bytes().replace(b"4. Execution", b"5. Execution"))
    case.refresh("owner_postlive")
    with pytest.raises(subject.ResultQAError, match="exactly ordered"):
        case.run()


def test_rejects_fact_marker_mismatch(case: Case) -> None:
    path = case.refs["task06_postlive"].path
    _write(path, path.read_bytes().replace(b'"requirement_count":131', b'"requirement_count":130'))
    case.refresh("task06_postlive")
    with pytest.raises(subject.ResultQAError, match="fact marker mismatch"):
        case.run()


def test_rejects_timer_over_1200_seconds(case: Case) -> None:
    case.observed_ns = case.timer_start + 1_200_000_000_001
    with pytest.raises(subject.ResultQAError, match="exceeded 1200"):
        case.run()


def test_rejects_1199_9_to_1201_crossing_before_publication(case: Case) -> None:
    samples = iter(
        (
            case.timer_start + 1_199_900_000_000,
            case.timer_start + 1_201_000_000_000,
        )
    )

    def monotonic_ns() -> int:
        return next(samples)

    with pytest.raises(subject.ResultQAError, match="final prepublication.*exceeded 1200"):
        subject.build(
            case.refs,
            FINAL_QA_PASS,
            case.timer_start,
            paths=case.paths,
            static_specs=case.static_specs,
            monotonic_ns=monotonic_ns,
            require_loaded_shared_io_match=False,
        )
    assert not case.paths.output.exists()


def test_rejects_runtime_symlink(case: Case) -> None:
    target = case.refs["owner_snapshot"].path
    link = case.refs["task05_snapshot"].path
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(subject.ResultQAError, match="O_NOFOLLOW read failed"):
        case.run()


def test_rejects_runtime_hardlink_alias(case: Case) -> None:
    target = case.refs["owner_snapshot"].path
    alias = case.refs["task05_snapshot"].path
    alias.unlink()
    os.link(target, alias)
    with pytest.raises(subject.ResultQAError, match="hardlink alias"):
        case.run()


def test_rejects_snapshot_mode_not_0444(case: Case) -> None:
    case.refs["task15_snapshot"].path.chmod(0o440)
    with pytest.raises(subject.ResultQAError, match="mode must be 0444"):
        case.run()


def test_rejects_explicit_ref_digest_drift(case: Case) -> None:
    ref = case.refs["final_metrics"]
    case.refs["final_metrics"] = subject.ExplicitRef(
        ref.role,
        ref.path,
        ref.bytes,
        "0" * 64,
    )
    with pytest.raises(subject.ResultQAError, match="same-FD O_NOFOLLOW read failed"):
        case.run()


def test_rejects_dynamic_nonpass(case: Case) -> None:
    payload = case.load_json("final_artifacts_qa")
    payload["overall_verdict"]["verdict"] = "HOLD_NOT_PASS"
    case.save_json("final_artifacts_qa", payload)
    with pytest.raises(subject.ResultQAError, match="lacks the exact expected"):
        case.run()


def test_rejects_dynamic_conflicting_disposition(case: Case) -> None:
    payload = case.load_json("final_artifacts_qa")
    payload["disposition"] = "HOLD_CONFLICT"
    case.save_json("final_artifacts_qa", payload)
    with pytest.raises(subject.ResultQAError, match="conflicting or abbreviated"):
        case.run()


def test_rejects_top_level_terminal_as_conflicting_qa_status(case: Case) -> None:
    payload = case.load_json("final_artifacts_qa")
    payload["status"] = subject.TERMINAL_STATUS
    case.save_json("final_artifacts_qa", payload)
    with pytest.raises(subject.ResultQAError, match="conflicting or abbreviated"):
        case.run()


def test_rejects_nonzero_admission(case: Case) -> None:
    payload = case.load_json("final_artifacts_qa")
    payload["terminal_subject"]["execution_gpu_pixel_queue_admission"] = 1
    case.save_json("final_artifacts_qa", payload)
    with pytest.raises(subject.ResultQAError, match="terminal lacks exact zero admission"):
        case.run()


def test_rejects_existing_output(case: Case) -> None:
    _write_json(case.paths.output, {"preexisting": True}, mode=0o440)
    with pytest.raises(subject.ResultQAError, match="exclusive QA publication failed"):
        case.run()


def test_parser_rejects_missing_runtime_role(case: Case) -> None:
    rows = [
        [role, str(ref.path), str(ref.bytes), ref.sha256]
        for role, ref in case.refs.items()
        if role != "final_pack"
    ]
    with pytest.raises(subject.ResultQAError, match="exactly the thirteen"):
        subject._parse_reference_rows(rows)
