#!/usr/bin/env python3
"""Freeze Robot15h evidence scopes and perform the final release audit.

This module intentionally has no algorithm implementation.  It reads the
SHA-bound terminal results of the current task's DAG dependencies and keeps
three facts separate throughout publication:

``executed``
    The algorithm actually ran for a session/scope.
``exported``
    A substantive algorithm artifact was published.
``quality_admitted``
    The artifact passed the quality gate for the narrowly named scope.

A structurally successful task, an eligibility ledger, or a blocker-only
terminal never implies algorithm or quality success.  The final-audit mode
also requires explicit, SHA-bound runtime, video, test, and commit evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet


ROOT = Path(__file__).resolve().parents[3]
WINDOW_CLOCK = ROOT / "_run/current/0915_robot15h_v1/WINDOW_CLOCK.json"
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INDEX = ROOT / "tasks/current/INDEX.json"
TASK_IDS = (
    "0915_robot15h_release_candidate_v1",
    "0915_robot15h_window_release_audit_v1",
)
TERMINAL = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
    "BUDGET_EXHAUSTED",
}
EXPECTED_OUTPUTS = {
    TASK_IDS[0]: {"CAPABILITY_MATRIX.json", "DEADLINE_AUDIT.json"},
    TASK_IDS[1]: {
        "CAPABILITY_MATRIX.json", "CAMPAIGN_COUNTS.json", "DEADLINE_AUDIT.json",
        "FINAL_AUDIT.json", "REFERENCE_AUDIT.json",
    },
}
SIDECAR_SCHEMAS = {
    "runtime": "0915-robot15h-final-runtime-evidence-v1",
    "video": "0915-robot15h-final-video-evidence-v1",
    "tests": "0915-robot15h-final-test-evidence-v1",
    "commit": "0915-robot15h-final-commit-evidence-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"regular artifact required: {candidate}")
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def verify_ref(value: Mapping[str, Any]) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(value):
        raise RuntimeError("artifact reference requires path/bytes/sha256")
    path = Path(str(value["path"])).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"regular artifact required: {path}")
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != str(value["sha256"]):
        raise RuntimeError(f"artifact reference drift: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"timezone-aware timestamp required: {value}")
    return parsed


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def infer_task(session_id: object) -> str:
    value = str(session_id or "")
    if value.startswith("play_cards_"):
        return "playing_cards"
    if value.startswith("get_potato_chips_"):
        return "potato_chips"
    return "campaign"


def capability_scope(task_id: str) -> tuple[str, str]:
    if "foundationstereo" in task_id:
        return "foundationstereo", "DEVELOPMENT_DEPTH_VISUAL_OBJECT6D_INPUT_ONLY"
    if "sam31_temporal_identity" in task_id:
        return "sam31_hand_temporal", "DIRECT_ANCHORED_HAND_INSTANCE_PROXY_ONLY"
    if "sam31_task_object" in task_id:
        return "sam31_task_object", "VISIBLE_TASK_OBJECT_MASK_PROXY_ONLY"
    if "sam31_waves" in task_id:
        return "sam31_waves", "RUNNER_DECLARED_SESSION_SCOPE"
    if "hawor" in task_id:
        return "hawor", "DEVELOPMENT_HUMAN_PRIOR_ONLY"
    if "kai22_r0" in task_id:
        return "kai22_r0", "WRIST_LOCAL_Q22_R0_ONLY"
    if "geometry_object6d" in task_id:
        return "object6d", "VISIBLE_SURFACE_OBJECT6D_ONLY"
    if "interaction_occlusion" in task_id:
        return "interaction_occlusion", "OFFLINE_RECONSTRUCTION_ONLY"
    if "contact_dual" in task_id:
        return "contact_dual", "OBSERVED_OR_HYPOTHESIS_SCOPE_AS_RECORDED"
    if "robot_relative" in task_id:
        return "kai22_r1", "R1_E_OR_R1_H_AS_RECORDED"
    if "robot_virtual_arm" in task_id:
        return "r2_virtual_arm", "FIXED_VIRTUAL_BASE_IK_ONLY"
    if "robot_waves" in task_id:
        return "robot_waves", "RUNNER_DECLARED_ROBOT_LAYER"
    if "scale_cause" in task_id:
        return "human_stereo_alignment", "DIAGNOSIS_NOT_CALIBRATION"
    return "governance_or_inventory", "NO_ALGORITHM_AUTHORITY"


# Later entries supersede earlier attempts only inside the same cohort/wave.
# Different waves remain jointly authoritative campaign evidence.  This is
# deliberately explicit: filesystem mtime and a top-level PASSED are not
# authority selection mechanisms, and W1 is not a retry of W0.
SOURCE_PRIORITY = {
    "hawor": (
        (
            "0915_robot15h_hawor_wave0_v1",
            "0915_robot15h_hawor_wave0_recovery_v1",
        ),
        ("0915_robot15h_hawor_waves_v1",),
    ),
    "foundationstereo": (
        (
            "0915_robot15h_foundationstereo_wave0_v1",
            "0915_robot15h_foundationstereo_wave0_recovery_v1",
        ),
        ("0915_robot15h_foundationstereo_waves_v1",),
    ),
    "sam31_hand_temporal": ((
        "0915_robot15h_sam31_temporal_identity_v1",
        "0915_robot15h_sam31_temporal_identity_recovery_v1",
        "0915_robot15h_sam31_temporal_identity_quality_v2",
    ),),
    "sam31_task_object": ((
        "0915_robot15h_sam31_task_object_wave0_v1",
        "0915_robot15h_sam31_task_object_wave0_recovery_v1",
    ),),
}


def select_authoritative_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence:
        capability, _ = capability_scope(str(item["task_id"]))
        grouped[capability].append(item)
    selected: list[Mapping[str, Any]] = []
    for capability in sorted(grouped):
        choices = grouped[capability]
        chains = SOURCE_PRIORITY.get(capability)
        if not chains:
            selected.extend(choices)
            continue
        governed_ids = {task_id for chain in chains for task_id in chain}
        for chain in chains:
            rank = {task_id: index for index, task_id in enumerate(chain)}
            cohort = [item for item in choices if str(item["task_id"]) in rank]
            if cohort:
                selected.append(max(cohort, key=lambda item: rank[str(item["task_id"])]))
        selected.extend(item for item in choices if str(item["task_id"]) not in governed_ids)
    return selected


def _session_rows(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    batch = item.get("batch")
    if isinstance(batch, Mapping):
        for key in ("sessions", "results", "rows"):
            rows = batch.get(key)
            if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
                return list(rows)
    return [item["result"]]


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def assess_row(task_id: str, row: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    capability, default_scope = capability_scope(task_id)
    capability = str(row.get("capability_id") or capability)
    scope = str(row.get("evidence_scope") or row.get("authorized_scope") or default_scope)
    status = str(row.get("status") or parent.get("status") or "UNKNOWN")
    task = str(row.get("task") or infer_task(row.get("session_id")))

    explicit_execution = any(row.get(key) is True for key in (
        "execution_completed", "algorithm_attempted", "model_execution_performed", "model_rerun_performed",
        "r0_exported", "r2_exported", "hand_temporal_artifact_exported",
    ))
    has_algorithm_artifact = any(key in row for key in ("npz", "semantic_archive", "states"))
    executed_status = status.startswith(("PASS", "REJECTED_QUALITY", "FAILED_QUALITY", "COMPLETED_DEVELOPMENT"))
    explicitly_not_executed = row.get("algorithm_attempted") is False or row.get("model_execution_performed") is False
    executed = bool(explicit_execution or has_algorithm_artifact or (row.get("session_id") and executed_status))
    if explicitly_not_executed and not explicit_execution and not has_algorithm_artifact:
        executed = False

    if capability == "hawor":
        exported = isinstance(row.get("npz"), Mapping)
        quality_units = 1 if status.startswith("PASS") else 0
    elif capability == "foundationstereo":
        exported = status == "PASSED" and isinstance(row.get("result"), Mapping)
        quality_units = int(exported)
    elif capability == "kai22_r0":
        exported = row.get("r0_exported") is True
        quality_units = int(row.get("r0_quality_admitted") is True)
    elif capability == "sam31_hand_temporal":
        exported = row.get("hand_temporal_artifact_exported") is True
        quality_units = _positive_int(row.get("hand_counts", {}).get("proxy_pass"))
    elif capability == "sam31_task_object":
        counts = row.get("instance_counts", {})
        exported = _positive_int(counts.get("raw_selected")) > 0
        quality_units = _positive_int(counts.get("stable_consumer_allowed"))
        if row.get("object_mask_consumer_allowed") is not True:
            quality_units = 0
    elif capability == "r2_virtual_arm":
        exported = row.get("r2_exported") is True
        quality_units = int(row.get("r2_quality_admitted") is True)
    elif capability == "human_stereo_alignment":
        exported = False
        quality_units = int(
            row.get("metric_translation_authorized") is True or row.get("correction_adopted") is True
        )
    else:
        exported = any(row.get(key) is True for key in (
            "algorithm_artifact_emitted", "object6d_artifact_emitted", "r1_e_candidate_emitted",
            "r1_h_candidate_emitted", "r0_exported", "r2_exported",
        ))
        quality_units = int(any(row.get(key) is True for key in (
            "quality_admitted", "consumer_allowed", "object_mask_consumer_allowed",
            "r0_quality_admitted", "r2_quality_admitted", "r1_e_adopted", "r1_h_adopted",
        )))
        if not quality_units and status.startswith("PASS_") and executed and exported:
            quality_units = 1

    return {
        "capability_id": capability,
        "task": task,
        "scope": scope,
        "session_id": row.get("session_id"),
        "source_status": status,
        "executed": executed,
        "exported": bool(exported),
        "quality_admitted": quality_units > 0,
        "quality_admitted_units": quality_units,
        "first_blocker": row.get("first_blocker") or parent.get("first_blocker"),
    }


def build_capability_matrix(
    evidence: Sequence[Mapping[str, Any]], *, freeze_deadline_met: bool = True,
) -> dict[str, Any]:
    selected = select_authoritative_evidence(evidence)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in selected:
        task_id = str(item["task_id"])
        parent = item["result"]
        terminal_status = str(item["terminal_status"])
        for assessed in (assess_row(task_id, row, parent) for row in _session_rows(item)):
            key = (assessed["capability_id"], assessed["task"], assessed["scope"])
            target = grouped.setdefault(key, {
                "capability_id": key[0],
                "task": key[1],
                "scope": key[2],
                "session_count": 0,
                "executed_count": 0,
                "exported_count": 0,
                "quality_admitted_count": 0,
                "quality_admitted_units": 0,
                "source_status_counts": {},
                "source_task_ids": [],
                "source_terminal_statuses": [],
                "source_results": [],
                "first_blockers": [],
            })
            target["session_count"] += 1
            target["executed_count"] += int(assessed["executed"])
            target["exported_count"] += int(assessed["exported"])
            target["quality_admitted_count"] += int(assessed["quality_admitted"])
            target["quality_admitted_units"] += assessed["quality_admitted_units"]
            source_status = assessed["source_status"]
            target["source_status_counts"][source_status] = target["source_status_counts"].get(source_status, 0) + 1
            if task_id not in target["source_task_ids"]:
                target["source_task_ids"].append(task_id)
            if terminal_status not in target["source_terminal_statuses"]:
                target["source_terminal_statuses"].append(terminal_status)
            result_ref = item["result_ref"]
            if result_ref not in target["source_results"]:
                target["source_results"].append(result_ref)
            blocker = assessed["first_blocker"]
            if blocker and blocker not in target["first_blockers"]:
                target["first_blockers"].append(blocker)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        row = grouped[key]
        row["executed"] = row["executed_count"] > 0
        row["exported"] = row["exported_count"] > 0
        row["quality_admitted"] = row["quality_admitted_count"] > 0
        if row["quality_admitted"]:
            row["decision"] = "QUALITY_SCOPE_ADMITTED"
        elif row["executed"] or row["exported"]:
            row["decision"] = "EXECUTED_NOT_QUALITY_ADMITTED"
        else:
            row["decision"] = "NOT_EXECUTED_OR_BLOCKED"
        row["w1_expansion_allowed"] = bool(row["quality_admitted"] and freeze_deadline_met)
        row["control_eligible"] = False
        row["training_eligible"] = False
        row["physical_deployment_authorized"] = False
        rows.append(row)

    return {
        "schema_version": "0915-robot15h-capability-task-scope-matrix-v1",
        "window_run_id": WINDOW_RUN_ID,
        "authority_selection": "EXPLICIT_SUCCESSOR_PRIORITY_NOT_MTIME_NOT_TOP_LEVEL_PASS",
        "freeze_deadline_met": freeze_deadline_met,
        "rows": rows,
        "counts": {
            "rows": len(rows),
            "executed_rows": sum(row["executed"] for row in rows),
            "exported_rows": sum(row["exported"] for row in rows),
            "quality_admitted_rows": sum(row["quality_admitted"] for row in rows),
            "w1_expansion_allowed_rows": sum(row["w1_expansion_allowed"] for row in rows),
        },
        "claim_limit": (
            "A matrix row preserves the source terminal and scope. Structural task PASS, "
            "eligibility, export, and execution do not imply quality admission. No row grants "
            "control, training, physical deployment, or authority outside its named scope."
        ),
    }


def build_campaign_counts(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Publish fixed-denominator execution counts without conflating terminal and success."""
    results = {str(item["task_id"]): item["result"] for item in evidence}

    def result(task_id: str) -> Mapping[str, Any]:
        value = results.get(task_id)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"campaign count source is absent: {task_id}")
        return value

    def counts(task_id: str) -> Mapping[str, Any]:
        value = result(task_id).get("counts")
        if not isinstance(value, Mapping):
            raise RuntimeError(f"campaign count object is absent: {task_id}")
        return value

    inventory = result("0915_robot15h_window_start_inventory_v1")
    r0 = counts("0915_robot15h_kai22_r0_wave0_v1")
    r1 = counts("0915_robot15h_robot_relative_refinement_v1")
    r2 = counts("0915_robot15h_robot_virtual_arm_v1")
    depth_w0 = counts("0915_robot15h_foundationstereo_wave0_recovery_v1")
    depth_w1 = counts("0915_robot15h_foundationstereo_waves_v1")
    robot_w1 = counts("0915_robot15h_robot_waves_v1")
    inventory_count = _positive_int(inventory.get("session_count"))
    selected_ids = inventory.get("w1_cumulative_session_ids")
    if inventory_count != 220 or not isinstance(selected_ids, list) or len(selected_ids) != 12:
        raise RuntimeError("Robot15h campaign denominator drift")
    if _positive_int(robot_w1.get("total")) != 8:
        raise RuntimeError("Robot W1 denominator drift")
    w1_blocked = _positive_int(robot_w1.get("blocked"))
    r1_sessions = _positive_int(r1.get("sessions_total"))
    summary = {
        "schema_version": "0915-robot15h-campaign-counts-v1",
        "window_run_id": WINDOW_RUN_ID,
        "denominators": {
            "inventory_sessions": inventory_count,
            "frozen_selected_sessions": len(selected_ids),
            "w0_sessions": len(inventory.get("w0_session_ids", [])),
            "w1_additional_sessions": len(selected_ids) - len(inventory.get("w0_session_ids", [])),
            "unselected_not_run": inventory_count - len(selected_ids),
        },
        "foundationstereo": {
            "w0": {
                "attempted": _positive_int(depth_w0.get("total")),
                "passed": _positive_int(depth_w0.get("passed")),
                "rejected_quality": _positive_int(depth_w0.get("rejected_quality")),
                "failed_runtime": _positive_int(depth_w0.get("failed_runtime")),
            },
            "w1": {
                "attempted": _positive_int(depth_w1.get("attempted")),
                "passed": _positive_int(depth_w1.get("passed")),
                "rejected_quality": _positive_int(depth_w1.get("rejected_quality")),
                "failed_runtime": _positive_int(depth_w1.get("failed_runtime")),
                "unrun": _positive_int(depth_w1.get("unrun")),
            },
        },
        "robot_layers": {
            "R0": {
                "denominator": len(selected_ids),
                "attempted": _positive_int(r0.get("attempted")),
                "exported": _positive_int(r0.get("r0_exported")),
                "quality_success": _positive_int(r0.get("r0_success")),
                "quality_rejected": _positive_int(r0.get("r0_rejected")),
                "blocked_upstream": w1_blocked,
                "unrun": 0,
            },
            "R1_E": {
                "denominator": len(selected_ids),
                "attempted": _positive_int(r1.get("r1_e_attempted")),
                "exported": _positive_int(r1.get("r1_e_exported")),
                "quality_success": _positive_int(r1.get("r1_e_adopted")),
                "quality_rejected": 0,
                "blocked_local_or_upstream": _positive_int(
                    r1.get("r1_e_blocked_local_evidence")
                ) + w1_blocked,
                "unrun": 0,
            },
            "R1_H": {
                "denominator": len(selected_ids),
                "attempted": _positive_int(r1.get("r1_h_attempted")),
                "exported": _positive_int(r1.get("r1_h_exported")),
                "quality_success": _positive_int(r1.get("r1_h_adopted")),
                "quality_rejected": _positive_int(r1.get("r1_h_attempted"))
                - _positive_int(r1.get("r1_h_adopted")),
                "blocked_local_or_upstream": r1_sessions
                - _positive_int(r1.get("r1_h_attempted")) + w1_blocked,
                "unrun": 0,
            },
            "R2": {
                "denominator": len(selected_ids),
                "attempted": _positive_int(r2.get("attempted")),
                "exported": _positive_int(r2.get("r2_exported")),
                "quality_success": _positive_int(r2.get("r2_quality_admitted")),
                "quality_rejected": _positive_int(r2.get("r2_quality_rejected")),
                "blocked_upstream": w1_blocked,
                "unrun": 0,
            },
        },
        "interpretation": {
            "all_terminal_is_not_all_success": True,
            "numeric_export_is_not_quality_success": True,
            "r1_h_training_eligible": False,
            "physical_deployment_authorized": False,
        },
    }
    for layer in summary["robot_layers"].values():
        terminal = (
            _positive_int(layer.get("quality_success"))
            + _positive_int(layer.get("quality_rejected"))
            + _positive_int(layer.get("blocked_upstream"))
            + _positive_int(layer.get("blocked_local_or_upstream"))
            + _positive_int(layer.get("unrun"))
        )
        if terminal != int(layer["denominator"]):
            raise RuntimeError(f"Robot layer terminal accounting drift: {layer}")
    return summary


def deadline_audit(task_id: str, clock: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    started = parse_time(str(clock["started_at"]))
    freeze = parse_time(str(clock["freeze_algorithm_candidates_at"]))
    stop_new = parse_time(str(clock["start_new_sessions_deadline_at"]))
    drain = parse_time(str(clock["gpu_drain_deadline_at"]))
    deadline = parse_time(str(clock["deadline_at"]))
    if now.tzinfo is None or now.utcoffset() is None:
        raise RuntimeError("timezone-aware current time required")
    if not started <= freeze <= stop_new <= drain <= deadline:
        raise RuntimeError("window clock ordering invalid")
    mode = "RELEASE_CANDIDATE" if task_id == TASK_IDS[0] else "FINAL_AUDIT"
    return {
        "schema_version": "0915-robot15h-release-deadline-audit-v1",
        "task_id": task_id,
        "mode": mode,
        "observed_at": now.isoformat(timespec="seconds"),
        "started_at": started.isoformat(timespec="seconds"),
        "freeze_algorithm_candidates_at": freeze.isoformat(timespec="seconds"),
        "start_new_sessions_deadline_at": stop_new.isoformat(timespec="seconds"),
        "gpu_drain_deadline_at": drain.isoformat(timespec="seconds"),
        "deadline_at": deadline.isoformat(timespec="seconds"),
        "freeze_deadline_met": now <= freeze,
        "new_session_cutoff_reached": now >= stop_new,
        "gpu_drain_deadline_reached": now >= drain,
        "window_deadline_met": now <= deadline,
        "phase_open": now <= freeze if task_id == TASK_IDS[0] else drain <= now <= deadline,
    }


def is_artifact_ref(value: object) -> bool:
    return isinstance(value, Mapping) and {"path", "bytes", "sha256"}.issubset(value)


def iter_artifact_refs(value: object, location: str = "$") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if is_artifact_ref(value):
        yield location, value  # type: ignore[misc]
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_artifact_refs(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_artifact_refs(child, f"{location}[{index}]")


def audit_references(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for document_index, value in enumerate(values):
        for location, artifact in iter_artifact_refs(value, f"$[{document_index}]"):
            identity = (str(artifact["path"]), int(artifact["bytes"]), str(artifact["sha256"]))
            if identity in seen:
                continue
            seen.add(identity)
            try:
                path = verify_ref(artifact)
                checked.append({"location": location, **ref(path)})
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(f"{location}: {error}")
    return {
        "schema_version": "0915-robot15h-reference-audit-v1",
        "status": "PASSED" if not errors else "REJECTED",
        "checked_count": len(checked),
        "checked": checked,
        "errors": errors,
    }


def audit_campaign_references(
    evidence: Sequence[Mapping[str, Any]], sidecars: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit immutable dependency bindings and current authoritative artifacts.

    Every dependency RESULT/batch binding remains part of the audit surface, even
    when a successor supersedes its algorithm authority.  Nested artifact refs,
    however, are recursively followed only from the selected authoritative
    evidence.  Superseded runners may legitimately clean their private staging
    trees after atomically publishing a successor, so those historical nested
    staging refs cannot be treated as current consumable authority.
    """
    dependency_bindings: list[dict[str, Any]] = []
    for item in evidence:
        binding: dict[str, Any] = {}
        for name in ("result_ref", "batch_ref"):
            value = item.get(name)
            if is_artifact_ref(value):
                binding[name] = value
        dependency_bindings.append(binding)
    authoritative = select_authoritative_evidence(evidence)
    return audit_references([
        *dependency_bindings,
        *authoritative,
        *sidecars.values(),
    ])


def validate_final_sidecars(
    sidecars: Mapping[str, Mapping[str, Any]], *, clock: Mapping[str, Any], now: datetime,
) -> dict[str, Any]:
    errors: list[str] = []
    for kind, schema in SIDECAR_SCHEMAS.items():
        value = sidecars.get(kind)
        if not isinstance(value, Mapping):
            errors.append(f"missing {kind} evidence")
        elif value.get("schema_version") != schema:
            errors.append(f"{kind} evidence schema mismatch")

    runtime = sidecars.get("runtime", {})
    observed_at: datetime | None = None
    try:
        observed_at = parse_time(str(runtime.get("observed_at")))
    except (RuntimeError, ValueError):
        errors.append("runtime observed_at invalid")
    if observed_at is not None:
        drain = parse_time(str(clock["gpu_drain_deadline_at"]))
        if observed_at < drain:
            errors.append("runtime evidence predates GPU drain deadline")
        if observed_at > now:
            errors.append("runtime evidence is from the future")
        if (now - observed_at).total_seconds() > 900:
            errors.append("runtime evidence is older than 15 minutes")
    for name in ("owned_processes", "gpu_leases", "active_writers"):
        if runtime.get(name) != []:
            errors.append(f"runtime {name} is not empty")
    for name in (
        "all_owned_tasks_terminal", "new_sessions_started_after_cutoff", "model_inference_active_after_drain",
    ):
        expected = False if name != "all_owned_tasks_terminal" else True
        if runtime.get(name) is not expected:
            errors.append(f"runtime {name} must be {expected}")
    source_integrity = runtime.get("selected_source_integrity")
    if not isinstance(source_integrity, Mapping):
        errors.append("runtime selected_source_integrity is absent")
    elif (
        source_integrity.get("status") != "PASSED"
        or source_integrity.get("sessions_checked") != 12
        or source_integrity.get("content_drift_count") != 0
        or source_integrity.get("0916_consumed") is not False
    ):
        errors.append("runtime selected source integrity did not pass")

    video = sidecars.get("video", {})
    videos = video.get("videos")
    if video.get("status") != "PASSED" or not isinstance(videos, list) or not videos:
        errors.append("video evidence must contain at least one passed full decode")
    elif isinstance(videos, list):
        for index, item in enumerate(videos):
            if not isinstance(item, Mapping):
                errors.append(f"video[{index}] is not an object")
                continue
            try:
                path = verify_ref(item)
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(f"video[{index}] reference invalid: {error}")
                continue
            if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
                errors.append(f"video[{index}] has unsupported suffix")
            expected_frames = _positive_int(item.get("expected_frames"))
            decoded_frames = _positive_int(item.get("decoded_frames"))
            if item.get("full_decode") is not True or expected_frames < 1 or decoded_frames != expected_frames:
                errors.append(f"video[{index}] lacks exact full-decode evidence")

    tests = sidecars.get("tests", {})
    checks = tests.get("checks")
    if tests.get("status") != "PASSED" or not isinstance(checks, list) or not checks:
        errors.append("test evidence lacks passed checks")
    elif isinstance(checks, list):
        names: set[str] = set()
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping):
                errors.append(f"test check[{index}] is not an object")
                continue
            names.add(str(check.get("kind")))
            if check.get("status") != "PASSED" or check.get("returncode") != 0 or not check.get("command"):
                errors.append(f"test check[{index}] did not pass")
        if not {"governance", "tests"}.issubset(names):
            errors.append("test evidence must include governance and tests kinds")

    commit = sidecars.get("commit", {})
    if commit.get("status") != "PASSED":
        errors.append("commit evidence status is not PASSED")
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit.get("commit", ""))):
        errors.append("commit evidence lacks a 40-hex commit")
    if commit.get("git_status_porcelain") != "":
        errors.append("commit evidence says the worktree is dirty")
    if commit.get("remote_push_performed") is not False:
        errors.append("commit evidence must state remote_push_performed=false")

    return {
        "schema_version": "0915-robot15h-final-sidecar-audit-v1",
        "status": "PASSED" if not errors else "REJECTED",
        "errors": errors,
        "checks": {
            "all_owned_tasks_terminal": runtime.get("all_owned_tasks_terminal") is True,
            "no_owned_processes": runtime.get("owned_processes") == [],
            "no_gpu_leases": runtime.get("gpu_leases") == [],
            "no_active_writers": runtime.get("active_writers") == [],
            "new_session_cutoff_respected": runtime.get("new_sessions_started_after_cutoff") is False,
            "gpu_drain_respected": runtime.get("model_inference_active_after_drain") is False,
            "selected_sources_unchanged": isinstance(source_integrity, Mapping)
            and source_integrity.get("status") == "PASSED"
            and source_integrity.get("content_drift_count") == 0,
            "videos_full_decode": isinstance(videos, list) and bool(videos) and not any(
                "video[" in error for error in errors
            ),
            "governance_and_tests_passed": isinstance(checks, list) and bool(checks) and not any(
                "test " in error for error in errors
            ),
            "clean_local_commit_no_push": not any("commit evidence" in error for error in errors),
        },
    }


def _batch_reference(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    # Model runners use capability-specific names when a generic batch could be
    # mistaken for authority outside that capability.  They are still the
    # authoritative per-session batch for release accounting and must not be
    # silently collapsed into a campaign-level zero row.
    for key in ("batch_result", "worker_batch", "object_identity_batch"):
        value = result.get(key)
        if is_artifact_ref(value):
            return value  # type: ignore[return-value]
    return None


def collect_dependency_evidence(
    state: Mapping[str, Any], dependencies: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = {str(row.get("task_id")): row for row in state.get("tasks", []) if isinstance(row, Mapping)}
    evidence: list[dict[str, Any]] = []
    bound_refs: list[dict[str, Any]] = []
    for task_id in dependencies:
        task = rows.get(task_id)
        if task is None:
            raise RuntimeError(f"dependency absent from task state: {task_id}")
        terminal_status = str(task.get("status"))
        if terminal_status not in TERMINAL:
            raise RuntimeError(f"dependency is not terminal: {task_id}={terminal_status}")
        result_value = task.get("result")
        if not is_artifact_ref(result_value):
            raise RuntimeError(f"dependency result reference absent: {task_id}")
        result_path = verify_ref(result_value)  # type: ignore[arg-type]
        result = load_json(result_path)
        if result.get("task_id") != task_id or result.get("status") != terminal_status:
            raise RuntimeError(f"dependency state/result mismatch: {task_id}")
        result_ref = dict(result_value)
        bound_refs.append(result_ref)
        batch_value = _batch_reference(result)
        batch: dict[str, Any] | None = None
        batch_ref: dict[str, Any] | None = None
        if batch_value is not None:
            batch_path = verify_ref(batch_value)
            batch = load_json(batch_path)
            batch_ref = dict(batch_value)
            bound_refs.append(batch_ref)
        evidence.append({
            "task_id": task_id,
            "terminal_status": terminal_status,
            "result_ref": result_ref,
            "result": result,
            "batch_ref": batch_ref,
            "batch": batch,
        })
    return evidence, bound_refs


def validate_route(task_id: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    packet_path = ROOT / f"tasks/current/{task_id}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(task_id) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current release-control packet differs from frozen CPU spec")
    missing_outputs = EXPECTED_OUTPUTS[task_id] - set(packet.get("required_outputs", []))
    if missing_outputs:
        raise RuntimeError(f"release-control packet is still an inert placeholder: {sorted(missing_outputs)}")
    runner_relative = str(Path(__file__).resolve().relative_to(ROOT))
    if runner_relative not in packet.get("read_set", []):
        raise RuntimeError("release-control runner is absent from task read_set")
    state = load_json(STATE)
    if state.get("next_task", {}).get("task_id") != task_id:
        raise RuntimeError("release-control task is not current next_task")
    index = load_json(INDEX)
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == task_id), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("release-control task is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("release-control packet SHA differs from route")
    return packet, packet_path, state


def heartbeat(task_id: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", task_id,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", task_id.upper()],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def _sidecar_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "runtime": args.runtime_evidence,
        "video": args.video_evidence,
        "tests": args.test_evidence,
        "commit": args.commit_evidence,
    }
    return {key: Path(value).resolve(strict=True) for key, value in values.items() if value is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_IDS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--video-evidence", type=Path)
    parser.add_argument("--test-evidence", type=Path)
    parser.add_argument("--commit-evidence", type=Path)
    args = parser.parse_args()

    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 characters required")
    packet, packet_path, state = validate_route(args.task_id)
    output = args.output_root.resolve()
    primary = Path(str(packet["write_set"][0]))
    expected_output = (primary if primary.is_absolute() else ROOT / primary).resolve()
    if output != expected_output:
        raise RuntimeError("output root differs from packet primary write root")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output root required: {output}")

    clock = load_json(WINDOW_CLOCK)
    if clock.get("run_id") != WINDOW_RUN_ID:
        raise RuntimeError("window clock run identity drift")
    now = datetime.now().astimezone()
    deadline = deadline_audit(args.task_id, clock, now)
    if args.task_id == TASK_IDS[1] and not deadline["gpu_drain_deadline_reached"]:
        raise RuntimeError("final audit cannot run before the GPU drain deadline")

    evidence, dependency_refs = collect_dependency_evidence(state, packet["dag_dependencies"])
    sidecar_paths = _sidecar_paths(args)
    sidecars: dict[str, dict[str, Any]] = {}
    if args.task_id == TASK_IDS[1]:
        if set(sidecar_paths) != set(SIDECAR_SCHEMAS):
            raise RuntimeError("final audit requires runtime, video, test, and commit evidence")
        packet_reads = {
            str((Path(value) if Path(value).is_absolute() else ROOT / value).resolve())
            for value in packet.get("read_set", [])
        }
        missing_reads = [str(path) for path in sidecar_paths.values() if str(path) not in packet_reads]
        if missing_reads:
            raise RuntimeError(f"final evidence sidecar is absent from packet read_set: {missing_reads}")
        sidecars = {kind: load_json(path) for kind, path in sidecar_paths.items()}

    matrix = build_capability_matrix(evidence, freeze_deadline_met=bool(deadline["freeze_deadline_met"]))
    input_refs = [ref(WINDOW_CLOCK), *dependency_refs, *(ref(path) for path in sidecar_paths.values())]
    signature_payload = {
        "schema_version": "0915-robot15h-release-control-run-signature-v1",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
        "weights": "ABSENT",
        "gpu_used": False,
        "executor_epoch": args.executor_epoch,
        "task_packet": ref(packet_path),
        "inputs": input_refs,
        "code": ref(Path(__file__)),
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}

    # All authority, clock, dependency, and explicit sidecar validation above is
    # complete before claiming a fresh immutable output root.
    output.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-release-control-writer-claim-v1",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output),
        "run_signature_sha256": signature["run_signature_sha256"],
    })
    heartbeat(args.task_id)
    atomic_json(output / "DEADLINE_AUDIT.json", deadline)
    atomic_json(output / "CAPABILITY_MATRIX.json", matrix)

    status = "PASSED"
    final_audit: dict[str, Any] | None = None
    reference_audit: dict[str, Any] | None = None
    if args.task_id == TASK_IDS[0]:
        if not deadline["freeze_deadline_met"]:
            status = "BUDGET_EXHAUSTED"
            for row in matrix["rows"]:
                row["w1_expansion_allowed"] = False
            matrix["counts"]["w1_expansion_allowed_rows"] = 0
            atomic_json(output / "CAPABILITY_MATRIX.json", matrix)
    else:
        campaign_counts = build_campaign_counts(evidence)
        atomic_json(output / "CAMPAIGN_COUNTS.json", campaign_counts)
        reference_audit = audit_campaign_references(evidence, sidecars)
        sidecar_audit = validate_final_sidecars(sidecars, clock=clock, now=now)
        final_errors = [*reference_audit["errors"], *sidecar_audit["errors"]]
        if not deadline["window_deadline_met"]:
            final_errors.append("final audit completed after H15 deadline")
            status = "BUDGET_EXHAUSTED"
        elif final_errors:
            status = "REJECTED_QUALITY"
        final_audit = {
            "schema_version": "0915-robot15h-window-final-audit-v1",
            "task_id": args.task_id,
            "window_run_id": WINDOW_RUN_ID,
            "status": "PASSED" if not final_errors else "REJECTED",
            "deadline": deadline,
            "sidecar_audit": sidecar_audit,
            "reference_status": reference_audit["status"],
            "errors": final_errors,
            "physical_deployable": 0,
            "remote_pushed": False,
        }
        atomic_json(output / "REFERENCE_AUDIT.json", reference_audit)
        atomic_json(output / "FINAL_AUDIT.json", final_audit)

    result = {
        "schema_version": "0915-robot15h-release-control-result-v1",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": status,
        "weights": "ABSENT",
        "completed_at": now.isoformat(timespec="seconds"),
        "release_status": (
            "FROZEN_QUALIFIED_SCOPES_ONLY" if args.task_id == TASK_IDS[0] and status == "PASSED"
            else "FINAL_EVIDENCE_AUDIT"
        ),
        "capability_matrix": ref(output / "CAPABILITY_MATRIX.json"),
        "campaign_counts": (
            ref(output / "CAMPAIGN_COUNTS.json") if args.task_id == TASK_IDS[1] else None
        ),
        "deadline_audit": ref(output / "DEADLINE_AUDIT.json"),
        "final_audit": ref(output / "FINAL_AUDIT.json") if final_audit is not None else None,
        "reference_audit": ref(output / "REFERENCE_AUDIT.json") if reference_audit is not None else None,
        "executed_rows": matrix["counts"]["executed_rows"],
        "exported_rows": matrix["counts"]["exported_rows"],
        "quality_admitted_rows": matrix["counts"]["quality_admitted_rows"],
        "w1_expansion_allowed_rows": matrix["counts"]["w1_expansion_allowed_rows"],
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Release-control accounting only. Every capability retains its source terminal, "
            "task, and evidence scope. This result grants no control, training, physical "
            "deployment, metric Contact, or cross-task authority."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-release-control-run-receipt-v1",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": status,
        "result": ref(output / "RESULT.json"),
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
    })
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
