#!/usr/bin/env python3
"""Audit the fixed current-only AI2 W0/A1/A2 cohort without model calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable
import uuid

from chaoyang.ops.audit_ai2_real_assets_v31 import (
    COHORT_ROOTS,
    ROOT,
    artifact_ref,
    atomic_json,
    build_audit,
)


SCHEMA_VERSION = "AI2_REAL_ASSETS_COHORT_AUDIT_V31"
SESSION_SCHEMA_VERSION = "AI2_REAL_ASSET_SESSION_TERMINAL_V31"
PARENT_TASK_ID = "three_stream_stable_baseline_v31"
LANE = "ai2"
OUTPUT_NAME = "AI2_REAL_ASSETS_COHORT_AUDIT_V31"
SESSION_SPECS = (
    {"cohort": "W0", "task": "poker", "session_id": "play_cards_0915_031"},
    {"cohort": "W0", "task": "poker", "session_id": "play_cards_0915_119"},
    {"cohort": "W0", "task": "potato_chips", "session_id": "get_potato_chips_0915_007"},
    {"cohort": "W0", "task": "potato_chips", "session_id": "get_potato_chips_0915_042"},
    {"cohort": "A1", "task": "poker", "session_id": "play_cards_0915_044"},
    {"cohort": "A1", "task": "potato_chips", "session_id": "get_potato_chips_0915_097"},
    {"cohort": "A2", "task": "poker", "session_id": "play_cards_0915_106"},
    {"cohort": "A2", "task": "potato_chips", "session_id": "get_potato_chips_0915_029"},
)


class CohortAuditError(RuntimeError):
    """Raised when the lane writer boundary or immutable cohort is invalid."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lane_root(root: Path) -> Path:
    return root / f"_run/current/{PARENT_TASK_ID}/attempts/attempt_0001/lanes/{LANE}"


def _assert_lane_identity(root: Path) -> Path:
    lane_root = _lane_root(root).resolve(strict=True)
    state_path = lane_root / "STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("lane") != LANE or state.get("parent_task_id") != PARENT_TASK_ID:
        raise CohortAuditError("AI2 lane STATE identity mismatch")
    if state.get("status") not in {
        "READY_CPU_PREFLIGHT",
        "CPU_PREFLIGHT_COMPLETE",
        "BLOCKED_CPU_PREFLIGHT",
        "REJECTED_CPU_PREFLIGHT",
    }:
        raise CohortAuditError(f"AI2 lane is not recordable: {state.get('status')}")
    return lane_root


def _paths(root: Path, spec: dict[str, str]) -> dict[str, Path | None]:
    cohort = spec["cohort"]
    task = spec["task"]
    session = spec["session_id"]
    hawor_base = root / COHORT_ROOTS[cohort]["hawor"] / "hawor/sessions" / task / session
    kai_base = root / COHORT_ROOTS[cohort]["kai22"] / "sessions" / task / session
    result: dict[str, Path | None] = {
        "hawor_npz": hawor_base / "HAWOR_RAW_MANO21.npz",
        "hawor_result": hawor_base / "RESULT.json",
        "kai22_npz": kai_base
        / (
            "KAI22_R0_BASELINE_V1.npz"
            if cohort == "W0"
            else "KAI22_R0_QUALITY_PROPAGATION_V1.npz"
        ),
        "kai22_result": kai_base / "RESULT.json",
        "bounded_npz": None,
        "bounded_result": None,
        "sam_result": None,
    }
    if cohort == "W0":
        bounded = root / COHORT_ROOTS[cohort]["kai22"] / "bounded_v2" / session
        sam = root / COHORT_ROOTS[cohort]["sam"] / "sessions" / task / session
        result.update(
            {
                "bounded_npz": bounded / "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
                "bounded_result": bounded / "RESULT.json",
                "sam_result": sam / "RESULT.json",
            }
        )
    return result


def _future_ref(staged: Path, final: Path) -> dict[str, Any]:
    reference = artifact_ref(staged)
    reference["path"] = str(final.resolve(strict=False))
    return reference


def _terminal_from_audit(audit: dict[str, Any]) -> tuple[str, list[str]]:
    blockers = sorted(set(str(value) for value in audit.get("blocker_codes", [])))
    if blockers:
        return "BLOCKED_EVIDENCE", blockers
    if audit.get("status") == "COMPLETED_FAIL_CLOSED_AUDIT":
        return "PASS_CPU_PREFLIGHT", []
    return "REJECTED_AUDIT_STATUS", ["UNEXPECTED_SESSION_AUDIT_STATUS"]


def _session_terminal(
    *,
    spec: dict[str, str],
    audit: dict[str, Any],
    audit_reference: dict[str, Any],
) -> dict[str, Any]:
    status, blockers = _terminal_from_audit(audit)
    tiers = {
        side: row.get("highest_admitted_level", "NONE")
        for side, row in audit.get("kai22_tier", {}).get("sides", {}).items()
    }
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": status,
        "cohort": spec["cohort"],
        "task": spec["task"],
        "session_id": spec["session_id"],
        "audit": audit_reference,
        "blocker_codes": blockers,
        "kai22_highest_admitted_level_by_side": tiers,
        "all_inputs_sha_bound": True,
        "current_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }


def _rejected_terminal(spec: dict[str, str], error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "REJECTED_INPUT_PRECONDITION",
        "cohort": spec["cohort"],
        "task": spec["task"],
        "session_id": spec["session_id"],
        "audit": None,
        "blocker_codes": ["INPUT_PATH_OR_SHA_BINDING_REJECTED"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "all_inputs_sha_bound": False,
        "current_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }


def run_cohort(
    *,
    root: Path = ROOT,
    audit_builder: Callable[..., dict[str, Any]] = build_audit,
) -> dict[str, Any]:
    """Audit all eight frozen sessions and atomically publish one lane result."""

    root = root.resolve(strict=True)
    lane_root = _assert_lane_identity(root)
    output_root = lane_root / OUTPUT_NAME
    if output_root.exists() or output_root.is_symlink():
        raise CohortAuditError(f"fresh output root required: {output_root}")
    staging = lane_root / f".{OUTPUT_NAME}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    rows: list[dict[str, Any]] = []
    try:
        for spec in SESSION_SPECS:
            session_stage = staging / "sessions" / spec["cohort"] / spec["session_id"]
            session_final = output_root / "sessions" / spec["cohort"] / spec["session_id"]
            session_stage.mkdir(parents=True)
            try:
                inputs = _paths(root, spec)
                audit = audit_builder(
                    cohort=spec["cohort"],
                    session_id=spec["session_id"],
                    hawor_npz=inputs["hawor_npz"],
                    hawor_result=inputs["hawor_result"],
                    kai22_npz=inputs["kai22_npz"],
                    kai22_result=inputs["kai22_result"],
                    bounded_npz=inputs["bounded_npz"],
                    bounded_result=inputs["bounded_result"],
                    sam_result=inputs["sam_result"],
                    root=root,
                )
                audit_path = session_stage / "AUDIT.json"
                atomic_json(audit_path, audit)
                terminal = _session_terminal(
                    spec=spec,
                    audit=audit,
                    audit_reference=_future_ref(audit_path, session_final / "AUDIT.json"),
                )
            except Exception as error:  # independent terminalization is intentional
                terminal = _rejected_terminal(spec, error)
            result_path = session_stage / "RESULT.json"
            atomic_json(result_path, terminal)
            rows.append(
                {
                    "cohort": spec["cohort"],
                    "task": spec["task"],
                    "session_id": spec["session_id"],
                    "status": terminal["status"],
                    "blocker_codes": terminal["blocker_codes"],
                    "result": _future_ref(result_path, session_final / "RESULT.json"),
                }
            )

        counts = {
            "total": len(rows),
            "pass": sum(row["status"].startswith("PASS") for row in rows),
            "blocked": sum(row["status"].startswith("BLOCKED") for row in rows),
            "rejected": sum(row["status"].startswith("REJECTED") for row in rows),
        }
        if counts["rejected"]:
            status = "REJECTED_AI2_CURRENT_ASSET_AUDIT"
        elif counts["blocked"]:
            status = "BLOCKED_AI2_EVIDENCE"
        else:
            status = "PASS_AI2_CURRENT_ASSET_AUDIT"
        manifest = [dict(spec) for spec in SESSION_SPECS]
        lane_result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "lane": LANE,
            "parent_task_id": PARENT_TASK_ID,
            "cohort_manifest": manifest,
            "cohort_manifest_sha256": _canonical_sha(manifest),
            "counts": counts,
            "sessions": rows,
            "all_sessions_terminal": len(rows) == len(SESSION_SPECS),
            "all_existing_inputs_sha_bound": counts["rejected"] == 0,
            "current_only": True,
            "archive_consumed": False,
            "weights": "ABSENT",
            "model_calls": 0,
            "gpu_calls": 0,
            "claims": {
                "PIPELINE_COMPLETE": False,
                "NUMERIC_QUALITY_PASS": False,
                "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
                "TRAINING_COMPLETE": False,
                "TRAINING_ELIGIBLE": False,
                "CONTROL_GROUND_TRUTH": False,
                "PHYSICAL_DEPLOYABLE": False,
            },
        }
        atomic_json(staging / "RESULT.json", lane_result)
        os.replace(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return json.loads((output_root / "RESULT.json").read_text(encoding="utf-8"))


def main() -> int:
    value = run_cohort()
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
