#!/usr/bin/env python3
"""Run the CPU-only AI2 asset-directory correction successor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable
import uuid

from chaoyang.governance.common import artifact_ref as governance_artifact_ref
from chaoyang.governance.common import load_json
from chaoyang.ops.audit_ai2_real_assets_v31 import ROOT, atomic_json, build_audit
from chaoyang.ops.preflight_ai2_current_asset_paths_v31 import build_preflight
from chaoyang.ops.run_ai2_real_assets_cohort_audit_v31 import (
    ASSET_TASK_DIRECTORIES,
    SESSION_SPECS,
    _canonical_sha,
    _future_ref,
    _paths,
    _rejected_terminal,
    _session_terminal,
)


TASK_ID = "ai2_current_asset_path_fix_v31"
SCHEMA_VERSION = "AI2_CURRENT_ASSET_PATH_FIX_V31_RESULT"
OUTPUT_ROOT_RELATIVE = f"_run/current/{TASK_ID}/attempts/attempt_0001"
PREDECESSOR_RELATIVE = (
    "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/"
    "lanes/ai2/AI2_REAL_ASSETS_COHORT_AUDIT_V31/RESULT.json"
)


class SuccessorRunError(RuntimeError):
    """Raised when the immutable successor cannot be run safely."""


def _assert_routed(root: Path) -> None:
    index = load_json(root / "tasks/current/INDEX.json")
    routes = index.get("task_packets", [])
    if len(routes) != 1 or routes[0].get("task_id") != TASK_ID:
        raise SuccessorRunError("AI2 path-fix successor is not the sole current route")
    if routes[0].get("execution_allowed") is not True:
        raise SuccessorRunError("AI2 path-fix successor is not execution_allowed")


def run_successor(
    *,
    root: Path = ROOT,
    audit_builder: Callable[..., dict[str, Any]] = build_audit,
    require_route: bool = True,
) -> dict[str, Any]:
    """Publish a fresh successor attempt while retaining the sealed predecessor."""

    root = root.resolve(strict=True)
    if require_route:
        _assert_routed(root)
    predecessor = (root / PREDECESSOR_RELATIVE).resolve(strict=True)
    predecessor_value = load_json(predecessor)
    if predecessor_value.get("schema_version") != "AI2_REAL_ASSETS_COHORT_AUDIT_V31":
        raise SuccessorRunError("sealed predecessor schema mismatch")
    output_root = root / OUTPUT_ROOT_RELATIVE
    if output_root.exists() or output_root.is_symlink():
        raise SuccessorRunError(f"fresh successor attempt required: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    rows: list[dict[str, Any]] = []
    try:
        preflight = build_preflight(root=root)
        atomic_json(staging / "CURRENT_ASSET_PATH_PREFLIGHT.json", preflight)
        if preflight["status"] != "PASS_8_CURRENT_ASSET_BINDINGS":
            raise SuccessorRunError("fixed current-asset preflight did not bind all 8 sessions")
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
            except Exception as error:  # independent fail-closed terminalization
                terminal = _rejected_terminal(spec, error)
            terminal["asset_directory"] = ASSET_TASK_DIRECTORIES[spec["task"]]
            result_path = session_stage / "RESULT.json"
            atomic_json(result_path, terminal)
            rows.append(
                {
                    "cohort": spec["cohort"],
                    "task": spec["task"],
                    "asset_directory": ASSET_TASK_DIRECTORIES[spec["task"]],
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
        terminal_status = "REJECTED_QUALITY" if counts["rejected"] else (
            "BLOCKED_PREREQ" if counts["blocked"] else "PASSED"
        )
        manifest = [
            {**spec, "asset_directory": ASSET_TASK_DIRECTORIES[spec["task"]]}
            for spec in SESSION_SPECS
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": terminal_status,
            "machine_status": (
                "REJECTED_AI2_CURRENT_ASSET_AUDIT"
                if counts["rejected"]
                else "BLOCKED_AI2_EVIDENCE"
                if counts["blocked"]
                else "PASS_AI2_CURRENT_ASSET_AUDIT"
            ),
            "predecessor": governance_artifact_ref(predecessor),
            "predecessor_preserved": True,
            "path_preflight": _future_ref(
                staging / "CURRENT_ASSET_PATH_PREFLIGHT.json",
                output_root / "CURRENT_ASSET_PATH_PREFLIGHT.json",
            ),
            "cohort_manifest": manifest,
            "cohort_manifest_sha256": _canonical_sha(manifest),
            "counts": counts,
            "sessions": rows,
            "all_sessions_terminal": len(rows) == len(SESSION_SPECS),
            "all_existing_inputs_sha_bound": counts["rejected"] == 0,
            "logical_poker_task_preserved": True,
            "poker_asset_directory": "playing_cards",
            "current_only": True,
            "archive_consumed": False,
            "weights": "ABSENT",
            "model_calls": 0,
            "gpu_calls": 0,
            "pipeline_complete": False,
            "training_complete": False,
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
        }
        atomic_json(staging / "RESULT.json", result)
        os.replace(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_json(output_root / "RESULT.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run_successor(root=args.root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
