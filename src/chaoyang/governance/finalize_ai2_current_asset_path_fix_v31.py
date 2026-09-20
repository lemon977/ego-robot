#!/usr/bin/env python3
"""CAS-finalize the narrow AI2 current-asset path correction successor."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import (
    REPO_ROOT,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_ai2_current_asset_path_fix_v31 import (
    EXECUTION_REVISION,
    PLAN_REVISION,
    TASK_ID,
)


LIVE = {"PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
TERMINAL = {"PASSED", "BLOCKED_PREREQ", "REJECTED_QUALITY"}


class FinalizeError(RuntimeError):
    """Raised when the successor result cannot be safely terminalized."""


def _paths(root: Path) -> dict[str, Path]:
    attempt = root / f"_run/current/{TASK_ID}/attempts/attempt_0001"
    return {
        "authority": root / "docs/governance/CURRENT_AUTHORITY_INDEX.json",
        "receipt": root / "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "state": root / "docs/governance/LONG_HORIZON_TASK_STATE.json",
        "index": root / "tasks/current/INDEX.json",
        "packet": root / f"tasks/current/{TASK_ID}/TASK_PACKET.json",
        "attempt": attempt,
        "result": attempt / "RESULT.json",
        "terminal_receipt": root
        / "tasks/receipts/AI2_CURRENT_ASSET_PATH_FIX_V31_RESULT.json",
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise FinalizeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def finalize(
    *,
    expected_revision: int,
    root: Path = REPO_ROOT,
    publisher: Callable[..., dict[str, Any]] = publish_bundle,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    paths = _paths(root)
    receipt = load_json(paths["receipt"])
    if int(receipt.get("governance_revision", -1)) != expected_revision:
        raise FinalizeError("CAS revision mismatch before AI2 successor finalization")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise FinalizeError("current receipt conflict: " + "; ".join(errors))
    result = load_json(paths["result"])
    if result.get("task_id") != TASK_ID or result.get("status") not in TERMINAL:
        raise FinalizeError("AI2 successor result identity or terminal status mismatch")
    if result.get("model_calls") != 0 or result.get("gpu_calls") != 0:
        raise FinalizeError("model/GPU calls are forbidden for the AI2 successor")
    if not result.get("all_sessions_terminal") or result.get("counts", {}).get("total") != 8:
        raise FinalizeError("all eight AI2 sessions must be terminal")
    if result.get("predecessor_preserved") is not True:
        raise FinalizeError("sealed predecessor preservation was not proven")

    state = load_json(paths["state"])
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID]
    if len(rows) != 1 or rows[0].get("status") not in LIVE:
        raise FinalizeError("AI2 successor is not the unique live task row")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise FinalizeError("AI2 successor is not current next_task")
    other_live = [
        row.get("task_id")
        for row in state.get("tasks", [])
        if row.get("task_id") != TASK_ID and row.get("status") in LIVE
    ]
    if other_live:
        raise FinalizeError(f"other live tasks prevent finalization: {other_live}")
    current = load_json(paths["index"])
    routes = current.get("task_packets", [])
    if len(routes) != 1 or routes[0].get("task_id") != TASK_ID:
        raise FinalizeError("AI2 successor is not the sole current route")
    route = routes[0]
    if route.get("execution_allowed") is not True:
        raise FinalizeError("AI2 successor route is not executable")
    packet_ref = artifact_ref(paths["packet"])
    if route.get("packet_sha256") != packet_ref["sha256"]:
        raise FinalizeError("AI2 successor task packet SHA mismatch")
    for path in (
        paths["attempt"] / "PREDECESSOR_ACTIVE_INDEX.json",
        paths["attempt"] / "RUN_RECEIPT.json",
        paths["terminal_receipt"],
    ):
        if path.exists() or path.is_symlink():
            raise FinalizeError(f"fresh finalization artifact required: {path}")

    predecessor_index = paths["attempt"] / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(paths["index"], predecessor_index)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{EXECUTION_REVISION}_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor_index),
        "task_packets": [],
        "claim_limit": (
            "AI2 path correction is terminal. Evidence blockers remain local; no model, "
            "training, control or deployment successor is authorized."
        ),
    }
    created = created_at or now_iso()
    terminal_receipt = {
        "schema_version": "ai2-current-asset-path-fix-v31-terminal-receipt-v1",
        "task_id": TASK_ID,
        "status": result["status"],
        "created_at": created,
        "result": artifact_ref(paths["result"]),
        "counts": result["counts"],
        "predecessor": result["predecessor"],
        "pipeline_complete": False,
        "training_complete": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    _write_once(paths["terminal_receipt"], terminal_receipt)
    task = rows[0]
    task.update(
        status=result["status"],
        updated_at=created,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        result=artifact_ref(paths["result"]),
    )
    state["next_task"] = None
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": None,
                "attempt": 1,
                "status": result["status"],
                "created_at": created,
                "message": "AI2 current-asset path correction reached a bounded terminal.",
                "result": artifact_ref(paths["result"]),
            }
        ]
    )[-100:]
    published = publisher(
        load_json(paths["authority"]),
        state,
        event_type=f"AI2_CURRENT_ASSET_PATH_FIX_V31_{result['status']}",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=paths["index"],
        task_packet_index_value=terminal_index,
    )
    run_receipt = {
        "schema_version": "ai2-current-asset-path-fix-v31-run-receipt-v1",
        "task_id": TASK_ID,
        "status": result["status"],
        "finalization_status": "PASSED_CAS_PUBLISHED",
        "created_at": created,
        "result": artifact_ref(paths["result"]),
        "terminal_receipt": artifact_ref(paths["terminal_receipt"]),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "current_index_status": "PASS_NO_ACTIVE_TASKS",
        "pipeline_complete": False,
    }
    _write_once(paths["attempt"] / "RUN_RECEIPT.json", run_receipt)
    return run_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(expected_revision=args.expected_revision), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
