#!/usr/bin/env python3
"""CAS-register the narrow CPU-only AI2 current-asset path correction."""

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
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "ai2_current_asset_path_fix_v31"
PLAN_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
EXECUTION_REVISION = "AI2_CURRENT_ASSET_PATH_FIX_V31"
PREDECESSOR_RELATIVE = (
    "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/"
    "lanes/ai2/AI2_REAL_ASSETS_COHORT_AUDIT_V31/RESULT.json"
)
CODE_PATHS = (
    "src/chaoyang/ops/run_ai2_real_assets_cohort_audit_v31.py",
    "src/chaoyang/ops/preflight_ai2_current_asset_paths_v31.py",
    "src/chaoyang/ops/run_ai2_current_asset_path_fix_v31.py",
    "src/chaoyang/governance/register_ai2_current_asset_path_fix_v31.py",
    "src/chaoyang/governance/finalize_ai2_current_asset_path_fix_v31.py",
)
LIVE = {"PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


class RegisterError(RuntimeError):
    """Raised when the successor cannot be registered from a clean terminal."""


def _paths(root: Path) -> dict[str, Path]:
    return {
        "authority": root / "docs/governance/CURRENT_AUTHORITY_INDEX.json",
        "receipt": root / "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "task_state": root / "docs/governance/LONG_HORIZON_TASK_STATE.json",
        "pointer": root / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json",
        "index": root / "tasks/current/INDEX.json",
        "packet": root / f"tasks/current/{TASK_ID}/TASK_PACKET.json",
        "registration": root / f"_run/current/{TASK_ID}/registration_0001",
        "predecessor": root / PREDECESSOR_RELATIVE,
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RegisterError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _packet(root: Path, predecessor: Path) -> dict[str, Any]:
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": (
            "Correct the explicit logical poker to playing_cards frozen-asset mapping, "
            "then rerun the fixed eight-session CPU-only fail-closed AI2 audit."
        ),
        "phase": "AI2_CURRENT_ASSET_PATH_FIX_V31",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "weights": "ABSENT",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/current/INDEX.json",
            str(predecessor.relative_to(root)),
            "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001",
            "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001",
            "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            f"tasks/current/{TASK_ID}",
            "tasks/receipts/AI2_CURRENT_ASSET_PATH_FIX_V31_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "sealed_predecessor_READ_ONLY",
            "current_frozen_assets_READ_ONLY",
        ],
        "required_outputs": [
            "attempts/attempt_0001/CURRENT_ASSET_PATH_PREFLIGHT.json",
            "attempts/attempt_0001/RESULT.json",
            "attempts/attempt_0001/RUN_RECEIPT.json",
        ],
        "budgets": {
            "attempt_max": 1,
            "cpu_workers_max": 1,
            "gpu_calls_max": 0,
            "model_calls_max": 0,
        },
        "stop_conditions": [
            "all_eight_sessions_terminal",
            "any_missing_current_binding_fails_closed",
            "old_sealed_attempt_is_never_modified",
        ],
        "code_closure": [artifact_ref(root / relative) for relative in CODE_PATHS],
        "claim_limit": (
            "CPU-only input-path correction and evidence audit. No algorithm rerun, "
            "numeric quality, training, control or deployment claim."
        ),
    }
    errors = _validate_packet(packet)
    if errors:
        raise RegisterError("invalid successor packet: " + "; ".join(errors))
    return packet


def register(
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
        raise RegisterError("CAS revision mismatch before AI2 successor registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RegisterError("current receipt conflict: " + "; ".join(errors))
    state = load_json(paths["task_state"])
    if state.get("next_task") is not None or any(
        row.get("status") in LIVE for row in state.get("tasks", [])
    ):
        raise RegisterError("registration requires no current live task")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RegisterError("AI2 successor already registered")
    current = load_json(paths["index"])
    if current.get("status") != "PASS_NO_ACTIVE_TASKS" or current.get("task_packets"):
        raise RegisterError("current task index is not an empty terminal index")
    pointer = load_json(paths["pointer"])
    pointer_path = Path(str(pointer.get("index_path", "")))
    pointer_path = pointer_path if pointer_path.is_absolute() else root / pointer_path
    if pointer_path.resolve() != paths["index"].resolve():
        raise RegisterError("current packet pointer does not target tasks/current/INDEX.json")
    current_ref = artifact_ref(paths["index"])
    if pointer.get("index_sha256") != current_ref["sha256"]:
        raise RegisterError("current packet pointer SHA mismatch")
    predecessor = paths["predecessor"].resolve(strict=True)
    if paths["registration"].exists() or paths["registration"].is_symlink():
        raise RegisterError("fresh registration output required")
    packet = _packet(root, predecessor)
    _write_once(paths["packet"], packet)
    paths["registration"].mkdir(parents=True)
    frozen = paths["registration"] / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(paths["index"], frozen)
    packet_ref = artifact_ref(paths["packet"])
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": EXECUTION_REVISION,
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen),
        "task_packets": [
            {
                "task_id": TASK_ID,
                "packet_path": str(paths["packet"].relative_to(root)),
                "packet_sha256": packet_ref["sha256"],
                "execution_class": "CURRENT_CPU_PREFLIGHT_ONLY",
                "execution_allowed": True,
                "weights": "ABSENT",
            }
        ],
        "claim_limit": "One CPU-only AI2 path correction; no model or GPU execution.",
    }
    created = created_at or now_iso()
    state["tasks"].append(
        {
            "task_id": TASK_ID,
            "phase": packet["phase"],
            "plan_execution_revision": EXECUTION_REVISION,
            "attempt": 1,
            "status": "PENDING",
            "updated_at": created,
            "heartbeat_at": None,
            "session": None,
            "pid": None,
            "proc_start_ticks": None,
            "gpu_id": None,
            "task_packet": packet_ref,
        }
    )
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_ONLY; one writer; weights ABSENT",
        "stop_condition": "Eight current sessions independently terminalized.",
    }
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": None,
                "attempt": 1,
                "status": "PENDING",
                "created_at": created,
                "message": "Registered narrow AI2 current-asset path correction.",
                "task_packet": packet_ref,
            }
        ]
    )[-100:]
    published = publisher(
        load_json(paths["authority"]),
        state,
        event_type="AI2_CURRENT_ASSET_PATH_FIX_V31_REGISTERED",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=paths["index"],
        task_packet_index_value=successor_index,
    )
    result = {
        "schema_version": "ai2-current-asset-path-fix-v31-registration-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "predecessor_audit": artifact_ref(predecessor),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    }
    _write_once(paths["registration"] / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(register(expected_revision=args.expected_revision), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
