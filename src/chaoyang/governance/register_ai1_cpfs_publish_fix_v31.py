#!/usr/bin/env python3
"""CAS-register the narrow CPU-only AI1 CPFS publication successor."""

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


TASK_ID = "ai1_cpfs_publish_fix_v31"
PLAN_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
EXECUTION_REVISION = "AI1_CPFS_PUBLISH_FIX_V31"
PREDECESSOR_RELATIVE = (
    "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/"
    "lanes/ai1/current_only_run_0001/RESULT.json"
)
CODE_PATHS = (
    "contracts/wiyh_ai1_lane_ledger_v31.schema.json",
    "contracts/wiyh_ai1_lane_result_v31.schema.json",
    "contracts/wiyh_wrist_dual_input_provenance_v1.schema.json",
    "contracts/wrist_dual_producer_config_v1.schema.json",
    "contracts/wrist_dual_representation_v1.schema.json",
    "src/chaoyang/ops/build_wiyh_wrist_dual_input_v1.py",
    "src/chaoyang/ops/build_wiyh_wrist_dual_representation_v1.py",
    "src/chaoyang/ops/run_wiyh_ai1_current_lane_v31.py",
    "src/chaoyang/research/world_in_your_hands/wrist_dual_representation_v1.py",
    "src/chaoyang/ops/run_ai1_cpfs_publish_fix_v31.py",
    "src/chaoyang/governance/register_ai1_cpfs_publish_fix_v31.py",
    "src/chaoyang/governance/finalize_ai1_cpfs_publish_fix_v31.py",
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
            "Retry only the AI1 current-only CPU lane after adding the portable CPFS "
            "writer fence, preserving the prior FAILED_RUNTIME bytes and all honest "
            "adoption-observation blockers."
        ),
        "phase": EXECUTION_REVISION,
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "weights": "ABSENT",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/current/INDEX.json",
            str(predecessor.relative_to(root)),
            "_run/current/worktrees/wiyh-hand-depth-v1/_run/current/experiments",
            "/mnt/data/egodata/datasets/ego/processed/"
            "chips_cards_handle_highview_0916/cleaned/playing_cards",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            f"tasks/current/{TASK_ID}",
            "tasks/receipts/AI1_CPFS_PUBLISH_FIX_V31_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "sealed_failed_predecessor_READ_ONLY",
            "current_fixed_sources_READ_ONLY",
        ],
        "required_outputs": [
            "attempts/attempt_0001/AI1_CURRENT_ONLY_LANE/RESULT.json",
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
            "current_only_lane_reaches_one_terminal",
            "missing_adoption_observations_remain_BLOCKED_PREREQ",
            "old_failed_attempt_is_never_modified",
        ],
        "code_closure": [artifact_ref(root / relative) for relative in CODE_PATHS],
        "claim_limit": (
            "CPU-only CPFS publication retry over the frozen current AI1 sources. "
            "No model, GPU, external metric, training, control or deployment claim."
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
        raise RegisterError("CAS revision mismatch before AI1 successor registration")
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
        raise RegisterError("AI1 successor already registered")
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
    predecessor_value = load_json(predecessor)
    if (
        predecessor_value.get("schema_version") != "chaoyang-wiyh-ai1-lane-result-v31"
        or predecessor_value.get("status") != "FAILED_RUNTIME"
    ):
        raise RegisterError("AI1 predecessor is not the sealed FAILED_RUNTIME result")
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
        "claim_limit": "One CPU-only AI1 CPFS publication retry; no model or GPU execution.",
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
        "stop_condition": "One truthful current-only AI1 lane terminal.",
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
                "message": "Registered narrow AI1 CPFS publication retry.",
                "task_packet": packet_ref,
            }
        ]
    )[-100:]
    published = publisher(
        load_json(paths["authority"]),
        state,
        event_type="AI1_CPFS_PUBLISH_FIX_V31_REGISTERED",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=paths["index"],
        task_packet_index_value=successor_index,
    )
    result = {
        "schema_version": "ai1-cpfs-publish-fix-v31-registration-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "predecessor_failed_result": artifact_ref(predecessor),
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
