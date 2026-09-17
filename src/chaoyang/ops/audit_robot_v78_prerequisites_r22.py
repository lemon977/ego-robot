#!/usr/bin/env python3
"""Audit existing v77 hard-pass evidence needed by the V78 reach canary.

The audit is CPU-only and metadata-only.  It does not solve, render, or derive
reach numbers.  It reports whether each metric can be computed later from
already published trajectories in a common world coordinate domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


TASK_ID = "robot_v78_prerequisite_closure_r22"


class ContractError(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def ref(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if expected is not None and result != expected:
        raise ContractError(f"artifact reference mismatch: {path}")
    return result


def write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def _object_evidence(root: Path, task: str, session: str) -> dict[str, Any]:
    result_path = root / task / session / "RESULT.json"
    if not result_path.is_file():
        return {"available": False, "reason": "OBJECT6D_RESULT_ABSENT"}
    result = load(result_path)
    trajectories = []
    has_world_pose = False
    has_valid = False
    frame_ids: set[int] = set()
    for instance in result.get("instances", {}).values():
        trajectory_ref = instance.get("trajectory")
        if not isinstance(trajectory_ref, dict):
            continue
        trajectory_path = Path(trajectory_ref["path"])
        checked = ref(trajectory_path, trajectory_ref)
        with np.load(trajectory_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"frame_indices", "valid", "T_object_to_world"}
            if required <= keys:
                has_world_pose = True
                valid = np.asarray(archive["valid"], dtype=bool)
                frames = np.asarray(archive["frame_indices"], dtype=int)
                if valid.shape == frames.shape and bool(valid.any()):
                    has_valid = True
                    frame_ids.update(int(value) for value in frames[valid])
            checked["keys"] = sorted(keys)
        trajectories.append(checked)
    return {
        "available": has_world_pose and has_valid,
        "result": ref(result_path),
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "world_pose_present": has_world_pose,
        "direct_valid_pose_present": has_valid,
        "valid_frame_ids": sorted(frame_ids),
    }


def audit_row(row: dict[str, Any], object_root: Path) -> dict[str, Any]:
    terminal_path = Path(row["result"]["path"])
    terminal_ref = ref(terminal_path, row["result"])
    terminal = load(terminal_path)
    arm_expected = terminal.get("evidence", {}).get("arm_result")
    if not isinstance(arm_expected, dict):
        raise ContractError(f"hard-pass terminal lacks arm_result: {row['session']}")
    arm_path = Path(arm_expected["path"])
    arm_ref = ref(arm_path, arm_expected)
    arm = load(arm_path)
    states_expected = arm.get("output_states")
    if not isinstance(states_expected, dict):
        raise ContractError(f"arm result lacks output_states: {row['session']}")
    states_path = Path(states_expected["path"])
    states_ref = ref(states_path, states_expected)
    with np.load(states_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        shapes = {key: list(archive[key].shape) for key in keys}
        action_ready = "q_arm" in keys and archive["q_arm"].ndim == 3 and archive["q_arm"].shape[0] >= 2
        ik_ready = (
            {"valid_side_frame", "source_frames"} <= keys
            and archive["valid_side_frame"].ndim == 2
            and archive["source_frames"].ndim == 1
        )
        wrist_ready = {"T_actual_hand_root_world", "source_frames"} <= keys
        robot_frames = set(int(value) for value in np.asarray(archive["source_frames"], dtype=int)) if "source_frames" in keys else set()

    objects = _object_evidence(object_root, row["task"], row["session"])
    object_frames = set(objects.pop("valid_frame_ids", []))
    overlap = sorted(robot_frames & object_frames)
    approach_ready = wrist_ready and objects["available"] and bool(overlap)
    readiness = {
        "action_amplitude": action_ready,
        "ik_frame_valid_rate": ik_ready,
        "wrist_object_approach": approach_ready,
    }
    return {
        "session": row["session"],
        "task": row["task"],
        "v77_terminal_status": row["terminal_status"],
        "metric_input_readiness": readiness,
        "all_metric_inputs_ready": all(readiness.values()),
        "world_frame_overlap_count": len(overlap),
        "evidence": {
            "terminal": terminal_ref,
            "arm_result": arm_ref,
            "arm_states": {**states_ref, "keys": sorted(keys), "shapes": shapes},
            "object6d": objects,
        },
        "claim_limit": "Input availability only; no reach metric was computed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--terminal-index", type=Path, required=True)
    parser.add_argument("--object6d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.task_packet.resolve(strict=True)
    packet = load(packet_path)
    if packet.get("task_id") != TASK_ID:
        raise ContractError(f"Task Packet must be {TASK_ID}")
    output = args.output.resolve()
    write_roots = [Path(value).resolve() for value in packet.get("write_set", [])]
    if not any(output.is_relative_to(root) for root in write_roots):
        raise ContractError("output outside declared write_set")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    index_path = args.terminal_index.resolve(strict=True)
    read_roots = [Path(value).resolve() for value in packet.get("read_set", [])]
    if index_path not in read_roots:
        raise ContractError("terminal index is outside declared read_set")
    object_root = args.object6d_root.resolve(strict=True)
    if not any(object_root == root or object_root.is_relative_to(root) for root in read_roots):
        raise ContractError("Object6D root is outside declared read_set")
    terminal_index = load(index_path)
    if terminal_index.get("status") != "PASSED_TERMINAL_COVERAGE":
        raise ContractError("complete v77 terminal index required")
    hard_rows = [row for row in terminal_index.get("rows", []) if row.get("terminal_status") == "PASSED"]
    if len(hard_rows) != 13:
        raise ContractError(f"expected 13 v77 hard-pass rows, got {len(hard_rows)}")
    rows = [audit_row(row, object_root) for row in hard_rows]
    poker_ready = sorted(row["session"] for row in rows if row["task"] == "poker" and row["all_metric_inputs_ready"])
    fully_ready = sum(row["all_metric_inputs_ready"] for row in rows)
    field_counts = {
        field: sum(row["metric_input_readiness"][field] for row in rows)
        for field in ("action_amplitude", "wrist_object_approach", "ik_frame_valid_rate")
    }
    output.mkdir(parents=True)
    result = {
        "schema_version": "robot-v78-prerequisite-closure-result-v1",
        "terminal_status": "PASSED",
        "status": "PASSED_CPU_ONLY_EVIDENCE_AVAILABILITY_AUDIT",
        "counts": {"v77_hard_pass": 13, "all_metric_inputs_ready": fully_ready, **field_counts},
        "legal_poker_regression_candidates": poker_ready,
        "rows": rows,
        "computation_performed": "METADATA_AND_SHAPE_AUDIT_ONLY",
        "solver_run": False,
        "metric_values_computed": False,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Evidence availability audit only; no V78 reach metric, optimizer result, or Robot authority.",
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    write_json(output / "METRICS.json", {"schema_version": "robot-v78-prerequisite-metrics-v1", **result["counts"], "legal_poker_regression_candidates": poker_ready})
    write_json(output / "RUN_RECEIPT.json", {"schema_version": "robot-v78-prerequisite-receipt-v1", "result": ref(result_path), "task_packet": ref(packet_path), "terminal_index": ref(index_path)})
    write_json(output / "ARTIFACT_MANIFEST.json", {"schema_version": "artifact-manifest-v1", "artifacts": [ref(result_path), ref(packet_path), ref(index_path)]})
    write_new(output / "DECISION.md", (
        f"# Decision\n\nAudited all 13 v77 hard-pass sessions without running a solver. "
        f"{fully_ready} contain all raw inputs needed for the three future reach metrics. "
        f"Legal Poker regression candidates: {poker_ready}. This does not compute or authorize V78.\n"
    ).encode())
    write_json(output / "NEXT_ACTION.json", {
        "next_task_id": "robot_v78_selection_revision_and_metric_extractor",
        "status": "READY" if poker_ready else "BLOCKED_PREREQ",
        "start_solver": False,
        "legal_poker_regression_candidates": poker_ready,
        "requirements": ["CAS-freeze exactly one listed Poker regression", "register a causal metric extractor/optimizer", "preserve all v77 hard gates"],
    })
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(json.dumps({"legal_poker_regression_candidates": poker_ready}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
