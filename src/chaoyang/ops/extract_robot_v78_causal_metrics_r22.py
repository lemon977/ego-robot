#!/usr/bin/env python3
"""Extract read-only Robot reach diagnostics from frozen NPZ trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


TASK_ID = "robot_v78_causal_metric_extractor_r22"
SIDES = ("left", "right")


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_new(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def _finite_transform(value: np.ndarray) -> bool:
    return value.shape == (4, 4) and bool(np.isfinite(value).all())


def extract_metrics(arm: dict[str, np.ndarray], obj: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_arm = {"q_arm", "T_actual_hand_root_world", "valid_side_frame", "source_frames"}
    required_object = {"frame_indices", "valid", "observed", "T_object_to_world"}
    if not required_arm <= set(arm):
        raise ContractError(f"arm NPZ missing: {sorted(required_arm - set(arm))}")
    if not required_object <= set(obj):
        raise ContractError(f"Object6D NPZ missing: {sorted(required_object - set(obj))}")
    q = np.asarray(arm["q_arm"], dtype=np.float64)
    wrist = np.asarray(arm["T_actual_hand_root_world"], dtype=np.float64)
    valid = np.asarray(arm["valid_side_frame"], dtype=bool)
    frames = np.asarray(arm["source_frames"], dtype=np.int64)
    if q.ndim != 3 or q.shape[1:] != (2, 7):
        raise ContractError(f"unexpected q_arm shape: {q.shape}")
    if wrist.shape != (q.shape[0], 2, 4, 4) or valid.shape != (2, q.shape[0]) or frames.shape != (q.shape[0],):
        raise ContractError("arm arrays are not frame-aligned")
    if len(set(int(frame) for frame in frames)) != len(frames):
        raise ContractError("duplicate source_frames")

    object_frames = np.asarray(obj["frame_indices"], dtype=np.int64)
    object_valid = np.asarray(obj["valid"], dtype=bool)
    object_observed = np.asarray(obj["observed"], dtype=bool)
    object_pose = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    if object_valid.shape != object_frames.shape or object_observed.shape != object_frames.shape or object_pose.shape != (len(object_frames), 4, 4):
        raise ContractError("Object6D arrays are not frame-aligned")
    object_by_frame = {
        int(frame): index for index, frame in enumerate(object_frames)
        if object_valid[index] and object_observed[index] and _finite_transform(object_pose[index])
    }

    rows: list[dict[str, Any]] = []
    effective_valid = np.zeros_like(valid, dtype=bool)
    q_totals = [0.0, 0.0]
    path_totals = [0.0, 0.0]
    consecutive_counts = [0, 0]
    distances: list[list[tuple[int, float]]] = [[], []]
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        object_index = object_by_frame.get(frame)
        row: dict[str, Any] = {
            "source_frame": frame,
            "object_direct_observed_valid": object_index is not None,
        }
        for side_index, side in enumerate(SIDES):
            current_valid = bool(valid[side_index, index]) and bool(np.isfinite(q[index, side_index]).all()) and _finite_transform(wrist[index, side_index])
            effective_valid[side_index, index] = current_valid
            row[f"{side}_ik_valid"] = current_valid
            q_step = None
            wrist_step = None
            if index > 0 and frame == int(frames[index - 1]) + 1:
                previous_valid = bool(valid[side_index, index - 1]) and bool(np.isfinite(q[index - 1, side_index]).all()) and _finite_transform(wrist[index - 1, side_index])
                if current_valid and previous_valid:
                    q_step = float(np.abs(q[index, side_index] - q[index - 1, side_index]).sum())
                    wrist_step = float(np.linalg.norm(wrist[index, side_index, :3, 3] - wrist[index - 1, side_index, :3, 3]))
                    q_totals[side_index] += q_step
                    path_totals[side_index] += wrist_step
                    consecutive_counts[side_index] += 1
            row[f"{side}_q_consecutive_total_variation_step_rad"] = q_step
            row[f"{side}_actual_wrist_step_m"] = wrist_step
            distance = None
            if current_valid and object_index is not None:
                distance = float(np.linalg.norm(wrist[index, side_index, :3, 3] - object_pose[object_index, :3, 3]))
                distances[side_index].append((frame, distance))
            row[f"{side}_wrist_object_distance_m"] = distance
        rows.append(row)

    side_metrics: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        series = distances[side_index]
        first = series[0][1] if series else None
        last = series[-1][1] if series else None
        minimum = min((value for _, value in series), default=None)
        side_metrics[side] = {
            "ik_valid_frames": int(effective_valid[side_index].sum()),
            "ik_valid_rate": float(effective_valid[side_index].mean()),
            "consecutive_valid_step_count": consecutive_counts[side_index],
            "q_arm_consecutive_valid_total_variation_rad": q_totals[side_index],
            "actual_wrist_consecutive_valid_path_length_m": path_totals[side_index],
            "wrist_object_same_frame_count": len(series),
            "wrist_object_first_distance_m": first,
            "wrist_object_last_distance_m": last,
            "wrist_object_min_distance_m": minimum,
            "wrist_object_delta_first_to_last_m": None if first is None else last - first,
            "wrist_object_approach_delta_m": None if first is None else first - minimum,
            "approach_delta_definition": "first_same_frame_distance_minus_minimum_later_or_same_distance; positive means closer at least once",
        }
    summary = {
        "frame_count": int(len(frames)),
        "source_frame_min": int(frames.min()),
        "source_frame_max": int(frames.max()),
        "ik_valid_rate_all_side_frames": float(effective_valid.mean()),
        "q_arm_consecutive_valid_total_variation_rad_all_sides": float(sum(q_totals)),
        "actual_wrist_consecutive_valid_path_length_m_all_sides": float(sum(path_totals)),
        "sides": side_metrics,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--arm-states", type=Path, required=True)
    parser.add_argument("--object-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.task_packet.resolve(strict=True)
    packet = load_json(packet_path)
    if packet.get("task_id") != TASK_ID:
        raise ContractError(f"Task Packet must be {TASK_ID}")
    selection_path = args.selection.resolve(strict=True)
    selection = load_json(selection_path)
    if selection.get("session") != "play_cards_0902_031" or selection.get("start_solver") is not False:
        raise ContractError("frozen Poker031 read-only selection required")
    read_set = {str(Path(path).resolve()) for path in packet.get("read_set", [])}
    input_paths = [selection_path, args.arm_states.resolve(strict=True), args.object_states.resolve(strict=True)]
    if any(str(path) not in read_set for path in input_paths):
        raise ContractError("input outside exact Task Packet read_set")
    output = args.output.resolve()
    write_roots = [Path(path).resolve() for path in packet.get("write_set", [])]
    if not any(output.is_relative_to(root) for root in write_roots):
        raise ContractError("output outside Task Packet write_set")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    with np.load(input_paths[1], allow_pickle=False) as archive:
        arm = {key: archive[key] for key in archive.files}
    with np.load(input_paths[2], allow_pickle=False) as archive:
        obj = {key: archive[key] for key in archive.files}
    rows, summary = extract_metrics(arm, obj)
    output.mkdir(parents=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    frame_csv = output / "FRAME_METRICS.csv"
    atomic_new(frame_csv, csv_buffer.getvalue().encode())
    inputs = {"task_packet": artifact_ref(packet_path), "selection": artifact_ref(selection_path), "arm_states": artifact_ref(input_paths[1]), "object_states": artifact_ref(input_paths[2])}
    result = {
        "schema_version": "robot-v78-causal-metric-result-v1",
        "terminal_status": "PASSED",
        "status": "PASSED_OFFLINE_DIAGNOSTIC_EXTRACTION",
        "session": selection["session"],
        "summary": summary,
        "frame_metrics": artifact_ref(frame_csv),
        "inputs": inputs,
        "solver_run": False,
        "optimizer_input": False,
        "aggregation_scope": "OFFLINE_FULL_SEQUENCE_DIAGNOSTIC_NOT_OPTIMIZER_INPUT",
        "causal_training_placement_changed": False,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Offline diagnostic metrics from frozen trajectories; not optimization input, control truth, or physical accuracy.",
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(output / "METRICS.json", {"schema_version": "robot-v78-causal-metrics-v1", **summary})
    metrics_path = output / "METRICS.json"
    receipt_path = output / "RUN_RECEIPT.json"
    decision_path = output / "DECISION.md"
    next_action_path = output / "NEXT_ACTION.json"
    atomic_json(receipt_path, {"schema_version": "robot-v78-causal-metric-receipt-v1", "result": artifact_ref(result_path), "inputs": inputs})
    atomic_new(decision_path, (
        "# Decision\n\nExtracted full-sequence diagnostics from frozen Poker031 trajectories. "
        "The aggregate values are offline reports only and were not used to choose placement, "
        "optimize a trajectory, relax a hard gate, or modify training input.\n"
    ).encode())
    atomic_json(next_action_path, {"next_task_id": "robot_v78_causal_target_reach_canary_implementation_r22", "status": "READY_FOR_REGISTERED_IMPLEMENTATION", "start_solver": False, "required_mode": "TASK_PRESET_OR_START_PREFIX_ONLY"})
    atomic_json(
        output / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "artifact-manifest-v1",
            "artifacts": [
                artifact_ref(path)
                for path in (result_path, metrics_path, receipt_path, decision_path, next_action_path, frame_csv)
            ],
            "inputs": list(inputs.values()),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
