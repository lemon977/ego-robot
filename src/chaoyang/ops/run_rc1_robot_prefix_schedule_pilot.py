#!/usr/bin/env python3
"""Produce causal Robot terminal states for a frozen list of H50 starts.

Each scheduled start is recomputed from the prefix ``0..t``.  The program
never reads a bidirectional/full-trajectory Robot artifact and only publishes
the state at ``t``.  This is a production-shape pilot, not Robot authority:
collision/render/compositor eligibility remains a separate downstream gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
ARM = PROJECT / "src/chaoyang/ops/run_robot_motion_transfer_arm_canary_v3.py"
HAND = PROJECT / "src/chaoyang/ops/run_robot_hand_fullsession_v2.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha(resolved)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def slice_prefix(source: dict[str, np.ndarray], frame_count: int, end: int) -> dict[str, np.ndarray]:
    count = end + 1
    output: dict[str, np.ndarray] = {}
    for key, value in source.items():
        if value.ndim >= 2 and value.shape[0] == 2 and value.shape[1] == frame_count:
            output[key] = np.asarray(value[:, :count]).copy()
        elif value.ndim >= 1 and value.shape[0] == frame_count:
            output[key] = np.asarray(value[:count]).copy()
        else:
            output[key] = np.asarray(value).copy()
    return output


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run(command: list[str], log: Path) -> tuple[int, float]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"solver runtime failure rc={completed.returncode}; see {log}")
    return completed.returncode, time.monotonic() - started


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--accepted-states", type=Path, required=True)
    parser.add_argument("--accepted-hawor", type=Path, required=True)
    parser.add_argument("--task-base-backoff-m", type=float, required=True)
    parser.add_argument("--starts", default="15,20,25")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    starts = sorted({int(value) for value in args.starts.split(",") if value.strip()})
    if not starts:
        raise ValueError("at least one scheduled start is required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    with np.load(args.hawor.resolve(strict=True), allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    frame_count = int(len(source["c2w"]))
    if any(start < 15 or start > frame_count - 51 or (start - 15) % 5 for start in starts):
        raise ValueError(f"starts violate RC1 schedule or H50 tail: starts={starts}, frame_count={frame_count}")

    rows = []
    for start in starts:
        # The pinned arm/hand executors intentionally enforce same-session
        # provenance by checking the input path.  Keep the immutable session
        # identity in every prefix attempt path.
        start_root = args.output_root / "starts" / args.session / f"frame_{start:06d}"
        start_root.mkdir(parents=True)
        prefix = slice_prefix(source, frame_count, start)
        hawor_path = start_root / "HAWOR_PREFIX.npz"
        write_npz(hawor_path, prefix)
        hawor_result = start_root / "HAWOR_PREFIX_RESULT.json"
        atomic_json(hawor_result, {
            "status": "PASS_PREFIX_FIXTURE", "session": args.session, "prefix_end": start,
            "input_mode": "CAUSAL_TRAINING_INPUT", "source_hawor": ref(args.hawor),
            "claim_limit": "Prefix fixture only; not HaWoR authority.",
        })
        arm_result = start_root / "arm" / "RESULT.json"; arm_result.parent.mkdir()
        arm_rc, arm_s = run([
            sys.executable, str(ARM), "--task", args.task, "--session", args.session,
            "--hawor", str(hawor_path), "--hawor-result", str(hawor_result),
            "--accepted-states", str(args.accepted_states), "--accepted-hawor", str(args.accepted_hawor),
            "--task-base-backoff-m", str(args.task_base_backoff_m), "--output", str(arm_result),
        ], start_root / "arm.log")
        hand_result = start_root / "hand" / "RESULT.json"; hand_result.parent.mkdir()
        hand_rc, hand_s = run([
            sys.executable, str(HAND), "--task", args.task, "--session", args.session,
            "--hawor", str(hawor_path), "--hawor-result", str(hawor_result),
            "--accepted-states", str(args.accepted_states), "--output", str(hand_result),
        ], start_root / "hand.log")
        arm = load(arm_result); hand = load(hand_result)
        arm_state_path = arm_result.with_name("ARM_CANARY_STATES.npz")
        hand_state_path = hand_result.with_name("HAND_STATES.npz")
        with np.load(arm_state_path, allow_pickle=False) as archive:
            terminal = {key: np.asarray(archive[key])[start] for key in (
                "q_arm", "T_target_hand_root_world", "T_actual_hand_root_world")}
            terminal["valid_side_frame"] = np.asarray(archive["valid_side_frame"])[:, start]
        with np.load(hand_state_path, allow_pickle=False) as archive:
            terminal["q_hand"] = np.asarray(archive["q_hand"])[start]
            hand_valid = np.asarray(archive["valid_side_frame"])[:, start]
        if not np.array_equal(terminal["valid_side_frame"], hand_valid):
            raise RuntimeError(f"arm/hand validity mismatch at start {start}")
        terminal_path = start_root / "ROBOT_TERMINAL_STATE.npz"
        write_npz(terminal_path, terminal)
        hard_structural = {
            "finite_on_valid_sides": bool(
                np.isfinite(terminal["q_arm"][terminal["valid_side_frame"]]).all()
                and np.isfinite(terminal["q_hand"][terminal["valid_side_frame"]]).all()
            ),
            "arm_temporal_contract": all(arm.get("gates", {}).get(key) is True for key in (
                "arm_velocity", "arm_acceleration", "missing_frames_unknown_not_filled")),
            "hand_temporal_contract": all(hand.get("gates", {}).get(key) is True for key in (
                "velocity", "acceleration", "missing_unknown_not_filled",
                "thumb_independent_q0_to_q5", "four_finger_chain_semantics")),
        }
        rows.append({
            "session": args.session, "task": args.task, "scheduled_start": start,
            "status": "PASSED_PREFIX_STATE_NO_COLLISION_GATE" if all(hard_structural.values()) else "FAILED_QUALITY_C",
            "hard_structural": hard_structural,
            "soft_pose": {
                "arm_pose_all_observed": arm.get("gates", {}).get("all_observed_rows_pose_branch"),
                "hand_anatomy_all_observed": hand.get("gates", {}).get("anatomy_all_observed"),
            },
            "collision_gate": "NOT_EVALUATED",
            "visual_aux_rc1_eligible": False,
            "arm_returncode": arm_rc, "hand_returncode": hand_rc,
            "wall_seconds": arm_s + hand_s,
            "terminal_state": ref(terminal_path), "arm_result": ref(arm_result), "hand_result": ref(hand_result),
        })

    passed = sum(row["status"].startswith("PASSED") for row in rows)
    result = {
        "schema_version": "chaoyang-rc1-robot-prefix-schedule-pilot-v1",
        "task_id": "rc1_robot30_prefix_schedule_pilot", "created_at": now(),
        "status": "PASSED_DEVELOPMENT" if passed == len(rows) else "FAILED_QUALITY_C",
        "session": args.session, "task": args.task, "frame_count": frame_count,
        "scheduled_starts": starts, "counts": {"starts": len(rows), "prefix_state_pass": passed},
        "rows": rows,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "collision_gate_complete": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "lineage": {
            "hawor": ref(args.hawor), "hawor_result": ref(args.hawor_result),
            "accepted_states": ref(args.accepted_states), "accepted_hawor": ref(args.accepted_hawor),
            "arm_runner": ref(ARM), "hand_runner": ref(HAND),
        },
        "claim_limit": "Causal scheduled-start Robot state pilot only. Collision/render/compositor gates are absent, so no window is training-eligible and no Robot authority is granted.",
    }
    result_path = args.output_root / "RESULT.json"; atomic_json(result_path, result)
    atomic_json(args.output_root / "METRICS.json", {"counts": result["counts"], "wall_seconds": [row["wall_seconds"] for row in rows]})
    atomic_json(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": result["status"], "session": args.session, "counts": result["counts"], "next_action": "ADD_DIGITAL_COLLISION_GATE_THEN_FREEZE_PILOT_P95"})
    atomic_json(args.output_root / "RUN_RECEIPT.json", {"status": result["status"], "result": ref(result_path), "created_at": now()})
    atomic_json(args.output_root / "ARTIFACT_MANIFEST.json", {"result": ref(result_path), "terminals": [row["terminal_state"] for row in rows]})
    (args.output_root / "DECISION.md").write_text("# RC1 Robot scheduled-start pilot\n\nOnly causal terminal states are produced. Collision/render/compositor gates remain required before training eligibility.\n", encoding="utf-8")
    atomic_json(args.output_root / "NEXT_ACTION.json", {"next": "ADD_DIGITAL_COLLISION_GATE_THEN_FREEZE_PILOT_P95"})
    print(json.dumps({"status": result["status"], "result": ref(result_path), "counts": result["counts"]}, ensure_ascii=False))
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
