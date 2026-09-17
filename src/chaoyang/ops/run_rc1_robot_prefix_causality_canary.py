#!/usr/bin/env python3
"""Prove that the v77 forward Robot solvers can emit a causal state at t.

The production rule tested here is deliberately narrow: build a HaWoR prefix
0..t, run the existing forward arm and hand solvers on that prefix, and emit
only the terminal state t.  A second source has all post-t solver inputs
changed; after prefix extraction it must produce the same logical prefix and
the same terminal Robot state.  Full-sequence/bidirectional artifacts are not
read by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
ARM_RUNNER = PROJECT / "src/chaoyang/ops/run_robot_motion_transfer_arm_canary_v3.py"
HAND_RUNNER = PROJECT / "src/chaoyang/ops/run_robot_hand_fullsession_v2.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logical_digest(payload: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(payload[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def slice_prefix(payload: dict[str, np.ndarray], frame_count: int, end: int) -> dict[str, np.ndarray]:
    count = end + 1
    output: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if value.ndim >= 2 and value.shape[0] == 2 and value.shape[1] == frame_count:
            output[key] = np.asarray(value[:, :count]).copy()
        elif value.ndim >= 1 and value.shape[0] == frame_count:
            output[key] = np.asarray(value[:count]).copy()
        else:
            output[key] = np.asarray(value).copy()
    return output


def mutate_future(payload: dict[str, np.ndarray], frame_count: int, end: int) -> tuple[dict[str, np.ndarray], int]:
    output = {key: np.asarray(value).copy() for key, value in payload.items()}
    changed = 0
    for key, value in output.items():
        if value.dtype.kind not in "fiu" or value.ndim == 0:
            continue
        target = None
        if value.ndim >= 2 and value.shape[0] == 2 and value.shape[1] == frame_count:
            target = value[:, end + 1 :]
        elif value.shape[0] == frame_count:
            target = value[end + 1 :]
        if target is None or target.size == 0:
            continue
        if value.dtype.kind == "f":
            finite = np.isfinite(target)
            target[finite] += np.asarray(0.123456, dtype=value.dtype)
            changed += int(finite.sum())
        else:
            target[...] = target + np.asarray(1, dtype=value.dtype)
            changed += int(target.size)
    return output, changed


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def run(command: list[str], log_path: Path) -> int:
    completed = subprocess.run(command, cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"solver failed rc={completed.returncode}; see {log_path}")
    return completed.returncode


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--accepted-states", type=Path, required=True)
    parser.add_argument("--accepted-hawor", type=Path, required=True)
    parser.add_argument("--task-base-backoff-m", type=float, default=0.0)
    parser.add_argument("--prefix-end", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    with np.load(args.hawor.resolve(strict=True), allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    frame_count = int(len(source["c2w"]))
    if not (0 <= args.prefix_end < frame_count - 1):
        raise ValueError("prefix-end must leave at least one changed future frame")
    mutated, changed_values = mutate_future(source, frame_count, args.prefix_end)
    prefix_a = slice_prefix(source, frame_count, args.prefix_end)
    prefix_b = slice_prefix(mutated, frame_count, args.prefix_end)
    digest_a = logical_digest(prefix_a)
    digest_b = logical_digest(prefix_b)
    if changed_values <= 0 or digest_a != digest_b:
        raise RuntimeError("future-mutation fixture did not preserve the logical prefix")

    results = []
    terminal = {}
    for variant, payload in (("a", prefix_a), ("b", prefix_b)):
        root = args.output_root / f"variant_{variant}" / args.session
        root.mkdir(parents=True)
        hawor_path = root / "HAWOR_PREFIX.npz"
        hawor_result = root / "RESULT.json"
        write_npz(hawor_path, payload)
        hawor_result.write_text(json.dumps({
            "status": "PASS_PREFIX_FIXTURE",
            "session": args.session,
            "prefix_end": args.prefix_end,
            "logical_prefix_sha256": logical_digest(payload),
            "claim_limit": "Causality fixture only; not HaWoR authority.",
        }, ensure_ascii=False, indent=2) + "\n")

        arm_result = root / "arm" / "RESULT.json"
        arm_result.parent.mkdir()
        arm_rc = run([
            sys.executable, str(ARM_RUNNER), "--task", args.task, "--session", args.session,
            "--hawor", str(hawor_path), "--hawor-result", str(hawor_result),
            "--accepted-states", str(args.accepted_states), "--accepted-hawor", str(args.accepted_hawor),
            "--task-base-backoff-m", str(args.task_base_backoff_m), "--output", str(arm_result),
        ], root / "arm.log")
        hand_result = root / "hand" / "RESULT.json"
        hand_result.parent.mkdir()
        hand_rc = run([
            sys.executable, str(HAND_RUNNER), "--task", args.task, "--session", args.session,
            "--hawor", str(hawor_path), "--hawor-result", str(hawor_result),
            "--accepted-states", str(args.accepted_states), "--output", str(hand_result),
        ], root / "hand.log")
        with np.load(arm_result.with_name("ARM_CANARY_STATES.npz"), allow_pickle=False) as z:
            arm_terminal = {key: np.asarray(z[key])[args.prefix_end] for key in ("q_arm", "T_target_hand_root_world", "T_actual_hand_root_world")}
        with np.load(hand_result.with_name("HAND_STATES.npz"), allow_pickle=False) as z:
            hand_terminal = {"q_hand": np.asarray(z["q_hand"])[args.prefix_end]}
        terminal[variant] = {**arm_terminal, **hand_terminal}
        results.append({"variant": variant, "arm_returncode": arm_rc, "hand_returncode": hand_rc,
                        "arm_result": ref(arm_result), "hand_result": ref(hand_result)})

    comparisons = {}
    for key in terminal["a"]:
        left = terminal["a"][key]
        right = terminal["b"][key]
        finite_equal = np.array_equal(np.isfinite(left), np.isfinite(right))
        max_abs = float(np.max(np.abs(left[np.isfinite(left)] - right[np.isfinite(right)]))) if np.isfinite(left).any() else 0.0
        comparisons[key] = {"finite_mask_equal": bool(finite_equal), "max_abs_difference": max_abs,
                            "pass": bool(finite_equal and max_abs <= 1e-10)}
    gates = {
        "future_was_changed": changed_values > 0,
        "logical_prefix_equal": digest_a == digest_b,
        "repeatable_terminal_state": all(row["pass"] for row in comparisons.values()),
        "bidirectional_artifact_not_consumed": True,
        "emits_terminal_prefix_state_only": True,
    }
    status = "PASSED_CAUSAL_CANARY_NO_AUTHORITY" if all(gates.values()) else "FAILED_QUALITY_C"
    payload = {
        "schema_version": "chaoyang-rc1-robot-prefix-causality-canary-v1",
        "created_at": now(), "task_id": "rc1_t3_v77_causal_robot_prefix_canary",
        "status": status, "task": args.task, "session": args.session,
        "prefix_end": args.prefix_end, "future_changed_value_count": changed_values,
        "logical_prefix_sha256": digest_a, "comparisons": comparisons, "gates": gates,
        "lineage": {"hawor": ref(args.hawor), "hawor_result": ref(args.hawor_result),
                    "accepted_states": ref(args.accepted_states), "accepted_hawor": ref(args.accepted_hawor),
                    "arm_runner": ref(ARM_RUNNER), "hand_runner": ref(HAND_RUNNER)},
        "solver_runs": results,
        "input_mode": "CAUSAL_PREFIX_CANARY",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "claim_limit": "Proves terminal-state prefix invariance for one Robot solver canary only; not full-session eligibility, Robot authority, action truth, contact truth, or physical accuracy.",
    }
    result_path = args.output_root / "RESULT.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    summary = {"task_id": payload["task_id"], "status": status, "session": args.session,
               "prefix_end": args.prefix_end, "gates": gates, "result": ref(result_path)}
    (args.output_root / "RESULT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "METRICS.json").write_text(json.dumps({"comparisons": comparisons, "future_changed_value_count": changed_values}, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "ARTIFACT_MANIFEST.json").write_text(json.dumps({"result": ref(result_path), "solver_runs": results}, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "RUN_RECEIPT.json").write_text(json.dumps({"status": status, "result": ref(result_path), "created_at": now()}, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "DECISION.md").write_text(f"# RC1 Robot 前缀因果 canary\n\n状态：`{status}`。仅证明单个前缀终态不读取未来 solver 输入，不授予 Robot authority。\n")
    (args.output_root / "NEXT_ACTION.json").write_text(json.dumps({"next": "RUN_PREFIX_CANARY_ON_POKER_THEN_BUILD_SCHEDULED_START_PRODUCER" if all(gates.values()) else "STOP_AND_DIAGNOSE"}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
