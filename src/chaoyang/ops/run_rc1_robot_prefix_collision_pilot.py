#!/usr/bin/env python3
"""Audit pinned digital self-collision at causal scheduled-start states."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.ops.audit_robot_geometry_self_collision_v71 import audit


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha(value)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    prefix = load(args.prefix_result)
    rows = []
    for row in prefix.get("rows", []):
        start = int(row["scheduled_start"])
        arm_result = Path(row["arm_result"]["path"])
        hand_result = Path(row["hand_result"]["path"])
        arm_path = arm_result.with_name("ARM_CANARY_STATES.npz").resolve(strict=True)
        hand_path = hand_result.with_name("HAND_STATES.npz").resolve(strict=True)
        with np.load(arm_path, allow_pickle=False) as source:
            arm = {key: np.asarray(source[key]) for key in source.files}
        with np.load(hand_path, allow_pickle=False) as source:
            hand = {key: np.asarray(source[key]) for key in source.files}
        if start >= len(arm["q_arm"]) or start >= len(hand["q_hand"]):
            raise RuntimeError(f"scheduled start outside prefix state: {start}")
        root = args.output_root / f"frame_{start:06d}"
        root.mkdir(parents=True)
        arm_one = root / "ARM_TERMINAL_STATE.npz"
        hand_one = root / "HAND_TERMINAL_STATE.npz"
        write_npz(
            arm_one,
            q_arm=arm["q_arm"][start:start + 1],
            T_world_base=arm["T_world_base"],
            T_tool_hand_root=arm["T_tool_hand_root"],
            T_target_hand_root_world=arm["T_target_hand_root_world"][start:start + 1],
            T_actual_hand_root_world=arm["T_actual_hand_root_world"][start:start + 1],
            valid_side_frame=arm["valid_side_frame"][:, start:start + 1],
            source_frames=arm["source_frames"][start:start + 1],
        )
        write_npz(
            hand_one,
            q_hand=hand["q_hand"][start:start + 1],
            valid_side_frame=hand["valid_side_frame"][:, start:start + 1],
            source_frames=hand["source_frames"][start:start + 1],
        )
        terminal_valid = np.asarray(arm["valid_side_frame"][:, start], dtype=bool)
        if not bool(np.all(terminal_valid)):
            rows.append({
                "session": prefix["session"], "task": prefix["task"], "scheduled_start": start,
                "status": "BLOCKED_PREREQ_BILATERAL_COLLISION_AUDIT",
                "valid_sides": terminal_valid.tolist(),
                "reason": "Pinned collision auditor requires two finite arm/hand states; the missing side stays UNKNOWN and is not replaced with an invented pose.",
                "collision": None,
            })
            continue
        value = audit(
            session_id=str(prefix["session"]), arm_path=arm_one, hand_path=hand_one,
            frame_count=1, penetration_tolerance_m=1e-4,
        )
        collision_path = root / "RESULT.json"; atomic_json(collision_path, value)
        passed = value.get("gates", {}).get("non_adjacent_self_intersection_absent") is True
        rows.append({
            "session": prefix["session"], "task": prefix["task"], "scheduled_start": start,
            "status": "PASSED_DIGITAL_COLLISION" if passed else "FAILED_QUALITY_C",
            "illegal_contact_count": value.get("illegal_contact_count"),
            "max_penetration_m": value.get("max_penetration_m"),
            "collision": ref(collision_path),
        })
    passed = sum(row["status"] == "PASSED_DIGITAL_COLLISION" for row in rows)
    quality_failed = sum(row["status"] == "FAILED_QUALITY_C" for row in rows)
    blocked = sum(row["status"].startswith("BLOCKED_PREREQ") for row in rows)
    status = "FAILED_QUALITY_C" if quality_failed else ("PASSED_DEVELOPMENT" if rows and passed == len(rows) else "PARTIAL_BLOCKED_PREREQ")
    result = {
        "schema_version": "chaoyang-rc1-robot-prefix-collision-pilot-v1",
        "task_id": "rc1_robot30_digital_collision_pilot", "created_at": now(),
        "status": status, "task": prefix["task"], "session": prefix["session"],
        "counts": {"starts": len(rows), "digital_collision_pass": passed, "blocked_prereq": blocked, "failed_quality_c": quality_failed}, "rows": rows,
        "input": ref(args.prefix_result),
        "render_gate_complete": False, "compositor_gate_complete": False,
        "training_eligible": False, "authority_promoted": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "claim_limit": "Pinned URDF digital self-collision at causal terminal states only; no object contact, camera z-buffer, control truth or physical collision accuracy.",
    }
    result_path = args.output_root / "RESULT.json"; atomic_json(result_path, result)
    atomic_json(args.output_root / "METRICS.json", {"status": status, "counts": result["counts"]})
    atomic_json(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": status, "session": prefix["session"], "counts": result["counts"], "next_action": "RENDER_AND_COMPOSITOR_PILOT" if passed == len(rows) else "STOP_FAILED_QUALITY_C"})
    atomic_json(args.output_root / "RUN_RECEIPT.json", {"status": status, "result": ref(result_path), "created_at": now()})
    atomic_json(args.output_root / "ARTIFACT_MANIFEST.json", {"result": ref(result_path), "collisions": [row["collision"] for row in rows if row.get("collision")]})
    (args.output_root / "DECISION.md").write_text("# Digital collision pilot\n\nThis checks pinned URDF self-collision only; it is not physical collision truth.\n", encoding="utf-8")
    atomic_json(args.output_root / "NEXT_ACTION.json", {"next": "RENDER_AND_COMPOSITOR_PILOT" if passed == len(rows) else "STOP_FAILED_QUALITY_C"})
    print(json.dumps({"status": status, "result": ref(result_path), "counts": result["counts"]}, ensure_ascii=False))
    return 2 if quality_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
