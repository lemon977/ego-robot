#!/usr/bin/env python3
"""Measure H50 eligibility before rendering expensive paired RGB bundles."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.human_ego.tools.build_visual_aux_session_bundle_v54 import arrays, exact, exact_from  # noqa: E402
from chaoyang.human_ego.tools.build_visual_retarget_projection_candidate_v52 import (  # noqa: E402
    future_arrays,
    project,
)
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as robot  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def source_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source video: {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def assess(
    path: Path,
    assets: Any,
    eligibility_mode: str = "BOTH_ENDPOINTS_40_OF_50",
) -> dict[str, Any]:
    review = load(path)
    numeric_review = review.get("status") == "PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
    pose_only = review.get("status") == "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY"
    if not (numeric_review or pose_only):
        raise RuntimeError(f"numeric-pass or pose-only visual review required: {path}")
    count = int(review["frame_count"])
    if pose_only:
        if (
            review.get("authority") is not False
            or review.get("control_ground_truth") is not False
            or review.get("metric_object_geometry") is not False
            or review.get("contact_frame_valid") is not False
        ):
            raise RuntimeError("pose-only claim boundary mismatch")
        root = path.parent
        sidecar = arrays(exact_from(root, review["outputs"]["trajectory_sidecar"], "trajectory sidecar"))
        lineage_path = exact_from(root, review["outputs"]["lineage"], "pose-only lineage")
        lineage = load(lineage_path)
        chain = lineage["input_manifest"]["hawor_chain"]
        temporal = Path(chain["temporal_output"]["path"])
        hawor_ref = chain["temporal_output"] if temporal.is_file() else chain["frozen_npz"]
        hawor = arrays(exact(hawor_ref, "HaWoR camera metadata"))
        source = exact(lineage["input_manifest"]["source_video"], "source video")
        roots = np.asarray(sidecar["T_actual_hand_root_world"], dtype=np.float64)
        side_valid = np.asarray(sidecar["valid_side_frame"], dtype=bool)
        if roots.shape != (count, 2, 4, 4) or side_valid.shape != (count, 2):
            raise RuntimeError("pose-only trajectory shape mismatch")
        source_mode = "POSE_ONLY_VISUAL_ROBOT"
    else:
        lineage = review["lineage"]
        hawor = arrays(exact(lineage["hawor_npz"], "HaWoR temporal"))
        arm = arrays(exact(lineage["arm_states"], "arm states"))
        source = exact(lineage["source_video"], "source video")
        world_base = np.asarray(arm["T_world_base"], dtype=np.float64)
        if world_base.ndim == 3:
            if not np.allclose(world_base, world_base[0], atol=1e-12, rtol=0):
                raise RuntimeError("Robot base moved within session")
            world_base = world_base[0]
        mounts = np.asarray(arm["T_tool_hand_root"], dtype=np.float64)
        roots = np.full((count, 2, 4, 4), np.nan, dtype=np.float64)
        for frame in range(count):
            values = {name: 0.0 for names in robot.official.ARM_JOINT_NAMES for name in names}
            for side in range(2):
                values.update(dict(zip(robot.official.ARM_JOINT_NAMES[side], arm["q_arm"][frame, side], strict=True)))
            fk = robot.official.forward_kinematics(assets.tianji, values)
            roots[frame, 0] = world_base @ fk["left_tool"] @ mounts[0]
            roots[frame, 1] = world_base @ fk["right_tool"] @ mounts[1]
        side_valid = np.ones((count, 2), dtype=bool)
        source_mode = "NUMERIC_RENDER_REVIEW"
    width, height = source_size(source)
    _, point_valid = project(
        np.asarray(hawor["c2w"], dtype=np.float64),
        np.asarray(hawor["intrinsics"], dtype=np.float64),
        roots,
        side_valid,
        width,
        height,
    )
    dummy_uv = np.zeros((count, 2, 2), dtype=np.float32)
    _, _, current_valid, starts = future_arrays(dummy_uv, point_valid, eligibility_mode)
    return {
        "task": review["task"],
        "session": review["session"],
        "review": artifact_ref(path),
        "frame_count": count,
        "trajectory_mode": source_mode,
        "source_domain": {"width": width, "height": height},
        "projected_valid_frames": {
            "physical_left": int(point_valid[:, 0].sum()),
            "physical_right": int(point_valid[:, 1].sum()),
            "both": int(point_valid.all(axis=1).sum()),
            "any": int(point_valid.any(axis=1).sum()),
            "current_mode_valid": int(current_valid.sum()),
        },
        "eligible_h50_window_count": len(starts),
        "h50_eligibility_mode": eligibility_mode,
        "status": "READY" if starts else "BLOCKED_PREREQ_ZERO_H50_WINDOWS",
        "claim_limit": "Projection eligibility only; not a training, Robot, contact, or physical-accuracy result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--h50-eligibility-mode",
        choices=("BOTH_ENDPOINTS_40_OF_50", "ANY_ENDPOINT_40_OF_50"),
        default="BOTH_ENDPOINTS_40_OF_50",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable output exists: {output}")
    assets = robot.load_pinned_robot_assets(PROJECT)
    rows = [assess(path.resolve(strict=True), assets, args.h50_eligibility_mode) for path in args.review_result]
    atomic_json(
        output,
        {
            "schema_version": "exact78-visual-aux-h50-preflight-v55-v1",
            "status": "PASS_PREFLIGHT_COMPLETE",
            "rows": rows,
            "counts": {
                "candidates": len(rows),
                "ready": sum(row["status"] == "READY" for row in rows),
                "blocked_zero_windows": sum(row["status"] == "BLOCKED_PREREQ_ZERO_H50_WINDOWS" for row in rows),
                "eligible_h50_windows": sum(int(row["eligible_h50_window_count"]) for row in rows),
            },
            "claim_limit": "CPU eligibility preflight only; no paired RGB or checkpoint was produced.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
