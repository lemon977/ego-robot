#!/usr/bin/env python3
"""Run the bounded HuRo-derived Kai22 hand-only comparison producer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    AUTHORITY,
    METHOD_ID,
    SCHEMA_VERSION,
    load_hand_model,
    solve_sequence,
)


PROJECT = Path(__file__).resolve().parents[3]
HURO_ROOT = PROJECT / "vendor/HuRo"
HURO_COMMIT = "033197778fcc30edc3631dddf3343a967683da09"
KAI_URDFS = (
    PROJECT
    / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)"
    / "urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    PROJECT
    / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)"
    / "urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
)


class RunError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def percentile(values: np.ndarray, q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.percentile(finite, q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor-npz", required=True, type=Path)
    parser.add_argument("--local-r0-npz", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-evaluations", type=int, default=80)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise RunError(f"refusing to overwrite output: {args.output_dir}")
    if not args.hawor_npz.is_file() or args.hawor_npz.is_symlink():
        raise RunError("ordinary HaWoR NPZ required")
    if args.local_r0_npz is not None and (
        not args.local_r0_npz.is_file() or args.local_r0_npz.is_symlink()
    ):
        raise RunError("ordinary local R0 NPZ required")
    required_vendor = (HURO_ROOT / "LICENSE", HURO_ROOT / "THIRD_PARTY_NOTICES.md")
    if any(not path.is_file() for path in required_vendor):
        raise RunError("pinned HuRo vendor source is incomplete")
    with np.load(args.hawor_npz, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("joints_3d_camera", "observed", "original_frame_indices", "fps"):
        if key not in hawor:
            raise RunError(f"HaWoR input missing {key}")
    hands = (
        load_hand_model(KAI_URDFS[0], "left"),
        load_hand_model(KAI_URDFS[1], "right"),
    )
    solved = solve_sequence(
        hands,
        hawor["joints_3d_camera"],
        hawor["observed"],
        max_evaluations=args.max_evaluations,
    )
    frames = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
    fps = float(np.asarray(hawor["fps"]).item())
    solved["source_frame_id"] = frames
    solved["timestamp_s"] = frames.astype(np.float64) / fps
    args.output_dir.mkdir(parents=True)
    state_path = args.output_dir / "HURO_DERIVED_HAND_ONLY_STATES.npz"
    np.savez_compressed(state_path, **solved)

    comparison: dict[str, Any] = {"status": "NOT_PROVIDED"}
    if args.local_r0_npz is not None:
        with np.load(args.local_r0_npz, allow_pickle=False) as archive:
            local = {key: np.asarray(archive[key]) for key in archive.files}
        if "q22" not in local or local["q22"].shape != solved["q22"].shape:
            raise RunError("local R0 q22 shape mismatch")
        common = solved["valid"].copy()
        if "q22_computed" in local:
            common &= np.asarray(local["q22_computed"], dtype=bool)
        difference_deg = np.degrees(local["q22"] - solved["q22"])
        comparison = {
            "status": "COMMON_FRAME_DIFFERENCE_ONLY_NOT_ACCURACY",
            "common_side_frames": int(np.sum(common)),
            "q_difference_rms_deg": (
                float(np.sqrt(np.mean(difference_deg[common] ** 2))) if np.any(common) else None
            ),
            "q_difference_p95_abs_deg": percentile(np.abs(difference_deg[common]), 95),
            "local_r0": evidence(args.local_r0_npz),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "authority": AUTHORITY,
        "status": "PASSED_DEVELOPMENT_HAND_ONLY_PRODUCTION",
        "session_id": args.session,
        "frame_count": int(len(frames)),
        "valid_side_frames": int(np.sum(solved["valid"])),
        "physical_left_valid": int(np.sum(solved["valid"][:, 0])),
        "physical_right_valid": int(np.sum(solved["valid"][:, 1])),
        "solver_metrics": {
            "local_direction_rms_p50": percentile(solved["local_direction_rms"], 50),
            "local_direction_rms_p95": percentile(solved["local_direction_rms"], 95),
            "tip_position_rms_mm_p50": (
                None
                if percentile(solved["tip_position_rms_m"], 50) is None
                else 1000.0 * float(percentile(solved["tip_position_rms_m"], 50))
            ),
            "temporal_delta_rms_rad_p95": percentile(
                solved["temporal_delta_rms_rad"], 95
            ),
        },
        "comparison": comparison,
        "inputs": {
            "hawor": evidence(args.hawor_npz),
            "kaihand_left_urdf": evidence(KAI_URDFS[0]),
            "kaihand_right_urdf": evidence(KAI_URDFS[1]),
            "huro_license": evidence(HURO_ROOT / "LICENSE"),
            "huro_third_party_notices": evidence(HURO_ROOT / "THIRD_PARTY_NOTICES.md"),
            "huro_upstream_commit": HURO_COMMIT,
        },
        "outputs": {"states": evidence(state_path)},
        "claims": {
            "official_full_huro_reproduction": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
            "stage9_native_overlay": "BLOCKED_HARDWARE_LICENSE",
            "stage10": "NOT_RUN_OUT_OF_SCOPE",
            "wuji20": "BLOCKED_ROBOT_ASSET",
        },
        "claim_limit": (
            "Root-relative Kai22 hand-shape development comparison only. It does not provide "
            "arm, mount, wrist/world, contact, control, deployment or external accuracy authority."
        ),
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

