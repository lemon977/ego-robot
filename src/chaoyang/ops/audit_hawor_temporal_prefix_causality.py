#!/usr/bin/env python3
"""Audit whether the current HaWoR temporal successor uses future samples.

This is a development diagnostic, not an authority publisher.  It rebuilds
the exact full-sequence temporal proposal and compares it with independently
rebuilt prefixes of the same bounded input.  The comparison is performed on
the parameter outputs before Robot IK, so a difference identifies the HaWoR
temporal stage rather than downstream Robot behavior.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops import run_hawor_bounded_parameter_successor as base_tool

# The historical temporal runner is also a direct CLI and imports its sibling
# by basename.  Add the repository tools directory so the exact production
# implementation can be audited without rewriting it.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from chaoyang.ops import run_hawor_temporal_jerk_successor as temporal


PARAMETERS = ("root_translation_camera", "root_orient_camera", "hand_pose_rotmat", "betas")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        return {key: np.asarray(value[key]) for key in value.files}


def _prefix_track(track: dict[str, np.ndarray], end_exclusive: int) -> dict[str, np.ndarray]:
    frame_count = int(track["observed"].shape[1])
    if not 4 <= end_exclusive <= frame_count:
        raise ValueError(f"invalid prefix end {end_exclusive}/{frame_count}")
    output: dict[str, np.ndarray] = {}
    for key in (*PARAMETERS, "observed", "detector_confidence"):
        value = track[key]
        if value.ndim < 2 or value.shape[1] != frame_count:
            raise RuntimeError(f"unexpected time axis for {key}: {value.shape}")
        output[key] = value[:, :end_exclusive].copy()
    return output


def _apply(track: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    return base_tool.interpolate_parameters(track, temporal.proposal(track), alpha)


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    flattened = relative.reshape(-1, 3, 3)
    finite = np.isfinite(flattened).all(axis=(1, 2))
    output = np.full(len(flattened), np.nan, dtype=np.float64)
    if finite.any():
        output[finite] = np.linalg.norm(
            Rotation.from_matrix(flattened[finite]).as_rotvec(), axis=1
        ) * 180.0 / np.pi
    return output.reshape(relative.shape[:-2])


def compare_prefix(full: dict[str, np.ndarray], prefix: dict[str, np.ndarray], observed: np.ndarray) -> dict[str, Any]:
    n = prefix["root_translation_camera"].shape[1]
    mask = observed[:, :n]
    translation = np.linalg.norm(full["root_translation_camera"][:, :n] - prefix["root_translation_camera"], axis=-1) * 1000.0
    root_rotation = _rotation_angle_deg(full["root_orient_camera"][:, :n], prefix["root_orient_camera"])
    pose_rotation = _rotation_angle_deg(full["hand_pose_rotmat"][:, :n], prefix["hand_pose_rotmat"])
    beta = np.linalg.norm(full["betas"][:, :n] - prefix["betas"], axis=-1)

    def max_valid(values: np.ndarray, valid: np.ndarray) -> float:
        chosen = values[valid & np.isfinite(values)]
        return float(np.max(chosen)) if chosen.size else 0.0

    terminal_mask = mask[:, -1]
    metrics = {
        "prefix_frames": n,
        "observed_rows": int(mask.sum()),
        "root_translation_max_mm": max_valid(translation, mask),
        "root_translation_terminal_mm": max_valid(translation[:, -1], terminal_mask),
        "root_rotation_max_deg": max_valid(root_rotation, mask),
        "root_rotation_terminal_deg": max_valid(root_rotation[:, -1], terminal_mask),
        "pose_rotation_max_deg": max_valid(pose_rotation, np.repeat(mask[..., None], pose_rotation.shape[2], axis=2)),
        "pose_rotation_terminal_deg": max_valid(pose_rotation[:, -1], np.repeat(terminal_mask[..., None], pose_rotation.shape[2], axis=1)),
        "betas_l2_max": max_valid(beta, mask),
    }
    metrics["prefix_invariant"] = all(
        metrics[key] <= threshold
        for key, threshold in (
            ("root_translation_max_mm", 1e-6),
            ("root_rotation_max_deg", 1e-8),
            ("pose_rotation_max_deg", 1e-8),
            ("betas_l2_max", 1e-10),
        )
    )
    return metrics


def audit(input_npz: Path, temporal_npz: Path, temporal_result: Path, cutoffs: list[int]) -> dict[str, Any]:
    result = json.loads(temporal_result.read_text(encoding="utf-8"))
    alpha = float(result["selected_alpha"])
    raw = _load_npz(input_npz)
    saved = _load_npz(temporal_npz)
    full = _apply(raw, alpha)
    reconstruction = {}
    for key in PARAMETERS:
        reconstruction[key] = float(np.nanmax(np.abs(full[key] - saved[key])))
    rows = []
    for cutoff in cutoffs:
        prefix_raw = _prefix_track(raw, cutoff)
        prefix = _apply(prefix_raw, alpha)
        rows.append(compare_prefix(full, prefix, raw["observed"]))
    any_future_dependency = any(not row["prefix_invariant"] for row in rows)
    return {
        "schema_version": "chaoyang-hawor-temporal-prefix-causality-audit-v1",
        "created_at": now_iso(),
        "status": "DEVELOPMENT_EVIDENCE_FUTURE_DEPENDENCY" if any_future_dependency else "PASSED_PREFIX_INVARIANCE_DIAGNOSTIC",
        "session": result.get("session"),
        "task": result.get("task"),
        "selected_alpha": alpha,
        "frame_count": int(raw["observed"].shape[1]),
        "cutoffs_end_exclusive": cutoffs,
        "comparisons": rows,
        "recomputed_full_vs_saved_max_abs": reconstruction,
        "full_reconstruction_matches_saved": all(value <= 1e-6 for value in reconstruction.values()),
        "future_dependency_detected": any_future_dependency,
        "inputs": {
            "bounded_input": artifact_ref(input_npz),
            "temporal_output": artifact_ref(temporal_npz),
            "temporal_result": artifact_ref(temporal_result),
            "temporal_code": artifact_ref(Path(temporal.__file__).resolve()),
        },
        "authority_promoted": False,
        "claim_limit": "Real-track parameter-stage causality diagnostic only; it does not measure external hand accuracy or quantify downstream Robot error.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--temporal-npz", type=Path, required=True)
    parser.add_argument("--temporal-result", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    value = audit(args.input_npz, args.temporal_npz, args.temporal_result, args.cutoff)
    atomic_json(args.output_root / "RESULT.json", value)
    atomic_json(args.output_root / "METRICS.json", {"comparisons": value["comparisons"]})
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
