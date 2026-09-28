"""Refresh old common-contract Local/HuRo metrics, preserving failure denominators."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/compare/common_old_contract_v1"
R2 = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
V5 = ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
CASES = {
    "get_potato_chips_0915_007": ("recovered_007_wave0", "007"),
    "play_cards_0915_031": ("recovered_031_wave0", "031"),
}


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _stats(values: np.ndarray, frame_id: np.ndarray, valid: np.ndarray, gate: float) -> dict:
    t, side = np.where(valid & np.isfinite(values))
    if not len(t):
        return {"count": 0, "missing_or_invalid": int(valid.size), "p50": None,
                "p95": None, "max": None, "max_frame": None, "max_side": None}
    selected = values[t, side]
    top = int(np.argmax(selected))
    return {"count": len(selected), "timeline_side_slots": int(valid.size),
            "missing_or_invalid": int(valid.size - len(selected)),
            "p50": float(np.percentile(selected, 50)), "p95": float(np.percentile(selected, 95)),
            "max": float(selected[top]), "max_frame": int(frame_id[t[top]]), "max_side": int(side[top]),
            "gate": gate, "gate_fail_count": int((selected > gate).sum())}


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    reports = []
    for session, (local_name, short) in CASES.items():
        local_path = R2 / f"lanes/lane2_motion/{local_name}/ROBOT_R0_V1.npz"
        huro_path = V5 / f"lanes/huro/full_0001/{session}/HURO_CORE_V1.npz"
        with np.load(local_path, allow_pickle=False) as archive:
            local = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(huro_path, allow_pickle=False) as archive:
            huro = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("frame_id", "timestamp_ns", "target_valid", "T_target_root_cam", "T_cam_base", "T_flange_hand", "human_to_physical"):
            if not np.array_equal(local[key], huro[key], equal_nan=True):
                raise ValueError(f"COMMON_TARGET_OR_MOUNT_DRIFT:{session}:{key}")
        valid = np.asarray(local["wrist_valid"], bool) & np.asarray(huro["wrist_valid"], bool)
        frame_id = np.asarray(local["frame_id"])
        difference = np.linalg.norm(local["actual21_camera_m"] - huro["actual21_camera_m"], axis=-1) * 1000
        curve_path = OUT / f"CURVES_{short}.npz"
        np.savez_compressed(curve_path, frame_id=frame_id, timestamp_ns=local["timestamp_ns"],
                            common_valid=valid, local_position_mm=local["position_residual_mm"],
                            huro_position_mm=huro["position_residual_mm"],
                            local_full_rotation_deg=local["rotation_residual_deg"],
                            huro_full_rotation_deg=huro["rotation_residual_deg"],
                            fk21_difference_mm=difference)
        positions = np.flatnonzero(valid.any(axis=1))
        reports.append({"session_id": session, "frame_count": len(frame_id),
                        "first_common_valid_frame": int(frame_id[positions[0]]) if len(positions) else None,
                        "common_valid_side_frames": int(valid.sum()),
                        "local_valid_side_frames": int(np.asarray(local["wrist_valid"], bool).sum()),
                        "huro_valid_side_frames": int(np.asarray(huro["wrist_valid"], bool).sum()),
                        "local_position_mm": _stats(local["position_residual_mm"], frame_id, valid, 20.),
                        "huro_position_mm": _stats(huro["position_residual_mm"], frame_id, valid, 20.),
                        "local_full_rotation_deg": _stats(local["rotation_residual_deg"], frame_id, valid, 15.),
                        "huro_full_rotation_deg": _stats(huro["rotation_residual_deg"], frame_id, valid, 15.),
                        "method_fk21_difference_mm": {
                            "p50": float(np.nanpercentile(difference[valid], 50)),
                            "p95": float(np.nanpercentile(difference[valid], 95))},
                        "curves": artifact_ref(curve_path),
                        "inputs": {"local": artifact_ref(local_path), "huro": artifact_ref(huro_path)},
                        "same_target_mount_and_evaluator": True,
                        "winner": None})
    result = {"schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_COMMON_COMPARE_V1",
              "task_id": TASK, "execution": "REUSED_SOLVES_REEVALUATED", "quality": "INCONCLUSIVE_NO_INDEPENDENT_TRUTH",
              "adoption": "NOT_ADOPTED", "new_solver_invocations": 0, "sessions": reports,
              "claim_limit": "Old common-target numeric contract only; maxima and denominators expose failures, not a winner or new solver improvement.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "sessions": [{"id": row["session_id"],
        "first_valid": row["first_common_valid_frame"], "local_max_frame": row["local_position_mm"]["max_frame"],
        "local_max_mm": row["local_position_mm"]["max"]} for row in reports],
        "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
