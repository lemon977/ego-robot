#!/usr/bin/env python3
"""Frozen three-frame Poker245 causal-atlas eligibility audit.

The audit does not tune scale or thresholds.  It checks the exact size semantics
implemented by the producing Object6D code, projects that geometry into the
selected RGB domain, and compares it with the independently frozen instance
mask for frames 0/32/98.  A doubled-size interpretation is reported only as a
disallowed diagnostic because the producer explicitly uses +/-0.5 * size_m.
"""

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
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import unions
from chaoyang.ops.run_clean20_cpu_contract_closure_r22 import mask_iou_for_quad, project_card_quad

FRAMES = (0, 32, 98)
BASE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
FRAME_MANIFEST = BASE / "clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json"
OBJECT_MASK_MANIFEST = BASE / "mask_task_object_identity_v1/poker/play_cards_0903_245/OBJECT_MASK_MANIFEST.json"
OBJECT6D_RESULT = BASE / "depth_object6d_stream_v1/object6d_observed_only_v1/poker/play_cards_0903_245/RESULT.json"
OBJECT6D_FRAME_MANIFEST = BASE / "depth_object6d_stream_v1/object6d_observed_only_v1/poker/play_cards_0903_245/FRAME_MANIFEST.json"
ADAPTER_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/selected_camera_adapter_batch_R7_3/poker/play_cards_0903_245/physical_object_0/RESULT.json"
HAWOR = BASE / "hawor_bounded_v2/poker/play_cards_0903_245/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
PRODUCER = ROOT / "src/chaoyang/ops/run_visual_fixed_instance_object6d.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path, verify: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if verify is not None and any(value[key] != verify[key] for key in value):
        raise RuntimeError(f"reference mismatch: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def face_label(provider: str) -> str:
    if "BACK" in provider:
        return "CARD_BACK_VISIBLE"
    if "REVEALED_FACE" in provider:
        return "CARD_FACE_VISIBLE"
    return "UNKNOWN"


def project_metrics(mask: np.ndarray, T: np.ndarray, K: np.ndarray, size: np.ndarray) -> dict[str, Any]:
    uv, points_camera = project_card_quad(T, K, size)
    iou, coverage = mask_iou_for_quad(mask, uv)
    mask_y, mask_x = np.where(mask)
    mask_center = np.array([mask_x.mean(), mask_y.mean()])
    return {
        "quad_xy": uv.tolist(),
        "iou": iou,
        "mask_coverage": coverage,
        "centroid_error_px": float(np.linalg.norm(uv.mean(axis=0) - mask_center)),
        "all_points_in_front": bool(np.all(points_camera[:, 2] > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    visuals = out / "visuals"
    visuals.mkdir()

    task_packet = ref(args.task_packet)
    rows = json.loads(FRAME_MANIFEST.read_text(encoding="utf-8"))["frames"]
    object_manifest = json.loads(OBJECT_MASK_MANIFEST.read_text(encoding="utf-8"))
    object_rows = object_manifest["frames"]
    object6d_result = json.loads(OBJECT6D_RESULT.read_text(encoding="utf-8"))
    object6d_rows = json.loads(OBJECT6D_FRAME_MANIFEST.read_text(encoding="utf-8"))["frames"]
    adapter_result = json.loads(ADAPTER_RESULT.read_text(encoding="utf-8"))
    trajectory_path = Path(object6d_result["artifacts"]["trajectory"]["path"])
    adapter_path = Path(adapter_result["outputs"][0]["path"])
    ref(trajectory_path, object6d_result["artifacts"]["trajectory"])
    ref(adapter_path, adapter_result["outputs"][0])
    with np.load(trajectory_path, allow_pickle=False) as legacy, np.load(adapter_path, allow_pickle=False) as adapter, np.load(HAWOR, allow_pickle=False) as hand:
        size = np.asarray(legacy["object_size_m"], np.float64)
        valid = np.asarray(legacy["valid"], bool)
        observed = np.asarray(legacy["observed"], bool)
        instance = np.asarray(legacy["physical_instance_id"], np.int32)
        transforms = np.asarray(adapter["T_object_to_selected_camera"], np.float64)
        adapter_valid = np.asarray(adapter["valid"], bool)
        intrinsics = np.asarray(hand["intrinsics"], np.float64)

    frame_reports = []
    for frame_id in FRAMES:
        if not (valid[frame_id] and observed[frame_id] and adapter_valid[frame_id] and instance[frame_id] == 0):
            raise RuntimeError(f"frame {frame_id} is not same-instance DIRECT_OBJECT6D")
        raw = cv2.imread(rows[frame_id]["source_rgb"]["path"], cv2.IMREAD_COLOR)
        mask = unions(rows[frame_id])[2]
        authoritative = project_metrics(mask, transforms[frame_id], intrinsics[frame_id], size)
        # Diagnostic only.  The producer proves this interpretation wrong.
        wrong_half_extent = project_metrics(mask, transforms[frame_id], intrinsics[frame_id], size * 2.0)
        provider = str(object_rows[frame_id]["provider"])
        normal_dot = float(np.dot(transforms[frame_id][:3, 2], -transforms[frame_id][:3, 3]))
        object6d_row = object6d_rows[frame_id]
        observed_span = float(object6d_row["near_far_observed_m"][1] - object6d_row["near_far_observed_m"][0])
        analytic_span = float(object6d_row["near_far_analytic_m"][1] - object6d_row["near_far_analytic_m"][0])

        overlay = raw.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
        full_quad = np.rint(np.asarray(authoritative["quad_xy"])).astype(np.int32)
        wrong_quad = np.rint(np.asarray(wrong_half_extent["quad_xy"])).astype(np.int32)
        cv2.polylines(overlay, [full_quad], True, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.polylines(overlay, [wrong_quad], True, (0, 165, 255), 2, cv2.LINE_AA)
        label = face_label(provider)
        cv2.putText(overlay, f"frame={frame_id}  visual={label}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"FULL-SIZE quad IoU={authoritative['iou']:.3f} (required aggregate P50>=0.70/P10>=0.50)", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"orange=DISALLOWED 2x diagnostic IoU={wrong_half_extent['iou']:.3f}", (24, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2, cv2.LINE_AA)
        path = visuals / f"frame_{frame_id:06d}_size_pose_face_audit.png"
        if not cv2.imwrite(str(path), overlay):
            raise RuntimeError(f"cannot write {path}")
        frame_reports.append({
            "frame_id": frame_id,
            "same_instance_id": 0,
            "evidence_type": "DIRECT_OBJECT6D",
            "allowed_target_frame_min": frame_id,
            "future_used": False,
            "attachment_used": False,
            "old_clean_used": False,
            "object_mask_provider": provider,
            "independent_visible_side_label": label,
            "authoritative_full_dimensions": authoritative,
            "disallowed_half_extent_interpretation_2x": wrong_half_extent,
            "object6d_plane_median_abs_residual_mm": float(object6d_row["plane_median_abs_residual_m"] * 1000.0),
            "observed_depth_span_mm": observed_span * 1000.0,
            "analytic_card_depth_span_mm": analytic_span * 1000.0,
            "pose_normal_dot_camera_vector": normal_dot,
            "pose_normal_sign": "CAMERA_FACING_CONVENTION",
            "visual": ref(path),
        })

    ious = np.array([row["authoritative_full_dimensions"]["iou"] for row in frame_reports])
    p50, p10 = float(np.percentile(ious, 50)), float(np.percentile(ious, 10))
    projection_pass = p50 >= 0.70 and p10 >= 0.50
    visual_sides = [row["independent_visible_side_label"] for row in frame_reports]
    visible_side_labels_closed = visual_sides == ["CARD_BACK_VISIBLE", "CARD_BACK_VISIBLE", "CARD_FACE_VISIBLE"]
    # The producer flips the fitted normal toward the camera every frame.  It is
    # therefore not a persistent object-side axis and cannot bind front/back.
    persistent_pose_face_mapping = False
    audit = {
        "schema_version": "poker245-min-atlas-canary-audit-r22-v1",
        "session": "play_cards_0903_245",
        "frames": list(FRAMES),
        "object_size_m": size.tolist(),
        "size_semantics": {
            "status": "CLOSED_FULL_DIMENSIONS",
            "producer": ref(PRODUCER),
            "code_contract": "GEOMETRY supplies [0.088,0.063,0.001]; fit_pose uses corners in {-0.5,+0.5} * size_m, so values are full dimensions, not half extents.",
            "doubling_size_authorized": False,
        },
        "projection_gate": {
            "thresholds_unchanged": True,
            "required_iou_p50": 0.70,
            "required_iou_p10": 0.50,
            "observed_iou_p50": p50,
            "observed_iou_p10": p10,
            "pass": projection_pass,
        },
        "face_identity": {
            "frozen_visual_side_labels_closed_for_three_frames": visible_side_labels_closed,
            "labels": visual_sides,
            "persistent_object_pose_axis_to_physical_face_closed": persistent_pose_face_mapping,
            "reason": "Object6D fit_pose forces the fitted plane normal toward the camera (normal[2] <= 0), so both card back and revealed face receive the same camera-facing normal convention.",
        },
        "depth_diagnostic": "Observed optical-Z spans are far larger than the 1 mm card thickness and the analytic pose spans; this is internal evidence of contaminated/unstable visible-surface pose support, not external accuracy.",
        "frame_reports": frame_reports,
        "atlas_eligibility": False,
        "status": "BLOCKED_PREREQ",
        "claim_limit": "Three-frame size/pose/face contract audit only; no atlas pixels, Clean authority or physical truth.",
    }
    write_json(out / "POKER245_MIN_ATLAS_CANARY_AUDIT.json", audit)

    result = {
        "schema_version": "poker245-min-atlas-canary-result-r22-v1",
        "task_id": "poker245_min_atlas_canary_r22",
        "attempt_id": out.name,
        "status": "BLOCKED_PREREQ",
        "generated_at": now(),
        "task_packet": task_packet,
        "audit": ref(out / "POKER245_MIN_ATLAS_CANARY_AUDIT.json"),
        "atlas_eligibility": False,
        "atlas_published": False,
        "clean_authority_promoted": False,
        "propainter_started": False,
        "future_used": False,
        "attachment_used": False,
        "old_clean_used": False,
        "claim_limit": audit["claim_limit"],
    }
    metrics = {
        "schema_version": "poker245-min-atlas-canary-metrics-r22-v1",
        "status": result["status"],
        "frames": frame_reports,
        "gates": {
            "full_dimension_semantics_closed": True,
            "thresholds_unchanged": True,
            "projection_iou_p50_pass": p50 >= 0.70,
            "projection_iou_p10_pass": p10 >= 0.50,
            "visual_side_label_closed_for_three_frames": visible_side_labels_closed,
            "persistent_pose_face_mapping_closed": persistent_pose_face_mapping,
            "future_use_zero": True,
            "attachment_use_zero": True,
            "old_clean_use_zero": True,
        },
    }
    write_json(out / "RESULT.json", result)
    write_json(out / "METRICS.json", metrics)
    write_json(out / "RUN_RECEIPT.json", {
        "schema_version": "poker245-min-atlas-canary-run-receipt-r22-v1",
        "task_id": result["task_id"], "attempt_id": out.name,
        "status": result["status"], "created_at": now(), "host": socket.gethostname(),
        "pid": os.getpid(), "gpu_lease": "NOT_ACQUIRED_CPU_ONLY", "authority_promoted": False,
    })
    write_json(out / "NEXT_ACTION.json", {
        "schema_version": "poker245-min-atlas-canary-next-r22-v1",
        "status": result["status"],
        "next": [
            "Repair or replace the DIRECT_OBJECT6D plane-pose estimator using depth-quality rejection and an image-supported card quadrilateral/PnP canary.",
            "Publish a persistent object-frame face-axis contract that does not flip the normal toward the camera every frame.",
            "Re-run these same frozen frames and unchanged 0.70/0.50 gates before accepting any atlas pixel."
        ],
        "do_not": ["Do not double object_size_m.", "Do not lower IoU gates.", "Do not use future frames, Attachment, old Clean or ProPainter."],
    })
    write_text(out / "DECISION.md", f"""# Poker245 三帧最小 Atlas Canary 决定\n\n状态：`BLOCKED_PREREQ`。\n\n`object_size_m=[0.088, 0.063, 0.001]` 的语义已由生产代码闭合为**完整尺寸**：代码以 `±0.5 × size_m` 构造角点，不允许为了提高 IoU 把尺寸翻倍。\n\n冻结帧 0/32/98 的公称尺寸投影 IoU 分别为 `{ious[0]:.4f}`、`{ious[1]:.4f}`、`{ious[2]:.4f}`；P50=`{p50:.4f}`、P10=`{p10:.4f}`，没有通过原门槛 0.70/0.50。门槛未调整。\n\n独立 Object Mask manifest 可以区分帧0/32为牌背、帧98为翻开牌面；但是 Object6D 生产代码每帧都把平面法向翻到朝向相机，因此 pose 中的法向不是固定的物体正反面轴，不能将两套纹理安全绑定到 canonical front/back atlas。\n\n三帧的 observed optical-Z 范围也明显大于 1 mm 牌厚和 analytic pose 范围，支持“当前平面深度/姿态受污染或不稳定”的内部诊断，但不构成外部精度结论。\n\n本任务没有使用未来帧、Attachment、旧 Clean，没有运行 ProPainter，也没有发布 atlas 或晋升 Clean authority。\n""")
    artifacts = {
        "task_packet": task_packet,
        "result": ref(out / "RESULT.json"),
        "metrics": ref(out / "METRICS.json"),
        "run_receipt": ref(out / "RUN_RECEIPT.json"),
        "decision": ref(out / "DECISION.md"),
        "next_action": ref(out / "NEXT_ACTION.json"),
        "audit": ref(out / "POKER245_MIN_ATLAS_CANARY_AUDIT.json"),
    }
    for row in frame_reports:
        artifacts[f"frame_{row['frame_id']}_visual"] = row["visual"]
    write_json(out / "ARTIFACT_MANIFEST.json", {
        "schema_version": "poker245-min-atlas-canary-artifact-manifest-r22-v1",
        "task_id": result["task_id"], "status": result["status"],
        "artifacts": artifacts, "authority_promoted": False,
    })
    print(json.dumps({"status": result["status"], "result": ref(out / "RESULT.json"), "iou_p50": p50, "iou_p10": p10}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
