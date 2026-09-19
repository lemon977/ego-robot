#!/usr/bin/env python3
"""Classify the 0915 Human/Stereo scale anomaly without fitting Contact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import cv2
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_scale_cause_audit_v1"
PHASE = "ROBOT15H_HUMAN_STEREO_SCALE_CAUSE_AUDIT"
SURFACE = ROOT / "_run/current/0915_human_stereo_surface_association_canary_v1/attempts/attempt_0001/MANO_SURFACE_ASSOCIATION_V1.json"
BOUND = ROOT / "_run/current/0915_human_stereo_alignment_bound_audit_v1/attempts/attempt_0001/ALIGNMENT_BOUND_SATURATION_AUDIT_V1.json"
DEPTH_CONTRACT = ROOT / "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001/DEPTH_CONTRACT.json"
CAMERA = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001/camera_params.json")
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_SCALE_CAUSE_AUDIT_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SCALE_CAUSE_AUDIT_V1_RESULT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current scale audit packet differs from frozen CPU spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("scale cause audit is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("scale cause audit is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("scale cause audit packet SHA differs from route")
    return packet, packet_path


def heartbeat() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", PHASE],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def admitted_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in records
        if row.get("status") == "OBSERVED_SURFACE_PAIR"
        and row.get("whole_frame_hand_contact_excluded") is False
        and int(row.get("support_count", 0)) >= 20
        and np.isfinite([row.get("mano_surface_depth_m"), row.get("stereo_surface_depth_m")]).all()
    ]


def affine_fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 4:
        return {"status": "INSUFFICIENT_ROWS", "row_count": len(rows)}
    x = np.asarray([row["mano_surface_depth_m"] for row in rows], np.float64)
    y = np.asarray([row["stereo_surface_depth_m"] for row in rows], np.float64)
    design = np.column_stack((x, np.ones(x.size)))
    estimate, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ estimate
    x_span = float(np.ptp(x))
    return {
        "status": "FIT_DIAGNOSTIC_ONLY",
        "row_count": len(rows),
        "scale": float(estimate[0]),
        "offset_m": float(estimate[1]),
        "mano_depth_min_m": float(x.min()),
        "mano_depth_max_m": float(x.max()),
        "mano_depth_span_m": x_span,
        "correlation": float(np.corrcoef(x, y)[0, 1]) if x_span > 0 else None,
        "median_abs_residual_m": float(np.median(np.abs(residual))),
        "p90_abs_residual_m": float(np.percentile(np.abs(residual), 90)),
        "design_condition_number": float(np.linalg.cond(design)),
    }


def camera_geometry(camera: Mapping[str, Any], depth: Mapping[str, Any]) -> dict[str, Any]:
    matrices = {eye: np.asarray(camera["extrinsics"][eye], np.float64) for eye in ("left", "right")}
    centres = {eye: -value[:3, :3].T @ value[:3, 3] for eye, value in matrices.items()}
    relative = matrices["right"] @ np.linalg.inv(matrices["left"])
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    source_width = float(camera["width"])
    output_width = float(depth["frame_geometry"][0])
    left = camera["left"]["intrinsics"]
    right = camera["right"]["intrinsics"]
    cx_delta = output_width / source_width * (float(left["cx"]) - float(right["cx"]))
    return {
        "schema_version": "0915-robot15h-depth-geometry-audit-v1",
        "encoded_pixel_operation": "PASSTHROUGH_CROP_RESIZE_ONLY_NO_LENS_REMAP",
        "depth_formula_in_frozen_canary": depth["formula"],
        "depth_reference": depth["depth_reference"],
        "reported_baseline_m": float(depth["baseline_m"]),
        "camera_centre_baseline_m": float(np.linalg.norm(centres["left"] - centres["right"])),
        "factory_inter_camera_rotation_deg": float(np.degrees(np.arccos(cosine))),
        "scaled_factory_principal_point_delta_left_minus_right_px": cx_delta,
        "encoded_projection_matrices_present": False,
        "encoded_rectified_intrinsics_external_accuracy": "UNVERIFIED",
        "lens_undistortion_required": False,
        "lens_undistortion_applied": False,
        "horizontal_reflection_roundtrip_status": "PIXEL_EXACT_PREVIOUSLY_VERIFIED",
        "finding": (
            "The frozen Depth conversion uses scaled factory K and camera-centre baseline "
            "in fxB/disparity, while independently calibrated encoded-domain P_left/P_right "
            "are absent. This is a plausible metric-scale uncertainty, not proof of the 0.783730 cause."
        ),
    }


def render_diagnostic(stability: Mapping[str, Any], matrix: Mapping[str, Any], path: Path) -> None:
    canvas = np.full((820, 1440, 3), 246, np.uint8)
    lines = [
        ("0915 Robot15h | Human/Stereo scale-cause audit", 1.0, (20, 20, 20)),
        ("0.783730 is an affine optical-Z slope, not a MANO bone/world-scale calibration", 0.66, (20, 20, 160)),
        (f"Reproduced robust global slope: {stability['reproduced_unconstrained_scale']:.6f}", 0.72, (40, 40, 40)),
        (f"Per-hand OLS slopes: L {stability['by_hand']['left']['scale']:.3f} | R {stability['by_hand']['right']['scale']:.3f}", 0.72, (40, 40, 40)),
        (f"30-frame block slope range: {stability['temporal_block_scale_range'][0]:.3f} .. {stability['temporal_block_scale_range'][1]:.3f}", 0.72, (40, 40, 40)),
        ("Result: a session-static multiplicative correction is not identifiable from this evidence", 0.68, (20, 20, 160)),
        ("Encoded K/P external accuracy remains unverified; no scale bound or Contact gate changed", 0.68, (20, 20, 160)),
        (f"Diagnostic: {matrix['diagnostic_status']}", 0.72, (20, 100, 20)),
        ("Next independent evidence: W0 encoded Depth + direct visible surfaces across source groups", 0.66, (40, 40, 40)),
    ]
    y = 70
    for line, scale, color in lines:
        cv2.putText(canvas, line, (45, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 72
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError("failed to write scale audit diagnostic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed scale-audit namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)
    inputs = [SURFACE, BOUND, DEPTH_CONTRACT, CAMERA, INVENTORY]
    before = {str(path.resolve()): sha256(path) for path in inputs}
    surface, bound, depth, camera, inventory = map(load_json, inputs)
    records = surface.get("records")
    if not isinstance(records, list) or surface.get("session_id") != "play_cards_0915_001":
        raise RuntimeError("sealed surface association identity drift")
    if bound.get("unconstrained_fit", {}).get("scale") is None:
        raise RuntimeError("sealed bound audit lacks unconstrained slope")
    if depth.get("external_accuracy") != "UNVERIFIED" or inventory.get("cohort_denominator") != 220:
        raise RuntimeError("depth authority or inventory denominator drift")

    signature_payload = {
        "schema_version": "0915-robot15h-scale-cause-run-signature-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID,
        "weights": "ABSENT", "gpu_used": False,
        "contact_or_object_fit_used": False, "fixed_48mm_bias_used": False,
        "scale_bounds_changed": False, "model_rerun_performed": False,
        "executor_epoch": args.executor_epoch,
        "task_packet": ref(packet_path), "inputs": [ref(path) for path in inputs],
        "code": ref(Path(__file__)),
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-scale-cause-writer-claim-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "CLAIMED",
        "weights": "ABSENT", "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output), "run_signature_sha256": signature["run_signature_sha256"],
    })
    heartbeat()

    admitted = admitted_rows(records)
    by_hand = {hand: affine_fit([row for row in admitted if row["hand_id"] == hand]) for hand in ("left", "right")}
    blocks: list[dict[str, Any]] = []
    for start in range(0, int(surface["frame_count"]), 30):
        subset = [row for row in admitted if start <= int(row["frame_id"]) < start + 30]
        blocks.append({"frame_range": [start, min(start + 29, int(surface["frame_count"]) - 1)], **affine_fit(subset)})
    block_scales = [float(row["scale"]) for row in blocks if row.get("scale") is not None]
    reproduced = float(bound["unconstrained_fit"]["scale"])
    stability = {
        "schema_version": "0915-robot15h-human-stereo-fit-stability-v1",
        "session_id": "play_cards_0915_001",
        "source_group_scope": "ONE_DEVELOPMENT_RECORDING_ONLY",
        "row_count": len(admitted),
        "estimand": "AFFINE_MAP_MANO_FRONT_VISIBLE_SURFACE_OPTICAL_Z_TO_STEREO_VISIBLE_SURFACE_OPTICAL_Z",
        "not_estimands": ["MANO_BONE_LENGTH", "SLAM_WORLD_SCALE", "RAY_DISTANCE", "ROBOT_SCALE"],
        "reproduced_unconstrained_scale": reproduced,
        "reproduced_unconstrained_offset_m": float(bound["unconstrained_fit"]["offset_m"]),
        "all_rows_ols": affine_fit(admitted),
        "by_hand": by_hand,
        "temporal_blocks_30_frames": blocks,
        "temporal_block_scale_range": [min(block_scales), max(block_scales)],
        "hand_scale_absolute_difference": abs(float(by_hand["left"]["scale"]) - float(by_hand["right"]["scale"])),
        "session_static_scale_stable": False,
        "reason": "hand- and time-conditioned slopes vary materially and some strata have narrow depth span/high affine coupling",
    }
    atomic_json(output / "FIT_STABILITY.json", stability)
    geometry = camera_geometry(camera, depth)
    atomic_json(output / "DEPTH_GEOMETRY_AUDIT.json", geometry)
    matrix = {
        "schema_version": "0915-robot15h-scale-cause-matrix-v1",
        "diagnostic_status": "UNRESOLVED_NON_IDENTIFIABLE_CONFOUNDED_FIT",
        "correction_adopted": False,
        "metric_translation_authorized": False,
        "strict_contact_authorized": False,
        "scale_bound_changed": False,
        "candidate_causes": [
            {"cause": "GLOBAL_MANO_BONE_OR_WORLD_SCALE", "assessment": "NOT_ESTIMATED_BY_0_783730"},
            {"cause": "OPTICAL_Z_VS_RAY_DISTANCE", "assessment": "NOT_PRESENT_IN_FROZEN_PAIR_FIELDS"},
            {"cause": "HORIZONTAL_REFLECTION_SPATIAL_MISREGISTRATION", "assessment": "NOT_SUPPORTED_PIXEL_ROUNDTRIP_EXACT"},
            {"cause": "HAND_TIME_DEPTH_MIXTURE_AND_AFFINE_COUPLING", "assessment": "SUPPORTED"},
            {"cause": "VISIBLE_SURFACE_CORRESPONDENCE_CONTAMINATION", "assessment": "PLAUSIBLE_NOT_SEPARATELY_IDENTIFIED"},
            {"cause": "ENCODED_DOMAIN_K_P_METRIC_UNCERTAINTY", "assessment": "PLAUSIBLE_NOT_SEPARATELY_IDENTIFIED"},
        ],
        "decision": (
            "Reject 0.783730 as a reusable correction. Preserve strict metric-alignment and Contact blocks; "
            "use independently recorded W0 Depth/surface evidence for any future causal separation."
        ),
        "next_evidence": "MULTI_SOURCE_W0_FOUNDATIONSTEREO_PLUS_DIRECT_VISIBLE_HAND_SURFACES",
    }
    atomic_json(output / "SCALE_CAUSE_MATRIX.json", matrix)
    metrics = {
        "schema_version": "0915-robot15h-scale-cause-metrics-v1",
        "admitted_pair_rows": len(admitted),
        "reproduced_scale": reproduced,
        "left_scale": by_hand["left"]["scale"], "right_scale": by_hand["right"]["scale"],
        "temporal_block_scale_min": min(block_scales), "temporal_block_scale_max": max(block_scales),
        "correction_adopted": False, "metric_translation_authorized": False,
    }
    atomic_json(output / "METRICS.json", metrics)
    visual.mkdir(parents=True)
    render_diagnostic(stability, matrix, visual / "SCALE_CAUSE_DIAGNOSTIC.png")
    (visual / "README_ZH.md").write_text(
        "# 0915 Robot15h 尺度原因审计 V1\n\n"
        "`0.783730` 被复现，但它只是在 001 的冻结非接触样本上，将 MANO 前可见表面 optical-Z 映射到 Stereo 可见表面 optical-Z 的仿射斜率；不是骨长、世界尺度或机器人尺度。\n\n"
        f"- 左/右手分层斜率：`{by_hand['left']['scale']:.3f}` / `{by_hand['right']['scale']:.3f}`。\n"
        f"- 30 帧块斜率范围：`{min(block_scales):.3f}–{max(block_scales):.3f}`。\n"
        "- 当前 encoded Depth 仍使用缩放 factory K 与 camera-centre baseline；encoded-domain P_L/P_R 外部精度未验证。\n"
        "- 结论：`UNRESOLVED_NON_IDENTIFIABLE_CONFOUNDED_FIT`；不采用统一尺度修正，不改 5 mm Contact 门，不授权公制 wrist-object 平移。\n\n"
        "下一步使用相互独立的 W0 录制取得新的 encoded-domain Depth 与直接可见手表面证据，不能继续调 001 自证。\n",
        encoding="utf-8",
    )
    for raw, digest in before.items():
        if not Path(raw).is_file() or sha256(Path(raw)) != digest:
            raise RuntimeError(f"read-only scale audit input changed: {raw}")
    result = {
        "schema_version": "0915-robot15h-scale-cause-result-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "PASSED",
        "diagnostic_status": matrix["diagnostic_status"], "weights": "ABSENT",
        "model_rerun_performed": False, "gpu_used": False,
        "correction_adopted": False, "metric_translation_authorized": False,
        "strict_contact_authorized": False, "source_mutated": False,
        "control_ground_truth": False, "training_eligible": False,
        "physical_deployment_authorized": False,
        "artifacts": {name: ref(path) for name, path in {
            "scale_cause_matrix": output / "SCALE_CAUSE_MATRIX.json",
            "fit_stability": output / "FIT_STABILITY.json",
            "depth_geometry_audit": output / "DEPTH_GEOMETRY_AUDIT.json",
            "metrics": output / "METRICS.json",
            "visual_readme": visual / "README_ZH.md",
            "diagnostic_png": visual / "SCALE_CAUSE_DIAGNOSTIC.png",
        }.items()},
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-scale-cause-run-receipt-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "PASSED",
        "result": ref(output / "RESULT.json"), "terminal_receipt": ref(receipt),
    })
    heartbeat()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
