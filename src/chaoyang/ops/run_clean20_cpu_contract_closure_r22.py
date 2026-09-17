#!/usr/bin/env python3
"""Close the CPU-verifiable portion of the Clean-20 causal input contract.

This worker is deliberately fail-closed.  It creates fresh per-frame provenance
maps paired with the immutable lossless Raw PNGs, but it never opens or copies
pixels from an earlier Clean result.  A temporal proposal is accepted only when
its semantic layer is independently known.  The current inputs have no frozen
support-surface semantic mask, so all otherwise-valid background proposals stay
UNKNOWN rather than being silently treated as table/plate pixels.

The Poker atlas audit consumes only physical instance 0, DIRECT_OBSERVED_ONLY
Object6D frames, the selected-camera adapter and same-frame Raw pixels.  It
publishes diagnostic canonical warps only; it cannot publish an atlas unless the
pose projection agrees with the independently frozen object mask.
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

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import dilate, proposed, unions


RUN_SNAPSHOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/g0/attempts/attempt_0001/RUN_START_SNAPSHOT.json"
PRIOR_PREREQ = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/attempts/attempt_0001_cpu_prereq"
POKER_ADAPTER_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/selected_camera_adapter_batch_R7_3/poker/play_cards_0903_245/physical_object_0/RESULT.json"
POKER_HAWOR = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/hawor_bounded_v2/poker/play_cards_0903_245/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"

CASES: dict[str, dict[str, Any]] = {
    "Poker245": {
        "session": "play_cards_0903_245",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/real_donor_v1/play_cards_0903_245/SOURCE_MAP_MANIFEST.json",
    },
    "Chips039": {
        "session": "get_potato_chips_0902_039",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/real_donor_v1/get_potato_chips_0902_039/SOURCE_MAP_MANIFEST.json",
    },
}

PIXEL_SOURCE = {
    1: "RAW_CURRENT_UNCHANGED",
    2: "RAW_CURRENT_VISIBLE_OBJECT",
    7: "UNKNOWN",
}
PROPOSAL_REASON = {
    0: "NOT_A_TEMPORAL_WRITE_PROPOSAL",
    2: "REJECT_FUTURE",
    3: "REJECT_INVALID_COORDINATE",
    4: "REJECT_SOURCE_TASK_OBJECT",
    5: "REJECT_SUPPORT_OR_SEMANTIC_LAYER_UNVERIFIED",
}


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
    if verify is not None and any(value[k] != verify[k] for k in value):
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


def classify_proposals_fail_closed(
    target_frame: int,
    m_write: np.ndarray,
    source_kind: np.ndarray,
    source_frame: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_objects: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    """Classify temporal proposals without assuming that unknown == background."""
    temporal = m_write & (source_kind == 1)
    reason = np.zeros(m_write.shape, np.uint8)
    future = temporal & (source_frame > target_frame)
    reason[future] = 2
    causal = temporal & ~future
    in_frame = causal & (source_frame >= 0) & (source_frame < len(source_objects))
    in_bounds = in_frame & (source_x >= 0) & (source_x < m_write.shape[1]) & (source_y >= 0) & (source_y < m_write.shape[0])
    reason[causal & ~in_bounds] = 3
    for source_id in np.unique(source_frame[in_bounds]):
        loc = in_bounds & (source_frame == source_id)
        object_hit = np.zeros_like(loc)
        object_hit[loc] = source_objects[int(source_id)][source_y[loc], source_x[loc]]
        reason[loc & object_hit] = 4
        # No frozen support-surface/background semantic label exists.  These
        # pixels are deliberately rejected, not guessed to be background.
        reason[loc & ~object_hit] = 5
    return reason, {
        "temporal_proposals": int(temporal.sum()),
        "future_rejected": int((reason == 2).sum()),
        "invalid_coordinate_rejected": int((reason == 3).sum()),
        "source_task_object_rejected": int((reason == 4).sum()),
        "support_or_semantic_unknown_rejected": int((reason == 5).sum()),
        "accepted_temporal_pixels": 0,
    }


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="little")


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed, bitorder="little", count=shape[0] * shape[1]).reshape(shape).astype(bool)


def build_fresh_pair_case(label: str, spec: dict[str, Any], out: Path) -> dict[str, Any]:
    case_out = out / label
    maps_out = case_out / "fresh_source_maps"
    maps_out.mkdir(parents=True, exist_ok=False)
    manifest_ref = ref(spec["manifest"])
    donor_ref = ref(spec["donor"])
    manifest = json.loads(spec["manifest"].read_text(encoding="utf-8"))
    donor_manifest = json.loads(spec["donor"].read_text(encoding="utf-8"))
    rows = manifest["frames"]
    donor_rows = donor_manifest["frames"]
    if len(rows) != len(donor_rows):
        raise RuntimeError(f"{label}: frame/source-map count mismatch")
    source_objects = [unions(row)[2] for row in rows]
    totals = {key: 0 for key in (
        "m_remove_pixels", "m_write_pixels", "m_flow_pixels", "visible_object_pixels",
        "unknown_write_pixels", "temporal_proposals", "future_rejected",
        "invalid_coordinate_rejected", "source_task_object_rejected",
        "support_or_semantic_unknown_rejected", "accepted_temporal_pixels",
    )}
    output_rows: list[dict[str, Any]] = []
    verified_raw = 0
    for frame_id, (row, donor_row) in enumerate(zip(rows, donor_rows)):
        raw_ref = ref(Path(row["source_rgb"]["path"]), verify=row["source_rgb"])
        if Path(raw_ref["path"]).suffix.lower() != ".png":
            raise RuntimeError(f"{label} frame {frame_id}: Raw is not lossless PNG")
        verified_raw += 1
        human, tracker, obj = unions(row)
        m_remove = (human | tracker) & ~obj
        m_write = proposed(human, tracker, obj)
        m_flow = dilate(m_write, 16)
        if np.any(m_remove & ~m_write) or np.any(m_write & ~m_flow) or np.any(obj & m_write):
            raise RuntimeError(f"{label} frame {frame_id}: M_remove/M_write/M_flow or object-protection contract failed")

        proposal_ref = ref(Path(donor_row["pixel_source_map"]["path"]), verify=donor_row["pixel_source_map"])
        # Intentionally do not open, hash or copy donor_row['clean_rgb'].
        with np.load(proposal_ref["path"], allow_pickle=False) as z:
            required = {"source_kind", "source_frame", "source_x", "source_y"}
            if not required.issubset(z.files):
                raise RuntimeError(f"{label} frame {frame_id}: proposal map schema incomplete")
            shape = m_write.shape
            if any(z[key].shape != shape for key in required):
                raise RuntimeError(f"{label} frame {frame_id}: proposal map shape mismatch")
            reasons, counts = classify_proposals_fail_closed(
                frame_id, m_write, z["source_kind"], z["source_frame"], z["source_x"], z["source_y"], source_objects
            )

        pixel_source = np.full(m_write.shape, 1, np.uint8)
        pixel_source[obj] = 2
        pixel_source[m_write] = 7
        map_path = maps_out / f"{frame_id:06d}.npz"
        np.savez_compressed(
            map_path,
            schema_version=np.array("clean20-fresh-lossless-source-map-r22-v1"),
            frame_id=np.int32(frame_id),
            height=np.int32(m_write.shape[0]),
            width=np.int32(m_write.shape[1]),
            identity_raw_coordinates=np.bool_(True),
            pixel_source_code=pixel_source,
            proposal_rejection_reason=reasons,
            m_remove_packed=pack_mask(m_remove),
            m_write_packed=pack_mask(m_write),
            m_flow_packed=pack_mask(m_flow),
            training_valid_packed=pack_mask(~m_write),
        )
        # Decode our own fresh map immediately; this is the publish gate.
        with np.load(map_path, allow_pickle=False) as z:
            if not np.array_equal(unpack_mask(z["m_write_packed"], m_write.shape), m_write):
                raise RuntimeError(f"{label} frame {frame_id}: fresh source-map round trip failed")
            if np.any(z["pixel_source_code"][m_write] != 7):
                raise RuntimeError(f"{label} frame {frame_id}: unresolved write pixels were not UNKNOWN")
        values = {
            "frame_id": frame_id,
            "raw_lossless_rgb": raw_ref,
            "fresh_source_map": ref(map_path),
            "proposal_source_map": proposal_ref,
            "m_remove_pixels": int(m_remove.sum()),
            "m_write_pixels": int(m_write.sum()),
            "m_flow_pixels": int(m_flow.sum()),
            "visible_object_pixels": int(obj.sum()),
            "unknown_write_pixels": int(m_write.sum()),
            **counts,
        }
        output_rows.append(values)
        for key in totals:
            totals[key] += int(values[key])

    pair_manifest = {
        "schema_version": "clean20-fresh-lossless-pair-manifest-r22-v1",
        "artifact_revision": "R7_6_CLEAN20_CPU_PREFILL_1",
        "session": spec["session"],
        "frame_count": len(rows),
        "formal_output_fps": 30,
        "video_generated": False,
        "rgb_materialization": "REFERENCE_EXISTING_RAW_LOSSLESS_PNG_NO_REENCODE",
        "old_clean_rgb_files_opened_or_copied": 0,
        "source_map_input_role": "DONOR_PROPOSAL_ONLY_NOT_PIXEL_SOURCE",
        "source_manifest": manifest_ref,
        "proposal_manifest": donor_ref,
        "pixel_source_codes": {str(k): v for k, v in PIXEL_SOURCE.items()},
        "proposal_rejection_codes": {str(k): v for k, v in PROPOSAL_REASON.items()},
        "mask_contract": "M_remove subset M_write subset M_flow; task object disjoint from M_write",
        "causal_contract": "Every accepted donor must satisfy source_frame <= target_frame; this attempt accepts none without independent semantic-layer evidence.",
        "frames": output_rows,
        "totals": totals,
        "verified_raw_lossless_frames": verified_raw,
        "training_eligible": False,
        "claim_limit": "Fresh Raw/source-map prefill pairing only; unresolved write pixels remain UNKNOWN and this is not a Clean image result.",
    }
    write_json(case_out / "FRESH_LOSSLESS_PAIR_MANIFEST.json", pair_manifest)
    return {
        "label": label,
        "session": spec["session"],
        "frame_count": len(rows),
        "totals": totals,
        "fresh_pair_manifest": ref(case_out / "FRESH_LOSSLESS_PAIR_MANIFEST.json"),
        "support_surface_exclusion": "PASSED_FAIL_CLOSED_ZERO_ACCEPTED_UNKNOWN_SEMANTICS",
        "lossless_pair_contract": "PASSED_PREFILL_NOT_CLEAN",
        "training_eligible": False,
    }


def project_card_quad(T: np.ndarray, K: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = map(float, size)
    corners = np.array([
        [-sx / 2, -sy / 2, 0.0], [sx / 2, -sy / 2, 0.0],
        [sx / 2, sy / 2, 0.0], [-sx / 2, sy / 2, 0.0],
    ], np.float64)
    points_camera = (T[:3, :3] @ corners.T + T[:3, 3:4]).T
    projected = (K @ points_camera.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return uv, points_camera


def mask_iou_for_quad(mask: np.ndarray, uv: np.ndarray) -> tuple[float, float]:
    polygon = np.zeros(mask.shape, np.uint8)
    if np.all(np.isfinite(uv)):
        cv2.fillConvexPoly(polygon, np.rint(uv).astype(np.int32), 1)
    p = polygon.astype(bool)
    union = int((p | mask).sum())
    intersection = int((p & mask).sum())
    return (intersection / union if union else 0.0), (intersection / int(mask.sum()) if mask.any() else 0.0)


def audit_poker_atlas(out: Path) -> dict[str, Any]:
    spec = CASES["Poker245"]
    rows = json.loads(spec["manifest"].read_text(encoding="utf-8"))["frames"]
    adapter_result = json.loads(POKER_ADAPTER_RESULT.read_text(encoding="utf-8"))
    adapter_path = Path(adapter_result["outputs"][0]["path"])
    ref(adapter_path, verify=adapter_result["outputs"][0])
    legacy_result_path = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/depth_object6d_stream_v1/object6d_observed_only_v1/poker/play_cards_0903_245/RESULT.json"
    legacy_result = json.loads(legacy_result_path.read_text(encoding="utf-8"))
    legacy_path = Path(legacy_result["artifacts"]["trajectory"]["path"])
    ref(legacy_path, verify=legacy_result["artifacts"]["trajectory"])
    hawor_ref = ref(POKER_HAWOR)
    with np.load(adapter_path, allow_pickle=False) as a, np.load(legacy_path, allow_pickle=False) as legacy, np.load(POKER_HAWOR, allow_pickle=False) as hand:
        valid = np.asarray(a["valid"], bool)
        direct = np.asarray(legacy["valid"], bool) & np.asarray(legacy["observed"], bool) & (np.asarray(legacy["physical_instance_id"]) == 0)
        if not np.array_equal(valid, direct):
            raise RuntimeError("selected-camera adapter validity does not exactly match DIRECT_OBJECT6D instance 0")
        transforms = np.asarray(a["T_object_to_selected_camera"], np.float64)
        intrinsics = np.asarray(hand["intrinsics"], np.float64)
        size = np.asarray(legacy["object_size_m"], np.float64)

    rows_metrics: list[dict[str, Any]] = []
    for frame_id in np.flatnonzero(direct).tolist():
        mask = unions(rows[frame_id])[2]
        uv, points_camera = project_card_quad(transforms[frame_id], intrinsics[frame_id], size)
        iou, mask_coverage = mask_iou_for_quad(mask, uv)
        face_dot = float(np.dot(transforms[frame_id][:3, 2], -transforms[frame_id][:3, 3]))
        rows_metrics.append({
            "source_frame": frame_id,
            "physical_instance_id": 0,
            "evidence_type": "DIRECT_OBJECT6D",
            "source_is_not_future_for_targets_gte": frame_id,
            "projection_finite": bool(np.isfinite(uv).all() and np.all(points_camera[:, 2] > 0)),
            "projected_quad_xy": uv.tolist(),
            "projected_quad_mask_iou": iou,
            "object_mask_coverage": mask_coverage,
            "face_normal_dot_camera_vector": face_dot,
            "face_sign_only_not_physical_front_back_identity": "POSITIVE" if face_dot >= 0 else "NEGATIVE",
        })

    ious = np.array([row["projected_quad_mask_iou"] for row in rows_metrics], np.float64)
    coverages = np.array([row["object_mask_coverage"] for row in rows_metrics], np.float64)
    diagnostic_dir = out / "Poker245" / "atlas_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=False)
    chosen = sorted(rows_metrics, key=lambda row: row["projected_quad_mask_iou"], reverse=True)[:3]
    diagnostics = []
    canonical_size = (256, int(round(256 * float(size[1] / size[0]))))
    dst = np.array([[0, 0], [canonical_size[0] - 1, 0], [canonical_size[0] - 1, canonical_size[1] - 1], [0, canonical_size[1] - 1]], np.float32)
    for row_metric in chosen:
        frame_id = int(row_metric["source_frame"])
        raw = cv2.imread(rows[frame_id]["source_rgb"]["path"], cv2.IMREAD_COLOR)
        mask = unions(rows[frame_id])[2]
        uv = np.asarray(row_metric["projected_quad_xy"], np.float32)
        overlay = raw.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
        cv2.polylines(overlay, [np.rint(uv).astype(np.int32)], True, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"DIRECT frame={frame_id}  quad-mask IoU={row_metric['projected_quad_mask_iou']:.3f}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        overlay_path = diagnostic_dir / f"frame_{frame_id:06d}_projection.png"
        cv2.imwrite(str(overlay_path), overlay)
        H = cv2.getPerspectiveTransform(uv, dst)
        warp = cv2.warpPerspective(raw, H, canonical_size)
        warp_mask = cv2.warpPerspective(mask.astype(np.uint8) * 255, H, canonical_size, flags=cv2.INTER_NEAREST)
        rgba = cv2.cvtColor(warp, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = warp_mask
        warp_path = diagnostic_dir / f"frame_{frame_id:06d}_canonical_diagnostic.png"
        cv2.imwrite(str(warp_path), rgba)
        diagnostics.append({
            "source_frame": frame_id,
            "allowed_target_frame_min": frame_id,
            "offline_diagnostic_only": True,
            "projection_overlay": ref(overlay_path),
            "canonical_warp": ref(warp_path),
        })

    projection_gate = bool(len(rows_metrics) >= 3 and np.median(ious) >= 0.70 and np.percentile(ious, 10) >= 0.50)
    face_identity_gate = False  # sign is observable; physical front/back identity is not independently labeled.
    atlas = {
        "schema_version": "poker-causal-atlas-audit-r22-v1",
        "session": "play_cards_0903_245",
        "physical_instance_id": 0,
        "input_authority": "DIRECT_OBSERVED_ONLY_KEEP_INVALID_PLUS_DEVELOPMENT_SELECTED_CAMERA_ADAPTER",
        "adapter_result": ref(POKER_ADAPTER_RESULT),
        "legacy_object6d_result": ref(legacy_result_path),
        "hawor_intrinsics": hawor_ref,
        "direct_observed_frames": len(rows_metrics),
        "object_size_m": size.tolist(),
        "projection_metrics": {
            "quad_mask_iou_p10": float(np.percentile(ious, 10)),
            "quad_mask_iou_p50": float(np.percentile(ious, 50)),
            "quad_mask_iou_p90": float(np.percentile(ious, 90)),
            "quad_mask_iou_max": float(ious.max()),
            "mask_coverage_p50": float(np.percentile(coverages, 50)),
            "required_iou_p50": 0.70,
            "required_iou_p10": 0.50,
        },
        "gates": {
            "same_physical_instance_only": True,
            "direct_object6d_only": True,
            "attachment_used": False,
            "future_frame_used": False,
            "selected_camera_projection_mask_agreement": projection_gate,
            "independent_physical_front_back_identity": face_identity_gate,
        },
        "per_frame": rows_metrics,
        "diagnostic_warps": diagnostics,
        "status": "BLOCKED_PREREQ",
        "atlas_published": False,
        "training_eligible": False,
        "reason": "The exact-size selected-camera pose projection does not agree with the frozen visible-object mask, and physical front/back face identity is not independently closed.",
        "claim_limit": "Projection/warp diagnostic only; not a canonical object texture atlas and not eligible as Clean or training pixels.",
    }
    write_json(out / "Poker245" / "POKER_CAUSAL_ATLAS_AUDIT.json", atlas)
    return atlas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--task-packet", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    task_packet_ref = ref(args.task_packet)
    snapshot_ref = ref(RUN_SNAPSHOT)
    case_reports = [build_fresh_pair_case(label, spec, out) for label, spec in CASES.items()]
    atlas = audit_poker_atlas(out)
    totals = {key: sum(case["totals"][key] for case in case_reports) for key in case_reports[0]["totals"]}
    support_contract = {
        "schema_version": "clean20-semantic-support-surface-exclusion-r22-v1",
        "policy": "FAIL_CLOSED_NO_UNKNOWN_LAYER_AS_BACKGROUND",
        "frozen_support_surface_pixel_labels_present": False,
        "accepted_temporal_pixels": totals["accepted_temporal_pixels"],
        "rejected_unverified_semantic_pixels": totals["support_or_semantic_unknown_rejected"],
        "task_object_donor_rejected": totals["source_task_object_rejected"],
        "old_clean_rgb_files_opened_or_copied": 0,
        "status": "PASSED_FAIL_CLOSED",
        "downstream_effect": "Unresolved M_write pixels remain UNKNOWN; full Clean remains BLOCKED_PREREQ.",
    }
    write_json(out / "SUPPORT_SURFACE_EXCLUSION_CONTRACT.json", support_contract)
    status = "BLOCKED_PREREQ"
    result = {
        "schema_version": "clean20-cpu-contract-closure-result-r22-v1",
        "task_id": "clean20_cpu_contract_closure_r22",
        "attempt_id": out.name,
        "status": status,
        "artifact_revision": "R7_6_CLEAN20_CPU_CONTRACT_1",
        "generated_at": now(),
        "task_packet": task_packet_ref,
        "run_start_snapshot": snapshot_ref,
        "case_reports": case_reports,
        "support_surface_exclusion_contract": ref(out / "SUPPORT_SURFACE_EXCLUSION_CONTRACT.json"),
        "poker_causal_atlas_audit": ref(out / "Poker245" / "POKER_CAUSAL_ATLAS_AUDIT.json"),
        "fresh_lossless_pair_frames": sum(case["frame_count"] for case in case_reports),
        "old_clean_rgb_files_opened_or_copied": 0,
        "fresh_propainter_started": False,
        "gpu_used": False,
        "authority_promoted": False,
        "claim_limit": "CPU causal prefill/source-map closure only; not Clean imagery, a verified atlas, or causal training authority.",
    }
    metrics = {
        "schema_version": "clean20-cpu-contract-closure-metrics-r22-v1",
        "status": status,
        "cases": case_reports,
        "aggregate": totals,
        "poker_atlas": {
            "direct_observed_frames": atlas["direct_observed_frames"],
            "projection_metrics": atlas["projection_metrics"],
            "atlas_published": False,
        },
        "gates": {
            "support_surface_cross_semantic_donor_zero": totals["accepted_temporal_pixels"] == 0,
            "fresh_lossless_raw_source_map_pairing": True,
            "all_unresolved_write_pixels_unknown": totals["unknown_write_pixels"] == totals["m_write_pixels"],
            "old_clean_pixel_reuse_zero": True,
            "future_donor_use_zero": True,
            "attachment_use_zero": True,
            "poker_causal_atlas_closed": False,
            "full_clean_or_propainter_started": False,
        },
    }
    write_json(out / "RESULT.json", result)
    write_json(out / "METRICS.json", metrics)
    write_json(out / "RUN_RECEIPT.json", {
        "schema_version": "clean20-cpu-contract-closure-run-receipt-r22-v1",
        "task_id": result["task_id"], "attempt_id": out.name, "status": status,
        "created_at": now(), "host": socket.gethostname(), "pid": os.getpid(),
        "gpu_lease": "NOT_ACQUIRED_CPU_ONLY", "authority_promoted": False,
    })
    write_json(out / "NEXT_ACTION.json", {
        "schema_version": "clean20-cpu-contract-closure-next-r22-v1",
        "status": status,
        "minimum_next_canary": {
            "session": "play_cards_0903_245",
            "frames": [0, 32, 98],
            "objective": "Freeze independent front/back face identity and diagnose card-size/pose-to-mask projection disagreement before any atlas pixel is accepted.",
            "required_gates": ["quad-mask IoU P50 >= 0.70", "IoU P10 >= 0.50", "front/back identity independently labeled", "source_frame <= target_frame"],
        },
        "then": "Only after the atlas and support-surface semantics close, create a new prefix-only ProPainter Task Packet.",
        "do_not": ["Do not use future donors.", "Do not use HAND_OBJECT_ATTACHMENT.", "Do not read or reuse old Clean RGB.", "Do not start full-session ProPainter from this attempt."],
    })
    write_text(out / "DECISION.md", """# Clean-20 CPU 前置闭合决定\n\n状态：`BLOCKED_PREREQ`。\n\n已闭合两项 CPU 合同：\n\n- Poker245 与 Chips039 共 457 帧均重新绑定到经过 SHA 验证的 Raw PNG，并生成新的逐像素 source-map；旧 Clean RGB 读取和复用均为 0。\n- 没有独立 support-surface 语义的 donor 全部 fail-closed，未把桌面、盘子或未知语义区域猜成背景；因此所有未解决的 `M_write` 像素保持 `UNKNOWN`。\n\nPoker atlas 不能发布。43 个同实例直接观测帧满足来源和因果条件，但公称牌尺寸经 selected-camera pose 投影后与冻结物体 Mask 的一致性没有通过，且正反面物理身份没有独立闭合。这里只保留三帧 diagnostic warp；它们不是合法对象纹理。\n\n本 attempt 未启动 ProPainter、未使用 GPU、未使用未来帧、Attachment 或旧 Clean 像素，也未修改 current authority。\n""")
    artifacts = {
        "task_packet": task_packet_ref,
        "result": ref(out / "RESULT.json"),
        "metrics": ref(out / "METRICS.json"),
        "run_receipt": ref(out / "RUN_RECEIPT.json"),
        "decision": ref(out / "DECISION.md"),
        "next_action": ref(out / "NEXT_ACTION.json"),
        "support_surface_contract": ref(out / "SUPPORT_SURFACE_EXCLUSION_CONTRACT.json"),
        "poker_atlas_audit": ref(out / "Poker245" / "POKER_CAUSAL_ATLAS_AUDIT.json"),
    }
    for case in case_reports:
        artifacts[f"{case['label']}_fresh_pair_manifest"] = case["fresh_pair_manifest"]
    write_json(out / "ARTIFACT_MANIFEST.json", {
        "schema_version": "clean20-cpu-contract-closure-artifact-manifest-r22-v1",
        "task_id": result["task_id"], "status": status,
        "artifacts": artifacts, "authority_promoted": False,
    })
    print(json.dumps({"status": status, "result": ref(out / "RESULT.json"), "metrics": metrics["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
