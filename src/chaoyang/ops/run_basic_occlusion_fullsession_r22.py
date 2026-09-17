#!/usr/bin/env python3
"""Build a full-session basic Robot/Object z-buffer diagnostic on Clean RGB.

This is the Contact-independent branch.  It uses only directly visible task
object pixels and registered Stereo depth.  Contact-aware refinement is
published separately as blocked unless a same-session CONTACT-10 result is
explicitly supplied by a future successor.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.occlusion_compositor_r22 import CompositePixelSource, derive_composite_domain
from chaoyang.pipeline.occlusion_compositor_v1 import (
    DepthQualityEvidence,
    ObjectPixelSource,
    Ownership,
    audit_frame,
    choose_object_pixels,
    resolve_ownership,
)
from chaoyang.ops.build_visible_surface_occlusion_canary_v71 import (
    exact,
    load,
    object_front_edge_support,
    object_union,
    protected_loss_reasons,
    quality_evidence,
    ref,
    title,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def instance_map(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    labels = np.full(shape, -1, dtype=np.int16)
    for raw_id, instance in sorted(row["physical_instances"].items(), key=lambda item: int(item[0])):
        if not instance.get("valid"):
            continue
        path = exact(instance["mask"], f"object mask {raw_id}")
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot decode object mask: {path}")
        selected = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.any((labels >= 0) & selected):
            raise RuntimeError("Chips physical-instance masks overlap")
        labels[selected] = int(raw_id)
    return labels


def open_capture(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    return cap, float(cap.get(cv2.CAP_PROP_FPS)), int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-zbuffer-result", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument("--object-mask-manifest", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--clean-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    z_result_path = args.robot_zbuffer_result.resolve(strict=True)
    depth_result_path = args.depth_result.resolve(strict=True)
    mask_path = args.object_mask_manifest.resolve(strict=True)
    raw_path = args.raw_video.resolve(strict=True)
    clean_result_path = args.clean_result.resolve(strict=True)
    z_result = load(z_result_path)
    depth_result = load(depth_result_path)
    mask_manifest = load(mask_path)
    clean_result = load(clean_result_path)
    if any(identity != args.session_id for identity in (
        z_result.get("session_id"), depth_result.get("session_id"), mask_manifest.get("session"),
        clean_result.get("session"),
    )):
        raise RuntimeError("same-session identity check failed")
    if depth_result.get("status") != "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION":
        raise RuntimeError("corrected metric depth is not passed")
    if clean_result.get("status") != "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B":
        raise RuntimeError("Clean structural baseline is not passed")
    clean_path = Path(clean_result["artifacts"]["clean_synthetic_master"]["path"]).resolve(strict=True)

    z_archive = exact(z_result["outputs"][0], "Robot z-buffer")
    depth_manifest_path = exact(depth_result["frame_manifest"], "depth frame manifest")
    depth_rows = {int(row["frame_id"]): row for row in load(depth_manifest_path)["frames"]}
    mask_rows = {int(row["source_frame"]): row for row in mask_manifest["frames"]}
    with np.load(z_archive, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        robot_rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        robot_depth = np.asarray(archive["depth_m"], dtype=np.float32)
        robot_label = np.asarray(archive["render_label"], dtype=np.int32)
        triangle_present = "triangle_id" in archive.files
    if not triangle_present:
        raise RuntimeError("Robot triangle provenance is absent")
    if not np.array_equal(frame_ids, np.arange(len(frame_ids), dtype=np.int64)):
        raise RuntimeError("full-session frame identity must be contiguous from zero")
    if robot_rgb.shape[0] != len(frame_ids) or robot_depth.shape != robot_label.shape:
        raise RuntimeError("Robot z-buffer array closure failed")

    raw_cap, raw_fps, raw_count = open_capture(raw_path)
    clean_cap, clean_fps, clean_count = open_capture(clean_path)
    if raw_count != clean_count or raw_count != len(frame_ids) or abs(raw_fps - clean_fps) > 1e-6:
        raise RuntimeError("Raw/Clean/Robot frame timing closure failed")
    shape = robot_depth.shape[1:]
    writer_path = output / f"{args.session_id}_Clean底图_Robot基础几何遮挡_全片.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), raw_fps, (shape[1] * 2, shape[0] * 2))
    if not writer.isOpened():
        raise RuntimeError("cannot open full-session review writer")

    ownership_all = np.empty((len(frame_ids), *shape), dtype=np.uint8)
    training_valid_all = np.empty((len(frame_ids), *shape), dtype=np.bool_)
    m_composite_all = np.empty((len(frame_ids), *shape), dtype=np.bool_)
    composite_source_all = np.empty((len(frame_ids), *shape), dtype=np.uint8)
    object_source_all = np.empty((len(frame_ids), *shape), dtype=np.uint8)
    object_instance_all = np.empty((len(frame_ids), *shape), dtype=np.int16)
    per_frame: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    protected_total = retained_total = contact_total = known_total = 0
    write_domain_violations = 0
    try:
        for slot, frame_id_value in enumerate(frame_ids):
            frame_id = int(frame_id_value)
            ok_raw, raw_full = raw_cap.read()
            ok_clean, clean_full = clean_cap.read()
            if not ok_raw or not ok_clean:
                raise RuntimeError(f"video decode failed at frame {frame_id}")
            raw = cv2.resize(raw_full, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            clean = cv2.resize(clean_full, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            row = mask_rows[frame_id]
            object_mask = object_union(row, shape)
            object_instances = instance_map(row, shape)
            depth_row = depth_rows[frame_id]
            depth_path = exact(
                {"path": depth_row["relative_path"], "bytes": depth_row["bytes"], "sha256": depth_row["sha256"]},
                f"depth frame {frame_id}", depth_manifest_path.parent,
            )
            with np.load(depth_path, allow_pickle=False) as depth_file:
                depth = np.asarray(depth_file["depth_m"], dtype=np.float32)
                disparity = np.asarray(depth_file["disparity_px"], dtype=np.float32)
                valid = np.asarray(depth_file["valid"], dtype=np.bool_)
            if depth.shape != shape:
                raise RuntimeError("Depth and Robot render grids differ")
            pixels, object_source = choose_object_pixels(raw_rgb=raw, raw_visible_mask=object_mask)
            robot_mask = robot_label[slot] >= 0
            decision = (cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (
                cv2.dilate(robot_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            )
            evidence = quality_evidence(depth, disparity, valid, object_mask, raw)
            supported_edge = object_front_edge_support(
                object_mask=object_mask, robot_mask=robot_mask, object_depth_m=depth,
                robot_depth_m=robot_depth[slot], evidence=evidence,
            )
            evidence = DepthQualityEvidence(
                disparity_finite=evidence.disparity_finite,
                registration_pass=evidence.registration_pass,
                away_from_occlusion_edge=evidence.away_from_occlusion_edge | supported_edge,
                texture_support=evidence.texture_support,
                local_consistency_pass=evidence.local_consistency_pass,
                in_valid_depth_range=evidence.in_valid_depth_range,
                depth_confidence_present=False,
            )
            robot_valid = robot_mask & np.isfinite(robot_depth[slot]) & (robot_depth[slot] > 0)
            result = resolve_ownership(
                human_mask=np.zeros(shape, dtype=np.bool_),
                object_amodal_mask=object_mask,
                object_depth_m=depth,
                object_depth_valid=valid & object_mask,
                robot_alpha_mask=robot_mask,
                robot_depth_m=robot_depth[slot],
                robot_depth_valid=robot_valid,
                stereo_depth_valid=valid,
                depth_quality_evidence=evidence,
                object_rgb=pixels,
                object_pixel_source=object_source,
                contact_decision_mask=decision,
            )
            domain = derive_composite_domain(result)
            protected = object_mask & valid & ((~robot_mask) | (depth + 0.003 < robot_depth[slot]))
            audit = audit_frame(result, protected_raw_object_mask=protected)
            protected_total += audit.protected_pixels
            retained_total += audit.retained_protected_pixels
            contact_total += audit.contact_pixels
            known_total += audit.contact_pixels - audit.unknown_contact_pixels

            composite = clean.copy()
            object_front = result.ownership == int(Ownership.OBJECT_FRONT)
            robot_front = result.ownership == int(Ownership.ROBOT_FRONT)
            unknown = result.ownership == int(Ownership.TIE_UNKNOWN)
            composite[object_front] = pixels[object_front]
            composite[robot_front] = robot_rgb[slot][robot_front]
            checker = (np.indices(shape).sum(axis=0) // 6) % 2 == 0
            composite[unknown & checker] = (255, 0, 255)
            composite[unknown & ~checker] = (25, 25, 25)
            write_domain_violations += int(np.count_nonzero((composite != clean).any(axis=2) & ~domain.m_composite))

            ownership_all[slot] = result.ownership
            training_valid_all[slot] = result.training_valid_mask
            m_composite_all[slot] = domain.m_composite
            composite_source_all[slot] = domain.pixel_source
            object_source_all[slot] = result.object_pixel_source
            object_instance_all[slot] = object_instances
            for source in CompositePixelSource:
                source_counts[source.name] += int(np.count_nonzero(domain.pixel_source == int(source)))
            per_frame.append({
                "frame_id": frame_id,
                "known_decision_coverage": audit.known_decision_coverage,
                "unknown_pixel_ratio": audit.unknown_pixel_ratio,
                "protected_retention": audit.protected_retention,
                "m_composite_pixels": int(np.count_nonzero(domain.m_composite)),
                "object_visible_pixels": int(np.count_nonzero(object_mask)),
                "robot_pixels": int(np.count_nonzero(robot_mask)),
                "direct_overlap_pixels": int(np.count_nonzero(object_mask & robot_mask)),
            })

            raw_mask = raw.copy()
            raw_mask[object_mask] = (0.45 * raw_mask[object_mask] + 0.55 * np.asarray((255, 180, 30))).astype(np.uint8)
            robot_view = robot_rgb[slot].copy()
            review = np.vstack([
                np.hstack([title(raw_mask, f"帧{frame_id} Raw＋三实例可见物体"), title(clean, "结构Clean底图（非语义通过）")]),
                np.hstack([title(robot_view, "统一Robot z-buffer"), title(composite, "基础几何遮挡；紫色=UNKNOWN")]),
            ])
            writer.write(review)
    finally:
        raw_cap.release()
        clean_cap.release()
        writer.release()

    if write_domain_violations:
        raise RuntimeError(f"M_composite write-domain violations: {write_domain_violations}")
    arrays_path = output / "BASIC_OCCLUSION_FULLSESSION.npz"
    np.savez_compressed(
        arrays_path,
        frame_ids=frame_ids,
        ownership=ownership_all,
        training_valid_mask=training_valid_all,
        M_composite=m_composite_all,
        composite_pixel_source=composite_source_all,
        object_pixel_source=object_source_all,
        object_instance_id=object_instance_all,
    )
    cap, _, decoded = open_capture(writer_path)
    cap.release()
    if decoded != len(frame_ids):
        raise RuntimeError(f"review decode closure failed: {decoded}/{len(frame_ids)}")
    write_json(output / "PIXEL_SOURCE_LABELS.json", {
        "composite_pixel_source": {str(int(item)): item.name for item in CompositePixelSource},
        "object_pixel_source": {str(int(item)): item.name for item in ObjectPixelSource},
        "ownership": {str(int(item)): item.name for item in Ownership},
        "M_composite": "independent final-compositor write domain; not Clean M_write",
    })
    metrics = {
        "schema_version": "BASIC_OCCLUSION_FULLSESSION_METRICS_R22",
        "session_id": args.session_id,
        "frame_count": len(frame_ids),
        "known_decision_coverage": known_total / contact_total if contact_total else 1.0,
        "unknown_pixel_ratio": 1.0 - known_total / contact_total if contact_total else 0.0,
        "protected_retention": retained_total / protected_total if protected_total else 1.0,
        "protected_pixels": protected_total,
        "retained_protected_pixels": retained_total,
        "M_composite_outside_write_violations": write_domain_violations,
        "composite_pixel_source_counts": dict(sorted(source_counts.items())),
        "accuracy_reported": False,
        "contact_aware_refinement_executed": False,
        "training_eligible": False,
        "training_blockers": ["STRUCTURAL_CLEAN_NOT_SEMANTICALLY_AUTHORIZED", "CONTACT_REFINEMENT_ABSENT"],
        "per_frame": per_frame,
    }
    write_json(output / "METRICS.json", metrics)
    write_json(output / "CONTACT_AWARE_REFINEMENT_RESULT.json", {
        "schema_version": "CONTACT_AWARE_OCCLUSION_REFINEMENT_RESULT_R22",
        "session_id": args.session_id,
        "terminal_status": "BLOCKED_PREREQ",
        "blocker": "SAME_SESSION_CONTACT10_RESULT_ABSENT",
        "basic_geometric_occlusion_remains_valid": True,
        "authority_promoted": False,
    })
    (output / "DECISION.md").write_text(
        "# Chips023基础几何Occlusion全片\n\n"
        "基础分支已完成全片：Robot、当前可见三实例物体、Depth与结构Clean通过统一z-buffer组合。"
        "`M_composite`独立于Clean `M_write`，每个像素保存最终layer provenance。"
        "同会话CONTACT-10缺失，因此contact-aware refinement明确阻塞；该视频不具训练、Silver、"
        "Gold、控制或部署authority。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "BASIC_OCCLUSION_FULLSESSION_NEXT_ACTION_R22",
        "status": "PASSED",
        "next_task_id": "CONTACT10_CHIPS023_THEN_CONTACT_AWARE_REFINEMENT",
        "prerequisites": ["same-session CONTACT-10", "semantic Clean successor"],
    })
    run_receipt = {
        "schema_version": "BASIC_OCCLUSION_FULLSESSION_RUN_RECEIPT_R22",
        "task_id": "BASIC-GEOMETRIC-OCCLUSION-FULLSESSION-R22",
        "artifact_revision": "R7_2_BASIC_OCCLUSION_FULLSESSION",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": [ref(z_result_path), ref(depth_result_path), ref(mask_path), ref(raw_path), ref(clean_result_path), ref(clean_path)],
        "producer": ref(Path(__file__)),
        "authority_promoted": False,
    }
    write_json(output / "RUN_RECEIPT.json", run_receipt)
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "BASIC_OCCLUSION_FULLSESSION_ARTIFACT_MANIFEST_R22",
        "artifacts": [
            ref(arrays_path), ref(writer_path), ref(output / "PIXEL_SOURCE_LABELS.json"),
            ref(output / "METRICS.json"), ref(output / "CONTACT_AWARE_REFINEMENT_RESULT.json"),
            ref(output / "DECISION.md"), ref(output / "NEXT_ACTION.json"), ref(output / "RUN_RECEIPT.json"),
        ],
        "authority_promoted": False,
    })
    result = {
        "schema_version": "BASIC_OCCLUSION_FULLSESSION_RESULT_R22",
        "task_id": "BASIC-GEOMETRIC-OCCLUSION-FULLSESSION-R22",
        "session_id": args.session_id,
        "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_BASIC_GEOMETRIC_OCCLUSION",
        "frame_count": len(frame_ids),
        "basic_occlusion_executed": True,
        "contact_aware_refinement_executed": False,
        "M_composite_present": True,
        "pixel_provenance_present": True,
        "review_video": ref(writer_path),
        "metrics": ref(output / "METRICS.json"),
        "authority_promoted": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "claim_limit": "Full-session visible-surface basic z-buffer diagnostic on structural Clean; no hidden-object, Contact, accuracy, Silver/Gold, training, control or physical authority.",
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
