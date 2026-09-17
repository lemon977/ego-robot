#!/usr/bin/env python3
from __future__ import annotations

"""Finalize the frozen R3 Mask challenger canary as development evidence."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mask_png(record: Mapping[str, Any]) -> np.ndarray:
    image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode mask: {record['path']}")
    return image > 0


def mask_npy(record: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(record, Mapping):
        return None
    return np.load(str(record["path"]), allow_pickle=False) > 0


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else int(np.count_nonzero(first & second)) / union


def candidate_metrics(baseline: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    baseline_frames = baseline["frames"]
    candidate_frames = output["frames"]
    if len(baseline_frames) != len(candidate_frames):
        raise ValueError("candidate/baseline frame count mismatch")
    overlap: list[float] = []
    post_reentry: list[float] = []
    invalid_total = invalid_known = unknown = 0
    areas: list[int] = []
    first_reentry = None
    previous_valid = True
    for baseline_frame, candidate_frame in zip(baseline_frames, candidate_frames):
        source_frame = int(baseline_frame["source_frame"])
        base_record = baseline_frame["physical_instances"]["0"]
        candidate_record = candidate_frame["instances"][0]
        candidate = mask_npy(candidate_record.get("mask"))
        if candidate is None:
            unknown += 1
        else:
            areas.append(int(candidate.sum()))
        base_valid = bool(
            base_record.get("observed") and base_record.get("valid")
            and int(base_record.get("area_px", 0)) > 0
        )
        if not base_valid:
            invalid_total += 1
            invalid_known += int(candidate is not None)
            previous_valid = False
            continue
        if not previous_valid and first_reentry is None:
            first_reentry = source_frame
        previous_valid = True
        if candidate is None:
            continue
        current_iou = iou(mask_png(base_record["mask"]), candidate)
        overlap.append(current_iou)
        if first_reentry is not None and source_frame >= first_reentry:
            post_reentry.append(current_iou)
    frame_count = len(candidate_frames)
    prompt_iou = None
    if candidate_frames:
        initial = baseline_frames[0]["physical_instances"]["0"]
        proposed = mask_npy(candidate_frames[0]["instances"][0].get("mask"))
        if proposed is not None:
            prompt_iou = iou(mask_png(initial["mask"]), proposed)
    return {
        "frames": frame_count,
        "known_decision_coverage": (frame_count - unknown) / frame_count,
        "unknown_frames": unknown,
        "identity_warning_count": int(output.get("identity_qa", {}).get("warning_count", 0)),
        "prompt_frame_iou": prompt_iou,
        "baseline_observed_common_frames": len(overlap),
        "iou_mean_on_baseline_observed_common": float(np.mean(overlap)) if overlap else None,
        "iou_p05_on_baseline_observed_common": float(np.percentile(overlap, 5)) if overlap else None,
        "first_predecessor_reentry_frame": first_reentry,
        "post_reentry_common_frames": len(post_reentry),
        "post_reentry_iou_mean": float(np.mean(post_reentry)) if post_reentry else None,
        "predecessor_invalid_frames": invalid_total,
        "candidate_known_in_predecessor_invalid_fraction": (
            invalid_known / invalid_total if invalid_total else None
        ),
        "candidate_area_median_known": float(np.median(areas)) if areas else None,
        "claim_limit": (
            "Agreement/recovery relative to the predecessor is development evidence, "
            "not correctness or Gold accuracy."
        ),
    }


def add_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 24)
    y = 8
    for line in lines:
        draw.rectangle((4, y - 2, image.width - 4, y + 30), fill=(0, 0, 0, 170))
        draw.text((12, y), line, font=font, fill=(255, 255, 255))
        y += 32
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def overlay(image: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    if mask is None:
        return result
    tint = np.zeros_like(result)
    tint[:] = color
    result[mask] = cv2.addWeighted(result[mask], 0.45, tint[mask], 0.55, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def render_review(
    *, rgb_manifest: Mapping[str, Any], baseline: Mapping[str, Any],
    candidate: Mapping[str, Any], output_path: Path,
) -> None:
    width, height = 640, 480
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError("cannot open MP4 writer")
    try:
        for rgb_row, base_row, candidate_row in zip(
            rgb_manifest["frames"], baseline["frames"], candidate["frames"]
        ):
            frame_id = int(rgb_row["frame_id"])
            image = cv2.imread(str(rgb_row["path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode RGB frame {frame_id}")
            base_record = base_row["physical_instances"]["0"]
            base = mask_png(base_record["mask"]) if base_record.get("valid") else None
            proposed_record = candidate_row["instances"][0]
            proposed = mask_npy(proposed_record.get("mask"))
            left = overlay(image, base, (0, 220, 0))
            right = overlay(image, proposed, (220, 0, 220))
            base_state = "可见" if base is not None else "前序UNKNOWN"
            candidate_state = "可见" if proposed is not None else "UNKNOWN"
            left = add_text(left, ["SAM3.1 当前基线（绿色）", f"帧 {frame_id} | {base_state}"])
            right = add_text(right, ["SAM2.1 challenger（紫色）", f"帧 {frame_id} | {candidate_state}"])
            if proposed is None:
                tile = 24
                for y in range(0, right.shape[0], tile):
                    for x in range(0, right.shape[1], tile):
                        if (x // tile + y // tile) % 2 == 0:
                            cv2.rectangle(right, (x, y), (x + tile, y + tile), (35, 35, 35), -1)
                right = add_text(right, ["SAM2.1 challenger", f"帧 {frame_id} | UNKNOWN（棋盘）"])
            left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(np.concatenate([left, right], axis=1))
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.attempt_root.resolve()
    preflight_path = root / "input_preflight/INPUT_PREFLIGHT.json"
    preflight = load(preflight_path)
    object_canary = next(
        row for row in preflight["rows"]
        if row["stage"] == "OBJECT_MASK" and row["selection_role"] == "CURRENT_C_CANARY"
    )
    session_id = object_canary["session_id"]
    sam_receipt_path = root / f"gpu/sam2_1/sessions/{session_id}/attempts/attempt_0001/RESULT.json"
    cutie_receipt_path = root / f"gpu/cutie/sessions/{session_id}/attempts/attempt_0001/RESULT.json"
    sam_receipt = load(sam_receipt_path)
    cutie_receipt = load(cutie_receipt_path)
    sam_output_path = Path(sam_receipt["payload"]["output"]["path"])
    sam_output = load(sam_output_path)
    source_result = load(Path(object_canary["result"]["path"]))
    baseline_path = Path(source_result["artifacts"]["manifest"]["path"])
    baseline = load(baseline_path)
    rgb_path = Path(object_canary["rgb_manifest"]["path"])
    rgb = load(rgb_path)
    sam_metrics = candidate_metrics(baseline, sam_output)
    sam_quality_gates = {
        "prompt_iou_at_least_0p95": bool((sam_metrics["prompt_frame_iou"] or 0) >= 0.95),
        "known_coverage_at_least_0p70": sam_metrics["known_decision_coverage"] >= 0.70,
        "post_reentry_common_frames_positive": sam_metrics["post_reentry_common_frames"] > 0,
        "post_reentry_iou_at_least_0p50": bool((sam_metrics["post_reentry_iou_mean"] or 0) >= 0.50),
        "identity_warnings_zero": sam_metrics["identity_warning_count"] == 0,
    }
    sam_quality_pass = all(sam_quality_gates.values())
    cutie_payload = cutie_receipt["payload"]

    visual_path = root / f"{session_id}_SAM31_vs_SAM21_全片开发复核.mp4"
    render_review(
        rgb_manifest=rgb, baseline=baseline, candidate=sam_output, output_path=visual_path
    )

    metrics = {
        "schema_version": "mask-challenger-bounded-metrics-r3-v1",
        "current_baseline": "SAM3.1",
        "role_mask": {
            "status": "BLOCKED_PREREQ",
            "selected_sessions": [
                row["session_id"] for row in preflight["rows"] if row["stage"] == "ROLE_MASK"
            ],
            "reason": "NO_REVIEWED_FOUR_ROLE_CAUSAL_PROMPT_AND_ADAPTER_CLOSURE",
        },
        "object_mask": {
            "canary_session": session_id,
            "sam2_1": {
                "execution_status": sam_receipt["payload"]["status"],
                "metrics": sam_metrics,
                "quality_gates": sam_quality_gates,
                "quality_pass": sam_quality_pass,
            },
            "cutie": {
                "execution_status": cutie_payload["status"],
                "execution_performed": cutie_payload["execution_performed"],
                "runtime_error": cutie_payload["runtime_error"],
            },
            "regression_execution": "NOT_RUN_CANARY_DID_NOT_PASS",
        },
        "authority_promoted": False,
        "gold_accuracy_reported": False,
    }
    metrics_path = root / "METRICS.json"
    atomic_json(metrics_path, metrics)

    rows = [
        {
            "stage": "ROLE_MASK", "backend": "SAM2.1/Cutie", "session": "get_potato_chips_0901_010",
            "selection_role": "CURRENT_C_CANARY", "execution_status": "BLOCKED_PREREQ",
            "quality_pass": False, "reason": "FOUR_ROLE_CAUSAL_PROMPT_SET_INCOMPLETE",
        },
        {
            "stage": "OBJECT_MASK", "backend": "SAM2.1", "session": session_id,
            "selection_role": "CURRENT_C_CANARY", "execution_status": sam_receipt["payload"]["status"],
            "quality_pass": sam_quality_pass, "reason": "CANARY_QUALITY_GATE_FAILED" if not sam_quality_pass else "PASS",
        },
        {
            "stage": "OBJECT_MASK", "backend": "Cutie", "session": session_id,
            "selection_role": "CURRENT_C_CANARY", "execution_status": cutie_payload["status"],
            "quality_pass": False, "reason": "CUDA_OOM",
        },
    ]
    csv_path = root / "MASK_CHALLENGER_COMPARISON.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())

    decision_path = root / "DECISION.md"
    decision_path.write_text(
        "# Mask challenger 有界结论 R3\n\n"
        "结论：`NO_GO`。SAM3.1 保持当前基线，不晋升 SAM2.1 或 Cutie。\n\n"
        "- Role Mask：Chips010 当前 C 在四个角色上均无合法非空 seed；现有 S1 输入是物体提示，不能冒充角色提示。\n"
        f"- Object Mask / SAM2.1：{sam_metrics['frames']} 帧结构运行通过，但 known coverage="
        f"{sam_metrics['known_decision_coverage']:.2%}，第一个前序重入后没有共同有效帧，"
        f"identity warning={sam_metrics['identity_warning_count']}，质量门失败。\n"
        "- Object Mask / Cutie：canary 实际执行但 CUDA OOM；lease 已释放，未运行回归。\n"
        "- 这些比较以旧基线为参照，不是人工 Gold accuracy。\n",
        encoding="utf-8",
    )
    next_path = root / "NEXT_ACTION.json"
    atomic_json(next_path, {
        "status": "NO_GO_STOP_BOUNDED_BRANCH",
        "actions": [
            "Freeze a reviewed four-role prompt/adapter contract before any Role challenger retry.",
            "For Object Mask, test bounded causal current-frame correction prompts at verified re-entry; do not expand one-shot SAM2.1.",
            "Profile/downscale Cutie memory in a new canary before any retry; never retry this signature automatically.",
        ],
        "full_batch_allowed": False,
        "authority_promotion_allowed": False,
    })
    task_packet_path = root / "TASK_PACKET.json"
    atomic_json(task_packet_path, {
        "task_id": "mask_challenger_bounded_r3",
        "attempt_id": "attempt_0002",
        "objective": "Bounded 1C+2A/B challenger decision",
        "selection": preflight["selection"],
        "write_set": [str(root)],
        "forbidden_write_set": [
            str((ROOT / "docs/governance").resolve()),
            str((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v76").resolve()),
        ],
        "current_baseline": "SAM3.1",
        "current_governance_update_allowed": False,
    })
    receipt_path = root / "RUN_RECEIPT.json"
    atomic_json(receipt_path, {
        "schema_version": "mask-challenger-bounded-run-receipt-r3-v1",
        "status": "FAILED_QUALITY_C",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "execution": {
            "sam2_1_canary": ref(sam_receipt_path),
            "cutie_canary": ref(cutie_receipt_path),
            "gpu_lease_final": load(ROOT / "_run/current/GPU_LEASE.json"),
        },
        "regressions_not_run_reason": "NO_BACKEND_PASSED_CANARY_QUALITY_GATE",
        "current_baseline_unchanged": "SAM3.1",
        "authority_promoted": False,
    })
    result_path = root / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "mask-challenger-bounded-result-r3-v1",
        "status": "FAILED_QUALITY_C",
        "task_id": "mask_challenger_bounded_r3",
        "attempt_id": "attempt_0002",
        "current_baseline": "SAM3.1",
        "challenger_authority_promoted": False,
        "role_mask_status": "BLOCKED_PREREQ",
        "object_mask_status": "FAILED_QUALITY_C",
        "go_no_go": "NO_GO",
        "outputs": [ref(metrics_path), ref(csv_path), ref(visual_path), ref(decision_path), ref(next_path), ref(receipt_path)],
        "claim_limit": "Bounded development challenger evidence only; not Gold accuracy or current Mask authority.",
    })
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    atomic_json(manifest_path, {
        "schema_version": "mask-challenger-bounded-artifact-manifest-r3-v1",
        "artifacts": [
            ref(preflight_path), ref(sam_receipt_path), ref(cutie_receipt_path), ref(sam_output_path),
            ref(metrics_path), ref(csv_path), ref(visual_path), ref(decision_path), ref(next_path),
            ref(receipt_path), ref(result_path), ref(task_packet_path),
        ],
        "authority_promoted": False,
    })
    print(json.dumps(load(result_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
