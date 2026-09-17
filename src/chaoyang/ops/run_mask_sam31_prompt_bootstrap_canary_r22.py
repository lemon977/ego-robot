#!/usr/bin/env python3
"""Run the two frozen SAM3.1 prompt-bootstrap canaries under an outer GPU lease.

Only the frozen anchor frame is evaluated. Every box and point-refinement
candidate is persisted on success or failure. No propagation, regression or
authority update is performed here.
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers


CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
ANNOTATIONS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json"
EVALUATION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/MASK_NEXT_EVALUATION_REFERENCE.json"
SELECTION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/MASK_NEXT_FROZEN_SELECTION.json"
PROMPT_CONTRACT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/SAM31_CURRENT_PROMPT_CONTRACT.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())


def norm_box(box: list[int], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise RuntimeError(f"invalid frozen box {box}")
    return [x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height]


def tracker_point_tensors(point: list[int], width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile one frozen pixel point into the official tracker tensor contract."""
    px, py = point
    if not (0 <= px < width and 0 <= py < height):
        raise RuntimeError(f"point outside image: {point}")
    points = torch.tensor([[px / width, py / height]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int32)
    if points.ndim != 2 or points.shape != (1, 2) or labels.ndim != 1 or labels.shape != (1,):
        raise RuntimeError("compiled tracker point tensor shape drift")
    return points, labels


def candidate_rows(masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, point: list[int], box: list[int], stage: str) -> list[dict[str, Any]]:
    px, py = point
    x0, y0, x1, y1 = box
    box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    rows = []
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        area = int(mask.sum())
        if area:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            bbox, cx, cy = None, None, None
        inside_box = int(mask[y0 : y1 + 1, x0 : x1 + 1].sum())
        row = {
            "stage": stage,
            "instance_index": index,
            "object_id": int(ids[index]),
            "model_score": float(scores[index]),
            "area_pixels": area,
            "bbox_xyxy": bbox,
            "positive_point_covered": bool(mask[py, px]) if area else False,
            "prompt_box_coverage": inside_box / box_area,
            "mask_fraction_inside_prompt_box": inside_box / max(area, 1),
            "centroid_xy": [cx, cy] if area else None,
            "centroid_inside_prompt_box": bool(area and x0 <= cx <= x1 and y0 <= cy <= y1),
        }
        row["eligible"] = bool(
            row["positive_point_covered"]
            and row["centroid_inside_prompt_box"]
            and 64 <= area <= box_area * 3.0
            and row["prompt_box_coverage"] >= 0.05
        )
        rows.append(row)
    return sorted(rows, key=lambda r: (r["eligible"], r["positive_point_covered"], r["model_score"]), reverse=True)


def run_prompt(adapter: Any, input_dir: Path, task_name: str, annotation: dict[str, Any], text: str | None) -> tuple[dict[str, Any], np.ndarray | None]:
    source = Path(annotation["source_rgb"]["path"])
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (960, 1280):
        raise RuntimeError(f"{task_name}: source image domain mismatch")
    if ref(source) != annotation["source_rgb"]:
        raise RuntimeError(f"{task_name}: frozen source changed")
    input_dir.mkdir(parents=True)
    os.symlink(source, input_dir / "00000.png")
    height, width = image.shape[:2]
    box = annotation["box_xyxy"]
    point = annotation["positive_point_xy"]
    session = f"{task_name}-prompt-bootstrap"
    sam_helpers.start_session(adapter, session, input_dir)
    all_rows: list[dict[str, Any]] = []
    chosen_mask = None
    try:
        state = adapter._session(session)
        _, box_outputs = adapter.model.add_prompt(
            inference_state=state,
            frame_idx=0,
            text_str=text or "visual",
            boxes_xywh=[norm_box(box, width, height)],
            box_labels=[1],
            output_prob_thresh=0.5,
        )
        masks, scores, ids = sam_helpers.normalize(box_outputs, height, width)
        box_rows = candidate_rows(masks, scores, ids, point, box, "BOX_AND_OPTIONAL_TEXT")
        all_rows.extend(box_rows)
        eligible = [row for row in box_rows if row["eligible"]]
        if not eligible:
            return {"task": task_name, "status": "FAILED_QUALITY_C", "reason": "NO_ELIGIBLE_BOX_SEED", "candidate_rows": all_rows}, None
        chosen = eligible[0]
        chosen_id = int(chosen["object_id"])
        point_tensor, point_label_tensor = tracker_point_tensors(point, width, height)
        _, point_outputs = adapter.model.add_prompt(
            inference_state=state,
            frame_idx=0,
            text_str=None,
            clear_old_points=True,
            points=point_tensor,
            point_labels=point_label_tensor,
            boxes_xywh=None,
            box_labels=None,
            clear_old_boxes=False,
            output_prob_thresh=0.5,
            obj_id=chosen_id,
            rel_coordinates=True,
        )
        refined_masks, refined_scores, refined_ids = sam_helpers.normalize(point_outputs, height, width)
        point_rows = candidate_rows(refined_masks, refined_scores, refined_ids, point, box, "POINT_REFINEMENT")
        all_rows.extend(point_rows)
        refined_eligible = [row for row in point_rows if row["eligible"] and row["object_id"] == chosen_id]
        if not refined_eligible:
            return {"task": task_name, "status": "FAILED_QUALITY_C", "reason": "POINT_REFINEMENT_REJECTED", "selected_box_object_id": chosen_id, "candidate_rows": all_rows}, None
        row = refined_eligible[0]
        index = next(i for i, object_id in enumerate(refined_ids) if int(object_id) == chosen_id)
        chosen_mask = refined_masks[index]
        return {"task": task_name, "status": "PASSED_DEVELOPMENT", "reason": "BOX_SEED_AND_POINT_REFINEMENT_ACCEPTED", "selected_object_id": chosen_id, "selected_row": row, "candidate_rows": all_rows}, chosen_mask
    finally:
        adapter.handle_request({"type": "close_session", "session_id": session})


def save_preview(path: Path, annotation: dict[str, Any], mask: np.ndarray | None, title: str) -> None:
    image = cv2.imread(annotation["source_rgb"]["path"], cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = annotation["box_xyxy"]
    px, py = annotation["positive_point_xy"]
    if mask is not None:
        image[mask] = (0.45 * image[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.circle(image, (px, py), 7, (0, 255, 0), -1)
    cv2.putText(image, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed writing preview {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("RESULT.json", "RUN_RECEIPT.json")):
        raise RuntimeError("immutable attempt terminal already exists")

    started = time.perf_counter()
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    inputs = {name: ref(path) for name, path in {"annotations": ANNOTATIONS, "evaluation": EVALUATION, "selection": SELECTION, "prompt_contract": PROMPT_CONTRACT}.items()}
    status = "FAILED_RUNTIME_FINAL"
    error = None
    tasks: dict[str, Any] = {}
    artifact_refs: dict[str, Any] = {}
    adapter = None
    try:
        if sha(CHECKPOINT) != CHECKPOINT_SHA:
            raise RuntimeError("SAM31_CHECKPOINT_SHA_MISMATCH")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        adapter, build = adapter_module.build_pinned_adapter(official_code_root=SAM_ROOT, checkpoint_path=CHECKPOINT)
        chips_ann = annotations["annotations"]["chips010"]
        chips_roles = {}
        chips_masks = {}
        for role, role_prompt in chips_ann["roles"].items():
            merged = {"source_rgb": chips_ann["source_rgb"], **role_prompt}
            result, mask = run_prompt(adapter, out / "input_frames" / f"chips_{role}", f"chips010_{role}", merged, "a person's hand and forearm")
            chips_roles[role] = result
            chips_masks[role] = mask
            preview = out / f"Chips010_{role}_SAM31提示bootstrap.png"
            save_preview(preview, merged, mask, f"Chips010 {role}: {result['status']}")
            artifact_refs[preview.name] = ref(preview)
        chips_status = "PASSED_DEVELOPMENT" if all(x["status"] == "PASSED_DEVELOPMENT" for x in chips_roles.values()) else "FAILED_QUALITY_C"
        tasks["chips010"] = {"status": chips_status, "roles": chips_roles, "propagation_executed": False}

        poker_ann = annotations["annotations"]["poker015"]
        poker_result, poker_mask = run_prompt(adapter, out / "input_frames" / "poker_task_object", "poker015_task_object", poker_ann, "a purple-backed playing card")
        preview = out / "Poker015_task_object_SAM31提示bootstrap.png"
        save_preview(preview, poker_ann, poker_mask, f"Poker015 target: {poker_result['status']}")
        artifact_refs[preview.name] = ref(preview)
        tasks["poker015"] = {"status": poker_result["status"], "object": poker_result, "propagation_executed": False}
        status = "PASSED_DEVELOPMENT" if all(x["status"] == "PASSED_DEVELOPMENT" for x in tasks.values()) else "FAILED_QUALITY_C"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "FAILED_RUNTIME_FINAL"
    finally:
        if adapter is not None:
            adapter.predictor.shutdown()

    candidates = {
        "schema_version": "mask-sam31-prompt-bootstrap-candidates-r22-v1",
        "created_at": now(),
        "current_baseline": "SAM3.1",
        "status": status,
        "tasks": tasks,
        "error": error,
        "all_candidate_rows_persisted": True,
        "propagation_executed": False,
        "authority_promoted": False,
    }
    write_json(out / "PROMPT_CANDIDATES.json", candidates)
    metrics = {
        "schema_version": "mask-sam31-prompt-bootstrap-metrics-r22-v1",
        "created_at": now(),
        "status": status,
        "chips010_status": tasks.get("chips010", {}).get("status", "NOT_RUN"),
        "poker015_status": tasks.get("poker015", {}).get("status", "NOT_RUN"),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "regressions_authorized": {"chips": tasks.get("chips010", {}).get("status") == "PASSED_DEVELOPMENT", "poker": tasks.get("poker015", {}).get("status") == "PASSED_DEVELOPMENT"},
        "claim_limit": "Prompt-bootstrap development metrics; no pixel accuracy or temporal identity result.",
    }
    write_json(out / "METRICS.json", metrics)
    next_action = {
        "schema_version": "mask-sam31-prompt-bootstrap-next-r22-v1",
        "created_at": now(),
        "status": status,
        "chips": "Freeze and run two A/B prompt-bootstrap regressions only if chips010_status=PASSED_DEVELOPMENT; otherwise no quality retry.",
        "poker": "Freeze and run two A/B prompt-bootstrap regressions only if poker015_status=PASSED_DEVELOPMENT; otherwise no quality retry.",
        "do_not": ["Do not run temporal propagation before seed pass.", "Do not promote authority.", "Do not replace SAM3.1 with SAM2.x."],
    }
    write_json(out / "NEXT_ACTION.json", next_action)
    decision = f"""# SAM3.1 提示 bootstrap canary

状态：`{status}`。

- Chips010：`{metrics['chips010_status']}`。
- Poker015：`{metrics['poker015_status']}`。
- 本轮只验证冻结 anchor 上的 box seed + point refinement；没有传播、没有时序身份结论、没有 Mask authority 晋升。
- 所有候选行均写入 `PROMPT_CANDIDATES.json`，包括质量失败路径。
- SAM3.1 仍是当前基线；SAM2.x 只允许作为 challenger。
"""
    write_text(out / "DECISION.md", decision)
    run_receipt = {
        "schema_version": "mask-sam31-prompt-bootstrap-run-r22-v1",
        "created_at": now(),
        "task_id": "mask_sam31_prompt_bootstrap_canary_r22",
        "attempt_id": out.name,
        "status": status,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "inputs": inputs,
        "checkpoint": ref(CHECKPOINT),
        "code": ref(Path(__file__)),
        "error": error,
        "authority_promoted": False,
    }
    write_json(out / "RUN_RECEIPT.json", run_receipt)
    manifest = {
        "schema_version": "mask-sam31-prompt-bootstrap-artifacts-r22-v1",
        "created_at": now(),
        "status": status,
        "artifacts": {**artifact_refs, "prompt_candidates": ref(out / "PROMPT_CANDIDATES.json"), "metrics": ref(out / "METRICS.json"), "next_action": ref(out / "NEXT_ACTION.json"), "decision": ref(out / "DECISION.md"), "run_receipt": ref(out / "RUN_RECEIPT.json")},
        "authority_promoted": False,
    }
    write_json(out / "ARTIFACT_MANIFEST.json", manifest)
    result = {
        "schema_version": "mask-sam31-prompt-bootstrap-result-r22-v1",
        "created_at": now(),
        "task_id": "mask_sam31_prompt_bootstrap_canary_r22",
        "attempt_id": out.name,
        "status": status,
        "artifact_revision": "R7_5_MASK_SAM31_PROMPT_BOOTSTRAP_1",
        "current_baseline": "SAM3.1",
        "metrics": ref(out / "METRICS.json"),
        "candidates": ref(out / "PROMPT_CANDIDATES.json"),
        "manifest": ref(out / "ARTIFACT_MANIFEST.json"),
        "next_action": ref(out / "NEXT_ACTION.json"),
        "error": error,
        "authority_promoted": False,
        "claim_limit": "Two frozen prompt-bootstrap development canaries only; no temporal identity, accuracy or Mask authority.",
    }
    write_json(out / "RESULT.json", result)
    print(json.dumps({"status": status, "result": ref(out / "RESULT.json")}, ensure_ascii=False))
    return 0 if status in {"PASSED_DEVELOPMENT", "FAILED_QUALITY_C"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
