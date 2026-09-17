"""Minimal integration of frozen SAM3.1 seed replay and causal propagation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from chaoyang.pipeline.sam31_causal_seed_propagation_adapter_r22 import (
    CausalSeedContract,
    collect_causal_forward,
)


class TemporalIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenSeedReplay:
    session_id: str
    accepted_object_id: int
    point_xy: tuple[int, int]
    box_xyxy: tuple[int, int, int, int]
    text: str | None
    start_frame_index: int
    frame_count: int
    source_frame_offset: int
    height: int
    width: int


def _rows(masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, spec: FrozenSeedReplay, stage: str) -> list[dict[str, Any]]:
    px, py = spec.point_xy
    x0, y0, x1, y1 = spec.box_xyxy
    result = []
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        inside = int(mask[y0 : y1 + 1, x0 : x1 + 1].sum())
        minimum_area = min(64, max(1, int(spec.height * spec.width * 0.01)))
        result.append({
            "stage": stage,
            "object_id": int(ids[index]),
            "model_score": float(scores[index]),
            "area_pixels": area,
            "positive_point_covered": bool(area and mask[py, px]),
            "mask_fraction_inside_box": inside / max(area, 1),
            "eligible": bool(area >= minimum_area and mask[py, px] and inside / max(area, 1) >= 0.25),
        })
    return sorted(result, key=lambda row: (row["eligible"], row["model_score"]), reverse=True)


def replay_seed_then_collect(
    adapter: Any,
    resource_path: Path,
    spec: FrozenSeedReplay,
    normalize_outputs: Callable[[dict[str, Any], int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    """Replay the frozen seed, require its accepted ID, then propagate forward."""
    x0, y0, x1, y1 = spec.box_xyxy
    if not (0 <= x0 < x1 < spec.width and 0 <= y0 < y1 < spec.height):
        raise TemporalIntegrationError("frozen box outside image")
    adapter.handle_request({"type": "start_session", "resource_path": str(resource_path.resolve()), "session_id": spec.session_id})
    seed_rows: list[dict[str, Any]] = []
    try:
        state = adapter._session(spec.session_id)
        box = [[x0 / spec.width, y0 / spec.height, (x1 - x0 + 1) / spec.width, (y1 - y0 + 1) / spec.height]]
        _, output = adapter.model.add_prompt(
            inference_state=state, frame_idx=spec.start_frame_index,
            text_str=spec.text or "visual", boxes_xywh=box, box_labels=[1],
            output_prob_thresh=0.5,
        )
        masks, scores, ids = normalize_outputs(output, spec.height, spec.width)
        seed_rows.extend(_rows(masks, scores, ids, spec, "BOX_SEED"))
        accepted = [row for row in seed_rows if row["eligible"] and row["object_id"] == spec.accepted_object_id]
        if not accepted:
            raise TemporalIntegrationError("accepted object ID absent from frozen box replay")
        points = torch.tensor([[spec.point_xy[0] / spec.width, spec.point_xy[1] / spec.height]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int32)
        _, output = adapter.model.add_prompt(
            inference_state=state, frame_idx=spec.start_frame_index,
            text_str=None, clear_old_points=True, points=points, point_labels=labels,
            boxes_xywh=None, box_labels=None, clear_old_boxes=False,
            output_prob_thresh=0.5, obj_id=spec.accepted_object_id,
            rel_coordinates=True,
        )
        masks, scores, ids = normalize_outputs(output, spec.height, spec.width)
        point_rows = _rows(masks, scores, ids, spec, "POINT_REFINEMENT")
        seed_rows.extend(point_rows)
        if not any(row["eligible"] and row["object_id"] == spec.accepted_object_id for row in point_rows):
            raise TemporalIntegrationError("accepted object ID absent after point refinement")
        contract = CausalSeedContract(
            session_id=spec.session_id,
            accepted_object_id=spec.accepted_object_id,
            start_frame_index=spec.start_frame_index,
            frame_count=spec.frame_count,
            source_frame_offset=spec.source_frame_offset,
            height=spec.height,
            width=spec.width,
        )
        propagation_rows, masks_by_source = collect_causal_forward(adapter, contract, normalize_outputs)
        return seed_rows, propagation_rows, masks_by_source
    finally:
        adapter.handle_request({"type": "close_session", "session_id": spec.session_id})


def replay_detector_then_track_stable_point_id(
    adapter: Any,
    resource_path: Path,
    spec: FrozenSeedReplay,
    normalize_outputs: Callable[[dict[str, Any], int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    tracking_object_id: int,
    priming_object_id: int | None = 9901,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    """Validate the frozen detector choice, then track one explicit stable ID.

    The pinned multiplex runtime suppresses its first point object.  A separate
    ignored priming object is therefore installed before the actual role.  The
    detector raw ID remains frozen evidence only; it is never assumed to be a
    stable tracker ID across frames.
    """
    x0, y0, x1, y1 = spec.box_xyxy
    if (priming_object_id is not None and tracking_object_id == priming_object_id) or tracking_object_id < 0:
        raise TemporalIntegrationError("invalid stable tracking object ID")
    detection_session = f"{spec.session_id}-detector"
    tracking_session = f"{spec.session_id}-stable-track"
    adapter.handle_request({"type": "start_session", "resource_path": str(resource_path.resolve()), "session_id": detection_session})
    seed_rows: list[dict[str, Any]] = []
    try:
        state = adapter._session(detection_session)
        box = [[x0 / spec.width, y0 / spec.height, (x1 - x0 + 1) / spec.width, (y1 - y0 + 1) / spec.height]]
        _, output = adapter.model.add_prompt(
            inference_state=state, frame_idx=spec.start_frame_index,
            text_str=spec.text or "visual", boxes_xywh=box, box_labels=[1],
            output_prob_thresh=0.5,
        )
        masks, scores, ids = normalize_outputs(output, spec.height, spec.width)
        seed_rows.extend(_rows(masks, scores, ids, spec, "FROZEN_DETECTOR_REPLAY"))
        if not any(row["eligible"] and row["object_id"] == spec.accepted_object_id for row in seed_rows):
            raise TemporalIntegrationError("frozen detector object ID absent")
    finally:
        adapter.handle_request({"type": "close_session", "session_id": detection_session})

    adapter.handle_request({"type": "start_session", "resource_path": str(resource_path.resolve()), "session_id": tracking_session})
    try:
        state = adapter._session(tracking_session)
        if priming_object_id is not None:
            adapter.model.add_prompt(
                inference_state=state, frame_idx=spec.start_frame_index,
                text_str=None, clear_old_points=True,
                points=torch.tensor([[0.02, 0.02]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int32),
                boxes_xywh=None, box_labels=None, obj_id=priming_object_id,
                rel_coordinates=True, output_prob_thresh=0.5,
            )
        box = [[x0 / spec.width, y0 / spec.height, (x1 - x0 + 1) / spec.width, (y1 - y0 + 1) / spec.height]]
        adapter.model.add_prompt(
            inference_state=state, frame_idx=spec.start_frame_index,
            text_str=None, clear_old_points=True, points=None,
            point_labels=None, boxes_xywh=box, box_labels=[1],
            clear_old_boxes=True, obj_id=tracking_object_id,
            rel_coordinates=True, output_prob_thresh=0.5,
        )
        points = torch.tensor([[spec.point_xy[0] / spec.width, spec.point_xy[1] / spec.height]], dtype=torch.float32)
        _, output = adapter.model.add_prompt(
            inference_state=state, frame_idx=spec.start_frame_index,
            text_str=None, clear_old_points=True, points=points,
            point_labels=torch.tensor([1], dtype=torch.int32),
            boxes_xywh=None, box_labels=None, clear_old_boxes=False,
            obj_id=tracking_object_id, rel_coordinates=True, output_prob_thresh=0.5,
        )
        masks, scores, ids = normalize_outputs(output, spec.height, spec.width)
        tracking_spec = FrozenSeedReplay(
            session_id=spec.session_id, accepted_object_id=tracking_object_id,
            point_xy=spec.point_xy, box_xyxy=spec.box_xyxy, text=spec.text,
            start_frame_index=spec.start_frame_index, frame_count=spec.frame_count,
            source_frame_offset=spec.source_frame_offset, height=spec.height, width=spec.width,
        )
        stable_rows = _rows(masks, scores, ids, tracking_spec, "STABLE_POINT_BOX_TRACK_SEED")
        seed_rows.extend(stable_rows)
        if not any(row["eligible"] and row["object_id"] == tracking_object_id for row in stable_rows):
            raise TemporalIntegrationError("stable point/box tracking ID absent at seed")
        contract = CausalSeedContract(
            session_id=tracking_session, accepted_object_id=tracking_object_id,
            start_frame_index=spec.start_frame_index, frame_count=spec.frame_count,
            source_frame_offset=spec.source_frame_offset, height=spec.height, width=spec.width,
            ignored_object_ids=(priming_object_id,) if priming_object_id is not None else (),
        )
        propagation_rows, masks_by_source = collect_causal_forward(adapter, contract, normalize_outputs)
        return seed_rows, propagation_rows, masks_by_source
    finally:
        adapter.handle_request({"type": "close_session", "session_id": tracking_session})


def persist_compact_evidence(
    output_dir: Path,
    seed_rows: list[dict[str, Any]],
    propagation_rows: list[dict[str, Any]],
    masks_by_source: dict[int, np.ndarray],
    frame_loader: Callable[[int], np.ndarray],
    maximum_failure_pngs: int = 8,
) -> dict[str, Any]:
    """Persist every compact row, packed masks, and bounded stream-failure PNGs."""
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_path = output_dir / "SEED_REPLAY_ROWS.json"
    seed_path.write_text(json.dumps(seed_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_path = output_dir / "COMPACT_PROPAGATION_ROWS.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in propagation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    frames = sorted(masks_by_source)
    stack = np.stack([masks_by_source[frame].reshape(-1) for frame in frames])
    packed_path = output_dir / "PACKED_MASKS.npz"
    np.savez_compressed(packed_path, source_frames=np.asarray(frames, np.int32), packed=np.packbits(stack, axis=1))
    failures = [row for row in propagation_rows if row["reason"] != "ACCEPTED_SEED_ID_PRESENT" or not row["present"]][:maximum_failure_pngs]
    failure_paths = []
    for row in failures:
        frame = int(row["source_frame"])
        image = np.asarray(frame_loader(frame)).copy()
        mask = masks_by_source[frame]
        image[mask] = (0.45 * image[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.putText(image, f"frame {frame}: {row['reason']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        path = output_dir / f"failure_{frame:05d}.png"
        if not cv2.imwrite(str(path), image):
            raise TemporalIntegrationError("failure PNG write failed")
        failure_paths.append(str(path))
    return {"seed_rows": str(seed_path), "compact_rows": str(rows_path), "packed_masks": str(packed_path), "failure_pngs": failure_paths}
