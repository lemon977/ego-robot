#!/usr/bin/env python3
"""Bounded Tracker re-entry canaries for the exact78 role-removal Mask line.

This runner does not relax the old temporal gate.  It changes the source of a
periodic/re-entry Tracker prompt from a stale warped centroid to the current
frame bounded-v2 wrist/palm geometry.  Masks are fail-closed to empty when the
hand is not observed or the predicted wrist-band centre is outside the image.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_assisted_bilateral_flow_refresh as flow  # noqa: E402
from tools import run_chips001_pico_mask_temporal as base  # noqa: E402
from tools import run_configurable_bilateral_mask_canary as canary  # noqa: E402
from tools import run_newtask_baseline_sam31_mask_probe as baseline  # noqa: E402


CONTROL = PROJECT / "tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
HAWOR_INDEX = CONTROL / "HAWOR_TERMINAL_INDEX.json"
LEASE = PROJECT / "_run/GPU_LEASE.json"
OUTPUT = CONTROL / "mask_role_reentry_bounded_v2_1_canary"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
CASES = {
    "play_cards_0903_223": {"task": "poker", "start": 40, "end": 72},
    "play_cards_0903_243": {"task": "poker", "start": 45, "end": 72},
}
COLORS = {"left_tracker": (255, 210, 0), "right_tracker": (255, 0, 220)}
HEIGHT, WIDTH = 960, 1280
CADENCE = 4
ROI_RADIUS = 78


class CanaryError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise CanaryError(f"regular file required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanaryError(f"JSON object required: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def mutate_lease(holder: str, acquire: bool, reason: str | None = None) -> None:
    descriptor = os.open(LEASE.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = load(LEASE)
        if acquire:
            if current.get("status") != "RELEASED":
                raise CanaryError(f"GPU lease busy: {current.get('holder')}")
            start = datetime.now().astimezone()
            value = {
                "schema_version": "gpu-lease-v1",
                "status": "ACQUIRED",
                "holder": holder,
                "holder_pid": os.getpid(),
                "requester": str(Path(__file__).resolve()),
                "since": start.isoformat(timespec="seconds"),
                "scope": {
                    "gpu_indices": [0],
                    "purpose": "bounded-v2 current-frame wrist/palm Tracker re-entry canaries",
                    "max_wall_seconds": 1800,
                    "expires_at": (start + timedelta(minutes=30)).isoformat(timespec="seconds"),
                },
                "claim_limit": "Poker223 and Poker243 difficult re-entry windows only.",
            }
        else:
            if current.get("status") != "ACQUIRED" or current.get("holder") != holder:
                raise CanaryError("refuse to release another GPU holder")
            value = {**current, "status": "RELEASED", "released_at": now(), "release_reason": reason}
        temporary = LEASE.parent / f".{LEASE.name}.{os.getpid()}.tmp"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, LEASE)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def hawor_result(session: str) -> tuple[Path, dict[str, Any]]:
    terminals = load(HAWOR_INDEX)["terminals"]
    terminal = next((item for item in terminals if item["session_id"] == session), None)
    if terminal is None or terminal.get("downstream_authorized") is not True:
        raise CanaryError(f"HaWoR A/B terminal required: {session}")
    result = load(Path(terminal["result"]["path"]))
    value = result["outputs"]["npz"]
    path = Path(value["path"])
    if ref(path) != value:
        raise CanaryError(f"HaWoR NPZ drift: {session}")
    return path, terminal


def frame_path(session: str, frame_id: int) -> Path:
    task = CASES[session]["task"]
    manifest = load(CONTROL.parent / "20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json")
    row = next(item for item in manifest["sessions"] if item["session_id"] == session and item["task"] == task)
    return (Path(row["raw_path"]) / "preprocess/all_data" / f"{frame_id:05d}" / "rgb.png").resolve(strict=True)


def tracker_geometry(joints: np.ndarray, observed: np.ndarray, side: int, frame_id: int) -> dict[str, Any]:
    anatomical = "left_tracker" if side == 0 else "right_tracker"
    if not bool(observed[side, frame_id]):
        return {"role": anatomical, "expected_visible": False, "reason": "HAWOR_NOT_OBSERVED"}
    wrist = joints[side, frame_id, 0].astype(float)
    palm = np.nanmean(joints[side, frame_id, [1, 5, 9, 13, 17]], axis=0)
    if not np.isfinite(np.concatenate((wrist, palm))).all():
        return {"role": anatomical, "expected_visible": False, "reason": "NONFINITE_WRIST_PALM"}
    direction = wrist - palm
    norm = float(np.linalg.norm(direction))
    if norm < 5.0:
        return {"role": anatomical, "expected_visible": False, "reason": "DEGENERATE_WRIST_DIRECTION"}
    unit = direction / norm
    tangent = np.asarray([-unit[1], unit[0]])
    center = wrist + unit * 35.0
    inside = bool(10 <= center[0] < WIDTH - 10 and 10 <= center[1] < HEIGHT - 10)
    if not inside:
        return {
            "role": anatomical,
            "expected_visible": False,
            "reason": "PREDICTED_TRACKER_CENTER_OFFSCREEN",
            "center": center.tolist(),
        }
    positives = [center - tangent * 7.0, center + tangent * 7.0]
    negatives = [palm, wrist - unit * 8.0, center + unit * 58.0]
    return {
        "role": anatomical,
        "expected_visible": True,
        "reason": "BOUNDED_V2_WRIST_PALM_VISIBLE",
        "center": center.tolist(),
        "positives": [point.tolist() for point in positives],
        "negatives": [point.tolist() for point in negatives],
    }


def local_component(mask: np.ndarray, center: np.ndarray) -> np.ndarray:
    y_grid, x_grid = np.ogrid[:HEIGHT, :WIDTH]
    local = mask & ((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= ROI_RADIUS ** 2)
    count, labels, _, centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask)
    distances = np.linalg.norm(centroids[1:] - center[None, :], axis=1)
    selected = int(np.argmin(distances)) + 1
    return labels == selected


def proposal_gate(mask: np.ndarray, geometry: dict[str, Any], human: np.ndarray) -> dict[str, Any]:
    center = np.asarray(geometry["center"], dtype=float)
    positives = np.asarray(geometry["positives"], dtype=float)
    negatives = np.asarray(geometry["negatives"], dtype=float)
    rounded_pos = np.rint(positives).astype(int)
    rounded_neg = np.rint(negatives).astype(int)
    area = int(mask.sum())
    centroid = flow.centroid(mask)
    distance = float(math.dist(centroid, center)) if centroid is not None else float("inf")
    own_distance = float(flow.min_distance(mask, human)) if mask.any() and human.any() else float("inf")
    negative_exclusion = bool(all(not mask[y, x] for x, y in rounded_neg if 0 <= x < WIDTH and 0 <= y < HEIGHT))
    checks = {
        "present": bool(mask.any()),
        "positive_coverage": bool(all(mask[y, x] for x, y in rounded_pos)),
        "bounded_area": bool(80 <= area <= 30000),
        "centroid_near_current_wrist_anchor": bool(distance <= 45.0),
        "adjacent_to_current_human": bool(own_distance <= 32.0),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "area_pixels": area,
        "centroid_distance_to_current_anchor_px": distance,
        "own_human_min_distance_px": own_distance,
        "diagnostic_negative_prompt_exclusion": negative_exclusion,
        "diagnostic_note": "Negative prompts still constrain SAM, but leakage at an adjacent hand/forearm point is not a hard failure after current-wrist ROI bounding; those pixels are independently in the role-removal human mask.",
    }


def warp_gate(mask: np.ndarray, geometry: dict[str, Any], human: np.ndarray) -> dict[str, Any]:
    center = np.asarray(geometry["center"], dtype=float)
    area = int(mask.sum())
    centroid = flow.centroid(mask)
    distance = float(math.dist(centroid, center)) if centroid is not None else float("inf")
    own_distance = float(flow.min_distance(mask, human)) if mask.any() and human.any() else float("inf")
    checks = {
        "present": bool(mask.any()),
        "bounded_area": bool(80 <= area <= 30000),
        "centroid_near_current_wrist_anchor": bool(distance <= 52.0),
        "adjacent_to_current_human": bool(own_distance <= 36.0),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "area_pixels": area,
        "centroid_distance_to_current_anchor_px": distance,
        "own_human_min_distance_px": own_distance,
    }


def sam_current_frame(model: Any, path: Path, geometry: dict[str, Any], obj_id: int, root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    os.symlink(path, root / "00000.png")
    positives = np.asarray(geometry["positives"], dtype=float)
    negatives = np.asarray(geometry["negatives"], dtype=float)
    state = model.init_state(resource_path=str(root), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        model.add_prompt(
            inference_state=state,
            frame_idx=0,
            points=torch.tensor(positives / np.asarray([WIDTH, HEIGHT]), dtype=torch.float32),
            point_labels=torch.ones(len(positives), dtype=torch.int32),
            obj_id=obj_id,
            rel_coordinates=True,
            clear_old_points=True,
            output_prob_thresh=0.5,
        )
        points = np.concatenate((positives, negatives), axis=0)
        labels = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives)))).astype(np.int32)
        _, outputs = model.add_prompt(
            inference_state=state,
            frame_idx=0,
            points=torch.tensor(points / np.asarray([WIDTH, HEIGHT]), dtype=torch.float32),
            point_labels=torch.tensor(labels, dtype=torch.int32),
            obj_id=obj_id,
            rel_coordinates=True,
            clear_old_points=True,
            output_prob_thresh=0.5,
        )
        raw = flow.extract_id(outputs, obj_id, HEIGHT, WIDTH)
    finally:
        state.clear()
    center = np.asarray(geometry["center"], dtype=float)
    return local_component(raw, center), {
        "raw_area_pixels": int(raw.sum()),
        "bounded_component_area_pixels": int(local_component(raw, center).sum()),
    }


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (HEIGHT, WIDTH):
        raise CanaryError(f"bad mask: {path}")
    return image > 0


def overlay(image: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    result = image.copy()
    for role, mask in masks.items():
        color = np.asarray(COLORS[role], dtype=np.float32)
        result[mask] = np.clip(result[mask].astype(np.float32) * 0.42 + color * 0.58, 0, 255).astype(np.uint8)
    return result


def render_video(session: str, records: list[dict[str, Any]], paths: list[Path], old_root: Path, masks: dict[str, dict[int, np.ndarray]], output: Path) -> None:
    process = canary.start_video(output, 1280, 440, 30.0)
    font = ImageFont.truetype(str(FONT), 21)
    small = ImageFont.truetype(str(FONT), 17)
    assert process.stdin is not None
    for local, (record, path) in enumerate(zip(records, paths)):
        frame_id = int(record["frame"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        old = {role: read_mask(old_root / role / f"{frame_id:05d}.png") for role in COLORS}
        new = {role: masks[role][local] for role in COLORS}
        panels = [image, overlay(image, old), overlay(image, new)]
        canvas = np.zeros((440, 1280, 3), np.uint8)
        for index, panel in enumerate(panels):
            canvas[80:400, index * 426:(index + 1) * 426] = cv2.resize(panel, (426, 320), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        draw.text((10, 7), f"{session}｜Tracker离屏/重入 bounded-v2困难窗｜帧{frame_id:03d}", font=font, fill=(255, 255, 255))
        draw.text((80, 48), "原图", font=small, fill=(230, 230, 230))
        draw.text((505, 48), "旧：上一warp质心漂移", font=small, fill=(255, 150, 150))
        draw.text((890, 48), "新：当前腕/掌重锚定", font=small, fill=(130, 255, 160))
        draw.text((10, 407), f"青=左Tracker  紫=右Tracker  当前provider: {record['providers']}", font=small, fill=(230, 230, 230))
        process.stdin.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).tobytes())
    canary.finish_video(process)


def run_case(model: Any, session: str, spec: dict[str, Any]) -> dict[str, Any]:
    output = OUTPUT / session
    if output.exists():
        raise CanaryError(f"no-clobber output exists: {output}")
    output.mkdir(parents=True)
    start, end = int(spec["start"]), int(spec["end"])
    frame_ids = list(range(start, end + 1))
    paths = [frame_path(session, frame_id) for frame_id in frame_ids]
    npz_path, _ = hawor_result(session)
    with np.load(npz_path, allow_pickle=False) as archive:
        joints = archive["joints_2d"].astype(float)
        observed = archive["observed"].astype(bool)
    old_root = CONTROL / "mask_role_removal_v1" / spec["task"] / session / "raw_role_masks"
    human = {
        role: {local: read_mask(old_root / role / f"{frame_id:05d}.png") for local, frame_id in enumerate(frame_ids)}
        for role in ("left_human", "right_human")
    }
    cache = flow.FlowCache(paths, 640, 480)
    masks: dict[str, dict[int, np.ndarray]] = {role: {} for role in COLORS}
    metadata: dict[str, Any] = {}
    frame_records = [{"frame": frame_id, "providers": {}} for frame_id in frame_ids]
    for side, role in enumerate(("left_tracker", "right_tracker")):
        current = np.zeros((HEIGHT, WIDTH), bool)
        previous_visible = False
        attempts, accepted, rejected, warped_accepted, empty_closed = 0, 0, 0, 0, 0
        detail = []
        for local, frame_id in enumerate(frame_ids):
            geometry = tracker_geometry(joints, observed, side, frame_id)
            if not geometry["expected_visible"]:
                current = np.zeros_like(current)
                masks[role][local] = current.copy()
                previous_visible = False
                empty_closed += 1
                provider = "EMPTY_FAIL_CLOSED_NOT_VISIBLE"
                detail.append({"frame": frame_id, "geometry": geometry, "provider": provider})
                frame_records[local]["providers"][role] = provider
                continue
            center = np.asarray(geometry["center"], dtype=float)
            if local and previous_visible and current.any():
                warped, flow_record = cache.warp(current, local - 1, local)
                warped = local_component(warped, center)
                current_warp_gate = warp_gate(warped, geometry, human[f"{'left' if side == 0 else 'right'}_human"][local])
            else:
                warped = np.zeros_like(current)
                flow_record = None
                current_warp_gate = {"pass": False, "reason": "NO_VISIBLE_PREDECESSOR"}
            refresh = bool((frame_id - start) % CADENCE == 0 or not previous_visible or not current.any())
            proposal = None
            current_proposal_gate = None
            sam_summary = None
            if refresh:
                attempts += 1
                proposal, sam_summary = sam_current_frame(
                    model,
                    paths[local],
                    geometry,
                    202 if side == 0 else 201,
                    output / "refresh_inputs" / role / f"{frame_id:05d}",
                )
                current_proposal_gate = proposal_gate(
                    proposal, geometry, human[f"{'left' if side == 0 else 'right'}_human"][local]
                )
            if proposal is not None and current_proposal_gate and current_proposal_gate["pass"]:
                current = proposal
                accepted += 1
                provider = "SAM31_CURRENT_BOUNDED_WRIST_PALM_ACCEPTED"
            elif current_warp_gate.get("pass"):
                current = warped
                warped_accepted += 1
                provider = "RAW_DIS_WARP_CURRENT_WRIST_ROI_ACCEPTED"
                if refresh:
                    rejected += 1
            else:
                current = np.zeros_like(current)
                empty_closed += 1
                provider = "EMPTY_FAIL_CLOSED_REFRESH_AND_WARP_REJECTED"
                if refresh:
                    rejected += 1
            masks[role][local] = current.copy()
            previous_visible = True
            detail.append({
                "frame": frame_id,
                "geometry": geometry,
                "refresh_attempted": refresh,
                "sam_summary": sam_summary,
                "proposal_gate": current_proposal_gate,
                "flow_record": flow_record,
                "warp_gate": current_warp_gate,
                "provider": provider,
                "area_pixels": int(current.sum()),
            })
            frame_records[local]["providers"][role] = provider
        expected = sum(item["geometry"]["expected_visible"] for item in detail)
        present = sum(bool(masks[role][local].any()) and item["geometry"]["expected_visible"] for local, item in enumerate(detail))
        drift = sum(bool(masks[role][local].any()) and not item["geometry"]["expected_visible"] for local, item in enumerate(detail))
        metadata[role] = {
            "window_full_denominator": len(frame_ids),
            "expected_visible_denominator": expected,
            "present_on_expected_visible": present,
            "visible_coverage_fraction": float(present / max(expected, 1)),
            "offscreen_or_unobserved_empty_frames": sum(not item["geometry"]["expected_visible"] for item in detail),
            "background_drift_on_not_visible_frames": drift,
            "refresh_attempt_count": attempts,
            "refresh_accept_count": accepted,
            "refresh_reject_count": rejected,
            "warp_accept_count": warped_accepted,
            "empty_fail_closed_count": empty_closed,
            "records": detail,
        }
    mask_root = output / "tracker_masks"
    for role, stream in masks.items():
        (mask_root / role).mkdir(parents=True)
        for local, frame_id in enumerate(frame_ids):
            if not cv2.imwrite(str(mask_root / role / f"{frame_id:05d}.png"), stream[local].astype(np.uint8) * 255):
                raise CanaryError("mask write failed")
    video = output / f"{session}_TRACKER_REENTRY_BOUNDED_V2_中文对比.mp4"
    render_video(session, frame_records, paths, old_root, masks, video)
    numeric_gates = {
        "full_window_denominator_recorded": all(value["window_full_denominator"] == len(frame_ids) for value in metadata.values()),
        "no_mask_when_tracker_expected_offscreen_or_unobserved": all(value["background_drift_on_not_visible_frames"] == 0 for value in metadata.values()),
        "visible_tracker_coverage_at_least_0p80": all(value["visible_coverage_fraction"] >= 0.80 for value in metadata.values()),
        "all_output_is_current_wrist_roi_bounded": True,
    }
    result = {
        "schema_version": "exact78-tracker-reentry-bounded-v2-canary-v1",
        "created_at": now(),
        "status": "PASS_NUMERIC_WAIT_AGENT_VISUAL" if all(numeric_gates.values()) else "HOLD_NUMERIC",
        "session": session,
        "task": spec["task"],
        "window": {"start": start, "end": end, "frame_count": len(frame_ids)},
        "method": "SAM31_PERIODIC_CURRENT_FRAME_BOUNDED_V2_WRIST_PALM_REANCHOR_WITH_RAW_DIS_WARP_AND_EMPTY_FAIL_CLOSED",
        "numeric_gates": numeric_gates,
        "tracker_summary": {role: {key: value for key, value in data.items() if key != "records"} for role, data in metadata.items()},
        "tracker_records": metadata,
        "artifacts": {"review_video": ref(video), "tracker_masks": str(mask_root)},
        "pins": {
            "runner": ref(Path(__file__)),
            "hawor_npz": ref(npz_path),
            "old_role_result": ref(old_root.parent / "RESULT.json"),
            "checkpoint_sha256": base.CHECKPOINT_SHA256,
        },
        "authorization": "No batch authorization until both numeric results and agent visual reviews pass.",
    }
    write_new(output / "NUMERIC_RESULT.json", result)
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8") as stream:
        for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
            stream.write(f"{sha(path)}  {path.relative_to(output)}\n")
    return {"session": session, "status": result["status"], "result": ref(output / "NUMERIC_RESULT.json"), "review_video": ref(video)}


def main() -> int:
    if OUTPUT.exists():
        raise CanaryError(f"no-clobber output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    holder = "exact78-role-bounded-v2-reentry-canaries"
    mutate_lease(holder, True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter = None
    try:
        adapter_module = baseline.load_adapter_module()
        if str(base.CODE_ROOT) not in sys.path:
            sys.path.insert(0, str(base.CODE_ROOT))
        adapter, build_evidence = adapter_module.build_pinned_adapter(
            official_code_root=base.CODE_ROOT, checkpoint_path=base.CHECKPOINT
        )
        cases = [run_case(adapter.model, session, spec) for session, spec in CASES.items()]
        aggregate = {
            "schema_version": "exact78-tracker-reentry-bounded-v2-canary-aggregate-v1",
            "created_at": now(),
            "status": "PASS_NUMERIC_WAIT_TWO_AGENT_VISUAL_REVIEWS" if all(item["status"].startswith("PASS") for item in cases) else "HOLD_NUMERIC",
            "cases": cases,
            "pins": {"runner": ref(Path(__file__)), "checkpoint_sha256": base.CHECKPOINT_SHA256, "sam_build_evidence": build_evidence},
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "authorization": "Role batch remains paused until both visual reviews explicitly pass.",
        }
        write_new(OUTPUT / "NUMERIC_RESULT.json", aggregate)
        return 0 if aggregate["status"].startswith("PASS") else 2
    finally:
        if adapter is not None:
            adapter.predictor.shutdown()
        mutate_lease(holder, False, "BOUNDED_V2_REENTRY_CANARIES_COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
