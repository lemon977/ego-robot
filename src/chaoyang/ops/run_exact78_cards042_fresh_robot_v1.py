"""Render the selected Cards042 Robot motion on the fresh Clean candidate.

The 76--91 visible-card support is rebuilt from current Raw pixels and current
role masks.  Depth/Object6D are deliberately not consumed: contact ownership
remains the selected two-dimensional review rule and has no physical authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.exact78_task_runtime_v1 import authorize, checked, verify_refs
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared
from chaoyang.pipeline import exact78_robot_raster_v1 as raster
from chaoyang.pipeline.exact78_visual_compositor_v1 import compose_visual_frame
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


FRAME_COUNT = 171
CONTACT = frozenset(range(76, 92))
ANCHORS = frozenset((0, 42, 76, 77, 82, 84, 87, 91, 96, 125, 137, 170))
WIDTH, HEIGHT = 640, 480


def arrays(reference: dict) -> dict[str, np.ndarray]:
    with np.load(checked(reference), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def image(reference: dict, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(checked(reference)), flag)
    if value is None:
        raise RuntimeError(f"IMAGE_DECODE:{reference['path']}")
    return value


def save(path: Path, value: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not cv2.imwrite(str(path), value):
        raise RuntimeError(f"NO_CLOBBER_IMAGE:{path}")
    return artifact_ref(path)


def encoder(path: Path, width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        cwd=REPO_ROOT,
    )


def close_encoder(worker: subprocess.Popen) -> None:
    assert worker.stdin is not None
    worker.stdin.close()
    if worker.wait(timeout=120):
        raise RuntimeError("VIDEO_ENCODE_FAILED")


def abort_encoder(worker: subprocess.Popen) -> None:
    if worker.poll() is None:
        worker.terminate()
        worker.wait(timeout=10)


def decode(path: Path, frames: int, size: tuple[int, int]) -> bool:
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, value = capture.read()
            if not ok:
                break
            if value.shape[1::-1] != size:
                return False
            count += 1
    finally:
        capture.release()
    return count == frames


def title(value: np.ndarray, text: str) -> np.ndarray:
    output = value.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def current_human_mask(role_row: dict) -> np.ndarray:
    output = np.zeros((960, 1280), dtype=bool)
    for role in ("left_human", "right_human"):
        output |= image(role_row["role_masks"][role], cv2.IMREAD_GRAYSCALE) > 0
    return output


def contact_cards(raw: np.ndarray, human: np.ndarray) -> tuple[np.ndarray, dict]:
    """Observed pixels of the three cards; no hidden-card geometry is invented."""
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    candidate = (
        (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 165)
        & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8)
    roi = np.zeros_like(candidate)
    roi[400:680, 620:1060] = 1
    candidate &= roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    selected = np.zeros_like(candidate)
    component_rows = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        stationary = area > 2000 and 660 < x < 850 and 430 < y < 470
        moving = area >= 50 and x >= 900 and y >= 500
        if stationary or moving:
            selected[labels == label] = 1
            component_rows.append({
                "kind": "STATIONARY" if stationary else "MOVING_VISIBLE",
                "bbox": [x, y, width, height],
                "blue_pixels": area,
            })
    expanded = cv2.dilate(
        selected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ) > 0
    visible = expanded & ~human
    return visible, {
        "components": component_rows,
        "component_count": len(component_rows),
        "visible_pixels": int(visible.sum()),
        "human_pixels_excluded": int(np.logical_and(expanded, human).sum()),
        "method": "FIXED_ROI_BLUE_ENTITY_COMPONENTS_DILATE10_MINUS_CURRENT_HUMAN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    supplied = load_json(args.packet)
    packet, root = authorize(supplied["task_id"])
    if packet != supplied:
        raise RuntimeError("PACKET_MISMATCH")
    verify_refs(packet["inputs"])
    verify_refs(packet["runtime_code"])

    clean_result = load_json(checked(packet["inputs"]["clean_result"]))
    if clean_result.get("status") != "PASS_EXECUTION_PENDING_VISUAL_REVIEW" or not all(clean_result.get("hard_gates", {}).values()):
        raise RuntimeError("FRESH_CLEAN_EXECUTION_PASS_REQUIRED")
    clean_manifest = load_json(checked(packet["inputs"]["clean_manifest"]))
    prepare = load_json(checked(packet["inputs"]["clean_prepare"]))
    role_manifest = load_json(checked(packet["inputs"]["role_manifest"]))
    object_manifest = load_json(checked(packet["inputs"]["object_manifest"]))
    clean_rows = {row["source_frame"]: row for row in clean_manifest["rows"]}
    prepare_rows = {row["source_frame"]: row for row in prepare["rows"]}
    role_rows = {row["source_frame"]: row for row in role_manifest["frames"]}
    object_rows = {row["source_frame"]: row for row in object_manifest["frames"]}
    expected = set(range(FRAME_COUNT))
    if any(set(rows) != expected for rows in (clean_rows, prepare_rows, role_rows, object_rows)):
        raise RuntimeError("FRAME_SET_DRIFT")

    hawor = arrays(packet["inputs"]["selected_hawor"])
    arm = arrays(packet["inputs"]["selected_arm"])
    hand = arrays(packet["inputs"]["selected_hand"])
    if not (len(hawor["c2w"]) == len(arm["q_arm"]) == len(hand["q_hand"]) == FRAME_COUNT):
        raise RuntimeError("MOTION_FRAME_COUNT_DRIFT")
    if not np.array_equal(arm["valid_side_frame"], hand["valid_side_frame"]):
        raise RuntimeError("ARM_HAND_VALIDITY_DRIFT")
    if not np.all(arm["valid_side_frame"]):
        raise RuntimeError("SELECTED_CANONICAL_MOTION_MUST_BE_COMPLETE")
    if not (np.isfinite(arm["q_arm"]).all() and np.isfinite(hand["q_hand"]).all()):
        raise RuntimeError("SELECTED_CANONICAL_STATE_NONFINITE")
    world_base = np.asarray(arm["T_world_base"])
    if world_base.ndim == 3:
        if not np.allclose(world_base, world_base[0], atol=1e-12):
            raise RuntimeError("ROBOT_BASE_MOVES")
        world_base = world_base[0]

    output = root / "robot"
    output.mkdir()
    composite_path = output / "cards042_fresh_clean_robot.mp4"
    review_path = output / "cards042_fresh_clean_robot_review_4panel.mp4"
    external_path = output / "cards042_selected_robot_external.mp4"
    contact_path = output / "cards042_contact_76_91_review.mp4"
    composite_video = encoder(composite_path, WIDTH, HEIGHT)
    review_video = encoder(review_path, WIDTH * 2, HEIGHT * 2)
    external_video = encoder(external_path, WIDTH, HEIGHT)
    contact_video = encoder(contact_path, WIDTH * 2, HEIGHT)

    assets = load_pinned_robot_assets(REPO_ROOT)
    flange = raster.naturalv2_local_triangles(checked(packet["inputs"]["flange_mesh"]))
    cache: dict = {}
    global_camera = shared.same_direction_global_camera(np.linalg.inv(hawor["c2w"][0]) @ world_base)
    global_k = np.asarray(((575.0, 0, 319.5), (0, 575.0, 239.5), (0, 0, 1.0)))
    rows = []
    contact_masks = []
    started = time.monotonic()
    try:
        for frame in range(FRAME_COUNT):
            raw = image(prepare_rows[frame]["raw"])
            clean = image(clean_rows[frame]["clean"])
            human_native = current_human_mask(role_rows[frame])
            object_native = image(object_rows[frame]["mask"], cv2.IMREAD_GRAYSCALE) > 0
            contact_native = np.zeros((960, 1280), dtype=bool)
            contact_evidence = None
            if frame in CONTACT:
                contact_native, contact_evidence = contact_cards(raw, human_native)
                contact_masks.append(contact_native)
            visible_native = (object_native | contact_native) & ~human_native
            restored_native = clean.copy()
            restored_native[visible_native] = raw[visible_native]

            clean_small = cv2.resize(restored_native, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            raw_small = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            visible = cv2.resize(visible_native.astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
            human_small = cv2.resize(human_native.astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
            unknown = np.zeros((HEIGHT, WIDTH), dtype=bool)

            camera = np.linalg.inv(hawor["c2w"][frame]) @ world_base
            intrinsics = np.asarray(hawor["intrinsics"][frame], dtype=np.float64).copy()
            intrinsics[0] *= WIDTH / 1280.0
            intrinsics[1] *= HEIGHT / 960.0
            robot, labels, buffers = raster.render_robot(
                raster, assets, arm["q_arm"][frame], hand["q_hand"][frame],
                arm["T_tool_hand_root"], camera, intrinsics, cache, flange,
                complete_robot=False, return_buffers=True, width=WIDTH, height=HEIGHT,
            )
            alpha = labels >= 0
            if not np.all(np.isfinite(buffers["depth_m"][alpha]) & (buffers["depth_m"][alpha] > 0)):
                raise RuntimeError(f"ROBOT_DEPTH_INVALID:{frame}")
            composed = compose_visual_frame(
                clean_small, robot, alpha, visible, human_small, unknown,
                robot_observed=True, masks_available=True,
            )
            final = composed["image"]
            if np.any(final[composed["object_front"]] != clean_small[composed["object_front"]]):
                raise RuntimeError(f"OBJECT_FRONT_PROTECTION_DRIFT:{frame}")

            external_rgb, external_labels, external_buffers = raster.render_robot(
                raster, assets, arm["q_arm"][frame], hand["q_hand"][frame],
                arm["T_tool_hand_root"], global_camera, global_k, cache, flange,
                complete_robot=True, return_buffers=True, width=WIDTH, height=HEIGHT,
            )
            external_alpha = external_labels >= 0
            if not np.all(np.isfinite(external_buffers["depth_m"][external_alpha]) & (external_buffers["depth_m"][external_alpha] > 0)):
                raise RuntimeError(f"EXTERNAL_ROBOT_DEPTH_INVALID:{frame}")
            external = np.full_like(external_rgb, 242)
            external[external_alpha] = external_rgb[external_alpha]
            robot_only = np.full_like(robot, 242)
            robot_only[alpha] = robot[alpha]
            review = np.concatenate(
                (
                    np.concatenate((title(clean_small, f"FRESH CLEAN {frame}"), title(robot_only, "FIRST-PERSON ROBOT")), axis=1),
                    np.concatenate((title(final, "CLEAN + ROBOT"), title(external, "INDEPENDENT WHOLE ROBOT")), axis=1),
                ),
                axis=0,
            )
            assert composite_video.stdin and review_video.stdin and external_video.stdin
            composite_video.stdin.write(final.tobytes())
            review_video.stdin.write(review.tobytes())
            external_video.stdin.write(external.tobytes())
            if frame in CONTACT:
                ownership = (clean_small.astype(np.float32) * .32).astype(np.uint8)
                ownership[composed["robot_front"]] = (30, 210, 30)
                ownership[composed["object_front"]] = (220, 80, 20)
                ownership[composed["boundary_unknown"]] = (30, 180, 245)
                contact_review = np.concatenate((title(raw_small, f"RAW {frame}"), title(final, "COMPOSITE + OWNERSHIP INSET")), axis=1)
                inset = cv2.resize(title(ownership, "GREEN ROBOT / BLUE CARD / AMBER UNKNOWN"), (480, 360))
                contact_review[60:420, 720:1200] = inset
                assert contact_video.stdin
                contact_video.stdin.write(contact_review.tobytes())

            row = {
                "frame_id": frame,
                "robot_pixels": int(alpha.sum()),
                "external_robot_pixels": int(external_alpha.sum()),
                "visible_object_pixels": int(visible.sum()),
                "robot_front_pixels": int(composed["robot_front"].sum()),
                "object_front_pixels": int(composed["object_front"].sum()),
                "unknown_boundary_pixels": int(composed["boundary_unknown"].sum()),
                "contact_rebuild": contact_evidence,
                "clean_source": clean_rows[frame]["clean"],
                "raw_source": prepare_rows[frame]["raw"],
            }
            if frame in ANCHORS:
                row["composite"] = save(output / "anchors/composite" / f"{frame:06d}.png", final)
                row["review"] = save(output / "anchors/review" / f"{frame:06d}.png", review)
                rgba = cv2.cvtColor(robot, cv2.COLOR_BGR2BGRA)
                rgba[..., 3] = alpha.astype(np.uint8) * 255
                row["robot_rgba"] = save(output / "anchors/robot_rgba" / f"{frame:06d}.png", rgba)
                if frame in CONTACT:
                    row["contact_mask"] = save(output / "anchors/contact_mask" / f"{frame:06d}.png", contact_native.astype(np.uint8) * 255)
            rows.append(row)
            if frame % 20 == 0:
                print(json.dumps({"frame": frame, "seconds": round(time.monotonic() - started, 1)}), flush=True)
        for worker in (composite_video, review_video, external_video, contact_video):
            close_encoder(worker)
    except BaseException:
        for worker in (composite_video, review_video, external_video, contact_video):
            abort_encoder(worker)
        raise

    consecutive_contact_iou = []
    for first, second in zip(contact_masks[:-1], contact_masks[1:], strict=True):
        consecutive_contact_iou.append(float(np.logical_and(first, second).sum() / max(int(np.logical_or(first, second).sum()), 1)))
    contact_rows = [row for row in rows if row["frame_id"] in CONTACT]
    hard_gates = {
        "frame_count_exact": len(rows) == FRAME_COUNT,
        "robot_present_all_frames": all(row["robot_pixels"] > 0 for row in rows),
        "whole_robot_present_all_frames": all(row["external_robot_pixels"] > 0 for row in rows),
        "contact_three_or_more_components": all(row["contact_rebuild"]["component_count"] >= 3 for row in contact_rows),
        "contact_visible_area_bounded": all(12000 <= row["contact_rebuild"]["visible_pixels"] <= 20000 for row in contact_rows),
        "contact_temporal_iou_min_0p75": min(consecutive_contact_iou) >= .75,
        "composite_video_full_decode": decode(composite_path, FRAME_COUNT, (WIDTH, HEIGHT)),
        "review_video_full_decode": decode(review_path, FRAME_COUNT, (WIDTH * 2, HEIGHT * 2)),
        "external_video_full_decode": decode(external_path, FRAME_COUNT, (WIDTH, HEIGHT)),
        "contact_video_full_decode": decode(contact_path, len(CONTACT), (WIDTH * 2, HEIGHT)),
        "metric_depth_not_consumed": True,
        "object6d_not_consumed": True,
    }
    passed = all(hard_gates.values())
    atomic_json(output / "FRAME_MANIFEST.json", {
        "schema_version": "exact78-cards042-fresh-robot-frame-manifest-v1",
        "session": "play_cards_0902_042",
        "rows": rows,
    })
    result = {
        "schema_version": "exact78-cards042-fresh-robot-result-v1",
        "status": "PASS_VISUAL_REVIEW_CANDIDATE" if passed else "FAILED_QUALITY_C",
        "hard_gates": hard_gates,
        "metrics": {
            "frame_count": len(rows),
            "contact_temporal_iou_min": min(consecutive_contact_iou),
            "contact_temporal_iou_median": float(np.median(consecutive_contact_iou)),
            "wall_seconds": time.monotonic() - started,
        },
        "inputs": packet["inputs"],
        "outputs": {
            "composite": artifact_ref(composite_path),
            "review_4panel": artifact_ref(review_path),
            "external_robot": artifact_ref(external_path),
            "contact_review": artifact_ref(contact_path),
            "frame_manifest": artifact_ref(output / "FRAME_MANIFEST.json"),
        },
        "motion_origin": "INDEPENDENTLY_MIGRATED_USER_SELECTED_CANONICAL_STATES_NOT_FRESH_HAWOR_INFERENCE",
        "clean_quality": "DEVELOPMENT_VISUAL_INPUT_INTERNAL_REVIEW_ACCEPTED",
        "contact_quality": "TWO_DIMENSIONAL_VISUAL_OWNERSHIP_NO_PHYSICAL_AUTHORITY",
        "scene_depth_consumed": False,
        "object6d_consumed": False,
        "human_review": "PENDING",
        "visual_quality_adopted": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_occlusion_authority": False,
        "deletion_authorized": False,
        "claim_limit": (
            "Cards042 development visual using fresh current Clean/masks and selected canonical motion. "
            "No metric depth, Object6D, physical contact, training, control or deletion authority."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(root / "RESULT.json", {
        "status": result["status"],
        "robot_result": artifact_ref(output / "RESULT.json"),
        "baseline_adopted": False,
        "deletion_authorized": False,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
