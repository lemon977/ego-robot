#!/usr/bin/env python3
"""Exact78 P1 HaWoR recovery ladder with per-frame visual provenance.

L0 is an immutable fresh HaWoR result.  L1 re-runs the pinned HaWoR regressor
from same-frame role-mask boxes.  L2 calls the pinned official infiller and
only admits short, bounded gaps.  Remaining bounded gaps are interpolated for
VISUAL use; prefix/suffix gaps are held.  Only L0 OBSERVED rows are metric
observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import cv2
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


HAWOR = REPO_ROOT / "vendor/HaWoR"
SIDES = ("left", "right")
ROLES = ("left_human", "right_human")
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
METRIC_REPROJECTION_P95_PX_MAX = 12.0
OFFICIAL_INFILLER_HORIZON = 120


def checked(path: Path, reference: dict | None = None) -> Path:
    path = path.resolve(strict=True)
    if reference is not None:
        actual = artifact_ref(path)
        if actual["bytes"] != reference["bytes"] or actual["sha256"] != reference["sha256"]:
            raise RuntimeError(f"INPUT_SHA_DRIFT:{path}")
    return path


def localize(reference: dict) -> Path:
    path = Path(reference["path"])
    if not path.is_file():
        prefix = "/path/to/chaoyang/tasks/control/runs/"
        if str(path).startswith(prefix):
            path = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
                    / str(path)[len(prefix):])
    return checked(path, reference)


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def segments(mask: np.ndarray) -> list[np.ndarray]:
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return [part for part in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(part)]


def role_masks(manifest_path: Path | None, frame_count: int) -> tuple[list[list[np.ndarray | None]], dict]:
    output: list[list[np.ndarray | None]] = [[None] * frame_count for _ in range(2)]
    if manifest_path is None:
        return output, {"available": False, "reason": "NO_ROLE_MANIFEST"}
    manifest = load_json(manifest_path)
    rows = {int(row["source_frame"]): row for row in manifest.get("frames", [])}
    if set(rows) != set(range(frame_count)):
        return output, {"available": False, "reason": "ROLE_FRAME_SET_INCOMPLETE",
                        "present": len(rows), "expected": frame_count}
    sha_verified = 0
    for frame in range(frame_count):
        for side, role in enumerate(ROLES):
            reference = rows[frame].get("role_masks", {}).get(role)
            if not reference:
                continue
            try:
                path = localize(reference)
            except (FileNotFoundError, RuntimeError):
                continue
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            output[side][frame] = mask > 0
            sha_verified += 1
    return output, {"available": True, "manifest": artifact_ref(manifest_path),
                    "mask_files_sha_verified": sha_verified}


def mask_visible(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    return int(mask.sum()) >= max(32, int(round(mask.size * 0.00002)))


def mask_box(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    return np.asarray((xx.min(), yy.min(), xx.max() + 1, yy.max() + 1), dtype=np.float32)


def project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    value = points @ intrinsic.T
    return value[:, :2] / value[:, 2:3]


def admission(joints_camera: np.ndarray, vertices_camera: np.ndarray, intrinsic: np.ndarray,
              own: np.ndarray | None, other: np.ndarray | None) -> dict:
    if not mask_visible(own) or not np.isfinite(joints_camera).all() or np.any(joints_camera[:, 2] <= 1e-6):
        return {"pass": False, "reason": "OWN_ROLE_MASK_OR_POSITIVE_DEPTH_ABSENT"}
    assert own is not None
    uv = project(joints_camera, intrinsic)
    height, width = own.shape
    rounded = np.rint(uv).astype(np.int64)
    inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < width)
              & (rounded[:, 1] >= 0) & (rounded[:, 1] < height))
    distance = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 3)
    errors = np.full(len(uv), np.hypot(width, height), dtype=np.float64)
    errors[inside] = distance[rounded[inside, 1], rounded[inside, 0]]
    reprojection_p95 = float(np.percentile(errors, 95))
    vertex_uv = project(vertices_camera, intrinsic)
    finite = np.isfinite(vertex_uv).all(axis=1)
    predicted = np.zeros_like(own, dtype=np.uint8)
    if int(finite.sum()) >= 3:
        hull = cv2.convexHull(np.rint(vertex_uv[finite]).astype(np.int32))
        cv2.fillConvexPoly(predicted, hull, 1)
    own_union = int(np.logical_or(predicted > 0, own).sum())
    own_iou = float(np.logical_and(predicted > 0, own).sum() / max(own_union, 1))
    other_iou = 0.0
    if mask_visible(other):
        assert other is not None
        other_union = int(np.logical_or(predicted > 0, other).sum())
        other_iou = float(np.logical_and(predicted > 0, other).sum() / max(other_union, 1))
    passed = reprojection_p95 <= METRIC_REPROJECTION_P95_PX_MAX and own_iou >= other_iou
    return {"pass": bool(passed), "reprojection_p95_px": reprojection_p95,
            "mask_iou": own_iou, "opposite_role_mask_iou": other_iou,
            "gate": "EXISTING_REPROJECTION_P95_LE_12_AND_OWN_IOU_GE_OPPOSITE_IOU"}


def mano_materialize(run_mano, trans: np.ndarray, root_rot: np.ndarray,
                     hand_rot: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    with torch.inference_mode():
        result = run_mano(
            torch.from_numpy(trans[None].astype(np.float32)),
            rotation_matrix_to_angle_axis(torch.from_numpy(root_rot[None].astype(np.float32))),
            rotation_matrix_to_angle_axis(torch.from_numpy(hand_rot[None].astype(np.float32))),
            betas=torch.from_numpy(beta[None].astype(np.float32)), use_cuda=False,
        )
    return (result["joints"][0].detach().cpu().numpy().astype(np.float32),
            result["vertices"][0].detach().cpu().numpy().astype(np.float32))


def recover_observed_translations(arr: dict[str, np.ndarray], mano_functions: tuple) -> None:
    """Recover HaWoR init_trans exactly from retained MANO parameters/joints."""
    for side in (0, 1):
        frames = np.flatnonzero(arr["metric_observed"][side])
        if not len(frames):
            continue
        zeros = np.zeros((len(frames), 3), dtype=np.float32)
        joints_zero, _ = mano_materialize(
            mano_functions[side], zeros,
            arr["root_orient_camera"][side, frames],
            arr["hand_pose_rotmat"][side, frames],
            arr["betas"][side, frames],
        )
        arr["root_translation_camera"][side, frames] = (
            arr["joints_3d_camera"][side, frames, 0] - joints_zero[:, 0]
        )


def fill_world(arr: dict[str, np.ndarray], side: int, frame: int,
               joints_cam: np.ndarray, root_cam: np.ndarray, hand_rot: np.ndarray,
               beta: np.ndarray, trans_cam: np.ndarray, source: str, confidence: float) -> None:
    c2w = arr["c2w"][frame]
    arr["joints_3d_camera"][side, frame] = joints_cam
    arr["joints_3d_world"][side, frame] = joints_cam @ c2w[:3, :3].T + c2w[:3, 3]
    arr["joints_2d"][side, frame] = project(joints_cam, arr["intrinsics"][frame])
    arr["root_orient_camera"][side, frame] = root_cam
    arr["hand_pose_rotmat"][side, frame] = hand_rot
    arr["betas"][side, frame] = beta
    arr["root_translation_camera"][side, frame] = trans_cam
    arr["visual_valid"][side, frame] = True
    arr["provenance"][side, frame] = source
    arr["detector_confidence"][side, frame] = confidence


def infer_mask_prompted(arr: dict[str, np.ndarray], masks: list[list[np.ndarray | None]],
                        image_files: np.ndarray, model, focal: float, center: list[float],
                        mano_functions: tuple, audit: list[dict]) -> None:
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
    for side in (0, 1):
        missing = ~arr["metric_observed"][side]
        frames = np.asarray([frame for frame in np.flatnonzero(missing)
                             if mask_visible(masks[side][frame])], dtype=np.int64)
        observed_betas = arr["betas"][side, arr["metric_observed"][side]]
        if not len(frames) or not len(observed_betas):
            continue
        beta = np.nanmedian(observed_betas, axis=0).astype(np.float32)
        boxes = np.stack([mask_box(masks[side][int(frame)]) for frame in frames])
        result = model.inference(image_files[frames], boxes, img_focal=focal,
                                 img_center=center, do_flip=side == 0)
        root = result["pred_rotmat"][:, 0].detach().cpu()
        hand = result["pred_rotmat"][:, 1:].detach().cpu()
        trans = result["pred_trans"][:, 0].detach().cpu().numpy().astype(np.float32)
        if side == 0:
            root_aa = rotation_matrix_to_angle_axis(root)
            hand_aa = rotation_matrix_to_angle_axis(hand)
            root_aa[..., 1:] *= -1
            hand_aa[..., 1:] *= -1
            root = angle_axis_to_rotation_matrix(root_aa)
            hand = angle_axis_to_rotation_matrix(hand_aa)
        root_np = root.numpy().astype(np.float32)
        hand_np = hand.numpy().astype(np.float32)
        for local, frame_value in enumerate(frames):
            frame = int(frame_value)
            joints, vertices = mano_materialize(mano_functions[side], trans[local:local+1],
                                                root_np[local:local+1], hand_np[local:local+1],
                                                beta[None])
            gate = admission(joints[0], vertices[0], arr["intrinsics"][frame],
                             masks[side][frame], masks[1-side][frame])
            audit.append({"level": "L1", "side": SIDES[side], "frame": frame, **gate})
            if not gate["pass"]:
                continue
            fill_world(arr, side, frame, joints[0], root_np[local], hand_np[local], beta,
                       trans[local], "OBSERVED_MASK_PROMPTED", 1.0)
            arr["detector_boxes_xyxy"][side, frame] = boxes[local]


def interior_short_gap_frames(valid: np.ndarray) -> set[int]:
    allowed: set[int] = set()
    for gap in segments(~valid):
        if int(gap[0]) > 0 and int(gap[-1]) + 1 < len(valid) and len(gap) < OFFICIAL_INFILLER_HORIZON:
            allowed.update(map(int, gap))
    return allowed


def official_infiller(arr: dict[str, np.ndarray], masks: list[list[np.ndarray | None]],
                      staged_video: Path, upstream_work: Path, mano_functions: tuple,
                      audit: list[dict]) -> dict:
    eligible = [interior_short_gap_frames(arr["visual_valid"][side]) for side in (0, 1)]
    if not any(eligible):
        return {"attempted": False, "reason": "NO_SHORT_BOUNDED_GAP_AFTER_L1"}
    from scripts.scripts_test_video.hawor_video import hawor_infiller
    from hawor.utils.rotation import angle_axis_to_rotation_matrix
    video_dir = upstream_work / staged_video.stem
    track_dirs = sorted(video_dir.glob("tracks_*_*"))
    if len(track_dirs) != 1:
        return {"attempted": False, "reason": "TRACK_DIRECTORY_IDENTITY", "count": len(track_dirs)}
    tokens = track_dirs[0].name.split("_")
    start, end = int(tokens[-2]), int(tokens[-1])
    chunks = joblib.load(track_dirs[0] / "frame_chunks_all.npy")
    slam = video_dir / "SLAM" / f"hawor_slam_w_scale_{start}_{end}.npz"
    slam.parent.mkdir(exist_ok=True)
    quaternions = Rotation.from_matrix(arr["c2w"][:, :3, :3]).as_quat()
    trajectory = np.concatenate((arr["c2w"][:, :3, 3], quaternions), axis=1)
    if not slam.exists():
        np.savez_compressed(slam, traj=trajectory, scale=np.asarray(1.0))
    class Args: pass
    args = Args()
    args.video_path = str(staged_video)
    args.infiller_weight = str(HAWOR / "weights/hawor/checkpoints/infiller.pt")
    try:
        trans_world, root_world_aa, hand_aa, infill_betas, pred_valid = hawor_infiller(
            args, start, end, chunks)
    except Exception as error:  # The ladder degrades to L3; the session continues.
        return {"attempted": True, "status": "FAILED_RUNTIME_DEGRADED_TO_L3",
                "error": f"{type(error).__name__}:{error}"}
    trans_world = np.asarray(trans_world)
    root_world = angle_axis_to_rotation_matrix(torch.as_tensor(np.asarray(root_world_aa))).numpy()
    hand_rot = angle_axis_to_rotation_matrix(
        torch.as_tensor(np.asarray(hand_aa)).reshape(2, len(arr["c2w"]), 15, 3)).numpy()
    infill_betas = np.asarray(infill_betas)
    pred_valid = np.asarray(pred_valid, dtype=bool)
    accepted = 0
    for side in (0, 1):
        observed_betas = arr["betas"][side, arr["metric_observed"][side]]
        if not len(observed_betas):
            continue
        beta = np.nanmedian(observed_betas, axis=0).astype(np.float32)
        for frame in sorted(eligible[side]):
            if arr["visual_valid"][side, frame] or not pred_valid[side, frame] or not mask_visible(masks[side][frame]):
                continue
            w2c = np.linalg.inv(arr["c2w"][frame])
            root_cam = w2c[:3, :3] @ root_world[side, frame]
            trans_cam = w2c[:3, :3] @ (trans_world[side, frame] - arr["c2w"][frame, :3, 3])
            joints, vertices = mano_materialize(mano_functions[side], trans_cam[None], root_cam[None],
                                                hand_rot[side, frame][None], beta[None])
            gate = admission(joints[0], vertices[0], arr["intrinsics"][frame],
                             masks[side][frame], masks[1-side][frame])
            audit.append({"level": "L2", "side": SIDES[side], "frame": frame, **gate})
            if not gate["pass"]:
                continue
            fill_world(arr, side, frame, joints[0], root_cam, hand_rot[side, frame], beta,
                       trans_cam, "MODEL_INFILLED", 0.0)
            accepted += 1
    return {"attempted": True, "status": "COMPLETE", "accepted_side_frames": accepted,
            "horizon": OFFICIAL_INFILLER_HORIZON, "weights": artifact_ref(Path(args.infiller_weight))}


def world_params(arr: dict[str, np.ndarray], side: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c2w = arr["c2w"][frame]
    root_world = c2w[:3, :3] @ arr["root_orient_camera"][side, frame]
    trans_world = c2w[:3, :3] @ arr["root_translation_camera"][side, frame] + c2w[:3, 3]
    return root_world, trans_world, arr["hand_pose_rotmat"][side, frame], arr["betas"][side, frame]


def materialize_visual(arr: dict[str, np.ndarray], masks: list[list[np.ndarray | None]],
                       mano_functions: tuple, side: int, frame: int, root_world: np.ndarray,
                       trans_world: np.ndarray, hand_rot: np.ndarray, beta: np.ndarray,
                       source: str) -> None:
    w2c = np.linalg.inv(arr["c2w"][frame])
    root_cam = w2c[:3, :3] @ root_world
    trans_cam = w2c[:3, :3] @ (trans_world - arr["c2w"][frame, :3, 3])
    joints, _ = mano_materialize(mano_functions[side], trans_cam[None], root_cam[None],
                                 hand_rot[None], beta[None])
    mask = masks[side][frame]
    if mask_visible(mask) and joints[0, 0, 2] > 1e-6:
        assert mask is not None
        yy, xx = np.nonzero(mask)
        wanted = np.asarray((float(np.median(xx)), float(np.median(yy))))
        actual = project(joints[0, 0:1], arr["intrinsics"][frame])[0]
        z = float(joints[0, 0, 2])
        trans_cam[0] += (wanted[0] - actual[0]) * z / arr["intrinsics"][frame, 0, 0]
        trans_cam[1] += (wanted[1] - actual[1]) * z / arr["intrinsics"][frame, 1, 1]
        joints, _ = mano_materialize(mano_functions[side], trans_cam[None], root_cam[None],
                                     hand_rot[None], beta[None])
    fill_world(arr, side, frame, joints[0], root_cam, hand_rot, beta, trans_cam, source, 0.0)


def visual_completion(arr: dict[str, np.ndarray], masks: list[list[np.ndarray | None]],
                      mano_functions: tuple) -> dict:
    counts = Counter()
    held_motion: list[dict] = []
    for side in (0, 1):
        initial = arr["visual_valid"][side].copy()
        observed_ids = np.flatnonzero(initial)
        if not len(observed_ids):
            arr["provenance"][side] = "NEUTRAL"
            counts["NEUTRAL"] += len(initial)
            continue
        for gap in segments(~initial):
            first, last = int(gap[0]), int(gap[-1])
            if first > 0 and last + 1 < len(initial):
                left, right = first - 1, last + 1
                r0, t0, h0, b0 = world_params(arr, side, left)
                r1, t1, h1, b1 = world_params(arr, side, right)
                root_slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((r0, r1))))
                hand_slerps = [Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((h0[j], h1[j]))))
                               for j in range(15)]
                for local, frame in enumerate(gap, start=1):
                    u = local / (len(gap) + 1.0)
                    smooth = u * u * (3.0 - 2.0 * u)
                    root = root_slerp([smooth]).as_matrix()[0]
                    hand = np.stack([value([smooth]).as_matrix()[0] for value in hand_slerps])
                    materialize_visual(arr, masks, mano_functions, side, int(frame), root,
                                       (1.0-smooth)*t0 + smooth*t1, hand,
                                       (1.0-smooth)*b0 + smooth*b1, "INTERPOLATED")
                    counts["INTERPOLATED"] += 1
            else:
                endpoint = last + 1 if first == 0 else first - 1
                root, trans, hand, beta = world_params(arr, side, endpoint)
                centroids = []
                for frame in gap:
                    mask = masks[side][int(frame)]
                    if mask_visible(mask):
                        assert mask is not None
                        yy, xx = np.nonzero(mask)
                        centroids.append((float(np.median(xx)), float(np.median(yy))))
                    materialize_visual(arr, masks, mano_functions, side, int(frame), root, trans,
                                       hand, beta, "HELD")
                    counts["HELD"] += 1
                if len(centroids) >= 2:
                    displacement = float(np.linalg.norm(np.asarray(centroids[-1]) - np.asarray(centroids[0])))
                    held_motion.append({"side": SIDES[side], "first": first, "last": last,
                                        "mask_centroid_displacement_px": displacement,
                                        "quality": "DIAGNOSTIC_IF_MASK_MOVES"})
    return {"filled": dict(counts), "held_mask_motion": held_motion}


def review_video(source_video: Path, destination: Path, arr: dict[str, np.ndarray]) -> None:
    capture = cv2.VideoCapture(str(source_video))
    fps = float(np.asarray(arr["fps"]).reshape(-1)[0])
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = ((255, 120, 20), (20, 30, 255))
    try:
        for frame in range(len(arr["c2w"])):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"SOURCE_VIDEO_ENDED:{frame}")
            for side in (0, 1):
                if not arr["visual_valid"][side, frame]:
                    continue
                uv = np.rint(arr["joints_2d"][side, frame]).astype(np.int32)
                for chain in CHAINS:
                    cv2.polylines(image, [uv[np.asarray(chain)]], False, colors[side], 3, cv2.LINE_AA)
                cv2.circle(image, tuple(uv[0]), 7, colors[side], -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (width, 70), (0, 0, 0), -1)
            cv2.putText(image, f"{frame:04d} | L={arr['provenance'][0, frame]} | R={arr['provenance'][1, frame]}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "VISUAL completion; METRIC accepts OBSERVED only",
                        (10, 57), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 230, 255), 2, cv2.LINE_AA)
            writer.write(image)
    finally:
        capture.release()
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--raw-npz", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--upstream-work", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    output.mkdir(parents=True)
    source_video = checked(args.source_video)
    raw_npz = checked(args.raw_npz)
    raw_result = checked(args.raw_result)
    upstream = checked(args.upstream_work)
    if args.session_id not in str(raw_npz) or load_json(raw_result).get("session_id") != args.session_id:
        raise RuntimeError("SESSION_IDENTITY_DRIFT")
    arr = arrays(raw_npz)
    frame_count = len(arr["c2w"])
    arr["metric_observed"] = np.asarray(arr["observed"], dtype=bool).copy()
    arr["visual_valid"] = arr["metric_observed"].copy()
    arr["provenance"] = np.where(arr["metric_observed"], "OBSERVED", "MISSING").astype("<U32")
    arr["root_translation_camera"] = np.full((2, frame_count, 3), np.nan, np.float32)
    masks, mask_receipt = role_masks(args.role_manifest.resolve(strict=True) if args.role_manifest else None,
                                     frame_count)

    sys.path.insert(0, str(HAWOR))
    from scripts.scripts_test_video.hawor_video import load_hawor
    from hawor.utils.process import run_mano_left, run_mano
    mano_functions = (run_mano_left, run_mano)
    recover_observed_translations(arr, mano_functions)
    model, _ = load_hawor(str(HAWOR / "weights/hawor/checkpoints/hawor.ckpt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sequence = upstream / source_video.stem / "extracted_images"
    image_files = np.asarray(sorted(map(str, sequence.glob("*.jpg"))))
    if len(image_files) != frame_count:
        raise RuntimeError(f"EXTRACTED_FRAME_COUNT:{len(image_files)}:{frame_count}")
    height, width = cv2.imread(image_files[0]).shape[:2]
    focal = float(np.mean(arr["intrinsics"][:, [0, 1], [0, 1]]))
    audit: list[dict] = []
    infer_mask_prompted(arr, masks, image_files, model, focal, [width / 2.0, height / 2.0],
                        mano_functions, audit)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    staged_video = upstream / source_video.name
    infiller = official_infiller(arr, masks, staged_video, upstream, mano_functions, audit)
    completion = visual_completion(arr, masks, mano_functions)

    output_npz = output / "HAWOR_RECOVERED_VISUAL_MANO21.npz"
    np.savez_compressed(
        output_npz,
        joints_3d_camera=arr["joints_3d_camera"], joints_3d_world=arr["joints_3d_world"],
        joints_2d=arr["joints_2d"], betas=arr["betas"], root_orient_camera=arr["root_orient_camera"],
        hand_pose_rotmat=arr["hand_pose_rotmat"], root_translation_camera=arr["root_translation_camera"],
        observed=arr["metric_observed"], metric_observed=arr["metric_observed"],
        visual_valid=arr["visual_valid"], provenance=arr["provenance"],
        detector_confidence=arr["detector_confidence"], detector_boxes_xyxy=arr["detector_boxes_xyxy"],
        c2w=arr["c2w"], intrinsics=arr["intrinsics"], original_frame_indices=arr["original_frame_indices"],
        fps=arr["fps"], mano_joint_names=arr["mano_joint_names"],
        anatomical_side_names=arr["anatomical_side_names"], mano_wrist_index=arr["mano_wrist_index"],
        mano_tip_indices=arr["mano_tip_indices"], mano_mcp_indices=arr["mano_mcp_indices"],
    )
    review = output / "HAWOR_RECOVERY_PROVENANCE_REVIEW.mp4"
    review_video(source_video, review, arr)
    atomic_json(output / "FRAME_ADMISSION_AUDIT.json", {"rows": audit})
    provenance_counts = {SIDES[side]: dict(Counter(map(str, arr["provenance"][side]))) for side in (0, 1)}
    visible = np.asarray([[mask_visible(masks[side][frame]) for frame in range(frame_count)] for side in (0, 1)])
    no_action_visible = visible & ~arr["visual_valid"]
    denom = int(visible.sum())
    result = {
        "schema_version": "exact78-hawor-recovery-v2", "status": "COMPLETE_PENDING_VISUAL_REVIEW",
        "session_id": args.session_id, "frame_count": frame_count,
        "inputs": {"raw_npz": artifact_ref(raw_npz), "raw_result": artifact_ref(raw_result),
                   "source_video": artifact_ref(source_video), "role_manifest": mask_receipt,
                   "hawor_checkpoint": artifact_ref(HAWOR / "weights/hawor/checkpoints/hawor.ckpt")},
        "ladder": {"L0": "STANDARD_DETECT_TRACK", "L1": "SAME_FRAME_ROLE_MASK_PROMPTED_HAWOR",
                   "L2": "PINNED_OFFICIAL_INFILLER_SHORT_BOUNDED_GAPS", "L3": "UNKNOWN_THEN_VISUAL_COMPLETION"},
        "official_infiller": infiller, "visual_completion": completion,
        "metrics": {"metric_observed_side_frames": int(arr["metric_observed"].sum()),
                    "visual_valid_side_frames": int(arr["visual_valid"].sum()),
                    "role_mask_visible_side_frames": denom,
                    "visible_without_visual_action_side_frames": int(no_action_visible.sum()),
                    "visible_without_visual_action_fraction": float(no_action_visible.sum() / denom) if denom else None,
                    "provenance_counts": provenance_counts,
                    "L1_L2_admission_pass": sum(row["pass"] for row in audit),
                    "L1_L2_admission_fail": sum(not row["pass"] for row in audit),
                    "wall_seconds": time.monotonic() - started},
        "outputs": {"npz": artifact_ref(output_npz), "review": artifact_ref(review),
                    "frame_admission_audit": artifact_ref(output / "FRAME_ADMISSION_AUDIT.json")},
        "metric_policy": "ONLY_OBSERVED", "visual_policy": "ALL_SIX_PROVENANCE_CLASSES_LABELED",
        "training_eligible": False, "control_ground_truth": False, "visual_qualified": False,
        "claim_limit": "P1 offline visual action recovery candidate; prompted/infilled/interpolated/held rows are never metric observations or training labels.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "session": args.session_id,
                      "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
