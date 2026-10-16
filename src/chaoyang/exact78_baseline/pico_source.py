"""PICO tracker session -> camera poses and hand/body motion in one world frame.

The raw sessions under ``chips_cards_tracker_0901/0902`` carry the headset's
own SLAM head pose, 26-joint OpenXR hand tracking and 24-joint body tracking,
plus a per-frame ``training_data.json`` written by the PICO dataset editor.

World frame ``W`` is the editor's ``first_frame_selected_camera_optical``.
Hand and body joints are read from the raw tracking file and mapped into ``W``
with the rigid transform that reproduces the editor's own world keypoints.

Camera poses are rebuilt as ``T_W_H0 @ head(i) @ X`` where ``X`` keeps the
editor's head->camera rotation but uses one global head->camera translation.
The 0902 editor build placed the camera at the stereo extrinsic centre, which
projects hands ~100 px off the image; the global value was fitted jointly on
five 0901+0902 sessions against HaWoR 2-D joints (21-36 px median residual on
every session, see docs/plans/EXACT78_TAKEOVER_20261004/20_PICO_SOURCE_ZH.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


RAW_ROOTS = (
    Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901"),
    Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902"),
)
HEAD_TO_CAMERA_TRANSLATION = np.asarray((-0.0377, 0.0044, -0.0713))
SIDES = ("left", "right")
# OpenXR-26 index for each MANO-21 joint (wrist, thumb cmc/mcp/ip/tip, 4x finger mcp/pip/dip/tip).
OPENXR_TO_MANO = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15,
                  17, 18, 19, 20, 22, 23, 24, 25)
BODY_JOINTS = {"left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
               "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
               "neck": 12, "head": 15, "spine3": 9, "pelvis": 0}
POSITION_VALID = 0x2  # XrSpaceLocationFlags bit, per joint
SHORT_GAP_SECONDS = 0.5
SMOOTH_WINDOW = 5


class PicoSourceError(ValueError):
    pass


@dataclass
class PicoSession:
    session_id: str
    raw_dir: Path
    video: Path
    fps: float
    frame_count: int
    width: int
    height: int
    K: np.ndarray                    # (3,3) rectified pinhole
    c2w: np.ndarray                  # (N,4,4) corrected camera->world
    c2w_editor: np.ndarray           # (N,4,4) as written by the editor
    hands: np.ndarray                # (2,N,21,3) MANO order, world, filled
    hand_observed: np.ndarray        # (2,N) bool
    hand_source: np.ndarray          # (2,N) label
    body: np.ndarray                 # (N,24,3) world, filled
    body_observed: np.ndarray        # (N,) bool
    gravity: np.ndarray              # (3,) unit vector in W
    calibration: dict[str, Any] = field(default_factory=dict)

    def camera_points(self, points_world: np.ndarray, frame: int) -> np.ndarray:
        w2c = np.linalg.inv(self.c2w[frame])
        return points_world @ w2c[:3, :3].T + w2c[:3, 3]

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        q = points_camera @ self.K.T
        return q[..., :2] / q[..., 2:3]


def find_session(session_id: str, roots: tuple[Path, ...] = RAW_ROOTS) -> Path:
    if os.environ.get("CHAOYANG_EXACT78_PREPARED_INPUTS"):
        from chaoyang.exact78_baseline.native_input import find_prepared
        return find_prepared(session_id)
    hits = [Path(p) for root in roots for p in glob.glob(str(root / "*" / session_id))]
    hits = [p for p in hits if (p / "clip_manifest.json").is_file()]
    if len(hits) != 1:
        raise PicoSourceError(f"SESSION_LOOKUP:{session_id}:{len(hits)}")
    return hits[0]


def list_sessions(roots: tuple[Path, ...] = RAW_ROOTS) -> list[str]:
    found = []
    for root in roots:
        for manifest in sorted(glob.glob(str(root / "*" / "*" / "clip_manifest.json"))):
            found.append(Path(manifest).parent.name)
    return found


def _pose7(text: str) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split(",")], dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise PicoSourceError("POSE7_INVALID")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def _procrustes(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid T with target ~ T @ source; returns (T, max residual metres)."""
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.diag((1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))))
    rotation = vt.T @ d @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ct - rotation @ cs
    residual = np.linalg.norm(source @ rotation.T + transform[:3, 3] - target, axis=1)
    return transform, float(residual.max())


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs where mask is False."""
    runs, start = [], None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def fill_track(values: np.ndarray, observed: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill unobserved frames of a (N, ...) track; never alters observed frames.

    Interior gaps are eased between the two bounding observations (cosine ramp, so
    the display never jumps); leading/trailing gaps hold the nearest observation.
    """
    count = len(values)
    filled = np.asarray(values, dtype=np.float64).copy()
    label = np.where(observed, "OBSERVED", "NONE").astype("<U16")
    if not observed.any():
        return filled, label
    short = max(1, int(round(SHORT_GAP_SECONDS * fps)))
    for start, stop in _segments(observed):
        if start == 0:
            filled[:stop] = filled[stop]
            label[:stop] = "EDGE_HOLD"
        elif stop == count:
            filled[start:] = filled[start - 1]
            label[start:] = "EDGE_HOLD"
        else:
            first, last = filled[start - 1], filled[stop]
            for index in range(start, stop):
                ratio = (index - start + 1) / (stop - start + 1)
                ease = 0.5 - 0.5 * math.cos(math.pi * ratio)
                filled[index] = (1.0 - ease) * first + ease * last
            label[start:stop] = "INTERP_SHORT" if stop - start <= short else "EASED_LONG"
    return filled, label


def smooth_observed(values: np.ndarray, observed: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centred moving average inside each observed run (edges use a shrunk window)."""
    result = np.asarray(values, dtype=np.float64).copy()
    half = window // 2
    runs = _segments(~observed)
    for start, stop in runs:
        for index in range(start, stop):
            radius = min(half, index - start, stop - 1 - index)
            result[index] = values[index - radius:index + radius + 1].mean(axis=0)
    return result


def load_session(session_id: str, *, roots: tuple[Path, ...] = RAW_ROOTS,
                 raw_dir: Path | None = None) -> PicoSession:
    root = raw_dir or find_session(session_id, roots)
    if (root / "SESSION_INPUT.json").is_file():
        from chaoyang.exact78_baseline.native_input import load_prepared
        return load_prepared(session_id, root)
    manifest = json.loads((root / "clip_manifest.json").read_text(encoding="utf-8"))
    video = root / manifest["files"]["video"]
    fps = float(manifest["video"]["fps"])
    width, height = (int(v) for v in manifest["video_rectification"]["resolution"])
    frame_files = sorted(glob.glob(str(root / "preprocess" / "all_data" / "*" / "training_data.json")))
    count = len(frame_files)
    if count != int(manifest["video"]["frame_count"]) or count == 0:
        raise PicoSourceError(f"FRAME_COUNT_DRIFT:{count}:{manifest['video']['frame_count']}")
    tracking_lines = (root / manifest["files"]["tracking"]).read_text(encoding="utf-8").splitlines()[1:]

    c2w_editor = np.zeros((count, 4, 4))
    head = np.zeros((count, 4, 4))
    raw_hands = np.full((2, count, 26, 3), np.nan)
    editor_world = np.full((2, count, 21, 3), np.nan)
    hand_ok = np.zeros((2, count), dtype=bool)
    body = np.full((count, 24, 3), np.nan)
    body_ok = np.zeros(count, dtype=bool)
    sync_ms = np.zeros(count)
    intrinsics = None
    pico_index = None
    for frame, path in enumerate(frame_files):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = payload["metadata"]
        if int(meta["idx"]) != frame:
            raise PicoSourceError(f"FRAME_INDEX_DRIFT:{frame}")
        c2w_editor[frame] = meta["c2w"]
        intrinsics = np.asarray(meta["k"], dtype=np.float64)
        sync_ms[frame] = float(meta.get("tracking_sync_error_ms", 0.0))
        row = json.loads(tracking_lines[int(meta["tracking_index"])])
        if int(row["timeStampNs"]) != int(meta["ts"]):
            raise PicoSourceError(f"TRACKING_TIMESTAMP_DRIFT:{frame}")
        head[frame] = _pose7(row["Head"]["pose"])
        for side, key in enumerate(("leftHand", "rightHand")):
            hand = row.get("Hand", {}).get(key, {})
            entity = payload["entities"]["hands"].get(SIDES[side]) or {}
            joints = hand.get("HandJointLocations") or []
            if (hand.get("isActive") and hand.get("poseValid", True) and len(joints) == 26
                    and all(int(joint.get("s", 0)) & POSITION_VALID for joint in joints)
                    and float(entity.get("confidence", 0.0)) > 0.5):
                raw_hands[side, frame] = [[float(v) for v in joint["p"].split(",")[:3]] for joint in joints]
                if entity.get("keypoints_3d_world") is not None:
                    editor_world[side, frame] = entity["keypoints_3d_world"]
                    pico_index = np.asarray(entity["pico_joint_indices"], dtype=int)
                hand_ok[side, frame] = bool(np.isfinite(raw_hands[side, frame]).all())
        tracking = row.get("bodyTracking", {})
        joints = row.get("Body", {}).get("joints") or []
        if tracking.get("isTracking") and len(joints) == 24:
            body[frame] = [[float(v) for v in joint["p"].split(",")[:3]] for joint in joints]
            body_ok[frame] = bool(np.isfinite(body[frame]).all())

    # World <- tracking-head-start transform from the editor's own world keypoints.
    pairs = [(raw_hands[s, f][pico_index], editor_world[s, f]) for s in range(2) for f in range(count)
             if hand_ok[s, f] and np.isfinite(editor_world[s, f]).all()] if pico_index is not None else []
    if pairs:
        T_W_H0, procrustes_max = _procrustes(np.concatenate([p[0] for p in pairs]),
                                             np.concatenate([p[1] for p in pairs]))
    else:
        flip = np.diag((1.0, -1.0, -1.0, 1.0))
        T_W_H0 = np.linalg.inv(head[0] @ flip) @ c2w_editor[0]
        procrustes_max = float("nan")
    X = np.asarray([np.linalg.inv(T_W_H0 @ head[f]) @ c2w_editor[f] for f in range(count)])
    X_editor = X[0]
    x_drift = float(np.max(np.abs(X - X_editor)))
    X_fixed = X_editor.copy()
    X_fixed[:3, 3] = HEAD_TO_CAMERA_TRANSLATION
    c2w = np.asarray([T_W_H0 @ head[f] @ X_fixed for f in range(count)])

    mano = raw_hands[:, :, list(OPENXR_TO_MANO)]
    hands_world = mano @ T_W_H0[:3, :3].T + T_W_H0[:3, 3]
    hands = np.empty_like(hands_world)
    labels = np.empty((2, count), dtype="<U16")
    for side in range(2):
        smoothed = smooth_observed(np.nan_to_num(hands_world[side]), hand_ok[side])
        hands[side], labels[side] = fill_track(smoothed, hand_ok[side], fps)
    body_world = body @ T_W_H0[:3, :3].T + T_W_H0[:3, 3]
    body_filled, _ = fill_track(np.nan_to_num(body_world), body_ok, fps)

    origin = _pose7(manifest["coordinate_system"]["origin_head_pose_in_source_world"])
    gravity = T_W_H0[:3, :3] @ (origin[:3, :3].T @ np.asarray((0.0, -1.0, 0.0)))
    gravity /= np.linalg.norm(gravity)
    if intrinsics is None:
        raise PicoSourceError("INTRINSICS_ABSENT")
    calibration = {
        "head_to_camera_translation_editor": X_editor[:3, 3].tolist(),
        "head_to_camera_translation_used": HEAD_TO_CAMERA_TRANSLATION.tolist(),
        "head_to_camera_rotation": X_editor[:3, :3].tolist(),
        "editor_head_to_camera_max_drift": x_drift,
        "world_from_tracking_procrustes_max_m": procrustes_max,
        "tracking_sync_error_ms_max_abs": float(np.max(np.abs(sync_ms))),
        "calibration_sha256_by_session": manifest.get("calibration_sha256_by_session"),
    }
    return PicoSession(session_id=session_id, raw_dir=root, video=video, fps=fps,
                       frame_count=count, width=width, height=height, K=intrinsics,
                       c2w=c2w, c2w_editor=c2w_editor, hands=hands, hand_observed=hand_ok,
                       hand_source=labels, body=body_filled, body_observed=body_ok,
                       gravity=gravity, calibration=calibration)
