"""Human hand/body tracks (world frame) -> Tianji q_arm and KaiHand q_hand.

* The robot base is fixed in the world: upright against PICO gravity, facing the
  mean horizontal head direction, shoulders a searched distance behind/below the
  tracked human shoulders (the Tianji arm is ~0.2 m longer than a human arm).
* Hand-root targets are absolute every frame: robot palm basis aligned to the
  human palm basis, robot palm centre on the human palm centre. No anchor frame,
  no constant offset absorbing IK error.
* Arm IK is warm-started frame to frame with a soft elbow hint that keeps the
  robot elbow on the same side of the shoulder-wrist line as the human elbow.
* Joint tracks are low-pass filtered; no frame is ever frozen by a speed gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from chaoyang.exact78_baseline.robot_model import FINGERS, SHOULDER_HEIGHT, Robot
from chaoyang.pipeline.robot_wrist_kai_adapter import final_v3_mano_palm_basis


SIDES = ("left", "right")
MANO_CHAINS = {"thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12),
               "ring": (0, 13, 14, 15, 16), "pinky": (0, 17, 18, 19, 20)}
TIP_WEIGHT, KNUCKLE_WEIGHT, HAND_CONTINUITY = 2.0, 1.0, 0.05
BODY = {"shoulder": (16, 17), "elbow": (18, 19), "wrist": (20, 21)}
POSITION_SCALE_M, ROTATION_SCALE_RAD = 0.004, 0.04
ELBOW_SCALE_M, CONTINUITY_SCALE_RAD = 0.08, 0.30
LIMIT_MARGIN_RAD = 0.02
BASE_BACK_M = (0.05, 0.12, 0.19, 0.26)
BASE_DOWN_M = (-0.06, 0.0, 0.06)
TRACK_POSITION_MM, TRACK_ROTATION_DEG = 20.0, 15.0
ARM_SPEED_RAD_S, ARM_ACCEL_RAD_S2 = 3.6, 54.0   # = 0.12 rad/frame, 0.06 rad/frame^2 at 30 fps


def _unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def _log_so3(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cosine)
    skew = np.asarray((matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]))
    if angle < 1e-6:
        return 0.5 * skew
    if angle > math.pi - 1e-4:
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0))
        axis *= np.sign(np.where(skew == 0, 1.0, skew))
        return angle * _unit(axis)
    return angle / (2.0 * math.sin(angle)) * skew


def _pose(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = position
    return value


# ---------------------------------------------------------------- hand-root targets

def hand_root_targets(hands: np.ndarray, robot: Robot) -> np.ndarray:
    """(2,N,21,3) MANO world joints -> (2,N,4,4) world hand-root poses."""
    count = hands.shape[1]
    targets = np.zeros((2, count, 4, 4))
    for side in range(2):
        contract = robot.hands[side]
        basis = final_v3_mano_palm_basis(hands[side], handedness=SIDES[side])        # (N,3,3)
        rotation = basis @ contract.basis.T
        centre = 0.5 * (hands[side][:, 0] + hands[side][:, [5, 9, 13, 17]].mean(axis=1))
        targets[side, :, :3, :3] = rotation
        targets[side, :, :3, 3] = centre - np.einsum("nij,j->ni", rotation, contract.palm_centre)
        targets[side, :, 3, 3] = 1.0
    return targets


# ---------------------------------------------------------------- base placement

def base_frame(gravity: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    up = -_unit(gravity)
    forward = c2w[:, :3, 2].mean(axis=0)
    forward = _unit(forward - forward.dot(up) * up)
    left = np.cross(up, forward)
    return np.stack((forward, left, up), axis=1)


def human_shoulders(body: np.ndarray, observed: np.ndarray, c2w: np.ndarray,
                    rotation: np.ndarray) -> np.ndarray:
    if observed.sum() >= 3:
        return np.median(body[observed][:, list(BODY["shoulder"])], axis=0)
    eye = c2w[:, :3, 3].mean(axis=0)
    forward, left, up = rotation.T
    return np.asarray([eye - 0.20 * up - 0.06 * forward + 0.19 * left,
                       eye - 0.20 * up - 0.06 * forward - 0.19 * left])


def base_pose(rotation: np.ndarray, shoulders: np.ndarray, back: float, down: float,
              lateral: float = 0.0) -> np.ndarray:
    forward, left, up = rotation.T
    middle = shoulders.mean(axis=0) - back * forward - down * up + lateral * left
    return _pose(rotation, middle - SHOULDER_HEIGHT * up)


# ---------------------------------------------------------------- arm IK

def elbow_hints(body: np.ndarray, observed: np.ndarray, T_W_B: np.ndarray,
                robot: Robot, root_base: np.ndarray) -> np.ndarray:
    """(2,N,3) soft elbow targets in the base frame."""
    inverse = np.linalg.inv(T_W_B)
    count = root_base.shape[1]
    hints = np.zeros((2, count, 3))
    down = np.asarray((0.0, 0.0, -1.0))
    for side in range(2):
        shoulder_r = robot.shoulder[side]
        outward = np.asarray((0.0, 1.0 if side == 0 else -1.0, 0.0))
        default = _unit(0.8 * down + 0.6 * outward)
        for frame in range(count):
            wrist_r = root_base[side, frame, :3, 3]
            axis = _unit(wrist_r - shoulder_r)
            offset = default
            if observed[frame]:
                s, e, w = (inverse[:3, :3] @ body[frame, BODY[k][side]] + inverse[:3, 3]
                           for k in ("shoulder", "elbow", "wrist"))
                human_axis = _unit(w - s)
                bend = (e - s) - (e - s).dot(human_axis) * human_axis
                if np.linalg.norm(bend) > 0.02:
                    offset = _unit(bend)
            offset = _unit(offset - offset.dot(axis) * axis)
            hints[side, frame] = 0.5 * (shoulder_r + wrist_r) + 0.12 * offset
    return hints


def solve_arm_frame(robot: Robot, side: int, target_root: np.ndarray, elbow: np.ndarray | None,
                    seed: np.ndarray, previous: np.ndarray | None, max_nfev: int = 120) -> np.ndarray:
    target = target_root @ np.linalg.inv(robot.flange_root[side])
    lower = robot.arm_lower[side] + LIMIT_MARGIN_RAD
    upper = robot.arm_upper[side] - LIMIT_MARGIN_RAD
    suffix = ("L", "R")[side]
    chain = robot.arms[side]

    def residual(q: np.ndarray) -> np.ndarray:
        frames = chain.fk(q)
        flange = frames[f"flange_{suffix}"]
        rows = [(flange[:3, 3] - target[:3, 3]) / POSITION_SCALE_M,
                _log_so3(target[:3, :3].T @ flange[:3, :3]) / ROTATION_SCALE_RAD]
        if elbow is not None:
            rows.append((frames[f"Link4_{suffix}"][:3, 3] - elbow) / ELBOW_SCALE_M)
        if previous is not None:
            rows.append((q - previous) / CONTINUITY_SCALE_RAD)
        return np.concatenate(rows)

    start = np.clip(seed, lower, upper)
    return least_squares(residual, start, bounds=(lower, upper), max_nfev=max_nfev,
                         ftol=1e-8, xtol=1e-8, gtol=1e-8).x


def arm_errors(robot: Robot, side: int, q: np.ndarray, target_root: np.ndarray) -> tuple[float, float]:
    actual = robot.arm_frames(side, q)["root"]
    position = float(np.linalg.norm(actual[:3, 3] - target_root[:3, 3]) * 1000.0)
    rotation = float(np.degrees(np.linalg.norm(_log_so3(target_root[:3, :3].T @ actual[:3, :3]))))
    return position, rotation


def _initial_seeds(robot: Robot, side: int) -> list[np.ndarray]:
    sign = 1.0 if side == 0 else -1.0
    middle = 0.5 * (robot.arm_lower[side] + robot.arm_upper[side])
    elbow_down = np.asarray((0.0, -1.0, 0.0, -1.4, 0.0, 0.0, 0.0)) * np.asarray((1, 1, sign, 1, sign, 1, 1))
    return [middle, elbow_down, np.zeros(7),
            np.asarray((-0.3 * sign, -1.1, -0.4 * sign, -1.1, 0.0, -0.5, 0.6 * sign))]


def solve_arm_track(robot: Robot, side: int, root_base: np.ndarray, hints: np.ndarray,
                    frames: list[int] | None = None, max_nfev: int = 120) -> np.ndarray:
    count = root_base.shape[0]
    order = list(range(count)) if frames is None else frames
    q = np.zeros((count, 7))
    previous = None
    for position, frame in enumerate(order):
        if previous is None:
            options = [solve_arm_frame(robot, side, root_base[frame], hints[frame], seed, None, 300)
                       for seed in _initial_seeds(robot, side)]
            scores = [sum(arm_errors(robot, side, option, root_base[frame])) for option in options]
            q[frame] = options[int(np.argmin(scores))]
        else:
            q[frame] = solve_arm_frame(robot, side, root_base[frame], hints[frame], previous,
                                       previous, max_nfev)
        previous = q[frame]
    return q


def place_base(robot: Robot, gravity: np.ndarray, c2w: np.ndarray, body: np.ndarray,
               body_observed: np.ndarray, root_world: np.ndarray,
               samples: int = 36) -> tuple[np.ndarray, dict[str, Any]]:
    rotation = base_frame(gravity, c2w)
    shoulders = human_shoulders(body, body_observed, c2w, rotation)
    count = root_world.shape[1]
    picks = sorted(set(np.linspace(0, count - 1, min(samples, count)).round().astype(int).tolist()))

    def evaluate(back: float, down: float, lateral: float) -> dict[str, Any]:
        T_W_B = base_pose(rotation, shoulders, back, down, lateral)
        inverse = np.linalg.inv(T_W_B)
        root_base = np.einsum("ij,snjk->snik", inverse, root_world)
        hints = elbow_hints(body, body_observed, T_W_B, robot, root_base)
        errors = []
        for side in range(2):
            q = solve_arm_track(robot, side, root_base[side], hints[side], picks, max_nfev=80)
            for frame in picks:
                position, rotation_deg = arm_errors(robot, side, q[frame], root_base[side, frame])
                errors.append(position + 0.5 * rotation_deg)
        errors = np.asarray(errors)
        return {"back_m": back, "down_m": down, "lateral_m": lateral,
                "score": float(np.percentile(errors, 90) + np.median(errors)),
                "median": float(np.median(errors)), "p90": float(np.percentile(errors, 90))}

    rows = [evaluate(back, down, 0.0) for back in BASE_BACK_M for down in BASE_DOWN_M]
    best = min(rows, key=lambda row: row["score"])
    for dback, ddown, dlat in ((.035, 0, 0), (-.035, 0, 0), (0, .03, 0), (0, -.03, 0), (0, 0, .03), (0, 0, -.03)):
        rows.append(evaluate(best["back_m"] + dback, best["down_m"] + ddown, best["lateral_m"] + dlat))
    best = min(rows, key=lambda row: row["score"])
    T_W_B = base_pose(rotation, shoulders, best["back_m"], best["down_m"], best["lateral_m"])
    return T_W_B, {"rule": "gravity_upright_heading_mean_camera_shoulder_prior_grid_v1",
                   "human_shoulders_world": shoulders.tolist(), "sample_frames": picks,
                   "candidates": rows, "selected": best, "T_world_base": T_W_B.tolist(),
                   "score_units": "p90+median of (position_mm + 0.5*rotation_deg)"}


# ---------------------------------------------------------------- hand IK

def _chain_features(local: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """``local``: (5,3) palm-basis points wrist, base, knuckle, joint, tip."""
    return {"bones": _unit(np.diff(local[1:], axis=0)),   # the three phalanx-level segments
            "tip": local[4] / scale, "knuckle": local[2] / scale}


def _human_features(points: np.ndarray, side: int) -> dict[str, dict[str, np.ndarray]]:
    basis = final_v3_mano_palm_basis(points, handedness=SIDES[side])
    local = (points - points[0]) @ basis
    scale = float(np.mean(np.linalg.norm(local[[5, 9, 13, 17]], axis=1)))
    return {f: _chain_features(local[list(MANO_CHAINS[f])], scale) for f in FINGERS}


def _robot_features(robot: Robot, side: int, finger: str, q_finger: np.ndarray) -> dict[str, np.ndarray]:
    contract = robot.hands[side]
    return _chain_features(robot.finger_points(side, finger, q_finger) @ contract.basis, contract.mcp_scale)


def solve_hand_track(robot: Robot, side: int, hands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-finger least squares; returns q (N,22) and per-frame max bone error (deg)."""
    contract = robot.hands[side]
    count = hands.shape[0]
    q = np.zeros((count, len(contract.names)))
    bone_error = np.zeros(count)
    previous = contract.neutral.copy()
    for frame in range(count):
        wanted = _human_features(hands[frame], side)
        current = previous.copy()
        worst = 0.0
        for finger in FINGERS:
            group = contract.groups[finger]
            lower, upper = contract.lower[group], contract.upper[group]
            seed = np.clip(previous[group], lower, upper)
            target = wanted[finger]

            knuckle_weight = KNUCKLE_WEIGHT if finger == "thumb" else 0.0

            def residual(local: np.ndarray) -> np.ndarray:
                actual = _robot_features(robot, side, finger, local)
                return np.concatenate(((actual["bones"] - target["bones"]).ravel(),
                                       TIP_WEIGHT * (actual["tip"] - target["tip"]),
                                       knuckle_weight * (actual["knuckle"] - target["knuckle"]),
                                       HAND_CONTINUITY * (local - seed)))

            free = upper - lower > 1e-9
            solved = seed.copy()
            if free.any():
                def residual_free(values: np.ndarray) -> np.ndarray:
                    whole = seed.copy()
                    whole[free] = values
                    return residual(whole)
                solved[free] = least_squares(residual_free, seed[free], bounds=(lower[free], upper[free]),
                                             max_nfev=60, ftol=1e-7, xtol=1e-7, gtol=1e-7).x
            current[group] = solved
            actual = _robot_features(robot, side, finger, solved)
            cosine = np.clip(np.sum(_unit(actual["bones"]) * _unit(target["bones"]), axis=-1), -1, 1)
            worst = max(worst, float(np.degrees(np.arccos(cosine)).max()))
        q[frame] = current
        bone_error[frame] = worst
        previous = current
    return q, bone_error


# ---------------------------------------------------------------- smoothing + metrics

def smooth_joints(q: np.ndarray, fps: float, lower: np.ndarray, upper: np.ndarray,
                  seconds: float = 0.23) -> np.ndarray:
    window = max(5, int(round(seconds * fps)) | 1)
    if len(q) < window:
        return q.copy()
    return np.clip(savgol_filter(q, window, 2, axis=0, mode="interp"), lower, upper)


@dataclass
class MotionResult:
    q_arm: np.ndarray            # (N,2,7)
    q_hand: np.ndarray           # (N,2,22)
    T_world_base: np.ndarray     # (4,4)
    root_world: np.ndarray       # (2,N,4,4) targets
    metrics: dict[str, Any]
    base_report: dict[str, Any]
    arm_position_mm: np.ndarray  # (2,N)
    arm_rotation_deg: np.ndarray # (2,N)
    hand_bone_deg: np.ndarray    # (2,N)


def retarget_session(session: Any, robot: Robot) -> MotionResult:
    count, fps = session.frame_count, session.fps
    root_world = hand_root_targets(session.hands, robot)
    T_W_B, base_report = place_base(robot, session.gravity, session.c2w, session.body,
                                    session.body_observed, root_world)
    inverse = np.linalg.inv(T_W_B)
    root_base = np.einsum("ij,snjk->snik", inverse, root_world)
    hints = elbow_hints(session.body, session.body_observed, T_W_B, robot, root_base)
    q_arm = np.zeros((count, 2, 7))
    q_hand = np.zeros((count, 2, 22))
    hand_bone = np.zeros((2, count))
    for side in range(2):
        raw = solve_arm_track(robot, side, root_base[side], hints[side])
        q_arm[:, side] = smooth_joints(raw, fps, robot.arm_lower[side], robot.arm_upper[side])
        contract = robot.hands[side]
        raw_hand, hand_bone[side] = solve_hand_track(robot, side, session.hands[side])
        q_hand[:, side] = smooth_joints(raw_hand, fps, contract.lower, contract.upper)
    position = np.zeros((2, count))
    rotation = np.zeros((2, count))
    for side in range(2):
        for frame in range(count):
            position[side, frame], rotation[side, frame] = arm_errors(
                robot, side, q_arm[frame, side], root_base[side, frame])
    velocity = np.abs(np.diff(q_arm, axis=0)) * fps if count > 1 else np.zeros((0, 2, 7))
    acceleration = np.abs(np.diff(q_arm, n=2, axis=0)) * fps * fps if count > 2 else np.zeros((0, 2, 7))
    observed = session.hand_observed
    tracked = (position <= TRACK_POSITION_MM) & (rotation <= TRACK_ROTATION_DEG)
    metrics = {
        "frame_count": count, "fps": fps,
        "gates": {"track_position_mm": TRACK_POSITION_MM, "track_rotation_deg": TRACK_ROTATION_DEG,
                  "arm_speed_rad_s": ARM_SPEED_RAD_S, "arm_accel_rad_s2": ARM_ACCEL_RAD_S2},
        "per_side": {},
    }
    for side, name in enumerate(SIDES):
        mask = observed[side]
        metrics["per_side"][name] = {
            "observed_frames": int(mask.sum()),
            "tracked_observed_frames": int((tracked[side] & mask).sum()),
            "tracking_pass_rate_observed": float((tracked[side] & mask).sum() / max(1, mask.sum())),
            "position_mm_median": float(np.median(position[side])),
            "position_mm_p95": float(np.percentile(position[side], 95)),
            "rotation_deg_median": float(np.median(rotation[side])),
            "rotation_deg_p95": float(np.percentile(rotation[side], 95)),
            "arm_speed_rad_s_max": float(velocity[:, side].max()) if len(velocity) else 0.0,
            "arm_accel_rad_s2_max": float(acceleration[:, side].max()) if len(acceleration) else 0.0,
            "speed_gate_violation_frames": int((velocity[:, side].max(axis=1) > ARM_SPEED_RAD_S).sum()) if len(velocity) else 0,
            "hand_bone_deg_median": float(np.median(hand_bone[side])),
            "hand_bone_deg_p95": float(np.percentile(hand_bone[side], 95)),
        }
    return MotionResult(q_arm=q_arm, q_hand=q_hand, T_world_base=T_W_B, root_world=root_world,
                        metrics=metrics, base_report=base_report, arm_position_mm=position,
                        arm_rotation_deg=rotation, hand_bone_deg=hand_bone)
