"""Bounded CPU-only four-lane quality increments; no model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle, validate_artifact_ref,
)
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets
from chaoyang.pipeline.v5_sensor import project
from chaoyang.ops.run_human_to_robot_10h_007_context import _independent_root_fk

TASK = "human_to_robot_four_lane_quality_increment_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PRIOR = REPO_ROOT / "_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001"
R2 = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
V5 = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"


def _new_dir(lane: str, name: str) -> Path:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    rows = index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != TASK or not rows[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    path = ATTEMPT / "lanes" / lane / name
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def _image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def scene() -> dict:
    """Separate actual mask consumption from residual 007 inpainting failure."""
    old = REPO_ROOT / ("_run/current/human_to_robot_007_device_donor_probe_20260923/"
                       "attempts/attempt_0001/lanes/scene/device_donor_probe_v1/RESULT.json")
    source = PRIOR / "lanes/scene/context_007_v1"
    complaints = load_json(old)["complaint_points"]
    review = load_json(source / "AI_VISUAL_REVIEW.json")
    if review["reviewed_source_frames"] != [181, 184, 190, 193, 196]:
        raise RuntimeError("SCENE_REVIEW_IDENTITY_DRIFT")
    # First attempt retained: it exposed a model-domain/raw-domain mix-up.
    out = _new_dir("scene", "complaint_consumption_v1_runtime_fix1")
    local = 184 - 160
    raw_path = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                            "lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw/000184.png")
    model_path = source / f"input/model_masks/{local:06d}.png"
    write_path = source / f"input/write/{local:06d}.png"
    protect_path = source / f"input/protect/{local:06d}.png"
    clean_path = source / f"clean/{184-181:06d}.png"
    raw, clean = _image(raw_path), _image(clean_path)
    model = _image(model_path, cv2.IMREAD_GRAYSCALE)
    write = _image(write_path, cv2.IMREAD_GRAYSCALE)
    protect = _image(protect_path, cv2.IMREAD_GRAYSCALE)
    if raw.shape != clean.shape or raw.shape != (960, 1280, 3) or model.shape != (720, 960):
        raise RuntimeError("SCENE_DOMAIN_DRIFT")
    panel = raw.copy()
    rows = []
    for entry in complaints:
        x, y = int(entry["x"]), int(entry["y"])
        model_y, model_x = min(719, int(y * .75)), min(959, int(x * .75))
        in_model, in_write, in_protect = bool(model[model_y, model_x]), bool(write[y, x]), bool(protect[y, x])
        classification = ("MODEL_AND_WRITE_CONSUMED" if in_model and in_write and not in_protect
                          else "OUTSIDE_MODEL_OR_WRITE" if not in_model or not in_write else "PROTECTED")
        color = (30, 220, 30) if classification == "MODEL_AND_WRITE_CONSUMED" else (0, 180, 255)
        cv2.circle(panel, (x, y), 12, color, 2)
        cv2.putText(panel, entry["name"], (max(5, x - 50), max(18, y - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2)
        rows.append({"name": entry["name"], "pixel_xy": [x, y], "model_pixel_xy": [model_x, model_y],
                     "in_model_mask": in_model, "in_write": in_write, "in_protect": in_protect,
                     "consumption": classification, "raw_bgr": raw[y, x].tolist(),
                     "new_clean_bgr": clean[y, x].tolist(),
                     "identity_authority": "FIXED_AI_COMPLAINT_POINT_NOT_PIXEL_GT"})
    overlay = out / "007_FRAME184_COMPLAINT_TO_MODEL_CONSUMPTION.png"
    if not cv2.imwrite(str(overlay), panel):
        raise RuntimeError("SCENE_OVERLAY_WRITE_FAILED")
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_SCENE_CONSUMPTION_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "source_frame_id": 184,
              "source_points": artifact_ref(old), "model_input": artifact_ref(model_path),
              "write_input": artifact_ref(write_path), "protect_input": artifact_ref(protect_path),
              "raw": artifact_ref(raw_path), "new_clean": artifact_ref(clean_path),
              "prior_visual_review": artifact_ref(source / "AI_VISUAL_REVIEW.json"),
              "rows": rows, "consumed_complaint_points": sum(r["consumption"] == "MODEL_AND_WRITE_CONSUMED" for r in rows),
              "outside_or_protected_points": sum(r["consumption"] != "MODEL_AND_WRITE_CONSUMED" for r in rows),
              "overlay": artifact_ref(overlay), "execution": "ACTUAL_INPUTS_READ",
              "quality": "NEW_CLEAN_REJECTED_PREVIOUSLY", "improvement": "NONE_CLAIMED",
              "adoption": "NOT_ADOPTED",
              "next_action": "Do not repeat mask-only fixes for the six already consumed device points; establish white-line identity before deletion.",
              "claim_limit": "Six fixed device complaint points reached model and write support but residual appearance remains; white-line identity is unresolved."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "SCENE_CONSUMPTION_CLASSIFIED", "result": str(out / "RESULT.json")}


SENSOR_CASES = (
    ("play_cards_0916_097", "run_097_wave0", 165),
    ("play_cards_0916_098", "run_098_wave1", 179),
    ("play_cards_0916_101", "run_101_wave1", 122),
)


def sensor() -> dict:
    """Report joint-level image-frustum losses, not a guessed wrist accuracy."""
    # First attempt retained: 098/101 were published under wave1, not wave0.
    out = _new_dir("sensor", "projection_frustum_v1_runtime_fix1")
    sessions = []
    for session, folder, frames in SENSOR_CASES:
        motion_path = R2 / "lanes/lane3_sensor" / folder / "HAND_MOTION_V1.npz"
        with np.load(motion_path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if data["frame_id"].tolist() != list(range(frames)):
            raise RuntimeError(f"SENSOR_FRAME_ID_DRIFT:{session}")
        points = data["joints21_camera"]
        source_valid = np.asarray(data["joint_valid"], bool)
        uv, positive = project(points, data["camera_K"])
        finite = np.isfinite(points).all(axis=-1) & np.isfinite(uv).all(axis=-1)
        in_frame = (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 960)
        states = {
            "SOURCE_INVALID": ~source_valid,
            "NONFINITE_SOURCE_OR_PROJECTION": source_valid & ~finite,
            "BEHIND_OR_NEAR_CAMERA": source_valid & finite & ~positive,
            "POSITIVE_Z_OUT_OF_FRAME": source_valid & finite & positive & ~in_frame,
            "PROJECTABLE_IN_FRAME": source_valid & finite & positive & in_frame,
        }
        if np.any(sum(state.astype(np.int8) for state in states.values()) != 1):
            raise RuntimeError(f"SENSOR_STATE_PARTITION:{session}")
        counts_by_frame = {key: value.sum(axis=(1, 2)).astype(np.int32) for key, value in states.items()}
        curves = out / f"{session}_PROJECTION_STATES.npz"
        np.savez_compressed(curves, frame_id=data["frame_id"], timestamp_ns=data["timestamp_ns"],
                            **counts_by_frame)
        sessions.append({"session_id": session, "frames": frames, "motion": artifact_ref(motion_path),
                         "curves": artifact_ref(curves),
                         "counts": {key: int(value.sum()) for key, value in states.items()},
                         "per_side_joint_counts": {key: value.sum(axis=0).astype(int).tolist()
                                                   for key, value in states.items()},
                         "worst_positive_z_out_of_frame": {
                             "frame": int(np.argmax(counts_by_frame["POSITIVE_Z_OUT_OF_FRAME"])),
                             "joint_side_slots": int(counts_by_frame["POSITIVE_Z_OUT_OF_FRAME"].max())}})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_SENSOR_FRUSTUM_V1", "task_id": TASK,
              "sessions": sessions, "execution": "ALL_SAVED_NATIVE_POINTS_PROJECTED",
              "structure": "PASS_PARTITION_AND_FRAME_MAP", "quality": "VISUAL_ALIGNMENT_INCONCLUSIVE",
              "adoption": "NOT_ADOPTED",
              "next_action": "Use independent RGB anatomical evidence for alignment; do not treat projected-outside points as drawing omissions.",
              "claim_limit": "Existing projection states only; no anatomical or external wrist accuracy claim."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "SENSOR_FRUSTUM_CLASSIFIED", "result": str(out / "RESULT.json")}


def motion() -> dict:
    """Expose frozen target and Robot discontinuity without a new IK solve."""
    out = _new_dir("motion_product", "target_continuity_v1")
    sessions = []
    for session, short, frames in (("get_potato_chips_0915_007", "007", 378),
                                   ("play_cards_0915_031", "031", 149)):
        source = R2 / f"lanes/lane2_motion/recovered_{short}_wave0/ROBOT_R0_V1.npz"
        with np.load(source, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if data["frame_id"].tolist() != list(range(frames)):
            raise RuntimeError(f"MOTION_FRAME_ID_DRIFT:{short}")
        valid = np.asarray(data["wrist_valid"], bool)
        target = data["T_target_root_cam"][..., :3, 3]
        actual = data["T_actual_root_cam"][..., :3, 3]
        dt = np.diff(data["timestamp_ns"].astype(np.float64)) * 1e-9
        if np.any(dt <= 0):
            raise RuntimeError(f"MOTION_NONPOSITIVE_DT:{short}")
        pair = valid[1:] & valid[:-1]
        target_step = np.linalg.norm(np.diff(target, axis=0), axis=-1) * 1000
        actual_step = np.linalg.norm(np.diff(actual, axis=0), axis=-1) * 1000
        target_step[~pair] = np.nan
        actual_step[~pair] = np.nan
        delta = np.linalg.norm(actual - target, axis=-1) * 1000
        delta[~valid] = np.nan
        curves = out / f"{session}_TARGET_AND_ROBOT_CONTINUITY.npz"
        np.savez_compressed(curves, frame_id=data["frame_id"], timestamp_ns=data["timestamp_ns"],
                            wrist_valid=valid, consecutive_pair_valid=pair,
                            target_step_mm=target_step, robot_step_mm=actual_step,
                            target_tracking_mm=delta)
        maxima = []
        for side in range(2):
            array = target_step[:, side]
            if np.isfinite(array).any():
                index = int(np.nanargmax(array))
                maxima.append({"side": side, "source_frame_pair": [index, index + 1],
                               "target_step_mm": float(array[index]),
                               "robot_step_mm": float(actual_step[index, side]),
                               "dt_s": float(dt[index]),
                               "tracking_mm_at_next": float(delta[index + 1, side])})
        sessions.append({"session_id": session, "frames": frames, "source": artifact_ref(source),
                         "curves": artifact_ref(curves), "valid_side_frames": valid.sum(axis=0).astype(int).tolist(),
                         "valid_consecutive_pairs": pair.sum(axis=0).astype(int).tolist(),
                         "max_target_steps": maxima,
                         "tracking_over_20mm_side_frames": int(np.count_nonzero((delta > 20) & valid))})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_MOTION_CONTINUITY_V1", "task_id": TASK,
              "sessions": sessions, "execution": "FROZEN_TARGET_AND_SAVED_ROBOT_CONSUMED",
              "structure": "PASS_FULL_TIMELINES", "quality": "PRODUCT_MOTION_NOT_PROMOTED",
              "adoption": "NOT_ADOPTED",
              "next_action": "Fix only a proven target producer/consumer discontinuity with frozen source evidence; no frame deletion or new IK here.",
              "claim_limit": "Adjacent source-valid steps and saved Robot tracking only; not external human motion truth."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "MOTION_CONTINUITY_CLASSIFIED", "result": str(out / "RESULT.json")}


def _root_cam(assets: object, data: dict[str, np.ndarray], frame: int, side: int,
              q: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    values = {name: float(neutral[s, j]) for s, group in enumerate(ARM_JOINT_NAMES)
              for j, name in enumerate(group)}
    values.update({name: float(q[j]) for j, name in enumerate(ARM_JOINT_NAMES[side])})
    fk = forward_kinematics(assets.tianji, values)
    return data["T_cam_base"] @ fk[("flange_L", "flange_R")[side]] @ data["T_flange_hand"][side]


def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(target) @ actual
    return (float(np.linalg.norm(delta[:3, 3]) * 1000),
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))))


def compare() -> dict:
    """One bounded, derived HuRo postprocess on two fixed 16-frame windows."""
    out = _new_dir("compare", "huro_limit_repair_window_v1")
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    neutral = .5 * (lower + upper)
    sessions = []
    for session, first, last in (("get_potato_chips_0915_007", 181, 196),
                                 ("play_cards_0915_031", 66, 81)):
        source = V5 / f"lanes/huro/full_0001/{session}/HURO_CORE_V1.npz"
        with np.load(source, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        frames = np.arange(first, last + 1)
        valid = np.asarray(data["wrist_valid"][frames], bool)
        original = np.asarray(data["q_arm"][frames], np.float64)
        repaired = np.full_like(original, np.nan)
        statuses = np.full(valid.shape, "SOURCE_INVALID", dtype="<U48")
        before_target = np.full((*valid.shape, 2), np.nan)
        after_target = np.full_like(before_target, np.nan)
        change = np.full(valid.shape, np.nan)
        rows = []
        for i, frame in enumerate(frames):
            for side in range(2):
                if not valid[i, side]:
                    continue
                q = original[i, side]
                if not np.isfinite(q).all():
                    statuses[i, side] = "VALID_Q_NONFINITE"
                    continue
                saved_root = np.asarray(data["T_actual_root_cam"][frame, side], np.float64)
                target = np.asarray(data["T_target_root_cam"][frame, side], np.float64)
                if not np.isfinite(saved_root).all() or not np.isfinite(target).all():
                    statuses[i, side] = "VALID_POSE_NONFINITE"
                    continue
                before_target[i, side] = _pose_error(saved_root, target)
                violation = np.maximum(np.maximum(lower[side] - q, q - upper[side]), 0)
                if not np.any(violation > 1e-9):
                    candidate = q.copy()
                    statuses[i, side] = "ALREADY_WITHIN_LIMITS"
                    nfev = 0
                else:
                    def residual(x: np.ndarray) -> np.ndarray:
                        root = _root_cam(assets, data, int(frame), side, x, neutral)
                        delta = np.linalg.inv(saved_root) @ root
                        return np.concatenate((delta[:3, 3] / .005,
                                               Rotation.from_matrix(delta[:3, :3]).as_rotvec() / np.deg2rad(2),
                                               .05 * (x - q) / .2))
                    seed = np.clip(q, lower[side] + 1e-8, upper[side] - 1e-8)
                    fit = least_squares(residual, seed, bounds=(lower[side], upper[side]),
                                        method="trf", max_nfev=40, ftol=1e-7, xtol=1e-7, gtol=1e-7)
                    candidate, nfev = fit.x, int(fit.nfev)
                    statuses[i, side] = "REPAIRED_NUMERIC" if fit.success else "SOLVER_DID_NOT_CONVERGE"
                if np.any(candidate < lower[side] - 1e-9) or np.any(candidate > upper[side] + 1e-9):
                    statuses[i, side] = "HARD_LIMIT_FAILURE"
                    continue
                root = _root_cam(assets, data, int(frame), side, candidate, neutral)
                repaired[i, side] = candidate
                after_target[i, side] = _pose_error(root, target)
                change[i, side] = float(np.linalg.norm(candidate - q))
                rows.append({"source_frame_id": int(frame), "side": side, "status": str(statuses[i, side]),
                             "original_max_violation_rad": float(violation.max()), "nfev": nfev,
                             "q_delta_l2_rad": float(change[i, side]),
                             "original_target_position_mm": float(before_target[i, side, 0]),
                             "candidate_target_position_mm": float(after_target[i, side, 0]),
                             "original_target_rotation_deg": float(before_target[i, side, 1]),
                             "candidate_target_rotation_deg": float(after_target[i, side, 1])})
        array = out / f"{session}_HURO_DERIVED_LIMIT_WINDOW.npz"
        np.savez_compressed(array, frame_id=frames, timestamp_ns=data["timestamp_ns"][frames],
                            source_valid=valid, q_original=original, q_derived=repaired,
                            status=statuses, before_target_position_rotation=before_target,
                            after_target_position_rotation=after_target, q_delta_l2_rad=change,
                            lower=lower, upper=upper)
        all_q = np.asarray(data["q_arm"], np.float64)
        all_valid = np.asarray(data["wrist_valid"], bool)
        violations = np.maximum(np.maximum(lower[None] - all_q, all_q - upper[None]), 0)
        sessions.append({"session_id": session, "source": artifact_ref(source), "fixed_window": [first, last],
                         "result_array": artifact_ref(array), "rows": rows,
                         "all_source_valid_side_frames": int(all_valid.sum()),
                         "all_source_valid_with_any_hard_limit_violation": int(np.count_nonzero(
                             all_valid & np.any(violations > 1e-9, axis=-1))),
                         "all_source_valid_violation_by_joint": [int(x) for x in np.count_nonzero(
                             all_valid[..., None] & (violations > 1e-9), axis=(0, 1))],
                         "window_source_valid": int(valid.sum()),
                         "window_derived_finite": int(np.count_nonzero(np.isfinite(repaired).all(axis=-1))),
                         "window_repaired": int(np.count_nonzero(statuses == "REPAIRED_NUMERIC"))})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_HURO_LIMIT_WINDOW_V1", "task_id": TASK,
              "sessions": sessions, "method": "DERIVED_BOUNDED_POSTPROCESS_PRESERVE_SAVED_WRIST_POSE",
              "execution": "ONE_FROZEN_CPU_POSTPROCESS", "structure": "NUMERIC_ONLY_WINDOW",
              "quality": "NOT_OFFICIAL_HURO_AND_NOT_PRODUCT_ADOPTED", "adoption": "NOT_ADOPTED",
              "next_action": "Evaluate hard-limit and target-cost tradeoff on fixed windows; full-session extension needs frozen quality and collision gates.",
              "claim_limit": "Original HuRo q immutable. Derived bounded q is not a HuRo core rerun, official method improvement or product acceptance."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "HURO_DERIVED_LIMIT_WINDOW_COMPLETE", "result": str(out / "RESULT.json")}


def compare_review() -> dict:
    """Reload derived q and independently recompute fixed-URDF FK before any claim."""
    source_result = ATTEMPT / "lanes/compare/huro_limit_repair_window_v1/RESULT.json"
    source = load_json(source_result)
    out = _new_dir("compare", "huro_limit_independent_review_v1")
    assets = load_pinned_robot_assets(REPO_ROOT)
    sessions = []
    for row in source["sessions"]:
        original_path = Path(row["source"]["path"])
        derived_path = Path(row["result_array"]["path"])
        with np.load(original_path, allow_pickle=False) as archive:
            original = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(derived_path, allow_pickle=False) as archive:
            derived = {key: np.asarray(archive[key]) for key in archive.files}
        frames = derived["frame_id"]
        finite = derived["source_valid"] & np.isfinite(derived["q_derived"]).all(axis=-1)
        evaluator_input = {"frame_id": frames, "wrist_valid": finite,
                           "q_arm": derived["q_derived"],
                           "T_cam_base": original["T_cam_base"],
                           "T_flange_hand": original["T_flange_hand"],
                           "T_actual_root_cam": original["T_actual_root_cam"][frames]}
        actual, _, status = _independent_root_fk(assets, evaluator_input)
        if np.any(status[finite] != "PASS"):
            raise RuntimeError(f"DERIVED_Q_FAILS_INDEPENDENT_FK:{row['session_id']}")
        pos = np.full(finite.shape, np.nan)
        rot = np.full(finite.shape, np.nan)
        for i, side in np.argwhere(finite):
            pos[i, side], rot[i, side] = _pose_error(actual[i, side],
                                                   original["T_target_root_cam"][frames[i], side])
        saved = derived["after_target_position_rotation"]
        if (np.nanmax(np.abs(pos - saved[..., 0])) > .01
                or np.nanmax(np.abs(rot - saved[..., 1])) > .01):
            raise RuntimeError(f"DERIVED_FK_METRIC_BINDING:{row['session_id']}")
        array = out / f"{row['session_id']}_DERIVED_INDEPENDENT_FK.npz"
        np.savez_compressed(array, frame_id=frames, source_valid=derived["source_valid"],
                            derived_valid=finite, q_derived=derived["q_derived"],
                            T_actual_root_cam=actual, fk_status=status,
                            target_position_mm=pos, target_rotation_deg=rot)
        image = np.full((660, 1280, 3), 25, np.uint8)
        cv2.putText(image, f"{row['session_id']} | derived HuRo q, fixed 16-frame window", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2)
        cv2.putText(image, "Blue: original target residual; orange: bounded-derived residual", (18, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (240, 240, 240), 1)
        before = derived["before_target_position_rotation"]
        for metric, y_top, maximum, title in ((0, 100, 1500.0, "position mm (internal target)"),
                                               (1, 360, 180.0, "full rotation deg")):
            cv2.putText(image, title, (18, y_top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        .56, (240, 240, 240), 1)
            cv2.line(image, (65, y_top + 210), (1220, y_top + 210), (120, 120, 120), 1)
            for side in range(2):
                offset = side * 3
                for values, color in ((before, (255, 130 + offset, 50)),
                                      (saved, (30, 150 + offset, 255))):
                    prior = None
                    for i, frame in enumerate(frames):
                        if not finite[i, side] or not np.isfinite(values[i, side, metric]):
                            prior = None
                            continue
                        xy = (65 + int(i * 1150 / 15),
                              y_top + 210 - int(min(max(values[i, side, metric], 0), maximum) * 180 / maximum))
                        if prior is not None:
                            cv2.line(image, prior, xy, color, 2)
                        cv2.circle(image, xy, 3, color, -1)
                        prior = xy
            cv2.putText(image, "dashed thresholds not shown: this is diagnostic, not adopted quality", (65, y_top + 238),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (180, 180, 180), 1)
        png = out / f"{row['session_id']}_DERIVED_LIMIT_TRADEOFF.png"
        if not cv2.imwrite(str(png), image):
            raise RuntimeError("DERIVED_REVIEW_IMAGE_WRITE")
        sessions.append({"session_id": row["session_id"], "frames": len(frames),
                         "source": artifact_ref(derived_path), "independent_fk": artifact_ref(array),
                         "review_image": artifact_ref(png), "source_valid_side_frames": int(derived["source_valid"].sum()),
                         "derived_fk_valid_side_frames": int(finite.sum()),
                         "derived_hard_limit_reject_side_frames": int(np.count_nonzero(status[finite] != "PASS")),
                         "max_position_cost_increase_mm": float(np.nanmax(saved[..., 0] - before[..., 0])),
                         "max_rotation_cost_increase_deg": float(np.nanmax(saved[..., 1] - before[..., 1]))})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_DERIVED_HURO_INDEPENDENT_REVIEW_V1",
              "task_id": TASK, "source": artifact_ref(source_result), "sessions": sessions,
              "execution": "INDEPENDENT_SAVED_Q_TO_FIXED_URDF_REPLAY",
              "structure": "PASS_FK_BINDING_FOR_DECLARED_WINDOW",
              "quality": "NOT_FULL_POSE_OR_COLLISION_ADOPTED", "adoption": "NOT_ADOPTED",
              "claim_limit": "A derived bounded postprocess makes reported window arm q FK legal, not the original HuRo solve or product quality."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "HURO_DERIVED_INDEPENDENT_REVIEW_COMPLETE", "result": str(out / "RESULT.json")}


def _timeline_canvas(title: str, frames: np.ndarray) -> np.ndarray:
    image = np.full((540, 1280, 3), 26, np.uint8)
    cv2.putText(image, title, (26, 42), cv2.FONT_HERSHEY_SIMPLEX, .72, (235, 235, 235), 2)
    cv2.line(image, (75, 450), (1220, 450), (150, 150, 150), 1)
    for fraction in (0, .25, .5, .75, 1):
        x = 75 + int(1145 * fraction)
        frame = int(frames[0] + (frames[-1] - frames[0]) * fraction)
        cv2.line(image, (x, 450), (x, 460), (150, 150, 150), 1)
        cv2.putText(image, str(frame), (x - 15, 480), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
    cv2.putText(image, "source frame id; gaps are not interpolated", (75, 515),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 180, 180), 1)
    return image


def _draw_curve(image: np.ndarray, values: np.ndarray, maximum: float, color: tuple[int, int, int],
                *, y_top: int = 100, y_bottom: int = 440) -> None:
    previous = None
    for i, value in enumerate(values):
        if not np.isfinite(value):
            previous = None
            continue
        x = 75 + int(i * 1145 / max(len(values) - 1, 1))
        y = y_bottom - int(min(max(float(value), 0), maximum) * (y_bottom - y_top) / maximum)
        if previous is not None:
            cv2.line(image, previous, (x, y), color, 2)
        cv2.circle(image, (x, y), 2, color, -1)
        previous = (x, y)


def sensor_review() -> dict:
    """Render the existing full-timeline frustum partition; no sensor recompute."""
    source = ATTEMPT / "lanes/sensor/projection_frustum_v1_runtime_fix1/RESULT.json"
    prior = load_json(source)
    out = _new_dir("sensor", "projection_frustum_review_v1")
    sessions = []
    for row in prior["sessions"]:
        with np.load(row["curves"]["path"], allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        frames = data["frame_id"]
        image = _timeline_canvas(f"{row['session_id']} | native joints: camera frustum", frames)
        cv2.putText(image, "orange: positive-Z outside image | cyan: in image (of 42 joint-side slots)",
                    (75, 75), cv2.FONT_HERSHEY_SIMPLEX, .54, (230, 230, 230), 1)
        _draw_curve(image, data["POSITIVE_Z_OUT_OF_FRAME"], 42, (20, 155, 255))
        _draw_curve(image, data["PROJECTABLE_IN_FRAME"], 42, (220, 220, 30))
        png = out / f"{row['session_id']}_FRUSTUM_TIMELINE.png"
        if not cv2.imwrite(str(png), image):
            raise RuntimeError("SENSOR_REVIEW_IMAGE_WRITE")
        sessions.append({"session_id": row["session_id"], "source": row["curves"],
                         "review_image": artifact_ref(png), "frames": len(frames)})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_SENSOR_REVIEW_V1", "task_id": TASK,
              "source": artifact_ref(source), "sessions": sessions,
              "execution": "FULL_TIMELINE_REVIEW_FROM_SAVED_COUNTS", "structure": "PASS",
              "quality": "ANATOMICAL_ALIGNMENT_NOT_ESTABLISHED", "adoption": "NOT_ADOPTED",
              "claim_limit": "Image-frustum occupancy is not hand-to-RGB anatomical accuracy."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "SENSOR_REVIEW_COMPLETE", "result": str(out / "RESULT.json")}


def scene_review() -> dict:
    """Consume every saved 007 canary frame; keep semantic residual judgment separate."""
    source = PRIOR / "lanes/scene/context_007_v1"
    out = _new_dir("scene", "full_window_consumption_review_v1")
    raw_root = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                            "lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw")
    frames = np.arange(181, 197)
    counts = {key: [] for key in ("model_and_write", "protected_write", "changed_in_write",
                                  "changed_outside_write", "changed_in_model_and_write")}
    for frame in frames:
        local = int(frame - 160)
        raw = _image(raw_root / f"{frame:06d}.png")
        clean = _image(source / f"clean/{frame-181:06d}.png")
        model = _image(source / f"input/model_masks/{local:06d}.png", cv2.IMREAD_GRAYSCALE)
        write = _image(source / f"input/write/{local:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
        protect = _image(source / f"input/protect/{local:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
        if raw.shape != clean.shape or model.shape != (720, 960) or write.shape != (960, 1280):
            raise RuntimeError(f"SCENE_WINDOW_DOMAIN_DRIFT:{frame}")
        model_raw = cv2.resize(model, (1280, 960), interpolation=cv2.INTER_NEAREST) > 0
        changed = np.any(raw != clean, axis=-1)
        counts["model_and_write"].append(int(np.count_nonzero(model_raw & write)))
        counts["protected_write"].append(int(np.count_nonzero(protect & write)))
        counts["changed_in_write"].append(int(np.count_nonzero(changed & write)))
        counts["changed_outside_write"].append(int(np.count_nonzero(changed & ~write)))
        counts["changed_in_model_and_write"].append(int(np.count_nonzero(changed & model_raw & write)))
    array = out / "007_181_196_MODEL_WRITE_CLEAN_CONSUMPTION.npz"
    np.savez_compressed(array, frame_id=frames, **{key: np.asarray(value) for key, value in counts.items()})
    image = _timeline_canvas("007 frames 181-196 | actual mask and Clean consumption", frames)
    cv2.putText(image, "orange: model AND write | cyan: changed inside write | red: changed outside write",
                (75, 75), cv2.FONT_HERSHEY_SIMPLEX, .52, (230, 230, 230), 1)
    maximum = max(float(max(values)) for values in counts.values())
    for key, color in (("model_and_write", (20, 155, 255)),
                       ("changed_in_write", (220, 220, 30)),
                       ("changed_outside_write", (40, 40, 255))):
        _draw_curve(image, np.asarray(counts[key]), max(maximum, 1), color)
    png = out / "007_181_196_CONSUMPTION_TIMELINE.png"
    if not cv2.imwrite(str(png), image):
        raise RuntimeError("SCENE_REVIEW_IMAGE_WRITE")
    old_video = source / "007_CONTEXT_OLD_NEW_CLEAN_16FRAME_REVIEW.mp4"
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_SCENE_WINDOW_REVIEW_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "frames": frames.tolist(),
              "counts": {key: {"total": int(sum(value)), "by_frame": value} for key, value in counts.items()},
              "source_video": artifact_ref(old_video), "array": artifact_ref(array),
              "review_image": artifact_ref(png), "execution": "ALL_16_SAVED_RAW_MASK_AND_CLEAN_FRAMES_CONSUMED",
              "structure": ("PASS_FIXED_WINDOW" if not any(counts["changed_outside_write"])
                            else "FAIL_OUTSIDE_WRITE_CHANGED"),
              "quality": "CLEAN_REJECTED_BY_PRIOR_VISUAL_REVIEW",
              "adoption": "NOT_ADOPTED",
              "claim_limit": "Pixel consumption is structural evidence, not proof that residual limbs, cable or card edges were removed."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "SCENE_WINDOW_REVIEW_COMPLETE", "result": str(out / "RESULT.json")}


def motion_review() -> dict:
    """Render preserved motion gaps and target/Robot residuals from saved arrays."""
    source = ATTEMPT / "lanes/motion_product/target_continuity_v1/RESULT.json"
    prior = load_json(source)
    out = _new_dir("motion_product", "target_continuity_review_v1")
    sessions = []
    for row in prior["sessions"]:
        with np.load(row["curves"]["path"], allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        frames = data["frame_id"]
        image = _timeline_canvas(f"{row['session_id']} | frozen target and saved Robot", frames)
        cv2.putText(image, "orange: target step (max side) | cyan: tracking error (max valid side)",
                    (75, 75), cv2.FONT_HERSHEY_SIMPLEX, .54, (230, 230, 230), 1)
        target = np.where(np.isfinite(data["target_step_mm"]), data["target_step_mm"], np.nan)
        tracking = np.where(np.isfinite(data["target_tracking_mm"]), data["target_tracking_mm"], np.nan)
        target_curve = np.full(len(frames), np.nan)
        target_curve[1:] = np.max(np.where(np.isfinite(target), target, -np.inf), axis=1)
        target_curve[~np.isfinite(target_curve)] = np.nan
        target_curve[target_curve == -np.inf] = np.nan
        tracking_curve = np.max(np.where(np.isfinite(tracking), tracking, -np.inf), axis=1)
        tracking_curve[tracking_curve == -np.inf] = np.nan
        maximum = max(20.0, float(np.nanmax(np.r_[target_curve, tracking_curve])))
        _draw_curve(image, target_curve, maximum, (20, 155, 255))
        _draw_curve(image, tracking_curve, maximum, (220, 220, 30))
        if row["session_id"] == "play_cards_0915_031":
            x = 75 + int(48 * 1145 / max(len(frames) - 1, 1))
            cv2.line(image, (x, 95), (x, 450), (100, 100, 255), 1)
            cv2.putText(image, "47->48 frozen target jump", (x + 8, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (120, 120, 255), 1)
        png = out / f"{row['session_id']}_TARGET_ROBOT_TIMELINE.png"
        if not cv2.imwrite(str(png), image):
            raise RuntimeError("MOTION_REVIEW_IMAGE_WRITE")
        sessions.append({"session_id": row["session_id"], "source": row["curves"],
                         "review_image": artifact_ref(png), "frames": len(frames),
                         "valid_side_frames": row["valid_side_frames"]})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_MOTION_REVIEW_V1", "task_id": TASK,
              "source": artifact_ref(source), "sessions": sessions,
              "execution": "FULL_TIMELINE_REVIEW_FROM_SAVED_TARGET_AND_ROBOT", "structure": "PASS",
              "quality": "PRODUCT_MOTION_NOT_PROMOTED", "adoption": "NOT_ADOPTED",
              "claim_limit": "Gaps stay absent and 031 frame 47->48 jump remains visible; no new IK solve."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "MOTION_REVIEW_COMPLETE", "result": str(out / "RESULT.json")}


def motion_source() -> dict:
    """Determine whether the 031 frozen-target jump precedes target building."""
    out = _new_dir("motion_product", "target_source_binding_v1")
    source = R2 / "lanes/lane2_motion/recovered_031_wave0"
    hand_path, robot_path = source / "HAND_MOTION_V1.npz", source / "ROBOT_R0_V1.npz"
    with np.load(hand_path, allow_pickle=False) as archive:
        hand = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {key: np.asarray(archive[key]) for key in archive.files}
    if (not np.array_equal(hand["frame_id"], robot["frame_id"])
            or not np.array_equal(hand["timestamp_ns"], robot["timestamp_ns"])):
        raise RuntimeError("MOTION_SOURCE_FRAME_OR_TIME_DRIFT")
    valid = np.asarray(robot["target_valid"], bool)
    if not np.array_equal(hand["position_valid"], valid):
        raise RuntimeError("MOTION_SOURCE_VALID_DRIFT")
    hand_t = hand["T_camera_wrist"]
    target_t = robot["T_target_root_cam"]
    position_delta = np.full(valid.shape, np.nan)
    rotation_delta = np.full(valid.shape, np.nan)
    position_delta[valid] = np.max(np.abs(hand_t[valid, :3, 3] - target_t[valid, :3, 3]), axis=-1)
    rotation_delta[valid] = np.max(np.abs(hand_t[valid, :3, :3] - target_t[valid, :3, :3]), axis=(-1, -2))
    if np.nanmax(position_delta) > 1e-10:
        raise RuntimeError("MOTION_SOURCE_TARGET_POSITION_NOT_IDENTICAL")
    if valid[:47].any() or not valid[47:149, 1].all() or valid[:, 0].any():
        raise RuntimeError("MOTION_SOURCE_COHORT_DRIFT")
    hand_step_mm = float(np.linalg.norm(hand_t[48, 1, :3, 3] - hand_t[47, 1, :3, 3]) * 1000)
    target_step_mm = float(np.linalg.norm(target_t[48, 1, :3, 3] - target_t[47, 1, :3, 3]) * 1000)
    array = out / "031_HAND_TO_TARGET_POSITION_BINDING.npz"
    np.savez_compressed(array, frame_id=hand["frame_id"], timestamp_ns=hand["timestamp_ns"],
                        source_valid=valid, max_position_element_delta_m=position_delta,
                        max_rotation_element_delta=rotation_delta)
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_TARGET_SOURCE_BINDING_V1", "task_id": TASK,
              "session_id": "play_cards_0915_031", "hand_motion": artifact_ref(hand_path),
              "robot_r0": artifact_ref(robot_path), "binding_array": artifact_ref(array),
              "source_valid_side_frames": valid.sum(axis=0).astype(int).tolist(),
              "max_hand_to_target_position_element_delta_m": float(np.nanmax(position_delta)),
              "max_hand_to_target_rotation_element_delta": float(np.nanmax(rotation_delta)),
              "frame47_to48_hand_wrist_step_mm": hand_step_mm,
              "frame47_to48_robot_target_step_mm": target_step_mm,
              "classification": "POSITION_JUMP_PRESENT_IN_SAVED_HAND_MOTION_BEFORE_TARGET_BUILDER",
              "execution": "FULL_149_FRAME_POSITION_SOURCE_TO_CONSUMER_BINDING_CHECK", "structure": "PASS_POSITION_BINDING",
              "quality": "UPSTREAM_HAND_WRIST_ESTIMATE_NOT_VALIDATED", "adoption": "NOT_ADOPTED",
              "claim_limit": "Target builder did not create this POSITION jump; saved HandMotion already contains it. Target rotations differ by contract and are not certified here. No raw-model cause or frame replacement authority."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "MOTION_SOURCE_BINDING_COMPLETE", "result": str(out / "RESULT.json")}


def motion_upstream() -> dict:
    """Bind the onset jump to actual HaWoR source boxes and raw RGB frames."""
    out = _new_dir("motion_product", "frame47_upstream_evidence_v1")
    hawor_path = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                              "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz")
    hand_path = R2 / "lanes/lane2_motion/recovered_031_wave0/HAND_MOTION_V1.npz"
    raw_root = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                            "lanes/exact78/prepare_full_v1/play_cards_0915_031/raw")
    with np.load(hawor_path, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(hand_path, allow_pickle=False) as archive:
        hand = {key: np.asarray(archive[key]) for key in archive.files}
    valid = np.asarray(hawor["predicted_valid"][1], bool)
    if not np.array_equal(valid, hand["position_valid"][:, 1]):
        raise RuntimeError("HAWOR_HAND_VALID_DRIFT")
    source_wrist = hawor["joints_3d_camera"][1, :, 0].astype(np.float64)
    saved_wrist = hand["T_camera_wrist"][:, 1, :3, 3]
    if np.max(np.abs(source_wrist[valid] - saved_wrist[valid])) > 1e-6:
        raise RuntimeError("HAWOR_HAND_WRIST_DRIFT")
    boxes = np.asarray(hawor["detector_boxes_xyxy"][1], np.float64)
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    if not valid[47] or valid[:47].any():
        raise RuntimeError("HAWOR_FIRST_VALID_DRIFT")
    montage = []
    source_refs = []
    for frame in (47, 48, 49):
        path = raw_root / f"{frame:06d}.png"
        image = _image(path)
        source_refs.append(artifact_ref(path))
        x1, y1, x2, y2 = np.rint(boxes[frame]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (15, 165, 255), 3)
        cv2.putText(image, f"frame {frame}: source box {int(width[frame])}x{int(height[frame])} px",
                    (25, 55), cv2.FONT_HERSHEY_SIMPLEX, .85, (250, 250, 250), 2)
        cv2.putText(image, f"HaWoR wrist optical Z {source_wrist[frame, 2]:.3f} m",
                    (25, 91), cv2.FONT_HERSHEY_SIMPLEX, .78, (250, 250, 250), 2)
        montage.append(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA))
    png = out / "031_FRAMES47_49_RAW_HAWOR_ONSET.png"
    if not cv2.imwrite(str(png), np.concatenate(montage, axis=1)):
        raise RuntimeError("MOTION_UPSTREAM_MONTAGE_WRITE")
    array = out / "031_HAWOR_BOX_AND_WRIST_TIMELINE.npz"
    np.savez_compressed(array, frame_id=hand["frame_id"], timestamp_ns=hand["timestamp_ns"],
                        source_valid=valid, detector_boxes_xyxy=boxes, width_px=width,
                        height_px=height, bottom_edge_at_960=boxes[:, 3] >= 960,
                        source_wrist_camera_m=source_wrist, saved_hand_wrist_camera_m=saved_wrist)
    step_mm = float(np.linalg.norm(source_wrist[48] - source_wrist[47]) * 1000)
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_031_HAWOR_ONSET_EVIDENCE_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031",
              "hawor_source": artifact_ref(hawor_path), "saved_hand_motion": artifact_ref(hand_path),
              "raw_frames": source_refs, "montage": artifact_ref(png), "timeline": artifact_ref(array),
              "first_valid_frame": 47, "first_valid_box_width_height_px": [int(width[47]), int(height[47])],
              "next_box_width_height_px": [int(width[48]), int(height[48])],
              "source_valid_frames": int(valid.sum()),
              "source_valid_bottom_edge_boxes": int(np.count_nonzero(valid & (boxes[:, 3] >= 960))),
              "max_hawor_to_saved_hand_wrist_element_delta_m": float(np.max(np.abs(source_wrist[valid] - saved_wrist[valid]))),
              "frame47_to48_source_wrist_step_mm": step_mm,
              "classification": "FIRST_VALID_TINY_BOTTOM_TRUNCATED_BOX_WITH_SOURCE_WRIST_JUMP",
              "execution": "SOURCE_ARRAY_AND_RAW_RGB_CONSUMED", "structure": "PASS_ONSET_BINDING",
              "quality": "SOURCE_WRIST_NOT_VALIDATED", "adoption": "NOT_ADOPTED",
              "claim_limit": "The onset box and source jump coexist, but this does not prove the detector or model is uniquely causal, nor authorize replacing or dropping frame 47."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "MOTION_UPSTREAM_EVIDENCE_COMPLETE", "result": str(out / "RESULT.json")}


def compare_tradeoff_review() -> dict:
    """Readable fixed-window before/after dashboard; no additional solve."""
    source = ATTEMPT / "lanes/compare/huro_limit_repair_window_v1/RESULT.json"
    prior = load_json(source)
    out = _new_dir("compare", "huro_limit_tradeoff_dashboard_v1")
    sessions = []
    for row in prior["sessions"]:
        with np.load(row["result_array"]["path"], allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        valid = data["source_valid"]
        before = data["before_target_position_rotation"]
        after = data["after_target_position_rotation"]
        lower, upper = data["lower"], data["upper"]
        q_before, q_after = data["q_original"], data["q_derived"]
        before_v = np.maximum(np.maximum(lower[None] - q_before, q_before - upper[None]), 0)
        after_v = np.maximum(np.maximum(lower[None] - q_after, q_after - upper[None]), 0)
        original_bad = valid & np.any(before_v > 1e-9, axis=-1)
        derived_bad = valid & np.any(after_v > 1e-9, axis=-1)
        if np.any(derived_bad):
            raise RuntimeError("DASHBOARD_DERIVED_HARD_LIMIT_DRIFT")
        image = np.full((780, 1300, 3), 27, np.uint8)
        cv2.putText(image, row["session_id"] + " | fixed 16-frame derived-q tradeoff", (22, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, (235, 235, 235), 2)
        headline = (f"original hard-limit rejects {int(original_bad.sum())}/{int(valid.sum())} valid side-frames"
                    f"  ->  derived {int(derived_bad.sum())}/{int(valid.sum())}")
        cv2.putText(image, headline, (22, 75), cv2.FONT_HERSHEY_SIMPLEX, .6, (70, 190, 255), 1)
        delta_pos = float(np.nanmax(after[..., 0] - before[..., 0]))
        delta_rot = float(np.nanmax(after[..., 1] - before[..., 1]))
        cv2.putText(image, f"max target cost increase: {delta_pos:.3f} mm, {delta_rot:.3f} deg; collision UNKNOWN",
                    (22, 108), cv2.FONT_HERSHEY_SIMPLEX, .58, (220, 220, 220), 1)
        for metric, top, height, scale, label in (
            (0, 160, 210, 110.0, "Target position residual, mm | blue original, orange derived"),
            (1, 410, 210, 65.0, "Full rotation residual, deg | blue original, orange derived"),
        ):
            cv2.putText(image, label, (75, top - 10), cv2.FONT_HERSHEY_SIMPLEX, .53, (225, 225, 225), 1)
            cv2.line(image, (75, top + height), (1230, top + height), (130, 130, 130), 1)
            for side in range(2):
                for values, color in ((before, (240, 130, 45)), (after, (20, 165, 255))):
                    previous = None
                    for i in range(len(data["frame_id"])):
                        if not valid[i, side]:
                            previous = None
                            continue
                        value = float(values[i, side, metric])
                        x = 75 + int(i * 1155 / 15)
                        y = top + height - int(min(max(value, 0), scale) * height / scale)
                        if previous is not None:
                            cv2.line(image, previous, (x, y), color, 2)
                        cv2.circle(image, (x, y), 3, color, -1)
                        previous = (x, y)
        cv2.putText(image, "orange bars: original limit violation by source frame; derived has zero", (75, 667),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (215, 215, 215), 1)
        by_frame = original_bad.sum(axis=1)
        for i, value in enumerate(by_frame):
            x = 75 + int(i * 1155 / 15)
            cv2.rectangle(image, (x - 8, 724 - int(value * 24)), (x + 8, 724), (20, 165, 255), -1)
            cv2.putText(image, str(int(data["frame_id"][i])), (x - 16, 750),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (185, 185, 185), 1)
        png = out / f"{row['session_id']}_HURO_DERIVED_TRADEOFF_DASHBOARD.png"
        if not cv2.imwrite(str(png), image):
            raise RuntimeError("DASHBOARD_IMAGE_WRITE")
        sessions.append({"session_id": row["session_id"], "review_image": artifact_ref(png),
                         "input": row["result_array"], "source_valid_side_frames": int(valid.sum()),
                         "original_limit_reject_side_frames": int(original_bad.sum()),
                         "derived_limit_reject_side_frames": int(derived_bad.sum()),
                         "max_target_position_cost_increase_mm": delta_pos,
                         "max_target_rotation_cost_increase_deg": delta_rot})
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_HURO_TRADEOFF_DASHBOARD_V1",
              "task_id": TASK, "source": artifact_ref(source), "sessions": sessions,
              "execution": "SAVED_Q_WINDOW_VISUAL_REVIEW", "structure": "PASS_FIXED_WINDOW",
              "quality": "COLLISION_AND_FULL_POSE_NOT_ADOPTED", "adoption": "NOT_ADOPTED",
              "claim_limit": "Readable derived-q kinematic tradeoff only; original HuRo and product quality remain rejected."}
    atomic_json(out / "RESULT.json", result)
    return {"status": "COMPARE_DASHBOARD_COMPLETE", "result": str(out / "RESULT.json")}


def progress() -> dict:
    """Single-publisher projection from completed lane receipts, not guessed success."""
    evidence = {
        "scene": ("full_window_consumption_review_v1", "CLEAN_REJECTED", "NONE_CLAIMED",
                  "Review prior 16-frame Raw/old/new video; no same-signature model retry."),
        "sensor": ("projection_frustum_review_v1", "ALIGNMENT_INCONCLUSIVE", "NONE_CLAIMED",
                   "Review native joint overlay against independent RGB anatomy before any alignment claim."),
        "motion_product": ("target_continuity_review_v1", "PRODUCT_MOTION_REJECTED", "NONE_CLAIMED",
                           "Trace frozen 031 frame 47->48 target source in a separately authorized producer fix."),
        "compare": ("huro_limit_independent_review_v1", "WINDOW_KINEMATIC_ONLY", "DERIVED_Q_BOUNDED_WINDOW",
                    "Check collision and full-session validity before any method or product adoption."),
    }
    now = now_iso()
    lane_refs = {}
    for lane, (stage, quality, improvement, action) in evidence.items():
        result_path = ATTEMPT / "lanes" / lane / stage / "RESULT.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = load_json(result_path)
        if result.get("task_id") != TASK:
            raise RuntimeError(f"LANE_RESULT_IDENTITY:{lane}")
        ref = artifact_ref(result_path)
        lane_refs[lane] = ref
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        state = load_json(path)
        state.update({"status": "COMPLETE_LIMITED_REVIEW", "execution": "EXECUTED",
                      "structure": result.get("structure", "PASS_FOR_DECLARED_SCOPE"),
                      "quality": quality, "improvement": improvement, "adoption": "NOT_ADOPTED",
                      "result": ref, "consumer": "FOUR_LANE_CURRENT_EVIDENCE_AND_DELIVERY",
                      "dependencies": [ref], "next_action": action, "updated_at": now})
        atomic_json(path, state)
    progress_path = ATTEMPT / "PROGRESS.json"
    if progress_path.exists():
        if load_json(progress_path).get("lane_results") != lane_refs:
            raise RuntimeError("PROGRESS_RETRY_EVIDENCE_DRIFT")
    else:
        atomic_json(progress_path, {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_PROGRESS_V1",
                                    "task_id": TASK, "created_at": now, "lane_results": lane_refs,
                                    "products_structure": "4/4", "products_quality": "0/4",
                                    "products_adopted": "0/4", "execution": "FOUR_CPU_LANES_EVIDENCE_COMPLETE",
                                    "quality": "LIMITED_NEW_NUMERIC_NO_PRODUCT_PROMOTION",
                                    "next_action": "Publish current evidence navigation, verify artifacts and retained 15 slots."})
    task_state = load_json(TASK_STATE_PATH)
    active = [row for row in task_state["tasks"] if row.get("task_id") == TASK]
    if len(active) != 1 or task_state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("PUBLISHER_TASK_NOT_ACTIVE")
    active[0].update({"status": "RUNNING", "attempt": 1, "updated_at": now,
                      "heartbeat_at": now, "progress": artifact_ref(progress_path)})
    revision = int(load_json(RECEIPT_PATH)["governance_revision"])
    receipt = publish_bundle(load_json(AUTHORITY_PATH), task_state,
                             event_type="HUMAN_TO_ROBOT_FOUR_LANE_EVIDENCE_PROGRESS",
                             expected_revision=revision, generator_path=Path(__file__))
    return {"status": "PROGRESS_PUBLISHED", "result": str(progress_path),
            "governance_revision": receipt["governance_revision"]}


def validate_delivery() -> dict:
    """Reload every new artifact and retain existing 15-slot validation as history."""
    path = ATTEMPT / "FINAL_VALIDATION.json"
    if path.exists():
        raise FileExistsError(path)
    stages = (
        "lanes/scene/complaint_consumption_v1_runtime_fix1/RESULT.json",
        "lanes/scene/full_window_consumption_review_v1/RESULT.json",
        "lanes/sensor/projection_frustum_v1_runtime_fix1/RESULT.json",
        "lanes/sensor/projection_frustum_review_v1/RESULT.json",
        "lanes/motion_product/target_continuity_v1/RESULT.json",
        "lanes/motion_product/target_continuity_review_v1/RESULT.json",
        "lanes/motion_product/target_source_binding_v1/RESULT.json",
        "lanes/motion_product/frame47_upstream_evidence_v1/RESULT.json",
        "lanes/compare/huro_limit_repair_window_v1/RESULT.json",
        "lanes/compare/huro_limit_independent_review_v1/RESULT.json",
        "lanes/compare/huro_limit_tradeoff_dashboard_v1/RESULT.json",
    )
    checked: dict[str, dict] = {}
    def visit(value: object) -> None:
        if isinstance(value, dict):
            if {"path", "sha256", "bytes"} <= set(value):
                checked[str(value["path"])] = value
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for name in stages:
        result_path = ATTEMPT / name
        result = load_json(result_path)
        if result.get("task_id") != TASK:
            raise RuntimeError(f"DELIVERY_STAGE_IDENTITY:{name}")
        checked[str(result_path)] = artifact_ref(result_path)
        visit(result)
    for ref in checked.values():
        errors = validate_artifact_ref(ref)
        if errors:
            raise RuntimeError("DELIVERY_ARTIFACT_DRIFT:" + ";".join(errors))
    npz_count = png_count = mp4_count = 0
    decoded = {}
    for name in checked:
        item = Path(name)
        if item.suffix == ".npz":
            with np.load(item, allow_pickle=False) as archive:
                for key in archive.files:
                    _ = archive[key].shape
            npz_count += 1
        elif item.suffix == ".png":
            if cv2.imread(name, cv2.IMREAD_UNCHANGED) is None:
                raise RuntimeError(f"DELIVERY_IMAGE_DECODE:{name}")
            png_count += 1
        elif item.suffix == ".mp4":
            capture = cv2.VideoCapture(name)
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or not frame.size:
                    raise RuntimeError(f"DELIVERY_VIDEO_FRAME:{name}:{frames}")
                frames += 1
            capture.release()
            if frames != 16:
                raise RuntimeError(f"DELIVERY_VIDEO_FRAME_COUNT:{name}:{frames}")
            decoded[name] = frames
            mp4_count += 1
    candidate = load_json(ATTEMPT / "lanes/compare/huro_limit_repair_window_v1/RESULT.json")
    gates = []
    for row in candidate["sessions"]:
        with np.load(row["result_array"]["path"], allow_pickle=False) as archive:
            valid = archive["source_valid"]
            costs = archive["after_target_position_rotation"]
        position = valid & (costs[..., 0] <= 20)
        rotation = valid & (costs[..., 1] <= 15)
        gates.append({"session_id": row["session_id"], "valid_side_frames": int(valid.sum()),
                      "position_le_20mm": int(position.sum()), "full_rotation_le_15deg": int(rotation.sum()),
                      "both": int(np.count_nonzero(position & rotation))})
    tests = ATTEMPT / "TARGETED_TESTS.xml"
    old_slots = PRIOR / "FINAL_VALIDATION.json"
    validation = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_FINAL_VALIDATION_V1",
                  "task_id": TASK, "status": "PASS_ARTIFACT_RELOAD_ONLY",
                  "stage_results": [artifact_ref(ATTEMPT / name) for name in stages],
                  "validated_unique_refs": len(checked), "npz_reloaded": npz_count,
                  "png_decoded": png_count, "mp4_fully_decoded": mp4_count,
                  "decoded_video_frames": decoded, "targeted_tests": artifact_ref(tests),
                  "prior_15_slot_validation": artifact_ref(old_slots), "derived_huro_gates": gates,
                  "quality": "FOUR_PRODUCT_QUALITY_0_OF_4_AND_DERIVED_FULL_POSE_NOT_PASSED",
                  "claim_limit": "Artifact integrity and declared window metrics only; reusing the prior 15-slot validation does not re-approve rejected video content."}
    atomic_json(path, validation)
    return {"status": "DELIVERY_ARTIFACTS_VALIDATED", "result": str(path),
            "checked_refs": len(checked)}


def finish() -> dict:
    """Close the finite CPU packet after all real lane outputs and validation."""
    result_path = ATTEMPT / "RESULT.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    validation = load_json(validation_path)
    if validation.get("status") != "PASS_ARTIFACT_RELOAD_ONLY" or len(validation["stage_results"]) != 11:
        raise RuntimeError("FOUR_LANE_VALIDATION_INCOMPLETE")
    if any(row["both"] for row in validation["derived_huro_gates"]):
        raise RuntimeError("DERIVED_FULL_POSE_RESULT_CHANGED")
    progress_path = ATTEMPT / "PROGRESS.json"
    progress = load_json(progress_path)
    if set(progress["lane_results"]) != {"scene", "sensor", "motion_product", "compare"}:
        raise RuntimeError("FOUR_LANE_PROGRESS_INCOMPLETE")
    lane_states = {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json")
                   for lane in progress["lane_results"]}
    for lane in lane_states:
        state = load_json(ATTEMPT / "lanes" / lane / "STATE.json")
        if state.get("status") != "COMPLETE_LIMITED_REVIEW" or state.get("adoption") != "NOT_ADOPTED":
            raise RuntimeError(f"LANE_NOT_FINITE_COMPLETE:{lane}")
    now = now_iso()
    result = {"schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_INCREMENT_TERMINAL_V1",
              "task_id": TASK, "status": "REJECTED_QUALITY", "terminal_at": now,
              "execution": "FOUR_LANES_COMPLETED_FOR_FROZEN_CPU_SCOPE",
              "structure": "NEW_NUMERIC_AND_REVIEW_ARTIFACTS_RELOAD_PASS",
              "quality": "FOUR_PRODUCTS_STILL_0_OF_4_AND_DERIVED_HURO_FULL_POSE_0_OF_48",
              "improvement": "DERIVED_HURO_WINDOW_HARD_LIMIT_ONLY_NO_PRODUCT_GAIN",
              "adoption": "0_OF_4_PRODUCTS", "review": "AI_EVIDENCE_USER_PENDING",
              "lanes": lane_states, "progress": artifact_ref(progress_path),
              "final_validation": artifact_ref(validation_path),
              "motion_source_binding": artifact_ref(ATTEMPT / "lanes/motion_product/target_source_binding_v1/RESULT.json"),
              "motion_upstream_evidence": artifact_ref(ATTEMPT / "lanes/motion_product/frame47_upstream_evidence_v1/RESULT.json"),
              "huro_tradeoff": artifact_ref(ATTEMPT / "lanes/compare/huro_limit_tradeoff_dashboard_v1/RESULT.json"),
              "prior_15_slot_delivery": artifact_ref(PRIOR / "delivery/RESULT.json"),
              "product_structure": "4/4", "product_quality": "0/4", "product_adoption": "0/4",
              "new_model_runs": 0, "new_031_ik_attempts": 0, "new_huro_core_solves": 0,
              "new_derived_huro_window_postprocesses": 1, "gpu_hours": 0,
              "contact_r1_executed": 0, "training_eligible": False,
              "control_ground_truth": False, "physical_deployable": False,
              "external_metric_authority": False,
              "user_device_transfer": False,
              "claim_limit": "Four bounded CPU lanes and new evidence completed; Scene Clean and all products remain quality rejected. Derived HuRo q only removes hard-limit violations on fixed windows, with zero full-pose passes and unknown collision. No new model, IK, training or physical authority."}
    atomic_json(result_path, result)
    task_state = load_json(TASK_STATE_PATH)
    task = next((item for item in task_state["tasks"] if item.get("task_id") == TASK), None)
    index_path = REPO_ROOT / "tasks/current/INDEX.json"
    index = load_json(index_path)
    if (task is None or task.get("status") != "RUNNING"
            or task_state.get("next_task", {}).get("task_id") != TASK
            or len(index.get("task_packets", [])) != 1
            or index["task_packets"][0].get("task_id") != TASK):
        raise RuntimeError("FOUR_LANE_TERMINAL_AUTHORITY_DRIFT")
    result_ref = artifact_ref(result_path)
    task.update(status="REJECTED_QUALITY", attempt=1, updated_at=now,
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None,
                result=result_ref, last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="CPU_INCREMENT_COMPLETE_NO_PRODUCT_QUALITY_PROMOTION")
    task_state["next_task"] = None
    task_state["recent_events"] = (task_state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "REJECTED_QUALITY", "created_at": now,
        "result": result_ref, "message": "Four bounded CPU lanes completed; derived HuRo limit repair remains window-only and products 0/4 quality.",
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": "HUMAN_TO_ROBOT_FOUR_LANE_INCREMENT_TERMINAL",
                 "plan_revision": "HUMAN_TO_ROBOT_FOUR_LANE_QUALITY_INCREMENT_20260924",
                 "execution_revision": "HUMAN_TO_ROBOT_FOUR_LANE_QUALITY_INCREMENT_20260924",
                 "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(index_path),
                 "task_packets": [],
                 "claim_limit": "Finite four-lane CPU task terminal; new model, IK, collision or product-quality work requires a new scoped task."}
    revision = int(load_json(RECEIPT_PATH)["governance_revision"])
    receipt = publish_bundle(load_json(AUTHORITY_PATH), task_state,
                             event_type="HUMAN_TO_ROBOT_FOUR_LANE_INCREMENT_TERMINAL",
                             expected_revision=revision, task_packet_index_path=index_path,
                             task_packet_index_value=successor, generator_path=Path(__file__))
    return {"status": result["status"], "result": str(result_path),
            "governance_revision": receipt["governance_revision"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("scene", "sensor", "motion", "compare", "compare-review",
                                            "scene-review", "sensor-review", "motion-review",
                                            "motion-source", "motion-upstream", "compare-dashboard",
                                            "progress", "validate", "finish"), required=True)
    args = parser.parse_args()
    value = {"scene": scene, "sensor": sensor, "motion": motion,
             "compare": compare, "compare-review": compare_review,
             "scene-review": scene_review, "sensor-review": sensor_review,
             "motion-review": motion_review, "motion-source": motion_source,
             "motion-upstream": motion_upstream,
             "compare-dashboard": compare_tradeoff_review,
             "progress": progress, "validate": validate_delivery, "finish": finish}[args.stage]()
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
