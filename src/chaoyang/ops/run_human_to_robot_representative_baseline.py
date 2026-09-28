"""Bounded consumers for the registered representative baseline attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.human_ego.preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar
from chaoyang.human_ego.utils.utils_math import o6d_to_rotmat, rotmat_to_o6d
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.robot_wrist_kai_adapter import final_v3_mano_palm_basis


TASK = "human_to_robot_representative_baseline_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
SELECTION = ATTEMPT / "selection/SELECTION.json"
CHIPS = REPO_ROOT / "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/hawor/sessions/potato_chips/get_potato_chips_0915_042/HAWOR_RAW_MANO21.npz"
CHIPS_R0 = REPO_ROOT / "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001/sessions/potato_chips/get_potato_chips_0915_042/KAI22_R0_BASELINE_V1.npz"
POKER = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/recovered_0902_042_v1"
SENSOR = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor"
SENSOR_HAND = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/sensor/run_0001"
SID = "get_potato_chips_0915_042"
SOURCE = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/potato_chips") / SID
HAWOR = CHIPS
R0 = CHIPS_R0
OUT = ATTEMPT
ADAPTER = ATTEMPT / "learning_adapter_v2" / SID / "09_humanego_adapter"
SIDECAR_ROOT = ATTEMPT / "learning_sidecars"
SIDECAR = SIDECAR_ROOT / "kai22" / SID / "sidecar.npz"
MANIFEST = ATTEMPT / "LEARNING_DIAGNOSTIC_BUNDLE_V2.json"
LEFT_VIDEO = REPO_ROOT / (
    "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/"
    f"prepared/{SID}/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
)
LEFT_META = REPO_ROOT / (
    "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/"
    f"prepared/{SID}/adapter_session/preprocess/all_data"
)
LEFT_ADAPTER = ATTEMPT / "learning_adapter_v3_physical_left" / SID / "09_humanego_adapter"
LEFT_MANIFEST = ATTEMPT / "LEARNING_DIAGNOSTIC_BUNDLE_V3_PHYSICAL_LEFT.json"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _windows(mask: np.ndarray, frame: np.ndarray, time: np.ndarray, horizon: int = 50) -> int:
    """Count source-frame H50 anchors without closing gaps or skipping records."""
    mask = np.asarray(mask, bool)
    n = len(frame)
    if mask.shape != (n,) or time.shape != (n,) or horizon < 1 or n <= horizon:
        return 0
    dt = np.diff(time.astype(np.int64))
    if np.any(dt <= 0):
        return 0
    step_good = (np.diff(frame.astype(np.int64)) == 1) & (
        dt <= 2.5 * np.median(dt)
    )
    count = 0
    for start in range(n - horizon):
        # The anchor and all fifty actual future source frames are required.
        if mask[start : start + horizon + 1].all() and step_good[start : start + horizon].all():
            count += 1
    return count


def _counts(name: str, frame: np.ndarray, time: np.ndarray,
            pos: np.ndarray, rot: np.ndarray, local: np.ndarray,
            q: np.ndarray, anchor_camera: np.ndarray, quality: dict,
            inputs: dict) -> dict:
    n = len(frame)
    for field in (pos, rot, local, q):
        if field.shape != (n, 2):
            raise ValueError(f"{name} field shape differs from ({n}, 2)")
    result = {"session_id": name, "time_axis_frames": n,
              "anchor_camera_transform_frames": int(np.asarray(anchor_camera, bool).sum()),
              "input_refs": inputs, "quality_authority": quality,
              "fields": {}, "complete_62d_structural_h50": 0,
              "complete_62d_training_eligible_h50": 0}
    for label, mask in (("wrist_position", pos), ("wrist_rotation", rot),
                        ("local_shape", local), ("kai22_q", q)):
        result["fields"][label] = {
            side: {"valid_frames": int(mask[:, index].sum()),
                   "structural_h50": _windows(mask[:, index], frame, time)}
            for index, side in enumerate(("left", "right"))
        }
    structural = pos & rot & q & anchor_camera[:, None]
    result["complete_62d_structural_h50"] = _windows(
        structural.all(axis=1), frame, time
    )
    # The existing receipts do not grant full-field label-quality authority.
    # This number must never be inferred from structural coverage alone.
    return result


def run_windows() -> dict:
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    route = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if not any(row.get("task_id") == TASK and row.get("execution_allowed") is True
               for row in route.get("task_packets", [])):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    selected = load_json(SELECTION)
    if (selected["chips"]["selected"], selected["poker"]["selected"]) != (
        "get_potato_chips_0915_042", "play_cards_0902_042"
    ) or packet["frozen_gates"]["horizon_frames"] != 50:
        raise RuntimeError("SELECTION_OR_HORIZON_DRIFT")
    rows = []

    source = _load(CHIPS)
    frame = source["original_frame_indices"].astype(np.int64)
    source_frames = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/potato_chips/get_potato_chips_0915_042/preprocess/all_data")
    metadata = [load_json(source_frames / f"{int(fid):05d}/training_data.json")["metadata"] for fid in frame]
    if any(int(row["idx"]) != int(fid) for row, fid in zip(metadata, frame, strict=True)):
        raise ValueError("CHIPS_SOURCE_FRAME_ID_DRIFT")
    time = np.asarray([int(row["ts"]) for row in metadata], dtype=np.int64)
    observed = source["observed"].T.astype(bool)
    pos = observed & np.isfinite(source["joints_3d_camera"][:, :, 0, :]).all(axis=-1).T
    rot = observed & np.isfinite(source["root_orient_camera"]).all(axis=(-1, -2)).T
    local = observed & np.isfinite(source["joints_3d_camera"]).all(axis=(-1, -2)).T
    c2w = np.isfinite(source["c2w"]).all(axis=(1, 2))
    chips_r0 = _load(CHIPS_R0)
    if not np.array_equal(chips_r0["frame_id"], frame) or not np.array_equal(chips_r0["timestamp_ns"], time):
        raise ValueError("CHIPS_KAI22_FRAME_OR_TIME_DRIFT")
    chips_mapping = chips_r0["human_to_physical"].astype(int)
    if sorted(chips_mapping.tolist()) != [0, 1]:
        raise ValueError("CHIPS_KAI22_SIDE_MAPPING")
    chips_q = chips_r0["valid_side_frame"][:, chips_mapping] & np.isfinite(
        chips_r0["q22_init"][:, chips_mapping]
    ).all(axis=-1)
    rows.append(_counts(selected["chips"]["selected"], frame, time, pos, rot, local,
                        chips_q, c2w,
                        {"hawor_numeric_mask_gate": "FAILED_QUALITY_C",
                         "kai22": "EXISTING_DEVELOPMENT_HAND_R0_NOT_TRAINING_ADMITTED",
                         "camera": "SAME_SESSION_ACQUISITION_C2W_DEVELOPMENT"},
                        {"hawor": artifact_ref(CHIPS), "kai22_r0": artifact_ref(CHIPS_R0)}))

    hand_path, robot_path = POKER / "HAND_MOTION_V1.npz", POKER / "ROBOT_R0_V1.npz"
    hand, robot = _load(hand_path), _load(robot_path)
    frame, time = hand["frame_id"], hand["timestamp_ns"]
    if not np.array_equal(frame, robot["frame_id"]) or not np.array_equal(time, robot["timestamp_ns"]):
        raise ValueError("POKER_ROBOT_TIME_DRIFT")
    pos = hand["position_valid"] & np.isfinite(hand["T_camera_wrist"][..., :3, 3]).all(axis=-1)
    rot = hand["rotation_valid"] & np.isfinite(hand["T_camera_wrist"][..., :3, :3]).all(axis=(-1, -2))
    local = hand["joint_valid"].all(axis=-1) & np.isfinite(hand["joints21_root"]).all(axis=(-1, -2))
    mapping = robot["human_to_physical"].astype(int)
    if sorted(mapping.tolist()) != [0, 1]:
        raise ValueError("POKER_SIDE_MAPPING")
    q = robot["finger_valid"][:, mapping] & np.isfinite(robot["q_hand22"][:, mapping]).all(axis=-1)
    rows.append(_counts(selected["poker"]["selected"], frame, time, pos, rot, local,
                        q, hand["world_valid"].astype(bool),
                        {"world_camera": "UNVERIFIED", "robot": "LEGACY_FAILED_NUMERIC_GATE",
                         "kai22": "KINEMATIC_CANDIDATE_NOT_TRAINING_ADMITTED"},
                        {"hand_motion": artifact_ref(hand_path), "robot_r0": artifact_ref(robot_path)}))

    for session, stage in (("play_cards_0916_097", "run_097_wave0"),
                           ("play_cards_0916_098", "run_098_wave1"),
                           ("play_cards_0916_101", "run_101_wave1")):
        hand_path = SENSOR_HAND / session / "HAND_MOTION_V1.npz"
        backend_path = SENSOR / stage / "KAI22_COMMON_BACKEND_V1.npz"
        hand, backend = _load(hand_path), _load(backend_path)
        frame, time = hand["frame_id"], hand["timestamp_ns"]
        if not np.array_equal(frame, backend["source_frame_id"]) or not np.array_equal(time, backend["timestamp_ns"]):
            raise ValueError(f"{session} SENSOR_BACKEND_TIME_DRIFT")
        pos = hand["position_valid"] & np.isfinite(hand["T_camera_wrist"][..., :3, 3]).all(axis=-1)
        rot = hand["rotation_valid"] & np.isfinite(hand["T_camera_wrist"][..., :3, :3]).all(axis=(-1, -2))
        local = hand["joint_valid"].all(axis=-1) & np.isfinite(hand["joints21_root"]).all(axis=(-1, -2))
        mapping = backend["human_to_physical"].astype(int)
        if sorted(mapping.tolist()) != [0, 1]:
            raise ValueError(f"{session} SIDE_MAPPING")
        q = backend["valid"][:, mapping] & np.isfinite(backend["q22"][:, mapping]).all(axis=-1)
        rows.append(_counts(session, frame, time, pos, rot, local, q,
                            hand["world_valid"].astype(bool),
                            {"world_camera": "DEVELOPMENT_ONLY_NOT_WORLD_AUTHORITY",
                             "kai22": "KINEMATIC_ONLY_PENDING_SELF_COLLISION",
                             "visual_alignment": "NOT_INDEPENDENTLY_PASSED"},
                            {"hand_motion": artifact_ref(hand_path),
                             "kai22_backend": artifact_ref(backend_path)}))

    output = ATTEMPT / "WINDOWS_H50_V2.json"
    if output.exists():
        raise FileExistsError(output)
    result = {"schema_version": "HUMAN_TO_ROBOT_FIELD_H50_WINDOWS_V1",
              "task_id": TASK, "horizon_frames": 50,
              "count_semantics": "anchor plus actual next 50 source frames, no missing-frame compaction",
              "selection": artifact_ref(SELECTION), "sessions": rows,
              "training_eligible": False,
              "supersedes": str(ATTEMPT / "WINDOWS_H50.json"),
              "next_action": "CHECK_EXISTING_CONSUMER_ON_STRUCTURAL_DATA_WITHOUT_QUALITY_PROMOTION"}
    atomic_json(output, result)
    return result


def run_chips_arm(*, continuity_weight: float = 0.0) -> dict:
    """Consume the frozen local Kai22 R0; solve only its missing virtual arm layer."""
    from chaoyang.pipeline.full_robot_review_v2 import solve_full_chain
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    source = _load(CHIPS_R0)
    if not np.array_equal(source["human_to_physical"], [1, 0]):
        raise ValueError("CHIPS_LOCAL_R0_SIDE_CONTRACT_CHANGED")
    mount_path = REPO_ROOT / (
        "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    )
    mount = load_json(mount_path)
    if mount.get("classification") != "REAL_CAD_GEOMETRY_WITH_UNMEASURED_VIRTUAL_INSTALLATION":
        raise ValueError("MOUNT_DEVELOPMENT_CONTRACT_DRIFT")
    flange_mounts = np.asarray([
        mount["transforms"]["left_flange_to_hand_root"],
        mount["transforms"]["right_flange_to_hand_root"],
    ], dtype=np.float64)
    arrays = {
        "q22": source["q22_init"],
        "finger_valid": source["valid_side_frame"],
        "wrist_valid": source["valid_side_frame"],
        "relative_wrist_T": source["relative_wrist_T"],
        "timestamp_ns": source["timestamp_ns"],
        "frame_id": source["frame_id"],
        "human_to_physical": source["human_to_physical"],
    }
    assets = load_pinned_robot_assets(REPO_ROOT)
    solved = solve_full_chain(arrays, assets, flange_mounts, max_nfev=80,
                              continuity_weight=continuity_weight)
    candidate = continuity_weight > 0
    stem = "get_potato_chips_0915_042_FULL_ARM_CONTINUITY_CANDIDATE" if candidate else "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT"
    output = ATTEMPT / f"{stem}.npz"
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **solved)
    with np.load(output, allow_pickle=False) as check:
        if not np.array_equal(check["frame_id"], source["frame_id"]):
            raise RuntimeError("CHIPS_ARM_RELOAD_FRAME_DRIFT")
    valid, passed = solved["wrist_valid"], solved["tolerance_pass"]
    receipt = {
        "schema_version": "REPRESENTATIVE_CHIPS_FULL_ARM_CONTINUITY_CANDIDATE_V1" if candidate else "REPRESENTATIVE_CHIPS_FULL_ARM_DEVELOPMENT_V1",
        "task_id": TASK,
        "session_id": "get_potato_chips_0915_042",
        "input_r0": artifact_ref(CHIPS_R0),
        "mount_contract": artifact_ref(mount_path),
        "output": artifact_ref(output),
        "full_timeline_frames": int(len(source["frame_id"])),
        "valid_physical_side_frames": valid.sum(axis=0).astype(int).tolist(),
        "virtual_pose_gate_pass_physical_side_frames": passed.sum(axis=0).astype(int).tolist(),
        "pose_gate": "DEVELOPMENT_ONLY_20MM_15DEG_IN_NEUTRAL_RELATIVE_BASE",
        "continuity_weight": continuity_weight,
        "continuity_objective": "L2_TO_PREVIOUS_VALID_SOLUTION_RESET_AT_GAP_WITH_ORIGINAL_5MM_2DEG_POSE_SCALES" if candidate else "NONE",
        "pre_registered_candidate_recipe": "ONE_CANDIDATE_WEIGHT_0P5_NO_GRID_SEARCH" if candidate else None,
        "source_numeric_quality": "HAWOR_FAILED_QUALITY_C",
        "real_world_wrist_authority": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "next_action": "COMPARE_FIXED_FRAME_SET_AND_ABSOLUTE_GATES_BEFORE_ANY_ADOPTION" if candidate else "RENDER_ACTUAL_SAVED_Q_WITH_FIXED_THIRD_PERSON_CAMERA_AND_EVALUATE_CONTINUITY",
    }
    atomic_json(ATTEMPT / f"{stem}_RESULT.json", receipt)
    return receipt


def run_robot_review(session: str) -> dict:
    """Render saved q through the current CAD-aware renderer, fixed third-person view."""
    import cv2
    from chaoyang.pipeline.v5_product import ProductRobotRenderer
    from chaoyang.ops.render_tianji_kai_mount_proxy_audit import body_bounds

    if session == "chips":
        session_id = "get_potato_chips_0915_042"
        motion_path = ATTEMPT / f"{session_id}_FULL_ARM_DEVELOPMENT.npz"
        source_video = REPO_ROOT / (
            "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/"
            f"prepared/{session_id}/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
        )
        motion = _load(motion_path)
        motion["q_hand22"] = motion["q22"]
        motion["T_cam_base"] = np.eye(4)
        source_policy = "EXISTING_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY"
    elif session == "poker":
        session_id = "play_cards_0902_042"
        motion_path = POKER / "ROBOT_R0_V1.npz"
        source_video = Path(
            "/mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/"
            "playing_cards/play_cards_0902_042/CameraRecord_play_cards_0902_042.mp4"
        )
        motion = _load(motion_path)
        source_policy = "SOURCEINDEX0_LEFT_HALF_FISHEYE_RAW_DIAGNOSTIC"
    else:
        raise ValueError("UNKNOWN_REVIEW_SESSION")
    if motion["q_arm"].shape[0] != len(motion["frame_id"]):
        raise ValueError("ARM_TIME_AXIS_DRIFT")
    if not np.array_equal(motion["frame_id"], np.arange(len(motion["frame_id"]))):
        raise ValueError("SOURCE_FRAME_MAPPING_NOT_IDENTITY")
    if not np.isfinite(motion["q_arm"][motion["wrist_valid"]]).all():
        raise ValueError("VALID_ARM_Q_NONFINITE")
    renderer = ProductRobotRenderer(
        REPO_ROOT, motion,
        {"width": 640, "height": 480,
         "K": [[580.0, 0.0, 320.0], [0.0, 580.0, 240.0], [0.0, 0.0, 1.0]]},
        include_adapter=True,
    )
    lower, upper = body_bounds(renderer.client, (renderer.robot,))
    center = ((lower + upper) * 0.5).astype(float)
    distance = max(2.1, float(np.linalg.norm(upper - lower)) * 1.08)
    renderer.view = renderer.b.computeViewMatrixFromYawPitchRoll(
        center.tolist(), distance, 40.0, -22.0, 0.0, 2
    )
    renderer.projection = renderer.b.computeProjectionMatrixFOV(50.0, 640.0 / 480.0, 0.02, 20.0)
    output = ATTEMPT / f"{session_id}_FIXED_THIRD_PERSON_FULL_ARM.mp4"
    if output.exists():
        raise FileExistsError(output)
    encoder = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "30", "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        renderer.close()
        encoder.kill()
        raise RuntimeError("SOURCE_RGB_VIDEO_UNREADABLE")
    frames = 0
    visible_pixels = 0
    try:
        for index in range(len(motion["frame_id"])):
            okay, source_bgr = capture.read()
            if not okay:
                raise RuntimeError(f"SOURCE_VIDEO_TRUNCATED_AT_{index}")
            if session == "poker":
                source_bgr = source_bgr[:, : source_bgr.shape[1] // 2]
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            h, w = source_rgb.shape[:2]
            scale = min(640 / w, 480 / h)
            rw, rh = round(w * scale), round(h * scale)
            raw_panel = np.full((480, 640, 3), 245, dtype=np.uint8)
            x, y = (640 - rw) // 2, (480 - rh) // 2
            raw_panel[y:y + rh, x:x + rw] = cv2.resize(source_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
            layers = renderer.frame_layers(index)
            robot_panel = layers.rgb.copy()
            robot_panel[~layers.alpha] = 245
            visible_pixels += int(layers.alpha.sum())
            canvas = np.concatenate((raw_panel, robot_panel), axis=1)
            cv2.putText(canvas, f"{session_id} | source frame {index} | saved q / FK", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1)
            cv2.putText(canvas, "FIXED THIRD-PERSON | virtual development mount | not control/training GT", (648, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (15, 15, 15), 1)
            cv2.putText(canvas, f"valid physical L/R: {motion['wrist_valid'][index].astype(int).tolist()}",
                        (648, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1)
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
            frames += 1
        assert encoder.stdin is not None
        encoder.stdin.close()
        error = encoder.stderr.read().decode() if encoder.stderr is not None else ""
        if encoder.wait() != 0:
            raise RuntimeError("REVIEW_FFMPEG_FAILED:" + error[-1000:])
    finally:
        capture.release()
        renderer.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    if visible_pixels == 0:
        raise RuntimeError("THIRD_PERSON_RENDER_NO_ROBOT_PIXELS")
    check = cv2.VideoCapture(str(output))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != frames or frames != len(motion["frame_id"]):
        raise RuntimeError("THIRD_PERSON_FULL_DECODE_COUNT_DRIFT")
    result = {
        "schema_version": "REPRESENTATIVE_FIXED_THIRD_PERSON_REVIEW_V1",
        "task_id": TASK, "session_id": session_id,
        "motion": artifact_ref(motion_path), "source_video": artifact_ref(source_video),
        "review_video": artifact_ref(output), "decoded_frames": decoded,
        "source_policy": source_policy,
        "view_policy": "FIXED_FROM_NEUTRAL_URDF_BOUNDS_ALL_FRAMES_NO_TARGET_FITTING",
        "view_center_base_m": center.tolist(), "view_distance_m": distance,
        "actual_saved_q_consumed": True, "adapter_visual_consumed": True,
        "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False,
    }
    atomic_json(ATTEMPT / f"{session_id}_THIRD_PERSON_RESULT.json", result)
    return result


def run_motion_metrics() -> dict:
    """Attribute visible discontinuities without changing either saved trajectory."""
    sessions = (
        ("get_potato_chips_0915_042", ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz",
         "q22", "T_target_root", "T_actual_root"),
        ("play_cards_0902_042", POKER / "ROBOT_R0_V1.npz",
         "q_hand22", "T_target_root_cam", "T_actual_root_cam"),
    )
    reports = []
    for session_id, path, finger_key, target_key, actual_key in sessions:
        data = _load(path)
        frame = data["frame_id"].astype(np.int64)
        time = data["timestamp_ns"].astype(np.int64)
        if len(frame) < 3 or not np.all(np.diff(time) > 0):
            raise ValueError("MOTION_TIME_AXIS_INVALID")
        delta_ns = np.diff(time)
        edge_time = (np.diff(frame) == 1) & (delta_ns <= 2.5 * np.median(delta_ns))
        dt = delta_ns.astype(float) / 1e9
        sides = []
        for side, name in enumerate(("physical_left", "physical_right")):
            valid = data["wrist_valid"][:, side].astype(bool)
            edges = edge_time & valid[:-1] & valid[1:]
            target = data[target_key][:, side, :3, 3]
            actual = data[actual_key][:, side, :3, 3]
            q_arm = data["q_arm"][:, side]
            q_finger = data[finger_key][:, side]
            finite = (np.isfinite(target).all(axis=1) & np.isfinite(actual).all(axis=1)
                      & np.isfinite(q_arm).all(axis=1) & np.isfinite(q_finger).all(axis=1))
            edges &= finite[:-1] & finite[1:]
            indices = np.flatnonzero(edges)
            if not len(indices):
                sides.append({"side": name, "valid_frames": int(valid.sum()), "valid_edges": 0})
                continue
            def distribution(values: np.ndarray) -> dict:
                values = np.asarray(values, dtype=float)[indices]
                worst = int(np.argmax(values))
                return {"p50": float(np.percentile(values, 50)),
                        "p95": float(np.percentile(values, 95)),
                        "p99": float(np.percentile(values, 99)),
                        "max": float(values[worst]),
                        "max_edge_source_frames": [int(frame[indices[worst]]),
                                                   int(frame[indices[worst] + 1])]}
            target_step = np.linalg.norm(np.diff(target, axis=0), axis=1) * 1000
            actual_step = np.linalg.norm(np.diff(actual, axis=0), axis=1) * 1000
            arm_step = np.max(np.abs(np.diff(q_arm, axis=0)), axis=1)
            finger_step = np.max(np.abs(np.diff(q_finger, axis=0)), axis=1)
            target_speed = target_step / dt
            arm_speed = arm_step / dt
            simultaneous = np.flatnonzero(edges & (target_step > np.percentile(target_step[indices], 95))
                                      & (arm_step > np.percentile(arm_step[indices], 95)))
            sides.append({
                "side": name, "valid_frames": int(valid.sum()), "valid_edges": int(len(indices)),
                "target_root_step_mm": distribution(target_step),
                "actual_root_step_mm": distribution(actual_step),
                "arm_max_joint_step_rad": distribution(arm_step),
                "finger_max_joint_step_rad": distribution(finger_step),
                "target_root_speed_mm_s": distribution(target_speed),
                "arm_max_joint_speed_rad_s": distribution(arm_speed),
                "simultaneous_upper_tail_target_and_arm_edges": int(len(simultaneous)),
                "simultaneous_upper_tail_source_frames": [int(frame[i + 1]) for i in simultaneous[:20]],
                "position_gate_pass_frames": int(np.sum(valid & (data["position_residual_mm"][:, side] <= 20))),
                "rotation_gate_pass_frames": int(np.sum(valid & (data["rotation_residual_deg"][:, side] <= 15))),
            })
        reports.append({"session_id": session_id, "motion": artifact_ref(path),
                        "frames": len(frame), "median_dt_ms": float(np.median(dt) * 1000),
                        "source_frame_gap_edges": int(np.sum(~edge_time)), "sides": sides})
    result = {"schema_version": "REPRESENTATIVE_MOTION_DISCONTINUITY_DIAGNOSTIC_V1",
              "task_id": TASK, "evaluation": "SAVED_Q_TARGET_AND_INDEPENDENT_FK_NO_ALGORITHM_CHANGE",
              "edge_policy": "BOTH_ENDPOINTS_VALID_CONSECUTIVE_SOURCE_FRAMES_DT_LE_2P5X_MEDIAN",
              "units": {"root_step": "mm", "joint_step": "rad", "dt": "ms"},
              "sessions": reports, "quality_authority": "DIAGNOSTIC_ONLY_NOT_STABILITY_PASS"}
    atomic_json(ATTEMPT / "MOTION_DISCONTINUITY_DIAGNOSTIC_V1.json", result)
    return result


def assess_chips_continuity_candidate() -> dict:
    """Apply frozen same-frame improvement and pose-gate comparison, never pick frames."""
    old_path = ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz"
    new_path = ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_CONTINUITY_CANDIDATE.npz"
    old, new = _load(old_path), _load(new_path)
    if not np.array_equal(old["frame_id"], new["frame_id"]) or not np.array_equal(
        old["timestamp_ns"], new["timestamp_ns"]
    ) or not np.array_equal(old["wrist_valid"], new["wrist_valid"]):
        raise ValueError("CANDIDATE_FRAME_OR_DENOMINATOR_DRIFT")
    if not np.allclose(old["T_target_root"], new["T_target_root"], equal_nan=True):
        raise ValueError("CANDIDATE_TARGET_CHANGED")
    dt = np.diff(old["timestamp_ns"]).astype(float) / 1e9
    time_edges = (np.diff(old["frame_id"]) == 1) & (dt > 0) & (
        dt <= 2.5 * np.median(dt)
    )
    rows = []
    for side, name in enumerate(("physical_left", "physical_right")):
        valid = old["wrist_valid"][:, side]
        edge = time_edges & valid[1:] & valid[:-1]
        jerk_edge = edge[2:] & edge[1:-1] & edge[:-2]
        metrics = []
        for data in (old, new):
            q = data["q_arm"][:, side]
            step = np.max(np.abs(np.diff(q, axis=0)), axis=1)
            vel = np.diff(q, axis=0) / dt[:, None]
            acc = np.diff(vel, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
            jerk = np.diff(acc, axis=0) / ((dt[2:] + dt[1:-1] + dt[:-2]) / 3)[:, None]
            metrics.append({
                "step_p99_rad": float(np.percentile(step[edge], 99)),
                "step_max_rad": float(step[edge].max()),
                "jerk_p95_rad_s3": float(np.percentile(np.max(np.abs(jerk[jerk_edge]), axis=1), 95)),
                "position_and_rotation_gate_pass_frames": int(data["tolerance_pass"][:, side].sum()),
            })
        previous, candidate = metrics
        rows.append({"side": name, "valid_frames": int(valid.sum()),
                     "same_valid_edges": int(edge.sum()), "old": previous, "candidate": candidate,
                     "step_p99_improved_5pct": candidate["step_p99_rad"] <= .95 * previous["step_p99_rad"],
                     "jerk_p95_improved_5pct": candidate["jerk_p95_rad_s3"] <= .95 * previous["jerk_p95_rad_s3"],
                     "max_step_not_worse": candidate["step_max_rad"] <= previous["step_max_rad"],
                     "tracking_gate_count_not_worse": candidate["position_and_rotation_gate_pass_frames"] >= previous["position_and_rotation_gate_pass_frames"]})
    accepted = all(r["step_p99_improved_5pct"] and r["jerk_p95_improved_5pct"]
                   and r["max_step_not_worse"] and r["tracking_gate_count_not_worse"]
                   for r in rows)
    result = {"schema_version": "CHIPS_ARM_CONTINUITY_CANDIDATE_ASSESSMENT_V1",
              "task_id": TASK, "session_id": "get_potato_chips_0915_042",
              "old_motion": artifact_ref(old_path), "new_motion": artifact_ref(new_path),
              "same_target_and_validity": True, "assessment": rows,
              "improvement_accepted": accepted, "adoption": "NOT_ADOPTED" if not accepted else "CANDIDATE_ONLY_PENDING_OTHER_GATES",
              "quality": "REJECTED_TRACKING_REGRESSION" if not accepted else "NOT_YET_COMPLETE",
              "amplitude_and_latency": "NOT_EVALUATED_AFTER_MANDATORY_TRACKING_GATE_FAILURE" if not accepted else "PENDING",
              "old_motion_unchanged": True, "training_eligible": False}
    atomic_json(ATTEMPT / "CHIPS_ARM_CONTINUITY_CANDIDATE_ASSESSMENT_V1.json", result)
    return result


def anchor_camera_wrist(anchor_c2w: np.ndarray, future_c2w: np.ndarray,
                        future_camera_wrist: np.ndarray) -> np.ndarray:
    """Express a future wrist in the current anchor camera, not its own camera."""
    transforms = [np.asarray(item, dtype=np.float64) for item in
                  (anchor_c2w, future_c2w, future_camera_wrist)]
    if any(item.shape != (4, 4) or not np.isfinite(item).all() for item in transforms):
        raise ValueError("CAMERA_OR_WRIST_TRANSFORM_INVALID")
    return np.linalg.inv(transforms[0]) @ transforms[1] @ transforms[2]


def prepare() -> dict:
    if MANIFEST.exists():
        raise FileExistsError("LEARNING_DIAGNOSTIC_BUNDLE_ALREADY_EXISTS")
    hawor, r0 = _load(HAWOR), _load(R0)
    n = len(r0["frame_id"])
    if n != 363 or not np.array_equal(r0["frame_id"], np.arange(n)):
        raise ValueError("CHIPS_FRAME_AXIS_DRIFT")
    mapping = r0["human_to_physical"].astype(int)
    if not np.array_equal(mapping, [1, 0]):
        raise ValueError("ANATOMICAL_TO_PHYSICAL_SIDE_DRIFT")
    observed = hawor["observed"].T.astype(bool)
    q = r0["q22_init"][:, mapping].astype(np.float32)
    valid = r0["valid_side_frame"][:, mapping].astype(bool) & observed
    camera_wrist = np.full((n, 2, 4, 4), np.nan, np.float32)
    for anatomical, name in enumerate(("left", "right")):
        for frame in np.flatnonzero(valid[:, anatomical]):
            joints = np.asarray(hawor["joints_3d_camera"][anatomical, frame], np.float64)
            if not np.isfinite(joints).all():
                valid[frame, anatomical] = False
                continue
            try:
                rotation = final_v3_mano_palm_basis(joints, handedness=name)
            except (ValueError, np.linalg.LinAlgError):
                valid[frame, anatomical] = False
                continue
            camera_wrist[frame, anatomical] = np.eye(4, dtype=np.float32)
            camera_wrist[frame, anatomical, :3, :3] = rotation
            camera_wrist[frame, anatomical, :3, 3] = joints[0]
    times = np.asarray(r0["timestamp_ns"], np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("CHIPS_NONMONOTONIC_TIME")
    edge = np.diff(times) <= 2.5 * np.median(np.diff(times))
    starts = [start for start in range(n - 50)
              if valid[start:start + 51].all() and edge[start:start + 50].all()]
    if not starts:
        raise RuntimeError("NO_STRUCTURAL_DUAL_HAND_H50_WINDOW")
    start = starts[0]
    frames = np.arange(start, start + 51, dtype=int)
    source_c2w_mismatch = []
    records: dict[str, dict] = {}
    for frame in frames:
        original = SOURCE / f"preprocess/all_data/{frame:05d}/training_data.json"
        source_image = original.with_name("rgb.png")
        data = load_json(original)
        meta = dict(data["metadata"])
        if int(meta["idx"]) != frame or int(meta["ts"]) != int(times[frame]):
            raise ValueError(f"SOURCE_TIMESTAMP_OR_FRAME_DRIFT_{frame}")
        # This is a byte-documented alias of the existing camera transform,
        # not an invented static world/camera calibration.
        meta["world_transforms"] = {"cam0": meta["c2w"]}
        data["metadata"] = meta
        # The source PICO26 record has no T_hand_to_world/grasp consumed by
        # this loader. Keep the original JSON as a signed input and mask the
        # unsupported ICT hand tokens; current Robot state remains a separate
        # valid token. Never invent a grasp observation.
        entities = dict(data.get("entities", {}))
        entities["hands"] = {}
        data["entities"] = entities
        source_c2w_mismatch.append(float(np.max(np.abs(
            np.asarray(meta["c2w"]) - np.asarray(hawor["c2w"][frame])
        ))))
        folder = ADAPTER / "preprocess" / "all_data" / f"{frame:05d}"
        folder.mkdir(parents=True, exist_ok=False)
        metadata_path = folder / "training_data.json"
        atomic_json(metadata_path, data)
        bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (int(meta["h"]), int(meta["w"])):
            raise ValueError(f"SOURCE_RGB_UNREADABLE_OR_DOMAIN_MISMATCH_{frame}")
        image_path = folder / "rgb.png"
        if not cv2.imwrite(str(image_path), cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)):
            raise RuntimeError(f"RGB_WRITE_FAILED_{frame}")
        records[f"{frame:05d}"] = {
            "metadata": artifact_ref(metadata_path), "image": artifact_ref(image_path),
            "original_metadata": artifact_ref(original), "original_rgb": artifact_ref(source_image),
        }
    assets = load_pinned_robot_assets(REPO_ROOT)
    models = [assets.right_hand, assets.left_hand]  # anatomical L/R -> physical R/L
    joint_names = np.asarray([[j.name for j in m.joints if j.joint_type != "fixed"] for m in models])
    lower = np.asarray([[j.lower for j in m.joints if j.joint_type != "fixed"] for m in models], np.float32)
    upper = np.asarray([[j.upper for j in m.joints if j.joint_type != "fixed"] for m in models], np.float32)
    wrist9 = np.full((51, 2, 9), np.nan, np.float32)
    for local, frame in enumerate(frames):
        for side in range(2):
            transform = camera_wrist[frame, side]
            wrist9[local, side, :3] = transform[:3, 3]
            wrist9[local, side, 3:] = rotmat_to_o6d(transform[:3, :3])
            if not np.allclose(o6d_to_rotmat(wrist9[local, side, 3:]),
                               transform[:3, :3], atol=2e-5):
                raise RuntimeError("WRIST_6D_ROUNDTRIP_FAILED")
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    if not SIDECAR.exists():
        np.savez_compressed(
            SIDECAR, schema_version=np.asarray("humanego-robot-sidecar-v1"),
            embodiment=np.asarray("kai22"), frame_names=np.asarray([f"{x:05d}" for x in frames]),
            timestamps_ns=times[frames], wrist_9d=wrist9,
            wrist_T_camera=camera_wrist[frames], q=q[frames], valid=valid[frames],
            confidence=np.zeros((51, 2), np.float32), grasp=np.zeros((51, 2), np.float32),
            joint_names=joint_names, joint_lower=lower, joint_upper=upper,
            canonical_sha256=np.asarray(artifact_ref(HAWOR)["sha256"]),
            retarget_confidence=np.zeros((51, 2), np.float32),
            failure_reason=np.full((51, 2), "DIAGNOSTIC_GRADE_C", dtype="<U20"),
            hand_object_T=np.full((51, 2, 4, 4), np.nan, np.float32),
        )
    else:
        with np.load(SIDECAR, allow_pickle=False) as old:
            if (not np.array_equal(old["frame_names"], [f"{x:05d}" for x in frames])
                    or not np.allclose(old["q"], q[frames], equal_nan=True, atol=1e-6)
                    or not np.allclose(old["wrist_T_camera"], camera_wrist[frames], equal_nan=True, atol=1e-6)):
                raise RuntimeError("EXISTING_DIAGNOSTIC_SIDECAR_DRIFT")
    sidecar_check = validate_sidecar(SIDECAR, EMBODIMENTS["kai22"])
    if sidecar_check["active_left"] != 51 or sidecar_check["active_right"] != 51:
        raise RuntimeError("SELECTED_WINDOW_NOT_FULL_STRUCTURAL")
    manifest = {
        "schema_version": "REPRESENTATIVE_LEARNING_DIAGNOSTIC_BUNDLE_V2",
        "session_id": SID, "frame_range_inclusive": [int(frames[0]), int(frames[-1])],
        "source_frame_count": n, "future_horizon": 50,
        "field_layout_62d": "Lpos3,Rpos3,Lrot6,Rrot6,Lq22,Rq22",
        "source_hawor": artifact_ref(HAWOR), "source_local_r0": artifact_ref(R0),
        "sidecar": artifact_ref(SIDECAR), "sidecar_validation": sidecar_check,
        "adapter_root": str(ADAPTER), "selector_root": str(ADAPTER),
        "selector_records": {SID: records},
        "original_image_domain": "0915_PHYSICAL_LEFT_RGB_PROCESSED_1280x960",
        "model_image_domain": "EXACT_SAME_SOURCE_RGB_RESIZE_ONLY_320x240",
        "source_vs_hawor_c2w_max_abs_difference": float(max(source_c2w_mismatch)),
        "camera_transform_authority": "UNVERIFIED_CROSS_PRODUCER_ALIGNMENT",
        "q_label_authority": "DEVELOPMENT_LOCAL_R0_FROM_FAILED_QUALITY_C_HAWOR",
        "ict_hand_tokens": "MASKED_UNSUPPORTED_PICO26_SCHEMA_NO_SYNTHETIC_GRASP",
        "supersedes_failed_interface_attempt": str(ATTEMPT / "LEARNING_DIAGNOSTIC_BUNDLE.json"),
        "confidence_semantics": "ZERO_UNCALIBRATED_NOT_QUALITY_PASS",
        "training_eligible": False, "control_ground_truth": False,
        "next_action": "RUN_EXISTING_DATALOADER_AND_RANDOM_WEIGHT_FORWARD_WITHOUT_OPTIMIZER",
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def prepare_left_eye() -> dict:
    """Versioned correction: pair the HaWoR left-eye labels with left-eye RGB/K/c2w."""
    if LEFT_MANIFEST.exists():
        raise FileExistsError("PHYSICAL_LEFT_LEARNING_BUNDLE_ALREADY_EXISTS")
    old = load_json(MANIFEST)
    if old["session_id"] != SID or old["frame_range_inclusive"] != [0, 50]:
        raise ValueError("LEARNING_V2_WINDOW_IDENTITY_DRIFT")
    hawor = _load(HAWOR)
    r0_time = _load(R0)["timestamp_ns"]
    capture = cv2.VideoCapture(str(LEFT_VIDEO))
    if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 363:
        raise ValueError("PHYSICAL_LEFT_SOURCE_VIDEO_UNREADABLE_OR_TRUNCATED")
    records = {}
    old_right_vs_left_rgb_mean = []
    max_left_c2w_delta = 0.0
    try:
        for frame in range(51):
            okay, left_bgr = capture.read()
            if not okay or left_bgr.shape[:2] != (960, 1280):
                raise ValueError(f"PHYSICAL_LEFT_FRAME_MISSING_{frame}")
            original = SOURCE / f"preprocess/all_data/{frame:05d}/training_data.json"
            source = load_json(original)
            source_meta = source["metadata"]
            prepared_path = LEFT_META / f"{frame:05d}/training_data.json"
            prepared = load_json(prepared_path)["metadata"]
            if prepared.get("camera_eye") != "physical_left" or prepared.get("camera_source_index") != 1:
                raise ValueError(f"PREPARED_CAMERA_EYE_DRIFT_{frame}")
            left_c2w = np.asarray(prepared["c2w"], dtype=np.float64)
            max_left_c2w_delta = max(max_left_c2w_delta,
                                     float(np.max(np.abs(left_c2w - hawor["c2w"][frame]))))
            if max_left_c2w_delta > 1e-8:
                raise ValueError(f"HAWOR_AND_RGB_LEFT_CAMERA_TRANSFORM_DRIFT_{frame}")
            if int(source_meta["idx"]) != frame or int(source_meta["ts"]) != int(r0_time[frame]):
                raise ValueError(f"SOURCE_FRAME_OR_TIMESTAMP_DRIFT_{frame}")
            right_bgr = cv2.imread(str(original.with_name("rgb.png")), cv2.IMREAD_COLOR)
            if right_bgr is None or right_bgr.shape != left_bgr.shape:
                raise ValueError(f"ORIGINAL_RIGHT_RGB_MISSING_{frame}")
            old_right_vs_left_rgb_mean.append(float(np.mean(cv2.absdiff(right_bgr, left_bgr))))
            data = dict(source)
            meta = dict(source_meta)
            meta.update(c2w=prepared["c2w"], k=prepared["k"],
                        world_transforms={"cam0": prepared["c2w"]},
                        image_domain="0915_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY",
                        camera_eye="physical_left", camera_source_index=1)
            data["metadata"] = meta
            entities = dict(data.get("entities", {}))
            entities["hands"] = {}  # unsupported PICO26 ICT fields stay masked
            data["entities"] = entities
            folder = LEFT_ADAPTER / "preprocess/all_data" / f"{frame:05d}"
            folder.mkdir(parents=True, exist_ok=False)
            metadata_path, image_path = folder / "training_data.json", folder / "rgb.png"
            atomic_json(metadata_path, data)
            if not cv2.imwrite(str(image_path), cv2.resize(
                left_bgr, (320, 240), interpolation=cv2.INTER_AREA
            )):
                raise RuntimeError("PHYSICAL_LEFT_RESIZED_RGB_WRITE_FAILED")
            records[f"{frame:05d}"] = {
                "metadata": artifact_ref(metadata_path), "image": artifact_ref(image_path),
                "original_metadata": artifact_ref(original),
                "prepared_left_metadata": artifact_ref(prepared_path),
                "prepared_left_video": artifact_ref(LEFT_VIDEO) if frame == 0 else None,
            }
    finally:
        capture.release()
    if min(old_right_vs_left_rgb_mean) < 1.0:
        raise RuntimeError("LEFT_RGB_NOT_DISTINCT_FROM_ORIGINAL_RIGHT_EYE")
    manifest = {
        "schema_version": "REPRESENTATIVE_LEARNING_DIAGNOSTIC_BUNDLE_V3_PHYSICAL_LEFT",
        "session_id": SID, "frame_range_inclusive": [0, 50], "future_horizon": 50,
        "field_layout_62d": "Lpos3,Rpos3,Lrot6,Rrot6,Lq22,Rq22",
        "source_hawor": artifact_ref(HAWOR), "source_local_r0": artifact_ref(R0),
        "sidecar": artifact_ref(SIDECAR), "adapter_root": str(LEFT_ADAPTER),
        "selector_root": str(LEFT_ADAPTER), "selector_records": {SID: records},
        "original_right_eye_metadata_source": artifact_ref(SOURCE / "preprocess/all_data/00000/training_data.json"),
        "physical_left_video": artifact_ref(LEFT_VIDEO),
        "source_eye": "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY_1280X960",
        "source_vs_left_eye_mean_pixel_absolute_difference_min": min(old_right_vs_left_rgb_mean),
        "source_vs_left_eye_mean_pixel_absolute_difference_max": max(old_right_vs_left_rgb_mean),
        "ha_wor_left_c2w_max_abs_difference": max_left_c2w_delta,
        "camera_transform_authority": "SAME_FROZEN_PHYSICAL_LEFT_PREPARED_DOMAIN_AND_HAWOR_C2W_DEVELOPMENT",
        "q_label_authority": "DEVELOPMENT_LOCAL_R0_FROM_FAILED_QUALITY_C_HAWOR",
        "ict_hand_tokens": "MASKED_UNSUPPORTED_PICO26_SCHEMA_NO_SYNTHETIC_GRASP",
        "supersedes_failed_domain_bundle": artifact_ref(MANIFEST),
        "training_eligible": False, "control_ground_truth": False,
        "next_action": "RUN_REAL_CONSUMER_AND_VERIFY_PHYSICAL_LEFT_FRAME_AND_ANCHOR_CAMERA_ROUNDTRIP",
    }
    atomic_json(LEFT_MANIFEST, manifest)
    return manifest


def check(*, corrected_eye: bool = False) -> dict:
    import torch
    from chaoyang.human_ego.training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions
    from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
    from chaoyang.human_ego.training.FlowMatchingTrainer import sanitize_flow_action_targets

    manifest_path = LEFT_MANIFEST if corrected_eye else MANIFEST
    manifest = load_json(manifest_path)
    adapter = Path(manifest["adapter_root"])
    if manifest["sidecar"] != artifact_ref(SIDECAR):
        raise ValueError("ROBOT_SIDECAR_REFERENCE_DRIFT")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    ds = FlowMatchingDataloader(
        sessions=[MPSSessions(mps_path=str(adapter))], image_size=(240, 320),
        pred_horizon=50, single_hand=False, img_name="rgb.png", centric_mode="ego_centric",
        frame_mode="camera_frame", action_mode="absolute",
        hand_action_representation="kaihand_joint_state", robot_sidecar_root=str(SIDECAR_ROOT),
        use_pcd_features=False, use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, enable_augmentation=False, seed=7,
        selector_records=manifest["selector_records"], selector_root=str(adapter),
        sidecar_sha256_by_session={SID: manifest["sidecar"]["sha256"]},
        allowed_window_starts={SID: {manifest["frame_range_inclusive"][0]}},
        hand_tracking_method="aria_mps",
    )
    if len(ds) != 1:
        raise RuntimeError("FROZEN_SINGLE_WINDOW_NOT_CONSUMED")
    sample = ds[0]
    y = sample["y_action"].unsqueeze(0)
    valid = sample["action_valid_mask"].unsqueeze(0)
    if y.shape != (1, 50, 62) or valid.shape != y.shape or not bool(valid.all()):
        raise RuntimeError("REAL_H50_FIELD_SHAPE_OR_VALIDITY_DRIFT")
    if not torch.isfinite(sample["x_rgb"]).all() or not torch.isfinite(sample["x_ict"]).all():
        raise RuntimeError("REAL_CURRENT_CONDITION_NONFINITE")
    if not bool(sample["robot_state_mask"].all()):
        raise RuntimeError("REAL_CURRENT_ROBOT_STATE_MISSING")
    model = FlowMatchingModel(
        single_hand=False, pred_horizon=50, img_size=(240, 320),
        vision_embed_dim=128, num_decoder_layers=2, num_heads=4, dropout=0.0,
        use_pcd_features=False, use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False, use_aux_temporal_contrastive=False,
        use_region_attn=False, hand_action_representation="kaihand_joint_state",
    ).eval()
    with torch.inference_mode():
        safe = sanitize_flow_action_targets(y, valid, torch.ones(1))
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(y.shape, generator=generator)
        t = torch.full((1, 1), 0.5)
        xt = (1 - t[:, :, None]) * noise + t[:, :, None] * safe
        pred = model(
            x_rgb=sample["x_rgb"].unsqueeze(0), x_ict=sample["x_ict"].unsqueeze(0),
            ict_mask=sample["ict_mask"].unsqueeze(0), x_t=xt, t=t,
            x_robot_state=sample["x_robot_state"].unsqueeze(0),
            robot_state_mask=sample["robot_state_mask"].unsqueeze(0),
        )
        pred["action_pred"] = xt + 0.5 * pred["v_pred"]
        losses = model.compute_loss(pred, {
            "v_target": safe - noise, "action_valid_mask": valid,
            "y_action": safe, "sample_weight": torch.ones(1),
            "joint_lower": sample["joint_lower"].unsqueeze(0),
            "joint_upper": sample["joint_upper"].unsqueeze(0),
            "y_done": sample["y_done"][-1:],
        })
    numbers = {key: float(value) for key, value in losses.items() if value.ndim == 0}
    if any(not np.isfinite(value) for value in numbers.values()):
        raise RuntimeError("REAL_CONSUMER_LOSS_NONFINITE")
    with np.load(SIDECAR, allow_pickle=False) as archive:
        exported_q = np.asarray(archive["q"], np.float32)
        exported_pose = np.asarray(archive["wrist_T_camera"], np.float32)
    source_q = _load(R0)["q22_init"][:, [1, 0]].astype(np.float32)
    start = manifest["frame_range_inclusive"][0]
    if not np.allclose(exported_q, source_q[start:start + 51], atol=1e-6):
        raise RuntimeError("KAI22_Q_ROUNDTRIP_DRIFT")
    if not np.allclose(sample["y_action"][:, 18:40], exported_q[1:, 0], atol=1e-6):
        raise RuntimeError("LOADER_LEFT_Q_ORDER_DRIFT")
    if not np.allclose(sample["y_action"][:, 40:62], exported_q[1:, 1], atol=1e-6):
        raise RuntimeError("LOADER_RIGHT_Q_ORDER_DRIFT")
    for side in range(2):
        if not np.allclose(o6d_to_rotmat(rotmat_to_o6d(exported_pose[0, side, :3, :3])),
                           exported_pose[0, side, :3, :3], atol=2e-5):
            raise RuntimeError("LOADER_WRIST_ROTATION_ROUNDTRIP_DRIFT")
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    assets = load_pinned_robot_assets(REPO_ROOT)
    models = (assets.right_hand, assets.left_hand)
    fk_max_link_position_delta_m = 0.0
    for side, model_hand in enumerate(models):
        names = [joint.name for joint in model_hand.joints if joint.joint_type != "fixed"]
        for future in (0, 24, 49):
            original = forward_kinematics(model_hand, {
                name: float(source_q[start + future + 1, side, joint])
                for joint, name in enumerate(names)
            })
            returned = forward_kinematics(model_hand, {
                name: float(sample["y_action"][future, (18 if side == 0 else 40) + joint])
                for joint, name in enumerate(names)
            })
            fk_max_link_position_delta_m = max(
                fk_max_link_position_delta_m,
                *(float(np.linalg.norm(original[link][:3, 3] - returned[link][:3, 3]))
                  for link in original)
            )
    if fk_max_link_position_delta_m > 1e-6:
        raise RuntimeError("LOADER_KAI22_INDEPENDENT_FK_ROUNDTRIP_DRIFT")
    c2w = [np.asarray(load_json(adapter / "preprocess/all_data" /
                                f"{int(fid):05d}/training_data.json")["metadata"]["c2w"],
                      dtype=np.float64)
           for fid in range(start, start + 51)]
    position_roundtrip_max_m = 0.0
    naive_own_camera_max_m = 0.0
    for future in range(50):
        for side in range(2):
            anchor_pose = anchor_camera_wrist(c2w[0], c2w[future + 1],
                                              exported_pose[future + 1, side])
            normalized = (anchor_pose[:3, 3] - ds.pos_mean) / ds.pos_std
            channel = 0 if side == 0 else 3
            position_roundtrip_max_m = max(position_roundtrip_max_m,
                float(np.max(np.abs((sample["y_action"][future, channel:channel + 3].numpy()
                                      - normalized) * ds.pos_std))))
            naive_own_camera_max_m = max(naive_own_camera_max_m,
                float(np.linalg.norm(anchor_pose[:3, 3]
                                     - exported_pose[future + 1, side, :3, 3])))
    if position_roundtrip_max_m > 1e-5:
        raise RuntimeError("LOADER_ANCHOR_CAMERA_POSITION_DRIFT")
    result = {
        "schema_version": "REPRESENTATIVE_REAL_FLOW_CONSUMER_CHECK_V4_PHYSICAL_LEFT" if corrected_eye else "REPRESENTATIVE_REAL_FLOW_CONSUMER_CHECK_V3",
        "bundle": artifact_ref(manifest_path), "session_id": SID,
        "selected_source_frame": int(sample["meta_t"]), "future_steps": 50,
        "actual_rgb_shape": list(sample["x_rgb"].shape),
        "actual_action_shape": list(y.shape),
        "actual_valid_supervision": int(valid.sum()),
        "actual_forward_shape": list(pred["v_pred"].shape),
        "losses_random_initialization": numbers,
        "independent_urdf_fk_max_link_position_roundtrip_delta_m": fk_max_link_position_delta_m,
        "anchor_camera_position_roundtrip_max_component_delta_m": position_roundtrip_max_m,
        "naive_future_own_camera_vs_anchor_max_position_delta_m": naive_own_camera_max_m,
        "supersedes": str(OUT / ("LEARNING_REAL_CONSUMER_RESULT_V3.json" if corrected_eye else "LEARNING_REAL_CONSUMER_RESULT_V2.json")),
        "optimizer_steps": 0, "checkpoint_loaded": False,
        "camera_transform_authority": manifest["camera_transform_authority"],
        "label_quality": "FAILED_UPSTREAM_NUMERIC_QUALITY_C" if corrected_eye else "FAILED_UPSTREAM_NUMERIC_AND_WRONG_EYE_RGB_CAMERA_DOMAIN",
        "interface_executable": True, "training_eligible": False,
        "learning_benefit_evaluated": False,
    }
    atomic_json(OUT / ("LEARNING_REAL_CONSUMER_RESULT_V4_PHYSICAL_LEFT.json" if corrected_eye else "LEARNING_REAL_CONSUMER_RESULT_V3.json"), result)
    return result


def validate_delivery() -> dict:
    """Reload actual numeric/video consumers without promoting their quality."""
    videos = (
        ("get_potato_chips_0915_042", ATTEMPT / "get_potato_chips_0915_042_FIXED_THIRD_PERSON_FULL_ARM.mp4", 363),
        ("play_cards_0902_042", ATTEMPT / "play_cards_0902_042_FIXED_THIRD_PERSON_FULL_ARM.mp4", 171),
        ("play_cards_0902_042_clean_rejected", REPO_ROOT / (
            "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/"
            "lanes/lane1_scene/clean_candidate_042_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4"), 171),
    )
    video_rows = []
    for label, path, expected in videos:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"DELIVERY_VIDEO_UNREADABLE:{label}")
        frames = 0
        while capture.read()[0]:
            frames += 1
        capture.release()
        if frames != expected:
            raise RuntimeError(f"DELIVERY_VIDEO_TRUNCATED:{label}:{frames}/{expected}")
        video_rows.append({"label": label, "decoded_frames": frames,
                           "video": artifact_ref(path)})
    old = _load(ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz")
    candidate = _load(ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_CONTINUITY_CANDIDATE.npz")
    if len(old["frame_id"]) != 363 or not np.array_equal(old["frame_id"], candidate["frame_id"]):
        raise RuntimeError("DELIVERY_CHIPS_NUMERIC_TIME_DRIFT")
    if load_json(ATTEMPT / "CHIPS_ARM_CONTINUITY_CANDIDATE_ASSESSMENT_V1.json")["adoption"] != "NOT_ADOPTED":
        raise RuntimeError("REJECTED_CANDIDATE_ACCIDENTALLY_ADOPTED")
    left = load_json(LEFT_MANIFEST)
    checked = 0
    for row in left["selector_records"][SID].values():
        for field in ("metadata", "image", "original_metadata", "prepared_left_metadata"):
            actual = artifact_ref(Path(row[field]["path"]))
            if actual != row[field]:
                raise RuntimeError(f"LEARNING_SELECTOR_REFERENCE_DRIFT:{field}")
            checked += 1
    consumer = load_json(ATTEMPT / "LEARNING_REAL_CONSUMER_RESULT_V4_PHYSICAL_LEFT.json")
    if (consumer["bundle"] != artifact_ref(LEFT_MANIFEST)
            or consumer["optimizer_steps"] != 0
            or consumer["training_eligible"]
            or consumer["actual_valid_supervision"] != 3100):
        raise RuntimeError("LEARNING_CONSUMER_AUTHORITY_DRIFT")
    status = load_json(REPO_ROOT / "docs/current/STATUS.json")
    if status["counts"] != {"products_structure": "4/4", "products_quality": "0/4",
                             "products_adopted": "0/4"}:
        raise RuntimeError("PRODUCT_QUALITY_COUNT_DRIFT")
    result = {
        "schema_version": "REPRESENTATIVE_BASELINE_DELIVERY_VALIDATION_V1", "task_id": TASK,
        "videos": video_rows, "chips_old_q": artifact_ref(ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz"),
        "chips_rejected_candidate_q": artifact_ref(ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_CONTINUITY_CANDIDATE.npz"),
        "learning_physical_left_bundle": artifact_ref(LEFT_MANIFEST),
        "learning_actual_consumer": artifact_ref(ATTEMPT / "LEARNING_REAL_CONSUMER_RESULT_V4_PHYSICAL_LEFT.json"),
        "learning_selector_refs_sha_checked": checked,
        "scene_chips_clean": "NOT_PRODUCED_NO_LEGAL_MODEL_MASK_OR_REFERENCE_SUPPORT",
        "scene_poker_clean": "REUSED_EXECUTED_REJECTED_QUALITY",
        "product_counts_unchanged": status["counts"],
        "technical_structure": "PASS_FOR_LISTED_ARTIFACTS",
        "algorithm_quality": "NOT_PASSED", "user_visual_review": "PENDING",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    atomic_json(ATTEMPT / "FINAL_VALIDATION.json", result)
    return result


def write_result() -> dict:
    """Close the finite producer work with separate execution and quality axes."""
    output = ATTEMPT / "RESULT.json"
    if output.exists():
        raise FileExistsError(output)
    validation = load_json(ATTEMPT / "FINAL_VALIDATION.json")
    if validation["algorithm_quality"] != "NOT_PASSED":
        raise RuntimeError("QUALITY_MUST_NOT_BE_PROMOTED_BY_STRUCTURE")
    run = load_json(ATTEMPT / "RUN_SIGNATURE.json")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "t0": run["t0"], "deadline_at": run["deadline_at"],
        "terminal_at": now_iso(), "status": "REJECTED_QUALITY_LIMITED_DELIVERY",
        "selection": artifact_ref(SELECTION),
        "robot": {
            "chips_saved_q_fk": artifact_ref(ATTEMPT / "get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz"),
            "chips_video": artifact_ref(ATTEMPT / "get_potato_chips_0915_042_FIXED_THIRD_PERSON_FULL_ARM.mp4"),
            "poker_video": artifact_ref(ATTEMPT / "play_cards_0902_042_FIXED_THIRD_PERSON_FULL_ARM.mp4"),
            "continuity": artifact_ref(ATTEMPT / "MOTION_DISCONTINUITY_DIAGNOSTIC_V1.json"),
            "candidate_rejected": artifact_ref(ATTEMPT / "CHIPS_ARM_CONTINUITY_CANDIDATE_ASSESSMENT_V1.json"),
            "structure": "FULL_TIMELINE_SAVED_Q_AND_FIXED_THIRD_PERSON_DECODED",
            "quality": "REJECTED_ARM_DISCONTINUITIES_AND_UPSTREAM_NUMERIC_QUALITY",
            "adoption": "NOT_ADOPTED",
        },
        "clean": {
            "chips": "NOT_PRODUCED_NO_INDEPENDENT_MODEL_MASK_PROTECT_REFERENCE_SUPPORT",
            "poker_previous_result": artifact_ref(REPO_ROOT / (
                "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/"
                "lanes/lane1_scene/clean_candidate_042_wave4/RESULT.json")),
            "poker_quality": "REUSED_REJECTED_QUALITY",
            "product_adoption": "NOT_ADOPTED",
        },
        "learning": {
            "field_h50": artifact_ref(ATTEMPT / "WINDOWS_H50_V2.json"),
            "physical_left_bundle": artifact_ref(LEFT_MANIFEST),
            "real_flow_consumer": artifact_ref(ATTEMPT / "LEARNING_REAL_CONSUMER_RESULT_V4_PHYSICAL_LEFT.json"),
            "fixed_error": "RIGHT_EYE_SOURCE_RGB_WAS_PAIRED_WITH_LEFT_EYE_HAWOR; V3_BINDS_LEFT_RGB_K_C2W",
            "interface_executable": True, "label_quality": "FAILED_UPSTREAM_HAWOR_NUMERIC_C",
            "training_eligible": False, "optimizer_steps": 0,
        },
        "sensor": {
            "existing_corrected_review": artifact_ref(REPO_ROOT / (
                "_run/current/human_to_robot_sensor_display_correction_20260923/"
                "attempts/attempt_0001/lanes/sensor/review_metric_v2/RESULT.json")),
            "new_backend_runs": 0, "structure": "REUSED_466_FRAMES",
            "wrist_visual_fit": "INCONCLUSIVE_NO_INDEPENDENT_GT",
            "kai22_training_eligible": False,
        },
        "huro": {
            "existing_common_comparison": artifact_ref(REPO_ROOT / (
                "_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/"
                "lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json")),
            "new_solver_runs": 0, "quality": "ORIGINAL_HURO_Q_HARD_LIMIT_FAILURE",
            "winner": "NONE",
        },
        "tests_and_own_temp_cleanup": artifact_ref(ATTEMPT / "TEST_AND_EPHEMERAL_CLEANUP_RESULT.json"),
        "validation": artifact_ref(ATTEMPT / "FINAL_VALIDATION.json"),
        "video_navigation": str(REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE/INDEX_ZH.md"),
        "product_counts": validation["product_counts_unchanged"],
        "claim_limit": "Executable interface and corrected eye identity do not repair upstream hand quality or prove stable Robot action. Third-person mount is virtual development, not measured installation. No full Clean/product adoption.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    atomic_json(output, result)
    return result



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("windows", "chips_arm", "chips_arm_candidate", "review_chips", "review_poker",
                                          "motion_metrics", "assess_chips_candidate",
                                          "learning_prepare", "learning_check", "learning_prepare_left_eye", "learning_check_left_eye",
                                          "validate", "finalize"))
    args = parser.parse_args()
    if args.stage == "windows":
        result = run_windows()
    elif args.stage == "chips_arm":
        result = run_chips_arm()
    elif args.stage == "chips_arm_candidate":
        result = run_chips_arm(continuity_weight=0.5)
    elif args.stage == "learning_prepare":
        result = prepare()
    elif args.stage == "learning_check":
        result = check()
    elif args.stage == "learning_prepare_left_eye":
        result = prepare_left_eye()
    elif args.stage == "learning_check_left_eye":
        result = check(corrected_eye=True)
    elif args.stage == "motion_metrics":
        result = run_motion_metrics()
    elif args.stage == "assess_chips_candidate":
        result = assess_chips_continuity_candidate()
    elif args.stage == "validate":
        result = validate_delivery()
    elif args.stage == "finalize":
        result = write_result()
    else:
        result = run_robot_review(args.stage.split("_", 1)[1])
    print(json.dumps({"stage": args.stage, "status": "EXECUTED",
                      "result_type": result["schema_version"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
