"""Registered CPU-only three-session PICO/MANUS motion and image-domain audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from chaoyang.pipeline.pico_manus_motion_v2 import (
    EDGES_21, FRAME_COUNTS, MANUS_NAMES, SESSION_IDS, SIDES,
    MotionContractError, compose_motion, pose_matrices, project_points,
)


def artifact(path):
    p = Path(path).resolve(strict=True)
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    after = p.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise MotionContractError(f"unstable input: {p}")
    return {"path": str(p), "bytes": after.st_size, "sha256": h.hexdigest()}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def apply_source_sampling_provenance(motion, attrs):
    """Downscope direct observations without changing geometry or computability."""
    if "resample timeline" not in str(attrs.get("timestamp_clock", "")):
        raise MotionContractError("source sampling semantics must be explicitly reviewed")
    cv = motion["controller_pose_observed"].copy()
    motion["joint_observed_local"] = np.zeros_like(motion["joint_observed_local"])
    motion["controller_pose_observed"] = np.zeros_like(cv)
    motion["provenance_v2_controller_pose_computable"] = cv
    motion["provenance_v2_controller_direct_observation_unverified"] = cv.copy()
    valid = motion["manus_local_joint_valid"]
    motion["provenance_v2_upstream_resampled"] = valid.copy()
    motion["provenance_v2_direct_observation_unverified"] = valid.copy()
    motion["provenance_v2_exact_interpolation_support_known"] = np.zeros_like(valid)
    motion["provenance_v2_source_clock"] = np.array("PICO_SENSOR_CLOCK_RESAMPLE_TIMELINE_NOT_RAW_ARRIVAL")
    motion["provenance_v2_inferred_scope"] = np.array("ORIGINAL_INFERRED_MASK_IS_ADAPTER_ONLY_NOT_UPSTREAM_SAMPLING")
    motion["provenance_v2_status"] = np.array("DIRECT_OBSERVATION_DOWNSCOPED_NO_NUMERICAL_CHANGE")


def load_session(source_root, processed_root, session_id, expected_count):
    import h5py
    if session_id not in SESSION_IDS:
        raise MotionContractError("session outside fixed 097/098/101 read-set")
    raw = Path(source_root) / "cards_130_0916" / session_id[-3:] / "dataset.hdf5"
    processed = Path(processed_root) / "cleaned" / "playing_cards" / session_id
    camera_path, clip_path = processed / "camera_params.json", processed / "clip_manifest.json"
    camera, clip = (json.loads(p.read_text(encoding="utf-8")) for p in (camera_path, clip_path))
    source_ref = artifact(raw)
    if clip.get("acquisition_aligned_hdf5_contract", {}).get("sha256") != source_ref["sha256"]:
        raise MotionContractError("processed frame mapping does not bind this HDF5 SHA")
    with h5py.File(raw, "r") as f:
        attrs = dict(f.attrs)
        if attrs.get("schema") != "egodex_v1" or attrs.get("pose_layout") != "pos(xyz) + quat(x,y,z,w)":
            raise MotionContractError("unsupported HDF5 pose schema")
        if not bool(attrs.get("all_exported_frames_complete")):
            raise MotionContractError("source does not declare complete exported frames")
        if attrs.get("hand_frame") != "wrist-local (relative to MANUS wrist root)":
            raise MotionContractError("MANUS root domain is not wrist-local")
        if attrs.get("controller_pose_frame") != "PICO world, right-handed X-forward Y-left Z-up, before wrist calibration":
            raise MotionContractError("controller coordinate frame mismatch")
        required = ["timestamp_ns", "head_pose", "segment_id", "video_frame_idx", "source_row_idx", "video_offset_ms"]
        required += [f"{s}_{suffix}" for s in SIDES for suffix in ("controller_pose", "wrist_pose", "hand_joints", "hand_valid")]
        data = {k: f[k][:] for k in required}
    n = len(data["timestamp_ns"])
    if n != expected_count or clip["video"]["frame_count"] != n:
        raise MotionContractError("frozen frame-count drift")
    prior = json.loads(attrs["controller_to_wrist_calibration"])
    motion = compose_motion(
        head_pose=data["head_pose"],
        controller_pose=np.stack([data[f"{s}_controller_pose"] for s in SIDES], axis=1),
        manus_local=np.stack([data[f"{s}_hand_joints"] for s in SIDES], axis=1),
        hand_valid=np.stack([data[f"{s}_hand_valid"] for s in SIDES], axis=1),
        timestamp_ns=data["timestamp_ns"], frame_id=np.arange(n, dtype=np.int64),
        segment_id=data["segment_id"], calibration=camera, prior=prior,
        joint_names=json.loads(attrs["joint_names"]),
    )
    apply_source_sampling_provenance(motion, attrs)
    motion["source_video_frame_id"] = data["video_frame_idx"]
    motion["source_row_id"] = data["source_row_idx"]
    motion["source_video_offset_ms"] = data["video_offset_ms"]
    motion["display_time_s"] = np.arange(n) / float(clip["video"]["fps"])
    motion["display_fps"] = np.array(float(clip["video"]["fps"]))
    motion["session_id"] = np.array(session_id)
    saved, saved_valid = pose_matrices(np.stack([data[f"{s}_wrist_pose"] for s in SIDES], axis=1))
    good = saved_valid & motion["wrist_world_valid"]
    residual = float(np.max(np.abs(saved[good] - motion["T_world_wrist"][good]))) if good.any() else None
    if residual is not None and residual > 1e-8:
        raise MotionContractError("saved wrist cannot be reproduced from controller and saved prior")
    if artifact(raw) != source_ref:
        raise MotionContractError("HDF5 changed while reading")
    video = processed / clip["files"]["source_stereo_video"]
    if not video.resolve().is_relative_to(processed.resolve()):
        raise MotionContractError("source video escapes session")
    return motion, video, {
        "session_id": session_id, "frame_count": n,
        "source_hdf5": source_ref, "camera_params": artifact(camera_path), "clip_manifest": artifact(clip_path),
        "source_stereo_video": artifact(video),
        "source_video_contract": {"physical_eye": "left", "source_index": 1, "crop": "second horizontal half", "resize": [1280, 960], "lens_remap": False},
        "legacy_processed_camera": clip["selection"],
        "installation_prior": prior, "installation_authority": "CAPTURE_RECORDED_FIXED_PRIOR_NOT_INDEPENDENT_CALIBRATION",
        "manus_glove_calibration_provenance": json.loads(attrs.get("manus_calibration", "{}")),
        "saved_wrist_reconstruction_max_abs": residual,
        "legacy_eye_to_current_eye_baseline_mm": float(np.linalg.norm(motion["T_leftcamera_rightcamera"][:3, 3]) * 1000),
        "wrist_world_xyz_peak_to_peak_mm": {s: (np.ptp(motion["T_world_wrist"][:, i, :3, 3][motion["wrist_world_valid"][:, i]], axis=0) * 1000).tolist() if motion["wrist_world_valid"][:, i].any() else None for i, s in enumerate(SIDES)},
        "observed_manus_hand_frames": {s: 0 for s in SIDES},
        "source_valid_manus_hand_frames": {s: int(motion["manus_hand_valid"][:, i].sum()) for i, s in enumerate(SIDES)},
        "direct_observation_policy": "UNVERIFIED_UPSTREAM_RESAMPLING_NOT_SENSOR_ABSENCE",
        "head_valid_frames": int(motion["head_valid"].sum()),
        "source_clock_dt_ms": {"min": float(np.min(np.diff(motion["timestamp_ns"])) / 1e6), "median": float(np.median(np.diff(motion["timestamp_ns"])) / 1e6), "max": float(np.max(np.diff(motion["timestamp_ns"])) / 1e6)},
        "time_transition_invalid_count_excluding_first": int((~motion["time_transition_valid"][1:]).sum()),
        "source_hawor_read": False, "source_data_modified": False,
        "wrist_semantics": "controller resampled source computable, direct sample unverified; anatomical wrist position and rotation derived from unverified installation prior",
        "surface_semantics": "UNKNOWN; MANUS joint nodes are not physical fingertip pad surfaces",
        "full_robot_status": "PENDING_SHARED_ROBOT_ADAPTER",
    }


def render_diagnostic(source_video, motion, target):
    """Finite full-frame sourceIndex=1 diagnostic; deliberately not calibration proof."""
    import cv2
    if not shutil.which("ffmpeg"):
        return {"status": "SKIP_TOOL_MISSING", "tool": "ffmpeg"}
    cv2.setNumThreads(2)
    n, fps = len(motion["frame_id"]), float(motion["display_fps"])
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise MotionContractError("cannot decode source stereo")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (source_width, source_height) != (4096, 1536):
        cap.release()
        raise MotionContractError(f"source stereo shape drift: {source_width}x{source_height}")
    # No inherited OpenCV scratch, no hardware encoder and no shell.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "1280x600", "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    colors = ((255, 100, 30), (40, 60, 255))
    started = time.monotonic()
    try:
        for frame in range(n):
            ok, stereo = cap.read()
            if not ok:
                raise MotionContractError(f"source truncated at frame {frame}")
            rgb = cv2.resize(stereo[:, source_width // 2:], (640, 480), interpolation=cv2.INTER_AREA)
            overlay = rgb.copy()
            uv, visible = project_points(motion["joints_camera_m"][frame], motion["camera_K"], motion["joint_camera_valid"][frame])
            uv = uv / 2.
            for side in range(2):
                color = colors[side]
                inside = visible[side] & np.isfinite(uv[side]).all(axis=-1) & (uv[side, :, 0] >= 0) & (uv[side, :, 0] < 640) & (uv[side, :, 1] >= 0) & (uv[side, :, 1] < 480)
                for a, b in EDGES_21:
                    if inside[a] and inside[b]:
                        cv2.line(overlay, tuple(np.rint(uv[side, a]).astype(int)), tuple(np.rint(uv[side, b]).astype(int)), color, 2, cv2.LINE_AA)
                for j in np.flatnonzero(inside):
                    cv2.circle(overlay, tuple(np.rint(uv[side, j]).astype(int)), 4 if j == 0 else 2, color, -1, cv2.LINE_AA)
            panel = np.zeros((600, 1280, 3), np.uint8)
            panel[:480, :640], panel[:480, 640:] = rgb, overlay
            lines = [
                f"{str(motion['session_id'])} | frame {frame}/{n-1} | capture t={motion['timestamp_s'][frame]:.6f}s | display {frame/fps:.3f}s",
                "LEFT: physical-left RGB sourceIndex=1 | RIGHT: Controller + MANUS; blue=L red=R",
                "FACTORY K UNVERIFIED FOR ENCODED PIXELS; FIXED WRIST INSTALLATION PRIOR, NOT MEASURED WRIST",
                "OFFLINE_VISUAL / NOT_FOR_TRAINING / SURFACE UNKNOWN / FULL ROBOT NOT SHOWN IN THIS DIAGNOSTIC",
            ]
            for line, text in enumerate(lines):
                cv2.putText(panel, text, (10, 504 + 25 * line), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
            if frame in {0, n // 2, n - 1}:
                snapshot = target.parent / f"preview_{frame:05d}.png"
                if snapshot.exists() or not cv2.imwrite(str(snapshot), panel):
                    raise MotionContractError("cannot create fresh preview")
            proc.stdin.write(panel.tobytes())
        if cap.read()[0]:
            raise MotionContractError("source stereo contains unexpected extra frames")
        proc.stdin.close()
        error = proc.stderr.read().decode("utf-8", errors="replace")
        rc = proc.wait(timeout=120)
        if rc:
            raise MotionContractError(f"ffmpeg failed {rc}: {error[-2000:]}")
    finally:
        cap.release()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
    verify = cv2.VideoCapture(str(target))
    count = 0
    try:
        while True:
            ok, img = verify.read()
            if not ok:
                break
            if img.shape[:2] != (600, 1280):
                raise MotionContractError("encoded dimensions changed")
            count += 1
    finally:
        verify.release()
    if count != n:
        raise MotionContractError(f"full decode count mismatch {count}!={n}")
    return {"status": "PASS_COMPLETE_DECODE_NOT_VISUAL_ACCEPTANCE", "video": artifact(target), "decoded_frames": count, "seconds": time.monotonic() - started, "frame_mapping": "output frame i = processed stereo row i; source_video_frame_id retained in NPZ", "playback_time_authority": "DISPLAY_CFR_ONLY_NOT_CAPTURE_CLOCK"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--skip-video", action="store_true", help="Explicitly record videos not executed")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU-only operation requires CUDA_VISIBLE_DEVICES=''")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    root = args.output_root.resolve()
    project = Path(__file__).resolve().parents[3]
    # Isolated worktrees still reside physically inside canonical chaoyang.
    canonical = next((p for p in project.parents if p.name == "chaoyang"), project)
    if not root.is_relative_to(canonical / "_run" / "current"):
        parser.error("output must stay in canonical chaoyang/_run/current")
    if (root / "RESULT.json").exists():
        parser.error("immutable RESULT already exists; use a new attempt")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.monotonic()
    for session_id, count in zip(SESSION_IDS, FRAME_COUNTS):
        folder = root / session_id
        folder.mkdir(exist_ok=False)
        motion, video, provenance = load_session(args.source_root, args.processed_root, session_id, count)
        npz = folder / "CONTROLLER_MANUS_MOTION_V2.npz"
        with npz.open("xb") as f:
            np.savez_compressed(f, **motion)
        provenance["motion_npz"] = artifact(npz)
        provenance["review"] = ({"status": "NOT_EXECUTED_EXPLICIT_SKIP_VIDEO"} if args.skip_video else render_diagnostic(video, motion, folder / "PICO_MANUS_LEFT_EYE_DIAGNOSTIC_V2.mp4"))
        write_json(folder / "RESULT.json", provenance)
        results.append(artifact(folder / "RESULT.json"))
    result = {
        "schema_version": "PICO_MANUS_MOTION_V2", "status": "PASS_MOTION_ROUTING_ENGINEERING_DIAGNOSTIC_PENDING_FULL_ROBOT",
        "session_results": results, "frame_count": sum(FRAME_COUNTS), "seconds": time.monotonic() - started,
        "claims": {"MOTION_ADAPTER_COMPLETE": True, "PIPELINE_COMPLETE": False, "NUMERIC_QUALITY_PASS": False, "TRAINING_ELIGIBLE": False, "CONTROL_GROUND_TRUTH": False, "VISUAL_REVIEW_STATUS": "NOT_REVIEWED"},
        "engineering_fix": "Physical-left sourceIndex=1 receives matching left extrinsic and K; legacy tracker-left slot actually used physical-right sourceIndex=0",
        "not_claimed": ["Anatomical wrist calibration", "Encoded camera metric accuracy", "Physical contact", "Complete full Robot before shared renderer", "Manual visual acceptance"],
        "execution": {"gpu_used": False, "hawor_read": False, "sessions_102_103_read": False, "source_data_modified": False},
    }
    write_json(root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
