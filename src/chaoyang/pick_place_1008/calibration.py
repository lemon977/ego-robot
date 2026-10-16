"""Development-only, raw-source projection comparisons; no fitting or selection.

Each native picture is decoded sequentially. Grid candidates vary physical-eye
calibration, SBS crop and upright basis independently. Only raw-composed valid
hand samples are drawn. Results are visual assumptions, not calibration truth.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from chaoyang.pipeline.pico_manus_motion_v2 import EDGES_21
from .camera import normalize_camera, sha256
from .source import (
    SourceContractError, _project_output, read_json, json_rows, timeline,
    read_installation, load_raw_motion, ref,
)

DEVELOPMENT_SESSIONS = ("001", "003", "008", "100")


def camera_candidates(params: dict, meta: dict, snapshot_sha: str) -> list[dict]:
    result = []
    for eye in ("left", "right"):
        for index in (0, 1):
            for upright in ("identity", "capture_sensor_from_upright"):
                result.append({"candidate_id": f"{eye}_sbs{index}_{upright}",
                               "device_serial": meta["device_serial"], "camera_sha256": snapshot_sha,
                               "eye": eye, "sbs_index": index, "upright_axes": upright,
                               "extrinsic_convention": "head_to_camera", "image_mode": "encoded_crop_resize_only",
                               "status": "DEVELOPMENT_CANDIDATE", "output_size": [1280, 960]})
    return result


def _manifest_rows(manifest):
    data = read_json(Path(manifest)) if isinstance(manifest, (str, Path)) else manifest
    rows = data.get("sessions", data.get("recordings", []))
    if not isinstance(rows, list):
        raise SourceContractError("PROJECTION_MANIFEST_SESSION_LIST_REQUIRED")
    return {str(row["session_id"]).rsplit("_", 1)[-1]: row for row in rows}


def _native_frames(path: Path, selected: list[int]) -> dict[int, np.ndarray]:
    import av
    output = {}
    needed = set(selected)
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(video=0)):
            if index in needed:
                output[index] = frame.to_ndarray(format="bgr24")
            if index >= max(needed):
                break
    if set(output) != needed:
        raise SourceContractError("PROJECTION_NATIVE_FRAME_DECODE_INCOMPLETE")
    return output


def _candidate_panel(stereo, camera, native, frame: int):
    import cv2
    x, y, w, h = camera.crop
    if stereo.shape[:2] != (h, 2*w):
        raise SourceContractError("PROJECTION_SBS_DOMAIN_MISMATCH")
    raw = cv2.resize(stereo[y:y+h, x:x+w], (640, 480), interpolation=cv2.INTER_AREA)
    panel = np.zeros((550, 640, 3), np.uint8)
    panel[:480] = raw
    stats = {"raw_hand_valid": native["hand_observed"][:, frame].tolist(),
             "head_valid": bool(native["head_valid"][frame]), "finite_joints": [0, 0],
             "positive_depth_joints": [0, 0], "in_view_joints": [0, 0]}
    if stats["head_valid"]:
        w2c = np.linalg.inv(native["head"][frame] @ camera.T_head_camera)
        points = native["hands"][:, frame] @ w2c[:3, :3].T + w2c[:3, 3]
        k = camera.K.copy()
        k[:2] *= .5
        for side, color in enumerate(((255, 100, 30), (40, 50, 255))):
            if not stats["raw_hand_valid"][side]:
                continue
            finite = np.isfinite(points[side]).all(axis=-1)
            front = finite & (points[side, :, 2] > 1e-6)
            uv = np.full((21, 2), np.nan)
            hpoints = points[side] @ k.T
            uv[front] = hpoints[front, :2] / hpoints[front, 2, None]
            inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
            stats["finite_joints"][side] = int(finite.sum())
            stats["positive_depth_joints"][side] = int(front.sum())
            stats["in_view_joints"][side] = int(inside.sum())
            stats.setdefault("projected_uv_640x480", {})[("left", "right")[side]] = [v.tolist() if np.isfinite(v).all() else None for v in uv]
            for a, b in EDGES_21:
                if inside[a] and inside[b]:
                    cv2.line(panel, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 2, cv2.LINE_AA)
            for joint in np.flatnonzero(inside):
                cv2.circle(panel, tuple(np.rint(uv[joint]).astype(int)), 4 if joint == 0 else 2, color, -1, cv2.LINE_AA)
    lines = [f"{camera.eye} | SBS {camera.sbs_index} | {camera.calibration['upright_axes']}",
             f"rawvalid L/R={stats['raw_hand_valid']} in-view={stats['in_view_joints']}",
             "RAW Controller+MANUS | installation prior | factory K | visual unverified"]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 497 + i*21), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
    return panel, stats


def run_projection_audit(source_root: Path, output_root: Path, manifest,
                         dev_sessions=DEVELOPMENT_SESSIONS) -> dict:
    import cv2
    cv2.setNumThreads(1)
    sessions = tuple(str(s).rsplit("_", 1)[-1] for s in dev_sessions)
    if not sessions or len(sessions) > 4 or any(s not in DEVELOPMENT_SESSIONS for s in sessions):
        raise SourceContractError("PROJECTION_FIXED_DEVELOPMENT_READSET")
    rows = _manifest_rows(manifest)
    source_root, output_root = Path(source_root).resolve(strict=True), _project_output(Path(output_root))
    output_root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "pick-place-development-projection-audit-v1",
              "status": "PROJECTION_AUDIT_COMPLETE", "qualification": "CANDIDATE_COMPARISON_NO_FIT_OR_SELECTION", "sessions": [],
              "development_sessions": list(sessions), "holdout_used": False,
              "lens_remap": False, "training_eligible": False, "control_ground_truth": False}
    for short_id in sessions:
        row = rows.get(short_id)
        if row is None:
            raise SourceContractError(f"PROJECTION_SESSION_NOT_IN_MANIFEST:{short_id}")
        source = Path(row.get("source_dir", source_root/short_id)).resolve(strict=True)
        if source != source_root/short_id:
            raise SourceContractError("PROJECTION_SOURCE_ROOT_IDENTITY_MISMATCH")
        pins = row.get("source_refs", row.get("inputs", []))
        pins = list(pins.values()) if isinstance(pins, dict) else pins
        for pin in pins:
            if isinstance(pin, str):
                continue
            if "path" in pin and ref(Path(pin["path"])) != {k: pin[k] for k in ("path", "bytes", "sha256")}:
                raise SourceContractError("PROJECTION_INPUT_SHA_DRIFT")
        params = read_json(source/"raw/camera_params.json")
        meta = read_json(source/"raw/camera_params.meta.json")
        sha = sha256(source/"raw/camera_params.json")
        serial = meta["device_serial"]
        if row.get("device_serial", serial) != serial:
            raise SourceContractError("PROJECTION_MANIFEST_DEVICE_MISMATCH")
        qpc = timeline([v for _, v in json_rows(source/"raw/vst.qpc.ts.jsonl")], "video")
        if row.get("raw_frame_count", row.get("expected_frames", len(qpc))) != len(qpc):
            raise SourceContractError("PROJECTION_MANIFEST_FRAME_COUNT_DRIFT")
        prior, prior_ref = read_installation(source)
        native, mapping, quality = load_raw_motion(source, qpc, expected_serial=serial, prior=prior)
        selected = sorted(set([0, (len(qpc)-1)//4, (len(qpc)-1)//2, 3*(len(qpc)-1)//4, len(qpc)-1]))
        frames = _native_frames(source/"raw/vst.h264", selected)
        candidates = camera_candidates(params, meta, sha)
        cameras = [normalize_camera(params, meta, device_serial=serial, camera_sha256=sha, config=c) for c in candidates]
        folder = output_root/short_id
        folder.mkdir()
        session_result = {"session_id": row["session_id"], "device_serial": serial,
                          "camera_snapshot": ref(source/"raw/camera_params.json"),
                          "installation": prior_ref, "installation_prior_used": prior,
                          "source_motion": "RAW_PICO_CONTROLLER_PLUS_RAW_MANUS",
                          "time_quality": quality, "raw_frames": len(qpc),
                          "selected_native_frame_ids": selected, "frames": []}
        for frame in selected:
            panels, comparisons = [], []
            for c, camera in zip(candidates, cameras):
                panel, stats = _candidate_panel(frames[frame], camera, native, frame)
                panels.append(panel)
                comparisons.append({"candidate": c, "camera": camera.calibration, "projection": stats})
            grid = np.concatenate([np.concatenate(panels[:4], axis=1), np.concatenate(panels[4:], axis=1)], axis=0)
            filename = folder/f"frame_{frame:05d}_eye_route_grid.png"
            if not cv2.imwrite(str(filename), grid):
                raise SourceContractError("PROJECTION_PNG_WRITE_FAILED")
            session_result["frames"].append({"native_frame_id": frame, "video_qpc_ns": int(qpc[frame]),
                                             "pico_source_line": int(mapping["pico_source_line"][frame]),
                                             "manus_left_source_line": int(mapping["manus_left_source_line"][frame]),
                                             "manus_right_source_line": int(mapping["manus_right_source_line"][frame]),
                                             "grid": ref(filename), "comparisons": comparisons})
        with (folder/"PROJECTION_COMPARISON.json").open("x", encoding="utf-8") as stream:
            json.dump(session_result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        result["sessions"].append({"session_id": row["session_id"], "result": ref(folder/"PROJECTION_COMPARISON.json"),
                                   "grid_paths": [f["grid"]["path"] for f in session_result["frames"]]})
    with (output_root/"RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result
