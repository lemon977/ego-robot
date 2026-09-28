"""Read-only HaWoR source observability diagnostic; never repairs or smooths."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

PARENT = "four_stream_completion_20260928"
TASK = PARENT + "_motion"
SESSION = "play_cards_0915_031"
OUTPUT = "M01_031_SOURCE_OBSERVABILITY_V1.json"


def _ref(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("UNSTABLE_INPUT")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(ref, repo, max_bytes=32 * 1024 * 1024):
    path = Path(ref["path"])
    if not path.is_absolute():
        path = repo / path
    if ".." in path.parts:
        raise RuntimeError("PATH_ESCAPE")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise RuntimeError("SYMLINK_INPUT")
    path = path.resolve(strict=True)
    if not path.is_relative_to(repo):
        raise RuntimeError("PATH_ESCAPE")
    if path.stat().st_size > max_bytes:
        raise RuntimeError("INPUT_TOO_LARGE")
    actual = _ref(path)
    expected = {"path": str(path), "bytes": ref["bytes"], "sha256": ref["sha256"]}
    if actual != expected:
        raise RuntimeError("INPUT_PIN_DRIFT")
    return path.read_bytes()


def _npz(ref, repo):
    with np.load(io.BytesIO(_read(ref, repo)), allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def analyze(source, roi, image_sizes, frames):
    required = {"joints_3d_camera", "joints_2d", "observed", "inferred", "predicted_valid",
                "original_frame_indices", "timestamp_ns", "anatomical_side_names"}
    if not required.issubset(source):
        raise ValueError("SOURCE_SCHEMA")
    names = source["anatomical_side_names"].tolist()
    if names != ["left", "right"]:
        raise ValueError("SOURCE_SIDE_ORDER")
    if roi["anatomical_side_names"].tolist() != names:
        raise ValueError("ROI_SIDE_ORDER")
    if source["joints_3d_camera"].shape != (2, 149, 21, 3) or source["joints_2d"].shape != (2, 149, 21, 2):
        raise ValueError("SOURCE_SHAPE")
    if roi["boxes_xyxy"].shape != (149, 2, 4) or roi["roi_valid"].shape != (149, 2):
        raise ValueError("ROI_SHAPE")
    if not np.array_equal(source["original_frame_indices"], np.arange(149)):
        raise ValueError("TIMELINE")
    if np.any(np.diff(source["timestamp_ns"]) <= 0):
        raise ValueError("TIMESTAMP_ORDER")
    rows = []
    for frame in frames:
        if frame not in image_sizes:
            raise ValueError("MISSING_PINNED_IMAGE_SIZE")
        width, height = image_sizes[frame]
        for side, name in enumerate(names):
            box = np.asarray(roi["boxes_xyxy"][frame, side], float)
            finite = bool(np.isfinite(box).all())
            bw = max(0.0, box[2] - box[0]) if finite else 0.0
            bh = max(0.0, box[3] - box[1]) if finite else 0.0
            points = np.asarray(source["joints_2d"][side, frame], float)
            inside = (np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] < width)
                      & (points[:, 1] >= 0) & (points[:, 1] < height))
            root = np.asarray(source["joints_3d_camera"][side, frame, 0], float)
            rows.append({
                "frame_id": frame, "side": name, "width_px": width, "height_px": height,
                "roi_valid": bool(roi["roi_valid"][frame, side]),
                "box_xyxy": box.tolist() if finite else None,
                "roi_width_px": bw, "roi_height_px": bh,
                "roi_area_fraction": float(bw * bh / (width * height)),
                "touches_left": finite and box[0] <= 0, "touches_top": finite and box[1] <= 0,
                "touches_right": finite and box[2] >= width, "touches_bottom": finite and box[3] >= height,
                "joints_inside_image": int(inside.sum()), "joints_total": 21,
                "root_camera_m": root.tolist() if np.isfinite(root).all() else None,
                "root_depth_m": float(root[2]) if np.isfinite(root).all() else None,
                "observed": bool(source["observed"][side, frame]),
                "inferred": bool(source["inferred"][side, frame]),
                "predicted_valid": bool(source["predicted_valid"][side, frame]),
            })
    full = []
    for side, name in enumerate(names):
        xyz = np.asarray(source["joints_3d_camera"][side, :, 0], float)
        delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        valid = np.asarray(source["predicted_valid"][side], bool) & np.isfinite(xyz).all(1)
        pairs = valid[:-1] & valid[1:]
        masked = np.where(pairs, delta, -np.inf)
        jump = None if not pairs.any() else int(np.argmax(masked)) + 1
        full.append({
            "side": name,
            "observed_frames": int(np.asarray(source["observed"][side]).sum()),
            "inferred_frames": int(np.asarray(source["inferred"][side]).sum()),
            "predicted_valid_frames": int(valid.sum()),
            "largest_consecutive_root_delta_m": None if jump is None else float(delta[jump - 1]),
            "largest_delta_from_frame": None if jump is None else jump - 1,
            "largest_delta_to_frame": jump,
        })
    return {
        "frame_rows": rows, "full_timeline": full,
        "interpretation": "OBSERVABILITY_EVIDENCE_ONLY_NO_VALIDITY_REWRITE",
        "prohibited_actions": ["NO_SMOOTHING_AS_ROOT_CAUSE_FIX", "NO_COPY_OTHER_SIDE", "NO_LONG_HOLD", "NO_HIDDEN_FRAME_REMOVAL"],
        "limitations": [
            "ROI border contact and zero projected joints diagnose image support, not independent 3D ground truth.",
            "Inference validity is reported unchanged and is not promoted to direct observation.",
        ],
    }


def guard(repo, attempt):
    repo = Path(repo).resolve(strict=True)
    if attempt != "attempt_0001":
        raise RuntimeError("ATTEMPT")
    index = json.loads((repo / "tasks/current/INDEX.json").read_text())["task_packets"]
    entries = {item["task_id"]: item for item in index}
    if not entries[PARENT]["execution_allowed"]:
        raise RuntimeError("PARENT_NOT_ROUTABLE")
    packets = {}
    for task in (PARENT, TASK):
        path = repo / entries[task]["packet_path"]
        if _ref(path)["sha256"] != entries[task]["packet_sha256"]:
            raise RuntimeError("PACKET_DRIFT")
        packets[task] = json.loads(path.read_text())
    if packets[PARENT]["writer"] != packets[TASK]["writer"]:
        raise RuntimeError("WRITER_DELEGATION")
    writer = packets[PARENT]["writer"]
    stat = Path(f"/proc/{writer['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    if int(stat[19]) != writer["proc_start_ticks"]:
        raise RuntimeError("WRITER_FENCE")
    lane = repo / "_run/current" / PARENT / "attempts" / attempt / "lanes/motion"
    if lane.resolve() != Path(packets[TASK]["output_root"]).resolve() or str(lane.relative_to(repo)) not in packets[TASK]["write_set"]:
        raise RuntimeError("WRITE_SET")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY")
    return lane, writer, index


def run(config):
    repo = Path(config["repo_root"]).resolve(strict=True)
    lane, writer, index = guard(repo, config["attempt"])
    if config.get("session_id") != SESSION or config.get("frames") != list(range(46, 61)):
        raise RuntimeError("FIXED_SCOPE")
    source = _npz(config["source"], repo)
    roi = _npz(config["roi"], repo)
    sizes = {}
    for item in config["frame_images"]:
        frame = item["frame_id"]
        raw = _read(item["ref"], repo, 4 * 1024 * 1024)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            sizes[frame] = image.size
    if set(sizes) != set(config["frames"]):
        raise RuntimeError("IMAGE_SCOPE")
    diagnostic = analyze(source, roi, sizes, config["frames"])
    result = {
        "schema_version": "M01_SOURCE_OBSERVABILITY_V1", "task_id": TASK, "session_id": SESSION,
        "execution": "EXECUTED", "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED", "gpu_used": False,
        "timeline_frame_count": 149, "fixed_diagnostic_frames": config["frames"], "inputs": config, "diagnostic": diagnostic,
    }
    if json.loads((repo / "tasks/current/INDEX.json").read_text())["task_packets"] != index:
        raise RuntimeError("ROUTING_CHANGED")
    for key in ("source", "roi"):
        _read(config[key], repo)
    for item in config["frame_images"]:
        _read(item["ref"], repo, 4 * 1024 * 1024)
    stat = Path(f"/proc/{writer['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    if int(stat[19]) != writer["proc_start_ticks"]:
        raise RuntimeError("WRITER_CHANGED")
    output = lane / OUTPUT
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.config.read_text()))
    print(json.dumps({"execution": result["execution"], "quality": result["quality"], "output": OUTPUT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
