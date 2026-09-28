"""One Poker input-repair Clean candidate using frozen ProPainter weights/config."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.v5_scene import composite_clean

TASK = "human_to_robot_result_breakthrough_20260924"
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/clean/POKER_076_091_INPUT_REPAIR"
OLD = REPO_ROOT / ("_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/"
                   "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN")
AUDIT = REPO_ROOT / ("_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/"
                     "lanes/scene/POKER_ACCEPTED_MASK_VS_CURRENT_AUDIT.json")
VENDOR = REPO_ROOT / "vendor/ProPainter"
WINDOW = range(76, 92)
COMPLAINT_REGIONS = {"left_wrist_band": (170, 680, 410, 910),
                     "right_wrist_band": (960, 670, 1270, 940)}


def _image(path: Path, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise RuntimeError(f"IMAGE_UNREADABLE:{path}")
    return image


def _save(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"IMAGE_WRITE_FAILED:{path}")


def _registered():
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    if packet["task_id"] != TASK:
        raise RuntimeError("TASK_NOT_REGISTERED")


def prepare():
    _registered()
    if OUT.exists():
        raise FileExistsError(OUT)
    old = load_json(OLD / "INPUT.json")
    audit = load_json(AUDIT)
    if old["source_frame_range_inclusive"] != [76, 91] or audit["frame_count"] != 171:
        raise RuntimeError("FROZEN_COMPARISON_DRIFT")
    for folder in ("frames", "model_masks", "write", "protect", "unknown"):
        (OUT / folder).mkdir(parents=True, exist_ok=False)
    rows = []
    for local, frame in enumerate(WINDOW):
        previous = old["rows"][local]
        historical = audit["rows"][frame]
        if previous["source_frame"] != frame or historical["frame_id"] != frame:
            raise RuntimeError("FRAME_MAP_DRIFT")
        raw = _image(Path(previous["raw"]["path"]), cv2.IMREAD_COLOR)
        source_frame = _image(Path(previous["model_frame"]["path"]), cv2.IMREAD_COLOR)
        old_write = _image(Path(previous["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        historical_write = _image(Path(historical["historical_support"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        visible_card = _image(Path(previous["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        if raw.shape != (960, 1280, 3) or source_frame.shape != (720, 960, 3):
            raise RuntimeError("IMAGE_DOMAIN_DRIFT")
        # Card evidence belongs to this frame only. A 5px boundary is UNKNOWN,
        # not a permanently protected region propagated from another frame.
        inner = visible_card
        expanded = cv2.dilate(inner.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        unknown = expanded & ~inner & (old_write | historical_write)
        write = (old_write | historical_write) & ~inner & ~unknown
        if np.any(write & inner) or not np.any(write):
            raise RuntimeError("PROTECT_WRITE_CONFLICT")
        model = cv2.resize(write.astype(np.uint8), (960, 720), interpolation=cv2.INTER_NEAREST) > 0
        paths = {key: OUT / key / f"{local:06d}.png"
                 for key in ("frames", "model_masks", "write", "protect", "unknown")}
        for key, pixels in (("frames", source_frame), ("model_masks", model.astype(np.uint8) * 255),
                            ("write", write.astype(np.uint8) * 255),
                            ("protect", inner.astype(np.uint8) * 255),
                            ("unknown", unknown.astype(np.uint8) * 255)):
            _save(paths[key], pixels)
        complaint = {}
        for name, (x0, y0, x1, y1) in COMPLAINT_REGIONS.items():
            old_region = old_write[y0:y1, x0:x1]
            new_region = write[y0:y1, x0:x1]
            complaint[name] = {"old_write": int(old_region.sum()),
                               "new_write": int(new_region.sum()),
                               "newly_covered": int((new_region & ~old_region).sum())}
        rows.append({"frame_id": frame, "local_index": local,
                     "raw": previous["raw"], "historical_support": historical["historical_support"],
                     "old_input": artifact_ref(OLD / "INPUT.json"),
                     **{key: artifact_ref(value) for key, value in paths.items()},
                     "new_write_px": int(write.sum()), "old_write_px": int(old_write.sum()),
                     "unknown_px": int(unknown.sum()), "complaint_regions": complaint})
    changed = sum(row["new_write_px"] - row["old_write_px"] for row in rows)
    complaint_added = sum(sum(item["newly_covered"] for item in row["complaint_regions"].values()) for row in rows)
    if changed <= 0 or complaint_added <= 0:
        raise RuntimeError("NO_PROVEN_COMPLAINT_SUPPORT_CHANGE")
    recipe = {"schema_version": "POKER_CLEAN_INPUT_REPAIR_V1", "task_id": TASK,
              "session_id": "play_cards_0902_042", "frame_range": [76, 91],
              "old_candidate": artifact_ref(OLD / "RESULT.json"), "audit": artifact_ref(AUDIT),
              "change": "old_current_remove_union_historical_full_body_support_with_current_frame_card_protect",
              "input_scope_limit": "historical geometry matches session/frame but source RGB differs up to 3 intensity levels; not a pure single-variable causal proof",
              "model_parameters": {"width": 960, "height": 720, "mask_dilation": 0,
                                   "ref_stride": 10, "neighbor_length": 10,
                                   "subvideo_length": 80, "raft_iter": 20, "fp16": True},
              "rows": rows, "newly_covered_complaint_px": complaint_added,
              "new_write_minus_old_px": changed, "quality": "INPUT_ONLY_NOT_CLEAN_PASS"}
    atomic_json(OUT / "INPUT.json", recipe)
    return {"input": str(OUT / "INPUT.json"), "frames": len(rows),
            "newly_covered_complaint_px": complaint_added}


def infer():
    _registered()
    recipe = load_json(OUT / "INPUT.json")
    if (OUT / "RESULT.json").exists():
        raise FileExistsError("CLEAN_RESULT_EXISTS")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:clean"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    weights = [VENDOR / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    if any(not p.is_file() for p in weights):
        raise RuntimeError("PINNED_WEIGHT_MISSING")
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(OUT / "frames"), "--mask", str(OUT / "model_masks"),
               "--output", str(OUT / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    atomic_json(OUT / "INVOCATION.json", {"task_id": TASK, "input": artifact_ref(OUT / "INPUT.json"),
                "weights": [artifact_ref(p) for p in weights], "code": artifact_ref(VENDOR / "inference_propainter.py"),
                "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                "lease_token": lease["fencing_token"]})
    with (OUT / "PROPAINTER.log").open("xb") as log:
        done = subprocess.run(command, cwd=VENDOR,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                   "PYTHONDONTWRITEBYTECODE": "1"},
                              stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"PROPAINTER_FAILED:{done.returncode}")
    generated = sorted((OUT / "upstream/frames/frames").glob("*.png"))
    if len(generated) != len(recipe["rows"]):
        raise RuntimeError("MODEL_OUTPUT_FRAME_COVERAGE")
    (OUT / "clean").mkdir()
    rows = []
    for source, prediction in zip(recipe["rows"], generated, strict=True):
        raw = _image(Path(source["raw"]["path"]), cv2.IMREAD_COLOR)
        fake = cv2.resize(_image(prediction, cv2.IMREAD_COLOR), (1280, 960), interpolation=cv2.INTER_LINEAR)
        write = _image(Path(source["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        protect = _image(Path(source["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        clean = composite_clean(raw, fake, write, protect)
        changed = np.any(clean != raw, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError("WRITE_BOUNDARY_VIOLATION")
        path = OUT / "clean" / f'{source["frame_id"]:06d}.png'
        _save(path, clean)
        rows.append({"frame_id": source["frame_id"], "clean": artifact_ref(path),
                     "changed_px": int(changed.sum()), "outside_write_changed_px": 0,
                     "protected_changed_px": 0})
    atomic_json(OUT / "RESULT.json", {"schema_version": "POKER_CLEAN_INPUT_REPAIR_RESULT_V1",
                "task_id": TASK, "execution": "REAL_PROPAINTER_SINGLE_CANDIDATE",
                "structure": "PASS", "quality": "PENDING_INDEPENDENT_REVIEW",
                "adoption": "NOT_ADOPTED", "input": artifact_ref(OUT / "INPUT.json"),
                "invocation": artifact_ref(OUT / "INVOCATION.json"),
                "model_log": artifact_ref(OUT / "PROPAINTER.log"), "rows": rows,
                "actual_mask_consumption": "vendor read_mask loads each local PNG; selected neighbor/reference indices require invocation trace review",
                "claim_limit": "16-frame candidate only; no full-session or product pass"})
    return {"result": str(OUT / "RESULT.json"), "frames": len(rows)}


def review():
    result = load_json(OUT / "RESULT.json")
    recipe = load_json(OUT / "INPUT.json")
    if (OUT / "VISUAL_REVIEW.json").exists():
        raise FileExistsError("REVIEW_EXISTS")
    frames = OUT / "review_frames"
    frames.mkdir()
    for local, row in enumerate(recipe["rows"]):
        raw = _image(Path(row["raw"]["path"]), cv2.IMREAD_COLOR)
        old = _image(OLD / "clean" / f'{row["frame_id"]:06d}.png', cv2.IMREAD_COLOR)
        new = _image(OUT / "clean" / f'{row["frame_id"]:06d}.png', cv2.IMREAD_COLOR)
        mask = _image(Path(row["write"]["path"]), cv2.IMREAD_GRAYSCALE)
        support = raw.copy()
        support[mask > 0] = (.45 * support[mask > 0] + .55 * np.array([0, 0, 255])).astype(np.uint8)
        tiles = [cv2.resize(x, (640, 480), interpolation=cv2.INTER_AREA)
                 for x in (raw, old, support, new)]
        canvas = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        for x, y, label in ((10, 28, "RAW"), (650, 28, "OLD REJECTED CLEAN"),
                            (10, 508, "NEW DELETE SUPPORT"), (650, 508, "NEW CANDIDATE")):
            cv2.rectangle(canvas, (x - 4, y - 22), (x + 265, y + 8), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .65,
                        (255, 255, 255), 2)
        _save(frames / f"{local:06d}.png", canvas)
    video = OUT / "POKER_076_091_OLD_NEW_CLEAN_REVIEW.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-framerate", "30", "-i", str(frames / "%06d.png"), "-an",
               "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "19",
               "-pix_fmt", "yuv420p", str(video)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("REVIEW_ENCODING_FAILED:" + done.stderr[-1000:])
    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != 16:
        raise RuntimeError("REVIEW_FRAME_COUNT")
    record = {"schema_version": "POKER_CLEAN_INPUT_REPAIR_REVIEW_V1", "task_id": TASK,
              "candidate": artifact_ref(OUT / "RESULT.json"), "video": artifact_ref(video),
              "decoded_frames": count, "reviewed_fixed_frames": [76, 80, 91],
              "observations": ["left forearm removal visibly improved in fixed frames",
                               "right hand remains as a large skin-colored blob on the card/table edge",
                               "card boundary and inpaint texture remain visibly invalid"],
              "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
              "full_171": "NOT_RUN_FIXED_WINDOW_FAILED",
              "review_limit": "three fixed-frame human-visible checks plus complete 16-frame synchronized video; not a full-session visual pass"}
    atomic_json(OUT / "VISUAL_REVIEW.json", record)
    return {"video": str(video), "quality": record["quality"], "full_171": record["full_171"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "review"])
    args = parser.parse_args()
    action = {"prepare": prepare, "infer": infer, "review": review}[args.stage]
    print(json.dumps(action(), ensure_ascii=False))


if __name__ == "__main__":
    main()
