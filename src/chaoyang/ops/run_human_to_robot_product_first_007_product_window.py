"""Render 007 old/new Clean through one fixed q/FK renderer, diagnostic only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK, RAW, OLD_PREP
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/product_007/window_v1"
NEW = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/clean_window_v1"
MOTION = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_007_wave0/ROBOT_R0_V1.npz"
PREP = OLD_PREP.parent / "SCENE_PREP_MANIFEST.json"


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    clean = load_json(NEW / "RESULT.json")
    if clean["source_frames"] != list(range(181, 197)):
        raise ValueError("CLEAN_WINDOW_DRIFT")
    with np.load(MOTION, allow_pickle=False) as archive:
        motion = {name: np.asarray(archive[name]) for name in archive.files}
    domain = load_json(Path(load_json(PREP)["source_domain"]["path"]))
    renderer = ProductRobotRenderer(ROOT, motion, domain, include_adapter=True)
    OUT.mkdir(parents=True)
    video = OUT / "007_OLD_NEW_CLEAN_ROBOT_WINDOW.mp4"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x480", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    rows = []
    try:
        for local, frame in enumerate(range(181, 197)):
            raw = _image(RAW / f"{frame:06d}.png")
            old_clean = _image(OLD_PREP.parent / "clean" / f"{frame:06d}.png")
            new_clean = _image(NEW / "clean" / f"{local:06d}.png")
            layer = renderer.frame_layers(frame)
            old_product = composite(old_clean, layer.rgb, layer.alpha)
            new_product = composite(new_clean, layer.rgb, layer.alpha)
            panels = [cv2.resize(value, (640, 480), interpolation=cv2.INTER_AREA)
                      for value in (raw, old_product, new_product)]
            canvas = np.concatenate(panels, axis=1)
            for x, title in ((8, "RAW"), (648, "OLD CLEAN + SAME ROBOT"), (1288, "NEW CLEAN + SAME ROBOT")):
                cv2.putText(canvas, title, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.putText(canvas, f"007 frame {frame} | OCCLUSION UNKNOWN | NOT ADOPTED", (8, 462),
                        cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 2)
            proc.stdin.write(canvas.tobytes())
            old_to_new = np.any(old_product != new_product, axis=2)
            rows.append({"source_frame_id": frame, "robot_pixels": int(layer.alpha.sum()),
                         "old_new_product_changed_pixels": int(old_to_new.sum()),
                         "clean_old_new_changed_pixels": int(np.any(old_clean != new_clean, axis=2).sum()),
                         "T_world_hand": layer.T_world_hand.tolist(),
                         "occlusion": "UNKNOWN_NO_007_STEREO_QUALIFICATION"})
        proc.stdin.close()
        errors = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ENCODE_FAILED:{errors[-1000:]}")
    finally:
        renderer.close()
        if proc.poll() is None:
            proc.terminate(); proc.wait()
    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != 16:
        raise ValueError(f"VIDEO_DECODE:{count}")
    result = {"schema_version": "HUMAN_TO_ROBOT_007_PRODUCT_FIRST_WINDOW_V1",
              "task_id": TASK, "session_id": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
              "execution": "ACTUAL_RENDERER_CONSUMED_NEW_CLEAN", "structure": "PASS",
              "quality": "REJECTED_QUALITY_FULL_SESSION_NOT_PROVEN", "adoption": "CANDIDATE_ONLY",
              "frozen_variables": ["q", "camera", "mount", "CAD", "renderer", "frame_set"],
              "only_change": "OLD_VS_NEW_CLEAN_BACKGROUND", "rows": rows,
              "video": {"path": str(video), "sha256": _sha(video), "decoded_frames": count},
              "inputs": {"new_clean": artifact_ref(NEW / "RESULT.json"), "old_clean": artifact_ref(OLD_PREP.parent / "RESULT.json"),
                         "motion": artifact_ref(MOTION), "prep": artifact_ref(PREP)},
              "claim_limit": "Actual fixed-16-frame Robot product candidate with 007 occlusion UNKNOWN. Not 378-frame Clean or product quality/adoption.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "decoded": count,
                      "changed_product_px": sum(row["old_new_product_changed_pixels"] for row in rows),
                      "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
