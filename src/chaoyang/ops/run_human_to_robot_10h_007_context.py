"""One frozen-center 007 ProPainter context candidate (160..216)."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso, validate_artifact_ref
from chaoyang.pipeline.v5_scene import composite_clean
from chaoyang.pipeline.robot_renderer_cycles import RendererError, forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

TASK = "human_to_robot_10h_delivery_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/context_007_v1"
BASELINE = REPO_ROOT / "_run/current/human_to_robot_007_attachment_clean_canary_20260923/attempts/attempt_0001/lanes/scene/attachment_clean_window_v1"
R2 = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_007_wave5"
RAW_FULL = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"
VENDOR = REPO_ROOT / "vendor/ProPainter"
START, STOP = 160, 216
EVAL_START, EVAL_STOP = 181, 196
KINDS = ("frames", "model_masks", "write", "protect")


def full_windows() -> list[tuple[int, int, int, int]]:
    """Return (output first,last,input first,last), inclusive, with 181..196 intact."""
    outputs = [(0, 4)] + [(start, min(start + 15, 377)) for start in range(5, 378, 16)]
    if [frame for first, last in outputs for frame in range(first, last + 1)] != list(range(378)):
        raise RuntimeError("FULL_WINDOW_COVERAGE")
    if (181, 196) not in outputs:
        raise RuntimeError("EVALUATION_WINDOW_NOT_PRESERVED")
    return [(first, last, max(0, first - 21), min(377, last + 20)) for first, last in outputs]


def model_reference_schedule(length: int = 57) -> list[dict]:
    """Mirror the pinned vendor get_ref_index/neighbor loop without altering selection."""
    rows = []
    stride = 5  # frozen neighbor_length=10
    for mid in range(0, length, stride):
        neighbors = list(range(max(0, mid - stride), min(length, mid + stride + 1)))
        refs = [i for i in range(0, length, 10) if i not in neighbors]
        rows.append({"model_mid_local_index": mid, "neighbor_local_indices": neighbors,
                     "reference_local_indices": refs,
                     "neighbor_source_frames": [START + i for i in neighbors],
                     "reference_source_frames": [START + i for i in refs]})
    return rows


def _image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _routable() -> None:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")


def _sources(frame: int) -> dict[str, Path]:
    if EVAL_START <= frame <= EVAL_STOP:
        number = frame - EVAL_START
        return {kind: BASELINE / "input" / kind / f"{number:06d}.png" for kind in KINDS}
    return {kind: R2 / "prep" / kind / f"{frame:06d}.png" for kind in KINDS}


def prepare() -> dict:
    _routable()
    target = DEST / "input"
    if DEST.exists():
        raise FileExistsError(DEST)
    baseline = load_json(BASELINE / "input/RESULT.json")
    r2 = load_json(R2 / "SCENE_PREP_MANIFEST.json")
    if baseline.get("source_frames") != list(range(EVAL_START, EVAL_STOP + 1)) or r2.get("frame_count") != 378:
        raise RuntimeError("FROZEN_FRAME_ID_MISMATCH")
    rows = []
    for frame in range(START, STOP + 1):
        source = _sources(frame)
        images = {kind: _image(path) for kind, path in source.items()}
        if (images["frames"].shape != (720, 960, 3)
                or images["model_masks"].shape != (720, 960)
                or any(images[kind].shape != (960, 1280) for kind in ("write", "protect"))):
            raise RuntimeError(f"FRAME_DOMAIN:{frame}")
        for kind in ("model_masks", "write", "protect"):
            if not set(np.unique(images[kind])).issubset({0, 255}):
                raise RuntimeError(f"NONBINARY_{kind}:{frame}")
        if np.any((images["write"] > 0) & (images["protect"] > 0)):
            raise RuntimeError(f"WRITE_PROTECT_CONFLICT:{frame}")
        if not images["model_masks"].any() or not images["write"].any():
            raise RuntimeError(f"EMPTY_REJECTED_MASK:{frame}")
        r2_raw = _image(R2 / "prep/frames" / f"{frame:06d}.png")
        if not np.array_equal(images["frames"], r2_raw):
            raise RuntimeError(f"RAW_SOURCE_DRIFT:{frame}")
        if EVAL_START <= frame <= EVAL_STOP:
            frozen_row = baseline["rows"][frame - EVAL_START]
            for kind in KINDS:
                if artifact_ref(source[kind]) != frozen_row["prepared"][kind]:
                    raise RuntimeError(f"CENTER_INPUT_DRIFT:{frame}:{kind}")
        r2_row = r2["rows"][frame]
        if int(r2_row["frame_id"]) != frame or int(r2_row["capture_time"]["source_frame"]) != frame:
            raise RuntimeError(f"CAPTURE_FRAME_DRIFT:{frame}")
        unknown = _image(R2 / "prep/unknown" / f"{frame:06d}.png")
        if unknown.shape != (960, 1280):
            raise RuntimeError(f"UNKNOWN_DOMAIN:{frame}")
        rows.append({"source_frame_id": frame, "local_index": frame - START,
                     "capture_time": r2_row["capture_time"],
                     "center_input_frozen": EVAL_START <= frame <= EVAL_STOP,
                     "input_sources": {kind: artifact_ref(path) for kind, path in source.items()},
                     "unknown_source": artifact_ref(R2 / "prep/unknown" / f"{frame:06d}.png"),
                     "unknown_pixels": int(np.count_nonzero(unknown)),
                     "model_mask_pixels": int(np.count_nonzero(images["model_masks"])),
                     "write_pixels": int(np.count_nonzero(images["write"])),
                     "protect_pixels": int(np.count_nonzero(images["protect"]))})
    target.mkdir(parents=True)
    for kind in KINDS:
        (target / kind).mkdir()
    for row in rows:
        local = row["local_index"]
        for kind, source_ref in row["input_sources"].items():
            output = target / kind / f"{local:06d}.png"
            shutil.copyfile(source_ref["path"], output)
            if artifact_ref(output)["sha256"] != source_ref["sha256"]:
                raise RuntimeError(f"COPIED_INPUT_MISMATCH:{local}:{kind}")
    schedule = model_reference_schedule()
    if not any(any(frame < EVAL_START or frame > EVAL_STOP for frame in row["reference_source_frames"])
               for row in schedule):
        raise RuntimeError("NO_NEW_CONTEXT_REFERENCES")
    result = {"schema_version": "HUMAN_TO_ROBOT_007_CONTEXT_INPUT_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "source_frames": [START, STOP],
              "evaluation_frames": [EVAL_START, EVAL_STOP], "evaluation_local_indices": [21, 36],
              "old_center_input": artifact_ref(BASELINE / "input/RESULT.json"),
              "context_source": artifact_ref(R2 / "SCENE_PREP_MANIFEST.json"),
              "rows": rows, "vendor_reference_schedule": schedule,
              "full_window_schedule": full_windows(),
              "reference_semantics": "PINNED_VENDOR_DETERMINISTIC_SELECTION_FROM_57_INPUTS",
              "unknown_semantics": "RECORDED_NOT_INTERPRETED_AS_CLEAN_BACKGROUND",
              "claim_limit": "Center pixels and masks byte-identical to old canary; new context is a model input, not a trusted hidden-scene observation."}
    path = target / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "PREPARED", "result": str(path), "rows": len(rows)}


def run_model() -> dict:
    _routable()
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != TASK + ":scene"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    source = DEST / "input/RESULT.json"
    prep = load_json(source)
    if prep.get("source_frames") != [START, STOP] or len(prep.get("rows", [])) != 57:
        raise RuntimeError("PREPARE_NOT_COMPLETE")
    if (DEST / "RESULT.json").exists() or (DEST / "upstream").exists():
        raise FileExistsError("MODEL_ATTEMPT_ALREADY_EXISTS")
    weights = [VENDOR / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(weight)
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(DEST / "input/frames"), "--mask", str(DEST / "input/model_masks"),
               "--output", str(DEST / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    invocation = DEST / "INVOCATION.json"
    invocation.write_text(json.dumps({"task_id": TASK, "started_at": datetime.now().astimezone().isoformat(),
                                      "input": artifact_ref(source), "command": command,
                                      "weights": [artifact_ref(path) for path in weights],
                                      "lease_fencing_token": lease["fencing_token"]},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log_path = DEST / "PROPAINTER.log"
    with log_path.open("xb") as log:
        run = subprocess.run(command, cwd=VENDOR, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT_{run.returncode}")
    generated = sorted((DEST / "upstream/frames/frames").glob("*.png"))
    if len(generated) != 57:
        raise RuntimeError(f"MODEL_FRAME_COUNT:{len(generated)}")
    (DEST / "clean").mkdir()
    video = DEST / "007_CONTEXT_OLD_NEW_CLEAN_16FRAME_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (2560, 480))
    if not writer.isOpened():
        raise RuntimeError("REVIEW_WRITER_NOT_OPEN")
    output_rows = []
    try:
        for frame in range(EVAL_START, EVAL_STOP + 1):
            local, old_local = frame - START, frame - EVAL_START
            raw = _image(RAW_FULL / f"{frame:06d}.png", cv2.IMREAD_COLOR)
            old = _image(BASELINE / "clean" / f"{old_local:06d}.png", cv2.IMREAD_COLOR)
            fake = _image(generated[local], cv2.IMREAD_COLOR)
            write = _image(DEST / "input/write" / f"{local:06d}.png") > 0
            protect = _image(DEST / "input/protect" / f"{local:06d}.png") > 0
            if raw.shape != (960, 1280, 3) or old.shape != raw.shape:
                raise RuntimeError(f"RAW_OR_OLD_DOMAIN:{frame}")
            clean = composite_clean(raw, cv2.resize(fake, (1280, 960)), write, protect)
            changed = np.any(clean != raw, axis=2)
            if np.any(changed & ~write) or np.any(changed & protect):
                raise RuntimeError(f"WRITE_PROTECT_VIOLATION:{frame}")
            output = DEST / "clean" / f"{old_local:06d}.png"
            if not cv2.imwrite(str(output), clean):
                raise RuntimeError(f"CLEAN_WRITE_FAILED:{frame}")
            support = raw.copy()
            support[write] = (support[write] * .55 + np.array([0, 0, 255]) * .45).astype(np.uint8)
            canvas = np.concatenate([cv2.resize(item, (640, 480)) for item in (raw, support, old, clean)], axis=1)
            for x, title in ((8, "RAW"), (648, "FROZEN WRITE"), (1288, "OLD REJECTED"), (1928, "NEW CONTEXT")):
                cv2.putText(canvas, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.putText(canvas, str(frame), (8, 460), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            writer.write(canvas)
            output_rows.append({"source_frame_id": frame, "model_local_index": local,
                                "clean": artifact_ref(output), "changed_pixels": int(changed.sum())})
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != 16:
        raise RuntimeError(f"REVIEW_DECODE:{decoded}")
    result = {"schema_version": "HUMAN_TO_ROBOT_007_CONTEXT_CLEAN_CANDIDATE_V1",
              "task_id": TASK, "session_id": "get_potato_chips_0915_007",
              "source_frames": [EVAL_START, EVAL_STOP], "execution": "EXECUTED",
              "structure": "PASS_16_FRAME", "quality": "PENDING_INDEPENDENT_REVIEW",
              "adoption": "CANDIDATE_ONLY", "input": artifact_ref(source),
              "invocation": artifact_ref(invocation), "model_log": artifact_ref(log_path),
              "review": {**artifact_ref(video), "decoded_frames": decoded}, "rows": output_rows,
              "claim_limit": "Same-center-input expanded-context synthetic Clean candidate only; no full-session or product promotion.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    output = DEST / "RESULT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "EXECUTED_CANDIDATE", "result": str(output), "review": str(video)}


def review() -> dict:
    """Independently accept/reject the frozen complaint window, not the full product."""
    _routable()
    output = DEST / "AI_VISUAL_REVIEW.json"
    if output.exists():
        raise FileExistsError(output)
    result_path = DEST / "RESULT.json"
    result = load_json(result_path)
    prepared = load_json(DEST / "input/RESULT.json")
    if result.get("structure") != "PASS_16_FRAME" or len(result.get("rows", [])) != 16:
        raise RuntimeError("CANDIDATE_STRUCTURE_NOT_READY")
    video = Path(result["review"]["path"])
    if artifact_ref(video)["sha256"] != result["review"]["sha256"]:
        raise RuntimeError("REVIEW_VIDEO_DRIFT")
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 16:
        raise RuntimeError(f"REVIEW_DECODE_DRIFT:{decoded}")
    rows = []
    for frame in range(EVAL_START, EVAL_STOP + 1):
        local = frame - EVAL_START
        raw = _image(RAW_FULL / f"{frame:06d}.png", cv2.IMREAD_COLOR)
        old = _image(BASELINE / "clean" / f"{local:06d}.png", cv2.IMREAD_COLOR)
        new_path = DEST / "clean" / f"{local:06d}.png"
        if artifact_ref(new_path) != result["rows"][local]["clean"]:
            raise RuntimeError(f"CANDIDATE_FRAME_DRIFT:{frame}")
        new = _image(new_path, cv2.IMREAD_COLOR)
        write = _image(DEST / "input/write" / f"{frame - START:06d}.png") > 0
        protect = _image(DEST / "input/protect" / f"{frame - START:06d}.png") > 0
        old_changed = np.any(old != raw, axis=2)
        new_changed = np.any(new != raw, axis=2)
        if np.any(new_changed & ~write) or np.any(new_changed & protect):
            raise RuntimeError(f"FROZEN_WRITE_OR_PROTECT_VIOLATION:{frame}")
        rows.append({"source_frame_id": frame, "write_pixels": int(write.sum()),
                     "old_changed_inside_write": int((old_changed & write).sum()),
                     "new_changed_inside_write": int((new_changed & write).sum()),
                     "old_new_different_inside_write": int((np.any(old != new, axis=2) & write).sum()),
                     "new_changed_outside_write": int((new_changed & ~write).sum()),
                     "new_changed_protected": int((new_changed & protect).sum())})
    used = []
    for step in prepared["vendor_reference_schedule"]:
        if 21 <= step["model_mid_local_index"] <= 36:
            used.append({"model_mid_local_index": step["model_mid_local_index"],
                         "neighbor_source_frames": step["neighbor_source_frames"],
                         "reference_source_frames": step["reference_source_frames"],
                         "reference_mask_sha256": [prepared["rows"][i - START]["input_sources"]["model_masks"]["sha256"]
                                                   for i in step["reference_source_frames"]]})
    payload = {
        "schema_version": "HUMAN_TO_ROBOT_007_CONTEXT_AI_REVIEW_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "review_authority": "AI_REVIEW_PROXY_NOT_HUMAN_GT",
        "reviewed_source_frames": [181, 184, 190, 193, 196], "full_16_frame_video_decoded": decoded,
        "candidate": artifact_ref(result_path), "input": artifact_ref(DEST / "input/RESULT.json"),
        "code": artifact_ref(Path(__file__)), "vendor_code": artifact_ref(VENDOR / "inference_propainter.py"),
        "model_execution_governance_revision": 14066,
        "model_execution_code_note": "Review-only function was appended after model exit; model invocation and pinned vendor/config/inputs are unchanged.",
        "video": artifact_ref(video), "model_execution_receipt": artifact_ref(DEST / "PROPAINTER_GPU_RECEIPT.json"),
        "effective_center_neighbor_reference_trace": used,
        "reference_trace_authority": "DETERMINISTIC_PINNED_VENDOR_CODE_AND_57_INPUTS_NOT_RUNTIME_INSTRUMENTED",
        "pixel_checks": rows, "execution": "EXECUTED", "structure": "PASS_16_FRAME",
        "quality": "REJECTED_QUALITY", "improvement": "NOT_ESTABLISHED",
        "adoption": "NOT_ADOPTED", "full_session_expansion_allowed": False,
        "observations": [
            "The same raw complaint region was compared against the old rejected canary and the new output at the five declared frames.",
            "A conspicuous left forearm-shaped brown ghost and attachment/yellow cable remnants remain at every reviewed frame; the right-hand/bowl region also retains yellow and grey hand-shaped residue.",
            "The new 57-frame context changes pixels but does not resolve the fixed complaint. At frames 184 and 190 the new right-hand region is visibly more hand-shaped than the old rejected output.",
            "The white line on the left remains visible; the quality rejection does not rely on file existence or a global pixel-difference score."
        ],
        "limits": ["Five AI-reviewed frames plus full video decode; not human inspection of all sixteen frames.",
                   "New context effects cannot be generalized beyond this input; the extra masks differ from center masks by source and remain development-only.",
                   "No 378-frame expansion, product adoption, geometry, Contact, training or metric authority."],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "REJECTED_QUALITY_NO_FULL_EXPANSION", "result": str(output)}


MOTION_SESSIONS = {
    "007": ("get_potato_chips_0915_007", 378, "formal_product_007/attempt_0002"),
    "031": ("play_cards_0915_031", 149, "formal_product_031/attempt_0004"),
}
MANO_EDGES = [(0, first) for first in (1, 5, 9, 13, 17)] + [
    (joint, joint + 1) for first in (1, 5, 9, 13, 17) for joint in range(first, first + 3)
]


def _project_hand(image: np.ndarray, joints: np.ndarray, valid: np.ndarray, k: np.ndarray,
                  color: tuple[int, int, int]) -> None:
    points: list[tuple[int, int] | None] = []
    for xyz, usable in zip(joints, valid):
        if not usable or not np.isfinite(xyz).all() or xyz[2] <= 1e-4:
            points.append(None)
            continue
        uv = k @ xyz
        x, y = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
        points.append((x, y) if -10000 < x < 10000 and -10000 < y < 10000 else None)
    for a, b in MANO_EDGES:
        if points[a] is None or points[b] is None:
            continue
        ok, clipped_a, clipped_b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), points[a], points[b])
        if ok:
            cv2.line(image, clipped_a, clipped_b, color, 2, cv2.LINE_AA)
    for point in points:
        if point is not None and 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)


def motion_review(short: str) -> dict:
    """Render existing Raw, actual HandMotion, and formal q-bound product together."""
    _routable()
    session, frames, product_part = MOTION_SESSIONS[short]
    out = ATTEMPT / "lanes/motion_product" / f"full_chain_{short}"
    if out.exists():
        raise FileExistsError(out)
    motion_root = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/"
                               f"lanes/lane2_motion/recovered_{short}_wave0")
    raw_root = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                            f"lanes/exact78/prepare_full_v1/{session}/raw")
    product_root = REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/"
                                f"attempt_0001/lanes/motion_product/{product_part}")
    motion_path, robot_path = motion_root / "HAND_MOTION_V1.npz", motion_root / "ROBOT_R0_V1.npz"
    product_result_path, product_video = product_root / "PRODUCT_RESULT.json", product_root / "robot.mp4"
    product_result = load_json(product_result_path)
    if (product_result["session_id"] != session or product_result["expected_frames"] != frames
            or product_result["robot_r0"] != artifact_ref(robot_path)):
        raise RuntimeError(f"FORMAL_PRODUCT_SOURCE_MISMATCH:{short}")
    with np.load(motion_path, allow_pickle=False) as source, np.load(robot_path, allow_pickle=False) as robot:
        frame_id = source["frame_id"].copy()
        timestamp_ns = source["timestamp_ns"].copy()
        joints = source["joints21_camera"].copy()
        valid = source["joint_valid"].copy()
        intrinsics = source["intrinsics"].copy()
        robot_frame_id = robot["frame_id"].copy()
        wrist_valid = robot["wrist_valid"].copy()
        position_mm = robot["position_residual_mm"].copy()
        rotation_deg = robot["rotation_residual_deg"].copy()
    if (frame_id.tolist() != list(range(frames)) or not np.array_equal(frame_id, robot_frame_id)
            or joints.shape != (frames, 2, 21, 3) or valid.shape != (frames, 2, 21)
            or wrist_valid.shape != (frames, 2) or position_mm.shape != (frames, 2)):
        raise RuntimeError(f"MOTION_TIMELINE_OR_SHAPE:{short}")
    capture = cv2.VideoCapture(str(product_video))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
        capture.release()
        raise RuntimeError(f"PRODUCT_VIDEO_FRAME_COUNT:{short}")
    out.mkdir(parents=True)
    video = out / f"{session}_RAW_MOTION_FORMAL_PRODUCT_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 600))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("MOTION_REVIEW_WRITER")
    rows = []
    try:
        for i in range(frames):
            raw = _image(raw_root / f"{i:06d}.png", cv2.IMREAD_COLOR)
            success, formal = capture.read()
            if not success or raw.shape != (960, 1280, 3) or formal.shape != raw.shape:
                raise RuntimeError(f"MOTION_REVIEW_FRAME:{short}:{i}")
            projected = raw.copy()
            for side, color in ((0, (255, 120, 20)), (1, (20, 100, 255))):
                _project_hand(projected, joints[i, side], valid[i, side], intrinsics[i], color)
            canvas = np.zeros((600, 1920, 3), dtype=np.uint8)
            for panel, source_image in enumerate((raw, projected, formal)):
                canvas[40:520, panel * 640:(panel + 1) * 640] = cv2.resize(source_image, (640, 480))
            for x, title in ((8, "RAW PHYSICAL-LEFT"), (648, "SAVED HAND MOTION PROJECTION"),
                             (1288, "FORMAL ROBOT - SAVED Q/FK")):
                cv2.putText(canvas, title, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            label = f"{session}  frame {i:04d}/{frames-1}  t_ns={int(timestamp_ns[i])}  REUSED Q / REJECTED PRODUCT"
            cv2.putText(canvas, label, (8, 547), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
            for side in (0, 1):
                side_label = "LEFT" if side == 0 else "RIGHT"
                value = float(position_mm[i, side])
                pos_text = f"{value:.1f} mm" if wrist_valid[i, side] and np.isfinite(value) else "N/A"
                rot = float(rotation_deg[i, side])
                rot_text = f"{rot:.1f} deg" if wrist_valid[i, side] and np.isfinite(rot) else "N/A"
                cv2.putText(canvas, f"{side_label}: motion={int(valid[i, side].any())} robot={int(wrist_valid[i, side])} pos={pos_text} rot={rot_text}",
                            (8 + side * 900, 576), cv2.FONT_HERSHEY_SIMPLEX, .57,
                            (255, 150, 40) if side == 0 else (40, 130, 255), 1)
            writer.write(canvas)
            rows.append({"source_frame_id": i, "timestamp_ns": int(timestamp_ns[i]),
                         "motion_joint_valid_per_side": valid[i].sum(axis=1).astype(int).tolist(),
                         "robot_wrist_valid": wrist_valid[i].astype(bool).tolist(),
                         "position_residual_mm": [float(x) if np.isfinite(x) else None for x in position_mm[i]],
                         "full_rotation_residual_deg": [float(x) if np.isfinite(x) else None for x in rotation_deg[i]]})
        if capture.read()[0]:
            raise RuntimeError(f"PRODUCT_VIDEO_EXTRA_FRAME:{short}")
    finally:
        writer.release()
        capture.release()
    check = cv2.VideoCapture(str(video))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != frames:
        raise RuntimeError(f"MOTION_REVIEW_DECODE:{short}:{decoded}")
    result = {"schema_version": "HUMAN_TO_ROBOT_10H_FULL_CHAIN_REVIEW_V1", "task_id": TASK,
              "session_id": session, "source_frames": frames, "decoded_frames": decoded,
              "provenance": "NEW_REVIEW_RENDER_REUSED_ALGORITHM_ARRAYS", "execution": "EXECUTED_REVIEW",
              "structure": "PASS_FULL_TIMELINE", "quality": "REJECTED_QUALITY_REUSED_PRODUCT",
              "review": "READY_FOR_USER_NOT_HUMAN_APPROVED", "improvement": "NONE_CLAIMED",
              "adoption": "NOT_ADOPTED", "raw_root": str(raw_root),
              "motion": artifact_ref(motion_path), "robot_r0": artifact_ref(robot_path),
              "formal_product": artifact_ref(product_result_path), "formal_video": artifact_ref(product_video),
              "review_video": artifact_ref(video), "rows": rows,
              "claim_limit": "Synchronized real Raw, saved HandMotion projection and the q-bound formal Robot; no new pose solve or product quality promotion.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    path = out / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "FULL_CHAIN_REVIEW_READY", "session": session, "frames": decoded, "result": str(path)}


def _independent_root_fk(assets: object, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = len(data["frame_id"])
    valid = np.asarray(data["wrist_valid"], bool)
    actual = np.full((frames, 2, 4, 4), np.nan, dtype=np.float64)
    saved_delta_mm = np.full((frames, 2), np.nan, dtype=np.float64)
    failure = np.full((frames, 2), "SOURCE_INVALID", dtype="<U160")
    for frame in range(frames):
        for side in range(2):
            if not valid[frame, side]:
                continue
            q = np.asarray(data["q_arm"][frame], np.float64)
            if not np.isfinite(q[side]).all():
                failure[frame, side] = "VALID_Q_NONFINITE"
                continue
            values = {name: float(q[side, j]) for j, name in enumerate(ARM_JOINT_NAMES[side])}
            # The other side is an FK API placeholder only. Its q (possibly
            # invalid/out of limits) cannot poison this side's independent FK.
            for group in ARM_JOINT_NAMES:
                for name in group:
                    values.setdefault(name, 0.0)
            try:
                fk = forward_kinematics(assets.tianji, values)
            except RendererError as error:
                failure[frame, side] = f"FIXED_URDF_HARD_LIMIT:{error}"
                continue
            t = data["T_cam_base"] @ fk[("flange_L", "flange_R")[side]] @ data["T_flange_hand"][side]
            actual[frame, side] = t
            saved_delta_mm[frame, side] = float(np.linalg.norm(
                t[:3, 3] - data["T_actual_root_cam"][frame, side, :3, 3]) * 1000)
            failure[frame, side] = "PASS"
    return actual, saved_delta_mm, failure


def _angle_degrees(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", target[..., :3, :3], actual[..., :3, :3])
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def _metric_summary(values: np.ndarray, valid: np.ndarray, gate: float) -> dict:
    indices = np.argwhere(valid & np.isfinite(values))
    if not len(indices):
        return {"count": 0, "max": None, "max_frame": None, "gate_fail_count": 0}
    chosen = values[indices[:, 0], indices[:, 1]]
    top = indices[int(np.argmax(chosen))]
    return {"count": int(len(chosen)), "p50": float(np.percentile(chosen, 50)),
            "p95": float(np.percentile(chosen, 95)), "max": float(chosen.max()),
            "max_frame": int(top[0]), "max_side": int(top[1]),
            "gate": gate, "gate_fail_count": int(np.count_nonzero(chosen > gate))}


def compare_review() -> dict:
    """One independent saved-q→fixed-URDF wrist FK evaluator for both old methods."""
    _routable()
    # The original path is kept as immutable evidence of the hard-limit runtime
    # rejection that prompted this bounded reporting fix.
    out = ATTEMPT / "lanes/compare/independent_fk_v1_runtime_fix1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    assets = load_pinned_robot_assets(REPO_ROOT)
    sessions = []
    for short, (session, frames, _) in MOTION_SESSIONS.items():
        local_path = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/"
                                  f"lanes/lane2_motion/recovered_{short}_wave0/ROBOT_R0_V1.npz")
        huro_path = REPO_ROOT / ("_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/"
                                 f"lanes/huro/full_0001/{session}/HURO_CORE_V1.npz")
        common = load_json(REPO_ROOT / ("_run/current/human_to_robot_product_first_cleanup_20260923/"
                                            "attempts/attempt_0001/lanes/compare/common_old_contract_v1/RESULT.json"))
        old_row = next(row for row in common["sessions"] if row["session_id"] == session)
        if old_row["inputs"]["local"] != artifact_ref(local_path) or old_row["inputs"]["huro"] != artifact_ref(huro_path):
            raise RuntimeError(f"OLD_COMPARE_INPUT_DRIFT:{short}")
        with np.load(local_path, allow_pickle=False) as archive:
            local = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(huro_path, allow_pickle=False) as archive:
            huro = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("frame_id", "timestamp_ns", "target_valid", "T_target_root_cam", "T_cam_base",
                    "T_flange_hand", "human_to_physical"):
            if not np.array_equal(local[key], huro[key], equal_nan=True):
                raise RuntimeError(f"COMMON_CONTRACT_DRIFT:{short}:{key}")
        if local["frame_id"].tolist() != list(range(frames)):
            raise RuntimeError(f"FRAME_ID_DRIFT:{short}")
        valid = np.asarray(local["wrist_valid"], bool) & np.asarray(huro["wrist_valid"], bool)
        fk_local, delta_local, reject_local = _independent_root_fk(assets, local)
        fk_huro, delta_huro, reject_huro = _independent_root_fk(assets, huro)
        kinematic_common = valid & (reject_local == "PASS") & (reject_huro == "PASS")
        if np.any(delta_local[reject_local == "PASS"] > .01) or np.any(delta_huro[reject_huro == "PASS"] > .01):
            raise RuntimeError(f"SAVED_FK_DOES_NOT_MATCH_INDEPENDENT_URDF:{short}")
        target = local["T_target_root_cam"]
        position_local = np.linalg.norm(fk_local[..., :3, 3] - target[..., :3, 3], axis=-1) * 1000
        position_huro = np.linalg.norm(fk_huro[..., :3, 3] - target[..., :3, 3], axis=-1) * 1000
        rotation_local = _angle_degrees(fk_local, target)
        rotation_huro = _angle_degrees(fk_huro, target)
        curves = out / f"CURVES_{short}.npz"
        np.savez_compressed(curves, frame_id=local["frame_id"], timestamp_ns=local["timestamp_ns"],
                            input_common_valid=valid, kinematic_common_valid=kinematic_common,
                            local_fk_status=reject_local, huro_fk_status=reject_huro,
                            local_position_mm=position_local,
                            huro_position_mm=position_huro, local_full_rotation_deg=rotation_local,
                            huro_full_rotation_deg=rotation_huro, local_saved_fk_delta_mm=delta_local,
                            huro_saved_fk_delta_mm=delta_huro)
        input_video = REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
                                   f"attempts/attempt_0001/lanes/compare/adapter_refresh_{short}/"
                                   "attempt_0001/LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4")
        capture = cv2.VideoCapture(str(input_video))
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
            capture.release()
            raise RuntimeError(f"COMPARE_VIDEO_FRAME_COUNT:{short}")
        video = out / f"{session}_SAME_TARGET_FK_REVIEW.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 600))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("COMPARE_REVIEW_WRITER")
        decoded_input = 0
        try:
            for frame in range(frames):
                ok, picture = capture.read()
                if not ok or picture.shape != (480, 1280, 3):
                    raise RuntimeError(f"COMPARE_VIDEO_FRAME:{short}:{frame}")
                decoded_input += 1
                canvas = np.zeros((600, 1280, 3), np.uint8)
                canvas[:480] = picture
                for side in (0, 1):
                    y = 503 + side * 26
                    for x, name, pos, rot in ((6, "LOCAL", position_local, rotation_local),
                                              (646, "HURO", position_huro, rotation_huro)):
                        status = reject_local[frame, side] if name == "LOCAL" else reject_huro[frame, side]
                        text_value = (f"{name} side={side}: pos={pos[frame, side]:.1f} mm "
                                      f"rot={rot[frame, side]:.1f} deg") if kinematic_common[frame, side] else f"{name} side={side}: {status[:35]}"
                        cv2.putText(canvas, text_value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .57,
                                    (235, 235, 235) if kinematic_common[frame, side] else (90, 90, 255), 1)
                cv2.putText(canvas, f"frame {frame}/{frames-1} | common target | 20 mm / 15 deg internal gates | no winner",
                            (8, 580), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 230, 230), 1)
                writer.write(canvas)
            if capture.read()[0]:
                raise RuntimeError(f"COMPARE_VIDEO_EXTRA_FRAME:{short}")
        finally:
            writer.release()
            capture.release()
        check = cv2.VideoCapture(str(video))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != frames or decoded_input != frames:
            raise RuntimeError(f"COMPARE_REVIEW_DECODE:{short}:{decoded}")
        sessions.append({"session_id": session, "frames": frames,
                         "input_common_valid_side_frames": int(valid.sum()),
                         "kinematic_common_valid_side_frames": int(kinematic_common.sum()),
                         "local": {"position_mm": _metric_summary(position_local, kinematic_common, 20.),
                                   "full_rotation_deg": _metric_summary(rotation_local, kinematic_common, 15.),
                                   "saved_fk_delta_mm_max": float(np.nanmax(delta_local[reject_local == "PASS"]))
                                   if np.any(reject_local == "PASS") else None,
                                   "hard_limit_rejected_source_valid": int(np.count_nonzero(valid & np.char.startswith(reject_local, "FIXED_URDF_HARD_LIMIT")))},
                         "huro": {"position_mm": _metric_summary(position_huro, kinematic_common, 20.),
                                  "full_rotation_deg": _metric_summary(rotation_huro, kinematic_common, 15.),
                                  "saved_fk_delta_mm_max": float(np.nanmax(delta_huro[reject_huro == "PASS"]))
                                  if np.any(reject_huro == "PASS") else None,
                                  "hard_limit_rejected_source_valid": int(np.count_nonzero(valid & np.char.startswith(reject_huro, "FIXED_URDF_HARD_LIMIT")))},
                         "local_q": artifact_ref(local_path), "huro_q": artifact_ref(huro_path),
                         "source_same_camera_video": artifact_ref(input_video),
                         "curves": artifact_ref(curves), "review_video": artifact_ref(video),
                         "review_decoded_frames": decoded, "same_target_mount_and_evaluator": True,
                         "winner": None})
    result = {"schema_version": "HUMAN_TO_ROBOT_10H_INDEPENDENT_COMMON_FK_REVIEW_V1",
              "task_id": TASK, "execution": "REUSED_SOLVES_INDEPENDENT_FK_REEVALUATED",
              "structure": "PASS_FOR_REPORTED_KINEMATIC_COMMON_FRAMES",
              "quality": "REJECTED_HARD_LIMIT_SCOPE_AND_INCONCLUSIVE_NO_INDEPENDENT_TRUTH",
              "review": "READY_FOR_USER_NOT_HUMAN_APPROVED", "improvement": "NONE_CLAIMED",
              "adoption": "NOT_ADOPTED", "new_solver_invocations": 0,
              "runtime_fix": "First independent_fk_v1 execution stopped on a real HuRo URDF hard-limit violation; that path is retained, and this one permitted reporting fix masks only the rejected method/side for residual evaluation.",
              "evaluator_code": artifact_ref(Path(__file__)), "sessions": sessions,
              "claim_limit": "Same frozen target/mount and independent fixed-URDF wrist FK for old Local/HuRo q; incomplete collision and no external accuracy or winner claim.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    path = out / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "INDEPENDENT_COMMON_FK_REVIEW_READY", "result": str(path), "sessions": len(sessions)}


def sensor_review() -> dict:
    """Review newest saved Sensor displays and expose 2D projection visibility limits."""
    _routable()
    source_path = REPO_ROOT / ("_run/current/human_to_robot_sensor_display_correction_20260923/"
                               "attempts/attempt_0001/lanes/sensor/review_metric_v2/RESULT.json")
    source = load_json(source_path)
    out = ATTEMPT / "lanes/sensor/independent_review_v1"
    if out.exists():
        raise FileExistsError(out)
    expected = {"play_cards_0916_097": 165, "play_cards_0916_098": 179, "play_cards_0916_101": 122}
    if {row["session_id"]: row["frames"] for row in source["sessions"]} != expected:
        raise RuntimeError("SENSOR_SESSION_OR_TIMELINE_DRIFT")
    out.mkdir(parents=True)
    rows = []
    for old in source["sessions"]:
        session, frames = old["session_id"], old["frames"]
        motion_path, robot_path = Path(old["source_motion"]["path"]), Path(old["source_robot_fk"]["path"])
        video = Path(old["video"]["path"])
        if (artifact_ref(motion_path) != old["source_motion"] or artifact_ref(robot_path) != old["source_robot_fk"]
                or artifact_ref(video)["sha256"] != old["video"]["sha256"]):
            raise RuntimeError(f"SENSOR_SOURCE_DRIFT:{session}")
        with np.load(motion_path, allow_pickle=False) as data, np.load(robot_path, allow_pickle=False) as backend:
            valid = np.asarray(data["joint_valid"], bool)
            xyz = np.asarray(data["joints21_camera"], np.float64)
            k = np.asarray(data["camera_K"], np.float64)
            size = np.asarray(data["camera_image_size"], int)
            time_ok = np.asarray(data["time_transition_valid"], bool)
            source_frame = np.asarray(data["source_video_frame_id"])
            robot_valid = np.asarray(backend["valid"], bool)
        if (valid.shape != (frames, 2, 21) or xyz.shape != (frames, 2, 21, 3)
                or size.tolist() != [1280, 960] or k.shape != (3, 3)
                or len(time_ok) != frames or robot_valid.shape != (frames, 2)):
            raise RuntimeError(f"SENSOR_ARRAY_SHAPE:{session}")
        finite = np.isfinite(xyz).all(axis=-1)
        z = xyz[..., 2]
        positive = valid & finite & (z > 0.01)
        denom = np.where(z > 0.01, z, 1.)
        u = k[0, 0] * xyz[..., 0] / denom + k[0, 2]
        v = k[1, 1] * xyz[..., 1] / denom + k[1, 2]
        inside = positive & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
        outside = positive & ~inside
        per_frame_outside = outside.sum(axis=(1, 2))
        abnormal = int(np.argmax(per_frame_outside))
        samples = sorted(set([0, frames // 2, frames - 1, abnormal]))
        capture = cv2.VideoCapture(str(video))
        decoded, sample_refs = 0, []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if decoded in samples:
                path = out / f"{session}_sample_{decoded:06d}.png"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"SENSOR_SAMPLE_WRITE:{session}:{decoded}")
                sample_refs.append({"frame_id": decoded, "image": artifact_ref(path)})
            decoded += 1
        capture.release()
        if decoded != frames or len(sample_refs) != len(samples):
            raise RuntimeError(f"SENSOR_REVIEW_DECODE:{session}:{decoded}")
        rows.append({"session_id": session, "frames": frames, "video": artifact_ref(video),
                     "decoded_frames": decoded, "motion": artifact_ref(motion_path),
                     "robot_fk": artifact_ref(robot_path), "sample_images": sample_refs,
                     "fixed_equal_time_frames": [0, frames // 2, frames - 1],
                     "abnormal_max_out_of_frame": {"frame_id": abnormal,
                                                   "joint_side_slots": int(per_frame_outside[abnormal])},
                     "complaint_window": "NO_SEPARATE_FROZEN_SENSOR_COMPLAINT_FRAME_IN_CURRENT_PACKET",
                     "pixel_visibility_denominator": int(valid.size),
                     "source_joint_invalid_slots": int((~valid).sum()),
                     "nonfinite_valid_slots": int((valid & ~finite).sum()),
                     "negative_or_near_z_valid_slots": int((valid & finite & (z <= .01)).sum()),
                     "positive_z_out_of_frame_slots": int(outside.sum()),
                     "positive_z_in_frame_slots": int(inside.sum()),
                     "time_transition_invalid_frames": int((~time_ok).sum()),
                     "robot_valid_side_frames": int(robot_valid.sum()),
                     "source_video_frame_min_max": [int(np.min(source_frame)), int(np.max(source_frame))],
                     "technical_visual_quality": "INCONCLUSIVE_NO_INDEPENDENT_ANATOMICAL_KEYPOINT_GT",
                     "review_authority": "AI_SAMPLE_REVIEW_PROXY_NOT_FULL_HUMAN_REVIEW",
                     "user_review": "PENDING"})
    result = {"schema_version": "HUMAN_TO_ROBOT_10H_SENSOR_INDEPENDENT_REVIEW_V1", "task_id": TASK,
              "execution": "REUSED_ARRAYS_AND_VIDEO_NEW_VISIBILITY_AUDIT", "structure": "PASS_466_FRAME_DECODE",
              "quality": "PASS_DISPLAY_ONLY_VISUAL_ALIGNMENT_INCONCLUSIVE", "review": "AI_SAMPLED_USER_PENDING",
              "improvement": "NO_NEW_SOLVE", "adoption": "NOT_ADOPTED", "source": artifact_ref(source_path),
              "sessions": rows, "claim_limit": "Full decode and fixed/anomaly image samples with independent image-domain validity counts, not a measured wrist alignment or human visual acceptance.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    path = out / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "SENSOR_INDEPENDENT_REVIEW_READY", "result": str(path), "sessions": len(rows)}


def delivery() -> dict:
    """Validate the 15 fixed slot identities without re-encoding or relabeling history."""
    _routable()
    out = ATTEMPT / "delivery"
    if out.exists():
        raise FileExistsError(out)
    scene_base = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
                              "attempts/attempt_0001/lanes/lane1_scene")
    formal_base = REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
                               "attempts/attempt_0001/lanes/motion_product")
    sensor_result_path = ATTEMPT / "lanes/sensor/independent_review_v1/RESULT.json"
    compare_result_path = ATTEMPT / "lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json"
    sensor_result, compare_result = load_json(sensor_result_path), load_json(compare_result_path)
    scene_defs = [
        ("SCENE_007", "get_potato_chips_0915_007", 378, "clean_candidate_007_wave4"),
        ("SCENE_031", "play_cards_0915_031", 149, "clean_candidate_031_wave5"),
        ("SCENE_103", "get_potato_chips_0902_103", 284, "clean_candidate_103_wave4"),
        ("SCENE_042", "play_cards_0902_042", 171, "clean_candidate_042_wave4"),
    ]
    slots = []
    for slot, session, frames, folder in scene_defs:
        slots.append((slot, session, frames, scene_base / folder / "SCENE_CLEAN_CANDIDATE_REVIEW.mp4",
                      "REUSED", "REJECTED_QUALITY", "SCENE_RAW_SUPPORT_CLEAN_FULL_REVIEW",
                      scene_base / folder / "RESULT.json"))
    for item in sensor_result["sessions"]:
        slots.append(("SENSOR_" + item["session_id"][-3:], item["session_id"], item["frames"],
                      Path(item["video"]["path"]), "REUSED", "PASS_DISPLAY_ONLY_ALIGNMENT_INCONCLUSIVE",
                      "SENSOR_RGB_WORLD_LOCAL_SAVED_FK_FULL_REVIEW", sensor_result_path))
    for short, (session, frames, product_part) in MOTION_SESSIONS.items():
        review_path = ATTEMPT / "lanes/motion_product" / f"full_chain_{short}/RESULT.json"
        review = load_json(review_path)
        slots.append((f"MOTION_{short}", session, frames, Path(review["review_video"]["path"]),
                      "NEW_REVIEW_RENDER_REUSED_Q", "REJECTED_QUALITY",
                      "RAW_HAND_MOTION_FORMAL_PRODUCT_FULL_CHAIN", review_path))
    for item in compare_result["sessions"]:
        short = item["session_id"][-3:]
        slots.append((f"COMPARE_{short}", item["session_id"], item["frames"],
                      Path(item["review_video"]["path"]), "NEW_REVIEW_RENDER_REUSED_SOLVES",
                      "REJECTED_HARD_LIMIT_SCOPE_NO_WINNER", "SAME_TARGET_LOCAL_HURO_NUMERIC_AND_VIDEO",
                      compare_result_path))
    product_defs = [
        ("PRODUCT_007", "get_potato_chips_0915_007", 378,
         REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/motion_product/formal_007_current/attempt_0001"),
        ("PRODUCT_031", "play_cards_0915_031", 149, formal_base / "formal_product_031/attempt_0004"),
        ("PRODUCT_103", "get_potato_chips_0902_103", 284,
         formal_base / "formal_product_get_potato_chips_0902_103/attempt_0002"),
        ("PRODUCT_042", "play_cards_0902_042", 171,
         formal_base / "formal_product_play_cards_0902_042/attempt_0002"),
    ]
    for slot, session, frames, root in product_defs:
        slots.append((slot, session, frames, root / "robot.mp4", "REUSED",
                      "REJECTED_QUALITY", "PURE_FORMAL_ROBOT_CANDIDATE", root / "PRODUCT_RESULT.json"))
    if len(slots) != 15 or len({item[0] for item in slots}) != 15:
        raise RuntimeError("FIXED_15_SLOT_SET_DRIFT")
    # The two all-chain slots are new synchronized reviews; old one-panel
    # robot_candidate.mp4 is intentionally not accepted for these slots.
    rows = []
    for slot, session, frames, video, provenance, quality, content, source_result in slots:
        if not video.is_file() or not source_result.is_file():
            raise FileNotFoundError((slot, video, source_result))
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded != frames:
            raise RuntimeError(f"SLOT_FULL_DECODE_OR_SESSION_TIMELINE:{slot}:{decoded}/{frames}")
        source = load_json(source_result)
        if "session_id" in source and source["session_id"] != session:
            raise RuntimeError(f"SLOT_SOURCE_SESSION:{slot}")
        rows.append({"slot": slot, "session_id": session, "artifact_type": "FULL_SESSION_REVIEW"
                     if not slot.startswith("PRODUCT_") else "FULL_SESSION_PRODUCT_CANDIDATE",
                     "source_frame_first_last": [0, frames - 1], "expected_frames": frames,
                     "decoded_frames": decoded, "video": artifact_ref(video),
                     "source_result": artifact_ref(source_result), "content_contract": content,
                     "provenance": provenance, "execution": "EXECUTED_THIS_ROUND" if provenance.startswith("NEW_") else "COMPLETED_PREVIOUSLY_REVALIDATED",
                     "structure": "PASS_FULL_DECODE", "quality": quality,
                     "review": "AI_SAMPLED_ONLY_USER_PENDING", "improvement": "NONE_CLAIMED",
                     "adoption": "NOT_ADOPTED"})
    out.mkdir(parents=True)
    result = {"schema_version": "HUMAN_TO_ROBOT_10H_15_SLOT_DELIVERY_V1", "task_id": TASK,
              "expected_slots": 15, "validated_slots": len(rows), "rows": rows,
              "auxiliary_007_context_canary": artifact_ref(DEST / "AI_VISUAL_REVIEW.json"),
              "sensor_independent_review": artifact_ref(sensor_result_path),
              "common_independent_fk": artifact_ref(compare_result_path),
              "product_structure": "4/4_PREVIOUSLY", "product_quality": "0/4", "product_adoption": "0/4",
              "review_transferred_to_user_device": False,
              "claim_limit": "All 15 fixed slots are real full-session files with correct session/timeline and SHA, but none is thereby quality-adopted. New reviews reuse old solves; 007 context canary failed and stays auxiliary.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    path = out / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "FIFTEEN_SLOTS_STRUCTURALLY_VALIDATED", "result": str(path), "slots": len(rows)}


def geometry_review() -> dict:
    """Display the two already executed 031 geometry consumers, without new inference."""
    _routable()
    base = REPO_ROOT / ("_run/current/human_to_robot_product_first_cleanup_20260923/"
                        "attempts/attempt_0001/lanes")
    patch = base / "scene/object_patch_031/window_v1"
    correspondence = base / "motion_product/robot_correspondence_031/window_v1"
    # The first attempt correctly exposed a missing terminal correspondence
    # overlay: source frame 81 has no t+1 frame. Keep that failed attempt.
    out = ATTEMPT / "lanes/motion_product/geometry_reuse_review_v1_runtime_fix1"
    if out.exists():
        raise FileExistsError(out)
    for root in (patch, correspondence):
        if not (root / "RESULT.json").is_file():
            raise FileNotFoundError(root / "RESULT.json")
    out.mkdir(parents=True)
    video = out / "031_PATCH_AND_ROBOT_CORRESPONDENCE_16FRAME_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("GEOMETRY_VIDEO_WRITER_UNAVAILABLE")
    inputs = []
    try:
        for frame in range(66, 82):
            left = patch / f"overlay_{frame:06d}.png"
            right = correspondence / f"correspondence_{frame:06d}.png"
            if frame == 81 and not right.exists():
                images = [_image(left, cv2.IMREAD_COLOR), np.zeros((960, 1280, 3), dtype=np.uint8)]
                cv2.putText(images[1], "NOT APPLICABLE: final frame has no t+1",
                            (60, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)
            else:
                images = [_image(path, cv2.IMREAD_COLOR) for path in (left, right)]
            if any(image.shape != (960, 1280, 3) for image in images):
                raise RuntimeError(f"GEOMETRY_REVIEW_FRAME_SHAPE:{frame}")
            panels = [cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA) for image in images]
            canvas = np.concatenate(panels, axis=1)
            cv2.putText(canvas, f"031 source frame {frame} | EXISTING WINDOW EVIDENCE ONLY",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
            cv2.putText(canvas, "object patch", (15, 455), cv2.FONT_HERSHEY_SIMPLEX,
                        .62, (255, 255, 255), 2)
            cv2.putText(canvas, "robot rigid correspondence", (655, 455), cv2.FONT_HERSHEY_SIMPLEX,
                        .62, (255, 255, 255), 2)
            writer.write(canvas)
            inputs.append({"source_frame_id": frame, "patch_image": artifact_ref(left),
                           "robot_image": artifact_ref(right) if right.exists() else None,
                           "robot_correspondence_status": "NO_T_PLUS_ONE" if frame == 81 else "EXECUTED"})
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != 16:
        raise RuntimeError(f"GEOMETRY_REVIEW_DECODE:{decoded}/16")
    result = {"schema_version": "HUMAN_TO_ROBOT_10H_GEOMETRY_REVIEW_V1", "task_id": TASK,
              "session_id": "play_cards_0915_031", "source_frame_first_last": [66, 81],
              "decoded_frames": decoded, "video": artifact_ref(video), "inputs": inputs,
              "object_patch_result": artifact_ref(patch / "RESULT.json"),
              "robot_correspondence_result": artifact_ref(correspondence / "RESULT.json"),
              "execution": "NEW_REVIEW_OF_REUSED_GEOMETRY_RESULTS",
              "quality": "LIMITED_WINDOW_ONLY_CONTACT_R1_NOT_ADOPTED",
              "claim_limit": "Patch and robot rigid correspondence are separate diagnostics; neither proves moving-scene correspondence, Contact, or whole-session occlusion."}
    path = out / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "GEOMETRY_16FRAME_REVIEW_COMPLETE", "result": str(path)}


def finalize() -> dict:
    """Publisher-side projection from immutable run evidence, without quality promotion."""
    _routable()
    result_path = ATTEMPT / "RESULT.json"
    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    if result_path.exists() or validation_path.exists():
        raise FileExistsError(result_path)
    evidence = {
        "scene": DEST / "AI_VISUAL_REVIEW.json",
        "sensor": ATTEMPT / "lanes/sensor/independent_review_v1/RESULT.json",
        "motion_product": ATTEMPT / "lanes/motion_product/geometry_reuse_review_v1_runtime_fix1/RESULT.json",
        "compare": ATTEMPT / "lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json",
        "delivery": ATTEMPT / "delivery/RESULT.json",
    }
    for path in evidence.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    delivery_result = load_json(evidence["delivery"])
    if delivery_result.get("validated_slots") != 15:
        raise RuntimeError("FIFTEEN_SLOT_DELIVERY_NOT_COMPLETE")
    for row in delivery_result["rows"]:
        for field in ("video", "source_result"):
            errors = validate_artifact_ref(row[field])
            if errors:
                raise RuntimeError(f"DELIVERY_ARTIFACT_DRIFT:{row['slot']}:{field}:{errors}")
        if row["decoded_frames"] != row["expected_frames"]:
            raise RuntimeError(f"DELIVERY_FRAME_MISMATCH:{row['slot']}")
    index_path = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_10H_DELIVERY/INDEX_ZH.md"
    junit_path = ATTEMPT / "REGRESSION_JUNIT.xml"
    junit = ET.parse(junit_path).getroot()
    suite = junit if junit.tag == "testsuite" else junit.find("testsuite")
    if suite is None:
        raise RuntimeError("REGRESSION_JUNIT_SUITE_MISSING")
    tests, failures, errors, skipped = (int(suite.get(name, "0"))
                                        for name in ("tests", "failures", "errors", "skipped"))
    if tests < 21 or failures or errors:
        raise RuntimeError(f"TARGETED_REGRESSION_FAILED:{tests}:{failures}:{errors}")
    validation = {
        "schema_version": "HUMAN_TO_ROBOT_10H_FINAL_VALIDATION_V1", "task_id": TASK,
        "validated_full_session_slots": 15,
        "source_timeline_and_sha": "PASS_FOR_15_DECLARED_SLOTS",
        "geometry_reuse_video_frames": 16,
        "007_new_clean_quality": "REJECTED_QUALITY_NO_FULL_EXPANSION",
        "regression": {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped,
                       "junit": artifact_ref(junit_path)},
        "cpfs_hardlink_transaction_tests": "NOT_EVALUATED_ENV",
        "navigation": artifact_ref(index_path),
        "user_device_transfer": "NOT_PERFORMED",
    }
    atomic_json(validation_path, validation)
    lane_status = {
        "scene": ("REJECTED_QUALITY", "REJECTED_QUALITY",
                  "No second ProPainter recipe or full expansion in this task."),
        "sensor": ("COMPLETE_LIMITED_REVIEW", "DISPLAY_PASS_ALIGNMENT_INCONCLUSIVE",
                   "Independent keypoint evidence or user visual review is needed for alignment authority."),
        "motion_product": ("COMPLETE_REJECTED_PRODUCT", "REJECTED_QUALITY",
                           "Motion quality or missing left hand requires new evidence; no new IK authorized."),
        "compare": ("COMPLETE_LIMITED_COMPARISON", "REJECTED_HARD_LIMIT_SCOPE",
                    "HuRo hard-limit failures prevent a method winner; no new solve authorized."),
    }
    for lane, (status, quality, future_need) in lane_status.items():
        state_path = ATTEMPT / "lanes" / lane / "STATE.json"
        current = load_json(state_path)
        if current.get("task_id") != TASK or current.get("status") != "PENDING":
            raise RuntimeError(f"LANE_STATE_UNEXPECTED:{lane}")
        current.update(status=status, execution="EXECUTED", structure="PASS_FOR_DECLARED_SCOPE",
                       quality=quality, improvement="NONE_CLAIMED", adoption="NOT_ADOPTED",
                       result=artifact_ref(evidence[lane]), consumer="FIFTEEN_SLOT_DELIVERY_AND_CURRENT_NAVIGATION",
                       dependencies=[artifact_ref(evidence["delivery"])],
                       next_action="NO_FURTHER_ACTION_UNDER_FROZEN_10H_SCOPE",
                       future_need=future_need, blocker=None, updated_at=now_iso())
        atomic_json(state_path, current)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_10H_TERMINAL_RESULT_V1", "task_id": TASK,
        "status": "TERMINAL_DELIVERY_QUALITY_REJECTED", "execution": "COMPLETE_FOR_AUTHORIZED_SCOPE",
        "structure": "15_OF_15_VIDEO_SLOTS_FULL_DECODE_AND_SHA_PASS",
        "quality": "PRODUCTS_0_OF_4_AND_NEW_007_CLEAN_REJECTED",
        "adoption": "0_OF_4_PRODUCTS", "improvement": "NO_PRODUCT_QUALITY_IMPROVEMENT_CLAIMED",
        "review": "AI_SAMPLED_USER_PENDING", "delivery": artifact_ref(evidence["delivery"]),
        "final_validation": artifact_ref(validation_path),
        "lanes": {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json") for lane in lane_status},
        "geometry_review": artifact_ref(evidence["motion_product"]),
        "formal_entry_resume": artifact_ref(REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
                             "attempts/attempt_0001/formal_entry_resume/attempt_0002/RESULT.json")),
        "new_model_runs": 1, "new_ik_solves": 0, "new_huro_solves": 0,
        "contact_r1_executed": 0, "product_structure": "4/4", "product_quality": "0/4",
        "product_adoption": "0/4", "user_device_transfer": False,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "Offline review delivery only; 007 context Clean and four products remain quality-rejected. Sensor alignment and full collision/Contact remain unproven.",
    }
    atomic_json(result_path, result)
    return {"status": result["status"], "result": str(result_path), "validated_slots": 15}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "model", "review", "motion-review", "compare-review", "sensor-review", "geometry-review", "delivery", "finalize"), required=True)
    parser.add_argument("--session", choices=tuple(MOTION_SESSIONS))
    args = parser.parse_args()
    if args.stage == "finalize":
        value = finalize()
    elif args.stage == "geometry-review":
        value = geometry_review()
    elif args.stage == "delivery":
        value = delivery()
    elif args.stage == "sensor-review":
        value = sensor_review()
    elif args.stage == "compare-review":
        value = compare_review()
    elif args.stage == "motion-review":
        if args.session is None:
            parser.error("motion-review requires --session")
        value = motion_review(args.session)
    else:
        value = {"prepare": prepare, "model": run_model, "review": review}[args.stage]()
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
