"""Trace the frozen Poker Clean stack and admit exactly one evidence-led repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.v5_scene import composite_clean


TASK = "human_to_robot_shared_hand_delivery_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANE = ATTEMPT / "lanes/clean"
OUT = LANE / "POKER_076_091_LAYER_TRACE_V1"
BASE = REPO_ROOT / (
    "_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/"
    "lanes/clean/POKER_076_091_INPUT_REPAIR"
)
VENDOR = REPO_ROOT / "vendor/ProPainter"
LAMA = REPO_ROOT / "assets/models/lama-onnx/lama_fp32.onnx"
LAMA_PIN = LAMA.parent / "ASSET_PIN.json"
LAMA_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
ORT_ROOT = ATTEMPT / "environments/onnxruntime-1.22.1"
SOURCE_MAP = OUT / "SOURCE_MAP.json"
TRACE = OUT / "trace"
TRACE_MANIFEST = TRACE / "TRACE_MANIFEST.json"
DECISION_V2 = OUT / "DECISION_V2.json"
CANDIDATE = LANE / "POKER_076_091_LAMA_RESIDUAL_V3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image(path: Path, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flag)
    if value is None:
        raise RuntimeError(f"IMAGE_UNREADABLE:{path}")
    return value


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"IMAGE_WRITE_FAILED:{path}")


def _registered() -> None:
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if packet.get("task_id") != TASK or not any(
        row.get("task_id") == TASK and row.get("execution_allowed") is True
        for row in index.get("task_packets", [])
    ):
        raise RuntimeError("TASK_NOT_ROUTABLE")


def _state(execution: str, *, next_action: str, result: Path | None = None,
           blocker: dict | None = None) -> None:
    payload = {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_CLEAN_STATE_V1",
        "task_id": TASK,
        "owner": "clean",
        "updated_at": _now(),
        "execution": execution,
        "structure": "PASS" if result is not None else "NOT_EVALUATED",
        "quality": "NOT_EVALUATED",
        "adoption": "NOT_ADOPTED",
        "result": artifact_ref(result) if result is not None else None,
        "next_action": next_action,
        "blocker": blocker,
        "dependencies": [artifact_ref(BASE / "INPUT.json"), artifact_ref(BASE / "RESULT.json")],
        "claim_limit": "Frozen Poker 76-91 diagnostic and one evidence-led Clean candidate only.",
    }
    LANE.mkdir(parents=True, exist_ok=True)
    atomic_json(LANE / "STATE.json", payload)


def _lama_gate() -> dict:
    manifest = load_json(REPO_ROOT / "assets/models/MANIFEST.json")
    rows = [row for row in manifest.get("entries", manifest.get("files", []))
            if row.get("path") == "assets/models/lama-onnx/lama_fp32.onnx"]
    pin = load_json(LAMA_PIN) if LAMA_PIN.exists() else {}
    pin_ok = (pin.get("asset") == "assets/models/lama-onnx/lama_fp32.onnx"
              and pin.get("sha256") == LAMA_SHA256
              and pin.get("bytes") == LAMA.stat().st_size
              and pin.get("source_revision") == "a3ee2fca54baebec351b8fa7786154ffa7555aa6"
              and pin.get("declared_license") == "Apache-2.0")
    ref = artifact_ref(LAMA)
    sha_ok = ref["sha256"] == LAMA_SHA256 and len(rows) == 1 and rows[0].get("sha256") == LAMA_SHA256
    return {
        "weight": ref,
        "manifest_rows": rows,
        "sha_registered": sha_ok,
        "asset_pin": artifact_ref(LAMA_PIN) if LAMA_PIN.exists() else None,
        "source_and_license_pin_valid": pin_ok,
        "runtime_present": (ORT_ROOT / "onnxruntime").is_dir(),
        "admission": "PASS" if sha_ok and pin_ok else "BLOCKED_LICENSE_OR_SOURCE_PROOF",
        "reason": None if sha_ok and pin_ok else
        "Weight SHA or local source/license pin is missing or inconsistent.",
    }


def preflight() -> dict:
    _registered()
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = load_json(BASE / "INPUT.json")
    if frozen.get("frame_range") != [76, 91] or len(frozen.get("rows", [])) != 16:
        raise RuntimeError("FROZEN_INPUT_DRIFT")
    OUT.mkdir(parents=True)
    source_map = []
    for expected, row in enumerate(frozen["rows"]):
        if row["local_index"] != expected or row["frame_id"] != expected + 76:
            raise RuntimeError("SOURCE_MAP_DRIFT")
        frame = Path(row["frames"]["path"])
        mask = Path(row["model_masks"]["path"])
        if _image(frame).shape != (720, 960, 3) or _image(mask, cv2.IMREAD_GRAYSCALE).shape != (720, 960):
            raise RuntimeError("MODEL_DOMAIN_DRIFT")
        source_map.append({
            "local_index": expected,
            "source_frame": row["frame_id"],
            "frame": artifact_ref(frame),
            "model_mask": artifact_ref(mask),
            "raw": row["raw"],
            "write": row["write"],
            "protect": row["protect"],
            "unknown": row["unknown"],
        })
    atomic_json(SOURCE_MAP, source_map)
    gate = _lama_gate()
    record = {
        "schema_version": "POKER_CLEAN_LAYER_TRACE_PREFLIGHT_V1",
        "task_id": TASK,
        "frozen_candidate": artifact_ref(BASE / "RESULT.json"),
        "frozen_input": artifact_ref(BASE / "INPUT.json"),
        "source_map": artifact_ref(SOURCE_MAP),
        "source_frames": list(range(76, 92)),
        "vendor": artifact_ref(VENDOR / "inference_propainter.py"),
        "weights": [artifact_ref(VENDOR / "weights" / name) for name in (
            "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")],
        "parameters": frozen["model_parameters"],
        "lama_gate": gate,
        "quality_authority": "DIAGNOSTIC_PRECHECK_ONLY",
    }
    atomic_json(OUT / "PREFLIGHT.json", record)
    _state("READY_TRACE", next_action="RUN_ONE_SAME_SIGNATURE_INSTRUMENTED_PROPAINTER_TRACE",
           result=OUT / "PREFLIGHT.json")
    return {"preflight": str(OUT / "PREFLIGHT.json"), "frames": len(source_map),
            "lama_admission": gate["admission"]}


def trace() -> dict:
    _registered()
    preflight_value = load_json(OUT / "PREFLIGHT.json")
    if TRACE.exists() or (OUT / "TRACE_RESULT.json").exists():
        raise FileExistsError("CLEAN_TRACE_EXISTS")
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease = load_json(lease_path)
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:clean"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("SHARED_CLEAN_GPU_LEASE_REQUIRED")
    command = [
        sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
        "--video", str(BASE / "frames"), "--mask", str(BASE / "model_masks"),
        "--output", str(OUT / "upstream"), "--width", "960", "--height", "720",
        "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
        "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
        "--save_frames", "--fp16", "--trace_dir", str(TRACE),
        "--trace_source_map", str(SOURCE_MAP),
    ]
    invocation = {
        "schema_version": "POKER_CLEAN_INSTRUMENTED_INVOCATION_V1",
        "task_id": TASK,
        "started_at": _now(),
        "purpose": "SAME_SIGNATURE_DIAGNOSTIC_REPRODUCTION_NOT_SECOND_QUALITY_CANDIDATE",
        "input": preflight_value["frozen_input"],
        "source_map": artifact_ref(SOURCE_MAP),
        "vendor": artifact_ref(VENDOR / "inference_propainter.py"),
        "command": command,
        "lease_token": lease["fencing_token"],
    }
    atomic_json(OUT / "INVOCATION.json", invocation)
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(REPO_ROOT / "_run/cache"))
    with (OUT / "PROPAINTER_TRACE.log").open("xb") as log:
        completed = subprocess.run(command, cwd=VENDOR, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"INSTRUMENTED_PROPAINTER_FAILED:{completed.returncode}")
    manifest = load_json(TRACE_MANIFEST)
    outputs = sorted((OUT / "upstream/frames/frames").glob("*.png"))
    if manifest.get("video_length") != 16 or len(outputs) != 16:
        raise RuntimeError("TRACE_OUTPUT_COVERAGE")
    result = {
        "schema_version": "POKER_CLEAN_INSTRUMENTED_TRACE_RESULT_V1",
        "task_id": TASK,
        "execution": "REAL_SAME_SIGNATURE_PROPAINTER_TRACE",
        "quality_candidate_count_increment": 0,
        "invocation": artifact_ref(OUT / "INVOCATION.json"),
        "trace_manifest": artifact_ref(TRACE_MANIFEST),
        "model_log": artifact_ref(OUT / "PROPAINTER_TRACE.log"),
        "output_frames": [artifact_ref(path) for path in outputs],
        "next_action": "VERIFY_ACTUAL_REFERENCE_MASKS_AND_SEPARATE_INTERNAL_OUTPUT_FROM_FINAL_PASTE",
    }
    atomic_json(OUT / "TRACE_RESULT.json", result)
    _state("TRACE_EXECUTED", next_action=result["next_action"], result=OUT / "TRACE_RESULT.json")
    return {"result": str(OUT / "TRACE_RESULT.json"), "frames": len(outputs),
            "trace_steps": len(manifest["steps"])}


def _last_visits(manifest: dict) -> dict[int, dict]:
    visits: dict[int, dict] = {}
    for step in manifest["steps"]:
        for visit in step["visits"]:
            visits[int(visit["local_index"])] = visit
    return visits


def expected_reference_schedule(length: int = 16, neighbor_length: int = 10,
                                ref_stride: int = 10) -> list[dict]:
    """Mirror the frozen vendor index policy without claiming actual consumption."""
    stride = neighbor_length // 2
    rows = []
    for midpoint in range(0, length, stride):
        neighbors = list(range(max(0, midpoint - stride), min(length, midpoint + stride + 1)))
        references = [index for index in range(0, length, ref_stride) if index not in neighbors]
        rows.append({"mid_neighbor_local_index": midpoint,
                     "neighbor_local_indices": neighbors,
                     "reference_local_indices": references})
    return rows


def build_visible_card_permissions(direct_human: np.ndarray, historical: np.ndarray,
                                   card: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (direct_human.dtype != np.bool_ or historical.dtype != np.bool_ or card.dtype != np.bool_
            or direct_human.shape != historical.shape or direct_human.shape != card.shape):
        raise ValueError("VISIBLE_CARD_PERMISSION_DOMAIN")
    support = direct_human | historical
    conflict = card & direct_human
    visible_card = card & ~direct_human
    expanded = cv2.dilate(visible_card.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    unknown = expanded & ~visible_card & support & ~direct_human
    write = support & ~visible_card & ~unknown
    return write, visible_card, unknown, conflict


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    value = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(value, (0, 0), (480, 34), (0, 0, 0), -1)
    cv2.putText(value, title, (9, 24), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (255, 255, 255), 2, cv2.LINE_AA)
    return value


def analyze() -> dict:
    _registered()
    if (OUT / "LAYER_ANALYSIS.json").exists():
        raise FileExistsError("LAYER_ANALYSIS_EXISTS")
    frozen = load_json(BASE / "INPUT.json")
    manifest = load_json(TRACE_MANIFEST)
    source_map = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    if not isinstance(source_map, list):
        raise RuntimeError("TRACE_SOURCE_MAP_NOT_ARRAY")
    if manifest["source_map"] != source_map or manifest["video_length"] != 16:
        raise RuntimeError("TRACE_SOURCE_IDENTITY_DRIFT")
    selected_rows = [slot for step in manifest["steps"] for slot in step["selected_slots"]]
    actual_schedule = [
        {key: step[key] for key in ("mid_neighbor_local_index", "neighbor_local_indices",
                                    "reference_local_indices")}
        for step in manifest["steps"]
    ]
    if actual_schedule != expected_reference_schedule():
        raise RuntimeError("ACTUAL_REFERENCE_SCHEDULE_DRIFT")
    if not selected_rows or any(slot["actual_mask_tensor"]["nonzero"] <= 0 for slot in selected_rows):
        raise RuntimeError("ACTUAL_MASK_TENSOR_NOT_CONSUMED")
    if any(step["neighbor_local_indices"] and
           set(step["neighbor_local_indices"]) & set(step["reference_local_indices"])
           for step in manifest["steps"]):
        raise RuntimeError("REFERENCE_NEIGHBOR_OVERLAP")
    visits = _last_visits(manifest)
    if sorted(visits) != list(range(16)):
        raise RuntimeError("INTERNAL_OUTPUT_COVERAGE")
    (OUT / "final_paste").mkdir()
    (OUT / "layer_review_frames").mkdir()
    rows = []
    reintroduced_total = 0
    for local, row in enumerate(frozen["rows"]):
        raw = _image(Path(row["raw"]["path"]))
        write = _image(Path(row["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        protect = _image(Path(row["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        unknown = _image(Path(row["unknown"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        actual_mask = _image(Path(row["model_masks"]["path"]), cv2.IMREAD_GRAYSCALE)
        upstream = _image(OUT / "upstream/frames/frames" / f"{local:04d}.png")
        cumulative = _image(TRACE / visits[local]["internal_comp_cumulative"])
        if not np.array_equal(upstream, cumulative):
            raise RuntimeError(f"INTERNAL_CUMULATIVE_OUTPUT_DRIFT:{local}")
        generated = cv2.resize(upstream, (1280, 960), interpolation=cv2.INTER_LINEAR)
        final = composite_clean(raw, generated, write, protect)
        final_path = OUT / "final_paste" / f"{row['frame_id']:06d}.png"
        _save(final_path, final)
        reintroduced = int(np.any(final != generated, axis=2)[write].sum())
        reintroduced_total += reintroduced
        overlay = cv2.resize(_image(Path(row["frames"]["path"])), (960, 720))
        overlay[actual_mask > 0] = (
            .40 * overlay[actual_mask > 0] + .60 * np.array([0, 0, 255])
        ).astype(np.uint8)
        pred = _image(TRACE / visits[local]["raw_prediction"])
        canvas = np.vstack((
            np.hstack((_panel(cv2.resize(raw, (960, 720)), f"RAW {row['frame_id']}"),
                       _panel(overlay, "ACTUAL MODEL MASK"))),
            np.hstack((_panel(pred, "MODEL INTERNAL PRED"),
                       _panel(upstream, "INTERNAL COMP"))),
            np.hstack((_panel(cv2.resize(final, (960, 720)), "FINAL PASTE"),
                       _panel(cv2.resize(unknown.astype(np.uint8) * 255, (960, 720)),
                              "UNKNOWN (NOT WRITTEN)"))),
        ))
        review_path = OUT / "layer_review_frames" / f"{local:06d}.png"
        _save(review_path, canvas)
        rows.append({
            "local_index": local,
            "source_frame": row["frame_id"],
            "last_internal_visit": visits[local],
            "actual_model_mask_nonzero": int((actual_mask > 0).sum()),
            "write_px": int(write.sum()),
            "protect_px": int(protect.sum()),
            "unknown_px": int(unknown.sum()),
            "final": artifact_ref(final_path),
            "review": artifact_ref(review_path),
            "compositor_reintroduced_pixels_inside_write": reintroduced,
            "outside_write_changed_pixels": int(np.any(final != raw, axis=2)[~write].sum()),
            "protected_changed_pixels": int(np.any(final != raw, axis=2)[protect].sum()),
        })
    video = OUT / "POKER_076_091_CLEAN_LAYER_TRACE.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-framerate", "10", "-i", str(OUT / "layer_review_frames/%06d.png"),
               "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", str(video)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("LAYER_REVIEW_ENCODE_FAILED:" + completed.stderr[-1000:])
    result = {
        "schema_version": "POKER_CLEAN_LAYER_ANALYSIS_V1",
        "task_id": TASK,
        "trace": artifact_ref(OUT / "TRACE_RESULT.json"),
        "actual_reference_steps": len(manifest["steps"]),
        "actual_selected_mask_slots": len(selected_rows),
        "all_selected_masks_nonempty": True,
        "all_frames_have_internal_output": True,
        "compositor_reintroduced_pixels_inside_write": reintroduced_total,
        "compositor_byte_exact_outside_write": all(row["outside_write_changed_pixels"] == 0 for row in rows),
        "compositor_byte_exact_protect": all(row["protected_changed_pixels"] == 0 for row in rows),
        "rows": rows,
        "review_video": artifact_ref(video),
        "failure_layer": "PENDING_FIXED_FRAME_VISUAL_DECISION",
        "next_action": "REVIEW_RAW_MASK_INTERNAL_PRED_INTERNAL_COMP_FINAL_PASTE_AND_RECORD_ONE_FAILURE_LAYER",
    }
    atomic_json(OUT / "LAYER_ANALYSIS.json", result)
    _state("LAYER_ANALYZED", next_action=result["next_action"], result=OUT / "LAYER_ANALYSIS.json")
    return {"analysis": str(OUT / "LAYER_ANALYSIS.json"),
            "review_video": str(video), "compositor_reintroduced": reintroduced_total}


def decide(failure_layer: str, evidence_frames: list[int]) -> dict:
    _registered()
    output = OUT / "DECISION.json"
    if output.exists():
        raise FileExistsError(output)
    analysis = load_json(OUT / "LAYER_ANALYSIS.json")
    if not evidence_frames or any(frame < 76 or frame > 91 for frame in evidence_frames):
        raise ValueError("DECISION_EVIDENCE_FRAMES")
    if failure_layer == "compositor" and analysis["compositor_reintroduced_pixels_inside_write"] == 0:
        raise RuntimeError("COMPOSITOR_ROUTE_CONTRADICTS_TRACE")
    if failure_layer == "internal_model" and not (
        analysis["all_selected_masks_nonempty"] and analysis["all_frames_have_internal_output"]
    ):
        raise RuntimeError("INTERNAL_MODEL_ROUTE_LACKS_INPUT_EVIDENCE")
    lama = _lama_gate()
    if failure_layer == "input_reference":
        next_action = "IMPLEMENT_ONE_PROPAINTER_INPUT_OR_REFERENCE_FIX"
        blocker = None
    elif failure_layer == "compositor":
        next_action = "IMPLEMENT_ONE_FINAL_PASTE_FIX_WITHOUT_RERUNNING_MODEL"
        blocker = None
    elif lama["admission"] == "PASS":
        next_action = "INSTALL_PINNED_PROJECT_LOCAL_ONNXRUNTIME_AND_IMPLEMENT_ONE_LAMA_RESIDUAL_HOLE_CANDIDATE"
        blocker = None
    else:
        next_action = "STOP_LAMA_BRANCH_UNTIL_WEIGHT_LICENSE_AND_SOURCE_PROOF_IS_PROVIDED"
        blocker = {
            "code": "BLOCKED_LOCAL_BACKEND_LICENSE_PROOF",
            "missing": "local license/source pin for assets/models/lama-onnx/lama_fp32.onnx",
            "consumer": "one LaMa residual-hole Clean candidate",
            "owner": "clean",
            "unblock": "add verified local asset provenance and license without replacing or downloading weights",
            "unaffected": ["hand_data", "robot", "delivery of rejected Clean evidence"],
        }
    record = {
        "schema_version": "POKER_CLEAN_SINGLE_ROUTE_DECISION_V1",
        "task_id": TASK,
        "failure_layer": failure_layer,
        "evidence_frames": sorted(set(evidence_frames)),
        "analysis": artifact_ref(OUT / "LAYER_ANALYSIS.json"),
        "lama_gate": lama,
        "next_action": next_action,
        "blocker": blocker,
        "candidate_budget_used": 0,
        "claim_limit": "Visual layer classification on the frozen 16-frame diagnostic only.",
    }
    atomic_json(output, record)
    _state("BLOCKED_UPSTREAM" if blocker else "ROUTE_FROZEN_READY_IMPLEMENTATION",
           next_action=next_action, result=output, blocker=blocker)
    return {"decision": str(output), "failure_layer": failure_layer,
            "next_action": next_action, "blocked": blocker is not None}


def candidate_prepare() -> dict:
    """Let direct current-frame human evidence override an amodal card mask."""
    _registered()
    decision = load_json(OUT / "DECISION.json")
    if decision["failure_layer"] != "input_reference":
        raise RuntimeError("CANDIDATE_ROUTE_NOT_INPUT_REFERENCE")
    if CANDIDATE.exists():
        raise FileExistsError(CANDIDATE)
    frozen = load_json(BASE / "INPUT.json")
    scene = load_json(SCENE_PREP)
    if len(scene.get("rows", [])) != 171:
        raise RuntimeError("SCENE_PREP_TIMELINE_DRIFT")
    for folder in ("model_masks", "write", "protect", "unknown"):
        (CANDIDATE / folder).mkdir(parents=True, exist_ok=False)
    rows = []
    source_map = []
    for local, base_row in enumerate(frozen["rows"]):
        frame = int(base_row["frame_id"])
        current_row = scene["rows"][frame]
        if frame != local + 76 or int(current_row["frame_id"]) != frame:
            raise RuntimeError("CANDIDATE_FRAME_MAP_DRIFT")
        direct_human = _image(Path(current_row["write"]), cv2.IMREAD_GRAYSCALE) > 0
        historical = _image(Path(base_row["historical_support"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        old_write = _image(Path(base_row["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        card = _image(Path(base_row["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        write, visible_card, unknown, conflict = build_visible_card_permissions(
            direct_human, historical, card)
        if np.any(write & visible_card) or np.any(write & unknown) or not np.any(write):
            raise RuntimeError(f"CANDIDATE_PERMISSION_CONFLICT:{frame}")
        # The repair must restore deletion permission precisely where the
        # current direct human support conflicts with the amodal card support.
        if not np.all(write[conflict]):
            raise RuntimeError(f"DIRECT_HUMAN_NOT_WRITABLE:{frame}")
        model = cv2.resize(write.astype(np.uint8), (960, 720),
                           interpolation=cv2.INTER_NEAREST) > 0
        paths = {key: CANDIDATE / key / f"{local:06d}.png"
                 for key in ("model_masks", "write", "protect", "unknown")}
        for key, value in (("model_masks", model), ("write", write),
                           ("protect", visible_card), ("unknown", unknown)):
            _save(paths[key], value.astype(np.uint8) * 255)
        source_map.append({
            "local_index": local,
            "source_frame": frame,
            "frame": base_row["frames"],
            "model_mask": artifact_ref(paths["model_masks"]),
        })
        rows.append({
            "local_index": local,
            "frame_id": frame,
            "raw": base_row["raw"],
            "model_frame": base_row["frames"],
            "current_direct_human_support": artifact_ref(Path(current_row["write"])),
            "historical_support": base_row["historical_support"],
            **{key: artifact_ref(path) for key, path in paths.items()},
            "old_write_px": int(old_write.sum()),
            "new_write_px": int(write.sum()),
            "card_human_conflict_to_write_px": int(conflict.sum()),
            "visible_card_protect_px": int(visible_card.sum()),
            "unknown_px": int(unknown.sum()),
        })
    restored = sum(row["card_human_conflict_to_write_px"] for row in rows)
    if restored <= 0 or not all(row["card_human_conflict_to_write_px"] > 0 for row in rows):
        raise RuntimeError("NO_CARD_HUMAN_CONFLICT_RESTORED")
    atomic_json(CANDIDATE / "SOURCE_MAP.json", source_map)
    record = {
        "schema_version": "POKER_VISIBLE_CARD_OCCLUSION_FIX_INPUT_V1",
        "task_id": TASK,
        "session_id": "play_cards_0902_042",
        "source_frame_range_inclusive": [76, 91],
        "frame_count": 16,
        "decision": artifact_ref(OUT / "DECISION.json"),
        "trace": artifact_ref(OUT / "LAYER_ANALYSIS.json"),
        "frozen_base": artifact_ref(BASE / "INPUT.json"),
        "scene_prep": artifact_ref(SCENE_PREP),
        "only_changed_variable": "DIRECT_CURRENT_HUMAN_SUPPORT_OVERRIDES_AMODAL_CARD_PROTECT",
        "permission_rule": {
            "support": "current_direct_human OR frozen_historical_support",
            "protect": "current_card_support AND NOT current_direct_human",
            "unknown": "protect_boundary AND support AND NOT current_direct_human",
            "write": "support AND NOT protect AND NOT unknown",
        },
        "model_parameters": frozen["model_parameters"],
        "restored_card_human_conflict_px": restored,
        "rows": rows,
        "quality": "INPUT_ONLY_NOT_CLEAN_PASS",
    }
    atomic_json(CANDIDATE / "INPUT.json", record)
    _state("CANDIDATE_INPUT_READY", next_action="RUN_ONE_FROZEN_PROPAINTER_INPUT_FIX_CANDIDATE",
           result=CANDIDATE / "INPUT.json")
    return {"input": str(CANDIDATE / "INPUT.json"), "frames": len(rows),
            "restored_card_human_conflict_px": restored}


def candidate_infer() -> dict:
    _registered()
    recipe = load_json(CANDIDATE / "INPUT.json")
    if (CANDIDATE / "RESULT.json").exists() or (CANDIDATE / "trace").exists():
        raise FileExistsError("CLEAN_CANDIDATE_EXISTS")
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease = load_json(lease_path)
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:clean"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("SHARED_CLEAN_GPU_LEASE_REQUIRED")
    trace_root = CANDIDATE / "trace"
    command = [
        sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
        "--video", str(BASE / "frames"), "--mask", str(CANDIDATE / "model_masks"),
        "--output", str(CANDIDATE / "upstream"), "--width", "960", "--height", "720",
        "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
        "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
        "--save_frames", "--fp16", "--trace_dir", str(trace_root),
        "--trace_source_map", str(CANDIDATE / "SOURCE_MAP.json"),
    ]
    atomic_json(CANDIDATE / "INVOCATION.json", {
        "schema_version": "POKER_VISIBLE_CARD_OCCLUSION_FIX_INVOCATION_V1",
        "task_id": TASK,
        "started_at": _now(),
        "input": artifact_ref(CANDIDATE / "INPUT.json"),
        "weights": [artifact_ref(VENDOR / "weights" / name) for name in (
            "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")],
        "vendor": artifact_ref(VENDOR / "inference_propainter.py"),
        "command": command,
        "lease_token": lease["fencing_token"],
        "quality_candidate_ordinal": 1,
    })
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(REPO_ROOT / "_run/cache"))
    with (CANDIDATE / "PROPAINTER.log").open("xb") as log:
        completed = subprocess.run(command, cwd=VENDOR, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"PROPAINTER_CANDIDATE_FAILED:{completed.returncode}")
    manifest = load_json(trace_root / "TRACE_MANIFEST.json")
    outputs = sorted((CANDIDATE / "upstream/frames/frames").glob("*.png"))
    if manifest.get("video_length") != 16 or len(outputs) != 16:
        raise RuntimeError("CANDIDATE_OUTPUT_COVERAGE")
    (CANDIDATE / "clean").mkdir()
    rows = []
    for row, generated_path in zip(recipe["rows"], outputs, strict=True):
        raw = _image(Path(row["raw"]["path"]))
        generated = cv2.resize(_image(generated_path), (1280, 960), interpolation=cv2.INTER_LINEAR)
        write = _image(Path(row["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        protect = _image(Path(row["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        clean = composite_clean(raw, generated, write, protect)
        changed = np.any(clean != raw, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError(f"CANDIDATE_WRITE_BOUNDARY:{row['frame_id']}")
        path = CANDIDATE / "clean" / f"{row['frame_id']:06d}.png"
        _save(path, clean)
        rows.append({"frame_id": row["frame_id"], "clean": artifact_ref(path),
                     "changed_px": int(changed.sum()), "outside_write_changed_px": 0,
                     "protected_changed_px": 0})
    result = {
        "schema_version": "POKER_VISIBLE_CARD_OCCLUSION_FIX_RESULT_V1",
        "task_id": TASK,
        "execution": "REAL_PROPAINTER_SINGLE_QUALITY_CANDIDATE",
        "structure": "PASS",
        "quality": "PENDING_FIXED_WINDOW_REVIEW",
        "adoption": "NOT_ADOPTED",
        "input": artifact_ref(CANDIDATE / "INPUT.json"),
        "invocation": artifact_ref(CANDIDATE / "INVOCATION.json"),
        "trace_manifest": artifact_ref(trace_root / "TRACE_MANIFEST.json"),
        "log": artifact_ref(CANDIDATE / "PROPAINTER.log"),
        "rows": rows,
        "claim_limit": "16-frame Poker input-permission fix; no full-session claim.",
        "next_action": "INDEPENDENT_FIXED_WINDOW_REVIEW_THEN_CONDITIONAL_FULL_171",
    }
    atomic_json(CANDIDATE / "RESULT.json", result)
    _state("CANDIDATE_EXECUTED", next_action=result["next_action"],
           result=CANDIDATE / "RESULT.json")
    return {"result": str(CANDIDATE / "RESULT.json"), "frames": len(rows)}


def candidate_review(quality: str, observations: list[str]) -> dict:
    _registered()
    if quality not in {"PASS", "REJECTED_QUALITY"}:
        raise ValueError("CANDIDATE_REVIEW_QUALITY")
    if not observations:
        raise ValueError("CANDIDATE_REVIEW_OBSERVATIONS")
    output = CANDIDATE / "VISUAL_REVIEW.json"
    if output.exists():
        raise FileExistsError(output)
    recipe = load_json(CANDIDATE / "INPUT.json")
    (CANDIDATE / "review_frames").mkdir()
    for local, row in enumerate(recipe["rows"]):
        raw = _image(Path(row["raw"]["path"]))
        old = _image(BASE / "clean" / f"{row['frame_id']:06d}.png")
        new = _image(CANDIDATE / "clean" / f"{row['frame_id']:06d}.png")
        old_mask = _image(BASE / "model_masks" / f"{local:06d}.png", cv2.IMREAD_GRAYSCALE)
        new_mask = _image(CANDIDATE / "model_masks" / f"{local:06d}.png", cv2.IMREAD_GRAYSCALE)
        conflict = (_image(Path(row["current_direct_human_support"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
                    ) & (_image(BASE / "protect" / f"{local:06d}.png", cv2.IMREAD_GRAYSCALE) > 0)
        conflict_view = cv2.resize(raw, (960, 720), interpolation=cv2.INTER_AREA)
        conflict_small = cv2.resize(conflict.astype(np.uint8), (960, 720),
                                    interpolation=cv2.INTER_NEAREST) > 0
        conflict_view[conflict_small] = (
            .35 * conflict_view[conflict_small] + .65 * np.array([0, 255, 255])
        ).astype(np.uint8)
        canvas = np.vstack((
            np.hstack((_panel(raw, f"RAW {row['frame_id']}"), _panel(old, "OLD REJECTED"))),
            np.hstack((_panel(old_mask, "OLD MODEL MASK"), _panel(new_mask, "NEW MODEL MASK"))),
            np.hstack((_panel(conflict_view, "HUMAN/CARD CONFLICT -> WRITE"),
                       _panel(new, "NEW CANDIDATE"))),
        ))
        _save(CANDIDATE / "review_frames" / f"{local:06d}.png", canvas)
    video = CANDIDATE / "POKER_076_091_VISIBLE_CARD_OCCLUSION_FIX_REVIEW.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-framerate", "10", "-i", str(CANDIDATE / "review_frames/%06d.png"),
               "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", str(video)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("CANDIDATE_REVIEW_ENCODE_FAILED:" + completed.stderr[-1000:])
    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != 16:
        raise RuntimeError("CANDIDATE_REVIEW_DECODE_COUNT")
    record = {
        "schema_version": "POKER_VISIBLE_CARD_OCCLUSION_FIX_REVIEW_V1",
        "task_id": TASK,
        "candidate": artifact_ref(CANDIDATE / "RESULT.json"),
        "video": artifact_ref(video),
        "decoded_frames": count,
        "reviewed_fixed_frames": [76, 80, 84, 91],
        "observations": observations,
        "quality": quality,
        "adoption": "NOT_ADOPTED",
        "full_171": "READY_CONDITIONAL_EXPANSION" if quality == "PASS" else
                    "NOT_RUN_FIXED_WINDOW_REJECTED",
        "next_action": "EXPAND_SAME_RECIPE_TO_171" if quality == "PASS" else
                       "STOP_CLEAN_RECIPE_AND_DELIVER_REJECTION",
        "review_limit": "fixed 16-frame visual review; not a full-session pass",
    }
    atomic_json(output, record)
    _state("CANDIDATE_WINDOW_PASS" if quality == "PASS" else "REJECTED_QUALITY",
           next_action=record["next_action"], result=output)
    return {"review": str(output), "video": str(video), "quality": quality,
            "full_171": record["full_171"]}


def freeze_internal_model_route() -> dict:
    """Supersede the coarse first decision without rewriting its evidence."""
    _registered()
    if DECISION_V2.exists():
        raise FileExistsError(DECISION_V2)
    analysis = load_json(OUT / "LAYER_ANALYSIS.json")
    if analysis["compositor_reintroduced_pixels_inside_write"] != 0:
        raise RuntimeError("INTERNAL_ROUTE_COMPOSITOR_NOT_EXCLUDED")
    gate = _lama_gate()
    if gate["admission"] != "PASS":
        raise RuntimeError("LAMA_ASSET_NOT_ADMITTED")
    record = {
        "schema_version": "POKER_CLEAN_SINGLE_ROUTE_DECISION_V2",
        "task_id": TASK,
        "supersedes_decision": artifact_ref(OUT / "DECISION.json"),
        "failure_layer": "internal_model",
        "evidence_frames": [80, 84, 91],
        "evidence": (
            "The right-hand/card-edge residue remains inside the red contour of the actual "
            "model mask in internal prediction/composition; final paste reintroduced zero "
            "pixels inside M_write."
        ),
        "lama_gate": gate,
        "only_candidate": "LAMA_ON_PROPAGATION_RESIDUAL_MASK",
        "candidate_budget_used": 0,
        "next_action": "INSTALL_PINNED_PROJECT_LOCAL_ONNXRUNTIME_THEN_RUN_ONE_LAMA_RESIDUAL_CANDIDATE",
    }
    atomic_json(DECISION_V2, record)
    _state("ROUTE_FROZEN_READY_RUNTIME", next_action=record["next_action"], result=DECISION_V2)
    return {"decision": str(DECISION_V2), "failure_layer": "internal_model"}


def install_lama_runtime() -> dict:
    _registered()
    if _lama_gate()["admission"] != "PASS":
        raise RuntimeError("LAMA_ASSET_NOT_ADMITTED")
    receipt = ORT_ROOT / "INSTALL_RECEIPT.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    ORT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = ATTEMPT / "cache/pip"
    cache.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "install", "--target", str(ORT_ROOT),
               "--cache-dir", str(cache), "--only-binary=:all:", "onnxruntime==1.22.1"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("ONNXRUNTIME_INSTALL_FAILED:" + completed.stderr[-1500:])
    metadata = sorted(ORT_ROOT.glob("*.dist-info/METADATA"))
    rows = []
    for path in metadata:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()
        name = next((line[6:] for line in first if line.startswith("Name: ")), path.parent.name)
        version = next((line[9:] for line in first if line.startswith("Version: ")), "UNKNOWN")
        rows.append({"name": name, "version": version, "metadata_sha256": digest})
    if not any(row["name"].lower() == "onnxruntime" and row["version"] == "1.22.1" for row in rows):
        raise RuntimeError("ONNXRUNTIME_VERSION_NOT_PINNED")
    atomic_json(receipt, {"schema_version": "PROJECT_LOCAL_ORT_RUNTIME_V1",
                          "task_id": TASK, "command": command, "packages": rows,
                          "root": str(ORT_ROOT), "asset_pin": artifact_ref(LAMA_PIN)})
    _state("LAMA_RUNTIME_READY", next_action="RUN_ONE_LAMA_RESIDUAL_CANDIDATE", result=receipt)
    return {"receipt": str(receipt), "packages": len(rows)}


def _lama_session():
    if not (ORT_ROOT / "INSTALL_RECEIPT.json").exists():
        raise RuntimeError("LAMA_RUNTIME_NOT_INSTALLED")
    sys.path.insert(0, str(ORT_ROOT))
    import onnxruntime as ort  # type: ignore
    session = ort.InferenceSession(str(LAMA), providers=["CPUExecutionProvider"])
    inputs = {value.name: value for value in session.get_inputs()}
    outputs = session.get_outputs()
    if set(inputs) != {"image", "mask"} or len(outputs) != 1:
        raise RuntimeError("LAMA_ONNX_IO_NAMES")
    if not _lama_shape_matches(inputs["image"].shape, 3) or inputs["image"].type != "tensor(float)":
        raise RuntimeError("LAMA_ONNX_IMAGE_CONTRACT")
    if not _lama_shape_matches(inputs["mask"].shape, 1) or inputs["mask"].type != "tensor(float)":
        raise RuntimeError("LAMA_ONNX_MASK_CONTRACT")
    if not _lama_shape_matches(outputs[0].shape, 3) or outputs[0].type != "tensor(float)":
        raise RuntimeError("LAMA_ONNX_OUTPUT_CONTRACT")
    return session, outputs[0].name


def _lama_shape_matches(shape, channels: int) -> bool:
    value = list(shape)
    return (len(value) == 4 and value[0] in (1, "batch")
            and value[1:] == [channels, 512, 512])


def _apply_lama_residual(session, output_name: str, propagated: np.ndarray,
                         residual: np.ndarray) -> np.ndarray:
    if propagated.shape != (720, 960, 3) or residual.shape != (720, 960):
        raise ValueError("LAMA_RESIDUAL_DOMAIN")
    mask = residual > 0
    if not mask.any():
        return propagated.copy()
    image_512 = cv2.resize(propagated, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
    image_tensor = (cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)
                    .transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    prediction = session.run([output_name], {"image": image_tensor,
                                              "mask": mask_512[None, None]})[0]
    if prediction.shape != (1, 3, 512, 512) or not np.isfinite(prediction).all():
        raise RuntimeError("LAMA_OUTPUT_INVALID")
    predicted = prediction[0].transpose(1, 2, 0)
    predicted = cv2.cvtColor(np.clip(predicted, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    predicted = cv2.resize(predicted, (960, 720), interpolation=cv2.INTER_LINEAR)
    output = propagated.copy()
    output[mask] = predicted[mask]
    return output


def lama_candidate_infer() -> dict:
    _registered()
    decision = load_json(DECISION_V2)
    if decision["failure_layer"] != "internal_model":
        raise RuntimeError("CANDIDATE_ROUTE_NOT_INTERNAL_MODEL")
    if CANDIDATE.exists():
        raise FileExistsError(CANDIDATE)
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:clean"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("SHARED_CLEAN_GPU_LEASE_REQUIRED")
    CANDIDATE.mkdir(parents=True)
    trace_root = CANDIDATE / "propagation_trace"
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(BASE / "frames"), "--mask", str(BASE / "model_masks"),
               "--output", str(CANDIDATE / "propainter"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16", "--trace_dir", str(trace_root),
               "--trace_source_map", str(SOURCE_MAP)]
    atomic_json(CANDIDATE / "INVOCATION.json", {"schema_version": "POKER_LAMA_RESIDUAL_INVOCATION_V1",
                 "task_id": TASK, "decision": artifact_ref(DECISION_V2), "command": command,
                 "vendor": artifact_ref(VENDOR / "inference_propainter.py"),
                 "lama": artifact_ref(LAMA), "asset_pin": artifact_ref(LAMA_PIN),
                 "runtime": artifact_ref(ORT_ROOT / "INSTALL_RECEIPT.json"),
                 "lease_token": lease["fencing_token"], "quality_candidate_ordinal": 1})
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(REPO_ROOT / "_run/cache"))
    with (CANDIDATE / "PROPAINTER_PROPAGATION.log").open("xb") as log:
        completed = subprocess.run(command, cwd=VENDOR, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"PROPAINTER_PROPAGATION_FAILED:{completed.returncode}")
    manifest = load_json(trace_root / "TRACE_MANIFEST.json")
    propagation = manifest.get("propagation_rows", [])
    if len(propagation) != 16 or any(row["residual_mask_tensor"]["nonzero"] <= 0 for row in propagation):
        raise RuntimeError("PROPAGATION_RESIDUAL_NOT_AVAILABLE")
    session, output_name = _lama_session()
    frozen = load_json(BASE / "INPUT.json")
    (CANDIDATE / "lama_internal").mkdir()
    (CANDIDATE / "clean").mkdir()
    rows = []
    for prop, row in zip(propagation, frozen["rows"], strict=True):
        generated = _apply_lama_residual(
            session, output_name, _image(trace_root / prop["propagated_frame"]),
            _image(trace_root / prop["residual_propagation_mask"], cv2.IMREAD_GRAYSCALE))
        internal_path = CANDIDATE / "lama_internal" / f"{row['frame_id']:06d}.png"
        _save(internal_path, generated)
        raw = _image(Path(row["raw"]["path"]))
        write = _image(Path(row["write"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        protect = _image(Path(row["protect"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
        clean = composite_clean(raw, cv2.resize(generated, (1280, 960)), write, protect)
        clean_path = CANDIDATE / "clean" / f"{row['frame_id']:06d}.png"
        _save(clean_path, clean)
        changed = np.any(clean != raw, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError(f"LAMA_WRITE_BOUNDARY:{row['frame_id']}")
        rows.append({"frame_id": row["frame_id"], "propagation": prop,
                     "lama_internal": artifact_ref(internal_path), "clean": artifact_ref(clean_path),
                     "changed_px": int(changed.sum())})
    result = {"schema_version": "POKER_LAMA_RESIDUAL_RESULT_V1", "task_id": TASK,
              "execution": "REAL_LAMA_ON_PROPAGATION_RESIDUAL", "structure": "PASS",
              "quality": "PENDING_FIXED_WINDOW_REVIEW", "adoption": "NOT_ADOPTED",
              "invocation": artifact_ref(CANDIDATE / "INVOCATION.json"), "rows": rows,
              "next_action": "INDEPENDENT_FIXED_WINDOW_REVIEW_THEN_CONDITIONAL_FULL_171"}
    atomic_json(CANDIDATE / "RESULT.json", result)
    _state("CANDIDATE_EXECUTED", next_action=result["next_action"], result=CANDIDATE / "RESULT.json")
    return {"result": str(CANDIDATE / "RESULT.json"), "frames": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("preflight")
    sub.add_parser("trace")
    sub.add_parser("analyze")
    sub.add_parser("freeze-internal-route")
    sub.add_parser("install-lama-runtime")
    sub.add_parser("lama-candidate-infer")
    decision = sub.add_parser("decide")
    decision.add_argument("--failure-layer", required=True,
                          choices=("input_reference", "compositor", "internal_model"))
    decision.add_argument("--evidence-frame", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "trace":
        result = trace()
    elif args.stage == "analyze":
        result = analyze()
    elif args.stage == "freeze-internal-route":
        result = freeze_internal_model_route()
    elif args.stage == "install-lama-runtime":
        result = install_lama_runtime()
    elif args.stage == "lama-candidate-infer":
        result = lama_candidate_infer()
    else:
        result = decide(args.failure_layer, args.evidence_frame)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
