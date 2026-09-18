#!/usr/bin/env python3
"""Run the bounded SAM3.1 weak-role successor for ``play_cards_0915_001``.

Only weak v5 roles are sent through the model.  The accepted v5 hand masks and
the first two card masks are loaded as SHA-guarded, read-only regression
evidence.  The parent process is route-gated and launches its worker only
through the repository's central GPU lease wrapper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import run_0915_sam31_strict_role_canary_v1 as base
from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module
from chaoyang.pipeline.sam31_0915_weak_role_contract_v1 import (
    FRAME_COUNT,
    FROZEN_WEAK_ROLE_PROMPTS,
    HEIGHT,
    IMAGE_DOMAIN,
    MODEL_WEIGHT,
    PHASE,
    REGRESSION_INPUT_POLICY,
    REGRESSION_INSTANCE_IDS,
    SESSION_ID,
    TARGET_INSTANCE_IDS,
    TASK_ID,
    WEIGHTS,
    WIDTH,
    build_prompt_plan,
    regression_paths,
)


ROOT = Path(__file__).resolve().parents[3]
VIDEO = base.VIDEO
HAWOR = base.HAWOR
CHECKPOINT = base.CHECKPOINT
CHECKPOINT_SHA256 = base.CHECKPOINT_SHA256
CODE_ROOT = base.CODE_ROOT
REGRESSION_ROOT = (
    ROOT / "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
)
OUTPUT_NAMESPACE = ROOT / "_run/current" / TASK_ID
CENTRAL_GPU_LEASE = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"

COLORS = {
    "left_hand_00": (255, 180, 40),
    "right_hand_00": (30, 90, 255),
    "playing_card_00": (60, 255, 60),
    "playing_card_01": (255, 80, 80),
    "left_forearm_00": (255, 230, 70),
    "right_forearm_00": (30, 210, 255),
    "left_finger_sleeve_visible_00": (255, 80, 220),
    "left_finger_sleeve_visible_01": (230, 70, 245),
    "left_finger_sleeve_visible_02": (205, 60, 255),
    "left_finger_sleeve_visible_03": (180, 50, 255),
    "right_finger_sleeve_visible_00": (160, 100, 255),
    "right_finger_sleeve_visible_01": (145, 75, 235),
    "right_finger_sleeve_visible_02": (130, 55, 220),
    "right_finger_sleeve_visible_03": (115, 40, 205),
    "left_yellow_cable_visible_00": (0, 220, 255),
    "right_yellow_cable_visible_00": (0, 150, 255),
    "playing_card_02": (255, 60, 200),
}


def validate_output_namespace(output: Path) -> None:
    resolved = output.resolve()
    if not resolved.is_relative_to(OUTPUT_NAMESPACE.resolve()):
        raise RuntimeError(f"output must stay inside the task namespace: {OUTPUT_NAMESPACE}")


def snapshot_regression_inputs(regression_root: Path) -> dict[str, dict[str, Any]]:
    """Hash the complete v5 regression read set without creating any files."""

    root = regression_root.resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for path in regression_paths(root):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError("regression path escaped its immutable root")
        result[str(resolved.relative_to(root))] = base.ref(resolved)
    return result


def assert_regression_inputs_unchanged(
    before: dict[str, dict[str, Any]], regression_root: Path,
) -> None:
    after = snapshot_regression_inputs(regression_root)
    if after != before:
        raise RuntimeError("read-only v5 regression evidence changed during weak-role run")


def _load_packed(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["frame_count"]) != FRAME_COUNT:
            raise RuntimeError(f"regression frame-count drift: {path}")
        if int(archive["height"]) != HEIGHT or int(archive["width"]) != WIDTH:
            raise RuntimeError(f"regression image-domain drift: {path}")
        if str(archive["bitorder"]) != "big":
            raise RuntimeError(f"regression bitorder drift: {path}")
        packed = np.asarray(archive["packed"], np.uint8)
    expected = (FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8)
    if packed.shape != expected:
        raise RuntimeError(f"regression packed-mask shape drift: {path}")
    return packed


def _unpack(packed: np.ndarray, frame: int) -> np.ndarray:
    return np.unpackbits(
        packed[frame], bitorder="big", count=HEIGHT * WIDTH,
    ).reshape(HEIGHT, WIDTH).astype(bool)


def load_regression_evidence(
    regression_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    manifest = base.load_json(regression_root / "ROLE_MANIFEST.json")
    temporal = base.load_json(regression_root / "TEMPORAL_STATE_LEDGER.json")
    manifest_rows = {row["instance_id"]: row for row in manifest["instances"]}
    temporal_rows = {row["instance_id"]: row["frames"] for row in temporal["instances"]}
    packed: dict[str, np.ndarray] = {}
    states: dict[str, list[dict[str, Any]]] = {}
    evidence_rows = []
    for instance_id in REGRESSION_INSTANCE_IDS:
        row = manifest_rows.get(instance_id)
        frames = temporal_rows.get(instance_id)
        if row is None or not isinstance(frames, list) or len(frames) != FRAME_COUNT:
            raise RuntimeError(f"missing v5 regression instance: {instance_id}")
        archive = regression_root / row["mask_archive"]
        packed[instance_id] = _load_packed(archive)
        states[instance_id] = frames
        evidence_rows.append({
            "instance_id": instance_id,
            "role": row["role"],
            "source": "V5_READ_ONLY_REGRESSION_EVIDENCE",
            "mask_archive": base.ref(archive),
            "state_counts": row["state_counts"],
            "initial_box_prompt": row["initial_box_prompt"],
            "reseed_count": row["reseed_count"],
        })
    return packed, states, evidence_rows


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = base.load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from its frozen specification")
    if packet.get("weights") != list(WEIGHTS) or len(packet["weights"]) != 1:
        raise RuntimeError("weak-role canary must bind exactly one SAM3.1 weight")
    state = base.load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("weak-role canary is not current next_task")
    task = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID), None,
    )
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE",
    }:
        raise RuntimeError("weak-role canary is not executable")
    index = base.load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize this canary")
    if route.get("packet_sha256") != base.sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat(status: str = "RUNNING") -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", status, "--phase", PHASE,
        *(["--gpu-id", "0"] if status == "RUNNING" else []),
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def _open_encoder(path: Path) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x480",
        "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ], stdin=subprocess.PIPE)


def render_review(
    *, video: Path, packed_by_id: dict[str, np.ndarray],
    states_by_id: dict[str, list[dict[str, Any]]], destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    encoder = _open_encoder(destination)
    assert encoder.stdin is not None
    frame = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            overlay = raw.copy()
            for instance_id, packed in packed_by_id.items():
                mask = _unpack(packed, frame)
                if mask.any():
                    color = np.asarray(COLORS[instance_id], np.float32)
                    overlay[mask] = (
                        0.48 * overlay[mask].astype(np.float32) + 0.52 * color
                    ).astype(np.uint8)
            canvas = np.hstack((
                cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA),
                cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA),
            ))
            cv2.rectangle(canvas, (0, 0), (1279, 72), (0, 0, 0), -1)
            unknown = sum(
                states_by_id[name][frame]["state"] == "unknown"
                for name in states_by_id
            )
            reseeded = [
                name for name in TARGET_INSTANCE_IDS
                if states_by_id[name][frame]["state"] == "reseeded"
            ]
            cv2.putText(
                canvas,
                f"{SESSION_ID} | resize-only | SAM3.1 weak roles | frame {frame:03d}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"left=RGB | right=weak targets + v5 regression | unknown={unknown}/{len(states_by_id)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"quality-triggered reseed={','.join(reseeded) or '-'} | empty means unknown, not absent",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1,
                cv2.LINE_AA,
            )
            encoder.stdin.write(canvas.tobytes())
            frame += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frame != FRAME_COUNT:
        raise RuntimeError("weak-role review encoding failed")
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(destination),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise RuntimeError("weak-role review full-decode failed")
    return {
        "video": base.ref(destination), "full_decode": True,
        "frame_count": FRAME_COUNT, "geometry": [1280, 480],
    }


def run_worker(
    output: Path, visual: Path, video: Path, hawor: Path,
    regression_root: Path,
) -> int:
    if base.sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    regression_before = snapshot_regression_inputs(regression_root)
    regression_packed, regression_states, regression_rows = load_regression_evidence(
        regression_root,
    )
    gate = base.video_gate(video)
    with np.load(hawor, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
    if joints.shape != (2, FRAME_COUNT, 21, 2) or observed.shape != (2, FRAME_COUNT):
        raise RuntimeError("accepted bounded HaWoR shape drift")
    if sides != ["left", "right"]:
        raise RuntimeError("bounded HaWoR anatomical side order drift")

    frames = output / "input_frames"
    base.extract_frames(video, frames)
    regression_hashes = {
        name: row["sha256"] for name, row in regression_before.items()
    }
    plan = build_prompt_plan(
        video_sha256=base.sha256(video), hawor_sha256=base.sha256(hawor),
        regression_sha256=regression_hashes,
    )
    base.atomic_json(output / "PROMPT_PLAN.json", plan)

    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT,
    )
    packed_by_id = dict(regression_packed)
    states_by_id = dict(regression_states)
    quality_instances = []
    target_rows = []
    human_equipment_union = np.zeros((FRAME_COUNT, HEIGHT, WIDTH), bool)
    for instance_id in ("left_hand_00", "right_hand_00"):
        for frame in range(FRAME_COUNT):
            human_equipment_union[frame] |= _unpack(regression_packed[instance_id], frame)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index, prompt in enumerate(FROZEN_WEAK_ROLE_PROMPTS, start=1):
            masks, ledger, evidence = base.process_instance(
                adapter.model, frames, prompt, joints, observed,
            )
            archive_path = output / "masks" / f"{prompt.instance_id}.npz"
            packed_by_id[prompt.instance_id] = base.save_packed(archive_path, masks)
            states_by_id[prompt.instance_id] = ledger
            if prompt.role != "task_object":
                human_equipment_union |= masks
            counts = Counter(row["state"] for row in ledger)
            state_counts = {
                name: int(counts.get(name, 0))
                for name in ("seeded", "tracked", "reseeded", "unknown")
            }
            target_rows.append({
                "instance_id": prompt.instance_id,
                "role": prompt.role,
                "source": "SAM3.1_WEAK_ROLE_TARGET",
                "mask_archive": str(archive_path.relative_to(output)),
                "state_counts": state_counts,
                "initial_visual_box": {
                    "frame_index": prompt.primary.frame_index,
                    "box_xywh": list(prompt.primary.box_xywh),
                    "runtime_semantics": "MULTIPLEX_GEOMETRIC_BOX",
                },
                "reseed_count": int(evidence["reseed_triggered"]),
                "physical_identity_policy": prompt.physical_identity_policy,
            })
            quality_instances.append(evidence)
            print(json.dumps({
                "completed_targets": index,
                "total_targets": len(FROZEN_WEAK_ROLE_PROMPTS),
                "instance_id": prompt.instance_id,
                "states": state_counts,
                "reseed_triggered": evidence["reseed_triggered"],
            }, ensure_ascii=False), flush=True)
            del masks
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        else:
            adapter.predictor.shutdown()

    union_path = output / "masks/HUMAN_EQUIPMENT_UNION.npz"
    base.save_packed(union_path, human_equipment_union)
    temporal = {
        "schema_version": "sam31-weak-role-temporal-state-ledger-v1",
        "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT,
        "state_semantics": {
            "seeded": "primary initial visual box is quality-admissible",
            "tracked": "quality-admissible propagation from primary or fallback",
            "reseeded": "fallback visual box after a quality trigger",
            "unknown": "no quality-admissible mask; not evidence of absence",
        },
        "instances": [
            {
                "instance_id": name,
                "source": (
                    "V5_READ_ONLY_REGRESSION_EVIDENCE"
                    if name in REGRESSION_INSTANCE_IDS else "SAM3.1_WEAK_ROLE_TARGET"
                ),
                "frames": states_by_id[name],
            }
            for name in (*REGRESSION_INSTANCE_IDS, *TARGET_INSTANCE_IDS)
        ],
        "empty_mask_semantics": "UNKNOWN_NOT_ABSENT",
    }
    base.atomic_json(output / "TEMPORAL_STATE_LEDGER.json", temporal)
    base.atomic_json(output / "QUALITY_TRIGGER_LEDGER.json", {
        "schema_version": "sam31-weak-role-quality-trigger-ledger-v1",
        "session_id": SESSION_ID,
        "policy": "QUALITY_TRIGGERED_NOT_PERIODIC_MAX_ONE_RESEED",
        "target_instances": quality_instances,
        "regression_instances_recomputed": False,
    })
    manifest = {
        "schema_version": "sam31-weak-role-canary-manifest-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT,
        "image_domain": IMAGE_DOMAIN,
        "model": {
            "identity": "SAM3.1",
            "weight": MODEL_WEIGHT,
            "weight_sha256": CHECKPOINT_SHA256,
            "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
            "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
        },
        "regression_input_policy": REGRESSION_INPUT_POLICY,
        "regression_instances": regression_rows,
        "target_instances": target_rows,
        "human_equipment_union": {
            "mask_archive": str(union_path.relative_to(output)),
            "direction": "DERIVED_FROM_V5_HANDS_AND_WEAK_TARGET_MASKS_ONLY",
        },
        "temporal_state_ledger": "TEMPORAL_STATE_LEDGER.json",
        "tracker_role_created": False,
        "controller_role_created": False,
        "pico26_consumed": False,
    }
    base.atomic_json(output / "WEAK_ROLE_MANIFEST.json", manifest)
    review = render_review(
        video=video, packed_by_id=packed_by_id, states_by_id=states_by_id,
        destination=visual / "0915_SAM31_WEAK_ROLE_REVIEW.mp4",
    )
    assert_regression_inputs_unchanged(regression_before, regression_root)
    base.atomic_json(output / "REGRESSION_GATES.json", {
        "schema_version": "0915-sam31-weak-role-regression-gates-v1",
        "status": "PASS",
        "policy": REGRESSION_INPUT_POLICY,
        "instances": list(REGRESSION_INSTANCE_IDS),
        "before": regression_before,
        "after": snapshot_regression_inputs(regression_root),
        "recomputed": False,
        "byte_identity_preserved": True,
    })
    worker_result = {
        "schema_version": "0915-sam31-weak-role-worker-result-v1",
        "status": "COMPLETED_DEVELOPMENT_CANARY",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT,
        "image_domain": IMAGE_DOMAIN,
        "video_gate": gate,
        "inputs": {
            "video": base.ref(video), "hawor": base.ref(hawor),
            "checkpoint": base.ref(CHECKPOINT),
            "v5_regression_root": str(regression_root.resolve()),
            "v5_regression_snapshot": regression_before,
        },
        "access_contract": {
            "allowed_algorithm_inputs": [
                str(video), str(hawor), str(CHECKPOINT),
                *(row["path"] for row in regression_before.values()),
            ],
            "regression_inputs": REGRESSION_INPUT_POLICY,
            "pico26": "NOT_CONSUMED", "controller_pose": "NOT_CONSUMED",
            "trackingData_hand": "NOT_CONSUMED", "source_tree_mutated": False,
        },
        "prompt_plan": base.ref(output / "PROMPT_PLAN.json"),
        "weak_role_manifest": base.ref(output / "WEAK_ROLE_MANIFEST.json"),
        "temporal_state_ledger": base.ref(output / "TEMPORAL_STATE_LEDGER.json"),
        "quality_trigger_ledger": base.ref(output / "QUALITY_TRIGGER_LEDGER.json"),
        "regression_gates": base.ref(output / "REGRESSION_GATES.json"),
        "review": review,
        "build_evidence": build_evidence,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "claim_limit": plan["claim_limit"],
    }
    base.atomic_json(output / "SAM31_WEAK_ROLE_WORKER_RESULT.json", worker_result)
    shutil.rmtree(frames)
    print(json.dumps({
        "status": worker_result["status"],
        "review": review["video"]["path"],
    }, ensure_ascii=False), flush=True)
    return 0


def write_terminal(
    output: Path, visual: Path, receipt: Path, packet: dict[str, Any],
    *, status: str, first_blocker: str | None, worker: dict[str, Any] | None,
) -> None:
    result = {
        "schema_version": "0915-sam31-weak-role-canary-result-v1",
        "task_id": TASK_ID,
        "status": status,
        "session_id": SESSION_ID,
        "weights": packet["weights"],
        "image_domain": IMAGE_DOMAIN,
        "session_admission": (
            "AWAITING_USER_VISUAL_REVIEW" if worker is not None else "NOT_PRODUCED"
        ),
        "first_blocker": first_blocker,
        "worker_result": base.ref(output / "SAM31_WEAK_ROLE_WORKER_RESULT.json")
        if worker is not None else None,
        "gpu_command_receipt": base.ref(output / "GPU_COMMAND_RECEIPT.json")
        if (output / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "visual": base.ref(visual / "0915_SAM31_WEAK_ROLE_REVIEW.mp4")
        if (visual / "0915_SAM31_WEAK_ROLE_REVIEW.mp4").is_file() else None,
        "source_mutated": False,
        "batch_started": False,
        "regression_instances_recomputed": False,
        "tracker_role_created": False,
        "controller_role_created": False,
        "pico26_consumed": False,
        "next_authority": "USER_VISUAL_REVIEW_REQUIRED_NO_AUTOMATIC_BATCH_SUCCESSOR",
        "claim_limit": packet["claim_limit"],
    }
    base.atomic_json(output / "RESULT.json", result)
    base.atomic_json(receipt, {**result, "result": base.ref(output / "RESULT.json")})
    base.atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-sam31-weak-role-canary-run-receipt-v1",
        "task_id": TASK_ID, "status": status,
        "result": base.ref(output / "RESULT.json"),
        "terminal_receipt": base.ref(receipt),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet = validate_route()
    output = args.output_root.resolve()
    validate_output_namespace(output)
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    for path in (VIDEO, HAWOR, CHECKPOINT, CENTRAL_GPU_LEASE):
        if not path.is_file():
            raise RuntimeError(f"fixed canary input is missing: {path}")
    snapshot_regression_inputs(REGRESSION_ROOT)
    output.mkdir(parents=True)
    heartbeat("WAIT_GPU_RESOURCE")
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output-root", str(output), "--visual-root", str(visual),
        "--video", str(VIDEO), "--hawor", str(HAWOR),
        "--regression-root", str(REGRESSION_ROOT),
    ]
    lease_command = [
        sys.executable, str(CENTRAL_GPU_LEASE),
        "--task-id", TASK_ID, "--attempt-id", output.name,
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds), "--wall-seconds", "10800",
        "--receipt", str(gpu_receipt), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    base.atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-sam31-weak-role-command-v1",
        "task_id": TASK_ID,
        "weights": packet["weights"],
        "worker_command": worker_command,
        "lease_command": lease_command,
        "regression_input_policy": REGRESSION_INPUT_POLICY,
    })
    with (output / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            time.sleep(30)
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = base.load_json(lease_path) if lease_path.is_file() else {}
            acquired = (
                lease.get("status") == "ACQUIRED"
                and lease.get("task_id") == TASK_ID
                and lease.get("attempt_id") == output.name
            )
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE")
    gpu = base.load_json(gpu_receipt) if gpu_receipt.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        blocked = gpu.get("status") == "BLOCKED_RESOURCE"
        write_terminal(
            output, visual, receipt, packet,
            status="BLOCKED_RESOURCE" if blocked else "FAILED_RUNTIME_FINAL",
            first_blocker=str(
                gpu.get("reason") or gpu.get("error") or "SAM31_WEAK_ROLE_RUNTIME_FAILED"
            ),
            worker=None,
        )
        return 3 if blocked else 2
    worker = base.load_json(output / "SAM31_WEAK_ROLE_WORKER_RESULT.json")
    write_terminal(
        output, visual, receipt, packet,
        status="PASSED", first_blocker=None, worker=worker,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--hawor", type=Path)
    parser.add_argument("--regression-root", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.video is None or args.hawor is None or args.regression_root is None:
            raise RuntimeError("worker requires --video, --hawor, and --regression-root")
        validate_output_namespace(args.output_root)
        return run_worker(
            args.output_root.resolve(), args.visual_root.resolve(),
            args.video.resolve(strict=True), args.hawor.resolve(strict=True),
            args.regression_root.resolve(strict=True),
        )
    if args.receipt is None:
        raise RuntimeError("orchestrator requires --receipt")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
