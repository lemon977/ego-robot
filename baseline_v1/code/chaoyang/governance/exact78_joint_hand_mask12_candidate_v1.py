"""Unregistered, read-only preflight for a four-development-session Mask A/B.

This module prepares one exact Task Packet.  It does not register it, import
SAM, evaluate historical Mask/Clean, or execute an algorithm.  A future
successor must publish the route and packet under governance CAS before the
operation can be called by ``chaoyang run``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, REPO_ROOT, artifact_ref, canonical_bytes,
    load_json, process_identity,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


OPERATION = "run_exact78_joint_hand_mask12_candidate_v1"
ROUTE_KEY = "exact78_joint_hand_mask12_candidate_v1"
TASK_ID = "exact78_joint_hand_mask12_four_dev_v1_20261001"
OLD_PROMPT = "a person's hand and forearm"
NEW_PROMPT = "a person's complete hand and forearm, including anything worn on or attached to them"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
VENDOR_MANIFEST_SHA = "1b42f75001be6aa1da3e64e20dd3589b4cf8bdebe84fecd8114675976a74f8ac"
WALL_SECONDS_MAX = 2700
RUNTIME_RELATIVE = (
    "src/chaoyang/governance/exact78_joint_hand_mask12_candidate_v1.py",
    "src/chaoyang/governance/execute_exact78_joint_hand_mask12_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_joint_hand_mask12_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v1.py",
    "src/chaoyang/pipeline/vst_encoded_video_domain.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/v52_contracts.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/paths.py",
    "src/chaoyang/cli/__init__.py",
)

# Exactly four development scenes.  Holdout IDs are intentionally absent.
FROZEN = (
    ("play_cards_0902_042", "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930",
     "TRACKER_0902", "poker", 171, (0, 85, 170),
     "c3edcc533d27a064e584bcaeeef683c00e52df6d2eafbf361ea52da9795c32d2",
     "d065012d7d774dd5c7f8b4d42be0555dc7f5f3d9c28e29a950be65de7e027684",
     "26bac49ce9526f34673219b3cdeb5c6afb038f6c0073d0e9921d516765d90e34",
     "f1baaa4c0ea6201b9ad00fc2947abd4b0816a6e4c9d38d4fae3df4f35de4f53d"),
    ("get_potato_chips_0902_103", "exact78_raw_to_robot_t2_0902_chips103_frontend_v3_mp4_retry1_20261001",
     "TRACKER_0902", "chips", 284, (0, 142, 283),
     "dcac04d55202062285a724e3e2bdfa52dc500d8e61d1b9248b30ba1e76ddcbe0",
     "7a6b70ce88a87cb21aee92e90a76966055921f6e49ba9c4d7a4ca83ae7738bf3",
     "df22ec569afb81b898445323f564ec25ffcfea4487dbae1afe34c18b2a9e9ae1",
     "ceaf897abd4a1cccf96e92bb02398e05431613b4f28e88a99354cb78af974f63"),
    ("get_potato_chips_0901_013", "exact78_raw_to_robot_t2_chips013_frontend_v1_20260930",
     "TRACKER_0901", "chips", 870, (0, 435, 869),
     "aed752e32412bb06d3e0983c17a6f788e74ca08f9edf3c25d1913fc2637f294b",
     "4bf0438733b4d2eef9d2b8bd7639cea6cfc09a8cd6ceb0099dadd3be177d55f1",
     "964a0e030193fbac66d34e3a4e93681da7cebf40e8dea46480c17b9a5b9a7a35",
     "9c88384f9c86377c71651f6a81e614a4ea15b6f0d2704de72d4b8f8ac256829b"),
    ("play_cards_0915_001", "exact78_raw_to_robot_t2_0915_cards001_frontend_v1_20260930",
     "VST_0915", "poker", 150, (0, 75, 149),
     "8f31f7676622d4b7d8a56655f58ab96d8f6e4614267eeca52f24d520ad7cff45",
     "858299610bb876b07a4cf786dbbed735d6f7e7a64bb371540a68b7e6b15de5c3",
     "4c2dd400b744c4385e7119fa80d56e253c701c73e45a1537ffca2051d1fb1d48",
     "7e12cde47c01f4a2e97125a3d19a0d10434825f84499363764febd6c58ef4b68"),
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _exact_ref(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise RuntimeError(f"INPUT_SYMLINK_OR_PATH_DRIFT:{path}")
    ref = artifact_ref(path)
    if ref["sha256"] != expected_sha:
        raise RuntimeError(f"INPUT_SHA_DRIFT:{path}")
    return ref


def _freeze_session(row: tuple[Any, ...]) -> dict[str, Any]:
    session, old_task, family, task, count, anchors, spec_sha, result_sha, raw_sha, video_sha = row
    attempt = REPO_ROOT / "_run/current" / old_task / "attempts/attempt_0001"
    spec_path, result_path = attempt / "FRONTEND_SPEC.json", attempt / "RESULT.json"
    raw_path = attempt / "candidate/RAW.json"
    spec_ref = _exact_ref(spec_path, spec_sha)
    result_ref = _exact_ref(result_path, result_sha)
    raw_ref = _exact_ref(raw_path, raw_sha)
    spec, result, raw = _object(spec_path), _object(result_path), _object(raw_path)
    if (spec.get("session_id") != session or spec.get("source_family") != family
            or spec.get("task") != task or spec.get("frame_count") != count
            or spec.get("task_id") != old_task or spec.get("output_root") != str(attempt / "candidate")
            or result.get("task_id") != old_task or result.get("status") != "PASSED"
            or result.get("source_session") != session or result.get("source_frames_expected") != count
            or result.get("execution_complete") is not True
            or raw.get("session_id") != session or raw.get("source_family") != family
            or raw.get("source_video") != spec.get("source_video")
            or len(raw.get("rows", [])) != count):
        raise RuntimeError(f"SEALED_SCENE_IDENTITY_DRIFT:{session}")
    video_ref = _exact_ref(Path(spec["source_video"]["path"]), video_sha)
    if video_ref != spec["source_video"]:
        raise RuntimeError(f"SOURCE_VIDEO_REF_DRIFT:{session}")
    if spec.get("sam_checkpoint", {}).get("sha256") != CHECKPOINT_SHA:
        raise RuntimeError(f"SAM_CHECKPOINT_SPEC_DRIFT:{session}")
    images = []
    for index in anchors:
        item = raw["rows"][index]
        path = attempt / "candidate/raw" / f"{index:06d}.png"
        if (item.get("decode_display_index") != index
                or item.get("model_input_index") != index
                or item.get("model_input", {}).get("path") != str(path)):
            raise RuntimeError(f"ANCHOR_FRAME_MAP_DRIFT:{session}:{index}")
        ref = _exact_ref(path, item["model_input"]["sha256"])
        if ref != item["model_input"]:
            raise RuntimeError(f"ANCHOR_IMAGE_REF_DRIFT:{session}:{index}")
        images.append({"frame": index, "raw_image": ref,
                       "time_mode": item.get("time_mode"),
                       "physical_eye": item.get("physical_eye")})
    return {"session_id": session, "source_family": family, "task": task,
            "frame_count": count, "source_fps": spec["source_fps"],
            "anchors": images, "source_video": video_ref,
            "frontend_spec": spec_ref, "sealed_result": result_ref,
            "sealed_raw": raw_ref, "sam_checkpoint": spec["sam_checkpoint"]}


def frozen_inputs() -> dict[str, Any]:
    sessions = [_freeze_session(row) for row in FROZEN]
    checkpoints = {item["sam_checkpoint"]["sha256"] for item in sessions}
    if checkpoints != {CHECKPOINT_SHA} or len({item["session_id"] for item in sessions}) != 4:
        raise RuntimeError("FOUR_DEV_SAM_IDENTITY_DRIFT")
    checkpoint = _exact_ref(Path(sessions[0]["sam_checkpoint"]["path"]), CHECKPOINT_SHA)
    if any(item["sam_checkpoint"] != checkpoint for item in sessions):
        raise RuntimeError("SAM_CHECKPOINT_REF_DRIFT")
    return {"sessions": sessions, "sam_checkpoint": checkpoint,
            "sam_vendor_manifest": _exact_ref(REPO_ROOT / "manifests/vendor.json",
                                              VENDOR_MANIFEST_SHA)}


def route_contract(runtime_code: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact route fields to stage later; this function does not publish them."""
    return {
        "execution_status": "CANDIDATE_FENCED_GPU_MASK12_FOUR_DEV",
        "candidate_operation": OPERATION,
        "task_packet_required": True,
        "code_closure": runtime_code,
        "output_schema": "exact78-joint-hand-mask12-result-v1",
        "model_calls_authorized": "FORTY_TWO_ANCHOR_ROLE_CALLS_ONLY_WHEN_EXACT_PACKET_ACTIVE",
        "render_calls_authorized": 0,
        "training_steps_authorized": 0,
        "current_runnable_operation": None,
        "claim_limit": "Twelve development Mask anchors only; no Clean, Robot, geometry, product, or training qualification.",
    }


def packet_value() -> dict[str, Any]:
    inputs = frozen_inputs()
    code = [artifact_ref(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE]
    attempt = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
    cache = attempt / "cache"
    command = (f"PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "
               f"TMPDIR={shlex.quote(str(cache))} TMP={shlex.quote(str(cache))} "
               f"TEMP={shlex.quote(str(cache))} XDG_CACHE_HOME={shlex.quote(str(cache))} "
               f"HF_HOME={shlex.quote(str(cache / 'huggingface'))} "
               f"TORCH_HOME={shlex.quote(str(cache / 'torch'))} "
               f"MPLCONFIGDIR={shlex.quote(str(cache / 'mpl'))} "
               f"CUDA_CACHE_PATH={shlex.quote(str(cache / 'cuda'))} "
               f"/usr/local/bin/python -B -m chaoyang.cli run {OPERATION} "
               f"--output-root {shlex.quote(str(attempt / 'candidate'))}")
    config = {
        "old_human_prompt": OLD_PROMPT, "new_human_prompt": NEW_PROMPT,
        "other_role_prompts": {
            "tracker_equipment": "a wrist worn tracking device",
            "vst_equipment": "a tactile glove", "vst_finger_sleeve": "a finger sleeve",
            "vst_yellow_cable": "a yellow cable", "poker_object": "a playing card",
            "chips_object": "a potato chip",
        },
        "output_prob_thresh": 0.5, "instance_score_gate": 0.5,
        "anchors_total": 12, "model_calls_max": 42,
        "cpu_threads_total": 2, "gpu_count": 1, "training_steps": 0,
        "session_attempts_max": 1,
        "clean_calls_authorized": 0, "old_mask_clean_algorithm_inputs": False,
    }
    signature = hashlib.sha256(canonical_bytes({
        "schema": "exact78-joint-hand-mask12-stage-v1", "inputs": inputs,
        "runtime_code": code, "config": config, "command": command,
        "weights": [inputs["sam_checkpoint"]], "calibration": "RAW_STAGE_ALREADY_SEALED",
    })).hexdigest()
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "One exact 12-anchor joint hand/attached-equipment prompt Mask experiment across four development scenes",
        "phase": "EXACT78_CLEAN_JOINT_HAND_MASK12_DEV_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": TASK_ID.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "read_set": [str(Path(item[name]["path"]).relative_to(REPO_ROOT))
                     for item in inputs["sessions"]
                     for name in ("frontend_spec", "sealed_raw")],
        "write_set": [str(attempt.relative_to(REPO_ROOT)),
                      f"tasks/current/{TASK_ID}/TASK_PACKET.json",
                      f"tasks/receipts/{TASK_ID.upper()}.json",
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "FOUR_DEV_SCENES_SEALED",
                          "OLD_MASK_CLEAN_EVALUATOR_ONLY", "ONE_GPU", "NO_HOLDOUT_TUNING",
                          "NO_0916", "NO_DELETION"],
        "required_outputs": ["candidate/MASK_ANCHOR.json", "candidate/RESULT.json",
                             "GPU_RECEIPT.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "model_calls_max": 42, "training_steps": 0,
                    "wall_seconds_max": WALL_SECONDS_MAX, "session_attempts_max": 1},
        "stop_conditions": ["INPUT_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "GPU_LEASE_UNAVAILABLE",
                            "UNSUPPORTED_SOURCE_OR_FRAME", "MASK_PROTECTION_GATE_FAILED",
                            "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Mask-only development candidate; no Clean, visual, Robot, geometry, product or training pass.",
        "writer": writer, "inputs": inputs, "runtime_code": code,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("MASK12_PACKET_INVALID:" + ";".join(errors))
    return packet


def route_status(packet: dict[str, Any]) -> str:
    contract = load_json(ALGORITHM_CONTRACT_PATH)
    routes = contract.get("special_status_contracts", {})
    route = routes.get(ROUTE_KEY) if isinstance(routes, dict) else None
    if route is None:
        return "ROUTE_NOT_STAGED_NO_EXECUTION"
    if route != route_contract(packet["runtime_code"]):
        raise RuntimeError("PUBLISHED_MASK12_ROUTE_DRIFT")
    return "PUBLISHED_ROUTE_PACKET_STILL_REQUIRED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    assert args.check_only
    packet = packet_value()
    print(json.dumps({"status": route_status(packet), "task_id": packet["task_id"],
                      "run_signature": packet["run_signature"],
                      "source_sessions": [row["session_id"] for row in packet["inputs"]["sessions"]],
                      "anchor_frames": {row["session_id"]: [item["frame"] for item in row["anchors"]]
                                        for row in packet["inputs"]["sessions"]},
                      "model_calls": 0, "registration_started": False,
                      "old_mask_clean_read": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
