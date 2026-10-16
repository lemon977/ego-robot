"""Read-only 201-frame camera-direction render candidate for new Raw Cards001.

This is a CPU preflight and non-executable Task Packet blueprint. It never
registers, writes, renders, or reads the predecessor videos. The later governed
worker may reuse the already frozen generic render-direction operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, canonical_bytes, load_json,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


TASK_ID = "exact78_newraw_0902_cards001_render_direction_only_v1_20261001"
SESSION_ID = "play_cards_0902_001"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
STAGE = (REPO_ROOT / "_run/current/exact78_newraw_robot_play_cards_0902_001_9541579d8de29395_v1"
         "/attempts/attempt_0001/candidate/robot/v1_stage_with_ffmpeg44_encoder/RESULT.json")
STAGE_SHA = "cc9b87a947bdc945fbfece7c96efa16d115c2fcc3560913565138d6b3553e2a1"
STATES_SHA = "1ab5b73f6e6d971b57fd24e76b1ff62da2ef0c5bf6399714a181f3c6dc471f76"
HAWOR_SHA = "8d0c65d95e54b4eec542a3b0dd2c5ead5fb45f20dd570e3d5ef2a6d55540165f"
RAW_SHA = "939459dd010084ab3665cdbeb117eeb2669c6b4a81ff4fa8fdbb8ffc5d6fcf9f"
ASSET_PIN_SHA = "4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330"
OUT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
SPEC = OUT / "RENDER_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1.json"
OPERATION = "run_exact78_robot_render_direction_v1"
CONTRACT_KEY = "exact78_robot_render_direction_newraw_cards001_candidate_v1_20261001"
COMMAND = f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run {OPERATION} --spec {SPEC}"
FRAMES = 201
FPS = 30.0
SAMPLE_FRAMES = (0, 100, 200)
VIDEO_FILES = {"ego_robot": "ego_robot_diagnostic.mp4",
               "external_full_robot": "fixed_external_full_robot.mp4",
               "raw_motion_source": "raw_motion_source_review.mp4"}


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def frozen_spec() -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    """Bind one sealed stage; generic preflight checks every source and code SHA."""
    if artifact_ref(STAGE)["sha256"] != STAGE_SHA:
        raise RuntimeError("SEALED_STAGE_SHA_DRIFT")
    stage = load_json(STAGE)
    if stage.get("session_id") != SESSION_ID or stage.get("frame_count") != FRAMES:
        raise RuntimeError("SEALED_STAGE_IDENTITY_DRIFT")
    states = stage.get("outputs", {}).get("states", {})
    inputs = stage.get("inputs", {})
    if (states.get("sha256") != STATES_SHA
            or inputs.get("hawor_npz", {}).get("sha256") != HAWOR_SHA
            or inputs.get("raw_video", {}).get("sha256") != RAW_SHA
            or artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")["sha256"] != ASSET_PIN_SHA):
        raise RuntimeError("SEALED_STATE_HAWOR_RAW_OR_ASSET_SHA_DRIFT")
    spec = {"task_id": TASK_ID, "session_id": SESSION_ID,
            "source_kind": "TRACKER_CLIP_RGB",
            "inputs": {"stage_result": artifact_ref(STAGE), "states": states,
                       **{key: inputs[key] for key in (
                           "hawor_npz", "frame_provenance", "input_audit", "raw_video")}},
            "output_root": str(OUT / "candidate")}
    fixed, intrinsics, fps = direction.preflight(spec)
    if (len(fixed["q_arm"]) != FRAMES or fps != FPS
            or fixed["display_joints_3d_camera"].shape[:2] != (2, FRAMES)
            or fixed["q_hand"].shape != (FRAMES, 2, 22)):
        raise RuntimeError("FROZEN_201_FRAME_DOMAIN_DRIFT")
    return spec, fixed, intrinsics


def _project(k: np.ndarray, point: np.ndarray) -> list[float] | None:
    if not np.isfinite(point).all() or point[2] <= 1e-9:
        return None
    pixel = k @ point
    if not np.isfinite(pixel).all() or pixel[2] <= 1e-9:
        return None
    return [round(float(pixel[0] / pixel[2]), 2),
            round(float(pixel[1] / pixel[2]), 2)]


def projection_observations(spec: dict[str, Any], states: dict[str, np.ndarray],
                            intrinsics: np.ndarray) -> list[dict[str, Any]]:
    """Read-only FK comparison; HaWoR model wrists are not direct observations."""
    stage = load_json(STAGE)
    frame_ref = stage["outputs"]["frame_states"]
    if artifact_ref(Path(frame_ref["path"])) != frame_ref:
        raise RuntimeError("SEALED_FRAME_STATES_SHA_DRIFT")
    frame_states = load_json(Path(frame_ref["path"]))
    by_frame_side = {(int(row["frame"]), row["side"]): row
                     for row in frame_states["rows"]}
    if len(by_frame_side) != 2 * FRAMES:
        raise RuntimeError("SEALED_FRAME_STATES_DOMAIN_DRIFT")
    assets = legacy.load_pinned_robot_assets(legacy.ROOT)
    base = states["T_camera_base"]
    inverse = np.linalg.inv(base)
    rows: list[dict[str, Any]] = []
    for frame in SAMPLE_FRAMES:
        for side, name in enumerate(("left", "right")):
            tool = legacy._tool_fk(assets, side, states["q_arm"][frame, side])
            local = tool @ states["T_tool_hand_root"][side]
            k = intrinsics[frame]
            predicted = _project(k, states["display_joints_3d_camera"][side, frame, 0])
            old = _project(k, (inverse @ local)[:3, 3])
            changed = _project(k, (base @ local)[:3, 3])
            status = by_frame_side[(frame, name)]["arm"]["status"]
            rows.append({
                "frame": frame, "side": name, "arm_status": status,
                "motion_source": str(states["motion_source"][side, frame]),
                "hawor_model_wrist_px": predicted,
                "old_inverse_camera_wrist_px": old,
                "candidate_base_to_camera_wrist_px": changed,
                "comparison_scope": "INTERNAL_MODEL_TARGET_ONLY_NOT_VISUAL_TRUTH",
                "arm_hold_not_action_evidence": not status.startswith("PASS"),
            })
    return rows


def packet_blueprint(spec: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """A complete signature/closure plan but deliberately no live writer token."""
    stamp = now or datetime.now().astimezone()
    if stamp >= CAMPAIGN_END:
        raise RuntimeError("CAMPAIGN_HARD_CUTOFF_PASSED")
    code = [Path(__file__), REPO_ROOT / "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
            REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
            *(REPO_ROOT / relative for relative in direction.PINNED_CODE_SHA),
            REPO_ROOT / "src/chaoyang/governance/common.py",
            REPO_ROOT / "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
            REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py",
            REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py"]
    runtime = [artifact_ref(path) for path in code]
    inputs = {"render_spec": _virtual_ref(SPEC, spec),
              "sealed_stage": spec["inputs"]["stage_result"],
              "fixed_inputs": spec["inputs"],
              "frame_states": load_json(STAGE)["outputs"]["frame_states"],
              "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")}
    config = {"session_id": SESSION_ID, "frames": FRAMES, "fps": FPS,
              "change_only": "EGO_CAMERA_INV_T_CAMERA_BASE_TO_T_CAMERA_BASE",
              "candidate_contract_key": CONTRACT_KEY,
              "flange_parent_changed": False, "q_base_mount_recomputed": False,
              "ik_calls": 0, "model_calls": 0, "gpu_count": 0,
              "numeric_gates_changed": False,
              "observation_plan": {"three_complete_videos": list(VIDEO_FILES.values()),
                                   "per_frame_ego_and_fixed_external_robot_pixels": True,
                                   "projection_frames": list(SAMPLE_FRAMES),
                                   "old_video_ab": "EVALUATOR_ONLY_AFTER_FRESH_SEAL",
                                   "ai_review": "PENDING", "human_review": "PENDING"}}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": COMMAND, "schema": "exact78-newraw-cards001-direction-only-blueprint-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "One 201-frame new Raw Cards001 render-only camera-direction candidate",
        "phase": "EXACT78_ROBOT_RENDER_DIRECTION_ONLY",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": None,
        "deadline_at": min(stamp + timedelta(seconds=3600), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(STAGE.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
                     "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
                     "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
                     "assets/robot/ROBOT_ASSET_PIN.json",
                     "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"],
        "expanded_read_set": [ref["path"] for ref in (*spec["inputs"].values(),
                                                       inputs["frame_states"])],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_201_FRAME_STATES",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "CPU_ONLY",
                          "NEW_SESSION_CANDIDATE_ROUTE_STAGED_WITH_PACKET",
                          "NO_MODEL", "NO_CLEAN_GEOMETRY_CLAIM", "NO_DELETION"],
        "required_outputs": ["RENDER_SPEC.json", "candidate/RESULT.json",
                             *(f"candidate/render/{name}" for name in VIDEO_FILES.values()),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 3600,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "DEADLINE", "VIDEO_DECODE_OR_TIME_FAIL", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Execution-only camera diagnostic; no flange, occlusion, visual, product, training or control qualification",
        "writer": None, "blueprint_only": True, "registration_allowed": False,
        "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_RENDER_DIRECTION_BLUEPRINT:" + ";".join(errors))
    return packet


def check_only(*, now: datetime | None = None) -> dict[str, Any]:
    spec, states, intrinsics = frozen_spec()
    observations = projection_observations(spec, states, intrinsics)
    packet = packet_blueprint(spec, now=now)
    state = load_json(TASK_STATE_PATH)
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    contract = load_json(ALGORITHM_CONTRACT_PATH)
    new_route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY)
    route_status = ("PUBLISHED_NEW_SESSION_CANDIDATE_ROUTE"
                    if isinstance(new_route, dict)
                    and new_route.get("candidate_operation") == OPERATION
                    and new_route.get("task_packet_required") is True
                    else "NEW_SESSION_CANDIDATE_ROUTE_NOT_PUBLISHED")
    busy = (bool(state.get("next_task"))
            or any(row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}
                   for row in state.get("tasks", []))
            or any(row.get("execution_allowed") for row in index["task_packets"]))
    return {"status": "DRY_RUN_EVIDENCE_READY_NOT_ROUTABLE",
            "publisher_busy": busy, "route_status": route_status,
            "registered": False, "rendered": False,
            "task_id": TASK_ID, "session_id": SESSION_ID, "frames": FRAMES,
            "nominal_fps": FPS, "spec_sha256": packet["inputs"]["render_spec"]["sha256"],
            "run_signature": packet["run_signature"],
            "packet_blueprint": packet,
            "projection_observations": observations,
            "claim_limit": packet["claim_limit"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
