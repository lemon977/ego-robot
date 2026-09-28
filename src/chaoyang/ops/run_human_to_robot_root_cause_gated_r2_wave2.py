#!/usr/bin/env python3
"""R2 wave 2: 0902 same-recipe regression, motion reuse and adapter truth."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import (
    ATTEMPT, REPO, TASK, V5, motion_rebind, ref, write_json,
)
from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave1 import scene_repair
from chaoyang.pipeline.v5_exact_motion import recover as recover_exact


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def adapter_receipt() -> dict:
    output = ATTEMPT / "lanes/lane2_motion/assembly_wave2/RESULT.json"
    step = REPO / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/received_design/KAI_HAND固定件.STEP"
    decode = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/decode_v1/RESULT.json"
    mount = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/robot/KAIHAND_ADAPTER_V3_MOUNT_CONTRACT.json"
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_ASSEMBLY_EVIDENCE_V1",
        "task_id": TASK, "execution": "EXECUTED", "structure": "PASS",
        "quality": "INCONCLUSIVE", "adoption": "CANDIDATE_ONLY",
        "adapter_cad": "PRESENT_CANDIDATE_GEOMETRY",
        "adapter_scene_loading": "AVAILABLE_FROM_DECODED_REVIEW_MESH",
        "measured_installation_transform": "ABSENT",
        "robot_tcp": "ABSENT", "camera_world_to_base": "ABSENT",
        "complete_assembly_adoption": False,
        "inputs": {"step": ref(step), "decode": ref(decode), "virtual_mount": ref(mount)},
        "claim_limit": "Decoded real STEP plus development virtual mount; not measured assembly or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "created_at": now(),
    }
    write_json(output, result)
    return result


def append_state(lane: str, artifacts: list[Path], action: str) -> None:
    path = ATTEMPT / "lanes" / lane / "STATE.json"
    state = json.loads(path.read_text())
    old = state.get("latest_artifacts", [])
    seen = {item["path"] for item in old if isinstance(item, dict) and "path" in item}
    for artifact in artifacts:
        item = ref(artifact)
        if item["path"] not in seen:
            old.append(item); seen.add(item["path"])
    state.update(latest_artifacts=old, current_action=action,
                 writer={"pid": os.getpid(), "proc_start_ticks": None}, updated_at=now())
    write_json(path, state)


def _sha_ref(path: Path) -> dict:
    return ref(path)


def exact_motion_rebind(short: str) -> dict:
    """Recover one frozen 0902 Exact motion with the schema-specific adapter.

    The 0902 artifacts are bounded historical inputs, not the ROI/raw HaWoR
    representation used by the 0915 sessions.  Keeping the adapters separate
    prevents a schema mismatch from being hidden as a quality rejection.
    """
    session = {
        "103": "get_potato_chips_0902_103",
        "042": "play_cards_0902_042",
    }[short]
    frames = {"103": 284, "042": 171}[short]
    if short == "103":
        bounded = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/hawor_bounded_v2/chips/get_potato_chips_0902_103/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
    else:
        bounded = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/play_cards_0902_042/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
    v3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
    exact_dir = v3 / "lanes/ai2/exact_bounded_camera_source_v3" / session
    robot_dir = v3 / "lanes/ai2/camera_exact_mount_v2" / session
    domain = v3 / "lanes/exact78/exact_domain_v1" / session / "DOMAIN_MANIFEST.json"
    config_path = ATTEMPT / "lanes/lane2_motion/configs" / f"recover_0902_{short}_wave2.json"
    output = ATTEMPT / "lanes/lane2_motion" / f"recovered_0902_{short}_wave2"
    config = {
        "schema_version": "HUMAN_TO_ROBOT_BASELINE_V1_EXACT_MOTION",
        "session_id": session,
        "frame_count": frames,
        "bounded_source": _sha_ref(bounded),
        "exact_source": _sha_ref(exact_dir / "BOUNDED_CAMERA_SOURCE_V3.npz"),
        "exact_result": _sha_ref(exact_dir / "RESULT.json"),
        "robot_r0": _sha_ref(robot_dir / "CAMERA_ROBOT_MOTION_V3.npz"),
        "robot_result": _sha_ref(robot_dir / "RESULT.json"),
        "domain_manifest": _sha_ref(domain),
        "mount": _sha_ref(v3 / "lanes/ai2/mount_v2_inputs/MOUNT_ARRAY_BINDING.json"),
        "asset_pin": _sha_ref(REPO / "assets/robot/ROBOT_ASSET_PIN.json"),
        "output": str(output),
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise FileExistsError(f"CONFIG_SIGNATURE_CHANGED:{config_path}")
    else:
        write_json(config_path, config)
    result = recover_exact(config_path)
    return {
        "session_id": session,
        "valid_side_frames": result["source_valid_left_right"],
        "quality": result["quality_status"],
        "result": result,
    }


def main() -> int:
    scene = [
        scene_repair("103", V5 / "lanes/scene/POST_103.json"),
        scene_repair("042", V5 / "lanes/scene/POST_042.json"),
    ]
    motion = [exact_motion_rebind("103"), exact_motion_rebind("042")]
    assembly = adapter_receipt()
    scene_paths = [ATTEMPT / "lanes/lane1_scene/conservative_repair_103_wave1/RESULT.json",
                   ATTEMPT / "lanes/lane1_scene/conservative_repair_042_wave1/RESULT.json"]
    motion_paths = [ATTEMPT / "lanes/lane2_motion/recovered_0902_103_wave2/RESULT.json",
                    ATTEMPT / "lanes/lane2_motion/recovered_0902_042_wave2/RESULT.json",
                    ATTEMPT / "lanes/lane2_motion/assembly_wave2/RESULT.json"]
    append_state("lane1_scene", scene_paths,
                 "Same conservative Scene repair executed on both 0902 regressions; Clean remains evidence-gated.")
    append_state("lane2_motion", motion_paths,
                 "Four Motion sessions rebound; real STEP present, measured installation remains absent.")
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1", "task_id": TASK,
        "stage": "WAVE2_0902_REGRESSION_AND_ASSEMBLY", "observed_at": now(),
        "scene_regressions": [{"session_id": row["session_id"], "quality": row["quality"],
                               "area_reduction_fraction": row["area_reduction_fraction"]} for row in scene],
        "motion_regressions": [{"session_id": row["session_id"],
                                "valid_side_frames": row["valid_side_frames"]} for row in motion],
        "assembly": {key: assembly[key] for key in ("adapter_cad", "measured_installation_transform",
                                                     "robot_tcp", "complete_assembly_adoption")},
        "clean_sessions": 0, "product_sessions": 0,
        "next": ["Scene G0 candidate inpaint under GPU lease only after reference input manifest",
                 "renderer four-way diagnostic", "per-session product remains independent"],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE2.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
