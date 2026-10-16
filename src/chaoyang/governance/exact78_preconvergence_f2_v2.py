"""Produce F2 evidence only; never register or execute the draft P1 packet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chaoyang.governance.common import (
    REPO_ROOT, artifact_ref, atomic_json, atomic_write, load_json,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


P0_ROOT = REPO_ROOT / "_run/current/exact78_general_p0_preflight_v1_20260929/attempts/attempt_0001"
F2_ROOT = REPO_ROOT / "_run/current/exact78_preconvergence_f2_v1_20260930/attempts/attempt_0001"
PRIOR_PACKET = REPO_ROOT / "tasks/current/exact78_general_p1_action_canary_v1_retry8_20260930/TASK_PACKET.json"
MASK_PACKET = REPO_ROOT / "tasks/current/exact78_cards042_role_mask_rebuild_v1_20260929/TASK_PACKET.json"
CLEAN_PACKET = REPO_ROOT / "tasks/current/exact78_cards042_fresh_clean_v1_20260929/TASK_PACKET.json"
PARENT_RESULT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PENDING_ID = "exact78_p1_cold_start_three_session_pending_v1_20260930"
PENDING_PACKET = REPO_ROOT / "tasks/current" / PENDING_ID / "TASK_PACKET.json"
SESSION_IDS = (
    "play_cards_0902_042",
    "get_potato_chips_0902_103",
    "get_potato_chips_0901_013",
)


def _pinned_ref(value: dict[str, Any], label: str) -> dict[str, Any]:
    errors = validate_artifact_ref(value)
    if errors:
        raise RuntimeError(f"SHA_DRIFT:{label}:{','.join(errors)}")
    return value


def _source_row(row: dict[str, Any], prior_video: dict[str, Any]) -> dict[str, Any]:
    session_id = row["session_id"]
    checks = row["manifest_checks"]
    errors = set(row["errors"])
    if errors - {"MONO_FPS", "STEREO_FPS"}:
        raise RuntimeError(f"SOURCE_PREFLIGHT_ERROR:{session_id}:{sorted(errors)}")
    fps = float(row["mono"]["fps"])
    if fps not in (25.0, 30.0) or abs(fps - float(row["stereo"]["fps"])) > 0.01:
        raise RuntimeError(f"SOURCE_FPS_MISMATCH:{session_id}")
    if abs(fps - float(checks["humanego_fps"])) > 0.01:
        raise RuntimeError(f"SOURCE_METADATA_FPS_MISMATCH:{session_id}")
    count = int(row["expected_frames"])
    if not all((
        row["mono"]["decode_ok"], row["stereo"]["decode_ok"],
        row["mono"]["decoded_frames"] == count,
        row["stereo"]["decoded_frames"] == count,
        checks["clip_frames"] == count, checks["humanego_frames"] == count,
        checks["clip_name_matches"], checks["K_valid"],
        row["frame_identity"]["contiguous"], row["frame_identity"]["metadata_idx"],
        row["frame_identity"]["c2w"], row["frame_identity"]["K"],
    )):
        raise RuntimeError(f"SOURCE_IDENTITY_OR_DECODE_FAILURE:{session_id}")
    source_root = Path(row["source_root"])
    mono = artifact_ref(Path(row["mono"]["path"]))
    if mono != prior_video:
        raise RuntimeError(f"SOURCE_VIDEO_SHA_DRIFT:{session_id}")
    camera = source_root / "camera_params.json"
    camera_value = load_json(camera)
    camera_ref = artifact_ref(camera)
    has_extrinsics = bool(checks["stereo_calibration_present"])
    expected_extrinsics = row["frozen_calibration_state"] == "CALIBRATED_METRIC_STEREO"
    if has_extrinsics != expected_extrinsics:
        raise RuntimeError(f"CALIBRATION_STATE_CONFLICT:{session_id}")
    calibration = (
        {
            "state": "SOURCE_EXTRINSICS_PRESENT",
            "source_file": camera_ref,
            "source_metadata": camera_value.get("source_calibration"),
            "metric_accuracy_externally_validated": False,
        }
        if has_extrinsics else
        {
            "state": "ABSENT",
            "source_file": camera_ref,
            "extrinsics_sha256": "ABSENT",
            "metric_layer_allowed": False,
        }
    )
    return {
        "session_id": session_id, "task": row["task"], "fps": fps,
        "frame_count": count,
        "raw_input_valid": True,
        "p0_30fps_only_errors_reinterpreted": sorted(errors),
        "input_identity_sha256": row["expected_input_identity_sha256"],
        "files": {
            "source_video": mono,
            "source_stereo_video": artifact_ref(Path(row["stereo"]["path"])),
            "camera_params": camera_ref,
            "clip_manifest": artifact_ref(source_root / "clip_manifest.json"),
            "humanego_manifest": artifact_ref(source_root / "preprocess/pico_humanego_manifest.json"),
            "slam_trajectory": artifact_ref(source_root / f"slam_trajectory_{session_id}.jsonl"),
        },
        "intrinsics": "PINNED_CAMERA_PARAMS_AND_HUMANEGO",
        "stereo_metric_calibration": calibration,
        "visual_layer_allowed_when_metric_absent": True,
    }


def make_f2_v2() -> tuple[dict[str, Path], dict[str, Any]]:
    """Write F2's immutable contract and non-runnable P1 draft in registered write paths."""
    contract_path = F2_ROOT / "COLD_START_CONTRACT.json"
    readiness_path = F2_ROOT / "READINESS_ZH.md"
    if contract_path.exists() or readiness_path.exists() or PENDING_PACKET.exists():
        raise RuntimeError("F2_OUTPUT_ALREADY_EXISTS")
    p0 = load_json(P0_ROOT / "SESSION_PREFLIGHT.json")
    p0_rows = {row["session_id"]: row for row in p0["sessions"]}
    prior = load_json(PRIOR_PACKET)
    manifest_ref = _pinned_ref(prior["inputs"]["manifest"], "prior_manifest")
    prior_manifest = load_json(Path(manifest_ref["path"]))
    prior_sources = {row["session_id"]: row["source_video"] for row in prior_manifest["sessions"]}
    sessions = [_source_row(p0_rows[session_id], prior_sources[session_id]) for session_id in SESSION_IDS]

    weights: dict[str, dict[str, Any]] = {}
    for name in ("hawor_checkpoint", "hawor_infiller"):
        weights[name] = _pinned_ref(prior["inputs"][name], name)
    mask = load_json(MASK_PACKET)
    clean = load_json(CLEAN_PACKET)
    for name, packet in (
        ("sam31_checkpoint", mask),
        ("propainter_weights", clean),
        ("flow_weights", clean),
        ("raft_weights", clean),
    ):
        weights[name] = _pinned_ref(packet["inputs"][name], name)

    candidate_code = [
        {**ref, "sha_status": "MATCH" if not validate_artifact_ref(ref) else "DRIFT"}
        for ref in prior["runtime_code"]
    ]
    candidate_configs = {
        "prior_p1_task_packet": artifact_ref(PRIOR_PACKET),
        "prior_p1_execution_plan": _pinned_ref(prior["inputs"]["execution_plan"], "execution_plan"),
        "prior_mount_visual_proxy_contract": _pinned_ref(prior["inputs"]["mount_proxy"], "mount_proxy"),
        "prior_mask_task_packet": artifact_ref(MASK_PACKET),
        "prior_clean_task_packet": artifact_ref(CLEAN_PACKET),
    }
    contract = {
        "schema_version": "exact78-three-session-cold-start-contract-v2",
        "status": "INPUTS_FROZEN_RUNNER_UNSELECTED",
        "p0_source_inventory": artifact_ref(P0_ROOT / "SESSION_PREFLIGHT.json"),
        "sessions": sessions,
        "models": weights,
        "candidate_code_audit_not_selected_runtime": candidate_code,
        "candidate_config_audit_not_selected_runtime": candidate_configs,
        "selected_runtime_code": None,
        "selected_runtime_config": None,
        "full_chain_dependency_closure": "INCOMPLETE",
        "closure_required_before_p1_activation": [
            "fresh_action_runner_code_and_config",
            "role_mask_and_object_mask_code_and_config",
            "clean_and_compositor_code_and_config",
            "robot_hand_arm_flange_geometry_and_render_config",
        ],
        "reuse_policy": "RAW_ONLY_NO_OLD_STATE_ARRAYS_NO_CARDS042_FITTED_TEMPLATE",
        "per_session_fit_policy": "derive base and offsets from same session only",
        "frame_contract": "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md#3",
        "fixed_numeric_gates": {
            "visible_without_action_fraction_max": 0.05,
            "observed_arm_reach_fraction_min": 0.90,
            "position_mm_max": 10.0, "rotation_deg_max": 5.0,
            "velocity_rad_per_frame_max": 0.12,
            "acceleration_rad_per_frame2_max": 0.06,
            "visual_session_qualified_frame_fraction_min": 0.80,
        },
        "outputs_required": [
            "per_frame_provenance", "per_frame_stage_status", "raw_robot_diagnostic",
            "independent_whole_robot", "final_first_person_clean_robot",
            "four_metric_report", "failure_cluster_table",
        ],
        "visual_parent_result": artifact_ref(PARENT_RESULT),
        "visual_parent_replayable": False,
        "no_training_or_physical_action_authority": True,
    }
    atomic_json(contract_path, contract)
    draft = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": PENDING_ID,
        "objective": "Select one fresh Raw-to-Robot runner and execute a three-session cold-start candidate",
        "phase": "EXACT78_P1_COLD_START_THREE_SESSION",
        "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        "execution_revision": "PENDING_NEW_QUALIFIED_RUNTIME_CODE",
        "read_set": [
            "AGENTS.md", "docs/plans/EXACT78_PRECONVERGENCE_V1_20260930/00_EXECUTION.md",
            "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md",
            str(contract_path.relative_to(REPO_ROOT)),
        ],
        "write_set": [
            f"_run/current/{PENDING_ID}/attempts/attempt_0001",
            f"tasks/current/{PENDING_ID}/TASK_PACKET.json", "docs/governance", "tasks/current/INDEX.json",
        ],
        "prerequisites": [
            "F0_F1_F2_F3_PASSED", "SELECTED_FRESH_RUNTIME_CODE_SHA",
            "FULL_CHAIN_DEPENDENCY_CLOSURE", "NEW_WRITER_FENCE", "ONE_GPU_RESOURCE",
        ],
        "required_outputs": contract["outputs_required"],
        "budgets": {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
            "training_steps": 0, "wall_seconds_max": 28800, "session_attempts_max": 1,
        },
        "stop_conditions": [
            "SOURCE_SHA_DRIFT", "MODEL_SHA_DRIFT", "CARDS042_FIT_READ",
            "DOUBLE_WRITER", "ONE_ATTEMPT_TERMINAL",
        ],
        "claim_limit": "Non-runnable draft only; a new immutable successor with complete dependency closure, selected code, writer, signature and command is required.",
        "execution_mode": "EXACT78_CANDIDATE_DRAFT",
        "execution_allowed": False,
        "inputs": {"cold_start_contract": artifact_ref(contract_path)},
        "runtime_code": [], "selected_runtime_code": None, "selected_runtime_config": None,
        "run_signature": None, "commands": [], "writer": None,
    }
    errors = _validate_packet(draft)
    if errors:
        raise RuntimeError("INVALID_PENDING_P1_PACKET:" + ";".join(errors))
    PENDING_PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PENDING_PACKET, draft)
    text = (
        "# 三会话冷启动合同\n\n"
        "Cards042、Chips103 与原生 25 FPS Chips013 的单目／双目 Raw 视频、相机参数、"
        "源清单、HumanEgo 元数据和 SLAM 轨迹已分别锁定 SHA。Chips013 缺少双目外参，"
        "米制标定记为 `ABSENT`；042／103 仅记录来源标定存在，不声称外部精度验证。\n\n"
        "历史动作代码只做 SHA 审计。Clean、合成、几何及配置的完整依赖闭包尚未选定；"
        "P1 草案的 `execution_allowed=false`、`selected_runtime_code=null`，不能运行模型。"
        "正式执行必须发布有完整闭包的新不可变 successor，不能复用 042 动作数组或拟合模板。\n"
    )
    atomic_write(readiness_path, text.encode("utf-8"))
    return {
        "contract": contract_path,
        "pending_p1_packet": PENDING_PACKET,
        "readiness_zh": readiness_path,
    }, {
        "sessions": list(SESSION_IDS),
        "runtime_selected": False,
        "full_chain_dependency_closure": "INCOMPLETE",
        "pending_p1_execution_allowed": False,
        "metric_calibration_absent_sessions": ["get_potato_chips_0901_013"],
        "source_stereo_sha_frozen_count": 3,
    }


__all__ = ["make_f2_v2"]
