#!/usr/bin/env python3
"""Terminalize S2 after evidence closure without promoting quality.

H12 remains the default latest checkpoint.  ``--allow-early-close`` is an
explicitly authorized event-driven close and still requires H3/H6/H9, final
validation, no live lane writer, no active S2 GPU lease, and the sole current
task packet.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
)


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT_OUT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"
PLAN = REPO_ROOT / "docs/current/PLAN.md"
WORK_ENTRY = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
BASELINE = REPO_ROOT / "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
VISUAL_INDEX = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md"
LANES = ("scene", "sensor", "motion_product", "compare")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _pid_live(value: object) -> bool:
    if not isinstance(value, int) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--allow-early-close", action="store_true")
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((item for item in state["tasks"] if item.get("task_id") == TASK), None)
    if task is None or task.get("status") not in LIVE:
        raise RuntimeError("S2_NOT_LIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("S2_NOT_CURRENT")
    elapsed = (datetime.now().astimezone() - datetime.fromisoformat(str(task["t0"]))).total_seconds()
    if elapsed < 12 * 3600 and not args.allow_early_close:
        raise RuntimeError(f"H12_NOT_REACHED:{elapsed:.1f}")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("S2_PACKET_NOT_SOLE_CURRENT")
    if (ROOT / "RESULT.json").exists() or RECEIPT_OUT.exists():
        raise RuntimeError("IMMUTABLE_TERMINAL_RESULT_EXISTS")
    validation_path = ROOT / "FINAL_VALIDATION.json"
    validation = load_json(validation_path)
    if validation.get("status") != "PASS_STRUCTURE_WITH_QUALITY_GAPS":
        raise RuntimeError("FINAL_VALIDATION_NOT_ACCEPTABLE")
    if validation.get("video_count") != 11 or validation.get("video_full_decode_pass") is not True:
        raise RuntimeError("FINAL_MEDIA_NOT_CLOSED")
    if validation.get("quality_summary", {}).get("formal_product_quality_pass") != 0:
        raise RuntimeError("UNSUPPORTED_PRODUCT_QUALITY_PROMOTION")
    for checkpoint in ("H3", "H6", "H9"):
        if not (ROOT / f"checkpoints/{checkpoint}_RESULT.json").is_file():
            raise RuntimeError(f"CHECKPOINT_MISSING:{checkpoint}")

    lane_values = {}
    for lane in LANES:
        path = ROOT / f"lanes/{lane}/STATE.json"
        value = load_json(path)
        writer = value.get("writer", {})
        if _pid_live(writer.get("pid")):
            raise RuntimeError(f"ACTIVE_LANE_WRITER:{lane}:{writer['pid']}")
        lane_values[lane] = value

    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    if lease_path.is_file():
        lease = load_json(lease_path)
        if lease.get("status") == "ACQUIRED" and str(lease.get("task_id", "")).startswith(TASK):
            raise RuntimeError("S2_GPU_LEASE_ACTIVE")

    lane_refs = {}
    for lane, value in lane_values.items():
        path = ROOT / f"lanes/{lane}/STATE.json"
        value["writer"] = {"pid": None, "proc_start_ticks": None, "executor_epoch": 8}
        value["status"] = "TERMINAL_WITH_QUALITY_GAPS"
        value["execution"] = "EXECUTED"
        value["structure"] = "PASS"
        value["adoption"] = "NOT_ADOPTED"
        value["updated_at"] = now_iso()
        atomic_json(path, value)
        lane_refs[lane] = artifact_ref(path)

    terminal_at = now_iso()
    result = {
        "schema_version": "HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_RESULT_V1",
        "task_id": TASK,
        "route": ROUTE,
        "execution_round": "S2",
        "status": "TERMINAL_WITH_QUALITY_GAPS",
        "task_terminal_status": "REJECTED_QUALITY",
        "release_status": "INCOMPLETE",
        "terminal_at": terminal_at,
        "elapsed_seconds": elapsed,
        "nominal_time_gate_seconds": 12 * 3600,
        "early_close_authorized": bool(args.allow_early_close and elapsed < 12 * 3600),
        "execution_summary": {
            "formal_product_candidates": 4,
            "formal_product_structure_pass": 4,
            "formal_product_quality_pass": 0,
            "formal_product_adopted": 0,
            "formal_entry_current_signature_resume": "4/4_PASS_NO_MUTATION",
            "sensor_sessions_reverified": 3,
            "local_huro_adapter_refresh_sessions": 2,
            "local_huro_winner": None,
            "adapter_visual_consumption": "PASS",
            "adapter_collision_scope": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
            "occlusion_031": "EXECUTED_INCONCLUSIVE",
            "occlusion_007_0902": "UNKNOWN_NO_QUALIFIED_DEPTH",
            "robot_r1_screened": 0,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
        },
        "quality_axes": {
            "execution": "EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE",
            "adoption": "NOT_ADOPTED",
        },
        "reason_codes": [
            "P031_FULLPOSE_TARGET_TRACKING_REJECTED",
            "P031_LEFT_HAND_MODEL_OUTPUT_ABSENT",
            "P031_TARGET_ROTATION_EXTERNAL_AUTHORITY_ABSENT",
            "C007_LONG_CABLE_SUPPORT_AND_CLEAN_QUALITY_UNRESOLVED",
            "C007_STEREO_MATCH_GATE_REJECTED_1021_OF_1500",
            "OCCLUSION_TEMPORAL_QUALITY_INCONCLUSIVE",
            "ADAPTER_COLLISION_GEOMETRY_UNVERIFIED",
            "CONTACT_ROBOT_R1_NOT_SCREENED_NO_ADMISSIBLE_WINDOW_ESTABLISHED",
            "LOCAL_HURO_INDEPENDENT_TRUTH_ABSENT",
            "FOURTEEN_HARDLINK_TRANSACTION_TESTS_NOT_EVALUATED_ON_CPFS",
        ],
        "checkpoints": {
            name: artifact_ref(ROOT / f"checkpoints/{name}_RESULT.json")
            for name in ("H3", "H6", "H9")
        },
        "final_validation": artifact_ref(validation_path),
        "formal_entry_resume": artifact_ref(ROOT / "formal_entry_resume/attempt_0002/RESULT.json"),
        "lane_states": lane_refs,
        "visual_index": artifact_ref(VISUAL_INDEX),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "S2 closes structural execution and evidence lineage only. All four formal products remain "
            "rejected-quality candidates; Contact/Robot R1 screening was not completed and no admissible "
            "window was established, no method winner, external metric, control, training, physical "
            "deployment or user adoption authority was obtained."
        ),
    }
    atomic_json(ROOT / "RESULT.json", result)
    result_ref = artifact_ref(ROOT / "RESULT.json")
    atomic_json(
        RECEIPT_OUT,
        {
            "schema_version": "HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_RECEIPT_V1",
            "task_id": TASK,
            "status": "REJECTED_QUALITY",
            "result": result_ref,
            "products_structure": "4/4",
            "products_quality": "0/4",
            "products_adopted": "0/4",
            "formal_resume": "4/4",
            "videos_full_decode": 11,
            "robot_r1_adopted": 0,
        },
    )

    task.update(
        status="REJECTED_QUALITY",
        phase="HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_TERMINAL",
        attempt=1,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        heartbeat_at=None,
        updated_at=terminal_at,
        result=result_ref,
        last_attempt_terminal="REJECTED_QUALITY",
        last_attempt_reason="STRUCTURE_AND_LINEAGE_CLOSED_WITH_PRODUCT_QUALITY_GAPS",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "REJECTED_QUALITY",
        "created_at": terminal_at,
        "message": "S2 terminal: structural/current-signature closure, 0/4 product quality pass and no authority promotion.",
        "result": result_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_20260923",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "S2 is terminal rejected-quality; any new partial-direction IK or cable tracker requires a new finite task.",
    }

    PLAN.write_text(
        "# 当前计划入口：Human→Robot S2 已终态\n\n"
        f"`{TASK}` 已按 H3/H6/H9/H12 收敛，终态为 `REJECTED_QUALITY`。"
        "四条正式候选结构完整，质量通过0/4、采用0/4；不得把终态化解释为产品成功。\n\n"
        f"机器结果：[RESULT]({result_ref['path']})；[视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n",
        encoding="utf-8",
    )
    WORK_ENTRY.write_text(
        "# 当前入口：Human→Robot S2 已终态\n\n"
        f"当前无活动任务。先读 [S2 RESULT]({result_ref['path']})、[当前状态](STATUS.json) 和 "
        "[视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n\n"
        "如需继续031部分方向IK、007线缆实例支持或连接件碰撞覆盖，必须登记新的有限任务；"
        "不得复活S2、降低现有门或覆盖旧attempt。\n",
        encoding="utf-8",
    )
    baseline_text = BASELINE.read_text(encoding="utf-8")
    marker = "## S2 终态"
    if marker not in baseline_text:
        baseline_text += (
            f"\n{marker}\n\nS2 已完成当前签名入口、完整解码和证据绑定，"
            "但四条产品质量通过为0/4。连接件只有视觉几何，031遮挡仍为inconclusive，"
            "Contact/Robot R1筛选、执行和采用均为0，尚未建立可用窗口结论；"
            "Local/HuRo无独立真值。终态不授予训练、控制或部署权限。\n"
        )
        BASELINE.write_text(baseline_text, encoding="utf-8")
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_EVIDENCE_UNLOCK_S2_TERMINAL_QUALITY_GAPS",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    print(json.dumps({"status": "REJECTED_QUALITY", "governance_revision": published["governance_revision"], "result": result_ref, "products_quality": "0/4", "videos": 11}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
