#!/usr/bin/env python3
"""Terminalize the finite convergence successor without promoting rejected quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_baseline_v1_convergence_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT_OUT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"
PLAN = REPO_ROOT / "docs/current/PLAN.md"
ENTRY = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
BASELINE = REPO_ROOT / "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
VIS = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md"

EVIDENCE = {
    "motion": ROOT / "lanes/motion_product/partial_direction_031/wave0/RESULT.json",
    "motion_review": ROOT / "lanes/motion_product/partial_direction_031/review_v1/RESULT.json",
    "left": ROOT / "lanes/motion_product/left_evidence_031/wave0/RESULT.json",
    "cable": ROOT / "lanes/scene/cable_007/wave0/RESULT.json",
    "contact": ROOT / "lanes/scene/contact_screen_031/wave0/RESULT.json",
    "occlusion": ROOT / "lanes/scene/occlusion_same_surface_031/wave0/RESULT.json",
    "collision": ROOT / "lanes/motion_product/adapter_collision/wave0/RESULT.json",
    "sensor": ROOT / "lanes/sensor/review_v1/RESULT.json",
    "compare": ROOT / "lanes/compare/numeric_v1/RESULT.json",
    "delivery": ROOT / "delivery/DELIVERY_MANIFEST.json",
}


def lane(name: str, quality: str, improvement: str, evidence: list[Path], blocker: dict) -> dict:
    return {
        "schema_version": "HUMAN_TO_ROBOT_CONVERGENCE_LANE_STATE_V1", "task_id": TASK,
        "lane": name, "status": "TERMINAL_WITH_QUALITY_GAPS", "execution": "EXECUTED",
        "structure": "PASS", "quality": quality, "improvement": improvement,
        "adoption": "NOT_ADOPTED", "evidence": [artifact_ref(p) for p in evidence],
        "blocker": blocker, "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 9},
        "updated_at": now_iso(), "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-revision", type=int, required=True)
    args = ap.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((r for r in state["tasks"] if r.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
        raise RuntimeError("TASK_NOT_LIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_SOLE_CURRENT")
    for path in EVIDENCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if (ROOT / "RESULT.json").exists() or RECEIPT_OUT.exists():
        raise FileExistsError("IMMUTABLE_TERMINAL_RESULT_EXISTS")

    motion = load_json(EVIDENCE["motion"]); contact = load_json(EVIDENCE["contact"])
    delivery = load_json(EVIDENCE["delivery"]); sensor = load_json(EVIDENCE["sensor"])
    if motion["window"]["gate_pass_count"] != 0 or motion["full_expansion"]["executed"]:
        raise RuntimeError("UNSUPPORTED_MOTION_PROMOTION")
    if not delivery["full_decode_pass"] or delivery["slot_count"] != 15:
        raise RuntimeError("DELIVERY_NOT_CLOSED")

    blockers = {
        "scene": {"missing": "independent 007 cable-to-human/device association; 031 persistent same-surface IDs; strict observed-hand Contact authority",
                  "consumer": "new Clean, temporal occlusion quality, Contact/R1", "owner": "scene",
                  "unblock_action": "supply frozen independent association/surface identity/strict observation evidence before a new finite successor",
                  "affected": ["007 Clean improvement", "031 same-surface occlusion", "Contact/R1"],
                  "unaffected": ["old Clean candidates", "pixel-screen diagnostic", "R0"]},
        "motion_product": {"missing": "031 longitudinal direction <=15deg, full rotation <=15deg, independent left ROI/model output, approved adapter collision semantics",
                           "consumer": "031 full expansion, complete two-hand product, complete-machine collision", "owner": "motion_product",
                           "unblock_action": "do not retune this rejected IK; obtain new target/observation or collision authority in a separately authorized successor",
                           "affected": ["new 031 motion", "left-hand product", "adapter collision"],
                           "unaffected": ["position feasibility", "old R0", "actual adapter visual rendering"]},
        "sensor": {"missing": "user visual acceptance and independent external wrist truth", "consumer": "visual adoption and absolute accuracy",
                   "owner": "sensor", "unblock_action": "user reviews the three full-session videos; external truth remains separate",
                   "affected": ["visual adoption", "absolute accuracy"], "unaffected": ["native arrays", "common backend", "kinematic review"]},
        "compare": {"missing": "independent truth and a shared accepted new target/background contract", "consumer": "method winner",
                    "owner": "compare", "unblock_action": "retain finite old-contract comparison; rerun both methods only under a newly frozen common contract",
                    "affected": ["winner claim"], "unaffected": ["finite differences", "existing review videos"]},
    }
    states = {
        "scene": lane("scene", "REJECTED_QUALITY_MIXED_NOT_EVALUATED",
                      "CONTACT_GEOMETRY_158_SAMPLES_AND_SCOPE_CORRECTIONS_NO_STRICT_CONTACT",
                      [EVIDENCE["cable"], EVIDENCE["contact"], EVIDENCE["occlusion"]], blockers["scene"]),
        "motion_product": lane("motion_product", "REJECTED_QUALITY",
                      "POSITION_FEASIBILITY_16_OF_16_WITH_DIRECTION_REJECTION_AND_NO_EXPANSION",
                      [EVIDENCE["motion"], EVIDENCE["motion_review"], EVIDENCE["left"], EVIDENCE["collision"]], blockers["motion_product"]),
        "sensor": lane("sensor", "PENDING_USER_VISUAL_REVIEW",
                      "THREE_NEW_FULL_SESSION_REVIEWS_466_FRAMES_NO_RESOLVE",
                      [EVIDENCE["sensor"]], blockers["sensor"]),
        "compare": lane("compare", "INCONCLUSIVE_NO_EXTERNAL_TRUTH",
                      "FINITE_OLD_COMMON_CONTRACT_COMPARISON_NO_WINNER",
                      [EVIDENCE["compare"]], blockers["compare"]),
    }
    for name, value in states.items():
        atomic_json(ROOT / f"lanes/{name}/STATE.json", value)

    terminal_at = now_iso()
    result = {
        "schema_version": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "TERMINAL_WITH_QUALITY_GAPS",
        "task_terminal_status": "REJECTED_QUALITY", "release_status": "INCOMPLETE",
        "terminal_at": terminal_at, "event_driven_early_close": True,
        "execution_summary": {
            "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
            "031_direction_qualified": "102/102_MODEL_DERIVED_NOT_MEASURED_WRIST",
            "031_ik_window_position_pass": "16/16", "031_ik_window_supported_direction_pass": "0/16",
            "031_full_expansion_executed": False, "031_left_sample": "16_RAW_FRAMES_INCONCLUSIVE_FULL_TIMELINE",
            "007_cable_top1": "16/16_AI_PROXY_ASSOCIATION_INCONCLUSIVE", "new_propainter_invocations": 0,
            "contact_qualification_checked": contact["counts"]["qualification_checked"],
            "contact_geometry_screened": contact["counts"]["geometry_screened"],
            "contact_strict_admissible": 0, "robot_r1_executed": 0,
            "adapter_collision": "NOT_EVALUATED_NO_APPROVED_COLLISION_SEMANTICS",
            "occlusion_same_surface": "NOT_EVALUATED_NO_TEMPORAL_SURFACE_IDENTITY",
            "sensor_reviews": f"3/3_{sum(int(r['frames']) for r in sensor['sessions'])}_FRAMES_NEW",
            "local_huro_winner": None, "delivery_slots": "15/15_DECODED_3_NEW_12_REUSED",
        },
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS",
                         "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE", "improvement": "PARTIAL_DIAGNOSTIC_GAINS",
                         "adoption": "NOT_ADOPTED"},
        "reason_codes": [
            "P031_SUPPORTED_LONGITUDINAL_DIRECTION_REJECTED", "P031_FULL_ROTATION_REJECTED",
            "P031_LEFT_HAND_FULL_TIMELINE_NOT_EVALUATED", "C007_CABLE_ASSOCIATION_AUTHORITY_ABSENT",
            "SCENE_CLEAN_QUALITY_REMAINS_REJECTED", "OCCLUSION_TEMPORAL_SURFACE_IDENTITY_ABSENT",
            "ADAPTER_COLLISION_SEMANTICS_UNAUTHORIZED", "CONTACT_STRICT_AUTHORITY_ABSENT",
            "ROBOT_R1_NO_ADMISSIBLE_WINDOW", "LOCAL_HURO_EXTERNAL_TRUTH_ABSENT",
        ],
        "evidence": {k: artifact_ref(v) for k, v in EVIDENCE.items()},
        "lane_states": {k: artifact_ref(ROOT / f"lanes/{k}/STATE.json") for k in states},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "Finite offline convergence execution only. Diagnostic improvements do not grant product quality, Contact, method-winner, training, metric, control or deployment authority.",
    }
    atomic_json(ROOT / "RESULT.json", result)
    result_ref = artifact_ref(ROOT / "RESULT.json")
    atomic_json(RECEIPT_OUT, {"schema_version": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_RECEIPT_V1",
                              "task_id": TASK, "status": "REJECTED_QUALITY", "result": result_ref,
                              "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
                              "delivery_slots": "15/15", "new_videos": 5})
    task.update(status="REJECTED_QUALITY", phase="CONVERGENCE_TERMINAL", attempt=1, pid=None,
                proc_start_ticks=None, gpu_id=None, heartbeat_at=None, updated_at=terminal_at,
                result=result_ref, last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="FINITE_WORK_EXECUTED_WITH_PRODUCT_QUALITY_GAPS")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": TASK, "attempt": 1,
        "status": "REJECTED_QUALITY", "created_at": terminal_at,
        "message": "Finite convergence work terminal: diagnostic gains, 0/4 product quality, 0/4 adoption.",
        "result": result_ref}])[-100:]
    successor_index = {"schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX), "task_packets": [],
        "claim_limit": "Convergence successor is terminal rejected-quality; a new task requires new evidence or a genuinely different authorized contract."}

    delivery_index = (ROOT / "delivery/INDEX_ZH.md").read_text(encoding="utf-8")
    VIS.parent.mkdir(parents=True, exist_ok=True)
    VIS.write_text(delivery_index + "\n## 本轮终态\n\n031新IK未扩片；严格Contact/R1为0；四产品质量0/4、采用0/4。\n", encoding="utf-8")
    PLAN.write_text("# 当前计划：Human→Robot Baseline v1 收敛任务已终态\n\n"
                    f"`{TASK}` 已完成所有可合法执行的有限动作，终态 `REJECTED_QUALITY`。"
                    "真实增量包括031位置可行性、158条Contact几何采样、三片Sensor新回放和15槽位闭合；产品质量仍0/4、采用0/4。\n\n"
                    f"[机器结果]({result_ref['path']}) · [15槽位视频索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)\n", encoding="utf-8")
    ENTRY.write_text("# 当前执行入口：无活动任务\n\n"
                     f"最近任务 `{TASK}` 已终态 `REJECTED_QUALITY`。不要原样重试031 IK、007 Clean或Contact/R1。"
                     "只有新的独立观测、线缆关联、持久表面身份或碰撞语义，才能登记有限后继。\n\n"
                     f"[最终结果]({result_ref['path']}) · [可视化交付](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)\n", encoding="utf-8")
    text = BASELINE.read_text(encoding="utf-8")
    marker = "## 收敛任务终态（2026-09-23）"
    if marker not in text:
        first = text.find("\n")
        section = (f"\n{marker}\n\n`{TASK}` 已终态：031固定窗位置16/16、支持方向0/16，因此未扩片；"
                   "Contact完成510条资格检查、158条实际几何采样，但严格Contact/R1仍为0；Sensor新增466帧完整回放；"
                   "15个交付槽位全部解码，质量通过与产品采用仍为0。详见[交付索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。\n\n")
        BASELINE.write_text(text[:first+1] + section + text[first+1:], encoding="utf-8")

    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_TERMINAL",
        expected_revision=args.expected_revision, task_packet_index_path=INDEX,
        task_packet_index_value=successor_index, generator_path=Path(__file__))
    print(json.dumps({"status": "REJECTED_QUALITY", "revision": published["governance_revision"],
                      "result": result_ref, "delivery": str(VIS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
