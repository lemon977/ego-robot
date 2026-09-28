#!/usr/bin/env python3
"""Publish the immutable S2 H3 evidence checkpoint through governance CAS."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    publish_bundle,
)


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
S1 = REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001"
R2 = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
DOC = REPO_ROOT / "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
VISUAL_INDEX = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md"
PLAN = REPO_ROOT / "docs/current/PLAN.md"
WORK_ENTRY = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
STATUS = REPO_ROOT / "docs/current/STATUS.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checked(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return artifact_ref(path)


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError(f"JUNIT_WITHOUT_SUITES:{path}")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--allow-early", action="store_true")
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    task_state = load_json(TASK_STATE_PATH)
    if task_state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("S2_IS_NOT_CURRENT")

    row = next(item for item in task_state["tasks"] if item.get("task_id") == TASK)
    t0 = datetime.fromisoformat(str(row["t0"]))
    elapsed = (datetime.now().astimezone() - t0).total_seconds()
    if elapsed < 3 * 3600 and not args.allow_early:
        raise RuntimeError(f"H3_NOT_REACHED:{elapsed:.1f}")

    paths = {
        "motion_audit_031": ROOT / "lanes/motion_product/h3_motion_audit_031/attempt_0002/RESULT.json",
        "position_only_031": ROOT / "lanes/motion_product/position_first_031/attempt_0002/RESULT.json",
        "two_stage_031": ROOT / "lanes/motion_product/two_stage_031/attempt_0001/RESULT.json",
        "source_provenance_031": ROOT / "lanes/motion_product/source_provenance_031/attempt_0001/RESULT.json",
        "target_authority_031": ROOT / "lanes/motion_product/target_authority_031/attempt_0001/RESULT.json",
        "adapter_007": ROOT / "lanes/motion_product/h3_adapter_ab/get_potato_chips_0915_007/attempt_0002/RESULT.json",
        "adapter_031": ROOT / "lanes/motion_product/h3_adapter_ab/play_cards_0915_031/attempt_0002/RESULT.json",
        "occlusion_031": ROOT / "lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json",
        "attachment_clean_007": S1 / "lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json",
        "sensor_reuse": ROOT / "lanes/sensor/reuse_audit_v1/RESULT.json",
        "compare_reuse": ROOT / "lanes/compare/reuse_audit_v1/RESULT.json",
        "compare_adapter_refresh_007": ROOT / "lanes/compare/adapter_refresh_007/attempt_0001/RESULT.json",
        "compare_adapter_refresh_031": ROOT / "lanes/compare/adapter_refresh_031/attempt_0001/RESULT.json",
        "contact_funnel": R2 / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json",
        "stereo_preflight_007": S1 / "lanes/geometry_contact/encoded_stereo_preflight_v1/get_potato_chips_0915_007/RESULT.json",
        "regression_project_local": ROOT / "PYTEST_PRE_H3_PROJECT_LOCAL.xml",
        "regression_collision": ROOT / "PYTEST_COLLISION_PRE_H3_FIXED.xml",
    }
    refs = {name: checked(path) for name, path in paths.items()}
    motion = load_json(paths["motion_audit_031"])
    two_stage = load_json(paths["two_stage_031"])
    source = load_json(paths["source_provenance_031"])
    target_authority = load_json(paths["target_authority_031"])
    occlusion = load_json(paths["occlusion_031"])
    stereo_007 = load_json(paths["stereo_preflight_007"])
    compare_adapter_007 = load_json(paths["compare_adapter_refresh_007"])
    compare_adapter_031 = load_json(paths["compare_adapter_refresh_031"])
    regression_project = junit_counts(paths["regression_project_local"])
    regression_collision = junit_counts(paths["regression_collision"])
    if regression_project != {"tests": 1490, "failures": 0, "errors": 0, "skipped": 1}:
        raise RuntimeError(f"PROJECT_LOCAL_REGRESSION_NOT_CLOSED:{regression_project}")
    if regression_collision != {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"COLLISION_REGRESSION_NOT_CLOSED:{regression_collision}")

    product_paths = {
        "get_potato_chips_0915_007": ROOT / "lanes/motion_product/formal_product_007/attempt_0001/PRODUCT_RESULT.json",
        "play_cards_0915_031": ROOT / "lanes/motion_product/formal_product_031/attempt_0003/PRODUCT_RESULT.json",
        "get_potato_chips_0902_103": ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0001/PRODUCT_RESULT.json",
        "play_cards_0902_042": ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0001/PRODUCT_RESULT.json",
    }
    products = {}
    for session_id, path in product_paths.items():
        value = load_json(path)
        if value.get("execution") != "EXECUTED" or value.get("structure") != "PASS":
            raise RuntimeError(f"PRODUCT_NOT_STRUCTURALLY_COMPLETE:{session_id}")
        products[session_id] = {
            "frames": f"{value['decoded_frames']}/{value['expected_frames']}",
            "quality": value["quality"],
            "adoption": value["adoption"],
            "adapter_visible_frames": value["adapter_visible_frames"],
            "geometry_mode": value.get("geometry_mode", value.get("occlusion_scope")),
            "known_decision_coverage": value["known_decision_coverage"],
            "unknown_decision_ratio": value["unknown_decision_ratio"],
            "result": checked(path),
            "video": value["product_video"],
        }

    checkpoint = {
        "schema_version": "HUMAN_TO_ROBOT_S2_H3_CHECKPOINT_V1",
        "task_id": TASK,
        "checkpoint": "H3",
        "created_at": now(),
        "elapsed_seconds": elapsed,
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE",
        "adoption": "NOT_ADOPTED",
        "products": products,
        "h3_findings": {
            "adapter": (
                "Real STEP-derived adapter is consumed by RGB/depth/component-ID and actual q/FK. "
                "It is visible in 007/0902; in the frozen 031 window it is outside view or hidden. "
                "Collision coverage remains UNVERIFIED_VISUAL_GEOMETRY_ONLY."
            ),
            "031_display_consistency": {
                "renderer_fk_vs_saved_actual_mm_p95": motion["renderer_fk_vs_saved_actual_mm"]["p95"],
                "producer_fk_vs_saved_actual_mm_p95": motion["producer_fk_vs_saved_actual_mm"]["p95"],
                "classification": "DISPLAY_USES_ACTUAL_Q_FK_NOT_TARGET_WRIST",
            },
            "031_target_tracking": {
                "full_pose_position_mm": motion["full_pose_target_residual_mm"],
                "position_only_best_mm": motion["position_only_best_residual_mm"],
                "two_stage_position_mm": two_stage["candidate_position_mm"],
                "two_stage_rotation_deg": two_stage["candidate_rotation_deg"],
                "two_stage_pose_gate_pass": two_stage["pose_gate_pass_count"],
                "arm_limit_hit_rows_two_stage": two_stage["arm_limit_hit_count"],
                "classification": (
                    "FIXED_BASE_FULL_POSE_TARGET_CONFLICTS_WITH_JOINT_CONSTRAINTS_OR_TARGET_ORIENTATION; "
                    "NOT_MOUNT_LENGTH_ONLY; NOT_RENDERER_DISPLAY_ERROR; PHYSICAL_UNREACHABILITY_NOT_PROVEN"
                ),
                "target_definition": target_authority["target_definition"],
                "target_authority_interpretation": target_authority["interpretation"],
            },
            "031_source": {
                "sides": source["side_rows"],
                "writer_semantics": source["writer_semantics"],
                "temporal_authority": source["temporal_authority"],
                "identity_authority": source["identity_authority"],
                "left_status": source["left_status"],
            },
            "031_occlusion": {
                "ownership_counts": occlusion["ownership_counts"],
                "known_decision_coverage": occlusion["known_decision_coverage"],
                "unknown_decision_ratio": occlusion["unknown_decision_ratio"],
                "continuous_switches": occlusion["consecutive_known_ownership_switches"],
                "continuous_comparable_pixels": occlusion["consecutive_known_comparable_pixels"],
                "quality": "INCONCLUSIVE_NO_FROZEN_TEMPORAL_SWITCH_GATE",
            },
            "007_scene": (
                "Bounded 16-frame attachment repair is structurally real and locally improved, but "
                "the long thin cable and full-session removal/protection evidence remain unresolved."
            ),
            "007_stereo": {
                "decision": stereo_007["stereo_geometry_check"]["decision"],
                "total_robust_matches": stereo_007["stereo_geometry_check"]["metrics"]["total_robust_matches"],
                "minimum_total_robust_matches": stereo_007["stereo_geometry_check"]["thresholds"]["minimum_total_robust_matches"],
                "successor_authorized": stereo_007["stereo_geometry_check"]["gpu_successor_authorized"],
                "action": "PRESERVE_REJECTION_DO_NOT_LOWER_GATE_OR_SELECT_NEW_FRAMES",
            },
            "contact_robot_r1": "0 screened / 0 executed / 0 adopted; no admissible metric window",
            "sensor": "3/3 frozen common-backend outputs and 466-frame videos reverified; no refit",
            "regression": {
                "project_local_main": regression_project,
                "collision_dedicated_fixture": regression_collision,
                "hardlink_transaction_tests": "14_NOT_EVALUATED_ON_CPFS_PROJECT_LOCAL_TMPDIR",
                "interpretation": "1495 passed and 1 skipped across the two legal project-local groups; this is not a full-suite PASS because 14 filesystem-transaction tests remain environment-unavailable.",
            },
            "local_huro": {
                "same_target_numeric_evidence": "REVERIFIED_NO_INDEPENDENT_TRUTH_NO_WINNER",
                "adapter_refresh_007": {
                    "decoded_frames": compare_adapter_007["decoded_frames"],
                    "adapter_visible_frames": compare_adapter_007["adapter_visible_frames"],
                    "real_adapter_consumed_both_methods": compare_adapter_007["real_adapter_consumed_both_methods"],
                },
                "adapter_refresh_031": {
                    "decoded_frames": compare_adapter_031["decoded_frames"],
                    "adapter_visible_frames": compare_adapter_031["adapter_visible_frames"],
                    "real_adapter_consumed_both_methods": compare_adapter_031["real_adapter_consumed_both_methods"],
                    "interpretation": "ZERO_VISIBLE_PIXELS_IN_FROZEN_VIEW_NOT_MISSING_ASSET_CONSUMPTION",
                },
                "collision_scope": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
                "occlusion_scope": "NOT_COMPARED_REJECTED_BACKGROUND_DIAGNOSTIC",
            },
        },
        "evidence": refs,
        "authority": {
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
        "next_actions": [
            "Do not adopt either 031 bounded IK candidate; retain actual-q/FK rendering.",
            "Do not expand the 007 attachment canary until frozen-window visual review supports it.",
            "Continue formal-entry/resume and artifact verification; do not invent Depth or Contact.",
            "At H6 decide whether any remaining full-session computation is evidence-justified.",
        ],
    }
    checkpoint_path = ROOT / "checkpoints/H3_RESULT.json"
    if checkpoint_path.exists():
        raise RuntimeError(f"IMMUTABLE_H3_EXISTS:{checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(checkpoint_path, checkpoint)
    required_pointer = ROOT / "H3_PROGRESS.json"
    if required_pointer.exists():
        raise RuntimeError(f"IMMUTABLE_H3_POINTER_EXISTS:{required_pointer}")
    atomic_json(required_pointer, {
        "schema_version": "HUMAN_TO_ROBOT_S2_H3_PROGRESS_POINTER_V1",
        "task_id": TASK,
        "checkpoint": "H3",
        "result": checked(checkpoint_path),
        "formal_product_candidates_complete": "4/4",
        "formal_product_quality_pass": "0/4",
        "claim_limit": "Navigation pointer only; the immutable H3 result is authoritative.",
    })

    # Current lane states are mutable projections; evidence remains immutable.
    for lane in ("scene", "motion_product", "sensor", "compare"):
        state_path = ROOT / f"lanes/{lane}/STATE.json"
        value = load_json(state_path)
        value["checkpoint"] = "H3"
        value["checkpoint_result"] = checked(checkpoint_path)
        value["updated_at"] = now()
        if lane == "motion_product":
            value["evidence"] += [refs["two_stage_031"], refs["source_provenance_031"]]
            value["evidence"] += [refs["target_authority_031"]]
            value["improvement"] = "FOUR_STRUCTURAL_PRODUCTS_REAL_ADAPTER_TRUE_FK_SOURCE_LINEAGE_CLOSED_NO_MOTION_QUALITY_PASS"
        if lane == "compare":
            value["evidence"] += [
                refs["compare_adapter_refresh_007"],
                refs["compare_adapter_refresh_031"],
            ]
            value["improvement"] = "LOCAL_HURO_RENDER_REFRESH_CONSUMES_REAL_ADAPTER_NO_NEW_SOLVER_NO_WINNER"
        atomic_json(state_path, value)

    marker = "## S2 H3 检查点"
    text = DOC.read_text(encoding="utf-8")
    if marker not in text:
        text += (
            f"\n{marker}\n\n"
            "H3 已完成装配、显示、动作跟踪与遮挡的分开验收。连接件已进入正式 renderer 的 RGB／深度／部件 ID；"
            "031 视频忠实消费实际 q/FK，但 full-pose 腕目标仍未跟上。position-only 与两阶段候选均被拒绝：前者牺牲旋转，"
            "后者回到原 full-pose 折衷且 85/102 帧触及手臂限位。来源链确认右手 102 帧是非直接 detector ROI 上的 HaWoR 模型输出，"
            "`inferred` 不等于 motion infiller；左手仍无 ROI、无模型输出。031 遮挡在固定窗有约82.87%已判归属，17.13% UNKNOWN，"
            "连续切换尚无冻结质量门，因此不采用。四条正式候选结构完成，质量通过仍为0/4。\n"
        )
        DOC.write_text(text, encoding="utf-8")

    visual_text = VISUAL_INDEX.read_text(encoding="utf-8")
    if marker not in visual_text:
        visual_text += (
            f"\n{marker}\n\n"
            f"- 证据快照：[{checkpoint_path}]({checkpoint_path})\n"
            f"- 031 来源旁车：[{paths['source_provenance_031']}]({paths['source_provenance_031']})\n"
            f"- 031 两阶段 IK 诊断：[{paths['two_stage_031']}]({paths['two_stage_031']})\n"
            f"- 031 腕目标权限：[{paths['target_authority_031']}]({paths['target_authority_031']})\n"
            f"- 007 Local/HuRo 真实连接件刷新：[{paths['compare_adapter_refresh_007']}]({paths['compare_adapter_refresh_007']})\n"
            f"- 031 Local/HuRo 真实连接件刷新：[{paths['compare_adapter_refresh_031']}]({paths['compare_adapter_refresh_031']})\n"
        )
        VISUAL_INDEX.write_text(visual_text, encoding="utf-8")

    PLAN.write_text(
        "# 当前计划入口：Human→Robot S2 执行中\n\n"
        f"当前唯一任务为 `{TASK}`，总预算12小时，H3检查点已形成。"
        "旧 V5、R1、R2、S1 均保持不可变；S2 只在当前有限范围内收敛装配、动作、遮挡和四支线证据。\n\n"
        "## H3事实\n\n"
        "- 四条正式产品候选均结构完成并完整解码；质量通过0/4，均未采用。\n"
        "- 正式renderer按实际q/FK工作并消费真实连接件；031冻结视角连接件不可见不等于未加载。\n"
        "- 031 full-pose腕目标仍有约77.5/104.4 mm的P50/P95残差；两个有界候选均被拒绝。\n"
        "- 031遮挡可判区域与UNKNOWN分开记录；007及0902没有合格Depth，不制造遮挡结论。\n"
        "- Sensor仅复核冻结共同后端；Local/HuRo无独立真值，不宣布胜者。\n\n"
        "## 后续\n\n"
        "H6只决定是否还有证据支持全片计算；H9冻结实际组件组合；H10后仅验收和收敛。"
        "不得降低Scene、Depth、Contact、碰撞或动作门，不得以视频存在代替质量通过。\n\n"
        f"机器证据：[H3_RESULT]({checkpoint_path})；[S2视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n",
        encoding="utf-8",
    )
    WORK_ENTRY.write_text(
        "# 当前入口：Human→Robot S2 H3\n\n"
        f"当前任务：`{TASK}`。先运行治理校验，再读 [H3结果]({checkpoint_path})、"
        "[当前状态](STATUS.json) 和 [视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n\n"
        "当前不是产品成功：4/4仅表示结构候选完整，质量通过0/4。031实际q/FK显示忠实，但full-pose目标冲突且"
        "目标旋转只具模型派生开发权限；007局部Clean仍残留长线缆；除031外无获准Depth；Robot R1为0。\n\n"
        "后续执行者只能完成已登记的H6/H9/最终验收动作。不得重跑同签名基础模型、降低1500匹配门、"
        "把视觉连接件升级为碰撞覆盖、把UNKNOWN算正确，或把Local/HuRo诊断写成胜负。"
        "所有写入继续限制在项目目录内。\n",
        encoding="utf-8",
    )

    timestamp = now()
    row["updated_at"] = timestamp
    task_state["recent_events"] = (
        task_state.get("recent_events", [])
        + [{
            "task_id": TASK,
            "attempt": 1,
            "status": "PENDING",
            "created_at": timestamp,
            "message": "S2 H3 checkpoint published; finite H6/H9 work remains.",
            "result": checked(checkpoint_path),
        }]
    )[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH), task_state,
        event_type="HUMAN_TO_ROBOT_S2_H3_PUBLISHED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    atomic_json(STATUS, {
        "schema_version": "HUMAN_TO_ROBOT_S2_CURRENT_STATUS_V1",
        "generated_at": now(),
        "governance_revision": published["governance_revision"],
        "status": "ACTIVE_H3_CHECKPOINT",
        "active_tasks": [TASK],
        "next_task": {"task_id": TASK, "checkpoint": "H6"},
        "checkpoint": checked(checkpoint_path),
        "counts": {
            "formal_product_candidates_complete": 4,
            "formal_product_quality_pass": 0,
            "formal_product_adopted": 0,
            "sensor_sessions_reverified": 3,
            "local_huro_adapter_refresh_sessions": 2,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
        },
        "quality_axes": {
            "execution": "EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE",
            "adoption": "NOT_ADOPTED",
        },
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    })
    print(json.dumps({
        "status": "PASS",
        "governance_revision": published["governance_revision"],
        "checkpoint": checked(checkpoint_path),
        "formal_products": "4/4",
        "quality_pass": "0/4",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
