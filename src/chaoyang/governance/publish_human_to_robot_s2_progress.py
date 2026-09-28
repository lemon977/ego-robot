#!/usr/bin/env python3
"""Publish evidence-backed S2 progress and four-lane states through CAS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    atomic_json, artifact_ref, load_json, publish_bundle,
)


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
R2 = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
DOC = REPO_ROOT / "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
VISUAL_INDEX = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checked(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return artifact_ref(path)


def state(*, lane: str, execution: str, structure: str, quality: str, adoption: str,
          improvement: str, evidence: list[dict[str, object]], blocker: dict[str, object] | None) -> dict:
    return {
        "schema_version": "HUMAN_TO_ROBOT_S2_LANE_STATE_V1", "task_id": TASK,
        "lane": lane, "status": "RUNNING", "execution": execution,
        "structure": structure, "quality": quality, "adoption": adoption,
        "improvement": improvement, "evidence": evidence, "blocker": blocker,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "updated_at": now(),
        "writer": {"executor_epoch": 8, "pid": os.getpid(), "proc_start_ticks": None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    task_state = load_json(TASK_STATE_PATH)
    if task_state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("S2_IS_NOT_CURRENT")

    current_products = {
        "get_potato_chips_0915_007": ROOT / "lanes/motion_product/formal_product_007/attempt_0001/PRODUCT_RESULT.json",
        "play_cards_0915_031": ROOT / "lanes/motion_product/formal_product_031/attempt_0003/PRODUCT_RESULT.json",
        "get_potato_chips_0902_103": ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0001/PRODUCT_RESULT.json",
        "play_cards_0902_042": ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0001/PRODUCT_RESULT.json",
    }
    products = []
    for sid, path in current_products.items():
        value = load_json(path)
        if value.get("session_id") != sid or value.get("execution") != "EXECUTED" or value.get("structure") != "PASS":
            raise RuntimeError(f"PRODUCT_NOT_STRUCTURALLY_EXECUTED:{sid}")
        products.append({
            "session_id": sid, "decoded_frames": value["decoded_frames"],
            "expected_frames": value["expected_frames"], "quality": value["quality"],
            "adoption": value["adoption"], "adapter_visible_frames": value["adapter_visible_frames"],
            "known_decision_coverage": value["known_decision_coverage"],
            "unknown_decision_ratio": value["unknown_decision_ratio"],
            "geometry_mode": value.get("geometry_mode", value["occlusion_scope"]),
            "result": checked(path),
            "video": value["product_video"],
        })

    evidence = {
        "adapter_ab_007": checked(ROOT / "lanes/motion_product/h3_adapter_ab/get_potato_chips_0915_007/attempt_0002/RESULT.json"),
        "adapter_ab_031": checked(ROOT / "lanes/motion_product/h3_adapter_ab/play_cards_0915_031/attempt_0002/RESULT.json"),
        "motion_audit_031": checked(ROOT / "lanes/motion_product/h3_motion_audit_031/attempt_0002/RESULT.json"),
        "position_candidate_031": checked(ROOT / "lanes/motion_product/position_first_031/attempt_0002/RESULT.json"),
        "occlusion_031": checked(ROOT / "lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json"),
        "sensor_reuse": checked(ROOT / "lanes/sensor/reuse_audit_v1/RESULT.json"),
        "compare_reuse": checked(ROOT / "lanes/compare/reuse_audit_v1/RESULT.json"),
        "attachment_clean_007": checked(REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json"),
        "contact_funnel": checked(R2 / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json"),
    }

    scene_state = state(
        lane="scene", execution="EXECUTED", structure="PASS", quality="REJECTED_QUALITY",
        adoption="NOT_ADOPTED", improvement="OCCLUSION_INTERFACE_EXECUTED_AND_UINT16_MASK_DECODE_FIXED_CLEAN_QUALITY_UNRECOVERED",
        evidence=[evidence["occlusion_031"], evidence["attachment_clean_007"], evidence["contact_funnel"]],
        blocker={
            "missing": "trusted full-session hand/device/cable removal and qualified depth outside 031",
            "consumer": "technical session adoption, visible-surface ownership, Contact/Robot R1",
            "owner": "scene", "unblock_action": "independent review of the bounded attachment repair; no full expansion unless it improves the frozen window; preserve 007 depth rejection",
            "affected": ["Clean adoption", "007/0902 occlusion", "Robot R1"],
            "unaffected": ["formal candidate rendering", "Motion/R0", "Sensor", "numeric Local/HuRo"],
        })
    motion_state = state(
        lane="motion_product", execution="EXECUTED", structure="PASS", quality="REJECTED_QUALITY",
        adoption="CANDIDATE_ONLY", improvement="FOUR_FORMAL_S2_PRODUCTS_WITH_REAL_ADAPTER_AND_NO_RGB_FALLBACK",
        evidence=[evidence["adapter_ab_007"], evidence["adapter_ab_031"], evidence["motion_audit_031"], evidence["position_candidate_031"]]
                 + [item["result"] for item in products],
        blocker={
            "missing": "031 full-pose target tracking and left-hand evidence; all Scene products remain quality rejected",
            "consumer": "session technical pass and product adoption", "owner": "motion_product",
            "unblock_action": "keep position-only candidate rejected; trace full-pose objective/optimizer and source qualification without moving the rendered hand away from actual FK",
            "affected": ["031 motion quality", "main product adoption"],
            "unaffected": ["actual-q/FK rendering", "007 and 0902 structural candidates", "Sensor"],
        })
    sensor_state = state(
        lane="sensor", execution="EXECUTED_REUSED_FROZEN_EVIDENCE", structure="PASS",
        quality="INCONCLUSIVE_MIXED_KINEMATIC_ONLY", adoption="CANDIDATE_ONLY",
        improvement="NO_NEW_MODEL_RUN_FROZEN_466_FRAME_COMMON_BACKEND_AND_VIDEO_DECODE_REVERIFIED",
        evidence=[evidence["sensor_reuse"]],
        blocker={
            "missing": "independent external wrist truth and user visual acceptance",
            "consumer": "visual alignment adoption", "owner": "sensor",
            "unblock_action": "review frozen videos; do not refit 101 or treat 098 as labelled",
            "affected": ["sensor visual adoption"], "unaffected": ["native MANUS arrays", "kinematic-only backend evidence"],
        })
    compare_state = state(
        lane="compare", execution="EXECUTED_REUSED_FROZEN_EVIDENCE", structure="PASS",
        quality="INCONCLUSIVE", adoption="NOT_ADOPTED",
        improvement="SAME_TARGET_DENOMINATOR_AND_FULL_VIDEO_DECODE_REVERIFIED_NO_WINNER",
        evidence=[evidence["compare_reuse"]],
        blocker={
            "missing": "adopted Clean and independent target truth; HuRo limits remain",
            "consumer": "method winner and final same-background comparison", "owner": "compare",
            "unblock_action": "retain numeric comparison; refresh visuals only after an adopted same-session Clean exists",
            "affected": ["winner claim", "final visual comparison"], "unaffected": ["same-target numeric diagnostics"],
        })
    for name, value in (("scene", scene_state), ("motion_product", motion_state),
                        ("sensor", sensor_state), ("compare", compare_state)):
        atomic_json(ROOT / f"lanes/{name}/STATE.json", value)

    progress = {
        "schema_version": "HUMAN_TO_ROBOT_S2_PROGRESS_V1", "task_id": TASK,
        "created_at": now(), "checkpoint": "EARLY_PROGRESS_BEFORE_H3",
        "products": products, "evidence": evidence,
        "summary": {
            "formal_product_candidates_complete": "4/4",
            "formal_product_quality_pass": "0/4",
            "adapter_real_consumption": "PASS_007_AND_0902;_031_OUT_OF_VIEW_OR_HIDDEN",
            "031_visible_surface_ownership_known_coverage": 0.8785457001350743,
            "031_visible_surface_unknown_ratio": 0.12145429986492572,
            "007_and_0902_occlusion": "UNKNOWN_NO_QUALIFIED_DEPTH",
            "robot_r1_windows_executed": 0, "robot_r1_windows_adopted": 0,
            "sensor_common_backend": "3/3_REUSED_AND_REVERIFIED",
            "local_huro_winner": None,
        },
        "authority": {"training_eligible": False, "control_ground_truth": False,
                      "physical_deployable": False, "external_metric_authority": False},
    }
    checkpoint = ROOT / "checkpoints/PROGRESS_LIVE_V1.json"
    if checkpoint.exists():
        raise RuntimeError(f"IMMUTABLE_PROGRESS_EXISTS:{checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(checkpoint, progress)

    VISUAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Human→Robot S2 当前视频索引", "",
        "> 实体只保存在对应 attempt；这里仅导航。当前 4/4 是结构完整候选，质量通过仍为 0/4。", "",
        "## S2 正式产品候选", "",
    ]
    for item in products:
        lines.append(f"- `{item['session_id']}`：[{item['decoded_frames']}帧 robot.mp4]({item['video']['path']})；quality=`{item['quality']}`，geometry=`{item['geometry_mode']}`。")
    lines += ["", "## 固定证据视频", "",
              f"- 031 H3 连接件/遮挡 A-B-C：[{(ROOT / 'lanes/scene/h3_occlusion_031/attempt_0003/H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4')}]({ROOT / 'lanes/scene/h3_occlusion_031/attempt_0003/H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4'})",
              f"- 007 局部附件 Clean canary：[{(REPO_ROOT / '_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/ATTACHMENT_CLEAN_REVIEW.mp4')}]({REPO_ROOT / '_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/ATTACHMENT_CLEAN_REVIEW.mp4'})",
              "", "Sensor 与 Local/HuRo 冻结视频由上述 reuse audit 中的 SHA 引用；没有复制第二份实体。", ""]
    VISUAL_INDEX.write_text("\n".join(lines), encoding="utf-8")

    marker = "## S2 进行中（2026-09-23）"
    text = DOC.read_text(encoding="utf-8")
    if marker not in text:
        insert = (
            f"\n{marker}\n\n"
            "S2 已完成真实连接件 renderer、深度/部件 ID 接口、visible-surface compositor 与正式 CLI 接入。"
            "007、031、0902_103、0902_042 已各生成一条完整可解码正式候选并通过同签名 resume；当前均因 Scene／运动／几何证据未满足而未采用。"
            "031 已证明画面忠实消费实际 q/FK，但原 full-pose 腕目标残差仍为 P50 77.5 mm、P95 104.4 mm；position-only 诊断改善位置却破坏旋转，因此被拒绝。"
            "031 可见表面遮挡覆盖约87.85%，其余保持 UNKNOWN；007及0902没有合格Depth，未制造排序。"
            "Sensor三片与Local/HuRo证据采用冻结复核，不重复推理，也不宣布外部精度或方法胜者。"
            f" [当前S2视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)\n\n"
        )
        first_newline = text.find("\n")
        DOC.write_text(text[:first_newline + 1] + insert + text[first_newline + 1:], encoding="utf-8")

    published = publish_bundle(
        load_json(AUTHORITY_PATH), task_state,
        event_type="HUMAN_TO_ROBOT_S2_PROGRESS_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PASS", "governance_revision": published["governance_revision"],
                      "checkpoint": str(checkpoint), "products": "4/4", "quality_pass": "0/4"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
