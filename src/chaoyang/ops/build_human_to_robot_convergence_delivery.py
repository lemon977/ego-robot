#!/usr/bin/env python3
"""Build the fixed 15-slot delivery index from unique, decoded artifacts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2

from chaoyang.governance.common import artifact_ref, load_json

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
ROOT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
S2 = REPO / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
INDEX = REPO / "tasks/current/INDEX.json"
OUT = ROOT / "delivery"


def decode(path: Path) -> int:
    cap = cv2.VideoCapture(str(path)); n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def slot(slot_id: str, group: str, session: str, expected: int, path: Path,
         provenance: str, quality: str, artifact_type: str = "FULL_SESSION") -> dict:
    if not path.is_file():
        return {"slot_id": slot_id, "group": group, "session_id": session,
                "status": "NOT_PRODUCED", "quality": "NOT_EVALUATED",
                "reason": f"MISSING_EXACT_ARTIFACT:{path}"}
    decoded = decode(path)
    return {"slot_id": slot_id, "group": group, "session_id": session,
            "artifact_type": artifact_type, "status": provenance, "quality": quality,
            "adoption": "NOT_ADOPTED", "expected_frames": expected, "decoded_frames": decoded,
            "frame_count_pass": decoded == expected, "video": artifact_ref(path)}


def main() -> int:
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    slots = [
        slot("SCENE_007", "SCENE_CLEAN_REVIEW", "get_potato_chips_0915_007", 378,
             R2/"lanes/lane1_scene/clean_candidate_007_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("SCENE_031", "SCENE_CLEAN_REVIEW", "play_cards_0915_031", 149,
             R2/"lanes/lane1_scene/clean_candidate_031_wave5/SCENE_CLEAN_CANDIDATE_REVIEW.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("SCENE_103", "SCENE_CLEAN_REVIEW", "get_potato_chips_0902_103", 284,
             R2/"lanes/lane1_scene/clean_candidate_103_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("SCENE_042", "SCENE_CLEAN_REVIEW", "play_cards_0902_042", 171,
             R2/"lanes/lane1_scene/clean_candidate_042_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("SENSOR_097", "SENSOR_REVIEW", "play_cards_0916_097", 165,
             ROOT/"lanes/sensor/review_v1/play_cards_0916_097/SENSOR_COMPLETE_REVIEW.mp4", "NEW", "PENDING_USER_VISUAL_REVIEW"),
        slot("SENSOR_098", "SENSOR_REVIEW", "play_cards_0916_098", 179,
             ROOT/"lanes/sensor/review_v1/play_cards_0916_098/SENSOR_COMPLETE_REVIEW.mp4", "NEW", "PENDING_USER_VISUAL_REVIEW"),
        slot("SENSOR_101", "SENSOR_REVIEW", "play_cards_0916_101", 122,
             ROOT/"lanes/sensor/review_v1/play_cards_0916_101/SENSOR_COMPLETE_REVIEW.mp4", "NEW", "PENDING_USER_VISUAL_REVIEW"),
        slot("MOTION_007", "MOTION_PRODUCT_REVIEW", "get_potato_chips_0915_007", 378,
             R2/"lanes/lane2_motion/product_candidate_007_wave6/robot_candidate.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("MOTION_031", "MOTION_PRODUCT_REVIEW", "play_cards_0915_031", 149,
             R2/"lanes/lane2_motion/product_candidate_031_wave6/robot_candidate.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("COMPARE_007", "LOCAL_HURO_REVIEW", "get_potato_chips_0915_007", 378,
             S2/"lanes/compare/adapter_refresh_007/attempt_0001/LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4", "REUSED", "INCONCLUSIVE_NO_EXTERNAL_TRUTH"),
        slot("COMPARE_031", "LOCAL_HURO_REVIEW", "play_cards_0915_031", 149,
             S2/"lanes/compare/adapter_refresh_031/attempt_0001/LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4", "REUSED", "INCONCLUSIVE_NO_EXTERNAL_TRUTH"),
        slot("PRODUCT_007", "PURE_ROBOT_PRODUCT", "get_potato_chips_0915_007", 378,
             S2/"lanes/motion_product/formal_product_007/attempt_0002/robot.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("PRODUCT_031", "PURE_ROBOT_PRODUCT", "play_cards_0915_031", 149,
             S2/"lanes/motion_product/formal_product_031/attempt_0004/robot.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("PRODUCT_103", "PURE_ROBOT_PRODUCT", "get_potato_chips_0902_103", 284,
             S2/"lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/robot.mp4", "REUSED", "REJECTED_QUALITY"),
        slot("PRODUCT_042", "PURE_ROBOT_PRODUCT", "play_cards_0902_042", 171,
             S2/"lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/robot.mp4", "REUSED", "REJECTED_QUALITY"),
    ]
    if len(slots) != 15:
        raise AssertionError(len(slots))
    manifest = {
        "schema_version": "HUMAN_TO_ROBOT_CONVERGENCE_15_SLOT_DELIVERY_V1", "task_id": TASK,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "slot_count": 15, "produced_or_reused": sum(s["status"] != "NOT_PRODUCED" for s in slots),
        "new": sum(s["status"] == "NEW" for s in slots), "reused": sum(s["status"] == "REUSED" for s in slots),
        "full_decode_pass": all(s.get("frame_count_pass", False) for s in slots if s["status"] != "NOT_PRODUCED"),
        "quality_pass": 0, "adopted": 0, "slots": slots,
        "supplementary_evidence": {
            "031_ik_window": artifact_ref(ROOT/"lanes/motion_product/partial_direction_031/review_v1/031_IK_WINDOW_OLD_NEW_REVIEW.mp4"),
            "031_left_raw_sample": artifact_ref(ROOT/"lanes/motion_product/left_evidence_031/wave0/031_LEFT_EVIDENCE_FIXED_SAMPLE_REVIEW.mp4"),
            "007_cable_window": artifact_ref(ROOT/"lanes/scene/cable_007/wave0/CABLE_TOP1_EVIDENCE_REVIEW.mp4"),
            "031_occlusion_abc": artifact_ref(S2/"lanes/scene/h3_occlusion_031/attempt_0003/H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4"),
        },
        "claim_limit": "Slots are navigational evidence, not 15 new runs. Reused and rejected artifacts remain so; no product quality or adoption is granted.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    OUT.mkdir(parents=True)
    result = OUT/"DELIVERY_MANIFEST.json"
    result.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    rows = []
    for s in slots:
        link = f"[{Path(s['video']['path']).name}]({s['video']['path']})" if "video" in s else "—"
        rows.append(f"| {s['slot_id']} | {s['session_id']} | {s['status']} | {s['quality']} | {s.get('decoded_frames','—')}/{s.get('expected_frames','—')} | {link} |")
    md = "# Human→Robot Baseline v1：15槽位交付索引\n\n"
    md += "> 这是唯一实体的导航表，不是15次新求解。所有槽位当前采用均为0。\n\n"
    md += "| 槽位 | 会话 | 新增/复用 | 质量 | 解码 | 视频 |\n|---|---|---|---|---:|---|\n" + "\n".join(rows) + "\n\n"
    md += "## 补充证据\n\n"
    for key, value in manifest["supplementary_evidence"].items():
        md += f"- {key}: [{Path(value['path']).name}]({value['path']})\n"
    md += "\n质量通过：0/15；产品采用：0/4。Sensor新回放仍待用户视觉审阅。\n"
    (OUT/"INDEX_ZH.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(result), "full_decode": manifest["full_decode_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
