#!/usr/bin/env python3
"""Build the bounded R2.2 recovery package from immutable worker receipts.

This tool is intentionally read-only with respect to the current governance
ledger.  It records missing results as pending and never upgrades development
evidence to authority.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h"
DEFAULT_OUT = RUN / "lanes/doc_r22/attempts/attempt_0003_final_recovery"

RESULTS = {
    "g0": RUN / "g0/attempts/attempt_0001/RESULT.json",
    "g0_snapshot_repair": RUN / "g0/attempts/attempt_0002_content_addressed_repair/G0_SNAPSHOT_REPAIR_RECEIPT.json",
    "robot_v77_adopt12": RUN / "lanes/robot_v77_adopt12_r22/attempts/attempt_0002/RESULT.json",
    "robot_v77_batch001": RUN / "lanes/robot_v77_batch_001_r22/attempts/attempt_0001/RESULT.json",
    "robot_v77_batch001_final": RUN / "lanes/robot_v77_batch_001_r22/attempts/attempt_0002/RESULT.json",
    "robot_v77_batch002": RUN / "lanes/robot_v77_batch_002_r22/attempts/attempt_0001/RESULT.json",
    "robot_v77_batch003": RUN / "lanes/robot_v77_batch_003_r22/attempts/attempt_0001/RESULT.json",
    "robot_v77_batch004": RUN / "lanes/robot_v77_batch_004_r22/attempts/attempt_0001/RESULT.json",
    "robot_v77_batch004_final": RUN / "lanes/robot_v77_batch_004_r22/attempts/attempt_0002/RESULT.json",
    "robot_v77_batch005": RUN / "lanes/robot_v77_batch_005_r22/attempts/attempt_0001/RESULT.json",
    "robot_v77_terminal_index": RUN / "lanes/robot_v77_terminal_index_r22/attempts/attempt_0003/RESULT.json",
    "robot_yield20": RUN / "lanes/robot_yield_20_r22/attempts/attempt_0003/RESULT.json",
    "robot_v78_target_reach_preflight": RUN / "lanes/robot_v78_target_reach_canary_r22/attempts/attempt_0001/RESULT.json",
    "mask_clean_prerequisites": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/attempts/attempt_0001_cpu_prereq/RESULT.json",
    "mask_chips010": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/chips010_role_sam31_r22/sessions/get_potato_chips_0901_010/attempts/attempt_0002/RESULT.json",
    "mask_poker015": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/poker015_object_sam31_r22/attempts/attempt_0002/RESULT.json",
    "mask_clean_lane_b_terminal": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/final_lane_b_terminal_v1/RESULT.json",
    "clean20_cpu_contract_closure": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_clean20_cpu_contract_v1/attempts/attempt_0001/RESULT.json",
    "poker245_min_atlas_canary": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_poker245_atlas_min_canary_v1/attempts/attempt_0001/RESULT.json",
    "mask_next_cpu_freeze": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/RESULT.json",
    "mask_prompt_bootstrap": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/attempts/attempt_0002/RESULT.json",
    "mask_temporal_identity_blocker": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_identity_canary_v1/attempts/attempt_0001/RESULT.json",
    "mask_temporal_adapter": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_adapter_v1/attempts/attempt_0001/RESULT.json",
    "mask_temporal_integration_blocker": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_identity_canary_v1/attempts/attempt_0002/RESULT.json",
    "mask_temporal_integration": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_integration_v1/attempts/attempt_0001/RESULT.json",
    "mask_temporal_scope_correction": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_integration_v1/attempts/attempt_0002/RESULT.json",
    "mask_temporal_real_cli": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_real_canary_cli_v1/attempts/attempt_0001/RESULT.json",
    "mask_temporal_real_canary": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_identity_canary_v1/attempts/attempt_0004/RESULT.json",
    "mask_temporal_failure_analysis": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_failure_analysis_v1/attempts/attempt_0001/RESULT.json",
    "mask_predictor_internal_instrumentation": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_predictor_internal_instrumentation_v1/attempts/attempt_0001/RESULT.json",
    "contact_readiness_matrix": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_contact_readiness_matrix_v1/attempts/attempt_0002/RESULT.json",
    "contact_poker031": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_contact10_poker031_v1/attempts/attempt_0001/RESULT.json",
    "robot_v78_prerequisite_closure": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_robot_v78_prerequisite_closure_v1/attempts/attempt_0001/RESULT.json",
    "robot_v78_causal_metric_extractor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_robot_v78_causal_metric_extractor_v1/attempts/attempt_0001/RESULT.json",
    "depth_rectification": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v1/depth10/play_cards_0910_001/attempt_0001/RESULT.json",
    "contact_poker245": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v1/contact10/play_cards_0903_245/attempt_0001/RESULT.json",
    "occlusion_chips023": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v1/occlusion_basic_object6d_gated/get_potato_chips_0902_023/attempt_0001/RESULT.json",
    "lane_cd_v2_summary": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v2/lane_summary/attempt_0001/RESULT.json",
    "occlusion_intersection_v2": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v2/intersection_matrix/attempt_0002/RESULT.json",
    "depth_selected_eye_audit_v2": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v2/depth_selected_eye_audit/play_cards_0910_001/attempt_0001/RESULT.json",
    "occlusion_poker227_v2": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v2/poker_basic_occlusion/play_cards_0903_227/attempt_0002/RESULT.json",
    "cleanup": RUN / "lanes/cleanup_current_only_r72_dryrun/CLEANUP_R22_COMBINED_RECEIPT.json",
    "cleanup_reference_audit": RUN / "lanes/cleanup_current_only_r72_dryrun/attempts/attempt_0006_reference_audit_corrected/RESULT.json",
    "visual_index": RUN / "lanes/doc_r22/attempts/attempt_0001/RESULT.json",
}

VISUALS = {
    "sensor_wrist_stereo_unavailable": ROOT / "docs/current/visuals/R22_手套_PlayCards0910_001_Controller_MANUS_HaWoR_Stereo不可用_绝对3D全片.mp4",
    "chips023_basic_occlusion": ROOT / "docs/current/visuals/R22_Chips023_Clean底图_Robot基础几何遮挡_全片.mp4",
    "chips010_sam31_role_mask_quality_c": ROOT / "docs/current/visuals/R22_Chips010_SAM31_RoleMask_全片质量C.mp4",
    "poker227_robot_v77_hard_geometry_pass": ROOT / "docs/current/visuals/R22_Poker227_Robot_v77_硬几何通过_全片.mp4",
    "poker227_basic_occlusion": ROOT / "archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/R22_Poker227_Clean底图_Robot基础几何遮挡_全片.mp4",
    "sensor_selected_eye_audit": ROOT / "archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/R22_PlayCards0910_001_SelectedEye与SBS证据审计.png",
}

OPTIONAL_EXPANSIONS = {
    "robot_v77_batch004",
    "robot_v77_batch004_final",
    "robot_v77_batch005",
    "robot_v77_terminal_index",
    "robot_yield20",
    "mask_predictor_internal_instrumentation",
    "robot_v78_prerequisite_closure",
    "robot_v78_causal_metric_extractor",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def status_of(data: dict) -> str:
    for key in ("status", "execution_status", "terminal_status", "outcome", "decision"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return "RECORDED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        raise RuntimeError(f"immutable attempt already exists: {out}")
    out.mkdir(parents=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    items = []
    for name, path in RESULTS.items():
        if not path.is_file():
            items.append({"name": name, "status": "PENDING", "expected_path": str(path.resolve())})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items.append({"name": name, "status": status_of(data), "result": ref(path)})

    governance = json.loads((ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text(encoding="utf-8"))
    package = {
        "schema_version": "chaoyang-r22-recovery-package-v1",
        "generated_at": generated_at,
        "plan_revision": "R2.2",
        "governance_revision": governance.get("governance_revision"),
        "authority_promoted": False,
        "items": items,
        "visuals": {
            name: ref(path) for name, path in VISUALS.items() if path.is_file()
        },
        "boundaries": [
            "Clean 58/58 denotes structural/provenance/decode closure only.",
            "Mask, Contact, Occlusion and Robot outputs remain development evidence.",
            "FoundationStereo was not run when rectification failed.",
            "No checkpoint training was started.",
            "No legacy runs or external datasets were permanently deleted.",
        ],
    }
    package_path = out / "RECOVERY_PACKAGE.json"
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# R2.2 四小时集成执行摘要",
        "",
        f"生成时间：`{generated_at}`",
        "",
        "> 本页由不可变 RESULT/receipt 自动汇总。PENDING 表示尚未发布终态；开发证据不等于正式 authority。",
        "",
        "| 子任务 | 状态 | 证据 |",
        "|---|---|---|",
    ]
    for item in items:
        evidence = item.get("result", {}).get("path", item.get("expected_path", ""))
        lines.append(f"| `{item['name']}` | `{item['status']}` | `{evidence}` |")
    lines += [
        "",
        "## 当前边界",
        "",
        "- Clean 58/58 仅代表结构、来源与解码闭合。",
        "- Mask、Contact、Occlusion、Robot 均未因本轮 canary 自动晋升 authority。",
        "- 新传感器 rectification 未过门时不运行 FoundationStereo。",
        "- 本轮不训练四支 Visual Aux checkpoint。",
        "- 清理只删除普通缓存；旧 run、历史文档、工具、测试、合同没有永久删除。",
    ]
    summary_path = out / "R22_EXECUTION_SUMMARY_ZH.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    required_pending = [
        item["name"] for item in items
        if item["status"] == "PENDING" and item["name"] not in OPTIONAL_EXPANSIONS
    ]
    metrics = {
        "schema_version": "chaoyang-r22-doc-metrics-v1",
        "generated_at": generated_at,
        "terminal_items": sum(item["status"] != "PENDING" for item in items),
        "pending_items": sum(item["status"] == "PENDING" for item in items),
        "required_pending_items": len(required_pending),
        "optional_expansion_pending_items": sum(
            item["status"] == "PENDING" and item["name"] in OPTIONAL_EXPANSIONS
            for item in items
        ),
        "total_items": len(items),
    }
    (out / "METRICS.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision = (
        "# 决定\n\n"
        "R2.2 在限定时间内以已有终态和明确阻塞封账；未终态 Robot/Mask 计算继续由当前 worker 完成，"
        "不因时间不足记为算法失败。所有扩展目标由 fresh Task Packet 启动。\n"
    )
    (out / "DECISION.md").write_text(decision, encoding="utf-8")
    next_action = {
        "schema_version": "chaoyang-r22-next-action-v1",
        "action": "WAIT_FOR_ACTIVE_WORKERS_THEN_REBUILD_PACKAGE_AND_CAS_TERMINALIZE",
        "pending": [item["name"] for item in items if item["status"] == "PENDING"],
        "forbidden": ["promote_development_authority", "start_checkpoint_training", "permanently_delete_legacy_assets"],
    }
    (out / "NEXT_ACTION.json").write_text(json.dumps(next_action, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact_paths = [package_path, summary_path, out / "METRICS.json", out / "DECISION.md", out / "NEXT_ACTION.json"]
    manifest = {"schema_version": "chaoyang-artifact-manifest-v1", "artifacts": [ref(path) for path in artifact_paths]}
    manifest_path = out / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "chaoyang-r22-doc-result-v2",
        "task_id": "doc_r22",
        "status": "PASSED" if metrics["required_pending_items"] == 0 else "RUNNING",
        "authority_promoted": False,
        "recovery_package": ref(package_path),
        "summary": ref(summary_path),
    }
    result_path = out / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_receipt = {
        "schema_version": "chaoyang-run-receipt-v1",
        "task_id": "doc_r22",
        "generated_at": generated_at,
        "result": ref(result_path),
        "artifact_manifest": ref(manifest_path),
    }
    (out / "RUN_RECEIPT.json").write_text(json.dumps(run_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
