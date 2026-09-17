#!/usr/bin/env python3
"""Compile immutable, evidence-bound RC1 research handoff without promotion."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chaoyang.governance.common import artifact_ref, atomic_json, atomic_write, now_iso

BASE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2"
OLD = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "research_scope": BASE / "RESEARCH_EXECUTION_SCOPE.json",
        "governance_receipt": ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "sam_legacy": OLD / "sam31_state_route/attempt_0001/RESULT.json",
        "sam_native_gpu_wait": OLD / "sam31_semantic_full_route/GPU_WRAPPER_ATTEMPT_0003.json",
        "sam_visual_comparison": BASE / "sam31_route_comparison/attempt_0001/RESULT.json",
        "sam_batchflag_code_unrun": ROOT / "src/chaoyang/ops/run_sam31_semantic_batchflag_ablation.py",
        "sam_native_code": ROOT / "src/chaoyang/ops/run_sam31_semantic_full_route_diagnostic.py",
        "sam_old_route_code": ROOT / "src/chaoyang/ops/run_sam31_state_route_diagnostic.py",
        "sam_checkpoint": ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
        "c2w_audit_code": ROOT / "src/chaoyang/ops/audit_rc1_pico_c2w_causality.py",
        "source_capacity_audit_code": ROOT / "src/chaoyang/ops/audit_rc1_original_source_capacity.py",
        "pico_blocked_video_code": ROOT / "src/chaoyang/ops/render_rc1_pico_c2w_blocked_diagnostic.py",
        "sam_comparison_video_code": ROOT / "src/chaoyang/ops/build_sam31_route_comparison_video.py",
        "chips_c2w": BASE / "c2w_index_correction/chips103/attempt_0003/RESULT.json",
        "poker_c2w": BASE / "c2w_index_correction/poker227/attempt_0003/RESULT.json",
        "chips_blocked_video": BASE / "pico_c2w_blocked_visual/chips103/attempt_0001/RESULT.json",
        "poker_blocked_video": BASE / "pico_c2w_blocked_visual/poker227/attempt_0001/RESULT.json",
        "source_capacity": BASE / "original_source_capacity/attempt_0002/RESULT.json",
    }
    refs = {name: artifact_ref(path) for name, path in paths.items()}
    scope = read(paths["research_scope"])
    receipt = read(paths["governance_receipt"])
    if (receipt["generation_id"], receipt["governance_revision"]) != (
        scope["governance_start"]["generation_id"], scope["governance_start"]["revision"]
    ):
        raise RuntimeError("current governance changed; freeze a fresh handoff")
    old = read(paths["sam_legacy"])
    sam_wait = read(paths["sam_native_gpu_wait"])
    sam_video = read(paths["sam_visual_comparison"])
    chips = read(paths["chips_c2w"])
    poker = read(paths["poker_c2w"])
    cap = read(paths["source_capacity"])
    if sam_wait["status"] != "BLOCKED_RESOURCE" or sam_video["status"] != "PASSED_VISUAL_DIAGNOSTIC_ONLY":
        raise RuntimeError("SAM diagnostic status mismatch")
    if chips["robot_causal_eligible"] or poker["robot_causal_eligible"]:
        raise RuntimeError("c2w producer has unexpectedly been promoted")
    if any(spec["capacity_16_train_3_validation_proven"] for spec in cap["summary"].values()):
        raise RuntimeError("source capacity handoff would contradict frozen pair block")
    legacy_rows = old["matrix"]["repeated_seed_8"]["适配器_无Priming"]["seed_rows"]
    detector_area = next(row["area_pixels"] for row in legacy_rows if row["stage"] == "FROZEN_DETECTOR_REPLAY" and row["eligible"])
    tracking_area = next(row["area_pixels"] for row in legacy_rows if row["stage"] == "STABLE_POINT_BOX_TRACK_SEED" and row["eligible"])
    result = {
        "schema_version": "chaoyang-rc1-unblock-handoff-v2",
        "created_at": now_iso(),
        "status": "PASSED_HANDOFF_WITH_RESEARCH_BLOCKERS",
        "governance_revision_frozen": receipt["governance_revision"],
        "authority_promoted": False,
        "training_eligibility_changed": False,
        "checkpoint_pairs_remain": {"chips": "BLOCKED_DATA_VOLUME", "poker": "BLOCKED_DATA_VOLUME"},
        "task_outcomes": {
            "A_SAM31_ROUTE": {
                "status": "BLOCKED_RESOURCE",
                "gpu_wait_seconds": sam_wait["wait_seconds"],
                "old_detector_seed_area_px": detector_area,
                "old_tracking_seed_area_px": tracking_area,
                "old_tracking_seed_ratio": tracking_area / detector_area,
                "native_visual_only": True,
                "one_factor_ablation_executed": False,
                "first_observable_divergence": "DETECTOR_TO_STABLE_TRACKING_SEED_AREA_COLLAPSE",
                "root_cause": "UNKNOWN",
            },
            "B_PICO_C2W_PRODUCER": {
                "status": "UNKNOWN_VERIFICATION_REQUIRED",
                "chips_indexed_timestamp_equal_frames": chips["metadata_timestamp_equals_indexed_tracking_frames"],
                "poker_indexed_timestamp_equal_frames": poker["metadata_timestamp_equals_indexed_tracking_frames"],
                "chips_tracker_state": chips["mapped_tracker_state_counts"],
                "poker_tracker_state": poker["mapped_tracker_state_counts"],
                "sync_error_observed_semantics": chips["stored_sync_error_observed_semantics"],
                "original_rgb_exposure_timestamp_proven": False,
                "old_producer_suffix_invariance_proven": False,
                "causal_robot_generated": False,
            },
            "C_ORIGINAL_SOURCE_CAPACITY": {
                "status": "PASSED_DIAGNOSTIC_WITH_PROVENANCE_GAP",
                "chips": cap["summary"]["chips"],
                "poker": cap["summary"]["poker"],
                "original_rgb_per_frame_lineage_proven": False,
            },
        },
        "facts": [
            "SAM old stable tracking seed area shrank from detector area before frame-1 loss.",
            "PICO tracking_index addresses pose rows after one physical merge-header line.",
            "PICO metadata.ts equals indexed tracker timestamp on all frozen Chips103/Poker227 frames.",
            "PICO tracking_sync_error_ms matches tracker timestamp minus nominal FPS grid, not independently proven camera exposure skew.",
            "Poker227 mapped TrackerState is notAccurate on every frame despite Head.status=3.",
            "Frozen T0 candidate connected source clusters are insufficient for 16 train plus 3 validation in both tasks.",
        ],
        "hypotheses": [
            "The old independent point/box tracking route may explain SAM seed collapse; no single-factor GPU result yet.",
            "Original acquisition lineages may be finer than T0 merged-run groups, but RGB frame-to-original-session identity is unproven.",
        ],
        "fixed": [
            "C2w timestamp audit excludes the trackingData merge-header line and supersedes its earlier mis-indexed diagnostic conclusion.",
            "Research video comparison and blocked-stage visualizations have immutable results and SHA receipts.",
        ],
        "unverified": [
            "SAM specific internal root cause and full-sequence identity/segmentation quality.",
            "Original RGB capture timestamp and old c2w producer suffix-invariance.",
            "Independent original RGB acquisition groups and quality-qualified RC1 source capacity.",
            "Causal HaWoR, causal Robot, paired smoke and four formal checkpoints.",
        ],
        "next_actions": [
            "When central GPU lease has >=16 GiB safe free memory, run frozen semantic batch-final one-factor ablation once; no forced takeover.",
            "Locate original PICO editor producer and RGB per-frame exposure/source mapping; otherwise keep c2w/Robot causal eligibility blocked.",
            "Do not revise T0 or checkpoint pair statuses without original RGB acquisition proof and governance CAS.",
        ],
        "reproduction_commands": {
            "A_batchflag_not_yet_run": "python -m tools.run_sam31_semantic_batchflag_ablation --output-root <fresh_attempt_dir>  # only under central GPU lease after resource gate",
            "B_chips_index_corrected": "python -m tools.audit_rc1_pico_c2w_causality --raw-root /mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/potato_chips/get_potato_chips_0902_103 --hawor-npz archive/baseline-20260917-0aa69e9/content/history/data/processed/chips/get_potato_chips_0902_103/hawor/20260908_hawor_world_consistent_prod_v2/HAWOR_WORLD_CONSISTENT_MANO21.npz --output-root <fresh_attempt_dir>",
            "B_poker_index_corrected": "python -m tools.audit_rc1_pico_c2w_causality --raw-root /mnt/data/egodata/datasets/ego/chips_cards_tracker_0903/playing_cards/play_cards_0903_227 --hawor-npz archive/baseline-20260917-0aa69e9/content/history/data/processed/poker/play_cards_0903_227/hawor/20260908_hawor_world_consistent_prod_v2/HAWOR_WORLD_CONSISTENT_MANO21.npz --output-root <fresh_attempt_dir>",
            "C_source_capacity": "python -m tools.audit_rc1_original_source_capacity --output-root <fresh_attempt_dir>",
        },
        "claim_limit": "Bounded research handoff only. No external accuracy, Mask authority, Clean recovery accuracy, causal Robot output, physical deployment or policy claim.",
        "evidence": refs,
    }
    atomic_json(output / "RESULT.json", result)
    markdown = [
        "# RC1 解阻研究 V2：事实、限制与下一步", "",
        f"生成时间：{result['created_at']}；冻结治理 revision：{receipt['governance_revision']}。本页是研究交接，不是 current authority。", "",
        "## 已确认事实", "",
    ]
    markdown.extend(f"- {item}" for item in result["facts"])
    for title, key in (("当前假设", "hypotheses"), ("已修正", "fixed"), ("尚未证明", "unverified"), ("下一步", "next_actions")):
        markdown += ["", f"## {title}", ""]
        markdown.extend(f"- {item}" for item in result[key])
    markdown += ["", "## 任务状态与证据", ""]
    for task, outcome in result["task_outcomes"].items():
        markdown.append(f"- {task}: `{outcome['status']}`。")
    markdown += ["", "## 复现命令", ""]
    markdown.extend(f"- `{name}`：`{command}`" for name, command in result["reproduction_commands"].items())
    markdown += ["", "机器结果：`RESULT.json`；该文件逐项绑定输入、代码、SAM3.1 权重和视频的 bytes/SHA。", ""]
    atomic_write(output / "DECISION.md", ("\n".join(markdown)).encode("utf-8"))
    atomic_json(output / "RUN_RECEIPT.json", {
        "status": result["status"], "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
        "result": artifact_ref(output / "RESULT.json"),
        "decision": artifact_ref(output / "DECISION.md"),
    })
    print(json.dumps({"status": result["status"], "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
