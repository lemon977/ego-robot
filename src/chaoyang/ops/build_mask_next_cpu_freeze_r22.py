#!/usr/bin/env python3
"""Build the immutable CPU-only next Mask canary evidence package.

This tool performs no model inference and never writes governance current files.
It verifies the sealed Chips010/Poker015 results, derives bounded failure
attribution, and freezes the next 1C+2AB selection and independent evaluation
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001"

PATHS = {
    "selection": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_mask_failure_clusters_r3/attempts/attempt_0001/MASK_CHALLENGER_FROZEN_SELECTION_R3.json",
    "chips_wrapper": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/chips010_role_sam31_r22/sessions/get_potato_chips_0901_010/attempts/attempt_0002/RESULT.json",
    "chips_result": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/chips010_role_sam31_r22/sessions/get_potato_chips_0901_010/attempts/attempt_0002/role_mask_output/RESULT.json",
    "chips_tracker": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/chips010_role_sam31_r22/sessions/get_potato_chips_0901_010/attempts/attempt_0002/role_mask_output/TRACKER_REENTRY_MANIFEST.json",
    "chips_config": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_bounded_v2_1_configs/get_potato_chips_0901_010.json",
    "poker_result": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/gpu_mask_canaries/poker015_object_sam31_r22/attempts/attempt_0002/RESULT.json",
    "poker_old_manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker/play_cards_0901_015/OBJECT_MASK_MANIFEST.json",
    "poker_runner": ROOT / "src/chaoyang/ops/run_sam31_poker_object_canary_r22.py",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_selection(selection: dict[str, Any], stage: str) -> dict[str, Any]:
    rows = [x for x in selection["selections"] if x["stage"] == stage and x.get("selection_eligible")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one eligible {stage} selection, got {len(rows)}")
    return rows[0]


def verify_frozen_result(entry: dict[str, Any]) -> dict[str, Any]:
    declared = entry["result"]
    actual = ref(Path(declared["path"]))
    if actual["bytes"] != declared["bytes"] or actual["sha256"] != declared["sha256"]:
        raise RuntimeError(f"frozen result changed: {declared['path']}")
    payload = load(Path(declared["path"]))
    return {
        **entry,
        "result": actual,
        "verified_frame_count": payload.get("frame_count"),
        "selection_frame_count_metadata_stale": entry.get("frame_count") != payload.get("frame_count"),
    }


def chips_tracker_metrics(tracker: dict[str, Any]) -> dict[str, Any]:
    right = tracker["roles"]["right_tracker"]
    records = right["records"]
    expected = [r for r in records if r["geometry"].get("expected_visible")]
    refresh = [r for r in expected if r.get("refresh_attempted")]
    present = [r for r in refresh if r.get("proposal_gate", {}).get("checks", {}).get("present")]
    near = [r for r in refresh if r.get("proposal_gate", {}).get("checks", {}).get("centroid_near_current_wrist_anchor")]
    bounded = [r for r in refresh if r.get("proposal_gate", {}).get("checks", {}).get("bounded_area")]
    positive = [r for r in refresh if r.get("proposal_gate", {}).get("checks", {}).get("positive_coverage")]
    adjacent = [r for r in refresh if r.get("proposal_gate", {}).get("checks", {}).get("adjacent_to_current_human")]
    pass_except_adjacency = []
    for record in refresh:
        checks = record.get("proposal_gate", {}).get("checks", {})
        if all(checks.get(k) for k in ("present", "positive_coverage", "bounded_area", "centroid_near_current_wrist_anchor")) and not checks.get("adjacent_to_current_human"):
            pass_except_adjacency.append(record)
    return {
        "expected_visible_frames": len(expected),
        "refresh_attempt_frames": len(refresh),
        "proposal_present_frames": len(present),
        "centroid_near_anchor_frames": len(near),
        "bounded_area_frames": len(bounded),
        "positive_coverage_frames": len(positive),
        "adjacent_to_human_frames": len(adjacent),
        "would_pass_except_empty_human_adjacency_frames": len(pass_except_adjacency),
        "accepted_frames": right.get("refresh_accept_count", 0),
        "empty_fail_closed_frames": right.get("empty_fail_closed_count", 0),
    }


def poker_observed_ranges(manifest: dict[str, Any]) -> list[list[int]]:
    observed = []
    for frame in manifest["frames"]:
        inst = frame.get("physical_instances", {}).get("0", {})
        if inst.get("observed") and inst.get("valid"):
            observed.append(int(frame["source_frame"]))
    ranges: list[list[int]] = []
    for value in observed:
        if not ranges or value != ranges[-1][1] + 1:
            ranges.append([value, value])
        else:
            ranges[-1][1] = value
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    for path in PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = load(PATHS["selection"])
    chips_wrapper = load(PATHS["chips_wrapper"])
    chips_result = load(PATHS["chips_result"])
    chips_tracker = load(PATHS["chips_tracker"])
    poker_result = load(PATHS["poker_result"])
    poker_old_manifest = load(PATHS["poker_old_manifest"])

    if selection.get("current_baseline") != "SAM3.1":
        raise RuntimeError("frozen selection does not identify SAM3.1 as current baseline")
    if chips_wrapper.get("status") != "FAILED_QUALITY_C":
        raise RuntimeError("unexpected sealed Chips010 status")
    if poker_result.get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("unexpected sealed Poker015 status")

    chips_sel = find_selection(selection, "ROLE_MASK")
    poker_sel = find_selection(selection, "OBJECT_MASK")
    chips_tracker_summary = chips_tracker_metrics(chips_tracker)
    poker_ranges = poker_observed_ranges(poker_old_manifest)

    frozen_selection = {
        "schema_version": "mask-next-frozen-selection-r22-v1",
        "created_at": now(),
        "status": "FROZEN_1C_2AB_PER_TASK",
        "current_baseline": "SAM3.1",
        "authority_promoted": False,
        "tasks": {
            "chips_role": {
                "failed_canary": {"latest_sealed_result": ref(PATHS["chips_wrapper"]), "baseline_terminal": verify_frozen_result(chips_sel["canary"])},
                "regressions": [verify_frozen_result(x) for x in chips_sel["regressions"]],
                "execution_order": "failed_canary_then_regressions_only_if_canary_passes",
            },
            "poker_object": {
                "failed_canary": {"latest_sealed_result": ref(PATHS["poker_result"]), "baseline_terminal": verify_frozen_result(poker_sel["canary"])},
                "regressions": [verify_frozen_result(x) for x in poker_sel["regressions"]],
                "execution_order": "failed_canary_then_regressions_only_if_canary_passes",
            },
        },
        "claim_limit": "Frozen development canary/regression selection only; no new Mask inference or authority.",
    }

    evaluation = {
        "schema_version": "mask-next-evaluation-reference-r22-v1",
        "created_at": now(),
        "status": "FROZEN_INDEPENDENT_OF_NEXT_CANDIDATE_OUTPUT",
        "pixel_gold_available": False,
        "accuracy_authorized": False,
        "candidate_may_modify_reference": False,
        "candidate_output_may_generate_reference": False,
        "current_masks_are_diagnostic_receipts_not_gold": True,
        "tasks": {
            "chips_role": {
                "diagnostic_frames": [43, 266, 713],
                "frame_rationale": {
                    "43": "early right-tracker expected-visible interval",
                    "266": "pre-existing human anchor/bootstrap frame",
                    "713": "late causal continuation diagnostic",
                },
                "candidate_source": "pre-existing HaWoR/config anatomical-side points plus independently frozen side-specific boxes; never candidate output",
                "reference_sources": [ref(PATHS["selection"]), ref(PATHS["chips_wrapper"]), ref(PATHS["chips_result"]), ref(PATHS["chips_tracker"]), ref(PATHS["chips_config"])],
                "development_metrics": ["human_seed_nonempty_per_side", "positive_point_coverage", "tracker_gate_conditioned_on_valid_human_seed", "offscreen_empty", "left_right_identity_continuity"],
            },
            "poker_object": {
                "diagnostic_frames": [0, 2, 3, 79, 80, 213, 214, 220, 221, 223],
                "frame_rationale": "seed, first loss, first re-entry, second loss and second re-entry boundaries from sealed old manifest",
                "observed_ranges_in_diagnostic_receipt": poker_ranges,
                "candidate_source": "independently frozen action-contract rightmost-card point+box; must be recorded before inference and must not be derived from evaluation masks",
                "reference_sources": [ref(PATHS["selection"]), ref(PATHS["poker_result"]), ref(PATHS["poker_old_manifest"])],
                "development_metrics": ["frame0_same_instance_seed_present", "all_prompt_candidate_rows_persisted", "causal_forward_identity", "reentry_latency", "no_instance_union"],
            },
        },
        "claim_limit": "Independent development reference and frame freeze. It is not pixel Gold and cannot support accuracy claims.",
    }

    attribution = {
        "schema_version": "mask-failure-attribution-r22-v1",
        "created_at": now(),
        "status": "SUPPORTED_DEVELOPMENT_DIAGNOSIS",
        "model_rerun": False,
        "current_baseline": "SAM3.1",
        "tasks": {
            "chips_role": {
                "session": "get_potato_chips_0901_010",
                "sealed_terminal_status": "FAILED_QUALITY_C",
                "primary_failure_axis": "PROMPT_BOOTSTRAP",
                "secondary_failure_axis": "GATE_DEPENDENCY_ON_EMPTY_HUMAN_MASK",
                "projection_axis": "NOT_PRIMARY_RETAIN_UNCHANGED_FOR_FIRST_NEXT_CANARY",
                "temporal_identity_axis": "NOT_REACHED_NO_ACCEPTED_SEED",
                "evidence": {
                    "human_left_pass_fraction": chips_result.get("metrics", {}).get("human_pass_fractions", {}).get("left_human"),
                    "human_right_pass_fraction": chips_result.get("metrics", {}).get("human_pass_fractions", {}).get("right_human"),
                    "right_tracker": chips_tracker_summary,
                    "reasoning": "Tracker projection produced nonempty proposals near the current wrist anchor, but the human seed masks were empty. 660 proposals satisfy all recorded checks except adjacency to the empty human mask; temporal identity cannot be tested without an accepted seed.",
                },
                "next_fix_order": ["SIDE_SPECIFIC_POINT_BOX_PROMPT_BOOTSTRAP", "CONDITIONALIZE_TRACKER_ADJACENCY_ON_VALID_HUMAN_SEED", "THEN_CAUSAL_TEMPORAL_IDENTITY"],
            },
            "poker_object": {
                "session": "play_cards_0901_015",
                "sealed_terminal_status": "FAILED_RUNTIME_FINAL",
                "semantic_failure_diagnosis": "SEED_PROMPT_SELECTION_NO_ELIGIBLE_FRAME0_CANDIDATE",
                "primary_failure_axis": "PROMPT_SELECTION",
                "projection_axis": "NOT_APPLICABLE_NO_GEOMETRIC_PROJECTOR_IN_THIS_CANARY",
                "temporal_identity_axis": "NOT_REACHED_NO_FRAME0_SEED",
                "evidence": {
                    "runtime_error": poker_result.get("error"),
                    "prompt_candidate_rows_persisted": False,
                    "old_diagnostic_observed_ranges": poker_ranges,
                    "reasoning": "The text-only frame-0 sweep terminated before propagation. Re-entry quality therefore cannot be attributed to temporal identity in this attempt.",
                },
                "next_fix_order": ["PERSIST_ALL_PROMPT_CANDIDATE_ROWS_ON_FAILURE", "INDEPENDENT_ACTION_CONTRACT_POINT_BOX_PROMPT", "THEN_CAUSAL_FORWARD_REENTRY"],
            },
        },
        "claim_limit": "Read-only attribution from sealed receipts. It does not reclassify terminals or measure Mask accuracy.",
    }

    prompt_contract = {
        "schema_version": "sam31-current-prompt-contract-r22-v1",
        "created_at": now(),
        "status": "CURRENT_BASELINE_BOUNDED_NEXT_CANARY_CONTRACT",
        "current_baseline": "SAM3.1",
        "checkpoint": poker_result["checkpoint"],
        "sam2x_role": "CHALLENGER_ONLY_NOT_CURRENT_BASELINE",
        "authority_change": False,
        "chips_role": {
            "current": {"human_text": "a person's hand and forearm", "tracker": "HaWoR/config point prompts plus tracker ROI gates"},
            "bounded_next_change": ["retain current projection for first diagnostic", "use anatomical-side point+box human bootstrap", "persist seed candidate rows", "emit BLOCKED_HUMAN_SEED instead of misclassifying tracker when human seed is empty"],
            "must_not_change_yet": ["projection transform", "temporal identity policy", "authority baseline"],
        },
        "poker_object": {
            "current_text_prompts": ["a playing card", "the playing card held by the person", "a purple-backed playing card"],
            "bounded_next_change": ["persist every text/point/box candidate row on failure", "freeze independent action-contract rightmost-card point+box before inference", "accept one same-instance seed before causal propagation"],
            "must_not_change_yet": ["causal-only temporal direction", "identity contract", "authority baseline"],
        },
        "next_inference_prerequisite": "Independent prompt annotations/boxes and evaluation reference must be frozen before GPU inference.",
        "claim_limit": "SAM3.1 development prompt contract only; no Mask authority or accuracy.",
    }

    data_files = {
        "MASK_FAILURE_ATTRIBUTION.json": attribution,
        "MASK_NEXT_FROZEN_SELECTION.json": frozen_selection,
        "MASK_NEXT_EVALUATION_REFERENCE.json": evaluation,
        "SAM31_CURRENT_PROMPT_CONTRACT.json": prompt_contract,
    }
    for name, payload in data_files.items():
        dump(out / name, payload)

    metrics = {
        "schema_version": "mask-next-cpu-freeze-metrics-r22-v1",
        "created_at": now(),
        "status": "PASSED",
        "model_runs": 0,
        "gpu_used": False,
        "frozen_failed_canaries": 2,
        "frozen_ab_regressions": 4,
        "chips": chips_tracker_summary,
        "poker": {"old_diagnostic_observed_ranges": poker_ranges, "frame0_prompt_candidate_found_latest": False},
        "decision": {"chips": "PROMPT_BOOTSTRAP_FIRST", "poker": "PROMPT_SELECTION_FIRST", "projection_first": False, "temporal_identity_first": False},
    }
    dump(out / "METRICS.json", metrics)

    next_action = {
        "schema_version": "mask-next-action-r22-v1",
        "created_at": now(),
        "status": "READY_AFTER_PROMPT_ANNOTATION_FREEZE",
        "task_id": "mask_sam31_prompt_bootstrap_canary_r22",
        "ordered_steps": [
            "Freeze independent Chips side-specific human point+box prompts and Poker action-contract rightmost-card point+box prompts before inference.",
            "Run Chips010 and Poker015 only with SAM3.1; persist every prompt candidate row, including failure paths.",
            "For Chips, evaluate tracker adjacency only after a valid same-side human seed exists.",
            "For Poker, start causal temporal/re-entry evaluation only after frame-0 same-instance seed exists.",
            "Run the two A/B regressions per task only if its failed canary passes; otherwise terminal quality C with no automatic retry.",
        ],
        "stop_conditions": ["PASSED_1C_2AB", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "gpu_required": True,
        "authority_promoted": False,
    }
    dump(out / "NEXT_ACTION.json", next_action)

    decision_md = """# Mask 下一步有界 canary 决定

状态：`PASSED`（本次仅 CPU 只读归因与冻结；没有运行模型、没有晋升 authority）。

## 结论

- Chips010 Role Mask 的首要问题是 **人手提示/seed bootstrap 失败**，不是当前投影首要失败。右 tracker 在 714 个预期可见帧都有非空 proposal，其中 660 帧满足记录中的 present、positive coverage、bounded area、near-anchor，唯独无法与空的人体 mask 建立 adjacency。没有有效 seed，因此还不能评价时序身份。
- Poker015 Object Mask 的密封终态仍是 `FAILED_RUNTIME_FINAL`。实际语义诊断是 frame 0 文本提示没有产生合格牌候选；传播根本未启动，因此不能把失败归因给时序重入，且没有几何投影可修。
- 两条下一轮都应先修 **提示选择与失败证据持久化**。投影不作为第一修改项；时序身份只在有效 seed 后测试。
- SAM3.1 仍是当前正式基线；SAM2.x 不在本次合同中晋升。

## 冻结实验

每任务固定 `1条失败 canary + 2条现有 A/B regression`。候选输出与 evaluation reference 严格分离；现有 mask 只作开发诊断，不是 pixel Gold，因此不得报告 accuracy。

## 下一步

先独立冻结 Chips 左右人体 point+box 与 Poker 动作合同右侧目标牌 point+box，再各跑一次 SAM3.1 失败 canary。只有 canary 通过才运行两条回归；失败即有限封账，不继续调投影或时序策略。
"""
    (out / "DECISION.md").write_text(decision_md, encoding="utf-8")

    manifest_items = [ref(out / name) for name in [*data_files, "METRICS.json", "NEXT_ACTION.json", "DECISION.md"]]
    manifest = {
        "schema_version": "mask-next-cpu-freeze-artifact-manifest-r22-v1",
        "created_at": now(),
        "status": "PASS",
        "artifacts": manifest_items,
        "authority_promoted": False,
    }
    dump(out / "ARTIFACT_MANIFEST.json", manifest)

    run_receipt = {
        "schema_version": "mask-next-cpu-freeze-run-receipt-r22-v1",
        "created_at": now(),
        "task_id": "mask_next_cpu_freeze_r22",
        "attempt_id": "attempt_0001",
        "status": "PASSED",
        "execution": "CPU_READ_ONLY_EVIDENCE_AUDIT",
        "model_inference": False,
        "gpu_used": False,
        "current_governance_modified": False,
        "inputs": [ref(path) for path in PATHS.values()],
        "code": ref(Path(__file__)),
        "claim_limit": "Evidence attribution, selection and prompt/evaluation contract only.",
    }
    dump(out / "RUN_RECEIPT.json", run_receipt)

    result = {
        "schema_version": "mask-next-cpu-freeze-result-r22-v1",
        "created_at": now(),
        "task_id": "mask_next_cpu_freeze_r22",
        "attempt_id": "attempt_0001",
        "status": "PASSED",
        "artifact_revision": "R7_4_MASK_NEXT_CPU_FREEZE_1",
        "current_baseline": "SAM3.1",
        "summary": "Both failures are prompt/seed-first. Projection is not the first Chips fix; temporal identity was not reached in either canary.",
        "metrics": ref(out / "METRICS.json"),
        "decision": ref(out / "DECISION.md"),
        "next_action": ref(out / "NEXT_ACTION.json"),
        "artifact_manifest": ref(out / "ARTIFACT_MANIFEST.json"),
        "authority_promoted": False,
        "claim_limit": "Read-only development diagnosis and immutable next-canary freeze; no Mask accuracy or authority.",
    }
    dump(out / "RESULT.json", result)
    print(json.dumps({"status": "PASSED", "output": str(out), "chips": chips_tracker_summary, "poker_ranges": poker_ranges}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
