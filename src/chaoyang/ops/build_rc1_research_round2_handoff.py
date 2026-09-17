#!/usr/bin/env python3
"""Small evidence-bound handoff for the next RC1 research worker."""

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

from chaoyang.governance.common import artifact_ref, atomic_json, atomic_write, now_iso, validate_artifact_ref

BASE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2"
OLD = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "sam_native": OLD / "sam31_semantic_full_route/attempt_0004/RESULT.json",
        "sam_batchflag": BASE / "sam31_batchflag_ablation/attempts/attempt_0002/RESULT.json",
        "sam_seed_stage": BASE / "sam31_seed_stage_probe/attempts/attempt_0002/RESULT.json",
        "rgb_producer": BASE / "rgb_producer_evidence/attempts/attempt_0001/RESULT.json",
        "expandable_candidates": BASE / "expandable_candidates/attempts/attempt_0001/RESULT.json",
    }
    refs = {key: artifact_ref(path) for key, path in paths.items()}
    current_receipt_path = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    current_min_path = ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json"
    receipt, current = load(current_receipt_path), load(current_min_path)
    min_binding = receipt.get("files", {}).get("rc1_status_min")
    if not isinstance(min_binding, dict) or validate_artifact_ref(min_binding):
        raise RuntimeError("current RC1 status min is not bound by the receipt")
    if current["governance_revision"] != receipt["governance_revision"] or current["generation_id"] != receipt["generation_id"]:
        raise RuntimeError("current RC1 projection is not bound to receipt")
    if current["release_flags"]["RC1_RELEASE_STATUS"] != "INCOMPLETE" or current["pair_status"] != {"chips": "BLOCKED_DATA_VOLUME", "poker": "BLOCKED_DATA_VOLUME"}:
        raise RuntimeError("current RC1 train/release boundary changed; re-audit first")
    native, batch, seed = (load(paths[key]) for key in ("sam_native", "sam_batchflag", "sam_seed_stage"))
    if native["status"] != "PASSED_DIAGNOSTIC" or seed["session_id"] != "play_cards_0901_015":
        raise RuntimeError("SAM frozen full-session diagnostic mismatch")
    if {key: value["present_frames"] for key, value in native["matrix"].items()} != {"repeated_seed_8": 8, "known_translation_8": 8, "real_prefix_16": 16}:
        raise RuntimeError("SAM native route result does not match frozen cases")
    if any(not value["comparison"]["same_pixel_masks"] for value in batch["matrix"].values()):
        raise RuntimeError("batch flag ablation is not pixel-identical")
    stages = {item["stage"]: item for item in seed["stages"]}
    if (stages["SEMANTIC_TEXT_BOX"]["selected_area_pixels"], stages["TRACKING_BOX_THEN_POINT"]["selected_area_pixels"]) != (9647, 1036):
        raise RuntimeError("seed-stage diagnostic mismatch")
    rgb_result, expansion = load(paths["rgb_producer"]), load(paths["expandable_candidates"])
    if rgb_result["status"] != "UNKNOWN_VERIFICATION_REQUIRED" or sorted(rgb_result["session_ids"]) != ["get_potato_chips_0902_103", "play_cards_0903_227"]:
        raise RuntimeError("RGB producer proof boundary changed")
    if expansion["status"] != "PASSED_RESEARCH_AUDIT" or sum(v["excluded_sessions"] for v in expansion["summary"].values()) != 111:
        raise RuntimeError("expandable candidate denominator mismatch")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Current files are mutable CAS projections. Bind byte-preserving snapshots
    # so this immutable research handoff survives later governance revisions.
    receipt_snapshot = output / "CURRENT_STATUS_RECEIPT_SNAPSHOT.json"
    min_snapshot = output / "CURRENT_RC1_STATUS_MIN_SNAPSHOT.json"
    atomic_write(receipt_snapshot, current_receipt_path.read_bytes())
    atomic_write(min_snapshot, current_min_path.read_bytes())
    if load(receipt_snapshot) != receipt or load(min_snapshot) != current:
        raise RuntimeError("current governance changed while freezing handoff snapshots")
    refs["governance_receipt_snapshot"] = artifact_ref(receipt_snapshot)
    refs["rc1_status_min_snapshot"] = artifact_ref(min_snapshot)
    created = now_iso()
    result = {
        "schema_version": "chaoyang-rc1-research-round2-handoff-v1",
        "created_at": created,
        "status": "PASSED_RESEARCH_HANDOFF_WITH_UNRESOLVED_PREREQUISITES",
        "governance_revision_observed": receipt["governance_revision"],
        "authority_promoted": False,
        "training_eligibility_changed": False,
        "checkpoint_pairs": current["pair_status"],
        "release_status": current["release_flags"]["RC1_RELEASE_STATUS"],
        "facts": [
            "play_cards_0901_015: native SAM3.1 semantic-full emitted the frozen ID on repeated 8/8, translated 8/8 and real 16/16 frames.",
            "play_cards_0901_015: is_last_batch true/false gave identical per-frame binary Mask SHA on repeated and real fixtures.",
            "play_cards_0901_015: semantic text+box seed=9647 px/full route; old independent tracking box-only emitted no object, point+box seed=1036 px/partial route.",
            "get_potato_chips_0902_103 has 284 frames; play_cards_0903_227 has 196. Their inspected clip assets do not provide raw per-frame RGB exposure time, frame-source map or replayable old exporter.",
            "The frozen shortlist has Chips 21/Poker 24 sessions and sourceSession-connected clusters Chips 12/Poker 9; 111 nonshortlist sessions have categorized first blockers.",
        ],
        "unverified": [
            "SAM same physical card identity and pixel segmentation quality beyond the short fixture; no official Mask successor.",
            "PICO old exporter suffix-invariance and RGB exposure-to-tracker sync; no causal Robot training eligibility.",
            "Original RGB independent acquisition groups, group-safe validation split and quality-qualified paired H50 windows.",
        ],
        "next_decisions": [
            "SAM: evaluate one minimally changed semantic-full challenger on frozen same-instance and boundary reference; stop if identity or regression fails.",
            "PICO: request or locate exact old pico_dataset_editor-1.1.0 exporter, raw RGB capture timestamp table and source-frame map; absent those, keep causal c2w blocked.",
            "Capacity: audit source evidence and group-safe splits first; prioritize calibration-only Visual Tier and metric-ready exclusions, then bounded Poker object-identity successor. Do not move a development group into validation.",
        ],
        "claim_limit": "Research handoff only. No Mask accuracy, external synchronization, causal Robot pass, dataset minimum, checkpoint or physical authority claim.",
        "evidence": refs,
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    lines = [
        "# RC1 解阻研究：下一 AI 的最小交接", "",
        f"生成时间：{created}。点时治理 revision：{receipt['governance_revision']}；使用前重新验证 current receipt。", "",
        "性质：研究收据，不晋升 Mask/Robot/训练 authority。", "", "## 新增事实", "",
        *[f"- {item}" for item in result["facts"]], "", "## 尚未证明", "",
        *[f"- {item}" for item in result["unverified"]], "", "## 下一决策", "",
        *[f"- {item}" for item in result["next_decisions"]], "",
        "详细路径与 bytes/SHA：`RESULT.json` 的 `evidence`。", "",
    ]
    atomic_write(output / "DECISION.md", "\n".join(lines).encode("utf-8"))
    atomic_json(output / "RUN_RECEIPT.json", {
        "status": result["status"], "created_at": created,
        "result": artifact_ref(output / "RESULT.json"),
        "decision": artifact_ref(output / "DECISION.md"),
        "code": artifact_ref(Path(__file__)),
    })
    print(json.dumps({"status": result["status"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
