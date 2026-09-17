"""Audit the metric-ready sessions outside RC1's frozen candidate shortlist.

This is a read-only research join. It does not update governance or infer training
eligibility from an offline Robot candidate or a sourceSession marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPAND = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/expandable_candidates/attempts/attempt_0001/EXPANDABLE_CANDIDATES.json"
MASTER = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/t0_freeze_capacity_split/attempts/attempt_0001/MASTER_LEDGER.json"
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/original_source_capacity/attempt_0002/RESULT.json"
CONVERSION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
ROBOT30 = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/robot30_causal_budget_closure/attempts/attempt_0001/ROBOT30_FINAL_TERMINAL_INDEX.json"
HARD_SOFT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_v76_r3/ROBOT_HARD_SOFT_CANDIDATE_INDEX.json"
OLD_ROBOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_terminals_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_ref(item: dict) -> None:
    path = Path(item["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Missing absolute evidence path: {path}")
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"Evidence bytes/SHA mismatch: {path}")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f"No-clobber: output exists: {out}")

    inputs = {name: ref(path) for name, path in {
        "expandable": EXPAND, "master": MASTER, "original_source": SOURCE,
        "conversion": CONVERSION, "robot30": ROBOT30, "hard_soft": HARD_SOFT,
        "generator": Path(__file__),
    }.items()}
    expansion = load(EXPAND)
    for name, embedded in expansion["inputs"].items():
        verify_ref(embedded)
        if name == "master" and embedded != inputs["master"]:
            raise ValueError("Master changed since expansion selection")
        if name == "source" and embedded != inputs["original_source"]:
            raise ValueError("Original-source audit changed since expansion selection")
        if name == "conversion" and embedded != inputs["conversion"]:
            raise ValueError("Conversion ledger changed since expansion selection")

    metric = [row for row in expansion["rows"] if row["first_blocker"] == "METRIC_GEOMETRY_READY"]
    if len(metric) != 26 or Counter(row["task"] for row in metric) != {"chips": 18, "poker": 8}:
        raise ValueError("Frozen metric-ready 18+8 population does not match")
    overlap = [row for row in metric if row["source_component_already_in_shortlist"] is True]
    new = [row for row in metric if row["source_component_already_in_shortlist"] is False]
    if len(overlap) != 2 or len(new) != 24 or any(row["task"] != "chips" for row in overlap):
        raise ValueError("Expected two Chips source overlaps and 24 non-overlaps")
    if len({row["session_id"] for row in metric}) != 26:
        raise ValueError("Duplicate metric-ready session")

    master = {row["session_id"]: row for row in load(MASTER)["rows"]}
    source = {row["session_id"]: row for row in load(SOURCE)["rows"]}
    conversion = {row["session_id"]: row for row in load(CONVERSION)["rows"]}
    robot = {row["session_id"]: row for row in load(ROBOT30)["rows"]}
    hard_soft = {row["session"]: row for row in load(HARD_SOFT)["rows"]}
    audited = []
    evidence_count = 0
    for selected in metric:
        sid = selected["session_id"]
        task = selected["task"]
        m, s, c = master[sid], source[sid], conversion[sid]
        if m["task"] != task or s["task"] != task or c["task"].lower() != task:
            raise ValueError(f"Task mismatch: {sid}")
        if m["rc1_split"] != "not_candidate" or s["rc1_split"] != "not_candidate":
            raise ValueError(f"Unexpected RC1 split: {sid}")
        if s["original_rgb_per_frame_source_proven"] or s["original_acquisition_independence_proven"]:
            raise ValueError(f"Original-source proof changed; re-audit required: {sid}")
        if c["first_blocker"] != "METRIC_GEOMETRY_READY":
            raise ValueError(f"Conversion mismatch: {sid}")
        if selected["assets"] != s["assets"]:
            raise ValueError(f"Asset reference mismatch: {sid}")
        for asset in selected["assets"].values():
            verify_ref(asset)
            evidence_count += 1

        prior_path = OLD_ROBOT / task / sid / "RESULT.json"
        prior_ref = ref(prior_path) if prior_path.is_file() else None
        prior = load(prior_path) if prior_ref else None
        rr = robot.get(sid)
        if rr and rr.get("result"):
            verify_ref(rr["result"])
        hs = hard_soft.get(sid)
        if hs:
            for item in hs.get("evidence", {}).values():
                if isinstance(item, dict) and {"path", "bytes", "sha256"} <= item.keys():
                    verify_ref(item)

        robot_status = rr["terminal_status"] if rr else "NOT_SELECTED"
        if robot_status in {"PASSED_OFFLINE_VISUAL_HARD_GEOMETRY", "PASSED_VERIFIED_PRIOR"}:
            disposition = "SUPPLEMENT_EVIDENCE_FOR_OFFLINE_HARD_PASS"
            next_action = "Prove original RGB acquisition identity and independently rebuild causal Clean/Robotized input; offline hard pass is not training eligibility."
        elif prior and prior["status"] == "FAILED_QUALITY_C":
            disposition = "REASSESS_PRIOR_QUALITY_C_HARD_GATES"
            next_action = "Separate prior pose-similarity soft failures from current digital hard gates, then run a bounded same-signature successor only if a hard-gate failure remains."
        elif robot_status == "NOT_EVALUATED_BUDGET":
            disposition = "NOT_EVALUATED_BUDGET_NONQUALITY"
            next_action = "Decide a new bounded evaluation budget before running Robot; no algorithm failure or training qualification has been measured."
        else:
            raise ValueError(f"Unhandled Robot evidence combination: {sid}: {robot_status}")

        if prior and prior["status"] != "FAILED_QUALITY_C":
            raise ValueError(f"Unexpected prior Robot status: {sid}")
        if rr and rr.get("training_eligible") is not False:
            raise ValueError(f"Robot30 training flag changed: {sid}")
        if selected["independent_acquisition_proven"] or selected["original_rgb_source_proven"]:
            raise ValueError(f"Expansion provenance unexpectedly proven: {sid}")

        audited.append({
            "session_id": sid,
            "task": task,
            "frame_count": selected["frame_count"],
            "scheduled_h50_start_count": selected["scheduled_h50_start_count"],
            "legacy_split": selected["legacy_split"],
            "rc1_split": m["rc1_split"],
            "shortlist_exclusion": "FROZEN_NOT_CANDIDATE_SELECTION; quality_evaluated=false; specific selection rationale not proven",
            "source_component_already_in_shortlist": selected["source_component_already_in_shortlist"],
            "source_component_upper_bound_key": selected["source_component_upper_bound_key"],
            "source_group_independence_proven": False,
            "original_rgb_per_frame_source_proven": False,
            "source_proof_status": s["proof_status"],
            "missing_original_proof": ["per-frame original RGB source mapping", "original acquisition identity/independence proof"],
            "conversion_first_blocker": c["first_blocker"],
            "prior_robot_terminal": prior["status"] if prior else "ABSENT",
            "prior_robot_reason_codes": prior.get("reason_codes", []) if prior else [],
            "robot30_terminal": robot_status,
            "robot30_hard_geometry_pass": rr.get("hard_geometry_pass") if rr else None,
            "robot30_input_mode": rr.get("input_mode") if rr else None,
            "hard_soft_v76_status": hs.get("terminal_status") if hs else "NOT_EVALUATED",
            "hard_soft_v76_hard_pass": hs.get("hard_geometry_pass") if hs else None,
            "disposition": disposition,
            "secondary_blockers": ["MISSING_ORIGINAL_RGB_PRODUCER_PROOF", "NOT_RC1_SHORTLIST", "NO_CAUSAL_PAIRED_BUNDLE"],
            "next_action": next_action,
            "train_eligible": False,
            "claim_limit": "Research routing only. SourceSession is not independent acquisition; offline Robot hard geometry and legacy split are not RC1 training eligibility.",
            "evidence": {
                "clip_manifest": selected["assets"]["clip_manifest"],
                "clip_rgb": selected["assets"]["clip_rgb"],
                "slam": selected["assets"]["slam"],
                "tracking": selected["assets"]["tracking"],
                "prior_robot": prior_ref,
                "robot30_result": rr.get("result") if rr else None,
            },
        })

    disposition_counts = Counter(row["disposition"] for row in audited)
    source_components = defaultdict(list)
    for row in audited:
        if not row["source_component_already_in_shortlist"]:
            source_components[(row["task"], row["source_component_upper_bound_key"])].append(row["session_id"])
    if len(source_components) != 20:
        raise ValueError("Source component upper-bound count changed")

    result = {
        "schema_version": "chaoyang-rc1-expand26-disposition-v1",
        "status": "PASSED_RESEARCH_AUDIT_WITH_CAPACITY_HOLD",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authority_promoted": False,
        "checkpoint_pair_status_change": False,
        "inputs": inputs,
        "population": {
            "metric_ready_outside_frozen_candidate_shortlist": 26,
            "chips": 18,
            "poker": 8,
            "sessions_with_source_component_outside_shortlist": 24,
            "sessions_sharing_shortlist_source_component": 2,
            "unproven_new_source_components_upper_bound": 20,
            "original_acquisition_independence_proven": 0,
            "validated_asset_references": evidence_count,
        },
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "claim_limit": "18 Chips + 8 Poker means 26 sessions outside the frozen RC1 shortlist, not 26 new independent sources: 2 Chips share shortlist source components and 24 others collapse to at most 20 unproven source components. Original RGB acquisition and causal paired input remain unproven; checkpoint pairs stay BLOCKED_DATA_VOLUME.",
        "rows": audited,
    }
    out.mkdir(parents=True)
    json_path = out / "EXPAND26_DISPOSITION.json"
    csv_path = out / "EXPAND26_DISPOSITION.csv"
    write_json(json_path, result)
    fields = [
        "session_id", "task", "frame_count", "scheduled_h50_start_count", "legacy_split", "rc1_split",
        "source_component_already_in_shortlist", "source_component_upper_bound_key", "source_group_independence_proven",
        "original_rgb_per_frame_source_proven", "prior_robot_terminal", "robot30_terminal",
        "robot30_hard_geometry_pass", "robot30_input_mode", "hard_soft_v76_status", "disposition", "train_eligible", "next_action",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fields} for row in audited)
    decision = out / "DECISION.md"
    decision.write_text(
        "# RC1 扩候选处置（研究证据）\n\n"
        "冻结 shortlist 外确有 Chips 18＋Poker 8＝26 条 metric-ready 会话；其中 Chips 2 条的来源组件已在 shortlist。"
        "剩余 24 条会话最多对应 20 个待证来源组件，不能写成 24 或 26 个独立采集。\n\n"
        f"处置计数：{dict(sorted(disposition_counts.items()))}。"
        "所有 26 条均缺原始逐帧 RGB 来源及采集独立性证明，T0 中均为 not_candidate。"
        "离线 Robot 硬几何通过仅为候选；旧 C 需按当前 hard/soft 合同复核，预算未评估不是算法失败。\n\n"
        "下一步先对离线 hard-pass 条目补源证据与因果 Clean/Robotized 链；"
        "旧 C 按固定小 canary 拆软门与硬门；预算未评估项另行冻结计算预算。"
        "不得改 RC1 split、容量门、checkpoint pair 或 current authority。\n",
        encoding="utf-8",
    )
    write_json(out / "RESULT.json", {
        "schema_version": "chaoyang-rc1-expand26-result-v1",
        "status": result["status"],
        "claim_limit": result["claim_limit"],
        "authority_promoted": False,
        "checkpoint_pair_status_change": False,
        "population": result["population"],
        "disposition_counts": result["disposition_counts"],
        "inputs": inputs,
        "outputs": {"matrix_json": ref(json_path), "matrix_csv": ref(csv_path), "decision": ref(decision)},
    })
    print(json.dumps({"status": result["status"], "population": result["population"], "disposition_counts": result["disposition_counts"], "result": str(out / "RESULT.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
