#!/usr/bin/env python3
"""Publish an immutable, evidence-bound research-round delivery index."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1"
GOV_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
QUEUE_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_b_regression_queue_v1/QUEUE_RESULT.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def newest(pattern: str) -> Path | None:
    values = sorted(RUN.glob(pattern))
    return values[-1] if values else None


def main() -> None:
    for name in ("FINAL_RESULT.json", "TASK_MATRIX_REV_0002.json", "VISUAL_REVIEW_INDEX_REV_0002.json"):
        if (RUN / name).exists():
            raise FileExistsError(RUN / name)
    snapshot = RUN / "RUN_START_SNAPSHOT.json"
    initial = RUN / "TASK_MATRIX_REV_0001.json"
    if artifact_ref(Path(load(snapshot)["refs"]["script"]["path"])) != load(snapshot)["refs"]["script"]:
        raise RuntimeError("start snapshot generator SHA changed")
    current = load(GOV_RECEIPT)
    if current.get("freshness", {}).get("status") != "FRESH":
        raise RuntimeError("formal governance not FRESH")
    rows = load(initial)["rows"]
    evidence: dict[str, dict] = {}
    videos: list[dict] = []
    for row in rows:
        lane = RUN / row["lane"]
        receipts = sorted(lane.glob("**/RESULT.json")) if lane.exists() else []
        summaries = sorted(lane.glob("**/RESULT_SUMMARY.json")) if lane.exists() else []
        row["result_refs"] = [artifact_ref(p) for p in receipts]
        row["summary_refs"] = [artifact_ref(p) for p in summaries]
        evidence[row["task_id"]] = {"result_refs": row["result_refs"], "summary_refs": row["summary_refs"]}
        for video in sorted(lane.glob("**/*.mp4")) if lane.exists() else []:
            videos.append({"lane": row["task_id"], "video": artifact_ref(video),
                           "verification": "LANE_RECEIPT_REQUIRED", "input_mode": "OFFLINE_VISUAL_OR_DEVELOPMENT"})
        if row["task_id"] == "A_MASK":
            q = load(QUEUE_RESULT)
            if q["status"] != "BLOCKED_RESOURCE":
                raise RuntimeError("existing SAM queue status changed; re-audit")
            row["status"] = "BLOCKED_RESOURCE_REGRESSION"
            row["existing_queue_result"] = artifact_ref(QUEUE_RESULT)
        elif row["task_id"] == "B_CLEAN":
            required = [
                lane / case / "attempt_0001/RESULT.json"
                for case in ("Poker245", "Chips039")
            ] + [lane / case / "closure_audit_0001/CLOSURE_AUDIT.json"
                 for case in ("Poker245", "Chips039")]
            row["status"] = "PASSED_OFFLINE_LINEAGE_DIAGNOSTIC" if all(p.exists() for p in required) else "PARTIAL_EVIDENCE"
            row["session_ids"] = ["play_cards_0903_245", "get_potato_chips_0902_039"]
            row["closure_refs"] = [artifact_ref(p) for p in required if p.exists()]
        elif row["task_id"] in {"C_DEPTH", "D_OBJECT_CONTACT"}:
            summary = lane / "RESULT_SUMMARY.json"
            row["status"] = "PASSED_DEVELOPMENT_DIAGNOSTIC" if summary.exists() else "PENDING_EVIDENCE"
            if summary.exists():
                row["summary"] = artifact_ref(summary)
                value = load(summary)
                if value.get("session_id"):
                    row["session_ids"] = [value["session_id"]]
        elif row["task_id"] == "E_ROBOT30":
            delivery = newest("lane_e_robot30/attempts/attempt_*/ROBOT30_VIDEO_DELIVERY_MATRIX.json")
            if delivery is None:
                row["status"] = "PENDING_EVIDENCE"
                continue
            delivery_rows = load(delivery)["rows"]
            if len(delivery_rows) != 60 or len({x["session_id"] for x in delivery_rows}) != 60:
                raise RuntimeError("Robot video matrix must have exactly 60 unique sessions")
            row["delivery_matrix"] = artifact_ref(delivery)
            row["session_ids"] = [x["session_id"] for x in delivery_rows]
            row["session_binding_status"] = "BOUND_60"
            counts = {"chips": 0, "poker": 0}
            for item in delivery_rows:
                ref = item.get("verified_video")
                if not ref:
                    continue
                if artifact_ref(Path(ref["path"])) != ref:
                    raise RuntimeError(f"Robot video SHA changed since full decode: {ref['path']}")
                counts[item["task"]] += 1
                videos.append({"lane": "E_ROBOT30", "session_id": item["session_id"],
                               "video": ref, "verification": "SHA_AND_FULL_DECODE_VERIFIED_BY_E_LANE",
                               "quality_status": item["robot30_terminal_status"],
                               "hard_geometry_pass": item["robot30_hard_geometry_pass"],
                               "input_mode": item["input_mode"], "training_eligible": item["training_eligible"]})
            row["verified_full_video_counts"] = counts
            row["status"] = "PASSED_VIDEO60" if counts == {"chips": 30, "poker": 30} else "PARTIAL_VIDEO_DELIVERY"
        elif row["task_id"] == "F_DOCS":
            row["status"] = "PASSED_RESEARCH_INDEX"
    mask_decision = RUN / "lane_a_mask/attempts/attempt_0001/POKER015_224_QUALITY_DECISION.json"
    if mask_decision.exists():
        decision = load(mask_decision)
        mask_video = decision["observed"]["full_review_video"]
        if artifact_ref(Path(mask_video["path"])) != mask_video:
            raise RuntimeError("Poker015 video changed since Mask quality decision")
        videos.append({"lane": "A_MASK", "session_id": decision["session_id"],
                       "video": mask_video, "verification": "SHA_AND_FULL_DECODE_VERIFIED_BY_A_LANE",
                       "quality_status": decision["quality_decision"],
                       "input_mode": decision["input_mode"], "training_eligible": False})
    matrix_path = RUN / "TASK_MATRIX_REV_0002.json"
    visual_path = RUN / "VISUAL_REVIEW_INDEX_REV_0002.json"
    atomic_json(matrix_path, {"schema_version": "rc1-multiline-task-matrix-v2", "created_at": now_iso(),
                              "start_snapshot": artifact_ref(snapshot), "previous": artifact_ref(initial), "rows": rows})
    atomic_json(visual_path, {"schema_version": "rc1-multiline-visual-index-v2", "created_at": now_iso(),
                              "start_snapshot": artifact_ref(snapshot), "videos": videos,
                              "claim_limit": "Full video delivery, quality and causal eligibility are separate fields."})
    atomic_json(RUN / "FINAL_RESULT.json", {
        "schema_version": "rc1-multiline-research-final-v1", "created_at": now_iso(),
        "status": "PARTIAL_WITH_EXECUTED_EVIDENCE", "formal_rc1_results_overwritten": False,
        "formal_authority_promoted": False, "checkpoint_training_started": False,
        "governance_at_finalization": artifact_ref(GOV_RECEIPT),
        "start_snapshot": artifact_ref(snapshot), "task_matrix": artifact_ref(matrix_path),
        "visual_review_index": artifact_ref(visual_path), "lane_evidence": evidence,
        "claim_limit": "Research delivery only. SAM regression was resource-blocked; Clean remains offline; Robot60 full videos do not imply 60 hard passes or causal training eligibility.",
        "code": artifact_ref(Path(__file__)),
    })
    print(json.dumps({"matrix": str(matrix_path), "visual_index": str(visual_path),
                      "video_entries": len(videos),
                      "robot_counts": next(x for x in rows if x["task_id"] == "E_ROBOT30").get("verified_full_video_counts")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
