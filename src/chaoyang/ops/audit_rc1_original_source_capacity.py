#!/usr/bin/env python3
"""Bounded exact78 source-lineage audit; never upgrades training capacity.

Read only frozen T0 manifest paths.  Tracker/SLAM sourceSession labels are
candidate lineage, not proof that original RGB assets are independent.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


MASTER = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/t0_freeze_capacity_split/attempts/attempt_0001/MASTER_LEDGER.json"


def checked_ref(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    actual = {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}
    if expected is not None and (actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256")):
        raise RuntimeError(f"frozen manifest closure mismatch: {path}")
    return actual


def lineage_from_jsonl(path: Path, source_field: str) -> dict[str, Any]:
    ids: set[str] = set()
    first_last: dict[str, list[int | None]] = {}
    records: dict[str, list[int | None]] = {}
    rows = 0
    unknown = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row if source_field == "direct" else row.get(source_field, {})
            source = meta.get("sourceSession")
            if not isinstance(source, str) or not source:
                # A merge-header row may list every source in the run.  It is
                # not evidence that every listed source belongs to this clip.
                unknown += 1
                continue
            rows += 1
            ids.add(source)
            stamp = meta.get("sourceTimeStampNs")
            index = meta.get("sourceRecordIndex")
            if source not in first_last:
                first_last[source] = [int(stamp) if stamp is not None else None, int(stamp) if stamp is not None else None]
                records[source] = [int(index) if index is not None else None, int(index) if index is not None else None]
            else:
                if stamp is not None:
                    first_last[source][0] = min(first_last[source][0], int(stamp)) if first_last[source][0] is not None else int(stamp)
                    first_last[source][1] = max(first_last[source][1], int(stamp)) if first_last[source][1] is not None else int(stamp)
                if index is not None:
                    records[source][0] = min(records[source][0], int(index)) if records[source][0] is not None else int(index)
                    records[source][1] = max(records[source][1], int(index)) if records[source][1] is not None else int(index)
    return {
        "source_sessions": sorted(ids),
        "rows_with_source": rows,
        "rows_without_source": unknown,
        "source_timestamp_ns_range": first_last,
        "source_record_index_range": records,
    }


def connected_lineage_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clips sharing any claimed original source must remain in one split."""
    parent = {row["session_id"]: row["session_id"] for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    owner: dict[str, str] = {}
    for row in rows:
        session = row["session_id"]
        for source in row["claimed_source_sessions"]:
            if source in owner:
                union(session, owner[source])
            else:
                owner[source] = session
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["claimed_source_sessions"]:
            components[find(row["session_id"])].append(row)
    result = []
    for members in components.values():
        tasks = sorted({row["task"] for row in members})
        splits = sorted({row["rc1_split"] for row in members if row["rc1_split"] in ("train", "validation")})
        result.append({
            "sessions": sorted(row["session_id"] for row in members),
            "tasks": tasks,
            "source_sessions": sorted({s for row in members for s in row["claimed_source_sessions"]}),
            "candidate_splits": splits,
            "split_conflict": len(splits) > 1,
            "independence_proven": False,
        })
    return sorted(result, key=lambda value: value["sessions"][0])


def audit(master_path: Path) -> dict[str, Any]:
    master = json.loads(master_path.read_text(encoding="utf-8"))
    frozen = master["rows"]
    if len(frozen) != 156 or len({row["session_id"] for row in frozen}) != 156:
        raise RuntimeError("T0 cohort must be 156 unique sessions")
    if {task: sum(row["task"] == task for row in frozen) for task in ("chips", "poker")} != {"chips": 78, "poker": 78}:
        raise RuntimeError("T0 task denominator mismatch")
    rows: list[dict[str, Any]] = []
    for frozen_row in frozen:
        session = frozen_row["session_id"]
        manifest_ref = frozen_row.get("source_group_evidence")
        if not isinstance(manifest_ref, dict):
            rows.append({
                "session_id": session, "task": frozen_row["task"],
                "rc1_split": frozen_row["rc1_split"],
                "legacy_candidate_split": frozen_row["legacy_candidate_split"],
                "claimed_source_sessions": [], "proof_status": "MISSING_CLIP_MANIFEST",
                "quality_evaluated": frozen_row["quality_evaluated"],
                "train_eligible": frozen_row["train_eligible"],
            })
            continue
        manifest_path = Path(manifest_ref["path"])
        verified_manifest = checked_ref(manifest_path, manifest_ref)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("clip_name") != session:
            raise RuntimeError(f"clip identity mismatch: {session}")
        files = data.get("files", {})
        video = manifest_path.parent / files["video"]
        tracking = manifest_path.parent / files["tracking"]
        slam = manifest_path.parent / files["slam"]
        refs = {
            "clip_manifest": verified_manifest,
            "clip_rgb": checked_ref(video),
            "tracking": checked_ref(tracking),
            "slam": checked_ref(slam),
        }
        tracking_lineage = lineage_from_jsonl(tracking, "_merge")
        slam_lineage = lineage_from_jsonl(slam, "direct")
        track_ids = set(tracking_lineage["source_sessions"])
        slam_ids = set(slam_lineage["source_sessions"])
        claimed = sorted(track_ids | slam_ids)
        exact_match = track_ids == slam_ids and bool(claimed)
        rows.append({
            "session_id": session,
            "task": frozen_row["task"],
            "rc1_split": frozen_row["rc1_split"],
            "legacy_candidate_split": frozen_row["legacy_candidate_split"],
            "scheduled_h50_start_count": frozen_row["scheduled_h50_start_count"],
            "t0_source_group_id": frozen_row["source_group_id"],
            "clip_selection": data.get("selection"),
            "merged_rgb_pieces": data.get("pieces", []),
            "assets": refs,
            "tracking_lineage": tracking_lineage,
            "slam_lineage": slam_lineage,
            "claimed_source_sessions": claimed,
            "tracking_slam_source_set_equal": exact_match,
            "original_rgb_per_frame_source_proven": False,
            "original_acquisition_independence_proven": False,
            "proof_status": "TRACKER_SLAM_SOURCE_MATCH_RGB_MERGED_ONLY" if exact_match else "TRACKER_SLAM_SOURCE_CONFLICT_OR_MISSING",
            "quality_evaluated": frozen_row["quality_evaluated"],
            "train_eligible": frozen_row["train_eligible"],
        })
    clusters = connected_lineage_clusters(rows)
    summary = {}
    for task in ("chips", "poker"):
        task_rows = [row for row in rows if row["task"] == task]
        candidate = [row for row in task_rows if row["legacy_candidate_split"] != "not_candidate"]
        qualified = [row for row in candidate if row["train_eligible"]]
        task_clusters = [cluster for cluster in clusters if task in cluster["tasks"]]
        candidate_ids = {row["session_id"] for row in candidate}
        candidate_clusters = [cluster for cluster in task_clusters if set(cluster["sessions"]) & candidate_ids]
        train_clusters = [cluster for cluster in candidate_clusters if "train" in cluster["candidate_splits"]]
        validation_clusters = [cluster for cluster in candidate_clusters if "validation" in cluster["candidate_splits"]]
        summary[task] = {
            "sessions": len(task_rows),
            "legacy_candidate_sessions": len(candidate),
            "quality_qualified_sessions_in_frozen_t0": len(qualified),
            "quality_not_evaluated_sessions_in_frozen_t0": sum(not row["quality_evaluated"] for row in task_rows),
            "claimed_source_session_ids_all": len({s for row in task_rows for s in row["claimed_source_sessions"]}),
            "claimed_source_session_ids_candidate": len({s for row in candidate for s in row["claimed_source_sessions"]}),
            "candidate_connected_lineage_clusters": len(candidate_clusters),
            "candidate_train_lineage_clusters": len(train_clusters),
            "candidate_validation_lineage_clusters": len(validation_clusters),
            "candidate_clusters_with_split_conflict": sum(cluster["split_conflict"] for cluster in candidate_clusters),
            "verified_independent_rgb_acquisitions": 0,
            "capacity_16_train_3_validation_proven": False,
            "claim_limit": "SourceSession and connected cluster counts are upper-bound candidates in the frozen T0 shortlist, not verified independent RGB acquisitions or current RC1 quality eligibility.",
        }
    return {
        "schema_version": "chaoyang-rc1-original-source-lineage-audit-v1",
        "created_at": now_iso(),
        "status": "PASSED_DIAGNOSTIC_WITH_PROVENANCE_GAP",
        "authority_promoted": False,
        "checkpoint_pair_status_change": False,
        "claim_limit": "Exact78 tracker/SLAM lineage and clip RGB SHA closure only. Original per-frame RGB source mapping and independent acquisition identity remain unproven; do not change official T0 or training pair status.",
        "inputs": {"master_ledger": artifact_ref(master_path)},
        "rows": rows,
        "connected_lineage_clusters": clusters,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = audit(MASTER)
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "task_id": "research_rc1_original_source_capacity",
        "status": result["status"],
        "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
        "input_master": result["inputs"]["master_ledger"],
        "result": artifact_ref(output / "RESULT.json"),
    })
    print(json.dumps({"status": result["status"], "summary": result["summary"], "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
