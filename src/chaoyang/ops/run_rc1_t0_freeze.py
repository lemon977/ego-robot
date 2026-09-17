#!/usr/bin/env python3
"""Freeze the exact78 RC1 denominator, source groups, starts and capacity."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
)


RUN_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
OUT = RUN_ROOT / "t0_freeze_capacity_split/attempts/attempt_0001"
FINAL_MATRIX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal/EXACT78_FINAL_TERMINAL_MATRIX.json"
CROSS_MATRIX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
OLD_CAPACITY = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_visual_aux_capacity_preflight_v71/RESULT.json"
CONTRACT = REPO_ROOT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json"


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"fresh immutable artifact required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"fresh immutable artifact required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, value.encode("utf-8"))


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_video_from_hawor(row: dict[str, Any]) -> Path | None:
    result_ref = (row.get("hawor") or {}).get("result")
    if not isinstance(result_ref, dict):
        return None
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file():
        return None
    value = load_json(result_path).get("inputs", {}).get("source_video")
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    return Path(str(value))


def _source_group(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    video = _source_video_from_hawor(row)
    if video is None:
        return "UNKNOWN_SOURCE_GROUP", None, "NO_HAWOR_SOURCE_VIDEO_EVIDENCE"
    manifest = video.parent / "clip_manifest.json"
    if not manifest.is_file():
        return "UNKNOWN_SOURCE_GROUP", None, "CLIP_MANIFEST_MISSING"
    data = load_json(manifest)
    pieces = []
    for piece in data.get("pieces", []):
        pieces.append({
            "session": str(piece.get("session", "")),
            "source_video": str(piece.get("source_video", "")),
        })
    identity = {
        "run_id": str(data.get("run_id", "")),
        "source_root": str(data.get("source_root", "")),
        "pieces": sorted(pieces, key=lambda x: (x["session"], x["source_video"])),
    }
    if not identity["run_id"] and not any(x["source_video"] for x in pieces):
        return "UNKNOWN_SOURCE_GROUP", artifact_ref(manifest), "MANIFEST_HAS_NO_ORIGINAL_ACQUISITION_IDENTITY"
    return f"source_{_canonical_sha(identity)[:20]}", artifact_ref(manifest), None


def _candidate_sessions(capacity: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {"chips": {}, "poker": {}}
    for task, spec in capacity.get("tasks", {}).items():
        for split in ("train", "validation"):
            for source in spec.get(split, {}).get("sources", {}).values():
                for session in source.get("sessions", []):
                    output[task][str(session)] = split
    return output


def _starts(frame_count: int) -> list[int]:
    return list(range(15, max(15, frame_count - 50), 5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh T0 attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("T0 CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    task = next((x for x in state.get("tasks", []) if x.get("task_id") == "rc1_t0_freeze_capacity_split"), None)
    if task is None or task.get("status") != "PENDING":
        raise RuntimeError("RC1-T0 must be the registered PENDING task")

    final = load_json(FINAL_MATRIX)
    cross = load_json(CROSS_MATRIX)
    old_capacity = load_json(OLD_CAPACITY)
    candidates = _candidate_sessions(old_capacity)
    cross_by_session = {str(x["session_id"]): x for x in cross["rows"]}
    rows = final.get("rows", [])
    if len(rows) != 156 or len({x["session_id"] for x in rows}) != 156:
        raise RuntimeError("exact78 denominator is not exactly 156 unique sessions")
    task_counts = {key: sum(x.get("task") == key for x in rows) for key in ("chips", "poker")}
    if task_counts != {"chips": 78, "poker": 78}:
        raise RuntimeError(f"task denominator mismatch: {task_counts}")

    master: list[dict[str, Any]] = []
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for terminal in sorted(rows, key=lambda x: (x.get("task", ""), x["session_id"])):
        session = str(terminal["session_id"])
        task_name = str(terminal["task"])
        cross_row = cross_by_session.get(session)
        if cross_row is None:
            raise RuntimeError(f"missing cross-stage row: {session}")
        group_id, evidence, group_error = _source_group(cross_row)
        frame_count = int(cross_row.get("frame_count") or 0)
        starts = _starts(frame_count)
        candidate_split = candidates.get(task_name, {}).get(session, "not_candidate")
        row = {
            "session_id": session,
            "task": task_name,
            "frame_count": frame_count,
            "scheduled_h50_starts": starts,
            "scheduled_h50_start_count": len(starts),
            "source_group_id": group_id,
            "source_group_evidence": evidence,
            "source_group_error": group_error,
            "legacy_candidate_split": candidate_split,
            "rc1_split": "not_candidate",
            "quality_evaluated": False,
            "train_eligible": False,
            "terminal_row": terminal,
        }
        master.append(row)
        if group_id != "UNKNOWN_SOURCE_GROUP":
            group_rows[group_id].append(row)

    # A source group may not cross splits. Any legacy conflict is conservatively
    # assigned to train/development; validation may only use untouched groups.
    group_split: dict[str, str] = {}
    for group_id, members in group_rows.items():
        requested = {x["legacy_candidate_split"] for x in members if x["legacy_candidate_split"] != "not_candidate"}
        if "train" in requested or len(requested) > 1:
            group_split[group_id] = "train"
        elif requested == {"validation"}:
            group_split[group_id] = "validation"
        else:
            group_split[group_id] = "not_candidate"
    for row in master:
        if row["source_group_id"] != "UNKNOWN_SOURCE_GROUP" and row["legacy_candidate_split"] != "not_candidate":
            row["rc1_split"] = group_split[row["source_group_id"]]

    source_groups = []
    for group_id, members in sorted(group_rows.items()):
        source_groups.append({
            "source_group_id": group_id,
            "rc1_split": group_split[group_id],
            "tasks": sorted({x["task"] for x in members}),
            "sessions": sorted(x["session_id"] for x in members),
            "session_count": len(members),
            "evidence": members[0]["source_group_evidence"],
        })

    capacity: dict[str, Any] = {}
    for task_name in ("chips", "poker"):
        candidates_for_task = [x for x in master if x["task"] == task_name and x["legacy_candidate_split"] != "not_candidate"]
        train_groups = {x["source_group_id"] for x in candidates_for_task if x["rc1_split"] == "train" and x["source_group_id"] != "UNKNOWN_SOURCE_GROUP"}
        validation_groups = {x["source_group_id"] for x in candidates_for_task if x["rc1_split"] == "validation" and x["source_group_id"] != "UNKNOWN_SOURCE_GROUP"}
        all_groups = train_groups | validation_groups
        scheduled_train = sum(x["scheduled_h50_start_count"] for x in candidates_for_task if x["rc1_split"] == "train")
        scheduled_validation = sum(x["scheduled_h50_start_count"] for x in candidates_for_task if x["rc1_split"] == "validation")
        minimum_met_upper_bound = (
            len(train_groups) >= 16 and len(validation_groups) >= 3
            and scheduled_train >= 256 and scheduled_validation >= 48
        )
        capacity[task_name] = {
            "candidate_sessions": len(candidates_for_task),
            "independent_source_groups_total": len(all_groups),
            "train_source_groups": len(train_groups),
            "validation_source_groups": len(validation_groups),
            "scheduled_train_windows_upper_bound": scheduled_train,
            "scheduled_validation_windows_upper_bound": scheduled_validation,
            "requirements": {"train_source_groups": 16, "validation_source_groups": 3, "train_windows": 256, "validation_windows": 48},
            "capacity_upper_bound_before_quality": minimum_met_upper_bound,
            "capacity_ready_under_rc1": False,
            "pair_terminal": "NOT_EVALUATED_QUALITY" if minimum_met_upper_bound else "BLOCKED_DATA_VOLUME",
            "claim_limit": "Source-group and scheduled-window upper bound only; no Mask/Clean/Robot quality was evaluated.",
        }

    created = now_iso()
    master_value = {
        "schema_version": "chaoyang-rc1-master-ledger-v1",
        "created_at": created,
        "rows": master,
        "counts": {"total": 156, **task_counts, "unknown_source_group": sum(x["source_group_id"] == "UNKNOWN_SOURCE_GROUP" for x in master)},
        "inputs": [artifact_ref(FINAL_MATRIX), artifact_ref(CROSS_MATRIX), artifact_ref(OLD_CAPACITY), artifact_ref(CONTRACT)],
        "claim_limit": "Frozen denominator and source lineage; quality and training eligibility remain untested.",
    }
    source_value = {
        "schema_version": "chaoyang-rc1-source-group-ledger-v1",
        "created_at": created,
        "groups": source_groups,
        "claim_limit": "Original-acquisition grouping derived from immutable clip manifests; session fragments are not independent groups.",
    }
    capacity_value = {
        "schema_version": "chaoyang-rc1-capacity-report-v1",
        "created_at": created,
        "tasks": capacity,
        "status": "PASSED_AUDIT_WITH_CAPACITY_BLOCK" if any(x["pair_terminal"] == "BLOCKED_DATA_VOLUME" for x in capacity.values()) else "PASSED",
        "claim_limit": "Pre-quality theoretical capacity only.",
    }
    master_path = OUT / "MASTER_LEDGER.json"
    source_path = OUT / "SOURCE_GROUP_LEDGER.json"
    capacity_path = OUT / "CAPACITY_REPORT.json"
    _write_json(master_path, master_value)
    _write_json(source_path, source_value)
    _write_json(capacity_path, capacity_value)
    csv_path = OUT / "MASTER_LEDGER.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "task", "frame_count", "source_group_id", "legacy_candidate_split", "rc1_split", "scheduled_h50_start_count"])
        writer.writeheader()
        for row in master:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    result_path = OUT / "RESULT.json"
    _write_json(result_path, {
        "schema_version": "chaoyang-rc1-t0-result-v1", "task_id": "rc1_t0_freeze_capacity_split",
        "status": "PASSED", "created_at": created, "capacity": capacity,
        "master_ledger": artifact_ref(master_path), "source_group_ledger": artifact_ref(source_path),
        "capacity_report": artifact_ref(capacity_path), "authority_promoted": False,
    })
    _write_json(OUT / "METRICS.json", {"status": "PASSED", "sessions": 156, "tasks": capacity})
    _write_json(OUT / "NEXT_ACTION.json", {"task_id": "rc1_t1_sam31_bounded_repair", "reason": "Continue operational pilot even when checkpoint capacity is blocked."})
    _write_text(OUT / "DECISION.md", "# RC1 T0 容量结论\n\n已冻结156条分母、原始采集source group与质量筛选前H50起点。source group容量不足时，checkpoint pair封为BLOCKED_DATA_VOLUME，但Mask/Clean/Robot/Smoke生产闭环继续。\n")
    manifest_path = OUT / "ARTIFACT_MANIFEST.json"
    _write_json(manifest_path, {"schema_version": "chaoyang-rc1-t0-artifact-manifest-v1", "artifacts": [artifact_ref(x) for x in (master_path, csv_path, source_path, capacity_path, result_path, OUT / "METRICS.json", OUT / "NEXT_ACTION.json", OUT / "DECISION.md")]})
    _write_json(OUT / "RUN_RECEIPT.json", {"schema_version": "chaoyang-rc1-t0-run-receipt-v1", "task_id": "rc1_t0_freeze_capacity_split", "status": "PASSED", "result": artifact_ref(result_path), "manifest": artifact_ref(manifest_path)})
    _write_json(OUT / "RESULT_SUMMARY.json", {"task_id": "rc1_t0_freeze_capacity_split", "status": "PASSED", "capacity": {k: v["pair_terminal"] for k, v in capacity.items()}, "next_task": "rc1_t1_sam31_bounded_repair"})

    task.update(status="PASSED", attempt=1, updated_at=created, result=artifact_ref(result_path), heartbeat_at=None, pid=None, proc_start_ticks=None)
    state["next_task"] = None
    state["rc1_pair_status"] = {key: value["pair_terminal"] for key, value in capacity.items()}
    state["rc1_release_flags"]["DATA_MINIMUM_MET_CHIPS"] = False
    state["rc1_release_flags"]["DATA_MINIMUM_MET_POKER"] = False
    state["rc1_release_flags"]["RC1_RELEASE_STATUS"] = "T0_COMPLETE_CAPACITY_LIMITED"
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_t0_freeze_capacity_split", "attempt": 1, "status": "PASSED", "created_at": created, "message": "RC1 denominator, source groups and scheduled starts frozen.", "result": artifact_ref(result_path)}])[-100:]
    published = publish_bundle(authority, state, event_type="RC1_T0_CAPACITY_FROZEN", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "capacity": capacity, "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
