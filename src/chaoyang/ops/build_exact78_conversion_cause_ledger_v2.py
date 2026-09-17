#!/usr/bin/env python3
"""Build the V7.1 exact78 conversion-cause ledger from pinned evidence.

The ledger is diagnostic only.  It classifies every frozen session once at the
first upstream blocker, then records the current downstream outcome separately
so infrastructure failures cannot be mistaken for algorithm quality failures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
BLOCKERS = (
    "HAWOR_C",
    "ROLE_AFTER_HAWOR",
    "OBJECT_AFTER_HAWOR_ROLE",
    "CALIBRATION_MISSING_AFTER_TRIPLE",
    "METRIC_READY",
)
FAILURE_CATEGORIES = (
    "ALGORITHM_FAILURE",
    "INFRASTRUCTURE_FAILURE",
    "PROVENANCE_MISSING",
)
NO_FAILURE = "NO_FAILURE_CURRENT_SNAPSHOT"
CSV_FIELDS = (
    "position",
    "session_id",
    "task",
    "date",
    "acquisition_contract",
    "split",
    "first_blocker",
    "first_blocker_category",
    "current_primary_cause_category",
    "current_primary_cause",
    "hawor_grade",
    "role_grade",
    "object_grade",
    "metric_ready",
    "clean_state",
    "robot_state",
    "robot_eligible",
    "robot_candidate",
    "successor_newly_passed",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_path(resolved),
    }


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percent(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def grade_ab(value: Any) -> bool:
    return value in {"A", "B"}


def first_blocker(row: Mapping[str, Any]) -> str:
    if not grade_ab(row["hawor"]["grade"]):
        return "HAWOR_C"
    if not grade_ab(row["role_mask"]["grade"]):
        return "ROLE_AFTER_HAWOR"
    if not grade_ab(row["object_mask"]["grade"]):
        return "OBJECT_AFTER_HAWOR_ROLE"
    if not bool(row["metric_ready_wave0"]):
        return "CALIBRATION_MISSING_AFTER_TRIPLE"
    return "METRIC_READY"


def blocker_category(blocker: str) -> str:
    if blocker in {"HAWOR_C", "ROLE_AFTER_HAWOR", "OBJECT_AFTER_HAWOR_ROLE"}:
        return "ALGORITHM_FAILURE"
    if blocker == "CALIBRATION_MISSING_AFTER_TRIPLE":
        return "PROVENANCE_MISSING"
    return NO_FAILURE


def clean_passed(state: Any) -> bool:
    return isinstance(state, str) and (state == "PASSED" or state.startswith("PASSED_"))


def robot_candidate(state: Any) -> bool:
    return state in {
        "POSE_ONLY_VISUAL_ROBOT",
        "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY",
        "METRIC_CONTACT_ROBOT",
    }


def primary_cause(row: Mapping[str, Any], blocker: str) -> tuple[str, str]:
    initial = blocker_category(blocker)
    if initial != NO_FAILURE:
        return initial, blocker
    clean_state = str(row.get("clean_state", "UNKNOWN"))
    robot_state = str(row.get("robot_current_state", "UNKNOWN"))
    if clean_state == "FAILED_RUNTIME_FINAL":
        return "INFRASTRUCTURE_FAILURE", "CLEAN_FAILED_RUNTIME_FINAL"
    if clean_state == "FAILED_QUALITY_C":
        return "ALGORITHM_FAILURE", "CLEAN_FAILED_QUALITY_C"
    if robot_state == "FAILED_RUNTIME_FINAL":
        return "INFRASTRUCTURE_FAILURE", "ROBOT_FAILED_RUNTIME_FINAL"
    if robot_state == "FAILED_QUALITY_C":
        return "ALGORITHM_FAILURE", "ROBOT_FAILED_QUALITY_C"
    return NO_FAILURE, "NONE_CURRENT_SNAPSHOT"


def extract_acquisition_contract(raw_path: str, date: str) -> str:
    parts = Path(raw_path).parts
    candidates = [part for part in parts if part.startswith("chips_cards_tracker_")]
    if len(candidates) != 1:
        raise ValueError(f"cannot derive unique acquisition contract from raw_path={raw_path!r}")
    contract = candidates[0]
    if not contract.endswith(f"_{date}"):
        raise ValueError(f"acquisition contract/date mismatch: {contract!r} versus {date!r}")
    return contract


def validate_matrix(matrix: Mapping[str, Any], expected_sessions: int) -> list[dict[str, Any]]:
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_sessions:
        raise ValueError(f"matrix must contain exactly {expected_sessions} rows")
    session_ids = [row.get("session_id") for row in rows]
    if len(set(session_ids)) != expected_sessions or None in session_ids:
        raise ValueError(f"matrix must contain exactly {expected_sessions} unique session_id values")
    for row in rows:
        blocker = first_blocker(row)
        if blocker == "METRIC_READY" and not bool(row.get("three_upstream_ab")):
            raise ValueError(f"metric-ready row is not triple A/B: {row['session_id']}")
    return rows


def validate_cohort(
    cohort: Mapping[str, Any], matrix_rows: Sequence[Mapping[str, Any]], expected_sessions: int
) -> dict[str, dict[str, Any]]:
    sessions = cohort.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != expected_sessions:
        raise ValueError(f"cohort must contain exactly {expected_sessions} sessions")
    by_id = {row.get("session_id"): row for row in sessions}
    if len(by_id) != expected_sessions or set(by_id) != {row["session_id"] for row in matrix_rows}:
        raise ValueError("cohort and matrix session sets differ")
    return by_id


def rate_record(stage: str, numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "numerator": numerator,
        "denominator": denominator,
        "rate_percent": percent(numerator, denominator),
    }


def row_grade(row: Mapping[str, Any], stage: str) -> Any:
    normalized_key = {"hawor": "hawor_grade", "role_mask": "role_grade", "object_mask": "object_grade"}[stage]
    if normalized_key in row:
        return row[normalized_key]
    return row[stage]["grade"]


def row_metric_ready(row: Mapping[str, Any]) -> bool:
    return bool(row["metric_ready"] if "metric_ready" in row else row["metric_ready_wave0"])


def conditional_rates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw = len(rows)
    hawor = sum(grade_ab(row_grade(row, "hawor")) for row in rows)
    hawor_role = sum(
        grade_ab(row_grade(row, "hawor")) and grade_ab(row_grade(row, "role_mask"))
        for row in rows
    )
    triple = sum(
        grade_ab(row_grade(row, "hawor"))
        and grade_ab(row_grade(row, "role_mask"))
        and grade_ab(row_grade(row, "object_mask"))
        for row in rows
    )
    metric = sum(row_metric_ready(row) for row in rows)
    clean = sum(row_metric_ready(row) and clean_passed(row.get("clean_state")) for row in rows)
    eligible = sum(
        row_metric_ready(row) and clean_passed(row.get("clean_state")) for row in rows
    )
    candidates = sum(
        row_metric_ready(row)
        and clean_passed(row.get("clean_state"))
        and robot_candidate(row.get("robot_state", row.get("robot_current_state")))
        for row in rows
    )
    return [
        rate_record("RAW_TO_HAWOR_AB", hawor, raw),
        rate_record("HAWOR_AB_TO_ROLE_AB", hawor_role, hawor),
        rate_record("HAWOR_ROLE_AB_TO_OBJECT_AB", triple, hawor_role),
        rate_record("TRIPLE_AB_TO_METRIC_READY", metric, triple),
        rate_record("METRIC_READY_TO_CLEAN_PASSED", clean, metric),
        rate_record("ROBOT_ELIGIBLE_TO_ROBOT_CANDIDATE", candidates, eligible),
    ]


def distribution(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(str(record[key]) for key in keys)].append(record)
    output = []
    for values in sorted(groups):
        selected = groups[values]
        blockers = Counter(str(record["first_blocker"]) for record in selected)
        item: dict[str, Any] = dict(zip(keys, values))
        item.update(
            {
                "total": len(selected),
                "first_blockers": {name: blockers[name] for name in BLOCKERS},
                "conditional_rates": conditional_rates(selected),
            }
        )
        output.append(item)
    return output


def successor_changes(
    baseline_rows: Sequence[Mapping[str, Any]], successor: Mapping[str, Any] | None
) -> tuple[dict[str, Any], set[str]]:
    if successor is None:
        return {
            "provided": False,
            "newly_passed_sessions": 0,
            "newly_passed_by_stage": {"hawor": 0, "role": 0, "object": 0, "metric_ready": 0},
            "sessions": [],
        }, set()
    successor_rows = validate_matrix(successor, len(baseline_rows))
    baseline_by_id = {row["session_id"]: row for row in baseline_rows}
    successor_by_id = {row["session_id"]: row for row in successor_rows}
    if set(baseline_by_id) != set(successor_by_id):
        raise ValueError("successor matrix session set differs from pinned baseline")
    stages = Counter()
    newly_passed: set[str] = set()
    details = []
    for session_id in sorted(baseline_by_id):
        old = baseline_by_id[session_id]
        new = successor_by_id[session_id]
        changed = []
        comparisons = {
            "hawor": (grade_ab(old["hawor"]["grade"]), grade_ab(new["hawor"]["grade"])),
            "role": (grade_ab(old["role_mask"]["grade"]), grade_ab(new["role_mask"]["grade"])),
            "object": (grade_ab(old["object_mask"]["grade"]), grade_ab(new["object_mask"]["grade"])),
            "metric_ready": (bool(old["metric_ready_wave0"]), bool(new["metric_ready_wave0"])),
        }
        for stage, (was_passed, is_passed) in comparisons.items():
            if not was_passed and is_passed:
                stages[stage] += 1
                changed.append(stage)
        if changed:
            newly_passed.add(session_id)
            details.append({"session_id": session_id, "newly_passed_stages": changed})
    return {
        "provided": True,
        "newly_passed_sessions": len(newly_passed),
        "newly_passed_by_stage": {stage: stages[stage] for stage in ("hawor", "role", "object", "metric_ready")},
        "sessions": details,
    }, newly_passed


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def csv_text(records: Iterable[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def publish_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(
    *,
    matrix_path: Path,
    cohort_path: Path,
    output_root: Path,
    successor_path: Path | None = None,
    expected_sessions: int = 156,
    created_at: str | None = None,
) -> dict[str, Any]:
    matrix_path = matrix_path.resolve(strict=True)
    cohort_path = cohort_path.resolve(strict=True)
    successor_path = successor_path.resolve(strict=True) if successor_path else None
    matrix = load_object(matrix_path)
    rows = validate_matrix(matrix, expected_sessions)
    cohort_by_id = validate_cohort(load_object(cohort_path), rows, expected_sessions)
    successor = load_object(successor_path) if successor_path else None
    successor_summary, successor_sessions = successor_changes(rows, successor)

    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["position"])):
        cohort_row = cohort_by_id[row["session_id"]]
        if cohort_row.get("task") != row.get("task") or cohort_row.get("split") != row.get("split"):
            raise ValueError(f"cohort metadata mismatch for {row['session_id']}")
        date = str(cohort_row["date"])
        blocker = first_blocker(row)
        category, cause = primary_cause(row, blocker)
        eligible = bool(row["metric_ready_wave0"]) and clean_passed(row.get("clean_state"))
        candidate = eligible and robot_candidate(row.get("robot_current_state"))
        records.append(
            {
                "position": int(row["position"]),
                "session_id": row["session_id"],
                "task": row["task"],
                "date": date,
                "acquisition_contract": extract_acquisition_contract(cohort_row["raw_path"], date),
                "split": row["split"],
                "first_blocker": blocker,
                "first_blocker_category": blocker_category(blocker),
                "current_primary_cause_category": category,
                "current_primary_cause": cause,
                "hawor_grade": row["hawor"]["grade"],
                "role_grade": row["role_mask"]["grade"],
                "object_grade": row["object_mask"]["grade"],
                "metric_ready": bool(row["metric_ready_wave0"]),
                "clean_state": row.get("clean_state"),
                "robot_state": row.get("robot_current_state"),
                "robot_eligible": eligible,
                "robot_candidate": candidate,
                "successor_newly_passed": row["session_id"] in successor_sessions,
            }
        )

    blocker_counts = Counter(record["first_blocker"] for record in records)
    if sum(blocker_counts.values()) != expected_sessions or set(blocker_counts) - set(BLOCKERS):
        raise AssertionError("exclusive blocker partition failed")
    category_counts = Counter(record["current_primary_cause_category"] for record in records)
    classified_failures = sum(category_counts[name] for name in FAILURE_CATEGORIES)
    category_report = {
        "classified_failure_denominator": classified_failures,
        "all_session_denominator": expected_sessions,
        "categories": {
            name: {
                "count": category_counts[name],
                "share_of_classified_failures_percent": percent(category_counts[name], classified_failures),
                "share_of_all_sessions_percent": percent(category_counts[name], expected_sessions),
            }
            for name in FAILURE_CATEGORIES
        },
        "no_failure_current_snapshot": category_counts[NO_FAILURE],
    }

    source_refs = {
        "pinned_cross_stage_matrix": file_ref(matrix_path),
        "frozen_cohort_manifest": file_ref(cohort_path),
    }
    if successor_path:
        source_refs["successor_matrix"] = file_ref(successor_path)
    input_manifest_sha = canonical_sha(source_refs)
    producer_inputs = {
        "tool": file_ref(Path(__file__)),
        "schema": file_ref(ROOT / "contracts/exact78_conversion_cause_ledger_v2.schema.json"),
        "sources": source_refs,
        "expected_sessions": expected_sessions,
    }
    producer_signature = canonical_sha(producer_inputs)
    generated_at = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    artifact_id = f"exact78-conversion-cause-ledger-v2-{input_manifest_sha[:16]}"
    ledger = {
        "schema_version": "exact78-conversion-cause-ledger-v2",
        "artifact_id": artifact_id,
        "artifact_revision": "R7_1",
        "input_artifact_ids": [matrix.get("schema_version", "UNKNOWN"), load_object(cohort_path).get("schema_version", "UNKNOWN")],
        "input_manifest_sha": input_manifest_sha,
        "producer_signature": producer_signature,
        "supersedes_artifact_id": None,
        "validity": "VALID_FOR_PINNED_REVISION",
        "created_at": generated_at,
        "status": "PASS_DIAGNOSTIC_ACCOUNTING",
        "authority": False,
        "sources": source_refs,
        "counts": {
            "sessions": expected_sessions,
            "exclusive_first_blockers": {name: blocker_counts[name] for name in BLOCKERS},
            "robot_eligible": sum(record["robot_eligible"] for record in records),
            "robot_candidates": sum(record["robot_candidate"] for record in records),
        },
        "conditional_conversion_rates": conditional_rates(records),
        "failure_cause_attribution": category_report,
        "successor_new_passes": successor_summary,
        "breakdowns": {
            "by_task": distribution(records, ("task",)),
            "by_date": distribution(records, ("date",)),
            "by_acquisition_contract": distribution(records, ("acquisition_contract",)),
            "by_task_date_acquisition_contract": distribution(
                records, ("task", "date", "acquisition_contract")
            ),
        },
        "rows": records,
        "claim_limit": (
            "Diagnostic accounting for pinned evidence only; no automatic authority promotion. "
            "Robot candidate rate uses robot_eligible, never Raw156, as its denominator."
        ),
    }

    output_root = output_root.resolve()
    json_path = output_root / "CONVERSION_CAUSE_LEDGER_V2.json"
    csv_path = output_root / "CONVERSION_CAUSE_LEDGER_V2.csv"
    publish_immutable(json_path, json_text(ledger))
    publish_immutable(csv_path, csv_text(records))
    result = {
        "schema_version": "exact78-conversion-cause-ledger-v2-result",
        "artifact_id": f"{artifact_id}-result",
        "artifact_revision": "R7_1",
        "input_artifact_ids": [artifact_id],
        "input_manifest_sha": input_manifest_sha,
        "producer_signature": producer_signature,
        "supersedes_artifact_id": None,
        "validity": "VALID_FOR_PINNED_REVISION",
        "created_at": generated_at,
        "status": "PASSED",
        "terminal": True,
        "task_id": "exact78_conversion_cause_ledger_v2",
        "outputs": {
            "json": file_ref(json_path),
            "csv": file_ref(csv_path),
        },
        "gates": {
            "156_unique_sessions": "PASS",
            "exclusive_first_blocker": "PASS",
            "robot_eligible_denominator": "PASS",
        },
        "counts": ledger["counts"],
        "claim_limit": ledger["claim_limit"],
    }
    publish_immutable(output_root / "RESULT.json", json_text(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--successor-matrix", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int, default=156)
    parser.add_argument("--created-at")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        matrix_path=args.matrix,
        cohort_path=args.cohort,
        successor_path=args.successor_matrix,
        output_root=args.output_root,
        expected_sessions=args.expected_sessions,
        created_at=args.created_at,
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
