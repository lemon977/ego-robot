"""Exact78 V3.2 frozen-cohort and paired-production contracts.

The functions in this module are CPU-only and side-effect free.  They keep the
historical 156 identities frozen while allowing current-path rebinding to fail
closed.  A missing member remains unresolved; a newly discovered session can
never replace it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


COHORT_SCHEMA = "EXACT78_COHORT_SPLIT_V32"
PAIR_LEDGER_SCHEMA = "EXACT78_PAIR_LEDGER_V32"
RAW_ONLY_SCHEMA = "EXACT78_RAW_ONLY_DEV_V1"
PAIR_CONTRACT = "EXACT78_V32_REAL_CAUSAL_PAIR"
SUFFIX_SCHEMA = "EXACT78_SUFFIX_INVARIANCE_V32"

EXPECTED_SPLIT_COUNTS = {
    "train": 59,
    "validation": 77,
    "development_final": 20,
}
EXPECTED_SOURCE_GROUP_COUNTS = {
    "train": 2,
    "validation": 1,
    "development_final": 1,
}
EXPECTED_SPLIT_DATES = {
    "train": "0901",
    "validation": "0902",
    "development_final": "0903",
}
REQUIRED_SUFFIX_FIELDS = {
    "human_raw_rgb",
    "robotized_rgb",
    "crop",
    "state",
    "confidence",
}
PRODUCTION_STAGES = (
    "HISTORICAL_IDENTITY_REBIND",
    "CURRENT_SOURCE_LINEAGE_SHA",
    "UPSTREAM_READINESS",
    "RAW_ROBOTIZED_PIXEL_LABEL_PRODUCTION",
    "SUFFIX_INVARIANCE_INDEPENDENT_REPLAY",
    "PAIR_LEDGER_VALIDATION",
)


class ContractError(RuntimeError):
    """Raised when a V3.2 frozen contract fails closed."""


def _session_task(session_id: str) -> str:
    if session_id.startswith("get_potato_chips_"):
        return "chips"
    if session_id.startswith("play_cards_"):
        return "poker"
    raise ContractError(f"unrecognized Exact78 identity: {session_id}")


def validate_cohort_split(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate and expand the compact frozen 156-member split manifest."""

    if payload.get("schema_version") != COHORT_SCHEMA:
        raise ContractError("Exact78 V3.2 cohort schema mismatch")
    if payload.get("replacement_policy") != "FORBIDDEN_KEEP_UNRESOLVED":
        raise ContractError("Exact78 V3.2 replacement policy must be fail-closed")
    if payload.get("development_final_policy") != "EXPOSED_NOT_FIRST_BLIND_TEST":
        raise ContractError("0903 must be declared exposed development final")
    historical = payload.get("historical_identity_authority")
    if not isinstance(historical, dict) or not isinstance(
        historical.get("sha256"), str
    ) or len(historical["sha256"]) != 64:
        raise ContractError("historical 156-member authority SHA is required")
    groups = payload.get("source_groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ContractError("Exact78 V3.2 requires exactly four source groups")

    index: dict[str, dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ContractError("source-group row must be an object")
        group_id = str(group.get("source_group_id", ""))
        split = str(group.get("split", ""))
        date = str(group.get("capture_date", ""))
        members = group.get("members")
        if (
            not group_id
            or group_id in group_ids
            or split not in EXPECTED_SPLIT_COUNTS
            or date != EXPECTED_SPLIT_DATES[split]
            or not isinstance(members, list)
            or not members
        ):
            raise ContractError(f"invalid source group: {group_id or '<missing>'}")
        group_ids.add(group_id)
        group_counts[split] += 1
        for member in members:
            if not isinstance(member, str) or member in index:
                raise ContractError(f"invalid/duplicate frozen identity: {member}")
            task = _session_task(member)
            marker = f"_{date}_"
            if marker not in member:
                raise ContractError(f"identity/date mismatch: {member} vs {date}")
            index[member] = {
                "session_id": member,
                "task": task,
                "capture_date": date,
                "split": split,
                "source_group_id": group_id,
                "current_rebind_status": "UNRESOLVED_UNTIL_BUILDER_VALIDATION",
            }
            split_counts[split] += 1
            task_counts[task] += 1

    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ContractError(
            f"Exact78 V3.2 split counts mismatch: {dict(split_counts)}"
        )
    if dict(group_counts) != EXPECTED_SOURCE_GROUP_COUNTS:
        raise ContractError(
            f"Exact78 V3.2 source-group counts mismatch: {dict(group_counts)}"
        )
    if task_counts != Counter({"chips": 78, "poker": 78}):
        raise ContractError(f"Exact78 task counts mismatch: {dict(task_counts)}")
    declared = payload.get("counts")
    expected_declared = {
        "members": 156,
        "tasks": {"chips": 78, "poker": 78},
        "splits": EXPECTED_SPLIT_COUNTS,
        "source_groups_by_split": EXPECTED_SOURCE_GROUP_COUNTS,
    }
    if declared != expected_declared:
        raise ContractError("declared cohort counts do not match expanded identities")
    return index


def validate_suffix_receipt(payload: Mapping[str, Any]) -> None:
    """Require two real, cache-isolated producer executions."""

    if (
        payload.get("schema_version") != SUFFIX_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("causal_current_inputs_authorized") is not True
        or payload.get("independent_replay") is not True
    ):
        raise ContractError("V3.2 suffix-invariance replay did not pass")
    runs = payload.get("production_runs")
    if not isinstance(runs, dict) or set(runs) != {"full", "prefix"}:
        raise ContractError("full and prefix production runs are required")
    cache_roots: set[str] = set()
    run_ids: set[str] = set()
    for name, run in runs.items():
        if not isinstance(run, dict) or run.get("actual_production") is not True:
            raise ContractError(f"{name} replay is not an actual production run")
        cache_root = str(run.get("cache_root", ""))
        run_id = str(run.get("run_id", ""))
        if not cache_root or not Path(cache_root).is_absolute() or not run_id:
            raise ContractError(f"{name} replay identity/cache root missing")
        cache_roots.add(cache_root)
        run_ids.add(run_id)
        for key in ("producer_sha256", "config_sha256", "input_sha256"):
            value = run.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ContractError(f"{name} replay {key} missing")
    if len(cache_roots) != 2 or len(run_ids) != 2:
        raise ContractError("suffix replay must use distinct run IDs and cache roots")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not REQUIRED_SUFFIX_FIELDS.issubset(fields):
        raise ContractError("suffix replay required fields missing")
    for name in REQUIRED_SUFFIX_FIELDS:
        row = fields[name]
        if (
            not isinstance(row, dict)
            or row.get("suffix_invariant") is not True
            or row.get("effective_temporal_authority") != "CAUSAL_CURRENT"
        ):
            raise ContractError(f"noncausal current input: {name}")


def validate_pair_production_readiness(
    payload: Mapping[str, Any],
    *,
    raw_frame_refs: Sequence[Mapping[str, Any]],
    robotized_frame_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the pair contains two real pixel branches, not renamed copies."""

    readiness = payload.get("production_readiness")
    if not isinstance(readiness, dict):
        raise ContractError("V3.2 production readiness is required")
    stages = readiness.get("stages")
    if not isinstance(stages, list) or [row.get("stage") for row in stages] != list(
        PRODUCTION_STAGES[:-1]
    ):
        raise ContractError("V3.2 production stages are missing or out of order")
    for row in stages:
        evidence_sha = row.get("evidence_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("status") != "PASS"
            or not isinstance(evidence_sha, str)
            or len(evidence_sha) != 64
        ):
            raise ContractError(f"production stage is not ready: {row}")
    if readiness.get("next_stage") != "PAIR_LEDGER_VALIDATION":
        raise ContractError("pair must enter the current ledger validator next")
    producer = payload.get("robotized_producer")
    if (
        not isinstance(producer, dict)
        or producer.get("producer_kind") != "CAUSAL_PIXEL_COMPOSITOR"
        or producer.get("actual_production") is not True
        or producer.get("copy_or_rename_only") is not False
    ):
        raise ContractError("Robotized pixels lack a real causal producer receipt")
    for key in ("producer_sha256", "config_sha256", "input_sha256"):
        value = producer.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"Robotized producer {key} missing")

    if not raw_frame_refs or len(raw_frame_refs) != len(robotized_frame_refs):
        raise ContractError("Raw/Robotized frame sets differ or are empty")
    differing = 0
    for index, (raw, robotized) in enumerate(
        zip(raw_frame_refs, robotized_frame_refs, strict=True)
    ):
        raw_path = str(raw.get("path", ""))
        robotized_path = str(robotized.get("path", ""))
        raw_sha = str(raw.get("sha256", ""))
        robotized_sha = str(robotized.get("sha256", ""))
        if not raw_path or not robotized_path or not raw_sha or not robotized_sha:
            raise ContractError(f"frame {index} lacks byte-bound RGB references")
        if Path(raw_path).resolve(strict=False) == Path(robotized_path).resolve(
            strict=False
        ):
            raise ContractError(f"frame {index} reuses one RGB path for both branches")
        differing += int(raw_sha != robotized_sha)
    if differing == 0:
        raise ContractError("Robotized branch is a byte-identical renamed Raw copy")
    return {
        "raw_frames": len(raw_frame_refs),
        "robotized_frames": len(robotized_frame_refs),
        "byte_different_frame_count": differing,
        "robotized_is_copy_or_rename": False,
        "production_chain": [
            *stages,
            {
                "stage": "PAIR_LEDGER_VALIDATION",
                "status": "PASS_BY_CURRENT_V32_BUILDER",
            },
        ],
    }


def validate_raw_only_contract(payload: Mapping[str, Any]) -> None:
    """Keep emergency Raw-only work explicitly outside the four-model A/B."""

    required = {
        "schema_version": RAW_ONLY_SCHEMA,
        "status": "DEVELOPMENT_ONLY",
        "branch": "HUMAN_RAW_RGB",
        "formal_ab_allowed": False,
        "four_model_completion": False,
        "robotized_comparator": "ABSENT",
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ContractError(f"Raw-only contract mismatch: {key}")
    if payload.get("cohort_split_schema") != COHORT_SCHEMA:
        raise ContractError("Raw-only run must remain bound to the V3.2 cohort")
