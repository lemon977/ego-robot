"""Fixed-denominator 0915 ledger; every aggregate is derived from session rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema


DATASET_ID = "chips_cards_hands__0915_v3"
FIXED_DENOMINATOR = 220
STAGE_ORDER = (
    "Raw", "HaWoR", "Mask", "Depth", "Object6D", "Clean", "Contact",
    "RobotVisual", "RobotContactAware",
)
STATUSES = frozenset({
    "PASS", "REJECTED_QUALITY", "BLOCKED_UPSTREAM", "BLOCKED_EXTERNAL",
    "FAILED_RUNTIME", "NOT_RUN",
})
BLOCKING_STATUSES = STATUSES - {"PASS", "NOT_RUN"}
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts/full_funnel_session_ledger_v1.schema.json"
)


class FullFunnelLedgerError(ValueError):
    pass


def empty_stage() -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "reason": None,
        "input_signature": None,
        "weight_identity": None,
        "calibration_identity": None,
        "artifact": None,
    }


def initialize(sessions: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    rows = []
    for source in sessions:
        rows.append({
            "session_id": str(source["session_id"]),
            "task": str(source["task"]),
            "stages": {stage: empty_stage() for stage in STAGE_ORDER},
            "first_blocker": None,
        })
    if len(rows) != FIXED_DENOMINATOR:
        raise FullFunnelLedgerError(
            f"0915 ledger denominator {len(rows)} != {FIXED_DENOMINATOR}"
        )
    identities = {(row["task"], row["session_id"]) for row in rows}
    if len(identities) != FIXED_DENOMINATOR:
        raise FullFunnelLedgerError("duplicate 0915 task/session identity")
    ledger = {
        "schema_version": "full-funnel-session-ledger-v1",
        "dataset_id": DATASET_ID,
        "fixed_denominator": FIXED_DENOMINATOR,
        "stage_order": list(STAGE_ORDER),
        "sessions": rows,
        "summary": {},
        "counts_generated": True,
    }
    refresh(ledger)
    validate(ledger)
    return ledger


def update_stage(
    ledger: dict[str, Any], *, task: str, session_id: str, stage: str,
    status: str, reason: str | None, input_signature: str | None,
    weight_identity: str | None, calibration_identity: str | None,
    artifact: Mapping[str, Any] | None,
) -> None:
    if stage not in STAGE_ORDER:
        raise FullFunnelLedgerError(f"unknown stage: {stage}")
    if status not in STATUSES:
        raise FullFunnelLedgerError(f"unknown status: {status}")
    matches = [
        row for row in ledger["sessions"]
        if row["task"] == task and row["session_id"] == session_id
    ]
    if len(matches) != 1:
        raise FullFunnelLedgerError(f"unknown/duplicate session: {task}/{session_id}")
    matches[0]["stages"][stage] = {
        "status": status,
        "reason": reason,
        "input_signature": input_signature,
        "weight_identity": weight_identity,
        "calibration_identity": calibration_identity,
        "artifact": dict(artifact) if artifact is not None else None,
    }
    refresh(ledger)
    validate(ledger)


def refresh(ledger: dict[str, Any]) -> None:
    summary: dict[str, dict[str, int]] = {}
    for stage in STAGE_ORDER:
        counts = {status: 0 for status in sorted(STATUSES)}
        for row in ledger["sessions"]:
            counts[row["stages"][stage]["status"]] += 1
        summary[stage] = counts
    for row in ledger["sessions"]:
        row["first_blocker"] = next((
            stage for stage in STAGE_ORDER
            if row["stages"][stage]["status"] in BLOCKING_STATUSES
        ), None)
    ledger["summary"] = summary
    ledger["counts_generated"] = True


def validate(ledger: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(ledger)
    if len({(row["task"], row["session_id"]) for row in ledger["sessions"]}) != FIXED_DENOMINATOR:
        raise FullFunnelLedgerError("session identities are not unique")
    regenerated = json.loads(json.dumps(ledger))
    refresh(regenerated)
    if regenerated["summary"] != ledger["summary"]:
        raise FullFunnelLedgerError("summary is not generated from session rows")
    if any(regenerated["sessions"][index]["first_blocker"] != row["first_blocker"]
           for index, row in enumerate(ledger["sessions"])):
        raise FullFunnelLedgerError("first_blocker is not generated from stage rows")
