#!/usr/bin/env python3
"""Fail-closed Robot preflight for one to three newly Clean-joined sessions.

This is the bounded-batch successor to the original ready-three preflight.  It
changes only batch cardinality; artifact, matrix, code, template, and claim
checks are delegated to the frozen v5.2 preflight implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import preflight_exact78_robot_ready_v52 as base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = base.load(args.contract)
    matrix = base.load(args.matrix)
    if contract.get("schema_version") != "exact78-robot-ready-input-v52-v1":
        raise SystemExit("unsupported contract schema")
    if contract.get("status") != "FROZEN_PREFLIGHT_PENDING":
        raise SystemExit("contract is not frozen for preflight")
    matrix_ref = base.require_ref(contract["matrix_snapshot"], "matrix_snapshot")
    if matrix_ref != base.ref(args.matrix):
        raise SystemExit("--matrix does not match contract matrix_snapshot")
    sessions = contract.get("sessions", [])
    if not 1 <= len(sessions) <= 3 or len({row.get("session") for row in sessions}) != len(sessions):
        raise SystemExit("contract must contain one to three unique sessions")
    matrix_by_session = {row["session_id"]: row for row in matrix.get("rows", [])}
    rows = [base.validate_session(row, matrix_by_session.get(row.get("session"), {})) for row in sessions]
    templates = {
        task: {
            name: base.require_ref(value, f"accepted_templates:{task}:{name}")
            for name, value in group.items()
        }
        for task, group in contract.get("accepted_templates", {}).items()
    }
    if set(templates) != {"chips", "poker"} or any(
        set(group) != {"states", "hawor"} for group in templates.values()
    ):
        raise SystemExit("both exact accepted template pairs are required")
    programs = {name: base.ref(PROJECT / "tools" / name) for name in base.REQUIRED_PROGRAMS}
    counts = {
        "sessions": len(rows),
        "chips": sum(row["task"] == "chips" for row in rows),
        "poker": sum(row["task"] == "poker" for row in rows),
    }
    result = {
        "schema_version": "exact78-robot-ready-batch-preflight-v52-v1",
        "created_at": base.now(),
        "status": "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT",
        "preflight_program": base.ref(Path(__file__)),
        "base_preflight_program": base.ref(Path(base.__file__)),
        "contract": base.ref(args.contract),
        "matrix_snapshot": matrix_ref,
        "counts": counts,
        "sessions": rows,
        "accepted_templates": templates,
        "placement_policy": contract["placement_policy"],
        "method_budget": contract["method_budget"],
        "programs": programs,
        "gpu_calls": 0,
        "claim_limit": (
            "Input and code closure for one-to-three Clean-joined sessions only; no IK/render was run, no "
            "method round was consumed, and no Robot/contact/action/deployment authority is published."
        ),
    }
    base.atomic_new(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "counts": counts, "output": str(args.output), "sha256": base.sha256(args.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
