#!/usr/bin/env python3
"""Fail-closed R2.2 V78 target/reach canary preflight.

This tool never runs a Robot optimizer.  It verifies that the frozen three-way
selection and the required framewise reach evidence exist.  If either is
missing it publishes an immutable BLOCKED_PREREQ six-file task result instead
of inventing a replacement session or an unregistered algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


TASK_ID = "robot_v78_target_reach_canary_r22"
REQUIRED_REACH_FIELDS = (
    "action_amplitude",
    "wrist_object_approach",
    "ik_frame_valid_rate",
)


class ContractError(RuntimeError):
    pass


def load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_new(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def evaluate(
    selection: dict[str, Any], terminal_index: dict[str, Any], robot_yield: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = terminal_index.get("rows", [])
    by_session = {row.get("session"): row for row in rows}
    selected = {
        "failure_canary": selection.get("v78_failure_canary"),
        "chips_hard_pass_regression": selection.get("v78_chips_regression"),
        "poker_hard_pass_regression": selection.get("v78_poker_regression"),
    }
    blockers: list[dict[str, str]] = []
    expected_tasks = {
        "failure_canary": "poker",
        "chips_hard_pass_regression": "chips",
        "poker_hard_pass_regression": "poker",
    }
    for role, session in selected.items():
        if not session:
            blockers.append({"code": "FROZEN_SELECTION_MISSING", "detail": role})
            continue
        row = by_session.get(session)
        if row is None:
            blockers.append({"code": "SESSION_NOT_IN_FROZEN_TERMINAL_INDEX", "detail": session})
            continue
        if row.get("task") != expected_tasks[role]:
            blockers.append({"code": "TASK_IDENTITY_MISMATCH", "detail": session})
        if role.endswith("hard_pass_regression") and row.get("terminal_status") != "PASSED":
            blockers.append({"code": "REGRESSION_NOT_HARD_PASS", "detail": session})
        if role == "failure_canary" and row.get("terminal_status") != "FAILED_QUALITY_C":
            blockers.append({"code": "FAILURE_CANARY_NOT_QUALITY_C", "detail": session})

    missing_profile_fields = set(robot_yield.get("missing_profile_fields", []))
    for field in REQUIRED_REACH_FIELDS:
        if field in missing_profile_fields:
            blockers.append({"code": "FRAMEWISE_REACH_FIELD_MISSING", "detail": field})

    return selected, blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--terminal-index", type=Path, required=True)
    parser.add_argument("--robot-yield", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet_path = args.task_packet.resolve(strict=True)
    selection_path = args.selection.resolve(strict=True)
    terminal_path = args.terminal_index.resolve(strict=True)
    yield_path = args.robot_yield.resolve(strict=True)
    packet = load_object(packet_path, "Task Packet")
    if packet.get("task_id") != TASK_ID:
        raise ContractError(f"Task Packet must be {TASK_ID}")
    output = args.output.resolve()
    writes = [Path(value).resolve() for value in packet.get("write_set", [])]
    if not any(output.is_relative_to(root) for root in writes):
        raise ContractError("output is outside Task Packet write_set")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    selection = load_object(selection_path, "frozen integration selection")
    terminal = load_object(terminal_path, "v77 terminal index")
    robot_yield = load_object(yield_path, "Robot Yield 20")
    if terminal.get("status") != "PASSED_TERMINAL_COVERAGE" or terminal.get("counts", {}).get("missing") != 0:
        raise ContractError("v77 frozen terminal index is not complete")
    selected, blockers = evaluate(selection, terminal, robot_yield)
    if not blockers:
        raise ContractError("preflight is READY; this fail-closed finalizer must not run computation")

    output.mkdir(parents=True)
    inputs = {
        "task_packet": artifact_ref(packet_path),
        "frozen_selection": artifact_ref(selection_path),
        "v77_terminal_index": artifact_ref(terminal_path),
        "robot_yield_20": artifact_ref(yield_path),
    }
    result = {
        "schema_version": "robot-v78-target-reach-preflight-result-v1",
        "terminal_status": "BLOCKED_PREREQ",
        "status": "BLOCKED_MISSING_FROZEN_REGRESSION_AND_REACH_IMPLEMENTATION_INPUTS",
        "selection": selected,
        "blockers": blockers,
        "causal_mode_required": "CAUSAL_TRAINING",
        "causal_contract": {
            "placement_sources_allowed": ["TASK_PRESET", "START_PREFIX"],
            "full_sequence_optimization_allowed": False,
        },
        "unchanged_hard_gates": ["finite", "collision", "joint_limits", "coordinate_chain"],
        "computation_started": False,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "inputs": inputs,
        "claim_limit": "Preflight and implementation-gap evidence only; no V78 Robot result or authority.",
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(output / "METRICS.json", {
        "schema_version": "robot-v78-target-reach-preflight-metrics-v1",
        "blocker_count": len(blockers),
        "selected_session_count": sum(value is not None for value in selected.values()),
        "required_session_count": 3,
        "required_framewise_reach_fields": list(REQUIRED_REACH_FIELDS),
        "missing_framewise_reach_fields": [
            row["detail"] for row in blockers if row["code"] == "FRAMEWISE_REACH_FIELD_MISSING"
        ],
    })
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "robot-v78-target-reach-preflight-receipt-v1",
        "result": artifact_ref(result_path),
        "inputs": inputs,
    })
    atomic_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "artifact-manifest-v1",
        "artifacts": [artifact_ref(result_path), *inputs.values()],
    })
    atomic_new(output / "DECISION.md", (
        "# Decision\n\nV78 target/reach canary is BLOCKED_PREREQ. The frozen R2.2 selection "
        "does not contain a Poker hard-pass regression, and the pinned v77 evidence does not "
        "contain the required framewise reach fields. No replacement session or optimizer was "
        "invented, and no hard gate was relaxed.\n"
    ).encode())
    atomic_json(output / "NEXT_ACTION.json", {
        "next_task_id": "robot_v78_target_reach_implementation_and_selection_revision",
        "status": "BLOCKED_PREREQ",
        "start_computation": False,
        "requirements": [
            "CAS-publish a new frozen selection containing a verified Poker hard-pass regression",
            "Register a causal V78 implementation that emits framewise action amplitude, wrist-object approach and IK validity",
            "Preserve finite, collision, joint-limit and coordinate-chain hard gates",
        ],
    })
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
