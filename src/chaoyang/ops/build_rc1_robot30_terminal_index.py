#!/usr/bin/env python3
"""Build a truthful Robot30 progress/terminal index from immutable evidence.

The index keeps selection, offline visual feasibility and causal training
eligibility separate.  In particular, a selected row is never counted as a
hard pass and an offline full-sequence result is never counted as RC1 causal
training input.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, now_iso


RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
SELECTION = RUN / "robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json"
V77_INDEX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/robot_v77_terminal_index_r22/attempts/attempt_0003/RESULT.json"
POKER189_V72 = RUN / "robot_gate_schema_v72/poker189/attempt_0001/RESULT.json"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _offline_results() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Return latest immutable batch row and its result reference per session."""
    output: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    candidates = sorted((RUN / "robot30_offline_visual").glob("fresh_batch_*/attempt_*/RESULT.json"))
    for path in candidates:
        payload = load(path)
        if payload.get("schema_version") != "chaoyang-rc1-robot30-offline-visual-batch-v1":
            continue
        if payload.get("input_mode") != "OFFLINE_BIDIRECTIONAL_VISUALIZATION":
            raise RuntimeError(f"unexpected input mode: {path}")
        if payload.get("training_eligible") is not False:
            raise RuntimeError(f"offline result incorrectly marked train eligible: {path}")
        for row in payload.get("rows", []):
            session = str(row.get("session", ""))
            if not session:
                continue
            output[session] = (row, artifact_ref(path))
    # Poker189 was produced before the wrapper adopted the v72 compatible
    # gate schema.  Its immutable v72 successor supersedes classification only.
    if POKER189_V72.is_file():
        payload = load(POKER189_V72)
        rows = payload.get("rows", [])
        if len(rows) != 1 or rows[0].get("session") != "play_cards_0903_189":
            raise RuntimeError("unexpected Poker189 v72 successor payload")
        row = rows[0]
        output["play_cards_0903_189"] = ({
            "session": row["session"],
            "task": row["task"],
            "terminal_status": row["terminal_status"],
            "hard_geometry_pass": row["hard_geometry_pass"],
            "strict_pose_match": row["strict_pose_match"],
            "offline_visual_only": True,
            "training_eligible": False,
            "hard_soft_audit": artifact_ref(POKER189_V72),
        }, artifact_ref(POKER189_V72))
    return output


def build_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    selection = load(SELECTION)
    selected = selection.get("rows", [])
    if len(selected) != 60 or len({row["session_id"] for row in selected}) != 60:
        raise RuntimeError("Robot30 selection must contain 60 unique rows")
    v77 = load(V77_INDEX)
    v77_rows = {str(row["session"]): row for row in v77.get("rows", [])}
    offline = _offline_results()
    rows: list[dict[str, Any]] = []
    counts = {
        task: {
            "selected": 0, "hard_geometry_pass_evidence": 0,
            "failed_quality_c": 0, "failed_runtime_final": 0,
            "pending_causal_production": 0, "pending_successor": 0,
        }
        for task in ("chips", "poker")
    }
    for selected_row in selected:
        session = str(selected_row["session_id"])
        task = str(selected_row["task"])
        route = str(selected_row["execution_route"])
        row: dict[str, Any] = {
            "session_id": session,
            "task": task,
            "target_rank": int(selected_row["target_rank"]),
            "execution_route": route,
            "terminal_status": None,
            "hard_geometry_pass": False,
            "input_mode": None,
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "result": None,
        }
        counts[task]["selected"] += 1
        if route == "ADOPT_VERIFIED_HARD_PASS":
            prior = v77_rows.get(session)
            if prior is None or prior.get("terminal_status") != "PASSED":
                raise RuntimeError(f"selection claims missing verified pass: {session}")
            row.update(
                terminal_status="PASSED_VERIFIED_PRIOR",
                hard_geometry_pass=True,
                input_mode="OFFLINE_BIDIRECTIONAL_VISUALIZATION",
                result=prior.get("result"),
            )
            counts[task]["hard_geometry_pass_evidence"] += 1
        elif route == "BOUNDED_SUCCESSOR_REQUIRED" and session in offline:
            candidate, result_ref = offline[session]
            terminal = str(candidate.get("terminal_status"))
            hard_pass = candidate.get("hard_geometry_pass") is True and terminal == "PASSED"
            row.update(
                terminal_status=("PASSED_OFFLINE_VISUAL_HARD_GEOMETRY" if hard_pass else terminal),
                hard_geometry_pass=hard_pass,
                input_mode="OFFLINE_BIDIRECTIONAL_VISUALIZATION",
                result=result_ref,
            )
            if hard_pass:
                counts[task]["hard_geometry_pass_evidence"] += 1
            elif terminal == "FAILED_QUALITY_C":
                counts[task]["failed_quality_c"] += 1
            elif terminal == "FAILED_RUNTIME_FINAL":
                counts[task]["failed_runtime_final"] += 1
            else:
                raise RuntimeError(f"unsupported successor terminal {terminal}: {session}")
        elif route == "BOUNDED_SUCCESSOR_REQUIRED":
            row["terminal_status"] = "PENDING_BOUNDED_SUCCESSOR"
            counts[task]["pending_successor"] += 1
        elif route == "FRESH_CAUSAL_PREFIX_RUN":
            row["terminal_status"] = "PENDING_CAUSAL_PRODUCTION"
            row["input_mode"] = "CAUSAL_TRAINING_INPUT_REQUIRED"
            counts[task]["pending_causal_production"] += 1
        else:
            raise RuntimeError(f"unsupported selection route {route}: {session}")
        rows.append(row)
    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    rows, counts = build_rows()
    args.output_root.mkdir(parents=True)
    json_path = args.output_root / "ROBOT30_TERMINAL_INDEX.json"
    csv_path = args.output_root / "ROBOT30_TERMINAL_INDEX.csv"
    value = {
        "schema_version": "chaoyang-rc1-robot30-terminal-index-v1",
        "created_at": now_iso(),
        "status": "PASSED_PROGRESS_SNAPSHOT",
        "counts": counts,
        "rows": rows,
        "inputs": {"selection": artifact_ref(SELECTION), "v77_terminal_index": artifact_ref(V77_INDEX)},
        "input_mode_boundary": {
            "offline_visual_hard_pass_is_training_eligible": False,
            "causal_production_required_for_training": True,
        },
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "authority_promoted": False,
        "claim_limit": "Robot30 selection/progress and digital hard-feasibility evidence only; not causal bundle, contact truth, control truth or deployment authority.",
    }
    atomic_json(json_path, value)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["session_id", "task", "target_rank", "execution_route", "terminal_status", "hard_geometry_pass", "input_mode", "training_eligible"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    result = {
        "schema_version": "chaoyang-rc1-robot30-terminal-index-result-v1",
        "created_at": now_iso(), "status": "PASSED",
        "counts": counts,
        "index": artifact_ref(json_path), "csv": artifact_ref(csv_path),
        "authority_promoted": False,
        "claim_limit": value["claim_limit"],
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
