#!/usr/bin/env python3
"""Complete method-1 fixed-placement selection for Robot-ready canaries.

The immutable 0.26 m prior result is the first candidate.  For a HOLD row the
remaining task-level candidate values are evaluated in increasing
``(|backoff - 0.26|, backoff)`` order.  Evaluation stops at the first strict
PASS because no unevaluated candidate can then win the frozen lexicographic
objective.  If every candidate is HOLD, the best numeric HOLD is retained only
as the forward seed for the second/final bidirectional method.

Placement search is one parameter family inside arm method 1, not a sequence
of new solver methods.  This wrapper is CPU-only and publishes no Robot,
contact, action, training, or deployment authority.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ARM_TOOL = PROJECT / "tools/run_robot_motion_transfer_arm_canary_v2.py"
SELECTOR_TOOL = PROJECT / "tools/select_robot_fixed_placement_v1.py"
PRIOR_M = 0.26
ALLOWED_PRIOR_STATUSES = {
    "PASS_ARM_ROUND1_PRIOR026",
    "HOLD_ARM_ROUND1_ELIGIBLE_ROUND2",
    "HOLD_ARM_PRIOR_DETERMINISTIC_INFEASIBLE_ELIGIBLE_PLACEMENT_SWEEP",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_ref(value: dict[str, Any], label: str) -> Path:
    if set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: exact ref required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = PROJECT / path
    actual = ref(path)
    if actual["bytes"] != value["bytes"] or actual["sha256"] != value["sha256"]:
        raise ValueError(f"{label}: ref mismatch")
    return Path(actual["path"])


def select_arm_tool(preflight: dict[str, Any]) -> Path:
    """Resolve the exact forward solver registered by the immutable preflight."""
    override = preflight.get("arm_forward_tool")
    if override is not None:
        path = verify_ref(override, "arm_forward_tool")
        if preflight.get("programs", {}).get(path.name) != ref(path):
            raise ValueError("arm forward override is not in preflight program closure")
        return path
    expected = preflight.get("programs", {}).get(DEFAULT_ARM_TOOL.name)
    if expected is None or ref(DEFAULT_ARM_TOOL) != expected:
        raise ValueError("default arm tool differs from preflight closure")
    return DEFAULT_ARM_TOOL


def numeric_score(result: dict[str, Any], backoff: float) -> tuple[bool, int, float, float, float]:
    metrics = result["metrics"]
    failed = int(metrics["failed_rows"])
    excess = max(0.0, float(metrics["position_mm_max"]) / 10.0 - 1.0)
    excess += max(0.0, float(metrics["rotation_deg_max"]) / 5.0 - 1.0)
    excess += max(0.0, -float(metrics["branch_margin_m_min"]) / 1e-3)
    excess += max(0.0, float(metrics["velocity_rad_per_frame_max_contiguous"]) / 0.12 - 1.0)
    excess += max(0.0, float(metrics["acceleration_rad_per_frame2_max_contiguous"]) / 0.06 - 1.0)
    strict_pass = (
        result.get("status") == "PASS_NUMERIC_CANARY_NO_AUTHORITY"
        and failed == 0
        and excess <= 1e-8
    )
    return (not strict_pass, failed, excess, abs(backoff - PRIOR_M), backoff)


def backoff_label(value: float) -> str:
    return (f"{value:.4f}".rstrip("0").rstrip(".")).replace(".", "p")


def deterministic_bounds_failure(returncode: int, result_exists: bool, log_text: str) -> bool:
    return (
        returncode != 0
        and not result_exists
        and "TemporalReviewError: empty previous-accepted bounds:" in log_text
    )


def publish_selector_receipt(
    *,
    task: str,
    session: str,
    session_root: Path,
    candidates: list[dict[str, Any]],
    expected_backoff: float,
) -> dict[str, Any]:
    spec_path = session_root / "PLACEMENT_SELECTION_SPEC.json"
    result_path = session_root / "PLACEMENT_SELECTION_RESULT.json"
    log_path = session_root / "placement_selector.log"
    atomic_new(
        spec_path,
        {
            "schema_version": "exact78-fixed-placement-selection-spec-v52-v1",
            "task": task,
            "mode": "per_session_same_algorithm",
            "regularization_prior_m": PRIOR_M,
            "candidate_universe_note": (
                "Candidates are evaluated in increasing (distance_to_prior, backoff) order; after the first "
                "strict PASS, farther candidates cannot win and are intentionally omitted. A candidate with "
                "deterministically empty continuity bounds is non-selectable and omitted from this strict selector."
            ),
            "candidates": [
                {"backoff_m": item["backoff_m"], "result": item["result"]["path"]}
                for item in candidates
                if "result" in item
            ],
        },
    )
    command = [sys.executable, str(SELECTOR_TOOL), "--spec", str(spec_path), "--output", str(result_path)]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.flush()
        os.fsync(log.fileno())
    if process.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"{session}: frozen placement selector failed")
    selected = load(result_path)
    if selected.get("status") != "PASS_NUMERIC_PLACEMENT_SELECTED_NO_AUTHORITY":
        raise RuntimeError(f"{session}: unexpected selector status")
    actual = float(selected["selected"][session]["backoff_m"])
    if abs(actual - expected_backoff) > 1e-12:
        raise RuntimeError(f"{session}: selector chose {actual}, expected {expected_backoff}")
    return {
        "spec": ref(spec_path),
        "result": ref(result_path),
        "execution_log": ref(log_path),
        "command": command,
    }


def run_session(
    row: dict[str, Any],
    preflight: dict[str, Any],
    temporal_by_session: dict[str, Any],
    output_root: Path,
    arm_tool: Path,
) -> dict[str, Any]:
    task = row["task"]
    session = row["session"]
    if row["status"] not in ALLOWED_PRIOR_STATUSES:
        raise RuntimeError(f"{session}: unexpected prior status {row['status']}")

    candidate_key = f"{task}_candidates_m"
    universe = sorted(
        {float(x) for x in preflight["placement_policy"][candidate_key]},
        key=lambda value: (abs(value - PRIOR_M), value),
    )
    if PRIOR_M not in universe:
        raise RuntimeError(f"{session}: prior missing from candidate universe")
    deterministic_prior = (
        row["status"] == "HOLD_ARM_PRIOR_DETERMINISTIC_INFEASIBLE_ELIGIBLE_PLACEMENT_SWEEP"
    )
    if deterministic_prior:
        candidates: list[dict[str, Any]] = [
            {
                "backoff_m": PRIOR_M,
                "status": "HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS",
                "returncode": int(row["returncode"]),
                "execution_log": row["execution_log"],
                "score": [True, 2147483647, 1000000000000.0, 0.0, PRIOR_M],
                "source": "immutable_prior_round1",
                "selectable_for_method2_seed": False,
                "reason": row["reason"],
            }
        ]
    else:
        prior_result_path = verify_ref(row["result"], f"{session}:prior_result")
        verify_ref(row["states"], f"{session}:prior_states")
        prior_result = load(prior_result_path)
        if abs(float(prior_result["contract"]["task_base_backoff_m"]) - PRIOR_M) > 1e-12:
            raise RuntimeError(f"{session}: prior result is not {PRIOR_M}")
        candidates = [
            {
                "backoff_m": PRIOR_M,
                "status": prior_result["status"],
                "returncode": int(row["returncode"]),
                "result": row["result"],
                "states": row["states"],
                "metrics": prior_result["metrics"],
                "score": list(numeric_score(prior_result, PRIOR_M)),
                "source": "immutable_prior_round1",
            }
        ]
    session_root = output_root / task / session
    session_root.mkdir(parents=True)
    if not deterministic_prior and not numeric_score(prior_result, PRIOR_M)[0]:
        selector = publish_selector_receipt(
            task=task,
            session=session,
            session_root=session_root,
            candidates=candidates,
            expected_backoff=PRIOR_M,
        )
        return {
            "task": task,
            "session": session,
            "status": "PASS_ARM_METHOD1_PLACEMENT_SELECTED",
            "arm_method_rounds_consumed": 1,
            "candidate_universe_m": universe,
            "evaluated_candidates": candidates,
            "remaining_candidates_not_run": universe[1:],
            "early_stop_proof": "0.26 m prior is already the closest strict PASS",
            "selected_backoff_m": PRIOR_M,
            "selected_result": row["result"],
            "selected_states": row["states"],
            "selector": selector,
        }

    temporal_session_result_path = verify_ref(
        temporal_by_session[session]["result"], f"{session}:temporal_result"
    )
    temporal_session_result = load(temporal_session_result_path)
    hawor_path = verify_ref(temporal_session_result["outputs"]["npz"], f"{session}:temporal_npz")
    accepted_states = verify_ref(
        preflight["accepted_templates"][task]["states"], f"{task}:accepted_states"
    )
    accepted_hawor = verify_ref(
        preflight["accepted_templates"][task]["hawor"], f"{task}:accepted_hawor"
    )

    runtime_failure: dict[str, Any] | None = None
    strict_pass_found = False
    for backoff in universe:
        if abs(backoff - PRIOR_M) <= 1e-12:
            continue
        candidate_root = session_root / f"arm_method1_backoff_{backoff_label(backoff)}"
        candidate_root.mkdir(parents=True)
        result_path = candidate_root / "RESULT.json"
        log_path = candidate_root / "execution.log"
        command = [
            sys.executable,
            str(arm_tool),
            "--task",
            task,
            "--session",
            session,
            "--hawor",
            str(hawor_path),
            "--hawor-result",
            str(temporal_session_result_path),
            "--accepted-states",
            str(accepted_states),
            "--accepted-hawor",
            str(accepted_hawor),
            "--output",
            str(result_path),
            "--motion-gain",
            "1.0",
            "--task-base-backoff-m",
            str(backoff),
        ]
        started = now()
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.flush()
            os.fsync(log.fileno())
        if process.returncode not in {0, 2} or not result_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if deterministic_bounds_failure(process.returncode, result_path.exists(), log_text):
                candidates.append(
                    {
                        "backoff_m": backoff,
                        "status": "HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS",
                        "returncode": process.returncode,
                        "started_at": started,
                        "finished_at": now(),
                        "command": command,
                        "execution_log": ref(log_path),
                        "score": [True, 2147483647, 1000000000000.0, abs(backoff - PRIOR_M), backoff],
                        "source": "method1_candidate_evaluation",
                        "selectable_for_method2_seed": False,
                        "reason": (
                            "hard joint, velocity, and acceleration bounds have an empty intersection; "
                            "the deterministic candidate cannot be a strict PASS"
                        ),
                    }
                )
                continue
            runtime_failure = {
                "backoff_m": backoff,
                "started_at": started,
                "finished_at": now(),
                "returncode": process.returncode,
                "command": command,
                "execution_log": ref(log_path),
                "error": "candidate command failed without a valid bounded result",
            }
            break
        result = load(result_path)
        if result.get("task") != task or result.get("session") != session:
            raise RuntimeError(f"{session}: candidate identity mismatch")
        if abs(float(result["contract"]["task_base_backoff_m"]) - backoff) > 1e-12:
            raise RuntimeError(f"{session}: candidate backoff mismatch")
        states_path = verify_ref(result["output_states"], f"{session}:{backoff}:states")
        score = numeric_score(result, backoff)
        if process.returncode == 0 and not score[0]:
            classification = "PASS_NUMERIC_CANARY_NO_AUTHORITY"
        elif process.returncode == 2 and result.get("status") == "HOLD_NUMERIC_CANARY" and score[0]:
            classification = "HOLD_NUMERIC_CANARY"
        else:
            raise RuntimeError(f"{session}: candidate result/returncode mismatch")
        candidates.append(
            {
                "backoff_m": backoff,
                "status": classification,
                "returncode": process.returncode,
                "started_at": started,
                "finished_at": now(),
                "command": command,
                "execution_log": ref(log_path),
                "result": ref(result_path),
                "states": ref(states_path),
                "metrics": result["metrics"],
                "score": list(score),
                "source": "method1_candidate_evaluation",
            }
        )
        if not score[0]:
            strict_pass_found = True
            break

    if runtime_failure is not None:
        return {
            "task": task,
            "session": session,
            "status": "FAILED_RUNTIME_RETRYABLE",
            "arm_method_rounds_consumed": 1,
            "candidate_universe_m": universe,
            "evaluated_candidates": candidates,
            "runtime_failure": runtime_failure,
        }

    selectable = [item for item in candidates if "result" in item and "states" in item]
    if not selectable:
        raise RuntimeError(f"{session}: no complete candidate remains selectable")
    best = min(selectable, key=lambda item: tuple(item["score"]))
    evaluated_backoffs = {float(item["backoff_m"]) for item in candidates}
    remaining = [value for value in universe if value not in evaluated_backoffs]
    if strict_pass_found:
        if tuple(best["score"])[0]:
            raise RuntimeError(f"{session}: strict pass found but selection is HOLD")
        selector = publish_selector_receipt(
            task=task,
            session=session,
            session_root=session_root,
            candidates=candidates,
            expected_backoff=float(best["backoff_m"]),
        )
        return {
            "task": task,
            "session": session,
            "status": "PASS_ARM_METHOD1_PLACEMENT_SELECTED",
            "arm_method_rounds_consumed": 1,
            "candidate_universe_m": universe,
            "evaluated_candidates": candidates,
            "remaining_candidates_not_run": remaining,
            "early_stop_proof": (
                "first strict PASS in increasing (distance_to_prior, backoff) order; all remaining candidates "
                "are farther from the frozen prior and cannot win the lexicographic objective"
            ),
            "selected_backoff_m": best["backoff_m"],
            "selected_result": best["result"],
            "selected_states": best["states"],
            "selector": selector,
        }
    if remaining:
        raise RuntimeError(f"{session}: no PASS but candidate universe was not exhausted")
    return {
        "task": task,
        "session": session,
        "status": "HOLD_ARM_METHOD1_ALL_PLACEMENTS_ELIGIBLE_METHOD2",
        "arm_method_rounds_consumed": 1,
        "candidate_universe_m": universe,
        "evaluated_candidates": candidates,
        "remaining_candidates_not_run": [],
        "best_hold_backoff_m": best["backoff_m"],
        "method2_forward_seed_result": best["result"],
        "method2_forward_seed_states": best["states"],
        "selection_note": "all frozen placement candidates were HOLD; best score is only the method-2 seed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--round1-prior-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")
    if not 1 <= args.max_workers <= 3:
        raise SystemExit("--max-workers must be in [1,3]")

    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    try:
        arm_tool = select_arm_tool(preflight)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    expected_selector = preflight.get("programs", {}).get(SELECTOR_TOOL.name)
    if expected_selector is None or ref(SELECTOR_TOOL) != expected_selector:
        raise SystemExit(f"{SELECTOR_TOOL.name} differs from preflight closure")

    round1 = load(args.round1_prior_result)
    if round1.get("status") != "PASS_BOUNDED_ROUND1_RESULTS_PUBLISHED":
        raise SystemExit("complete round-1 prior aggregate required")
    if round1.get("preflight") != ref(args.preflight):
        raise SystemExit("round-1/preflight closure mismatch")
    temporal_root_path = verify_ref(round1["temporal_root_result"], "temporal_root_result")
    temporal_root = load(temporal_root_path)
    temporal_by_session = {row["session"]: row for row in temporal_root.get("sessions", [])}
    round1_rows = round1.get("sessions", [])
    if set(temporal_by_session) != {row["session"] for row in round1_rows}:
        raise SystemExit("round-1/temporal session set mismatch")

    args.output_root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_session, row, preflight, temporal_by_session, args.output_root, arm_tool): row["session"]
            for row in round1_rows
        }
        by_session: dict[str, dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(futures):
            session = futures[future]
            by_session[session] = future.result()
    rows = [by_session[row["session"]] for row in round1_rows]
    counts = {
        "method1_pass": sum(row["status"] == "PASS_ARM_METHOD1_PLACEMENT_SELECTED" for row in rows),
        "method1_hold_all_placements": sum(
            row["status"] == "HOLD_ARM_METHOD1_ALL_PLACEMENTS_ELIGIBLE_METHOD2" for row in rows
        ),
        "runtime_failed_retryable": sum(row["status"] == "FAILED_RUNTIME_RETRYABLE" for row in rows),
    }
    status = (
        "PASS_METHOD1_PLACEMENT_EVALUATION_PUBLISHED"
        if counts["runtime_failed_retryable"] == 0
        else "FAILED_RUNTIME_RETRYABLE"
    )
    aggregate = {
        "schema_version": "exact78-robot-arm-placement-sweep-v52-v1",
        "created_at": now(),
        "status": status,
        "preflight": ref(args.preflight),
        "round1_prior_result": ref(args.round1_prior_result),
        "arm_tool": ref(arm_tool),
        "selector_tool": ref(SELECTOR_TOOL),
        "placement_policy": preflight["placement_policy"],
        "parallel_workers": args.max_workers,
        "counts": counts,
        "sessions": rows,
        "gpu_calls": 0,
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "Bounded method-1 arm placement selection only. Exhaustive HOLD rows may consume the second/final "
            "bidirectional method. No hand, render, contact, action, training, deployment, or Robot authority."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(json.dumps({"status": status, "counts": counts, "result": ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0 if counts["runtime_failed_retryable"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
