#!/usr/bin/env python3
"""Audit a reduced Robot base-backoff candidate set against completed sweeps.

This tool is deliberately read-only with respect to the active Robot run.  It
simulates selection from already evaluated candidates and writes one immutable
development receipt for a future revision decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_score(candidate: dict[str, Any]) -> tuple[Any, ...] | None:
    score = candidate.get("score")
    if isinstance(score, list):
        return tuple(score)
    if isinstance(score, dict):
        # Compatibility for a possible future named representation.
        names = ("infeasible", "failed_rows", "position_mm_max", "branch_margin_m", "backoff_m")
        if all(name in score for name in names):
            return tuple(score[name] for name in names)
    return None


def select_reduced(row: dict[str, Any], allowed: Iterable[float]) -> dict[str, Any] | None:
    allowed_rounded = {round(float(value), 9) for value in allowed}
    candidates = []
    for candidate in row.get("evaluated_candidates", []):
        backoff = candidate.get("backoff_m")
        score = normalized_score(candidate)
        if backoff is None or score is None or round(float(backoff), 9) not in allowed_rounded:
            continue
        candidates.append((score, float(backoff), candidate))
    if not candidates:
        return None
    score, backoff, candidate = min(candidates, key=lambda item: item[0])
    return {"backoff_m": backoff, "score": list(score), "status": candidate.get("status")}


def audit_rows(rows: list[dict[str, Any]], reduced: list[float]) -> dict[str, Any]:
    sessions = []
    for row in rows:
        simulated = select_reduced(row, reduced)
        original = float(row["best_hold_backoff_m"])
        match = simulated is not None and abs(simulated["backoff_m"] - original) <= 1e-9
        sessions.append({
            "task": row.get("task"),
            "session": row.get("session"),
            "original_best_backoff_m": original,
            "reduced_best": simulated,
            "exact_winner_match": match,
            "full_evaluated_candidate_count": len(row.get("evaluated_candidates", [])),
            "reduced_candidate_count": sum(
                round(float(candidate.get("backoff_m", -1)), 9) in {round(v, 9) for v in reduced}
                for candidate in row.get("evaluated_candidates", [])
            ),
        })
    match_count = sum(bool(row["exact_winner_match"]) for row in sessions)
    full_count = sum(row["full_evaluated_candidate_count"] for row in sessions)
    reduced_count = sum(row["reduced_candidate_count"] for row in sessions)
    tasks = sorted({str(row["task"]) for row in sessions if row.get("task")})
    return {
        "sessions": sessions,
        "counts": {
            "sessions": len(sessions),
            "exact_winner_matches": match_count,
            "full_candidate_evaluations": full_count,
            "reduced_candidate_evaluations": reduced_count,
        },
        "exact_winner_agreement": match_count / len(sessions) if sessions else 0.0,
        "candidate_evaluation_reduction_ratio": 1.0 - reduced_count / full_count if full_count else 0.0,
        "covered_tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--reduced-backoff-m", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_paths = sorted(args.robot_root.resolve(strict=True).glob("batch_*/arm_placement/RESULT.json"))
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(value.get("sessions", []))
    audit = audit_rows(rows, args.reduced_backoff_m)
    sufficient = audit["counts"]["sessions"] >= 6 and audit["exact_winner_agreement"] == 1.0
    task_general = len(audit["covered_tasks"]) >= 2
    payload = {
        "schema_version": "robot-placement-candidate-reduction-audit-v71-v1",
        "status": "PASS_DEVELOPMENT_REDUCED_SET_CANARY" if sufficient else "INSUFFICIENT_DEVELOPMENT_EVIDENCE",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reduced_backoff_m": sorted(set(args.reduced_backoff_m)),
        "audit": audit,
        "task_generality_demonstrated": task_general,
        "recommendation": (
            "Run an immutable Poker canary before adopting the reduced set in a future Robot revision."
            if sufficient and not task_general
            else "Do not change the active Robot revision from this receipt alone."
        ),
        "inputs": [ref(path) for path in result_paths],
        "code": ref(Path(__file__)),
        "authority": False,
        "claim_limit": "Retrospective selection agreement on completed sweeps only; candidate-count reduction is not measured wall-clock speedup, Robot quality authority, or evidence for unseen sessions/tasks.",
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "counts": audit["counts"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
