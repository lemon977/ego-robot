#!/usr/bin/env python3
"""Deterministically select a fixed full-trajectory placement from numeric results.

The selector never branches on a session identifier. Candidate generation and
IK evaluation are external and expensive; this closure tool verifies immutable
candidate results and applies one common lexicographic objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score(result: dict, backoff: float, prior: float) -> tuple:
    m = result["metrics"]
    failed = int(m["failed_rows"])
    excess = max(0.0, float(m["position_mm_max"]) / 10.0 - 1.0)
    excess += max(0.0, float(m["rotation_deg_max"]) / 5.0 - 1.0)
    excess += max(0.0, -float(m["branch_margin_m_min"]) / 1e-3)
    excess += max(0.0, float(m["velocity_rad_per_frame_max_contiguous"]) / 0.12 - 1.0)
    excess += max(0.0, float(m["acceleration_rad_per_frame2_max_contiguous"]) / 0.06 - 1.0)
    strict_pass = result["status"] == "PASS_NUMERIC_CANARY_NO_AUTHORITY" and failed == 0 and excess <= 1e-8
    return (not strict_pass, failed, excess, abs(backoff - prior), backoff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    spec = json.load(args.spec.open())
    prior = float(spec.get("regularization_prior_m", 0.26))
    candidates = []
    for item in spec["candidates"]:
        path = Path(item["result"])
        path.resolve(strict=True)
        result = json.load(path.open())
        backoff = float(item["backoff_m"])
        candidates.append({"backoff_m": backoff, "result": str(path.resolve()),
                           "result_sha256": sha(path), "status": result["status"],
                           "metrics": result["metrics"], "score": score(result, backoff, prior)})
    # For task-shared mode, a backoff is admissible only if it covers every
    # session. For per-session mode, each session is selected independently.
    selected = {}
    if spec["mode"] == "task_shared":
        sessions = sorted({json.load(Path(c["result"]).open())["session"] for c in candidates})
        by_backoff = {}
        for c in candidates:
            sid = json.load(Path(c["result"]).open())["session"]
            by_backoff.setdefault(c["backoff_m"], {})[sid] = c
        aggregate = []
        for backoff, group in by_backoff.items():
            if sorted(group) != sessions:
                continue
            scores = [group[s]["score"] for s in sessions]
            aggregate.append((max(x[0] for x in scores), sum(x[1] for x in scores),
                              max(x[2] for x in scores), abs(backoff - prior), backoff, group))
        if not aggregate:
            raise RuntimeError("no task-shared candidate covers every session")
        *_, backoff, group = min(aggregate, key=lambda x: x[:5])
        if any(group[s]["score"][0] for s in sessions):
            raise RuntimeError("no task-shared strict PASS placement")
        selected = {s: group[s] for s in sessions}
    elif spec["mode"] == "per_session_same_algorithm":
        groups = {}
        for c in candidates:
            sid = json.load(Path(c["result"]).open())["session"]
            groups.setdefault(sid, []).append(c)
        for sid, group in groups.items():
            choice = min(group, key=lambda x: tuple(x["score"]))
            if choice["score"][0]:
                raise RuntimeError(f"no strict PASS candidate for {sid}")
            selected[sid] = choice
    else:
        raise RuntimeError("mode must be task_shared or per_session_same_algorithm")
    payload = {
        "schema_version": "robot-fixed-fulltrajectory-placement-selection-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_NUMERIC_PLACEMENT_SELECTED_NO_AUTHORITY",
        "task": spec["task"], "mode": spec["mode"],
        "algorithm": {
            "candidate_transform": "one fixed world-base backoff transform for the complete session; target hand roots are unchanged",
            "objective": "lexicographic(strict-pass, failed rows, normalized gate excess, |backoff-prior|, backoff)",
            "strict_gates": "10mm/5deg/branch>=0; arm velocity<=0.12 and acceleration<=0.06; unchanged",
            "identity_independence": "the selector reads metrics and numeric backoff only; no session-name rule",
            "regularization_prior_m": prior,
        },
        "spec": {"path": str(args.spec.resolve()), "sha256": sha(args.spec)},
        "candidates": candidates, "selected": selected,
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "Numeric fixed-placement selection for review; not Robot/action/contact authority or human visual approval.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "selected": {k: v["backoff_m"] for k, v in selected.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
