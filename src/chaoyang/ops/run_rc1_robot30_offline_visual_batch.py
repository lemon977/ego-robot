#!/usr/bin/env python3
"""Run pinned v77 full-session Robot batches for offline visual review.

This deliberately does not claim RC1 causal-training eligibility.  It reuses
the immutable v77 phase DAG to produce session-level Robot geometry, digital
collision evidence and review media for the user's 30+30 visual target.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from chaoyang.ops import run_exact78_robot_microbatches_v77 as v77


ROOT = Path(__file__).resolve().parents[3]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha(value)}


def atomic_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", required=True, help="Comma-separated frozen session ids")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--contract-mode",
        choices=("fresh", "allow-existing"),
        default="fresh",
        help="fresh lets the pinned phase DAG create its contract and phase receipt; allow-existing is only for an explicitly proven legacy-contract successor",
    )
    args = parser.parse_args()
    sessions = [value.strip() for value in args.sessions.split(",") if value.strip()]
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("non-empty unique sessions required")
    selection = json.loads(args.selection.resolve(strict=True).read_text(encoding="utf-8"))
    selected = {row["session_id"]: row for row in selection["rows"]}
    missing = [session for session in sessions if session not in selected]
    if missing:
        raise RuntimeError(f"sessions outside frozen Robot30 selection: {missing}")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    legacy = v77.legacy
    legacy.heartbeat = lambda *_args, **_kwargs: None
    legacy.RUN_ROOT = output
    legacy.TASK_ID = "rc1_robot30_offline_visual_batch"
    legacy.PHASE_TIMEOUT_SECONDS = args.phase_timeout_seconds
    legacy.POLL_SECONDS = 25.0
    legacy.POSE_ONLY_ROOT = output / "legacy_final/pose_only"
    legacy.TERMINAL_BASE = output / "legacy_final/quality_c"
    rows = []
    started = time.monotonic()
    for session in sessions:
        task = str(selected[session]["task"])
        work = output / "work_sessions" / session
        session_started = time.monotonic()
        try:
            if args.contract_mode == "allow-existing":
                # This mode remains fail-closed unless a separate successor
                # supplies a complete legacy-contract proof.  Merely passing
                # the flag is not enough to bypass v77 phase receipts.
                raise RuntimeError(
                    f"{session}: allow-existing requires a dedicated immutable adoption successor"
                )
            # For a current-READY session the pinned v77 phase DAG must create
            # both ROBOT_READY_INPUT.json and its phase-success receipt.  Do
            # not pre-create the contract: v53 deliberately rejects an output
            # without the exact command-bound receipt.
            v77._run_phase_dag_without_legacy_finalizer(work, [session], args.matrix.resolve(strict=True))
            audit_root = output / "hard_soft/sessions" / session
            # Produce the pinned v7.1 audit/collision evidence first, then
            # publish a schema-compatible v7.2 successor.  The successor only
            # resolves verified v77 arm gate aliases; it does not change any
            # solver state, threshold, collision or joint-limit test.
            v77.audit_batch(work, audit_root, output / "visual_candidates_v71" / session)
            audit_result = audit_root / "RESULT_V72.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chaoyang.ops.audit_robot_hard_soft_gate_v72",
                    "--batch-root",
                    str(work),
                    "--collision-root",
                    str(audit_root),
                    "--output",
                    str(audit_result),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            audit_log = audit_root / "logs/hard_soft_v72.log"
            audit_log.parent.mkdir(parents=True, exist_ok=True)
            audit_log.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or not audit_result.is_file():
                raise RuntimeError(f"{session}: v72 hard/soft audit failed; see {audit_log}")
            audit = json.loads(audit_result.read_text(encoding="utf-8"))
            audit_rows = [row for row in audit.get("rows", []) if row.get("session") == session]
            if len(audit_rows) != 1:
                raise RuntimeError(f"{session}: expected one hard/soft row, got {len(audit_rows)}")
            row = audit_rows[0]
            terminal = "PASSED" if row.get("hard_geometry_pass") is True else "FAILED_QUALITY_C"
            rows.append({
                "session": session, "task": task, "terminal_status": terminal,
                "hard_geometry_pass": row.get("hard_geometry_pass"),
                "strict_pose_match": row.get("strict_pose_match"),
                "offline_visual_only": True, "training_eligible": False,
                "wall_seconds": time.monotonic() - session_started,
                "hard_soft_audit": ref(audit_result),
                "candidate_result": row.get("evidence", {}).get("review_result"),
            })
        except Exception as error:
            failure = output / "failures" / session / "RESULT.json"
            atomic_json(failure, {
                "schema_version": "chaoyang-rc1-robot30-offline-runtime-failure-v1",
                "session": session, "task": task, "status": "FAILED_RUNTIME_FINAL",
                "created_at": now(), "error_type": type(error).__name__, "error": str(error),
                "partial_output_present": work.exists(),
                "claim_limit": "Runtime failure receipt only; partial output is not a terminal Robot result.",
            })
            rows.append({"session": session, "task": task, "terminal_status": "FAILED_RUNTIME_FINAL", "wall_seconds": time.monotonic() - session_started, "failure": ref(failure), "training_eligible": False})
    counts = {
        "sessions": len(rows),
        "hard_geometry_pass": sum(row["terminal_status"] == "PASSED" for row in rows),
        "failed_quality_c": sum(row["terminal_status"] == "FAILED_QUALITY_C" for row in rows),
        "failed_runtime_final": sum(row["terminal_status"] == "FAILED_RUNTIME_FINAL" for row in rows),
    }
    result = {
        "schema_version": "chaoyang-rc1-robot30-offline-visual-batch-v1",
        "task_id": "rc1_robot30_offline_visual_batch", "created_at": now(),
        "status": "PASSED_TERMINAL_COVERAGE", "counts": counts, "rows": rows,
        "wall_seconds": time.monotonic() - started,
        "inputs": {"selection": ref(args.selection), "matrix": ref(args.matrix)},
        "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployment_authorized": False, "authority_promoted": False,
        "claim_limit": "Pinned v77 session-level offline visual Robot terminals only; not RC1 causal input, contact truth, control truth or physical deployment authority.",
    }
    result_path = output / "RESULT.json"; atomic_json(result_path, result)
    atomic_json(output / "METRICS.json", counts | {"wall_seconds": result["wall_seconds"]})
    atomic_json(output / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": result["status"], "counts": counts, "next_action": "NEXT_IMMUTABLE_BATCH"})
    atomic_json(output / "RUN_RECEIPT.json", {"status": result["status"], "result": ref(result_path), "created_at": now()})
    atomic_json(output / "ARTIFACT_MANIFEST.json", {"result": ref(result_path), "terminal_evidence": [row.get("hard_soft_audit") or row.get("failure") for row in rows]})
    (output / "DECISION.md").write_text("# Robot30 offline visual batch\n\nThese full-session v77 outputs are review-only and may use full-sequence processing. They are never accepted by the RC1 causal loader.\n", encoding="utf-8")
    atomic_json(output / "NEXT_ACTION.json", {"next": "NEXT_IMMUTABLE_BATCH"})
    print(json.dumps({"status": result["status"], "result": ref(result_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
