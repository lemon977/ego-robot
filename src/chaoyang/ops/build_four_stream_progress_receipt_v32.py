#!/usr/bin/env python3
"""Build one immutable time-point receipt for the V3.2 four-stream run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


TASK_ID = "four_stream_pretraining_baseline_v32"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
LANES = ("exact78", "ai1", "ai2", "ai4_huro")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable progress receipt conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def build_receipt(*, checkpoint_hour: int, observed_at: str, attempt_root: Path) -> dict[str, Any]:
    run_signature_path = attempt_root / "RUN_SIGNATURE.json"
    run_signature = load_json(run_signature_path)
    if run_signature.get("task_id") != TASK_ID:
        raise RuntimeError("run signature task mismatch")
    t0 = _parse_time(str(run_signature["t0"]))
    observed = _parse_time(observed_at)
    elapsed_seconds = (observed - t0).total_seconds()
    if elapsed_seconds < checkpoint_hour * 3600:
        raise RuntimeError("checkpoint cannot be published before its scheduled time")

    lanes: dict[str, Any] = {}
    for lane in LANES:
        state_path = attempt_root / "lanes" / lane / "STATE.json"
        state = load_json(state_path)
        if state.get("lane") != lane or state.get("parent_task_id") != TASK_ID:
            raise RuntimeError(f"lane identity mismatch: {lane}")
        lanes[lane] = {
            "status": state.get("status"),
            "machine_result_status": state.get("machine_result_status"),
            "blocker": state.get("blocker"),
            "claims": state.get("claims"),
            "latest_artifacts": state.get("latest_artifacts", []),
            "state": artifact_ref(state_path),
        }

    gpu_receipts = []
    observed_wait_seconds = 0.0
    for path in sorted(attempt_root.rglob("*GPU_COMMAND_RECEIPT.json")):
        payload = load_json(path)
        if payload.get("status") != "PASSED" or int(payload.get("returncode", -1)) != 0:
            continue
        seconds = float(payload.get("wait_seconds_observed", 0.0))
        observed_wait_seconds += seconds
        gpu_receipts.append({
            "attempt_id": payload.get("attempt_id"),
            "observed_wait_seconds": seconds,
            "receipt": artifact_ref(path),
        })

    status_porcelain = _git("status", "--porcelain")
    return {
        "schema_version": "chaoyang-four-stream-progress-receipt-v32",
        "task_id": TASK_ID,
        "checkpoint_hour": checkpoint_hour,
        "t0": run_signature["t0"],
        "observed_at": observed_at,
        "elapsed_seconds": elapsed_seconds,
        "status": "PROGRESS_SNAPSHOT_NOT_FINAL",
        "run_signature": artifact_ref(run_signature_path),
        "lanes": lanes,
        "resources": {
            "gpu_command_receipts": gpu_receipts,
            "gpu_observed_wait_seconds_total": observed_wait_seconds,
            "gpu_accounting_note": (
                "Append-only observation from successful command receipts; the immutable initial "
                "GPU queue ledger retains its creation-time zero counters."
            ),
        },
        "repository": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "tracked_and_untracked_clean": status_porcelain == "",
            "status_porcelain": status_porcelain.splitlines(),
            "pushed": False,
        },
        "authority": {
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
            "visual_review_complete": False,
            "training_complete": False,
        },
        "next_frozen_gates": {
            "t_plus_4_exact_raw_only_fallback": "NOT_DUE" if checkpoint_hour < 4 else "EVALUATE",
            "ai1": "KEEP_M0_AND_102_103_CLOSED_WITHOUT_ADMISSIBLE_STATIC_CANDIDATE",
            "ai2": "KEEP_KINEMATIC_ONLY_UNTIL_ZERO_CLIP_AND_OBSERVABILITY_GATES_PASS",
            "ai4": "RENDER_REMAINING_THREE_FROZEN_NUMERIC_COMPARISONS",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-hour", type=int, choices=(2, 4, 8, 24), required=True)
    parser.add_argument("--attempt-root", type=Path, default=ATTEMPT_ROOT)
    parser.add_argument("--observed-at", default=None)
    args = parser.parse_args()
    observed_at = args.observed_at or now_iso()
    output = args.attempt_root.resolve() / "progress_receipts" / f"T_PLUS_{args.checkpoint_hour:02d}H.json"
    value = build_receipt(
        checkpoint_hour=args.checkpoint_hour,
        observed_at=observed_at,
        attempt_root=args.attempt_root.resolve(),
    )
    _write_once(output, value)
    print(json.dumps({"status": "PASSED", "output": artifact_ref(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
