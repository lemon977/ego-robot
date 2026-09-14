from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.governance.common import AUTHORITY_PATH, TASK_STATE_PATH, freshness, load_json, now_iso, process_identity, publish_bundle


def gpu_pids() -> set[int] | None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def safe_recovery_candidate(task: dict[str, object], seen_gpu: set[int] | None) -> bool:
    """Return true only when process identity and GPU evidence both permit recovery.

    A GPU task whose recorded parent disappeared is *not* safe to recover while
    any compute process remains visible: that process may be an unrecorded
    launcher child.  This deliberately prefers a resource hold over reclaiming
    another live workload.  A later lease-specific auditor may prove ownership
    using worker PID/startticks and narrow this conservative rule.
    """
    pid = task.get("pid")
    if pid is None:
        return False
    identity = process_identity(int(pid))
    mismatch = not identity["alive"] or (
        task.get("proc_start_ticks") is not None
        and identity["start_ticks"] != task["proc_start_ticks"]
    )
    if not mismatch:
        return False
    if task.get("gpu_id") is None:
        return True
    return seen_gpu is not None and not seen_gpu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-revision", type=int)
    args = parser.parse_args()
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    seen_gpu = gpu_pids()
    candidates = []
    for task in state["tasks"]:
        if task["status"] not in {"CLAIMED", "RUNNING"} or task.get("pid") is None:
            continue
        if safe_recovery_candidate(task, seen_gpu):
            candidates.append(task["task_id"])
            if args.apply:
                task["status"] = "FAILED_RUNTIME_RETRYABLE"
                task["updated_at"] = now_iso()
                task["heartbeat_at"] = None
    result = {"freshness": freshness(state), "candidates": candidates, "gpu_query_available": seen_gpu is not None, "applied": args.apply}
    if args.apply and candidates:
        if args.expected_revision is None:
            raise SystemExit("--expected-revision is required with --apply")
        publish_bundle(authority, state, event_type="STALE_TASK_RECOVERY", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
