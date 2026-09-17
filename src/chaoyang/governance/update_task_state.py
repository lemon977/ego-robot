from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import AUTHORITY_PATH, TASK_STATE_PATH, artifact_ref, load_json, now_iso, publish_bundle
from chaoyang.governance.heartbeat_task import (
    CLEAN_RECOVERY_TASK_ID,
    CLEAN_TASK_ID,
    reconcile_clean_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--session")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--proc-start-ticks", type=int)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--clear-runtime",
        action="store_true",
        help="Clear heartbeat/PID/startticks/GPU fields for a non-active terminal or recovery state.",
    )
    parser.add_argument("--message", default="")
    parser.add_argument("--result")
    parser.add_argument("--blocker-json", type=Path)
    parser.add_argument(
        "--remove-blocker",
        action="append",
        default=[],
        help="Remove a resolved blocker by its exact name. May be repeated.",
    )
    parser.add_argument("--next-task-json", type=Path)
    parser.add_argument(
        "--clear-next-task",
        action="store_true",
        help="Clear next_task when the bounded queue has no registered runnable successor.",
    )
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    selected = next((item for item in state["tasks"] if item["task_id"] == args.task_id), None)
    if selected is None:
        selected = {"task_id": args.task_id, "phase": args.phase or "unspecified", "attempt": 0, "status": "PENDING", "updated_at": now_iso(), "heartbeat_at": None, "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None}
        state["tasks"].append(selected)
    selected["status"] = args.status
    selected["updated_at"] = now_iso()
    if args.status in {"CLAIMED", "RUNNING"}:
        selected["heartbeat_at"] = now_iso()
    if args.clear_runtime:
        if args.status in {"CLAIMED", "RUNNING"}:
            raise SystemExit("--clear-runtime is invalid for CLAIMED/RUNNING")
        selected.update(
            heartbeat_at=None,
            pid=None,
            proc_start_ticks=None,
            gpu_id=None,
        )
    if args.phase is not None:
        selected["phase"] = args.phase
    for key in ("session", "attempt", "pid", "proc_start_ticks", "gpu_id"):
        value = getattr(args, key)
        if value is not None:
            selected[key] = value
    event = {"task_id": args.task_id, "session": selected.get("session"), "attempt": selected["attempt"], "status": args.status, "created_at": now_iso(), "message": args.message}
    if args.result:
        event["result"] = artifact_ref(args.result)
        selected["result"] = event["result"]
    if args.blocker_json:
        blocker = json.loads(args.blocker_json.read_text(encoding="utf-8"))
        if not isinstance(blocker, dict):
            raise SystemExit("--blocker-json must contain an object")
        state["blockers"] = [item for item in state.get("blockers", []) if item.get("name") != blocker.get("name")]
        state["blockers"].append(blocker)
    if args.remove_blocker:
        requested = set(args.remove_blocker)
        present = {str(item.get("name")) for item in state.get("blockers", [])}
        missing = sorted(requested - present)
        if missing:
            raise SystemExit(f"cannot remove absent blocker(s): {', '.join(missing)}")
        state["blockers"] = [
            item for item in state.get("blockers", []) if item.get("name") not in requested
        ]
    if args.next_task_json and args.clear_next_task:
        raise SystemExit("--next-task-json and --clear-next-task are mutually exclusive")
    if args.next_task_json:
        next_task = json.loads(args.next_task_json.read_text(encoding="utf-8"))
        if not isinstance(next_task, dict):
            raise SystemExit("--next-task-json must contain an object")
        state["next_task"] = next_task
    elif args.clear_next_task:
        state["next_task"] = None
    # A terminal Clean update must reconcile the final immutable session
    # receipt too.  Previously only live heartbeats did this, so the last
    # session could finish after its final heartbeat and leave current at
    # 57/58 forever even though the immutable combined matrix was 58/58.
    if args.task_id in {CLEAN_TASK_ID, CLEAN_RECOVERY_TASK_ID}:
        if not reconcile_clean_authority(authority, active=False):
            raise SystemExit("Clean terminal authority reconciliation failed closed")
    state["recent_events"] = (state.get("recent_events", []) + [event])[-100:]
    receipt = publish_bundle(authority, state, event_type=f"TASK_{args.status}", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
