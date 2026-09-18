from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import sys
import time
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
    sha256_file,
)


REPO_ROOT = AUTHORITY_PATH.parents[2]
CLEAN_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
CLEAN_RECOVERY_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53"
CLEAN_SELECTION = CLEAN_ROOT / "EXACT78_WAVE0_SELECTION.json"
CLEAN_SELECTION_SHA256 = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
CLEAN_TASK_ID = "exact78_wave0_clean_v1"
CLEAN_RECOVERY_TASK_ID = "exact78_wave0_clean_runtime_recovery_v53"
LIVE_TASK_STATUSES = frozenset({
    "PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE",
})


def heartbeat_guard_reason(state: dict, task_id: str) -> str | None:
    """Return why a heartbeat must not mutate the current task state.

    A finalizer and a worker heartbeat can race on the same CAS revision.  The
    retrying heartbeat must re-check both liveness and routing after every
    reload; otherwise it can resurrect a terminal row after the finalizer has
    cleared ``next_task``.
    """
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    if not matches:
        return "UNKNOWN_TASK"
    if len(matches) != 1:
        return "DUPLICATE_TASK_ROWS"
    if matches[0].get("status") not in LIVE_TASK_STATUSES:
        return "TASK_NOT_LIVE"
    next_task = state.get("next_task")
    if not isinstance(next_task, dict) or next_task.get("task_id") != task_id:
        return "TASK_NOT_CURRENT"
    return None


def reconcile_clean_authority(
    authority: dict,
    *,
    active: bool,
    clean_root: Path = CLEAN_ROOT,
    recovery_root: Path | None = CLEAN_RECOVERY_ROOT,
    expected_selection_sha256: str = CLEAN_SELECTION_SHA256,
) -> bool:
    """Refresh lightweight Clean counts from immutable session terminals.

    The recovery audit establishes the initial count, but a long-running guardian
    can publish additional immutable terminals afterwards.  Heartbeats therefore
    rescan only the 58 small RESULT files when the count changes.  They never hash
    the large media payloads and never use the guardian's mutable STATE.json as
    authority evidence.
    """
    selection_path = clean_root / "EXACT78_WAVE0_SELECTION.json"
    if not selection_path.is_file() or sha256_file(selection_path) != expected_selection_sha256:
        return False
    selection = load_json(selection_path)
    rows = selection.get("sessions", [])
    if len(rows) != 58 or len({row.get("session_id") for row in rows}) != 58:
        return False

    counts = {
        "passed": 0,
        "failed_quality_c": 0,
        "failed_runtime_final": 0,
        "blocked_prereq": 0,
        "pending": 0,
    }
    evidence = [artifact_ref(selection_path)]
    recovery_ids: set[str] = set()
    if recovery_root is not None:
        recovery_selection_path = recovery_root / "RECOVERY_19_SELECTION.json"
        if recovery_selection_path.is_file():
            recovery_selection = load_json(recovery_selection_path)
            recovery_rows = recovery_selection.get("sessions", [])
            recovery_ids = {str(row.get("session")) for row in recovery_rows}
            if len(recovery_rows) != 19 or len(recovery_ids) != 19 or "None" in recovery_ids:
                return False
            evidence.append(artifact_ref(recovery_selection_path))
    for row in rows:
        session = row.get("session_id")
        if row.get("existing_clean") is not None:
            counts["passed"] += 1
            continue
        recovery_passed_path = (
            recovery_root / "propainter_v1" / str(session) / "RESULT.json"
            if recovery_root is not None and str(session) in recovery_ids
            else None
        )
        recovery_terminal_path = (
            recovery_root / "clean_terminals" / str(session) / "RESULT.json"
            if recovery_root is not None and str(session) in recovery_ids
            else None
        )
        passed_path = clean_root / "propainter_v1" / str(session) / "RESULT.json"
        terminal_path = clean_root / "clean_terminals" / str(session) / "RESULT.json"
        effective_passed_path = (
            recovery_passed_path
            if recovery_passed_path is not None and recovery_passed_path.is_file()
            else passed_path
        )
        if effective_passed_path.is_file():
            result = load_json(effective_passed_path)
            valid = (
                result.get("status") == "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B"
                and result.get("grade") == "B"
                and result.get("downstream_authorized") is True
                and result.get("session") == session
                and result.get("task") == row.get("task")
                and result.get("frame_count") == row.get("frame_count")
                and all(value == "PASS" for value in result.get("hard_gates", {}).values())
            )
            if not valid:
                return False
            counts["passed"] += 1
            evidence.append(artifact_ref(effective_passed_path))
            continue
        if recovery_terminal_path is not None and recovery_terminal_path.is_file():
            terminal = load_json(recovery_terminal_path)
            status = terminal.get("status")
            key = {
                "FAILED_QUALITY_C": "failed_quality_c",
                "FAILED_RUNTIME_FINAL": "failed_runtime_final",
                "BLOCKED_PREREQ": "blocked_prereq",
            }.get(status)
            if (
                key is None
                or terminal.get("terminal") is not True
                or terminal.get("session") != session
                or terminal.get("task") != row.get("task")
                or terminal.get("downstream_authorized") is not False
            ):
                return False
            counts[key] += 1
            evidence.append(artifact_ref(recovery_terminal_path))
            continue
        # The old runtime terminal is superseded by this frozen recovery set.
        # Until the successor publishes a new terminal it is recovery-pending,
        # not a current final quality result.
        if str(session) in recovery_ids:
            counts["pending"] += 1
            continue
        if terminal_path.is_file():
            terminal = load_json(terminal_path)
            status = terminal.get("status")
            key = {
                "FAILED_QUALITY_C": "failed_quality_c",
                "FAILED_RUNTIME_FINAL": "failed_runtime_final",
                "BLOCKED_PREREQ": "blocked_prereq",
            }.get(status)
            if (
                key is None
                or terminal.get("terminal") is not True
                or terminal.get("session") != session
                or terminal.get("task") != row.get("task")
                or terminal.get("downstream_authorized") is not False
            ):
                return False
            counts[key] += 1
            evidence.append(artifact_ref(terminal_path))
            continue
        counts["pending"] += 1

    if sum(counts.values()) != 58:
        return False
    waves = authority.setdefault("waves", {})
    clean_stage = next((stage for stage in authority.get("stages", []) if stage.get("stage") == "Clean"), None)
    if clean_stage is None:
        return False
    running = 1 if active and counts["pending"] > 0 else 0
    blocked = 58 - counts["passed"] - counts["failed_quality_c"] - running
    if blocked < 0:
        return False
    changed = (
        waves.get("wave0_clean_passed") != counts["passed"]
        or waves.get("wave0_clean_pending") != counts["pending"]
        or clean_stage.get("grade_c") != counts["failed_quality_c"]
        or clean_stage.get("running") != running
        or clean_stage.get("blocked") != blocked
    )
    if not changed:
        return True
    waves["wave0_clean_passed"] = counts["passed"]
    waves["wave0_clean_pending"] = counts["pending"]
    waves["wave0_clean_failed_quality_c"] = counts["failed_quality_c"]
    waves["wave0_clean_failed_runtime_final"] = counts["failed_runtime_final"]
    waves["wave0_clean_blocked_prereq"] = counts["blocked_prereq"]
    clean_stage.update(
        total=58,
        passed=counts["passed"],
        grade_c=counts["failed_quality_c"],
        running=running,
        blocked=blocked,
        authority_scope="EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_SESSION_TERMINALS",
        denominator_semantics=(
            "Wave0 frozen 58; existing Clean refs plus immutable per-session Grade-B or explicit "
            "non-pass terminals determine current counts. RUNNING is the single active guardian row."
        ),
        evidence=evidence,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--status", choices=["CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"], default="RUNNING")
    parser.add_argument("--phase")
    parser.add_argument("--session")
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    if not identity["alive"]:
        raise SystemExit(f"pid is not alive: {args.pid}")
    # Multiple independent workers heartbeat through one CAS-published bundle.
    # A short fixed five-attempt window can phase-lock with 25/30-second worker
    # heartbeats and incorrectly turn healthy computation into a runtime error.
    # Retry for a bounded ~15 seconds; non-CAS validation failures still fail
    # immediately and therefore cannot be hidden by this contention handling.
    for attempt in range(20):
        receipt = load_json(RECEIPT_PATH)
        authority = load_json(AUTHORITY_PATH)
        state = load_json(TASK_STATE_PATH)
        guard_reason = heartbeat_guard_reason(state, args.task_id)
        if guard_reason is not None:
            if guard_reason == "UNKNOWN_TASK":
                raise SystemExit(f"unknown task: {args.task_id}")
            print(
                f"heartbeat ignored fail-closed: {args.task_id} {guard_reason}",
                file=sys.stderr,
            )
            return 0
        selected = next((item for item in state["tasks"] if item["task_id"] == args.task_id), None)
        assert selected is not None
        selected.update(
            status=args.status,
            pid=args.pid,
            proc_start_ticks=identity["start_ticks"],
            heartbeat_at=now_iso(),
            updated_at=now_iso(),
        )
        if args.phase:
            selected["phase"] = args.phase
        if args.session:
            selected["session"] = args.session
            # While a long-lived task advances through immutable session
            # batches, keep the human/AI-facing next-task pointer aligned with
            # the live batch.  Previously only the task row advanced, leaving
            # next_task.session pointing at an already completed batch.
            next_task = state.get("next_task")
            if isinstance(next_task, dict) and next_task.get("task_id") == args.task_id:
                next_task["session"] = args.session
        if args.gpu_id is not None:
            selected["gpu_id"] = args.gpu_id
        if args.task_id in {CLEAN_TASK_ID, CLEAN_RECOVERY_TASK_ID}:
            reconciled = reconcile_clean_authority(
                authority,
                active=args.status in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"},
            )
            if not reconciled:
                print("warning: Clean authority count reconciliation skipped fail-closed", file=sys.stderr)
        try:
            publish_bundle(
                authority,
                state,
                event_type="HEARTBEAT",
                expected_revision=int(receipt["governance_revision"]),
                generator_path=Path(__file__),
                append_event=False,
            )
            return 0
        except RuntimeError as error:
            if "CAS revision mismatch" not in str(error) or attempt == 19:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1.0))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
