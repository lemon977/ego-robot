from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tools.governance.common import (
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
CLEAN_ROOT = REPO_ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
CLEAN_SELECTION = CLEAN_ROOT / "EXACT78_WAVE0_SELECTION.json"
CLEAN_SELECTION_SHA256 = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
CLEAN_TASK_ID = "exact78_wave0_clean_v1"


def reconcile_clean_authority(
    authority: dict,
    *,
    active: bool,
    clean_root: Path = CLEAN_ROOT,
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
    for row in rows:
        session = row.get("session_id")
        if row.get("existing_clean") is not None:
            counts["passed"] += 1
            continue
        passed_path = clean_root / "propainter_v1" / str(session) / "RESULT.json"
        terminal_path = clean_root / "clean_terminals" / str(session) / "RESULT.json"
        if passed_path.is_file():
            result = load_json(passed_path)
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
            evidence.append(artifact_ref(passed_path))
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
    for attempt in range(5):
        receipt = load_json(RECEIPT_PATH)
        authority = load_json(AUTHORITY_PATH)
        state = load_json(TASK_STATE_PATH)
        selected = next((item for item in state["tasks"] if item["task_id"] == args.task_id), None)
        if selected is None:
            raise SystemExit(f"unknown task: {args.task_id}")
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
        if args.gpu_id is not None:
            selected["gpu_id"] = args.gpu_id
        if args.task_id == CLEAN_TASK_ID:
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
            if "CAS revision mismatch" not in str(error) or attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
