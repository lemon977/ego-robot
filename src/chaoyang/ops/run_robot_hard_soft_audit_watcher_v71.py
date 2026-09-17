#!/usr/bin/env python3
"""Incrementally audit finished Robot batches without blocking their producer."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import status_specific_artifacts  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 3):
        attempt_log = log.with_name(f"{log.stem}.attempt_{attempt:04d}{log.suffix}")
        if attempt_log.exists():
            continue
        with attempt_log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if completed.returncode in {0, 2}:
            return
    raise RuntimeError(f"two runtime attempts exhausted: {command}")


def exact(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference["path"])).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def bilateral_count(state_path: Path) -> int:
    with np.load(state_path, allow_pickle=False) as source:
        valid = np.asarray(source["valid_side_frame"], dtype=bool)
    count = int(np.all(valid, axis=0).sum())
    if count <= 0:
        raise RuntimeError(f"no bilateral-valid frame: {state_path}")
    return count


def adopted_source_batches(robot_root: Path, expected_set: set[str]) -> dict[Path, set[str]]:
    """Resolve V7.5 exact-adopt rows to immutable source batches."""

    batches: dict[Path, set[str]] = {}
    for adopted_path in sorted((robot_root / "adopted").glob("*/RESULT.json")):
        adopted = load(adopted_path)
        session = str(adopted.get("session", ""))
        if session not in expected_set:
            raise RuntimeError(f"unexpected adopted session: {session}")
        source = adopted.get("source_artifacts")
        if not isinstance(source, dict) or "preflight_result" not in source:
            raise RuntimeError(f"{session}: adopted source preflight is missing")
        preflight = exact(source["preflight_result"], f"{session}.preflight")
        batches.setdefault(preflight.parent.parent, set()).add(session)
    return batches


def audit_batch(
    batch: Path,
    output: Path,
    candidate_root: Path,
    allowed_sessions: set[str] | None = None,
) -> Path:
    result = output / "RESULT.json"
    if not result.is_file():
        arm = load(batch / "arm_round2/RESULT.json")
        hand = load(batch / "hand_round2/RESULT.json")
        arm_rows = {row["session"]: row for row in arm["sessions"]}
        hand_rows = {row["session"]: row for row in hand["sessions"]}
        if set(arm_rows) != set(hand_rows):
            raise RuntimeError(f"batch session mismatch: {batch}")
        for session in sorted(arm_rows):
            if allowed_sessions is not None and session not in allowed_sessions:
                continue
            collision = output / session / "collision_full/RESULT.json"
            if collision.is_file():
                continue
            _, arm_states_ref = status_specific_artifacts(arm_rows[session], kind="arm", session=session)
            _, hand_states_ref = status_specific_artifacts(hand_rows[session], kind="hand", session=session)
            arm_state = exact(arm_states_ref, f"{session}.arm")
            hand_state = exact(hand_states_ref, f"{session}.hand")
            count = bilateral_count(arm_state)
            run(
                [
                    sys.executable,
                    "src/chaoyang/ops/audit_robot_geometry_self_collision_v71.py",
                    "--session-id",
                    session,
                    "--arm-states",
                    str(arm_state),
                    "--hand-states",
                    str(hand_state),
                    "--frame-count",
                    str(count),
                    "--output-dir",
                    str(collision.parent),
                ],
                output / "logs" / f"collision_{session}.log",
            )
        run(
            [
                sys.executable,
                "src/chaoyang/ops/audit_robot_hard_soft_gate_v71.py",
                "--batch-root",
                str(batch),
                "--collision-root",
                str(output),
                "--output",
                str(result),
            ],
            output / "logs" / "hard_soft.log",
        )
    value = load(result)
    for row in value["rows"]:
        if allowed_sessions is not None and row.get("session") not in allowed_sessions:
            continue
        if row.get("hard_geometry_pass") is not True:
            continue
        session = row["session"]
        candidate = candidate_root / session / "RESULT.json"
        if candidate.is_file():
            continue
        blocker = output / session / "candidate_adoption_blocker.json"
        if blocker.is_file():
            continue
        review = exact(row["evidence"]["review_result"], f"{session}.review")
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "chaoyang.ops.adopt_exact78_pose_only_visual_robot_v52",
                    "--session",
                    session,
                    "--legacy-result",
                    str(review),
                    "--hard-soft-audit",
                    str(result),
                    "--output-root",
                    str(candidate_root),
                ],
                output / "logs" / f"adopt_{session}.log",
            )
        except RuntimeError as error:
            # A hard-feasible Robot trajectory can still be ineligible for a
            # Robotized visual candidate (for example when the frozen matrix
            # does not bind a Clean result).  That is a per-session
            # downstream-prerequisite failure, not a reason to crash the
            # whole watcher or discard the geometry audit.
            atomic_json(
                blocker,
                {
                    "schema_version": "robot-visual-candidate-blocker-v71-v1",
                    "status": "BLOCKED_PREREQ",
                    "session": session,
                    "reason": str(error),
                    "hard_soft_audit": ref(result),
                    "review_result": ref(review),
                    "authority": False,
                    "claim_limit": "Robot geometry diagnostic remains valid; no Robotized visual candidate is published.",
                },
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    args = parser.parse_args()
    robot_root = args.robot_root.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = load(robot_root / "SELECTION.json")
    expected_sessions = list(selection["sessions"])
    expected_set = set(expected_sessions)
    if len(expected_set) != len(expected_sessions):
        raise RuntimeError("duplicate expected session in Robot selection")
    state_path = output / "AUTOMATION_STATE.json"
    started = time.monotonic()
    while True:
        batch_results = []
        # V7.5 exact-adopts complete immutable V7.4 batches instead of copying
        # their multi-GB phase trees. Audit those source batches explicitly and
        # restrict publication to the sessions named by the V7.5 selection.
        adopted_batches = adopted_source_batches(robot_root, expected_set)
        for batch, allowed in sorted(adopted_batches.items(), key=lambda item: str(item[0])):
            prerequisites = (
                batch / "arm_round2/RESULT.json",
                batch / "hand_round2/RESULT.json",
                batch / "render/RESULT.json",
            )
            if all(path.is_file() for path in prerequisites):
                batch_results.append(
                    (
                        audit_batch(
                            batch,
                            output / f"adopted_{batch.name}",
                            output / "candidates",
                            allowed_sessions=allowed,
                        ),
                        allowed,
                    )
                )
        for batch in sorted(robot_root.glob("batch_*")):
            prerequisites = (
                batch / "arm_round2/RESULT.json",
                batch / "hand_round2/RESULT.json",
                batch / "render/RESULT.json",
            )
            if all(path.is_file() for path in prerequisites):
                batch_results.append((audit_batch(batch, output / batch.name, output / "candidates"), None))
        processed = []
        rows = []
        for path, allowed in batch_results:
            value = load(path)
            rows.extend(
                row for row in value["rows"]
                if row.get("session") in expected_set
                and (allowed is None or row.get("session") in allowed)
            )
            processed.append(ref(path))
        unique = {row["session"] for row in rows}
        if len(unique) != len(rows):
            raise RuntimeError("duplicate session in hard/soft batch audits")
        robot_result = robot_root / "RESULT.json"
        failure_receipts = sorted(robot_root.parent.glob("FAILED_RUNTIME_FINAL*.json"))
        root_succeeded = robot_result.is_file()
        root_failed = bool(failure_receipts)
        coverage_complete = set(expected_sessions) == unique
        root_terminal = root_succeeded or root_failed
        state = {
            "schema_version": "robot-hard-soft-audit-watcher-v71-v1",
            "status": "TERMINAL" if (root_succeeded and coverage_complete) or root_failed else "RUNNING",
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "processed_sessions": len(unique),
            "expected_sessions": len(expected_sessions),
            "robot_root_terminal": root_terminal,
            "robot_root_succeeded": root_succeeded,
            "robot_root_failed": root_failed,
            "claim_limit": "Incremental digital hard/soft Robot audit only; no authority.",
        }
        atomic_json(state_path, state)
        if state["status"] == "TERMINAL":
            index = output / "ROBOT_HARD_SOFT_CANDIDATE_INDEX.json"
            atomic_json(index, {
                "schema_version": "robot-hard-soft-candidate-index-v71-v1",
                "status": (
                    "PASSED_TERMINAL_AUDIT_COVERAGE"
                    if root_succeeded and coverage_complete
                    else "BLOCKED_UPSTREAM_RUNTIME_FAILURE"
                ),
                "created_at": state["updated_at"],
                "robot_root": ref(robot_result) if root_succeeded else None,
                "upstream_failure": ref(failure_receipts[-1]) if root_failed else None,
                "selection": ref(robot_root / "SELECTION.json"),
                "counts": {
                    "expected_sessions": len(expected_sessions),
                    "sessions": len(rows),
                    "missing_sessions": len(expected_set - unique),
                    "hard_geometry_pass": sum(row["hard_geometry_pass"] for row in rows),
                    "strict_pose_match": sum(row["strict_pose_match"] for row in rows),
                },
                "batch_results": processed,
                "visual_aux_candidates": [
                    ref(output / "candidates" / session / "RESULT.json")
                    for session in sorted(unique)
                    if (output / "candidates" / session / "RESULT.json").is_file()
                ],
                "rows": rows,
                "authority": False,
                "claim_limit": "Digital URDF hard-feasibility candidate index; incomplete upstream runs remain incomplete and unified z-buffer/visual review are still required.",
            })
            state["result"] = ref(index)
            atomic_json(state_path, state)
            return 0
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
