#!/usr/bin/env python3
"""Run the frozen 1+2 Role Mask R7_2 regression bundle and CAS its terminal state."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

from chaoyang.ops.run_post_clean_s1_automation_v71 import (
    ROLE_REGRESSIONS,
    load_json,
    publish_role_bundle,
    run_role_regression,
)
from chaoyang.governance.common import atomic_json


ROOT = Path(__file__).resolve().parents[3]
ROLE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/role_mask_runtime_contract_repair_poker042/R7_2"
AUTOMATION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/role_mask_r72_regressions"
RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def publish_delta(rows: list[dict], bundle_path: Path) -> Path:
    path = ROLE_ROOT / "WAVE_DELTA.json"
    value = {
        "schema_version": "exact78-role-mask-wave-delta-v71",
        "artifact_revision": "R7_2",
        "status": "PASS_SUCCESSOR_GATE_CANDIDATE",
        "baseline_revision": "R7_0",
        "wave0_unchanged": True,
        "authority": False,
        "eligible_for_bounded_expansion": True,
        "candidate_sessions": [{
            "session_id": "play_cards_0901_042",
            "predecessor_first_blocker": "ROLE_MASK_C",
            "successor_status": "PASSED",
            "result": ref(Path(rows[0]["result"])),
        }],
        "frozen_regressions": [
            {"session_id": session_id, "status": row["status"], "result": ref(Path(row["result"]))}
            for (session_id, _, _), row in zip(ROLE_REGRESSIONS, rows[1:], strict=True)
        ],
        "bundle": ref(bundle_path),
        "requires_aggregator_review_before_authority": True,
        "claim_limit": "One recovered Role Mask candidate plus two frozen prior-B regressions; not batch authority and not a Wave0 mutation.",
    }
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError("immutable Role Mask R7.2 WAVE_DELTA conflict")
    else:
        atomic_json(path, value)
    return path


def publish_result(bundle_path: Path, delta_path: Path) -> Path:
    path = ROLE_ROOT / "RESULT.json"
    value = {
        "schema_version": "exact78-role-mask-successor-result-v71",
        "artifact_revision": "R7_2",
        "status": "PASSED",
        "gate": "ONE_FAILURE_CANARY_PLUS_TWO_FROZEN_PRIOR_B_REGRESSIONS",
        "bundle": ref(bundle_path),
        "wave_delta": ref(delta_path),
        "authority": False,
        "batch_expansion_permitted": True,
        "claim_limit": "Bounded Role Mask successor gate passed; no automatic current-authority promotion or R7_0 overwrite.",
    }
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError("immutable Role Mask R7.2 RESULT conflict")
    else:
        atomic_json(path, value)
    return path


def update(status: str, result: Path) -> None:
    for _ in range(12):
        revision = int(load_json(RECEIPT)["governance_revision"])
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", "successor_role_mask_v71", "--status", status,
            "--phase", "r72_frozen_one_plus_two_bundle_terminal", "--clear-runtime",
            "--result", str(result.resolve()),
            "--message", "R7.2 frozen failure canary plus two prior-B regressions; no automatic authority promotion",
            "--expected-revision", str(revision),
        ]
        done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if done.returncode == 0:
            return
        if "revision conflict" not in (done.stdout + done.stderr).lower():
            raise RuntimeError((done.stdout + done.stderr)[-2000:])
        time.sleep(1)
    raise RuntimeError("governance CAS retry budget exhausted")


def main() -> int:
    canary = ROLE_ROOT / "sessions/play_cards_0901_042/attempts/attempt_0001/RESULT.json"
    if not canary.is_file() or load_json(canary).get("status") != "PASSED":
        return 3
    rows = [{"task": "role_mask_runtime_contract_repair", "status": "PASSED", "result": str(canary)}]
    epoch = int(time.time())
    for session_id, config, config_sha in ROLE_REGRESSIONS:
        row = run_role_regression(
            session_id=session_id, config=config, config_sha=config_sha,
            automation_root=AUTOMATION, executor_epoch=epoch,
            role_root=ROLE_ROOT, artifact_revision="R7_2",
        )
        rows.append(row)
        if row.get("status") != "PASSED":
            break
    bundle = publish_role_bundle(rows, role_root=ROLE_ROOT, artifact_revision="R7_2")
    delta = publish_delta(rows, Path(bundle["result"]))
    result = publish_result(Path(bundle["result"]), delta)
    status = "PASSED" if bundle["status"] == "PASSED" else bundle["status"]
    update(status, result)
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
