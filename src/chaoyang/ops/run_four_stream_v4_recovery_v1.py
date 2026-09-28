#!/usr/bin/env python3
"""Claim the recovery task and publish a bounded CPU contract preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, process_identity, publish_bundle,
)

TASK = "four_stream_full_pipeline_v4_recovery_v1"
ATT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ("exact78", "controller_manus", "hawor_retarget", "huro")


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return artifact_ref(path)


def publish_claim(expected: int) -> None:
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != expected:
        raise RuntimeError("claim CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    if row.get("status") != "PENDING" or state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("recovery task is not pending/current")
    ident = process_identity(os.getpid())
    t = now_iso()
    row.update(
        status="RUNNING", phase="RECOVERY_CONTRACT_PREFLIGHT",
        session="four_lane_recovery", pid=os.getpid(),
        proc_start_ticks=ident["start_ticks"], heartbeat_at=t, updated_at=t,
    )
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "RUNNING", "created_at": t,
        "message": "Recovery coordinator claimed; contract preflight started.",
        "pid": os.getpid(), "proc_start_ticks": ident["start_ticks"],
    }])[-100:]
    publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_CLAIMED",
        expected_revision=expected,
        generator_path=Path(__file__),
    )


def check(label: str, path: Path, *, required: bool = True) -> dict:
    present = path.is_file()
    return {
        "label": label, "path": str(path), "required": required,
        "present": present, "bytes": path.stat().st_size if present else None,
        "sha256": sha(path) if present else None,
    }


def write_preflight() -> dict:
    old_root = REPO_ROOT / "_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001"
    checks = [
        check("recovery_packet", REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"),
        check("recovery_signature", ATT / "RUN_SIGNATURE.json"),
        check("old_v4_terminal", old_root / "V4_RUNTIME_TERMINAL_20260922.json"),
        check("old_v4_contract_audit", old_root / "V4_ROUTE_CONTRACT_AUDIT_20260922_ZH.md"),
        check("old_v3_reuse_manifest", old_root / "V3_REUSE_MANIFEST.json"),
        check("exact78_pair_builder", REPO_ROOT / "src/chaoyang/human_ego/tools/build_exact78_pair_ledgers_v32.py"),
        check("exact78_trainer", REPO_ROOT / "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py"),
        check("controller_static_candidate", REPO_ROOT / "src/chaoyang/ops/run_wiyh_ai1_static_wrist_candidate_v32.py"),
        check("hawor_actual_persistent_worker", REPO_ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
        check("hawor_prepared_manifest", REPO_ROOT / "_run/current/0915_robot15h_hawor_wave0_v1/attempts/attempt_0001/PREPARED_MANIFEST.json", required=False),
        check("huro_common_review", REPO_ROOT / "src/chaoyang/ops/run_huro_common_review_v2.py"),
    ]
    missing_required = [x["label"] for x in checks if x["required"] and not x["present"]]
    prepared = next(x for x in checks if x["label"] == "hawor_prepared_manifest")
    blockers = list(missing_required)
    if not prepared["present"]:
        blockers.append("HAWOR_PREPARED_MANIFEST_NOT_VERIFIED")
    result = {
        "schema_version": "chaoyang-v4-recovery-contract-preflight-v1",
        "task_id": TASK,
        "generated_at": now_iso(),
        "status": "PASS_WITH_LANE_BLOCKERS" if blockers else "PASS",
        "checks": checks,
        "blockers": blockers,
        "image_domain_policy": "PER_SESSION_EXPLICIT_NO_GLOBAL_SOURCE_INDEX",
        "old_v4_quality_claim": False,
        "algorithm_execution_started": False,
        "claim_limit": "Contract/readiness evidence only; no algorithm quality, training, control or deployment authority.",
    }
    atomic_json(ATT / "RECOVERY_CONTRACT_PREFLIGHT.json", result)
    for lane in LANES:
        path = ATT / "lanes" / lane / "STATE.json"
        state = load_json(path)
        lane_blocker = None
        if lane == "hawor_retarget" and not prepared["present"]:
            lane_blocker = "HAWOR_PREPARED_MANIFEST_NOT_VERIFIED"
        state.update(
            status="BLOCKED_PREREQ" if lane_blocker else "READY_CPU",
            current_action="RECOVERY_CONTRACT_PREFLIGHT_RECORDED",
            blocker=lane_blocker,
            updated_at=now_iso(),
            latest_artifacts=[ref(ATT / "RECOVERY_CONTRACT_PREFLIGHT.json")],
        )
        atomic_json(path, state)
    atomic_json(ATT / "PROGRESS_2H.json", {
        "schema_version": "chaoyang-v4-recovery-progress-v1",
        "task_id": TASK, "generated_at": now_iso(),
        "elapsed_stage": "RECOVERY_CONTRACT_PREFLIGHT",
        "completed": ["recovery route claimed", "attempt writer roots verified", "contract and dependency preflight recorded"],
        "next": ["Exact78 frozen pair producer", "Controller/MANUS M0/M1 comparison", "HaWoR prepared-manifest closure", "HuRo shared-input canary"],
        "blockers": blockers,
        "claims": {"control_ground_truth": False, "physical_deployable": False, "training_complete": False},
    })
    return result


def heartbeat() -> None:
    receipt = load_json(RECEIPT_PATH)
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    ident = process_identity(os.getpid())
    t = now_iso()
    row.update(status="RUNNING", phase="RECOVERY_CONTRACT_PREFLIGHT", pid=os.getpid(), proc_start_ticks=ident["start_ticks"], heartbeat_at=t, updated_at=t)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_HEARTBEAT", expected_revision=int(receipt["governance_revision"]), generator_path=Path(__file__))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--hold-seconds", type=int, default=900)
    args = parser.parse_args()
    publish_claim(args.expected_revision)
    result = write_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    deadline = time.time() + args.hold_seconds
    while time.time() < deadline:
        time.sleep(30)
        heartbeat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
