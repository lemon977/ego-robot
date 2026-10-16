#!/usr/bin/env python3
"""Registered F2/F3 successor executor; v1 F0/F1 source is left immutable."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets

from chaoyang.governance import execute_exact78_preconvergence_v1 as base
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


BASE_SOURCE = Path(base.__file__)
THIS_SOURCE = Path(__file__)


def register(phase: str) -> dict:
    if phase not in {"F2", "F3"}:
        raise ValueError("v2 executor is reserved for F2/F3")
    base.verify_fixed_parent()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = base.index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    previous = base.PHASES[base.PHASES.index(phase) - 1]
    previous_result = load_json(base.root(previous) / "RESULT.json")
    if previous_result["status"] != "PASSED":
        raise RuntimeError("PREVIOUS_PHASE_NOT_PASSED")
    task = base.task_id(phase)
    output = base.root(phase)
    packet_path = base.packet_path(phase)
    receipt_path = base.receipt_path(phase)
    if output.exists() or packet_path.exists() or receipt_path.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    helper = REPO_ROOT / "src/chaoyang/governance" / (
        "exact78_preconvergence_f2_v2.py" if phase == "F2" else "exact78_preconvergence_f3_v2.py"
    )
    if not helper.is_file():
        raise RuntimeError(f"MISSING_PHASE_HELPER:{helper}")
    runtime_code = [artifact_ref(path) for path in (BASE_SOURCE, THIS_SOURCE, helper)]
    inputs = {
        "plan": artifact_ref(base.PLAN),
        "route_before": artifact_ref(base.ROUTE),
        "visual_pointer": artifact_ref(base.POINTER),
        "visual_parent_result": artifact_ref(base.PARENT),
        "governance_receipt_before": artifact_ref(RECEIPT_PATH),
        "previous_result": artifact_ref(base.root(previous) / "RESULT.json"),
    }
    if phase == "F2":
        inputs.update({
            "p0_session_preflight": artifact_ref(base.P0 / "SESSION_PREFLIGHT.json"),
            "candidate_packet_for_sha_audit": artifact_ref(REPO_ROOT / "tasks/current/exact78_general_p1_action_canary_v1_retry8_20260930/TASK_PACKET.json"),
            "mask_packet_for_sha_audit": artifact_ref(REPO_ROOT / "tasks/current/exact78_cards042_role_mask_rebuild_v1_20260929/TASK_PACKET.json"),
            "clean_packet_for_sha_audit": artifact_ref(REPO_ROOT / "tasks/current/exact78_cards042_fresh_clean_v1_20260929/TASK_PACKET.json"),
        })
    else:
        inputs["f0_inventory"] = artifact_ref(base.root("F0") / "INVENTORY.json")
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime_code, "phase": phase,
        "models": "NOT_RUN", "calibration": "NOT_CONSUMED",
        "schema": "exact78-preconvergence-v2",
    })).hexdigest()
    output.mkdir(parents=True)
    writer = {
        "pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": task,
    }
    writes = [
        str(output.relative_to(REPO_ROOT)), str(packet_path.relative_to(REPO_ROOT)),
        str(receipt_path.relative_to(REPO_ROOT)), "docs/governance", "tasks/current/INDEX.json",
    ]
    if phase == "F2":
        writes.append(str(base.PENDING_PACKET.relative_to(REPO_ROOT)))
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": task,
        "objective": (
            "Freeze three Raw cold-start sessions, including stereo SHA, without selecting a runner"
            if phase == "F2" else
            "Audit full Exact78 task-directory archive eligibility without moving or deleting data"
        ),
        "phase": f"EXACT78_PRECONVERGENCE_{phase}",
        "plan_revision": "EXACT78_PRECONVERGENCE_V1_20260930",
        "execution_revision": f"EXACT78_PRECONVERGENCE_{phase}_V2",
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=base.LIMIT_SECONDS[phase])).isoformat(),
        "read_set": [
            "AGENTS.md", str(base.PLAN.relative_to(REPO_ROOT)),
            str(base.ROUTE.relative_to(REPO_ROOT)), str(base.POINTER.relative_to(REPO_ROOT)),
            str(helper.relative_to(REPO_ROOT)),
        ],
        "write_set": writes,
        "prerequisites": [
            "GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V2_PARENT_SHA_PINNED",
            "NO_MODEL_RUN", "NO_DELETION", "ROBOT100_READ_ONLY",
        ],
        "required_outputs": (
            ["three_session_raw_manifest", "model_and_code_audit", "nonrunnable_p1_packet"]
            if phase == "F2" else
            ["reverse_reference_audit", "pid_fd_cwd_audit", "git_and_license_audit", "size_report"]
        ),
        "budgets": {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
            "training_steps": 0, "wall_seconds_max": base.LIMIT_SECONDS[phase],
            "session_attempts_max": 1,
        },
        "stop_conditions": [
            "SHA_DRIFT", "DOUBLE_WRITER", "V2_POINTER_CHANGE", "0916_WRITE",
            "DELETION_ATTEMPT", "DEADLINE", "ONE_ATTEMPT_TERMINAL",
        ],
        "claim_limit": f"{phase} preparatory evidence only; no general baseline, training, deletion, model or threshold authority.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code,
        "run_signature": signature,
        "commands": [f"PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_preconvergence_v2 --phase {phase}"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    atomic_json(output / "WRITER.json", writer)
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": task, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "Exact78_preconvergence",
        "updated_at": now, "heartbeat_at": now, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(packet_path), "run_signature": signature,
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": task, "session": "Exact78_preconvergence",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one immutable phase result or deadline",
    }
    index["task_packets"].append({
        "task_id": task, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(
        packet_revision=f"EXACT78_PRECONVERGENCE_{phase}_V2_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(base.INDEX),
    )
    publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_PRECONVERGENCE_{phase}_V2_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=THIS_SOURCE, task_packet_index_path=base.INDEX,
        task_packet_index_value=index,
    )
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("F2", "F3"), required=True)
    phase = parser.parse_args().phase
    packet = register(phase)
    print(json.dumps({"phase": phase, "registered": True, "task_id": base.task_id(phase)}), flush=True)
    if phase == "F2":
        from chaoyang.governance.exact78_preconvergence_f2_v2 import make_f2_v2
        outputs, detail = make_f2_v2()
    else:
        from chaoyang.governance.exact78_preconvergence_f3_v2 import build_archive_eligibility
        audit = build_archive_eligibility(REPO_ROOT, load_json(base.root("F0") / "INVENTORY.json"))
        output = base.root("F3")
        eligibility = output / "ARCHIVE_ELIGIBILITY.json"
        eligibility_zh = output / "ARCHIVE_ELIGIBILITY_ZH.md"
        atomic_json(eligibility, audit)
        counts = audit["counts"]
        base.write_text(eligibility_zh, f"""# Exact78 历史任务目录逻辑隔离初筛

- 审计任务根 {len(audit['rows'])} 个，表观大小 {audit['total_apparent_bytes']} 字节。
- 必须保留 {counts['MUST_KEEP']} 个；引用证明不足 {counts['BLOCKED_REFERENCE_PROOF']} 个；已获归档授权 0 个。
- 此清单覆盖无 `RESULT.json` 的 partial，扫描当前权威来源及历史任务包／收据，报告进程 FD/CWD 和 Git ignored 范围；详细证据在 `ARCHIVE_ELIGIBILITY.json`。
- 许可证、二进制 sidecar 及不可访问进程造成的证明缺口仍保守阻断。无文件移动、删除或物理归档；后续清理须另立任务并取得用户同意。
""")
        outputs = {"eligibility": eligibility, "eligibility_zh": eligibility_zh}
        detail = {**counts, "task_roots": len(audit["rows"]), "total_apparent_bytes": audit["total_apparent_bytes"]}
    base.__file__ = str(THIS_SOURCE)
    base.finish(phase, packet, outputs, detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
