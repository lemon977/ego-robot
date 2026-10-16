"""Finite selection-only maintenance using the existing governance publisher."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import signal
import sys
import tarfile
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
)
from chaoyang.governance.publish_project_branch_convergence_v1 import index_body, refs, validate_current
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "dual_branch_phase_v1_20261008"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PACKET = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
RECEIPT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = "docs/plans/DUAL_BRANCH_PHASE_V1_20261008"
WINDOW = 2700
EDITS = (
    "src/chaoyang/governance/current_r3_contracts.py",
    "scripts/takeover/autopilot.py", "scripts/takeover/autopilot_start.sh",
    "README.md", "docs/README.md", "docs/current/README_ZH.md",
    "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/PROJECT_LAYOUT_ZH.md",
    "docs/current/V5_SENSOR.md", "docs/current/V5_HURO.md", "docs/current/V5_MOTION.md",
    "docs/current/AI4_HURO.md", "review/README.md",
)
NEW = (
    "src/chaoyang/governance/dual_branch_phase_v1.py",
    "tests/unit/test_dual_branch_phase_v1.py",
    "docs/current/DUAL_BRANCH_BASELINE_V1.json", "docs/current/DUAL_BRANCH_BASELINE_V1_ZH.md",
)
OLD_PLAN = REPO_ROOT / "docs/plans/EXACT78_TAKEOVER_20261004/AUTOPILOT_PLAN.json"
OLD_STATE = OLD_PLAN.with_name("AUTOPILOT_STATE.json")


def register():
    validate_current()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(p.exists() for p in (ATTEMPT, PACKET, RECEIPT)):
        raise RuntimeError("FRESH_TASK_REQUIRED")
    parent = REPO_ROOT / "_run/current/exact78_runtime_snapshot_repair_v1_20261008/attempts/attempt_0001"
    protected = load_json(parent / "RESULT.json")["protected_after"]
    keep_path = REPO_ROOT / "_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/DEPENDENCY_KEEP_SET.json"
    snapshot_refs = {ref["path"]: ref for ref in load_json(parent / "CODE_SNAPSHOT_MANIFEST.json")["files"]}
    historical_keep_mismatches = []
    for ref in load_json(keep_path)["files"]:
        name = str(Path(ref["path"]).relative_to(REPO_ROOT))
        if name not in EDITS:
            current = artifact_ref(REPO_ROOT / name)
            if current != ref:
                # Old keep-set is a reachability list, not today's code authority.
                # Accept only already-sealed control-plane updates, never algorithm drift.
                if (name not in {"src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
                                 "src/chaoyang/governance/current_baseline_v2.py"}
                        or snapshot_refs.get(current["path"]) != current):
                    raise RuntimeError("UNREVIEWED_KEEP_DEPENDENCY_DRIFT:" + name)
                historical_keep_mismatches.append({"historical": ref, "current": current,
                    "scope": "SEALED_CONTROL_PLANE_UPDATE_NOT_NUMERICAL_BASELINE"})
            protected[name] = current
    protected.update(refs(tuple(str((parent / name).relative_to(REPO_ROOT)) for name in
                                ("RESULT.json", "CODE_SNAPSHOT.tar.gz", "CODE_SNAPSHOT_MANIFEST.json"))))
    protected.update(refs((str(keep_path.relative_to(REPO_ROOT)), "docs/current/PROJECT_BRANCH_MAP.json")))
    if refs(tuple(protected)) != protected:
        raise RuntimeError("PROTECTED_PARENT_DRIFT")
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"protected": protected, "edit_before": refs(EDITS), "publisher": artifact_ref(Path(__file__)),
              "plan": artifact_ref(REPO_ROOT / PLAN / "00_EXECUTION_ZH.md"),
              "historical_keep_sha_mismatches": historical_keep_mismatches,
              "old_autopilot_plan": artifact_ref(OLD_PLAN), "old_autopilot_pid": 644502,
              "old_autopilot_start_ticks": 352475492}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Select two phase-V1 aliases from adopted v13/v16; isolate historical defaults and retire completed autopilot",
        "phase": "CPU_DUAL_BRANCH_PHASE_CONVERGENCE", "plan_revision": TASK.upper(), "execution_revision": TASK.upper(),
        "session_id": None, "deadline_at": (datetime.now().astimezone() + timedelta(seconds=WINDOW+180)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md", PLAN + "/00_EXECUTION_ZH.md",
            "docs/governance/DOC_AUTHORITY_MAP.json", "docs/governance/ALGORITHM_CONTRACT.json",
            "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
            "docs/current/EXACT78_VISUAL_BASELINE_V1.json", "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json"],
        "expanded_read_set": [*protected, *EDITS, str(OLD_PLAN.relative_to(REPO_ROOT)), str(OLD_STATE.relative_to(REPO_ROOT))],
        "read_expansion_reason": "Concrete historical-reference ambiguity and completed autopilot identity conflict require dependency and entry audits.",
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)), str(RECEIPT.relative_to(REPO_ROOT)),
            "tasks/current/INDEX.json", "docs/governance", PLAN, "_run/caches/dual_branch_phase_v1", *EDITS, *NEW],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "PROTECTED_SHA_VERIFIED"],
        "required_outputs": ["RESULT.json", "REGRESSION.json", "AUTOPILOT_STOP.json", "CODE_SNAPSHOT.tar.gz", "CODE_SNAPSHOT_MANIFEST.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0, "model_calls_max": 0,
                    "training_steps": 0, "wall_seconds_max": WINDOW+180, "session_attempts_max": 1},
        "stop_conditions": ["protected_sha_drift", "double_writer", "autopilot_identity_or_plan_drift", "deadline"],
        "claim_limit": "Selection and history isolation only; no baseline algorithm, numeric gate, media, MANUS implementation, visual/product/training promotion or deletion.",
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": [f"{sys.executable} -B -m chaoyang.governance.publish_dual_branch_phase_v1"]}
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    ATTEMPT.mkdir(parents=True); PACKET.parent.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", writer); atomic_json(PACKET, packet)
    for name in EDITS:
        dst = ATTEMPT / "before" / name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, dst)
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "phase_selection_only", "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None, "task_packet": artifact_ref(PACKET),
        "run_signature": signature, "quality": "NOT_ASSESSED", "adoption": "ORIGINAL_BASELINES_UNCHANGED", "review": "NOT_APPLICABLE"})
    state["next_task"] = {"task_id": TASK, "session": "phase_selection_only", "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_TWO_THREADS", "stop_condition": "one terminal publication"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="DUAL_BRANCH_PHASE_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def stop_old_autopilot(packet):
    pid = packet["inputs"]["old_autopilot_pid"]
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        atomic_json(ATTEMPT / "AUTOPILOT_STOP.json", {"status": "ALREADY_ABSENT", "pid": pid})
        return
    if process_identity(pid)["start_ticks"] != packet["inputs"]["old_autopilot_start_ticks"]:
        raise RuntimeError("AUTOPILOT_IDENTITY_DRIFT")
    cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    if cmd != "/usr/local/bin/python -B scripts/takeover/autopilot.py":
        raise RuntimeError("AUTOPILOT_COMMAND_DRIFT")
    children = (proc / f"task/{pid}/children").read_text().strip()
    plan, state = load_json(OLD_PLAN), load_json(OLD_STATE)
    if children or artifact_ref(OLD_PLAN) != packet["inputs"]["old_autopilot_plan"] or not plan["steps"]:
        raise RuntimeError("AUTOPILOT_PLAN_OR_CHILD_DRIFT")
    if any(state["steps"].get(row["id"], {}).get("status") != "DONE" for row in plan["steps"]):
        raise RuntimeError("AUTOPILOT_NONTERMINAL_STEP")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or (proc / "stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("AUTOPILOT_DID_NOT_EXIT")
    atomic_json(ATTEMPT / "AUTOPILOT_STOP.json", {"status": "STOPPED_COMPLETED_LEGACY_AUTOPILOT", "pid": pid,
        "proc_start_ticks": packet["inputs"]["old_autopilot_start_ticks"], "command": cmd,
        "steps_done": len(plan["steps"]), "children": [], "signal": "SIGTERM", "other_processes_stopped": 0})


def finish(packet, failure=None):
    if refs(tuple(packet["inputs"]["protected"])) != packet["inputs"]["protected"]:
        failure = "PROTECTED_SHA_DRIFT"
    outputs = {}
    if failure is None:
        from chaoyang.governance.dual_branch_phase_v1 import validate_phase
        validate_phase(REPO_ROOT)
        if load_json(ATTEMPT / "REGRESSION.json")["status"] != "PASS":
            raise RuntimeError("REGRESSION_NOT_PASS")
        nav = importlib.reload(importlib.import_module("chaoyang.governance.project_branch_navigation_v1"))
        paths = sorted(set(nav.snapshot_paths(REPO_ROOT)) | {REPO_ROOT / p for p in (*EDITS, *NEW, PLAN+"/00_EXECUTION_ZH.md", str(Path(__file__).relative_to(REPO_ROOT)))})
        rows = [artifact_ref(p) for p in paths]
        atomic_json(ATTEMPT / "CODE_SNAPSHOT_MANIFEST.json", {"schema_version": "DUAL_BRANCH_PHASE_SOURCE_OVERLAY_V1",
            "files": rows, "contains_models_videos_external_data": False, "claim_limit": "Source/navigation overlay only; not a full environment, dependency, media or governance backup."})
        with tarfile.open(ATTEMPT / "CODE_SNAPSHOT.tar.gz", "x:gz") as tar:
            for row in rows:
                path = Path(row["path"])
                if artifact_ref(path) != row or path.is_symlink():
                    raise RuntimeError("SNAPSHOT_DRIFT")
                tar.add(path, arcname=str(path.relative_to(REPO_ROOT)), recursive=False)
        outputs = {n: artifact_ref(ATTEMPT / n) for n in ("REGRESSION.json", "AUTOPILOT_STOP.json", "CODE_SNAPSHOT_MANIFEST.json", "CODE_SNAPSHOT.tar.gz")}
    result = {"schema_version": "DUAL_BRANCH_PHASE_RESULT_V1", "task_id": TASK, "status": "PASSED" if failure is None else "FAILED_RUNTIME_FINAL",
        "failure_code": failure, "protected_before": packet["inputs"]["protected"], "protected_after": refs(tuple(packet["inputs"]["protected"])),
        "edit_before": packet["inputs"]["edit_before"], "edit_after": refs(EDITS), "outputs": outputs,
        "files_deleted": 0, "files_moved": 0, "models_run": 0, "numeric_gates_changed": False, "existing_media_overwritten": False,
        "baseline_pointers_changed": False, "claim_limit": packet["claim_limit"]}
    atomic_json(ATTEMPT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "DUAL_BRANCH_PHASE_RECEIPT_V1", "status": result["status"], "task_id": TASK,
        "result": artifact_ref(ATTEMPT / "RESULT.json"), "baseline_pointers_changed": False})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    row = next(r for r in state["tasks"] if r["task_id"] == TASK)
    row.update(status=result["status"], pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(ATTEMPT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT), quality="SELECTION_ONLY")
    state["next_task"] = None
    next(r for r in index["task_packets"] if r["task_id"] == TASK)["execution_allowed"] = False
    index.update(status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    importlib.reload(importlib.import_module("chaoyang.governance.current_r3_contracts"))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="DUAL_BRANCH_PHASE_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    atomic_json(ATTEMPT / "POST_PUBLICATION_VALIDATION.json", validate_current())
    print(json.dumps({"status": result["status"], "result": artifact_ref(ATTEMPT / "RESULT.json")}), flush=True)


def main():
    packet = register()
    try:
        stop_old_autopilot(packet)
        print("READY", flush=True); deadline = time.monotonic() + WINDOW
        while time.monotonic() < deadline:
            ready, _, _ = select.select([sys.stdin], [], [], 30)
            if ready:
                if sys.stdin.readline().strip() != "PUBLISH":
                    raise RuntimeError("ABORTED")
                finish(packet); return
            state = copy.deepcopy(load_json(TASK_STATE_PATH))
            next(r for r in state["tasks"] if r["task_id"] == TASK).update(heartbeat_at=now_iso(), updated_at=now_iso())
            publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="DUAL_BRANCH_PHASE_HEARTBEAT",
                expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__))
        raise RuntimeError("DEADLINE")
    except BaseException as exc:
        if not (ATTEMPT / "RESULT.json").exists():
            finish(packet, str(exc))
        raise


if __name__ == "__main__":
    main()
