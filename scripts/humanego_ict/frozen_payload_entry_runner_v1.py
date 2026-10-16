"""Thin finite publisher for an explicitly adopted frozen-payload resume.

No model code runs here. Reuses the existing publisher, storage observation,
registry and central GPU lease mechanisms, with one new finite Task per call.
Never takes over a live training worker or an active continuation dispatcher.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, publish_bundle)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.humanego_ict.frozen_payload_entry_cli import (
    OPERATION, ROUTE, EXECUTION_MODE, SOURCE_POLICY, TRAINING_SCOPE,
    PERIODIC_CONFIG_SCHEMA, PERIODIC_INITIALIZATION, NORMAL_ROLES, PERIODIC_ROLES,
    _engineering, _ref, _json, require)
from chaoyang.humanego_ict.production_resume import _project_path

ORIGINAL_PUBLISHER_SHA = "d3fb4d710501c1b2da4c943f84094fe8e0abd03cbdb9bc9e4d0e06c5ba2980dd"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def _infrastructure():
    path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    require(_ref(path, REPO_ROOT)["sha256"] == ORIGINAL_PUBLISHER_SHA, "ORIGINAL_PUBLISHER_SOURCE_DRIFT")
    spec = importlib.util.spec_from_file_location("_frozen_payload_publisher_infrastructure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pid_identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return int(fields[19]), fields[0]


def _group_members(group):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            text = (entry / 'stat').read_text()
            fields = text[text.rfind(')') + 2:].split()
            if int(fields[2]) == group:
                members.append({'pid': int(entry.name), 'state': fields[0],
                                'proc_start_ticks': int(fields[19]), 'pgid': group})
        except FileNotFoundError:
            continue
    return members


def _owned_child_reaped(child):
    return child is None or (child.poll() is not None and not _group_members(child.pid))


def _handoff(reference):
    value = _json(reference, REPO_ROOT)
    require(value.get("schema") == "ict-continuation-owner-handoff-v1"
        and value.get("status") == "CONTROLLER_RETIRED_SAME_GOAL_HANDOFF"
        and value.get("original_controller_reaped") is True
        and value.get("in_flight") is None and value.get("registered_successors") == 0,
        "EXPLICIT_FINITE_CONTROLLER_HANDOFF_REQUIRED")
    owner = value["original_controller"]
    try:
        ticks, state = _pid_identity(owner["pid"])
    except FileNotFoundError:
        return value
    require(ticks != owner["proc_start_ticks"], "ORIGINAL_CONTROLLER_STILL_EXISTS")
    return value


def _normal_parent(infra, parent_ref, parent, branch, state, index):
    config = _json(parent['inputs']['training_config'], REPO_ROOT)
    if config.get('schema_version') != PERIODIC_CONFIG_SCHEMA:
        return infra.parent_bindings(Path(parent_ref['path']), branch, state=state, index=index)
    from chaoyang.humanego_ict.frozen_payload_state import validate_frozen_normal_parent
    rows = [r for r in state['tasks'] if r.get('task_id') == parent['task_id']]
    items = [r for r in index['task_packets'] if r.get('task_id') == parent['task_id']]
    require(len(rows) == len(items) == 1 and rows[0].get('status') == 'PASSED'
        and rows[0].get('task_packet') == parent_ref and items[0].get('execution_allowed') is False
        and items[0].get('packet_sha256') == parent_ref['sha256'], 'SEALED_FROZEN_NORMAL_PARENT_REQUIRED')
    outer_ref = rows[0]['result']
    outer = _json(outer_ref, REPO_ROOT)
    stages = []
    for ref in outer.get('stage_results', []):
        value = _json(ref, REPO_ROOT)
        if (value.get('task_id') == parent['task_id'] and value.get('run_signature') == parent['run_signature']
                and value.get('branch') == branch and type(value.get('optimizer_updates')) is int):
            stages.append((value['optimizer_updates'], ref, value))
    require(bool(stages), 'FROZEN_NORMAL_INNER_RESULT_REQUIRED')
    _, inner_ref, inner = max(stages, key=lambda v: v[0])
    cp = _ref(inner['last_checkpoint'], REPO_ROOT)
    require(cp in outer.get('complete_checkpoints', []), 'FINAL_FROZEN_NORMAL_CHECKPOINT_REQUIRED')
    refs = {'task_packet': parent_ref, 'outer_result': outer_ref, 'inner_result': inner_ref,
            'training_config': parent['inputs']['training_config'], 'checkpoint': cp,
            'checkpoint_sidecar': _ref(cp['path'] + '.ref.json', REPO_ROOT)}
    evidence = validate_frozen_normal_parent(refs, root=REPO_ROOT)
    require(evidence['branch'] == branch and 0 < inner['optimizer_updates'] < 20000,
            'FROZEN_PARENT_BRANCH_OR_REMAINING_BUDGET')
    return refs, evidence


def _publish(infra, state, index, event, revision):
    index["supersedes_index"] = artifact_ref(INDEX)
    return publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
        expected_revision=revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)


def _summary(infra, packet, directory, result=None):
    update = infra._tail_update(directory / "output/updates.jsonl")
    status = result["task_terminal_status"] if result else "RUNNING"
    atomic_json(directory / "CONTINUE_HERE_ZH.json", {
        "task_id": packet["task_id"], "updated_at": now_iso(), "status": status,
        "source_policy": SOURCE_POLICY, "resume_mode": packet["resume_mode"],
        "branch": packet["branch"], "parent_optimizer_updates": packet["parent_optimizer_updates"],
        "last_logged_optimizer_updates": update.get("optimizer_update") if update else None,
        "last_logged_loss": update.get("flow_loss") if update else None,
        "sealed_actual_optimizer_updates": result.get("optimizer_updates") if result else None,
        "log_interval_updates": 20, "unlogged_updates": "UNKNOWN",
        "deadline_at": packet["deadline_at"], "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "next_action_zh": "本任务达到20k后审计完整checkpoint/导出；未到则仅以封存完整同分支父登记下一有限任务。"})


def _seal(infra, packet, directory, result, env):
    # New-route alias of the actual collected facts, not a rewritten status.
    # Original publishers may have only failure_codes; preserve that field.
    result['errors'] = list(result['failure_codes'])
    result.update(source_policy=SOURCE_POLICY, resume_mode=packet["resume_mode"],
        current_raw_source_replay_or_stat_equivalence=False,
        snapshot_map_ref=packet["inputs"]["snapshot_map"],
        parent_evidence_role="NORMAL_FINAL" if packet["resume_mode"] == "normal" else "PERIODIC_FAILED_PARENT")
    result_ref = infra._json_once(directory / "RESULT.json", result)
    receipt, state, index = infra._snapshot()
    row, item = infra._own_row(packet, state, index)
    terminal = result["task_terminal_status"]
    row.update(status=terminal, pid=None, proc_start_ticks=None, result=result_ref,
               updated_at=now_iso(), heartbeat_at=now_iso())
    state["next_task"] = None
    item.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    tp = REPO_ROOT / "tasks/receipts" / (packet["task_id"].upper() + ".json")
    infra._json_once(tp, {"task_id": packet["task_id"], "status": terminal, "result": result_ref,
                        "claim_limit": packet["claim_limit"]})
    row["receipt"] = artifact_ref(tp)
    _summary(infra, packet, directory, result)
    _publish(infra, state, index, "ICT_FROZEN_PAYLOAD_RESUME_SEALED", receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-m", "chaoyang.cli", "validate-governance"],
                           cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    infra._json_once(directory / "POST_VALIDATION.json", {"exit_code": check.returncode,
                    "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps(result), flush=True)
    return 0 if terminal == "PASSED" and check.returncode == 0 else 2


def register_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-task-packet", type=Path, required=True)
    parser.add_argument("--branch", choices=("raw", "robotized"), required=True)
    parser.add_argument("--snapshot-map", type=Path, required=True)
    parser.add_argument("--controller-handoff", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--mode", choices=("normal", "periodic"), default="normal")
    parser.add_argument("--periodic-parent-refs", type=Path)
    parser.add_argument("--periodic-evidence", type=Path)
    args = parser.parse_args(argv)
    require(args.attempt > 0 and 1500 < args.wall_seconds <= 43200, "FRESH_FINITE_BUDGET_REQUIRED")
    infra = _infrastructure()
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    handoff_ref = _ref(args.controller_handoff, REPO_ROOT)
    _handoff(handoff_ref)
    contract = _json(receipt["files"]["algorithm_contract"], REPO_ROOT)
    route = contract.get("special_status_contracts", {}).get(ROUTE)
    require(route and route.get("candidate_operation") == OPERATION
        and route.get("source_policy") == SOURCE_POLICY and route.get("current_runnable_operation") is None,
        "ADOPTED_CANDIDATE_SNAPSHOT_ROUTE_REQUIRED")
    closure = route["code_closure"]
    for ref in closure:
        require(_ref(ref["path"], REPO_ROOT) == ref, "EXACT_ROUTE_SHA_DRIFT")
    parent_ref = _ref(args.parent_task_packet, REPO_ROOT)
    parent = _json(parent_ref, REPO_ROOT)
    if args.mode == "normal":
        require(args.periodic_parent_refs is None and args.periodic_evidence is None,
                "NORMAL_MODE_MUST_NOT_HAVE_PERIODIC_EVIDENCE")
        refs, evidence = _normal_parent(infra, parent_ref, parent, args.branch, state, index)
        step = evidence["inner_result"]["optimizer_updates"]
        parent_config = evidence["training_config"]
        periodic = None
    else:
        require(args.periodic_parent_refs is not None and args.periodic_evidence is not None,
                "EXPLICIT_PERIODIC_REFERENCES_REQUIRED")
        refs = _json(_ref(args.periodic_parent_refs, REPO_ROOT), REPO_ROOT)
        require(set(refs) == set(PERIODIC_ROLES) and refs["task_packet"] == parent_ref,
                "EXACT_FIVE_PERIODIC_PARENT_REFS_REQUIRED")
        periodic = _json(_ref(args.periodic_evidence, REPO_ROOT), REPO_ROOT)
        from chaoyang.humanego_ict.frozen_payload_state import validate_periodic_evidence
        evidence = validate_periodic_evidence(refs, periodic, root=REPO_ROOT)
        step = evidence["optimizer_updates"]
        parent_config = _json(refs["training_config"], REPO_ROOT)
    require(parent.get("branch") == args.branch and 0 < step < 20000, "SAME_BRANCH_ACTUAL_PARENT_STEP_REQUIRED")
    require(all(ref in closure for ref in parent["runtime_code"]), "PARENT_CLOSURE_NOT_PRESERVED")
    snapshot_ref = _ref(args.snapshot_map, REPO_ROOT)
    snapshot = _json(snapshot_ref, REPO_ROOT)
    require(snapshot.get("branch") == args.branch and snapshot.get("split") == "train"
        and snapshot.get("production_manifest_ref") == parent["inputs"]["production_manifest"],
        "SAME_ADMITTED_BRANCH_SNAPSHOT_REQUIRED")
    er = route.get("sealed_adoption_task_refs")
    require(isinstance(er, dict) and set(er) == {"entry_engineering_task_packet", "entry_engineering_task_result"},
            "FORMAL_SEALED_ADOPTION_TASK_REFS_REQUIRED")
    _engineering(REPO_ROOT, {"inputs": er, "runtime_code": closure}, route, state, index)
    task_id = f"pick_place_humanego_ict_frozen_payload_{args.branch}_a{args.attempt:02d}_20261012"
    directory = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    require(not directory.exists() and not packet_path.exists(), "FRESH_ATTEMPT_REQUIRED")
    directory.mkdir(parents=True)
    inputs = {k: parent["inputs"][k] for k in ("production_manifest", "prepared_pair", "pretrained", "pretrained_config")}
    roles = NORMAL_ROLES if args.mode == "normal" else PERIODIC_ROLES
    inputs.update({roles[k]: v for k, v in refs.items()})
    inputs.update(er, snapshot_map=snapshot_ref, controller_handoff=handoff_ref)
    if periodic:
        inputs.update({"periodic_" + k: v for k, v in periodic.items()})
    measure_storage, lease_update = infra._existing_helpers()
    inputs["storage_capacity_observation"] = measure_storage(directory, task_id)
    config = copy.deepcopy(parent_config)
    if args.mode == "normal":
        config = infra.make_resume_config(parent_config, refs, args.wall_seconds)
    else:
        config.update(schema_version=PERIODIC_CONFIG_SCHEMA, initialization=PERIODIC_INITIALIZATION,
                      resume_parent_bindings=copy.deepcopy(refs))
        config["training"]["wall_seconds_max"] = args.wall_seconds - 1500
    from chaoyang.humanego_ict.frozen_payload_entry_cli import validate_config
    validate_config(config, args.mode)
    inputs["training_config"] = infra._json_once(directory / "TRAINING_CONFIG.json", config)
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(seconds=args.wall_seconds)
    output = directory / "output"
    writer = infra._writer_identity(task_id, args.attempt)
    command_args = ["train", "--task-packet", str(packet_path), "--manifest", inputs["production_manifest"]["path"],
        "--config", inputs["training_config"]["path"], "--checkpoint", inputs["resume_checkpoint"]["path"],
        "--output-dir", str(output), "--branch", args.branch]
    command = shlex.join([sys.executable, "-m", "chaoyang.cli", "run", OPERATION, *command_args])
    budgets = {"wall_seconds_max": args.wall_seconds, "gpu_count": 1, "model_calls": 160001,
        "training_steps": 20000, "optimizer_updates_max": 20000, "cpu_workers_max": 2, "cpu_threads_per_worker": 2}
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Continue own complete ICT state to actual20k using explicitly frozen admitted project payload",
        "phase": "HUMANEGO_ICT_FROZEN_PAYLOAD_RESUME", "plan_revision": "ICT_FROZEN_PAYLOAD_RESEARCH_20261012",
        "execution_revision": task_id.upper(), "execution_mode": EXECUTION_MODE, "source_policy": SOURCE_POLICY,
        "training_scope": TRAINING_SCOPE, "branch": args.branch, "resume_mode": args.mode,
        "parent_optimizer_updates": step, "admitted_parent_step": step, "periodic_evidence": periodic,
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
                     "docs/governance/ALGORITHM_CONTRACT.json", "tasks/current/INDEX.json"],
        "read_set_expansions": ["Explicit signed parent/adoption/snapshot and exact route refs only; noNAS sources"],
        "write_set": [str(directory.relative_to(REPO_ROOT)), str(packet_path.relative_to(REPO_ROOT)),
            "tasks/current/INDEX.json", "tasks/receipts/" + task_id.upper() + ".json", "docs/governance",
            "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "OWNER_HANDOFF", "NO_ACTIVE_WRITER", "SEALED_CPU_ADOPTION",
            "SAME_BRANCH_COMPLETE_STATE", "ACTUAL_PROJECT_PAYLOAD_SHA", "EXACT_ROUTE_CLOSURE", "RECENT_CAPACITY"],
        "required_outputs": ["RESULT.json", "POST_VALIDATION.json", "CONTINUE_HERE_ZH.json"],
        "budgets": budgets, "stop_conditions": ["fixed deadline", "SHA drift", "double writer", "ENOSPC"],
        "claim_limit": "Frozen admitted snapshot research; no current NAS raw-source equivalence, validation accuracy or physical robot authorization",
        "inputs": inputs, "runtime_code": closure, "writer": writer,
        "writer_file": str(directory / "WRITER.json"), "packet_path": str(packet_path), "output_dir": str(output),
        "commands": [command], "execution_started_at": now.isoformat(), "deadline_at": deadline.isoformat(),
        "allocated_gpu_ids": [0], "gpu_lease_verified": True, "gpu_lease_path": str(GPU_LEASE)}
    packet["run_signature"] = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": closure,
        "budgets": budgets, "commands": [command], "execution_mode": EXECUTION_MODE,
        "source_policy": SOURCE_POLICY, "resume_mode": args.mode})).hexdigest()
    require(not _validate_packet(packet), "INVALID_FINITE_TASK_PACKET")
    env = infra._environment(output)
    child, registered, leased, errors = None, False, False, []
    started = time.monotonic()
    handlers, stopped = {}, [None]
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, lambda s, f: stopped.__setitem__(0, s))
    try:
        receipt, state, index = infra._snapshot()
        infra._no_active_writer(state, index)
        _handoff(handoff_ref)
        from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire
        acquired = try_acquire(SimpleNamespace(task_id=task_id, attempt_id="attempt_0001",
            executor_epoch=writer["executor_epoch"], priority="CHECKPOINT_TRAINING", gpu_id=0,
            min_free_mib=61440, lease=GPU_LEASE, lock=GPU_LOCK), writer["fencing_token"])
        require(acquired is not None, "WAIT_GPU_RESOURCE_NO_ALGORITHM_STARTED")
        leased = True
        lease_update(packet)
        infra._json_once(directory / "GPU_LEASE_ACQUIRED.json", load_json(GPU_LEASE))
        packet_path.parent.mkdir(parents=True, exist_ok=False)
        infra._json_once(packet_path, packet)
        infra._json_once(directory / "WRITER.json", writer)
        infra._json_once(directory / "FROZEN_INPUTS.json", inputs)
        stamp = now_iso()
        state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": args.attempt,
            "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
            "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": 0,
            "task_packet": artifact_ref(packet_path), "run_signature": packet["run_signature"]})
        state["next_task"] = {"task_id": task_id, "expected_resource": "GPU"}
        index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": artifact_ref(packet_path)["sha256"], "execution_allowed": True,
            "execution_class": "CURRENT_LEDGER_ROUTABLE"})
        _summary(infra, packet, directory)
        _publish(infra, state, index, "ICT_FROZEN_PAYLOAD_RESUME_REGISTERED", receipt["governance_revision"])
        registered = True
        require(stopped[0] is None, "PUBLISHER_SIGNAL_BEFORE_EXECUTION")
        with (directory / "EXECUTION.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-m", "chaoyang.cli", "run", OPERATION, *command_args],
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lease_update(packet, worker_pid=child.pid)
            print(json.dumps({"status": "REGISTERED", "task_id": task_id, "child_pid": child.pid,
                              "parent_optimizer_updates": step, "deadline_at": packet["deadline_at"]}), flush=True)
            last = time.monotonic()
            while child.poll() is None:
                require(stopped[0] is None, "PUBLISHER_SIGNAL_" + str(stopped[0]))
                require(datetime.now(timezone.utc) < deadline, "FINITE_TASK_DEADLINE_EXCEEDED")
                if time.monotonic() - last >= 60:
                    lease_update(packet, worker_pid=child.pid)
                    receipt, state, index = infra._snapshot()
                    row, _ = infra._own_row(packet, state, index)
                    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                    _summary(infra, packet, directory)
                    _publish(infra, state, index, "ICT_FROZEN_PAYLOAD_RESUME_HEARTBEAT", receipt["governance_revision"])
                    last = time.monotonic()
                time.sleep(2)
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            infra._stop_child(child)
        except BaseException as error:
            errors.append("CHILD_CLEANUP:" + str(error))
        reaped = _owned_child_reaped(child)
        if leased and reaped:
            try:
                lease_update(packet, release_reason="FINITE_FROZEN_PAYLOAD_EXIT_" + str(child.returncode if child else "NO_CHILD"))
                infra._json_once(directory / "GPU_LEASE_RELEASED.json", load_json(GPU_LEASE))
            except BaseException as error:
                errors.append("LEASE_RELEASE:" + str(error))
    try:
        if not reaped:
            # Do NOT mark this occupied resource released or clear the live
            # task/index. A separate explicit reaping/reconciliation must
            # prove the entire owned group gone before any successor starts.
            infra._json_once(directory / 'CHILD_REAP_PENDING.json', {
                'status': 'REAP_PENDING_NO_RESOURCE_RELEASE', 'task_id': task_id,
                'run_signature': packet['run_signature'], 'child_pid': child.pid,
                'owned_group_members': _group_members(child.pid), 'failure_codes': errors,
                'lease_release_skipped': True, 'ledger_not_cleared': True,
                'requires_explicit_resource_reconciliation': True})
            return 2
        if not registered:
            infra._json_once(directory / "REGISTRATION_FAILURE.json", {"task_id": task_id, "algorithm_started": False,
                            "failure_codes": errors, "status": "FAILED_REGISTRATION"})
            return 2
        try:
            result = infra._collect_result(packet, directory, started, child, errors, step)
        except BaseException as error:
            # Preserve failed readback without clearing the writer or inventing
            # a checkpoint. Explicit reconciliation owns any later release.
            infra._json_once(directory / 'FINALIZATION_ERROR.json', {
                'status': 'FAILED_RESULT_COLLECTION_RECONCILIATION_REQUIRED',
                'task_id': task_id, 'run_signature': packet['run_signature'],
                'child_reaped': True, 'failure_codes': errors + [type(error).__name__ + ':' + str(error)],
                'ledger_not_cleared': True})
            return 2
        try:
            return _seal(infra, packet, directory, result, env)
        except BaseException as error:
            infra._json_once(directory / 'PUBLICATION_ERROR.json', {
                'status': 'FAILED_PUBLICATION_RECONCILIATION_REQUIRED',
                'task_id': task_id, 'run_signature': packet['run_signature'],
                'failure_codes': [type(error).__name__ + ':' + str(error)],
                'recheck_actual_ledger_and_result_before_reconciliation': True})
            return 2
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(register_main())
