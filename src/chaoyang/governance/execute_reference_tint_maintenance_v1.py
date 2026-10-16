"""Independent finite reference-tint Render-only maintenance executor. Original executor is immutable.

One spec file under ``docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/specs`` describes one
finite task.  ``--execute`` registers the packet through the governance CAS,
runs exactly one ``chaoyang run`` command as a fenced descendant, heartbeats,
and seals an immutable terminal result.  ``--recover`` seals a task whose writer
died (container restart, OOM, lost session) so that a successor can be routed.
"""

from __future__ import annotations

import argparse

import copy

import hashlib

import json

import os

from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

from pathlib import Path

import secrets

import shlex

import signal

import subprocess

import time

from typing import Any

from chaoyang.governance.common_candidate_optimization_v1 import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)

from chaoyang.governance.register_single_task_packet import _validate_packet

SCRIPT = Path(__file__)

ROUTE_KEY = "pick_place_reference_tint_render_only_maintenance_v1"

TASK_PREFIX = "exact78_pick_place_reference_tint_render_only_hold007199_maintenance_"

OPERATION_PREFIX = "run_exact78_pick_place_"

ALLOWED_OPERATIONS = frozenset({"run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v1"})

PLAN_REVISION = "PICK_PLACE_1008_RAW_TO_ROBOT_V1"

SPEC_ROOT = REPO_ROOT / "docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/specs"

PACKAGE_ROOT = REPO_ROOT / "src/chaoyang/pick_place_1008"

INDEX = REPO_ROOT / "tasks/current/INDEX.json"

PYTHON = "/usr/local/bin/python"

HEARTBEAT_SECONDS = 50

ACTIVE_STATUSES = ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")

GOVERNANCE_CODE = (
    "src/chaoyang/governance/execute_pick_place_1008_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/exact78_visual_selection_v1.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/__main__.py",
    "src/chaoyang/paths.py",
)

SPEC_FIELDS = {
    "task_id", "operation", "args", "objective", "phase", "read_set",
    "budgets", "required_outputs", "claim_limit", "stop_conditions",
}

class WorkerNotStoppedError(RuntimeError):
    pass

def _paths(task_id: str) -> dict[str, Path]:
    output = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    return {
        "output": output,
        "stage": output / "candidate",
        "packet": REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json",
        "receipt": REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json",
    }

def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

def _project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise RuntimeError(f"PICK_PLACE_PATH_ESCAPES_PROJECT:{raw}")
    return resolved

def _input_ref(raw: str) -> dict[str, Any]:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return artifact_ref(path)

def _governance_pass() -> dict[str, Any]:
    result = subprocess.run(
        [PYTHON, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, timeout=600, check=False)
    try:
        value = json.loads(result.stdout)
    except ValueError:
        value = {"status": "UNPARSEABLE", "stdout": result.stdout[-400:], "stderr": result.stderr[-400:]}
    return value

def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}

def register(spec: dict[str, Any], packet: dict[str, Any]) -> None:
    task_id = packet["task_id"]
    paths = _paths(task_id)
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": spec.get("session"),
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": spec.get("gpu_id"), "task_packet": _virtual_ref(paths["packet"], packet),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": task_id, "session": spec.get("session"),
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": f"GPU_{packet['budgets'].get('gpu_count', 0)}",
                          "stop_condition": "one attempt or hard deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("PICK_PLACE_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    index["task_packets"].append({
        "task_id": task_id, "packet_path": str(paths["packet"].relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(paths["packet"], packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=f"{task_id.upper()}_RUNNING", plan_revision=PLAN_REVISION,
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    paths["output"].mkdir(parents=True)
    atomic_json(paths["output"] / "WRITER.json", packet["writer"])
    paths["packet"].parent.mkdir(parents=True)
    atomic_json(paths["packet"], packet)
    if artifact_ref(paths["packet"])["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("PICK_PLACE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="PICK_PLACE_TASK_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)

def verify_fence(packet: dict[str, Any]) -> None:
    task_id = packet["task_id"]
    writer = packet["writer"]
    paths = _paths(task_id)
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(paths["output"] / "WRITER.json") != writer):
        raise RuntimeError("PICK_PLACE_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"] if row["status"] in ACTIVE_STATUSES]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != task_id
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != task_id
            or routed[0]["packet_sha256"] != artifact_ref(paths["packet"])["sha256"]):
        raise RuntimeError("PICK_PLACE_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")

def heartbeat(packet: dict[str, Any]) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="PICK_PLACE_TASK_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)

def _stop_fenced(worker: subprocess.Popen[Any], ticks: int) -> None:
    if worker.poll() is not None:
        return
    try:
        if process_identity(worker.pid)["start_ticks"] != ticks:
            raise WorkerNotStoppedError("PICK_PLACE_WORKER_IDENTITY_LOST")
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if process_identity(worker.pid)["start_ticks"] != ticks:
                raise WorkerNotStoppedError("PICK_PLACE_WORKER_IDENTITY_LOST")
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
    except (ProcessLookupError, ChildProcessError, OSError, KeyError,
            ValueError, subprocess.TimeoutExpired):
        if worker.poll() is None:
            raise WorkerNotStoppedError("PICK_PLACE_WORKER_STOP_UNPROVEN")

class GpuResourceWaitError(RuntimeError):
    """The finite resource wait ended before any algorithm worker began."""

def _resource_status(packet: dict[str, Any], status: str, reason: str | None) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    if row["status"] == status:
        return
    row.update(status=status, updated_at=now_iso(), heartbeat_at=now_iso(),
               resource_wait_reason=reason, algorithm_attempts_consumed=0)
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=f"PICK_PLACE_TASK_{status}",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)

def _gpu_wait_reason(packet: dict[str, Any]) -> str | None:
    if int(packet["budgets"].get("gpu_count", 0)) == 0:
        return None
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    if lease_path.is_file():
        lease = load_json(lease_path)
        if lease.get("status") != "RELEASED":
            return "CENTRAL_GPU_LEASE_UNAVAILABLE"
    from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import (
        query_gpu_memory_mib, query_gpu_pids,
    )
    pids = query_gpu_pids()
    memory = query_gpu_memory_mib(int(packet["config"].get("gpu_id", 0)))
    if pids is None or memory is None:
        return "PHYSICAL_GPU_EVIDENCE_UNAVAILABLE"
    if memory["free_mib"] < int(packet["config"].get("gpu_min_free_mib", 40960)):
        return "INSUFFICIENT_GPU_FREE_MEMORY"
    return None

def wait_for_gpu_resource(packet: dict[str, Any], deadline_monotonic: float) -> None:
    last_heartbeat = time.monotonic()
    while reason := _gpu_wait_reason(packet):
        _resource_status(packet, "WAIT_GPU_RESOURCE", reason)
        if time.monotonic() >= deadline_monotonic:
            raise GpuResourceWaitError(f"WAIT_GPU_RESOURCE_BUDGET_EXHAUSTED:{reason}")
        if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
            heartbeat(packet)
            last_heartbeat = time.monotonic()
        time.sleep(min(5.0, max(0.1, deadline_monotonic - time.monotonic())))
    _resource_status(packet, "RUNNING", None)

def _resource_wait_evidence(paths: dict[str, Path], code: int,
                            reason: str | None) -> bool:
    if reason and reason.startswith("GpuResourceWaitError:"):
        return True
    resource_path = paths["stage"] / "RESOURCE_WAIT.json"
    if code != 75 or not resource_path.is_file():
        return False
    try:
        value = load_json(resource_path)
    except (OSError, ValueError):
        return False
    return (value.get("status") == "WAIT_GPU_RESOURCE"
            and value.get("algorithm_started") is False)

def _packet_command_tokens(packet: dict[str, Any]) -> list[str]:
    tokens = shlex.split(packet["commands"][0])
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens.pop(0)
    return tokens

def _write_worker_identity(packet: dict[str, Any], worker_pid: int,
                           ticks: int, command: list[str]) -> None:
    path = _paths(packet["task_id"])["output"] / "WORKER.json"
    if path.exists():
        raise RuntimeError("PICK_PLACE_WORKER_IDENTITY_ALREADY_EXISTS")
    atomic_json(path, {
        "schema_version": "pick-place-fenced-worker-v1", "task_id": packet["task_id"],
        "pid": worker_pid, "proc_start_ticks": ticks, "command": command,
        "packet_command": packet["commands"][0], "run_signature": packet["run_signature"],
        "writer": packet["writer"], "recorded_at": now_iso(),
    })

def _verify_recovery_worker(packet: dict[str, Any], paths: dict[str, Path]
                            ) -> dict[str, Any] | None:
    path = paths["output"] / "WORKER.json"
    if not path.is_file():
        if (paths["output"] / "worker.log").exists():
            raise RuntimeError("PICK_PLACE_RECOVER_WORKER_IDENTITY_UNPROVEN")
        return None
    worker = load_json(path)
    if (worker.get("task_id") != packet["task_id"]
            or worker.get("run_signature") != packet["run_signature"]
            or worker.get("command") != _packet_command_tokens(packet)
            or worker.get("packet_command") != packet["commands"][0]
            or worker.get("writer") != packet["writer"]
            or type(worker.get("pid")) is not int or worker["pid"] <= 0
            or type(worker.get("proc_start_ticks")) is not int or worker["proc_start_ticks"] < 0):
        raise RuntimeError("PICK_PLACE_RECOVER_WORKER_IDENTITY_DRIFT")
    identity = process_identity(worker["pid"])
    if identity["alive"] and identity["start_ticks"] == worker["proc_start_ticks"]:
        raise RuntimeError("PICK_PLACE_RECOVER_WORKER_ALIVE")
    return {"worker": artifact_ref(path), "recorded_identity": worker,
            "current_identity": identity, "original_worker_stopped": True}

def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    paths = _paths(packet["task_id"])
    budgets = packet["budgets"]
    cache = paths["output"] / "cache"
    cache.mkdir()
    threads = str(min(2, int(budgets.get("cpu_threads_total", 4))))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
               NUMBA_CACHE_DIR=str(cache / "numba"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
               OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    if int(budgets.get("gpu_count", 0)) == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    wall = int(budgets.get("wall_seconds_max", 1800))
    budget_started = time.monotonic()
    wait_for_gpu_resource(packet, budget_started + wall)
    deadline = datetime.fromisoformat(packet["deadline_at"])
    log = paths["output"] / "worker.log"
    tokens = _packet_command_tokens(packet)
    started = budget_started
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(tokens, cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ticks = process_identity(worker.pid)["start_ticks"]
            _write_worker_identity(packet, worker.pid, ticks, tokens)
        except (OSError, KeyError, ValueError) as exc:
            if worker.poll() is None:
                raise WorkerNotStoppedError("PICK_PLACE_WORKER_START_IDENTITY_LOST") from exc
            ticks = -1
        last = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= wall or datetime.now().astimezone() >= deadline:
                    raise RuntimeError("PICK_PLACE_HARD_DEADLINE")
                if elapsed - last >= HEARTBEAT_SECONDS:
                    heartbeat(packet)
                    last = elapsed
                time.sleep(2)
        finally:
            _stop_fenced(worker, ticks)
    return worker.returncode, log

def _seal(packet: dict[str, Any], status: str, payload: dict[str, Any], event: str) -> dict[str, Any]:
    task_id = packet["task_id"]
    paths = _paths(task_id)
    if (paths["output"] / "RESULT.json").exists() or paths["receipt"].exists():
        raise RuntimeError("PICK_PLACE_DUPLICATE_TERMINAL")
    result = {"schema_version": "pick-place-1008-execution-result-v1", "task_id": task_id,
              "status": status, "task_packet": artifact_ref(paths["packet"]),
              "run_signature": packet["run_signature"], "visual_qualified": False,
              "training_eligible": False, "control_ground_truth": False,
              "adoption": "NOT_ADOPTED", "user_review": "PENDING",
              "files_deleted": False, **payload}
    atomic_json(paths["output"] / "RESULT.json", result)
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
    atomic_json(paths["receipt"], {"schema_version": "pick-place-1008-receipt-v1",
                                   "task_id": task_id, "status": status,
                                   "result": artifact_ref(paths["output"] / "RESULT.json"),
                                   "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == task_id)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(paths["output"] / "RESULT.json"),
               terminal_receipt=artifact_ref(paths["receipt"]),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if status == "PASSED" else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    if (state.get("next_task") or {}).get("task_id") == task_id:
        state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("PICK_PLACE_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == task_id:
            item["execution_allowed"] = False
            item["result"] = artifact_ref(paths["output"] / "RESULT.json")
            item["terminal_receipt"] = artifact_ref(paths["receipt"])
            item["required_outputs"] = copy.deepcopy(payload.get("required_outputs", {}))
    index.update(packet_revision=f"{task_id.upper()}_TERMINAL", plan_revision=PLAN_REVISION,
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=SCRIPT, task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(paths["output"] / "RESULT.json")}

def recover(task_id: str) -> dict[str, Any]:
    paths = _paths(task_id)
    if not task_id.startswith(TASK_PREFIX):
        raise RuntimeError("PICK_PLACE_TASK_PREFIX")
    packet = load_json(paths["packet"])
    state = load_json(TASK_STATE_PATH)
    row = next((item for item in state["tasks"] if item["task_id"] == task_id), None)
    if row is None or row["status"] not in ACTIVE_STATUSES:
        raise RuntimeError("PICK_PLACE_RECOVER_TASK_NOT_ACTIVE")
    identity = process_identity(int(row["pid"])) if row.get("pid") is not None else {"alive": False}
    if identity["alive"] and identity["start_ticks"] == row.get("proc_start_ticks"):
        raise RuntimeError("PICK_PLACE_RECOVER_WRITER_ALIVE")
    if (packet.get("task_id") != task_id
            or row.get("run_signature") != packet.get("run_signature")
            or row.get("task_packet") != artifact_ref(paths["packet"])
            or load_json(paths["output"] / "WRITER.json") != packet.get("writer")):
        raise RuntimeError("PICK_PLACE_RECOVER_PACKET_FENCE_DRIFT")
    worker_evidence = _verify_recovery_worker(packet, paths)
    outputs, output_errors = _collect_required_outputs(packet, paths)
    worker_log = paths["output"] / "worker.log"
    payload = {"worker_returncode": None, "failure_code": "INTERRUPTED_WRITER_LOST",
               "validation_errors": output_errors, "recovered_at": now_iso(),
               "required_outputs": outputs, "worker_evidence": worker_evidence,
               "stage_result": artifact_ref(paths["stage"] / "RESULT.json") if (paths["stage"] / "RESULT.json").is_file() else None,
               "lost_writer": {"pid": row.get("pid"), "proc_start_ticks": row.get("proc_start_ticks"),
                               "last_heartbeat_at": row.get("heartbeat_at")},
               "worker_log": artifact_ref(worker_log) if worker_log.is_file() else None,
               "partial_outputs_are_successor_inputs": False,
               "next_action": "Publish a new successor task id; reuse only sealed SHA-matched inputs"}
    return _seal(packet, "FAILED_RUNTIME_FINAL", payload, "PICK_PLACE_TASK_RECOVERED_INTERRUPTED")

def _route_drift_files() -> set[str]:
    files = {str(path.resolve()) for path in (*takeover_operations(), *package_code())}
    files |= {str((REPO_ROOT / name).resolve()) for name in (
        "src/chaoyang/governance/current_r3_contracts.py",
        "src/chaoyang/governance/execute_pick_place_1008_v1.py",
        "src/chaoyang/governance/exact78_visual_selection_v1.py")}
    return files

from chaoyang.governance.reference_tint_maintenance_contract_v1 import (OPERATION,TASK,FAMILY,BUDGETS,approved_context)

def load_spec(spec_path):
    return approved_context(spec_path)['spec']

def packet_from_context(context, *, preparation=False):
    from chaoyang.governance.reference_tint_maintenance_contract_v1 import expected_runtime, plan_shape
    spec, plan = context['spec'], context['plan']
    plan_shape(plan, spec, verify=not preparation)
    if not preparation and plan['publication_status']!='ROOT_PLANNED_FINITE_OPERATION_APPROVED':
        raise ValueError('TINT_ROOT_APPROVAL_PENDING')
    runtime=expected_runtime(context,verify=not preparation)
    paths=_paths(TASK);budgets=dict(BUDGETS)
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    writer={'pid':os.getpid(),'proc_start_ticks':process_identity(os.getpid())['start_ticks'],'executor_epoch':1,'fencing_token':secrets.token_hex(24),'scope':TASK}
    tail=[OPERATION,'--maintenance-spec',plan['approved_spec']['path']]
    command=f'PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run '+shlex.join(tail)
    inputs={'spec':plan['approved_spec'],'files':[spec['contract']],'maintenance_render_plan':context['plan_ref']}
    config={'session':'HOLD007199_RENDER_ONLY','budgets':budgets,'numeric_gates_changed':False,'deletion_authorized':False,'gpu_id':0,'gpu_min_free_mib':0}
    signature=hashlib.sha256(canonical_bytes({'inputs':inputs,'runtime_code':runtime,'config':config,'command':command,'schema':'reference-tint-render-only-maintenance-task-v1'})).hexdigest()
    packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK,'objective':'Finite HOLD007199 no-fit reference-tint lookup maintenance with full native pixel/PTS/contact/flags/FK equality','phase':'pick_place_1008_reference_tint_render_only_maintenance','plan_revision':PLAN_REVISION,'execution_revision':TASK.upper(),'execution_mode':'EXACT78_CANDIDATE','registered_at':now.isoformat(),'deadline_at':(now+timedelta(seconds=BUDGETS['wall_seconds_max']+300)).isoformat(),'read_set':[plan['approved_spec']['path'],context['plan_ref']['path'],spec['contract']['path'],context['contract']['original_queue']['path'],context['contract']['manifest']['path'],context['contract']['camera_config']['path'],context['contract']['original_HOLD_producer']['packet']['path'],context['contract']['original_HOLD_producer']['outer']['path']],'write_set':[str(paths['output'].relative_to(REPO_ROOT)),str(paths['packet'].relative_to(REPO_ROOT)),str(paths['receipt'].relative_to(REPO_ROOT)),'tasks/current/INDEX.json','docs/governance','review/pick_place_1008'],'prerequisites':['GOVERNANCE_PASS','NO_ACTIVE_TASK','STRICT_RELEASED_GPU_LEASE','ROOT_EXPLICIT_NEW_FAMILY_DESCRIPTOR','FULL_OLD_SOURCE_HOLD_GLOBAL_PASSED','EXACT_OLD93_PLUS_SIX_PLUS_SEVEN_RUNTIME','HOLD_NO_FIT','NO_DELETION'],'required_outputs':['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'budgets':budgets,'stop_conditions':['CODE_OR_INPUT_SHA_DRIFT','DOUBLE_WRITER','DEADLINE','ONE_ATTEMPT_TERMINAL','ANY_NATIVE_PIXEL_PTS_CONTACT_FLAGS_FK_DIFFERENCE','NO_AUTOMATIC_AI_ADOPTION_OR_RETRY'],'claim_limit':'Render-only offline maintenance equality candidate; source/q/Clean/masks/camera unchanged; contact UNKNOWN, human pending, no control/physical/training qualification.','writer':writer,'inputs':inputs,'runtime_code':runtime,'config':config,'run_signature':signature,'commands':[command]}
    if errors:=_validate_packet(packet):raise RuntimeError('TINT_ACTUAL_PACKET_SCHEMA_PREFLIGHT:'+';'.join(errors))
    return packet

def packet_value(spec_path,spec):
    context=approved_context(spec_path)
    if context['spec']!=spec:raise ValueError('TINT_APPROVED_SPEC_VALUE_DRIFT')
    return packet_from_context(context)

def readiness(spec):
    if spec.get('task_id')!=TASK:raise ValueError('TINT_ONLY_EXPLICIT_SINGLE_TASK')
    from chaoyang.governance.candidate_optimization_family_v1 import assert_registration_boundary
    # Registering a task retains the already published descriptor; no family is added.
    state=load_json(TASK_STATE_PATH)
    if state.get('next_task') or any(r['status'] in ACTIVE_STATUSES for r in state['tasks']):raise RuntimeError('TINT_ACTIVE_TASK_CONFLICT')
    if any(r.get('execution_allowed') for r in load_json(INDEX)['task_packets']):raise RuntimeError('TINT_ACTIVE_PACKET_CONFLICT')
    if load_json(REPO_ROOT/'_run/current/GPU_LEASE.json').get('status')!='RELEASED':raise RuntimeError('TINT_STRICT_RELEASED_LEASE_REQUIRED')
    if _governance_pass().get('status')!='PASS':raise RuntimeError('TINT_GOVERNANCE_NOT_PASS')
    paths=_paths(TASK)
    if any(paths[k].exists() for k in ('output','packet','receipt')):raise RuntimeError('TINT_FRESH_ONE_ATTEMPT_REQUIRED')

def _collect_required_outputs(packet,paths):
    outputs,errors={},[]
    for name in packet['required_outputs']:
        if not name.startswith('candidate/'):continue # RESULT/terminal_receipt are envelope refs, independently sealed.
        p=(paths['output']/name).resolve()
        if not p.is_relative_to(paths['output'].resolve()):raise RuntimeError('TINT_REQUIRED_OUTPUT_ESCAPES_ATTEMPT')
        if p.is_file():outputs[name]=artifact_ref(p)
        else:errors.append('PICK_PLACE_REQUIRED_OUTPUT_MISSING:'+name)
    return outputs,errors

def finish(packet,code,log,elapsed,reason=None):
    verify_fence(packet)
    paths=_paths(packet['task_id']);outputs,errors=_collect_required_outputs(packet,paths)
    for pin in packet['runtime_code']+[packet['inputs']['spec'],packet['inputs']['maintenance_render_plan'],*packet['inputs']['files']]:
        errors.extend(validate_artifact_ref(pin))
    validation=None;collector_elapsed=0.0
    if code==0 and not errors and reason is None:
        t=time.monotonic()
        try:
            from chaoyang.governance.collect_reference_tint_maintenance_v1 import collect
            validation=collect(packet)
            cp=paths['output']/'INDEPENDENT_MAINTENANCE_COLLECTOR_VALIDATION.json'
            if cp.exists():raise RuntimeError('TINT_COLLECTOR_SINGLE_SEAL_REQUIRED')
            atomic_json(cp,validation)
        except Exception as exc:errors.append('TINT_INDEPENDENT_COLLECTOR_FAILED:'+type(exc).__name__+':'+str(exc)[:600])
        collector_elapsed=time.monotonic()-t
    total=elapsed+collector_elapsed
    if total>BUDGETS['wall_seconds_max']:errors.append('TINT_OUTER_WALL_BUDGET_EXCEEDED')
    status='PASSED' if code==0 and not errors and reason is None and validation is not None else 'FAILED_RUNTIME_FINAL'
    payload={'worker_returncode':code,'failure_code':reason,'validation_errors':errors,'elapsed_seconds':round(total,6),'worker_and_guard_wall_seconds':round(elapsed,6),'independent_collector_wall_seconds':round(collector_elapsed,6),'required_outputs':outputs,'algorithm_attempts_consumed':1,'resource_wait':None,'stage_result':artifact_ref(paths['stage']/'RESULT.json') if (paths['stage']/'RESULT.json').is_file() else None,'worker_log':artifact_ref(log),'maintenance_collector':artifact_ref(paths['output']/'INDEPENDENT_MAINTENANCE_COLLECTOR_VALIDATION.json') if validation is not None and not errors else None,'new_visual_quality':'PENDING_SERVER_LOCAL_ACTUAL_REVIEW','full_pipeline_speedup':'UNKNOWN_NOT_MEASURED','next_action':'Root review independent equality and actual timing; no automatic visual adoption or retry'}
    return _seal(packet,status,payload,'REFERENCE_TINT_MAINTENANCE_'+status)

def publish_route(descriptor_path):
    from chaoyang.governance.candidate_optimization_family_v1 import descriptor,ref
    pin=ref(Path(descriptor_path));d=descriptor(pin)
    if d['family_key']!=FAMILY:raise ValueError('TINT_PUBLISHER_ONLY_OWN_EXPLICIT_FAMILY')
    receipt=publish_bundle(load_json(AUTHORITY_PATH),copy.deepcopy(load_json(TASK_STATE_PATH)),event_type='ROOT_EXPLICIT_REFERENCE_TINT_MAINTENANCE_FAMILY_PUBLISHED',expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=SCRIPT,planned_family_descriptor_ref=pin)
    after=_governance_pass()
    return {'status':after.get('status'),'errors_after':after.get('errors',[]),'governance_revision':receipt['governance_revision']}

def preparation_check(spec_path,draft_plan_path):
    spec=load_json(spec_path);plan=load_json(draft_plan_path)
    if plan.get('publication_status')!='DRAFT_ROOT_APPROVAL_REQUIRED_NOT_AUTHORIZED':raise ValueError('TINT_PREPARATION_ONLY_DRAFT_REQUIRED')
    from chaoyang.governance.reference_tint_maintenance_contract_v1 import plan_shape
    plan_shape(plan,spec,verify=False)
    context={'spec':spec,'plan':plan,'plan_ref':artifact_ref(draft_plan_path),'contract':load_json(Path(spec['contract']['path']))}
    packet=packet_from_context(context,preparation=True)
    return packet,{'status':'PRIVATE_PACKET_SCHEMA_PASSED_FUTURE_ROOT_APPROVAL_AND_INSTALLATION_PENDING','formal_READY':False,'task_id':TASK,'budgets':packet['budgets'],'runtime_code_files':len(packet['runtime_code']),'actual_R22_packet_validator_errors':_validate_packet(packet),'operation':OPERATION,'command':packet['commands'][0],'publish_called':False,'register_called':False,'model_calls':0,'render_calls':0}

def main(argv=None):
    parser=argparse.ArgumentParser(description='Independent single finite Root-approved Render-only maintenance task')
    c=parser.add_mutually_exclusive_group(required=True)
    c.add_argument('--check-only',type=Path)
    c.add_argument('--execute',type=Path)
    c.add_argument('--recover')
    c.add_argument('--publish-route',type=Path,metavar='ROOT_APPROVED_DESCRIPTOR')
    c.add_argument('--preparation-check',type=Path)
    parser.add_argument('--draft-plan',type=Path)
    args=parser.parse_args(argv)
    if args.preparation_check:
        if not args.draft_plan:parser.error('--draft-plan is required for preparation-only CPU check')
        packet,value=preparation_check(args.preparation_check,args.draft_plan)
        print(json.dumps(value,ensure_ascii=False,sort_keys=True));return 0
    if args.draft_plan:parser.error('--draft-plan is only accepted with --preparation-check')
    if args.recover:
        if args.recover!=TASK:raise ValueError('TINT_RECOVER_ONLY_EXPLICIT_TASK')
        print(json.dumps(recover(args.recover),sort_keys=True));return 0
    if args.publish_route:
        value=publish_route(args.publish_route);print(json.dumps(value,sort_keys=True));return 0 if value['status']=='PASS' else 1
    path=(args.check_only or args.execute).resolve()
    try:
        spec=load_spec(path);packet=packet_value(path,spec);readiness(spec)
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        if not args.check_only:raise
        print(json.dumps({'status':'DRY_RUN_NOT_AUTHORIZED_OR_NOT_ROUTABLE','formal_READY':False,'reason':str(exc)[:600],'task_id':TASK,'register_called':False,'model_calls':0,'render_calls':0}));return 0
    if args.check_only:
        print(json.dumps({'status':'DRY_RUN_READY','formal_READY':True,'task_id':TASK,'command':packet['commands'][0],'budgets':packet['budgets'],'run_signature':packet['run_signature'],'runtime_code_files':len(packet['runtime_code'])},sort_keys=True));return 0
    register(spec,packet);started=time.monotonic()
    try:
        from chaoyang.cli import _maintained_operations
        if OPERATION not in _maintained_operations():raise RuntimeError('TINT_OPERATION_NOT_IN_ROOT_PUBLISHED_CURRENT_CLI')
        code,log=run_worker(packet);reason=None
    except Exception as exc:
        if isinstance(exc,WorkerNotStoppedError):raise
        code=1;reason=type(exc).__name__+':'+str(exc)[:600];log=_paths(TASK)['output']/'executor_error.log'
        with log.open('a') as f:f.write(reason+'\n')
    value=finish(packet,code,log,time.monotonic()-started,reason)
    print(json.dumps(value,sort_keys=True));return 0 if value['status']=='PASSED' else 1

if __name__=='__main__':raise SystemExit(main())
