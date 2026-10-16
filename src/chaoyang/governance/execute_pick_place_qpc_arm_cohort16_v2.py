"""Generic fenced Task Packet executor for the Exact78 takeover line (2026-10-04).

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

from chaoyang.governance.common_candidate_optimization_v3 import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


SCRIPT = Path(__file__)
ROUTE_KEY = 'pick_place_qpc_arm_cohort16_cpu_candidate_v1'
TASK_PREFIX = "exact78_pick_place_"
OPERATION_PREFIX = 'run_pick_place_qpc_arm_cohort16_'
ALLOWED_OPERATIONS = frozenset(['run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1', 'run_pick_place_qpc_arm_cohort16_motion_render_candidate_v1'])
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



EXTRA_RUNTIME_PATHS=(
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/__init__.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/approval.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/runtime.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/bindings.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/motion_render.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/collector.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/release_v2.py",
 "src/chaoyang/pick_place_qpc_arm_cohort16_v1/output_audit.py",
 "src/chaoyang/ops/run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1.py",
 "src/chaoyang/ops/run_pick_place_qpc_arm_cohort16_motion_render_candidate_v1.py",
 "src/chaoyang/governance/execute_pick_place_qpc_arm_cohort16_v2.py",
 "src/chaoyang/governance/common_candidate_optimization_v3.py",
 "src/chaoyang/governance/current_candidate_optimization_contracts_v2.py",
 "src/chaoyang/governance/candidate_optimization_family_v2.py")
def family_code():return [REPO_ROOT/x for x in EXTRA_RUNTIME_PATHS]
def fixed_existing_code():
 from chaoyang.pick_place_qpc_arm_cohort16_v1.approval import fixed_existing_code as pins
 return [Path(x['path']) for x in pins()]
def approved_spec_inputs(spec):
 from chaoyang.pick_place_qpc_arm_cohort16_v1.approval import verify_runtime_approval,ref,EVALUATION,MOTION_RENDER
 def arg(k):
  if spec['args'].count(k)!=1:raise RuntimeError('QP_EXACT_ARG_REQUIRED:'+k)
  return spec['args'][spec['args'].index(k)+1]
 approval,contract=verify_runtime_approval(arg('--root-runtime-approval'),argv=spec['args'])
 if spec['task_id']!=approval['expected_task_id'] or spec['operation']!=approval['operation'] or spec['budgets']!=approval['expected_budgets'] or ref(arg('--contract'))!=approval['contract_ref'] or ref(arg('--cohort'))!=contract['cohort']:
  raise RuntimeError('QP_FORMAL_SPEC_NOT_ROOT_PLAN')
 if sorted(approval['exact_extra_runtime_code'],key=lambda x:x['path'])!=sorted([ref(x) for x in family_code()],key=lambda x:x['path']):raise RuntimeError('QP_EXACT_NEW_RUNTIME_WHITELIST_DRIFT')
 if any(ref(p) not in [ref(_project_path(x)) for x in spec.get('inputs',[])] for p in fixed_existing_code()) or approval['operation_plan'] not in [ref(_project_path(x)) for x in spec.get('inputs',[])]:raise RuntimeError('QP_FIXED_EXECUTED_CODE_AND_PLAN_NOT_DIRECT_SPEC_INPUTS')
 if spec.get('code') or len(spec['read_set'])>10 or spec['required_outputs']!=['INDEX.json','RESULT.json']:raise RuntimeError('QP_SPEC_EXTRA_OR_OUTPUT_BOUNDS')
 if approval['operation']==MOTION_RENDER:
  from chaoyang.pick_place_qpc_arm_cohort16_v1.collector import evaluation_selection
  evaluation_selection(approval)
 return approval,contract
def approved_release_boundary(path):
 from chaoyang.pick_place_qpc_arm_cohort16_v1.release_v2 import approved_idle
 result=approved_idle(path)
 before=_governance_pass()
 if before.get('status')!='PASS':raise RuntimeError('QP_RELEASE_GOVERNANCE_NOT_PASS:'+json.dumps(before))
 return result

def takeover_operations() -> list[Path]:
    return sorted(path for path in (REPO_ROOT / "src/chaoyang/ops").glob(f"{OPERATION_PREFIX}*.py")
                  if path.stem in ALLOWED_OPERATIONS)


def package_code() -> list[Path]:
    if not PACKAGE_ROOT.is_dir():
        return []
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def route_contract(repo_root: Path) -> dict[str, Any]:
    del repo_root
    closure=[artifact_ref(x) for x in (*takeover_operations(),*package_code(),*family_code(),*fixed_existing_code())]
    return {'execution_status':'ROOT_APPROVED_ONE_FINITE_CPU_PACKET_ONLY','candidate_operations':sorted(ALLOWED_OPERATIONS),
        'task_packet_required':True,'scope':'FIXED16_2578_QPC_ARM_CPU_CANDIDATE','plan_revision':PLAN_REVISION,
        'executor':str(SCRIPT.resolve()),'code_closure':list({x['path']:x for x in closure}.values()),
        'output_schema':'pick-place-1008-execution-result-v1','GPU_authorized':0,'model_calls_authorized':0,
        'render_calls_authorized':'ONLY_EXPLICIT_MOTION_RENDER_PACKET_MAX5156','training_steps_authorized':0,
        'current_runnable_operation':None,'source_GPU_exact_reuse_only':True,
        'newMotion_Render_Contact_same_saved_q_required':True,
        'maintenance_render_operation_allowlist':{op:{'plan':artifact_ref(REPO_ROOT/'review/pick_place_1008/QPC_ARM_COHORT16_GOVERNED_PREPARATION_V1'/('ROOT_EVALUATION_OPERATION_PLAN_REQUIRED_V1.json' if 'cpu_evaluation' in op else 'ROOT_MOTION_RENDER_OPERATION_PLAN_REQUIRED_V1.json')),'execution_allowed':True} for op in sorted(ALLOWED_OPERATIONS)},
        'claim_limit':'Internal derived-target/rate candidates; no physical/contact/control/training/visual acceptance.'}


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


def load_spec(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    if spec_path.parent != SPEC_ROOT.resolve():
        raise RuntimeError("PICK_PLACE_SPEC_OUTSIDE_SPEC_ROOT")
    spec = load_json(spec_path)
    if spec.get("preparation_only") or spec.get("execution_allowed") is False:
        raise RuntimeError("QP_PRIVATE_DRAFT_IS_NOT_EXECUTABLE")
    missing = sorted(SPEC_FIELDS - set(spec))
    if missing:
        raise RuntimeError(f"PICK_PLACE_SPEC_MISSING:{missing}")
    if not str(spec["task_id"]).startswith(TASK_PREFIX):
        raise RuntimeError("PICK_PLACE_TASK_PREFIX")
    if spec["operation"] not in ALLOWED_OPERATIONS:
        raise RuntimeError("PICK_PLACE_OPERATION_NOT_ALLOWED")
    if any(arg == "--output-root" or str(arg).startswith("--output-root=")
           for arg in spec["args"]):
        raise RuntimeError("PICK_PLACE_OUTPUT_ROOT_IS_EXECUTOR_OWNED")
    if spec_path.stem != spec["task_id"]:
        raise RuntimeError("PICK_PLACE_SPEC_NAME_MISMATCH")
    return spec


def parent_algorithm_code() -> list[Path]:
    """Freeze selected parent algorithms without editing their published bytes."""
    manifest = load_json(REPO_ROOT / "baseline_v1/MANIFEST.json")
    paths = []
    for row in manifest["sources"]:
        if row["role"] == "RUNTIME_PLATFORM_DEPENDENCY_ONLY":
            continue
        if errors := validate_artifact_ref(row["source"]):
            raise RuntimeError("PICK_PLACE_PARENT_ALGORITHM_SHA_DRIFT:" + ";".join(errors))
        paths.append(_project_path(row["source"]["path"]))
    return paths


def packet_value(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    task_id = str(spec["task_id"])
    paths = _paths(task_id)
    operation_file = REPO_ROOT / "src/chaoyang/ops" / f"{spec['operation']}.py"
    if not operation_file.is_file():
        raise RuntimeError(f"PICK_PLACE_OPERATION_MISSING:{operation_file}")
    code_files = [REPO_ROOT / name for name in GOVERNANCE_CODE]
    code_files += [operation_file, *package_code(), *parent_algorithm_code(), *family_code(), *fixed_existing_code()]
    code_files += [_project_path(name) for name in spec.get("code", [])]
    seen: set[Path] = set()
    runtime = []
    for path in code_files:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            runtime.append(artifact_ref(path))
    approved_spec_inputs(spec)
    inputs = {"spec": artifact_ref(spec_path),
              "files": [_input_ref(name) for name in spec.get("inputs", [])]}
    budgets = dict(spec["budgets"])
    wall = int(budgets.get("wall_seconds_max", 1800))
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = now + timedelta(seconds=wall + 300)
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": task_id}
    tail = [str(spec["operation"]), *map(str, spec["args"]), "--output-root", str(paths["stage"])]
    command = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run " + shlex.join(tail)
    config = {"session": spec.get("session"), "budgets": budgets,
              "numeric_gates_changed": False, "deletion_authorized": False,
              "gpu_id": int(spec.get("gpu_id", 0)),
              "gpu_min_free_mib": int(spec.get("gpu_min_free_mib", 40960))}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "pick-place-1008-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": spec["objective"], "phase": spec["phase"],
        "plan_revision": PLAN_REVISION, "execution_revision": task_id.upper(),
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(),
        "read_set": list(spec["read_set"]),
        "write_set": [str(paths["output"].relative_to(REPO_ROOT)),
                      str(paths["packet"].relative_to(REPO_ROOT)),
                      str(paths["receipt"].relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock",
                      "review/pick_place_1008",
                      "docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "CODE_SHA_PINNED",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": [f"candidate/{name}" for name in spec["required_outputs"]]
                            + ["RESULT.json", "terminal_receipt"],
        "budgets": budgets,
        "stop_conditions": list(spec["stop_conditions"]) + [
            "CODE_OR_INPUT_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": spec["claim_limit"],
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("PICK_PLACE_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


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


def readiness(spec: dict[str, Any]) -> None:
    paths = _paths(str(spec["task_id"]))
    if any(path.exists() for path in (paths["output"], paths["packet"], paths["receipt"])):
        raise RuntimeError("PICK_PLACE_FRESH_TASK_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") or any(row["status"] in ACTIVE_STATUSES for row in state["tasks"]):
        raise RuntimeError("PICK_PLACE_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("PICK_PLACE_ACTIVE_ROUTE_CONFLICT")
    governance = _governance_pass()
    if governance.get("status") != "PASS":
        raise RuntimeError("PICK_PLACE_GOVERNANCE_NOT_PASS:" + json.dumps(governance.get("errors", governance))[:400])
    approved_release_boundary(_project_path(spec["root_publication_approval"]))
    from chaoyang.cli import _maintained_operations
    if spec['operation'] not in _maintained_operations():raise RuntimeError('QP_OPERATION_NOT_CURRENT_MAINTAINED')
    current=load_json(REPO_ROOT/'docs/governance/ALGORITHM_CONTRACT.json')
    publication=load_json(_project_path(spec['root_publication_approval']))
    if current.get('special_status_contracts',{}).get(ROUTE_KEY,{}).get('root_registered_descriptor')!=publication['family_descriptor']:
        raise RuntimeError('QP_CURRENT_ROOT_DESCRIPTOR_NOT_PUBLISHED')


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


def _collect_required_outputs(packet: dict[str, Any], paths: dict[str, Path]
                              ) -> tuple[dict[str, Any], list[str]]:
    outputs, errors = {}, []
    for name in packet["required_outputs"]:
        if not name.startswith("candidate/"):
            continue
        path = (paths["output"] / name).resolve()
        if not path.is_relative_to(paths["output"].resolve()):
            raise RuntimeError("PICK_PLACE_REQUIRED_OUTPUT_ESCAPES_ATTEMPT")
        if path.is_file():
            outputs[name] = artifact_ref(path)
        else:
            errors.append(f"PICK_PLACE_REQUIRED_OUTPUT_MISSING:{name}")
    return outputs, errors


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
               OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, NUMBA_NUM_THREADS=threads)
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


def finish(packet: dict[str, Any], code: int, log: Path, elapsed: float,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    paths = _paths(packet["task_id"])
    outputs, errors = _collect_required_outputs(packet, paths)
    for ref in packet["runtime_code"]:
        errors.extend(validate_artifact_ref(ref))
    errors.extend(validate_artifact_ref(packet["inputs"]["spec"]))
    for ref in packet["inputs"]["files"]:
        errors.extend(validate_artifact_ref(ref))
    resource_wait = _resource_wait_evidence(paths, code, reason)
    signature_errors = [error for error in errors if not error.startswith("PICK_PLACE_REQUIRED_OUTPUT_MISSING:")]
    status = ("BLOCKED_RESOURCE" if resource_wait and not signature_errors else
              "PASSED" if code == 0 and not errors and reason is None else "FAILED_RUNTIME_FINAL")
    stage_result = paths["stage"] / "RESULT.json"
    payload = {"worker_returncode": code, "failure_code": reason, "validation_errors": errors,
               "elapsed_seconds": round(elapsed, 3), "required_outputs": outputs,
               "algorithm_attempts_consumed": 0 if resource_wait else 1,
               "resource_wait": resource_wait,
               "stage_result": artifact_ref(stage_result) if stage_result.is_file() else None,
               "worker_log": artifact_ref(log),
               "next_action": ("Review the sealed stage result; successors are new task ids"
                               if status == "PASSED" else "Inspect failure; publish a new successor task id")}
    return _seal(packet, status, payload, f"PICK_PLACE_TASK_{status}")


def recover(task_id: str) -> dict[str, Any]:
    fixed={
      'exact78_pick_place_qpc_arm_cohort16_cpu_candidate_revision1_20261010':'run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1',
      'exact78_pick_place_qpc_arm_cohort16_motion_render_candidate_revision1_20261010':'run_pick_place_qpc_arm_cohort16_motion_render_candidate_v1'}
    if task_id not in fixed:raise RuntimeError('QP_RECOVER_ONLY_FIXED_TWO_QPC_TASKS')
    packet_guard=load_json(_paths(task_id)['packet'])
    from chaoyang.governance.exact78_operation_gate import _command_tail
    from chaoyang.pick_place_qpc_arm_cohort16_v1.approval import verify_runtime_approval
    command=_command_tail(packet_guard['commands'][0])
    if command[0]!=fixed[task_id] or command[0] not in ALLOWED_OPERATIONS or packet_guard['task_id']!=task_id:
        raise RuntimeError('QP_RECOVER_FIXED_OPERATION_AND_TASK_IDENTITY')
    spec_guard=load_json(_project_path(packet_guard['inputs']['spec']['path']))
    if errors:=validate_artifact_ref(packet_guard['inputs']['spec']):raise RuntimeError('QP_RECOVER_SPEC_SHA_DRIFT:'+str(errors))
    approved_spec_inputs(spec_guard)
    args=command[1:]
    if args.count('--root-runtime-approval')!=1:raise RuntimeError('QP_RECOVER_EXPLICIT_APPROVAL_REQUIRED')
    verify_runtime_approval(args[args.index('--root-runtime-approval')+1],argv=args,packet=packet_guard)
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


def publish_route(root_publication_approval: Path) -> dict[str, Any]:
    if SCRIPT.resolve()!=REPO_ROOT/'src/chaoyang/governance/execute_pick_place_qpc_arm_cohort16_v2.py':
        raise RuntimeError('QP_CANONICAL_INSTALLED_EXECUTOR_REQUIRED')
    boundary=approved_release_boundary(root_publication_approval)
    approval=load_json(root_publication_approval)
    published=publish_bundle(load_json(AUTHORITY_PATH),copy.deepcopy(load_json(TASK_STATE_PATH)),
        event_type='ROOT_APPROVED_FIXED16_QPC_CPU_NEW_FAMILY_PUBLISHED',
        expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=SCRIPT,
        planned_family_descriptor_ref=approval['family_descriptor'])
    after=_governance_pass()
    return {'status':after.get('status'),'errors_after':after.get('errors',[]),'governance_revision':published['governance_revision'],'actual_release_boundary':boundary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", type=Path, metavar="SPEC")
    choice.add_argument("--execute", type=Path, metavar="SPEC")
    choice.add_argument("--recover", metavar="TASK_ID")
    choice.add_argument("--publish-route", type=Path, metavar="ROOT_PUBLICATION_APPROVAL")
    args = parser.parse_args(argv)
    if args.recover:
        print(json.dumps(recover(args.recover), ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.publish_route:
        result = publish_route(args.publish_route)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result["status"] == "PASS" else 1
    spec_path = (args.check_only or args.execute).resolve()
    spec = load_spec(spec_path)
    packet = packet_value(spec_path, spec)
    if args.check_only:
        try:
            readiness(spec)
            ready = "DRY_RUN_READY"
        except RuntimeError as exc:
            ready = f"DRY_RUN_NOT_ROUTABLE:{exc}"
        print(json.dumps({"status": ready, "task_id": spec["task_id"],
                          "command": packet["commands"][0], "run_signature": packet["run_signature"],
                          "runtime_code_files": len(packet["runtime_code"]),
                          "budgets": packet["budgets"]}, ensure_ascii=False, sort_keys=True))
        return 0
    readiness(spec)
    register(spec, packet)
    started = time.monotonic()
    try:
        from chaoyang.cli import _maintained_operations
        if spec["operation"] not in _maintained_operations():
            raise RuntimeError("PICK_PLACE_OPERATION_NOT_IN_PUBLISHED_CONTRACT")
        code, log = run_worker(packet)
        failure = None
    except Exception as exc:
        if isinstance(exc, WorkerNotStoppedError):
            raise
        code, failure = 1, f"{type(exc).__name__}:{exc}"
        log = _paths(packet["task_id"])["output"] / "executor_error.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    result = finish(packet, code, log, time.monotonic() - started, failure)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
