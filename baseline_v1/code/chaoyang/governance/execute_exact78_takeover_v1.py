"""Generic fenced Task Packet executor for the Exact78 takeover line (2026-10-04).

One spec file under ``docs/plans/EXACT78_TAKEOVER_20261004/specs`` describes one
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
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


SCRIPT = Path(__file__)
ROUTE_KEY = "exact78_takeover_v1_20261004"
TASK_PREFIX = "exact78_takeover_"
OPERATION_PREFIX = "run_exact78_takeover_"
PLAN_REVISION = "EXACT78_TAKEOVER_20261004"
SPEC_ROOT = REPO_ROOT / "docs/plans/EXACT78_TAKEOVER_20261004/specs"
PACKAGE_ROOT = REPO_ROOT / "src/chaoyang/exact78_baseline"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PYTHON = "/usr/local/bin/python"
HEARTBEAT_SECONDS = 50
ACTIVE_STATUSES = ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")
GOVERNANCE_CODE = (
    "src/chaoyang/governance/execute_exact78_takeover_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
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


def takeover_operations() -> list[Path]:
    return sorted((REPO_ROOT / "src/chaoyang/ops").glob(f"{OPERATION_PREFIX}*.py"))


def package_code() -> list[Path]:
    if not PACKAGE_ROOT.is_dir():
        return []
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def route_contract(repo_root: Path) -> dict[str, Any]:
    """Algorithm-contract entry; every run still needs its own live Task Packet."""
    del repo_root
    closure = [artifact_ref(path) for path in (*takeover_operations(), *package_code())]
    return {
        "execution_status": "CANDIDATE_FENCED_TAKEOVER_TASK_PACKET_ONLY",
        "candidate_operations": [path.stem for path in takeover_operations()],
        "task_packet_required": True,
        "executor": str(SCRIPT.resolve()),
        "code_closure": closure,
        "output_schema": "exact78-takeover-execution-result-v1",
        "model_calls_authorized": "ONLY_AS_DECLARED_IN_ACTIVE_PACKET_BUDGET",
        "render_calls_authorized": "ONLY_AS_DECLARED_IN_ACTIVE_PACKET_BUDGET",
        "training_steps_authorized": 0,
        "current_runnable_operation": None,
        "claim_limit": "Takeover development candidates only; no production route, visual/product qualification, control or training authority.",
    }


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
        raise RuntimeError(f"TAKEOVER_PATH_ESCAPES_PROJECT:{raw}")
    return resolved


def _input_ref(raw: str) -> dict[str, Any]:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return artifact_ref(path)


def load_spec(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    if spec_path.parent != SPEC_ROOT.resolve():
        raise RuntimeError("TAKEOVER_SPEC_OUTSIDE_SPEC_ROOT")
    spec = load_json(spec_path)
    missing = sorted(SPEC_FIELDS - set(spec))
    if missing:
        raise RuntimeError(f"TAKEOVER_SPEC_MISSING:{missing}")
    if not str(spec["task_id"]).startswith(TASK_PREFIX):
        raise RuntimeError("TAKEOVER_TASK_PREFIX")
    if not str(spec["operation"]).startswith(OPERATION_PREFIX):
        raise RuntimeError("TAKEOVER_OPERATION_PREFIX")
    if "--output-root" in spec["args"]:
        raise RuntimeError("TAKEOVER_OUTPUT_ROOT_IS_EXECUTOR_OWNED")
    if spec_path.stem != spec["task_id"]:
        raise RuntimeError("TAKEOVER_SPEC_NAME_MISMATCH")
    return spec


def packet_value(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    task_id = str(spec["task_id"])
    paths = _paths(task_id)
    operation_file = REPO_ROOT / "src/chaoyang/ops" / f"{spec['operation']}.py"
    if not operation_file.is_file():
        raise RuntimeError(f"TAKEOVER_OPERATION_MISSING:{operation_file}")
    code_files = [REPO_ROOT / name for name in GOVERNANCE_CODE]
    code_files += [operation_file, *package_code()]
    code_files += [_project_path(name) for name in spec.get("code", [])]
    seen: set[Path] = set()
    runtime = []
    for path in code_files:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            runtime.append(artifact_ref(path))
    inputs = {"spec": artifact_ref(spec_path),
              "files": [_input_ref(name) for name in spec.get("inputs", [])]}
    budgets = dict(spec["budgets"])
    wall = int(budgets.get("wall_seconds_max", 1800))
    now = datetime.now().astimezone()
    deadline = now + timedelta(seconds=wall + 300)
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": task_id}
    tail = [str(spec["operation"]), *map(str, spec["args"]), "--output-root", str(paths["stage"])]
    command = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run " + shlex.join(tail)
    config = {"session": spec.get("session"), "budgets": budgets,
              "numeric_gates_changed": False, "deletion_authorized": False}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-takeover-task-v1"})).hexdigest()
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
                      "tasks/current/INDEX.json", "docs/governance"],
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
        raise RuntimeError("TAKEOVER_PACKET_PREFLIGHT:" + ";".join(errors))
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
        raise RuntimeError("TAKEOVER_FRESH_TASK_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") or any(row["status"] in ACTIVE_STATUSES for row in state["tasks"]):
        raise RuntimeError("TAKEOVER_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("TAKEOVER_ACTIVE_ROUTE_CONFLICT")
    governance = _governance_pass()
    if governance.get("status") != "PASS":
        raise RuntimeError("TAKEOVER_GOVERNANCE_NOT_PASS:" + json.dumps(governance.get("errors", governance))[:400])


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
        raise RuntimeError("TAKEOVER_STATE_PREFLIGHT:" + ";".join(errors))
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
        raise RuntimeError("TAKEOVER_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_TAKEOVER_TASK_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    task_id = packet["task_id"]
    writer = packet["writer"]
    paths = _paths(task_id)
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(paths["output"] / "WRITER.json") != writer):
        raise RuntimeError("TAKEOVER_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"] if row["status"] in ACTIVE_STATUSES]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != task_id
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != task_id
            or routed[0]["packet_sha256"] != artifact_ref(paths["packet"])["sha256"]):
        raise RuntimeError("TAKEOVER_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def heartbeat(packet: dict[str, Any]) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_TAKEOVER_TASK_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)


def _stop_fenced(worker: subprocess.Popen[Any], ticks: int) -> None:
    if worker.poll() is not None:
        return
    try:
        if process_identity(worker.pid)["start_ticks"] != ticks:
            raise WorkerNotStoppedError("TAKEOVER_WORKER_IDENTITY_LOST")
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if process_identity(worker.pid)["start_ticks"] != ticks:
                raise WorkerNotStoppedError("TAKEOVER_WORKER_IDENTITY_LOST")
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
    except (ProcessLookupError, ChildProcessError, OSError, KeyError,
            ValueError, subprocess.TimeoutExpired):
        if worker.poll() is None:
            raise WorkerNotStoppedError("TAKEOVER_WORKER_STOP_UNPROVEN")


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    paths = _paths(packet["task_id"])
    budgets = packet["budgets"]
    cache = paths["output"] / "cache"
    cache.mkdir()
    threads = str(int(budgets.get("cpu_threads_total", 4)))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
               NUMBA_CACHE_DIR=str(cache / "numba"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
               OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    if int(budgets.get("gpu_count", 0)) == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    wall = int(budgets.get("wall_seconds_max", 1800))
    deadline = datetime.fromisoformat(packet["deadline_at"])
    log = paths["output"] / "worker.log"
    tokens = shlex.split(packet["commands"][0])
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens.pop(0)
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(tokens, cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ticks = process_identity(worker.pid)["start_ticks"]
        except (OSError, KeyError, ValueError) as exc:
            if worker.poll() is None:
                raise WorkerNotStoppedError("TAKEOVER_WORKER_START_IDENTITY_LOST") from exc
            ticks = -1
        last = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= wall or datetime.now().astimezone() >= deadline:
                    raise RuntimeError("TAKEOVER_HARD_DEADLINE")
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
        raise RuntimeError("TAKEOVER_DUPLICATE_TERMINAL")
    result = {"schema_version": "exact78-takeover-execution-result-v1", "task_id": task_id,
              "status": status, "task_packet": artifact_ref(paths["packet"]),
              "run_signature": packet["run_signature"], "visual_qualified": False,
              "training_eligible": False, "control_ground_truth": False,
              "adoption": "NOT_ADOPTED", "user_review": "PENDING",
              "files_deleted": False, **payload}
    atomic_json(paths["output"] / "RESULT.json", result)
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
    atomic_json(paths["receipt"], {"schema_version": "exact78-takeover-receipt-v1",
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
        raise RuntimeError("TAKEOVER_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == task_id:
            item["execution_allowed"] = False
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
    errors: list[str] = []
    outputs: dict[str, Any] = {}
    for name in packet["required_outputs"]:
        if not name.startswith("candidate/"):
            continue
        path = paths["output"] / name
        if path.is_file():
            outputs[name] = artifact_ref(path)
        else:
            errors.append(f"TAKEOVER_REQUIRED_OUTPUT_MISSING:{name}")
    for ref in packet["runtime_code"]:
        errors.extend(validate_artifact_ref(ref))
    errors.extend(validate_artifact_ref(packet["inputs"]["spec"]))
    for ref in packet["inputs"]["files"]:
        errors.extend(validate_artifact_ref(ref))
    status = "PASSED" if code == 0 and not errors and reason is None else "FAILED_RUNTIME_FINAL"
    stage_result = paths["stage"] / "RESULT.json"
    payload = {"worker_returncode": code, "failure_code": reason, "validation_errors": errors,
               "elapsed_seconds": round(elapsed, 3), "required_outputs": outputs,
               "stage_result": artifact_ref(stage_result) if stage_result.is_file() else None,
               "worker_log": artifact_ref(log),
               "next_action": ("Review the sealed stage result; successors are new task ids"
                               if status == "PASSED" else "Inspect failure; publish a new successor task id")}
    return _seal(packet, status, payload, f"EXACT78_TAKEOVER_TASK_{status}")


def recover(task_id: str) -> dict[str, Any]:
    paths = _paths(task_id)
    if not task_id.startswith(TASK_PREFIX):
        raise RuntimeError("TAKEOVER_TASK_PREFIX")
    packet = load_json(paths["packet"])
    state = load_json(TASK_STATE_PATH)
    row = next((item for item in state["tasks"] if item["task_id"] == task_id), None)
    if row is None or row["status"] not in ACTIVE_STATUSES:
        raise RuntimeError("TAKEOVER_RECOVER_TASK_NOT_ACTIVE")
    identity = process_identity(int(row["pid"])) if row.get("pid") is not None else {"alive": False}
    if identity["alive"] and identity["start_ticks"] == row.get("proc_start_ticks"):
        raise RuntimeError("TAKEOVER_RECOVER_WRITER_ALIVE")
    worker_log = paths["output"] / "worker.log"
    payload = {"worker_returncode": None, "failure_code": "INTERRUPTED_WRITER_LOST",
               "validation_errors": [], "recovered_at": now_iso(),
               "lost_writer": {"pid": row.get("pid"), "proc_start_ticks": row.get("proc_start_ticks"),
                               "last_heartbeat_at": row.get("heartbeat_at")},
               "worker_log": artifact_ref(worker_log) if worker_log.is_file() else None,
               "partial_outputs_are_successor_inputs": False,
               "next_action": "Publish a new successor task id; reuse only sealed SHA-matched inputs"}
    return _seal(packet, "FAILED_RUNTIME_FINAL", payload, "EXACT78_TAKEOVER_TASK_RECOVERED_INTERRUPTED")


def _route_drift_files() -> set[str]:
    files = {str(path.resolve()) for path in (*takeover_operations(), *package_code())}
    files |= {str((REPO_ROOT / name).resolve()) for name in (
        "src/chaoyang/governance/current_r3_contracts.py",
        "src/chaoyang/governance/execute_exact78_takeover_v1.py")}
    return files


def publish_route() -> dict[str, Any]:
    """Republish the algorithm contract after takeover code edits; no task routing."""
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") or any(row["status"] in ACTIVE_STATUSES for row in state["tasks"]):
        raise RuntimeError("TAKEOVER_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("TAKEOVER_ACTIVE_ROUTE_CONFLICT")
    before = _governance_pass()
    allowed = _route_drift_files()
    errors = list(before.get("errors", []))
    if before.get("status") != "PASS":
        foreign = [error for error in errors
                   if not (error.startswith("algorithm_contract.") and "code_closure" in error
                           and any(name in error for name in allowed))]
        if foreign or not errors:
            raise RuntimeError("TAKEOVER_ROUTE_PUBLISH_FOREIGN_ERRORS:" + json.dumps(foreign or before)[:800])
    published = publish_bundle(load_json(AUTHORITY_PATH), copy.deepcopy(state),
                               event_type="EXACT78_TAKEOVER_ROUTE_PUBLISHED",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=SCRIPT)
    after = _governance_pass()
    return {"status": after.get("status"), "errors_before": len(errors),
            "errors_after": after.get("errors", []),
            "governance_revision": published["governance_revision"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", type=Path, metavar="SPEC")
    choice.add_argument("--execute", type=Path, metavar="SPEC")
    choice.add_argument("--recover", metavar="TASK_ID")
    choice.add_argument("--publish-route", action="store_true")
    args = parser.parse_args(argv)
    if args.recover:
        print(json.dumps(recover(args.recover), ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.publish_route:
        result = publish_route()
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
            raise RuntimeError("TAKEOVER_OPERATION_NOT_IN_PUBLISHED_CONTRACT")
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
