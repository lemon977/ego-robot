"""One finite publisher for a sealed complete-state ICT successor.

The old producer/trainer and its eight-hour publisher are unchanged.  Only two
old infrastructure helpers are reused: the measured storage probe and central
GPU lease heartbeat/release.  All model work goes through the registered new
``chaoyang run`` operation, never through this publisher directly.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, atomic_write, canonical_bytes, load_json, now_iso, publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/checkpoint_continue_20k_20261011"
ENGINEERING_TASK = "pick_place_humanego_ict_checkpoint_continuation_cpu_v1_20261011"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"
MAX_WALL_SECONDS = 43200
SUMMARY_FILENAME = "CONTINUE_HERE_ZH.md"


def _existing_helpers():
    path = REPO_ROOT / "scripts/humanego_ict/production_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_sealed_production_infrastructure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Never call register_main or inherit module.HARD_DEADLINE.
    return module._measure_storage, module._lease_update


def _json_once(path, value):
    if path.exists():
        raise RuntimeError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def _project_path(path, *, existing=True):
    value = Path(path).resolve(strict=existing)
    if not value.is_relative_to(REPO_ROOT.resolve(strict=True)):
        raise ValueError("PROJECT_ONLY_RUNTIME_PATH_REQUIRED")
    return value


def _verify(reference):
    path = _project_path(reference["path"])
    if errors := validate_artifact_ref(reference):
        raise ValueError("FROZEN_REFERENCE_SHA_DRIFT:" + str(errors))
    return path


def _snapshot():
    """Use the existing publisher lock; one coherent authority snapshot."""
    lock_path = REPO_ROOT / "_run/current/locks/governance.lock"
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            receipt = load_json(RECEIPT_PATH)
            state_ref = receipt["files"]["task_state"]
            index_ref = receipt["files"]["task_packet_index_current"]
            state = load_json(_verify(state_ref))
            index = load_json(_verify(index_ref))
            return receipt, state, index
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _no_active_writer(state, index):
    if (state.get("next_task")
            or any(row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}
                   for row in state["tasks"])
            or any(row.get("execution_allowed") is True for row in index["task_packets"])):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")


def _publish(state, index, event, *, expected_revision):
    index["supersedes_index"] = artifact_ref(INDEX)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
        expected_revision=expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)


def _engineering_refs(state):
    rows = [row for row in state["tasks"] if row.get("task_id") == ENGINEERING_TASK]
    if len(rows) != 1 or rows[0].get("status") != "PASSED":
        raise ValueError("SEALED_PASSED_CPU_RESUME_ENGINEERING_REQUIRED")
    refs = {"engineering_task_packet": rows[0]["task_packet"],
            "engineering_task_result": rows[0]["result"]}
    for ref in refs.values():
        _verify(ref)
    return refs


def parent_bindings(parent_packet_path, branch, *, state, index):
    """Select only the normal sealed final state, never a partial or best-loss file."""
    from chaoyang.humanego_ict.production_resume import validate_parent_evidence
    path = _project_path(parent_packet_path)
    if not path.is_relative_to(REPO_ROOT / "tasks/current"):
        raise ValueError("REGISTERED_PROJECT_PARENT_PACKET_REQUIRED")
    packet_ref, packet = artifact_ref(path), load_json(path)
    task_id = packet.get("task_id")
    rows = [row for row in state["tasks"] if row.get("task_id") == task_id]
    pointers = [row for row in index["task_packets"] if row.get("task_id") == task_id]
    if (packet.get("branch") != branch or len(rows) != 1
            or rows[0].get("status") != "PASSED"
            or rows[0].get("task_packet") != packet_ref
            or rows[0].get("run_signature") != packet.get("run_signature")
            or len(pointers) != 1 or pointers[0].get("execution_allowed") is not False
            or pointers[0].get("packet_sha256") != packet_ref["sha256"]
            or (REPO_ROOT / pointers[0]["packet_path"]).resolve() != path):
        raise ValueError("SEALED_PASSED_SAME_BRANCH_PARENT_REQUIRED")
    outer_ref = rows[0]["result"]
    outer = load_json(_verify(outer_ref))
    stages = []
    for reference in outer.get("stage_results", []):
        value = load_json(_verify(reference))
        if (value.get("task_id") == task_id and value.get("run_signature") == packet.get("run_signature")
                and value.get("branch") == branch and type(value.get("optimizer_updates")) is int):
            stages.append((value["optimizer_updates"], reference, value))
    if not stages:
        raise ValueError("SEALED_PARENT_INNER_RESULT_REQUIRED")
    _, inner_ref, inner = max(stages, key=lambda item: item[0])
    checkpoint_path = _project_path(inner["last_checkpoint"])
    checkpoint_ref = artifact_ref(checkpoint_path)
    if checkpoint_ref not in outer.get("complete_checkpoints", []):
        raise ValueError("PARENT_FINAL_CHECKPOINT_NOT_SEALED_COMPLETE")
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".ref.json")
    refs = {"task_packet": packet_ref, "outer_result": outer_ref, "inner_result": inner_ref,
            "training_config": packet["inputs"]["training_config"],
            "checkpoint": checkpoint_ref, "checkpoint_sidecar": artifact_ref(sidecar_path)}
    evidence = validate_parent_evidence(refs, root=REPO_ROOT)
    if evidence["branch"] != branch or not 0 < inner["optimizer_updates"] < 20000:
        raise ValueError("PARENT_BRANCH_OR_REMAINING_UPDATE_BUDGET_INVALID")
    for reference in packet.get("runtime_code", []):
        _verify(reference)
    return refs, evidence


def make_resume_config(parent_config, refs, wall_seconds):
    from chaoyang.humanego_ict.production_resume_gate import CONFIG_SCHEMA, INITIALIZATION
    if type(wall_seconds) is not int or not 1500 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("FINITE_RESUME_AND_EXPORT_BUDGET_REQUIRED")
    value = copy.deepcopy(parent_config)
    value.update(schema_version=CONFIG_SCHEMA, initialization=INITIALIZATION,
                 resume_parent_bindings=copy.deepcopy(refs))
    value["training"]["wall_seconds_max"] = wall_seconds - 1500
    return value


def _writer_identity(task_id, attempt):
    value = Path("/proc/self/stat").read_text()
    return {"pid": int(value.split(" ", 1)[0]),
        "proc_start_ticks": int(value[value.rfind(")") + 2:].split()[19]),
        "executor_epoch": attempt, "fencing_token": secrets.token_hex(24), "scope": task_id}


def _environment(output):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO_ROOT / "src"),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0")
    for name, sub in (("TMPDIR", "tmp"), ("TORCH_HOME", "torch"),
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
            ("XDG_CACHE_HOME", "xdg"), ("NUMBA_CACHE_DIR", "numba"),
            ("CUDA_CACHE_PATH", "nvidia_compute"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
            ("HF_HOME", "huggingface"), ("HUGGINGFACE_HUB_CACHE", "huggingface_hub"),
            ("MPLCONFIGDIR", "matplotlib")):
        path = output / "runtime_cache" / sub
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(_project_path(path))
    env["TMP"] = env["TEMP"] = env["TMPDIR"]
    return env


def _own_row(packet, state, index):
    rows = [row for row in state["tasks"] if row.get("task_id") == packet["task_id"]]
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    pointers = [row for row in index["task_packets"] if row.get("task_id") == packet["task_id"]]
    writer = packet["writer"]
    if (len(rows) != 1 or len(active) != 1 or active[0] is not rows[0]
            or rows[0].get("status") != "RUNNING" or rows[0].get("pid") != writer["pid"]
            or rows[0].get("proc_start_ticks") != writer["proc_start_ticks"]
            or rows[0].get("run_signature") != packet["run_signature"]
            or len(pointers) != 1 or pointers[0].get("execution_allowed") is not True
            or pointers[0].get("packet_sha256") != artifact_ref(Path(packet["packet_path"]))["sha256"]
            or load_json(Path(packet["writer_file"])) != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    return rows[0], pointers[0]


def _tail_update(path):
    """Read at most 64 KiB, ignore an incomplete in-flight JSONL tail."""
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        offset = max(0, size - 65536)
        stream.seek(offset)
        data = stream.read(65536)
    if offset:
        data = data.partition(b"\n")[2]
    for line in reversed(data.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            continue
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if (isinstance(value, dict) and type(value.get("optimizer_update")) is int
                and 0 <= value["optimizer_update"] <= 20000):
            return value
    return None


def _published_checkpoint(output, parent_step):
    """Cheap published sidecar/size observation, not a per-minute tensor rehash.

    The immutable final publisher performs the full SHA readback.  Distinguish
    this heartbeat observation from that final verification in the document.
    """
    best = None
    for path in output.glob("checkpoint_*.pt.ref.json"):
        checkpoint = Path(str(path)[:-len(".ref.json")])
        try:
            step = int(checkpoint.stem[len("checkpoint_"):])
            sidecar = load_json(path)
            digest = sidecar["sha256"]
            if (not parent_step < step <= 20000 or type(sidecar.get("bytes")) is not int
                    or checkpoint.stat().st_size != sidecar["bytes"]
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                continue
            ref = dict(sidecar, path=str(_project_path(checkpoint)))
            if best is None or step > best[0]:
                best = (step, ref)
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return best


def _sealed_branch_progress(state, branch):
    candidates = []
    for row in state["tasks"]:
        task_id = row.get("task_id", "")
        if (row.get("status") != "PASSED" or not task_id.startswith((
                "pick_place_humanego_ict_production_train_" + branch + "_",
                "pick_place_humanego_ict_production_resume_" + branch + "_"))):
            continue
        outer = load_json(_verify(row["result"]))
        for reference in outer.get("stage_results", []):
            inner = load_json(_verify(reference))
            step = inner.get("optimizer_updates")
            if (type(step) is int and inner.get("branch") == branch
                    and inner.get("task_id") == task_id
                    and inner.get("run_signature") == row.get("run_signature")):
                ref = next((item for item in outer.get("complete_checkpoints", [])
                            if item["path"] == inner.get("last_checkpoint")), None)
                if ref is not None:
                    candidates.append({"step": step, "checkpoint": ref,
                        "task_id": task_id, "status": "PASSED", "verification": "封存时完整 SHA 读回通过"})
    return max(candidates, key=lambda item: item["step"], default=None)


def _continue_summary(packet, directory, state, result=None):
    """Render a bounded factual restart card, not a quality or deployment PASS."""
    parents = load_json(_verify(packet["inputs"]["engineering_task_packet"]))["inputs"]["parents"]
    branches = {}
    for branch in ("raw", "robotized"):
        initial = load_json(_verify(parents[branch]["inner_result"]))["optimizer_updates"]
        observed = _sealed_branch_progress(state, branch)
        branches[branch] = dict(observed or {"step": initial,
            "checkpoint": parents[branch]["checkpoint"], "task_id": "父八小时任务",
            "status": "PASSED", "verification": "已封存父 checkpoint"}, initial_step=initial)
    branch = packet["branch"]
    start = load_json(_verify(packet["inputs"]["parent_inner_result"]))["optimizer_updates"]
    update = _tail_update(directory / "output/updates.jsonl")
    logged_step = update["optimizer_update"] if update and update["optimizer_update"] >= start else None
    current = branches[branch]
    sidecar = _published_checkpoint(directory / "output", start)
    if sidecar is not None:
        current.update(step=max(current["step"], sidecar[0]), checkpoint=sidecar[1],
            verification="旁票已发布且字节数一致；本次心跳未重算完整 SHA")
    current.update(step=max(current["step"], start, logged_step or start),
        task_id=packet["task_id"], status=result["task_terminal_status"] if result else "RUNNING")
    if result and type(result.get("optimizer_updates")) is int:
        current["step"] = result["optimizer_updates"]
        complete = result.get("complete_checkpoints", [])
        if complete:
            current["checkpoint"] = max(complete, key=lambda item: int(Path(item["path"]).stem[len("checkpoint_"):]))
            current["verification"] = "封存时完整 SHA 读回通过"
    if result:
        next_action = ("Raw 已正常封存；独立登记 Robotized 续训，不覆盖 Raw 或原八小时结果。"
            if branch == "raw" and result["task_terminal_status"] == "PASSED" else
            "两分支本轮有限任务已封存；Root 复核实际累计步数与两份推理导出，未满 20000 则另立 successor。"
            if branch == "robotized" and result["task_terminal_status"] == "PASSED" else
            "本任务失败已隔离；Root 根据 RESULT 的失败码登记有限修复，不消费未验证 partial。")
    else:
        next_action = ("继续 Raw 至累计 20000 或本任务正常墙钟停止，保存导出并封存后独立登记 Robotized。"
            if branch == "raw" else "继续 Robotized 至累计 20000 或本任务正常墙钟停止，保存导出后验收两分支。")
    lines = ["# HumanEgo ICT 20000 步续训进度", "",
        "程序维护的有限任务续接摘要；父 Raw 为 4376 步，父 Robotized 为 4768 步，原结果只读保留。", "",
        f"更新时间：{now_iso()}。当前任务：`{packet['task_id']}`；状态：`{current['status']}`。",
        f"登记 writer PID：{packet['writer']['pid']}；固定截止：`{packet['deadline_at']}`。",
        f"本次恢复起点 {start} 步；最近有证据的累计步数 {current['step']}，本任务新增 {current['step'] - start} 步。",
        f"最近完整日志：{logged_step if logged_step is not None else '尚无新增日志'}；日志每 20 步记录，不冒充实时逐步计数。", "",
        "| 分支 | 原父步数 | 最新有证据累计步数 | 距 20000 | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("raw", "robotized"):
        value = branches[name]
        lines.append(f"| {name} | {value['initial_step']} | {value['step']} | {max(0, 20000 - value['step'])} | {value['status']} |")
    loss = update.get("flow_loss") if update else None
    if isinstance(loss, (int, float)) and math.isfinite(loss):
        lines.extend(["", f"最近日志训练 flow loss：{loss:.8g}（{logged_step} 步）；不是验证精度。"])
    lines.extend(["", "## 已发布 checkpoint", ""])
    for name in ("raw", "robotized"):
        value, label = branches[name], "Raw" if name == "raw" else "Robotized"
        reference = value["checkpoint"]
        lines.extend([f"- {label}：[训练文件]({reference['path']})；SHA `{reference['sha256']}`；{value['verification']}。"])
    lines.extend(["", "## 下一动作", "", next_action, "",
        "本任务结果：[输出目录](" + str(directory / "output") + ")；任务封存结果位于同级 RESULT.json。",
        "", "动作仍为 DERIVED_RETARGET 研究监督，非真实 Revo2 控制真值；数值门、源数据、模型和组划分不变。Robotized 已审样本视觉 HIGH／FAILED 及未看范围 UNKNOWN 保留。文件执行通过、累计步数完成、视觉质量、验证精度和真机资格互不替代。",
        "", "执行合同见[续训计划](EXECUTION_ZH.md)，原交付见[八小时交接](../checkpoint_delivery_8h_20261011/CONTINUE_HERE_ZH.md)。", ""])
    return "\n".join(lines)


def _write_continue_summary(packet, directory, state, result=None):
    path = _project_path(REPO_ROOT / PLAN / SUMMARY_FILENAME)
    atomic_write(path, _continue_summary(packet, directory, state, result).encode("utf-8"))


def _stop_child(child):
    """Terminate only the process group created by this publisher, wait/reap it."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=30)
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=30)


def _collect_result(packet, directory, started, child, errors, parent_step):
    output = Path(packet["output_dir"])
    stages, checkpoints = [], []
    for path in sorted(output.glob("RESULT_*.json")):
        try:
            value = load_json(path)
            if (value.get("task_id") != packet["task_id"]
                    or value.get("run_signature") != packet["run_signature"]
                    or value.get("branch") != packet["branch"]
                    or value.get("control_ground_truth") is not False
                    or value.get("physical_deployment_authorized") is not False):
                raise ValueError("INNER_RESULT_TASK_BINDING_DRIFT")
            stages.append((artifact_ref(path), value))
        except Exception as error:
            errors.append("RESULT_COLLECTION:" + type(error).__name__ + ":" + str(error))
    for path in sorted(output.glob("checkpoint_*.pt")):
        sidecar = path.with_suffix(path.suffix + ".ref.json")
        if not sidecar.is_file():
            continue
        try:
            ref = dict(load_json(sidecar), path=str(path.resolve()))
            _verify(ref)
            checkpoints.append(ref)
        except Exception as error:
            errors.append("CHECKPOINT_COLLECTION:" + type(error).__name__ + ":" + str(error))
    latest = max((value for _, value in stages), key=lambda value: value.get("optimizer_updates", 0), default={})
    updates = latest.get("optimizer_updates", parent_step)
    export_path = output / "INFERENCE_EXPORT.json"
    export_ref = artifact_ref(export_path) if export_path.is_file() else None
    final_checkpoint = latest.get("last_checkpoint")
    final_is_complete = any(ref["path"] == final_checkpoint for ref in checkpoints)
    success = bool(not errors and child is not None and child.returncode == 0 and stages
        and export_ref and type(updates) is int and parent_step < updates <= 20000 and final_is_complete)
    return {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "branch": packet["branch"], "child_exit_code": child.returncode if child else None,
        "elapsed_seconds": time.monotonic() - started,
        "stage_results": [ref for ref, _ in stages], "inference_export": export_ref,
        "complete_checkpoints": checkpoints, "start_optimizer_updates": parent_step,
        "optimizer_updates": updates, "new_optimizer_updates": updates - parent_step,
        "completed_update_budget": updates == 20000,
        "task_terminal_status": "PASSED" if success else "FAILED_RUNTIME_FINAL",
        "failure_codes": list(errors), "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": packet["claim_limit"]}


def _seal(packet, directory, result, env):
    reference = _json_once(directory / "RESULT.json", result)
    receipt, state, index = _snapshot()
    row, item = _own_row(packet, state, index)
    terminal = result["task_terminal_status"]
    row.update(status=terminal, pid=None, proc_start_ticks=None, result=reference,
               updated_at=now_iso(), heartbeat_at=now_iso())
    state["next_task"] = None
    item.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    task_receipt = REPO_ROOT / "tasks/receipts" / (packet["task_id"].upper() + ".json")
    _json_once(task_receipt, {"task_id": packet["task_id"], "status": terminal,
        "result": reference, "claim_limit": packet["claim_limit"]})
    row["receipt"] = artifact_ref(task_receipt)
    _write_continue_summary(packet, directory, state, result)
    _publish(state, index, "HUMANEGO_ICT_PRODUCTION_RESUME_SEALED",
             expected_revision=receipt["governance_revision"])
    validation = subprocess.run([sys.executable, "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    _json_once(directory / "POST_VALIDATION.json", {"exit_code": validation.returncode,
        "stdout": validation.stdout, "stderr": validation.stderr})
    print(json.dumps(result), flush=True)
    return 0 if terminal == "PASSED" and validation.returncode == 0 else 2


def register_main(argv=None):
    parser = argparse.ArgumentParser(description="Fresh finite exact-state ICT continuation; not a queue")
    parser.add_argument("--parent-task-packet", type=Path, required=True)
    parser.add_argument("--branch", choices=("raw", "robotized"), required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args(argv)
    if args.attempt < 1 or not 1500 < args.wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("FRESH_FINITE_RESUME_BUDGET_REQUIRED")
    from chaoyang.humanego_ict.production_resume_gate import (
        OPERATION, ROUTE, EXECUTION_MODE, TRAINING_SCOPE, PARENT_INPUT_ROLES,
        _sealed_engineering,
    )
    receipt, state, index = _snapshot()
    _no_active_writer(state, index)
    parent_refs, evidence = parent_bindings(args.parent_task_packet, args.branch, state=state, index=index)
    route = load_json(_verify(receipt["files"]["algorithm_contract"]))["special_status_contracts"].get(ROUTE)
    if (not route or route.get("candidate_operation") != OPERATION
            or route.get("resume_support") is not True
            or route.get("parent_sealed_checkpoint_required") is not True
            or route.get("current_runnable_operation") is not None):
        raise ValueError("REGISTERED_CANDIDATE_RESUME_ROUTE_REQUIRED")
    closure = route["code_closure"]
    for ref in closure:
        _verify(ref)
    if any(ref not in closure for ref in evidence["packet"].get("runtime_code", [])):
        raise ValueError("PARENT_CODE_CLOSURE_NOT_PRESERVED")
    engineering_refs = _engineering_refs(state)
    provisional = {"inputs": engineering_refs, "runtime_code": closure, "task_id": "unregistered_successor"}
    _sealed_engineering(REPO_ROOT, provisional, state, index)
    task_id = f"pick_place_humanego_ict_production_resume_{args.branch}_a{args.attempt:02d}_20261011"
    directory = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    if directory.exists() or packet_path.exists():
        raise ValueError("FRESH_ATTEMPT_REQUIRED_NO_DUPLICATE_SUBMISSION")
    directory.mkdir(parents=True)
    inputs = {role: evidence["packet"]["inputs"][role]
              for role in ("production_manifest", "prepared_pair", "pretrained", "pretrained_config")}
    inputs.update({PARENT_INPUT_ROLES[role]: reference for role, reference in parent_refs.items()})
    inputs.update(engineering_refs)
    inputs["plan"] = artifact_ref(REPO_ROOT / PLAN / "EXECUTION_ZH.md")
    measure_storage, lease_update = _existing_helpers()
    inputs["storage_capacity_observation"] = measure_storage(directory, task_id)
    # Admission and storage probing happen before the task clock; their records
    # remain immutable and no GPU/model call has occurred.
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(seconds=args.wall_seconds)
    config = make_resume_config(evidence["training_config"], parent_refs, args.wall_seconds)
    inputs["training_config"] = _json_once(directory / "TRAINING_CONFIG.json", config)
    output = directory / "output"
    writer = _writer_identity(task_id, args.attempt)
    command_args = ["train", "--task-packet", str(packet_path), "--manifest", inputs["production_manifest"]["path"],
        "--config", inputs["training_config"]["path"], "--checkpoint", inputs["resume_checkpoint"]["path"],
        "--output-dir", str(output), "--branch", args.branch]
    command = shlex.join([sys.executable, "-m", "chaoyang.cli", "run", OPERATION, *command_args])
    budgets = {"wall_seconds_max": args.wall_seconds, "gpu_count": 1, "model_calls": 160001,
        "training_steps": 20000, "optimizer_updates_max": 20000, "cpu_workers_max": 2,
        "cpu_threads_per_worker": 2}
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Resume the complete sealed training state to cumulative 20000 updates: " + args.branch,
        "phase": "HUMANEGO_ICT_PRODUCTION_RESUME", "plan_revision": "CHECKPOINT_CONTINUE_20K_20261011",
        "execution_revision": task_id.upper(), "execution_mode": EXECUTION_MODE,
        "training_scope": TRAINING_SCOPE, "branch": args.branch,
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/ALGORITHM_CONTRACT.json", "tasks/current/INDEX.json", PLAN + "/EXECUTION_ZH.md"],
        "read_set_expansions": ["Explicit signed parent/source/engineering refs and frozen route closure only; NAS sources readonly"],
        "write_set": [str(directory.relative_to(REPO_ROOT)), str(packet_path.relative_to(REPO_ROOT)),
            "tasks/current/INDEX.json", "tasks/receipts/" + task_id.upper() + ".json", "docs/governance",
            PLAN + "/CONTINUE_HERE_ZH.md", "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "SEALED_PASSED_CPU_ENGINEERING",
            "SEALED_PASSED_SAME_BRANCH_PARENT", "COMPLETE_STATE_RESUME", "EXACT_ROUTE_CLOSURE", "RECENT_MEASURED_STORAGE"],
        "required_outputs": ["RESULT.json", "POST_VALIDATION.json"], "budgets": budgets,
        "stop_conditions": ["fixed deadline", "input or code SHA drift", "double writer", "unrecoverable ENOSPC"],
        "claim_limit": "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING; cumulative actual updates only, not measured Revo2 controls, visual quality, validation accuracy or physical deployment approval",
        "inputs": inputs, "runtime_code": closure, "writer": writer,
        "writer_file": str(directory / "WRITER.json"), "packet_path": str(packet_path),
        "output_dir": str(output), "commands": [command], "execution_started_at": now.isoformat(),
        "deadline_at": deadline.isoformat(), "allocated_gpu_ids": [0],
        "gpu_lease_verified": True, "gpu_lease_path": str(GPU_LEASE)}
    packet["run_signature"] = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": closure,
        "budgets": budgets, "commands": [command], "execution_mode": EXECUTION_MODE})).hexdigest()
    if errors := _validate_packet(packet):
        raise ValueError(str(errors))
    env = _environment(output)
    child, registered, leased, errors = None, False, False, []
    started = time.monotonic()
    old_handlers = {}
    stop_requested = [None]

    def interrupted(signum, frame):
        # Do not raise inside Popen/lease publication: a signal between fork
        # and assignment could otherwise hide the newly-created child from
        # finally cleanup.  The bounded two-second loop performs shutdown.
        stop_requested[0] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        # A second fresh authority check prevents a competing writer appearing
        # while the same-filesystem storage probe was in flight.
        receipt, state, index = _snapshot()
        _no_active_writer(state, index)
        from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire
        acquired = try_acquire(SimpleNamespace(task_id=task_id, attempt_id="attempt_0001",
            executor_epoch=writer["executor_epoch"], priority="CHECKPOINT_TRAINING", gpu_id=0,
            min_free_mib=61440, lease=GPU_LEASE, lock=GPU_LOCK), writer["fencing_token"])
        if acquired is None:
            raise RuntimeError("WAIT_GPU_RESOURCE_NO_ALGORITHM_STARTED")
        leased = True
        lease_update(packet)
        _json_once(directory / "GPU_LEASE_ACQUIRED.json", load_json(GPU_LEASE))
        packet_path.parent.mkdir(parents=True, exist_ok=False)
        _json_once(packet_path, packet)
        _json_once(directory / "WRITER.json", writer)
        _json_once(directory / "FROZEN_INPUTS.json", inputs)
        stamp = now_iso()
        state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": args.attempt,
            "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
            "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": 0,
            "task_packet": artifact_ref(packet_path), "run_signature": packet["run_signature"]})
        state["next_task"] = {"task_id": task_id, "expected_resource": "GPU"}
        index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": artifact_ref(packet_path)["sha256"], "execution_allowed": True,
            "execution_class": "CURRENT_LEDGER_ROUTABLE"})
        _write_continue_summary(packet, directory, state)
        _publish(state, index, "HUMANEGO_ICT_PRODUCTION_RESUME_REGISTERED",
                 expected_revision=receipt["governance_revision"])
        registered = True
        if stop_requested[0] is not None:
            raise RuntimeError("PUBLISHER_SIGNAL_" + str(stop_requested[0]))
        print(json.dumps({"status": "REGISTERED", "task_id": task_id, "pid": writer["pid"],
            "deadline_at": packet["deadline_at"], "parent_optimizer_updates": evidence["inner_result"]["optimizer_updates"]}), flush=True)
        with (directory / "EXECUTION.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-m", "chaoyang.cli", "run", OPERATION, *command_args],
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lease_update(packet, worker_pid=child.pid)
            print(json.dumps({"child_pid": child.pid, "log": str(directory / "EXECUTION.log")}), flush=True)
            last = time.monotonic()
            while child.poll() is None:
                if stop_requested[0] is not None:
                    raise RuntimeError("PUBLISHER_SIGNAL_" + str(stop_requested[0]))
                if datetime.now(timezone.utc) >= deadline:
                    raise RuntimeError("FINITE_TASK_DEADLINE_EXCEEDED")
                if time.monotonic() - last >= 60:
                    lease_update(packet, worker_pid=child.pid)
                    receipt, state, index = _snapshot()
                    row, _ = _own_row(packet, state, index)
                    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                    _write_continue_summary(packet, directory, state)
                    _publish(state, index, "HUMANEGO_ICT_PRODUCTION_RESUME_HEARTBEAT",
                             expected_revision=receipt["governance_revision"])
                    atomic_json(directory / "HEARTBEAT.json", {"task_id": task_id,
                        "child_pid": child.pid, "elapsed_seconds": time.monotonic() - started,
                        "deadline_at": packet["deadline_at"]})
                    last = time.monotonic()
                time.sleep(2)
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        # Once shutdown begins, another ordinary stop signal must not interrupt
        # lease release or leave the ledger claiming an already-reaped writer.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        # Heartbeat, Popen, source-gate and signal failures all reap the child
        # before releasing its lease.  Never leave an unowned training process.
        try:
            _stop_child(child)
        except BaseException as error:
            errors.append("CHILD_CLEANUP:" + type(error).__name__ + ":" + str(error))
        if leased:
            try:
                lease_update(packet, release_reason="FINITE_RESUME_EXIT_" + str(child.returncode if child else "NO_CHILD"))
            except BaseException as error:
                errors.append("LEASE_RELEASE:" + type(error).__name__ + ":" + str(error))
    try:
        if not registered:
            status = "WAIT_GPU_RESOURCE" if any("WAIT_GPU_RESOURCE" in value for value in errors) else "FAILED_REGISTRATION"
            _json_once(directory / "REGISTRATION_FAILURE.json", {"task_id": task_id,
                "algorithm_started": False, "failure_codes": errors, "status": status})
            print(json.dumps({"status": status, "task_id": task_id, "failure_codes": errors}), flush=True)
            return 2
        try:
            result = _collect_result(packet, directory, started, child, errors,
                                     evidence["inner_result"]["optimizer_updates"])
        except Exception as error:
            # Even a readback failure must close this owned task as failure,
            # rather than bypassing finalization after a clean child shutdown.
            result = {"task_id": task_id, "run_signature": packet["run_signature"],
                "branch": args.branch, "child_exit_code": child.returncode if child else None,
                "elapsed_seconds": time.monotonic() - started, "stage_results": [],
                "complete_checkpoints": [], "inference_export": None,
                "task_terminal_status": "FAILED_RUNTIME_FINAL",
                "failure_codes": errors + ["RESULT_READBACK:" + type(error).__name__ + ":" + str(error)],
                "control_ground_truth": False, "physical_deployment_authorized": False}
        try:
            return _seal(packet, directory, result, env)
        except BaseException as error:
            # A changed writer fence must not be overwritten to make cleanup
            # look successful.  All owned compute is reaped and lease release
            # attempted; retain an immutable reconciliation receipt.
            _json_once(directory / "FINALIZATION_ERROR.json", {"task_id": task_id,
                "failure": type(error).__name__ + ":" + str(error),
                "owned_child_reaped": child is None or child.poll() is not None,
                "result_exists": (directory / "RESULT.json").is_file(),
                "requires_governance_reconciliation": True})
            print(json.dumps({"status": "FAILED_FINALIZATION", "task_id": task_id,
                "failure": type(error).__name__ + ":" + str(error)}), flush=True)
            return 2
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def continue_pair_main(argv):
    """Explicit sequential calls, two separate packets/results, no new queue."""
    parser = argparse.ArgumentParser(description="Raw then Robotized, independent finite fenced tasks")
    parser.add_argument("--continue-pair", action="store_true", required=True)
    parser.add_argument("--raw-parent-task-packet", type=Path, required=True)
    parser.add_argument("--robotized-parent-task-packet", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True,
                        help="Per-branch fixed task budget, at most 43200 seconds")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args(argv)
    if args.attempt < 1 or not 1500 < args.wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("FRESH_FINITE_RESUME_BUDGET_REQUIRED")
    for branch, parent in (("raw", args.raw_parent_task_packet),
                           ("robotized", args.robotized_parent_task_packet)):
        code = register_main(["--parent-task-packet", str(parent), "--branch", branch,
            "--wall-seconds", str(args.wall_seconds), "--attempt", str(args.attempt)])
        if code != 0:
            return code
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return continue_pair_main(argv) if "--continue-pair" in argv else register_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
