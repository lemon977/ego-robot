"""Finite, unregistered GPU executor candidate for four development Mask anchors.

The read-only ``--check-only`` never registers or invokes a model.  The
``--register-and-run`` path is fail-closed until the exact CLI operation and
algorithm-contract route are staged for publication in the same governance
CAS as its Task Packet.  Do not use this module as a production entry.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    load_json, now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance import exact78_equipment_prompt_mask13_candidate_v1 as design


INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / design.TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / design.TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{design.TASK_ID.upper()}.json"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"
PYTHON = Path("/usr/local/bin/python")
ACTIVE_STATUSES = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
MASK_ROLES = frozenset({"human", "object", "equipment_original", "equipment_added",
                        "equipment", "human_equipment", "unknown", "protected", "removal"})
GPU_ATTEMPT_ID = "attempt_0001_equipment_prompt_mask13"
DEADLINE_CLEANUP_RESERVE_SECONDS = 40


class CleanupUnproven(RuntimeError):
    """No new Task Packet may be registered until an independent cleanup audit."""


@dataclass(frozen=True)
class ProcessGroupIdentity:
    pid: int
    start_ticks: int
    parent_pid: int
    process_group: int
    session_id: int
    state: str


def _proc_evidence(pid: int) -> ProcessGroupIdentity | None:
    """Read Linux process identity without misparsing spaces in comm."""
    path = Path(f"/proc/{pid}/stat")
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    close = raw.rfind(")")
    if close < 0:
        raise CleanupUnproven(f"MASK13_PROC_STAT_INVALID:{pid}")
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        raise CleanupUnproven(f"MASK13_PROC_STAT_SHORT:{pid}")
    return ProcessGroupIdentity(pid, int(fields[19]), int(fields[1]),
                                int(fields[2]), int(fields[3]), fields[0])


def _group_members(identity: ProcessGroupIdentity) -> list[ProcessGroupIdentity]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        observed = _proc_evidence(int(entry.name))
        if (observed is not None and observed.state not in {"Z", "X"}
                and observed.process_group == identity.pid
                and observed.session_id == identity.pid):
            members.append(observed)
    return members


def _signal_exact_group(identity: ProcessGroupIdentity, sig: int) -> bool:
    members = _group_members(identity)
    if not members:
        return False
    leader = next((item for item in members if item.pid == identity.pid), None)
    if (identity.process_group != identity.pid or identity.session_id != identity.pid
            or (leader is not None and leader.start_ticks != identity.start_ticks)
            or any(item.start_ticks < identity.start_ticks for item in members)):
        raise CleanupUnproven(f"MASK13_PROCESS_GROUP_IDENTITY_DRIFT:{identity.pid}")
    try:
        os.killpg(identity.pid, sig)
    except ProcessLookupError:
        return False
    return True


def _wait_group_empty(identity: ProcessGroupIdentity, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while _group_members(identity):
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)
    return True


def _capture_owned_gpu_child(
    wrapper: ProcessGroupIdentity,
    known: ProcessGroupIdentity | None,
) -> tuple[ProcessGroupIdentity | None, dict[str, Any] | None]:
    if not GPU_LEASE.is_file():
        return known, None
    lease = load_json(GPU_LEASE)
    if (lease.get("task_id") != design.TASK_ID
            or lease.get("attempt_id") != GPU_ATTEMPT_ID):
        return known, None
    if (lease.get("schema_version") != "chaoyang-gpu-lease-v71"
            or lease.get("pid") != wrapper.pid
            or lease.get("process_startticks") != wrapper.start_ticks):
        raise CleanupUnproven("MASK13_OWNED_LEASE_WRAPPER_IDENTITY_DRIFT")
    child_pid = lease.get("gpu_process_pid")
    if child_pid is None:
        return known, lease
    if type(child_pid) is not int or child_pid <= 1:
        raise CleanupUnproven("MASK13_GPU_CHILD_PID_INVALID")
    observed = _proc_evidence(child_pid)
    if observed is not None and observed.state not in {"Z", "X"}:
        same_process = (known is None or
                        (observed.pid == known.pid
                         and observed.start_ticks == known.start_ticks
                         and observed.process_group == known.process_group
                         and observed.session_id == known.session_id))
        if (observed.parent_pid != wrapper.pid
                or observed.process_group != child_pid or observed.session_id != child_pid
                or observed.start_ticks < wrapper.start_ticks
                or not same_process):
            raise CleanupUnproven("MASK13_GPU_CHILD_IDENTITY_DRIFT")
        known = observed
    elif known is not None and known.pid != child_pid:
        raise CleanupUnproven("MASK13_GPU_CHILD_PID_DRIFT")
    return known, lease


def _lease_released_or_safely_release(
    wrapper: ProcessGroupIdentity,
    child: ProcessGroupIdentity | None,
    tracked_lease: dict[str, Any] | None,
) -> bool:
    from chaoyang.governance.gpu_lease_v71 import query_gpu_pids, release

    if _group_members(wrapper):
        return False
    snapshot = load_json(GPU_LEASE) if GPU_LEASE.is_file() else {}
    snapshot_ours = (snapshot.get("task_id") == design.TASK_ID
                     and snapshot.get("attempt_id") == GPU_ATTEMPT_ID)
    if not snapshot_ours and tracked_lease is None:
        receipt_path = ATTEMPT / "GPU_RECEIPT.json"
        receipt = load_json(receipt_path) if receipt_path.is_file() else {}
        receipt_matches = (receipt.get("schema_version") == "v71-gpu-command-receipt-v1"
                           and receipt.get("task_id") == design.TASK_ID
                           and receipt.get("attempt_id") == GPU_ATTEMPT_ID)
        blocked_without_lease = receipt_matches and receipt.get("status") == "BLOCKED_RESOURCE"
        wrapper_completed = (receipt_matches and receipt.get("status") in
                             {"PASSED", "FAILED_RUNTIME_FINAL"})
        return ((child is None or not _group_members(child))
                and (blocked_without_lease or wrapper_completed))
    gpu_pids = query_gpu_pids()
    if gpu_pids is None:
        return False
    if child is not None and (child.pid in gpu_pids or _group_members(child)):
        return False
    if not GPU_LEASE.is_file():
        return False
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load_json(GPU_LEASE)
        ours = (current.get("task_id") == design.TASK_ID
                and current.get("attempt_id") == GPU_ATTEMPT_ID)
        if not ours:
            return tracked_lease is None
        if (current.get("pid") != wrapper.pid
                or current.get("process_startticks") != wrapper.start_ticks
                or (tracked_lease is not None
                    and current.get("fencing_token") != tracked_lease.get("fencing_token"))):
            return False
        child_pid = current.get("gpu_process_pid")
        if child_pid is not None:
            if type(child_pid) is not int or child_pid in gpu_pids:
                return False
            if child is None:
                observed = _proc_evidence(child_pid)
                if observed is not None and observed.state not in {"Z", "X"}:
                    return False
            elif child_pid != child.pid:
                return False
        if current.get("status") == "RELEASED":
            return True
        if current.get("status") != "ACQUIRED":
            return False
        safe = release(current, pid=wrapper.pid,
                       fencing_token=current["fencing_token"],
                       reason="MASK13_DEADLINE_VERIFIED_GROUP_EXIT")
        atomic_json(GPU_LEASE, safe)
        return load_json(GPU_LEASE).get("status") == "RELEASED"


def _cleanup_deadline_impl(
    process: subprocess.Popen[Any], wrapper: ProcessGroupIdentity,
    child: ProcessGroupIdentity | None, tracked_lease: dict[str, Any] | None,
) -> None:
    """Bounded TERM/KILL; fail closed if identity, physical GPU, or lease is uncertain."""
    failures: list[str] = []
    signals: list[dict[str, Any]] = []
    try:
        child, observed_lease = _capture_owned_gpu_child(wrapper, child)
        tracked_lease = observed_lease or tracked_lease
    except (CleanupUnproven, OSError, ValueError) as exc:
        failures.append(str(exc))
    for identity in (child, wrapper):
        if identity is None:
            continue
        try:
            sent = _signal_exact_group(identity, signal.SIGTERM)
            signals.append({"pid": identity.pid, "signal": "TERM", "sent": sent})
            if not _wait_group_empty(identity, 6):
                sent = _signal_exact_group(identity, signal.SIGKILL)
                signals.append({"pid": identity.pid, "signal": "KILL", "sent": sent})
                if not _wait_group_empty(identity, 4):
                    failures.append(f"MASK13_GROUP_STILL_ALIVE:{identity.pid}")
        except (CleanupUnproven, OSError, ValueError) as exc:
            failures.append(str(exc))
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        failures.append("MASK13_WRAPPER_NOT_REAPED")
    try:
        if not _lease_released_or_safely_release(wrapper, child, tracked_lease):
            failures.append("MASK13_GPU_OR_LEASE_RELEASE_UNPROVEN")
    except (CleanupUnproven, OSError, ValueError, RuntimeError) as exc:
        failures.append(f"MASK13_LEASE_AUDIT_ERROR:{exc}")
    wrapper_empty = not _group_members(wrapper)
    gpu_empty = child is None or not _group_members(child)
    if not wrapper_empty or not gpu_empty:
        failures.append("MASK13_POST_AUDIT_PROCESS_GROUP_NOT_EMPTY")
    audit = {
        "schema_version": "exact78-equipment-prompt-mask13-deadline-cleanup-audit-v1",
        "task_id": design.TASK_ID,
        "status": "PROVEN" if not failures else "UNPROVEN",
        "wrapper": asdict(wrapper), "gpu_child": asdict(child) if child is not None else None,
        "signals": signals, "failures": failures,
        "wrapper_group_empty": wrapper_empty,
        "gpu_group_empty": gpu_empty,
        "gpu_lease": artifact_ref(GPU_LEASE) if GPU_LEASE.is_file() else "ABSENT",
        "new_task_packet_allowed": not failures,
    }
    try:
        atomic_json(ATTEMPT / "DEADLINE_CLEANUP_AUDIT.json", audit)
    except (OSError, ValueError, RuntimeError) as exc:
        raise CleanupUnproven(f"MASK13_CLEANUP_AUDIT_WRITE_FAILED:{exc}") from exc
    if failures:
        raise CleanupUnproven(";".join(failures))


def _cleanup_deadline(
    process: subprocess.Popen[Any], wrapper: ProcessGroupIdentity,
    child: ProcessGroupIdentity | None, tracked_lease: dict[str, Any] | None,
) -> None:
    try:
        _cleanup_deadline_impl(process, wrapper, child, tracked_lease)
    except CleanupUnproven:
        raise
    except Exception as exc:
        raise CleanupUnproven(f"MASK13_CLEANUP_UNEXPECTED_ERROR:{type(exc).__name__}:{exc}") from exc


def expected_prompts(scene: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    other = config["other_role_prompts"]
    prompts = {
        "human": config["new_human_prompt"],
        "object": other["poker_object" if scene["task"] == "poker" else "chips_object"],
        "equipment_original": other["vst_equipment" if scene["source_family"] == "VST_0915"
                                    else "tracker_equipment"],
        "equipment_added": config["equipment_prompt_added"],
    }
    if scene["source_family"] == "VST_0915":
        prompts.update(finger_sleeve=other["vst_finger_sleeve"],
                       yellow_cable=other["vst_yellow_cable"])
    return prompts


def check_mask_algebra(images: dict[str, Any], source_family: str,
                       session: str, frame: int) -> int:
    import numpy as np

    if not np.array_equal(images["equipment"],
                          images["equipment_original"] | images["equipment_added"]):
        raise RuntimeError(f"MASK13_EQUIPMENT_PIXEL_SOURCE_FAIL:{session}:{frame}")
    equipment = images["equipment"].copy()
    if source_family == "VST_0915":
        equipment |= images["finger_sleeve"] | images["yellow_cable"]
    combined = images["human"] | equipment
    object_mask = images["object"]
    expected = {
        "human_equipment": combined,
        "unknown": combined & object_mask,
        "protected": object_mask & ~combined,
        "removal": combined & ~object_mask,
    }
    if any(not np.array_equal(images[name], value) for name, value in expected.items()):
        raise RuntimeError(f"MASK13_PROTECTION_ALGEBRA_FAIL:{session}:{frame}")
    if np.any(images["removal"] & object_mask):
        raise RuntimeError(f"MASK13_VISIBLE_OBJECT_REMOVAL:{session}:{frame}")
    return int(object_mask.sum())


def source_quality_failures(gate: dict[str, Any]) -> list[str]:
    failures = []
    if gate["equipment_empty_frames"]:
        failures.append("MASK_DEVICE_UNVERIFIED")
    if gate["object_empty_frames"]:
        failures.append("MASK_VISIBLE_OBJECT_UNPROTECTED")
    if gate["source_family"] == "VST_0915" and (
            gate["finger_sleeve_empty_frames"] or gate["yellow_cable_empty_frames"]):
        failures.append("MASK_SLEEVE_CABLE_UNVERIFIED")
    return failures


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def route_readiness(packet: dict[str, Any]) -> str:
    """Require an exact staged or published candidate route and CLI dispatch."""
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    expected = design.route_contract(packet["runtime_code"])
    published = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts", {}).get(design.ROUTE_KEY)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH)).get("special_status_contracts", {}).get(design.ROUTE_KEY)
    if staged != expected:
        raise RuntimeError("MASK13_ROUTE_NOT_STAGED_OR_CODE_SHA_DRIFT")
    if published is None:
        if design.OPERATION in _maintained_operations():
            raise RuntimeError("MASK13_UNPUBLISHED_OPERATION_ALREADY_MAINTAINED")
        return "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"
    if design.OPERATION not in _maintained_operations():
        raise RuntimeError("MASK13_PUBLISHED_OPERATION_NOT_MAINTAINED")
    if published != expected:
        raise RuntimeError("MASK13_PUBLISHED_ROUTE_DRIFT")
    return "PUBLISHED_EXACT_CANDIDATE_ROUTE"


def readiness(packet: dict[str, Any]) -> str:
    if any(path.exists() for path in (ATTEMPT, PACKET, RECEIPT)):
        raise RuntimeError("MASK13_FRESH_ATTEMPT_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    if (state.get("next_task") or any(item.get("status") in ACTIVE_STATUSES for item in state["tasks"])
            or any(item.get("execution_allowed") is True for item in index["task_packets"])):
        raise RuntimeError("MASK13_ACTIVE_TASK_CONFLICT")
    if any(item.get("task_id") == design.TASK_ID for item in state["tasks"]):
        raise RuntimeError("MASK13_PRIOR_TASK_REQUIRES_NEW_SUCCESSOR")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("MASK13_RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    return route_readiness(packet)


def register(packet: dict[str, Any]) -> None:
    """Publish route and exact Task Packet through one governance CAS only."""
    readiness(packet)
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    stamp = now_iso()
    state["tasks"].append({
        "task_id": design.TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "FOUR_DEV_MASK13", "updated_at": stamp,
        "heartbeat_at": stamp, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": 0,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": design.TASK_ID, "session": "FOUR_DEV_MASK13",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "12 Mask anchors/54 role calls or 45 minute deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("MASK13_TASK_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": design.TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision=packet["execution_revision"] + "_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != identity["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or artifact_ref(PACKET)["sha256"] != next(
                (row.get("packet_sha256") for row in load_json(INDEX)["task_packets"]
                 if row.get("task_id") == design.TASK_ID and row.get("execution_allowed") is True), None)):
        raise RuntimeError("MASK13_WRITER_OR_PACKET_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row.get("status") in ACTIVE_STATUSES]
    if (len(active) != 1 or active[0].get("task_id") != design.TASK_ID
            or active[0].get("run_signature") != packet["run_signature"]):
        raise RuntimeError("MASK13_DOUBLE_WRITER_OR_STATE_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("MASK13_RUNTIME_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == design.TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("MASK13_HEARTBEAT_STATE_INVALID:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__))
    number = int(elapsed // 600)
    path = ATTEMPT / f"RESUME_SUMMARY_{number:04d}.json"
    if number and not path.exists():
        atomic_json(path, {
            "schema_version": "exact78-equipment-prompt-mask13-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "mask_anchor_sealed": (ATTEMPT / "candidate/MASK_ANCHOR.json").is_file(),
            "next_action": "Wait for fenced worker; do not consume partial Mask or run Clean.",
        })


def worker_command(packet: dict[str, Any], wait_seconds: int, wall_seconds: int) -> list[str]:
    return [str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
            "run_gpu_command_with_v71_lease", "--task-id", design.TASK_ID,
            "--attempt-id", GPU_ATTEMPT_ID, "--priority", "CANARY",
            "--gpu-id", "0", "--min-free-mib", "20000",
            "--wait-seconds", str(wait_seconds), "--wall-seconds", str(wall_seconds),
            "--receipt", str(ATTEMPT / "GPU_RECEIPT.json"),
            "--claim-limit", packet["claim_limit"], "--", str(PYTHON), "-B", "-m",
            "chaoyang.cli", "run", design.OPERATION, "--output-root",
            str(ATTEMPT / "candidate")]


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = ATTEMPT / "cache"
    cache.mkdir()
    for child in ("huggingface", "torch", "mpl", "cuda"):
        (cache / child).mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
               TORCH_HOME=str(cache / "torch"), MPLCONFIGDIR=str(cache / "mpl"),
               CUDA_CACHE_PATH=str(cache / "cuda"), OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if any(not path.resolve().is_relative_to(REPO_ROOT) for path in
           (cache, *(cache / name for name in ("huggingface", "torch", "mpl", "cuda")))):
        raise RuntimeError("MASK13_CACHE_OUTSIDE_PROJECT")
    remaining = int((datetime.fromisoformat(packet["deadline_at"]) - datetime.now().astimezone()).total_seconds())
    if remaining < 300:
        raise RuntimeError("MASK13_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(300, remaining // 8)
    wall_seconds = remaining - wait_seconds - 75
    if wall_seconds <= 0:
        raise RuntimeError("MASK13_WORKER_WALL_BUDGET_EXHAUSTED")
    command = worker_command(packet, wait_seconds, wall_seconds)
    log = ATTEMPT / "worker.log"
    began = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            wrapper = _proc_evidence(process.pid)
        except (OSError, ValueError) as exc:
            raise CleanupUnproven(f"MASK13_WRAPPER_PROC_EVIDENCE_UNAVAILABLE:{exc}") from exc
        if (wrapper is None or wrapper.parent_pid != os.getpid()
                or wrapper.process_group != process.pid
                or wrapper.session_id != process.pid):
            raise CleanupUnproven("MASK13_WRAPPER_INDEPENDENT_GROUP_IDENTITY_UNPROVEN")
        child: ProcessGroupIdentity | None = None
        tracked_lease: dict[str, Any] | None = None
        last_beat = began
        cleanup_proven = False
        cleanup_attempted = False
        try:
            while process.poll() is None:
                now = time.monotonic()
                child, observed_lease = _capture_owned_gpu_child(wrapper, child)
                tracked_lease = observed_lease or tracked_lease
                if now - last_beat >= 50:
                    heartbeat(packet, now - began)
                    last_beat = now
                if time.monotonic() - began >= remaining - DEADLINE_CLEANUP_RESERVE_SECONDS:
                    cleanup_attempted = True
                    _cleanup_deadline(process, wrapper, child, tracked_lease)
                    cleanup_proven = True
                    raise RuntimeError("MASK13_HARD_DEADLINE_CLEANUP_PROVEN")
                time.sleep(2)
            child, observed_lease = _capture_owned_gpu_child(wrapper, child)
            tracked_lease = observed_lease or tracked_lease
            if not _lease_released_or_safely_release(wrapper, child, tracked_lease):
                raise CleanupUnproven("MASK13_NORMAL_EXIT_GPU_OR_LEASE_RELEASE_UNPROVEN")
            cleanup_proven = True
        except Exception:
            if not cleanup_proven and not cleanup_attempted:
                _cleanup_deadline(process, wrapper, child, tracked_lease)
            raise
    return process.returncode, log


def validate_mask_output(packet: dict[str, Any]) -> dict[str, Any]:
    """Structural and per-pixel Mask gates; visible-object review stays human."""
    import cv2
    import numpy as np

    record_path = ATTEMPT / "candidate/MASK_ANCHOR.json"
    result_path = ATTEMPT / "candidate/RESULT.json"
    record, result = load_json(record_path), load_json(result_path)
    if (record.get("schema_version") != "exact78-equipment-prompt-mask13-v1"
            or record.get("status") != "MASK_CANDIDATE_UNREVIEWED"
            or record.get("model_calls") != 54
            or len(record.get("rows", [])) != 12
            or result.get("schema_version") != "exact78-equipment-prompt-mask13-result-v1"
            or result.get("status") != "MASK_CANDIDATE_UNREVIEWED"
            or result.get("mask_anchor") != artifact_ref(record_path)
            or result.get("run_signature") != packet["run_signature"]
            or result.get("anchors_total") != 12
            or result.get("model_calls") != 54
            or result.get("clean_called") is not False):
        raise RuntimeError("MASK13_UNSEALED_OR_COUNT_DRIFT")
    expected = [(scene["session_id"], frame["frame"], frame["raw_image"])
                for scene in packet["inputs"]["sessions"] for frame in scene["anchors"]]
    source_gates: dict[str, dict[str, Any]] = {
        scene["session_id"]: {"source_family": scene["source_family"],
                              "equipment_empty_frames": [], "object_empty_frames": [],
                              "finger_sleeve_empty_frames": [], "yellow_cable_empty_frames": [],
                              "equipment_union_pixels": 0, "object_candidate_pixels": 0}
        for scene in packet["inputs"]["sessions"]}
    for row, (session, frame, raw_ref) in zip(record["rows"], expected):
        if (row.get("session_id") != session or row.get("frame") != frame
                or row.get("raw_image") != raw_ref
                or row.get("review") != "PENDING_HUMAN_VISIBLE_OBJECT_AND_DEVICE"):
            raise RuntimeError(f"MASK13_ROW_IDENTITY_DRIFT:{session}:{frame}")
        masks = row.get("masks", {})
        source = next(item for item in packet["inputs"]["sessions"] if item["session_id"] == session)
        if row.get("prompts") != expected_prompts(source, packet["config"]):
            raise RuntimeError(f"MASK13_SINGLE_PROMPT_DELTA_DRIFT:{session}:{frame}")
        if (set(row.get("instance_counts", {})) != set(row["prompts"])
                or any(type(value) is not int or value < 0
                       for value in row["instance_counts"].values())
                or set(row.get("selected_scores", {})) != set(row["prompts"])
                or any(len(row["selected_scores"][role]) != count or
                       any(type(value) is not float or not math.isfinite(value) or value < 0.5
                           for value in row["selected_scores"][role])
                       for role, count in row["instance_counts"].items())):
            raise RuntimeError(f"MASK13_INSTANCE_COUNT_DRIFT:{session}:{frame}")
        sam_input = row.get("sam_input", {})
        expected_jpg = ATTEMPT / "candidate/sam_inputs" / session / f"{frame:06d}/000000.jpg"
        if (sam_input.get("path") != str(expected_jpg)
                or (issues := validate_artifact_ref(sam_input))):
            raise RuntimeError(f"MASK13_SAM_INPUT_SHA_DRIFT:{session}:{frame}:{issues if sam_input.get('path') == str(expected_jpg) else 'PATH'}")
        roles = MASK_ROLES | ({"finger_sleeve", "yellow_cable"} if source["source_family"] == "VST_0915" else set())
        if set(masks) != roles:
            raise RuntimeError(f"MASK13_MASK_ROLE_DRIFT:{session}:{frame}")
        images = {}
        for role, ref in masks.items():
            expected_path = ATTEMPT / "candidate/masks" / session / role / f"{frame:06d}.png"
            if Path(ref["path"]) != expected_path or expected_path.resolve(strict=True) != expected_path:
                raise RuntimeError(f"MASK13_MASK_PATH_OR_SYMLINK_DRIFT:{session}:{frame}:{role}")
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"MASK13_MASK_SHA_DRIFT:{session}:{frame}:{role}:{';'.join(errors)}")
            image = cv2.imread(ref["path"], cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (960, 1280) or not np.isin(image, (0, 255)).all():
                raise RuntimeError(f"MASK13_MASK_DOMAIN:{session}:{frame}:{role}")
            images[role] = image > 0
        check_mask_algebra(images, source["source_family"], session, frame)
        original, added = images["equipment_original"], images["equipment_added"]
        measured = {"original_only": int(np.count_nonzero(original & ~added)),
                    "added_only": int(np.count_nonzero(added & ~original)),
                    "both": int(np.count_nonzero(original & added)),
                    "union": int(np.count_nonzero(images["equipment"]))}
        if row.get("equipment_source_pixels") != measured:
            raise RuntimeError(f"MASK13_PIXEL_SOURCE_COUNT_DRIFT:{session}:{frame}")
        gate = source_gates[session]
        gate["equipment_union_pixels"] += measured["union"]
        gate["object_candidate_pixels"] += int(images["object"].sum())
        if measured["union"] == 0:
            gate["equipment_empty_frames"].append(frame)
        if not images["object"].any():
            gate["object_empty_frames"].append(frame)
        if source["source_family"] == "VST_0915":
            for role in ("finger_sleeve", "yellow_cable"):
                if not images[role].any():
                    gate[f"{role}_empty_frames"].append(frame)
    for gate in source_gates.values():
        gate["quality_failures"] = source_quality_failures(gate)
        gate["clean_admitted"] = False
    return {"anchors_validated": 12, "model_calls": 54,
            "source_gates": source_gates,
            "mask_quality_gate": "REJECTED" if any(
                gate["quality_failures"] for gate in source_gates.values()) else "PENDING_HUMAN",
            "visible_object_review": "PENDING_HUMAN",
            "protected_pixel_algebra": "PASS", "clean_authorized": False}


def apply_unproven_cleanup_hold(state: dict[str, Any]) -> None:
    """Keep the failed task as a non-runnable routing hold until separate proof."""
    state["blockers"].append({
        "name": "EXACT78_MASK13_GPU_CLEANUP_UNPROVEN",
        "status": "OPEN", "scope": "ALL_NEW_TASK_PACKETS",
        "resolution": "Independent PID/startticks/group/GPU/lease audit, then separate governance CAS; no automatic retry or successor.",
    })
    state["next_task"] = {
        "task_id": design.TASK_ID, "session": "FOUR_DEV_MASK13",
        "prerequisites": ["PROVE_GPU_CHILD_EXIT", "PROVE_GPU_LEASE_RELEASE"],
        "expected_resource": "NONE_UNTIL_INDEPENDENT_CLEANUP_AUDIT",
        "stop_condition": "Do not register any new Task Packet until a separate cleanup proof is published.",
    }


def finish(packet: dict[str, Any], code: int, log: Path, reason: str | None = None,
           *, cleanup_proven: bool = True) -> dict[str, Any]:
    verify_fence(packet)
    if (ATTEMPT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("MASK13_TERMINAL_ALREADY_EXISTS")
    gpu_path = ATTEMPT / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    errors: list[str] = []
    metrics = None
    if code == 0 and gpu.get("status") == "PASSED" and reason is None:
        try:
            metrics = validate_mask_output(packet)
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            errors.append(f"MASK_GATE:{type(exc).__name__}:{exc}")
    for scene in packet["inputs"]["sessions"]:
        for name in ("source_video", "frontend_spec", "sealed_result", "sealed_raw"):
            if issues := validate_artifact_ref(scene[name]):
                errors.append(f"INPUT_SHA:{scene['session_id']}:{name}:{';'.join(issues)}")
        for frame in scene["anchors"]:
            if issues := validate_artifact_ref(frame["raw_image"]):
                errors.append(f"RAW_ANCHOR_SHA:{scene['session_id']}:{frame['frame']}:{';'.join(issues)}")
    for name in ("sealed_mask12_packet_comparison_only", "sealed_mask12_result_comparison_only"):
        if issues := validate_artifact_ref(packet["inputs"][name]):
            errors.append(f"MASK12_COMPARATOR_SHA:{name}:{';'.join(issues)}")
    if issues := validate_artifact_ref(packet["inputs"]["sam_checkpoint"]):
        errors.append("SAM_CHECKPOINT_SHA:" + ";".join(issues))
    if issues := validate_artifact_ref(packet["inputs"]["sam_vendor_manifest"]):
        errors.append("SAM_VENDOR_MANIFEST_SHA:" + ";".join(issues))
    complete = (cleanup_proven and code == 0 and gpu.get("status") == "PASSED"
                and reason is None and not errors and metrics is not None)
    status = ("FAILED_RUNTIME_FINAL" if not cleanup_proven else "PASSED" if complete else
              "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL")
    if not log.is_file():
        log.write_text(reason or "worker did not start\n", encoding="utf-8")
    result = {
        "schema_version": "exact78-equipment-prompt-mask13-execution-result-v1",
        "task_id": design.TASK_ID, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "errors": errors,
        "deadline_cleanup_proven": cleanup_proven,
        "new_task_packet_allowed": cleanup_proven,
        "run_signature": packet["run_signature"],
        "mask_anchor": artifact_ref(ATTEMPT / "candidate/MASK_ANCHOR.json") if complete else None,
        "candidate_result": artifact_ref(ATTEMPT / "candidate/RESULT.json") if complete else None,
        "metrics": metrics, "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
        "deadline_cleanup_audit": artifact_ref(ATTEMPT / "DEADLINE_CLEANUP_AUDIT.json")
            if (ATTEMPT / "DEADLINE_CLEANUP_AUDIT.json").is_file() else None,
        "worker_log": artifact_ref(log), "mask_structural_gate": "PASS" if complete else "NOT_PASSED",
        "mask_quality_gate": metrics["mask_quality_gate"] if complete else "NOT_ADMITTED",
        "visible_object_review": "PENDING_HUMAN" if complete else "NOT_ADMITTED",
        "clean_called": False, "visual_qualified": False, "training_eligible": False,
        "claim_limit": packet["claim_limit"],
        "next_action": ("Resolve each source gate and review all twelve visible-object/device masks; Clean remains unadmitted."
                        if cleanup_proven else
                        "HOLD ALL NEW TASK PACKETS: independently prove GPU child exit and lease release first."),
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {
        "schema_version": "exact78-equipment-prompt-mask13-receipt-v1",
        "task_id": design.TASK_ID, "status": status,
        "result": artifact_ref(ATTEMPT / "RESULT.json"), "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == design.TASK_ID)
    row.update(status=status if cleanup_proven else "FAILED_RUNTIME_RETRYABLE",
               pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality=("MASK_QUALITY_REJECTED" if metrics["mask_quality_gate"] == "REJECTED"
                        else "MASK_STRUCTURE_ONLY_REVIEW_PENDING") if complete else
                       "GPU_CLEANUP_UNPROVEN_GLOBAL_HOLD" if not cleanup_proven else
                       "MASK_FAILED_OR_RESOURCE_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    if cleanup_proven:
        state["next_task"] = None
    else:
        apply_unproven_cleanup_hold(state)
    if errors := validate_task_state(state):
        raise RuntimeError("MASK13_TERMINAL_STATE_INVALID:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item.get("task_id") == design.TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] +
                 ("_CLEANUP_HOLD" if not cleanup_proven else "_TERMINAL"),
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="BLOCKED_GPU_CLEANUP_PROOF" if not cleanup_proven else "PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type=packet["execution_revision"] + "_" + status,
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(ATTEMPT / "RESULT.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    packet = design.packet_value()
    if args.check_only:
        try:
            status = readiness(packet)
        except RuntimeError as exc:
            status = str(exc)
        print(json.dumps({
            "status": status, "task_id": design.TASK_ID,
            "run_signature": packet["run_signature"],
            "route_key": design.ROUTE_KEY, "candidate_operation": design.OPERATION,
            "source_sessions": [row["session_id"] for row in packet["inputs"]["sessions"]],
            "anchor_count": 12, "model_calls_max": 54,
            "runtime_code_refs": len(packet["runtime_code"]),
            "registered": False, "gpu_called": False, "old_mask_clean_read": False,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    register(packet)
    try:
        code, log = run_worker(packet)
        reason, cleanup_proven = None, True
    except CleanupUnproven as exc:
        code, reason, cleanup_proven = 2, f"{type(exc).__name__}:{exc}", False
        log = ATTEMPT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    except Exception as exc:
        code, reason, cleanup_proven = 2, f"{type(exc).__name__}:{exc}", True
        log = ATTEMPT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    outcome = finish(packet, code, log, reason, cleanup_proven=cleanup_proven)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if outcome["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
