"""Finite executor for the unregistered three-frame Prompt14 experiment.

``--check-only`` is read-only. ``--run-registered`` requires a separately
published active Task Packet, live fenced writer, exact runtime closure and
one GPU lease. This module has no registration or governance publication path.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, REPO_ROOT, artifact_ref, atomic_json, canonical_bytes,
    load_json, process_identity,
    validate_artifact_ref,
)
from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep


PACKET = REPO_ROOT / "tasks/current" / prep.TASK_ID / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_ATTEMPT_ID = "attempt_0001_vst_finger_sleeve_prompt14"
RUNTIME_RELATIVE = (
    "src/chaoyang/governance/exact78_vst_finger_sleeve_prompt14_cpu_prep_v1.py",
    "src/chaoyang/governance/execute_exact78_vst_finger_sleeve_prompt14_candidate_v1.py",
    "src/chaoyang/governance/publish_exact78_vst_finger_sleeve_prompt14_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_vst_finger_sleeve_prompt14_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v1.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    "src/chaoyang/pipeline/vst_encoded_video_domain.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/v52_contracts.py",
    "src/chaoyang/governance/current_baseline_v2.py",
    "src/chaoyang/governance/algorithm_baseline_v2_contract.py",
    "src/chaoyang/governance/full_pipeline_v3_contract.py",
    "src/chaoyang/governance/build_algorithm_baseline_v2_status.py",
    "src/chaoyang/governance/build_full_pipeline_v3_status.py",
    "src/chaoyang/governance/build_full_pipeline_v4_status.py",
    "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/paths.py",
    "src/chaoyang/cli/__init__.py",
)


def runtime_code() -> list[dict[str, Any]]:
    return [artifact_ref(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE]


def run_signature(packet: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes({
        "schema": "exact78-vst-finger-sleeve-prompt14-stage-v1",
        "inputs": packet["inputs"], "runtime_code": packet["runtime_code"],
        "config": packet["config"], "budgets": packet["budgets"],
        "commands": packet["commands"],
        "weights": packet["inputs"]["sam_checkpoint"],
        "calibration": "ABSENT", "output_schema": "exact78-vst-finger-sleeve-prompt14-result-v1",
    })).hexdigest()


def route_contract(code: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact candidate route to stage with a future Task Packet CAS."""
    return {
        "execution_status": "CANDIDATE_FENCED_GPU_PROMPT14_THREE_FRAMES",
        "candidate_operation": prep.OPERATION, "task_packet_required": True,
        "code_closure": code,
        "output_schema": "exact78-vst-finger-sleeve-prompt14-result-v1",
        "model_calls_authorized": "THREE_FROZEN_0915_CARDS001_SLEEVE_CALLS_ONLY",
        "render_calls_authorized": 0, "training_steps_authorized": 0,
        "current_runnable_operation": None,
        "claim_limit": "Three development sleeve masks only; yellow cable unresolved; no Clean, Robot, geometry, product or training pass.",
    }


def expected_command() -> str:
    cache = prep.ATTEMPT / "cache"
    return ("PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "
            f"TMPDIR={cache} TMP={cache} TEMP={cache} XDG_CACHE_HOME={cache} "
            f"HF_HOME={cache / 'huggingface'} TORCH_HOME={cache / 'torch'} "
            f"MPLCONFIGDIR={cache / 'mpl'} CUDA_CACHE_PATH={cache / 'cuda'} "
            f"/usr/local/bin/python -B -m chaoyang.cli run {prep.OPERATION} "
            f"--output-root {prep.ATTEMPT / 'candidate'}")


def _writer_descendant(writer_pid: int, writer_ticks: int, caller_pid: int) -> bool:
    identity = process_identity(writer_pid)
    if not identity["alive"] or identity["start_ticks"] != writer_ticks:
        return False
    pid, seen = caller_pid, set()
    while pid > 1 and pid not in seen:
        if pid == writer_pid:
            return True
        seen.add(pid)
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        pid = int(raw[raw.rfind(")") + 2:].split()[1])
    return False


def verify_packet(packet: dict[str, Any], *, caller_pid: int | None = None) -> None:
    """Revalidate every publication-sensitive value before launching a worker."""
    if (packet.get("task_id") != prep.TASK_ID
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("candidate_operation") != prep.OPERATION
            or packet.get("route_key") != prep.ROUTE_KEY
            or packet.get("inputs") != prep.frozen_inputs()
            or packet.get("source_session") != prep.SOURCE_SESSION
            or packet.get("anchor_frames") != list(prep.FRAMES)
            or packet.get("single_change") != {"role": prep.ROLE, "from": prep.OLD_PROMPT,
                                                "to": prep.NEW_PROMPT}
            or packet.get("config") != {"output_prob_thresh": 0.5, "instance_score_gate": 0.5}
            or packet.get("commands") != [expected_command()]
            or packet.get("runtime_code") != runtime_code()
            or packet.get("run_signature") != run_signature(packet)):
        raise RuntimeError("PROMPT14_PACKET_OR_RUNTIME_SHA_DRIFT")
    budgets = packet.get("budgets", {})
    expected_budget = {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                       "model_calls_max": 3, "render_calls": 0, "clean_calls": 0,
                       "training_steps": 0, "wall_seconds_max": 900,
                       "session_attempts_max": 1}
    if budgets != expected_budget or prep.ATTEMPT.relative_to(REPO_ROOT).as_posix() not in packet.get("write_set", []):
        raise RuntimeError("PROMPT14_BUDGET_OR_WRITE_SET_DRIFT")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if deadline.tzinfo is None or datetime.now(deadline.tzinfo) >= deadline:
        raise RuntimeError("PROMPT14_DEADLINE_EXPIRED")
    writer = packet.get("writer", {})
    keys = ("pid", "proc_start_ticks", "executor_epoch", "fencing_token", "scope")
    if (writer.get("scope") != prep.TASK_ID or type(writer.get("executor_epoch")) is not int
            or writer["executor_epoch"] < 1 or not isinstance(writer.get("fencing_token"), str)
            or len(writer["fencing_token"]) < 32
            or not all(writer.get(key) == load_json(prep.ATTEMPT / "WRITER.json").get(key)
                       for key in keys)
            or not _writer_descendant(writer["pid"], writer["proc_start_ticks"],
                                      caller_pid or os.getpid())):
        raise RuntimeError("PROMPT14_DOUBLE_WRITER_OR_FENCE_DRIFT")
    active = [row for row in load_json(INDEX).get("task_packets", [])
              if row.get("execution_allowed") is True]
    if (len(active) != 1 or active[0].get("task_id") != prep.TASK_ID
            or active[0].get("packet_path") != str(PACKET.relative_to(REPO_ROOT))
            or active[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("PROMPT14_ACTIVE_PACKET_REQUIRED")
    route = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts", {}).get(prep.ROUTE_KEY)
    if route != route_contract(packet["runtime_code"]):
        raise RuntimeError("PROMPT14_ALGORITHM_ROUTE_NOT_PUBLISHED")


def validate_candidate(packet: dict[str, Any]) -> dict[str, Any]:
    """CPU-only postcondition: exactly three new masks, frozen refs, no Clean claim."""
    import cv2
    import numpy as np

    root = prep.ATTEMPT / "candidate"
    anchors_path, result_path = root / "FINGER_SLEEVE_ANCHORS.json", root / "RESULT.json"
    anchors, result = load_json(anchors_path), load_json(result_path)
    if (anchors.get("task_id") != prep.TASK_ID or anchors.get("model_calls") != 3
            or result.get("task_id") != prep.TASK_ID or result.get("model_calls") != 3
            or result.get("run_signature") != packet["run_signature"]
            or result.get("anchors") != artifact_ref(anchors_path)
            or anchors.get("clean_called") is not False
            or result.get("clean_called") is not False
            or anchors.get("yellow_cable") != "SEALED_ZERO_DETECTION_HOLD"
            or result.get("yellow_cable") != "UNRESOLVED_QUALITY_HOLD"):
        raise RuntimeError("PROMPT14_CANDIDATE_RESULT_DRIFT")
    rows = anchors.get("rows", [])
    if [row.get("frame") for row in rows] != list(prep.FRAMES):
        raise RuntimeError("PROMPT14_THREE_ROWS_REQUIRED")
    frozen = prep.frozen_inputs()
    diagnostics = []
    for row, pinned in zip(rows, frozen["anchors"], strict=True):
        frame = pinned["frame"]
        expected_path = root / "finger_sleeve" / f"{frame:06d}.png"
        ref = row.get("mask", {})
        if (row.get("sam_input") != pinned["refs"]["sam_input"]
                or row.get("prompt") != prep.NEW_PROMPT
                or ref.get("path") != str(expected_path)
                or expected_path.is_symlink() or expected_path.resolve(strict=True) != expected_path
                or validate_artifact_ref(ref)):
            raise RuntimeError(f"PROMPT14_MASK_REF_DRIFT:{frame}")
        mask = cv2.imread(str(expected_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (960, 1280) or not np.isin(mask, (0, 255)).all():
            raise RuntimeError(f"PROMPT14_MASK_DOMAIN:{frame}")
        selected = row.get("selected_scores", [])
        if (type(row.get("instance_count")) is not int
                or row["instance_count"] != len(selected)
                or any(type(score) is not float or not math.isfinite(score) or score < 0.5
                       for score in selected)
                or row.get("pixels") != int(np.count_nonzero(mask))):
            raise RuntimeError(f"PROMPT14_SCORE_OR_PIXEL_DRIFT:{frame}")
        masks = {}
        for name in ("human", "object", "equipment", "yellow_cable"):
            sealed = pinned["refs"][name]
            if validate_artifact_ref(sealed):
                raise RuntimeError(f"PROMPT14_SEALED_MASK_DRIFT:{frame}:{name}")
            image = cv2.imread(sealed["path"], cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != mask.shape or not np.isin(image, (0, 255)).all():
                raise RuntimeError(f"PROMPT14_SEALED_MASK_DOMAIN:{frame}:{name}")
            masks[name] = image > 0
        sleeve = mask > 0
        occupied = masks["human"] | masks["equipment"] | sleeve | masks["yellow_cable"]
        object_mask = masks["object"]
        diagnostics.append({"frame": frame, "new_sleeve_pixels": int(sleeve.sum()),
                            "unknown_overlap_pixels": int((occupied & object_mask).sum()),
                            "protected_visible_pixels": int((object_mask & ~occupied).sum()),
                            "removal_preview_pixels": int((occupied & ~object_mask).sum()),
                            "review": "PENDING_HUMAN"})
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.png"))
    expected = [f"finger_sleeve/{frame:06d}.png" for frame in prep.FRAMES]
    if actual != expected:
        raise RuntimeError("PROMPT14_UNEXPECTED_MASK_OUTPUT")
    return {"status": "STRUCTURE_PASS_QUALITY_HOLD", "model_calls": 3,
            "diagnostics": diagnostics, "clean_admitted": False,
            "yellow_cable": "UNRESOLVED_QUALITY_HOLD"}


def _group_members(leader_pid: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2:].split()
            if fields[0] not in {"Z", "X"} and int(fields[2]) == leader_pid and int(fields[3]) == leader_pid:
                members.append(int(entry.name))
        except (FileNotFoundError, ValueError, IndexError):
            continue
    return members


def _terminate_group(process: subprocess.Popen[Any], start_ticks: int) -> bool:
    identity = process_identity(process.pid)
    if identity["alive"] and identity["start_ticks"] != start_ticks:
        return False
    if _group_members(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=4)
    else:
        process.wait(timeout=2)
    return not _group_members(process.pid)


def _terminate_owned_gpu_child(wrapper_pid: int, wrapper_ticks: int) -> bool:
    """Kill only the exact child recorded by our fenced lease."""
    lease = load_json(GPU_LEASE) if GPU_LEASE.is_file() else {}
    if lease.get("task_id") != prep.TASK_ID or lease.get("attempt_id") != GPU_ATTEMPT_ID:
        return False
    if (lease.get("pid") != wrapper_pid
            or lease.get("process_startticks") != wrapper_ticks):
        return False
    child = lease.get("gpu_process_pid")
    if type(child) is not int or child <= 1:
        return child is None
    raw_path = Path(f"/proc/{child}/stat")
    if not raw_path.is_file():
        return True
    raw = raw_path.read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2:].split()
    if (int(fields[1]) != wrapper_pid or int(fields[2]) != child
            or int(fields[3]) != child or int(fields[19]) < wrapper_ticks):
        return False
    os.killpg(child, signal.SIGTERM)
    until = time.monotonic() + 6
    while _group_members(child) and time.monotonic() < until:
        time.sleep(0.1)
    if _group_members(child):
        os.killpg(child, signal.SIGKILL)
        until = time.monotonic() + 4
        while _group_members(child) and time.monotonic() < until:
            time.sleep(0.1)
    return not _group_members(child)


def run_registered(packet: dict[str, Any]) -> dict[str, Any]:
    verify_packet(packet)
    if prep.ATTEMPT.exists() and any((prep.ATTEMPT / name).exists() for name in
                                     ("RESULT.json", "GPU_RECEIPT.json", "candidate")):
        raise RuntimeError("PROMPT14_ONE_ATTEMPT_ALREADY_USED")
    cache = prep.ATTEMPT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("huggingface", "torch", "mpl", "cuda"):
        (cache / name).mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH="src", PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
               TORCH_HOME=str(cache / "torch"), MPLCONFIGDIR=str(cache / "mpl"),
               CUDA_CACHE_PATH=str(cache / "cuda"))
    output = prep.ATTEMPT / "candidate"
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.ops.run_gpu_command_with_v71_lease",
               "--task-id", prep.TASK_ID, "--attempt-id", GPU_ATTEMPT_ID,
               "--executor-epoch", str(packet["writer"]["executor_epoch"]),
               "--priority", "CANARY", "--wait-seconds", "0", "--wall-seconds", "800",
               "--receipt", str(prep.ATTEMPT / "GPU_RECEIPT.json"),
               "--claim-limit", packet["claim_limit"], "--",
               "/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run", prep.OPERATION,
               "--output-root", str(output)]
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    start_ticks = process_identity(process.pid)["start_ticks"]
    remaining = (datetime.fromisoformat(packet["deadline_at"]) -
                 datetime.now(datetime.fromisoformat(packet["deadline_at"]).tzinfo)).total_seconds()
    try:
        stdout, stderr = process.communicate(timeout=min(max(1, remaining - 40), 850))
        cleanup_proven = not _group_members(process.pid)
    except subprocess.TimeoutExpired:
        try:
            child_empty = _terminate_owned_gpu_child(process.pid, start_ticks)
        except (OSError, ValueError, IndexError):
            child_empty = False
        try:
            wrapper_empty = _terminate_group(process, start_ticks)
        except (OSError, subprocess.TimeoutExpired):
            wrapper_empty = False
        cleanup_proven = child_empty and wrapper_empty
        stdout, stderr = "", "PROMPT14_DEADLINE_TIMEOUT"
    receipt_path = prep.ATTEMPT / "GPU_RECEIPT.json"
    receipt = load_json(receipt_path) if receipt_path.is_file() else {}
    lease = load_json(GPU_LEASE) if GPU_LEASE.is_file() else {}
    gpu_pids = None
    child = None
    if lease.get("task_id") == prep.TASK_ID and lease.get("attempt_id") == GPU_ATTEMPT_ID:
        from chaoyang.governance.gpu_lease_v71 import query_gpu_pids
        gpu_pids = query_gpu_pids()
        child = lease.get("gpu_process_pid")
        cleanup_proven = (cleanup_proven and lease.get("status") == "RELEASED"
                          and gpu_pids is not None and child not in gpu_pids
                          and (child is None or not _group_members(child)))
    elif receipt.get("status") != "BLOCKED_RESOURCE":
        cleanup_proven = False
    cleanup_audit = {
        "schema_version": "exact78-vst-finger-sleeve-prompt14-gpu-cleanup-audit-v1",
        "task_id": prep.TASK_ID, "status": "PROVEN" if cleanup_proven else "UNPROVEN",
        "wrapper_pid": process.pid, "wrapper_startticks": start_ticks,
        "wrapper_group_empty": not _group_members(process.pid),
        "gpu_child_pid": child,
        "gpu_child_group_empty": child is None or not _group_members(child),
        "physical_gpu_evidence_available": gpu_pids is not None,
        "gpu_child_absent_from_physical_gpu": gpu_pids is not None and child not in gpu_pids,
        "lease_status": lease.get("status", "ABSENT"),
        "gpu_receipt_status": receipt.get("status", "ABSENT"),
        "new_task_packet_allowed": cleanup_proven,
    }
    atomic_json(prep.ATTEMPT / "GPU_CLEANUP_AUDIT.json", cleanup_audit)
    diagnostics = None
    validation_error = None
    if cleanup_proven and process.returncode == 0 and receipt.get("status") == "PASSED":
        try:
            diagnostics = validate_candidate(packet)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            validation_error = f"{type(exc).__name__}:{exc}"
    status = ("PASSED" if diagnostics is not None else
              "BLOCKED_RESOURCE" if cleanup_proven and receipt.get("status") == "BLOCKED_RESOURCE"
              else "FAILED_RUNTIME_FINAL")
    result = {"schema_version": "exact78-vst-finger-sleeve-prompt14-execution-v1",
              "task_id": prep.TASK_ID, "status": status, "execution_complete": diagnostics is not None,
              "run_signature": packet["run_signature"], "gpu_receipt": artifact_ref(receipt_path)
              if receipt_path.is_file() else None, "cleanup_proven": cleanup_proven,
              "gpu_cleanup_audit": artifact_ref(prep.ATTEMPT / "GPU_CLEANUP_AUDIT.json"),
              "worker_returncode": process.returncode, "worker_stdout_tail": stdout[-2000:],
              "worker_stderr_tail": stderr[-2000:], "validation_error": validation_error,
              "diagnostics": diagnostics,
              "candidate_result": artifact_ref(output / "RESULT.json") if diagnostics else None,
              "clean_called": False, "training_eligible": False,
              "mask_quality_gate": "YELLOW_CABLE_UNRESOLVED",
              "next_action": ("Review three sleeve boundaries, visible card fragments, and missing yellow cable"
                              if diagnostics else "Audit GPU cleanup and terminalize the active packet")}
    atomic_json(prep.ATTEMPT / "RESULT.json", result)
    atomic_json(prep.ATTEMPT / "TERMINAL_RECEIPT.json", {
        "schema_version": "exact78-vst-finger-sleeve-prompt14-local-terminal-receipt-v1",
        "task_id": prep.TASK_ID, "status": status,
        "result": artifact_ref(prep.ATTEMPT / "RESULT.json"),
        "governance_cas_published": False, "deletion_authorized": False,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run-registered", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        frozen = prep.frozen_inputs()
        code = runtime_code()
        print(json.dumps({"status": "READY_FOR_SEPARATE_CAS_PUBLICATION",
                          "task_id": prep.TASK_ID, "source_session": frozen["session_id"],
                          "anchor_frames": list(prep.FRAMES), "model_calls": 0,
                          "runtime_code_refs": len(code), "registered": False,
                          "gpu_called": False}, ensure_ascii=False, sort_keys=True))
        return 0
    packet = load_json(PACKET)
    result = run_registered(packet)
    print(json.dumps({"status": result["status"], "cleanup_proven": result["cleanup_proven"]},
                     sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
