"""Finite Mask14 governance executor; dry run until an exact route exists."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
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
from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design


INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / design.TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / design.TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{design.TASK_ID.upper()}.json"
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
GPU_ATTEMPT_ID = "attempt_0001_chips_object_prompt_mask14"


class WorkerStopped(RuntimeError):
    def __init__(self, reason: str, log: Path, *, cleanup_proven: bool) -> None:
        super().__init__(reason)
        self.log = log
        self.cleanup_proven = cleanup_proven


def route_readiness(packet: dict[str, Any]) -> str:
    """A missing route is a dry-run state, never implicit permission."""
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    expected = design.route_contract(packet["runtime_code"])
    published = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts", {}).get(design.ROUTE_KEY)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH)).get("special_status_contracts", {}).get(design.ROUTE_KEY)
    if published is None:
        if staged is None:
            return "DRY_RUN_NOT_ROUTABLE"
        if staged != expected or design.OPERATION in _maintained_operations():
            raise RuntimeError("MASK14_STAGED_ROUTE_OR_CLI_CODE_SHA_DRIFT")
        return "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"
    if staged != expected or published != expected or design.OPERATION not in _maintained_operations():
        raise RuntimeError("MASK14_ROUTE_OR_CLI_CODE_SHA_DRIFT")
    return "PUBLISHED_EXACT_CANDIDATE_ROUTE"


def readiness(packet: dict[str, Any]) -> str:
    route = route_readiness(packet)
    if route not in {"PUBLISHED_EXACT_CANDIDATE_ROUTE",
                     "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"}:
        return route
    if any(path.exists() for path in (ATTEMPT, PACKET, RECEIPT)):
        raise RuntimeError("MASK14_FRESH_ATTEMPT_REQUIRED")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if (state.get("next_task") or any(row.get("status") in ACTIVE for row in state["tasks"])
            or any(row.get("execution_allowed") is True for row in index["task_packets"])):
        raise RuntimeError("MASK14_ACTIVE_TASK_CONFLICT")
    if any(row.get("task_id") == design.TASK_ID for row in state["tasks"]):
        raise RuntimeError("MASK14_PRIOR_TASK_REQUIRES_SUCCESSOR")
    for ref in packet["runtime_code"]:
        if issues := validate_artifact_ref(ref):
            raise RuntimeError("MASK14_RUNTIME_SHA_DRIFT:" + ";".join(issues))
    return route


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def register(packet: dict[str, Any]) -> None:
    if readiness(packet) not in {"PUBLISHED_EXACT_CANDIDATE_ROUTE",
                               "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"}:
        raise RuntimeError("MASK14_DRY_RUN_NOT_ROUTABLE")
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    stamp = now_iso()
    state["tasks"].append({
        "task_id": design.TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "SIX_CHIPS_MASK14", "updated_at": stamp,
        "heartbeat_at": stamp, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": 0,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": design.TASK_ID, "session": "SIX_CHIPS_MASK14",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "Six object calls or 30 minute deadline",
    }
    if issues := validate_task_state(state):
        raise RuntimeError("MASK14_TASK_STATE_INVALID:" + ";".join(issues))
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
    identity, writer = process_identity(os.getpid()), packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != identity["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or artifact_ref(PACKET)["sha256"] != next(
                (row.get("packet_sha256") for row in load_json(INDEX)["task_packets"]
                 if row.get("task_id") == design.TASK_ID and row.get("execution_allowed") is True), None)):
        raise RuntimeError("MASK14_WRITER_OR_PACKET_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row.get("status") in ACTIVE]
    if (len(active) != 1 or active[0].get("task_id") != design.TASK_ID
            or active[0].get("run_signature") != packet["run_signature"]):
        raise RuntimeError("MASK14_DOUBLE_WRITER_OR_STATE_DRIFT")
    for ref in packet["runtime_code"]:
        if issues := validate_artifact_ref(ref):
            raise RuntimeError("MASK14_RUNTIME_SHA_DRIFT:" + ";".join(issues))


def heartbeat(packet: dict[str, Any]) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == design.TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    if issues := validate_task_state(state):
        raise RuntimeError("MASK14_HEARTBEAT_STATE_INVALID:" + ";".join(issues))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__))


def _proc_start(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return int(value[value.rfind(")") + 2:].split()[19])


def _tracked_child(pid: int, start: int) -> dict[str, int] | None:
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease = load_json(lease_path) if lease_path.is_file() else {}
    if lease.get("task_id") != design.TASK_ID or lease.get("attempt_id") != GPU_ATTEMPT_ID:
        return None
    if lease.get("pid") != pid or lease.get("process_startticks") != start:
        raise RuntimeError("MASK14_GPU_LEASE_IDENTITY_DRIFT")
    child = lease.get("gpu_process_pid")
    if child is None:
        return None
    observed = _proc_start(child)
    return {"pid": child, "start": observed} if observed is not None else None


def _stop_exact_group(pid: int, start: int) -> bool:
    if _proc_start(pid) != start:
        return _proc_start(pid) is None
    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise RuntimeError("MASK14_PROCESS_GROUP_IDENTITY_DRIFT")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(40):
        if _proc_start(pid) is None:
            return True
        time.sleep(0.1)
    if _proc_start(pid) == start:
        os.killpg(pid, signal.SIGKILL)
    for _ in range(40):
        if _proc_start(pid) is None:
            return True
        time.sleep(0.1)
    return False


def _lease_released(wrapper_pid: int, wrapper_start: int,
                    child: dict[str, int] | None) -> bool:
    from chaoyang.governance.gpu_lease_v71 import query_gpu_pids

    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease = load_json(lease_path) if lease_path.is_file() else {}
    if (lease.get("task_id"), lease.get("attempt_id")) != (design.TASK_ID, GPU_ATTEMPT_ID):
        receipt_path = ATTEMPT / "GPU_RECEIPT.json"
        receipt = load_json(receipt_path) if receipt_path.is_file() else {}
        return receipt.get("status") == "BLOCKED_RESOURCE" and child is None
    if (lease.get("pid") != wrapper_pid
            or lease.get("process_startticks") != wrapper_start
            or lease.get("status") != "RELEASED"):
        return False
    gpu_pids = query_gpu_pids()
    child_pid = lease.get("gpu_process_pid")
    if gpu_pids is None or (type(child_pid) is int and child_pid in gpu_pids):
        return False
    if child is not None and _proc_start(child["pid"]) == child["start"]:
        return False
    return True


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = ATTEMPT / "cache"
    cache.mkdir()
    for name in ("huggingface", "torch", "mpl", "cuda"):
        (cache / name).mkdir()
    if not cache.resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError("MASK14_CACHE_OUTSIDE_PROJECT")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
               TORCH_HOME=str(cache / "torch"), MPLCONFIGDIR=str(cache / "mpl"),
               CUDA_CACHE_PATH=str(cache / "cuda"), OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    remaining = int((datetime.fromisoformat(packet["deadline_at"]) - datetime.now().astimezone()).total_seconds())
    if remaining < 300:
        raise RuntimeError("MASK14_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(300, remaining // 8)
    wall_seconds = remaining - wait_seconds - 75
    cmd = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
           "run_gpu_command_with_v71_lease", "--task-id", design.TASK_ID,
           "--attempt-id", GPU_ATTEMPT_ID, "--priority", "CANARY",
           "--gpu-id", "0", "--min-free-mib", "20000",
           "--wait-seconds", str(wait_seconds), "--wall-seconds", str(wall_seconds),
           "--receipt", str(ATTEMPT / "GPU_RECEIPT.json"),
           "--claim-limit", packet["claim_limit"], "--", "/usr/local/bin/python",
           "-B", "-m", "chaoyang.cli", "run", design.OPERATION,
           "--output-root", str(ATTEMPT / "candidate")]
    log = ATTEMPT / "worker.log"
    start = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        identity = _proc_start(process.pid)
        if identity is None or os.getpgid(process.pid) != process.pid:
            raise RuntimeError("MASK14_WRAPPER_IDENTITY_UNPROVEN")
        child: dict[str, int] | None = None
        last_beat = start
        try:
            while process.poll() is None:
                observed = _tracked_child(process.pid, identity)
                if observed:
                    if child is not None and observed != child:
                        raise RuntimeError("MASK14_GPU_CHILD_IDENTITY_DRIFT")
                    child = observed
                now = time.monotonic()
                if now - last_beat >= 50:
                    heartbeat(packet)
                    last_beat = now
                if now - start >= remaining - 40:
                    raise RuntimeError("MASK14_HARD_DEADLINE")
                time.sleep(2)
            if child and _proc_start(child["pid"]) == child["start"]:
                raise RuntimeError("MASK14_GPU_CHILD_STILL_ALIVE")
            if not _lease_released(process.pid, identity, child):
                raise RuntimeError("MASK14_GPU_LEASE_RELEASE_UNPROVEN")
        except Exception as exc:
            cleanup = True
            for target in (child, {"pid": process.pid, "start": identity}):
                if target is not None:
                    try:
                        cleanup &= _stop_exact_group(target["pid"], target["start"])
                    except (OSError, RuntimeError):
                        cleanup = False
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cleanup = False
            cleanup &= _lease_released(process.pid, identity, child)
            raise WorkerStopped(f"{type(exc).__name__}:{exc}", log,
                                cleanup_proven=cleanup) from exc
    return process.returncode, log


def validate_mask_output(packet: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import numpy as np

    record_path = ATTEMPT / "candidate/MASK_ANCHOR.json"
    result_path = ATTEMPT / "candidate/RESULT.json"
    record, result = load_json(record_path), load_json(result_path)
    if (record.get("schema_version") != "exact78-chips-object-prompt-mask14-v1"
            or record.get("model_calls") != 6 or len(record.get("rows", [])) != 6
            or result.get("mask_anchor") != artifact_ref(record_path)
            or result.get("run_signature") != packet["run_signature"]
            or result.get("model_calls") != 6 or result.get("anchors_total") != 6
            or result.get("clean_called") is not False):
        raise RuntimeError("MASK14_UNSEALED_OR_COUNT_DRIFT")
    old_rows = design.frozen_six()
    empty = []
    for row, old, anchor in zip(record["rows"], old_rows, packet["inputs"]["anchors"]):
        session, frame = anchor["session_id"], anchor["frame"]
        if (row.get("session_id") != session or row.get("frame") != frame
                or row.get("raw_image") != anchor["raw_image"]
                or row.get("sam_input") != anchor["sam_input"]
                or row.get("fixed_masks") != anchor["fixed_masks"]
                or row.get("old_object") != anchor["old_object"]
                or row.get("prompts") != design.object_only_prompts(old["prompts"])):
            raise RuntimeError(f"MASK14_ROW_IDENTITY_DRIFT:{session}:{frame}")
        if row.get("instance_count") != len(row.get("selected_scores", [])):
            raise RuntimeError("MASK14_INSTANCE_COUNT_DRIFT")
        if any(type(score) is not float or not np.isfinite(score) or score < 0.5
               for score in row["selected_scores"]):
            raise RuntimeError("MASK14_INSTANCE_SCORE_DRIFT")
        fixed, generated = {}, {}
        for role, ref in anchor["fixed_masks"].items():
            if issues := validate_artifact_ref(ref):
                raise RuntimeError("MASK14_FIXED_MASK_SHA_DRIFT:" + ";".join(issues))
            fixed[role] = cv2.imread(ref["path"], cv2.IMREAD_GRAYSCALE) > 0
        if set(row.get("new_masks", {})) != {"object", "human_equipment", "unknown", "protected", "removal"}:
            raise RuntimeError("MASK14_GENERATED_ROLE_DRIFT")
        for role, ref in row["new_masks"].items():
            path = ATTEMPT / "candidate/masks" / session / role / f"{frame:06d}.png"
            if ref.get("path") != str(path) or path.resolve(strict=True) != path:
                raise RuntimeError("MASK14_NEW_MASK_PATH_DRIFT")
            if issues := validate_artifact_ref(ref):
                raise RuntimeError("MASK14_NEW_MASK_SHA_DRIFT:" + ";".join(issues))
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (960, 1280) or not np.isin(image, (0, 255)).all():
                raise RuntimeError("MASK14_NEW_MASK_DOMAIN_DRIFT")
            generated[role] = image > 0
        expected = design.derive_masks(fixed["human"], fixed["equipment_original"],
                                       fixed["equipment_added"], generated["object"])
        if not np.array_equal(expected["equipment"], fixed["equipment"]):
            raise RuntimeError("MASK14_EQUIPMENT_SOURCE_DRIFT")
        if any(not np.array_equal(generated[role], expected[role]) for role in generated):
            raise RuntimeError("MASK14_PROTECTION_ALGEBRA_FAIL")
        if (row.get("object_pixels") != int(generated["object"].sum())
                or row.get("protected_pixels") != int(generated["protected"].sum())
                or row.get("removal_pixels") != int(generated["removal"].sum())
                or row.get("unknown_overlap_pixels") != int(generated["unknown"].sum())):
            raise RuntimeError("MASK14_PIXEL_COUNT_DRIFT")
        if not generated["object"].any():
            empty.append((session, frame))
    return {"anchors_validated": 6, "model_calls": 6,
            "object_empty_anchors": empty,
            "mask_quality_gate": "REJECTED" if empty else "PENDING_HUMAN",
            "visible_object_review": "PENDING_HUMAN",
            "protected_pixel_algebra": "PASS", "clean_authorized": False}


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
        print(json.dumps({"status": status, "task_id": design.TASK_ID,
                          "run_signature": packet["run_signature"],
                          "anchor_count": 6, "model_calls_max": 6,
                          "registered": False, "gpu_called": False}, sort_keys=True))
        return 0
    if readiness(packet) not in {"PUBLISHED_EXACT_CANDIDATE_ROUTE",
                               "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"}:
        raise RuntimeError("MASK14_DRY_RUN_NOT_ROUTABLE")
    register(packet)
    reason = None
    cleanup_proven = True
    try:
        code, log = run_worker(packet)
    except WorkerStopped as exc:
        code, log, reason = 2, exc.log, str(exc)
        cleanup_proven = exc.cleanup_proven
    except Exception as exc:
        code, reason, cleanup_proven = 2, f"{type(exc).__name__}:{exc}", False
        log = ATTEMPT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    gpu_path = ATTEMPT / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    try:
        metrics = validate_mask_output(packet) if code == 0 and gpu.get("status") == "PASSED" else None
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        metrics = None
        reason = f"MASK_GATE:{type(exc).__name__}:{exc}"
    # Terminal publication is intentionally fenced by the same writer and
    # complete structural validation. A failed attempt remains terminal.
    verify_fence(packet)
    status = ("FAILED_RUNTIME_FINAL" if not cleanup_proven else
              "PASSED" if code == 0 and metrics is not None else
              "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else
              "FAILED_RUNTIME_FINAL")
    result = {"schema_version": "exact78-chips-object-prompt-mask14-execution-result-v1",
              "task_id": design.TASK_ID, "status": status,
              "execution_complete": status == "PASSED",
              "run_signature": packet["run_signature"], "metrics": metrics,
              "reason": reason, "deadline_cleanup_proven": cleanup_proven,
              "new_task_packet_allowed": cleanup_proven,
              "mask_anchor": artifact_ref(ATTEMPT / "candidate/MASK_ANCHOR.json") if metrics else None,
              "candidate_result": artifact_ref(ATTEMPT / "candidate/RESULT.json") if metrics else None,
              "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
              "worker_log": artifact_ref(log),
              "mask_structural_gate": "PASS" if metrics else "NOT_PASSED",
              "mask_quality_gate": metrics["mask_quality_gate"] if metrics else "NOT_ADMITTED",
              "clean_called": False, "visual_qualified": False,
              "training_eligible": False, "claim_limit": packet["claim_limit"],
              "next_action": "Review six chip masks; Clean remains unadmitted."}
    atomic_json(ATTEMPT / "RESULT.json", result)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {"schema_version": "exact78-chips-object-prompt-mask14-receipt-v1",
                          "task_id": design.TASK_ID, "status": status,
                          "result": artifact_ref(ATTEMPT / "RESULT.json"),
                          "deletion_authorized": False})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    row = next(item for item in state["tasks"] if item["task_id"] == design.TASK_ID)
    row.update(status=status if cleanup_proven else "FAILED_RUNTIME_RETRYABLE",
               pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT), quality="MASK_STRUCTURE_ONLY_REVIEW_PENDING"
               if metrics else "GPU_CLEANUP_UNPROVEN_GLOBAL_HOLD" if not cleanup_proven
               else "MASK_FAILED_OR_RESOURCE_HOLD", adoption="NOT_ADOPTED",
               review="MACHINE_AUDITED_USER_PENDING")
    if cleanup_proven:
        state["next_task"] = None
    else:
        state["blockers"].append({
            "name": "EXACT78_MASK14_GPU_CLEANUP_UNPROVEN", "status": "OPEN",
            "scope": "ALL_NEW_TASK_PACKETS",
            "resolution": "Independent PID/startticks/group/GPU/lease audit before new Task Packet.",
        })
        state["next_task"] = {
            "task_id": design.TASK_ID, "session": "SIX_CHIPS_MASK14",
            "prerequisites": ["PROVE_GPU_CHILD_EXIT", "PROVE_GPU_LEASE_RELEASE"],
            "expected_resource": "NONE_UNTIL_INDEPENDENT_CLEANUP_AUDIT",
            "stop_condition": "No new Task Packet before independent cleanup proof.",
        }
    if issues := validate_task_state(state):
        raise RuntimeError("MASK14_TERMINAL_STATE_INVALID:" + ";".join(issues))
    for item in index["task_packets"]:
        if item.get("task_id") == design.TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] +
                 ("_TERMINAL" if cleanup_proven else "_CLEANUP_HOLD"),
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS" if cleanup_proven else
                 "BLOCKED_GPU_CLEANUP_PROOF", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_" + status,
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    print(json.dumps({"status": status, "result": artifact_ref(ATTEMPT / "RESULT.json")},
                     sort_keys=True))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
