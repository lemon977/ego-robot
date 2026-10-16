"""Finite, unregistered CPU render-only successor for sealed 0902 Cards001.

Only a future idle governance publisher may stage the dedicated CLI route and
this packet in one CAS. ``--check-only`` never creates files or renders.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
    validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance import prepare_exact78_newraw_0902_cards001_render_direction_v1 as prep
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


TASK = prep.TASK_ID
OUT, SPEC, PACKET, RECEIPT = prep.OUT, prep.SPEC, prep.PACKET, prep.RECEIPT
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SCRIPT = Path(__file__)
OPERATION = "run_exact78_newraw_0902_cards001_render_direction_v1"
CONTRACT_KEY = prep.CONTRACT_KEY
PYTHON = Path("/usr/local/bin/python")
COMMAND = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {SPEC}"
WALL_SECONDS = 1800
VIDEO_FILES = prep.VIDEO_FILES
SCHEMA = "exact78-newraw-0902-cards001-render-direction-packet-v1"


class WorkerGroupUncleared(RuntimeError):
    """Do not terminalize the writer when an encoder may still write output."""


def virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def runtime_refs() -> list[dict[str, Any]]:
    files = [SCRIPT, REPO_ROOT / "src/chaoyang/ops" / f"{OPERATION}.py",
             Path(prep.__file__), REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
             *(REPO_ROOT / relative for relative in direction.PINNED_CODE_SHA),
             REPO_ROOT / "src/chaoyang/governance/common.py",
             REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py",
             REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py",
             REPO_ROOT / "src/chaoyang/cli/__init__.py",
             REPO_ROOT / "src/chaoyang/ops/immutable_artifact_io.py"]
    for relative, sha in direction.PINNED_CODE_SHA.items():
        if artifact_ref(REPO_ROOT / relative)["sha256"] != sha:
            raise RuntimeError(f"PINNED_RENDER_CODE_DRIFT:{relative}")
    return [artifact_ref(path) for path in files]


def packet_value(spec: dict[str, Any], *, stamp: datetime | None = None,
                 writer: dict[str, Any] | None = None) -> dict[str, Any]:
    now = stamp or datetime.now().astimezone()
    if now >= prep.CAMPAIGN_END:
        raise RuntimeError("CAMPAIGN_HARD_CUTOFF_PASSED")
    if writer is None:
        writer = {"pid": os.getpid(),
                  "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
                  "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
                  "scope": TASK}
    inputs = {"render_spec": virtual_ref(SPEC, spec),
              "sealed_stage": spec["inputs"]["stage_result"],
              "fixed_inputs": spec["inputs"],
              "frame_states": load_json(prep.STAGE)["outputs"]["frame_states"],
              "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")}
    runtime = runtime_refs()
    config = {"session_id": prep.SESSION_ID, "frames": prep.FRAMES, "fps": prep.FPS,
              "change_only": "EGO_CAMERA_INV_T_CAMERA_BASE_TO_T_CAMERA_BASE",
              "candidate_contract_key": CONTRACT_KEY,
              "flange_parent_changed": False, "q_base_mount_recomputed": False,
              "ik_calls": 0, "model_calls": 0, "gpu_count": 0,
              "numeric_gates_changed": False,
              "projection_frames": list(prep.SAMPLE_FRAMES),
              "old_video_ab": "EVALUATOR_ONLY_AFTER_FRESH_SEAL"}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": COMMAND, "schema": SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full 201-frame 0902 Cards001 render-direction-only diagnostic",
        "phase": "EXACT78_ROBOT_RENDER_DIRECTION_ONLY",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(seconds=WALL_SECONDS), prep.CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": prep.CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(prep.STAGE.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/" + OPERATION + ".py",
                     "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
                     "assets/robot/ROBOT_ASSET_PIN.json",
                     "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"],
        "expanded_read_set": [ref["path"] for ref in (*spec["inputs"].values(), inputs["frame_states"])],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_201_FRAME_STATES",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "CPU_ONLY",
                          "NEW_SESSION_CANDIDATE_ROUTE_STAGED_WITH_PACKET", "NO_MODEL",
                          "NO_CLEAN_GEOMETRY_CLAIM", "NO_DELETION"],
        "required_outputs": ["RENDER_SPEC.json", "WORKER_PROCESS.json",
                             "WORKER_GROUP_CLEANUP.json", "candidate/RESULT.json",
                             *(f"candidate/render/{name}" for name in VIDEO_FILES.values()),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": WALL_SECONDS,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "VIDEO_DECODE_OR_TIME_FAIL", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Execution-only camera diagnostic; no flange, occlusion, visual, product, training or control qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_RENDER_DIRECTION_PACKET:" + ";".join(errors))
    return packet


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    expected = [artifact_ref(REPO_ROOT / relative) for relative in (
        f"src/chaoyang/ops/{OPERATION}.py",
        "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
        *direction.PINNED_CODE_SHA)]
    return (route.get("candidate_operation") == OPERATION
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-robot-render-direction-only-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in expected))


def route_readiness() -> tuple[bool, str]:
    """A staged route and packet are published together by register()."""
    from chaoyang.cli import _maintained_operations
    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CANDIDATE_ROUTE") if _route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    return ((True, "READY_TO_PUBLISH_ROUTE_WITH_PACKET") if _route_valid(staged)
            else (False, "CANDIDATE_ROUTE_NOT_STAGED"))


def readiness() -> tuple[bool, str]:
    if datetime.now().astimezone() >= prep.CAMPAIGN_END:
        return False, "CAMPAIGN_HARD_CUTOFF_PASSED"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        return False, "FRESH_ATTEMPT_REQUIRED"
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if (state.get("next_task") or any(row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}
                                      for row in state["tasks"])
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        return False, "ACTIVE_TASK_CONFLICT"
    return route_readiness()


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    for ref in (*packet["runtime_code"], *spec["inputs"].values(),
                packet["inputs"]["frame_states"], packet["inputs"]["robot_asset_pin"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("SHA_DRIFT_BEFORE_CAS:" + ";".join(errors))
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["render_spec"]:
        raise RuntimeError("SPEC_SHA_DRIFT_BEFORE_CAS")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": spec["session_id"], "updated_at": now,
        "heartbeat_at": now, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": spec["session_id"],
        "prerequisites": packet["prerequisites"], "expected_resource": "TWO_CPU_THREADS_NO_GPU",
        "stop_condition": "one full 201-frame result or 30-minute/campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    if (packet["writer"]["pid"] != os.getpid()
            or packet["writer"]["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != packet["writer"]):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    if (len(active) != 1 or active[0]["task_id"] != TASK
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(routes) != 1 or routes[0]["task_id"] != TASK
            or routes[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    summary = OUT / f"RESUME_SUMMARY_{number:04d}.json"
    if number and not summary.exists():
        atomic_json(summary, {"schema_version": "exact78-render-direction-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "candidate_result_exists": (OUT / "candidate/RESULT.json").is_file(),
            "next_action": "Wait for fenced CPU worker or inspect its terminal receipt"})


def _group_members(pgid: int) -> list[dict[str, Any]]:
    """Read exact Linux process-group membership; zombies cannot write files."""
    rows: list[dict[str, Any]] = []
    try:
        processes = list(Path("/proc").iterdir())
    except OSError as error:
        raise WorkerGroupUncleared(f"WORKER_GROUP_PROC_SCAN_FAILED:{error}") from error
    for proc in processes:
        if not proc.name.isdecimal():
            continue
        try:
            raw = (proc / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue  # exited during the scan
        except (OSError, ValueError) as error:
            raise WorkerGroupUncleared(f"WORKER_GROUP_PROC_UNREADABLE:{proc}:{error}") from error
        fields = raw[raw.rfind(")") + 2:].split()
        if len(fields) < 20:
            raise WorkerGroupUncleared(f"WORKER_GROUP_PROC_MALFORMED:{proc}")
        if int(fields[2]) == pgid and fields[0] != "Z":
            rows.append({"pid": int(proc.name), "state": fields[0],
                         "start_ticks": int(fields[19])})
    return rows


def _assert_group_identity(pid: int, start_ticks: int) -> None:
    identity = process_identity(pid)
    if identity["alive"] and identity["start_ticks"] != start_ticks:
        raise WorkerGroupUncleared("WORKER_GROUP_LEADER_PID_REUSED")


def _settle_group(process: subprocess.Popen[Any], start_ticks: int,
                  *, term_seconds: float = 3.0, kill_seconds: float = 3.0) -> dict[str, Any]:
    """TERM then bounded KILL the exact dedicated session, and prove no live member."""
    pgid = process.pid  # start_new_session=True makes PID=PGID=SID
    if os.getpgrp() == pgid:
        raise WorkerGroupUncleared("WORKER_GROUP_NOT_ISOLATED")
    _assert_group_identity(pgid, start_ticks)
    before = _group_members(pgid)
    signals: list[str] = []
    for sig, seconds in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
        if not _group_members(pgid):
            break
        _assert_group_identity(pgid, start_ticks)
        try:
            os.killpg(pgid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()  # reap the leader; descendants are checked separately
            if not _group_members(pgid):
                break
            time.sleep(0.05)
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    remaining = _group_members(pgid)
    if remaining or process.poll() is None:
        raise WorkerGroupUncleared(f"WORKER_GROUP_NOT_CLEARED:{remaining}")
    # A second scan protects against a process briefly hidden by a concurrent
    # /proc exit/reparent event. It does not treat zombie encoders as writers.
    if residual := _group_members(pgid):
        raise WorkerGroupUncleared(f"WORKER_GROUP_REAPPEARED:{residual}")
    return {"pgid": pgid, "leader_start_ticks": start_ticks,
            "members_at_cleanup": before, "signals": signals,
            "live_members_after_cleanup": 0}


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("BUDGET_TOO_SHORT")
    cache = OUT / "cache"
    cache.mkdir()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
            OPERATION, "--spec", str(SPEC)], cwd=REPO_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        if not identity["alive"] or not isinstance(identity["start_ticks"], int):
            raise WorkerGroupUncleared("WORKER_GROUP_LEADER_IDENTITY_UNVERIFIED")
        if os.getpgid(process.pid) != process.pid:
            raise WorkerGroupUncleared("WORKER_GROUP_NOT_ISOLATED")
        atomic_json(OUT / "WORKER_PROCESS.json", {
            "schema_version": "exact78-render-worker-process-v1",
            "pid": process.pid, "proc_start_ticks": identity["start_ticks"],
            "pgid": process.pid, "sid": process.pid,
            "cleanup_required_before_terminal": True})
        last_beat = 0.0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("RENDER_DIRECTION_DEADLINE")
                if elapsed - last_beat >= 50:
                    heartbeat(packet, elapsed)
                    last_beat = elapsed
                time.sleep(2)
        finally:
            try:
                cleanup = _settle_group(process, identity["start_ticks"])
            except Exception as error:
                raise WorkerGroupUncleared(f"WORKER_GROUP_CLEANUP_UNPROVEN:{error}") from error
            atomic_json(OUT / "WORKER_GROUP_CLEANUP.json", cleanup)
    return process.returncode, log


def verify_candidate(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    path = OUT / "candidate/RESULT.json"
    if not path.is_file():
        return None, ["CANDIDATE_RESULT_MISSING"]
    try:
        from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
        result = load_json(path)
        if (result.get("status") != "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS"
                or result.get("session_id") != prep.SESSION_ID
                or result.get("frame_count") != prep.FRAMES
                or result.get("inputs") != spec["inputs"]
                or len(result.get("frame_rows", [])) != prep.FRAMES
                or result.get("q_base_mount_recomputed") is not False
                or result.get("numeric_gates_changed") is not False
                or result.get("ego_camera_rule") != "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION"
                or result.get("external_camera_rule") != "PINNED_FIXED_EXTERNAL_FROM_BASE_TO_CAMERA"
                or result.get("clean_or_depth_consumed") is not False
                or any(result.get(key) is not False for key in (
                    "visual_qualified", "product_qualified", "training_eligible", "control_ground_truth"))):
            raise RuntimeError("CANDIDATE_IDENTITY_OR_SCOPE_DRIFT")
        for index, row in enumerate(result["frame_rows"]):
            if (row.get("frame") != index or row.get("raw_decoded_display_index") != index
                    or row.get("geometry_decision") != "UNKNOWN"
                    or row.get("display_policy") != "DIAGNOSTIC_ROBOT_OVER_RAW_NO_OCCLUSION_PROOF"
                    or not isinstance(row.get("ego_robot_pixels"), int)
                    or not isinstance(row.get("external_robot_pixels"), int)
                    or row["ego_robot_pixels"] < 0 or row["external_robot_pixels"] < 0
                    or not isinstance(row.get("final_visible"), bool)):
                raise RuntimeError(f"FRAME_OBSERVATION_DRIFT:{index}")
            if row["external_robot_pixels"] == 0:
                raise RuntimeError(f"EXTERNAL_ROBOT_ZERO_PIXELS:{index}")
        if set(result.get("videos", {})) != set(VIDEO_FILES):
            raise RuntimeError("VIDEO_SET_DRIFT")
        for role, filename in VIDEO_FILES.items():
            path = OUT / "candidate/render" / filename
            checked = legacy._verify_video(path, prep.FRAMES,
                                           (legacy.WIDTH, legacy.HEIGHT), prep.FPS)
            if result["videos"][role] != checked:
                raise RuntimeError(f"VIDEO_SHA_OR_TIME_DRIFT:{role}")
        for role, ref in spec["inputs"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"INPUT_SHA_DRIFT:{role}:{';'.join(errors)}")
        return result, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"CANDIDATE_VALIDATION:{error}"]


def finish(packet: dict[str, Any], spec: dict[str, Any], code: int,
           log: Path, reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    cleanup_path = OUT / "WORKER_GROUP_CLEANUP.json"
    identity_path = OUT / "WORKER_PROCESS.json"
    if not cleanup_path.is_file() or not identity_path.is_file():
        raise WorkerGroupUncleared("WORKER_GROUP_CLEANUP_PROOF_ABSENT")
    cleanup, identity = load_json(cleanup_path), load_json(identity_path)
    if (cleanup.get("live_members_after_cleanup") != 0
            or cleanup.get("pgid") != identity.get("pgid")
            or cleanup.get("leader_start_ticks") != identity.get("proc_start_ticks")
            or identity.get("pid") != identity.get("pgid")):
        raise WorkerGroupUncleared("WORKER_GROUP_CLEANUP_PROOF_DRIFT")
    _assert_group_identity(identity["pid"], identity["proc_start_ticks"])
    if _group_members(identity["pgid"]):
        raise WorkerGroupUncleared("WORKER_GROUP_REAPPEARED_BEFORE_TERMINAL")
    candidate, errors = verify_candidate(spec)
    complete = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    result = {"schema_version": "exact78-newraw-0902-cards001-render-direction-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "validation_errors": errors,
        "source_session": prep.SESSION_ID, "source_frames_expected": prep.FRAMES,
        "render_spec": artifact_ref(SPEC),
        "candidate_result": artifact_ref(OUT / "candidate/RESULT.json") if candidate else None,
        "worker_log": artifact_ref(log), "q_base_mount_recomputed": False,
        "worker_process": artifact_ref(identity_path),
        "worker_group_cleanup": artifact_ref(cleanup_path),
        "flange_parent_changed": False, "numeric_gates_changed": False,
        "zero_visible_frame_indices": ([row["frame"] for row in candidate["frame_rows"]
                                       if not row["final_visible"]] if candidate else []),
        "per_frame_robot_pixel_counts": "SEE_CANDIDATE_RESULT_FRAME_ROWS",
        "projection_review_frames": list(prep.SAMPLE_FRAMES),
        "projection_review_status": "PENDING_FIXED_Q_VS_HAWOR_MODEL_WRIST_EVALUATOR",
        "old_video_ab_status": "NOT_READ_EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "ai_review": "PENDING", "user_review": "PENDING",
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "source_mutated": False, "files_deleted": False,
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-newraw-0902-cards001-render-direction-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TERMINAL_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_NEWRAW_0902_CARDS001_RENDER_DIRECTION_V1_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec, _, _ = prep.frozen_spec()
    packet = packet_value(spec)
    ready, reason = readiness()
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": TASK, "route_key": CONTRACT_KEY,
            "command": COMMAND, "spec_sha256": packet["inputs"]["render_spec"]["sha256"],
            "run_signature": packet["run_signature"], "frames": prep.FRAMES,
            "gpu_count": 0, "model_calls": 0, "wall_seconds_max": WALL_SECONDS,
            "new_attempt_write_set": packet["write_set"][0]}, ensure_ascii=False))
        return 0
    if not ready:
        raise RuntimeError(reason)
    register(packet, spec)
    try:
        code, log = run_worker(packet)
        failure = None
    except WorkerGroupUncleared as error:
        atomic_json(OUT / "WORKER_GROUP_HOLD.json", {
            "schema_version": "exact78-render-worker-group-hold-v1",
            "reason": str(error), "writer_remains_active": True,
            "next_action": "Prove exact process group termination before manually terminalizing this packet"})
        raise  # no finish(): an unknown live encoder must retain the sole writer
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = OUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    try:
        terminal = finish(packet, spec, code, log, failure)
    except WorkerGroupUncleared as error:
        if not (OUT / "WORKER_GROUP_HOLD.json").exists():
            atomic_json(OUT / "WORKER_GROUP_HOLD.json", {
                "schema_version": "exact78-render-worker-group-hold-v1",
                "reason": str(error), "writer_remains_active": True,
                "next_action": "Prove exact process group termination before manually terminalizing this packet"})
        raise
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
