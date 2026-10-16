"""Prepared finite executor for one 0901 Chips001 CPU display candidate.

The public module entry stays check-only. A future staged route and explicit
caller may invoke execute_once; this file does not register or run on import.
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
import time
from typing import Any

import numpy as np

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH as GOVERNANCE_RECEIPT_PATH, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.ops import run_exact78_chips001_first_anchor_display_cpu_v1 as operation


ROOT = operation.ROOT
TASK_ID = operation.TASK_ID
PACKET_PATH = ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT_PATH = ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
ATTEMPT = ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
CODE = (
    "src/chaoyang/governance/execute_exact78_chips001_first_anchor_display_cpu_v1.py",
    "src/chaoyang/governance/publish_exact78_chips001_first_anchor_display_cpu_v1.py",
    "src/chaoyang/ops/run_exact78_chips001_first_anchor_display_cpu_v1.py",
    "src/chaoyang/pipeline/exact78_chips001_first_anchor_display_safety_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "tests/unit/test_exact78_chips001_first_anchor_display_cpu_prep_v1.py",
    "tests/unit/test_exact78_chips001_first_anchor_display_safety_v1.py",
)


class WorkerNotStoppedError(RuntimeError):
    """A fenced worker may still be alive, so terminal publication is unsafe."""


def packet_value() -> dict[str, Any]:
    """Construct a finite, reviewable packet in memory; never publish it."""
    operation.check_only()
    inputs = operation._refs()
    runtime = [file_ref(ROOT / relative) for relative in CODE]
    now = datetime.now().astimezone()
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run "
               + operation.OPERATION + " --output-root " + str(operation.OUTPUT))
    config = {
        "session_id": operation.SESSION, "frame_count": operation.FRAMES,
        "fps_nominal": operation.FPS, "source_kind": "tracker",
        "display_policy": "FIRST_LAST_EDGE_HOLD_TWO_END_INTERNAL_NEUTRAL_NO_ANCHOR",
        "short_gap_rule": "10/30_SECONDS_25FPS_8_FRAMES",
        "ik_changed": False, "numeric_gates_changed": False,
        "old_robot_q_action_input": False,
        "source_ik_role": "SEALED_STRUCTURAL_V2_FRESH_HAWOR_DIAGNOSTIC_Q_ONLY",
        "gpu_count": 0, "model_calls": 0, "render_calls": 0,
        "training_steps": 0, "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC",
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command,
        "schema": "exact78-chips001-first-anchor-display-cpu-task-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "draft_status": "PREP_ONLY_NOT_REGISTERED_NOT_EXECUTABLE",
        "task_id": TASK_ID,
        "objective": "One sealed 0901 Chips001 799x2 CPU display-source and continuity diagnostic",
        "phase": "EXACT78_0901_CHIPS001_FIRST_ANCHOR_DISPLAY_CPU_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_0901_CHIPS001_FIRST_ANCHOR_DISPLAY_CPU_V1",
        "execution_mode": "EXACT78_CANDIDATE",
        "prepared_at": now.isoformat(),
        "registered_at": None,
        "deadline_at": None,
        "deadline_policy": "15_MINUTES_FROM_FUTURE_REGISTRATION",
        "read_set": [
            "AGENTS.md",
            *(str(Path(inputs[key]["path"]).relative_to(ROOT)) for key in (
                "v2_outer", "v2_candidate", "stage_result", "states",
                "frame_rows", "hawor_outer", "hawor_npz")),
        ],
        "write_set": [
            str(ATTEMPT.relative_to(ROOT)), str(PACKET_PATH.relative_to(ROOT)),
            str(RECEIPT_PATH.relative_to(ROOT)), "tasks/current/INDEX.json",
            "docs/governance",
        ],
        "prerequisites": [
            "GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_0901_CHIPS001_V2",
            "FRESH_HAWOR_AND_IK_SHA_PINNED", "CPU_ONLY", "NO_OLD_ROBOT_Q_ACTION",
            "NO_RENDER_OR_MODEL", "EXTERNAL_DATA_READ_ONLY", "NO_DELETION",
        ],
        "required_outputs": [
            "candidate/display_cpu/DISPLAY_Q_DIAGNOSTIC.npz",
            "candidate/display_cpu/DISPLAY_FRAME_STATES.json",
            "candidate/display_cpu/RESULT.json", "RESULT.json", "terminal_receipt",
        ],
        "budgets": {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
            "model_calls": 0, "render_calls": 0, "training_steps": 0,
            "wall_seconds_max": 900, "session_attempts_max": 1,
        },
        "stop_conditions": [
            "SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
            "MISSING_FRESH_1598_FRAME_SIDE_GRID", "NONFINITE_DISPLAY_Q",
            "DISPLAY_OR_SOURCE_POLICY_DRIFT", "ONE_ATTEMPT_TERMINAL",
        ],
        "claim_limit": "CPU diagnostic display q/source/step ledger only; no video, visual, product, motion, control or training qualification",
        "writer": None, "fenced_writer_required_at_registration": True,
        "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("CHIPS001_DRAFT_PACKET_INVALID:" + ";".join(errors))
    return packet


def check_only() -> dict[str, Any]:
    packet = packet_value()
    return {
        "status": "PREP_ONLY_REVIEWABLE_PACKET_NOT_REGISTERED_NOT_EXECUTABLE",
        "task_id": TASK_ID, "run_signature": packet["run_signature"],
        "source_states_sha256": packet["inputs"]["states"]["sha256"],
        "frame_count": packet["config"]["frame_count"],
        "attempts_max": packet["budgets"]["session_attempts_max"],
        "gpu_count": 0, "model_calls": 0, "render_calls": 0,
        "active_packet_written": False, "candidate_executed": False,
    }


def finalize_packet(draft: dict[str, Any]) -> dict[str, Any]:
    """Give a fresh in-memory draft one live writer and a 900-second deadline."""
    if (draft.get("draft_status") != "PREP_ONLY_NOT_REGISTERED_NOT_EXECUTABLE"
            or draft.get("writer") is not None or draft.get("registered_at") is not None):
        raise RuntimeError("CHIPS001_FINALIZE_REQUIRES_UNPUBLISHED_DRAFT")
    packet = copy.deepcopy(draft)
    now = datetime.now().astimezone()
    packet.pop("draft_status")
    packet["registered_at"] = now.isoformat()
    packet["deadline_at"] = (now + timedelta(seconds=900)).isoformat()
    packet["campaign_deadline_at"] = packet["deadline_at"]
    packet["writer"] = {
        "pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID,
    }
    packet["fenced_writer_required_at_registration"] = False
    if errors := _validate_packet(packet):
        raise RuntimeError("CHIPS001_FINAL_PACKET_INVALID:" + ";".join(errors))
    return packet


def verify_fence(packet: dict[str, Any]) -> None:
    from chaoyang.governance import publish_exact78_chips001_first_anchor_display_cpu_v1 as publisher

    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET_PATH) != packet):
        raise RuntimeError("CHIPS001_WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or {}).get("task_id") != TASK_ID:
        raise RuntimeError("CHIPS001_CURRENT_TASK_FENCE_DRIFT")
    active = [row for row in state["tasks"] if row["status"] in
              ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    routed = [row for row in load_json(publisher.INDEX)["task_packets"]
              if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != TASK_ID
            or active[0]["run_signature"] != packet["run_signature"]
            or active[0]["pid"] != writer["pid"]
            or active[0]["proc_start_ticks"] != writer["proc_start_ticks"]
            or len(routed) != 1 or routed[0]["task_id"] != TASK_ID
            or routed[0]["packet_sha256"] != artifact_ref(PACKET_PATH)["sha256"]):
        raise RuntimeError("CHIPS001_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    if packet["inputs"] != operation._refs():
        raise RuntimeError("CHIPS001_ACTIVE_INPUT_SHA_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("CHIPS001_ACTIVE_CODE_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_0901_CHIPS001_FIRST_ANCHOR_CPU_HEARTBEAT",
                   expected_revision=load_json(GOVERNANCE_RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__))
    segment = int(elapsed // 600)
    resume = ATTEMPT / f"RESUME_SUMMARY_{segment:04d}.json"
    if segment and not resume.exists():
        atomic_json(resume, {"schema_version": "exact78-chips001-cpu-resume-v1",
                             "elapsed_seconds": round(elapsed, 1),
                             "candidate_result_exists": (operation.OUTPUT / "RESULT.json").is_file(),
                             "next_action": "Continue fenced worker or terminal failure"})


def _stop_fenced_group(worker: subprocess.Popen[Any], start_ticks: int) -> None:
    if worker.poll() is None:
        try:
            current = process_identity(worker.pid)["start_ticks"]
            group = os.getpgid(worker.pid)
        except (OSError, KeyError, ValueError) as error:
            raise WorkerNotStoppedError("CHIPS001_WORKER_IDENTITY_LOST") from error
        if current != start_ticks or group != worker.pid:
            raise WorkerNotStoppedError("CHIPS001_WORKER_GROUP_FENCE_DRIFT")
        os.killpg(group, signal.SIGTERM)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if (process_identity(worker.pid)["start_ticks"] != start_ticks
                    or os.getpgid(worker.pid) != worker.pid):
                raise WorkerNotStoppedError("CHIPS001_WORKER_KILL_FENCE_DRIFT")
            os.killpg(worker.pid, signal.SIGKILL)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise WorkerNotStoppedError("CHIPS001_WORKER_SURVIVED_KILL") from error
    if worker.poll() is None:
        raise WorkerNotStoppedError("CHIPS001_WORKER_STILL_LIVE")
    for _ in range(20):
        try:
            os.killpg(worker.pid, 0)
        except ProcessLookupError:
            return
        except OSError as error:
            raise WorkerNotStoppedError("CHIPS001_GROUP_PROBE_FAILED") from error
        time.sleep(.1)
    raise WorkerNotStoppedError("CHIPS001_GROUP_STILL_LIVE")


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = ATTEMPT / "cache"
    cache.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
               NUMBA_CACHE_DIR=str(cache / "numba"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 60:
        raise RuntimeError("CHIPS001_BUDGET_TOO_SHORT")
    log = ATTEMPT / "worker.log"
    start = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(
            ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
             operation.OPERATION, "--output-root", str(operation.OUTPUT)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            start_ticks = process_identity(worker.pid)["start_ticks"]
        except (OSError, KeyError, ValueError) as error:
            if worker.poll() is None:
                raise WorkerNotStoppedError("CHIPS001_WORKER_START_IDENTITY_LOST") from error
            start_ticks = -1
        last_heartbeat = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - start
                if datetime.now().astimezone() >= deadline or elapsed >= 900:
                    raise RuntimeError("CHIPS001_HARD_DEADLINE")
                if elapsed - last_heartbeat >= 55:
                    heartbeat(packet, elapsed)
                    last_heartbeat = elapsed
                time.sleep(2)
        finally:
            _stop_fenced_group(worker, start_ticks)
    return worker.returncode, log


def verify_candidate(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage

    path = operation.OUTPUT / "RESULT.json"
    if not path.is_file():
        return None, ["CHIPS001_CANDIDATE_MISSING"]
    try:
        value = load_json(path)
        if (value.get("schema_version") != "exact78-chips001-display-cpu-result-v1"
                or value.get("status") != "COMPLETE_CPU_DIAGNOSTIC_NOT_VISUAL_PASS"
                or value.get("session_id") != operation.SESSION
                or value.get("frame_count") != 799 or value.get("frame_side_rows") != 1598
                or value.get("source_refs") != packet["inputs"]
                or value.get("old_robot_q_action_input") is not False
                or value.get("model_calls") != 0 or value.get("render_calls") != 0
                or value.get("visual_qualified") is not False
                or value.get("training_eligible") is not False
                or value.get("control_ground_truth") is not False):
            raise RuntimeError("CHIPS001_CANDIDATE_SCOPE_DRIFT")
        expected = {"display_q": operation.OUTPUT / "DISPLAY_Q_DIAGNOSTIC.npz",
                    "frame_states": operation.OUTPUT / "DISPLAY_FRAME_STATES.json"}
        if set(value.get("outputs", {})) != set(expected):
            raise RuntimeError("CHIPS001_OUTPUT_REF_SET_DRIFT")
        for role, target in expected.items():
            ref = value["outputs"][role]
            if ref.get("path") != str(target) or validate_artifact_ref(ref):
                raise RuntimeError("CHIPS001_OUTPUT_SHA_DRIFT:" + role)
        frame_rows = load_json(expected["frame_states"])
        rows = frame_rows.get("rows", [])
        grid = [(frame, side) for frame in range(799) for side in ("left", "right")]
        if (frame_rows.get("session_id") != operation.SESSION
                or len(rows) != 1598
                or [(row.get("frame"), row.get("side")) for row in rows] != grid
                or any(row.get("robot_drawable") is not True
                       or row.get("display_arm_visible") is not True
                       or row.get("motion_evidence_eligible") is not False
                       or row.get("visual_qualified") is not False
                       or row.get("training_eligible") is not False
                       or row.get("control_ground_truth") is not False for row in rows)):
            raise RuntimeError("CHIPS001_DISPLAY_GRID_OR_QUALIFICATION_DRIFT")
        with np.load(expected["display_q"], allow_pickle=False) as archive:
            if set(archive.files) != {"q_arm_display", "q_hand_display_diagnostic",
                                      "prediction_valid", "original_frame_indices"}:
                raise RuntimeError("CHIPS001_DISPLAY_STATE_KEYS_DRIFT")
            q = archive["q_arm_display"]
            if (q.shape != (799, 2, 7)
                    or archive["q_hand_display_diagnostic"].shape != (799, 2, 22)
                    or archive["prediction_valid"].shape != (2, 799)
                    or not np.array_equal(archive["original_frame_indices"], np.arange(799))
                    or not np.isfinite(q).all()):
                raise RuntimeError("CHIPS001_DISPLAY_STATES_INVALID")
            for row in rows:
                frame = row["frame"]
                side = 0 if row["side"] == "left" else 1
                step = None if frame == 0 else float(np.max(np.abs(
                    q[frame, side] - q[frame - 1, side])))
                if row.get("display_minus_previous_display_max_abs_rad") != step:
                    raise RuntimeError("CHIPS001_DISPLAY_STEP_DRIFT")
                expected_gate = ("NO_PREVIOUS_FRAME" if step is None else "PASS"
                                 if step <= stage.ARM_VELOCITY_GATE_RAD + 1e-9 else "FAIL")
                if row.get("display_step_existing_velocity_gate") != expected_gate:
                    raise RuntimeError("CHIPS001_DISPLAY_STEP_GATE_DRIFT")
        return value, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"CHIPS001_VALIDATION:{error}"]


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    from chaoyang.governance import publish_exact78_chips001_first_anchor_display_cpu_v1 as publisher

    verify_fence(packet)
    candidate, errors = verify_candidate(packet)
    try:
        if packet["inputs"] != operation._refs():
            errors.append("CHIPS001_POST_INPUT_REF_DRIFT")
    except Exception as error:
        errors.append("CHIPS001_POST_INPUT_VERIFY:" + type(error).__name__)
    for ref in packet["runtime_code"]:
        if ref_errors := validate_artifact_ref(ref):
            errors.append("CHIPS001_POST_CODE_REF_DRIFT:" + ";".join(ref_errors))
    passed = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if passed else "FAILED_RUNTIME_FINAL"
    if (ATTEMPT / "RESULT.json").exists() or RECEIPT_PATH.exists():
        raise RuntimeError("CHIPS001_DUPLICATE_TERMINAL")
    outer = {
        "schema_version": "exact78-chips001-display-cpu-execution-result-v1",
        "task_id": TASK_ID, "status": status,
        "source_session": operation.SESSION, "source_frames_expected": 799,
        "worker_returncode": code, "failure_code": reason,
        "validation_errors": errors,
        "stage_result": (artifact_ref(operation.OUTPUT / "RESULT.json")
                         if candidate is not None else None),
        "worker_log": artifact_ref(log),
        "fresh_v2_ik_states": packet["inputs"]["states"],
        "old_robot_q_action_input": False,
        "model_calls": 0, "render_calls": 0, "training_steps": 0,
        "motion_evidence_eligible": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "control_ground_truth": False, "user_review": "PENDING",
        "partial_candidate_preserved": bool(not passed and operation.OUTPUT.exists()),
        "next_action": ("Review 1598 CPU rows; render only under a separate task"
                        if passed else "Inspect preserved partial and use a new successor"),
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(ATTEMPT / "RESULT.json", outer)
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT_PATH, {
        "schema_version": "exact78-chips001-display-cpu-receipt-v1",
        "task_id": TASK_ID, "status": status,
        "result": artifact_ref(ATTEMPT / "RESULT.json"),
        "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT_PATH),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if passed else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if state_errors := validate_task_state(state):
        raise RuntimeError("CHIPS001_TERMINAL_STATE_PREFLIGHT:" + ";".join(state_errors))
    index = publisher._index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_0901_CHIPS001_FIRST_ANCHOR_CPU_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(publisher.INDEX))
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_0901_CHIPS001_FIRST_ANCHOR_CPU_{status}",
        expected_revision=load_json(GOVERNANCE_RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=publisher.INDEX,
        task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(ATTEMPT / "RESULT.json")}


def execute_once() -> dict[str, Any]:
    """Future explicit one-attempt entry; currently fails before publication."""
    from chaoyang.governance import publish_exact78_chips001_first_anchor_display_cpu_v1 as publisher

    publication = publisher.publish()
    packet = publication["packet"]
    try:
        code, log = run_worker(packet)
        reason = None
    except WorkerNotStoppedError:
        raise  # Preserve active fence and partial until process identity is resolved.
    except Exception as error:
        code = 1
        reason = f"{type(error).__name__}:{error}"
        log = ATTEMPT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    return finish(packet, code, log, reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        from chaoyang.governance import publish_exact78_chips001_first_anchor_display_cpu_v1 as publisher
        value = publisher.check_only()
    else:
        value = execute_once()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0 if value.get("status") != "FAILED_RUNTIME_FINAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
