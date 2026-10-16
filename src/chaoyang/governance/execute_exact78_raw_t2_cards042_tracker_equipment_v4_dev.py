"""Finite, unregistered Cards042 developer retry of tracker equipment Mask.

``--check-only`` is read-only. ``--execute`` requires a staged/published v4
candidate route and no active task; it publishes that route with its packet in
one CAS, then runs exactly one full 171-frame fenced GPU attempt. Never use a
holdout or an earlier Clean/Mask as producer input.
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
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH, RECEIPT_PATH,
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes,
    load_json, now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance import execute_exact78_raw_t2_cards042_frontend_v3_mp4_retry1 as base


TASK = "exact78_raw_to_robot_t2_cards042_tracker_equipment_v4_dev_20261001"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = OUT / "FRONTEND_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T2_CARDS042_TRACKER_EQUIPMENT_V4_DEV.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = base.SESSION
PLAN = base.PLAN
T0 = base.T0
AUDIT = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/16_CARDS042_CLEAN_FAILURE_CLUSTER_AUDIT_ZH.md"
PREP = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/17_TRACKER_DEVICE_PROMPT_UNION_CANDIDATE_ZH.md"
OLD_RESULT = REPO_ROOT / "_run/current/exact78_raw_to_robot_t2_cards042_frontend_v1_20260930/attempts/attempt_0001/RESULT.json"
CAMPAIGN_END = base.CAMPAIGN_END
OPERATION = "run_exact78_fresh_frontend_v4_tracker_equipment"
CONTRACT_KEY = "exact78_tracker_equipment_prompt_union_cards042_dev_v4_20261001"
PYTHON = base.PYTHON
COMMAND = (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} "
           f"--spec {SPEC} --phase all")
STAGES = base.STAGES
V4_PATH = REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v4_tracker_equipment.py"
V4_SHA = "f285a8429c1eb6bc1309c293d53c00cbfc98ede5e158f240020036a1a11d597b"


def frozen_inputs() -> dict[str, Any]:
    inputs = base.frozen_inputs()
    if artifact_ref(OLD_RESULT)["sha256"] != "d065012d7d774dd5c7f8b4d42be0555dc7f5f3d9c28e29a950be65de7e027684":
        raise RuntimeError("CARDS042_SEALED_OLD_RESULT_DRIFT")
    inputs["clean_failure_audit"] = artifact_ref(AUDIT)
    inputs["prompt_union_prep"] = artifact_ref(PREP)
    return inputs


def spec_value(inputs: dict[str, Any]) -> dict[str, Any]:
    spec = base.spec_value(inputs)
    spec.update(task_id=TASK, output_root=str(OUT / "candidate"))
    from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec
    validate_spec(spec)
    return spec


def runtime_refs() -> list[dict[str, Any]]:
    if artifact_ref(V4_PATH)["sha256"] != V4_SHA:
        raise RuntimeError("V4_ALGORITHM_SHA_DRIFT")
    refs = [artifact_ref(SCRIPT), artifact_ref(V4_PATH), *base.runtime_refs()]
    paths = [ref["path"] for ref in refs]
    if len(paths) != len(set(paths)):
        raise RuntimeError("V4_RUNTIME_CLOSURE_DUPLICATE")
    return refs


def packet_value(inputs: dict[str, Any], spec_ref: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = runtime_refs()
    all_inputs = {**inputs, "frontend_spec": spec_ref}
    config = {"session_id": SESSION.name, "frame_count": 171, "fps_nominal": 30,
              "stages": list(STAGES), "gpu_lease_count": 1, "training_steps": 0,
              "algorithm_delta": "TRACKER_EQUIPMENT_PROMPT_UNION_ONLY",
              "prompt_roles": ["equipment_tracker", "equipment_wristband"],
              "probability_threshold": 0.5,
              "mp4_acceptance": "EXACT_STREAM_TICKS_AND_MVHD_QUANTIZATION_V2",
              "visual_quality": "PENDING_REVIEW", "geometry": "NOT_IN_T2"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": all_inputs, "runtime_code": runtime, "config": config,
        "command": COMMAND, "schema": "exact78-cards042-tracker-equipment-v4-dev-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full Cards042 development raw MAP→CLEAN retry of tracker equipment synonym union",
        "phase": "EXACT78_RAW_TO_ROBOT_T2_TRACKER_EQUIPMENT_DEV",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=2), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(AUDIT.relative_to(REPO_ROOT)), str(PREP.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT)), str(SCRIPT.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance", "_run/current/GPU_LEASE.json",
                      "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "PREVIOUS_CARDS042_T2_TERMINAL",
                          "NO_ACTIVE_TASK", "ONE_GPU_LEASE", "SOURCE_SHA_PINNED",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["FRONTEND_SPEC.json", "candidate/CAMERA_MAPS.npz",
                             *(f"candidate/{name}.json" for name in STAGES),
                             *(f"candidate/SEAL_{name}.json" for name in STAGES),
                             "candidate/propainter_model/frames/masked_in.mp4",
                             "candidate/propainter_model/frames/inpaint_out.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 7200, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "STAGE_SEAL_FAIL", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards042 development Mask synonym candidate only; nonzero device pixels do not confer Clean quality, geometry, Robot, product or training qualification",
        "writer": writer, "inputs": all_inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
        "post_result_ab_reference": {"path": str(OLD_RESULT),
                                     "use": "EVALUATOR_ONLY_AFTER_FRESH_RESULT"},
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_V4_PACKET:" + ";".join(errors))
    return packet


def route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    required = [artifact_ref(path) for path in (V4_PATH,
        REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v3.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
        REPO_ROOT / "src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py")]
    return (route.get("candidate_operation") == OPERATION
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-fresh-frontend-stage-seal-v2"
            and route.get("model_calls_authorized") == "ONE_FULL_CARDS042_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE"
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in required))


def candidate_route_readiness() -> tuple[bool, str]:
    from chaoyang.cli import _maintained_operations
    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CANDIDATE_ROUTE") if route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "V4_PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    return ((True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET") if route_valid(staged)
            else (False, "V4_CANDIDATE_ROUTE_NOT_STAGED"))


def readiness() -> tuple[bool, str]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        return False, "FRESH_V4_TASK_REQUIRED"
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    prior = [row for row in state["tasks"] if row["task_id"] == base.TASK]
    original = [row for row in state["tasks"] if row["task_id"] == "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930"]
    if (not original or original[0]["status"] != "PASSED"
            or (prior and prior[0]["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"))):
        return False, "CARDS042_PREVIOUS_TASK_NOT_TERMINAL"
    return candidate_route_readiness()


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = base._index_copy()
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["frontend_spec"]:
        raise RuntimeError("FRONTEND_SPEC_PRECOMPUTED_SHA_DRIFT")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": SESSION.name, "updated_at": now, "heartbeat_at": now,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": 0, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": SESSION.name,
        "prerequisites": packet["prerequisites"], "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full Cards042 v4 frontend result or campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V4_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("V4_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("V4_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    route = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if len(route) != 1 or route[0]["task_id"] != TASK or route[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("V4_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("V4_CODE_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-cards042-tracker-v4-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "sealed_stages": [name for name in STAGES if (OUT / "candidate" / f"SEAL_{name}.json").is_file()],
            "next_action": "Wait for fenced worker or inspect terminal receipt"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = OUT / "cache"
    cache.mkdir()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    remaining = int((datetime.fromisoformat(packet["deadline_at"]) - datetime.now().astimezone()).total_seconds())
    if remaining < 300:
        raise RuntimeError("V4_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(600, max(0, remaining // 10))
    wall_seconds = min(5400, remaining - wait_seconds - 60)
    lease_command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
        "run_gpu_command_with_v71_lease", "--task-id", TASK,
        "--attempt-id", "attempt_0001_cards042_tracker_equipment_v4", "--priority", "CANARY",
        "--gpu-id", "0", "--min-free-mib", "60000", "--wait-seconds", str(wait_seconds),
        "--wall-seconds", str(wall_seconds), "--receipt", str(OUT / "GPU_RECEIPT.json"),
        "--claim-limit", "One Cards042 developer Mask synonym retry; no visual or training authority",
        "--", str(PYTHON), "-B", "-m", "chaoyang.cli", "run", OPERATION,
        "--spec", str(SPEC), "--phase", "all"]
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(lease_command, cwd=REPO_ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        last_beat = started
        while process.poll() is None:
            if time.monotonic() - last_beat >= 50:
                heartbeat(packet, time.monotonic() - started)
                last_beat = time.monotonic()
            time.sleep(2)
    return process.returncode, log


def verify_mask_and_protection(output: Path) -> dict[str, int]:
    record = load_json(output / "MASK.json")
    clean = load_json(output / "CLEAN.json")
    if (record.get("schema_version") != "exact78-fresh-frontend-mask-v4-tracker-equipment"
            or len(record["rows"]) != 171 or len(clean["rows"]) != 171):
        raise RuntimeError("V4_MASK_SCHEMA_OR_FRAME_COUNT")
    detected = 0
    protected_total = 0
    for index, (row, clean_row) in enumerate(zip(record["rows"], clean["rows"])):
        if row["decode_display_index"] != index or clean_row["decode_display_index"] != index:
            raise RuntimeError(f"V4_MASK_FRAME_MAPPING:{index}")
        images = {}
        for key, ref in row["masks"].items():
            value = cv2.imread(ref["path"], cv2.IMREAD_GRAYSCALE)
            if value is None:
                raise RuntimeError(f"V4_MASK_IMAGE_MISSING:{index}:{key}")
            images[key] = value > 0
        if any(value.shape != (960, 1280) for value in images.values()):
            raise RuntimeError(f"V4_MASK_IMAGE_DOMAIN:{index}")
        equipment = images["equipment_tracker"] | images["equipment_wristband"]
        human_equipment = images["human"] | equipment
        obj = images["object"]
        if (not np.array_equal(images["equipment"], equipment)
                or not np.array_equal(images["human_equipment"], human_equipment)
                or not np.array_equal(images["protected"], obj & ~human_equipment)
                or not np.array_equal(images["removal"], human_equipment & ~obj)
                or not np.array_equal(images["unknown"], human_equipment & obj)):
            raise RuntimeError(f"V4_OBJECT_PROTECTION_DRIFT:{index}")
        if row["equipment_candidate_pixels"]["equipment_union"] != int(equipment.sum()):
            raise RuntimeError(f"V4_DEVICE_PIXEL_COUNT_DRIFT:{index}")
        if row["role_status"] != ("MODEL_OUTPUT_UNREVIEWED" if equipment.any() else "MASK_DEVICE_UNVERIFIED"):
            raise RuntimeError(f"V4_DEVICE_STATUS_DRIFT:{index}")
        raw = cv2.imread(row["model_input"]["path"])
        cleaned = cv2.imread(clean_row["clean"]["path"])
        if (raw is None or cleaned is None or raw.shape != (960, 1280, 3)
                or cleaned.shape != raw.shape
                or not np.array_equal(raw[images["protected"]], cleaned[images["protected"]])):
            raise RuntimeError(f"V4_PROTECTED_OBJECT_PIXEL_DRIFT:{index}")
        detected += bool(equipment.any())
        protected_total += int(images["protected"].sum())
    return {"equipment_nonempty_frames": detected,
            "equipment_empty_frames": 171 - detected,
            "protected_object_pixels_raw_preserved": protected_total}


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    stage_refs = {}
    stage_errors = []
    previous_ref: dict[str, Any] | str = "ABSENT"
    for name in STAGES:
        path = OUT / "candidate" / f"SEAL_{name}.json"
        if path.is_file():
            seal = load_json(path)
            ref = seal.get("result")
            if (seal.get("session_id") != SESSION.name or seal.get("stage") != name
                    or seal.get("frontend_spec") != packet["inputs"]["frontend_spec"]
                    or seal.get("upstream_seal") != previous_ref
                    or seal.get("frame_count") != (0 if name == "MAP" else 171)
                    or not isinstance(ref, dict) or validate_artifact_ref(ref)):
                stage_errors.append(f"STAGE_SEAL_DRIFT:{name}")
            stage_refs[name] = artifact_ref(path)
            previous_ref = stage_refs[name]
        else:
            stage_errors.append(f"STAGE_SEAL_MISSING:{name}")
    metrics = None
    if "CLEAN" in stage_refs:
        try:
            base.verify_internal_video_records(OUT / "candidate")
            metrics = verify_mask_and_protection(OUT / "candidate")
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            stage_errors.append(f"CLEAN_OR_MASK_VALIDATION:{error}")
    input_errors = []
    for key, ref in packet["inputs"]["source_files"].items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{key}:{';'.join(errors)}")
    for ref in (packet["inputs"]["models"]["sam_checkpoint"],
                *packet["inputs"]["models"]["propainter_weights"].values()):
        if errors := validate_artifact_ref(ref):
            input_errors.append("model:" + ";".join(errors))
    gpu_path = OUT / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    complete = (code == 0 and gpu.get("status") == "PASSED" and not stage_errors
                and not input_errors and reason is None and metrics is not None)
    status = "PASSED" if complete else (
        "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL")
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_V4_TERMINAL_RESULT")
    result = {"schema_version": "exact78-cards042-tracker-equipment-v4-dev-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason,
        "stage_seal_errors": stage_errors, "post_input_sha_errors": input_errors,
        "source_session": SESSION.name, "source_frames_expected": 171,
        "frontend_spec": artifact_ref(SPEC), "stage_seals": stage_refs,
        "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
        "worker_log": artifact_ref(log), "mask_metrics": metrics,
        "mask_device_status": ("MODEL_OUTPUT_UNREVIEWED" if metrics and metrics["equipment_nonempty_frames"]
                               else "MASK_DEVICE_UNVERIFIED"),
        "mask_quality": "NOT_YET_ADOPTED", "clean_quality": "NOT_YET_ADOPTED",
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "source_mutated": False, "files_deleted": False,
        "old_result_ab_status": "NOT_READ_EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "next_action": ("If device still empty, stop this prompt candidate; otherwise independently review visible device coverage, cards and Clean before any adoption"),
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-cards042-tracker-equipment-v4-dev-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_OR_RESOURCE_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("V4_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = base._index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_CARDS042_TRACKER_EQUIPMENT_V4_DEV_{status}",
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
    inputs = frozen_inputs()
    spec = spec_value(inputs)
    spec_ref = base.virtual_json_ref(SPEC, spec)
    packet = packet_value(inputs, spec_ref)
    ready, reason = readiness()
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": TASK, "command": COMMAND,
            "spec_sha256": spec_ref["sha256"], "run_signature": packet["run_signature"],
            "source_frames": 171, "gpu_lease_count": 1,
            "old_result_use": "AB_ONLY_AFTER_FRESH_RESULT"}, ensure_ascii=False))
        return 0
    if not ready:
        raise RuntimeError(reason)
    register(packet, spec)
    try:
        code, log = run_worker(packet)
        failure = None
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = OUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    terminal = finish(packet, code, log, failure)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
