"""CPU-only, unregistered Mask14 object-query candidate boundary.

This module binds the six chip anchors to sealed Mask13 evidence and defines
the only permitted prompt delta and pixel algebra. It never invokes a model,
publishes a route or packet, or creates an output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
from datetime import datetime, timedelta
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, REPO_ROOT, artifact_ref, canonical_bytes,
    load_json, process_identity,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


BASE = (REPO_ROOT / "_run/current/exact78_equipment_prompt_mask13_four_dev_v1_20261001"
        / "attempts/attempt_0001")
ANCHOR_PATH = BASE / "candidate/MASK_ANCHOR.json"
PACKET_PATH = (REPO_ROOT / "tasks/current"
               / "exact78_equipment_prompt_mask13_four_dev_v1_20261001/TASK_PACKET.json")
ANCHOR_SHA = "fe1cfe09e1199c11350e37bff5c2f226c1f0c69ec152aa5214f3579482006a96"
PACKET_SHA = "bfd27f8b67b199aeb1f9de20738253e929dcd7c2c6ada181f36aa7aa9329c8f7"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
OLD_OBJECT = "a potato chip"
NEW_OBJECT = "a yellow potato chip"
ANCHORS = {
    "get_potato_chips_0902_103": (0, 142, 283),
    "get_potato_chips_0901_013": (0, 435, 869),
}
FROZEN_MASKS = ("human", "equipment_original", "equipment_added", "equipment")
THRESHOLD = 0.5
TASK_ID = "exact78_chips_object_prompt_mask14_six_dev_v1_20261001"
OPERATION = "run_exact78_chips_object_prompt_mask14_candidate_v1"
ROUTE_KEY = "exact78_chips_object_prompt_mask14_candidate_v1"
WALL_SECONDS_MAX = 1800
RUNTIME_RELATIVE = (
    "src/chaoyang/governance/exact78_chips_object_prompt_mask14_candidate_v1.py",
    "src/chaoyang/governance/execute_exact78_chips_object_prompt_mask14_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_chips_object_prompt_mask14_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/cli/__init__.py",
)


def _load_exact(path: Path, sha: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise RuntimeError(f"MASK14_PATH_DRIFT:{path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError(f"MASK14_SHA_DRIFT:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_six() -> list[dict[str, Any]]:
    """Return exact Mask13 rows after validating all reused image/mask refs."""
    record = _load_exact(ANCHOR_PATH, ANCHOR_SHA)
    packet = _load_exact(PACKET_PATH, PACKET_SHA)
    config = packet["config"]
    if (record.get("model_calls") != 54 or len(record.get("rows", [])) != 12
            or config["other_role_prompts"]["chips_object"] != OLD_OBJECT
            or config["output_prob_thresh"] != THRESHOLD
            or config["instance_score_gate"] != THRESHOLD
            or packet["inputs"]["sam_checkpoint"]["sha256"] != CHECKPOINT_SHA):
        raise RuntimeError("MASK14_BASELINE_CONFIG_DRIFT")
    selected = [row for row in record["rows"] if row["session_id"] in ANCHORS]
    actual = [(row["session_id"], row["frame"]) for row in selected]
    expected = [(session, frame) for session, frames in ANCHORS.items() for frame in frames]
    if actual != expected:
        raise RuntimeError("MASK14_SIX_ANCHOR_DRIFT")
    for row in selected:
        if (row["prompts"]["object"] != OLD_OBJECT
                or row["prompts"]["human"] != config["new_human_prompt"]
                or row["prompts"]["equipment_original"] !=
                config["other_role_prompts"]["tracker_equipment"]
                or row["prompts"]["equipment_added"] != config["equipment_prompt_added"]):
            raise RuntimeError("MASK14_ROLE_PROMPT_DRIFT")
        for ref in (row["raw_image"], row["sam_input"], row["masks"]["object"],
                    *(row["masks"][role] for role in FROZEN_MASKS)):
            path = Path(ref["path"])
            if not path.is_relative_to(REPO_ROOT) or path.is_symlink():
                raise RuntimeError("MASK14_REF_OUTSIDE_REPO_OR_SYMLINK")
            if path.stat().st_size != ref["bytes"]:
                raise RuntimeError(f"MASK14_REF_SIZE_DRIFT:{path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                raise RuntimeError(f"MASK14_REF_SHA_DRIFT:{path}")
    return selected


def object_only_prompts(baseline: dict[str, str]) -> dict[str, str]:
    """Preserve every Mask13 role text except the chip object query."""
    if baseline.get("object") != OLD_OBJECT or set(baseline) != {
        "human", "object", "equipment_original", "equipment_added"
    }:
        raise RuntimeError("MASK14_PROMPT_SET_DRIFT")
    return {**baseline, "object": NEW_OBJECT}


def derive_masks(human: Any, equipment_original: Any, equipment_added: Any,
                 object_new: Any) -> dict[str, Any]:
    """Recompute Mask13 algebra with only the independently inferred new O."""
    import numpy as np

    arrays = [np.asarray(x) for x in
              (human, equipment_original, equipment_added, object_new)]
    if any(x.dtype != np.bool_ or x.shape != arrays[0].shape for x in arrays):
        raise RuntimeError("MASK14_MASK_DOMAIN_DRIFT")
    h, original, added, obj = arrays
    equipment = original | added
    combined = h | equipment
    return {"equipment": equipment, "human_equipment": combined,
            "object": obj, "unknown": combined & obj,
            "protected": obj & ~combined, "removal": combined & ~obj}


def route_contract(runtime_code: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "execution_status": "CANDIDATE_FENCED_GPU_MASK14_SIX_DEV",
        "candidate_operation": OPERATION,
        "task_packet_required": True,
        "code_closure": runtime_code,
        "output_schema": "exact78-chips-object-prompt-mask14-result-v1",
        "model_calls_authorized": "SIX_OBJECT_ONLY_ANCHOR_CALLS_WHEN_EXACT_PACKET_ACTIVE",
        "render_calls_authorized": 0,
        "training_steps_authorized": 0,
        "current_runnable_operation": None,
        "claim_limit": "Six chip development Mask anchors only; no Clean, Robot, geometry, product or training qualification.",
    }


def packet_value() -> dict[str, Any]:
    rows = frozen_six()
    mask13 = _load_exact(PACKET_PATH, PACKET_SHA)
    checkpoint = mask13["inputs"]["sam_checkpoint"]
    if artifact_ref(Path(checkpoint["path"])) != checkpoint:
        raise RuntimeError("MASK14_CHECKPOINT_SHA_DRIFT")
    inputs = {
        "sealed_mask13_anchor": artifact_ref(ANCHOR_PATH),
        "sealed_mask13_packet": artifact_ref(PACKET_PATH),
        "sam_checkpoint": checkpoint,
        "sam_vendor_manifest": mask13["inputs"]["sam_vendor_manifest"],
        "anchors": [{"session_id": row["session_id"], "frame": row["frame"],
                     "raw_image": row["raw_image"], "sam_input": row["sam_input"],
                     "fixed_masks": {role: row["masks"][role] for role in FROZEN_MASKS},
                     "old_object": row["masks"]["object"]} for row in rows],
    }
    code = [artifact_ref(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE]
    attempt = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
    cache = attempt / "cache"
    command = ("PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "
               f"TMPDIR={shlex.quote(str(cache))} TMP={shlex.quote(str(cache))} "
               f"TEMP={shlex.quote(str(cache))} XDG_CACHE_HOME={shlex.quote(str(cache))} "
               f"HF_HOME={shlex.quote(str(cache / 'huggingface'))} "
               f"TORCH_HOME={shlex.quote(str(cache / 'torch'))} "
               f"MPLCONFIGDIR={shlex.quote(str(cache / 'mpl'))} "
               f"CUDA_CACHE_PATH={shlex.quote(str(cache / 'cuda'))} "
               f"/usr/local/bin/python -B -m chaoyang.cli run {OPERATION} "
               f"--output-root {shlex.quote(str(attempt / 'candidate'))}")
    config = {"old_object_prompt": OLD_OBJECT, "new_object_prompt": NEW_OBJECT,
              "mask13_config": mask13["config"],
              "output_prob_thresh": THRESHOLD, "instance_score_gate": THRESHOLD,
              "anchors_total": 6, "model_calls_max": 6,
              "cpu_threads_total": 2, "gpu_count": 1, "training_steps": 0,
              "session_attempts_max": 1, "clean_calls_authorized": 0,
              "render_calls_authorized": 0}
    if (config["mask13_config"]["output_prob_thresh"] != THRESHOLD
            or config["mask13_config"]["instance_score_gate"] != THRESHOLD):
        raise RuntimeError("MASK14_NUMERIC_THRESHOLD_DRIFT")
    signature = hashlib.sha256(canonical_bytes({
        "schema": "exact78-chips-object-prompt-mask14-stage-v1", "inputs": inputs,
        "runtime_code": code, "config": config, "command": command,
        "weights": [checkpoint], "calibration": "RAW_STAGE_ALREADY_SEALED",
    })).hexdigest()
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "One six-anchor chip object-only Mask14 prompt A/B against sealed Mask13",
        "phase": "EXACT78_CHIPS_OBJECT_PROMPT_MASK14_DEV_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": TASK_ID.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "read_set": [str(Path(inputs[name]["path"]).relative_to(REPO_ROOT))
                     for name in ("sealed_mask13_anchor", "sealed_mask13_packet",
                                  "sam_checkpoint", "sam_vendor_manifest")],
        "write_set": [str(attempt.relative_to(REPO_ROOT)),
                      f"tasks/current/{TASK_ID}/TASK_PACKET.json",
                      f"tasks/receipts/{TASK_ID.upper()}.json",
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_MASK13_STRUCTURE",
                          "ONE_GPU", "NO_HOLDOUT_TUNING", "NO_0916", "NO_DELETION"],
        "required_outputs": ["candidate/MASK_ANCHOR.json", "candidate/RESULT.json",
                             "GPU_RECEIPT.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "model_calls_max": 6, "training_steps": 0,
                    "wall_seconds_max": WALL_SECONDS_MAX, "session_attempts_max": 1},
        "stop_conditions": ["INPUT_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "UNSUPPORTED_SOURCE_OR_FRAME",
                            "MASK_PROTECTION_GATE_FAILED", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": route_contract(code)["claim_limit"],
        "writer": writer, "inputs": inputs, "runtime_code": code,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("MASK14_PACKET_INVALID:" + ";".join(errors))
    return packet


def route_status(packet: dict[str, Any]) -> str:
    route = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts", {}).get(ROUTE_KEY)
    if route is None:
        return "DRY_RUN_NOT_ROUTABLE"
    if route != route_contract(packet["runtime_code"]):
        raise RuntimeError("MASK14_PUBLISHED_ROUTE_DRIFT")
    return "PUBLISHED_ROUTE_PACKET_STILL_REQUIRED"
