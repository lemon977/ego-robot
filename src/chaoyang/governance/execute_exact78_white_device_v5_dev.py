"""Unregistered finite white-device frontend candidate for four development clips.

``--check-only`` is read-only.  Execution stays fenced off until a separate
candidate CLI route is staged/published; route and one exact Task Packet must
be published by the same governance CAS.  No holdout or historical derived
media is a producer input.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
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


SCRIPT = Path(__file__)
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PREP = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/29_THREE_SOURCE_WHITE_DEVICE_PROMPT_DEV_PREP_ZH.md"
AUDIT = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/27_THREE_SOURCE_CLEAN_FAILURE_CLUSTER_AND_NEXT_SINGLE_CHANGE_ZH.md"
V5_PATH = REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v5_white_device.py"
V5_SHA256 = "3627cffe8501fb7bad5b42b90fe322cff3f5019f7d9f69b35300c0e071b6a12a"
OPERATION = "run_exact78_fresh_frontend_v5_white_device"
CONTRACT_KEY = "exact78_three_source_white_device_prompt_candidate_v5_20261001"
PROMPT = "a white device attached to a hand or wrist"
PYTHON = Path("/usr/local/bin/python")
STAGES = ("MAP", "RAW", "MASK", "PREPARE", "CLEAN")
SESSION_CONFIGS = {
    "0902_cards042": {
        "base": "execute_exact78_raw_t2_cards042_frontend_v3_mp4_retry1",
        "task": "exact78_raw_to_robot_t2_white_device_v5_0902_cards042_dev_20261001",
        "frames": 171, "fps": 30, "family": "TRACKER_0902", "session": "play_cards_0902_042",
        "baseline_task": "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930",
        "wall_seconds": 900,
    },
    "0902_chips103": {
        "base": "execute_exact78_raw_t2_0902_chips103_frontend_v3_mp4_retry1",
        "task": "exact78_raw_to_robot_t2_white_device_v5_0902_chips103_dev_20261001",
        "frames": 284, "fps": 30, "family": "TRACKER_0902", "session": "get_potato_chips_0902_103",
        "baseline_task": "exact78_raw_to_robot_t2_0902_chips103_frontend_v3_mp4_retry1_20261001",
        "wall_seconds": 1200,
    },
    "0901_chips013": {
        "base": "execute_exact78_raw_t2_chips013_frontend_v1",
        "task": "exact78_raw_to_robot_t2_white_device_v5_0901_chips013_dev_20261001",
        "frames": 870, "fps": 25, "family": "TRACKER_0901", "session": "get_potato_chips_0901_013",
        "baseline_task": "exact78_raw_to_robot_t2_chips013_frontend_v1_20260930",
        "wall_seconds": 2400,
    },
    "0915_cards001": {
        "base": "execute_exact78_raw_t2_0915_cards001_frontend_v1",
        "task": "exact78_raw_to_robot_t2_white_device_v5_0915_cards001_dev_20261001",
        "frames": 150, "fps": 30, "family": "VST_0915", "session": "play_cards_0915_001",
        "baseline_task": "exact78_raw_to_robot_t2_0915_cards001_frontend_v1_20260930",
        "wall_seconds": 900,
    },
}


def context(key: str) -> dict[str, Any]:
    config = SESSION_CONFIGS[key]
    base = importlib.import_module(f"chaoyang.governance.{config['base']}")
    task = config["task"]
    out = REPO_ROOT / "_run/current" / task / "attempts/attempt_0001"
    return {"key": key, "config": config, "base": base, "task": task, "out": out,
            "spec": out / "FRONTEND_SPEC.json",
            "packet": REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json",
            "receipt": REPO_ROOT / "tasks/receipts" / f"{task.upper()}.json",
            "command": f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {out / 'FRONTEND_SPEC.json'} --phase all"}


def frozen_inputs(ctx: dict[str, Any]) -> dict[str, Any]:
    inputs = ctx["base"].frozen_inputs()
    # Evidence documents are admission references only, never Mask/Clean input.
    inputs.update(white_device_prep=artifact_ref(PREP), clean_failure_audit=artifact_ref(AUDIT))
    return inputs


def spec_value(ctx: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec
    spec = ctx["base"].spec_value(inputs)
    spec.update(task_id=ctx["task"], output_root=str(ctx["out"] / "candidate"))
    if (spec["session_id"] != ctx["config"]["session"]
            or spec["source_family"] != ctx["config"]["family"]
            or spec["frame_count"] != ctx["config"]["frames"]
            or spec["source_fps"] != ctx["config"]["fps"]):
        raise RuntimeError("DEVELOPMENT_SPEC_IDENTITY_DRIFT")
    if ctx["key"] == "0915_cards001" and (spec["source_index"] != 1
            or spec["frame_timing"]["sha256"] != inputs["t0_frame_time_map"]["sha256"]):
        raise RuntimeError("VST_PHYSICAL_LEFT_OR_TIMEBASE_DRIFT")
    validate_spec(spec)
    return spec


def virtual_json_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    blob = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def runtime_refs(ctx: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    if artifact_ref(V5_PATH)["sha256"] != V5_SHA256:
        raise RuntimeError("V5_ALGORITHM_SHA_DRIFT")
    base = ctx["base"]
    baseline_spec = base.spec_value(inputs)
    baseline_packet = base.packet_value(inputs, base.virtual_json_ref(base.SPEC, baseline_spec))
    refs = [artifact_ref(SCRIPT), artifact_ref(V5_PATH), *baseline_packet["runtime_code"]]
    for relative in ("src/chaoyang/ops/run_exact78_fresh_frontend_v3.py",
                     "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
                     "src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py"):
        refs.append(artifact_ref(REPO_ROOT / relative))
    unique: dict[str, dict[str, Any]] = {}
    for ref in refs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise RuntimeError("RUNTIME_CLOSURE_SHA_CONFLICT")
        unique[ref["path"]] = ref
    return list(unique.values())


def packet_value(ctx: dict[str, Any], inputs: dict[str, Any], spec_ref: dict[str, Any]
                 ) -> dict[str, Any]:
    config = ctx["config"]
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": ctx["task"]}
    runtime = runtime_refs(ctx, inputs)
    all_inputs = {**inputs, "frontend_spec": spec_ref}
    cfg = {"session_id": config["session"], "frame_count": config["frames"],
           "fps_nominal": config["fps"], "stages": list(STAGES), "gpu_lease_count": 1,
           "training_steps": 0, "algorithm_delta": "ONE_SHARED_WHITE_DEVICE_TEXT_PROMPT_ONLY",
           "equipment_prompt": PROMPT, "probability_threshold": 0.5,
           "mp4_acceptance": "EXACT_STREAM_TICKS_AND_MVHD_QUANTIZATION_V2",
           "preclean_protection": "ZERO_BASELINE_PROTECTED_OBJECT_CANDIDATE_PIXEL_LOSS",
           "visual_quality": "PENDING_REVIEW", "geometry": "NOT_IN_T2"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": all_inputs, "runtime_code": runtime, "config": cfg,
        "command": ctx["command"], "schema": "exact78-three-source-white-device-v5-dev-v1",
    })).hexdigest()
    # Governance caps this human-facing read_set at eight; complete machine
    # dependencies remain SHA-bound in inputs/runtime_code instead.
    base = ctx["base"]
    read_set = ["AGENTS.md", str(base.PLAN.relative_to(REPO_ROOT)),
                str((base.T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                str((base.T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                str(PREP.relative_to(REPO_ROOT)), str(AUDIT.relative_to(REPO_ROOT)),
                str(ctx["spec"].relative_to(REPO_ROOT)),
                str(SCRIPT.relative_to(REPO_ROOT))]
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": ctx["task"],
        "objective": f"One full raw MAP→CLEAN white-device prompt retry of {config['session']}",
        "phase": "EXACT78_RAW_TO_ROBOT_T2_WHITE_DEVICE_DEV",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_THREE_SOURCE_WHITE_DEVICE_V5_DEV",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(seconds=config["wall_seconds"] + 900),
                           ctx["base"].CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": ctx["base"].CAMPAIGN_END.isoformat(),
        "read_set": read_set,
        "write_set": [str(ctx["out"].relative_to(REPO_ROOT)),
                      str(ctx["packet"].relative_to(REPO_ROOT)),
                      str(ctx["receipt"].relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "T0_IDENTITY_FROZEN", "NO_ACTIVE_TASK",
                          "ONE_GPU_LEASE", "SOURCE_SHA_PINNED", "EXTERNAL_DATA_READ_ONLY",
                          "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["FRONTEND_SPEC.json", "candidate/CAMERA_MAPS.npz",
                             *(f"candidate/{stage}.json" for stage in STAGES),
                             *(f"candidate/SEAL_{stage}.json" for stage in STAGES),
                             "candidate/propainter_model/frames/masked_in.mp4",
                             "candidate/propainter_model/frames/inpaint_out.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": config["wall_seconds"] + 900,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "STAGE_SEAL_FAIL",
                            "VISIBLE_OBJECT_PROTECTION_REGRESSION", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One development Clean prompt candidate only; no visual, geometry, Robot, product or training qualification",
        "writer": writer, "inputs": all_inputs, "runtime_code": runtime,
        "config": cfg, "run_signature": signature, "commands": [ctx["command"]],
        "post_result_ab_reference": {"path": str(REPO_ROOT / "_run/current" /
                                                 config["baseline_task"] / "attempts/attempt_0001/RESULT.json"),
                                     "use": "EVALUATOR_ONLY_AFTER_FRESH_RESULT"},
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_V5_PACKET:" + ";".join(errors))
    return packet


def route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    required = [artifact_ref(path) for path in (
        V5_PATH, REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v3.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
        REPO_ROOT / "src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py")]
    return (route.get("candidate_operation") == OPERATION
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-fresh-frontend-stage-seal-v2"
            and route.get("model_calls_authorized") ==
                "ONE_DEV_SESSION_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE"
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in required))


def candidate_route_readiness() -> tuple[bool, str]:
    from chaoyang.cli import _maintained_operations
    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CANDIDATE_ROUTE") if route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "V5_PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    return ((True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET") if route_valid(staged)
            else (False, "V5_CANDIDATE_ROUTE_NOT_STAGED"))


def readiness(ctx: dict[str, Any]) -> tuple[bool, str]:
    if datetime.now().astimezone() >= ctx["base"].CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(path.exists() for path in (ctx["out"], ctx["packet"], ctx["receipt"])):
        return False, "FRESH_V5_TASK_REQUIRED"
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    previous = [row for row in state["tasks"]
                if row["task_id"] == ctx["config"]["baseline_task"]]
    if not previous or previous[0]["status"] != "PASSED":
        return False, "BASELINE_TASK_NOT_TERMINAL"
    return candidate_route_readiness()


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def register(ctx: dict[str, Any], packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness(ctx)
    if not ready:
        raise RuntimeError(reason)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    out, spec_path, packet_path = ctx["out"], ctx["spec"], ctx["packet"]
    out.mkdir(parents=True)
    atomic_json(spec_path, spec)
    if artifact_ref(spec_path) != packet["inputs"]["frontend_spec"]:
        raise RuntimeError("FRONTEND_SPEC_PRECOMPUTED_SHA_DRIFT")
    atomic_json(out / "WRITER.json", packet["writer"])
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    now = now_iso()
    state["tasks"].append({"task_id": ctx["task"], "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": ctx["config"]["session"], "updated_at": now,
        "heartbeat_at": now, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": 0,
        "task_packet": artifact_ref(packet_path), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": ctx["task"], "session": ctx["config"]["session"],
        "prerequisites": packet["prerequisites"], "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full development V5 frontend result or finite deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V5_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": ctx["task"],
        "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_WHITE_DEVICE_V5_DEV_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_WHITE_DEVICE_V5_DEV_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(ctx: dict[str, Any], packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(ctx["out"] / "WRITER.json") != writer):
        raise RuntimeError("V5_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != ctx["task"] or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("V5_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    route = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if len(route) != 1 or route[0]["task_id"] != ctx["task"] or route[0]["packet_sha256"] != artifact_ref(ctx["packet"])["sha256"]:
        raise RuntimeError("V5_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("V5_CODE_SHA_DRIFT:" + ";".join(errors))


def heartbeat(ctx: dict[str, Any], packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(ctx, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == ctx["task"])
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_WHITE_DEVICE_V5_DEV_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    summary = ctx["out"] / f"RESUME_SUMMARY_{number:04d}.json"
    if number and not summary.exists():
        atomic_json(summary, {"schema_version": "exact78-white-device-v5-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "sealed_stages": [name for name in STAGES
                              if (ctx["out"] / "candidate" / f"SEAL_{name}.json").is_file()],
            "next_action": "Wait for fenced worker; do not consume unsealed partial"})


def run_worker(ctx: dict[str, Any], packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(ctx, packet)
    cache = ctx["out"] / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    remaining = int((datetime.fromisoformat(packet["deadline_at"]) - datetime.now().astimezone()).total_seconds())
    if remaining < 300:
        raise RuntimeError("V5_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(600, max(0, remaining // 10))
    wall_seconds = min(ctx["config"]["wall_seconds"], remaining - wait_seconds - 60)
    command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
        "run_gpu_command_with_v71_lease", "--task-id", ctx["task"],
        "--attempt-id", f"attempt_0001_{ctx['key']}_white_device_v5", "--priority", "CANARY",
        "--gpu-id", "0", "--min-free-mib", "60000", "--wait-seconds", str(wait_seconds),
        "--wall-seconds", str(wall_seconds), "--receipt", str(ctx["out"] / "GPU_RECEIPT.json"),
        "--claim-limit", packet["claim_limit"], "--", str(PYTHON), "-B", "-m",
        "chaoyang.cli", "run", OPERATION, "--spec", str(ctx["spec"]), "--phase", "all"]
    log = ctx["out"] / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        last_beat = started
        while process.poll() is None:
            if time.monotonic() - last_beat >= 50:
                heartbeat(ctx, packet, time.monotonic() - started)
                last_beat = time.monotonic()
            time.sleep(2)
    return process.returncode, log


def verify_internal_videos(ctx: dict[str, Any]) -> None:
    from chaoyang.ops.run_exact78_fresh_frontend_v3 import verify_internal_mp4_v3
    output = ctx["out"] / "candidate"
    videos = load_json(output / "CLEAN.json")["internal_videos"]
    if set(videos) != {"masked_in.mp4", "inpaint_out.mp4"}:
        raise RuntimeError("INTERNAL_VIDEO_SET")
    for name in sorted(videos):
        path = output / "propainter_model/frames" / name
        expected = verify_internal_mp4_v3(path, ctx["config"]["frames"], ctx["config"]["fps"])
        if videos[name] != expected:
            raise RuntimeError(f"INTERNAL_VIDEO_REF_DRIFT:{name}")


def verify_masks_and_protection(ctx: dict[str, Any]) -> dict[str, int]:
    from chaoyang.ops.run_exact78_fresh_frontend_v5_white_device import compose_roles
    output = ctx["out"] / "candidate"
    record, clean = load_json(output / "MASK.json"), load_json(output / "CLEAN.json")
    frames = ctx["config"]["frames"]
    if (record.get("schema_version") != "exact78-fresh-frontend-mask-v5-white-device"
            or len(record["rows"]) != frames or len(clean["rows"]) != frames):
        raise RuntimeError("V5_MASK_SCHEMA_OR_FRAME_COUNT")
    detected = protected_total = white_new_total = 0
    for index, (row, clean_row) in enumerate(zip(record["rows"], clean["rows"])):
        if row["decode_display_index"] != index or clean_row["decode_display_index"] != index:
            raise RuntimeError(f"V5_FRAME_MAPPING:{index}")
        images = {}
        for name, ref in row["masks"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"V5_MASK_SHA:{index}:{name}:{';'.join(errors)}")
            value = cv2.imread(ref["path"], cv2.IMREAD_GRAYSCALE)
            if value is None or value.shape != (960, 1280):
                raise RuntimeError(f"V5_MASK_IMAGE_DOMAIN:{index}:{name}")
            images[name] = value > 0
        roles = {key: images[key] for key in ("human", "object", "equipment_original",
                 "equipment_white_device", *(("finger_sleeve", "yellow_cable")
                    if ctx["config"]["family"] == "VST_0915" else ())) }
        expected, counts = compose_roles(roles, ctx["config"]["family"])
        if set(images) != set(expected) or any(not np.array_equal(images[key], value)
                                               for key, value in expected.items()):
            raise RuntimeError(f"V5_MASK_ALGEBRA_DRIFT:{index}")
        if row["equipment_candidate_pixels"] != counts:
            raise RuntimeError(f"V5_MASK_COUNTS_DRIFT:{index}")
        raw = cv2.imread(row["model_input"]["path"])
        cleaned = cv2.imread(clean_row["clean"]["path"])
        if (raw is None or cleaned is None or raw.shape != (960, 1280, 3)
                or cleaned.shape != raw.shape
                or not np.array_equal(raw[images["protected"]], cleaned[images["protected"]])):
            raise RuntimeError(f"V5_PROTECTED_RAW_PIXEL_DRIFT:{index}")
        detected += bool(images["equipment"].any())
        protected_total += int(images["protected"].sum())
        white_new_total += counts["equipment_white_device_new_pixels"]
    return {"equipment_nonempty_frames": detected,
            "equipment_empty_frames": frames - detected,
            "white_device_new_pixels": white_new_total,
            "protected_visible_object_candidate_pixels_raw_preserved": protected_total}


def finish(ctx: dict[str, Any], packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(ctx, packet)
    stage_refs: dict[str, Any] = {}
    stage_errors: list[str] = []
    previous: dict[str, Any] | str = "ABSENT"
    for name in STAGES:
        path = ctx["out"] / "candidate" / f"SEAL_{name}.json"
        if path.is_file():
            seal = load_json(path)
            result = seal.get("result")
            if (seal.get("session_id") != ctx["config"]["session"] or seal.get("stage") != name
                    or seal.get("frontend_spec") != packet["inputs"]["frontend_spec"]
                    or seal.get("upstream_seal") != previous
                    or seal.get("frame_count") != (0 if name == "MAP" else ctx["config"]["frames"])
                    or not isinstance(result, dict) or validate_artifact_ref(result)):
                stage_errors.append(f"STAGE_SEAL_DRIFT:{name}")
            stage_refs[name] = artifact_ref(path)
            previous = stage_refs[name]
        else:
            stage_errors.append(f"STAGE_SEAL_MISSING:{name}")
    metrics = None
    if "CLEAN" in stage_refs:
        try:
            verify_internal_videos(ctx)
            metrics = verify_masks_and_protection(ctx)
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            stage_errors.append(f"CLEAN_OR_MASK_VALIDATION:{error}")
    input_errors = []
    for key, ref in packet["inputs"]["source_files"].items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{key}:{';'.join(errors)}")
    models = packet["inputs"]["models"]
    for ref in (models["sam_checkpoint"], *models["propainter_weights"].values()):
        if errors := validate_artifact_ref(ref):
            input_errors.append("model:" + ";".join(errors))
    gpu_path = ctx["out"] / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    complete = (code == 0 and gpu.get("status") == "PASSED" and not stage_errors
                and not input_errors and reason is None and metrics is not None)
    status = "PASSED" if complete else (
        "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL")
    if (ctx["out"] / "RESULT.json").exists() or ctx["receipt"].exists():
        raise RuntimeError("DUPLICATE_V5_TERMINAL_RESULT")
    result = {"schema_version": "exact78-white-device-v5-dev-result-v1",
        "task_id": ctx["task"], "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "stage_seal_errors": stage_errors,
        "post_input_sha_errors": input_errors, "source_session": ctx["config"]["session"],
        "source_frames_expected": ctx["config"]["frames"], "source_fps": ctx["config"]["fps"],
        "frontend_spec": artifact_ref(ctx["spec"]), "stage_seals": stage_refs,
        "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
        "worker_log": artifact_ref(log), "mask_metrics": metrics,
        "mask_quality": "NOT_YET_ADOPTED", "clean_quality": "NOT_YET_ADOPTED",
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "source_mutated": "UNKNOWN" if input_errors else False,
        "files_deleted": False, "old_result_ab_status": "NOT_READ_EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "next_action": "Review device-mask expansion, object protection and Clean on fixed development frames; do not tune with holdouts",
        "claim_limit": packet["claim_limit"]}
    atomic_json(ctx["out"] / "RESULT.json", result)
    atomic_json(ctx["receipt"], {"schema_version": "exact78-white-device-v5-dev-receipt-v1",
        "task_id": ctx["task"], "status": status,
        "result": artifact_ref(ctx["out"] / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == ctx["task"])
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(ctx["out"] / "RESULT.json"),
               terminal_receipt=artifact_ref(ctx["receipt"]),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_OR_RESOURCE_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("V5_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == ctx["task"]:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_WHITE_DEVICE_V5_DEV_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_WHITE_DEVICE_V5_DEV_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(ctx["out"] / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=tuple(SESSION_CONFIGS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ctx = context(args.session)
    inputs = frozen_inputs(ctx)
    spec = spec_value(ctx, inputs)
    spec_ref = virtual_json_ref(ctx["spec"], spec)
    packet = packet_value(ctx, inputs, spec_ref)
    ready, reason = readiness(ctx)
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": ctx["task"], "command": ctx["command"],
            "source_session": ctx["config"]["session"], "source_frames": ctx["config"]["frames"],
            "fps": ctx["config"]["fps"], "source_video": spec["source_video"],
            "spec_sha256": spec_ref["sha256"], "run_signature": packet["run_signature"],
            "write_set": packet["write_set"], "gpu_lease_count": 1,
            "historical_derived_input": False}, ensure_ascii=False))
        return 0
    if not ready:
        raise RuntimeError(reason)
    register(ctx, packet, spec)
    try:
        code, log = run_worker(ctx, packet)
        failure = None
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = ctx["out"] / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    terminal = finish(ctx, packet, code, log, failure)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
