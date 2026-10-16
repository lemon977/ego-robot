"""Prepare two sequential, single-variable Cards042 CPU diagnostic packets.

This file is a candidate executor. --check-only is read-only. Neither task may
register until the independent algorithm-contract route is staged/published;
this code-prep task does not call --execute.
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

import numpy as np

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as flange
from chaoyang.pipeline import exact78_cards042_cpu_motion_diagnostic_v1 as cpu


SCRIPT = Path(__file__)
OPERATION = "run_exact78_cards042_cpu_motion_diagnostic_v1"
ROUTE_KEY = "exact78_cards042_cpu_motion_diagnostic_candidate_v1_20261001"
PYTHON = Path("/usr/local/bin/python")
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
MODES = {
    "FIRST_ANCHOR_ONLY": "exact78_cards042_first_anchor_cpu_v1_20261001",
    "BASE_RERANK_ONLY": "exact78_cards042_continuity_base_cpu_v1_20261001",
}
CODE_PATHS = (
    "src/chaoyang/governance/execute_exact78_cards042_cpu_motion_diagnostic_v1.py",
    "src/chaoyang/ops/run_exact78_cards042_cpu_motion_diagnostic_v1.py",
    "src/chaoyang/pipeline/exact78_cards042_cpu_motion_diagnostic_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_continuity_base_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_first_anchor_continuity_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)


def paths(mode: str) -> dict[str, Path]:
    if mode not in MODES:
        raise RuntimeError("CARDS042_CPU_MODE_UNSUPPORTED")
    task = MODES[mode]
    out = REPO_ROOT / "_run/current" / task / "attempts/attempt_0001"
    return {"task": task, "out": out, "spec": out / "CPU_DIAGNOSTIC_SPEC.json",
            "packet": REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json",
            "receipt": REPO_ROOT / "tasks/receipts" / f"{task.upper()}.json"}


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def frozen_spec(mode: str) -> dict[str, Any]:
    p = paths(mode)
    sealed = flange._SEALED_DEVELOPMENT[cpu.SESSION]
    hawor = REPO_ROOT / sealed["root"]
    audit = load_json(hawor / "INPUT_AUDIT.json")
    raw = Path(audit["raw_source_ref"]["path"])
    flange.validate_sealed_development(
        hawor_npz=hawor / "HAWOR_CAMERA_PREDICTIONS.npz",
        frame_provenance=hawor / "FRAME_PROVENANCE.json",
        input_audit=hawor / "INPUT_AUDIT.json", raw_video=raw,
        source_kind="tracker", session_id=cpu.SESSION)
    refs = {
        "hawor_result": artifact_ref(hawor / "RESULT.json"),
        "hawor_npz": artifact_ref(hawor / "HAWOR_CAMERA_PREDICTIONS.npz"),
        "frame_provenance": artifact_ref(hawor / "FRAME_PROVENANCE.json"),
        "input_audit": artifact_ref(hawor / "INPUT_AUDIT.json"),
        "hawor_seal": artifact_ref(hawor.parent / "HAWOR_STAGE_SEAL.json"),
        "native_open_audit": artifact_ref(hawor.parent / "NATIVE_OPEN_AUDIT.json"),
        "raw_video": artifact_ref(raw),
        "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
        "mount_proxy": artifact_ref(flange.stage_v1.MOUNT),
    }
    if refs["raw_video"] != audit["raw_source_ref"]:
        raise RuntimeError("CARDS042_CPU_RAW_SOURCE_REF_DRIFT")
    return {
        "schema_version": "exact78-cards042-cpu-diagnostic-spec-v1",
        "task_id": p["task"], "mode": mode,
        "session_id": cpu.SESSION, "source_kind": "tracker",
        "source_role": "DEVELOPMENT_ONLY", "frame_count": 171,
        "fps_nominal": 30.0, "inputs": refs,
        "output_root": str(p["out"] / "candidate"),
        "single_change": cpu.MODES[mode],
        "model_calls": 0, "render_calls": 0,
        "numeric_gates_changed": False,
        "base_selection_changed": mode == "BASE_RERANK_ONLY",
        "first_anchor_rule_changed": mode == "FIRST_ANCHOR_ONLY",
        "old_robot_result_input": False,
    }


def runtime_refs() -> list[dict[str, Any]]:
    cpu._check_pins()
    return [artifact_ref(REPO_ROOT / relative) for relative in CODE_PATHS]


def _first_anchor_parent_ref() -> dict[str, Any] | None:
    first = paths("FIRST_ANCHOR_ONLY")
    receipt, result = first["receipt"], first["out"] / "RESULT.json"
    if not receipt.is_file() or not result.is_file():
        return None
    value = load_json(receipt)
    if value.get("status") != "PASSED" or value.get("result") != artifact_ref(result):
        return None
    terminal = load_json(result)
    if (terminal.get("status") != "PASSED" or terminal.get("execution_complete") is not True
            or terminal.get("mode") != "FIRST_ANCHOR_ONLY"):
        return None
    return artifact_ref(result)


def packet_value(mode: str, spec: dict[str, Any]) -> dict[str, Any]:
    p = paths(mode)
    now = datetime.now().astimezone()
    deadline = min(now + timedelta(hours=2), CAMPAIGN_END)
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": p["task"]}
    inputs = {"cpu_spec": _virtual_ref(p["spec"], spec),
              "source_refs": spec["inputs"]}
    if mode == "BASE_RERANK_ONLY":
        parent = _first_anchor_parent_ref()
        if parent is None:
            raise RuntimeError("WAIT_FIRST_ANCHOR_SEALED_RESULT")
        inputs["comparison_parent_evidence_only"] = parent
    runtime = runtime_refs()
    command = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {p['spec']}"
    config = {
        "mode": mode, "frame_count": 171, "fps_nominal": 30.0,
        "source_role": "DEVELOPMENT_ONLY", "model_calls": 0, "render_calls": 0,
        "single_change": cpu.MODES[mode], "numeric_gates_changed": False,
        "robot_model_weights": "NOT_CONSUMED", "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC",
        "old_robot_result_input": False,
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-cards042-cpu-motion-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": p["task"],
        "objective": "Full 171-frame, both-side sealed HaWoR CPU motion diagnostic; one variable only",
        "phase": "EXACT78_CARDS042_CPU_MOTION_DIAGNOSTIC",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": f"EXACT78_CARDS042_CPU_{mode}_V1",
        "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(), "deadline_at": deadline.isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(p["spec"].relative_to(REPO_ROOT)),
                     *(str(Path(spec["inputs"][role]["path"]).relative_to(REPO_ROOT))
                       for role in ("hawor_result", "hawor_npz", "frame_provenance",
                                    "input_audit", "hawor_seal", "native_open_audit"))],
        "write_set": [str(p["out"].relative_to(REPO_ROOT)),
                      str(p["packet"].relative_to(REPO_ROOT)),
                      str(p["receipt"].relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK",
                          "SEALED_CARDS042_V3_HAWOR", "SOURCE_SHA_PINNED",
                          "ROBOT_ASSETS_PINNED", "CPU_ONLY", "EXTERNAL_DATA_READ_ONLY",
                          "0916_READ_ONLY", "NO_DELETION"]
                         + (["FIRST_ANCHOR_CPU_SEALED"] if mode == "BASE_RERANK_ONLY" else []),
        "required_outputs": ["CPU_DIAGNOSTIC_SPEC.json", "candidate/RESULT.json",
                             "candidate/FRAME_NUMERIC_DIAGNOSTIC.json",
                             "candidate/BASE_SELECTION.json",
                             "candidate/ROBOT_NUMERIC_STATES.npz",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0,
                    "training_steps": 0, "wall_seconds_max": 7200,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "CAMPAIGN_OR_TASK_DEADLINE", "INCOMPLETE_342_FRAME_SIDE_GRID",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards042 development-only CPU camera diagnostic; no rendered video, world, visual, product, control or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("CARDS042_CPU_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


def _route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(ROUTE_KEY, {})
    expected = [artifact_ref(REPO_ROOT / relative) for relative in CODE_PATHS
                if relative.startswith("src/chaoyang/ops/") or
                relative.startswith("src/chaoyang/pipeline/")]
    return (route.get("candidate_operation") == OPERATION
            and route.get("execution_status") == "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION"
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-cards042-cpu-motion-result-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("render_calls_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in expected))


def route_readiness() -> tuple[bool, str]:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.common import ALGORITHM_CONTRACT_PATH, BASELINE_REGISTRY_PATH

    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CPU_CANDIDATE_ROUTE") if _route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "CPU_PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    if not _route_valid(staged):
        return False, "CPU_CANDIDATE_ROUTE_NOT_STAGED"
    return True, "READY_TO_PUBLISH_CPU_ROUTE_WITH_PACKET"


def readiness(mode: str) -> tuple[bool, str]:
    p = paths(mode)
    if datetime.now().astimezone() >= CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(p[key].exists() for key in ("out", "packet", "receipt")):
        return False, "FRESH_CPU_TASK_REQUIRED"
    if mode == "BASE_RERANK_ONLY" and _first_anchor_parent_ref() is None:
        return False, "WAIT_FIRST_ANCHOR_SEALED_RESULT"
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row["status"] in
                                  ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")
                                  for row in state["tasks"])):
        return False, "ACTIVE_TASK_CONFLICT"
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    return route_readiness()


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _registration_values(mode: str, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    p = paths(mode)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    stamp = now_iso()
    state["tasks"].append({
        "task_id": p["task"], "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": cpu.SESSION,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": _virtual_ref(p["packet"], packet),
        "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {
        "task_id": p["task"], "session": cpu.SESSION,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "TWO_CPU_THREADS_NO_GPU",
        "stop_condition": "one full CPU diagnostic or two-hour/campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("CARDS042_CPU_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": p["task"], "packet_path": str(p["packet"].relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(p["packet"], packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=f"EXACT78_CARDS042_CPU_{mode}_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    return state, index


def register(mode: str, packet: dict[str, Any], spec: dict[str, Any]) -> None:
    p = paths(mode)
    ready, reason = readiness(mode)
    if not ready:
        raise RuntimeError(reason)
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
        raise RuntimeError("CARDS042_CPU_TASK_DEADLINE_EXPIRED")
    for ref in [*packet["runtime_code"], *spec["inputs"].values()]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("CARDS042_CPU_PRE_REGISTER_SHA_DRIFT:" + ";".join(errors))
    state, index = _registration_values(mode, packet)  # preflight before writes
    p["out"].mkdir(parents=True)
    atomic_json(p["spec"], spec)
    if artifact_ref(p["spec"]) != packet["inputs"]["cpu_spec"]:
        raise RuntimeError("CARDS042_CPU_SPEC_VIRTUAL_SHA_DRIFT")
    atomic_json(p["out"] / "WRITER.json", packet["writer"])
    p["packet"].parent.mkdir(parents=True)
    atomic_json(p["packet"], packet)
    if artifact_ref(p["packet"])["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("CARDS042_CPU_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_CARDS042_CPU_{mode}_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX,
        task_packet_index_value=index)


def verify_fence(mode: str, packet: dict[str, Any]) -> None:
    p = paths(mode)
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(p["out"] / "WRITER.json") != writer):
        raise RuntimeError("CARDS042_CPU_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != p["task"]
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("CARDS042_CPU_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"]
              if row.get("execution_allowed")]
    if (len(routes) != 1 or routes[0]["task_id"] != p["task"]
            or routes[0]["packet_sha256"] != artifact_ref(p["packet"])["sha256"]):
        raise RuntimeError("CARDS042_CPU_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("CARDS042_CPU_RUNTIME_CODE_DRIFT:" + ";".join(errors))


def heartbeat(mode: str, packet: dict[str, Any], elapsed: float) -> None:
    p = paths(mode)
    verify_fence(mode, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == p["task"])
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_CARDS042_CPU_{mode}_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (p["out"] / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(p["out"] / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-cards042-cpu-resume-v1",
            "mode": mode, "elapsed_seconds": round(elapsed, 1),
            "candidate_result_exists": (p["out"] / "candidate/RESULT.json").is_file(),
            "next_action": "Wait for fenced CPU worker or inspect terminal receipt"})


def run_worker(mode: str, packet: dict[str, Any]) -> tuple[int, Path]:
    p = paths(mode)
    verify_fence(mode, packet)
    cache = p["out"] / "cache"
    cache.mkdir()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("CARDS042_CPU_BUDGET_TOO_SHORT")
    log = p["out"] / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
            OPERATION, "--spec", str(p["spec"])], cwd=REPO_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT)
        last_beat = 0.0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("CARDS042_CPU_HARD_TASK_DEADLINE")
                if elapsed - last_beat >= 50:
                    heartbeat(mode, packet, elapsed)
                    last_beat = elapsed
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    return process.returncode, log


def verify_candidate(mode: str, spec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    p = paths(mode)
    path = p["out"] / "candidate/RESULT.json"
    if not path.is_file():
        return None, ["CARDS042_CPU_CANDIDATE_RESULT_MISSING"]
    try:
        result = load_json(path)
        if (result.get("status") != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
                or result.get("session_id") != cpu.SESSION
                or result.get("mode") != mode or result.get("frame_count") != 171
                or result.get("single_change") != cpu.MODES[mode]
                or result.get("model_calls") != 0 or result.get("render_calls") != 0
                or result.get("visual_qualified") is not False
                or result.get("product_qualified") is not False
                or result.get("training_eligible") is not False
                or result.get("control_ground_truth") is not False):
            raise RuntimeError("CARDS042_CPU_CANDIDATE_IDENTITY_OR_SCOPE_DRIFT")
        for role, key in (("hawor_result", "sealed_hawor_result"),
                          ("hawor_seal", "hawor_stage_seal"),
                          ("native_open_audit", "hawor_native_open_audit")):
            if result.get(key) != spec["inputs"][role]:
                raise RuntimeError(f"CARDS042_CPU_SOURCE_REF_DRIFT:{role}")
        for key, filename in (("frame_numeric_diagnostic", "FRAME_NUMERIC_DIAGNOSTIC.json"),
                              ("base_selection", "BASE_SELECTION.json"),
                              ("numeric_states", "ROBOT_NUMERIC_STATES.npz")):
            ref = result.get(key)
            if (not isinstance(ref, dict)
                    or Path(ref.get("path", "")).resolve() != p["out"] / "candidate" / filename
                    or validate_artifact_ref(ref)):
                raise RuntimeError(f"CARDS042_CPU_OUTPUT_SHA_DRIFT:{key}")
        frame_doc = load_json(p["out"] / "candidate/FRAME_NUMERIC_DIAGNOSTIC.json")
        rows = frame_doc.get("rows", [])
        identities = {(row.get("frame"), row.get("side")) for row in rows}
        if (frame_doc.get("mode") != mode or len(rows) != 342
                or identities != {(frame, side) for frame in range(171) for side in ("left", "right")}
                or any("branch_margin_m_min" not in row or
                       "candidate_max_abs_dq_rad" not in row or
                       "arm_pass_numeric" not in row or
                       "hand_pass_numeric" not in row for row in rows)):
            raise RuntimeError("CARDS042_CPU_INCOMPLETE_FRAME_GRID")
        with np.load(p["out"] / "candidate/ROBOT_NUMERIC_STATES.npz", allow_pickle=False) as archive:
            if (archive["q_arm"].shape != (171, 2, 7)
                    or archive["q_hand"].shape[0:2] != (171, 2)
                    or archive["T_camera_base"].shape != (4, 4)
                    or not np.isfinite(archive["q_arm"]).all()):
                raise RuntimeError("CARDS042_CPU_NUMERIC_STATE_SHAPE_DRIFT")
        summary = result.get("summary", {})
        if (summary.get("frame_side_count") != 342
                or summary.get("worst_side_arm_pass") != min(
                    sum(row["arm_pass_numeric"] for row in rows if row["side"] == side)
                    for side in ("left", "right"))):
            raise RuntimeError("CARDS042_CPU_SUMMARY_DRIFT")
        base = load_json(p["out"] / "candidate/BASE_SELECTION.json")
        if (mode == "FIRST_ANCHOR_ONLY" and
                base.get("rule") != "same_frozen_grid_for_all_sources_v1"):
            raise RuntimeError("CARDS042_CPU_BASE_CHANGED_IN_FIRST_ANCHOR_TASK")
        if (mode == "BASE_RERANK_ONLY" and
                (base.get("approximation") != "TOP4_ONLY_NO_GLOBAL_GRID_OPTIMALITY_CLAIM"
                 or base.get("rule") != "same_frozen_grid_original_score_top4_full_chronology_v1")):
            raise RuntimeError("CARDS042_CPU_BASE_RERANK_RULE_DRIFT")
        for role, ref in spec["inputs"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"CARDS042_CPU_INPUT_SHA_DRIFT:{role}:{';'.join(errors)}")
        return result, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"CARDS042_CPU_CANDIDATE_VALIDATION:{error}"]


def finish(mode: str, packet: dict[str, Any], spec: dict[str, Any], code: int,
           log: Path, reason: str | None = None) -> dict[str, Any]:
    p = paths(mode)
    verify_fence(mode, packet)
    candidate, errors = verify_candidate(mode, spec)
    complete = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (p["out"] / "RESULT.json").exists() or p["receipt"].exists():
        raise RuntimeError("CARDS042_CPU_DUPLICATE_TERMINAL_RESULT")
    result = {
        "schema_version": "exact78-cards042-cpu-task-result-v1",
        "task_id": p["task"], "mode": mode, "status": status,
        "execution_complete": complete, "worker_returncode": code,
        "reason": reason, "validation_errors": errors,
        "source_session": cpu.SESSION, "source_frames_expected": 171,
        "cpu_spec": artifact_ref(p["spec"]),
        "candidate_result": artifact_ref(p["out"] / "candidate/RESULT.json") if candidate else None,
        "worker_log": artifact_ref(log), "single_change": spec["single_change"],
        "model_calls": 0, "render_calls": 0,
        "old_robot_result_input": False, "old_video_ab_status": "EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "ai_review": "PENDING", "user_review": "PENDING",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "source_mutated": False, "files_deleted": False,
        "next_action": "If first-anchor task sealed, register base-rerank as a separate task; otherwise inspect failed frame-side evidence",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(p["out"] / "RESULT.json", result)
    p["receipt"].parent.mkdir(parents=True, exist_ok=True)
    atomic_json(p["receipt"], {
        "schema_version": "exact78-cards042-cpu-task-receipt-v1",
        "task_id": p["task"], "status": status,
        "result": artifact_ref(p["out"] / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == p["task"])
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(p["out"] / "RESULT.json"),
               terminal_receipt=artifact_ref(p["receipt"]),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("CARDS042_CPU_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == p["task"]:
            item["execution_allowed"] = False
    index.update(packet_revision=f"EXACT78_CARDS042_CPU_{mode}_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_CARDS042_CPU_{mode}_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX,
        task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(p["out"] / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    spec = frozen_spec(mode)
    ready, reason = readiness(mode)
    if args.check_only:
        packet = packet_value(mode, spec) if mode == "FIRST_ANCHOR_ONLY" or _first_anchor_parent_ref() else None
        print(json.dumps({
            "status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": paths(mode)["task"],
            "mode": mode, "frame_count": 171, "frame_side_count": 342,
            "model_calls": 0, "render_calls": 0, "gpu_count": 0,
            "packet_preflight": "PASS" if packet is not None else "WAIT_FIRST_ANCHOR_SEALED_RESULT",
            "run_signature": packet["run_signature"] if packet else None,
            "candidate_route": route_readiness()[1],
        }, ensure_ascii=False))
        return 0
    if not ready:
        raise RuntimeError(reason)
    packet = packet_value(mode, spec)
    p = paths(mode)
    register(mode, packet, spec)
    try:
        code, log = run_worker(mode, packet)
        failure = None
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = p["out"] / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    terminal = finish(mode, packet, spec, code, log, failure)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
