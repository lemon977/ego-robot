#!/usr/bin/env python3
"""Four separately registered, bounded Exact78 preconvergence tasks."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, atomic_write, canonical_bytes, load_json,
    now_iso, process_identity, publish_bundle, validate_artifact_ref,
)
from chaoyang.governance.consolidate_exact78_baseline_route_v2 import STAGE_TASKS, audit_refs
from chaoyang.governance.register_single_task_packet import _validate_packet


PLAN = REPO_ROOT / "docs/plans/EXACT78_PRECONVERGENCE_V1_20260930/00_EXECUTION.md"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
POINTER = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LINEAGE = REPO_ROOT / "docs/current/EXACT78_ACTIVE_LINEAGE_V1.json"
P0 = REPO_ROOT / "_run/current/exact78_general_p0_preflight_v1_20260929/attempts/attempt_0001"
P1_STOP = REPO_ROOT / "_run/current/exact78_general_p1_cross_session_replay_v1_retry1_20260930/attempts/attempt_0001/RESULT.json"
PENDING_ID = "exact78_p1_cold_start_three_session_pending_v1_20260930"
PENDING_PACKET = REPO_ROOT / "tasks/current" / PENDING_ID / "TASK_PACKET.json"
PHASES = ("F0", "F1", "F2", "F3")
LIMIT_SECONDS = {"F0": 10800, "F1": 10800, "F2": 7200, "F3": 7200}


def task_id(phase: str) -> str:
    return f"exact78_preconvergence_{phase.lower()}_v1_20260930"


def root(phase: str) -> Path:
    return REPO_ROOT / "_run/current" / task_id(phase) / "attempts/attempt_0001"


def packet_path(phase: str) -> Path:
    return REPO_ROOT / "tasks/current" / task_id(phase) / "TASK_PACKET.json"


def receipt_path(phase: str) -> Path:
    return REPO_ROOT / "tasks/receipts" / f"EXACT78_PRECONVERGENCE_{phase}_V1.json"


def write_text(path: Path, content: str) -> None:
    atomic_write(path, content.rstrip().encode("utf-8") + b"\n")


def index_copy() -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }


def verify_fixed_parent() -> None:
    if artifact_ref(PARENT)["sha256"] != PARENT_SHA:
        raise RuntimeError("V2_PARENT_RESULT_SHA_DRIFT")
    pointer = load_json(POINTER)
    if pointer["baseline_id"] != "EXACT78_CARDS042_VISUAL_BASELINE_V2":
        raise RuntimeError("V2_POINTER_DRIFT")
    if pointer["stages"]["robot"]["result_sha256"] != PARENT_SHA:
        raise RuntimeError("V2_POINTER_RESULT_DRIFT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("UNEXPECTED_GENERAL_OPERATION")


def register(phase: str) -> dict[str, Any]:
    verify_fixed_parent()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    task, output, packet, receipt = task_id(phase), root(phase), packet_path(phase), receipt_path(phase)
    if output.exists() or packet.exists() or receipt.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    if phase != "F0":
        previous = PHASES[PHASES.index(phase) - 1]
        previous_result = load_json(root(previous) / "RESULT.json")
        if previous_result["status"] != "PASSED":
            raise RuntimeError("PREVIOUS_PHASE_NOT_PASSED")
    output.mkdir(parents=True)
    writer = {
        "pid": os.getpid(),
        "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1,
        "fencing_token": secrets.token_hex(24),
        "scope": task,
    }
    inputs: dict[str, Any] = {
        "plan": artifact_ref(PLAN),
        "route_before": artifact_ref(ROUTE),
        "visual_pointer": artifact_ref(POINTER),
        "visual_parent_result": artifact_ref(PARENT),
        "governance_receipt_before": artifact_ref(RECEIPT_PATH),
    }
    if phase != "F0":
        inputs["previous_result"] = artifact_ref(root(PHASES[PHASES.index(phase) - 1]) / "RESULT.json")
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": artifact_ref(Path(__file__)),
        "phase": phase, "models": "NOT_RUN", "calibration": "NOT_CONSUMED",
        "schema": "exact78-preconvergence-v1",
    })).hexdigest()
    writes = [
        str(output.relative_to(REPO_ROOT)), str(packet.relative_to(REPO_ROOT)),
        str(receipt.relative_to(REPO_ROOT)), "docs/governance", "tasks/current/INDEX.json",
    ]
    if phase == "F0":
        writes += [str(LINEAGE.relative_to(REPO_ROOT))]
    elif phase == "F1":
        writes += [
            "src/chaoyang/cli/__init__.py",
            "src/chaoyang/governance/exact78_operation_gate.py",
            "tests/unit/test_exact78_operation_gate.py",
            "review/README.md", "docs/current/AI_WORK_ENTRY_ZH.md",
        ]
    elif phase == "F2":
        writes += [str(PENDING_PACKET.relative_to(REPO_ROOT))]
    packet_value = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": task,
        "objective": {
            "F0": "Freeze exact Exact78 visual-parent, candidate, historical and unresolved evidence lineage",
            "F1": "Reject unrouted Exact78 CLI operations before module import and correct shallow review navigation",
            "F2": "Freeze three-session Raw cold-start inputs and publish a non-runnable P1 draft",
            "F3": "Prove archive eligibility without moving or deleting any historical file",
        }[phase],
        "phase": f"EXACT78_PRECONVERGENCE_{phase}",
        "plan_revision": "EXACT78_PRECONVERGENCE_V1_20260930",
        "execution_revision": f"EXACT78_PRECONVERGENCE_{phase}_V1",
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=LIMIT_SECONDS[phase])).isoformat(),
        "read_set": [
            "AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
            str(ROUTE.relative_to(REPO_ROOT)), str(POINTER.relative_to(REPO_ROOT)),
        ],
        "write_set": writes,
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V2_PARENT_SHA_PINNED", "NO_MODEL_RUN", "NO_DELETION", "ROBOT100_READ_ONLY"],
        "required_outputs": {
            "F0": ["machine_lineage_lock", "stage_sha_closure", "Chinese_selection_report"],
            "F1": ["preimport_cli_guard", "rejected_review_label", "guard_tests"],
            "F2": ["three_session_raw_manifest", "model_and_code_audit", "nonrunnable_p1_packet"],
            "F3": ["reverse_reference_audit", "pid_fd_cwd_audit", "git_and_license_audit", "size_report"],
        }[phase],
        "budgets": {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
            "training_steps": 0, "wall_seconds_max": LIMIT_SECONDS[phase], "session_attempts_max": 1,
        },
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "V2_POINTER_CHANGE", "0916_WRITE", "DELETION_ATTEMPT", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": f"{phase} preparatory evidence only; no general baseline, training, deletion, model or threshold authority.",
        "writer": writer,
        "inputs": inputs,
        "runtime_code": [artifact_ref(Path(__file__))],
        "run_signature": signature,
        "commands": [f"PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_preconvergence_v1 --phase {phase}"],
    }
    errors = _validate_packet(packet_value)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    atomic_json(output / "WRITER.json", writer)
    packet.parent.mkdir(parents=True)
    atomic_json(packet, packet_value)
    now = now_iso()
    state["tasks"].append({
        "task_id": task, "phase": packet_value["phase"], "attempt": 1,
        "status": "RUNNING", "session": "Exact78_preconvergence", "updated_at": now,
        "heartbeat_at": now, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(packet), "run_signature": signature,
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": task, "session": "Exact78_preconvergence",
        "prerequisites": packet_value["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one immutable phase result or deadline",
    }
    index["task_packets"].append({
        "task_id": task, "packet_path": str(packet.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(
        packet_revision=f"EXACT78_PRECONVERGENCE_{phase}_V1_RUNNING",
        plan_revision=packet_value["plan_revision"], execution_revision=packet_value["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX),
    )
    publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_PRECONVERGENCE_{phase}_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index,
    )
    return packet_value


def heartbeat(phase: str) -> None:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == task_id(phase))
    if row["pid"] != os.getpid() or row["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    row["heartbeat_at"] = now_iso()
    row["updated_at"] = now_iso()
    publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_PRECONVERGENCE_{phase}_V1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
    )


def finish(phase: str, packet: dict[str, Any], outputs: dict[str, Path], detail: dict[str, Any]) -> None:
    verify_fixed_parent()
    output = root(phase)
    result = {
        "schema_version": "exact78-preconvergence-phase-result-v1",
        "task_id": task_id(phase), "phase": phase, "status": "PASSED",
        "execution_complete": True, "general_raw_to_robot_qualified": False,
        "current_general_operation": None, "visual_qualified": False,
        "quality_passed": False, "training_eligible": False,
        "selected_v2_robot_result": artifact_ref(PARENT),
        "outputs": {name: artifact_ref(path) for name, path in outputs.items()},
        "detail": detail, "models_run": False, "training_run": False,
        "files_moved": False, "files_deleted": False, "robot100_modified": False,
        "next_action": {
            "F0": "Register F1 exact78 pre-import operation guard",
            "F1": "Register F2 three-session cold-start contract freeze",
            "F2": "Register F3 archive eligibility proof; keep P1 draft non-runnable",
            "F3": "Only a new finite P1 task may select and test a fresh cold-start runner on three sessions",
        }[phase],
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    receipt = receipt_path(phase)
    atomic_json(receipt, {
        "schema_version": "exact78-preconvergence-receipt-v1", "task_id": task_id(phase),
        "status": "PASSED", "result": artifact_ref(output / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"],
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == task_id(phase))
    if row["pid"] != os.getpid() or row["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    row.update(
        status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(output / "RESULT.json"), terminal_receipt=artifact_ref(receipt),
        quality="PREPARATORY_AUDIT_ONLY", adoption="NOT_ADOPTED", review="MACHINE_AUDITED",
    )
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == task_id(phase):
            item["execution_allowed"] = False
    if phase == "F2":
        pending_ref = artifact_ref(PENDING_PACKET)
        if not any(item["task_id"] == PENDING_ID for item in index["task_packets"]):
            index["task_packets"].append({
                "task_id": PENDING_ID,
                "packet_path": str(PENDING_PACKET.relative_to(REPO_ROOT)),
                "packet_sha256": pending_ref["sha256"],
                "execution_class": "FUTURE_BLOCKED_DRAFT", "execution_allowed": False,
            })
            state["tasks"].append({
                "task_id": PENDING_ID, "phase": "EXACT78_P1_COLD_START_THREE_SESSION",
                "attempt": 0, "status": "PENDING", "session": "Cards042_Chips103_Chips013",
                "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": None,
                "proc_start_ticks": None, "gpu_id": None, "task_packet": pending_ref,
                "run_signature": "UNSELECTED_RUNTIME_CODE", "quality": "NOT_RUN",
                "adoption": "NOT_ADOPTED", "review": "PENDING_NEW_RUNNER",
            })
    index.update(
        packet_revision=f"EXACT78_PRECONVERGENCE_{phase}_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_PRECONVERGENCE_{phase}_V1_COMPLETE",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index,
    )
    validation = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    atomic_json(output / "POST_VALIDATION.json", {
        "returncode": validation.returncode, "stdout": validation.stdout, "stderr": validation.stderr,
    })
    if validation.returncode:
        raise RuntimeError("POST_VALIDATION_FAILED:" + validation.stdout[-3000:] + validation.stderr[-3000:])
    print(json.dumps({
        "phase": phase, "status": "PASSED", "governance_revision": published["governance_revision"],
        "result": artifact_ref(output / "RESULT.json"),
    }, ensure_ascii=False, indent=2), flush=True)


def make_f0() -> tuple[dict[str, Path], dict[str, Any]]:
    output = root("F0")
    pointer = load_json(POINTER)
    selected = {
        name: {**ref, "sha_status": "MATCH" if not validate_artifact_ref(ref) else "DRIFT"}
        for name, ref in pointer["selected_artifacts"].items()
    }
    if any(row["sha_status"] != "MATCH" for row in selected.values()):
        raise RuntimeError("SELECTED_V2_ARTIFACT_DRIFT")
    audits = {stage: audit_refs(REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json")
              for stage, task in STAGE_TASKS.items()}
    failures = [{"stage": stage, **item} for stage, audit in audits.items() for item in audit["failures"]]
    if len(failures) != 9:
        raise RuntimeError(f"V2_CLOSURE_FAILURE_COUNT_DRIFT:{len(failures)}")
    packets = sorted((REPO_ROOT / "tasks/current").glob("exact78_*/TASK_PACKET.json"))
    results = sorted((REPO_ROOT / "_run/current").glob("exact78_*/**/RESULT.json"))
    scripts = sorted((REPO_ROOT / "src/chaoyang").glob("**/*exact78*.py"))
    inventory = {
        "schema_version": "exact78-preconvergence-inventory-v1",
        "visual_reference": {"result": artifact_ref(PARENT), "selected_artifacts": selected},
        "next_candidate_inputs": {"p0_result": artifact_ref(P0 / "RESULT.json"), "p0_session_preflight": artifact_ref(P0 / "SESSION_PREFLIGHT.json")},
        "historical_evidence": {
            "task_packets": [artifact_ref(path) for path in packets],
            "result_files": [artifact_ref(path) for path in results],
            "code_files": [artifact_ref(path) for path in scripts],
        },
        "blocked_reference_proof": failures,
        "stage_audits": audits,
        "stopped_p1_diagnostic": artifact_ref(P1_STOP),
    }
    atomic_json(output / "INVENTORY.json", inventory)
    protected = sorted(path.stem for path in (REPO_ROOT / "src/chaoyang/ops").glob("*exact78*.py"))
    lock = {
        "schema_version": "exact78-active-lineage-v1",
        "status": "VISUAL_PARENT_ONLY_NO_GENERAL_EXECUTOR",
        "selected_visual_parent": "EXACT78_CARDS042_VISUAL_BASELINE_V2",
        "selected_visual_parent_result": artifact_ref(PARENT),
        "selected_visual_artifacts": selected,
        "selected_visual_scope": "play_cards_0902_042_only",
        "strict_replay_closure": "FAILED_9_HISTORICAL_REFERENCES",
        "historical_dependency_failures": failures,
        "general_runnable_operation": None,
        "latest_p1_diagnostic": {"classification": "REJECTED_EVIDENCE_ONLY", "result": artifact_ref(P1_STOP)},
        "operation_protection": {"rule": "all operation names containing exact78", "known_operations": protected},
        "candidate_source_inventory": artifact_ref(P0 / "SESSION_PREFLIGHT.json"),
        "training_eligible": False, "deletion_authorized": False,
        "claim_limit": "Selected V2 visual reference only; historical scripts and results are evidence, not implicit execution authority.",
    }
    atomic_json(LINEAGE, lock)
    write_text(output / "SELECTION_ZH.md", f"""# Exact78 当前选用表

- 可引用的视觉父结果：Cards042 V2，Robot RESULT SHA `{PARENT_SHA}`；只限 042，171 帧。
- 当前通用 Raw→Robot operation：`null`。没有可调用的生产算法。
- 下轮候选输入：P0 冻结的 156 条源清单，三条冷启动输入另由 F2 锁定。
- P1 诊断片：已取消并拒绝视觉晋升，只可作失败证据。
- 历史闭包：四个 V2 阶段包共有 {len(failures)} 处 SHA 漂移或失踪；旧包不能按名称复活。
- 本次列出 Exact78 任务包 {len(packets)} 个、结果文件 {len(results)} 个、相关代码文件 {len(scripts)} 个。详细 SHA 在 `INVENTORY.json`。

缺引用证明的文件不归档、不删除；0916 Robot100 不在本清单中。
""")
    return {
        "inventory": output / "INVENTORY.json", "selection_zh": output / "SELECTION_ZH.md",
        "active_lineage": LINEAGE,
    }, {"historical_packet_count": len(packets), "historical_result_count": len(results),
        "code_count": len(scripts), "v2_closure_failure_count": len(failures)}


def make_f1() -> tuple[dict[str, Path], dict[str, Any]]:
    output = root("F1")
    cli = REPO_ROOT / "src/chaoyang/cli/__init__.py"
    gate = REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py"
    tests = REPO_ROOT / "tests/unit/test_exact78_operation_gate.py"
    review = REPO_ROOT / "review/README.md"
    entry = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
    if "authorize_exact78_operation" not in cli.read_text(encoding="utf-8"):
        raise RuntimeError("CLI_PREIMPORT_GUARD_MISSING")
    if "已拒绝" not in review.read_text(encoding="utf-8"):
        raise RuntimeError("REVIEW_REJECTION_LABEL_MISSING")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "TMPDIR": str(output)}
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "tests/unit/test_cli_maintained_operations.py", "tests/unit/test_exact78_operation_gate.py"]
    test = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    atomic_json(output / "TEST_REPORT.json", {
        "command": command, "returncode": test.returncode, "stdout": test.stdout, "stderr": test.stderr,
    })
    if test.returncode:
        raise RuntimeError("F1_TEST_FAILURE:" + test.stdout[-2500:] + test.stderr[-2500:])
    return {
        "cli": cli, "gate": gate, "tests": tests,
        "review_zh": review, "ai_entry": entry, "test_report": output / "TEST_REPORT.json",
    }, {"guard_tests_passed": True, "historic_operation_without_packet_rejected": True}


def make_f2() -> tuple[dict[str, Path], dict[str, Any]]:
    output = root("F2")
    p0 = load_json(P0 / "SESSION_PREFLIGHT.json")
    rows = {row["session_id"]: row for row in p0["sessions"]}
    ids = ["play_cards_0902_042", "get_potato_chips_0902_103", "get_potato_chips_0901_013"]
    prior = load_json(REPO_ROOT / "tasks/current/exact78_general_p1_action_canary_v1_retry8_20260930/TASK_PACKET.json")
    prior_sources = {row["session_id"]: row["source_video"] for row in load_json(Path(prior["inputs"]["manifest"]["path"]))["sessions"]}
    sessions = []
    for session_id in ids:
        row = rows[session_id]
        source = Path(row["mono"]["path"])
        video = artifact_ref(source)
        if video != prior_sources[session_id]:
            raise RuntimeError(f"SOURCE_VIDEO_SHA_DRIFT:{session_id}")
        source_root = Path(row["source_root"])
        files = {
            "source_video": video,
            "camera_params": artifact_ref(source_root / "camera_params.json"),
            "clip_manifest": artifact_ref(source_root / "clip_manifest.json"),
            "slam_trajectory": artifact_ref(source_root / f"slam_trajectory_{session_id}.jsonl"),
        }
        sessions.append({
            "session_id": session_id, "task": row["task"], "fps": row["mono"]["fps"],
            "frame_count": row["mono"]["decoded_frames"],
            "input_identity_sha256": row["expected_input_identity_sha256"],
            "files": files,
            "intrinsics": "PINNED_CAMERA_PARAMS",
            "stereo_metric_calibration": "PRESENT" if row["frozen_calibration_state"] == "CALIBRATED_METRIC_STEREO" else "ABSENT",
            "visual_layer_allowed_when_metric_absent": True,
        })
    weights = {}
    for name in ("hawor_checkpoint", "hawor_infiller"):
        pinned = prior["inputs"][name]
        if artifact_ref(Path(pinned["path"])) != pinned:
            raise RuntimeError(f"MODEL_SHA_DRIFT:{name}")
        weights[name] = pinned
    mask_packet = load_json(REPO_ROOT / "tasks/current/exact78_cards042_role_mask_rebuild_v1_20260929/TASK_PACKET.json")
    clean_packet = load_json(REPO_ROOT / "tasks/current/exact78_cards042_fresh_clean_v1_20260929/TASK_PACKET.json")
    for name, pinned in (
        ("sam31_checkpoint", mask_packet["inputs"]["sam31_checkpoint"]),
        ("propainter_weights", clean_packet["inputs"]["propainter_weights"]),
        ("flow_weights", clean_packet["inputs"]["flow_weights"]),
        ("raft_weights", clean_packet["inputs"]["raft_weights"]),
    ):
        if artifact_ref(Path(pinned["path"])) != pinned:
            raise RuntimeError(f"MODEL_SHA_DRIFT:{name}")
        weights[name] = pinned
    candidate_code = []
    for ref in prior["runtime_code"]:
        candidate_code.append({**ref, "sha_status": "MATCH" if not validate_artifact_ref(ref) else "DRIFT"})
    contract = {
        "schema_version": "exact78-three-session-cold-start-contract-v1",
        "status": "INPUTS_FROZEN_RUNNER_UNSELECTED",
        "sessions": sessions, "models": weights,
        "candidate_code_audit_not_selected_runtime": candidate_code,
        "selected_runtime_code": None,
        "reuse_policy": "RAW_ONLY_NO_OLD_STATE_ARRAYS_NO_CARDS042_FITTED_TEMPLATE",
        "per_session_fit_policy": "derive base and offsets from same session only",
        "frame_contract": "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md#3",
        "fixed_numeric_gates": {
            "visible_without_action_fraction_max": 0.05,
            "observed_arm_reach_fraction_min": 0.90,
            "position_mm_max": 10.0, "rotation_deg_max": 5.0,
            "velocity_rad_per_frame_max": 0.12, "acceleration_rad_per_frame2_max": 0.06,
            "visual_session_qualified_frame_fraction_min": 0.80,
        },
        "outputs_required": ["per_frame_provenance", "per_frame_stage_status", "raw_robot_diagnostic", "independent_whole_robot", "final_first_person_clean_robot", "four_metric_report", "failure_cluster_table"],
        "visual_parent_result": artifact_ref(PARENT),
        "visual_parent_replayable": False,
        "no_training_or_physical_action_authority": True,
    }
    atomic_json(output / "COLD_START_CONTRACT.json", contract)
    draft = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": PENDING_ID,
        "objective": "Select one fresh Raw-to-Robot runner and then execute a three-session cold-start candidate",
        "phase": "EXACT78_P1_COLD_START_THREE_SESSION",
        "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        "execution_revision": "PENDING_NEW_QUALIFIED_RUNTIME_CODE",
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md",
                     str((output / "COLD_START_CONTRACT.json").relative_to(REPO_ROOT))],
        "write_set": [f"_run/current/{PENDING_ID}/attempts/attempt_0001",
                      f"tasks/current/{PENDING_ID}/TASK_PACKET.json", "docs/governance", "tasks/current/INDEX.json"],
        "prerequisites": ["F0_F1_F2_F3_PASSED", "SELECTED_FRESH_RUNTIME_CODE_SHA", "NEW_WRITER_FENCE", "ONE_GPU_RESOURCE"],
        "required_outputs": contract["outputs_required"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 28800, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_SHA_DRIFT", "MODEL_SHA_DRIFT", "CARDS042_FIT_READ", "DOUBLE_WRITER", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Non-runnable draft only. A new immutable successor with selected code, writer, signature and command is required before execution.",
        "execution_mode": "EXACT78_CANDIDATE",
        "inputs": {"cold_start_contract": artifact_ref(output / "COLD_START_CONTRACT.json")},
        "runtime_code": [], "selected_runtime_code": None,
        "run_signature": None, "commands": [], "writer": None,
    }
    if _validate_packet(draft):
        raise RuntimeError("INVALID_PENDING_P1_PACKET")
    PENDING_PACKET.parent.mkdir(parents=True)
    atomic_json(PENDING_PACKET, draft)
    write_text(output / "READINESS_ZH.md", """# 三会话冷启动合同

固定 Cards042、Chips103、原生 25 FPS Chips013 的 Raw 视频、相机参数、源清单、SLAM 轨迹和模型权重，具体 SHA 见 `COLD_START_CONTRACT.json`。Chips013 有相机内参，但缺米制双目标定，明确记为 `ABSENT`；VISUAL 可继续，METRIC 不得通过。

历史 P1 代码仅作 SHA 审计，**未选作下一轮运行器**。当前待执行任务包没有 command、writer 或运行签名，`execution_allowed=false`；必须另立不可变 successor 才能跑模型。禁止复用任何旧动作数组或 Cards042 拟合模板。
""")
    return {"contract": output / "COLD_START_CONTRACT.json", "pending_p1_packet": PENDING_PACKET,
            "readiness_zh": output / "READINESS_ZH.md"}, {
                "sessions": ids, "runtime_selected": False,
                "pending_p1_execution_allowed": False,
                "metric_calibration_absent_sessions": ["get_potato_chips_0901_013"],
            }


def make_f3() -> tuple[dict[str, Path], dict[str, Any]]:
    output = root("F3")
    inventory = load_json(root("F0") / "INVENTORY.json")
    authoritative = [
        REPO_ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json",
        REPO_ROOT / "docs/governance/DOC_AUTHORITY_MAP.json",
        REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json",
        REPO_ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
        REPO_ROOT / "docs/governance/CURRENT_REGRESSION_MANIFEST.json",
        ROUTE, POINTER, LINEAGE, INDEX,
    ]
    texts = {str(path): path.read_text(encoding="utf-8", errors="replace") for path in authoritative}
    proc_rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = os.readlink(proc / "cwd")
            fds = [os.readlink(item) for item in (proc / "fd").iterdir()]
        except (PermissionError, ProcessLookupError, FileNotFoundError, OSError):
            continue
        held = [value for value in [cwd, *fds] if "/path/to/chaoyang/" in value and "/exact78_" in value]
        if held:
            proc_rows.append({"pid": int(proc.name), "cwd": cwd, "held_paths": held[:25]})
    git = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    if git.returncode:
        raise RuntimeError("GIT_STATUS_FAILED")
    dirty = [line[3:] for line in git.stdout.splitlines() if len(line) >= 4]
    license_files = [str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.glob("LICENSE*") if path.is_file()]
    rows = []
    for ref in inventory["historical_evidence"]["result_files"]:
        path = Path(ref["path"])
        task_dir = path.parts[path.parts.index("current") + 1] if "current" in path.parts else "UNKNOWN"
        refs = [name for name, content in texts.items() if str(path) in content or task_dir in content]
        held = [row["pid"] for row in proc_rows if any(task_dir in value for value in row["held_paths"])]
        dirty_match = [item for item in dirty if task_dir in item or str(path.relative_to(REPO_ROOT)) in item]
        if task_dir.startswith("exact78_cards042_fresh_robot") or refs:
            category = "MUST_KEEP"
            reason = "current_authority_or_selected_parent_reference"
        elif held or dirty_match or not license_files:
            category = "BLOCKED_REFERENCE_PROOF"
            reason = "open_process_or_dirty_untracked_or_license_unknown"
        else:
            category = "ARCHIVE_CANDIDATE"
            reason = "no_current_authority_reference_or_open_process_detected"
        rows.append({"result": ref, "task_dir": task_dir, "category": category,
                     "reason": reason, "current_refs": refs, "open_pids": held, "dirty_matches": dirty_match})
    counts = {key: sum(row["category"] == key for row in rows)
              for key in ("MUST_KEEP", "ARCHIVE_CANDIDATE", "BLOCKED_REFERENCE_PROOF")}
    audit = {
        "schema_version": "exact78-archive-eligibility-audit-v1",
        "scope": "Exact78 result files only; no move or delete",
        "authoritative_sources": [artifact_ref(path) for path in authoritative],
        "rows": rows, "counts": counts,
        "file_bytes_by_category": {key: sum(row["result"]["bytes"] for row in rows if row["category"] == key) for key in counts},
        "process_holders": proc_rows,
        "git_dirty_untracked_count": len(dirty), "git_dirty_untracked_paths": dirty,
        "license_files": license_files,
        "reverse_reference_scope": "receipt, authority map, algorithm contract, current registry, regression manifest, route, V2 pointer, lineage lock, task index",
        "limitation": "Current-authority reverse-reference scan does not prove absence of references in historical packets or binary sidecars. Only an explicit future cleanup packet can broaden proof and authorize movement.",
        "files_moved": False, "files_deleted": False,
    }
    atomic_json(output / "ARCHIVE_ELIGIBILITY.json", audit)
    write_text(output / "ARCHIVE_ELIGIBILITY_ZH.md", f"""# Exact78 历史结果清理资格

- 必须保留：{counts['MUST_KEEP']} 个 RESULT 文件；可归档候选：{counts['ARCHIVE_CANDIDATE']}；引用证明不足：{counts['BLOCKED_REFERENCE_PROOF']}。
- Git dirty/untracked 路径：{len(dirty)}；项目根许可证文件：{len(license_files)}；持有相关路径的进程：{len(proc_rows)}。
- 完整分类、反向引用、PID/FD/CWD 和大小统计见 `ARCHIVE_ELIGIBILITY.json`。
- 本任务仅给出资格，不移动、不删除任何文件。候选还须在独立清理任务里扩展历史与二进制引用证明，并得到用户许可。
""")
    return {"eligibility": output / "ARCHIVE_ELIGIBILITY.json",
            "eligibility_zh": output / "ARCHIVE_ELIGIBILITY_ZH.md"}, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    phase = args.phase
    packet = register(phase)
    print(json.dumps({"phase": phase, "registered": True, "task_id": task_id(phase),
                      "governance_revision": load_json(RECEIPT_PATH)["governance_revision"]}), flush=True)
    if phase == "F1":
        print("F1_WAITING_FOR_PATCH_AND_FINISH", flush=True)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 25)
            if ready:
                line = sys.stdin.readline().strip()
                if line == "FINISH":
                    break
                if not line:
                    raise RuntimeError("F1_STDIN_CLOSED_BEFORE_FINISH")
            heartbeat(phase)
        outputs, detail = make_f1()
    else:
        outputs, detail = {"F0": make_f0, "F2": make_f2, "F3": make_f3}[phase]()
    finish(phase, packet, outputs, detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
