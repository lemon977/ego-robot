"""Register, execute and terminalize Exact78 general baseline P1."""
from __future__ import annotations

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

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_general_p1_action_canary_v1_20260930"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"
COHORT = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
          / "20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json")
MATRIX = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
          / "20260915_exact78_robot_expansion_v76/attempts/attempt_0005_execution/run/input_snapshot"
          / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json")
P0 = REPO_ROOT / "_run/current/exact78_general_p0_preflight_v1_20260929/attempts/attempt_0001/RESULT.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PARENT_SHA256 = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
CANONICAL_ROOT = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/poker/runs/pipeline"
                  / "20260905_hawor_0902_v1/play_cards_0902_042__attempt2")
CANONICAL_NPZ = CANONICAL_ROOT / "output/HAWOR_RAW_MANO21_CANARY.npz"
CANONICAL_RESULT = CANONICAL_ROOT / "RESULT.json"
CARDS_ROLE = (REPO_ROOT / "_run/current/exact78_cards042_role_mask_rebuild_v1_20260929/attempts/attempt_0001"
              / "runner/sessions/play_cards_0902_042/attempts/attempt_0001/role_mask_output/FRAME_MANIFEST.json")
MOUNT = (REPO_ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only"
         / "MOUNT_VISUAL_PROXY_CONTRACT.json")
SELECTED = (
    ("play_cards_0902_042", "fixed_cards042"),
    ("get_potato_chips_0902_103", "fixed_chips103"),
    ("get_potato_chips_0901_010", "historical_role_grade_c"),
    ("get_potato_chips_0901_013", "longest_chips"),
    ("get_potato_chips_0903_007", "longest_historical_two_hand_bbox_overlap_chips"),
    ("play_cards_0901_002", "historical_hawor_role_grade_c"),
    ("play_cards_0901_003", "longest_poker"),
    ("play_cards_0901_016", "longest_historical_two_hand_bbox_overlap_poker"),
)
ROBOT100_GUARDS = [
    REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md",
    REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_world_first_v10.py",
]


def index_copy() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def historical(reference: dict) -> Path:
    path = Path(reference["path"])
    if not path.is_file():
        prefix = "/path/to/chaoyang/tasks/control/runs/"
        if str(path).startswith(prefix):
            path = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
                    / str(path)[len(prefix):])
    actual = artifact_ref(path)
    if actual["bytes"] != reference["bytes"] or actual["sha256"] != reference["sha256"]:
        raise RuntimeError(f"HISTORICAL_REF_DRIFT:{path}")
    return path


def role_manifest(matrix_row: dict, session_id: str) -> dict | None:
    if session_id == "play_cards_0902_042":
        return artifact_ref(CARDS_ROLE)
    result_ref = matrix_row.get("role_mask", {}).get("result")
    if not result_ref:
        return None
    try:
        result_path = historical(result_ref)
    except (FileNotFoundError, RuntimeError):
        return None
    result = load_json(result_path)
    reference = result.get("artifacts", {}).get("frame_manifest")
    if not isinstance(reference, dict):
        return None
    try:
        return artifact_ref(historical(reference))
    except (FileNotFoundError, RuntimeError):
        return None


def build_manifest() -> Path:
    cohort = {row["session_id"]: row for row in load_json(COHORT)["sessions"]}
    matrix = {row["session_id"]: row for row in load_json(MATRIX)["rows"]}
    rows = []
    overlap = {"get_potato_chips_0903_007": {"overlap_frames": 3, "longest_run": 2},
               "play_cards_0901_016": {"overlap_frames": 39, "longest_run": 39}}
    for session_id, reason in SELECTED:
        frozen = cohort[session_id]
        source_root = Path(frozen["raw_path"]).resolve(strict=True)
        source = source_root / f"CameraRecord_{session_id}.mp4"
        row = {"session_id": session_id, "task": frozen["task"], "frame_count": frozen["frame_count"],
               "source_root": str(source_root), "source_video": artifact_ref(source),
               "selection_reason": reason, "historical_hawor_grade": matrix[session_id]["hawor"]["grade"],
               "historical_role_grade": matrix[session_id]["role_mask"]["grade"],
               "role_manifest": role_manifest(matrix[session_id], session_id),
               "overlap_rank_evidence": overlap.get(session_id)}
        rows.append(row)
    manifest = ROOT / "CANARY_MANIFEST.json"
    atomic_json(manifest, {"schema_version": "exact78-general-p1-canary-manifest-v1",
                           "task_id": TASK, "selection_policy": "fixed_two_plus_per_task_grade_c_longest_overlap",
                           "overlap_metric": "longest contiguous positive overlap of historical projected MANO21 bboxes",
                           "sessions": rows})
    return manifest


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    for path in (PLAN, COHORT, MATRIX, P0, PARENT, CANONICAL_NPZ, CANONICAL_RESULT, CARDS_ROLE, MOUNT):
        path.resolve(strict=True)
    if artifact_ref(PARENT)["sha256"] != PARENT_SHA256:
        raise RuntimeError("PARENT_ROBOT_RESULT_SHA_DRIFT")
    ROOT.mkdir(parents=True)
    cache = ROOT / "cache"; cache.mkdir()
    manifest = build_manifest()
    master_operation = os.environ.get(
        "EXACT78_P1_MASTER_OPERATION", "run_exact78_general_p1_action_canary_v1")
    runtime_paths = [
        REPO_ROOT / f"src/chaoyang/ops/{master_operation}.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_raw_v1.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v2.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v1.py",
        REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v1.py",
        REPO_ROOT / "src/chaoyang/ops/run_robot_hand_fullsession_anatomy_repair_v3.py",
        REPO_ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
        REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    ]
    if master_operation in {
            "run_exact78_general_p1_action_canary_v2",
            "run_exact78_general_p1_action_canary_v3",
            "run_exact78_general_p1_action_canary_v4",
            "run_exact78_general_p1_action_canary_v5",
            "run_exact78_general_p1_action_canary_v6",
            "run_exact78_general_p1_action_canary_v7",
            "run_exact78_general_p1_action_canary_v8",
            "run_exact78_general_p1_action_canary_v9",
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v1.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v3.py",
        ))
    if master_operation in {
            "run_exact78_general_p1_action_canary_v4",
            "run_exact78_general_p1_action_canary_v5",
            "run_exact78_general_p1_action_canary_v6",
            "run_exact78_general_p1_action_canary_v7",
            "run_exact78_general_p1_action_canary_v8",
            "run_exact78_general_p1_action_canary_v9",
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.append(REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v4.py")
    if master_operation in {
            "run_exact78_general_p1_action_canary_v5",
            "run_exact78_general_p1_action_canary_v6",
            "run_exact78_general_p1_action_canary_v7",
            "run_exact78_general_p1_action_canary_v8",
            "run_exact78_general_p1_action_canary_v9",
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.append(REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v2.py")
    if master_operation == "run_exact78_general_p1_action_canary_v6":
        runtime_paths.append(REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v5.py")
    if master_operation == "run_exact78_general_p1_action_canary_v7":
        runtime_paths.append(REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v6.py")
    if master_operation in {
            "run_exact78_general_p1_action_canary_v8",
            "run_exact78_general_p1_action_canary_v9",
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_recovery_v6.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_raw_v2.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v3.py",
        ))
    if master_operation in {
            "run_exact78_general_p1_action_canary_v9",
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v8.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v4.py",
        ))
    if master_operation in {
            "run_exact78_general_p1_action_canary_v10",
            "run_exact78_general_p1_action_canary_v11",
            "run_exact78_general_p1_action_canary_v12",
            "run_exact78_general_p1_action_canary_v13"}:
        runtime_paths.append(
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v9.py")
    if master_operation == "run_exact78_general_p1_action_canary_v11":
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v10.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v5.py",
            REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v2.py",
        ))
    if master_operation == "run_exact78_general_p1_action_canary_v12":
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v10.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v11.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v5.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v6.py",
            REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v2.py",
        ))
    if master_operation == "run_exact78_general_p1_action_canary_v13":
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v10.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v11.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v12.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v5.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v6.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v7.py",
            REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v2.py",
        ))
    if master_operation in {
            "run_exact78_general_p1_fast_correction_v1",
            "run_exact78_general_p1_fast_correction_v2"}:
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_arm_causal_visual_v2.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v7.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_action_canary_v8.py",
            REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v2.py",
        ))
    if master_operation == "run_exact78_general_p1_fast_correction_v2":
        runtime_paths.extend((
            REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_fast_correction_v1.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_arm_causal_visual_v3.py",
        ))
    runtime_code = [artifact_ref(path) for path in runtime_paths]
    inputs = {"execution_plan": artifact_ref(PLAN), "manifest": artifact_ref(manifest),
              "p0_result": artifact_ref(P0), "parent_robot_result": artifact_ref(PARENT),
              "canonical_raw_hawor": artifact_ref(CANONICAL_NPZ),
              "canonical_raw_result": artifact_ref(CANONICAL_RESULT), "mount_proxy": artifact_ref(MOUNT),
              "hawor_checkpoint": artifact_ref(REPO_ROOT / "vendor/HaWoR/weights/hawor/checkpoints/hawor.ckpt"),
              "hawor_infiller": artifact_ref(REPO_ROOT / "vendor/HaWoR/weights/hawor/checkpoints/infiller.pt")}
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    quality_rounds = int(os.environ.get("EXACT78_P1_QUALITY_MODIFICATION_ROUNDS", "0"))
    config = {"canaries": len(SELECTED), "gpu_count": 1, "cpu_threads_total": 2,
              "wall_seconds_max": 28800, "position_gate_mm": 10.0, "rotation_gate_deg": 5.0,
              "motion_missing_in_view_fraction_max": 0.05, "observed_reach_fraction_min": 0.90,
              "external_data": "READ_ONLY", "cards042_fit_read": "FORBIDDEN",
              "quality_modification_rounds": quality_rounds}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime_code,
                                                "config": config, "schema": "exact78-general-p1-v1"})).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
              "objective": f"Run {len(SELECTED)} Exact78 action canaries with HaWoR recovery, per-session base solve and two diagnostic views",
              "phase": "EXACT78_GENERAL_P1_ACTION_CANARY", "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
              "execution_revision": "EXACT78_GENERAL_P1_ACTION_CANARY_V1",
              "deadline_at": (datetime.now().astimezone() + timedelta(hours=8)).isoformat(),
              "read_set": ["AGENTS.md", "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"],
              "write_set": [str(ROOT)],
              "prerequisites": ["P0_TERMINAL", "ONE_GPU", "EXTERNAL_DATA_READ_ONLY", "CARDS042_FIT_COLD_START"],
              "required_outputs": ["terminal_action_results", "per_frame_motion_provenance",
                                   "per_session_base_solve", "raw_robot_overlay", "independent_robot_view",
                                   "four_metric_report", "problem_table"],
              "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                          "training_steps": 0, "wall_seconds_max": 28800,
                          "session_attempts_max": 1,
                          "quality_modification_rounds_max": quality_rounds},
              "stop_conditions": ["source_sha_drift", "writer_conflict", "deadline", "external_write_attempt"],
              "claim_limit": "P1 offline visual action canary only; no training, physical action, visual adoption, deletion or Robot100 authority.",
              "writer": writer, "inputs": inputs, "runtime_code": runtime_code,
              "config": config, "run_signature": signature,
              "commands": [f"chaoyang run {master_operation} --manifest {manifest}"]}
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError(str(errors))
    robot100_before = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    atomic_json(ROOT / "ROBOT100_GUARD_BEFORE.json", robot100_before)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
                           "session": f"exact78_p1_{len(SELECTED)}_canaries", "updated_at": now, "heartbeat_at": now,
                           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": 0,
                           "task_packet": artifact_ref(PACKET), "run_signature": signature,
                           "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "PENDING"})
    state["next_task"] = {"task_id": TASK, "session": f"exact78_p1_{len(SELECTED)}_canaries",
                          "prerequisites": packet["prerequisites"], "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
                          "stop_condition": f"{len(SELECTED)} terminal canaries or eight-hour deadline"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_GENERAL_P1_ACTION_CANARY_V1_RUNNING",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_GENERAL_P1_ACTION_CANARY_REGISTERED",
                   expected_revision=revision, generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                   "TMPDIR": str(cache), "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache),
                   "NUMBA_CACHE_DIR": str(cache), "TORCH_EXTENSIONS_DIR": str(cache),
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run",
               master_operation, "--manifest", str(manifest)]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while True:
            try:
                code = worker.wait(timeout=30); break
            except subprocess.TimeoutExpired:
                current = copy.deepcopy(load_json(TASK_STATE_PATH))
                if ((current.get("next_task") or {}).get("task_id") != TASK
                        or time.monotonic() - started > config["wall_seconds_max"]):
                    os.killpg(worker.pid, signal.SIGTERM); code = worker.wait(timeout=30); break
                row = next(item for item in current["tasks"] if item["task_id"] == TASK)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                publish_bundle(load_json(AUTHORITY_PATH), current,
                               event_type="EXACT78_GENERAL_P1_ACTION_CANARY_HEARTBEAT",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index_copy())
    result_path = ROOT / "RESULT.json"
    if not result_path.is_file():
        atomic_json(result_path, {"status": "FAILED_RUNTIME_FINAL", "returncode": code,
                                  "execution_complete": False, "visual_qualified": False,
                                  "quality_passed": False, "deletion_authorized": False})
    guard_after = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    guard_ok = robot100_before == guard_after
    atomic_json(ROOT / "ROBOT100_GUARD_AFTER.json", {"unchanged": guard_ok, "files": guard_after})
    if not guard_ok:
        raise RuntimeError("ROBOT100_GUARD_CHANGED")
    result = load_json(result_path)
    terminal = "PASSED" if code == 0 and result.get("execution_complete") else "FAILED_RUNTIME_FINAL"
    receipt = ROOT / "TERMINAL_RECEIPT.json"
    atomic_json(receipt, {"task_id": TASK, "status": terminal, "result": artifact_ref(result_path),
                          "claim_limit": packet["claim_limit"], "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("FINALIZE_ROUTE_CONFLICT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=terminal, updated_at=now_iso(), heartbeat_at=now_iso(), pid=None,
               proc_start_ticks=None, result=artifact_ref(result_path), terminal_receipt=artifact_ref(receipt),
               quality="ACTION_CANARY_COMPLETE_VISUAL_REVIEW_PENDING" if terminal == "PASSED" else terminal)
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_GENERAL_P1_ACTION_CANARY_V1_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="EXACT78_GENERAL_P1_ACTION_CANARY_TERMINAL",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    print(json.dumps({"status": terminal, "revision": published["governance_revision"],
                      "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0 if terminal == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
