"""Register, execute, and terminalize the current Cards042 Robot replay."""
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
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_cards042_fresh_robot_v1_20260929"
SESSION = "play_cards_0902_042"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_FRESH_ROBOT_V1.json"


def index_body() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")

    clean_root = REPO_ROOT / "_run/current/exact78_cards042_fresh_clean_v1_20260929/attempts/attempt_0001/clean"
    clean = load_json(clean_root / "RESULT.json")
    role_result_path = (REPO_ROOT / "_run/current/exact78_cards042_role_mask_rebuild_v1_20260929/attempts/attempt_0001"
                        / "runner/sessions" / SESSION / "attempts/attempt_0001/role_mask_output/RESULT.json")
    obj_result_path = (REPO_ROOT / "_run/current/exact78_cards042_direct_object_rebuild_v2_20260929"
                       / "attempts/attempt_0001/object_identity/RESULT.json")
    role, obj = load_json(role_result_path), load_json(obj_result_path)
    bundle = (REPO_ROOT / "_run/current/exact78_cards042_visual_baseline_v2_publish_20260929"
              / "attempts/attempt_0001/dependencies/selected_motion_bundle")
    inputs = {
        "clean_result": artifact_ref(clean_root / "RESULT.json"),
        "clean_manifest": clean["artifacts"]["frame_manifest"],
        "clean_prepare": clean["artifacts"]["prepare"],
        "role_result": artifact_ref(role_result_path),
        "role_manifest": role["artifacts"]["frame_manifest"],
        "object_result": artifact_ref(obj_result_path),
        "object_manifest": obj["artifacts"]["frame_manifest"],
        "selected_motion_bundle": artifact_ref(bundle / "SESSION.json"),
        "selected_hawor": artifact_ref(bundle / "inputs/HAWOR.npz"),
        "selected_arm": artifact_ref(bundle / "inputs/ARM.npz"),
        "selected_hand": artifact_ref(bundle / "inputs/HAND.npz"),
        "flange_mesh": artifact_ref(bundle / "assets/flange.STL"),
    }
    runtime_paths = (
        "src/chaoyang/ops/run_exact78_cards042_fresh_robot_v1.py",
        "src/chaoyang/pipeline/exact78_task_runtime_v1.py",
        "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
        "src/chaoyang/pipeline/exact78_visual_compositor_v1.py",
        "src/chaoyang/ops/render_poker_same_side_outward_frame0.py",
        "src/chaoyang/pipeline/robot_renderer_cycles.py",
        "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    )
    runtime_code = [artifact_ref(REPO_ROOT / path) for path in runtime_paths]
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    config = {
        "frame_count": 171, "render_size": [640, 480], "contact_window": [76, 91],
        "contact_method": "FIXED_ROI_BLUE_ENTITY_COMPONENTS_DILATE10_MINUS_CURRENT_HUMAN",
        "ownership_near_far_px": [6, 12], "depth_consumed": False, "object6d_consumed": False,
    }
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime_code, "config": config})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Render a complete Cards042 review from current fresh Clean/masks and selected canonical motion",
        "phase": "ROBOT_VISUAL_REPLAY", "plan_revision": "EXACT78_INPAINT_GENERAL_V1",
        "execution_revision": "EXACT78_CARDS042_FRESH_ROBOT_V1", "session_id": SESSION,
        "deadline_at": (datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_DEPENDENCY_MIGRATION_ZH.md"],
        "write_set": [str(ROOT)],
        "prerequisites": ["NO_ACTIVE_WRITER", "FRESH_CLEAN_EXECUTION_PASS", "ROLE_AND_OBJECT_GRADE_B",
                          "SELECTED_CANONICAL_MOTION_INDEPENDENT_COPY", "EXTERNAL_RGB_READ_ONLY"],
        "required_outputs": ["171_composite", "171_four_panel_review", "171_external_robot",
                             "16_contact_review", "frame_manifest", "quality_result"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0, "training_steps": 0,
                    "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "writer_conflict", "deadline", "quality_C_no_auto_retry"],
        "claim_limit": ("Cards042 development visual: current fresh Clean/masks plus independently migrated selected "
                        "canonical motion. No metric depth, Object6D, physical contact, training, control or deletion authority."),
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code, "config": config,
        "run_signature": signature,
        "commands": [f"chaoyang run run_exact78_cards042_fresh_robot_v1 --packet {PACKET}"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING", "session": SESSION,
        "updated_at": now, "heartbeat_at": now, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature, "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "PENDING",
    })
    state["next_task"] = {"task_id": TASK, "session": SESSION, "prerequisites": packet["prerequisites"],
                          "expected_resource": "TWO_CPU_THREADS_NO_GPU", "stop_condition": "one terminal attempt or deadline"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CARDS042_FRESH_ROBOT_V1_RUNNING", status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_FRESH_ROBOT_V1_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)

    cache = ROOT / "cache"
    cache.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "CUDA_VISIBLE_DEVICES": "", **{key: str(cache) for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")},
           **{key: "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", "run_exact78_cards042_fresh_robot_v1",
               "--packet", str(PACKET)]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        started = time.monotonic()
        while True:
            try:
                returncode = worker.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                state = copy.deepcopy(load_json(TASK_STATE_PATH))
                if (state.get("next_task") or {}).get("task_id") != TASK or time.monotonic() - started > 3600:
                    os.killpg(worker.pid, signal.SIGTERM)
                    returncode = worker.wait(timeout=30)
                    break
                row = next(item for item in state["tasks"] if item["task_id"] == TASK)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_FRESH_ROBOT_V1_HEARTBEAT",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=index_body())
    if not (ROOT / "RESULT.json").is_file():
        atomic_json(ROOT / "RESULT.json", {"status": "FAILED_RUNTIME_FINAL", "returncode": returncode})
    status = "PASSED" if returncode == 0 else "FAILED_QUALITY_C" if returncode == 2 else "FAILED_RUNTIME_FINAL"
    atomic_json(RECEIPT, {"task_id": TASK, "status": status, "result": artifact_ref(ROOT / "RESULT.json"),
                          "human_review": "PENDING" if status == "PASSED" else "NOT_APPLICABLE",
                          "baseline_adopted": False, "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("FINALIZE_ROUTE_CONFLICT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ROOT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT),
               quality="VISUAL_REVIEW_CANDIDATE" if status == "PASSED" else status,
               review="PENDING" if status == "PASSED" else "NOT_APPLICABLE")
    state["next_task"] = None
    index = index_body()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_FRESH_ROBOT_V1_TERMINAL", status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_FRESH_ROBOT_V1_TERMINAL",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=index)
    print(json.dumps({"status": status, "result": artifact_ref(ROOT / "RESULT.json"),
                      "revision": published["governance_revision"]}))
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
