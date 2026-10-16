"""Register, execute, and terminalize the fresh Cards042 Clean candidate."""
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
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    canonical_bytes,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_cards042_fresh_clean_v1_20260929"
SESSION = "play_cards_0902_042"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_FRESH_CLEAN_V1.json"


def index_body() -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")

    role_result_path = (
        REPO_ROOT
        / "_run/current/exact78_cards042_role_mask_rebuild_v1_20260929/attempts/attempt_0001"
        / "runner/sessions"
        / SESSION
        / "attempts/attempt_0001/role_mask_output/RESULT.json"
    )
    object_result_path = (
        REPO_ROOT
        / "_run/current/exact78_cards042_direct_object_rebuild_v2_20260929"
        / "attempts/attempt_0001/object_identity/RESULT.json"
    )
    role = load_json(role_result_path)
    obj = load_json(object_result_path)
    if role.get("grade") != "B" or not role.get("downstream_authorized"):
        raise RuntimeError("ROLE_GRADE_B_REQUIRED")
    if obj.get("grade") != "B" or not obj.get("downstream_authorized"):
        raise RuntimeError("OBJECT_GRADE_B_REQUIRED")

    raw_root = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards") / SESSION / "preprocess/all_data"
    raw_frames = {str(frame): artifact_ref(raw_root / f"{frame:05d}/rgb.png") for frame in range(171)}
    inputs = {
        "role_result": artifact_ref(role_result_path),
        "role_manifest": role["artifacts"]["frame_manifest"],
        "object_result": artifact_ref(object_result_path),
        "object_manifest": obj["artifacts"]["frame_manifest"],
        "raw_frames": raw_frames,
        "propainter_entry": artifact_ref(REPO_ROOT / "vendor/ProPainter/inference_propainter.py"),
        "raft_weights": artifact_ref(REPO_ROOT / "assets/models/vendor/propainter/raft-things.pth"),
        "flow_weights": artifact_ref(REPO_ROOT / "assets/models/vendor/propainter/recurrent_flow_completion.pth"),
        "propainter_weights": artifact_ref(REPO_ROOT / "assets/models/vendor/propainter/ProPainter.pth"),
    }
    runtime_paths = (
        "src/chaoyang/ops/run_exact78_cards042_fresh_clean_v1.py",
        "src/chaoyang/pipeline/exact78_task_runtime_v1.py",
        "src/chaoyang/pipeline/causal_modal_mask_gpu_adapter_v71.py",
        "vendor/ProPainter/inference_propainter.py",
    )
    runtime_code = [artifact_ref(REPO_ROOT / path) for path in runtime_paths]
    writer = {
        "pid": os.getpid(),
        "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1,
        "fencing_token": secrets.token_hex(24),
        "scope": TASK,
    }
    config = {
        "frame_count": 171,
        "role_radii_px": {"left_human": 24, "right_human": 18, "left_tracker": 60, "right_tracker": 60},
        "temporal_radius_frames": 1,
        "propainter_size": [960, 720],
        "propainter_mask_dilation": 4,
        "object_protection": "current_physical_object_mask",
    }
    signature = hashlib.sha256(
        canonical_bytes({"inputs": inputs, "runtime_code": runtime_code, "config": config})
    ).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK,
        "objective": "Build one fresh Cards042 Clean candidate from admitted current role/object masks",
        "phase": "CLEAN_SYNTHETIC_REBUILD",
        "plan_revision": "EXACT78_INPAINT_GENERAL_V1",
        "execution_revision": "EXACT78_CARDS042_FRESH_CLEAN_V1",
        "session_id": SESSION,
        "deadline_at": (datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_BASELINE_CLOSURE_REVIEW_ZH.md"],
        "write_set": [str(ROOT)],
        "prerequisites": [
            "NO_ACTIVE_WRITER",
            "ROLE_MASK_GRADE_B",
            "OBJECT_MASK_GRADE_B",
            "PINNED_PROPAINTER",
            "EXTERNAL_RGB_READ_ONLY",
        ],
        "required_outputs": [
            "171_clean_frames",
            "171_pixel_source_maps",
            "clean_video",
            "three_panel_review_video",
            "quality_result",
        ],
        "budgets": {
            "workers_max": 1,
            "cpu_threads_total": 2,
            "gpu_count": 1,
            "training_steps": 0,
            "wall_seconds_max": 3600,
            "session_attempts_max": 1,
        },
        "stop_conditions": [
            "source_sha_drift",
            "writer_conflict",
            "gpu_lease_unavailable",
            "deadline",
            "quality_C_no_auto_retry",
        ],
        "claim_limit": (
            "One fresh synthetic Clean candidate. Human visual review remains mandatory; "
            "no observed-background, baseline adoption, deletion, training or geometry authority."
        ),
        "writer": writer,
        "inputs": inputs,
        "runtime_code": runtime_code,
        "config": config,
        "run_signature": signature,
        "commands": [f"chaoyang run run_exact78_cards042_fresh_clean_v1 --packet {PACKET}"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError(str(errors))

    ROOT.mkdir(parents=True)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append(
        {
            "task_id": TASK,
            "phase": packet["phase"],
            "attempt": 1,
            "status": "RUNNING",
            "session": SESSION,
            "updated_at": now,
            "heartbeat_at": now,
            "pid": writer["pid"],
            "proc_start_ticks": writer["proc_start_ticks"],
            "gpu_id": 0,
            "task_packet": artifact_ref(PACKET),
            "run_signature": signature,
            "quality": "PENDING",
            "adoption": "NOT_ADOPTED",
            "review": "PENDING",
        }
    )
    state["next_task"] = {
        "task_id": TASK,
        "session": SESSION,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one terminal attempt or deadline",
    }
    index["task_packets"].append(
        {
            "task_id": TASK,
            "packet_path": str(PACKET.relative_to(REPO_ROOT)),
            "packet_sha256": artifact_ref(PACKET)["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
        }
    )
    index.update(
        packet_revision="EXACT78_CARDS042_FRESH_CLEAN_V1_RUNNING",
        status="PASS",
        supersedes_index=artifact_ref(INDEX),
    )
    publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="EXACT78_CARDS042_FRESH_CLEAN_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )

    cache = ROOT / "cache"
    cache.mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        **{key: str(cache) for key in ("TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")},
        **{key: "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = [
        sys.executable,
        "-B",
        "-m",
        "chaoyang.cli",
        "run",
        "run_exact78_cards042_fresh_clean_v1",
        "--packet",
        str(PACKET),
    ]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
                publish_bundle(
                    load_json(AUTHORITY_PATH),
                    state,
                    event_type="EXACT78_CARDS042_FRESH_CLEAN_V1_HEARTBEAT",
                    expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                    generator_path=Path(__file__),
                    task_packet_index_path=INDEX,
                    task_packet_index_value=index_body(),
                )

    if not (ROOT / "RESULT.json").is_file():
        atomic_json(ROOT / "RESULT.json", {"status": "FAILED_RUNTIME_FINAL", "returncode": returncode})
    result = load_json(ROOT / "RESULT.json")
    status = "PASSED" if returncode == 0 else "FAILED_QUALITY_C" if returncode == 2 else "FAILED_RUNTIME_FINAL"
    atomic_json(
        RECEIPT,
        {
            "task_id": TASK,
            "status": status,
            "result": artifact_ref(ROOT / "RESULT.json"),
            "clean_stage_status": result.get("status"),
            "human_review": "PENDING" if status == "PASSED" else "NOT_APPLICABLE",
            "deletion_authorized": False,
            "baseline_adopted": False,
            "claim_limit": packet["claim_limit"],
        },
    )
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("FINALIZE_ROUTE_CONFLICT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(
        status=status,
        pid=None,
        proc_start_ticks=None,
        updated_at=now_iso(),
        heartbeat_at=now_iso(),
        result=artifact_ref(ROOT / "RESULT.json"),
        terminal_receipt=artifact_ref(RECEIPT),
        quality="EXECUTION_PASS_VISUAL_REVIEW_PENDING" if status == "PASSED" else status,
        review="PENDING" if status == "PASSED" else "NOT_APPLICABLE",
    )
    state["next_task"] = None
    index = index_body()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(
        packet_revision="EXACT78_CARDS042_FRESH_CLEAN_V1_TERMINAL",
        status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="EXACT78_CARDS042_FRESH_CLEAN_V1_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    print(json.dumps({"status": status, "result": artifact_ref(ROOT / "RESULT.json"), "revision": published["governance_revision"]}))
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
