"""Register, own and close one Cards042 four-role SAM3.1 reconstruction."""
from __future__ import annotations

import ast
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


TASK = "exact78_cards042_role_mask_rebuild_v1_20260929"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = "play_cards_0902_042"
SOURCE = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards") / SESSION
MOTION_BUNDLE = (REPO_ROOT / "_run/current/exact78_cards042_visual_baseline_v2_publish_20260929"
                 / "attempts/attempt_0001/dependencies/selected_motion_bundle")
HAWOR = MOTION_BUNDLE / "inputs/HAWOR.npz"
HAWOR_RESULT = MOTION_BUNDLE / "SESSION.json"
OLD_CONFIG = (
    REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
    / "20260908_two_task_e2e_baseline_v1/mask_fresh_fullsession_v1/config/POKER042.json"
)
ADMISSION = (
    REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
    / "20260909_exact78_current_baseline_batch_v1/MASK_ROLE_BOUNDED_V2_1_AUTHORITY.json"
)


def index_copy():
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def local_code_closure() -> list[dict]:
    queue = [
        REPO_ROOT / "src/chaoyang/ops/run_role_mask_runtime_contract_repair_v71.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_fullsession_role_mask.py",
    ]
    seen: set[Path] = set()
    while queue:
        path = queue.pop().resolve(strict=True)
        if path in seen or path.suffix != ".py":
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("chaoyang"):
                modules = [node.module, *(node.module + "." + item.name for item in node.names)]
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names if item.name.startswith("chaoyang")]
            for module in modules:
                candidate = REPO_ROOT / "src" / (module.replace(".", "/") + ".py")
                if candidate.is_file() and candidate.parent.name != "governance":
                    queue.append(candidate)
    return [artifact_ref(path) for path in sorted(seen)]


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    motion = load_json(HAWOR_RESULT)
    if motion.get("session_id") != SESSION or motion.get("frame_count") != 171:
        raise RuntimeError("SELECTED_MOTION_BUNDLE_NOT_COMPLETE")
    frames = [SOURCE / "preprocess/all_data" / f"{frame:05d}" / "rgb.png" for frame in range(171)]
    if not all(path.is_file() for path in frames):
        raise RuntimeError("LOSSLESS_RGB_FRAME_SET_INCOMPLETE")
    ROOT.mkdir(parents=True)
    input_manifest = {
        "schema_version": "exact78-cards042-role-mask-input-v1",
        "session_id": SESSION,
        "frame_count": 171,
        "lossless_rgb_frames": [artifact_ref(path) for path in frames],
        "raw_video": artifact_ref(SOURCE / f"CameraRecord_{SESSION}.mp4"),
        "current_temporal_hawor": artifact_ref(HAWOR),
        "current_temporal_result": artifact_ref(HAWOR_RESULT),
        "accepted_prompt_source": artifact_ref(OLD_CONFIG),
        "admission_contract": artifact_ref(ADMISSION),
    }
    atomic_json(ROOT / "INPUT_MANIFEST.json", input_manifest)
    config = {
        "schema_version": "exact78-fullsession-role-mask-config-v1",
        "task_id": "poker",
        "session_id": SESSION,
        "raw_all_data": str(SOURCE / "preprocess/all_data"),
        "raw_video": input_manifest["raw_video"],
        "hawor_npz": input_manifest["current_temporal_hawor"],
        "frame_count": 171,
        "anchor_frame": 125,
        "fps": 30.0,
        "human_anchors": {"left_human": [525, 730], "right_human": [730, 315]},
        "tracker_roles": {
            "upper_tracker_wearable": {
                "id": 201, "name": "upper_tracker_wearable",
                "positive": [[895, 140], [910, 190]],
                "negative": [[730, 315], [525, 730]],
            },
            "lower_tracker_wearable": {
                "id": 202, "name": "lower_tracker_wearable",
                "positive": [[265, 810], [235, 840]],
                "negative": [[525, 730], [730, 315]],
            },
        },
        "tracker_anatomical": {"upper_tracker_wearable": "right_tracker", "lower_tracker_wearable": "left_tracker"},
        "tracker_human_internal": {"upper_human_core": "right_human", "lower_human_core": "left_human"},
        "tracker_prompt_contract": "SAME_ID_POSITIVE_BOOTSTRAP_THEN_FULL_REFINE",
        "refresh_cadence_frames": 4,
        "review_language": "zh-CN",
        "admission_canaries": input_manifest["admission_contract"],
    }
    atomic_json(ROOT / "ROLE_MASK_CONFIG.json", config)
    runtime_code = local_code_closure()
    writer = {
        "pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK,
    }
    inputs = {
        "manifest": artifact_ref(ROOT / "INPUT_MANIFEST.json"),
        "config": artifact_ref(ROOT / "ROLE_MASK_CONFIG.json"),
        "sam31_checkpoint": artifact_ref(REPO_ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"),
        "algorithm_contract": artifact_ref(REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json"),
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime_code, "config": config,
        "schema": "exact78-cards042-four-role-mask-rebuild-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Rebuild Cards042 left/right human and tracker role masks with the maintained SAM3.1 bounded-role algorithm",
        "phase": "MASK_ROLE_REBUILD", "plan_revision": "EXACT78_INPAINT_GENERAL_V1",
        "execution_revision": "EXACT78_CARDS042_ROLE_MASK_REBUILD_V1",
        "deadline_at": (datetime.now().astimezone() + timedelta(hours=2)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_BASELINE_CLOSURE_REVIEW_ZH.md"],
        "write_set": [str(ROOT)],
        "prerequisites": ["NO_ACTIVE_WRITER", "PINNED_SAM31", "CURRENT_TEMPORAL_HAWOR_COMPLETE", "EXTERNAL_RGB_READ_ONLY"],
        "required_outputs": ["four_role_frame_manifest", "tracker_reentry_manifest", "full_review_video", "quality_result"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 7200, "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "writer_conflict", "gpu_lease_unavailable", "deadline", "quality_C_no_auto_retry"],
        "claim_limit": "One Cards042 four-role Mask reconstruction. Grade B is only a downstream candidate; no Clean, baseline adoption, deletion or training authority.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code, "run_signature": signature,
        "commands": [
            f"chaoyang run run_role_mask_runtime_contract_repair_v71 --output-root {ROOT / 'runner'} "
            f"--config {ROOT / 'ROLE_MASK_CONFIG.json'} --confirm-config-sha {inputs['config']['sha256']} "
            f"--attempt-id attempt_0001 --artifact-revision R7_3 --executor-epoch 1 --fencing-token {writer['fencing_token']}"
        ],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError(str(errors))
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": SESSION, "updated_at": now, "heartbeat_at": now, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": 0, "task_packet": artifact_ref(PACKET),
        "run_signature": signature, "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "PENDING",
    })
    state["next_task"] = {
        "task_id": TASK, "session": SESSION, "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS", "stop_condition": "one terminal attempt or deadline",
    }
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE",
        "execution_allowed": True,
    })
    index.update(packet_revision="EXACT78_CARDS042_ROLE_MASK_REBUILD_V1_RUNNING", status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_ROLE_MASK_REBUILD_REGISTERED",
        expected_revision=revision, generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)
    cache = ROOT / "cache"
    cache.mkdir()
    environment = {
        **os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(cache), "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "hf"),
        "MPLCONFIGDIR": str(cache), "NUMBA_CACHE_DIR": str(cache), "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", "run_role_mask_runtime_contract_repair_v71",
        "--output-root", str(ROOT / "runner"), "--config", str(ROOT / "ROLE_MASK_CONFIG.json"),
        "--confirm-config-sha", inputs["config"]["sha256"], "--attempt-id", "attempt_0001",
        "--artifact-revision", "R7_3", "--executor-epoch", "1", "--fencing-token", writer["fencing_token"]]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while True:
            try:
                code = worker.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                state = copy.deepcopy(load_json(TASK_STATE_PATH))
                if (state.get("next_task") or {}).get("task_id") != TASK or time.monotonic() - started > 7200:
                    os.killpg(worker.pid, signal.SIGTERM)
                    code = worker.wait(timeout=30)
                    break
                row = next(item for item in state["tasks"] if item["task_id"] == TASK)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_ROLE_MASK_HEARTBEAT",
                    expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                    task_packet_index_path=INDEX, task_packet_index_value=index_copy())
    wrapper_result = ROOT / "runner/sessions" / SESSION / "attempts/attempt_0001/RESULT.json"
    if wrapper_result.is_file():
        wrapped = load_json(wrapper_result)
        role_ref = wrapped.get("role_mask_result")
        result = {
            "status": wrapped.get("status"), "returncode": code, "wrapper_result": artifact_ref(wrapper_result),
            "role_mask_result": role_ref, "baseline_adopted": False, "deletion_authorized": False,
            "next_action": "OBJECT_IDENTITY_AND_EXPANDED_HANDOFF" if wrapped.get("status") == "PASSED" else "STOP_QUALITY_C",
        }
    else:
        result = {"status": "FAILED_RUNTIME_FINAL", "returncode": code, "baseline_adopted": False,
                  "deletion_authorized": False, "next_action": "DIAGNOSE_RUNTIME"}
    atomic_json(ROOT / "RESULT.json", result)
    terminal = "PASSED" if result["status"] == "PASSED" else result["status"]
    receipt_path = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_ROLE_MASK_REBUILD_V1.json"
    atomic_json(receipt_path, {"task_id": TASK, "status": terminal, "result": artifact_ref(ROOT / "RESULT.json"),
                              "claim_limit": packet["claim_limit"], "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("FINALIZE_ROUTE_CONFLICT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=terminal, updated_at=now_iso(), heartbeat_at=now_iso(), pid=None, proc_start_ticks=None,
               result=artifact_ref(ROOT / "RESULT.json"), terminal_receipt=artifact_ref(receipt_path),
               quality="GRADE_B_CANDIDATE" if terminal == "PASSED" else terminal)
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_ROLE_MASK_REBUILD_V1_TERMINAL",
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_CARDS042_ROLE_MASK_REBUILD_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    print(json.dumps({"status": terminal, "revision": published["governance_revision"],
                      "result": artifact_ref(ROOT / "RESULT.json")}, ensure_ascii=False))
    return 0 if terminal == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
