"""Register and freeze a new 100-session Robot development-video batch."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
import secrets

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "robot100_batch_baseline_20260928"
PLAN_REVISION = "ROBOT100_BATCH_BASELINE_V1_20260928"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET_PATH = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
DATA = Path("/path/to/private_data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned")
OPERATION = REPO_ROOT / "src/chaoyang/ops/run_robot100_batch_baseline_v1.py"


def _ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])


def _candidates(task: str) -> list[dict]:
    root = DATA / task
    expected_prefix = "play_cards_0916_" if task == "playing_cards" else "get_potato_chips_0916_"
    rows = []
    for session in sorted(root.iterdir(), key=lambda path: path.name):
        if not session.is_dir() or not session.name.startswith(expected_prefix):
            continue
        manifest_path = session / "clip_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        manifest = load_json(manifest_path)
        contract = manifest.get("acquisition_aligned_hdf5_contract", {})
        modality = manifest.get("modality_contract", {})
        if (
            manifest.get("processing_status") != "ready"
            or modality.get("manus") != "PRESENT"
            or contract.get("schema") != "egodex_v1"
            or contract.get("all_exported_frames_complete") is not True
        ):
            continue
        video = session / manifest["files"]["video"]
        hdf5 = Path(contract["path"])
        if not video.is_file() or video.is_symlink() or not hdf5.is_file() or hdf5.is_symlink():
            continue
        hdf5_ref = artifact_ref(hdf5)
        if hdf5_ref["sha256"] != contract["sha256"]:
            raise RuntimeError(f"HDF5_MANIFEST_SHA_DRIFT:{session.name}")
        frame_count = int(manifest["video"]["frame_count"])
        fps = float(manifest["video"]["fps"])
        if frame_count < 2 or fps <= 0:
            continue
        rows.append({
            "session_id": session.name,
            "task": task,
            "processed_session": str(session.resolve()),
            "clip_manifest": artifact_ref(manifest_path),
            "video": artifact_ref(video),
            "hdf5": hdf5_ref,
            "frames": frame_count,
            "fps": fps,
            "selection_reason": "FIRST_50_LEXICOGRAPHIC_WITH_COMPLETE_ACQUISITION_ALIGNED_CONTROLLER_MANUS_BY_TASK",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--writer-pid", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    writer_ticks = _ticks(args.writer_pid)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    old_index = load_json(INDEX)
    active = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    if any(row.get("status") in active for row in state.get("tasks", [])):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if old_index.get("task_packets"):
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    if ATTEMPT.exists() or PACKET_PATH.parent.exists():
        raise RuntimeError("FRESH_ATTEMPT_REQUIRED")
    if not OPERATION.is_file():
        raise RuntimeError("OPERATION_MISSING")

    cards = _candidates("playing_cards")
    chips = _candidates("potato_chips")
    if len(cards) < 50 or len(chips) < 50:
        raise RuntimeError(f"INSUFFICIENT_ELIGIBLE_SESSIONS:cards={len(cards)} chips={len(chips)}")
    selected = cards[:50] + chips[:50]
    if len({row["session_id"] for row in selected}) != 100:
        raise RuntimeError("SELECTION_NOT_100_UNIQUE")

    ATTEMPT.mkdir(parents=True)
    selection_path = ATTEMPT / "SELECTION.json"
    atomic_json(selection_path, {
        "schema_version": "CHAOYANG_ROBOT100_FROZEN_SELECTION_V1",
        "task_id": TASK_ID,
        "created_at": now_iso(),
        "count": 100,
        "by_task": {"playing_cards": 50, "potato_chips": 50},
        "cohort": "chips_cards_handle_highview_0916",
        "selection_policy": "Deterministic lexicographic first 50 per task after complete Controller+MANUS/HDF5/video contract validation; no visual-quality cherry-picking.",
        "rows": selected,
        "claim_limit": "Input inventory only; session inclusion is not an algorithm-quality, physical-accuracy, contact, control, or training claim.",
    })
    writer = {
        "pid": args.writer_pid,
        "proc_start_ticks": writer_ticks,
        "executor_epoch": 1,
        "fencing_token": secrets.token_hex(24),
        "scope": TASK_ID,
        "created_at": now_iso(),
    }
    atomic_json(ATTEMPT / "WRITER.json", writer)
    deadline = (datetime.now().astimezone() + timedelta(days=3)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": "Produce at least 100 distinct full-session standardised Tianji+KaiHand+adapter Robot development videos from the frozen 0916 Controller+MANUS cohort.",
        "phase": "ROBOT100_BATCH_VISUAL_BASELINE_V1",
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "read_set": [
            "AGENTS.md",
            "docs/governance/ALGORITHM_CONTRACT.json",
            str(selection_path.relative_to(REPO_ROOT)),
            "src/chaoyang/ops/run_robot100_batch_baseline_v1.py",
            "src/chaoyang/pipeline/full_robot_review_v2.py",
            "src/chaoyang/pipeline/v5_product.py",
        ],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT))],
        "prerequisites": ["governance_PASS", "single_writer", "frozen_100_session_selection", "registered_operation"],
        "required_outputs": ["100 session RESULT.json files", "100 distinct MP4 videos", "RESULT.json", "VIDEO_INDEX.json", "先看这里.md"],
        "budgets": {
            "cpu_threads_total": 8,
            "workers_max": 2,
            "gpu_count": 0,
            "training_steps": 0,
            "session_attempts_max": 2,
            "wall_seconds_max": 259200,
        },
        "stop_conditions": [
            "100_unique_session_videos_verified",
            "unrecoverable_input_integrity_failure",
            "writer_identity_or_scope_conflict",
        ],
        "selection": artifact_ref(selection_path),
        "writer": writer,
        "deadline_at": deadline,
        "commands": [
            "chaoyang validate-governance",
            "chaoyang run run_robot100_batch_baseline_v1 --selection <frozen> --output-root <attempt>/sessions --worker-count 2 --worker-index 0",
            "chaoyang run run_robot100_batch_baseline_v1 --selection <frozen> --output-root <attempt>/sessions --worker-count 2 --worker-index 1",
        ],
        "claim_limit": "Standardised offline Robot development visualisation only; virtual placement and MANUS retargeting are not physical calibration, contact truth, control truth, training labels, or deployment authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    PACKET_PATH.parent.mkdir(parents=True)
    atomic_json(PACKET_PATH, packet)
    packet_ref = artifact_ref(PACKET_PATH)
    created = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID,
        "phase": packet["phase"],
        "plan_execution_revision": PLAN_REVISION,
        "attempt": 1,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": created,
        "session": "100_frozen_0916_sessions",
        "pid": args.writer_pid,
        "proc_start_ticks": writer_ticks,
        "gpu_id": None,
        "task_packet": packet_ref,
        "deadline_at": deadline,
        "quality": "NOT_EVALUATED",
        "adoption": "NOT_ADOPTED",
        "review": "NOT_REVIEWED",
    })
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": "100_frozen_0916_sessions",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_ONLY_TWO_WORKERS",
        "stop_condition": "100 unique fully decoded Robot videos or explicit terminal input-integrity failure",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "attempt": 1,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered new user-requested 100-session Robot development-video batch; prior completion campaign remains terminal.",
        "task_packet": packet_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": PLAN_REVISION,
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{
            "task_id": TASK_ID,
            "packet_path": str(PACKET_PATH.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
        }],
        "claim_limit": packet["claim_limit"],
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="ROBOT100_BATCH_BASELINE_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    atomic_json(ATTEMPT / "REGISTRATION.json", {
        "schema_version": "CHAOYANG_ROBOT100_REGISTRATION_V1",
        "task_id": TASK_ID,
        "status": "REGISTERED",
        "governance_revision": published["governance_revision"],
        "selection": artifact_ref(selection_path),
        "task_packet": packet_ref,
        "writer": writer,
        "deadline_at": deadline,
    })
    print(json.dumps({"status": "REGISTERED", "task_id": TASK_ID, "revision": published["governance_revision"], "sessions": 100}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
