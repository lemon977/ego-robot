"""Register and seal the CPU-only Exact78 three-source T0 input freeze."""

from __future__ import annotations

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

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_raw_to_robot_t0_input_freeze_v1_20260930"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
SCRIPT = Path(__file__)
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T0_V1.json"
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
F2 = REPO_ROOT / "_run/current/exact78_preconvergence_f2_v1_20260930/attempts/attempt_0001/COLD_START_CONTRACT.json"
GROUP_RECEIPT = REPO_ROOT / "tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json"
TRACKER = Path("/path/to/private_data/egodata/datasets/ego")
RAW_0915 = TRACKER / "chips_cards_hands_0915"
DEV = {
    "0901": ["get_potato_chips_0901_013"],
    "0902": ["play_cards_0902_042", "get_potato_chips_0902_103"],
    "0915": ["play_cards_0915_001"],
}
FORBIDDEN = [
    "trackingData*.txt", "raw/pico.jsonl", "raw/tactile.jsonl",
    "raw/tactile.meta.json", "preprocess/all_data/**/hand*",
    "**/HAWOR*.npz", "**/HAND*.npz", "**/ARM*.npz",
    "**/roi*.json", "**/mask*.npy", "**/clean*.mp4",
    "**/robot_q*.npz", "**/contact_window*", "**/placement*",
]


def tracker_group(manifest: dict[str, Any]) -> dict[str, Any]:
    pieces = manifest.get("pieces") or []
    videos = sorted({str(x.get("source_video")) for x in pieces if x.get("source_video")})
    return {"source_root": manifest.get("source_root"), "run_id": manifest.get("run_id"),
            "piece_source_videos": videos}


def inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read identity and completeness metadata only; never inspect hand motion."""
    cohorts: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for source in ("0901", "0902"):
        root = TRACKER / f"chips_cards_tracker_{source}"
        rows = []
        for manifest_path in sorted(root.glob("*/*/clip_manifest.json"), key=lambda p: str(p.parent)):
            session = manifest_path.parent
            manifest = load_json(manifest_path)
            name = session.name
            video = session / manifest.get("files", {}).get("video", f"CameraRecord_{name}.mp4")
            stereo = session / manifest.get("files", {}).get("source_stereo_video", f"source_stereo/CameraRecord_{name}_stereo.mp4")
            camera = session / "camera_params.json"
            slam = session / f"slam_trajectory_{name}.jsonl"
            humanego = session / "preprocess/pico_humanego_manifest.json"
            required = [video, stereo, camera, manifest_path, slam, humanego]
            group = tracker_group(manifest)
            rows.append({"session_id": name, "raw_path": str(session),
                         "group": group, "group_key": hashlib.sha256(canonical_bytes(group)).hexdigest(),
                         "complete": all(p.is_file() and p.stat().st_size > 0 for p in required),
                         "missing": [str(p) for p in required if not p.is_file() or p.stat().st_size == 0],
                         "clip_manifest": artifact_ref(manifest_path),
                         "video": str(video), "stereo": str(stereo), "camera": str(camera),
                         "slam": str(slam), "humanego": str(humanego),
                         "fps": manifest.get("video", {}).get("fps"),
                         "frame_count": manifest.get("video", {}).get("frame_count"),
                         "capture_label": manifest.get("created_at")})
        cohorts[source] = rows
        dev_groups = {r["group_key"] for r in rows if r["session_id"] in DEV[source]}
        for name in DEV[source]:
            matches = [r for r in rows if r["session_id"] == name]
            if len(matches) != 1 or not matches[0]["complete"]:
                raise RuntimeError(f"DEVELOPMENT_INPUT_INCOMPLETE:{name}")
            selected[name] = {**matches[0], "role": "development"}
        holdout = next((r for r in rows if r["complete"] and r["group_key"] not in dev_groups), None)
        if holdout is not None:
            selected[holdout["session_id"]] = {**holdout, "role": "holdout"}
        cohorts[source] = {"root": str(root), "rows": rows,
                           "distinct_group_keys": sorted({r["group_key"] for r in rows}),
                           "holdout_session_id": holdout["session_id"] if holdout else None,
                           "holdout_status": "FROZEN_INDEPENDENT_GROUP" if holdout else "SPLIT_NOT_ESTABLISHED"}

    receipt = load_json(GROUP_RECEIPT)
    if validate_artifact_ref(receipt["artifacts"]["source_groups"]):
        raise RuntimeError("0915_SOURCE_GROUP_PROOF_SHA_DRIFT")
    group_manifest = load_json(Path(receipt["artifacts"]["source_groups"]["path"]))
    known = {r["raw_root"]: r for r in group_manifest["sessions"]}
    rows = []
    for folder in sorted(RAW_0915.glob("*/*"), key=str):
        if not folder.is_dir():
            continue
        raw = folder / "raw"
        required = [raw / name for name in ("vst.h264", "vst.qpc.ts.jsonl", "vst.ts.jsonl",
                                             "camera_params.json", "camera_params.meta.json")]
        proof = known.get(str(folder))
        rows.append({"session_id": proof["session_id"] if proof else None,
                     "raw_path": str(folder), "group": proof.get("source_group") if proof else None,
                     "group_proof": proof.get("independence") if proof else "SPLIT_NOT_ESTABLISHED",
                     "frame_count": proof.get("frame_count") if proof else None,
                     "group_evidence": {
                         "vst_sha256": proof.get("independence_evidence", {}).get("vst_sha256"),
                         "vst_qpc_sha256": proof.get("independence_evidence", {}).get("vst_qpc_sha256"),
                         "vst_qpc_first": proof.get("independence_evidence", {}).get("vst_qpc_first"),
                         "vst_qpc_last": proof.get("independence_evidence", {}).get("vst_qpc_last"),
                     } if proof else None,
                     "complete": all(p.is_file() and p.stat().st_size > 0 for p in required),
                     "missing": [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]})
    dev = next((r for r in rows if r["session_id"] == DEV["0915"][0]), None)
    if not dev or not dev["complete"]:
        raise RuntimeError("DEVELOPMENT_INPUT_INCOMPLETE:play_cards_0915_001")
    selected[dev["session_id"]] = {**dev, "role": "development"}
    holdout = next((r for r in rows if r["complete"] and r["group"] != dev["group"]
                    and r["group_proof"] == "USER_CONFIRMED_AND_METADATA_CORROBORATED"), None)
    if holdout:
        selected[holdout["session_id"]] = {**holdout, "role": "holdout"}
    cohorts["0915"] = {"root": str(RAW_0915), "rows": rows,
                       "source_group_proof": artifact_ref(GROUP_RECEIPT),
                       "source_group_manifest": artifact_ref(Path(receipt["artifacts"]["source_groups"]["path"])),
                       "holdout_session_id": holdout["session_id"] if holdout else None,
                       "holdout_status": "FROZEN_INDEPENDENT_GROUP" if holdout else "SPLIT_NOT_ESTABLISHED"}
    return cohorts, selected


def _ref(path: Path) -> dict[str, Any] | str:
    return artifact_ref(path) if path.is_file() else "ABSENT"


def selected_inputs(selected: dict[str, Any]) -> dict[str, Any]:
    answer = {}
    for name, row in sorted(selected.items()):
        if "0915" in name:
            raw = Path(row["raw_path"]) / "raw"
            files = {key: _ref(raw / key) for key in ("vst.h264", "vst.qpc.ts.jsonl",
                                                    "vst.ts.jsonl", "camera_params.json",
                                                    "camera_params.meta.json")}
        else:
            files = {key: _ref(Path(row[key])) for key in ("video", "stereo", "camera", "slam", "humanego")}
            files["clip_manifest"] = row["clip_manifest"]
        answer[name] = {"role": row["role"],
                        "prior_exposure": "UNKNOWN" if row["role"] == "holdout" else "DEVELOPMENT",
                        "split_claim": "PRE_FROZEN_CROSS_GROUP_NOT_BLIND" if row["role"] == "holdout" else "DEVELOPMENT",
                        "source_path": row["raw_path"],
                        "source_group": row["group"], "files": files,
                        "calibration": files.get("camera", files.get("camera_params.json", "ABSENT"))}
    return answer


def _index_copy() -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in load_json(INDEX).items()
            if k not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [r for r in state["tasks"] if r["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0].get("run_signature") != packet["run_signature"]:
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def heartbeat(packet: dict[str, Any]) -> None:
    _fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(r for r in state["tasks"] if r["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_RAW_T0_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)


def register(cohorts: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    if artifact_ref(PARENT)["sha256"] != PARENT_SHA or load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("VISUAL_PARENT_OR_ROUTE_DRIFT")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if OUT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    prior = load_json(F2)
    models = prior["models"]
    for name, ref in models.items():
        if validate_artifact_ref(ref):
            raise RuntimeError(f"MODEL_SHA_DRIFT:{name}")
    code = [_ref(SCRIPT), _ref(REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_raw_v1.py"),
            _ref(REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v3.py"),
            _ref(REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_raster_v1.py")]
    metadata = {source: [row["clip_manifest"] for row in cohort["rows"]]
                for source, cohort in cohorts.items() if source != "0915"}
    metadata["0915"] = [cohorts["0915"]["source_group_proof"],
                        cohorts["0915"]["source_group_manifest"]]
    inputs = {"plan": artifact_ref(PLAN), "route": artifact_ref(ROUTE),
              "visual_parent": artifact_ref(PARENT), "f2_contract": artifact_ref(F2),
              "cohort_metadata": metadata, "selected_sources": selected_inputs(selected),
              "models": models, "calibration_rule": "same-session-or-ABSENT"}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": code,
        "schema": "exact78-raw-t0-v1", "config": "physical-left-resize-only-no-hand-data"})).hexdigest()
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Freeze Exact78 three-source raw identities, groups, allowed inputs and 0915 frame/time evidence",
        "phase": "EXACT78_RAW_TO_ROBOT_T0", "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T0_V1", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(hours=2)).isoformat(),
        "campaign_deadline_at": (now + timedelta(hours=24)).isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)), str(ROUTE.relative_to(REPO_ROOT)),
                     str(F2.relative_to(REPO_ROOT)), str(GROUP_RECEIPT.relative_to(REPO_ROOT)),
                     str(SCRIPT.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V2_PARENT_SHA_PINNED",
                          "SOURCE_READ_ONLY", "NO_MODEL_RUN", "NO_DELETION", "0916_READ_ONLY"],
        "required_outputs": ["INPUT_FREEZE.json", "FRAME_TIME_MAP.json", "RESULT.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 7200, "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "T0 source and time evidence only; no visual, product, geometry, training or deletion pass",
        "writer": writer, "inputs": inputs, "runtime_code": code, "run_signature": signature,
        "forbidden_production_reads": FORBIDDEN,
        "camera_pose_exception": "Only an explicitly audited non-hand head/camera-pose field from the same acquisition may be extracted from raw/pico.jsonl in a later task; this T0 does not read the file and does not grant whole-file producer access.",
        "commands": ["PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_raw_t0_v1"]}
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    OUT.mkdir(parents=True)
    atomic_json(OUT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "Exact78_three_source", "updated_at": now.isoformat(),
        "heartbeat_at": now.isoformat(), "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": "Exact78_three_source",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one immutable T0 result or two-hour deadline"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE",
        "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T0_V1_RUNNING", plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_RAW_T0_V1_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def _ffprobe(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,time_base,duration",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    frames = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts,pkt_dts,best_effort_timestamp,pict_type,key_frame",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(stream.stdout).get("streams", [])
    return (streams[0] if streams else {}), json.loads(frames.stdout).get("frames", [])


def _ticks(path: Path) -> list[int]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(int(line.strip()))
    return values


def frame_time(selected: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    sessions = {}
    last_heartbeat = time.monotonic()
    for name, row in sorted(selected.items()):
        _fence(packet)
        if time.monotonic() - last_heartbeat >= 55:
            heartbeat(packet)
            last_heartbeat = time.monotonic()
        if "0915" in name:
            raw = Path(row["raw_path"]) / "raw"
            video = raw / "vst.h264"
            qpc, capture_ts = _ticks(raw / "vst.qpc.ts.jsonl"), _ticks(raw / "vst.ts.jsonl")
            camera = load_json(raw / "camera_params.json")
            if camera.get("left", {}).get("sourceIndex") != 1:
                raise RuntimeError(f"PHYSICAL_LEFT_SOURCE_INDEX_DRIFT:{name}")
            width = int(camera["width"])
            eye = 1
            crop = [eye * width, 0, width, int(camera["height"])]
            source = "raw_vst_h264_physical_left_resize_only"
        else:
            video = Path(row["video"])
            qpc, capture_ts, crop = [], [], None
            source = "tracker_session_rgb_clipped_video"
        stream, decoded = _ffprobe(video)
        count = len(decoded)
        expected = row.get("frame_count")
        if expected is not None and count != int(expected):
            raise RuntimeError(f"DECODE_FRAME_COUNT_MISMATCH:{name}:{count}:{expected}")
        if qpc and len(qpc) != count or capture_ts and len(capture_ts) != count:
            raise RuntimeError(f"TIMESTAMP_FRAME_COUNT_MISMATCH:{name}")
        records = []
        for i, decoded_frame in enumerate(decoded):
            records.append({"capture_frame_id": None, "decoded_display_index": i,
                "sbs_physical_eye": "left_sourceIndex_1" if crop else "tracker_session_left_RGB",
                "crop_xywh": crop, "model_input_index": None,
                "qpc_raw_tick": qpc[i] if qpc else None,
                "capture_timestamp_raw": capture_ts[i] if capture_ts else None,
                "pts": decoded_frame.get("pts"),
                "best_effort_timestamp": decoded_frame.get("best_effort_timestamp"),
                "pict_type": decoded_frame.get("pict_type"),
                "output_index": None,
                "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED" if crop else "CLIP_INDEX_ONLY"})
        dup_qpc = len(qpc) - len(set(qpc)) if qpc else 0
        neg_qpc = sum(b <= a for a, b in zip(qpc, qpc[1:]))
        sessions[name] = {"role": row["role"], "source": source,
            "stream": stream, "source_video": artifact_ref(video), "decoded_frames": count,
            "nominal_fps_from_capture_metadata": row.get("fps") if not crop else camera.get("fps"),
            "raw_qpc_count": len(qpc), "raw_capture_timestamp_count": len(capture_ts),
            "duplicate_qpc": dup_qpc, "nonincreasing_qpc": neg_qpc,
            "qpc_unit": "UNVERIFIED", "qpc_frequency": "UNVERIFIED",
            "qpc_sampling_location": "UNVERIFIED", "head_pose_clock_domain": "UNVERIFIED",
            "time_mode": "TIMEBASE_UNVERIFIED" if crop else "TRACKER_CLIP_METADATA_ONLY",
            "output_duration_seconds": None, "output_frame_count": 0,
            "encoded_pts_missing": sum(r["pts"] is None for r in records),
            "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED" if crop else "CLIP_INDEX_ONLY",
            "frame_rows": records}
    return {"schema_version": "exact78-t0-frame-time-map-v1",
            "claim_limit": "Input/display-order mapping only; output columns remain null until fresh model execution. QPC-to-PTS and head-pose synchronization are unverified.",
            "sessions": sessions}


def seal(packet: dict[str, Any], status: str, outputs: dict[str, Path], reason: str | None = None) -> dict[str, Any]:
    _fence(packet)
    if artifact_ref(PARENT)["sha256"] != PARENT_SHA:
        raise RuntimeError("V2_PARENT_RESULT_SHA_DRIFT")
    result_path = OUT / "RESULT.json"
    if result_path.exists() or RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_RESULT_ALREADY_EXISTS")
    result = {"schema_version": "exact78-raw-t0-result-v1", "task_id": TASK,
        "status": status, "execution_complete": status == "PASSED",
        "registered_at": packet["registered_at"],
        "campaign_deadline_at": packet["campaign_deadline_at"],
        "outputs": {name: artifact_ref(path) for name, path in outputs.items()},
        "reason": reason, "models_run": False, "training_run": False,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "general_operation": None,
        "next_action": "T1 register a finite candidate entrypoint/recipe task; resolve source split and timebase blockers locally",
        "claim_limit": packet["claim_limit"]}
    atomic_json(result_path, result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t0-receipt-v1", "task_id": TASK,
        "status": status, "result": artifact_ref(result_path),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(r for r in state["tasks"] if r["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
        heartbeat_at=now_iso(), result=artifact_ref(result_path), terminal_receipt=artifact_ref(RECEIPT),
        quality="INPUT_EVIDENCE_ONLY", adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T0_V1_TERMINAL", plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T0_V1_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(result_path)}


def main() -> int:
    cohorts, selected = inventory()
    packet = register(cohorts, selected)
    try:
        calibration_states = {}
        for name, row in selected.items():
            camera_path = (Path(row["raw_path"]) / "raw/camera_params.json" if "0915" in name
                           else Path(row["camera"]))
            camera = load_json(camera_path)
            left, right = camera.get("left", {}), camera.get("right", {})
            present = all(bool(side.get("extrinsic") or side.get("extrinsics") or side.get("hasExtrinsics"))
                          for side in (left, right))
            calibration_states[name] = {"camera_params": artifact_ref(camera_path),
                "extrinsics": "SOURCE_PRESENT" if present else "ABSENT",
                "metric_geometry": "UNVERIFIED" if present else "BLOCKED_CALIBRATION_ABSENT",
                "visual_robot": "NOT_BLOCKED_BY_STEREO_EXTRINSICS"}
        freeze = {"schema_version": "exact78-raw-t0-input-freeze-v1",
            "selection_rule": "first complete session by original raw path in independent source group outside development groups",
            "cohorts": cohorts, "selected": selected_inputs(selected),
            "calibration_states": calibration_states,
            "development_sessions": DEV, "forbidden_production_reads": FORBIDDEN,
            "camera_pose_exception": packet["camera_pose_exception"],
            "models": {**packet["inputs"]["models"],
                       "foundation_stereo_weights": "UNRESOLVED_NOT_SELECTED"},
            "model_closure": "PARTIAL_DEPTH_MODEL_NOT_SELECTED",
            "runtime_code": packet["runtime_code"],
            "source_group_limits": {source: cohort["holdout_status"] for source, cohort in cohorts.items()},
            "geometry_consumers": calibration_states,
            "quality_claims": {"raw_cold_start": "NOT_RUN", "visual_robot": "NOT_RUN",
                               "clean": "NOT_RUN", "depth": "NOT_RUN", "object6d": "NOT_RUN",
                               "occlusion": "NOT_RUN", "product": "NOT_RUN", "training": "NOT_ELIGIBLE"}}
        atomic_json(OUT / "INPUT_FREEZE.json", freeze)
        heartbeat(packet)
        mapping = frame_time(selected, packet)
        atomic_json(OUT / "FRAME_TIME_MAP.json", mapping)
        result = seal(packet, "PASSED", {"input_freeze": OUT / "INPUT_FREEZE.json",
                                         "frame_time_map": OUT / "FRAME_TIME_MAP.json"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        try:
            outputs = {name: path for name, path in (("input_freeze", OUT / "INPUT_FREEZE.json"),
                ("frame_time_map", OUT / "FRAME_TIME_MAP.json")) if path.exists()}
            print(json.dumps(seal(packet, "FAILED_RUNTIME", outputs,
                                  f"{type(error).__name__}:{error}"), ensure_ascii=False), flush=True)
        except Exception as seal_error:
            print(f"T0_SEAL_FAILED:{type(seal_error).__name__}:{seal_error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
