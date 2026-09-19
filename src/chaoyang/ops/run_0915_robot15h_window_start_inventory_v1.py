#!/usr/bin/env python3
"""Freeze the 0915 Robot15h inventory, source groups, split and W0/W1.

This first node is deliberately CPU/read-only.  It proves routing and input
identity before any model receives GPU time.  It never reads PICO/tracking hand
results and never writes below the source or processed dataset roots.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

from chaoyang.governance.robot15h_task_specs_v1 import (
    WINDOW_CLOCK,
    WINDOW_RUN_ID,
    build_packet,
    frozen_dag,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_window_start_inventory_v1"
PHASE = "ROBOT15H_WINDOW_START_INVENTORY"
DATASET_ROOT = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915")
RAW_ROOT = Path("/mnt/data/egodata/datasets/ego/chips_cards_hands_0915")
OUTPUT = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json"
SPLIT_SALT = "chaoyang_robot_15h_v1_source_split_v1"

W0 = (
    "play_cards_0915_031",
    "play_cards_0915_119",
    "get_potato_chips_0915_007",
    "get_potato_chips_0915_042",
)
W1_ADDITIONS = (
    "play_cards_0915_044",
    "play_cards_0915_106",
    "get_potato_chips_0915_097",
    "get_potato_chips_0915_029",
    "play_cards_0915_054",
    "play_cards_0915_003",
    "get_potato_chips_0915_068",
    "get_potato_chips_0915_056",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def split_for_source_group(source_group: str) -> tuple[str, int]:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{source_group}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "development", bucket
    if bucket < 85:
        return "validation", bucket
    return "final_holdout", bucket


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current task packet differs from frozen Robot15h specification")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("inventory node is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("current packet index does not authorize inventory node")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("current task packet SHA differs from route")
    return packet, packet_path


def heartbeat(status: str = "RUNNING") -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            TASK_ID,
            "--pid",
            str(os.getpid()),
            "--status",
            status,
            "--phase",
            PHASE,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def _session_paths(task: str, number: str) -> tuple[str, str, Path]:
    if task == "playing_cards":
        return "cards_120_0915", f"play_cards_0915_{number}", DATASET_ROOT / "cleaned/playing_cards" / f"play_cards_0915_{number}"
    if task == "potato_chips":
        return "chips_100_0915", f"get_potato_chips_0915_{number}", DATASET_ROOT / "cleaned/potato_chips" / f"get_potato_chips_0915_{number}"
    raise RuntimeError(f"unsupported 0915 task: {task}")


def _qpc_range(path: Path) -> tuple[int, int, int]:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"empty VST QPC timeline: {path}")
    return values[0], values[-1], len(values)


def inventory_sessions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_result_path = DATASET_ROOT / "DATASET_RESULT.json"
    dataset = load_json(dataset_result_path)
    results = dataset.get("results")
    if (
        dataset.get("completed") != 220
        or dataset.get("cleaned") != 220
        or dataset.get("rejected") != 0
        or dataset.get("failed") != 0
        or not isinstance(results, list)
        or len(results) != 220
    ):
        raise RuntimeError("0915 processed dataset is not the frozen 220/220 clean cohort")

    sessions: list[dict[str, Any]] = []
    capture_ids: set[str] = set()
    vst_shas: set[str] = set()
    qpc_shas: set[str] = set()
    camera_captured_at: set[str] = set()
    source_snapshot_byte_mismatches: list[str] = []
    qpc_count_mismatches: list[str] = []
    for index, row in enumerate(results):
        if index and index % 40 == 0:
            heartbeat()
        number = str(row.get("session_id", "")).zfill(3)
        task = str(row.get("task"))
        batch, session_id, processed = _session_paths(task, number)
        raw_session = RAW_ROOT / batch / number
        if not processed.is_dir() or not raw_session.is_dir():
            raise RuntimeError(f"materialized source/processed session missing: {session_id}")
        clip_path = processed / "clip_manifest.json"
        conversion_path = processed / "CONVERSION_RESULT.json"
        clip = load_json(clip_path)
        conversion = load_json(conversion_path)
        tactile_meta_path = raw_session / "raw/tactile.meta.json"
        camera_meta_path = raw_session / "raw/camera_params.meta.json"
        qpc_path = raw_session / "raw/vst.qpc.ts.jsonl"
        tactile_meta = load_json(tactile_meta_path)
        camera_meta = load_json(camera_meta_path)
        capture_id = str(tactile_meta.get("capture_id"))
        if tactile_meta.get("capture_scope") != "tactile_only" or not capture_id:
            raise RuntimeError(f"invalid tactile capture provenance: {session_id}")
        source_snapshot = conversion.get("source_snapshot")
        if not isinstance(source_snapshot, dict) or "raw/vst.h264" not in source_snapshot:
            raise RuntimeError(f"missing conversion source snapshot: {session_id}")
        for relative, item in source_snapshot.items():
            source_file = raw_session / relative
            if not source_file.is_file() or source_file.stat().st_size != int(item.get("bytes", -1)):
                source_snapshot_byte_mismatches.append(f"{session_id}:{relative}")
        vst_sha = str(source_snapshot["raw/vst.h264"].get("sha256"))
        qpc_sha = str(source_snapshot["raw/vst.qpc.ts.jsonl"].get("sha256"))
        qpc_first, qpc_last, qpc_count = _qpc_range(qpc_path)
        source_frames = int(conversion.get("source_video_frame_count", -1))
        if qpc_count != source_frames:
            qpc_count_mismatches.append(session_id)
        timeline = clip.get("timeline_resampling", {})
        segment_counts = timeline.get("segment_counts")
        no_cross_segment_merge = isinstance(segment_counts, dict) and set(segment_counts) == {"0"}
        anomaly = not (
            int(timeline.get("first_source_index", -1)) == 0
            and int(timeline.get("repeated_transitions", -1)) == 0
            and int(timeline.get("skipped_transitions", -1)) == 0
            and int(timeline.get("negative_transitions", -1)) == 0
            and int(timeline.get("max_forward_step", -1)) == 1
        )
        if not no_cross_segment_merge or int(timeline.get("negative_transitions", -1)) != 0:
            raise RuntimeError(f"cross-segment/negative timeline evidence: {session_id}")
        source_group = f"0915:{batch}:{number}:{capture_id}"
        split, bucket = split_for_source_group(source_group)
        if session_id == "play_cards_0915_001":
            split = "development"
        source_stereo_rel = str(clip.get("files", {}).get("source_stereo_video"))
        source_stereo = processed / source_stereo_rel
        if not source_stereo.is_file():
            raise RuntimeError(f"source stereo missing: {session_id}")
        record = {
            "session_id": session_id,
            "task": task,
            "batch": batch,
            "number": number,
            "frame_count": int(clip.get("source_stereo_video", {}).get("frame_count", -1)),
            "processed_root": str(processed),
            "raw_root": str(raw_session),
            "source_stereo": str(source_stereo),
            "source_group": source_group,
            "independence": "USER_CONFIRMED_AND_METADATA_CORROBORATED",
            "independence_evidence": {
                "capture_id": capture_id,
                "capture_scope": "tactile_only",
                "vst_sha256": vst_sha,
                "vst_qpc_sha256": qpc_sha,
                "vst_qpc_first": qpc_first,
                "vst_qpc_last": qpc_last,
                "camera_captured_at_utc": camera_meta.get("captured_at_utc"),
                "source_snapshot_sha256": canonical_sha(source_snapshot),
                "no_cross_segment_merge": no_cross_segment_merge,
            },
            "timeline_resampling": timeline,
            "timeline_anomaly": anomaly,
            "split": split,
            "split_bucket": bucket,
            "split_salt": SPLIT_SALT,
            "algorithm_input_domain": {
                "physical_left_source_index": 1,
                "operation": "CROP_THEN_RESIZE_ONLY",
                "lens_undistortion": False,
                "legacy_mono_allowed": False,
            },
        }
        sessions.append(record)
        if capture_id in capture_ids or vst_sha in vst_shas or qpc_sha in qpc_shas:
            raise RuntimeError(f"source identity collision: {session_id}")
        capture_ids.add(capture_id)
        vst_shas.add(vst_sha)
        qpc_shas.add(qpc_sha)
        captured = str(camera_meta.get("captured_at_utc"))
        if captured in camera_captured_at:
            raise RuntimeError(f"camera capture timestamp collision: {session_id}")
        camera_captured_at.add(captured)

    if source_snapshot_byte_mismatches or qpc_count_mismatches:
        raise RuntimeError(
            "source closure mismatch: "
            f"bytes={source_snapshot_byte_mismatches[:5]} qpc={qpc_count_mismatches[:5]}"
        )
    ordered = sorted(sessions, key=lambda item: int(item["independence_evidence"]["vst_qpc_first"]))
    overlaps = []
    gaps_ns = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = int(current["independence_evidence"]["vst_qpc_first"]) - int(previous["independence_evidence"]["vst_qpc_last"])
        gaps_ns.append(gap)
        if gap <= 0:
            overlaps.append([previous["session_id"], current["session_id"], gap])
    if overlaps:
        raise RuntimeError(f"VST QPC recording intervals overlap: {overlaps[:5]}")

    sessions.sort(key=lambda item: (item["task"], item["number"]))
    counts = {
        "sessions": len(sessions),
        "tasks": dict(Counter(item["task"] for item in sessions)),
        "splits": {
            task: dict(Counter(item["split"] for item in sessions if item["task"] == task))
            for task in ("playing_cards", "potato_chips")
        },
        "timeline_anomalies": sum(bool(item["timeline_anomaly"]) for item in sessions),
        "capture_ids_unique": len(capture_ids),
        "vst_sha256_unique": len(vst_shas),
        "vst_qpc_sha256_unique": len(qpc_shas),
        "camera_capture_timestamps_unique": len(camera_captured_at),
        "qpc_interval_overlaps": len(overlaps),
        "minimum_inter_recording_gap_ns": min(gaps_ns),
    }
    return sessions, counts


def _resource_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lease_path = ROOT / "_run/current/GPU_LEASE.json"
    return {
        "schema_version": "0915-robot15h-resource-snapshot-v1",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "gpu_query_returncode": gpu.returncode,
        "gpus": [line.strip() for line in gpu.stdout.splitlines() if line.strip()],
        "compute_processes": [line.strip() for line in compute.stdout.splitlines() if line.strip()],
        "gpu_lease": load_json(lease_path) if lease_path.is_file() else None,
        "resource_policy": {
            "cpu_lanes_max": 3,
            "coordinator_cpu_reserve_fraction": 0.25,
            "gpu_single_lease": True,
            "do_not_stop_external_processes": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()

    packet, packet_path = validate_route()
    output = args.output_root.resolve()
    receipt_path = args.receipt.resolve()
    if output != OUTPUT.resolve() or receipt_path != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed attempt/receipt namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("executor epoch and fencing token are invalid")
    for path in (output, receipt_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)

    clock_path = ROOT / WINDOW_CLOCK
    authorization_path = ROOT / "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json"
    dataset_result_path = DATASET_ROOT / "DATASET_RESULT.json"
    relocation_path = ROOT / "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json"
    input_refs = [ref(path) for path in (clock_path, authorization_path, dataset_result_path, relocation_path)]
    signature_payload = {
        "schema_version": "0915-robot15h-window-start-inventory-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "processed_root": str(DATASET_ROOT),
        "raw_root": str(RAW_ROOT),
        "source_mutation_allowed": False,
        "0916_consumption_allowed": False,
        "task_packet": ref(packet_path),
        "inputs": input_refs,
        "code": [ref(Path(__file__)), ref(ROOT / "src/chaoyang/governance/robot15h_task_specs_v1.py")],
    }
    signature_sha = canonical_sha(signature_payload)
    atomic_json(output / "RUN_SIGNATURE.json", {**signature_payload, "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-window-start-inventory-writer-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output),
        "run_signature_sha256": signature_sha,
        "task_packet": ref(packet_path),
    })
    heartbeat("CLAIMED")

    sessions, counts = inventory_sessions()
    by_id = {str(item["session_id"]): item for item in sessions}
    if set(W0 + W1_ADDITIONS) - set(by_id):
        raise RuntimeError("frozen W0/W1 session is absent from inventory")
    if any(by_id[item]["timeline_anomaly"] for item in W0 + W1_ADDITIONS):
        raise RuntimeError("frozen W0/W1 contains a timeline anomaly")
    if any(by_id[item]["split"] != "development" for item in W0):
        raise RuntimeError("W0 must contain only pre-registered development sessions")
    expected_w1_splits = {
        "play_cards_0915_044": "validation",
        "play_cards_0915_106": "validation",
        "get_potato_chips_0915_097": "validation",
        "get_potato_chips_0915_029": "validation",
        "play_cards_0915_054": "final_holdout",
        "play_cards_0915_003": "final_holdout",
        "get_potato_chips_0915_068": "final_holdout",
        "get_potato_chips_0915_056": "final_holdout",
    }
    if any(by_id[item]["split"] != expected for item, expected in expected_w1_splits.items()):
        raise RuntimeError("W1 validation/holdout split differs from frozen selection")

    source_manifest = {
        "schema_version": "0915-robot15h-source-group-manifest-v1",
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASS_USER_CONFIRMED_AND_METADATA_CORROBORATED",
        "processed_root": str(DATASET_ROOT),
        "raw_root": str(RAW_ROOT),
        "source_group_formula": "0915:<batch>:<NNN>:<tactile_capture_id>",
        "capture_id_scope": "TACTILE_ONLY_NOT_CAMERA_RECORDING_UUID",
        "independence_basis": [
            "USER_CONFIRMATION_SEPARATELY_RECORDED",
            "UNIQUE_TACTILE_CAPTURE_ID",
            "UNIQUE_VST_SHA256",
            "UNIQUE_VST_QPC_SHA256",
            "NON_OVERLAPPING_VST_QPC_INTERVAL",
            "UNIQUE_CAMERA_CAPTURE_TIMESTAMP",
            "NO_CROSS_SEGMENT_MERGE_EVIDENCE",
        ],
        "forbidden_grouping_fields": ["pairing_session_id", "collector_instance_id", "device_serial"],
        "split_policy": {
            "salt": SPLIT_SALT,
            "function": "uint32(sha256(salt + NUL + source_group)[:8]) mod 100",
            "development": "0..69",
            "validation": "70..84",
            "final_holdout": "85..99",
            "forced_override": {"play_cards_0915_001": "development"},
        },
        "counts": counts,
        "sessions": sessions,
        "claim_limit": (
            "Recording independence is user-confirmed and corroborated by multiple source "
            "metadata signals; no single camera recording UUID exists.  This inventory alone "
            "does not establish algorithm generalization."
        ),
    }
    atomic_json(output / "SOURCE_GROUP_MANIFEST.json", source_manifest)

    clock = load_json(clock_path)
    batch_sessions = []
    for session_id in W0 + W1_ADDITIONS:
        row = by_id[session_id]
        source_stereo = Path(str(row["source_stereo"]))
        item = {
            "session_id": session_id,
            "source_group": row["source_group"],
            "task": row["task"],
            "split": row["split"],
            "frame_count": row["frame_count"],
            "source_stereo": ref(source_stereo) if session_id in W0 else {
                "path": str(source_stereo.resolve()),
                "bytes": source_stereo.stat().st_size,
                "sha256": None,
                "sha_policy": "COMPUTE_BEFORE_SCHEDULING",
            },
            "wave": "W0" if session_id in W0 else "W1",
            "earliest_consumption_at": (
                clock["freeze_algorithm_candidates_at"]
                if row["split"] == "final_holdout"
                else clock["started_at"]
            ),
            "input_domain": row["algorithm_input_domain"],
        }
        batch_sessions.append(item)
    batch_manifest = {
        "schema_version": "0915-robot15h-batch-manifest-v1",
        "window_run_id": WINDOW_RUN_ID,
        "status": "FROZEN",
        "cohort_denominator": 220,
        "w0_target": 4,
        "w1_cumulative_target": 12,
        "expansion_cap": 220,
        "selection_uses_algorithm_results": False,
        "selection_basis": "PRE_REGISTERED_SOURCE_SPLIT_AND_TIMELINE_INTEGRITY",
        "sessions": batch_sessions,
        "final_holdout_policy": "NO_CONSUMPTION_BEFORE_H9_ALGORITHM_FREEZE",
        "play_cards_0915_001_policy": "DEVELOPMENT_ONLY_NOT_W0_NOT_FINAL_HOLDOUT",
        "0916_policy": "FORBIDDEN",
    }
    atomic_json(output / "BATCH_MANIFEST.json", batch_manifest)
    atomic_json(output / "FROZEN_DAG.json", frozen_dag())
    atomic_json(output / "RESOURCE_SNAPSHOT.json", _resource_snapshot())
    atomic_json(output / "WINDOW_CLOCK_REF.json", {
        "schema_version": "0915-robot15h-window-clock-reference-v1",
        "window_run_id": WINDOW_RUN_ID,
        "clock": ref(clock_path),
        "authorization": ref(authorization_path),
    })
    metrics = {
        "schema_version": "0915-robot15h-window-start-inventory-metrics-v1",
        **counts,
        "w0_sessions": len(W0),
        "w0_frames": sum(int(by_id[item]["frame_count"]) for item in W0),
        "w1_cumulative_sessions": len(W0) + len(W1_ADDITIONS),
        "w1_cumulative_frames": sum(int(by_id[item]["frame_count"]) for item in W0 + W1_ADDITIONS),
    }
    atomic_json(output / "METRICS.json", metrics)
    result = {
        "schema_version": "0915-robot15h-window-start-inventory-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "weights": "ABSENT",
        "gpu_used": False,
        "model_execution_performed": False,
        "source_mutated": False,
        "processed_mutated": False,
        "0916_consumed": False,
        "source_group_status": source_manifest["status"],
        "session_count": len(sessions),
        "timeline_anomaly_count": counts["timeline_anomalies"],
        "w0_session_ids": list(W0),
        "w1_cumulative_session_ids": list(W0 + W1_ADDITIONS),
        "artifacts": {
            "source_groups": ref(output / "SOURCE_GROUP_MANIFEST.json"),
            "batch_manifest": ref(output / "BATCH_MANIFEST.json"),
            "frozen_dag": ref(output / "FROZEN_DAG.json"),
            "resource_snapshot": ref(output / "RESOURCE_SNAPSHOT.json"),
            "metrics": ref(output / "METRICS.json"),
        },
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt_path, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-window-start-inventory-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt_path),
        "gpu_used": False,
    })
    heartbeat("RUNNING")
    print(json.dumps({
        "status": "PASSED",
        "run_id": WINDOW_RUN_ID,
        "sessions": len(sessions),
        "timeline_anomalies": counts["timeline_anomalies"],
        "w0": list(W0),
        "w1_cumulative": len(W0) + len(W1_ADDITIONS),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
