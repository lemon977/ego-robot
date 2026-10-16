"""Thin finite-task publisher and SHA-bound causal source producer.

Reuses the repository ledger/publisher. It does not create a queue, mutate
original captures, reuse old masks/motion, or launch unregistered algorithms.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from chaoyang.governance.common import (AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT,
    TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json,
    now_iso, publish_bundle, validate_artifact_ref)
from chaoyang.governance.register_single_task_packet import _validate_packet

HARD_DEADLINE = datetime.fromisoformat("2026-10-11T01:46:32+00:00")
PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/checkpoint_delivery_8h_20261011"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"


def _measure_storage(directory, task_id):
    """Release only our O_EXCL-created scratch, never existing project data."""
    probe = directory / "TASK_OWNED_WRITE_READBACK_4G.bin"
    size = 4 * 1024**3
    observed = {"schema": "task-local-write-capacity-observation-v1", "task_id": task_id,
        "probe_path": str(probe), "required_bytes": size, "device_id": directory.stat().st_dev,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "measured_capacity_not_general_free_space_guarantee": True,
        "source_data_or_existing_files_removed": False}
    capacity = os.statvfs(directory)
    observed["statvfs_free_before"] = capacity.f_bavail * capacity.f_frsize
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    identity = os.fstat(descriptor)
    written, digest = 0, hashlib.sha256()
    try:
        chunk = os.urandom(8 * 1024**2)
        while written < size:
            pending = memoryview(chunk)[:min(len(chunk), size-written)]
            while pending:
                count = os.write(descriptor, pending)
                if count <= 0:
                    raise OSError("ZERO_LENGTH_RESOURCE_PROBE_WRITE")
                digest.update(pending[:count]); written += count; pending = pending[count:]
        os.fsync(descriptor)
        stat = os.fstat(descriptor)
        os.close(descriptor); descriptor = None
        readback = hashlib.sha256()
        with probe.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                readback.update(block)
        if stat.st_size != size or digest.hexdigest() != readback.hexdigest():
            raise RuntimeError("RESOURCE_PROBE_READBACK_MISMATCH")
        observed.update(status="PASSED_MEASURED_WRITE_READBACK", written_bytes=written,
            fsynced_bytes=stat.st_size, write_sha256=digest.hexdigest(), read_sha256=readback.hexdigest(),
            reported_allocated_bytes=stat.st_blocks*512)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        current = probe.stat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise RuntimeError("RESOURCE_PROBE_TARGET_REPLACED")
        probe.unlink()
    capacity = os.statvfs(directory)
    observed.update(task_owned_probe_released=True, completed_at=datetime.now(timezone.utc).isoformat(),
                    statvfs_free_after=capacity.f_bavail * capacity.f_frsize)
    return _json_once(directory / "STORAGE_CAPACITY_OBSERVATION.json", observed)


def _lease_update(packet, *, worker_pid=None, release_reason=None):
    from chaoyang.governance.gpu_lease_v71 import heartbeat, release
    from chaoyang.ops.run_gpu_command_with_v71_lease import load
    writer = packet["writer"]
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(GPU_LEASE)
        if (current.get("pid") != writer["pid"]
                or current.get("fencing_token") != writer["fencing_token"]
                or current.get("process_startticks") != writer["proc_start_ticks"]):
            raise RuntimeError("CENTRAL_GPU_LEASE_FENCE_DRIFT")
        if release_reason is not None:
            updated = release(current, pid=writer["pid"],
                fencing_token=writer["fencing_token"], reason=release_reason)
        else:
            updated = heartbeat(current, pid=writer["pid"], fencing_token=writer["fencing_token"])
            # This finite trainer has fixed authorization, not a rolling model
            # callback. Its lease ends at the task deadline, never beyond it.
            updated["expires_at"] = packet["deadline_at"]
            if worker_pid is not None:
                updated["gpu_process_pid"] = worker_pid
        atomic_json(GPU_LEASE, updated)


def _fresh(packet, root):
    from chaoyang.humanego_ict.production_training_gate import authorize_production_prepare
    args = ["--task-packet", packet["packet_path"], "--output-dir", packet["output_dir"]]
    value = authorize_production_prepare(root, Path(packet["packet_path"]), args,
                                        output=Path(packet["output_dir"]))
    return dict(value, cpu_renderer_allowed=True, live_writer_verified=True)


def _json_once(path, value):
    if path.exists():
        raise RuntimeError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def _image_once(path, rgb):
    from PIL import Image
    if path.exists():
        raise RuntimeError("IMMUTABLE_IMAGE_ALREADY_EXISTS")
    Image.fromarray(rgb).save(path, format="PNG")
    return artifact_ref(path)


def _frames(record, anchors):
    """Decode native display order. SBS1 is a recorded visual prior, not calibration."""
    import av
    import cv2
    from chaoyang.humanego_ict.runtime_gate import verify_ref
    ref = record["source_refs"]["raw_vst"]
    source = verify_ref(ref)
    before = source.stat()
    width, height = record["reference_eye_size"]
    selected, count = set(anchors), 0
    with av.open(str(source)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for i, frame in enumerate(container.decode(video=0)):
            count = i + 1
            if (frame.width, frame.height) != (2 * width, height):
                raise RuntimeError("SOURCE_SBS_PIXEL_DIMENSION_MISMATCH")
            if i in selected:
                rgb = frame.to_ndarray(format="rgb24")[:, width:2 * width]
                yield i, cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("RAW_VIDEO_CHANGED_DURING_DECODE")
    if count != record["raw_frame_count"]:
        raise RuntimeError(f"SOURCE_FULL_DECODE_COUNT_MISMATCH:{count}:{record['raw_frame_count']}")


def prepare(packet, root):
    """Actual CPU producer, callable only by the registered prepare operation."""
    import numpy as np
    from chaoyang.humanego_ict.data import ACTION_LAYOUT, TARGET
    from chaoyang.humanego_ict.production_source import (replay_session, prepare_anchor,
        fixed_anchor_ids, aggregate_group_plan)
    from chaoyang.humanego_ict.production_data import (SCHEMA, TRAINING_SCOPE,
        NORMALIZATION, SELECTION_POLICY, validate_production_manifest)
    from chaoyang.humanego_ict.production_renderer import CausalRobotizedProducer
    _fresh(packet, root)
    output, inputs = Path(packet["output_dir"]), packet["inputs"]
    output.mkdir(parents=True, exist_ok=False)
    contract = load_json(Path(inputs["source_contract"]["path"]))
    recipe = load_json(Path(inputs["recipe"]["path"]))
    for ref in (inputs["source_contract"], inputs["recipe"], inputs["resolved_input_manifest"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("FROZEN_INPUT_DRIFT:" + str(errors))
    records = sorted(load_json(Path(inputs["resolved_input_manifest"]["path"]))["sessions"],
                     key=lambda r: r["source_refs"]["source_manifest"]["path"])
    if len(records) != 199:
        raise RuntimeError("ALL_199_ORIGINAL_CAPTURES_REQUIRED")
    mapper_ref, calibration_ref = inputs["legacy_mapper_refs"][:2]
    replays = []
    for record in records:
        _fresh(packet, root)
        replay = replay_session(record, mapper_ref=mapper_ref,
            calibration_ref=calibration_ref, source_contract=contract)
        replays.append(replay)
    plan = aggregate_group_plan([r["metadata"] for r in replays])
    candidate_rows = []
    for replay in replays:
        for frame in fixed_anchor_ids(replay):
            value = prepare_anchor(replay, frame)
            candidate_rows.append({"session_id": replay["record"]["session_id"],
                "source_frame_id": frame, "eligible": value["eligible"],
                "missing_items": value["missing_items"]})
    _json_once(output / "CANDIDATE_ANCHORS.json", candidate_rows)
    _json_once(output / "GROUP_PLAN.json", plan)

    def session_worker(replay):
        sid = replay["record"]["session_id"]
        directory = output / "sessions" / sid
        directory.mkdir(parents=True, exist_ok=False)
        results = {f: prepare_anchor(replay, f) for f in fixed_anchor_ids(replay)}
        needed = [f for f, result in results.items() if result["eligible"]]
        producer = CausalRobotizedProducer(root, recipe=recipe,
                                          runtime_authorizer=lambda: _fresh(packet, root))
        samples = []
        for frame, raw in _frames(replay["record"], needed):
            _fresh(packet, root)
            value = results[frame]
            sample_id = f"pick_place_1008_{sid}_f{frame:06d}"
            prefix = directory / f"f{frame:06d}"
            labels_path = prefix.with_name(prefix.name + "_labels.npz")
            if labels_path.exists():
                raise RuntimeError("IMMUTABLE_LABEL_ALREADY_EXISTS")
            np.savez_compressed(labels_path, **value["labels"])
            labels_ref = artifact_ref(labels_path)
            trace_ref = _json_once(prefix.with_name(prefix.name + "_trace.json"), value["trace"])
            raw_ref = _image_once(prefix.with_name(prefix.name + "_raw.png"), raw)
            current = dict(value["current"], raw_rgb_frame_id=frame,
                raw_rgb_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
            rgb = {"raw": raw_ref}
            causal = {}
            admission = {"raw": True, "robotized": False}
            missing = []
            render_trace_ref = diagnostic_ref = None
            evidence = replay["actual_input_refs"] + [inputs["source_contract"], trace_ref]
            def receipt(branch, image, extra):
                body = {"schema": "causal-training-input-v1", "status": "PASSED",
                    "input_class": "CAUSAL_TRAINING_INPUT", "sample_id": sample_id,
                    "source_frame_id": frame, "branch": branch, "image_sha256": image["sha256"],
                    "labels_sha256": labels_ref["sha256"], "anchor_qpc_ns": int(value["labels"]["anchor_qpc_ns"]),
                    "timestamp_unit": "nanoseconds", "clock_domain": contract["clock_domain"],
                    "donor_frame_ids": [frame], "evidence_refs": evidence + extra,
                    "source_signature": replay["metadata"]["source_signature"], "source_session_id": sid,
                    "time_trust_boundary": contract["clock_trust_boundary"],
                    "physical_eye_identity_verified": False, "geometry_qualification": "UNKNOWN",
                    "control_ground_truth": False, "physical_deployment_authorized": False}
                return _json_once(prefix.with_name(prefix.name + f"_{branch}_causal.json"), body)
            causal["raw"] = receipt("raw", raw_ref, [])
            try:
                rendered = producer.produce_anchor(raw, current)
                robot_ref = _image_once(prefix.with_name(prefix.name + "_robotized.png"), rendered.rgb)
                rgb["robotized"] = robot_ref
                render_trace_ref = _json_once(prefix.with_name(prefix.name + "_render_trace.json"), rendered.input_trace)
                diagnostic_ref = _json_once(prefix.with_name(prefix.name + "_render_diagnostics.json"), rendered.diagnostics)
                causal["robotized"] = receipt("robotized", robot_ref, [raw_ref, render_trace_ref, diagnostic_ref, inputs["recipe"]])
                tracking = rendered.diagnostics["motion"]["per_side_tracking"]
                admission["robotized"] = all(v["numeric_tracking_status"] == "PASSED" for v in tracking)
                if not admission["robotized"]:
                    missing = ["FAILED_ROBOT_NUMERIC_TRACKING_GATE"]
                if frame == needed[0]:
                    _image_once(directory / "review_raw_mask_clean_robot.png", np.concatenate((raw,
                        np.repeat(rendered.mask[..., None] * np.uint8(255), 3, axis=2),
                        rendered.clean, rendered.rgb), axis=1))
            except (ValueError, RuntimeError, OSError) as error:
                missing = [type(error).__name__ + ":" + str(error)]
                _json_once(prefix.with_name(prefix.name + "_render_failure.json"),
                    {"session_id": sid, "source_frame_id": frame, "missing_items": missing,
                     "numerical_status": "FAILED", "geometry": "UNKNOWN"})
            sample = {"sample_id": sample_id, "session_id": sid, "source_frame_id": frame,
                "source_group_id": plan[sid]["source_group_id"], "split": plan[sid]["split"],
                "labels": labels_ref, "derivation_trace_ref": trace_ref, "rgb": rgb,
                "causal_receipts": causal, "branch_admission": admission, "sample_weight": 1.0}
            if missing:
                sample["robotized_missing_items"] = missing
            if render_trace_ref is not None:
                sample["robotized_render_trace_ref"] = render_trace_ref
                sample["robotized_render_diagnostics_ref"] = diagnostic_ref
            samples.append(sample)
        _fresh(packet, root)
        _json_once(directory / "SESSION_RESULT.json", {"session_id": sid, "sample_count": len(samples),
            "robotized_admitted": sum(s["branch_admission"]["robotized"] for s in samples),
            "raw_full_decode": "PASSED", "source_metadata": replay["metadata"], "samples": samples})
        print(json.dumps({"session_id": sid, "raw_anchors": len(samples),
            "robotized_anchors": sum(s["branch_admission"]["robotized"] for s in samples)}), flush=True)
        return samples

    with ThreadPoolExecutor(max_workers=2) as pool:
        all_samples = [sample for chunk in pool.map(session_worker, replays) for sample in chunk]
    _fresh(packet, root)
    pose_clock = {"position_unit": "meters", "pose_frame_id": "REP103_X_FORWARD_Y_LEFT_Z_UP",
        "camera_frame_id": "ANCHOR_PAST_HEAD_FACTORY_CAMERA_PRIOR",
        "tcp_frame_ids": ["LEFT_INSTALLATION_WRIST_PRIOR", "RIGHT_INSTALLATION_WRIST_PRIOR"],
        "timestamp_unit": "nanoseconds", "clock_domain": contract["clock_domain"]}
    manifest = {"schema": SCHEMA, "training_scope": TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "target": TARGET,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "action_layout": ACTION_LAYOUT, "normalization": NORMALIZATION,
        "source_profile": contract["source_profile"], "selection_policy": SELECTION_POLICY,
        "pair_id": packet["run_signature"], "source_contract_ref": inputs["source_contract"],
        "mapper_ref": mapper_ref, "calibration_ref": calibration_ref,
        "pose_clock": pose_clock, "samples": all_samples, "group_plan": plan,
        "candidate_anchors": candidate_rows, "producer_packet": artifact_ref(Path(packet["packet_path"])),
        "rgb_recipe_ref": inputs["recipe"], "physical_eye_identity_verified": False,
        "visual_quality": "UNREVIEWED_CAUSAL_SPATIAL_CANDIDATE_NOT_V9_ORIGINAL_EXECUTION"}
    manifest_ref = _json_once(output / "PRODUCTION_MANIFEST.json", manifest)
    qualifications = {}
    for branch in ("raw", "robotized"):
        try:
            admitted = validate_production_manifest(Path(manifest_ref["path"]), branch)
            qualifications[branch] = {"status": "PASSED_NON_DEPLOYABLE_DERIVED_SOURCE",
                "sample_counts": admitted["sample_counts"], "valid_action_counts": admitted["valid_action_counts"],
                "source_group_splits": admitted["source_group_splits"]}
        except (ValueError, RuntimeError, OSError) as error:
            qualifications[branch] = {"status": "FAILED", "reason": str(error)}
    result = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "task_terminal_status": "PASSED" if qualifications["raw"]["status"].startswith("PASSED") else "FAILED_RUNTIME_FINAL",
        "production_manifest": manifest_ref, "branch_qualification": qualifications,
        "execution_completed": True, "visual_qualified": False, "product_quality_passed": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "raw_sessions": len(records), "candidate_anchors": len(candidate_rows), "raw_eligible_anchors": len(all_samples)}
    _json_once(output / "RESULT.json", result)
    return result


def _publish(state, index, event):
    index["supersedes_index"] = artifact_ref(INDEX)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)


def register_main(argv=None):
    parser = argparse.ArgumentParser(description="One fresh finite production task, no new queue")
    parser.add_argument("--kind", choices=("prepare", "train"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--mapper", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--branch", choices=("raw", "robotized"))
    parser.add_argument("--storage-observation", type=Path, required=True)
    args = parser.parse_args(argv)
    from chaoyang.humanego_ict.production_training_gate import (PREPARE_ROUTE, ROUTE,
        PREPARE, OPERATION, PREPARE_MODE, EXECUTION_MODE, TRAINING_SCOPE)
    from chaoyang.humanego_ict.production_training_cli import make_training_config
    root = REPO_ROOT
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), copy.deepcopy(load_json(INDEX))
    if (state.get("next_task") or any(r["status"] in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for r in state["tasks"])
            or any(r.get("execution_allowed") for r in index["task_packets"])):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    now = datetime.now(timezone.utc)
    deadline = min(HARD_DEADLINE, now + timedelta(seconds=args.wall_seconds))
    if args.attempt < 1 or args.wall_seconds < 1 or deadline <= now:
        raise RuntimeError("FRESH_FINITE_BUDGET_REQUIRED")
    suffix = "prepare" if args.kind == "prepare" else "train_" + str(args.branch)
    task_id = f"pick_place_humanego_ict_production_{suffix}_a{args.attempt:02d}_20261011"
    directory = root / "_run/current" / task_id / "attempts/attempt_0001"
    packet_path = root / "tasks/current" / task_id / "TASK_PACKET.json"
    output = directory / "output"
    if directory.exists() or packet_path.exists():
        raise RuntimeError("FRESH_ATTEMPT_REQUIRED")
    directory.mkdir(parents=True)
    receipt = load_json(RECEIPT_PATH)
    route = load_json(Path(receipt["files"]["algorithm_contract"]["path"]))["special_status_contracts"][PREPARE_ROUTE if args.kind == "prepare" else ROUTE]
    for ref in route["code_closure"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("ROUTE_SHA_DRIFT:" + str(errors))
    stat = Path("/proc/self/stat").read_text()
    writer = {"pid": int(stat.split(" ", 1)[0]), "proc_start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
        "executor_epoch": args.attempt, "fencing_token": secrets.token_hex(24), "scope": task_id}
    inputs = {"plan": artifact_ref(root / PLAN / "EXECUTION_ZH.md"),
              "prior_storage_observation": artifact_ref(args.storage_observation.resolve(strict=True)),
              "storage_capacity_observation": _measure_storage(directory, task_id)}
    command_args = ["--task-packet", str(packet_path), "--output-dir", str(output)]
    operation = PREPARE
    config = None
    if args.kind == "prepare":
        if any(v is None for v in (args.source_contract, args.recipe, args.mapper, args.calibration)):
            raise RuntimeError("EXPLICIT_PREPARATION_INPUTS_REQUIRED")
        contract = load_json(args.source_contract)
        inputs.update(source_contract=artifact_ref(args.source_contract), recipe=artifact_ref(args.recipe),
            resolved_input_manifest=contract["input_manifest_ref"],
            legacy_mapper_refs=[artifact_ref(args.mapper), artifact_ref(args.calibration)])
    else:
        if args.manifest is None or args.branch is None:
            raise RuntimeError("ONE_TRAINING_BRANCH_AND_MANIFEST_REQUIRED")
        inputs.update(production_manifest=artifact_ref(args.manifest), prepared_pair=artifact_ref(args.manifest),
            pretrained=artifact_ref(root / "assets/models/humanego/pretrained_humanego/latest.pt"),
            pretrained_config=artifact_ref(root / "assets/models/humanego/pretrained_humanego/config.json"))
        # Leave 20min export and 5min admission slack, never consume the other branch's allocation.
        seconds = int((deadline - now).total_seconds()) - 1500
        if seconds <= 0:
            raise RuntimeError("TRAINING_PLUS_EXPORT_BUDGET_TOO_SMALL")
        config = make_training_config(seconds, manifest_ref=inputs["production_manifest"],
            pretrained_ref=inputs["pretrained"], pretrained_config_ref=inputs["pretrained_config"])
        operation = OPERATION
        command_args = ["train", "--task-packet", str(packet_path), "--manifest", str(args.manifest.resolve()),
            "--config", str(directory / "TRAINING_CONFIG.json"), "--pretrained", inputs["pretrained"]["path"],
            "--output-dir", str(output), "--branch", args.branch]
    packet_path.parent.mkdir(parents=True)
    if config is not None:
        inputs["training_config"] = _json_once(directory / "TRAINING_CONFIG.json", config)
    budget = {"wall_seconds_max": int((deadline - now).total_seconds()), "gpu_count": 0 if config is None else 1,
        "model_calls": 0 if config is None else 160001, "training_steps": 0 if config is None else 20000,
        "optimizer_updates_max": 0 if config is None else 20000, "cpu_workers_max": 2, "cpu_threads_per_worker": 2}
    import shlex
    command = shlex.join([sys.executable, "-m", "chaoyang.cli", "run", operation, *command_args])
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Produce causally bound paired data" if config is None else "Train and export one genuine ICT checkpoint: " + args.branch,
        "phase": "HUMANEGO_ICT_PRODUCTION_PREPARE" if config is None else "HUMANEGO_ICT_PRODUCTION_TRAIN",
        "plan_revision": "CHECKPOINT_DELIVERY_8H_20261011", "execution_revision": task_id.upper(),
        "execution_mode": PREPARE_MODE if config is None else EXECUTION_MODE, "training_scope": TRAINING_SCOPE,
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json", "docs/governance/ALGORITHM_CONTRACT.json",
            "tasks/current/INDEX.json", str((root / PLAN / "EXECUTION_ZH.md").relative_to(root))],
        "read_set_expansions": ["Only explicit signed input refs and route code/assets closure; original sources read-only"],
        "write_set": [str(directory.relative_to(root)), str(packet_path.relative_to(root)), "tasks/current/INDEX.json",
            "tasks/receipts/" + task_id.upper() + ".json", "docs/governance", PLAN + "/CONTINUE_HERE_ZH.md",
            "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "EXACT_ROUTE_CLOSURE", "RECENT_MEASURED_STORAGE"],
        "required_outputs": ["RESULT.json", "POST_VALIDATION.json"], "budgets": budget,
        "stop_conditions": ["deadline", "input SHA drift", "double writer", "unrecoverable ENOSPC"],
        "claim_limit": "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING; actual steps only; geometry/physical clock implementation/eye qualification UNKNOWN; not robot controls or visual quality approval",
        "inputs": inputs, "writer": writer, "writer_file": str(directory / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(output), "runtime_code": route["code_closure"],
        "commands": [command], "execution_started_at": now.isoformat(), "deadline_at": deadline.isoformat()}
    if config is not None:
        packet.update(branch=args.branch, allocated_gpu_ids=[0], gpu_lease_verified=True,
                      gpu_lease_path=str(GPU_LEASE))
        from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire
        acquired = try_acquire(SimpleNamespace(task_id=task_id, attempt_id="attempt_0001",
            executor_epoch=writer["executor_epoch"], priority="CHECKPOINT_TRAINING", gpu_id=0,
            min_free_mib=61440, lease=GPU_LEASE, lock=GPU_LOCK), writer["fencing_token"])
        if acquired is None:
            raise RuntimeError("WAIT_GPU_RESOURCE_NO_ALGORITHM_STARTED")
        _lease_update(packet)
        _json_once(directory / "GPU_LEASE_ACQUIRED.json", load_json(GPU_LEASE))
    packet["run_signature"] = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": packet["runtime_code"],
        "budgets": budget, "commands": [command], "execution_mode": packet["execution_mode"]})).hexdigest()
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    atomic_json(packet_path, packet)
    atomic_json(directory / "WRITER.json", writer)
    atomic_json(directory / "FROZEN_INPUTS.json", inputs)
    stamp = now_iso()
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": args.attempt,
        "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "gpu_id": None if config is None else 0, "task_packet": artifact_ref(packet_path), "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU" if config is None else "GPU"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(root)),
        "packet_sha256": artifact_ref(packet_path)["sha256"], "execution_allowed": True,
        "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    _publish(state, index, "HUMANEGO_ICT_PRODUCTION_REGISTERED")
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "pid": writer["pid"], "deadline_at": packet["deadline_at"]}), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    for name, sub in (("TMPDIR", "tmp"), ("TORCH_HOME", "torch"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                      ("TRITON_CACHE_DIR", "triton"), ("XDG_CACHE_HOME", "xdg"), ("NUMBA_CACHE_DIR", "numba"),
                      ("CUDA_CACHE_PATH", "nvidia_compute"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                      ("HF_HOME", "huggingface"), ("HUGGINGFACE_HUB_CACHE", "huggingface_hub"),
                      ("MPLCONFIGDIR", "matplotlib")):
        path = output / "runtime_cache" / sub
        # Preparation owns output.mkdir; put its caches beside output, not prematurely inside it.
        if config is None:
            path = directory / "runtime_cache" / sub
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["TMP"] = env["TEMP"] = env["TMPDIR"]
    if config is not None:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    started, last = time.monotonic(), time.monotonic()
    with (directory / "EXECUTION.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, "-m", "chaoyang.cli", "run", operation, *command_args],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if config is not None:
            _lease_update(packet, worker_pid=child.pid)
        print(json.dumps({"child_pid": child.pid, "log": str(directory / "EXECUTION.log")}), flush=True)
        while child.poll() is None:
            if datetime.now(timezone.utc) >= deadline:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                break
            if time.monotonic() - last >= 60:
                if config is not None:
                    _lease_update(packet, worker_pid=child.pid)
                state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
                row = next(r for r in state["tasks"] if r["task_id"] == task_id)
                if row["status"] != "RUNNING" or row["pid"] != writer["pid"] or row["run_signature"] != packet["run_signature"]:
                    child.terminate()
                    raise RuntimeError("WRITER_FENCE_DRIFT")
                row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                _publish(state, index, "HUMANEGO_ICT_PRODUCTION_HEARTBEAT")
                atomic_json(directory / "HEARTBEAT.json", {"task_id": task_id, "child_pid": child.pid,
                    "elapsed_seconds": time.monotonic()-started, "deadline_at": packet["deadline_at"]})
                last = time.monotonic()
            time.sleep(2)
    if config is not None:
        _lease_update(packet, release_reason=f"FINITE_TRAINER_EXIT_{child.returncode}")
    result_files = sorted(output.glob("RESULT*.json")) if output.exists() else []
    complete_checkpoints = []
    for path in sorted(output.glob("checkpoint_*.pt")):
        sidecar = path.with_suffix(path.suffix + ".ref.json")
        if not sidecar.exists():
            continue
        reference = dict(load_json(sidecar), path=str(path.resolve()))
        if not validate_artifact_ref(reference):
            complete_checkpoints.append(reference)
    export_path = output / "INFERENCE_EXPORT.json"
    successful = child.returncode == 0 and bool(result_files)
    if config is not None:
        successful = bool(successful and export_path.exists()
            and any(load_json(path).get("optimizer_updates", 0) > 0 for path in result_files))
    elif result_files:
        successful = successful and load_json(result_files[-1]).get("task_terminal_status") == "PASSED"
    result = {"task_id": task_id, "run_signature": packet["run_signature"], "child_exit_code": child.returncode,
        "elapsed_seconds": time.monotonic()-started, "stage_results": [artifact_ref(p) for p in result_files],
        "inference_export": artifact_ref(export_path) if export_path.exists() else None,
        "complete_checkpoints": complete_checkpoints,
        "task_terminal_status": "PASSED" if successful else "FAILED_RUNTIME_FINAL",
        "control_ground_truth": False, "physical_deployment_authorized": False}
    result_ref = _json_once(directory / "RESULT.json", result)
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    row = next(r for r in state["tasks"] if r["task_id"] == task_id)
    if row["pid"] != writer["pid"] or row["run_signature"] != packet["run_signature"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    terminal = result["task_terminal_status"]
    row.update(status=terminal, pid=None, proc_start_ticks=None, result=result_ref,
               updated_at=now_iso(), heartbeat_at=now_iso())
    state["next_task"] = None
    item = next(r for r in index["task_packets"] if r["task_id"] == task_id)
    item.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    task_receipt = root / "tasks/receipts" / (task_id.upper() + ".json")
    atomic_json(task_receipt, {"task_id": task_id, "status": terminal, "result": result_ref, "claim_limit": packet["claim_limit"]})
    row["receipt"] = artifact_ref(task_receipt)
    _publish(state, index, "HUMANEGO_ICT_PRODUCTION_SEALED")
    validation = subprocess.run([sys.executable, "-m", "chaoyang.cli", "validate-governance"],
        cwd=root, env=env, capture_output=True, text=True)
    _json_once(directory / "POST_VALIDATION.json", {"exit_code": validation.returncode,
        "stdout": validation.stdout, "stderr": validation.stderr})
    print(json.dumps(result), flush=True)
    return 0 if terminal == "PASSED" and validation.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(register_main())
