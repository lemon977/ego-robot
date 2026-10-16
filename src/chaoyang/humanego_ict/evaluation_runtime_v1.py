"""Finite governed real-model evaluation; raw predictions precede strict decoding.

Importing this module constructs no model and imports no Torch or hardware SDK.
This is a candidate source until separately adopted and registered by its owner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import shlex
import socket
import sys

from chaoyang.governance.common import REPO_ROOT
from chaoyang.humanego_ict.derived_runtime import _json, _path, write_bytes_once, write_once
from chaoyang.humanego_ict.runtime_gate import _is_descendant, bind_cuda_visible_devices, file_ref, verify_ref

OPERATION = "infer_pick_place_humanego_ict_revo2_evaluation_v1"
MODE = "HUMANEGO_ICT_REVO2_OFFLINE_EVALUATION"
ROUTE = "pick_place_humanego_ict_revo2_evaluation_v1"
REQUEST_SCHEMA = "humanego-ict-revo2-offline-evaluation-batch-request-v1"
SOURCE_POLICY = "FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY"
CURRENT_KEYS = {"x_rgb", "wrist_pose_world", "wrist_valid", "reference_camera_to_world",
                "source_qpc_ns", "anchor_qpc_ns", "source_frame_id"}


def _project_ref(ref, root):
    _declared_ref(ref)
    root = Path(root).resolve(strict=True)
    path = Path(ref["path"])
    if path == root or not path.is_relative_to(root):
        raise ValueError("PROJECT_ONLY_FILE_REF_REQUIRED")
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("SYMLINK_CANDIDATE_PATH_FORBIDDEN")
    _path(path, root, existing=True)
    return ref


def _declared_ref(ref):
    """Validate external provenance declarations without stat/read/resolve."""
    if (not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}
            or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
            or ".." in Path(ref["path"]).parts
            or type(ref["bytes"]) is not int or ref["bytes"] < 0
            or not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("EXACT_PROJECT_FILE_REF_REQUIRED")
    return ref


def authorize_operation(root, packet_path, arguments, *, request, output):
    """Real receipt/index/state/fence/lease checks, not a projected old packet."""
    root = Path(root).resolve(strict=True)
    packet_path = _path(Path(packet_path), root, existing=True)
    request = _path(Path(request), root, existing=True)
    output = _path(Path(output), root)
    if not packet_path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_EVALUATION_PACKET_REQUIRED")
    receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text())
    route = _json(receipt["files"]["algorithm_contract"])["special_status_contracts"].get(ROUTE)
    if (not isinstance(route, dict) or OPERATION not in route.get("candidate_operations", [])
            or route.get("task_packet_required") is not True
            or route.get("control_ground_truth") is not False
            or route.get("physical_deployment_authorized") is not False):
        raise ValueError("EVALUATION_OPERATION_NOT_REGISTERED")
    packet_ref = file_ref(packet_path)
    packet = _json(packet_ref)
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (packet.get("execution_mode") != MODE or deadline.tzinfo is None
            or deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc)
            or packet.get("control_ground_truth") is not False
            or packet.get("physical_deployment_authorized") is not False
            or packet.get("source_policy") != SOURCE_POLICY):
        raise ValueError("FINITE_EVALUATION_TASK_MODE_OR_SCOPE_REQUIRED")
    index = _json(receipt["files"]["task_packet_index_current"])
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != packet["task_id"]
            or allowed[0]["packet_sha256"] != packet_ref["sha256"]
            or (root / allowed[0]["packet_path"]).resolve() != packet_path):
        raise ValueError("UNIQUE_ACTIVE_EVALUATION_PACKET_REQUIRED")
    state = _json(receipt["files"]["task_state"])
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    writer = packet["writer"]
    if (len(active) != 1 or active[0]["task_id"] != packet["task_id"]
            or active[0]["status"] != "RUNNING"
            or active[0].get("run_signature") != packet.get("run_signature")
            or active[0].get("pid") != writer.get("pid")
            or active[0].get("proc_start_ticks") != writer.get("proc_start_ticks")
            or writer.get("scope") != packet["task_id"] or not _is_descendant(writer)):
        raise ValueError("LIVE_EVALUATION_WRITER_ANCESTRY_REQUIRED")
    if json.loads(_path(Path(packet["writer_file"]), root, existing=True).read_text()) != writer:
        raise ValueError("EVALUATION_FENCING_TOKEN_DRIFT")
    matched = False
    for command in packet.get("commands", []):
        tokens = shlex.split(command)
        for i in range(len(tokens) - 2):
            if tokens[i:i + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[i + 3:] == [OPERATION, *arguments]
    if not matched:
        raise ValueError("EXACT_EVALUATION_COMMAND_NOT_PINNED")
    if packet.get("inputs", {}).get("request") != file_ref(Path(request)):
        raise ValueError("EVALUATION_REQUEST_SHA_MISMATCH")
    if _path(Path(packet["output_dir"]), root) != output:
        raise ValueError("EVALUATION_OUTPUT_BINDING_MISMATCH")
    closure = route.get("code_closure", [])
    runtime = packet.get("runtime_code", [])
    if not closure or any(ref not in runtime for ref in closure):
        raise ValueError("EVALUATION_CODE_CLOSURE_MISMATCH")
    for ref in runtime:
        _project_ref(ref, root)
        verify_ref(ref)
    ids = packet.get("allocated_gpu_ids", [])
    budgets = packet.get("budgets", {})
    lease = json.loads(_path(Path(packet["gpu_lease_path"]), root, existing=True).read_text())
    lease_expiry = datetime.fromisoformat(lease["expires_at"])
    if (packet.get("gpu_lease_verified") is not True or len(ids) != 1
            or type(ids[0]) is not int or ids[0] < 0 or budgets.get("gpu_count") != 1
            or type(budgets.get("model_calls_max")) is not int or not 1 <= budgets["model_calls_max"] <= 1140
            or type(budgets.get("samples_max")) is not int or not 1 <= budgets["samples_max"] <= 57
            or type(budgets.get("wall_seconds_max")) is not int or not 1 <= budgets["wall_seconds_max"] <= 7200
            or type(budgets.get("cpu_threads_per_worker")) is not int or not 1 <= budgets["cpu_threads_per_worker"] <= 2
            or budgets.get("optimizer_updates") != 0 or budgets.get("hardware_or_network_calls") != 0
            or lease.get("status") != "ACQUIRED" or lease.get("task_id") != packet["task_id"]
            or lease.get("hostname") != socket.gethostname() or lease.get("pid") != writer["pid"]
            or lease.get("process_startticks") != writer["proc_start_ticks"]
            or lease.get("fencing_token") != writer["fencing_token"]
            or lease.get("executor_epoch") != writer["executor_epoch"]
            or lease.get("gpu_id") != ids[0] or lease_expiry.tzinfo is None
            or lease_expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc)):
        raise ValueError("FINITE_EVALUATION_GPU_LEASE_REQUIRED")
    packet["admitted_authorization_expires_at"] = min(deadline.timestamp(), lease_expiry.timestamp())
    packet["admitted_context"] = {"root": str(root), "packet_path": str(packet_path),
        "arguments": list(arguments), "request": str(request), "output": str(output)}
    return packet


def validate_request(request, packet):
    required = {"schema", "branch", "source_policy", "inference_contract", "samples", "sampling"}
    if (set(request) != required or request["schema"] != REQUEST_SCHEMA
            or request["branch"] not in {"raw", "robotized"}
            or request["branch"] != packet.get("branch") or request["source_policy"] != SOURCE_POLICY):
        raise ValueError("EXPLICIT_SINGLE_BRANCH_BATCH_REQUEST_REQUIRED")
    if request["sampling"] != {"method": "euler", "steps": 20, "seed": 7}:
        raise ValueError("FROZEN_EULER20_SEED7_REQUIRED")
    samples = request["samples"]
    if (not isinstance(samples, list) or not 1 <= len(samples) <= packet["budgets"]["samples_max"]
            or 20 * len(samples) > packet["budgets"]["model_calls_max"]):
        raise ValueError("FINITE_SAMPLE_AND_MODEL_CALL_BUDGET_REQUIRED")
    ids = []
    for sample in samples:
        if (set(sample) != {"sample_id", "group_id", "current_input_npz", "input_receipt"}
                or type(sample["sample_id"]) is not str or not sample["sample_id"]
                or type(sample["group_id"]) is not str or not sample["group_id"]):
            raise ValueError("EXPLICIT_SAMPLE_AND_GROUP_ID_REQUIRED")
        ids.append(sample["sample_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("DUPLICATE_SAMPLE_FORBIDDEN")
    root = Path(packet["admitted_context"]["root"])
    contract = _json(_project_ref(request["inference_contract"], root))
    for key in ("checkpoint_ref", "normalization_ref", "model_config_ref"):
        _project_ref(contract[key], root)
    provenance = contract.get("training_provenance", {})
    if (provenance.get("branch") != request["branch"] or provenance.get("optimizer_updates") != 20000
            or provenance.get("source_policy") != SOURCE_POLICY or contract.get("fixture_only") is not False
            or contract.get("state_dict_key") != "model" or contract.get("control_ground_truth") is not False
            or contract.get("physical_deployment_authorized") is not False):
        raise ValueError("ACTUAL20K_PUREMODEL_SAME_BRANCH_PROVENANCE_REQUIRED")
    return contract


def prepare_current(sample, contract, *, root=REPO_ROOT, branch=None):
    """Only the original seven current-state fields reach conditioning/model."""
    import numpy as np
    from chaoyang.humanego_ict.data import _npz
    from chaoyang.humanego_ict.inference import _conditioning, build_hand_ict
    inputs = _npz(_project_ref(sample["current_input_npz"], root))
    if set(inputs) != CURRENT_KEYS:
        raise ValueError("CURRENT_STATE_ONLY_NO_FUTURE_LABELS_REQUIRED")
    anchor, frame = inputs["anchor_qpc_ns"], inputs["source_frame_id"]
    if (anchor.shape != () or frame.shape != () or anchor.dtype.kind not in "iu"
            or frame.dtype.kind not in "iu" or not 0 <= int(anchor) <= np.iinfo(np.int64).max
            or not 0 <= int(frame) <= np.iinfo(np.int64).max):
        raise ValueError("EXPLICIT_INTEGER_NATIVE_FRAME_TIME_REQUIRED")
    receipt = _json(_project_ref(sample["input_receipt"], root))
    expected = {"schema": "humanego-ict-causal-inference-input-v1", "status": "PASSED",
        "input_class": "CAUSAL_TRAINING_INPUT", "time_mode": "CAUSAL",
        "input_npz_sha256": sample["current_input_npz"]["sha256"], "source_frame_id": int(frame),
        "anchor_qpc_ns": int(anchor), "timestamp_unit": "nanoseconds",
        "image_preprocessing": "RGB_RESIZE_BILINEAR_320_240", "position_unit": "meters",
        "pose_semantics": "DERIVED_RETARGET_ANCHOR"}
    if (any(receipt.get(k) != v for k, v in expected.items()) or receipt.get("fixture_only") is True
            or type(receipt.get("anchor_qpc_ns")) is not int or type(receipt.get("source_frame_id")) is not int):
        raise ValueError("CAUSAL_INFERENCE_INPUT_RECEIPT_REQUIRED")
    if (receipt.get("branch") != branch or receipt.get("sample_id") != sample["sample_id"]
            or receipt.get("source_group_id") != sample["group_id"]
            or receipt.get("control_ground_truth") is not False
            or receipt.get("physical_deployment_authorized") is not False):
        raise ValueError("EXACT_SAME_BRANCH_SAMPLE_GROUP_RECEIPT_REQUIRED")
    manifest_ref = _project_ref(receipt["production_manifest_ref"], root)
    if manifest_ref != contract["training_provenance"]["production_manifest_ref"]:
        raise ValueError("EXACT_FROZEN_PRODUCTION_MANIFEST_REQUIRED")
    manifest = _json(manifest_ref)
    matches = [s for s in manifest["samples"] if s.get("sample_id") == sample["sample_id"]]
    if (manifest.get("schema") != "pick-place-humanego-ict-production-paired-v1"
            or manifest.get("control_ground_truth") is not False
            or manifest.get("physical_deployment_authorized") is not False
            or len(matches) != 1):
        raise ValueError("FROZEN_PRODUCTION_SAMPLE_REQUIRED")
    source = matches[0]
    if (source.get("split") != "validation" or source.get("source_group_id") != sample["group_id"]
            or source.get("source_frame_id") != int(frame)
            or any(source.get("branch_admission", {}).get(b) is not True for b in ("raw", "robotized"))
            or source["labels"] != receipt["source_labels_ref"]
            or source["rgb"][branch] != receipt["source_rgb_ref"]
            or source["causal_receipts"][branch] != receipt["source_rgb_causal_receipt_ref"]):
        raise ValueError("EXACT_COMMON_VALIDATION_BRANCH_SOURCE_BINDINGS_REQUIRED")
    for key in ("source_labels_ref", "source_rgb_ref", "source_rgb_causal_receipt_ref"):
        verify_ref(_project_ref(receipt[key], root))
    for key in ("clock_domain", "pose_frame_id", "camera_frame_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError("EXPLICIT_INFERENCE_FRAME_CLOCK_REQUIRED")
    if not isinstance(receipt.get("evidence_refs"), list) or not receipt["evidence_refs"]:
        raise ValueError("INFERENCE_INPUT_EVIDENCE_REQUIRED")
    external = []
    for ref in receipt["evidence_refs"]:
        _declared_ref(ref)
        # Lexical containment precedes all filesystem operations: NAS is never
        # stat'ed or read in frozen-snapshot research evaluation.
        if Path(ref["path"]).is_relative_to(Path(root)):
            verify_ref(_project_ref(ref, root))
        else:
            external.append(ref)
    tcp = receipt.get("tcp_frame_ids")
    if (not isinstance(tcp, list) or len(tcp) != 2 or len(set(tcp)) != 2
            or any(not isinstance(v, str) or not v.strip() for v in tcp)
            or contract.get("target_frame_contract") != {"position_unit": "meters", "tcp_frame_ids": tcp,
                "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}):
        raise ValueError("INFERENCE_TARGET_FRAME_TCP_MISMATCH")
    declaration = {k: receipt[k] for k in ("clock_domain", "pose_frame_id", "camera_frame_id",
                                            "position_unit", "tcp_frame_ids", "pose_semantics")}
    norm = _json(_project_ref(contract["normalization_ref"], root))
    current = build_hand_ict(inputs["wrist_pose_world"], inputs["wrist_valid"],
        inputs["reference_camera_to_world"], normalization=norm, source_qpc_ns=inputs["source_qpc_ns"],
        anchor_qpc_ns=int(anchor), frame_declaration=declaration)
    rgb = inputs["x_rgb"]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    elif rgb.dtype.kind != "f":
        raise ValueError("RGB_UINT8_OR_FLOAT_REQUIRED")
    _conditioning(rgb, current["x_ict"][None], current["ict_mask"][None])
    return {"rgb": rgb, "current": current, "camera": inputs["reference_camera_to_world"],
            "frame": int(frame), "anchor": int(anchor), "frame_declaration": declaration, "normalization": norm,
            "unreplayed_external_source_refs": external}


def diagnose_actions(actions, camera, normalization):
    """REJECT remains the only qualification; no clipping or implicit repair."""
    import numpy as np
    from chaoyang.humanego_ict.inference import decode_actions
    a = np.asarray(actions)
    if a.shape != (50, 30) or not np.isfinite(a).all():
        raise ValueError("FINITE_RAW_H50_ACTION30_REQUIRED")
    hand = a[:, 18:30]
    diagnostics = {"hand_min": float(hand.min()), "hand_max": float(hand.max()),
                   "hand_out_of_range_values": int(((hand < 0) | (hand > 1)).sum()),
                   "command_policy": "REJECT", "clipping_applied": False}
    try:
        decoded = decode_actions(a, camera, normalization=normalization, command_policy="REJECT")
    except ValueError as error:
        return {**diagnostics, "strict_decode_status": "FAILED_STRICT_REJECT", "failure_codes": [str(error)],
                "qualified_for_assumed_bridge": False}
    return {**diagnostics, "strict_decode_status": "PASSED_STRICT_DECODE_ONLY", "failure_codes": [],
            "qualified_for_assumed_bridge": True,
            "wrist_pose_reference_camera": decoded["wrist_pose_reference_camera"].tolist(),
            "hand_command": decoded["hand_command"].tolist()}


def execute_request(request, packet):
    import numpy as np
    from chaoyang.humanego_ict.inference import load_candidate_policy, sample_actions
    contract = validate_request(request, packet)
    context = packet["admitted_context"]
    root, output = Path(context["root"]), Path(context["output"])
    # Validate every sample before the first real model construction or load.
    prepared = [prepare_current(sample, contract, root=root, branch=request["branch"])
                for sample in request["samples"]]
    calls_used = 0
    contract_ref = request["inference_contract"]

    def verify_authorization(ref, policy_contract):
        current_packet = authorize_operation(root, Path(context["packet_path"]), context["arguments"],
            request=Path(context["request"]), output=output)
        if (current_packet["writer"] != packet["writer"]
                or current_packet["run_signature"] != packet["run_signature"]
                or ref != contract_ref or policy_contract != contract
                or ref != file_ref(Path(ref["path"]))):
            raise ValueError("EVALUATION_TASK_OR_POLICY_FENCE_DRIFT")
        if calls_used >= packet["budgets"]["model_calls_max"]:
            raise ValueError("EVALUATION_MODEL_CALL_BUDGET_EXHAUSTED")
        return {"status": "PASSED", "model_execution_allowed": True, "contract_sha256": ref["sha256"],
                "execution_mode": "REAL_MODEL_OFFLINE", "gpu_lease_verified": True, "device": "cuda:0"}

    bind_cuda_visible_devices(packet)
    reservation_path = _path(output / "MODEL_INVOCATION_RESERVED.json", root)
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    reservation = {"task_id": packet["task_id"], "writer": packet["writer"],
        "run_signature": packet["run_signature"], "request": packet["inputs"]["request"],
        "inference_contract": contract_ref, "model_calls_reserved": 20 * len(prepared),
        "status": "RESERVED_NOT_SUCCESS", "failure_requires_new_attempt": True}
    try:
        with reservation_path.open("xb") as stream:
            stream.write((json.dumps(reservation, sort_keys=True, indent=2) + "\n").encode())
    except FileExistsError as error:
        raise ValueError("MODEL_INVOCATION_ALREADY_RESERVED_NEW_ATTEMPT_REQUIRED") from error
    policy = load_candidate_policy(contract_ref, authorization_verifier=verify_authorization)

    def velocity(**kwargs):
        nonlocal calls_used
        # The policy checks authorization before its forward; the attempt is
        # counted even if that forward raises, and never reset on a rerun.
        try:
            return policy.predict_velocity(**kwargs)
        finally:
            calls_used += 1

    results = []
    for index, (sample, current) in enumerate(zip(request["samples"], prepared)):
        for key in ("current_input_npz", "input_receipt"):
            verify_ref(_project_ref(sample[key], root))
        # Identical per-sample seed/noise across separately registered branches.
        noise = np.random.default_rng(7).standard_normal((1, 50, 30)).astype(np.float32)
        actions = sample_actions(velocity, x_rgb=current["rgb"], x_ict=current["current"]["x_ict"][None],
            ict_mask=current["current"]["ict_mask"][None], initial_noise=noise, steps=20, method="euler")[0]
        encoded = io.BytesIO()
        np.savez(encoded, actions_50x30=actions)
        raw_ref = write_bytes_once(output / f"sample_{index:03d}_raw_actions.npz", encoded.getvalue(), root)
        for key in ("current_input_npz", "input_receipt"):
            verify_ref(_project_ref(sample[key], root))
        # Never lose raw predictions because a range/rot6 strict decode fails.
        diagnostic = diagnose_actions(actions, current["camera"], current["normalization"])
        result = {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
            "current_input_npz": sample["current_input_npz"], "input_receipt": sample["input_receipt"],
            "source_frame_id": current["frame"], "anchor_qpc_ns": current["anchor"],
            "frame_declaration": current["frame_declaration"], "raw_actions_ref": raw_ref,
            "unreplayed_external_source_refs": current["unreplayed_external_source_refs"],
            "source_replay_status": "SOURCE_REPLAY_NOT_PERFORMED",
            "model_calls": 20, **diagnostic}
        write_once(output / f"sample_{index:03d}_RESULT.json", result, root)
        results.append(result)
    # Publication is also bounded by the actual current Task and lease.
    authorize_operation(root, Path(context["packet_path"]), context["arguments"],
                        request=Path(context["request"]), output=output)
    return {"schema": "humanego-ict-revo2-offline-evaluation-result-v1", "task_id": packet["task_id"],
        "status": "COMPLETED_REAL_MODEL_OFFLINE_DIAGNOSTICS_NOT_DEPLOYMENT", "branch": request["branch"],
        "source_policy": SOURCE_POLICY, "inference_contract": contract_ref,
        "request": packet["inputs"]["request"], "run_signature": packet["run_signature"],
        "sampling": request["sampling"], "samples": results, "samples_completed": len(results),
        "groups_completed": len({s["group_id"] for s in results}), "model_calls": calls_used,
        "strict_decode_passed": sum(s["qualified_for_assumed_bridge"] for s in results),
        "strict_decode_failed": sum(not s["qualified_for_assumed_bridge"] for s in results),
        "model_invocation_reservation": file_ref(reservation_path), "future_labels_provided_to_model": False,
        "clipping_applied": False, "output_time_status": "UNKNOWN_REQUIRES_NATIVE_H50_MAPPING",
        "source_replay_status": "SOURCE_REPLAY_NOT_PERFORMED", "current_nas_equivalence_claimed": False,
        "grasp_status": "UNKNOWN", "object_status": "UNKNOWN_MASKED",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "optimizer_updates": 0, "robot_commands_sent": 0, "hardware_or_network_calls": 0}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        packet = authorize_operation(REPO_ROOT, args.task_packet, argv, request=args.request, output=args.output_dir)
        if (args.output_dir / "RESULT.json").exists():
            raise ValueError("IMMUTABLE_COMPLETED_EVALUATION_NO_REEXECUTION")
        result = execute_request(_json(file_ref(args.request)), packet)
        write_once(args.output_dir / "RESULT.json", result, Path(REPO_ROOT))
        print(json.dumps({"status": result["status"], "model_calls": result["model_calls"],
                          "strict_decode_failed": result["strict_decode_failed"]}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "failure_code": str(error),
                          "physical_deployment_authorized": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
