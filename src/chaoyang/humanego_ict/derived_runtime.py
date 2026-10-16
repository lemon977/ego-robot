"""Governed CPU preparation and offline model admission, with no hardware I/O.

These operations are deliberately separate from the REAL_CONTROL training route.
Importing this module never imports torch, an SDK, IK or a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shlex
import socket
import sys

from .derived_contract import INFER, PREPARE, ROUTE
from .runtime_gate import _is_descendant, bind_cuda_visible_devices, file_ref, verify_ref


def _path(path: Path, root: Path, *, existing: bool = False) -> Path:
    path = path.absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("SYMLINK_CANDIDATE_PATH_FORBIDDEN")
    path = path.resolve(strict=existing)
    if path == root or not path.is_relative_to(root):
        raise ValueError("CANDIDATE_PATH_MUST_BE_IN_PROJECT")
    return path


def _json(ref: dict) -> dict:
    path = verify_ref(ref)
    # Bind the parsed bytes, not just the preceding SHA read.
    content = path.read_bytes()
    import hashlib
    if len(content) != ref["bytes"] or hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError("INPUT_CHANGED_DURING_PARSE")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("CANDIDATE_JSON_OBJECT_REQUIRED")
    return value


def authorize_operation(root: Path, packet_path: Path, operation: str,
                        arguments: list[str], *, request: Path, output: Path) -> dict:
    """Refuse missing/unmatched packets before importing numerical consumers."""
    if operation not in {PREPARE, INFER}:
        raise ValueError("UNKNOWN_DERIVED_OPERATION")
    root = root.resolve(strict=True)
    packet_path = _path(packet_path, root, existing=True)
    if not packet_path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_CANDIDATE_PACKET_REQUIRED")
    output = _path(output, root)
    request_ref = file_ref(request)
    receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text())
    route = _json(receipt["files"]["algorithm_contract"]).get("special_status_contracts", {}).get(ROUTE)
    if (not isinstance(route, dict) or operation not in route.get("candidate_operations", [])
            or route.get("task_packet_required") is not True
            or route.get("control_ground_truth") is not False
            or route.get("physical_deployment_authorized") is not False):
        raise ValueError("DERIVED_CANDIDATE_CONTRACT_NOT_REGISTERED")
    packet_ref = file_ref(packet_path)
    packet = _json(packet_ref)
    mode = ("HUMANEGO_ICT_DERIVED_LABEL_PREPARATION" if operation == PREPARE
            else "HUMANEGO_ICT_REVO2_OFFLINE_INFERENCE")
    if packet.get("execution_mode") != mode:
        raise ValueError("OPERATION_TASK_MODE_MISMATCH")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if deadline.tzinfo is None or deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("CANDIDATE_TASK_EXPIRED")
    index = _json(receipt["files"]["task_packet_index_current"])
    allowed = [r for r in index["task_packets"] if r.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != packet["task_id"]
            or allowed[0]["packet_sha256"] != packet_ref["sha256"]
            or (root / allowed[0]["packet_path"]).resolve() != packet_path):
        raise ValueError("UNIQUE_ACTIVE_CANDIDATE_PACKET_REQUIRED")
    state = _json(receipt["files"]["task_state"])
    active = [r for r in state["tasks"] if r.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    if (len(active) != 1 or active[0]["task_id"] != packet["task_id"]
            or active[0]["status"] != "RUNNING"
            or active[0].get("run_signature") != packet.get("run_signature")):
        raise ValueError("LIVE_CANDIDATE_STATE_REQUIRED")
    writer = packet["writer"]
    if (active[0].get("pid") != writer.get("pid")
            or active[0].get("proc_start_ticks") != writer.get("proc_start_ticks")
            or writer.get("scope") != packet["task_id"]
            or not _is_descendant(writer)):
        raise ValueError("LIVE_CANDIDATE_WRITER_ANCESTRY_REQUIRED")
    if json.loads(_path(Path(packet["writer_file"]), root, existing=True).read_text()) != writer:
        raise ValueError("CANDIDATE_FENCING_TOKEN_DRIFT")
    expected = [operation, *arguments]
    matched = False
    for command in packet.get("commands", []):
        tokens = shlex.split(command)
        for i in range(len(tokens) - 2):
            if tokens[i:i + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[i + 3:] == expected
    if not matched:
        raise ValueError("EXACT_CANDIDATE_COMMAND_NOT_PINNED")
    if packet.get("inputs", {}).get("request") != request_ref:
        raise ValueError("CANDIDATE_REQUEST_SHA_MISMATCH")
    if _path(Path(packet["output_dir"]), root) != output:
        raise ValueError("CANDIDATE_OUTPUT_BINDING_MISMATCH")
    closure = route.get("code_closure", [])
    runtime = packet.get("runtime_code", [])
    if not closure or any(ref not in runtime for ref in closure):
        raise ValueError("CANDIDATE_CODE_CLOSURE_MISMATCH")
    for ref in runtime:
        verify_ref(ref)
    expiry = deadline.timestamp()
    if operation == INFER:
        lease = json.loads(_path(Path(packet["gpu_lease_path"]), root, existing=True).read_text())
        ids = packet.get("allocated_gpu_ids", [])
        budgets = packet.get("budgets", {})
        lease_expiry = datetime.fromisoformat(lease["expires_at"])
        if (packet.get("gpu_lease_verified") is not True
                or not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or i < 0 for i in ids)
                or budgets.get("gpu_count") != len(ids)
                or type(budgets.get("model_calls_max")) is not int or budgets["model_calls_max"] < 1
                or lease.get("status") != "ACQUIRED"
                or lease.get("task_id") != packet["task_id"]
                or lease.get("hostname") != socket.gethostname()
                or lease.get("pid") != writer["pid"]
                or lease.get("process_startticks") != writer["proc_start_ticks"]
                or lease.get("fencing_token") != writer["fencing_token"]
                or lease.get("executor_epoch") != writer["executor_epoch"]
                or lease.get("gpu_id") != ids[0]
                or lease_expiry.tzinfo is None
                or lease_expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc)):
            raise ValueError("FINITE_OFFLINE_MODEL_GPU_LEASE_REQUIRED")
        expiry = min(expiry, lease_expiry.timestamp())
    packet["admitted_authorization_expires_at"] = expiry
    packet["admitted_context"] = {"root": str(root), "packet_path": str(packet_path),
                                  "operation": operation, "arguments": list(arguments),
                                  "request": str(request), "output": str(output)}
    return packet


def write_once(path: Path, value: dict, root: Path) -> None:
    """Candidate result only, never a current baseline or mutable input."""
    path = _path(path, root)
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    write_bytes_once(path, content, root)


def write_bytes_once(path: Path, content: bytes, root: Path) -> dict:
    path = _path(path, root)
    if path.exists():
        if path.read_bytes() == content:
            return file_ref(path)
        raise ValueError("IMMUTABLE_CANDIDATE_OUTPUT_CONFLICT")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return file_ref(path)


def main(operation: str, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    try:
        packet = authorize_operation(root, args.task_packet, operation, argv,
                                     request=args.request, output=args.output_dir)
        existing = args.output_dir / "RESULT.json"
        if existing.exists():
            completed = _json(file_ref(existing))
            if (completed.get("task_id") != packet["task_id"]
                    or completed.get("request") != file_ref(args.request)
                    or completed.get("control_ground_truth") is not False
                    or completed.get("physical_deployment_authorized") is not False):
                raise ValueError("EXISTING_CANDIDATE_RESULT_BINDING_CONFLICT")
            print(json.dumps({"status": "ATTACHED_TO_COMPLETED_CANDIDATE", "result": str(existing),
                              "model_calls_this_invocation": 0, "training_started": False}))
            return 0
        request = _json(file_ref(args.request))
        if operation == PREPARE:
            result = execute_preparation_request(request, packet, root)
        else:
            result = execute_inference_request(request, packet)
        result["request"] = file_ref(args.request)
        if datetime.now(timezone.utc).timestamp() >= packet["admitted_authorization_expires_at"]:
            raise ValueError("CANDIDATE_AUTHORIZATION_EXPIRED_BEFORE_PUBLICATION")
        write_once(args.output_dir / "RESULT.json", result, root)
        print(json.dumps({"status": result["status"], "result": str(args.output_dir / "RESULT.json"),
                          "training_started": False, "physical_deployment_authorized": False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "training_started": False, "physical_deployment_authorized": False}), file=sys.stderr)
        return 2


def _alive(packet: dict) -> None:
    if datetime.now(timezone.utc).timestamp() >= packet["admitted_authorization_expires_at"]:
        raise ValueError("CANDIDATE_AUTHORIZATION_EXPIRED")


def _claim_model_invocation(packet: dict, contract_ref: dict, calls_reserved: int) -> dict:
    """One reservation per attempt, including failures; no budget reset on rerun."""
    context = packet["admitted_context"]
    root = Path(context["root"]).resolve(strict=True)
    path = _path(Path(context["output"]) / "MODEL_INVOCATION_RESERVED.json", root)
    reservation = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                   "request": packet["inputs"]["request"], "inference_contract": contract_ref,
                   "writer": packet["writer"], "model_calls_reserved": calls_reserved,
                   "status": "RESERVED_NOT_SUCCESS", "failure_requires_new_attempt": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write((json.dumps(reservation, sort_keys=True, indent=2) + "\n").encode())
    except FileExistsError as error:
        raise ValueError("MODEL_INVOCATION_ALREADY_RESERVED_NEW_ATTEMPT_REQUIRED") from error
    return file_ref(path)


def execute_preparation_request(request: dict, packet: dict, root: Path) -> dict:
    """Produce source-recomputed H50 labels or audit an already prepared pair."""
    if request.get("schema") != "humanego-ict-derived-preparation-request-v1":
        raise ValueError("INVALID_DERIVED_PREPARATION_REQUEST")
    mode = request.get("mode")
    result = {"task_id": packet["task_id"], "supervision_class": "DERIVED_RETARGET",
              "control_ground_truth": False, "physical_deployment_authorized": False,
              "training_started": False, "training_start_authorized": False}
    if mode == "AUDIT_PAIRED_MANIFEST":
        from .derived_data import validate_derived_preparation_manifest
        manifest = verify_ref(request["paired_manifest"])
        summary = validate_derived_preparation_manifest(manifest)
        _alive(packet)
        return {**result, "status": "DERIVED_PREPARATION_AUDITED", "qualification": summary,
                "paired_manifest": request["paired_manifest"]}
    if mode != "BUILD_LABELS":
        raise ValueError("EXPLICIT_BUILD_OR_AUDIT_PREPARATION_MODE_REQUIRED")
    anchors = request.get("anchor_rows")
    limit = packet.get("budgets", {}).get("label_windows_max")
    if (not isinstance(anchors, list) or not anchors or type(limit) is not int or limit < 1
            or len(anchors) > limit or len(set(anchors)) != len(anchors)
            or any(type(row) is not int or row < 0 for row in anchors)):
        raise ValueError("FINITE_UNIQUE_LABEL_ANCHOR_BUDGET_REQUIRED")
    from .derived_data import recompute_derived_labels
    from .data import VerifiedInputCache
    import numpy as np
    cache = VerifiedInputCache()
    entries = []
    missing = set()
    for anchor in anchors:
        _alive(packet)
        prepared = recompute_derived_labels(request["human_source_sidecar"], request["retarget_contract"],
            anchor, retarget_code_ref=request["retarget_code"], _input_cache=cache)
        missing.update(prepared["missing_items"])
        labels = prepared["labels"]
        encoded = io.BytesIO()
        np.savez(encoded, **labels)
        ref = write_bytes_once(Path(packet["output_dir"]) / f"labels_anchor_{anchor:06d}.npz",
                               encoded.getvalue(), root)
        entries.append({"anchor_row": anchor, "labels": ref,
                        "valid_action_counts": labels["action_valid_mask"].sum(axis=0).tolist(),
                        "conditioning_valid_count": int(labels["ict_mask"].sum()),
                        "missing_items": prepared["missing_items"]})
    _alive(packet)
    return {**result, "status": "DERIVED_LABELS_WRITTEN_NOT_TRAINING_ADMISSION",
            "human_source_sidecar": request["human_source_sidecar"],
            "retarget_contract": request["retarget_contract"], "retarget_code": request["retarget_code"],
            "label_windows": entries, "missing_items": sorted(missing),
            "all_labels_have_shared_branch_identity": True,
            "pixel_and_group_qualification": "NOT_GRANTED_BY_LABEL_PRODUCER"}


def execute_inference_request(request: dict, packet: dict) -> dict:
    """One causal RGB/current-state sample, H50; no implicit hardware adapter."""
    if request.get("schema") != "humanego-ict-revo2-offline-request-v1":
        raise ValueError("INVALID_OFFLINE_INFERENCE_REQUEST")
    from .data import _npz
    import numpy as np
    inputs = _npz(request["current_input_npz"])
    required = {"x_rgb", "wrist_pose_world", "wrist_valid", "reference_camera_to_world",
                "source_qpc_ns", "anchor_qpc_ns", "source_frame_id"}
    if set(inputs) != required:
        raise ValueError("INFERENCE_REQUIRES_CURRENT_STATE_ONLY_NO_FUTURE_LABELS")
    anchor = inputs["anchor_qpc_ns"]
    frame = inputs["source_frame_id"]
    if (anchor.shape != () or frame.shape != () or anchor.dtype.kind not in "iu"
            or frame.dtype.kind not in "iu" or int(anchor) < 0 or int(frame) < 0
            or int(anchor) > np.iinfo(np.int64).max or int(frame) > np.iinfo(np.int64).max):
        raise ValueError("EXPLICIT_INTEGER_NATIVE_FRAME_TIME_REQUIRED")
    receipt = _json(request["input_receipt"])
    required_receipt = {
        "schema": "humanego-ict-causal-inference-input-v1", "status": "PASSED",
        "input_class": "CAUSAL_TRAINING_INPUT", "time_mode": "CAUSAL",
        "input_npz_sha256": request["current_input_npz"]["sha256"],
        "source_frame_id": int(frame), "anchor_qpc_ns": int(anchor),
        "timestamp_unit": "nanoseconds", "image_preprocessing": "RGB_RESIZE_BILINEAR_320_240",
        "position_unit": "meters", "pose_semantics": "DERIVED_RETARGET_ANCHOR",
    }
    if (any(receipt.get(k) != v for k, v in required_receipt.items())
            or receipt.get("fixture_only") is True
            or type(receipt.get("anchor_qpc_ns")) is not int
            or type(receipt.get("source_frame_id")) is not int):
        raise ValueError("CAUSAL_OFFLINE_INFERENCE_INPUT_RECEIPT_REQUIRED")
    for key in ("clock_domain", "pose_frame_id", "camera_frame_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError("EXPLICIT_INFERENCE_FRAME_CLOCK_DECLARATION_REQUIRED")
    if not isinstance(receipt.get("evidence_refs"), list) or not receipt["evidence_refs"]:
        raise ValueError("INFERENCE_INPUT_UNDERLYING_EVIDENCE_REQUIRED")
    for ref in receipt["evidence_refs"]:
        verify_ref(ref)
    sampling = request["sampling"]
    steps, method, seed = sampling.get("steps"), sampling.get("method"), sampling.get("seed")
    if (type(steps) is not int or not 1 <= steps <= 1000 or method not in {"euler", "heun"}
            or type(seed) is not int or seed < 0):
        raise ValueError("FINITE_EXPLICIT_SAMPLING_CONFIG_REQUIRED")
    calls_required = steps * (2 if method == "heun" else 1)
    if calls_required > packet["budgets"]["model_calls_max"]:
        raise ValueError("SAMPLER_EXCEEDS_REGISTERED_MODEL_CALL_BUDGET")
    rgb = inputs["x_rgb"]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    elif rgb.dtype.kind != "f":
        raise ValueError("RGB_REQUIRES_UINT8_OR_FLOAT_INPUT")
    contract_ref = request["inference_contract"]
    calls_used = 0
    device_ids = bind_cuda_visible_devices(packet)
    # Namespace-mapped first allocated GPU is CUDA index 0, not its host ID.
    del device_ids
    from .inference import _conditioning, build_hand_ict, decode_actions, load_candidate_policy

    policy_contract = _json(contract_ref)
    frame_contract = policy_contract.get("target_frame_contract")
    tcp_frames = receipt.get("tcp_frame_ids")
    if (not isinstance(tcp_frames, list) or len(tcp_frames) != 2 or len(set(tcp_frames)) != 2
            or any(not isinstance(frame_id, str) or not frame_id.strip() for frame_id in tcp_frames)
            or frame_contract != {"position_unit": "meters", "tcp_frame_ids": tcp_frames,
                                  "pose_semantics": "DERIVED_RETARGET_ANCHOR",
                                  "action_reference_frame": "REFERENCE_CAMERA"}):
        raise ValueError("INFERENCE_TARGET_FRAME_UNIT_TCP_CONTRACT_MISMATCH")
    frame_declaration = {key: receipt[key] for key in
                         ("clock_domain", "pose_frame_id", "camera_frame_id", "position_unit",
                          "tcp_frame_ids", "pose_semantics")}
    normalization = _json(policy_contract["normalization_ref"])
    current = build_hand_ict(inputs["wrist_pose_world"], inputs["wrist_valid"],
                            inputs["reference_camera_to_world"], normalization=normalization,
                            source_qpc_ns=inputs["source_qpc_ns"], anchor_qpc_ns=int(anchor),
                            frame_declaration=frame_declaration)
    # Validate all conditioning before constructing or loading the real model.
    _conditioning(rgb, current["x_ict"][None], current["ict_mask"][None])

    def verify_authorization(ref, contract):
        _alive(packet)
        context = packet["admitted_context"]
        current_packet = authorize_operation(Path(context["root"]), Path(context["packet_path"]),
            context["operation"], context["arguments"], request=Path(context["request"]),
            output=Path(context["output"]))
        if (current_packet["run_signature"] != packet["run_signature"]
                or current_packet["writer"] != packet["writer"]):
            raise ValueError("OFFLINE_MODEL_TASK_FENCE_CHANGED")
        if ref != contract_ref or contract_ref != file_ref(Path(ref["path"])):
            raise ValueError("OFFLINE_POLICY_CONTRACT_SHA_DRIFT")
        if calls_used >= packet["budgets"]["model_calls_max"]:
            raise ValueError("OFFLINE_MODEL_CALL_BUDGET_EXHAUSTED")
        lease = json.loads(Path(packet["gpu_lease_path"]).read_text())
        if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != packet["task_id"]
                or lease.get("pid") != packet["writer"]["pid"]
                or lease.get("process_startticks") != packet["writer"]["proc_start_ticks"]
                or lease.get("fencing_token") != packet["writer"]["fencing_token"]
                or lease.get("executor_epoch") != packet["writer"]["executor_epoch"]
                or datetime.fromisoformat(lease["expires_at"]).timestamp() <= datetime.now(timezone.utc).timestamp()):
            raise ValueError("OFFLINE_GPU_LEASE_LOST")
        return {"status": "PASSED", "model_execution_allowed": True,
                "contract_sha256": ref["sha256"], "execution_mode": "REAL_MODEL_OFFLINE",
                "gpu_lease_verified": True, "device": "cuda:0"}

    invocation_reservation = _claim_model_invocation(packet, contract_ref, calls_required)
    policy = load_candidate_policy(contract_ref, authorization_verifier=verify_authorization)
    noise = np.random.default_rng(seed).standard_normal((1, 50, 30)).astype(np.float32)

    def bounded_velocity(**kwargs):
        nonlocal calls_used
        result = policy.predict_velocity(**kwargs)
        calls_used += 1
        return result

    from .inference import sample_actions
    actions = sample_actions(bounded_velocity, x_rgb=rgb,
                             x_ict=current["x_ict"][None], ict_mask=current["ict_mask"][None],
                             initial_noise=noise, steps=steps, method=method)
    decoded = decode_actions(actions[0], inputs["reference_camera_to_world"],
                             normalization=policy.normalization, command_policy="REJECT")
    _alive(packet)
    return {"task_id": packet["task_id"], "status": "OFFLINE_PREDICTION_NOT_DEPLOYMENT",
            "inference_contract": contract_ref, "current_input_npz": request["current_input_npz"],
            "input_receipt": request["input_receipt"], "sampling": sampling,
            "model_calls": calls_used, "source_frame_id": int(frame), "anchor_qpc_ns": int(anchor),
            "actions_50x30": actions[0].tolist(),
            "wrist_pose_reference_camera": decoded["wrist_pose_reference_camera"].tolist(),
            "hand_command": decoded["hand_command"].tolist(),
            "hand_command_unit": decoded["hand_command_unit"], "frame_id": decoded["frame_id"],
            "frame_declaration": frame_declaration, "model_invocation_reservation": invocation_reservation,
            "output_time_status": "UNKNOWN_REQUIRES_NATIVE_H50_MAPPING",
            "grasp_status": "UNKNOWN", "object_status": "UNKNOWN_MASKED",
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "training_started": False, "robot_commands_sent": 0,
            "ik_stage": "SEPARATE_CALIBRATED_OFFLINE_BRIDGE_REQUIRED"}
