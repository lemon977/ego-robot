"""Finite current-only input publisher; no model, hardware, or automatic run.

The real API needs a registered CPU task and a renewing authorization callback.
The frozen five-to-seven-field adapter is imported only after admission. Source
label NPZ files may contain future members: only the five current members are
decoded, and no source label archive is copied into the model request.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

OPERATION = "prepare_pick_place_humanego_ict_current_inputs_v1"
ROUTE = "humanego_ict_current_inputs_v1"
EXECUTION_MODE = "HUMANEGO_ICT_CURRENT_INPUT_PRODUCTION_CPU"
SPEC_SCHEMA = "humanego-ict-inference-evaluation-preparation-v1"
MANIFEST_SCHEMA = "pick-place-humanego-ict-production-paired-v1"
OUTPUT_SCHEMA = "humanego-ict-revo2-batch-current-inputs-v1"
BUILDER_SHA256 = "4413542df6f70aed16e6e1f9b49319d6a695fce22ba9f38ddd0bbedae694332e"
CURRENT_KEYS = frozenset({"x_rgb", "wrist_pose_world", "wrist_valid",
    "reference_camera_to_world", "source_qpc_ns", "anchor_qpc_ns", "source_frame_id"})
TRUST_KEYS = ("physical_eye_identity_verified", "geometry_qualification", "time_trust_boundary")


class CurrentProducerError(ValueError):
    pass


def _need(condition, code):
    if not condition:
        raise CurrentProducerError(code)


def _project_path(path, root, *, existing=False):
    path = Path(path)
    _need(path.is_absolute(), "ABSOLUTE_PROJECT_PATH_REQUIRED")
    for component in (path, *path.parents):
        _need(not component.is_symlink(), "SYMLINK_PATH_FORBIDDEN")
    resolved = path.resolve(strict=existing)
    _need(resolved != root and resolved.is_relative_to(root), "PROJECT_SUBPATH_REQUIRED")
    return resolved


def _reference_shape(ref):
    _need(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "EXACT_FILE_REFERENCE_REQUIRED")
    _need(isinstance(ref["path"], str) and Path(ref["path"]).is_absolute(), "ABSOLUTE_REFERENCE_REQUIRED")
    _need(type(ref["bytes"]) is int and ref["bytes"] >= 0
          and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
          "SHA_AND_BYTE_COUNT_REQUIRED")


def _read(ref, root):
    _reference_shape(ref)
    path = _project_path(ref["path"], root, existing=True)
    blob = path.read_bytes()
    _need(len(blob) == ref["bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"],
          "SOURCE_SHA_OR_SIZE_DRIFT:" + str(path))
    return blob


def _json_bytes(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    value = json.loads(blob, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(CurrentProducerError("NONFINITE_JSON")))
    _need(isinstance(value, dict), "JSON_OBJECT_REQUIRED")
    return value


def _json_ref(ref, root):
    return _json_bytes(_read(ref, root))


def _file_ref(path):
    blob = path.read_bytes()
    return {"path": str(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def _write(path, blob, root):
    path = _project_path(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    ref = _file_ref(path)
    _need(ref["bytes"] == len(blob) and ref["sha256"] == hashlib.sha256(blob).hexdigest(),
          "OUTPUT_WRITE_READBACK_FAILED")
    return ref


def _write_json(path, value, root):
    blob = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return _write(path, blob, root)


def _real_authorization(verifier, request, root):
    _need(callable(verifier), "REGISTERED_CPU_AUTHORIZER_REQUIRED_BEFORE_NUMERICAL_IMPORT")
    authority = verifier(dict(request))
    _need(isinstance(authority, dict) and authority.get("status") == "PASSED"
          and authority.get("execution_mode") == EXECUTION_MODE
          and authority.get("production_inputs_write_allowed") is True
          and authority.get("fixture_only") is not True, "FINITE_CPU_AUTHORIZATION_REQUIRED")
    packet_ref = authority.get("task_packet_ref")
    packet = _json_ref(packet_ref, root)
    packet_path = _project_path(packet_ref["path"], root, existing=True)
    _need(packet_path.is_relative_to(root / "tasks/current"), "REGISTERED_TASK_PACKET_REQUIRED")
    _need(packet.get("execution_mode") == EXECUTION_MODE
          and packet.get("branch") == request["branch"]
          and packet.get("physical_deployment_authorized") is False
          and packet.get("control_ground_truth") is False,
          "EXACT_CPU_TASK_MODE_BRANCH_SCOPE_REQUIRED")
    _need(packet.get("inputs", {}).get("evaluation_input_spec") == request["evaluation_input_spec_ref"]
          and packet.get("inputs", {}).get("production_manifest") == request["production_manifest_ref"]
          and packet.get("output_dir") == request["output_root"], "TASK_SOURCE_OUTPUT_BINDING_REQUIRED")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    _need(deadline.tzinfo is not None and deadline > datetime.now(timezone.utc), "CPU_TASK_EXPIRED_OR_NAIVE")
    budgets = packet.get("budgets", {})
    _need(type(budgets.get("gpu_count")) is int and budgets["gpu_count"] == 0
          and type(budgets.get("model_calls_max")) is int and budgets["model_calls_max"] == 0
          and type(budgets.get("wall_seconds_max")) is int and 0 < budgets["wall_seconds_max"] <= 7200
          and type(budgets.get("cpu_threads_per_worker")) is int
          and 0 < budgets["cpu_threads_per_worker"] <= 2, "FINITE_CPU_ONLY_BUDGET_REQUIRED")
    writer = packet.get("writer", {})
    _need(type(writer.get("pid")) is int and writer["pid"] > 0
          and type(writer.get("proc_start_ticks")) is int and writer["proc_start_ticks"] > 0
          and type(writer.get("executor_epoch")) is int and writer["executor_epoch"] > 0
          and isinstance(writer.get("fencing_token"), str) and bool(writer["fencing_token"])
          and writer.get("scope") == packet["task_id"], "ACTUAL_WRITER_IDENTITY_REQUIRED")
    _need(all(authority.get(key) == packet.get(key) for key in ("task_id", "run_signature", "writer"))
          and isinstance(packet.get("run_signature"), str)
          and re.fullmatch(r"[0-9a-f]{64}", packet["run_signature"]), "AUTHORITY_PACKET_SIGNATURE_BINDING")
    receipt = _json_bytes((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_bytes())
    index = _json_ref(receipt["files"]["task_packet_index_current"], root)
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    _need(len(allowed) == 1 and allowed[0]["task_id"] == packet["task_id"]
          and allowed[0]["packet_sha256"] == packet_ref["sha256"]
          and _project_path(root / allowed[0]["packet_path"], root, existing=True) == packet_path,
          "UNIQUE_REGISTERED_CPU_PACKET_REQUIRED")
    state = _json_ref(receipt["files"]["task_state"], root)
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    _need(len(active) == 1 and active[0]["task_id"] == packet["task_id"]
          and active[0]["status"] == "RUNNING"
          and active[0].get("run_signature") == packet["run_signature"]
          and active[0].get("pid") == writer["pid"]
          and active[0].get("proc_start_ticks") == writer["proc_start_ticks"], "LIVE_UNIQUE_CPU_WRITER_REQUIRED")
    from .runtime_gate import _is_descendant
    _need(_is_descendant(writer), "ACTUAL_WRITER_ANCESTRY_REQUIRED")
    _need(_json_bytes(_project_path(packet["writer_file"], root, existing=True).read_bytes()) == writer,
          "CPU_WRITER_FENCE_DRIFT")
    contract = _json_ref(receipt["files"]["algorithm_contract"], root)
    route = contract.get("special_status_contracts", {}).get(ROUTE, {})
    _need(route.get("candidate_operation") == OPERATION and route.get("task_packet_required") is True
          and route.get("physical_deployment_authorized") is False, "REGISTERED_CPU_ROUTE_REQUIRED")
    closure = route.get("code_closure", [])
    _need(bool(closure) and all(ref in packet.get("runtime_code", []) for ref in closure), "CPU_CODE_CLOSURE_REQUIRED")
    for ref in packet["runtime_code"]:
        _read(ref, root)
    required_args = [OPERATION, "--task-packet", str(packet_path),
        "--spec", request["evaluation_input_spec_ref"]["path"],
        "--manifest", request["production_manifest_ref"]["path"], "--branch", request["branch"],
        "--output-dir", request["output_root"]]
    matched = False
    for command in packet.get("commands", []):
        tokens = shlex.split(command)
        for start in range(len(tokens) - 2):
            if tokens[start:start + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[start + 3:] == required_args
    _need(matched, "EXACT_CPU_COMMAND_REQUIRED")
    builders = [ref for ref in packet["runtime_code"]
                if Path(ref["path"]).name == "evaluation_inputs_v1.py"
                and ref["sha256"] == BUILDER_SHA256]
    _need(len(builders) == 1, "EXACT_FROZEN_CURRENT_ADAPTER_REFERENCE_REQUIRED")
    return {"task_packet_ref": packet_ref, "task_id": packet["task_id"],
            "run_signature": packet["run_signature"], "writer": writer,
            "current_input_builder_ref": builders[0]}


def _real_dependencies(root, builder_ref):
    # A future packet may explicitly bind the existing immutable preparation
    # source. It is never discovered by scanning or placed on PYTHONPATH.
    _need(builder_ref["sha256"] == BUILDER_SHA256, "FROZEN_CURRENT_ADAPTER_SHA_DRIFT")
    source_bytes = _read(builder_ref, root)
    path = _project_path(builder_ref["path"], root, existing=True)
    module_spec = importlib.util.spec_from_file_location(
        "_chaoyang_frozen_current_input_builder_" + BUILDER_SHA256, path)
    _need(module_spec is not None and module_spec.loader is not None, "FROZEN_ADAPTER_IMPORT_SPEC_REQUIRED")
    builder = importlib.util.module_from_spec(module_spec)
    # Execute the exact verified source bytes, never an unbound pyc sibling.
    exec(compile(source_bytes, str(path), "exec"), builder.__dict__)
    _read(builder_ref, root)
    import numpy as np
    from PIL import Image
    return SimpleNamespace(np=np, Image=Image,
        read_current_fields=builder.read_current_fields_npz,
        build_current_arrays=builder.build_current_input_arrays)


def produce_current_inputs(evaluation_input_spec_ref, production_manifest_ref, *, branch,
                           output_root, project_root, authorization_verifier):
    """Registered CPU production only; no automatic registration or model call."""
    root = Path(project_root).resolve(strict=True)
    _need(root == Path(__file__).resolve().parents[3], "ACTUAL_SOURCE_PROJECT_ROOT_REQUIRED")
    output = _project_path(output_root, root)
    _need(branch in {"raw", "robotized"}, "EXPLICIT_BRANCH_REQUIRED")
    request = {"evaluation_input_spec_ref": evaluation_input_spec_ref,
        "production_manifest_ref": production_manifest_ref, "branch": branch, "output_root": str(output)}
    admitted = _real_authorization(authorization_verifier, request, root)
    deps = _real_dependencies(root, admitted["current_input_builder_ref"])

    def recheck():
        _need(_real_authorization(authorization_verifier, request, root) == admitted, "CPU_AUTHORITY_CHANGED")
    return _produce(request, root, output, deps, admitted, recheck, fixture_only=False)


def produce_synthetic_fixture_inputs(evaluation_input_spec_ref, production_manifest_ref, *, branch,
                                    output_root, project_root, dependencies):
    """Explicit synthetic CPU path, never usable as a real admission authority."""
    root = Path(project_root).resolve(strict=True)
    _need(root.is_relative_to(Path(__file__).resolve().parents[3]), "FIXTURE_MUST_STAY_IN_PROJECT")
    output = _project_path(output_root, root)
    _need(branch in {"raw", "robotized"}, "EXPLICIT_BRANCH_REQUIRED")
    for key in ("read_current_fields", "build_current_arrays"):
        _need(callable(getattr(dependencies, key, None)), "EXPLICIT_SYNTHETIC_DEPENDENCIES_REQUIRED")
    request = {"evaluation_input_spec_ref": evaluation_input_spec_ref,
        "production_manifest_ref": production_manifest_ref, "branch": branch, "output_root": str(output)}
    identity = {"task_id": "synthetic_current_input_fixture", "fixture_only": True}
    return _produce(request, root, output, dependencies, identity, lambda: None, fixture_only=True)


def _produce(request, root, output, deps, admitted, recheck, *, fixture_only):
    spec = _json_ref(request["evaluation_input_spec_ref"], root)
    manifest = _json_ref(request["production_manifest_ref"], root)
    _need(spec.get("schema") == SPEC_SCHEMA and manifest.get("schema") == MANIFEST_SCHEMA,
          "FROZEN_SOURCE_SCHEMAS_REQUIRED")
    _need(spec.get("manifest_ref") == request["production_manifest_ref"], "SPEC_MANIFEST_BINDING_REQUIRED")
    _need((spec.get("fixture_only") is True) == fixture_only
          and (manifest.get("fixture_only") is True) == fixture_only, "FIXTURE_SCOPE_MUST_NOT_BECOME_REAL")
    _need(manifest.get("control_ground_truth") is False and manifest.get("physical_deployment_authorized") is False
          and manifest.get("supervision_class") == "DERIVED_RETARGET"
          and manifest.get("training_scope") == "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING",
          "DERIVED_NON_DEPLOYMENT_SOURCE_REQUIRED")
    normalization = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
    _need(manifest.get("normalization") == normalization and spec.get("normalization") == normalization,
          "FROZEN_IDENTITY_NORMALIZATION_REQUIRED")
    rows = spec.get("common_validation_rows", [])
    _need(isinstance(rows, list) and all(isinstance(row, dict)
          and set(row) == {"sample_id", "session_id", "source_frame_id", "source_group_id"}
          for row in rows), "EXACT_FROZEN_ROW_FIELDS_NO_FUTURE_MEMBERS")
    _need(all(all(isinstance(row[key], str) and bool(row[key]) for key in
                  ("sample_id", "session_id", "source_group_id"))
              and type(row["source_frame_id"]) is int and 0 <= row["source_frame_id"] <= 2**63 - 1
              for row in rows), "NATIVE_ANCHOR_IDENTITY_REQUIRED")
    _need(bool(rows) and len({row["sample_id"] for row in rows}) == len(rows), "UNIQUE_FROZEN_ANCHORS_REQUIRED")
    groups = {row["source_group_id"] for row in rows}
    _need(fixture_only or (len(rows) == 57 and len(groups) == 20), "FIXED57_ANCHORS20_GROUPS_REQUIRED")
    source = {}
    for sample in manifest.get("samples", []):
        _need(sample["sample_id"] not in source, "DUPLICATE_PRODUCTION_SAMPLE")
        source[sample["sample_id"]] = sample
    for row in rows:
        sid = row["sample_id"]
        _need(isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", sid), "UNSAFE_SAMPLE_ID")
        sample = source.get(sid, {})
        _need(sample.get("split") == "validation"
              and all(sample.get(key) == row[key] for key in ("sample_id", "session_id", "source_frame_id", "source_group_id"))
              and all(sample.get("branch_admission", {}).get(branch) is True for branch in ("raw", "robotized")),
              "FROZEN_COMMON_VALIDATION_IDENTITY_REQUIRED")
        _need(type(row["source_frame_id"]) is int and row["source_frame_id"] >= 0
              and isinstance(row["session_id"], str) and bool(row["session_id"])
              and isinstance(row["source_group_id"], str) and bool(row["source_group_id"]), "NATIVE_ANCHOR_IDENTITY_REQUIRED")
    recheck()
    reservation = _write_json(output / "CURRENT_INPUTS_RESERVED.json", {
        "schema": "humanego-current-input-production-reservation-v1", **admitted,
        "branch": request["branch"], "evaluation_input_spec_ref": request["evaluation_input_spec_ref"],
        "production_manifest_ref": request["production_manifest_ref"],
        "status": "RESERVED_NOT_COMPLETE", "fixture_only": fixture_only}, root)
    np, Image = deps.np, deps.Image
    results = []
    for row in rows:
        recheck()
        sample = source[row["sample_id"]]
        branch = request["branch"]
        labels_ref, rgb_ref = sample["labels"], sample["rgb"][branch]
        causal_ref = sample["causal_receipts"][branch]
        causal = _json_ref(causal_ref, root)
        expected = {"schema": "causal-training-input-v1", "status": "PASSED", "branch": branch,
            "sample_id": row["sample_id"], "source_session_id": row["session_id"],
            "source_frame_id": row["source_frame_id"], "image_sha256": rgb_ref["sha256"],
            "labels_sha256": labels_ref["sha256"], "input_class": "CAUSAL_TRAINING_INPUT",
            "timestamp_unit": "nanoseconds", "control_ground_truth": False, "physical_deployment_authorized": False}
        _need(all(causal.get(key) == value for key, value in expected.items())
              and type(causal.get("source_frame_id")) is int, "SOURCE_CAUSAL_IDENTITY_REQUIRED")
        _need((causal.get("fixture_only") is True) == fixture_only, "SOURCE_CAUSAL_FIXTURE_SCOPE_REQUIRED")
        _need(all(key in causal for key in TRUST_KEYS) and type(causal["physical_eye_identity_verified"]) is bool,
              "SOURCE_TRUST_QUALIFIERS_REQUIRED")
        _need(isinstance(causal.get("evidence_refs"), list) and bool(causal["evidence_refs"]), "BOUND_SOURCE_EVIDENCE_REQUIRED")
        # Original evidence is carried verbatim, not silently requalified. The
        # downstream strict gate independently revalidates its exact allowlist.
        for reference in causal["evidence_refs"]:
            _reference_shape(reference)
        _read(labels_ref, root)  # SHA only; future ZIP members are never decoded.
        fields = deps.read_current_fields(labels_ref, project_root=root)
        _need(set(fields) == {"x_ict", "ict_mask", "anchor_qpc_ns", "conditioning_source_qpc_ns",
                              "reference_camera_to_world"}, "FIVE_CURRENT_SOURCE_FIELDS_ONLY")
        anchor = fields["anchor_qpc_ns"]
        _need(anchor.shape == () and anchor.dtype.kind in "iu" and type(causal.get("anchor_qpc_ns")) is int
              and int(anchor) == causal["anchor_qpc_ns"] and 0 <= int(anchor) <= np.iinfo(np.int64).max,
              "SOURCE_CURRENT_ANCHOR_TIME_MISMATCH")
        donors = causal.get("donor_frame_ids")
        _need(isinstance(donors, list) and bool(donors)
              and all(type(frame) is int and 0 <= frame <= row["source_frame_id"] for frame in donors),
              "FUTURE_OR_INVALID_RGB_DONOR")
        pose_clock = manifest.get("pose_clock", {})
        _need(pose_clock.get("clock_domain") == causal.get("clock_domain")
              and pose_clock.get("timestamp_unit") == "nanoseconds"
              and pose_clock.get("position_unit") == "meters", "SOURCE_FRAME_CLOCK_CONTRACT_REQUIRED")
        declaration = {key: pose_clock.get(key) for key in ("clock_domain", "pose_frame_id", "camera_frame_id", "position_unit", "tcp_frame_ids")}
        _need(all(isinstance(declaration[key], str) and bool(declaration[key])
                  for key in ("clock_domain", "pose_frame_id", "camera_frame_id"))
              and isinstance(declaration["tcp_frame_ids"], list) and len(declaration["tcp_frame_ids"]) == 2
              and len(set(declaration["tcp_frame_ids"])) == 2, "EXPLICIT_FRAME_DECLARATION_REQUIRED")
        declaration["pose_semantics"] = "DERIVED_RETARGET_ANCHOR"
        with Image.open(io.BytesIO(_read(rgb_ref, root))) as image:
            image.load()  # Actual complete image decode, not file existence.
            original_size = list(image.size)
            original_mode = image.mode
            rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR),
                             dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255.)
        arrays = deps.build_current_arrays(fields, x_rgb=rgb, source_frame_id=row["source_frame_id"],
            normalization=normalization, frame_declaration=declaration)
        _need(set(arrays) == CURRENT_KEYS, "OUTPUT_REQUIRES_EXACT_SEVEN_CURRENT_KEYS")
        encoded = io.BytesIO()
        np.savez(encoded, **arrays)
        blob = encoded.getvalue()
        with np.load(io.BytesIO(blob), allow_pickle=False) as check:
            _need(len(check.files) == 7 and set(check.files) == CURRENT_KEYS
                  and all(np.array_equal(check[key], arrays[key]) for key in CURRENT_KEYS), "CURRENT_NPZ_SAFE_ROUNDTRIP_FAILED")
        _read(labels_ref, root); _read(rgb_ref, root); _read(causal_ref, root)
        recheck()
        npz_ref = _write(output / (row["sample_id"] + ".current.npz"), blob, root)
        refs = {"production_manifest_ref": request["production_manifest_ref"], "source_labels_ref": labels_ref,
            "source_rgb_ref": rgb_ref, "source_rgb_causal_receipt_ref": causal_ref}
        evidence = []
        for reference in list(refs.values()) + causal["evidence_refs"]:
            if reference not in evidence:
                evidence.append(reference)
        receipt = {"schema": "humanego-ict-causal-inference-input-v1", "status": "PASSED",
            "input_class": "CAUSAL_TRAINING_INPUT", "time_mode": "CAUSAL", "branch": branch,
            "sample_id": row["sample_id"], "source_group_id": row["source_group_id"],
            "source_frame_id": row["source_frame_id"], "anchor_qpc_ns": int(anchor),
            "input_npz_sha256": npz_ref["sha256"], "timestamp_unit": "nanoseconds",
            "image_preprocessing": "RGB_RESIZE_BILINEAR_320_240", **declaration, **refs,
            **{key: causal[key] for key in TRUST_KEYS}, "evidence_refs": evidence,
            "fixture_only": fixture_only, "control_ground_truth": False, "physical_deployment_authorized": False,
            "producer": admitted, "source_rgb_decoded": True, "source_rgb_size": original_size,
            "source_rgb_original_mode": original_mode,
            "future_target_members_decoded": [], "source_label_archive_copied": False,
            "underlying_source_evidence": "CARRIED_FROM_FROZEN_CAUSAL_RECEIPT_NOT_REQUALIFIED_BY_PRODUCER"}
        receipt_ref = _write_json(output / (row["sample_id"] + ".input_receipt.json"), receipt, root)
        results.append({**row, "anchor_qpc_ns": int(anchor), "current_input_npz": npz_ref, "input_receipt": receipt_ref})
    # No manifest is published if any row or authority fails. Partial files are
    # not a complete current-input stage and cannot be consumed as such.
    _read(request["evaluation_input_spec_ref"], root)
    _read(request["production_manifest_ref"], root)
    for row in results:
        for key in ("current_input_npz", "input_receipt"):
            _read(row[key], root)
        receipt = _json_ref(row["input_receipt"], root)
        for key in ("source_labels_ref", "source_rgb_ref", "source_rgb_causal_receipt_ref"):
            _read(receipt[key], root)
    recheck()
    manifest_ref = _write_json(output / "CURRENT_INPUTS_MANIFEST.json", {
        "schema": OUTPUT_SCHEMA, "branch": request["branch"], "evaluation_input_spec_ref": request["evaluation_input_spec_ref"],
        "production_manifest_ref": request["production_manifest_ref"], "samples": results,
        "producer": admitted, "reservation_ref": reservation, "fixture_only": fixture_only,
        "control_ground_truth": False, "physical_deployment_authorized": False}, root)
    return {"status": "CURRENT_INPUTS_PRODUCED_NOT_MODEL_EVALUATED", "branch": request["branch"],
        "sample_count": len(results), "source_group_count": len(groups), "manifest_ref": manifest_ref,
        "fixture_only": fixture_only, "future_target_members_decoded": [], "model_forward_calls": 0,
        "gpu_calls": 0, "robot_commands_sent": 0, "control_ground_truth": False, "physical_deployment_authorized": False}
