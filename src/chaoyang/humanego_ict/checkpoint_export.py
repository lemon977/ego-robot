"""SHA-bound live training state → safe offline inference candidate export.

No resume/legacy checkpoint reader is provided. Torch is imported only after
runtime export authorization; tensor-only output is round-tripped with
weights_only=True. Export success never grants training or robot deployment.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from .data import ACTION_LAYOUT, reference_bytes
from .inference import (HAND_CHANNELS, HAND_UNIT, INFERENCE_SCHEMA, ROT6_LAYOUT,
                        TENSOR_CHECKPOINT_SCHEMA, _normalization)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHITECTURE_KEYS = {"patch_size", "vision_embed_dim", "num_decoder_layers", "num_heads",
                     "mlp_ratio", "dropout", "horizon_weighting", "horizon_beta",
                     "use_pre_norm", "use_ctx_norm"}


class CheckpointExportError(ValueError):
    pass


def _reference(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _json_ref(reference):
    try:
        value = json.loads(reference_bytes(reference))
    except (TypeError, ValueError, OSError) as error:
        raise CheckpointExportError("invalid SHA-bound JSON source") from error
    if not isinstance(value, dict):
        raise CheckpointExportError("JSON source must contain an object")
    return value


def _output(path):
    path = Path(path)
    if not path.is_absolute():
        raise CheckpointExportError("absolute project-local export directory required")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise CheckpointExportError("symbolic-link export path forbidden")
    path = path.resolve()
    if path == PROJECT_ROOT or not path.is_relative_to(PROJECT_ROOT):
        raise CheckpointExportError("export must stay inside project")
    return path


def _write_once(path, content):
    _output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CheckpointExportError("immutable tensor export conflict")
    else:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return _reference(path)


def _json_once(path, value):
    return _write_once(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def _model_config(value):
    if "model" in value:
        source = value["model"]
        if not isinstance(source, dict):
            raise CheckpointExportError("training model config must be an object")
        if (source.get("pred_horizon") != 50 or source.get("max_ict") != 8
                or source.get("img_size") != [240, 320]):
            raise CheckpointExportError("training config must fix H50/ICT8/RGB240x320")
        architecture = {key: item for key, item in source.items()
                        if key not in {"pred_horizon", "max_ict", "img_size"}}
    else:
        if (value.get("horizon") != 50 or value.get("action_dim") != 30
                or value.get("ict_dim") != 29 or value.get("image_size") != [240, 320]):
            raise CheckpointExportError("explicit H50/30D/ICT29 inference model config required")
        architecture = value.get("architecture")
    if not isinstance(architecture, dict) or set(architecture) - ARCHITECTURE_KEYS:
        raise CheckpointExportError("unsupported candidate architecture override")
    width = architecture.get("vision_embed_dim", 384)
    if type(width) is not int or width <= 0:
        raise CheckpointExportError("positive integer vision width required")
    return {"schema": "humanego-ict-revo2-model-config-v1", "horizon": 50,
            "action_dim": 30, "ict_dim": 29, "image_size": [240, 320],
            "architecture": architecture}


def _target_frame(value):
    if (not isinstance(value, Mapping) or set(value) != {
            "position_unit", "tcp_frame_ids", "pose_semantics", "action_reference_frame"}
            or value.get("position_unit") != "meters"
            or value.get("pose_semantics") != "DERIVED_RETARGET_ANCHOR"
            or value.get("action_reference_frame") != "REFERENCE_CAMERA"):
        raise CheckpointExportError("explicit derived reference-camera target contract required")
    frames = value.get("tcp_frame_ids")
    if (not isinstance(frames, list) or len(frames) != 2
            or any(not isinstance(frame, str) or not frame.strip() for frame in frames)
            or frames[0] == frames[1]):
        raise CheckpointExportError("ordered distinct left/right target frame IDs required")
    return dict(value)


def _provenance(value, model_config_ref, normalization, state_dict_key, fixture_only):
    if not isinstance(value, Mapping):
        raise CheckpointExportError("live training provenance required")
    try:
        source = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise CheckpointExportError("training provenance must contain JSON primitives only") from error
    for name in ("task_id", "run_signature"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise CheckpointExportError(f"training provenance missing {name}")
    signature = value["run_signature"]
    if len(signature) != 64 or any(char not in "0123456789abcdef" for char in signature):
        raise CheckpointExportError("frozen run_signature must be a SHA256")
    if (value.get("branch") not in {"raw", "robotized"}
            or value.get("supervision_class") != "DERIVED_RETARGET"
            or type(value.get("optimizer_updates")) is not int or value["optimizer_updates"] < 1):
        raise CheckpointExportError("explicit trained derived branch/update provenance required")
    if value.get("state_origin") != ("LIVE_MODEL_STATE" if state_dict_key == "model" else "LIVE_EMA_STATE"):
        raise CheckpointExportError("model/EMA selection requires explicit live state origin")
    dataset = _json_ref(value["dataset_ref"])
    config = _json_ref(value["training_config_ref"])
    result = _json_ref(value["training_result_ref"])
    if (dataset.get("schema") != "pick-place-humanego-ict-revo2-derived-paired-v1"
            or dataset.get("supervision_class") != "DERIVED_RETARGET"
            or dataset.get("control_ground_truth") is not False
            or dataset.get("physical_deployment_authorized") is not False
            or dataset.get("action_layout") != ACTION_LAYOUT
            or dataset.get("normalization") != normalization
            or (dataset.get("fixture_only") is True) != fixture_only):
        raise CheckpointExportError("dataset is not the selected derived non-deployment contract")
    if (_model_config(config) != _model_config(_json_ref(model_config_ref))
            or config.get("physical_deployment_authorized") is not False):
        raise CheckpointExportError("model config differs from actual training config")
    maximum = config.get("training", {}).get("max_updates")
    if (type(maximum) is not int or not 1 <= value["optimizer_updates"] <= maximum
            or result.get("optimizer_updates") != value["optimizer_updates"]
            or result.get("control_ground_truth") is not False
            or result.get("physical_deployment_authorized") is not False):
        raise CheckpointExportError("training update/result provenance mismatch")
    # The vanilla trainer result has no branch/signature fields. If a wrapper
    # supplies them, they must match; task authority binds the complete refs.
    for key in ("task_id", "run_signature", "branch"):
        if key in result and result[key] != value[key]:
            raise CheckpointExportError(f"training result provenance mismatch: {key}")
    refs = value.get("code_refs")
    if not isinstance(refs, list) or not refs:
        raise CheckpointExportError("export requires frozen training code refs")
    for reference in refs:
        reference_bytes(reference)
    return source, dataset


def _authorized(verifier, provenance, output_dir, fixture_only, *, require_gpu_lease=False):
    if not callable(verifier):
        raise CheckpointExportError("runtime export authority required before Torch")
    authority = verifier(provenance)
    mode = "SYNTHETIC_FIXTURE_ONLY" if fixture_only else "TRAINING_OWNED_LIVE_STATE"
    if (not isinstance(authority, Mapping) or authority.get("status") != "PASSED"
            or authority.get("export_authorized") is not True
            or authority.get("task_id") != provenance["task_id"]
            or authority.get("run_signature") != provenance["run_signature"]
            or authority.get("execution_mode") != mode
            or authority.get("output_dir") != str(output_dir)):
        raise CheckpointExportError("live-state export authorization rejected")
    if require_gpu_lease and authority.get("gpu_lease_verified") is not True:
        raise CheckpointExportError("GPU live snapshot requires verified current lease")
    return dict(authority)


def export_tensor_candidate(state_dict: Mapping, *, model_config_ref, normalization_ref,
                            provenance: Mapping, output_dir: Path, runtime_authorizer: Callable,
                            target_frame_contract: Mapping, state_dict_key="model", fixture_only=False):
    """Export a live completed training snapshot; never read a .pt input file.

    Passing a trusted in-memory EMA is supported only with LIVE_EMA_STATE
    provenance. No attempt is made to find EMA in a pickle checkpoint or to
    present transferred single-hand pretrained weights as a trained candidate.
    The caller owns task/data/GPU qualification and must provide a renewing
    runtime_authorizer. Every artifact remains an offline non-ground-truth
    candidate even after a successful safe-load round trip.
    """
    if state_dict_key not in {"model", "ema"} or type(fixture_only) is not bool:
        raise CheckpointExportError("explicit model/EMA selection and fixture marker required")
    output_dir = _output(output_dir)
    if not isinstance(provenance, Mapping):
        raise CheckpointExportError("training provenance object required")
    # Reject before reading tensor values or importing Torch.
    initial = _authorized(runtime_authorizer, provenance, output_dir, fixture_only)
    gpu_required = False

    def authorize():
        current = _authorized(runtime_authorizer, provenance, output_dir, fixture_only,
                              require_gpu_lease=gpu_required)
        for key in ("task_id", "run_signature", "execution_mode", "output_dir"):
            if current[key] != initial[key]:
                raise CheckpointExportError("export writer/task authorization changed")
        return current

    source_model_config = _json_ref(model_config_ref)
    model_config = _model_config(source_model_config)
    source_norm = _json_ref(normalization_ref)
    normalization = source_norm.get("normalization", source_norm)
    _normalization(normalization)
    source, dataset = _provenance(provenance, model_config_ref, normalization, state_dict_key, fixture_only)
    frame_contract = _target_frame(target_frame_contract)
    if dataset.get("pose_clock", {}).get("tcp_frame_ids") != frame_contract["tcp_frame_ids"]:
        raise CheckpointExportError("target TCP frames differ from training dataset")
    if not isinstance(state_dict, Mapping) or not state_dict or any(not isinstance(key, str) for key in state_dict):
        raise CheckpointExportError("live tensor state mapping required, not checkpoint path")
    authorize()
    import torch
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise CheckpointExportError("only live tensor values are exportable")
    if any(value.layout != torch.strided or value.is_meta for value in state_dict.values()):
        raise CheckpointExportError("dense materialized tensor state required")
    gpu_required = any(value.is_cuda for value in state_dict.values())
    authorize()
    width = model_config["architecture"].get("vision_embed_dim", 384)
    shapes = {"action_proj.weight": (width, 30), "action_proj.bias": (width,),
              "head_v.weight": (30, width), "head_v.bias": (30,),
              "state_proj.weight": (width, 29), "state_proj.bias": (width,),
              "pos_embed": (50, width), "state_pos_emb": (1, 8, width)}
    for name, expected in shapes.items():
        if (name not in state_dict or tuple(state_dict[name].shape) != expected
                or not state_dict[name].is_floating_point()):
            raise CheckpointExportError(f"not an H50/30D/ICT29 candidate tensor state: {name}")
    authorize()
    versions = {key: value._version for key, value in state_dict.items()}
    snapshot = {}
    for key in sorted(state_dict):
        authorize()
        value = state_dict[key].detach().to(device="cpu").contiguous().clone()
        if not torch.isfinite(value).all():
            raise CheckpointExportError(f"nonfinite live tensor: {key}")
        snapshot[key] = value
    if any(state_dict[key]._version != version for key, version in versions.items()):
        raise CheckpointExportError("live training state changed during snapshot")
    authorize()
    config_ref = _json_once(output_dir / "model_config.json", model_config)
    norm_ref = _json_once(output_dir / "normalization.json", normalization)
    payload = {"schema": TENSOR_CHECKPOINT_SCHEMA,
               "binding": {"config_sha256": config_ref["sha256"],
                           "normalization_sha256": norm_ref["sha256"], "action_layout": ACTION_LAYOUT},
               state_dict_key: snapshot, "training_provenance": source,
               "control_ground_truth": False, "physical_deployment_authorized": False,
               "fixture_only": fixture_only}
    # Metadata must consist solely of JSON values, never NumPy RNG/optimizer
    # objects. Encoding once also rejects accidental arbitrary Python objects.
    json.dumps(source, allow_nan=False)
    stream = io.BytesIO()
    authorize()
    torch.save(payload, stream)
    encoded = stream.getvalue()
    authorize()
    restored = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    if (restored["schema"] != TENSOR_CHECKPOINT_SCHEMA or restored["binding"] != payload["binding"]
            or set(restored[state_dict_key]) != set(snapshot)
            or any(not torch.equal(restored[state_dict_key][key], value) for key, value in snapshot.items())):
        raise CheckpointExportError("safe tensor checkpoint round trip failed")
    authorize()
    checkpoint_ref = _write_once(output_dir / "candidate_tensor.pt", encoded)
    inference_contract = {"schema": INFERENCE_SCHEMA, "horizon": 50, "action_dim": 30,
        "ict_dim": 29, "image_size": [240, 320], "action_layout": ACTION_LAYOUT,
        "rotation_layout": ROT6_LAYOUT, "hand_command_unit": HAND_UNIT,
        "hand_channel_order": list(HAND_CHANNELS), "control_ground_truth": False,
        "physical_deployment_authorized": False, "state_dict_key": state_dict_key,
        "checkpoint_ref": checkpoint_ref, "model_config_ref": config_ref,
        "normalization_ref": norm_ref, "target_frame_contract": frame_contract,
        "supervision_class": "DERIVED_RETARGET", "training_provenance": source,
        "fixture_only": fixture_only}
    authorize()
    contract_ref = _json_once(output_dir / "inference_contract.json", inference_contract)
    receipt = {"schema": "humanego-ict-tensor-export-receipt-v1",
        "status": "TENSOR_EXPORT_SAFE_ROUNDTRIP_VERIFIED_NOT_MODEL_QUALITY",
        "checkpoint_ref": checkpoint_ref, "inference_contract_ref": contract_ref,
        "state_dict_key": state_dict_key, "tensor_count": len(snapshot),
        "optimizer_updates": source["optimizer_updates"], "branch": source["branch"],
        "training_provenance": source, "model_source_ref": model_config_ref,
        "normalization_source_ref": normalization_ref,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "fixture_only": fixture_only}
    authorize()
    receipt_ref = _json_once(output_dir / "EXPORT_RECEIPT.json", receipt)
    return {"checkpoint_ref": checkpoint_ref, "inference_contract_ref": contract_ref,
            "export_receipt_ref": receipt_ref, "model_config_ref": config_ref,
            "normalization_ref": norm_ref, "state_dict_key": state_dict_key}
