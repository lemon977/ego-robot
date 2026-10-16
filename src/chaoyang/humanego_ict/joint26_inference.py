"""Read-only, SHA-bound loader and offline sampler for joint26 ICT exports.

There is deliberately no device driver, control publisher or pi0.5 loader here.
Torch/model imports occur only after the four exported file signatures and the
new interface have been checked. A portable export keeps member basenames while
every byte count/SHA remains unchanged after copying the directory.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

from .joint26_contract import (
    ACTION_DIM, ARM_INDICES, HAND_INDICES, HORIZON, interface_contract,
)


EXPORT_SCHEMA = "humanego-ict-joint26-tensor-export-manifest-v1"
INFERENCE_SCHEMA = "humanego-ict-joint26-inference-contract-v1"
_MEMBERS = {
    "tensor": "candidate_tensor.pt", "model_config": "model_config.json",
    "normalization": "normalization.json", "inference_contract": "inference_contract.json",
}
_DERIVED_CLASSES = {"DERIVED_IK_RESEARCH", "DERIVED_RETARGET_ASSUMED_MODEL_JOINTS"}


class Joint26InferenceError(ValueError):
    """A candidate cannot be loaded or its output cannot be admitted."""


class Joint26PredictionRejected(Joint26InferenceError):
    """Preserves unclipped diagnostics without granting control permission."""

    def __init__(self, diagnostic: dict):
        self.diagnostic = diagnostic
        super().__init__(f"JOINT26_PREDICTION_REJECTED: {diagnostic['status']}")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Joint26InferenceError(f"JOINT26_JSON_OBJECT_REQUIRED: {path.name}")
    return value


def _sha(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _verify_member(directory: Path, name: str, reference: Mapping) -> Path:
    if not isinstance(reference, Mapping) or not _sha(reference.get("sha256")):
        raise Joint26InferenceError(f"JOINT26_INVALID_EXPORT_REFERENCE: {name}")
    original = reference.get("path")
    if not isinstance(original, str) or Path(original).name != name:
        raise Joint26InferenceError(f"JOINT26_WRONG_EXPORT_MEMBER: {name}")
    expected_size = reference.get("bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 1:
        raise Joint26InferenceError(f"JOINT26_INVALID_EXPORT_BYTES: {name}")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
        raise Joint26InferenceError(f"JOINT26_ORDINARY_EXPORT_MEMBER_REQUIRED: {name}")
    if path.stat().st_size != expected_size or _digest(path) != reference["sha256"]:
        raise Joint26InferenceError(f"JOINT26_EXPORT_SHA_DRIFT: {name}")
    return path


def _check_interface(manifest: dict, contract: dict, model_config: dict, normalization: dict):
    if manifest.get("schema") != EXPORT_SCHEMA or contract.get("schema") != INFERENCE_SCHEMA:
        raise Joint26InferenceError("JOINT26_NEW_EXPORT_AND_INFERENCE_SCHEMA_REQUIRED")
    expected = interface_contract()
    for key, value in expected.items():
        if key == "supervision_class":
            continue  # Preserve the actual, explicitly derived producer class.
        if contract.get(key) != value:
            raise Joint26InferenceError(f"JOINT26_INTERFACE_MISMATCH: {key}")
    if contract.get("output_shape") != [HORIZON, ACTION_DIM]:
        raise Joint26InferenceError("JOINT26_H30_26_REQUIRED_NOT_H50_30")
    if (contract.get("model_config") != model_config
            or contract.get("normalization") != normalization
            or normalization != {"method": "IDENTITY_NO_FIT"}):
        raise Joint26InferenceError("JOINT26_CONFIG_OR_NORMALIZATION_BINDING_MISMATCH")
    if (manifest.get("source_state_dict_key") != "model"
            or manifest.get("model_source") != "MODEL_NOT_EMA"
            or contract.get("state_dict_key") != "model"):
        raise Joint26InferenceError("JOINT26_ONLY_DECLARED_MODEL_TENSORS_NOT_EMA")
    requested = manifest.get("requested_optimizer_updates")
    actual = manifest.get("optimizer_updates")
    if type(requested) is not int or requested <= 0:
        raise Joint26InferenceError("JOINT26_POSITIVE_REQUESTED_UPDATE_BUDGET_REQUIRED")
    if type(actual) is not int or actual <= 0 or actual != requested:
        raise Joint26InferenceError("JOINT26_COMPLETED_REQUESTED_UPDATE_BUDGET_REQUIRED")
    if "completed_update_budget" in manifest and manifest["completed_update_budget"] is not True:
        raise Joint26InferenceError("JOINT26_COMPLETED_REQUESTED_UPDATE_BUDGET_REQUIRED")
    for value in (manifest, contract):
        if (value.get("control_ground_truth") is not False
                or value.get("physical_deployment_authorized") is not False
                or value.get("supervision_class") not in _DERIVED_CLASSES):
            raise Joint26InferenceError("JOINT26_DERIVED_RESEARCH_SCOPE_REQUIRED")
    if manifest["supervision_class"] != contract["supervision_class"]:
        raise Joint26InferenceError("JOINT26_SUPERVISION_CLASS_BINDING_MISMATCH")
    if (contract.get("hand_boundary_policy") != "REJECT_WITHOUT_SILENT_CLAMP"
            or contract.get("pi05_loader_compatible") is not False):
        raise Joint26InferenceError("JOINT26_REJECT_POLICY_AND_ICT_LOADER_REQUIRED")


@dataclass
class Joint26Candidate:
    """Offline candidate and evidence; never an executable robot controller."""

    model: object
    contract: dict
    manifest: dict
    model_config: dict
    normalization: dict
    manifest_sha256: str
    external_manifest_sha_verified: bool
    device: object

    def sample_diagnostic(self, *, x_rgb, x_ict, ict_mask, x_state26, state_mask,
                          steps: int = 20, seed: int = 7,
                          target_qpc_ns=None, anchor_qpc_ns=None) -> dict:
        """Return raw, unclipped candidate arrays with explicit numeric status.

        QPC support is optional for diagnostics, never invented from nominal
        FPS. Missing timing/limits prevents the qualified ``predict`` method
        from accepting an action chunk. This method never sends the arrays.
        """
        import torch

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise Joint26InferenceError("JOINT26_POSITIVE_ODE_STEPS_REQUIRED")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise Joint26InferenceError("JOINT26_VALID_SEED_REQUIRED")
        values = {"x_rgb": x_rgb, "x_ict": x_ict, "ict_mask": ict_mask,
                  "x_state26": x_state26, "state_mask": state_mask}
        if any(not isinstance(value, torch.Tensor) for value in values.values()):
            raise Joint26InferenceError("JOINT26_TENSOR_INPUTS_REQUIRED_NO_IMPLICIT_PREPROCESSING")
        if (x_rgb.ndim != 4 or not x_rgb.is_floating_point()
                or not bool(torch.isfinite(x_rgb).all())
                or bool(((x_rgb < 0) | (x_rgb > 1)).any())):
            raise Joint26InferenceError("JOINT26_RGB_FLOAT_0_1_REQUIRED")
        batch = x_rgb.shape[0]
        if tuple(state_mask.shape) != (batch, ACTION_DIM) or state_mask.dtype != torch.bool:
            raise Joint26InferenceError("JOINT26_EXPLICIT_BOOLEAN_STATE_MASK_REQUIRED")
        if tuple(x_state26.shape) != (batch, ACTION_DIM):
            raise Joint26InferenceError("JOINT26_EXPLICIT_CURRENT_STATE26_REQUIRED")
        hand_mask = state_mask[:, list(HAND_INDICES)]
        hand_value = x_state26[:, list(HAND_INDICES)]
        if bool((hand_value[~hand_mask] != 0).any()):
            raise Joint26InferenceError("JOINT26_UNKNOWN_CURRENT_HAND_MUST_BE_ZERO_AND_FALSE_MASK")
        if not bool(torch.isfinite(hand_value[hand_mask]).all()) or bool(
                ((hand_value[hand_mask] < 0) | (hand_value[hand_mask] > 1)).any()):
            raise Joint26InferenceError("JOINT26_KNOWN_CURRENT_HAND_OUTSIDE_0_1")
        values = {name: value.to(self.device) for name, value in values.items()}
        generator = torch.Generator(device=self.device).manual_seed(seed)
        current = torch.randn((batch, HORIZON, ACTION_DIM), generator=generator,
                              device=self.device, dtype=values["x_rgb"].dtype)
        self.model.eval()
        with torch.no_grad():
            for index in range(steps):
                time = torch.full((batch,), index / steps, device=self.device, dtype=current.dtype)
                velocity = self.model(**values, x_t=current, t=time)["v_pred"]
                if not bool(torch.isfinite(velocity).all()):
                    raise Joint26InferenceError("JOINT26_NONFINITE_ODE_VECTOR_FIELD")
                current = current + velocity / steps
        if not bool(torch.isfinite(current).all()):
            raise Joint26InferenceError("JOINT26_NONFINITE_PREDICTED_ACTION")
        hand = current[:, :, list(HAND_INDICES)]
        hand_valid = (hand >= 0) & (hand <= 1)
        hand_passed = bool(hand_valid.all())
        arms_present = bool(values["state_mask"][:, list(ARM_INDICES)].all())
        numeric_status = self._joint_checks(current, values["x_state26"],
                                            target_qpc_ns, anchor_qpc_ns)
        accepted = hand_passed and arms_present and numeric_status == "PASSED_ASSUMED_NUMERICAL_LIMITS"
        return {
            "actions": current,
            "output_shape": [HORIZON, ACTION_DIM],
            "action_layout": self.contract["action_layout"],
            "status": ("ACCEPTED_OFFLINE_NUMERICAL_ONLY" if accepted else
                       "REJECTED_HAND_BOUNDARY" if not hand_passed else
                       "REJECTED_CURRENT_ARM_STATE_UNKNOWN" if not arms_present else
                       numeric_status),
            "accepted_for_offline_replay": accepted,
            "hand_boundary_passed": hand_passed,
            "hand_value_valid_mask": hand_valid,
            "current_arm_state_valid": arms_present,
            "joint_numerical_status": numeric_status,
            "hand_boundary_policy": "REJECT_WITHOUT_SILENT_CLAMP",
            "ode_steps": steps,
            "seed": seed,
            "supervision_class": self.contract["supervision_class"],
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "pi05_loader_compatible": False,
        }

    def _joint_checks(self, actions, state, target_qpc_ns, anchor_qpc_ns) -> str:
        import torch

        limits = self.contract.get("numerical_limits")
        if not isinstance(limits, Mapping):
            return "NOT_EVALUATED_JOINT_LIMITS_ABSENT"
        try:
            lower, upper, speed, step = [torch.as_tensor(limits[name], device=self.device,
                                                       dtype=actions.dtype) for name in (
                "joint_lower_rad", "joint_upper_rad", "joint_speed_max_rad_s", "joint_step_max_rad",
            )]
        except (KeyError, TypeError, ValueError):
            raise Joint26InferenceError("JOINT26_EXPORTED_NUMERICAL_LIMITS_INVALID") from None
        if (any(tuple(value.shape) != (14,) or not bool(torch.isfinite(value).all())
                for value in (lower, upper, speed, step)) or bool((lower >= upper).any())
                or bool((speed <= 0).any()) or bool((step <= 0).any())):
            raise Joint26InferenceError("JOINT26_EXPORTED_NUMERICAL_LIMITS_INVALID")
        q = actions[:, :, list(ARM_INDICES)]
        if bool(((q < lower) | (q > upper)).any()):
            return "REJECTED_ASSUMED_JOINT_BOUNDS"
        if target_qpc_ns is None or anchor_qpc_ns is None:
            return "NOT_EVALUATED_TARGET_QPC_MAPPING_ABSENT"
        target = torch.as_tensor(target_qpc_ns, device=self.device)
        anchor = torch.as_tensor(anchor_qpc_ns, device=self.device)
        if (tuple(target.shape) != tuple(actions.shape[:2]) or tuple(anchor.shape) != (actions.shape[0],)
                or target.dtype != torch.int64 or anchor.dtype != torch.int64):
            raise Joint26InferenceError("JOINT26_EXPLICIT_INT64_TARGET_QPC_MAPPING_REQUIRED")
        timestamps = torch.cat((anchor[:, None], target), dim=1)
        dt_ns = timestamps[:, 1:] - timestamps[:, :-1]
        if bool((dt_ns <= 0).any()):
            raise Joint26InferenceError("JOINT26_TARGET_QPC_MUST_BE_STRICTLY_FUTURE_MONOTONIC")
        trajectory = torch.cat((state[:, None, list(ARM_INDICES)], q), dim=1)
        delta = trajectory[:, 1:] - trajectory[:, :-1]
        if bool((delta.abs() > step).any()):
            return "REJECTED_ASSUMED_JOINT_STEP"
        if bool((delta.abs() > speed * dt_ns.to(actions.dtype).unsqueeze(-1) / 1e9).any()):
            return "REJECTED_ASSUMED_JOINT_SPEED"
        return "PASSED_ASSUMED_NUMERICAL_LIMITS"

    def predict(self, **inputs):
        """Only return a numerically admitted chunk; still not physical approval."""
        diagnostic = self.sample_diagnostic(**inputs)
        if not diagnostic["accepted_for_offline_replay"]:
            raise Joint26PredictionRejected(diagnostic)
        return diagnostic["actions"]


def load_joint26_candidate(export_directory, *, device="cpu", expected_manifest_sha256=None) -> Joint26Candidate:
    """Verify the export before importing Torch or allocating a candidate model.

    An external sealed manifest SHA is optional for diagnostic use, but callers
    must supply it to establish the manifest's provenance. File SHA verification
    alone is integrity, not proof of calibration or source/data qualification.
    """
    directory = Path(export_directory).resolve()
    manifest_path = directory / "EXPORT_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise Joint26InferenceError("JOINT26_EXPORT_MANIFEST_REQUIRED")
    digest = _digest(manifest_path)
    if expected_manifest_sha256 is not None:
        if not _sha(expected_manifest_sha256) or digest != expected_manifest_sha256:
            raise Joint26InferenceError("JOINT26_TRUSTED_MANIFEST_SHA_DRIFT")
    manifest = _json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_MEMBERS):
        raise Joint26InferenceError("JOINT26_EXACT_FOUR_EXPORT_FILES_REQUIRED")
    members = {key: _verify_member(directory, name, files[key]) for key, name in _MEMBERS.items()}
    contract = _json(members["inference_contract"])
    model_config = _json(members["model_config"])
    normalization = _json(members["normalization"])
    _check_interface(manifest, contract, model_config, normalization)

    import torch
    from .joint26_model import build_joint26_model

    resolved_device = torch.device(device)
    tensor_blob = members["tensor"].read_bytes()
    tensor_ref = files["tensor"]
    if (len(tensor_blob) != tensor_ref["bytes"]
            or hashlib.sha256(tensor_blob).hexdigest() != tensor_ref["sha256"]):
        raise Joint26InferenceError("JOINT26_EXPORT_SHA_DRIFT_DURING_TENSOR_LOAD")
    state = torch.load(io.BytesIO(tensor_blob), map_location="cpu", weights_only=True)
    # Keep JSON bytes/signatures frozen as well, rather than permitting a hot
    # switch between initial verification and creating the model.
    for key, name in _MEMBERS.items():
        if key != "tensor":
            _verify_member(directory, name, files[key])
    if _digest(manifest_path) != digest:
        raise Joint26InferenceError("JOINT26_MANIFEST_CHANGED_DURING_LOAD")
    if not isinstance(state, Mapping) or not state or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise Joint26InferenceError("JOINT26_FLAT_TENSOR_ONLY_MODEL_REQUIRED")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise Joint26InferenceError("JOINT26_NONFINITE_EXPORTED_MODEL_TENSOR")
    model = build_joint26_model(model_config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Joint26InferenceError("JOINT26_TENSOR_SHAPE_OR_MODEL_CONFIG_MISMATCH") from error
    model = model.to(resolved_device).eval()
    return Joint26Candidate(model=model, contract=contract, manifest=manifest, model_config=model_config,
                            normalization=normalization, manifest_sha256=digest,
                            external_manifest_sha_verified=expected_manifest_sha256 is not None,
                            device=resolved_device)
