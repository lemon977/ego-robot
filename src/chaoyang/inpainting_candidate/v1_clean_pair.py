"""Fresh, sequential A/B Clean calls on one already loaded frozen ProPainter.

This module grants no execution authority and never loads a model. Production
calls must come from the registered MCO consumer. CPU tests use synthetic
providers with the unchanged ProtectedProPainter inference path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
import random
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import canonical_sha
from .v1_clean_ablation import make_pixel_propagation_ablation

PAIR_SCHEMA = "inpainting-v1-fresh-clean-pair-v1"


def array_signature(value: np.ndarray) -> dict:
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("V1_CLEAN_PAIR_OBJECT_ARRAY_FORBIDDEN")
    contiguous = np.ascontiguousarray(value)
    record = {"shape": list(value.shape), "dtype": str(value.dtype),
              "bytes": int(value.nbytes),
              "data_sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()}
    return {**record, "sha256": canonical_sha(record)}


def runtime_backend(torch: Any, device: Any) -> dict:
    """Observe current flags, never infer them from historical execution."""
    device = torch.device(device)
    cuda = device.type == "cuda"
    gpu = None
    if cuda:
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise ValueError("V1_CLEAN_PAIR_ONE_LOGICAL_CUDA_CARD_REQUIRED")
        properties = torch.cuda.get_device_properties(0)
        gpu = {"logical_index": 0, "name": str(properties.name),
               "uuid": str(getattr(properties, "uuid", "UNAVAILABLE")),
               "compute_capability": list(torch.cuda.get_device_capability(0)),
               "total_memory": int(properties.total_memory)}
    flags = {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "default_dtype": str(torch.get_default_dtype()),
        "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
        "cpu_autocast_dtype": str(torch.get_autocast_dtype("cpu")),
    }
    for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if hasattr(torch.backends.cuda.matmul, name):
            flags[name] = bool(getattr(torch.backends.cuda.matmul, name))
    if hasattr(torch.backends.cudnn, "benchmark_limit"):
        flags["cudnn_benchmark_limit"] = int(torch.backends.cudnn.benchmark_limit)
    record = {"torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(), "device": str(device),
              "gpu": gpu, "flags": flags,
              "environment": {name: os.environ.get(name) for name in
                  ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE")}}
    return {**record, "sha256": canonical_sha(record)}


def require_same_backend(expected: Mapping, actual: Mapping) -> None:
    if dict(expected) != dict(actual):
        raise ValueError("V1_CLEAN_PAIR_BACKEND_DRIFT")


def model_identity(parent: Any) -> dict:
    """Record shared module dtype/device and detect ordinary tensor mutation."""
    records = {}
    for name in ("raft", "flow", "model"):
        module = getattr(parent, name)
        tensors = []
        for kind, method in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
            getter = getattr(module, method, None)
            if getter is not None:
                for key, tensor in getter():
                    tensors.append({"kind": kind, "name": key, "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype), "device": str(tensor.device),
                        "version": int(tensor._version), "requires_grad": bool(tensor.requires_grad)})
        records[name] = {"python_object_id": id(module),
                         "training": getattr(module, "training", None), "tensors": tensors}
    record = {"settings": dict(parent.settings), "weights": dict(parent.weights),
              "parent_device": str(parent.device), "modules": records,
              "inference_dtype": "torch.float16" if parent.settings["fp16"] else "torch.float32"}
    return {**record, "sha256": canonical_sha(record)}


class RandomState:
    """Actual Python, legacy NumPy, Torch CPU and active CUDA generator states."""

    def __init__(self, torch: Any, device: Any):
        self.torch = torch
        self.cuda = torch.device(device).type == "cuda"
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.cpu = torch.get_rng_state().clone()
        self.gpu = torch.cuda.get_rng_state(0).clone() if self.cuda else None

    def arrays(self) -> dict[str, np.ndarray]:
        result = {"python_mt_state": np.asarray(self.python[1], np.uint64),
                  "numpy_mt_state": self.numpy[1].copy(),
                  "torch_cpu_rng": self.cpu.numpy().copy()}
        if self.gpu is not None:
            result["torch_cuda_0_rng"] = self.gpu.cpu().numpy().copy()
        return result

    def metadata(self) -> dict:
        record = {"python_version": self.python[0], "python_gaussian_cache": self.python[2],
                  "numpy_generator": self.numpy[0], "numpy_position": int(self.numpy[2]),
                  "numpy_has_gaussian": int(self.numpy[3]), "numpy_gaussian_cache": float(self.numpy[4]),
                  "cuda_generators": [0] if self.cuda else [],
                  "arrays": {key: array_signature(value) for key, value in self.arrays().items()}}
        return {**record, "sha256": canonical_sha(record)}

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        self.torch.set_rng_state(self.cpu.clone())
        if self.gpu is not None:
            self.torch.cuda.set_rng_state(self.gpu.clone(), 0)


def run_fresh_clean_pair(parent: Any, raw: np.ndarray, removal: np.ndarray,
                         protected: np.ndarray, *, authority: Callable[[], Any],
                         on_call: Callable[[str], Any],
                         backend_before_load: Mapping | None = None) -> dict:
    """Two actual calls; reset RNG before each; abort on flags/input drift.

    ``on_call`` is the registered consumer's budget counter. No old Clean array
    is accepted here: A always runs original image propagation afresh.
    """
    from chaoyang.pick_place_1008.visual import ProtectedProPainter, effective_clean_masks

    candidate, _ = make_pixel_propagation_ablation(parent)
    if (candidate.raft is not parent.raft or candidate.flow is not parent.flow
            or candidate.model.wrapped_model is not parent.model
            or candidate.weights is not parent.weights):
        raise ValueError("V1_CLEAN_PAIR_SHARED_MODEL_IDENTITY_REQUIRED")
    if raw.ndim != 4 or raw.shape[-1] != 3 or raw.dtype != np.uint8 or len(raw) < 2:
        raise ValueError("V1_CLEAN_PAIR_RGB_DOMAIN")
    if any(value.shape != raw.shape[:3] or value.dtype != bool for value in (removal, protected)):
        raise ValueError("V1_CLEAN_PAIR_MASK_DOMAIN")
    effective, _ = effective_clean_masks(removal, protected)
    inputs = {"raw_rgb": raw, "removal": removal, "protected_visible": protected,
              "effective_removal": effective}
    signatures = {name: array_signature(value) for name, value in inputs.items()}
    backend = runtime_backend(parent.torch, parent.device)
    if backend_before_load is not None:
        require_same_backend(backend_before_load, backend)
    identity = model_identity(parent)
    initial_rng = RandomState(parent.torch, parent.device)
    initial_signature = initial_rng.metadata()
    observations, outputs, validations = [], {}, {}

    for variant, provider in (("A", parent), ("B", candidate)):
        authority()
        require_same_backend(backend, runtime_backend(parent.torch, parent.device))
        if model_identity(parent) != identity:
            raise ValueError("V1_CLEAN_PAIR_MODEL_STATE_DRIFT")
        # Budget/fence callbacks may use randomness; finish them before reset.
        on_call(variant)
        initial_rng.restore()
        actual_rng = RandomState(parent.torch, parent.device).metadata()
        if actual_rng != initial_signature:
            raise ValueError("V1_CLEAN_PAIR_RANDOM_STATE_RESET_FAILED")
        if {name: array_signature(value) for name, value in inputs.items()} != signatures:
            raise ValueError("V1_CLEAN_PAIR_ACTUAL_INPUT_DRIFT")
        painter = ProtectedProPainter(provider)
        outputs[variant] = painter.inpaint(raw, removal, protected)
        authority()
        actual_backend = runtime_backend(parent.torch, parent.device)
        require_same_backend(backend, actual_backend)
        if model_identity(parent) != identity:
            raise ValueError("V1_CLEAN_PAIR_MODEL_STATE_DRIFT")
        if {name: array_signature(value) for name, value in inputs.items()} != signatures:
            raise ValueError("V1_CLEAN_PAIR_ACTUAL_INPUT_DRIFT")
        if outputs[variant].shape != raw.shape or outputs[variant].dtype != np.uint8:
            raise ValueError("V1_CLEAN_PAIR_OUTPUT_DOMAIN")
        changed = np.any(outputs[variant] != raw, axis=-1)
        validation = deepcopy(painter.last_validation)
        independent_protected = np.count_nonzero(changed & protected, axis=(1, 2)).tolist()
        independent_outside = np.count_nonzero(changed & ~effective, axis=(1, 2)).tolist()
        if (validation.get("status") != "PASS" or any(independent_protected) or any(independent_outside)
                or validation.get("protected_changed_pixels_per_frame") != independent_protected
                or validation.get("outside_delete_changed_pixels_per_frame") != independent_outside
                or not np.array_equal(painter.effective_masks, effective)):
            raise ValueError("V1_CLEAN_PAIR_ORIGINAL_PIXEL_PROTECTION_FAILED")
        validation.update(independent_protected_changed_pixels_per_frame=independent_protected,
                          independent_outside_effective_changed_pixels_per_frame=independent_outside)
        validations[variant] = validation
        observations.append({"variant": variant, "actual_backend": actual_backend,
            "random_state_before": actual_rng,
            "random_state_after": RandomState(parent.torch, parent.device).metadata(),
            "output": array_signature(outputs[variant]), "actual_inputs": signatures})

    difference = np.any(outputs["A"] != outputs["B"], axis=-1)
    receipt = {"schema": PAIR_SCHEMA, "status": "PAIRED_COMPLETE", "model_calls": 2,
        "fresh_A": True, "fresh_B": True, "single_model_load": True,
        "single_change": "BYPASS_IMAGE_PIXEL_PROPAGATION_ONLY",
        "same_weights_raft_flow_feature_reference_subvideo_precision": True,
        "sam_loaded_by_pair": False, "actual_inputs": signatures,
        "backend_before_load": dict(backend_before_load) if backend_before_load is not None else None,
        "actual_backend": backend, "historical_backend_inferred": False,
        "deterministic_execution_claim": False, "model_identity": identity,
        "random_state_initial": initial_signature, "calls": observations,
        "B_ablation": candidate.model.metadata(),
        "A_B_difference_pixels_per_frame": np.count_nonzero(difference, axis=(1, 2)).tolist(),
        "A_B_difference_outside_effective_per_frame": np.count_nonzero(difference & ~effective, axis=(1, 2)).tolist(),
        "native_frames": len(raw), "completed_frame_indices": list(range(len(raw))),
        "history_clean_role": "EVALUATION_ONLY_NOT_A", "visual_quality_pass": False,
        "human_review": "PENDING", "candidate_adopted": False,
        "training_eligible": False, "control_ground_truth": False}
    return {"A": outputs["A"], "B": outputs["B"], "effective_removal": effective,
            "validations": validations, "receipt": receipt, "rng_arrays": initial_rng.arrays()}


def comparison_outside_titles(columns: list[np.ndarray], titles: list[str],
                               indices: list[int], sid: str,
                               label: Callable[[np.ndarray, str], np.ndarray],
                               header_height: int = 32) -> np.ndarray:
    """Put Chinese headings in new rows, keeping every source pixel intact."""
    if (not columns or len(columns) != len(titles) or header_height < 32
            or any(value.shape != columns[0].shape or value.dtype != np.uint8 for value in columns)
            or len(columns[0]) != len(indices)):
        raise ValueError("V1_CLEAN_PAIR_COMPARISON_DOMAIN")
    _, _, width, _ = columns[0].shape
    return np.stack([np.concatenate([
        np.concatenate((label(np.zeros((header_height, width, 3), np.uint8),
                              f"{sid} 第{frame}帧｜{title}"), pixels[output]), axis=0)
        for pixels, title in zip(columns, titles)], axis=1)
        for output, frame in enumerate(indices)])
