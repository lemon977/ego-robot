"""Explicit candidate SAM/ProPainter bridges; importing this loads no models.

The caller owns the existing Task Packet fence and card lease. Every model call
invokes its authority callback. G0 CPU development is not model authorization.
The bridge writes only a new project-local stage directory, sealing its receipt
last; failed directories are never consumable. No historical seeds or imagery
are accepted. These are offline visual candidates, not training/control truth.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .automatic_masks import (build_automatic_masks,
                              build_automatic_masks_with_source_aware_veto, collect_sam_rgb_proposals,
                              frames_sha256, refine_sam_rgb_proposals,
                              refine_sam_rgb_proposals_with_text_retention,
                              resolve_controller_text_prompt, resolve_glove_text_prompt,
                              resolve_text_wearer_retention,
                              resolve_source_aware_retention_veto)
from .clean_references import validate_clean_preservation
from .contracts import SCHEMA, canonical_sha, file_ref, verify_ref

PROJECT_ROOT = Path(__file__).resolve().parents[3]
Authority = Callable[[], Mapping[str, Any]]


class ModelStageError(ValueError):
    pass


def _gate(authority: Authority) -> dict:
    """Recheck the caller's real registry/writer/lease gate before each call."""
    if not callable(authority):
        raise ModelStageError("REGISTERED_MODEL_AUTHORITY_REQUIRED")
    row = dict(authority())
    if (row.get("execution_allowed") is not True
            or row.get("algorithm_execution_allowed") is not True
            or row.get("worker_logical_index") != 0
            or not isinstance(row.get("task_id"), str) or not row["task_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("run_signature", "")))
            or not row.get("fencing_token")
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", str(row.get("gpu_uuid", "")))):
        raise ModelStageError("REGISTERED_GPU_MODEL_AUTHORITY_REQUIRED")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != row["gpu_uuid"]:
        raise ModelStageError("PHYSICAL_UUID_TO_SINGLE_LOGICAL_ZERO_REQUIRED")
    return row


def _same_gate(authority: Authority, initial: Mapping) -> dict:
    current = _gate(authority)
    if any(current[key] != initial[key] for key in (
            "task_id", "run_signature", "fencing_token", "gpu_uuid", "worker_logical_index")):
        raise ModelStageError("MODEL_TASK_WRITER_OR_CARD_IDENTITY_DRIFT")
    return current


class _AuthorizedPointModel:
    def __init__(self, provider, authority, owner):
        self.provider, self.authority, self.owner = provider, authority, owner

    def __getattr__(self, name):
        original = getattr(self.provider, name)
        if name not in {"init_state", "add_prompt"}:
            return original
        def call(*args, **kwargs):
            _same_gate(self.authority, self.owner)
            result = original(*args, **kwargs)
            _same_gate(self.authority, self.owner)
            return result
        return call

    def propagate_in_video(self, *args, **kwargs):
        _same_gate(self.authority, self.owner)
        iterator = iter(self.provider.propagate_in_video(*args, **kwargs))
        while True:
            _same_gate(self.authority, self.owner)
            try:
                row = next(iterator)
            except StopIteration:
                return
            _same_gate(self.authority, self.owner)
            yield row


class _AuthorizedSam:
    def __init__(self, provider, authority, owner):
        self.provider, self.authority, self.owner = provider, authority, owner

    def __getattr__(self, name):
        value = getattr(self.provider, name)
        return (_AuthorizedPointModel(value, self.authority, self.owner)
                if name == "model" else value)

    def track_text(self, *args, **kwargs):
        _same_gate(self.authority, self.owner)
        result = self.provider.track_text(*args, **kwargs)
        _same_gate(self.authority, self.owner)
        return result


def _within_project(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == PROJECT_ROOT.resolve() or not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ModelStageError("OUTPUT_OR_CACHE_OUTSIDE_PROJECT")
    return resolved


def configure_project_caches(output_root: str | Path) -> dict:
    """Configure before importing Torch/SAM; never redirect writes to /tmp."""
    root = _within_project(output_root)
    env = {}
    for name, relative in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "TORCH_HOME": "torch", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton", "HF_HOME": "huggingface",
        "XDG_CACHE_HOME": "xdg", "CUDA_CACHE_PATH": "cuda",
    }.items():
        target = _within_project(root / "model_cache" / relative)
        target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    os.environ.update(env)
    tempfile.tempdir = env["TMPDIR"]
    sys.dont_write_bytecode = True
    return env


def load_frozen_models(repo_root: str | Path, output_root: str | Path, *,
                       authorization: Authority, frozen_inputs: Sequence[Mapping],
                       need_sam: bool = True, need_clean: bool = False) -> dict:
    """Load only the already installed SAM3.1/PP, after actual closure checks.

    ``frozen_inputs`` must be the actual frozen code/weights files, not a single
    checkpoint declaration. The parent runner additionally verifies its complete
    component closure. Existing default filenames are required, never downloads.
    """
    owner = _gate(authorization)
    root = Path(repo_root).resolve(strict=True)
    if root != PROJECT_ROOT.resolve() or not frozen_inputs:
        raise ModelStageError("PROJECT_FROZEN_MODEL_CLOSURE_REQUIRED")
    refs = [dict(value) for value in frozen_inputs]
    for row in refs:
        if not Path(row["path"]).resolve(strict=True).is_relative_to(root):
            raise ModelStageError("MODEL_INPUT_OUTSIDE_PROJECT")
        verify_ref(row)
    paths = {Path(row["path"]).resolve() for row in refs}
    required = []
    if need_sam:
        required += ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
                     "src/chaoyang/exact78_baseline/mask.py",
                     "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"]
        if not any(path.is_relative_to(root / "vendor/SAM3") for path in paths):
            raise ModelStageError("SAM_CODE_CLOSURE_REQUIRED")
    if need_clean:
        required += ["src/chaoyang/exact78_baseline/clean.py",
                     "src/chaoyang/pick_place_1008/visual.py"]
        required += ["vendor/ProPainter/weights/" + name for name in
                     ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")]
        if not any(path.is_relative_to(root / "vendor/ProPainter/model") for path in paths):
            raise ModelStageError("PROPAINTER_CODE_CLOSURE_REQUIRED")
    if any((root / relative).resolve() not in paths for relative in required):
        raise ModelStageError("FROZEN_MODEL_INPUT_CLOSURE_INCOMPLETE")
    cache = configure_project_caches(output_root)
    result, timings = {}, {}
    if need_sam or need_clean:
        import torch
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise ModelStageError("MODEL_WORKER_REQUIRES_ONE_LOGICAL_CUDA_CARD")
    # Loading neither model is valid for a caller doing resource preparation.
    if need_sam:
        _same_gate(authorization, owner)
        from chaoyang.exact78_baseline.mask import Sam3Video
        start = time.monotonic()
        result["sam"] = Sam3Video(root)
        _synchronize(result["sam"])
        timings["SAM31_constructor_wall_seconds"] = time.monotonic() - start
        _same_gate(authorization, owner)
    if need_clean:
        _same_gate(authorization, owner)
        from chaoyang.exact78_baseline.clean import ProPainter
        start = time.monotonic()
        result["propainter"] = ProPainter(root)
        _synchronize(result["propainter"])
        timings["ProPainter_constructor_wall_seconds"] = time.monotonic() - start
        _same_gate(authorization, owner)
    for row in refs:
        verify_ref(row)
    result["load_receipt"] = {"schema": "inpainting-frozen-model-load-v1",
        "authority": owner, "frozen_inputs": refs, "cache_environment": cache,
        "timings": timings, "new_model_or_download": False,
        "quality_claim": "NONE_CONSTRUCTORS_NOT_VISUAL_VALIDATION"}
    return result


def _synchronize(provider: Any) -> None:
    # Actual installed providers use Torch; synthetic API tests do not. Do not
    # import Torch merely to time a fake provider or a CPU-only import.
    torch = getattr(provider, "torch", None)
    if torch is None and type(provider).__module__ == "chaoyang.exact78_baseline.mask":
        import torch
    if torch is not None and getattr(torch, "cuda", None) is not None:
        torch.cuda.synchronize(0)


def _json(path: Path, value: Any) -> dict:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def _npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict:
    if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
        raise ModelStageError("OBJECT_ARRAY_FORBIDDEN")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def _seal_rgb_proposals(root: Path, stage: str, proposals: Sequence[Any],
                        native_shape: tuple[int, ...]) -> list[dict]:
    """Archive actual pre/post-point pixels, never infer them from final roles.

    This CPU diagnostic does not submit prompts, change ownership, or provide
    inputs to the model. Stable role/id bindings and explicit empty native
    frames make later failure attribution possible without a hidden cache.
    """
    if stage not in {"TEXT", "POINT"} or len(native_shape) != 3:
        raise ModelStageError("PROPOSAL_DIAGNOSTIC_DOMAIN_INVALID")
    arrays, rows = {}, []
    seen = set()
    for index, proposal in enumerate(proposals):
        identity = (proposal.role, int(proposal.track_id))
        masks = np.asarray(proposal.masks)
        if identity in seen or masks.dtype != np.bool_ or masks.shape != native_shape:
            raise ModelStageError("PROPOSAL_DIAGNOSTIC_ID_OR_NATIVE_MASK_INVALID")
        seen.add(identity)
        key = f"proposal_{index:04d}"
        arrays[key] = np.packbits(masks.reshape(native_shape[0], -1), axis=1)
        row = {"array_key": key, "role": proposal.role, "track_id": int(proposal.track_id),
            "source": proposal.source, "native_shape": list(native_shape),
            "frames_sha256": proposal.frames_sha256,
            "raw_source_sha256": proposal.raw_source_sha256,
            "frame_map_sha256": proposal.frame_map_sha256,
            "model_sha256": proposal.model_sha256,
            "supported_mask_pixels_per_frame": masks.sum(axis=(1, 2)).tolist(),
            "refinement_status": proposal.refinement_status,
            "refinement_receipt_payload_sha256": proposal.refinement_receipt_payload_sha256}
        for name in ("unsupported_frames", "prior_unsupported_frames"):
            flags = getattr(proposal, name)
            if flags is None:
                row[name + "_array_key"] = None
                continue
            flags = np.asarray(flags)
            if flags.dtype != np.bool_ or flags.shape != (native_shape[0],):
                raise ModelStageError("PROPOSAL_DIAGNOSTIC_UNSUPPORTED_FLAGS_INVALID")
            flag_key = key + "_" + name
            arrays[flag_key] = flags
            row[name + "_array_key"] = flag_key
        # The post-POINT archive may contain an explicitly declared union.
        # Preserve each pixel origin independently; its TEXT pixels must never
        # be retrospectively represented as a fresh POINT model prediction.
        for name in ("retained_text_masks", "point_masks", "retention_conflict_masks"):
            origin = getattr(proposal, name, None)
            if origin is None:
                continue
            origin = np.asarray(origin)
            if origin.dtype != np.bool_ or origin.shape != native_shape:
                raise ModelStageError("PROPOSAL_DIAGNOSTIC_ORIGIN_DOMAIN_INVALID")
            origin_key = key + "_" + name
            arrays[origin_key] = np.packbits(origin.reshape(native_shape[0], -1), axis=1)
            row[name + "_array_key"] = origin_key
        rows.append(row)
    pixels_ref = _npz(root / f"SAM_{stage}_PROPOSALS.npz", arrays)
    metadata_ref = _json(root / f"SAM_{stage}_PROPOSALS.json", {
        "schema": "inpainting-sam-native-proposal-diagnostic-v1", "stage": stage,
        "coordinate_domain": "NATIVE_RAW_RGB", "native_shape": list(native_shape),
        "packing": "NUMPY_PACKBITS_C_ORDER_AXIS1_BIG_BITORDER", "arrays": pixels_ref,
        "proposals": rows, "claim_limit": "Model predictions, not observed geometry or manual annotations",
        "algorithm_gain": False, "feeds_producer": False})
    return [pixels_ref, metadata_ref]


def _inputs(frames_rgb: np.ndarray, raw_source_ref: Mapping,
            frame_map: Sequence[Mapping], component_signature: str) -> tuple[np.ndarray, list[dict]]:
    raw = np.asarray(frames_rgb)
    if (raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[-1] != 3
            or not len(raw) or min(raw.shape[1:3]) < 1):
        raise ModelStageError("NATIVE_RAW_RGB_UINT8_REQUIRED")
    if not re.fullmatch(r"[0-9a-f]{64}", str(component_signature)):
        raise ModelStageError("FROZEN_COMPONENT_SIGNATURE_REQUIRED")
    verify_ref(raw_source_ref)
    source = Path(raw_source_ref["path"])
    domain = source.relative_to(PROJECT_ROOT) if source.is_relative_to(PROJECT_ROOT) else source
    if {piece.lower() for piece in domain.parts} & {"_run", "review", "processed", "attempts"}:
        raise ModelStageError("HISTORICAL_DERIVED_RAW_INPUT_FORBIDDEN")
    mapping = [dict(row) for row in frame_map]
    if len(mapping) != len(raw):
        raise ModelStageError("FULL_NATIVE_FRAME_MAP_COUNT_REQUIRED")
    for index, row in enumerate(mapping):
        if (type(row.get("native_frame_index")) is not int
                or row["native_frame_index"] != index
                or type(row.get("decoded_display_index")) is not int
                or row["decoded_display_index"] != index):
            raise ModelStageError("NATIVE_TO_DECODER_ONE_TO_ONE_MAPPING_REQUIRED")
    canonical_sha(mapping)  # Also reject NaN or unserializable mapping.
    return raw, mapping


def _new_stage(output_dir: str | Path) -> Path:
    root = _within_project(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _failure(root: Path, stage: str, error: Exception) -> None:
    _json(root / "FAILED.json", {"stage": stage, "status": "FAILED_PARTIAL_NOT_CONSUMABLE",
        "error_type": type(error).__name__, "error": str(error),
        "visual_quality": "NOT_EVALUATED", "training_eligible": False})


def run_automatic_mask_stage(frames_rgb: np.ndarray, *, sam: Any,
        output_dir: str | Path, raw_source_ref: Mapping, frame_map: Sequence[Mapping],
        model_ref: Mapping, component_signature: str, authorization: Authority,
        projection_hints: np.ndarray | None = None,
        controller_text_prompt: str | None = None,
        text_wearer_retention: str | None = None,
        source_aware_retention_veto: str | None = None,
        glove_text_prompt: str | None = None) -> dict:
    """Run actual text/device plus automatic RGB-point stages, then seal all T.

    The same fixed recipe is used without task name, manual seed or session ID.
    Raw/map arrays are archived so conclusions can trace actual consumed pixels.
    Hardware hints can associate tracks, never act as visible mask evidence.
    """
    owner = _gate(authorization)
    resolve_controller_text_prompt(controller_text_prompt)
    resolve_glove_text_prompt(glove_text_prompt)
    resolve_text_wearer_retention(text_wearer_retention)
    resolve_source_aware_retention_veto(source_aware_retention_veto)
    prompt_options = ({} if controller_text_prompt is None else
                      {"controller_text_prompt": controller_text_prompt})
    if glove_text_prompt is not None:
        prompt_options["glove_text_prompt"] = glove_text_prompt
    retention_options = ({} if text_wearer_retention is None else
                         {"text_wearer_retention": text_wearer_retention})
    veto_options = ({} if source_aware_retention_veto is None else
                    {"source_aware_retention_veto": source_aware_retention_veto})
    raw, mapping = _inputs(frames_rgb, raw_source_ref, frame_map, component_signature)
    verify_ref(model_ref)
    root = _new_stage(output_dir)
    start = time.monotonic()
    try:
        authorized_sam = _AuthorizedSam(sam, authorization, owner)
        outputs = [_npz(root / "RAW_RGB.npz", {"raw_rgb": raw}),
                   _json(root / "FRAME_MAP.json", mapping)]
        signatures = {"raw_source_sha256": raw_source_ref["sha256"],
                      "frame_map_sha256": canonical_sha(mapping),
                      "model_sha256": model_ref["sha256"]}
        timing = {}
        _same_gate(authorization, owner)
        begun = time.monotonic()
        proposals, text_receipt = collect_sam_rgb_proposals(
            authorized_sam, raw, **signatures, **prompt_options)
        _synchronize(sam)
        timing["SAM31_text_device_wall_seconds"] = time.monotonic() - begun
        _same_gate(authorization, owner)
        outputs.append(_json(root / "SAM_TEXT.json", text_receipt))
        outputs.extend(_seal_rgb_proposals(root, "TEXT", proposals, raw.shape[:3]))
        begun = time.monotonic()
        refine = (refine_sam_rgb_proposals if text_wearer_retention is None else
                  refine_sam_rgb_proposals_with_text_retention)
        refined, point_receipt = refine(
            authorized_sam, raw, proposals, **signatures, projection_hints=projection_hints,
            **retention_options)
        _synchronize(sam)
        timing["SAM31_automatic_point_wall_seconds"] = time.monotonic() - begun
        _same_gate(authorization, owner)
        outputs.append(_json(root / "SAM_POINT.json", point_receipt))
        outputs.extend(_seal_rgb_proposals(root, "POINT", refined, raw.shape[:3]))
        build = (build_automatic_masks if source_aware_retention_veto is None else
                 build_automatic_masks_with_source_aware_veto)
        masks = build(raw, refined, **signatures,
                     projection_hints=projection_hints, **veto_options)
        if glove_text_prompt is not None:
            masks["metadata"]["prompts"]["glove"] = glove_text_prompt
            masks["metadata"]["glove_text_prompt"] = glove_text_prompt
        arrays = {name: value for name, value in masks.items() if isinstance(value, np.ndarray)}
        arrays.update({"equipment__" + name: value
                       for name, value in masks["equipment_roles"].items()})
        if projection_hints is not None:
            arrays["projection_association_hints"] = np.asarray(projection_hints)
        outputs.append(_npz(root / "MASK_ARRAYS.npz", arrays))
        outputs.append(_json(root / "MASK_METADATA.json", masks["metadata"]))
        outputs.append(_json(root / "AUTOMATIC_POINTS.json", masks["automatic_points"]))
        status = [{"native_frame_index": index,
            "wearer_support": masks["metadata"]["wearer_support_status_by_native_frame"][index],
            "unknown_pixels": int(masks["unknown"][index].sum()),
            "protected_pixels": int(masks["protected_visible"][index].sum()),
            "effective_removal_pixels": int(masks["effective_removal"][index].sum()),
            "stage_execution": "MODEL_PROVIDER_RETURNED_NATIVE_FRAME",
            "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
            "geometry": "UNKNOWN", "time_mode": "OFFLINE_VISUAL",
            "training_eligible": False} for index in range(len(raw))]
        outputs.append(_json(root / "FRAME_STATUS.json", status))
        verify_ref(raw_source_ref)
        verify_ref(model_ref)
        _same_gate(authorization, owner)
        timing["whole_stage_wall_seconds_including_IO"] = time.monotonic() - start
        receipt = {"schema": SCHEMA, "stage": "Mask", "status": "SEALED_COMPLETE",
            "component_signature": component_signature, "outputs": outputs,
            "actual_inputs": {"raw_source": dict(raw_source_ref), "model": dict(model_ref),
                              "frames_sha256": frames_sha256(raw),
                              "frame_map_sha256": canonical_sha(mapping)},
            "authority": owner, "native_input_frames": len(raw), "native_output_frames": len(raw),
            "timings": timing, "text_prompt_invocations": len(text_receipt["calls"]),
            "automatic_points_submitted": point_receipt["automatic_points_submitted"],
            "manual_points_submitted": 0, "producer_manual_seed_reads": 0,
            "model_provider": type(sam).__module__ + "." + type(sam).__qualname__,
            "model_proof_scope": "PROVIDER_INVOCATIONS_FULL_TASK_INPUT_AUDIT_REQUIRED",
            "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False,
            "completed_at": datetime.now(timezone.utc).isoformat()}
        if controller_text_prompt is not None:
            receipt["actual_inputs"]["controller_text_prompt"] = controller_text_prompt
        if glove_text_prompt is not None:
            receipt["actual_inputs"]["glove_text_prompt"] = glove_text_prompt
        if text_wearer_retention is not None:
            receipt["actual_inputs"]["text_wearer_retention"] = text_wearer_retention
        if source_aware_retention_veto is not None:
            receipt["actual_inputs"]["source_aware_retention_veto"] = source_aware_retention_veto
        _json(root / "RECEIPT.json", receipt)
        return receipt
    except Exception as error:
        _failure(root, "Mask", error)
        raise


def _checked_sealed_effective(removal: np.ndarray, protected: np.ndarray,
                              effective: np.ndarray) -> tuple[np.ndarray, dict]:
    """Use a signed hole subset without changing frozen dilation or protection."""
    if (not isinstance(effective, np.ndarray) or effective.dtype != np.bool_
            or effective.shape != removal.shape):
        raise ModelStageError("CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED")
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    baseline, validation = effective_clean_masks(removal, protected)
    if np.any(effective & protected) or np.any(effective & ~baseline):
        raise ModelStageError("CLEAN_SEALED_EFFECTIVE_PROTECTION_OR_DILATION_VIOLATION")
    validation["effective_delete_pixels_per_frame"] = np.count_nonzero(
        effective, axis=(1, 2)).tolist()
    validation["effective_protected_overlap_per_frame"] = np.count_nonzero(
        effective & protected, axis=(1, 2)).tolist()
    validation["sealed_effective_removal_consumed"] = True
    return effective.copy(), validation


class ReferenceSelectingProtectedProPainter:
    """Same frozen PP bytecode; instance-local refs or sealed hole binding.

    With no selector, diagnostic hook or sealed mask, invokes the original
    ProtectedProPainter method directly.
    For G2, a copied function-global namespace redirects exactly the unique
    `_get_ref_index` call. Original function bytecode, models, settings, neighbors,
    propagation, output ownership and averaging are unchanged. No monkeypatch
    mutates the original module or another worker. An explicit effective mask
    replaces only the mask helper in this function's private namespace; the
    original settings, protection and inference function bytecode remain.
    """
    def __init__(self, parent: Any, selector: Callable | None = None, *,
                 record_reference_selection: bool = False,
                 sealed_effective_removal: np.ndarray | None = None):
        from chaoyang.pick_place_1008.visual import ProtectedProPainter
        self.wrapped = ProtectedProPainter(parent)
        self.selector = selector
        self.record_reference_selection = record_reference_selection
        self.sealed_effective_removal = (None if sealed_effective_removal is None
                                          else sealed_effective_removal.copy()
                                          if isinstance(sealed_effective_removal, np.ndarray)
                                          else sealed_effective_removal)
        self.calls: list[dict] = []
        original = ProtectedProPainter.inpaint
        source = inspect.getsource(original)
        import textwrap
        tree = ast.parse(textwrap.dedent(source))
        sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "_get_ref_index"]
        if len(sites) != 1 or len(sites[0].args) != 5 or sites[0].keywords:
            raise ModelStageError("FROZEN_PP_REFERENCE_CALLSITE_DRIFT")
        self.original_function_sha256 = hashlib.sha256(source.encode()).hexdigest()
        namespace = dict(original.__globals__)
        if selector is not None or record_reference_selection:
            namespace["_get_ref_index"] = self._select
        if sealed_effective_removal is not None:
            namespace["effective_clean_masks"] = self._effective
        hooked = types.FunctionType(original.__code__, namespace, original.__name__,
                                    original.__defaults__, original.__closure__)
        hooked.__kwdefaults__ = original.__kwdefaults__
        self._hooked = types.MethodType(hooked, self.wrapped)

    def _effective(self, removal, protected, settings):
        if settings != self.wrapped.settings:
            raise ModelStageError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
        return _checked_sealed_effective(removal, protected, self.sealed_effective_removal)

    def _select(self, mid, neighbors, length, stride, ref_num):
        from chaoyang.exact78_baseline.clean import _get_ref_index
        original = _get_ref_index(mid, neighbors, length, stride, ref_num)
        selected = original if self.selector is None else self.selector(
            mid=mid, neighbors=list(neighbors), length=length, stride=stride,
            ref_num=ref_num, original_references=list(original))
        selected = list(selected)
        if (len(selected) != len(original) or len(set(selected)) != len(selected)
                or selected != sorted(selected)
                or any(type(index) is not int or index in neighbors or not 0 <= index < length
                       for index in selected)):
            raise ModelStageError("PP_REFERENCE_COUNT_OR_NATIVE_DOMAIN_CHANGED")
        allowed = (set(range(length)) if ref_num == -1 else
                   set(range(max(0, mid - stride * (ref_num // 2)),
                             min(length, mid + stride * (ref_num // 2))))) - set(neighbors)
        if not set(selected).issubset(allowed):
            raise ModelStageError("PP_REFERENCE_CONTEXT_CHANGED")
        self.calls.append({"target_frame": mid, "neighbors": list(neighbors),
            "original_references": original, "references": selected,
            "reference_count": len(original), "settings_unchanged": True})
        return selected

    def inpaint(self, raw, removal, protected):
        self.calls.clear()
        if (self.selector is None and not self.record_reference_selection
                and self.sealed_effective_removal is None):
            return self.wrapped.inpaint(raw, removal, protected)
        return self._hooked(raw, removal, protected)


class OnlineReferenceContext:
    """Bind same-inference Raw-flow evidence before the first refs selection.

    The frozen PP has already finished its Raw RAFT calls at this point. A
    producer can seal those actual calls here instead of rerunning A merely to
    obtain flow. This adapter does not manufacture support, invoke models or
    qualify geometry. The provider must return immutable, signed evidence.
    """
    def __init__(self, selector, provider, context, authorization, owner):
        if not callable(selector) or not callable(provider):
            raise ModelStageError("ONLINE_REFERENCE_CALLBACKS_REQUIRED")
        self.selector, self.provider = selector, provider
        self.context, self.authorization, self.owner = context, authorization, owner
        self.resolved = False

    def __call__(self, **kwargs):
        _same_gate(self.authorization, self.owner)
        if not self.resolved:
            refs = self.provider()
            if not isinstance(refs, Mapping) or set(refs) != {"raw_flow_receipt", "reference_evidence"}:
                raise ModelStageError("SIGNED_ONLINE_FLOW_AND_REFERENCE_CONTEXT_REQUIRED")
            for value in refs.values():
                path = Path(value["path"])
                if path.resolve(strict=True) != path or not path.is_relative_to(PROJECT_ROOT):
                    raise ModelStageError("CLEAN_CONTEXT_PROJECT_REAL_PATH_REQUIRED")
                verify_ref(value)
            flow = json.loads(Path(refs["raw_flow_receipt"]["path"]).read_text())
            evidence = json.loads(Path(refs["reference_evidence"]["path"]).read_text())
            from .contracts import verify_sealed_stage
            for payload, key, stage in ((flow, "raw_flow_receipt", "RawFlow"),
                                         (evidence, "reference_evidence", "ReferenceEvidence")):
                if payload.get("stage") != stage:
                    raise ModelStageError("ONLINE_REFERENCE_CONTEXT_STAGE_MISMATCH")
                verify_sealed_stage(payload, payload.get("component_signature"),
                                    output_root=Path(refs[key]["path"]).parent)
            if (flow.get("actual_inputs", {}).get("capture_config") != self.context["capture_config"]
                    or evidence.get("actual_inputs", {}).get("raw_flow_receipt") != refs["raw_flow_receipt"]):
                raise ModelStageError("ONLINE_REFERENCE_CAPTURE_OR_FLOW_BINDING_MISMATCH")
            self.context.update({key: dict(value) for key, value in refs.items()})
            self.resolved = True
        for key in ("raw_flow_receipt", "reference_evidence"):
            verify_ref(self.context[key])
        result = self.selector(**kwargs)
        _same_gate(self.authorization, self.owner)
        return result


def run_clean_stage(frames_rgb: np.ndarray, *, masks: Mapping[str, np.ndarray],
        mask_receipt_ref: Mapping, parent_propainter: Any, output_dir: str | Path,
        raw_source_ref: Mapping, frame_map: Sequence[Mapping], model_refs: Sequence[Mapping],
        component_signature: str, authorization: Authority,
        reference_selector: Callable | None = None,
        clean_context_refs: Mapping[str, Mapping] | None = None,
        record_reference_selection: bool = False,
        sealed_effective_removal: np.ndarray | None = None,
        online_reference_context_provider: Callable | None = None) -> dict:
    """Frozen protected PP with optional signed refs or exact sealed hole subset."""
    owner = _gate(authorization)
    raw, mapping = _inputs(frames_rgb, raw_source_ref, frame_map, component_signature)
    if len(raw) < 2:
        raise ModelStageError("PROPAINTER_SINGLE_FRAME_UNSUPPORTED")
    verify_ref(mask_receipt_ref)
    parent_receipt = json.loads(Path(mask_receipt_ref["path"]).read_text())
    from .contracts import verify_sealed_stage
    verify_sealed_stage(parent_receipt, parent_receipt.get("component_signature"),
                        output_root=Path(mask_receipt_ref["path"]).parent)
    if (parent_receipt.get("stage") != "Mask"
            or parent_receipt.get("actual_inputs", {}).get("frames_sha256") != frames_sha256(raw)
            or parent_receipt.get("actual_inputs", {}).get("frame_map_sha256") != canonical_sha(mapping)
            or parent_receipt.get("actual_inputs", {}).get("raw_source") != dict(raw_source_ref)):
        raise ModelStageError("CLEAN_UPSTREAM_RAW_OR_MAP_IDENTITY_CHANGED")
    if not model_refs:
        raise ModelStageError("PROPAINTER_WEIGHT_REFS_REQUIRED")
    for ref in model_refs:
        verify_ref(ref)
    context = {key: dict(ref) for key, ref in (clean_context_refs or {}).items()}
    if set(context) - {"capture_config", "raw_flow_receipt", "reference_evidence"}:
        raise ModelStageError("CLEAN_CONTEXT_ROLE_UNSUPPORTED")
    if type(record_reference_selection) is not bool:
        raise ModelStageError("CLEAN_REFERENCE_DIAGNOSTIC_MODE_REQUIRED")
    online = None
    if online_reference_context_provider is not None:
        if set(context) != {"capture_config"}:
            raise ModelStageError("SIGNED_ONLINE_CAPTURE_CONFIG_REQUIRED")
        online = OnlineReferenceContext(reference_selector, online_reference_context_provider,
                                        context, authorization, owner)
        reference_selector = online
    if (online is None and reference_selector is not None
            and not {"raw_flow_receipt", "reference_evidence"} <= set(context)):
        raise ModelStageError("SIGNED_FLOW_AND_REFERENCE_CONTEXT_REQUIRED")
    for ref in context.values():
        path = Path(ref["path"])
        if path.resolve(strict=True) != path or not path.is_relative_to(PROJECT_ROOT):
            raise ModelStageError("CLEAN_CONTEXT_PROJECT_REAL_PATH_REQUIRED")
        verify_ref(ref)
    removal, protected = masks.get("removal"), masks.get("protected_visible")
    if (not isinstance(removal, np.ndarray) or not isinstance(protected, np.ndarray)
            or removal.dtype != np.bool_ or protected.dtype != np.bool_
            or removal.shape != raw.shape[:3] or protected.shape != raw.shape[:3]):
        raise ModelStageError("CLEAN_FULL_NATIVE_MASK_DOMAIN_REQUIRED")
    mask_array_refs = [ref for ref in parent_receipt["outputs"]
                       if Path(ref["path"]).name == "MASK_ARRAYS.npz"]
    if len(mask_array_refs) != 1:
        raise ModelStageError("SEALED_MASK_ARRAYS_REQUIRED")
    with np.load(mask_array_refs[0]["path"], allow_pickle=False) as sealed:
        if any(name not in sealed or not np.array_equal(sealed[name], masks[name])
               for name in ("removal", "protected_visible")):
            raise ModelStageError("CLEAN_MASK_ARRAYS_DIFFER_FROM_SEALED_UPSTREAM")
        if sealed_effective_removal is not None:
            if (not isinstance(sealed_effective_removal, np.ndarray)
                    or sealed_effective_removal.dtype != np.bool_
                    or sealed_effective_removal.shape != raw.shape[:3]):
                raise ModelStageError("CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED")
            if ("effective_removal" not in sealed
                    or sealed["effective_removal"].dtype != np.bool_
                    or sealed["effective_removal"].shape != raw.shape[:3]
                    or not np.array_equal(sealed["effective_removal"], sealed_effective_removal)):
                raise ModelStageError("CLEAN_SEALED_EFFECTIVE_DIFFERS_FROM_UPSTREAM")
            _checked_sealed_effective(removal, protected, sealed_effective_removal)
    effective_options = ({} if sealed_effective_removal is None else
                         {"sealed_effective_removal": sealed_effective_removal})
    root = _new_stage(output_dir)
    begun = time.monotonic()
    try:
        wrapped = ReferenceSelectingProtectedProPainter(parent_propainter, reference_selector,
            record_reference_selection=record_reference_selection, **effective_options)
        _same_gate(authorization, owner)
        model_start = time.monotonic()
        clean = wrapped.inpaint(raw, removal, protected)
        if online is not None and not online.resolved:
            raise ModelStageError("ONLINE_REFERENCE_EVIDENCE_NOT_CONSUMED")
        _synchronize(parent_propainter)
        model_wall = time.monotonic() - model_start
        _same_gate(authorization, owner)
        effective = wrapped.wrapped.effective_masks
        if sealed_effective_removal is not None and (
                not isinstance(effective, np.ndarray) or effective.dtype != np.bool_
                or effective.shape != sealed_effective_removal.shape
                or not np.array_equal(effective, sealed_effective_removal)):
            raise ModelStageError("CLEAN_SEALED_EFFECTIVE_MODEL_DOMAIN_CHANGED")
        validation = validate_clean_preservation(raw, clean, effective, protected)
        if validation["status"] != "PASS":
            raise ModelStageError("CLEAN_PROTECTION_OR_OUTSIDE_EFFECTIVE_DOMAIN_CHANGED")
        outputs = [_npz(root / "CLEAN_ARRAYS.npz", {"clean_rgb": clean,
                    "effective_removal": effective, "protected_visible": protected}),
                   _json(root / "FRAME_MAP.json", mapping),
                   _json(root / "REFERENCE_SELECTION.json", wrapped.calls),
                   _json(root / "PRESERVATION.json", validation)]
        status = [{"native_frame_index": index, "stage_execution": "MODEL_PROVIDER_RETURNED",
                   "visual_quality": "NOT_EVALUATED", "training_eligible": False,
                   "outside_effective_delete_changed_pixels": validation[
                       "outside_delete_changed_pixels_per_frame"][index],
                   "protected_changed_pixels": validation["protected_changed_pixels_per_frame"][index]}
                  for index in range(len(raw))]
        outputs.append(_json(root / "FRAME_STATUS.json", status))
        for ref in [raw_source_ref, mask_receipt_ref, *model_refs, *context.values()]:
            verify_ref(ref)
        _same_gate(authorization, owner)
        receipt = {"schema": SCHEMA, "stage": "Clean", "status": "SEALED_COMPLETE",
            "component_signature": component_signature, "outputs": outputs,
            "actual_inputs": {"raw_source": dict(raw_source_ref),
                              "mask_receipt": dict(mask_receipt_ref),
                              "models": [dict(value) for value in model_refs],
                              "clean_context_refs": context,
                              "frames_sha256": frames_sha256(raw),
                              "frame_map_sha256": canonical_sha(mapping)},
            "authority": owner, "native_input_frames": len(raw), "native_output_frames": len(clean),
            "settings": dict(wrapped.wrapped.settings),
            "original_inpaint_function_sha256": wrapped.original_function_sha256,
            "reference_mode": "FROZEN_BASELINE" if reference_selector is None else "SINGLE_REF_SELECTION_CHANGE",
            "reference_selection_diagnostic_hook": record_reference_selection,
            "timings": {"ProPainter_inference_wall_seconds": model_wall,
                        "whole_stage_wall_seconds_including_IO": time.monotonic() - begun},
            "validation": validation, "visual_quality": "NOT_EVALUATED",
            "time_mode": "OFFLINE_BIDIRECTIONAL_PROPAINTER", "causal_training_input": False,
            "human_review": "PENDING", "training_eligible": False, "control_ground_truth": False,
            "fill_source": "MODEL_PROPAGATED_OR_GENERATED_NOT_HIDDEN_OBJECT_TRUTH"}
        if sealed_effective_removal is not None:
            receipt["actual_inputs"]["effective_removal_source"] = {
                "mask_arrays": dict(mask_array_refs[0]), "array_key": "effective_removal",
                "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}
        _json(root / "RECEIPT.json", receipt)
        return receipt
    except Exception as error:
        _failure(root, "Clean", error)
        raise
