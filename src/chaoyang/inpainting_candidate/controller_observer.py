"""Temporary, instance-local telemetry for one unchanged SAM 3.1 text replay.

No vendor import, model load, inference, threshold change or output replacement
is performed here. Raw detector masks retain their detector resolution; their
positive-logit decoding is not a new detection or quality gate.
"""
from __future__ import annotations

import inspect
import sys
from collections import Counter

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    # NumPy has no torch bfloat16 representation. float() is an out-of-place
    # conversion of the detached view, preserving the original model tensor.
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(item) for item in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        return _json(_array(value).tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


class Sam31ControllerObserver:
    """Observe actual frame identities, copying only sampled mask arrays.

    ``model`` is the already loaded SAM multiplex predictor, not its adapter.
    Telemetry failures are reported in ``map_errors`` and never replace model
    results. Context exit restores both instance methods and the detector's
    already imported NMS alias, including after an inference exception.
    """
    PARAMETER_NAMES = (
        "score_threshold_detection", "new_det_thresh", "det_nms_thresh",
        "det_nms_use_iom", "image_only_det_thresh", "hotstart_delay",
        "hotstart_unmatch_thresh", "hotstart_dup_thresh", "max_num_objects",
        "masklet_confirmation_enable", "masklet_confirmation_consecutive_det_thresh",
        "suppress_unmatched_only_within_hotstart", "suppress_det_close_to_boundary",
        "postprocess_batch_size", "use_batched_grounding", "batched_grounding_batch_size",
        "running_in_prod", "compile_model", "rank", "world_size",
    )

    def __init__(self, model, *, native_frames, sample_native_frames, bindings=None):
        if type(native_frames) is not int or native_frames < 1:
            raise ValueError("OBSERVER_NATIVE_FRAME_COUNT_REQUIRED")
        samples = tuple(sample_native_frames)
        if any(type(frame) is not int or not 0 <= frame < native_frames for frame in samples):
            raise ValueError("OBSERVER_SAMPLE_NATIVE_FRAME_DOMAIN")
        self.model = model
        self.native_frames = native_frames
        self.sample_native_frames = tuple(sorted(set(samples)))
        self.bindings = _json({} if bindings is None else bindings)
        self.records, self.masks, self.map_errors = [], {}, []
        self.parameters = {name: _json(getattr(model, name)) for name in self.PARAMETER_NAMES
                           if hasattr(model, name)}
        self._patches, self._branches = [], Counter()
        self._base_context, self._chunk_context = None, None
        self._out_frames, self._permutations = {}, {}
        self._observed, self._active = set(), False
        self._mask_serial = 0

    def _error(self, code, **details):
        self.map_errors.append({"code": code, **_json(details)})

    def _safe(self, callback, *args):
        try:
            callback(*args)
        except Exception as exc:
            self._error("TELEMETRY_CAPTURE_FAILED", operation=callback.__name__,
                        exception=type(exc).__name__, detail=str(exc))

    @staticmethod
    def _arguments(original, args, kwargs):
        try:
            return dict(inspect.signature(original).bind_partial(*args, **kwargs).arguments)
        except (TypeError, ValueError):
            return dict(kwargs)

    def _native(self, value, phase):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            self._error("NATIVE_FRAME_MAPPING_MISSING", phase=phase, value=_json(value))
            return None
        frame = int(value)
        if not 0 <= frame < self.native_frames:
            self._error("NATIVE_FRAME_OUT_OF_DOMAIN", phase=phase, frame=frame)
            return None
        return frame

    def _record(self, phase, frame, **details):
        self.records.append({"phase": phase, "native_frame_index": frame,
                             **_json(details)})

    def _mask(self, phase, frame, value, *, kind):
        if frame not in self.sample_native_frames:
            return None
        self._mask_serial += 1
        key = f"f{frame:06d}_{phase}_{kind}_{self._mask_serial:04d}"
        array = _array(value)
        if kind == "decoder_positive_logits":
            array = array > 0
        self.masks[key] = array.copy()
        return {"array_key": key, "shape": list(array.shape), "dtype": str(array.dtype),
                "domain": "DETECTOR_NATIVE_RESOLUTION" if phase == "raw_detector"
                else "VENDOR_OUTPUT_RESOLUTION", "pixels_source": "ORIGINAL_MODEL_OUTPUT"}

    def _patch(self, owner, name, handler):
        original = getattr(owner, name, None)
        if not callable(original):
            return False
        own = name in getattr(owner, "__dict__", {})

        def wrapped(*args, **kwargs):
            self._branches[name] += 1
            values = self._arguments(original, args, kwargs)
            return handler(original, args, kwargs, values)

        setattr(owner, name, wrapped)
        self._patches.append((owner, name, original, own))
        return True

    def __enter__(self):
        if self._active or getattr(self.model, "_chaoyang_controller_observer", None) is not None:
            raise ValueError("OBSERVER_ALREADY_ACTIVE")
        self._active = True
        setattr(self.model, "_chaoyang_controller_observer", self)
        try:
            self._patch(self.model, "run_backbone_and_detection", self._detection)
            self._patch(self.model, "run_tracker_update_planning_phase", self._planning)
            self._patch(self.model, "_run_single_frame_inference", self._single_frame)
            self._patch(self.model, "_postprocess_output", self._postprocess)
            self._patch(self.model, "_postprocess_output_batched", self._postprocess_batched)
            detector = getattr(self.model, "detector", None)
            if detector is not None:
                self._patch(detector, "_build_multigpu_buffer_next_chunk", self._single_chunk)
                self._patch(detector, "_process_grounding_chunk_batched", self._batch_chunk)
                self._patch(detector, "forward_grounding", self._raw_grounding)
                module = sys.modules.get(type(detector).__module__)
                if module is not None:
                    self._patch(module, "nms_masks", self._nms)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback):
        for owner, name, original, own in reversed(self._patches):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._patches.clear()
        if getattr(self.model, "_chaoyang_controller_observer", None) is self:
            delattr(self.model, "_chaoyang_controller_observer")
        self._active = False
        return False

    def _single_chunk(self, original, args, kwargs, values):
        previous = self._chunk_context
        begin, end = values.get("frame_idx_begin"), values.get("frame_idx_end")
        rank = getattr(getattr(self.model, "detector", None), "rank", None)
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool)
               for value in (begin, end, rank)) and end > begin:
            self._chunk_context = [min(int(begin) + int(rank), int(end) - 1)]
        else:
            self._chunk_context = None
            self._error("SINGLE_CHUNK_FRAME_MAPPING_MISSING")
        try:
            return original(*args, **kwargs)
        finally:
            self._chunk_context = previous

    def _batch_chunk(self, original, args, kwargs, values):
        previous = self._chunk_context
        start, end = values.get("chunk_start"), values.get("chunk_end")
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool)
               for value in (start, end)) and end >= start:
            self._chunk_context = list(range(int(start), int(end)))
        else:
            self._chunk_context = None
            self._error("BATCH_CHUNK_FRAME_MAPPING_MISSING")
        try:
            return original(*args, **kwargs)
        finally:
            self._chunk_context = previous

    def _raw_grounding(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_raw, result)
        return result

    def _capture_raw(self, result):
        out = result[0] if isinstance(result, tuple) else result
        logits = _array(out["pred_logits"])
        frames = self._chunk_context
        if frames is None or logits.ndim != 3 or len(frames) != len(logits):
            self._error("RAW_BATCH_NATIVE_FRAME_MAPPING_MISSING", shape=list(logits.shape),
                        chunk_frames=frames)
            return
        for batch, candidate in enumerate(frames):
            frame = self._native(candidate, "raw_detector")
            if frame is None:
                continue
            raw = logits[batch, :, 0]
            # Numerically stable sigmoid; records preserve logits as well.
            scores = np.exp(-np.logaddexp(0, -raw.astype(np.float64)))
            mask = self._mask("raw_detector", frame, out["pred_masks"][batch],
                              kind="decoder_positive_logits")
            presence = out.get("presence_logit_dec")
            self._record("raw_detector", frame, query_indices=list(range(len(raw))),
                         pred_logits=raw, scores=scores,
                         presence_logit_dec=None if presence is None else presence[batch],
                         pred_boxes_xyxy=out.get("pred_boxes_xyxy", [None] * len(frames))[batch],
                         mask=mask, reverse=None if self._base_context is None else self._base_context[1],
                         score_basis="VENDOR_PRESENCE_JOINT_PRED_LOGITS_BEFORE_NMS")

    def _nms(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_nms, result, values)
        return result

    def _capture_nms(self, result, values):
        keep = _array(result)
        rows = keep[None] if keep.ndim == 1 else keep
        frames = self._chunk_context
        if frames is None or len(rows) != len(frames):
            self._error("NMS_NATIVE_FRAME_MAPPING_MISSING", shape=list(keep.shape), chunk_frames=frames)
            return
        for candidate, row in zip(frames, rows):
            frame = self._native(candidate, "nms")
            if frame is not None:
                self._record("nms", frame, keep=row, prob_threshold=values.get("prob_threshold"),
                             iou_threshold=values.get("iou_threshold"),
                             nms_use_iom=values.get("nms_use_iom"))

    def _detection(self, original, args, kwargs, values):
        frame = self._native(values.get("frame_idx"), "detection")
        reverse = bool(values.get("reverse", False))
        previous = self._base_context
        self._base_context = (frame, reverse)
        try:
            result = original(*args, **kwargs)
            self._safe(self._capture_detection, result, frame, reverse)
            return result
        finally:
            self._base_context = previous

    def _capture_detection(self, result, frame, reverse):
        if frame is None:
            return
        det_out, keep = result
        row = keep[0] if len(getattr(keep, "shape", ())) == 2 else keep
        if hasattr(row, "detach"):
            permutation = _array(row.argsort(descending=True))
            basis = "VENDOR_POS_PRED_MASK_ARGSORT_DESCENDING"
        else:
            permutation = np.argsort(-_array(row).astype(np.int8), kind="stable")
            basis = "SYNTHETIC_NUMPY_STABLE_DESCENDING"
        self._permutations[(frame, reverse)] = permutation
        self._observed.add(frame)
        threshold = getattr(self.model, "score_threshold_detection", None)
        score_keep = None if threshold is None else det_out["scores"] > threshold
        self._record("detection", frame, reverse=reverse, scores=det_out.get("scores"),
                     bbox=det_out.get("bbox"), score_gate_keep=score_keep,
                     combined_keep_after_score_boundary=row, post_score_keep=row,
                     query_permutation=permutation, permutation_basis=basis,
                     query_indices_domain="ORIGINAL_DETECTOR_ORDER", runtime_threshold=self.parameters.get("score_threshold_detection"))

    def _planning(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_planning, result, values)
        return result

    def _capture_planning(self, result, values):
        frame = self._native(values.get("frame_idx"), "planning")
        if frame is None:
            return
        reverse = bool(values.get("reverse", False))
        plan, metadata = result
        keys = ("new_det_fa_inds", "new_det_obj_ids", "new_det_gpu_ids", "unmatched_trk_obj_ids",
                "det_to_matched_trk_obj_ids", "obj_ids_newly_removed", "num_obj_dropped_due_to_limit",
                "reconditioned_obj_ids")
        selected = {name: plan[name] for name in keys if name in plan}
        permutation = self._permutations.get((frame, reverse))
        indices = _array(plan.get("new_det_fa_inds", [])).astype(int)
        raw_indices = None
        if permutation is not None and (not len(indices) or (indices.min() >= 0 and indices.max() < len(permutation))):
            raw_indices = permutation[indices]
        else:
            self._error("PLANNING_QUERY_PERMUTATION_MISSING", frame=frame)
        rank0 = metadata.get("rank0_metadata", {})
        matched_original = {}
        for index, ids in plan.get("det_to_matched_trk_obj_ids", {}).items():
            if (permutation is not None and isinstance(index, (int, np.integer))
                    and not isinstance(index, bool) and 0 <= index < len(permutation)):
                matched_original[str(int(permutation[index]))] = ids
            else:
                self._error("MATCHED_PLANNING_QUERY_PERMUTATION_MISSING", frame=frame, index=_json(index))
        self._record("planning", frame, reverse=reverse, plan=selected,
                     new_det_original_query_indices=raw_indices, det_keep=values.get("det_keep"),
                     det_to_matched_original_query_indices=matched_original,
                     sorted_scores=values.get("det_out", {}).get("scores"),
                     object_ids=metadata.get("obj_ids_all_gpu"),
                     removed_obj_ids=rank0.get("removed_obj_ids"),
                     suppressed_obj_ids=rank0.get("suppressed_obj_ids", {}).get(frame))

    def _single_frame(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_single, result, values)
        return result

    def _capture_single(self, out, values):
        frame = self._native(values.get("frame_idx"), "object_pre_postprocess")
        if frame is None:
            return
        self._out_frames[id(out)] = (frame, bool(values.get("reverse", False)))
        self._record("object_pre_postprocess", frame, object_ids=sorted(out.get("obj_id_to_mask", {})),
                     scores=out.get("obj_id_to_score"), sam2_scores=out.get("obj_id_to_sam2_score"),
                     removed_obj_ids=out.get("removed_obj_ids"), suppressed_obj_ids=out.get("suppressed_obj_ids"),
                     unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"), frame_stats=out.get("frame_stats"))

    def _postprocess(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_post, result, values.get("out"), values)
        return result

    def _capture_post(self, result, out, values):
        mapped = self._out_frames.pop(id(out), None)
        if mapped is None:
            self._error("POSTPROCESS_OUT_NATIVE_FRAME_MAPPING_MISSING")
            return
        frame, reverse = mapped
        mask = self._mask("object_postprocess", frame, result["out_binary_masks"], kind="binary_masks")
        self._record("object_postprocess", frame, reverse=reverse,
                     object_ids=result.get("out_obj_ids"), scores=result.get("out_probs"),
                     mask=mask, removed_obj_ids=values.get("removed_obj_ids"),
                     suppressed_obj_ids=values.get("suppressed_obj_ids"),
                     unconfirmed_obj_ids=values.get("unconfirmed_obj_ids"))

    def _postprocess_batched(self, original, args, kwargs, values):
        result = original(*args, **kwargs)
        self._safe(self._capture_post_batch, result, values)
        return result

    def _capture_post_batch(self, result, values):
        batched = values.get("batched_outs", [])
        if len(batched) != len(result):
            self._error("POSTPROCESS_BATCH_LENGTH_MISMATCH")
            return
        for output, row in zip(result, batched):
            out, removed, suppressed, unconfirmed = row
            self._capture_post(output, out, {"removed_obj_ids": removed,
                "suppressed_obj_ids": suppressed, "unconfirmed_obj_ids": unconfirmed})

    def summary(self):
        phase_counts = Counter(row["phase"] for row in self.records)
        frames = {phase: sorted({row["native_frame_index"] for row in self.records if row["phase"] == phase})
                  for phase in phase_counts}
        losses = []
        for row in self.records:
            phase = row["phase"]
            if phase == "nms":
                keep = np.asarray(row["keep"], bool)
                if (~keep).any():
                    losses.append({"native_frame_index": row["native_frame_index"], "stage": "NMS",
                                   "rejected_query_indices": np.flatnonzero(~keep).tolist()})
            if phase == "detection":
                keep = np.asarray(row["post_score_keep"], bool)
                if (~keep).any():
                    losses.append({"native_frame_index": row["native_frame_index"], "stage": "POST_SCORE_AND_BOUNDARY",
                                   "rejected_query_indices": np.flatnonzero(~keep).tolist()})
            if phase == "object_postprocess":
                hidden = {name: row.get(name) for name in ("removed_obj_ids", "suppressed_obj_ids", "unconfirmed_obj_ids")
                          if row.get(name)}
                if hidden:
                    losses.append({"native_frame_index": row["native_frame_index"], "stage": "OBJECT_POSTPROCESS",
                                   "vendor_hide_reasons": hidden})
        return {"schema": "inpainting-sam31-controller-observer-v1",
                "total_native_frames": self.native_frames,
                "sample_native_frames": list(self.sample_native_frames),
                "observed_native_frames": sorted(self._observed),
                "coverage": {"complete": len(self._observed) == self.native_frames,
                             "missing_native_frames": sorted(set(range(self.native_frames)) - self._observed),
                             "phase_native_frames": frames},
                "parameters": self.parameters, "bindings": self.bindings,
                "branches": dict(self._branches), "phase_event_counts": dict(phase_counts),
                "loss_stages": losses, "map_errors": list(self.map_errors),
                "mask_array_count": len(self.masks),
                "mask_bytes": sum(value.nbytes for value in self.masks.values()),
                "observer_active": self._active,
                "thresholds_changed": False, "model_calls_added": 0,
                "outputs_returned_without_replacement": True,
                "guarantee_limit": "WRAPPERS_PASS_ORIGINAL_ARGUMENTS_AND_RETURN_ORIGINAL_OBJECTS; NO_CROSS_REPLAY_DETERMINISM_CLAIM",
                "masks_are_physical_ground_truth": False,
                "telemetry_is_model_prediction_only": True}
