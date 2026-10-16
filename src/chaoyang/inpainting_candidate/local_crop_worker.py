"""Frozen three-real-frame crop experiment; never publishes deletion masks."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, now_iso
from .automatic_masks import frames_sha256
from .contracts import canonical_sha, verify_ref, verify_sealed_stage
from .controller_observer import Sam31ControllerObserver, _json
from .model_stages import configure_project_caches
from .resources import CANDIDATE_CARD_LOCK_ROOT

SCHEMA = "inpainting-local-crop-model-result-v1"
SCOPE = "AUTOMATIC_LOCAL_CROP_CONTROLLER_ONE_CANDIDATE"
CALL_UNIT = "SAM_ADAPTER_INFERENCE_API_INVOCATION"
PLANNED_CROPS = 61
OUTPUT_LIMIT = 128 * 1024 * 1024
FINAL_RESERVE = 2 * 1024 * 1024
EXPECTED_PARAMETERS = {
    "score_threshold_detection": 0.4,
    "new_det_thresh": 0.65,
    "image_only_det_thresh": 0.5,
    "masklet_confirmation_enable": True,
    "masklet_confirmation_consecutive_det_thresh": 3,
    "hotstart_delay": 15,
}


def _require(value, code):
    if not value:
        raise ValueError(code)


def _stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.tzinfo is not None, "LOCAL_CROP_DEADLINE_TIMEZONE")
    return result


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _read_ref(reference):
    verify_ref(reference)
    return json.loads(Path(reference["path"]).read_text())


def _output_named(receipt, name):
    refs = [row for row in receipt["outputs"] if Path(row["path"]).name == name]
    _require(len(refs) == 1, "LOCAL_CROP_UNIQUE_SOURCE_OUTPUT:" + name)
    return refs[0]


def _bytes_on_disk(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _publish(root, folder, payloads, *, reserve=FINAL_RESERVE):
    """All payload sizes are known before any member of this bundle is written."""
    _require(_bytes_on_disk(root) + sum(map(len, payloads.values())) + reserve
             <= OUTPUT_LIMIT, "LOCAL_CROP_OUTPUT_CAP")
    folder.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, payload in payloads.items():
        _require(Path(name).name == name, "LOCAL_CROP_SAFE_OUTPUT_NAME")
        path = folder / name
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        refs.append(artifact_ref(path))
    return refs


class ScopedCropObserver(Sam31ControllerObserver):
    """Add call boundaries and confirmation data without changing inference."""
    def __init__(self, model, row, bindings):
        super().__init__(model, native_frames=3, sample_native_frames=(0, 1, 2),
                         bindings=bindings)
        self.row = row
        self.call_scope = "OUTSIDE_INFERENCE"

    def _record(self, phase, frame, **details):
        super()._record(phase, frame, **details)
        self.records[-1].update(
            event_index=len(self.records) - 1, call_scope=self.call_scope,
            local_frame_index=frame,
            source_native_frame_index=self.row["native_frames"][frame],
            crop_id=self.row["crop_id"])

    def _capture_planning(self, result, values):
        before = len(self.records)
        super()._capture_planning(result, values)
        if len(self.records) > before:
            metadata = result[1]
            self.records[-1]["masklet_confirmation"] = _json(
                metadata.get("rank0_metadata", {}).get("masklet_confirmation"))
            self.records[-1]["confirmation_object_ids"] = _json(
                metadata.get("obj_ids_all_gpu"))

    @contextmanager
    def _scope(self, value):
        previous = self.call_scope
        self.call_scope = value
        try:
            yield
        finally:
            self.call_scope = previous

    def _initial(self, original, args, kwargs, values):
        state = values.get("inference_state")
        _require(isinstance(state, dict) and state.get("num_frames") == 3
                 and state.get("is_image_only") is False,
                 "LOCAL_CROP_THREE_REAL_FRAME_VIDEO_REQUIRED")
        with self._scope("INITIAL_ADD_PROMPT"):
            return original(*args, **kwargs)

    def _propagate(self, original, args, kwargs, values):
        _require(values.get("reverse", False) is False,
                 "LOCAL_CROP_FORWARD_ONLY")
        with self._scope("PROPAGATE_FORWARD"):
            return (yield from original(*args, **kwargs))

    def __enter__(self):
        super().__enter__()
        try:
            _require(callable(getattr(self.model, "add_prompt", None))
                     and callable(getattr(self.model, "propagate_in_video", None)),
                     "LOCAL_CROP_PUBLIC_SCOPE_HOOKS_REQUIRED")
            self._patch(self.model, "add_prompt", self._initial)
            self._patch(self.model, "propagate_in_video", self._propagate)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise


def _load_sources(packet):
    """Reuse only exact Source and TEXT bytes; never open POINT/E/Clean."""
    sources = {}
    model_shas = {row["sha256"] for row in packet["model_weights"]}
    for row in packet["sessions"]:
        key = row["session_key"]
        _require(key not in sources, "LOCAL_CROP_DUPLICATE_SESSION")
        source = _read_ref(row["source_receipt"])
        verify_sealed_stage(source, source["component_signature"],
                            output_root=Path(row["source_receipt"]["path"]).parent)
        _require(source.get("stage") == "Source", "LOCAL_CROP_SOURCE_STAGE")
        refs = {name: _output_named(source, name) for name in
                ("RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json")}
        mapping = _read_ref(refs["FRAME_MAP.json"])
        report = _read_ref(refs["SOURCE_REPORT.json"])
        with np.load(refs["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
            rgb = archive["rgb"].copy()
        _require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3
                 and len(rgb) == row["native_frames"] == len(mapping),
                 "LOCAL_CROP_FULL_SOURCE_DOMAIN")
        for index, frame in enumerate(mapping):
            _require(frame["native_frame_index"] == index
                     and frame["pixel_sha256"] == hashlib.sha256(rgb[index].tobytes()).hexdigest(),
                     "LOCAL_CROP_SOURCE_NATIVE_PIXEL_MAP")
        text = _read_ref(row["text_receipt"])
        proposals = _read_ref(row["text_proposals"])
        verify_ref(row["text_arrays"])
        _require(proposals.get("stage") == "TEXT"
                 and proposals.get("arrays") == row["text_arrays"]
                 and proposals.get("coordinate_domain") == "NATIVE_RAW_RGB",
                 "LOCAL_CROP_TEXT_SEARCH_SOURCE")
        expected = {"frames_sha256": frames_sha256(rgb),
                    "frame_map_sha256": canonical_sha(mapping),
                    "raw_source_sha256": report["actual_inputs"]["video"]["sha256"]}
        _require(all(text.get(k) == v for k, v in expected.items())
                 and text.get("model_sha256") in model_shas,
                 "LOCAL_CROP_TEXT_SOURCE_BINDING")
        for proposal in proposals["proposals"]:
            _require(proposal.get("source") == "SAM31_RGB_TEXT"
                     and proposal.get("native_shape") == list(rgb.shape[:3])
                     and all(proposal.get(k) == v for k, v in expected.items())
                     and proposal.get("model_sha256") == text["model_sha256"]
                     and proposal.get("refinement_status") is None,
                     "LOCAL_CROP_TEXT_PROPOSAL_BINDING")
        sources[key] = {"rgb": rgb, "row": row, "mapping": mapping,
                        "source_refs": refs, "text": text}
    return sources


def _validate_crop(row, sources):
    _require(isinstance(row["crop_id"], str)
             and re.fullmatch(r"[A-Za-z0-9_-]+", row["crop_id"]) is not None,
             "LOCAL_CROP_SAFE_ID")
    _require(row["session_key"] in sources, "LOCAL_CROP_UNKNOWN_SESSION")
    rgb = sources[row["session_key"]]["rgb"]
    frames = row["native_frames"]
    rect = row["rect_xyxy"]
    _require(isinstance(frames, list) and len(frames) == 3
             and all(type(value) is int for value in frames)
             and frames == list(range(frames[0], frames[0] + 3))
             and 0 <= frames[0] <= frames[-1] < len(rgb),
             "LOCAL_CROP_REAL_CONTIGUOUS_NATIVE_FRAMES")
    _require(isinstance(rect, list) and len(rect) == 4
             and all(type(value) is int for value in rect),
             "LOCAL_CROP_INTEGER_RECTANGLE")
    x0, y0, x1, y1 = rect
    _require(0 <= x0 < x1 <= rgb.shape[2] and 0 <= y0 < y1 <= rgb.shape[1],
             "LOCAL_CROP_IN_BOUNDS")
    crops = np.ascontiguousarray(rgb[frames, y0:y1, x0:x1])
    _require(list(crops.shape) == row["crop_shape"]
             and [hashlib.sha256(image.tobytes()).hexdigest() for image in crops]
             == row["frame_pixel_sha256"], "LOCAL_CROP_MANIFEST_PIXEL_DRIFT")
    _require(isinstance(row.get("provenance"), list) and bool(row["provenance"]),
             "LOCAL_CROP_SEARCH_PROVENANCE_REQUIRED")
    return crops


def _pack_masks(obs, tracks, row, source_shape):
    """Keep raw masks on their own grid; exact output remaps have separate keys."""
    arrays, index, hashes = {}, {}, {}

    def add(key, mask, *, domain, source_key=None):
        value = np.asarray(mask)
        _require(value.dtype == np.bool_, "LOCAL_CROP_BOOLEAN_MASK_REQUIRED")
        bits = np.packbits(value.reshape(-1), bitorder="big")
        digest = hashlib.sha256(str(value.shape).encode() + bits.tobytes()).hexdigest()
        storage = hashes.get(digest)
        if storage is None:
            storage = "mask_" + str(len(arrays)).zfill(6)
            arrays[storage] = bits
            hashes[digest] = storage
        index[key] = {"array_key": storage, "shape": list(value.shape), "dtype": "bool",
                      "packing": "NUMPY_PACKBITS_C_ORDER_FLAT_BIG_BITORDER",
                      "domain": domain, "source_key": source_key}
        return key

    for key, value in obs.masks.items():
        add(key, value, domain=("DETECTOR_NATIVE_RESOLUTION"
            if "_raw_detector_" in key else "CROP_RGB_OUTPUT_RESOLUTION"))
    _, height, width, _ = row["crop_shape"]
    native_h, native_w = source_shape[1:3]
    x0, y0, x1, y1 = row["rect_xyxy"]

    def remap(key, mask):
        _require(mask.shape[-2:] == (height, width), "LOCAL_CROP_REMAP_GEOMETRY")
        native = np.zeros((*mask.shape[:-2], native_h, native_w), dtype=bool)
        native[..., y0:y1, x0:x1] = mask
        add(key + "_native", native, domain="NATIVE_RAW_RGB",
            source_key=key)

    for event in obs.records:
        if event["phase"] == "object_postprocess" and event.get("mask") is not None:
            key = event["mask"]["array_key"]
            remap(key, obs.masks[key])
            event["native_output_mask_key"] = key + "_native"
    final_map = []
    for obj, frames in sorted(tracks.items()):
        for frame, packed in sorted(frames.items()):
            _require(type(frame) is int and 0 <= frame < 3, "LOCAL_CROP_FINAL_FRAME")
            data = np.asarray(packed)
            _require(data.dtype == np.uint8 and data.ndim == 1
                     and data.size == (height * width + 7) // 8,
                     "LOCAL_CROP_FINAL_PACKED_MASK")
            mask = np.unpackbits(data, count=height * width).reshape(height, width).astype(bool)
            key = "final_obj_" + str(obj) + "_local_" + str(frame)
            add(key, mask, domain="CROP_RGB_OUTPUT_RESOLUTION")
            remap(key, mask)
            final_map.append({"object_id": int(obj), "local_frame_index": frame,
                              "source_native_frame_index": row["native_frames"][frame],
                              "array_key": key, "native_array_key": key + "_native"})
    return arrays, {"schema": "inpainting-local-crop-mask-index-v1",
        "masks": index, "final_tracks": final_map,
        "final_tracks_include_initial_output": True,
        "raw_mask_remapping_performed": False, "physical_ground_truth": False}


def summarize_crop(obs, tracks, row):
    per_frame = [{"local_frame_index": f, "source_native_frame_index": row["native_frames"][f],
                  "raw_joint_score_max": None, "presence_probabilities": [],
                  "initial_output_ids": [], "propagated_output_ids": []} for f in range(3)]
    supports, confirmed, propagated, initial, born = {}, set(), {}, set(), set()
    for event in obs.records:
        frame = event["local_frame_index"]
        current = per_frame[frame]
        phase = event["phase"]
        if phase == "raw_detector":
            scores = np.asarray(event["scores"], dtype=float)
            score = float(scores.max()) if scores.size else None
            if score is not None:
                old = current["raw_joint_score_max"]
                current["raw_joint_score_max"] = score if old is None else max(old, score)
            if event.get("presence_logit_dec") is not None:
                values = np.asarray(event["presence_logit_dec"], dtype=float)
                current["presence_probabilities"].append(
                    np.exp(-np.logaddexp(0, -values)).reshape(-1).tolist())
        if phase == "planning":
            plan = event.get("plan", {})
            new_ids = plan.get("new_det_obj_ids", [])
            for obj in new_ids:
                obj = int(obj)
                born.add(obj)
                supports.setdefault(obj, set()).add(frame)
            for ids in event.get("det_to_matched_original_query_indices", {}).values():
                for obj in ids:
                    supports.setdefault(int(obj), set()).add(frame)
        if phase == "object_pre_postprocess":
            confirmed.update(set(map(int, event.get("object_ids", [])))
                             - set(map(int, event.get("unconfirmed_obj_ids", []))))
        if phase == "object_postprocess":
            ids = list(map(int, event.get("object_ids", [])))
            if event["call_scope"] == "INITIAL_ADD_PROMPT":
                initial.update(ids)
                current["initial_output_ids"] = ids
            elif event["call_scope"] == "PROPAGATE_FORWARD":
                current["propagated_output_ids"] = ids
                for obj in ids:
                    propagated.setdefault(obj, set()).add(frame)
    final_ids = set(map(int, tracks))
    three_support = {obj for obj, frames in supports.items() if frames == {0, 1, 2}}
    three_output = {obj for obj, frames in propagated.items() if frames == {0, 1, 2}}
    return {"schema": "inpainting-local-crop-summary-v1", "crop_id": row["crop_id"],
        "session_key": row["session_key"], "native_frames": row["native_frames"],
        "rect_xyxy": row["rect_xyxy"], "per_frame": per_frame,
        "new_instance_ids": sorted(born), "vendor_confirmed_ids_observed": sorted(confirmed),
        "distinct_native_detection_support": {str(obj): sorted(row["native_frames"][f] for f in frames)
                                               for obj, frames in sorted(supports.items())},
        "propagated_output_native_frames": {str(obj): sorted(row["native_frames"][f] for f in frames)
                                             for obj, frames in sorted(propagated.items())},
        "three_real_frame_supported_and_propagated_ids": sorted(
            born & confirmed & three_support & three_output),
        "initial_only_ids": sorted(initial - set(propagated)),
        "final_tracks_ids": sorted(final_ids),
        "final_tracks_contains_initial_only_ids": sorted(final_ids & (initial - set(propagated))),
        "final_tracks_nonempty_is_quality_pass": False,
        "observer": obs.summary(), "visual_identity_review": "PENDING",
        "visual_quality": "NOT_REVIEWED", "candidate_adopted": False,
        "threshold_changes": False, "point_calls": 0, "clean_calls": 0, "robot_calls": 0}


def execute_local_crop(packet):
    started = time.monotonic()
    bounds = packet["budgets"]
    _require(packet["experiment_scope"] == SCOPE and packet["config"]["text"] == "VR controller",
             "LOCAL_CROP_SCOPE_OR_PROMPT")
    _require(bounds["model_calls_unit"] == CALL_UNIT
             and bounds["model_calls_max"] == 80
             and 0 < bounds["wall_seconds_max"] <= 900
             and bounds["output_bytes_estimate"] == OUTPUT_LIMIT,
             "LOCAL_CROP_FINITE_BUDGET")
    root = Path(packet["output_dir"])
    _require(root.is_absolute() and root.resolve() == root
             and root.is_relative_to(REPO_ROOT) and root != REPO_ROOT,
             "LOCAL_CROP_PROJECT_OUTPUT")
    deadline = min(_stamp(packet["deadline_at"]), _stamp(packet["global_deadline_at"]))
    definition = importlib.util.spec_from_file_location("_local_crop_worker_gate",
        REPO_ROOT / "scripts/maintenance/inpainting_local_crop_v1.py")
    gate = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(gate)
    infra, helpers = gate._infra(), gate._model_helpers()

    def authority():
        _require(datetime.now(timezone.utc) < deadline
                 and time.monotonic() - started < bounds["wall_seconds_max"],
                 "LOCAL_CROP_FINITE_DEADLINE")
        _, state, task_index = infra._snapshot()
        ledger, _ = helpers.own_row(packet, state, task_index)
        _require(ledger["status"] == "RUNNING", "LOCAL_CROP_RUNNING_FENCE")
        required = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                    "fencing_token": packet["writer"]["fencing_token"],
                    "pid": packet["writer"]["pid"],
                    "proc_start_ticks": packet["writer"]["proc_start_ticks"],
                    "uuid": packet["gpu_mapping"]["uuid"],
                    "physical_index": packet["gpu_mapping"]["physical_index"],
                    "worker_logical_index": 0}
        lease = json.loads((CANDIDATE_CARD_LOCK_ROOT / (required["uuid"] + ".json")).read_text())
        _require(all(lease.get(k) == v for k, v in required.items())
                 and os.environ.get("CUDA_VISIBLE_DEVICES") == required["uuid"],
                 "LOCAL_CROP_CARD_LEASE_DRIFT")
        return {**required, "execution_allowed": True, "algorithm_execution_allowed": True}

    authority()
    manifest_ref = packet["inputs"]["roi_manifest"]
    manifest = _read_ref(manifest_ref)
    rows = manifest["crops"]
    _require(len(rows) == PLANNED_CROPS
             and len({row["crop_id"] for row in rows}) == len(rows),
             "LOCAL_CROP_EXACT_FROZEN_61")
    sources = _load_sources(packet)
    for row in rows:
        _validate_crop(row, sources)
    authority()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "CropDiagnostics"
    output.mkdir(exist_ok=False)
    configure_project_caches(root)
    attempted, completed, summaries, output_refs, error = 0, 0, [], [], None
    sam = None
    try:
        from chaoyang.exact78_baseline.mask import Sam3Video
        loaded = time.monotonic()
        sam = Sam3Video(REPO_ROOT)
        output_refs += _publish(root, output, {"MODEL_LOAD.json": _json_bytes({
            "model_class": type(sam.model).__name__, "adapter_evidence": sam.evidence,
            "model_weights": packet["model_weights"], "loaded_once": True,
            "load_wall_seconds": time.monotonic() - loaded})})
        authority()
        for row in rows:
            authority()
            _require(attempted < bounds["model_calls_max"], "LOCAL_CROP_API_CAP")
            _require(_bytes_on_disk(root) + FINAL_RESERVE < OUTPUT_LIMIT, "LOCAL_CROP_OUTPUT_CAP")
            crops = _validate_crop(row, sources)
            bindings = {"run_signature": packet["run_signature"],
                        "roi_manifest": manifest_ref, "crop": row,
                        "source_receipt": sources[row["session_key"]]["row"]["source_receipt"],
                        "observer_frame_domain": "LOCAL_CROP_INDEX_MAPPED_TO_REAL_NATIVE"}
            obs = ScopedCropObserver(sam.model, row, bindings)
            _require(all(obs.parameters.get(k) == v for k, v in EXPECTED_PARAMETERS.items()),
                     "LOCAL_CROP_FROZEN_THRESHOLDS_DRIFT")
            before_parameters = dict(obs.parameters)
            call_start = time.monotonic()
            attempted += 1
            with obs:
                tracks = sam.track_text(list(crops), "VR controller", 0)
            after = {name: _json(getattr(sam.model, name)) for name in before_parameters}
            _require(after == before_parameters, "LOCAL_CROP_PARAMETER_MUTATION")
            coverage = obs.summary()["coverage"]
            phases = ("raw_detector", "nms", "detection", "planning",
                      "object_pre_postprocess", "object_postprocess")
            _require(not obs.map_errors and coverage["complete"]
                     and all(coverage["phase_native_frames"].get(phase) == [0, 1, 2]
                             for phase in phases), "LOCAL_CROP_TELEMETRY_INCOMPLETE")
            _require(not sam.aborted, "LOCAL_CROP_PROPAGATION_ABORTED")
            authority()
            arrays, mask_index = _pack_masks(obs, tracks, row,
                                             sources[row["session_key"]]["rgb"].shape)
            summary = summarize_crop(obs, tracks, row)
            summary.update(execution_complete=True, model_api_calls=1,
                           inference_wall_seconds=time.monotonic() - call_start,
                           propagation_aborted=list(sam.aborted))
            folder = output / row["crop_id"]
            folder.mkdir(exist_ok=False)
            files = _publish(root, folder, {
                "INPUT.json": _json_bytes(bindings),
                "EVENTS.json": _json_bytes(obs.records),
                "MASKS.npz": _npz_bytes(arrays),
                "MASK_INDEX.json": _json_bytes(mask_index),
                "SUMMARY.json": _json_bytes(summary)})
            receipt = {"schema": "inpainting-candidate-component-v1",
                "stage": "LocalCropController", "status": "SEALED_COMPLETE",
                "component_signature": canonical_sha({"run_signature": packet["run_signature"],
                    "crop": row, "roi_manifest": manifest_ref}),
                "native_input_frames": 3, "native_output_frames": 3,
                "actual_inputs": bindings, "outputs": files,
                "visual_quality": "NOT_REVIEWED", "candidate_adopted": False,
                "manual_seed_reads": 0, "point_calls": 0, "clean_calls": 0,
                "robot_calls": 0, "threshold_changes": False, "completed_at": now_iso()}
            child = _publish(root, folder, {"RECEIPT.json": _json_bytes(receipt)})[0]
            output_refs.extend([*files, child])
            completed += 1
            summaries.append({"crop_id": row["crop_id"], "session_key": row["session_key"],
                "receipt": child,
                "three_real_frame_supported_and_propagated_ids":
                    summary["three_real_frame_supported_and_propagated_ids"],
                "new_instance_ids": summary["new_instance_ids"]})
            print(json.dumps({"stage": "LOCAL_CROP_COMPLETED", "crop_id": row["crop_id"],
                              "completed": completed, "planned": len(rows),
                              "api_calls": attempted}, ensure_ascii=False), flush=True)
            del crops, arrays, mask_index, obs, tracks, summary
        authority()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        if sam is not None:
            sam.close()
    complete = error is None and completed == len(rows)
    index_ref = _publish(root, output, {"CROP_INDEX.json": _json_bytes(summaries)}, reserve=0)[0]
    receipt = {"schema": "inpainting-candidate-component-v1", "stage": "LocalCropController",
        "status": "SEALED_COMPLETE" if complete else "PARTIAL_FAILED",
        "component_signature": canonical_sha({"run_signature": packet["run_signature"],
            "roi_manifest": manifest_ref, "stage": "LocalCropController"}),
        "native_input_frames": 3 * len(rows), "native_output_frames": 3 * completed,
        "actual_inputs": {"roi_manifest": manifest_ref,
                         "source_receipts": [row["source_receipt"] for row in packet["sessions"]]},
        "outputs": [index_ref, *output_refs],
        "planned_crops": len(rows), "completed_crops": completed,
        "visual_quality": "NOT_REVIEWED", "candidate_adopted": False,
        "threshold_changes": False, "point_calls": 0, "clean_calls": 0,
        "robot_calls": 0, "manual_seed_reads": 0, "error": error,
        "partial_consumable": False, "completed_at": now_iso()}
    diagnostics_ref = _publish(root, output, {"RECEIPT.json": _json_bytes(receipt)}, reserve=0)[0]
    result = {"schema": SCHEMA, "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "experiment_scope": SCOPE,
        "planned_crops": len(rows), "completed_crops": completed,
        "diagnostics_receipt": diagnostics_ref, "execution_complete": complete,
        "model_call_accounting": {"unit": CALL_UNIT, "attempted_api_calls": attempted,
                                  "maximum": bounds["model_calls_max"]},
        "point_calls": 0, "clean_calls": 0, "robot_calls": 0, "manual_seed_reads": 0,
        "threshold_changes": False, "candidate_adopted": False,
        "visual_quality": "NOT_REVIEWED", "files_deleted": 0, "training_steps": 0,
        "error": error, "whole_wall_seconds": time.monotonic() - started,
        "completed_at": now_iso()}
    _publish(root, root, {"MODEL_RESULT.json": _json_bytes(result)}, reserve=0)
    return 0 if complete else 2

