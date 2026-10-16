"""One finite frozen G2 Clean experiment, dispatched through the existing gate.

A runs actual ProtectedProPainter on all 238 frames. Actual Raw RAFT/FB
diagnostics are sealed separately. Background registration remains ABSENT;
therefore B is skipped, not counted as another model execution or improvement.
This module is not a standalone production or model-running entry.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np

from chaoyang.exact78_baseline.clean import SETTINGS
from .automatic_masks import frames_sha256
from .clean_flow import (CAPTURE_SCHEMA, RawRaftCaptureReplay, output_named,
                         sealed_receipt, seal_reference_evidence, write_json)
from .contracts import ContractError, canonical_sha, file_ref, verify_ref
from .model_stages import _gate, _same_gate, load_frozen_models, run_clean_stage
from .source_frames import write_review_video

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "inpainting-g2-clean-batch-result-v1"
STAGE_SCOPE = "G2_CLEAN_REFERENCE_SELECTION_AB_ONLY"
MODEL_CALL_UNIT = "PROPAINTER_INPAINT_API_INVOCATION"
NATIVE_SCOPE = "INPAINTING_NATIVE_BASELINE_CLEAN"
NATIVE_RESULT_SCHEMA = "inpainting-native-clean-batch-result-v1"
NATIVE_REFERENCE_POLICY = "RAW_VISIBLE_FB_RANKED_REFERENCE_EXPERIMENT"


def _packet_contract(packet: Mapping) -> None:
    if packet.get("experiment_scope") == NATIVE_SCOPE:
        from .native_contracts import validate_native_clean_spec
        validate_native_clean_spec(packet, repo_root=PROJECT_ROOT)
        if packet["clean_config"]["frozen_settings"] != SETTINGS:
            raise ContractError("NATIVE_CLEAN_FROZEN_SETTINGS_DRIFT")
        return
    if (packet.get("spec_schema") != "inpainting-finite-g2-clean-spec-v1"
            or packet.get("model_stages") != ["Clean"]
            or packet.get("actual_stage_scope") != STAGE_SCOPE):
        raise ContractError("FINITE_G2_CLEAN_PACKET_REQUIRED")
    sessions = packet.get("sessions", [])
    if len(sessions) != 1 or sessions[0].get("session_key") != "001" or sessions[0].get("native_frames") != 238:
        raise ContractError("FIXED_FULL_001_238_REQUIRED")
    config = packet.get("clean_config", {})
    if (config.get("frozen_settings") != SETTINGS or config.get("background_registration") != "ABSENT"
            or config.get("reference_policy") != "RAW_FB_DIAGNOSTIC_UNKNOWN_NO_QUALIFIED_SUPPORT"
            or config.get("evaluation_anchor_frames") != [0, 50, 119, 132, 202, 237]
            or config.get("candidates") != ["A", "B_IF_REFS_CHANGED"]):
        raise ContractError("FROZEN_G2_CLEAN_RECIPE_DRIFT")
    budget = packet.get("budgets", {})
    if (budget.get("model_calls_unit") != MODEL_CALL_UNIT or type(budget.get("model_calls_max")) is not int
            or not 1 <= budget["model_calls_max"] <= 2):
        raise ContractError("PROPAINTER_OUTER_CALL_BUDGET_REQUIRED")
    root = Path(packet["output_dir"]).resolve()
    if root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ContractError("CLEAN_BATCH_OUTPUT_OUTSIDE_PROJECT")


def clean_component_signature(packet: Mapping, session: Mapping, context: Mapping, *, candidate: str = "A") -> str:
    return canonical_sha({"schema": RESULT_SCHEMA, "stage": "Clean", "candidate": candidate,
        "source_receipt": session["source_receipt"], "mask_receipt": session["mask_receipt"],
        "raw_source_ref": session["raw_source_ref"], "frame_map_ref": session["frame_map_ref"],
        "native_frames": session["native_frames"], "runtime_code": packet["runtime_code"],
        "model_weights": packet["model_weights"], "config": packet["config"],
        "clean_config": packet["clean_config"], "clean_context_refs": dict(context)})


def load_native_inputs(packet: Mapping) -> tuple[np.ndarray, list[dict], dict, dict]:
    row = packet["sessions"][0]
    source = sealed_receipt(row["source_receipt"], expected_stage="Source")
    mask = sealed_receipt(row["mask_receipt"], expected_stage="Mask")
    raw_ref = output_named(source, "RAW_RGB.npz")
    map_ref = output_named(source, "FRAME_MAP.json")
    if source.get("native_frames") != 238 or map_ref != row["frame_map_ref"]:
        raise ContractError("CLEAN_SOURCE_FULL_NATIVE_MAP_REQUIRED")
    for ref in source.get("actual_inputs", []):
        verify_ref(ref)
    report = json.loads(Path(output_named(source, "SOURCE_REPORT.json")["path"]).read_text(encoding="utf-8"))
    for ref in report["actual_inputs"].values():
        verify_ref(ref)
    if report["actual_inputs"]["video"] != row["raw_source_ref"]:
        raise ContractError("CLEAN_RAW_SOURCE_BINDING_MISMATCH")
    with np.load(raw_ref["path"], allow_pickle=False) as archive:
        rgb = archive["rgb"]
    mapping = json.loads(Path(map_ref["path"]).read_text(encoding="utf-8"))
    if (rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != 238 or rgb.shape[-1] != 3
            or len(mapping) != 238 or not isinstance(mapping, list)):
        raise ContractError("CLEAN_FULL_NATIVE_RGB_AND_MAP_REQUIRED")
    for index, value in enumerate(mapping):
        if any(value.get(field) != index for field in (
                "native_frame_index", "decoded_display_index", "model_input_frame", "output_frame")):
            raise ContractError("CLEAN_NATIVE_FRAME_MAPPING_CHANGED")
    actual = mask.get("actual_inputs", {})
    if (mask.get("native_input_frames") != 238 or mask.get("native_output_frames") != 238
            or actual.get("raw_source") != row["raw_source_ref"]
            or actual.get("frames_sha256") != frames_sha256(rgb)
            or actual.get("frame_map_sha256") != canonical_sha(mapping)):
        raise ContractError("CLEAN_MASK_SOURCE_OR_FRAME_IDENTITY_CHANGED")
    with np.load(output_named(mask, "MASK_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        keys = ("removal", "effective_removal", "protected_visible", "core", "equipment", "unknown", "held")
        masks = {key: archive[key] for key in keys if key in archive}
    if not {"removal", "effective_removal", "protected_visible"} <= set(masks):
        raise ContractError("SEALED_NATIVE_CLEAN_MASKS_REQUIRED")
    if any(value.dtype != np.bool_ or value.shape != rgb.shape[:3] for value in masks.values()):
        raise ContractError("CLEAN_NATIVE_MASK_DOMAIN_CHANGED")
    capture_ref = packet["clean_config"]["capture_config_ref"]
    verify_ref(capture_ref)
    capture = json.loads(Path(capture_ref["path"]).read_text(encoding="utf-8"))
    if (capture.get("schema") != CAPTURE_SCHEMA or capture.get("session_id") != report["session_id"]
            or capture.get("native_shape") != list(rgb.shape[:3])
            or capture.get("source_receipt") != row["source_receipt"]
            or capture.get("raw_source_ref") != row["raw_source_ref"]
            or capture.get("raw_rgb_ref") != raw_ref or capture.get("frame_map_ref") != map_ref
            or capture.get("runtime_code") != packet["runtime_code"]
            or capture.get("raft_weight_ref") not in packet["model_weights"]):
        raise ContractError("CAPTURE_CONFIG_NOT_CURRENT_SOURCE_CLOSURE")
    return rgb, mapping, masks, report


def clean_review(raw: np.ndarray, masks: Mapping, clean: np.ndarray) -> np.ndarray:
    overlay = raw.copy()
    for name, color in (("effective_removal", (255, 60, 60)), ("protected_visible", (60, 255, 60)),
                        ("unknown", (255, 220, 40))):
        mask = masks.get(name)
        if mask is not None:
            overlay[mask] = ((overlay[mask].astype(np.uint16) + np.array(color, np.uint16)) // 2).astype(np.uint8)
    return np.concatenate((raw, overlay, clean), axis=2)


def run_clean_batch(packet: Mapping, *, authority: Callable) -> dict:
    """The existing operation gate/lease must precede this dispatch."""
    if packet.get("experiment_scope") == NATIVE_SCOPE:
        return run_native_clean_batch(packet, authority=authority)
    _packet_contract(packet)
    owner = _gate(authority)
    if owner["task_id"] != packet["task_id"] or owner["run_signature"] != packet["run_signature"]:
        raise ContractError("CLEAN_BATCH_AUTHORITY_IDENTITY_DRIFT")
    root = Path(packet["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_refs = [*packet["runtime_code"], *packet["model_weights"]]
    for ref in frozen_refs:
        verify_ref(ref)
    raw, mapping, masks, report = load_native_inputs(packet)
    row = packet["sessions"][0]
    out = root / "sessions" / row["session_key"]
    out.mkdir(parents=True, exist_ok=False)
    attempted, begun = 0, time.monotonic()
    status = {"session_key": "001", "session_id": report["session_id"], "native_frames": 238,
        "source": row["source_receipt"], "mask": row["mask_receipt"],
        "raw_source": row["raw_source_ref"], "frame_map": row["frame_map_ref"],
        "candidates": {}, "shared": {}, "visual_quality": "NOT_REVIEWED",
        "training_eligible": False, "control_ground_truth": False,
        "render": "NOT_EXECUTED", "composite": "NOT_EXECUTED", "motion": "NOT_EXECUTED",
        "depth": "NOT_CONSUMED", "object6d": "NOT_CONSUMED", "manual_seed_reads": 0}
    capture = None
    try:
        models = load_frozen_models(PROJECT_ROOT, root / "model_load", authorization=authority,
            frozen_inputs=frozen_refs, need_sam=False, need_clean=True)
        write_json(root / "MODEL_LOAD.json", {"task_id": packet["task_id"],
            "run_signature": packet["run_signature"], "load_receipt": models["load_receipt"],
            "inference_api_calls": 0, "visual_quality": "NOT_EVALUATED"})
        capture = RawRaftCaptureReplay(models["propainter"], raw_rgb=raw, frame_map=mapping,
            capture_config_ref=packet["clean_config"]["capture_config_ref"],
            output_dir=out / "RawFlow", authorization=authority)
        context = {"capture_config": packet["clean_config"]["capture_config_ref"]}
        signature = clean_component_signature(packet, row, context)
        attempted += 1  # Count attempted outer API, including a failed call.
        clean = run_clean_stage(raw, masks=masks, mask_receipt_ref=row["mask_receipt"],
            parent_propainter=capture.parent, output_dir=out / "A/Clean", raw_source_ref=row["raw_source_ref"],
            frame_map=mapping, model_refs=packet["model_weights"], component_signature=signature,
            authorization=authority, reference_selector=None, clean_context_refs=context,
            record_reference_selection=True)
        clean_ref = file_ref(out / "A/Clean/RECEIPT.json")
        status["candidates"]["A"] = {"execution": "COMPLETE_CLEAN", "receipt": clean_ref,
            "component_signature": signature, "model_calls": 1, "visual_quality": "NOT_REVIEWED"}
        with np.load(output_named(clean, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
            clean_rgb, actual_e = archive["clean_rgb"], archive["effective_removal"]
        if not np.array_equal(actual_e, masks["effective_removal"]):
            raise ContractError("A_EFFECTIVE_DOMAIN_DIFFERS_FROM_FROZEN_MASK")
        review = write_review_video(out / "A/raw_mask_clean_review.mp4", clean_review(raw, masks, clean_rgb), mapping)
        review.update(frame_map_ref=output_named(clean, "FRAME_MAP.json"),
                      source_frame_map_ref=row["frame_map_ref"], native_frames=238)
        status["candidates"]["A"]["review"] = review
        _same_gate(authority, owner)
        capture.seal()
        flow_ref = file_ref(out / "RawFlow/RECEIPT.json")
        status["shared"]["raw_flow"] = flow_ref
        calls = json.loads(Path(output_named(clean, "REFERENCE_SELECTION.json")["path"]).read_text(encoding="utf-8"))
        evidence = seal_reference_evidence(raw_flow_receipt_ref=flow_ref,
            mask_receipt_ref=row["mask_receipt"], clean_receipt_ref=clean_ref, masks=masks,
            reference_calls=calls, output_dir=out / "ReferenceEvidence", parent=models["propainter"],
            authorization=authority)
        evidence_ref = file_ref(out / "ReferenceEvidence/RECEIPT.json")
        status["shared"]["reference_evidence"] = evidence_ref
        if evidence["qualified_reference_pairs"] != 0 or evidence["changed_reference_calls"] != 0:
            raise ContractError("ABSENT_BACKGROUND_REGISTRATION_CANNOT_ENABLE_B")
        status["candidates"]["B"] = {"execution": "NOT_EXECUTED_NO_REFERENCE_CHANGE", "model_calls": 0,
            "reason": "UNKNOWN_BACKGROUND_REGISTRATION_ABSENT", "reference_evidence": evidence_ref,
            "visual_improvement": "NOT_ESTABLISHED"}
        for ref in [*frozen_refs, row["source_receipt"], row["mask_receipt"], row["raw_source_ref"],
                    row["frame_map_ref"], packet["clean_config"]["capture_config_ref"]]:
            verify_ref(ref)
        _same_gate(authority, owner)
        status["execution"] = "COMPLETE_CLEAN_WITH_REFERENCE_DIAGNOSTIC"
    except Exception as error:
        status["execution"] = ("PARTIAL_A_COMPLETE_SHARED_EVIDENCE_FAILED" if
            status["candidates"].get("A", {}).get("execution") == "COMPLETE_CLEAN" else "FAILED_RUNTIME")
        status["failure"] = type(error).__name__ + ":" + str(error)
        status["partial_consumable"] = False
        status["candidates"].setdefault("B", {"execution": "NOT_EXECUTED_SHARED_EVIDENCE_UNAVAILABLE", "model_calls": 0})
        # Already sealed A is independent evidence, even if later capture,
        # registration diagnostics or review encoding failed. Never rewrite it.
    status["timings"] = {"whole_session_wall_seconds_including_IO": time.monotonic() - begun}
    status["model_api_calls_attempted"] = attempted
    if capture is not None:
        status["raw_raft_subcalls_observed"] = len(capture.calls)
        status["capture_failure"] = capture.error
    write_json(out / "SESSION_RESULT.json", status)
    result = {"schema": RESULT_SCHEMA, "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "stage_scope": STAGE_SCOPE, "sessions": [status], "planned_sessions": 1,
        "completed_clean_sessions": int(status["execution"] == "COMPLETE_CLEAN_WITH_REFERENCE_DIAGNOSTIC"),
        "completed_model_candidates": int(status["candidates"].get("A", {}).get("execution") == "COMPLETE_CLEAN"),
        "model_call_accounting": {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
            "attempted_api_calls": attempted, "by_api": {"propainter_inpaint": attempted}},
        "candidate_adopted": False, "visual_quality": "NOT_REVIEWED", "visual_improvement": "NOT_ESTABLISHED",
        "training_eligible": False, "control_ground_truth": False, "files_deleted": 0, "training_steps": 0}
    write_json(root / "MODEL_RESULT.json", result)
    return result


def validate_clean_result(packet: Mapping, value: Mapping, *, decoder: Callable[[Path], int]) -> dict:
    """CPU-only independent seal/identity/actual decoding checks; not AI review."""
    if packet.get("experiment_scope") == NATIVE_SCOPE:
        return validate_native_clean_result(packet, value, decoder=decoder)
    _packet_contract(packet)
    if (value.get("schema") != RESULT_SCHEMA or value.get("task_id") != packet["task_id"]
            or value.get("run_signature") != packet["run_signature"] or value.get("stage_scope") != STAGE_SCOPE
            or value.get("planned_sessions") != 1 or value.get("completed_clean_sessions") != 1
            or value.get("completed_model_candidates") != 1 or len(value.get("sessions", [])) != 1):
        raise ContractError("G2_CLEAN_RESULT_IDENTITY_OR_COMPLETION_FAILED")
    accounting = value.get("model_call_accounting", {})
    if accounting != {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
                       "attempted_api_calls": 1, "by_api": {"propainter_inpaint": 1}}:
        raise ContractError("G2_CLEAN_ACTUAL_OUTER_MODEL_CALL_ACCOUNTING_FAILED")
    if any(value.get(key) is not False for key in ("candidate_adopted", "training_eligible", "control_ground_truth")):
        raise ContractError("G2_CLEAN_UNAUTHORIZED_CLAIM")
    raw, mapping, masks, report = load_native_inputs(packet)
    planned, session = packet["sessions"][0], value["sessions"][0]
    if (session.get("session_key") != "001" or session.get("session_id") != report["session_id"]
            or session.get("native_frames") != 238
            or session.get("execution") != "COMPLETE_CLEAN_WITH_REFERENCE_DIAGNOSTIC"
            or session.get("source") != planned["source_receipt"] or session.get("mask") != planned["mask_receipt"]
            or session.get("frame_map") != planned["frame_map_ref"] or session.get("raw_source") != planned["raw_source_ref"]):
        raise ContractError("G2_CLEAN_SESSION_SOURCE_IDENTITY_FAILED")
    root = Path(packet["output_dir"]).resolve()
    candidate = session["candidates"]["A"]
    receipt = sealed_receipt(candidate["receipt"], expected_stage="Clean")
    context = {"capture_config": packet["clean_config"]["capture_config_ref"]}
    signature = clean_component_signature(packet, planned, context)
    if (candidate.get("execution") != "COMPLETE_CLEAN" or candidate.get("model_calls") != 1
            or candidate.get("component_signature") != signature or receipt["component_signature"] != signature
            or receipt.get("native_input_frames") != 238 or receipt.get("native_output_frames") != 238
            or receipt.get("settings") != SETTINGS or receipt.get("reference_mode") != "FROZEN_BASELINE"
            or receipt.get("reference_selection_diagnostic_hook") is not True
            or receipt["actual_inputs"].get("raw_source") != planned["raw_source_ref"]
            or receipt["actual_inputs"].get("mask_receipt") != planned["mask_receipt"]
            or receipt["actual_inputs"].get("models") != packet["model_weights"]
            or receipt["actual_inputs"].get("clean_context_refs") != context
            or receipt["actual_inputs"].get("frames_sha256") != frames_sha256(raw)
            or receipt["actual_inputs"].get("frame_map_sha256") != canonical_sha(mapping)
            or receipt.get("authority", {}).get("task_id") != packet["task_id"]
            or receipt.get("authority", {}).get("run_signature") != packet["run_signature"]):
        raise ContractError("G2_A_FROZEN_CLEAN_SIGNATURE_OR_ACTUAL_INPUTS_FAILED")
    validation = receipt.get("validation", {})
    if validation.get("status") != "PASS" or any(validation.get(key) != 0 for key in (
            "outside_delete_changed_pixels", "protected_changed_pixels")):
        raise ContractError("G2_A_PIXEL_PRESERVATION_FAILED")
    clean_map = output_named(receipt, "FRAME_MAP.json")
    if json.loads(Path(clean_map["path"]).read_text(encoding="utf-8")) != mapping:
        raise ContractError("G2_A_OUTPUT_FRAME_MAP_CHANGED")
    with np.load(output_named(receipt, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        from .clean_references import validate_clean_preservation
        independent = validate_clean_preservation(raw, archive["clean_rgb"], archive["effective_removal"], archive["protected_visible"])
        if (independent["status"] != "PASS"
                or not np.array_equal(archive["effective_removal"], masks["effective_removal"])
                or not np.array_equal(archive["protected_visible"], masks["protected_visible"])):
            raise ContractError("G2_A_ACTUAL_ARRAY_PRESERVATION_FAILED")
    review = candidate["review"]
    verify_ref(review["video"])
    if (not Path(review["video"]["path"]).is_relative_to(root) or review.get("frame_map_ref") != clean_map
            or review.get("source_frame_map_ref") != planned["frame_map_ref"]
            or review.get("native_frames") != 238 or decoder(Path(review["video"]["path"])) != 238):
        raise ContractError("G2_A_REVIEW_ACTUAL_DECODE_OR_MAP_FAILED")
    shared = session["shared"]
    flow = sealed_receipt(shared["raw_flow"], expected_stage="RawFlow")
    evidence = sealed_receipt(shared["reference_evidence"], expected_stage="ReferenceEvidence")
    for ref in (candidate["receipt"], shared["raw_flow"], shared["reference_evidence"]):
        if not Path(ref["path"]).is_relative_to(root):
            raise ContractError("G2_RESULT_OUTSIDE_TASK_WRITE_SET")
    if (flow["actual_inputs"].get("capture_config") != context["capture_config"]
            or flow["actual_inputs"].get("source_receipt") != planned["source_receipt"]
            or flow.get("actual_native_shape") != list(raw.shape[:3]) or flow.get("adjacent_edges") != 237
            or flow.get("model_subcalls") != {"RAFT_bi": 20, "fix_raft_forward_backward": 40}
            or flow.get("background_geometry_qualified") is not False
            or evidence.get("actual_inputs") != {"raw_flow_receipt": shared["raw_flow"],
                "mask_receipt": planned["mask_receipt"], "clean_baseline_receipt": candidate["receipt"]}
            or evidence.get("actual_native_shape") != list(raw.shape[:3])
            or evidence.get("background_registration") != "ABSENT"
            or evidence.get("support_status") != "UNKNOWN_BACKGROUND_REGISTRATION_ABSENT"
            or evidence.get("qualified_reference_pairs") != 0 or evidence.get("qualified_visible_background_pixels") != 0
            or evidence.get("changed_reference_calls") != 0 or evidence.get("reference_calls") != 48):
        raise ContractError("G2_SHARED_FLOW_OR_UNKNOWN_REFERENCE_QUALIFICATION_FAILED")
    with np.load(output_named(evidence, "SUPPORT_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        if (archive["qualified_visible_background"].shape != raw.shape[:3]
                or archive["qualified_visible_background"].dtype != np.bool_
                or archive["qualified_visible_background"].any()
                or archive["background_support_unknown"].shape != raw.shape[:3]
                or archive["background_support_unknown"].dtype != np.bool_
                or not archive["background_support_unknown"].all()):
            raise ContractError("G2_UNKNOWN_SUPPORT_CANNOT_BECOME_QUALIFIED")
    b = session["candidates"]["B"]
    if (b.get("execution") != "NOT_EXECUTED_NO_REFERENCE_CHANGE" or b.get("model_calls") != 0
            or b.get("reference_evidence") != shared["reference_evidence"] or "receipt" in b):
        raise ContractError("G2_B_NOOP_CANNOT_COUNT_AS_EXECUTED")
    for ref in [*packet["runtime_code"], *packet["model_weights"], context["capture_config"]]:
        verify_ref(ref)
    return {"validation": "PASS_EXECUTION_STRUCTURE_ONLY", "actual_decoded_frames": 238,
        "completed_model_candidates": 1, "B": "NOT_EXECUTED_NO_REFERENCE_CHANGE",
        "visual_quality": "NOT_REVIEWED", "visual_improvement": "NOT_ESTABLISHED",
        "training_eligible": False}


def load_native_clean_inputs(packet: Mapping, row: Mapping):
    """New native branch: current Source/Mask identity, not equal lengths alone."""
    from .native_contracts import source_binding, stage_binding
    source, report, mapping = source_binding(row, PROJECT_ROOT)
    mask = stage_binding(row, "Mask", PROJECT_ROOT)
    source_refs = {Path(ref["path"]).name: ref for ref in source["outputs"]}
    with np.load(source_refs["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
        raw = archive["rgb"]
    n = row["native_frames"]
    if (raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[0] != n or raw.shape[-1] != 3
            or len(mapping) != n or report["actual_inputs"]["video"] != row["raw_source_ref"]):
        raise ContractError("NATIVE_CLEAN_CURRENT_SOURCE_DOMAIN_DRIFT")
    actual = mask.get("actual_inputs", {})
    if (actual.get("raw_source") != row["raw_source_ref"]
            or actual.get("frames_sha256") != frames_sha256(raw)
            or actual.get("frame_map_sha256") != canonical_sha(mapping)):
        raise ContractError("NATIVE_CLEAN_MASK_RAW_PIXEL_OR_MAP_DRIFT")
    with np.load(output_named(mask, "MASK_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        keys = ("removal", "effective_removal", "protected_visible", "core", "equipment", "unknown", "held")
        masks = {key: archive[key] for key in keys if key in archive}
    if not {"removal", "effective_removal", "protected_visible"} <= set(masks) or any(
            a.dtype != np.bool_ or a.shape != raw.shape[:3] for a in masks.values()):
        raise ContractError("NATIVE_CLEAN_SEALED_MASK_ARRAY_DOMAIN_DRIFT")
    from .model_stages import _checked_sealed_effective
    _checked_sealed_effective(masks["removal"], masks["protected_visible"], masks["effective_removal"])
    return raw, mapping, masks, report


def _fatal_native_failure(error):
    return any(code in str(error) for code in ("SHA_DRIFT", "FENCE", "LEASE", "DEADLINE", "ACTIVE_PACKET"))


class NativeRawVisibleReferenceContext:
    """One PP invocation: observe its original RAFT and prepare support once.

    Raw unremoved support is not registered/static background or hidden truth.
    Evidence is sealed before selection; downstream statistics cannot become an
    input of the same Clean receipt. No NN call is introduced by this adapter.
    """
    def __init__(self, *, packet, row, raw, mapping, masks, parent, target, authority):
        from .clean_flow import RawVisibleReferenceBuilder
        self.builder_type = RawVisibleReferenceBuilder
        self.packet, self.row, self.raw, self.mapping, self.masks = packet, row, raw, mapping, masks
        self.target, self.authority, self.owner = target, authority, _gate(authority)
        self.capture = RawRaftCaptureReplay(parent, raw_rgb=raw, frame_map=mapping,
            capture_config_ref=row["capture_config_ref"], output_dir=target / "B/RawFlow",
            authorization=authority)
        self.parent, self.builder, self.refs, self.calls = self.capture.parent, None, None, []

    def provide(self):
        from .clean_flow import array_sha, fresh_directory, vendor_fb_diagnostics, write_npz
        if self.refs is not None:
            return dict(self.refs)
        _same_gate(self.authority, self.owner)
        flow = self.capture.seal()
        flow_ref = file_ref(self.capture.root / "RECEIPT.json")
        folder = fresh_directory(self.target / "B/ReferenceEvidence")
        outputs, forward, backward, consistent_f, consistent_b, index = [], [], [], [], [], []
        for chunk in self.capture.calls:
            _same_gate(self.authority, self.owner)
            with np.load(chunk["arrays"]["path"], allow_pickle=False) as archive:
                f, b = archive["forward"], archive["backward"]
            values = vendor_fb_diagnostics(f, b, self.capture.original_parent)
            expected = (f.shape[1], *self.raw.shape[1:3])
            names = {"fb_residual_squared_forward", "fb_residual_squared_backward",
                "vendor_fb_consistent_forward", "vendor_fb_consistent_backward"}
            if set(values) != names or any(a.shape != expected or
                    (a.dtype != np.bool_ if "consistent" in name else a.dtype != np.float32)
                    for name, a in values.items()):
                raise ContractError("NATIVE_REFERENCE_FB_DOMAIN_DRIFT")
            output = write_npz(folder / f"fb_{chunk['chunk_index']:04d}.npz", **values)
            outputs.append(output)
            index.append({"first_edge": chunk["first_edge"], "end_edge_exclusive": chunk["end_edge_exclusive"],
                "raw_flow_arrays": chunk["arrays"], "diagnostic_arrays": output})
            forward.append(f[0]); backward.append(b[0])
            consistent_f.append(values["vendor_fb_consistent_forward"])
            consistent_b.append(values["vendor_fb_consistent_backward"])
        cf, cb = np.concatenate(consistent_f), np.concatenate(consistent_b)
        binding = {"source_receipt": self.row["source_receipt"], "mask_receipt": self.row["mask_receipt"],
            "raw_flow_receipt": flow_ref, "fb_consistent_forward_sha256": array_sha(cf),
            "fb_consistent_backward_sha256": array_sha(cb),
            "vendor_fb_diagnostic_policy": "EXISTING_VENDOR_DEFAULTS_NOT_BACKGROUND_CONFIDENCE"}
        self.builder = self.builder_type(session_id=self.row["session_id"], raw_rgb=self.raw,
            frame_map=self.mapping, masks=self.masks, forward=np.concatenate(forward),
            backward=np.concatenate(backward), fb_consistent_forward=cf, fb_consistent_backward=cb,
            binding=binding)
        outputs += [write_json(folder / "FB_INDEX.json", index), write_json(folder / "BINDING.json", binding)]
        actual = {"source_receipt": self.row["source_receipt"], "mask_receipt": self.row["mask_receipt"],
            "raw_flow_receipt": flow_ref, "capture_config": self.row["capture_config_ref"],
            "reference_policy": NATIVE_REFERENCE_POLICY, "binding_sha256": canonical_sha(binding)}
        from .contracts import SCHEMA
        receipt = {"schema": SCHEMA, "stage": "ReferenceEvidence", "status": "SEALED_COMPLETE",
            "component_signature": canonical_sha({"stage": "ReferenceEvidence", "actual_inputs": actual}),
            "session_id": self.row["session_id"], "native_input_frames": len(self.raw),
            "actual_inputs": actual, "outputs": outputs, "support_source": "RAW_UNREMOVED_VISIBLE_SUPPORT",
            "background_geometry_qualified": False, "static_background_truth": False,
            "training_eligible": False, "control_ground_truth": False}
        evidence_ref = write_json(folder / "RECEIPT.json", receipt)
        self.refs = {"raw_flow_receipt": flow_ref, "reference_evidence": evidence_ref}
        _same_gate(self.authority, self.owner)
        return dict(self.refs)

    def select(self, *, mid, neighbors, length, stride, ref_num, original_references):
        from .clean_references import select_reference_indices
        from .clean_flow import write_npz
        if self.builder is None or length != len(self.raw) or stride != SETTINGS["ref_stride"] or ref_num != (
                SETTINGS["subvideo_length"] // stride if length > SETTINGS["subvideo_length"] else -1):
            raise ContractError("NATIVE_REFERENCE_ONLINE_ORDER_OR_SETTINGS_DRIFT")
        start, end = (0, length) if ref_num == -1 else (
            max(0, mid - stride * (ref_num // 2)), min(length, mid + stride * (ref_num // 2)))
        evidence = self.builder.build(mid, [i for i in range(start, end) if i not in neighbors])
        plan = select_reference_indices(mid, neighbors, length, session_id=self.row["session_id"],
            effective_delete_mask=self.masks["effective_removal"][mid], evidence=evidence)
        if plan["original_references"] != list(original_references):
            raise ContractError("NATIVE_REFERENCE_ORIGINAL_PLAN_DRIFT")
        chosen = [e for e in evidence if e.reference_frame in plan["references"]]
        support_ref = write_npz(self.target / "B/ReferenceEvidence" / f"selected_{len(self.calls):04d}.npz",
            reference_frames=np.asarray([e.reference_frame for e in chosen], np.int64),
            support=np.stack([e.visible_background for e in chosen]) if chosen else np.empty((0, *self.raw.shape[1:3]), bool),
            flow_error=np.stack([e.flow_error for e in chosen]) if chosen else np.empty((0, *self.raw.shape[1:3]), np.float32))
        self.calls.append({"target_frame": mid, "neighbors": list(neighbors), **plan,
            "selected_support_arrays": support_ref,
            "references_changed": plan["references"] != plan["original_references"]})
        return plan["references"]

    def seal_statistics(self, clean):
        recorded = json.loads(Path(output_named(clean, "REFERENCE_SELECTION.json")["path"]).read_text())
        for actual, plan in zip(recorded, self.calls):
            if any(actual.get(k) != plan[k] for k in ("target_frame", "neighbors", "original_references", "references", "reference_count")):
                raise ContractError("NATIVE_REFERENCE_ACTUAL_SELECTION_DRIFT")
        if len(recorded) != len(self.calls) or not self.calls:
            raise ContractError("NATIVE_REFERENCE_ACTUAL_SELECTION_COUNT_DRIFT")
        value = {"schema": "inpainting-native-raw-visible-reference-statistics-v1",
            "reference_policy": NATIVE_REFERENCE_POLICY, "clean_context_refs": {
                "capture_config": self.row["capture_config_ref"], **self.refs}, "calls": self.calls,
            "reference_calls": len(self.calls), "changed_reference_calls": sum(c["references_changed"] for c in self.calls),
            "rank_not_applied_calls": sum(c["status"].startswith("BASELINE_") for c in self.calls),
            "support_source": "RAW_UNREMOVED_VISIBLE_SUPPORT", "background_geometry_qualified": False,
            "static_background_truth": False, "visual_improvement": "NOT_ESTABLISHED",
            "training_eligible": False, "control_ground_truth": False}
        return write_json(self.target / "B/REFERENCE_SELECTION_STATS.json", value)


def _run_native_reference_clean_batch(packet, *, authority):
    owner = _gate(authority)
    if owner["task_id"] != packet["task_id"] or owner["run_signature"] != packet["run_signature"]:
        raise ContractError("NATIVE_CLEAN_AUTHORITY_IDENTITY_DRIFT")
    root = Path(packet["output_dir"]).resolve()
    if root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ContractError("NATIVE_CLEAN_OUTPUT_OUTSIDE_PROJECT")
    root.mkdir(parents=True, exist_ok=True)
    frozen = [*packet["runtime_code"], *packet["model_weights"]]
    for ref in frozen:
        verify_ref(ref)
    rows, attempted, models = [], 0, None
    for row in packet["sessions"]:
        _same_gate(authority, owner)
        target = root / "sessions" / row["session_key"]
        target.mkdir(parents=True, exist_ok=False)
        status = {"session_key": row["session_key"], "session_id": row["session_id"],
            "native_frames": row["native_frames"], "experiment_scope": NATIVE_SCOPE,
            "model_api_calls_attempted": 0, "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False, "partial_consumable": False}
        for key, source in (("source", "source_receipt"), ("frame_map", "frame_map_ref")):
            if source in row:
                status[key] = row[source]
        if row["upstream_status"] == "BLOCKED":
            verify_ref(row["diagnostic_ref"])
            status.update(execution="BLOCKED_UPSTREAM", upstream_status="BLOCKED",
                failure_code=row["failure_code"], diagnostic_ref=row["diagnostic_ref"])
        else:
            started = time.monotonic()
            try:
                raw, mapping, masks, report = load_native_clean_inputs(packet, row)
                verify_ref(row["capture_config_ref"])
                if models is None:
                    models = load_frozen_models(PROJECT_ROOT, root / "model_load", authorization=authority,
                        frozen_inputs=frozen, need_sam=False, need_clean=True)
                    write_json(root / "MODEL_LOAD.json", {"task_id": packet["task_id"],
                        "run_signature": packet["run_signature"], "load_receipt": models["load_receipt"],
                        "inference_api_calls": 0, "visual_quality": "NOT_EVALUATED"})
                context = NativeRawVisibleReferenceContext(packet=packet, row=row, raw=raw, mapping=mapping,
                    masks=masks, parent=models["propainter"], target=target, authority=authority)
                if attempted >= packet["budgets"]["model_calls_max"]:
                    raise ContractError("NATIVE_CLEAN_MODEL_CALL_BUDGET_EXHAUSTED")
                attempted += 1
                status["model_api_calls_attempted"] = 1
                clean = run_clean_stage(raw, masks=masks, mask_receipt_ref=row["mask_receipt"],
                    parent_propainter=context.parent, output_dir=target / "B/Clean", raw_source_ref=row["raw_source_ref"],
                    frame_map=mapping, model_refs=packet["model_weights"], component_signature=row["clean_signature"],
                    authorization=authority, reference_selector=context.select,
                    clean_context_refs={"capture_config": row["capture_config_ref"]}, record_reference_selection=True,
                    sealed_effective_removal=masks["effective_removal"], online_reference_context_provider=context.provide)
                stats_ref = context.seal_statistics(clean)
                receipt_ref = file_ref(target / "B/Clean/RECEIPT.json")
                with np.load(output_named(clean, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
                    clean_rgb = archive["clean_rgb"]
                    if any(not np.array_equal(archive[k], masks[k]) for k in ("effective_removal", "protected_visible")):
                        raise ContractError("NATIVE_CLEAN_ACTUAL_E_OR_P_DIFFERS_FROM_CURRENT_MASK")
                review = write_review_video(target / "B/raw_mask_clean_review.mp4", clean_review(raw, masks, clean_rgb), mapping)
                review.update(native_frames=row["native_frames"], source_frame_map_ref=row["frame_map_ref"],
                    frame_map_ref=output_named(clean, "FRAME_MAP.json"))
                for ref in [*frozen, row["source_receipt"], row["mask_receipt"], row["raw_source_ref"], row["frame_map_ref"], row["capture_config_ref"]]:
                    verify_ref(ref)
                _same_gate(authority, owner)
                status.update(execution="COMPLETE_CLEAN_REFERENCE_EXPERIMENT", upstream_status="READY",
                    mask=row["mask_receipt"], raw_source=row["raw_source_ref"], clean=receipt_ref,
                    clean_receipt=receipt_ref, component_signature=row["clean_signature"], review=review,
                    capture_config=row["capture_config_ref"], **context.refs, reference_selection_stats=stats_ref,
                    timebase_status=report.get("timebase_status", "TIMEBASE_UNVERIFIED"))
            except Exception as error:
                status.update(execution="FAILED_RUNTIME", failure=type(error).__name__ + ":" + str(error))
                if _fatal_native_failure(error):
                    write_json(target / "SESSION_RESULT.json", status)
                    raise
            status["timings"] = {"whole_session_wall_seconds_including_IO": time.monotonic() - started}
        write_json(target / "SESSION_RESULT.json", status)
        rows.append(status)
    completed = sum(r["execution"] == "COMPLETE_CLEAN_REFERENCE_EXPERIMENT" for r in rows)
    blocked = sum(r["execution"] == "BLOCKED_UPSTREAM" for r in rows)
    failed = len(rows) - completed - blocked
    result = {"schema": NATIVE_RESULT_SCHEMA, "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "experiment_scope": NATIVE_SCOPE, "stage_scope": NATIVE_SCOPE, "sessions": rows,
        "planned_sessions": len(packet["sessions"]), "completed_clean_sessions": completed,
        "completed_model_candidates": completed, "blocked_sessions": blocked, "failed_sessions": failed,
        "execution_diagnostic_complete": completed > 0 and failed == 0 and len(rows) == len(packet["sessions"]),
        "model_call_accounting": {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
            "attempted_api_calls": attempted, "completed_api_calls": completed, "by_api": {"propainter_inpaint": attempted}},
        "reference_policy": NATIVE_REFERENCE_POLICY, "candidate_adopted": False,
        "visual_quality": "NOT_REVIEWED", "visual_improvement": "NOT_ESTABLISHED",
        "training_eligible": False, "control_ground_truth": False, "files_deleted": 0, "training_steps": 0}
    write_json(root / "MODEL_RESULT.json", result)
    return result


def run_native_clean_batch(packet: Mapping, *, authority: Callable) -> dict:
    """One unchanged baseline PP call per READY native session; no G2 A/B."""
    _packet_contract(packet)
    if packet["clean_config"]["reference_policy"] == NATIVE_REFERENCE_POLICY:
        return _run_native_reference_clean_batch(packet, authority=authority)
    owner = _gate(authority)
    if owner["task_id"] != packet["task_id"] or owner["run_signature"] != packet["run_signature"]:
        raise ContractError("NATIVE_CLEAN_AUTHORITY_IDENTITY_DRIFT")
    root = Path(packet["output_dir"]).resolve()
    if root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ContractError("NATIVE_CLEAN_OUTPUT_OUTSIDE_PROJECT")
    root.mkdir(parents=True, exist_ok=True)
    frozen = [*packet["runtime_code"], *packet["model_weights"]]
    for reference in frozen:
        verify_ref(reference)
    rows, attempted, models = [], 0, None
    for row in packet["sessions"]:
        _same_gate(authority, owner)
        target = root / "sessions" / row["session_key"]
        target.mkdir(parents=True, exist_ok=False)
        status = {"session_key": row["session_key"], "session_id": row["session_id"],
            "native_frames": row["native_frames"], "experiment_scope": NATIVE_SCOPE,
            "model_api_calls_attempted": 0, "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False, "partial_consumable": False}
        if "source_receipt" in row:
            status["source"] = row["source_receipt"]
        if "frame_map_ref" in row:
            status["frame_map"] = row["frame_map_ref"]
        if row["upstream_status"] == "BLOCKED":
            verify_ref(row["diagnostic_ref"])
            status.update(execution="BLOCKED_UPSTREAM", upstream_status="BLOCKED",
                failure_code=row["failure_code"], diagnostic_ref=row["diagnostic_ref"])
            write_json(target / "SESSION_RESULT.json", status)
            rows.append(status)
            continue
        started = time.monotonic()
        try:
            raw, mapping, masks, report = load_native_clean_inputs(packet, row)
            if models is None:
                models = load_frozen_models(PROJECT_ROOT, root / "model_load", authorization=authority,
                    frozen_inputs=frozen, need_sam=False, need_clean=True)
                write_json(root / "MODEL_LOAD.json", {"task_id": packet["task_id"],
                    "run_signature": packet["run_signature"], "load_receipt": models["load_receipt"],
                    "inference_api_calls": 0, "visual_quality": "NOT_EVALUATED"})
            if attempted >= packet["budgets"]["model_calls_max"]:
                raise ContractError("NATIVE_CLEAN_MODEL_CALL_BUDGET_EXHAUSTED")
            attempted += 1
            status["model_api_calls_attempted"] = 1
            clean = run_clean_stage(raw, masks=masks, mask_receipt_ref=row["mask_receipt"],
                parent_propainter=models["propainter"], output_dir=target / "A/Clean",
                raw_source_ref=row["raw_source_ref"], frame_map=mapping, model_refs=packet["model_weights"],
                component_signature=row["clean_signature"], authorization=authority,
                reference_selector=None, clean_context_refs=None, record_reference_selection=False,
                sealed_effective_removal=masks["effective_removal"])
            receipt_ref = file_ref(target / "A/Clean/RECEIPT.json")
            with np.load(output_named(clean, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
                clean_rgb = archive["clean_rgb"]
                if (not np.array_equal(archive["effective_removal"], masks["effective_removal"])
                        or not np.array_equal(archive["protected_visible"], masks["protected_visible"])):
                    raise ContractError("NATIVE_CLEAN_ACTUAL_E_OR_P_DIFFERS_FROM_CURRENT_MASK")
            review = write_review_video(target / "A/raw_mask_clean_review.mp4", clean_review(raw, masks, clean_rgb), mapping)
            review.update(native_frames=row["native_frames"], source_frame_map_ref=row["frame_map_ref"],
                frame_map_ref=output_named(clean, "FRAME_MAP.json"))
            for reference in [*frozen, row["source_receipt"], row["mask_receipt"], row["raw_source_ref"], row["frame_map_ref"]]:
                verify_ref(reference)
            _same_gate(authority, owner)
            status.update(execution="COMPLETE_CLEAN_BASELINE", upstream_status="READY",
                mask=row["mask_receipt"], raw_source=row["raw_source_ref"], clean=receipt_ref,
                clean_receipt=receipt_ref, component_signature=row["clean_signature"], review=review,
                timebase_status=report.get("timebase_status", "TIMEBASE_UNVERIFIED"))
        except Exception as error:
            status.update(execution="FAILED_RUNTIME", failure=type(error).__name__ + ":" + str(error))
            if _fatal_native_failure(error):
                write_json(target / "SESSION_RESULT.json", status)
                rows.append(status)
                raise
        status["timings"] = {"whole_session_wall_seconds_including_IO": time.monotonic() - started}
        write_json(target / "SESSION_RESULT.json", status)
        rows.append(status)
    completed = sum(row["execution"] == "COMPLETE_CLEAN_BASELINE" for row in rows)
    blocked = sum(row["execution"] == "BLOCKED_UPSTREAM" for row in rows)
    failed = len(rows) - completed - blocked
    result = {"schema": NATIVE_RESULT_SCHEMA, "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "experiment_scope": NATIVE_SCOPE, "stage_scope": NATIVE_SCOPE, "sessions": rows,
        "planned_sessions": len(packet["sessions"]), "completed_clean_sessions": completed,
        "completed_model_candidates": completed, "blocked_sessions": blocked, "failed_sessions": failed,
        "execution_diagnostic_complete": completed > 0 and failed == 0 and len(rows) == len(packet["sessions"]),
        "model_call_accounting": {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
            "attempted_api_calls": attempted, "completed_api_calls": completed, "by_api": {"propainter_inpaint": attempted}},
        "candidate_adopted": False, "visual_quality": "NOT_REVIEWED", "visual_improvement": "NOT_ESTABLISHED",
        "training_eligible": False, "control_ground_truth": False, "files_deleted": 0, "training_steps": 0}
    write_json(root / "MODEL_RESULT.json", result)
    return result


def _validate_native_reference_clean_result(packet, value, *, decoder):
    from .clean_references import validate_clean_preservation
    from chaoyang.exact78_baseline.clean import _get_ref_index
    planned, rows = packet["sessions"], value.get("sessions", [])
    if (value.get("schema") != NATIVE_RESULT_SCHEMA or value.get("task_id") != packet["task_id"]
            or value.get("run_signature") != packet["run_signature"] or value.get("experiment_scope") != NATIVE_SCOPE
            or value.get("reference_policy") != NATIVE_REFERENCE_POLICY
            or [r.get("session_key") for r in rows] != [r["session_key"] for r in planned]
            or value.get("planned_sessions") != len(planned)):
        raise ContractError("NATIVE_REFERENCE_RESULT_IDENTITY_FAILED")
    root, completed, blocked = Path(packet["output_dir"]), 0, 0
    for plan, row in zip(planned, rows):
        if (row.get("native_frames") != plan["native_frames"] or row.get("session_id") != plan["session_id"]
                or ("source_receipt" in plan and row.get("source") != plan["source_receipt"])
                or ("frame_map_ref" in plan and row.get("frame_map") != plan["frame_map_ref"])):
            raise ContractError("NATIVE_REFERENCE_SESSION_IDENTITY_FAILED")
        if plan["upstream_status"] == "BLOCKED":
            verify_ref(plan["diagnostic_ref"])
            if (row.get("execution") != "BLOCKED_UPSTREAM" or row.get("model_api_calls_attempted") != 0
                    or row.get("diagnostic_ref") != plan["diagnostic_ref"] or row.get("failure_code") != plan["failure_code"]
                    or any(k in row for k in ("clean", "clean_receipt", "review", "raw_flow_receipt"))):
                raise ContractError("NATIVE_BLOCKED_CLEAN_CANNOT_CLAIM_STAGE")
            blocked += 1
            continue
        raw, mapping, masks, _ = load_native_clean_inputs(packet, plan)
        n, target = plan["native_frames"], root / "sessions" / plan["session_key"]
        if (row.get("execution") != "COMPLETE_CLEAN_REFERENCE_EXPERIMENT" or row.get("model_api_calls_attempted") != 1
                or row.get("clean") != row.get("clean_receipt") or row.get("mask") != plan["mask_receipt"]
                or row.get("raw_source") != plan["raw_source_ref"] or row.get("capture_config") != plan["capture_config_ref"]
                or Path(row["clean_receipt"]["path"]) != target / "B/Clean/RECEIPT.json"):
            raise ContractError("NATIVE_REFERENCE_READY_STAGE_OR_PATH_DRIFT")
        context = {"capture_config": plan["capture_config_ref"],
            "raw_flow_receipt": row["raw_flow_receipt"], "reference_evidence": row["reference_evidence"]}
        for key, relative in (("raw_flow_receipt", "B/RawFlow/RECEIPT.json"),
                ("reference_evidence", "B/ReferenceEvidence/RECEIPT.json"),
                ("reference_selection_stats", "B/REFERENCE_SELECTION_STATS.json")):
            if Path(row[key]["path"]) != target / relative:
                raise ContractError("NATIVE_REFERENCE_CONTEXT_OUTPUT_PATH_DRIFT")
            verify_ref(row[key])
        flow = sealed_receipt(context["raw_flow_receipt"], expected_stage="RawFlow")
        evidence = sealed_receipt(context["reference_evidence"], expected_stage="ReferenceEvidence")
        flow_actual, ev_actual = flow.get("actual_inputs", {}), evidence.get("actual_inputs", {})
        if (flow_actual.get("capture_config") != plan["capture_config_ref"]
                or flow_actual.get("source_receipt") != plan["source_receipt"]
                or flow_actual.get("raw_source") != plan["raw_source_ref"]
                or flow_actual.get("frame_map") != plan["frame_map_ref"]
                or flow_actual.get("runtime_code") != packet["runtime_code"]
                or flow_actual.get("frames_sha256") != frames_sha256(raw)
                or flow_actual.get("frame_map_sha256") != canonical_sha(mapping)
                or flow.get("actual_native_shape") != list(raw.shape[:3])
                or flow.get("flow_source") != "RAW_RGB_FORWARD_BACKWARD_FLOW"
                or flow.get("completed_inpainted_flow_consumed") is not False
                or evidence.get("support_source") != "RAW_UNREMOVED_VISIBLE_SUPPORT"
                or ev_actual.get("source_receipt") != plan["source_receipt"]
                or ev_actual.get("mask_receipt") != plan["mask_receipt"]
                or ev_actual.get("raw_flow_receipt") != context["raw_flow_receipt"]
                or ev_actual.get("capture_config") != plan["capture_config_ref"]
                or ev_actual.get("reference_policy") != NATIVE_REFERENCE_POLICY
                or any("clean" in k.lower() for k in ev_actual)
                or evidence.get("background_geometry_qualified") is not False
                or evidence.get("static_background_truth") is not False):
            raise ContractError("NATIVE_REFERENCE_FLOW_OR_SUPPORT_SOURCE_DRIFT")
        from .clean_flow import array_sha, chunk_plan
        binding = json.loads(Path(output_named(evidence, "BINDING.json")["path"]).read_text())
        fb_index = json.loads(Path(output_named(evidence, "FB_INDEX.json")["path"]).read_text())
        raw_chunks = json.loads(Path(output_named(flow, "CHUNK_INDEX.json")["path"]).read_text())
        cf, cb, covered = [], [], []
        if (len(fb_index) != len(raw_chunks) or len(raw_chunks) != len(chunk_plan(n))
                or binding.get("source_receipt") != plan["source_receipt"]
                or binding.get("mask_receipt") != plan["mask_receipt"]
                or binding.get("raw_flow_receipt") != context["raw_flow_receipt"]
                or binding.get("vendor_fb_diagnostic_policy") != "EXISTING_VENDOR_DEFAULTS_NOT_BACKGROUND_CONFIDENCE"
                or ev_actual.get("binding_sha256") != canonical_sha(binding)):
            raise ContractError("NATIVE_REFERENCE_FB_BINDING_DRIFT")
        for diagnostic, chunk in zip(fb_index, raw_chunks):
            first, end = chunk["first_edge"], chunk["end_edge_exclusive"]
            if (diagnostic.get("first_edge") != first or diagnostic.get("end_edge_exclusive") != end
                    or diagnostic.get("raw_flow_arrays") != chunk["arrays"]
                    or diagnostic.get("diagnostic_arrays") not in evidence["outputs"]):
                raise ContractError("NATIVE_REFERENCE_FB_EDGE_PROVENANCE_DRIFT")
            with np.load(diagnostic["diagnostic_arrays"]["path"], allow_pickle=False) as archive:
                forward, backward = archive["vendor_fb_consistent_forward"], archive["vendor_fb_consistent_backward"]
                if any(a.dtype != np.bool_ or a.shape != (end-first, *raw.shape[1:3]) for a in (forward, backward)):
                    raise ContractError("NATIVE_REFERENCE_FB_DOMAIN_DRIFT")
                cf.append(forward); cb.append(backward)
            covered.extend(range(first, end))
        if (covered != list(range(n-1)) or array_sha(np.concatenate(cf)) != binding["fb_consistent_forward_sha256"]
                or array_sha(np.concatenate(cb)) != binding["fb_consistent_backward_sha256"]):
            raise ContractError("NATIVE_REFERENCE_FB_ARRAY_SHA_DRIFT")
        stage = sealed_receipt(row["clean_receipt"], expected_stage="Clean")
        actual = stage.get("actual_inputs", {})
        mask_stage = sealed_receipt(plan["mask_receipt"], expected_stage="Mask")
        effective_source = {"mask_arrays": output_named(mask_stage, "MASK_ARRAYS.npz"),
            "array_key": "effective_removal", "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}
        if (stage.get("component_signature") != plan["clean_signature"]
                or stage.get("native_input_frames") != n or stage.get("native_output_frames") != n
                or stage.get("settings") != SETTINGS or stage.get("reference_mode") != "SINGLE_REF_SELECTION_CHANGE"
                or stage.get("reference_selection_diagnostic_hook") is not True
                or actual.get("raw_source") != plan["raw_source_ref"] or actual.get("mask_receipt") != plan["mask_receipt"]
                or actual.get("models") != packet["model_weights"] or actual.get("clean_context_refs") != context
                or actual.get("effective_removal_source") != effective_source
                or actual.get("frames_sha256") != frames_sha256(raw) or actual.get("frame_map_sha256") != canonical_sha(mapping)
                or stage.get("authority", {}).get("task_id") != packet["task_id"]
                or stage.get("authority", {}).get("run_signature") != packet["run_signature"]):
            raise ContractError("NATIVE_REFERENCE_CLEAN_SIGNATURE_OR_CONTEXT_DRIFT")
        if json.loads(Path(output_named(stage, "FRAME_MAP.json")["path"]).read_text()) != mapping:
            raise ContractError("NATIVE_CLEAN_OUTPUT_MAP_DRIFT")
        states = json.loads(Path(output_named(stage, "FRAME_STATUS.json")["path"]).read_text())
        if not isinstance(states, list) or len(states) != n or any(s.get("native_frame_index") != i for i, s in enumerate(states)):
            raise ContractError("NATIVE_CLEAN_FULL_FRAME_STATUS_DRIFT")
        with np.load(output_named(stage, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
            if (validate_clean_preservation(raw, archive["clean_rgb"], archive["effective_removal"], archive["protected_visible"])["status"] != "PASS"
                    or any(not np.array_equal(archive[k], masks[k]) for k in ("effective_removal", "protected_visible"))):
                raise ContractError("NATIVE_CLEAN_ACTUAL_PIXEL_PRESERVATION_DRIFT")
        calls = json.loads(Path(output_named(stage, "REFERENCE_SELECTION.json")["path"]).read_text())
        stats = json.loads(Path(row["reference_selection_stats"]["path"]).read_text())
        plans = stats.get("calls", [])
        if (not calls or len(calls) != len(plans) or stats.get("reference_calls") != len(calls)
                or stats.get("clean_context_refs") != context or stats.get("reference_policy") != NATIVE_REFERENCE_POLICY
                or stats.get("changed_reference_calls") != sum(p.get("references_changed") is True for p in plans)
                or stats.get("rank_not_applied_calls") != sum(p.get("status", "").startswith("BASELINE_") for p in plans)
                or stats.get("support_source") != "RAW_UNREMOVED_VISIBLE_SUPPORT"
                or stats.get("visual_improvement") != "NOT_ESTABLISHED"
                or any(stats.get(k) is not False for k in ("background_geometry_qualified", "static_background_truth", "training_eligible", "control_ground_truth"))):
            raise ContractError("NATIVE_REFERENCE_SELECTION_STATISTICS_DRIFT")
        for call, selected in zip(calls, plans):
            mid, neighbors = call["target_frame"], call["neighbors"]
            stride = SETTINGS["ref_stride"]
            ref_num = SETTINGS["subvideo_length"] // stride if n > SETTINGS["subvideo_length"] else -1
            original = _get_ref_index(mid, neighbors, n, stride, ref_num)
            start, end = (0, n) if ref_num == -1 else (max(0, mid-stride*(ref_num//2)), min(n, mid+stride*(ref_num//2)))
            refs = call["references"]
            if (not 0 <= mid < n or mid not in neighbors or any(not 0 <= i < n for i in neighbors)
                    or refs != sorted(set(refs)) or len(refs) != len(original) or call["original_references"] != original
                    or any(i in neighbors or not start <= i < end for i in refs)
                    or any(call.get(k) != selected.get(k) for k in ("target_frame", "neighbors", "original_references", "references", "reference_count"))
                    or selected.get("references_changed") != (refs != original)
                    or (selected["status"].startswith("BASELINE_") and refs != original)):
                raise ContractError("NATIVE_REFERENCE_FROZEN_COUNT_WINDOW_OR_PLAN_DRIFT")
            verify_ref(selected["selected_support_arrays"])
            if not Path(selected["selected_support_arrays"]["path"]).is_relative_to(target / "B/ReferenceEvidence"):
                raise ContractError("NATIVE_REFERENCE_SUPPORT_ARRAY_PATH_DRIFT")
            with np.load(selected["selected_support_arrays"]["path"], allow_pickle=False) as archive:
                donors, support, errors = archive["reference_frames"], archive["support"], archive["flow_error"]
                if (donors.dtype != np.int64 or donors.tolist() != refs
                        or support.dtype != np.bool_ or support.shape != (len(refs), *raw.shape[1:3])
                        or errors.dtype != np.float32 or errors.shape != support.shape):
                    raise ContractError("NATIVE_REFERENCE_SELECTED_SUPPORT_DOMAIN_DRIFT")
                for index, donor in enumerate(refs):
                    ranked = next((r for r in selected["ranked_evidence"] if r["frame"] == donor), None)
                    if ranked is not None:
                        pixels = support[index] & masks["effective_removal"][mid] & np.isfinite(errors[index])
                        meta = ranked.get("evidence_metadata", {})
                        if (ranked.get("raw_support_sha256") != array_sha(support[index])
                                or ranked.get("raw_flow_sha256") != context["raw_flow_receipt"]["sha256"]
                                or ranked.get("supported_hole_pixels") != int(pixels.sum())
                                or not pixels.any() or ranked.get("median_raw_flow_error") != float(np.median(errors[index][pixels]))
                                or meta.get("binding_sha256") != canonical_sha(binding)
                                or any(meta.get(k) != binding[k] for k in ("source_receipt", "mask_receipt", "raw_flow_receipt"))):
                            raise ContractError("NATIVE_REFERENCE_SELECTED_SUPPORT_SHA_OR_ORIGIN_DRIFT")
        review = row.get("review", {})
        verify_ref(review["video"])
        if (not Path(review["video"]["path"]).is_relative_to(target / "B") or review.get("native_frames") != n
                or decoder(Path(review["video"]["path"])) != n or review.get("source_frame_map_ref") != plan["frame_map_ref"]
                or review.get("frame_map_ref") != output_named(stage, "FRAME_MAP.json")):
            raise ContractError("NATIVE_CLEAN_ACTUAL_REVIEW_DECODE_OR_MAP_DRIFT")
        completed += 1
    accounting = {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
        "attempted_api_calls": completed, "completed_api_calls": completed, "by_api": {"propainter_inpaint": completed}}
    if (not completed or value.get("completed_clean_sessions") != completed or value.get("completed_model_candidates") != completed
            or value.get("blocked_sessions") != blocked or value.get("failed_sessions") != 0
            or value.get("execution_diagnostic_complete") is not True or value.get("model_call_accounting") != accounting
            or any(value.get(k) is not False for k in ("candidate_adopted", "training_eligible", "control_ground_truth"))):
        raise ContractError("NATIVE_CLEAN_SCOPE_OR_MODEL_ACCOUNTING_DRIFT")
    return {"validation": "PASS_EXECUTION_STRUCTURE_ONLY", "completed_clean_sessions": completed,
        "blocked_sessions": blocked, "visual_quality": "NOT_REVIEWED", "training_eligible": False}


def validate_native_clean_result(packet: Mapping, value: Mapping, *, decoder: Callable[[Path], int]) -> dict:
    """Validate sealed current-Mask Clean outputs at each true native length."""
    _packet_contract(packet)
    if packet["clean_config"]["reference_policy"] == NATIVE_REFERENCE_POLICY:
        return _validate_native_reference_clean_result(packet, value, decoder=decoder)
    planned, rows = packet["sessions"], value.get("sessions", [])
    if (value.get("schema") != NATIVE_RESULT_SCHEMA or value.get("task_id") != packet["task_id"]
            or value.get("run_signature") != packet["run_signature"] or value.get("experiment_scope") != NATIVE_SCOPE
            or [r.get("session_key") for r in rows] != [r["session_key"] for r in planned]
            or value.get("planned_sessions") != len(planned)):
        raise ContractError("NATIVE_CLEAN_RESULT_IDENTITY_FAILED")
    root, completed, blocked, attempts = Path(packet["output_dir"]), 0, 0, 0
    for plan, row in zip(planned, rows):
        if (row.get("native_frames") != plan["native_frames"] or row.get("session_id") != plan["session_id"]
                or ("source_receipt" in plan and row.get("source") != plan["source_receipt"])
                or ("frame_map_ref" in plan and row.get("frame_map") != plan["frame_map_ref"])):
            raise ContractError("NATIVE_CLEAN_SESSION_SOURCE_IDENTITY_FAILED")
        if plan["upstream_status"] == "BLOCKED":
            verify_ref(plan["diagnostic_ref"])
            if (row.get("execution") != "BLOCKED_UPSTREAM" or row.get("model_api_calls_attempted") != 0
                    or row.get("diagnostic_ref") != plan["diagnostic_ref"] or row.get("failure_code") != plan["failure_code"]
                    or any(k in row for k in ("clean", "clean_receipt", "review"))):
                raise ContractError("NATIVE_BLOCKED_CLEAN_CANNOT_CLAIM_STAGE")
            blocked += 1
            continue
        if row.get("execution") != "COMPLETE_CLEAN_BASELINE" or row.get("model_api_calls_attempted") != 1:
            raise ContractError("NATIVE_READY_CLEAN_NOT_COMPLETE")
        raw, mapping, masks, _ = load_native_clean_inputs(packet, plan)
        n = plan["native_frames"]
        reference = row.get("clean_receipt")
        if (row.get("clean") != reference or not isinstance(reference, dict)
                or Path(reference["path"]) != root / "sessions" / plan["session_key"] / "A/Clean/RECEIPT.json"):
            raise ContractError("NATIVE_CLEAN_RESULT_WRITE_SET_IDENTITY_DRIFT")
        stage = sealed_receipt(reference, expected_stage="Clean")
        actual = stage.get("actual_inputs", {})
        mask_stage = sealed_receipt(plan["mask_receipt"], expected_stage="Mask")
        effective_source = {"mask_arrays": output_named(mask_stage, "MASK_ARRAYS.npz"),
            "array_key": "effective_removal", "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}
        if (stage.get("component_signature") != plan["clean_signature"]
                or stage.get("native_input_frames") != n or stage.get("native_output_frames") != n
                or stage.get("settings") != SETTINGS or stage.get("reference_mode") != "FROZEN_BASELINE"
                or stage.get("reference_selection_diagnostic_hook") is not False
                or actual.get("raw_source") != plan["raw_source_ref"] or actual.get("mask_receipt") != plan["mask_receipt"]
                or actual.get("models") != packet["model_weights"] or actual.get("clean_context_refs") != {}
                or actual.get("effective_removal_source") != effective_source
                or actual.get("frames_sha256") != frames_sha256(raw)
                or actual.get("frame_map_sha256") != canonical_sha(mapping)
                or stage.get("authority", {}).get("task_id") != packet["task_id"]
                or stage.get("authority", {}).get("run_signature") != packet["run_signature"]):
            raise ContractError("NATIVE_CLEAN_STAGE_SIGNATURE_OR_ACTUAL_INPUTS_DRIFT")
        if json.loads(Path(output_named(stage, "FRAME_MAP.json")["path"]).read_text()) != mapping:
            raise ContractError("NATIVE_CLEAN_OUTPUT_MAP_DRIFT")
        states = json.loads(Path(output_named(stage, "FRAME_STATUS.json")["path"]).read_text())
        if not isinstance(states, list) or len(states) != n or any(
                state.get("native_frame_index") != i for i, state in enumerate(states)):
            raise ContractError("NATIVE_CLEAN_FULL_FRAME_STATUS_DRIFT")
        with np.load(output_named(stage, "CLEAN_ARRAYS.npz")["path"], allow_pickle=False) as archive:
            from .clean_references import validate_clean_preservation
            if (validate_clean_preservation(raw, archive["clean_rgb"], archive["effective_removal"], archive["protected_visible"])["status"] != "PASS"
                    or not np.array_equal(archive["effective_removal"], masks["effective_removal"])
                    or not np.array_equal(archive["protected_visible"], masks["protected_visible"])):
                raise ContractError("NATIVE_CLEAN_ACTUAL_PIXEL_PRESERVATION_DRIFT")
        review = row.get("review", {})
        verify_ref(review["video"])
        if (not Path(review["video"]["path"]).is_relative_to(root)
                or review.get("native_frames") != n or decoder(Path(review["video"]["path"])) != n
                or review.get("source_frame_map_ref") != plan["frame_map_ref"]
                or review.get("frame_map_ref") != output_named(stage, "FRAME_MAP.json")):
            raise ContractError("NATIVE_CLEAN_ACTUAL_REVIEW_DECODE_OR_MAP_DRIFT")
        completed += 1
        attempts += 1
    expected = {"unit": MODEL_CALL_UNIT, "maximum": packet["budgets"]["model_calls_max"],
        "attempted_api_calls": attempts, "completed_api_calls": completed, "by_api": {"propainter_inpaint": attempts}}
    if (completed == 0 or value.get("completed_clean_sessions") != completed
            or value.get("completed_model_candidates") != completed or value.get("blocked_sessions") != blocked
            or value.get("failed_sessions") != 0 or value.get("execution_diagnostic_complete") is not True
            or value.get("model_call_accounting") != expected
            or any(value.get(k) is not False for k in ("candidate_adopted", "training_eligible", "control_ground_truth"))):
        raise ContractError("NATIVE_CLEAN_SCOPE_OR_MODEL_ACCOUNTING_DRIFT")
    return {"validation": "PASS_EXECUTION_STRUCTURE_ONLY", "completed_clean_sessions": completed,
        "blocked_sessions": blocked, "visual_quality": "NOT_REVIEWED", "training_eligible": False}
