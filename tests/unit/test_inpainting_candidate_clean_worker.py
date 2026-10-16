"""CPU synthetic G2 worker boundaries; never load weights, Torch or CUDA.

Native length remains 238. Tiny 8x8 fixtures do not claim real model results:
the inpainting provider and codec are fake, while provenance receipts, Raw RAFT
capture, reference diagnostics and final validation use the real implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.inpainting_candidate import clean_flow as flow
from chaoyang.inpainting_candidate import clean_worker as worker
from chaoyang.inpainting_candidate import model_stages
from chaoyang.inpainting_candidate.automatic_masks import frames_sha256
from chaoyang.inpainting_candidate.clean_references import validate_clean_preservation
from chaoyang.inpainting_candidate.contracts import SCHEMA, canonical_sha, file_ref, verify_sealed_stage
from chaoyang.governance.inpainting_candidate_registration import (CLEAN_MODEL_CALLS_UNIT, CLEAN_SPEC_SCHEMA,
                                                                   CLEAN_STAGE_SCOPE, VISUAL_OPERATION)


def json_ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_ref(path)


def array_ref(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    return file_ref(path)


def fake_fb(forward, backward, parent):
    shape = (forward.shape[1], *forward.shape[-2:])
    return {"fb_residual_squared_forward": np.zeros(shape, np.float32),
            "fb_residual_squared_backward": np.zeros(shape, np.float32),
            "vendor_fb_consistent_forward": np.ones(shape, bool),
            "vendor_fb_consistent_backward": np.ones(shape, bool)}


@pytest.fixture
def full_native_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(worker, "PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(worker, "REPO_ROOT", tmp_path, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-clean-cpu-fixture")
    count, height, width = 238, 8, 8
    rgb = np.stack([np.full((height, width, 3), i % 256, np.uint8) for i in range(count)])
    mapping = [{"native_frame_index": i, "decoded_display_index": i,
                "model_input_frame": i, "output_frame": i,
                "qpc_declared_ns": 1_000_000_000 + i * 40_000_000,
                "decoder_pts": None} for i in range(count)]
    raw_path = tmp_path / "raw/001/vst.h264"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"CPU synthetic raw descriptor, not decoded or used by a model")
    raw_ref = file_ref(raw_path)
    source_dir = tmp_path / "sealed_source/001"
    rgb_ref = array_ref(source_dir / "RAW_RGB.npz", rgb=rgb)
    map_ref = json_ref(source_dir / "FRAME_MAP.json", mapping)
    report_ref = json_ref(source_dir / "SOURCE_REPORT.json", {"session_id": "pick_place_1008_001",
        "actual_inputs": {"video": raw_ref}, "native_sidecar_frames": count,
        "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS"})
    source_ref = json_ref(source_dir / "RECEIPT.json", {"schema": SCHEMA, "stage": "Source",
        "status": "SEALED_COMPLETE", "component_signature": "c" * 64, "native_frames": count,
        "actual_inputs": [], "outputs": [rgb_ref, map_ref, report_ref]})
    removal = np.zeros((count, height, width), bool)
    removal[:, 3:5, 3:5] = True
    protected = np.zeros_like(removal)
    protected[:, 0, 0] = True
    masks = {"removal": removal, "effective_removal": removal.copy(), "protected_visible": protected,
             "core": removal.copy(), "equipment": np.zeros_like(removal),
             "held": np.zeros_like(removal), "unknown": np.zeros_like(removal)}
    mask_dir = tmp_path / "sealed_mask/001"
    mask_array_ref = array_ref(mask_dir / "MASK_ARRAYS.npz", **masks)
    mask_map_ref = json_ref(mask_dir / "FRAME_MAP.json", mapping)
    mask_ref = json_ref(mask_dir / "RECEIPT.json", {"schema": SCHEMA, "stage": "Mask",
        "status": "SEALED_COMPLETE", "component_signature": "d" * 64,
        "native_input_frames": count, "native_output_frames": count,
        "manual_points_submitted": 0, "producer_manual_seed_reads": 0,
        "actual_inputs": {"raw_source": raw_ref, "frames_sha256": frames_sha256(rgb),
                          "frame_map_sha256": canonical_sha(mapping)},
        "outputs": [mask_array_ref, mask_map_ref]})
    weights = []
    for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
        path = tmp_path / "assets/models/vendor/propainter" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("CPU synthetic weight reference, never loaded:" + name).encode())
        weights.append(file_ref(path))
    code_path = tmp_path / "fake_runtime.py"
    code_path.write_text("# CPU signed synthetic runtime declaration, not executed\n")
    runtime = [file_ref(code_path)]
    config_ref = json_ref(tmp_path / "capture_config.json", {"schema": flow.CAPTURE_SCHEMA,
        "session_id": "pick_place_1008_001", "native_shape": [count, height, width],
        "source_receipt": source_ref, "raw_source_ref": raw_ref, "raw_rgb_ref": rgb_ref,
        "frame_map_ref": map_ref, "raw_frames_sha256": frames_sha256(rgb),
        "frame_map_payload_sha256": canonical_sha(mapping), "raft_weight_ref": weights[0],
        "runtime_code": runtime, "raft_iter": 20, "chunk_length": 12, "preprocessing": flow.PREPROCESSING})
    output = tmp_path / "output"
    output.mkdir()
    packet = {"task_id": "inpainting_generalization_g2_clean_cpu_fixture", "run_signature": "a" * 64,
        "spec_schema": CLEAN_SPEC_SCHEMA, "actual_stage_scope": CLEAN_STAGE_SCOPE,
        "model_stages": ["Clean"], "operation": VISUAL_OPERATION, "algorithm_execution_allowed": True,
        "output_dir": str(output), "runtime_code": runtime, "model_weights": weights,
        "config": {"candidate_id": "G2_FULL_NATIVE_REFS_ONLY", "manual_seed_inputs": 0},
        "budgets": {"model_calls_max": 2, "model_calls_unit": CLEAN_MODEL_CALLS_UNIT,
                    "gpu_count": 1, "wall_seconds_max": 28800, "training_steps_max": 0, "candidates_max": 2},
        "clean_config": {"capture_config_ref": config_ref, "frozen_settings": dict(SETTINGS),
            "background_registration": "ABSENT", "reference_policy": "RAW_FB_DIAGNOSTIC_UNKNOWN_NO_QUALIFIED_SUPPORT",
            "evaluation_anchor_frames": [0, 50, 119, 132, 202, 237], "candidates": ["A", "B_IF_REFS_CHANGED"]},
        "sessions": [{"session_key": "001", "native_frames": count, "source_receipt": source_ref,
                      "mask_receipt": mask_ref, "raw_source_ref": raw_ref, "frame_map_ref": map_ref}]}
    owner = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
             "fencing_token": "CPU-synthetic-fence", "gpu_uuid": "GPU-clean-cpu-fixture",
             "worker_logical_index": 0, "execution_allowed": True, "algorithm_execution_allowed": True}
    calls = {"load": 0, "inpaint": 0, "raw_raft": 0, "video_frames": []}

    class Parent:
        settings = dict(SETTINGS)

        def raft(self, tensor, *, iters):
            calls["raw_raft"] += 1
            assert iters == 20 and tensor.dtype == np.float32
            shape = (1, tensor.shape[1] - 1, 2, height, width)
            return np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def load_models(*args, **kwargs):
        calls["load"] += 1
        assert kwargs["need_sam"] is False and kwargs["need_clean"] is True
        return {"propainter": Parent(), "load_receipt": {"synthetic_cpu": True}}

    def fake_stage(frames_rgb, *, masks, parent_propainter, output_dir, raw_source_ref, frame_map,
                   model_refs, component_signature, mask_receipt_ref, authorization,
                   clean_context_refs=None, reference_selector=None, record_reference_selection=False):
        calls["inpaint"] += 1
        assert len(frames_rgb) == 238 and list(frame_map) == mapping
        stage_owner = authorization()
        for plan in flow.chunk_plan(len(frames_rgb)):
            raw = frames_rgb[plan["first_frame"]:plan["end_frame_exclusive"]]
            tensor = (raw.astype(np.float32).transpose(0, 3, 1, 2)[None] / np.float32(127.5) - np.float32(1))
            parent_propainter.raft(tensor, iters=20)
        clean = frames_rgb.copy()
        clean[masks["effective_removal"]] = 117
        validation = validate_clean_preservation(frames_rgb, clean, masks["effective_removal"], masks["protected_visible"])
        refs = []
        for mid in range(0, 238, 5):
            neighbors = list(range(max(0, mid - 5), min(238, mid + 6)))
            original = _get_ref_index(mid, neighbors, 238, 10, 8)
            selected = original if reference_selector is None else reference_selector(mid=mid,
                neighbors=neighbors, length=238, stride=10, ref_num=8, original_references=original)
            refs.append({"target_frame": mid, "neighbors": neighbors, "original_references": original,
                         "references": selected, "reference_count": len(original), "settings_unchanged": True})
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        outputs = [array_ref(root / "CLEAN_ARRAYS.npz", clean_rgb=clean,
                    effective_removal=masks["effective_removal"], protected_visible=masks["protected_visible"]),
                   json_ref(root / "FRAME_MAP.json", frame_map), json_ref(root / "REFERENCE_SELECTION.json", refs),
                   json_ref(root / "PRESERVATION.json", validation),
                   json_ref(root / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(238)])]
        receipt = {"schema": SCHEMA, "stage": "Clean", "status": "SEALED_COMPLETE",
            "component_signature": component_signature, "native_input_frames": 238, "native_output_frames": 238,
            "authority": stage_owner,
            "settings": dict(SETTINGS), "outputs": outputs, "actual_inputs": {
                "raw_source": dict(raw_source_ref), "mask_receipt": dict(mask_receipt_ref),
                "models": [dict(ref) for ref in model_refs], "clean_context_refs": dict(clean_context_refs or {}),
                "frames_sha256": frames_sha256(frames_rgb), "frame_map_sha256": canonical_sha(frame_map)},
            "reference_mode": "FROZEN_BASELINE" if reference_selector is None else "SINGLE_REF_SELECTION_CHANGE",
            "reference_selection_diagnostic_hook": record_reference_selection, "validation": validation,
            "time_mode": "OFFLINE_BIDIRECTIONAL_PROPAINTER", "causal_training_input": False,
            "training_eligible": False, "control_ground_truth": False, "visual_quality": "NOT_EVALUATED"}
        json_ref(root / "RECEIPT.json", receipt)
        return receipt

    def video(path, frames, frame_map):
        assert len(frames) == 238 and frame_map == mapping
        calls["video_frames"].append(len(frames))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"CPU fake review video: real decode separately substituted")
        return {"video": file_ref(path), "decoded_frames": 238, "native_frames": 238}

    monkeypatch.setattr(worker, "load_frozen_models", load_models)
    monkeypatch.setattr(worker, "run_clean_stage", fake_stage)
    monkeypatch.setattr(worker, "write_review_video", video)
    monkeypatch.setattr(flow, "vendor_fb_diagnostics", fake_fb)
    return SimpleNamespace(packet=packet, rgb=rgb, mapping=mapping, masks=masks, owner=owner,
                           authority=lambda: dict(owner), calls=calls, output=output,
                           source_ref=source_ref, mask_ref=mask_ref, config_ref=config_ref)


def test_full_native_a_sealed_noop_b_is_one_model_execution(full_native_fixture):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    assert result["schema"] == "inpainting-g2-clean-batch-result-v1"
    assert result["planned_sessions"] == result["completed_clean_sessions"] == 1
    assert result["completed_model_candidates"] == 1
    row = result["sessions"][0]
    assert row["session_key"] == "001" and row["native_frames"] == 238
    assert row["source"] == f.source_ref and row["mask"] == f.mask_ref
    assert row["candidates"]["B"]["execution"] == "NOT_EXECUTED_NO_REFERENCE_CHANGE"
    assert row["candidates"]["B"]["model_calls"] == 0
    assert f.calls["inpaint"] == 1 and f.calls["raw_raft"] == 20
    accounting = result["model_call_accounting"]
    assert accounting["unit"] == CLEAN_MODEL_CALLS_UNIT and accounting["attempted_api_calls"] == 1
    assert accounting["by_api"] == {"propainter_inpaint": 1}
    seen = []
    worker.validate_clean_result(f.packet, result, decoder=lambda path: seen.append(path) or 238)
    assert len(seen) >= 1 and f.calls["video_frames"]
    assert json.loads((f.output / "MODEL_RESULT.json").read_text()) == result


def test_actual_decode_failure_not_self_reported_count(full_native_fixture):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    with pytest.raises(ValueError):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 237)


def test_denied_authority_does_not_load_any_model(full_native_fixture):
    f = full_native_fixture

    def deny():
        raise RuntimeError("NO_FENCED_PACKET")

    with pytest.raises(RuntimeError, match="NO_FENCED_PACKET"):
        worker.run_clean_batch(f.packet, authority=deny)
    assert f.calls["load"] == f.calls["inpaint"] == 0


@pytest.mark.parametrize("field,value", [("model_stages", ["Mask"]),
    ("actual_stage_scope", "FULL_RAW_TO_ROBOT"), ("spec_schema", "other"),
    ("sessions", [{"session_key": "001", "native_frames": 237}])])
def test_invalid_scope_never_starts_model(full_native_fixture, field, value):
    f = full_native_fixture
    f.packet[field] = value
    with pytest.raises((ValueError, KeyError)):
        worker.run_clean_batch(f.packet, authority=f.authority)
    assert f.calls["load"] == f.calls["inpaint"] == 0


@pytest.mark.parametrize("field,value", [("task_id", "different"), ("run_signature", "b" * 64),
    ("completed_model_candidates", 2), ("completed_clean_sessions", 0), ("stage_scope", "G1_AUTOMATIC_MASK_ONLY")])
def test_result_identity_counts_or_scope_not_a_pass(full_native_fixture, field, value):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    result[field] = value
    with pytest.raises(ValueError):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


@pytest.mark.parametrize("artifact", ["source", "mask", "clean", "video", "flow", "reference"])
def test_sealed_actual_artifact_sha_drift_is_rejected(full_native_fixture, artifact):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    row = result["sessions"][0]
    refs = {"source": row["source"], "mask": row["mask"], "clean": row["candidates"]["A"]["receipt"],
            "video": row["candidates"]["A"]["review"]["video"],
            "flow": row["shared"]["raw_flow"], "reference": row["shared"]["reference_evidence"]}
    Path(refs[artifact]["path"]).write_bytes(b"changed after sealed CPU fixture")
    with pytest.raises(ValueError):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


def test_failed_capture_keeps_legal_a_but_cannot_seal_complete_batch(full_native_fixture, monkeypatch):
    f = full_native_fixture

    def fail_capture(self):
        raise ValueError("CPU_CAPTURE_ARCHIVE_FAILED")

    monkeypatch.setattr(flow.RawRaftCaptureReplay, "seal", fail_capture)
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 0
    assert result["completed_model_candidates"] == 1
    row = result["sessions"][0]
    assert row["candidates"]["A"]["execution"] == "COMPLETE_CLEAN"
    ref = row["candidates"]["A"]["receipt"]
    receipt = json.loads(Path(ref["path"]).read_text())
    verify_sealed_stage(receipt, receipt["component_signature"], output_root=Path(ref["path"]).parent)
    assert row["candidates"]["B"]["execution"] == "NOT_EXECUTED_SHARED_EVIDENCE_UNAVAILABLE"
    assert f.calls["inpaint"] == 1
    with pytest.raises(ValueError):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


@pytest.mark.parametrize("field,value", [("model_calls_unit", "SAM_ADAPTER_INFERENCE_API_INVOCATION"),
    ("model_calls_max", 0), ("model_calls_max", 3), ("model_calls_max", True)])
def test_invalid_outer_api_budget_rejected_before_constructor(full_native_fixture, field, value):
    f = full_native_fixture
    f.packet["budgets"][field] = value
    with pytest.raises(ValueError, match="BUDGET"):
        worker.run_clean_batch(f.packet, authority=f.authority)
    assert f.calls["load"] == f.calls["inpaint"] == 0


@pytest.mark.parametrize("field", ["native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"])
def test_resigned_but_non_native_frame_map_rejected_before_constructor(full_native_fixture, field):
    f = full_native_fixture
    row = f.packet["sessions"][0]
    changed = [dict(item) for item in f.mapping]
    changed[1][field] = 0  # A duplicate, not a SHA-only corruption test.
    new_map_ref = json_ref(Path(row["frame_map_ref"]["path"]), changed)
    source_path = Path(row["source_receipt"]["path"])
    source = json.loads(source_path.read_text())
    source["outputs"] = [new_map_ref if Path(ref["path"]).name == "FRAME_MAP.json" else ref
                         for ref in source["outputs"]]
    row["source_receipt"] = json_ref(source_path, source)
    row["frame_map_ref"] = new_map_ref
    with pytest.raises(ValueError, match="NATIVE_FRAME_MAPPING"):
        worker.run_clean_batch(f.packet, authority=f.authority)
    assert f.calls["load"] == f.calls["inpaint"] == 0


@pytest.mark.parametrize("field", ["frames_sha256", "frame_map_sha256"])
def test_resigned_mask_with_other_pixel_or_frame_identity_cannot_start(full_native_fixture, field):
    f = full_native_fixture
    row = f.packet["sessions"][0]
    path = Path(row["mask_receipt"]["path"])
    mask = json.loads(path.read_text())
    mask["actual_inputs"][field] = "e" * 64
    row["mask_receipt"] = json_ref(path, mask)
    with pytest.raises(ValueError, match="MASK_SOURCE_OR_FRAME_IDENTITY"):
        worker.run_clean_batch(f.packet, authority=f.authority)
    assert f.calls["load"] == f.calls["inpaint"] == 0


@pytest.mark.parametrize("field,value", [("execution", "COMPLETE_CLEAN"), ("model_calls", 1), ("receipt", "fake")])
def test_noop_b_cannot_be_relabelled_second_execution(full_native_fixture, field, value):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    result["sessions"][0]["candidates"]["B"][field] = value
    with pytest.raises(ValueError, match="B_NOOP"):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


def test_unknown_background_receipt_not_promoted_by_resigning_metadata(full_native_fixture):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    row = result["sessions"][0]
    path = Path(row["shared"]["reference_evidence"]["path"])
    evidence = json.loads(path.read_text())
    evidence["qualified_visible_background_pixels"] = 1
    changed_ref = json_ref(path, evidence)
    row["shared"]["reference_evidence"] = changed_ref
    row["candidates"]["B"]["reference_evidence"] = changed_ref
    with pytest.raises(ValueError, match="UNKNOWN_REFERENCE_QUALIFICATION"):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


@pytest.mark.parametrize("name", ["capture_config", "raw_flow_receipt", "reference_evidence"])
def test_clean_component_signature_tracks_each_actual_context_reference(full_native_fixture, name):
    f = full_native_fixture
    context = {name: f.config_ref}
    original = worker.clean_component_signature(f.packet, f.packet["sessions"][0], context)
    changed = {name: {**f.config_ref, "sha256": "b" * 64}}
    assert original != worker.clean_component_signature(f.packet, f.packet["sessions"][0], changed)


def test_resigned_clean_protection_array_must_equal_frozen_mask_not_just_preserve_pixels(full_native_fixture):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    candidate = result["sessions"][0]["candidates"]["A"]
    path = Path(candidate["receipt"]["path"])
    receipt = json.loads(path.read_text())
    original = next(ref for ref in receipt["outputs"] if Path(ref["path"]).name == "CLEAN_ARRAYS.npz")
    with np.load(original["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["protected_visible"] = np.zeros_like(arrays["protected_visible"])
    changed = array_ref(Path(original["path"]), **arrays)
    receipt["outputs"] = [changed if ref == original else ref for ref in receipt["outputs"]]
    candidate["receipt"] = json_ref(path, receipt)
    # RGB remains pixel-preserving and the SHA chain is valid: P identity itself
    # must reject this before any downstream evidence-reference difference.
    with pytest.raises(ValueError, match="ACTUAL_ARRAY_PRESERVATION"):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)


def test_resigned_unknown_support_array_needs_exact_native_domain(full_native_fixture):
    f = full_native_fixture
    result = worker.run_clean_batch(f.packet, authority=f.authority)
    row = result["sessions"][0]
    path = Path(row["shared"]["reference_evidence"]["path"])
    receipt = json.loads(path.read_text())
    original = next(ref for ref in receipt["outputs"] if Path(ref["path"]).name == "SUPPORT_ARRAYS.npz")
    with np.load(original["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["background_support_unknown"] = np.ones((237, 8, 8), bool)
    changed = array_ref(Path(original["path"]), **arrays)
    receipt["outputs"] = [changed if ref == original else ref for ref in receipt["outputs"]]
    changed_receipt = json_ref(path, receipt)
    row["shared"]["reference_evidence"] = changed_receipt
    row["candidates"]["B"]["reference_evidence"] = changed_receipt
    with pytest.raises(ValueError, match="UNKNOWN_SUPPORT"):
        worker.validate_clean_result(f.packet, result, decoder=lambda path: 238)
