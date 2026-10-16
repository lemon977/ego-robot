"""Synthetic CPU-only bridges; no model, physical geometry or video claims."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.inpainting_candidate import clean_flow as flow
from chaoyang.inpainting_candidate import model_stages
from chaoyang.inpainting_candidate.automatic_masks import frames_sha256
from chaoyang.inpainting_candidate.contracts import SCHEMA, ContractError, canonical_sha, file_ref, verify_sealed_stage


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-fake")
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True,
             "task_id": "CPU_FAKE_NOT_REAL_MODEL_EXECUTION", "run_signature": "a" * 64,
             "fencing_token": "fake", "worker_logical_index": 0, "gpu_uuid": "GPU-cpu-fake"}
    raw = np.arange(25 * 8 * 8 * 3, dtype=np.int64).reshape(25, 8, 8, 3).astype(np.uint8)
    mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
                "output_frame": i, "acquisition_frame_id": "UNKNOWN"} for i in range(25)]
    folder = tmp_path / "source"
    folder.mkdir()
    video = tmp_path / "raw/vst.h264"
    video.parent.mkdir()
    video.write_bytes(b"FAKE CPU SOURCE, NOT A REAL VIDEO")
    weight = tmp_path / "raft-things.pth"
    weight.write_bytes(b"FAKE CPU WEIGHT NOT LOADED")
    code = tmp_path / "code.py"
    code.write_bytes(b"# fake frozen code identity\n")
    raw_ref = flow.write_npz(folder / "RAW_RGB.npz", rgb=raw)
    map_ref = flow.write_json(folder / "FRAME_MAP.json", mapping)
    source = {"schema": SCHEMA, "stage": "Source", "status": "SEALED_COMPLETE",
              "component_signature": "b" * 64, "native_frames": len(raw),
              "outputs": [raw_ref, map_ref]}
    source_ref = flow.write_json(folder / "RECEIPT.json", source)
    config = {"schema": flow.CAPTURE_SCHEMA, "session_id": "fake001", "native_shape": list(raw.shape[:3]),
        "source_receipt": source_ref, "raw_source_ref": file_ref(video), "raw_rgb_ref": raw_ref,
        "frame_map_ref": map_ref, "raw_frames_sha256": frames_sha256(raw),
        "frame_map_payload_sha256": canonical_sha(mapping), "raft_weight_ref": file_ref(weight),
        "runtime_code": [file_ref(code)], "raft_iter": 20, "chunk_length": 12,
        "preprocessing": flow.PREPROCESSING}
    config_ref = flow.write_json(tmp_path / "CAPTURE_CONFIG.json", config)
    calls = []
    def raft(value, *, iters):
        calls.append((value, iters))
        shape = (1, value.shape[1] - 1, 2, 8, 8)
        result = (np.zeros(shape, np.float32), np.zeros(shape, np.float32))
        returned.append(result)
        return result
    returned = []
    parent = SimpleNamespace(raft=raft, settings=dict(SETTINGS))
    return SimpleNamespace(root=tmp_path, raw=raw, mapping=mapping, parent=parent,
        config=config, config_ref=config_ref, owner=owner, authority=lambda: dict(owner),
        original_calls=calls, original_returns=returned)


def observer(env, name="capture", **kwargs):
    return flow.RawRaftCaptureReplay(env.parent, raw_rgb=env.raw, frame_map=env.mapping,
        capture_config_ref=env.config_ref, output_dir=env.root / name, authorization=env.authority, **kwargs)


def tensor_for(env, plan):
    raw = env.raw[plan["first_frame"]:plan["end_frame_exclusive"]]
    return raw.transpose(0, 3, 1, 2).astype(np.float32)[None] / 127.5 - 1


def run_capture(env):
    capture = observer(env)
    for plan in capture.plan:
        result = capture.parent.raft(tensor_for(env, plan), iters=20)
        assert result is env.original_returns[-1]
    receipt = capture.seal()
    return capture, receipt, file_ref(capture.root / "RECEIPT.json")


def test_import_no_top_level_torch_and_capture_has_no_model_constructor():
    code = inspect.getsource(flow)
    assert "import torch" not in "\n".join(line for line in code.splitlines() if not line.startswith(" "))
    assert "load_frozen_models" not in code


def test_full238_frozen_twenty_chunks_cover237_edges_once():
    plan = flow.chunk_plan(238)
    assert len(plan) == 20
    edges = [i for row in plan for i in range(row["first_edge"], row["end_edge_exclusive"])]
    assert edges == list(range(237))
    assert plan[0]["first_frame"] == 0 and plan[1]["first_frame"] == 11
    assert plan[-1]["end_frame_exclusive"] == 238


def test_parent_is_private_copy_and_raw_model_tuple_is_unmodified(env):
    original_raft = env.parent.raft
    capture, receipt, ref = run_capture(env)
    assert capture.parent is not env.parent
    assert env.parent.raft is original_raft
    assert capture.parent.raft is capture
    assert len(env.original_calls) == 3
    assert receipt["returned_original_output_identity"] is True
    assert receipt["adjacent_edges"] == 24
    assert receipt["model_subcalls"] == {"RAFT_bi": 3, "fix_raft_forward_backward": 6}
    assert receipt["background_geometry_qualified"] is False
    verify_sealed_stage(receipt, receipt["component_signature"], output_root=capture.root)
    assert ref["path"] == str(capture.root / "RECEIPT.json")


def test_replay_complete_same_input_avoids_all_raft_calls_and_seals(env):
    _, receipt, ref = run_capture(env)
    before = len(env.original_calls)
    replay = observer(env, "replay", replay_receipt_ref=ref)
    for plan in replay.plan:
        result = replay.parent.raft(tensor_for(env, plan), iters=20)
        assert len(result) == 2 and result[0].dtype == np.float32
        assert result[0].shape[1] == plan["end_edge_exclusive"] - plan["first_edge"]
    assert replay.seal() == receipt
    assert len(env.original_calls) == before


def test_replay_exact_actual_float_hash_not_only_recovered_raw_identity(env):
    _, _, ref = run_capture(env)
    replay = observer(env, "replay", replay_receipt_ref=ref)
    tensor = tensor_for(env, replay.plan[0]).copy()
    tensor[0, 0, 0, 0, 1] = np.nextafter(tensor[0, 0, 0, 0, 1], np.float32(0))
    with pytest.raises(ContractError, match="REPLAY_ACTUAL_INPUT_SHA_CHANGED"):
        replay.parent.raft(tensor, iters=20)


def test_capture_float_rounding_inverse_does_not_require_cpu_normalized_hash(env):
    capture = observer(env)
    tensor = tensor_for(env, capture.plan[0]).copy()
    tensor[0, 0, 0, 0, 1] = np.nextafter(tensor[0, 0, 0, 0, 1], np.float32(0))
    capture.parent.raft(tensor, iters=20)
    assert capture.error is None


@pytest.mark.parametrize("change,code", [
    ("dtype", "TENSOR_DOMAIN"), ("raw", "NOT_NATIVE_RAW_CHUNK"), ("iters", "SETTINGS_OR_COUNT"),
    ("shape", "TENSOR_DOMAIN"), ("nonfinite", "TENSOR_DOMAIN")])
def test_non_native_inputs_refused_before_original_model(env, change, code):
    capture = observer(env)
    value = tensor_for(env, capture.plan[0]).copy()
    iters = 20
    if change == "dtype":
        value = value.astype(np.float16)
    elif change == "raw":
        value[0, 0, 0, 0, 1] = 1
    elif change == "iters":
        iters = 21
    elif change == "shape":
        value = value[:, :-1]
    elif change == "nonfinite":
        value[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ContractError, match=code):
        capture.parent.raft(value, iters=iters)
    assert not env.original_calls


def test_incomplete_capture_has_no_consumable_receipt(env):
    capture = observer(env)
    capture.parent.raft(tensor_for(env, capture.plan[0]), iters=20)
    with pytest.raises(ContractError, match="CAPTURE_INCOMPLETE"):
        capture.seal()
    assert not (capture.root / "RECEIPT.json").exists()


def test_diagnostic_archive_error_returns_original_model_result_and_no_seal(env, monkeypatch):
    capture = observer(env)
    def fail(*args, **kwargs):
        raise OSError("synthetic archive failure")
    monkeypatch.setattr(flow, "write_npz", fail)
    for row in capture.plan:
        result = capture.parent.raft(tensor_for(env, row), iters=20)
        assert result is env.original_returns[-1]
    assert len(env.original_calls) == 3
    assert "archive failure" in capture.error
    with pytest.raises(ContractError, match="CAPTURE_INCOMPLETE"):
        capture.seal()


def test_config_sha_drift_stops_before_original_model(env):
    capture = observer(env)
    Path(env.config_ref["path"]).write_bytes(b"drift")
    with pytest.raises(ContractError, match="SHA_DRIFT"):
        capture.parent.raft(tensor_for(env, capture.plan[0]), iters=20)
    assert not env.original_calls


@pytest.mark.parametrize("drift", [{"algorithm_execution_allowed": False}, {"fencing_token": "other"}])
def test_cpu_or_changed_writer_cannot_call_raft(env, drift):
    capture = observer(env)
    env.owner.update(drift)
    with pytest.raises(ValueError, match="AUTHORITY_REQUIRED|IDENTITY_DRIFT"):
        capture.parent.raft(tensor_for(env, capture.plan[0]), iters=20)
    assert not env.original_calls


def test_historical_partial_flow_not_replayable(env):
    path = env.root / "partial"
    path.mkdir()
    ref = flow.write_json(path / "RECEIPT.json", {"schema": SCHEMA, "stage": "RawFlow",
        "status": "PARTIAL", "component_signature": "f" * 64, "outputs": []})
    with pytest.raises(ContractError, match="PARTIAL_NOT_CONSUMABLE"):
        observer(env, "replay", replay_receipt_ref=ref)


def unknown_evidence(env, *, change_calls=False, provider=None):
    capture, _, flow_ref = run_capture(env)
    folder = env.root / "mask"
    folder.mkdir()
    masks = {key: np.zeros(env.raw.shape[:3], bool) for key in
             ("effective_removal", "protected_visible", "removal", "core", "equipment", "unknown", "held")}
    masks["effective_removal"][:, 2:4, 2:4] = True
    masks["removal"][:] = masks["effective_removal"]
    arrays = flow.write_npz(folder / "MASK_ARRAYS.npz", **masks)
    mask_ref = flow.write_json(folder / "RECEIPT.json", {"schema": SCHEMA, "stage": "Mask",
        "status": "SEALED_COMPLETE", "component_signature": "c" * 64, "outputs": [arrays],
        "actual_inputs": {"frames_sha256": frames_sha256(env.raw), "frame_map_sha256": canonical_sha(env.mapping)}})
    calls = []
    for mid in range(0, 25, 5):
        neighbors = list(range(max(0, mid - 5), min(25, mid + 6)))
        refs = _get_ref_index(mid, neighbors, 25, 10, -1)
        calls.append({"target_frame": mid, "neighbors": neighbors,
                      "original_references": refs, "references": refs, "reference_count": len(refs)})
    clean_folder = env.root / "clean"
    clean_folder.mkdir()
    calls_ref = flow.write_json(clean_folder / "REFERENCE_SELECTION.json", calls)
    clean_ref = flow.write_json(clean_folder / "RECEIPT.json", {"schema": SCHEMA, "stage": "Clean",
        "status": "SEALED_COMPLETE", "component_signature": "d" * 64, "outputs": [calls_ref]})
    def fake_diagnostics(f, b, parent):
        shape = (f.shape[1], *f.shape[-2:])
        return {"fb_residual_squared_forward": np.zeros(shape, np.float32),
            "fb_residual_squared_backward": np.zeros(shape, np.float32),
            "vendor_fb_consistent_forward": np.ones(shape, bool),
            "vendor_fb_consistent_backward": np.ones(shape, bool)}
    if change_calls:
        calls[0]["neighbors"] = [0]
    return flow.seal_reference_evidence(raw_flow_receipt_ref=flow_ref, mask_receipt_ref=mask_ref,
        clean_receipt_ref=clean_ref, masks=masks, reference_calls=calls, output_dir=env.root / "evidence",
        parent=env.parent, authorization=env.authority, diagnostic_provider=provider or fake_diagnostics)


def test_perfect_raw_fb_still_zero_qualified_donors_no_changed_refs(env):
    evidence = unknown_evidence(env)
    assert evidence["qualified_reference_pairs"] == evidence["qualified_visible_background_pixels"] == 0
    assert evidence["changed_reference_calls"] == 0
    assert evidence["support_status"] == "UNKNOWN_BACKGROUND_REGISTRATION_ABSENT"
    with np.load(flow.output_named(evidence, "SUPPORT_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        assert archive["raw_unremoved_candidate"].any()
        assert not archive["qualified_visible_background"].any()
        assert archive["background_support_unknown"].all()
    plans = json.loads(Path(flow.output_named(evidence, "REFERENCE_PLAN.json")["path"]).read_text())
    assert all(row["references"] == row["original_references"] for row in plans)
    assert all(row["status"].startswith("BASELINE_") for row in plans)


def test_reference_calls_must_be_from_sealed_A_not_caller_self_report(env):
    with pytest.raises(ContractError, match="NOT_FROM_SEALED_A"):
        unknown_evidence(env, change_calls=True)


def test_invalid_fb_diagnostics_no_evidence_receipt(env):
    with pytest.raises(ContractError, match="FB_DIAGNOSTIC_FIELDS_INVALID"):
        unknown_evidence(env, provider=lambda *args: {})
    assert not (env.root / "evidence/RECEIPT.json").exists()
