"""CPU checks of frozen pair inputs and MCO consumer integration."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chaoyang.governance.common import artifact_ref
from chaoyang.inpainting_candidate import v1_clean_pair, v1_mco_worker
from chaoyang.pick_place_1008.visual import effective_clean_masks
from tests.test_inpainting_v1_clean_pair import fixture_arrays, synthetic_parent


@pytest.fixture
def source_binding(monkeypatch, tmp_path):
    monkeypatch.setattr(v1_mco_worker, "REPO_ROOT", tmp_path)
    raw, _, _ = fixture_arrays(4)
    np.savez_compressed(tmp_path / "RAW_RGB.npz", rgb=raw)
    session = SimpleNamespace(qpc_ns=np.arange(4, dtype=np.int64) + 100)
    mapping = [{"native_frame_index": i, "output_frame": i,
                "qpc_declared_ns": int(session.qpc_ns[i])} for i in range(4)]
    (tmp_path / "FRAME_MAP.json").write_text(json.dumps(mapping))
    packet = {"inputs": {"frozen_source_raw": artifact_ref(tmp_path / "RAW_RGB.npz"),
                         "frozen_frame_map": artifact_ref(tmp_path / "FRAME_MAP.json")}}
    return packet, raw, session, mapping, tmp_path


def test_source_binding_checks_actual_rgb_and_frame_map(source_binding):
    packet, raw, session, _, _ = source_binding
    refs = v1_mco_worker._require_pair_source_inputs(packet, raw, session)
    assert refs == (packet["inputs"]["frozen_source_raw"], packet["inputs"]["frozen_frame_map"])


def test_changed_source_pixels_rejected_even_with_valid_own_sha(source_binding):
    packet, raw, session, _, _ = source_binding
    raw[0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="SOURCE_RGB_MISMATCH"):
        v1_mco_worker._require_pair_source_inputs(packet, raw, session)


@pytest.mark.parametrize("change", ["count", "qpc", "native", "output"])
def test_valid_sha_but_wrong_native_frame_mapping_rejected(source_binding, change):
    packet, raw, session, mapping, root = source_binding
    if change == "count":
        mapping.pop()
    elif change == "qpc":
        mapping[1]["qpc_declared_ns"] += 1
    elif change == "native":
        mapping[1]["native_frame_index"] = 0
    else:
        mapping[1]["output_frame"] = 0
    (root / "FRAME_MAP.json").write_text(json.dumps(mapping))
    packet["inputs"]["frozen_frame_map"] = artifact_ref(root / "FRAME_MAP.json")
    with pytest.raises(ValueError, match="SOURCE_FRAME_MAP_MISMATCH"):
        v1_mco_worker._require_pair_source_inputs(packet, raw, session)


@pytest.mark.parametrize("paired_clean,history,sid,native_frames", [
    (True, False, "050", 128), (True, True, "050", 128), (False, False, "050", 128),
    (True, False, "001", 238)])
def test_full_native_worker_one_load_fresh_calls_and_sealed_artifact_roles(
        monkeypatch, tmp_path, paired_clean, history, sid, native_frames):
    from chaoyang.inpainting_candidate import model_stages, source_frames
    from chaoyang.storage_relocation import pick_place_original199_v1

    monkeypatch.setattr(v1_mco_worker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(v1_mco_worker, "_authority", lambda packet: lambda: {"execution_allowed": True})
    monkeypatch.setattr(model_stages, "configure_project_caches", lambda root: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    backend_observer = v1_clean_pair.runtime_backend
    monkeypatch.setattr(v1_clean_pair, "runtime_backend", lambda torch, device: backend_observer(torch, "cpu"))
    monkeypatch.setattr(v1_mco_worker, "_label", lambda pixels, text: pixels)
    raw, removal, protected = fixture_arrays(native_frames)
    effective, _ = effective_clean_masks(removal, protected)
    masks = {name: np.zeros_like(removal) for name in
             ("equipment", "unknown", "cup_observed", "semantic_conflict")}
    masks.update(removal=removal, core=removal, sam_core=removal,
                 protected_visible=protected, effective_removal=effective)
    session = SimpleNamespace(qpc_ns=np.arange(native_frames, dtype=np.int64) + 1000)
    mapping = [{"native_frame_index": i, "output_frame": i,
                "qpc_declared_ns": int(session.qpc_ns[i])} for i in range(native_frames)]
    np.savez_compressed(tmp_path / "RAW_RGB.npz", rgb=raw)
    np.savez_compressed(tmp_path / "MASK_ARRAYS.npz", **masks)
    (tmp_path / "FRAME_MAP.json").write_text(json.dumps(mapping))
    mask_ref = artifact_ref(tmp_path / "MASK_ARRAYS.npz")
    mask_result = {"schema": v1_mco_worker.RESULT_SCHEMA, "status": "SEALED_COMPLETE", "mode": "Mask",
        "candidate_adopted": False, "control_ground_truth": False, "artifacts": [mask_ref],
        "sessions": [{"session_key": sid, "native_frames": native_frames,
                      "completed_frame_indices": list(range(native_frames)), "execution_complete": True}]}
    (tmp_path / "MASK_RESULT.json").write_text(json.dumps(mask_result))
    original_row = {"session_id": sid, "stage_files": {"gpu": {"raw_rgb.npz": artifact_ref(tmp_path / "RAW_RGB.npz")}}}
    fake_reader = SimpleNamespace(binding={"source_manifest": {}},
        read_json=lambda ref: {"sessions": [original_row]})
    monkeypatch.setattr(pick_place_original199_v1, "Original199Reader", lambda ref: fake_reader)
    monkeypatch.setattr(v1_mco_worker, "_original", lambda reader, row:
        (raw, np.zeros_like(raw), dict(masks), {}, session, mapping))
    parent = synthetic_parent()
    loads = []
    def load_models(repo, output, **kwargs):
        loads.append(kwargs)
        assert kwargs["need_sam"] is False and kwargs["need_clean"] is True
        return {"propainter": parent, "load_receipt": {"synthetic": True}}
    monkeypatch.setattr(model_stages, "load_frozen_models", load_models)
    videos = []
    def video(path, pixels, frames):
        path.write_bytes(b"SYNTHETIC_VIDEO_METADATA_TEST_ONLY")
        videos.append((path.name, pixels.shape, len(frames)))
        return {"video": artifact_ref(path)}
    monkeypatch.setattr(source_frames, "write_review_video", video)
    output = tmp_path / "output"
    output.mkdir()
    packet = {"task_id": "inpainting_v1_mco_paired050_cpu_fixture", "run_signature": "0" * 64,
        "mode": "Clean", "output_dir": str(output), "review_dir": str(tmp_path / "review"),
        "config": {"paired_clean": paired_clean}, "runtime_code": [], "model_weights": [],
        "inputs": {"storage_binding": artifact_ref(tmp_path / "MASK_RESULT.json"),
                   "frozen_source_raw": artifact_ref(tmp_path / "RAW_RGB.npz"),
                   "frozen_frame_map": artifact_ref(tmp_path / "FRAME_MAP.json")},
        "sessions": [{"session_key": sid, "native_frames": native_frames, "review_anchors": [0, 64, 127],
                      "original_row": original_row, "mask_input": mask_ref,
                      "mask_result": artifact_ref(tmp_path / "MASK_RESULT.json")}],
        "budgets": {"model_calls_max": 2 if paired_clean else 1, "output_bytes_estimate": 512 << 20}}
    if history:
        history_pixels = raw.copy()
        history_pixels[effective] = 19
        np.savez_compressed(tmp_path / "HISTORY_CLEAN.npz", rgb=history_pixels)
        history_ref = artifact_ref(tmp_path / "HISTORY_CLEAN.npz")
        history_result = {**mask_result, "mode": "Clean", "artifacts": [history_ref]}
        (tmp_path / "HISTORY_RESULT.json").write_text(json.dumps(history_result))
        packet["sessions"][0].update(evaluation_clean_input=history_ref,
            evaluation_clean_result=artifact_ref(tmp_path / "HISTORY_RESULT.json"))
    assert v1_mco_worker.execute_v1_mco_stage(packet) == 0
    assert len(loads) == 1
    result = json.loads((output / "MODEL_RESULT.json").read_text())
    assert result["schema"] == v1_mco_worker.RESULT_SCHEMA
    assert result["model_calls"] == (2 if paired_clean else 1) and result["paired_clean"] is paired_clean
    assert result["sessions"][0]["completed_frame_indices"] == list(range(native_frames))
    pair_root = output / "sessions" / sid / "Clean"
    if not paired_clean:
        assert not (pair_root / "CLEAN_PAIR_RECEIPT.json").exists()
        assert (f"{sid}_Clean_comparison.mp4", (native_frames, 24, 64, 3), native_frames) in videos
        assert parent.model.pixel_calls == 0
        return
    receipt = json.loads((pair_root / "CLEAN_PAIR_RECEIPT.json").read_text())
    assert receipt["fresh_A"] is True and receipt["historical_evaluation_column_present"] is history
    assert receipt["output_arrays"]["B"]["path"] == str(pair_root / "CLEAN_RGB.npz")
    assert receipt["frozen_source_raw"] == packet["inputs"]["frozen_source_raw"]
    assert receipt["mask_input"] == mask_ref
    assert (f"{sid}_Clean_comparison.mp4", (native_frames, 56, 96 if history else 64, 3), native_frames) in videos
    if history:
        with np.load(pair_root / "CLEAN_A_RGB.npz") as archive:
            assert not np.array_equal(archive["rgb"], history_pixels)
    for path in ("CLEAN_A_RGB.npz", "CLEAN_RGB.npz", "FROZEN_INPUT_ARRAYS.npz", "RANDOM_STATE_ARRAYS.npz"):
        assert str(pair_root / path) in {ref["path"] for ref in result["artifacts"]}
