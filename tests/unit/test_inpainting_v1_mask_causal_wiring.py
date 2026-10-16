"""Synthetic CPU verification of the existing MCO mask-only Clean route."""
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
def causal_fixture(monkeypatch, tmp_path):
    from chaoyang.inpainting_candidate import model_stages, source_frames
    from chaoyang.storage_relocation import pick_place_original199_v1

    monkeypatch.setattr(v1_mco_worker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(v1_mco_worker, "_authority", lambda packet: lambda: {"execution_allowed": True})
    monkeypatch.setattr(v1_mco_worker, "_label", lambda pixels, title: pixels)
    monkeypatch.setattr(model_stages, "configure_project_caches", lambda root: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    backend = v1_clean_pair.runtime_backend
    monkeypatch.setattr(v1_clean_pair, "runtime_backend", lambda torch, device: backend(torch, "cpu"))
    raw, removal, protected = fixture_arrays(4)
    effective, _ = effective_clean_masks(removal, protected)
    base = {name: np.zeros_like(removal) for name in ("equipment", "unknown", "cup_observed", "semantic_conflict")}
    base.update(removal=removal.copy(), core=removal.copy(), sam_core=removal.copy(),
                protected_visible=protected, effective_removal=effective)
    repaired = {name: value.copy() for name, value in base.items()}
    for name in ("removal", "core", "sam_core"):
        repaired[name][:, 2:4, 2:4] = True
    repaired["effective_removal"], _ = effective_clean_masks(repaired["removal"], protected)
    session = SimpleNamespace(qpc_ns=np.arange(4, dtype=np.int64) + 100)
    mapping = [{"native_frame_index": i, "output_frame": i, "qpc_declared_ns": int(session.qpc_ns[i])}
               for i in range(4)]

    def save_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return artifact_ref(path)

    np.savez_compressed(tmp_path / "RAW_RGB.npz", rgb=raw)
    np.savez_compressed(tmp_path / "BASE.npz", **base)
    np.savez_compressed(tmp_path / "REPAIRED.npz", **repaired)
    np.savez_compressed(tmp_path / "PACKED_ORIGINAL.npz", **{
        name: np.packbits(value.reshape(4, -1), axis=1) for name, value in base.items()})
    source_ref = artifact_ref(tmp_path / "RAW_RGB.npz")
    frame_ref = save_json("FRAME_MAP.json", mapping)
    base_ref, repaired_ref = artifact_ref(tmp_path / "BASE.npz"), artifact_ref(tmp_path / "REPAIRED.npz")
    original_ref = artifact_ref(tmp_path / "PACKED_ORIGINAL.npz")
    common = {"schema": "inpainting-candidate-component-v1", "status": "SEALED_COMPLETE",
              "session_id": "synthetic_session", "native_frames": 4,
              "candidate_adopted": False, "control_ground_truth": False}
    source_receipt = save_json("SOURCE_RECEIPT.json", {**common, "stage": "Source", "outputs": [source_ref, frame_ref]})
    mask_receipt = save_json("MASK_RECEIPT.json", {**common, "stage": "Mask", "outputs": [base_ref, repaired_ref],
        "upstream": {"Source": source_receipt}})
    gpu_receipt = save_json("ORIGINAL_RECEIPT.json", {"schema": "pick_place_1008_sealed_stage_v1", "status": "COMPLETE",
        "binding": {"session_id": "synthetic_session", "stage": "gpu"}, "files": {"masks.npz": original_ref}})
    original_row = {"session_id": "synthetic_session", "stage_files": {"gpu": {
        "raw_rgb.npz": source_ref, "masks.npz": original_ref}}, "stage_receipts": {"gpu": gpu_receipt}}
    reader = SimpleNamespace(binding={"source_manifest": {}}, read_json=lambda ref: {"sessions": [original_row]})
    monkeypatch.setattr(pick_place_original199_v1, "Original199Reader", lambda ref: reader)
    original = (raw, raw.copy(), base, {}, session, mapping)
    monkeypatch.setattr(v1_mco_worker, "_original", lambda reader, row: original)
    parent, loads = synthetic_parent(), []

    def load_models(repo, output, **kwargs):
        loads.append(kwargs)
        assert kwargs["need_sam"] is False and kwargs["need_clean"] is True
        return {"propainter": parent, "load_receipt": {"synthetic": True}}

    monkeypatch.setattr(model_stages, "load_frozen_models", load_models)
    videos = []

    def video(path, pixels, frames):
        path.write_bytes(b"SYNTHETIC_VIDEO_METADATA_ONLY")
        videos.append((path.name, pixels.shape, len(frames)))
        return {"video": artifact_ref(path)}

    monkeypatch.setattr(source_frames, "write_review_video", video)
    output = tmp_path / "output"
    output.mkdir()
    packet = {"task_id": "inpainting_v1_mco_mask_causal_cpu", "run_signature": "0" * 64, "mode": "Clean",
        "output_dir": str(output), "review_dir": str(tmp_path / "review"), "runtime_code": [], "model_weights": [],
        "config": {"mask_causal_pair": True, "paired_clean": False, "pixel_propagation": "FROZEN"},
        "inputs": {"storage_binding": gpu_receipt, "frozen_base_masks": base_ref, "frozen_repaired_masks": repaired_ref,
            "frozen_mask_receipt": mask_receipt, "frozen_source_raw": source_ref, "frozen_frame_map": frame_ref,
            "frozen_source_receipt": source_receipt, "original_v1_masks": original_ref, "original_gpu_receipt": gpu_receipt},
        "sessions": [{"session_key": "synthetic_session", "native_frames": 4, "review_anchors": [0, 3],
                      "original_row": original_row}],
        "budgets": {"model_calls_max": 3, "render_calls_max": 0, "training_steps_max": 0,
                    "output_bytes_estimate": 512 << 20}}
    return SimpleNamespace(packet=packet, raw=raw, base=base, repaired=repaired, original=original,
        root=tmp_path, output=output, loads=loads, parent=parent, videos=videos, save_json=save_json)


def test_causal_route_uses_one_load_and_three_original_pp_calls(causal_fixture):
    fixture = causal_fixture
    assert v1_mco_worker.execute_v1_mco_stage(fixture.packet) == 0
    assert len(fixture.loads) == 1 and fixture.parent.model.pixel_calls == 3
    result = json.loads((fixture.output / "MODEL_RESULT.json").read_text())
    assert result["model_calls"] == 3 and result["render_calls"] == 0
    assert result["mask_causal_pair"] is True and result["paired_clean"] is False
    assert result["sessions"][0]["completed_frame_indices"] == list(range(4))
    root = fixture.output / "sessions/synthetic_session/Clean"
    receipt = json.loads((root / "MASK_CLEAN_CAUSAL_RECEIPT.json").read_text())
    assert receipt["single_change"] == "REMOVAL_MASK_ONLY_ORIGINAL_PIXEL_PROPAGATION"
    assert [call["variant"] for call in receipt["calls"]] == ["A", "B", "A_repeat"]
    assert receipt["frozen_mask_receipt"] == fixture.packet["inputs"]["frozen_mask_receipt"]
    assert receipt["original_gpu_receipt"] == fixture.packet["inputs"]["original_gpu_receipt"]
    assert receipt["output_arrays"]["B"] == artifact_ref(root / "CLEAN_RGB.npz")
    assert receipt["A_repeat_reproducibility"]["exact_equal"] is True
    assert ("synthetic_session_Clean_comparison.mp4", (4, 56, 96, 3), 4) in fixture.videos
    with np.load(root / "FROZEN_INPUT_ARRAYS.npz") as arrays:
        assert np.array_equal(arrays["base_removal"], fixture.base["removal"])
        assert np.array_equal(arrays["repaired_removal"], fixture.repaired["removal"])
    assert not (root / "PIXEL_PROPAGATION_ABLATION.json").exists()


@pytest.mark.parametrize("change", ["base_pixels", "protected_pixels", "partial_receipt", "missing_output", "source_producer", "original_producer", "rgb", "qpc"])
def test_wrong_causal_input_rejected_before_model_load(causal_fixture, change):
    fixture, packet = causal_fixture, causal_fixture.packet
    receipt_path = fixture.root / "MASK_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if change in {"base_pixels", "protected_pixels"}:
        name, key = ("BASE.npz", "frozen_base_masks") if change == "base_pixels" else ("REPAIRED.npz", "frozen_repaired_masks")
        masks = {k: v.copy() for k, v in (fixture.base if change == "base_pixels" else fixture.repaired).items()}
        masks["removal" if change == "base_pixels" else "protected_visible"][0, 0, 0] ^= True
        np.savez_compressed(fixture.root / name, **masks)
        previous = packet["inputs"][key]
        packet["inputs"][key] = artifact_ref(fixture.root / name)
        receipt["outputs"] = [packet["inputs"][key] if item == previous else item for item in receipt["outputs"]]
    elif change == "partial_receipt":
        receipt["status"] = "PARTIAL"
    elif change == "missing_output":
        receipt["outputs"].pop()
    elif change == "source_producer":
        receipt["upstream"]["Source"] = packet["inputs"]["original_gpu_receipt"]
    elif change == "original_producer":
        packet["inputs"]["original_gpu_receipt"] = packet["inputs"]["frozen_source_receipt"]
    elif change == "rgb":
        pixels = fixture.raw.copy()
        pixels[0, 0, 0, 0] ^= 1
        np.savez_compressed(fixture.root / "RAW_RGB.npz", rgb=pixels)
        packet["inputs"]["frozen_source_raw"] = artifact_ref(fixture.root / "RAW_RGB.npz")
    else:
        path = fixture.root / "FRAME_MAP.json"
        mapping = json.loads(path.read_text())
        mapping[0]["qpc_declared_ns"] += 1
        packet["inputs"]["frozen_frame_map"] = fixture.save_json("FRAME_MAP.json", mapping)
    packet["inputs"]["frozen_mask_receipt"] = fixture.save_json("MASK_RECEIPT.json", receipt)
    with pytest.raises(ValueError, match="MCO_"):
        v1_mco_worker.execute_v1_mco_stage(packet)
    assert fixture.loads == []


@pytest.mark.parametrize("field,value", [("calls", 2), ("calls", 4), ("propagation", "BYPASS"),
                                        ("paired", True), ("crop", [0])])
def test_causal_mode_rejects_changed_budget_ablation_or_crop_before_load(causal_fixture, field, value):
    fixture, packet = causal_fixture, causal_fixture.packet
    if field == "calls":
        packet["budgets"]["model_calls_max"] = value
    elif field == "propagation":
        packet["config"]["pixel_propagation"] = value
    elif field == "paired":
        packet["config"]["paired_clean"] = value
    else:
        packet["sessions"][0]["output_frames"] = value
    with pytest.raises(ValueError, match="MCO_"):
        v1_mco_worker.execute_v1_mco_stage(packet)
    assert fixture.loads == []
