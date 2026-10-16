"""Native B wiring only: real immutable CPU fake receipts, NN/GPU calls zero."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.inpainting_candidate import clean_worker as worker
from chaoyang.inpainting_candidate import clean_flow as flow
from chaoyang.inpainting_candidate import model_stages as stages
from chaoyang.inpainting_candidate.automatic_masks import frames_sha256
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref

spec = importlib.util.spec_from_file_location("_native_reference_small_fixture",
    Path(__file__).parent / "unit/test_inpainting_candidate_native_clean_composite.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    f = fixtures.native.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    baseline_pp = worker.run_clean_stage
    f.calls["raft"] = 0
    f.calls["fb"] = 0

    def raft(tensor, *, iters):
        f.calls["raft"] += 1
        assert iters == 20
        shape = (1, tensor.shape[1]-1, 2, *tensor.shape[-2:])
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def diagnostics(a, b, parent):
        f.calls["fb"] += 1
        shape = (a.shape[1], *a.shape[-2:])
        return {"fb_residual_squared_forward": np.zeros(shape, np.float32),
            "fb_residual_squared_backward": np.zeros(shape, np.float32),
            "vendor_fb_consistent_forward": np.ones(shape, bool),
            "vendor_fb_consistent_backward": np.ones(shape, bool)}

    def load(*args, **kwargs):
        f.calls["load"] += 1
        return {"propainter": SimpleNamespace(raft=raft), "load_receipt": {"cpu_fake": True}}

    def pp(rgb, *, masks, output_dir, frame_map, raw_source_ref, mask_receipt_ref,
            model_refs, component_signature, authorization, **options):
        if options["reference_selector"] is None:
            return baseline_pp(rgb, masks=masks, output_dir=output_dir, frame_map=frame_map,
                raw_source_ref=raw_source_ref, mask_receipt_ref=mask_receipt_ref, model_refs=model_refs,
                component_signature=component_signature, authorization=authorization, **options)
        f.calls["pp"].append(len(rgb))
        parent = options["parent_propainter"]
        for chunk in flow.chunk_plan(len(rgb)):
            pixels = rgb[chunk["first_frame"]:chunk["end_frame_exclusive"]]
            tensor = pixels.astype(np.float32).transpose(0, 3, 1, 2)[None] / np.float32(127.5) - np.float32(1)
            parent.raft(tensor, iters=20)
        context = dict(options["clean_context_refs"])
        selected = stages.OnlineReferenceContext(options["reference_selector"],
            options["online_reference_context_provider"], context, authorization, authorization())
        calls = []
        for mid in (0, 10):
            neighbors = list(range(max(0, mid-5), min(len(rgb), mid+6)))
            original = _get_ref_index(mid, neighbors, len(rgb), 10, -1)
            refs = selected(mid=mid, neighbors=neighbors, length=len(rgb), stride=10,
                ref_num=-1, original_references=original)
            calls.append({"target_frame": mid, "neighbors": neighbors, "original_references": original,
                "references": refs, "reference_count": len(refs), "settings_unchanged": True})
        folder = Path(output_dir)
        fixed = rgb.copy(); fixed[masks["effective_removal"]] = 91
        mask = flow.sealed_receipt(mask_receipt_ref, expected_stage="Mask")
        receipt = fixtures.seal(folder, "Clean", component_signature,
            [fixtures.arrays(folder / "CLEAN_ARRAYS.npz", clean_rgb=fixed,
                effective_removal=masks["effective_removal"], protected_visible=masks["protected_visible"]),
             fixtures.put(folder / "FRAME_MAP.json", frame_map),
             fixtures.put(folder / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(len(rgb))]),
             fixtures.put(folder / "REFERENCE_SELECTION.json", calls)], len(rgb),
            settings=dict(SETTINGS), reference_mode="SINGLE_REF_SELECTION_CHANGE", reference_selection_diagnostic_hook=True,
            authority=authorization(), actual_inputs={"raw_source": raw_source_ref, "mask_receipt": mask_receipt_ref,
                "models": model_refs, "clean_context_refs": context, "frames_sha256": frames_sha256(rgb),
                "frame_map_sha256": canonical_sha(frame_map), "effective_removal_source": {
                    "mask_arrays": flow.output_named(mask, "MASK_ARRAYS.npz"), "array_key": "effective_removal",
                    "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}})
        return json.loads(Path(receipt["path"]).read_text())

    monkeypatch.setattr(worker, "load_frozen_models", load)
    monkeypatch.setattr(worker, "run_clean_stage", pp)
    monkeypatch.setattr(flow, "vendor_fb_diagnostics", diagnostics)

    def packet(*, blocked=False, policy=True, support=False):
        sample = f.source("one", 21)
        if support:
            for key in ("removal", "effective_removal", "core", "human"):
                sample.masks[key][1:] = False
            mr = sample.row["mask_receipt"]
            mask = json.loads(Path(mr["path"]).read_text())
            mask["outputs"][0] = fixtures.arrays(Path(mask["outputs"][0]["path"]), **sample.masks)
            sample.row["mask_receipt"] = fixtures.put(Path(mr["path"]), mask)
        value = f.packet([sample.row, *([f.blocked("blocked", 7)] if blocked else [])])
        f.owner["task_id"] = value["task_id"]
        if policy:
            value["config"]["stage"] = "NATIVE_REFERENCE_CLEAN"
            value["clean_config"].update(reference_policy=worker.NATIVE_REFERENCE_POLICY, candidates=["B"])
            source = flow.sealed_receipt(sample.row["source_receipt"], expected_stage="Source")
            config = {"schema": flow.CAPTURE_SCHEMA, "session_id": sample.row["session_id"],
                "native_shape": list(sample.raw.shape[:3]), "source_receipt": sample.row["source_receipt"],
                "mask_receipt": sample.row["mask_receipt"], "raw_source_ref": sample.row["raw_source_ref"],
                "raw_rgb_ref": flow.output_named(source, "RAW_RGB.npz"), "frame_map_ref": sample.row["frame_map_ref"],
                "raw_frames_sha256": frames_sha256(sample.raw), "frame_map_payload_sha256": canonical_sha(sample.mapping),
                "runtime_code": value["runtime_code"], "raft_weight_ref": value["model_weights"][0],
                "raft_iter": 20, "chunk_length": 12, "preprocessing": flow.PREPROCESSING}
            sample.row["capture_config_ref"] = fixtures.put(tmp_path / "capture.json", config)
        return value

    f.make = packet
    f.decode = lambda path: json.loads(path.read_text())["fake_codec_frames"]
    return f


def test_b_real_observation_once_and_cached_two_reference_calls(wired):
    f = wired; packet = f.make(blocked=True)
    result = worker.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 1, result
    assert f.calls["load"] == 1 and f.calls["pp"] == [21]
    assert f.calls["raft"] == f.calls["fb"] == len(flow.chunk_plan(21))
    row = result["sessions"][0]
    assert "/B/Clean/" in row["clean"]["path"]
    stats = json.loads(Path(row["reference_selection_stats"]["path"]).read_text())
    assert stats["reference_calls"] == 2 and stats["changed_reference_calls"] == 0
    assert stats["rank_not_applied_calls"] == 2
    assert worker.validate_native_clean_result(packet, result, decoder=f.decode)["blocked_sessions"] == 1
    evidence = flow.sealed_receipt(row["reference_evidence"], expected_stage="ReferenceEvidence")
    assert not any("clean" in k.lower() for k in evidence["actual_inputs"])


def test_b_supported_refs_change_without_background_or_benefit_claim(wired):
    f = wired; packet = f.make(support=True)
    result = worker.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 1, result
    stats = json.loads(Path(result["sessions"][0]["reference_selection_stats"]["path"]).read_text())
    assert stats["changed_reference_calls"] == 1
    assert stats["static_background_truth"] is False and stats["visual_improvement"] == "NOT_ESTABLISHED"
    worker.validate_native_clean_result(packet, result, decoder=f.decode)


def test_old_a_dispatch_and_options_unchanged(wired):
    f = wired; packet = f.make(policy=False)
    result = worker.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 1
    assert result["sessions"][0]["execution"] == "COMPLETE_CLEAN_BASELINE"
    assert f.calls["raft"] == f.calls["fb"] == 0
    worker.validate_native_clean_result(packet, result, decoder=f.decode)


@pytest.mark.parametrize("fault", ["wrong_path", "context", "stats", "video", "count"])
def test_b_validator_rejects_actual_binding_and_accounting_drift(wired, fault):
    f = wired; packet = f.make()
    result = worker.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 1, result
    row = result["sessions"][0]
    if fault == "wrong_path": row["clean_receipt"] = row["raw_flow_receipt"]
    elif fault == "context": row["reference_evidence"] = row["raw_flow_receipt"]
    elif fault == "count": result["model_call_accounting"]["attempted_api_calls"] = 2
    elif fault == "video": row["review"]["native_frames"] = 20
    else:
        ref = row["reference_selection_stats"]
        value = json.loads(Path(ref["path"]).read_text()); value["changed_reference_calls"] = 999
        row["reference_selection_stats"] = fixtures.put(Path(ref["path"]), value)
    with pytest.raises(ValueError):
        worker.validate_native_clean_result(packet, result, decoder=f.decode)


def test_bad_capture_config_rejected_before_model_load(wired):
    f = wired; packet = f.make()
    ref = packet["sessions"][0]["capture_config_ref"]
    value = json.loads(Path(ref["path"]).read_text()); value["raft_iter"] = 19
    packet["sessions"][0]["capture_config_ref"] = fixtures.put(Path(ref["path"]), value)
    with pytest.raises(ValueError):
        worker.run_clean_batch(packet, authority=f.authority)
    assert f.calls["load"] == f.calls["raft"] == 0
