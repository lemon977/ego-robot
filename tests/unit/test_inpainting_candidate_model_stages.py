"""CPU fake-provider bridges, not visual quality or actual weight execution."""
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.inpainting_candidate.contracts import file_ref, verify_sealed_stage
from chaoyang.inpainting_candidate import model_stages as stages


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-card")
    source = tmp_path / "raw" / "vst.h264"
    source.parent.mkdir()
    source.write_bytes(b"synthetic original video bytes, not a real decode")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"fake weights not actually loaded")
    raw = np.zeros((4, 48, 64, 3), np.uint8)
    mapping = [{"native_frame_index": i, "decoded_display_index": i,
                "capture_frame_id": "UNKNOWN", "qpc": 100 + i * 3,
                "timebase_status": "TIMEBASE_UNVERIFIED"} for i in range(4)]
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True,
             "task_id": "CPU_FAKE_PROVIDER_TEST_NOT_MODEL_TASK",
             "run_signature": "a" * 64, "fencing_token": "test-fence",
             "gpu_uuid": "GPU-test-card", "worker_logical_index": 0}
    return SimpleNamespace(root=tmp_path, raw=raw, mapping=mapping,
                           source=file_ref(source), weight=file_ref(weight),
                           owner=owner, authority=lambda: dict(owner), calls=[])


class FakePointModel:
    def __init__(self, mask):
        self.mask = mask
        self.submitted = []

    def init_state(self, **kwargs):
        self.count = len(kwargs["resource_path"])
        return {"ids": [], "cached_frame_outputs": {}}

    def _cache_frame_outputs(self, state, frame, outputs):
        state["cached_frame_outputs"][frame] = outputs

    def add_prompt(self, **kwargs):
        state, index, obj = kwargs["inference_state"], kwargs["frame_idx"], kwargs["obj_id"]
        self.submitted.append(kwargs)
        if obj not in state["ids"]:
            state["ids"].append(obj)
        return index, self._out(state)

    def _out(self, state):
        return {"out_binary_masks": np.stack([self.mask] * len(state["ids"])),
                "out_obj_ids": np.asarray(state["ids"], np.int64)}

    def propagate_in_video(self, **kwargs):
        indices = range(self.count) if not kwargs["reverse"] else range(kwargs["start_frame_idx"], -1, -1)
        for i in indices:
            yield i, self._out(kwargs["inference_state"])


class FakeSam:
    def __init__(self, env):
        self.calls = []
        self.mask = np.zeros(env.raw.shape[1:3], bool)
        self.mask[10:30, 20:30] = True
        self.model = FakePointModel(self.mask)
        self.aborted = []

    def track_text(self, frames, prompt, anchor):
        self.calls.append((prompt, anchor, len(frames)))
        if prompt == "a person's hand and forearm":
            return {1: {i: np.packbits(self.mask) for i in range(len(frames))}}
        return {}


@pytest.fixture
def fake_tensors(monkeypatch):
    class Tensor:
        def __init__(self, values, dtype):
            self.values = np.asarray(values, dtype=dtype)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=lambda values, dtype: Tensor(values, dtype), float32=np.float32, int64=np.int64))


def mask_run(env, **kwargs):
    return stages.run_automatic_mask_stage(env.raw, sam=FakeSam(env),
        output_dir=env.root / "mask", raw_source_ref=env.source, frame_map=env.mapping,
        model_ref=env.weight, component_signature="b" * 64,
        authorization=env.authority, **kwargs)


def test_module_import_has_no_torch_or_model_constructor_path():
    source = inspect.getsource(stages)
    assert "import torch" not in "\n".join(line for line in source.splitlines() if not line.startswith(" "))
    assert "manual_seed" not in inspect.signature(stages.run_automatic_mask_stage).parameters
    assert "session_id" not in inspect.signature(stages.run_automatic_mask_stage).parameters


@pytest.mark.parametrize("changed", [
    {"execution_allowed": False}, {"algorithm_execution_allowed": False},
    {"worker_logical_index": 1}, {"run_signature": "bad"}, {"fencing_token": ""},
    {"gpu_uuid": "0"}, {"task_id": ""},
])
def test_cpu_only_or_unfenced_call_refused_before_model(env, changed):
    env.owner.update(changed)
    with pytest.raises(stages.ModelStageError, match="AUTHORITY_REQUIRED"):
        mask_run(env)
    assert not (env.root / "mask").exists()


def test_wrong_physical_card_refused(env, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(stages.ModelStageError, match="PHYSICAL_UUID"):
        mask_run(env)


def test_actual_text_and_point_bridge_seals_complete_native_masks(env, fake_tensors):
    sam = FakeSam(env)
    receipt = stages.run_automatic_mask_stage(env.raw, sam=sam,
        output_dir=env.root / "mask", raw_source_ref=env.source, frame_map=env.mapping,
        model_ref=env.weight, component_signature="b" * 64, authorization=env.authority)
    assert len(sam.calls) == 6
    assert len(sam.model.submitted) == 2
    assert all(call["text_str"] is None and call["point_labels"].values.tolist() == [1]
               for call in sam.model.submitted)
    assert receipt["native_input_frames"] == receipt["native_output_frames"] == 4
    assert receipt["producer_manual_seed_reads"] == receipt["manual_points_submitted"] == 0
    assert receipt["visual_quality"] == "NOT_EVALUATED"
    assert receipt["training_eligible"] is False
    refs = verify_sealed_stage(receipt, "b" * 64, output_root=env.root / "mask")
    assert len(refs) == 12
    with np.load(env.root / "mask" / "MASK_ARRAYS.npz", allow_pickle=False) as arrays:
        assert arrays["removal"].shape == env.raw.shape[:3]
        assert "equipment__controller" in arrays
        assert not np.any(arrays["effective_removal"] & arrays["protected_visible"])
    point = json.loads((env.root / "mask" / "SAM_POINT.json").read_text())
    meta = json.loads((env.root / "mask" / "MASK_METADATA.json").read_text())
    assert meta["upstream_point_refinement_receipts"][0]["payload_sha256"] == point["receipt_payload_sha256"]
    assert json.loads((env.root / "mask" / "FRAME_MAP.json").read_text()) == env.mapping
    for step in ("TEXT", "POINT"):
        archive = json.loads((env.root / "mask" / f"SAM_{step}_PROPOSALS.json").read_text())
        assert archive["feeds_producer"] is False and archive["algorithm_gain"] is False
        row, = archive["proposals"]
        assert row["role"] == "human" and row["track_id"] == 1
        with np.load(archive["arrays"]["path"], allow_pickle=False) as pixels:
            unpacked = np.unpackbits(pixels[row["array_key"]], axis=1,
                count=48 * 64).reshape(4, 48, 64).astype(bool)
            assert np.array_equal(unpacked, np.broadcast_to(sam.mask, (4, 48, 64)))


def test_duplicate_stage_refused_no_overwrite(env, fake_tensors):
    mask_run(env)
    before = (env.root / "mask" / "RECEIPT.json").read_bytes()
    with pytest.raises(FileExistsError):
        mask_run(env)
    assert (env.root / "mask" / "RECEIPT.json").read_bytes() == before


@pytest.mark.parametrize("prompt", [None, "VR controller"])
def test_controller_prompt_stage_records_only_explicit_candidate_actual_parameter(env, fake_tensors, prompt):
    from chaoyang.inpainting_candidate.automatic_masks import PROMPTS
    defaults = dict(PROMPTS)
    sam = FakeSam(env)
    receipt = stages.run_automatic_mask_stage(env.raw, sam=sam,
        output_dir=env.root / "mask", raw_source_ref=env.source, frame_map=env.mapping,
        model_ref=env.weight, component_signature="b" * 64, authorization=env.authority,
        controller_text_prompt=prompt)
    expected = dict(defaults)
    if prompt is not None:
        expected["controller"] = prompt
        assert receipt["actual_inputs"]["controller_text_prompt"] == prompt
    else:
        assert "controller_text_prompt" not in receipt["actual_inputs"]
    assert sam.calls == [(value, 0, len(env.raw)) for value in expected.values()]
    text = json.loads((env.root / "mask/SAM_TEXT.json").read_text())
    assert [(row["role"], row["prompt"]) for row in text["calls"]] == list(expected.items())
    assert PROMPTS == defaults
    assert len(sam.model.submitted) == 2
    assert all(call["text_str"] is None for call in sam.model.submitted)
    assert receipt["manual_points_submitted"] == receipt["producer_manual_seed_reads"] == 0
    verify_sealed_stage(receipt, "b" * 64, output_root=env.root / "mask")


@pytest.mark.parametrize("bad", ["", "custom", 0, True, [], {}])
def test_controller_prompt_invalid_before_stage_creation_or_model(env, bad):
    from chaoyang.inpainting_candidate.automatic_masks import AutomaticMaskContractError
    sam = FakeSam(env)
    with pytest.raises(AutomaticMaskContractError, match="CONTROLLER_TEXT_PROMPT"):
        stages.run_automatic_mask_stage(env.raw, sam=sam, output_dir=env.root / "mask",
            raw_source_ref=env.source, frame_map=env.mapping, model_ref=env.weight,
            component_signature="b" * 64, authorization=env.authority,
            controller_text_prompt=bad)
    assert sam.calls == sam.model.submitted == []
    assert not (env.root / "mask").exists()


def test_default_and_controller_candidate_preserve_all_masks_and_point_rules_when_provider_same(env, fake_tensors):
    receipts = []
    sam_calls = []
    for name, prompt in [("default", None), ("candidate", "VR controller")]:
        sam = FakeSam(env)
        receipts.append(stages.run_automatic_mask_stage(env.raw, sam=sam,
            output_dir=env.root / name, raw_source_ref=env.source, frame_map=env.mapping,
            model_ref=env.weight, component_signature="b" * 64,
            authorization=env.authority, controller_text_prompt=prompt))
        sam_calls.append(sam.model.submitted)
    with np.load(env.root / "default/MASK_ARRAYS.npz") as old, np.load(env.root / "candidate/MASK_ARRAYS.npz") as new:
        assert old.files == new.files
        for name in old.files:
            assert np.array_equal(old[name], new[name]), name
    for name in ["SAM_POINT.json", "AUTOMATIC_POINTS.json", "MASK_METADATA.json"]:
        assert json.loads((env.root / "default" / name).read_text()) == json.loads((env.root / "candidate" / name).read_text())
    assert [len(calls) for calls in sam_calls] == [2, 2]
    assert receipts[0]["text_prompt_invocations"] == receipts[1]["text_prompt_invocations"] == 6


def test_diagnostic_empty_proposals_are_explicit_not_hallucinated(env):
    refs = stages._seal_rgb_proposals(env.root, "TEXT", [], env.raw.shape[:3])
    assert len(refs) == 2
    metadata = json.loads(Path(refs[1]["path"]).read_text())
    assert metadata["proposals"] == [] and metadata["feeds_producer"] is False
    with np.load(refs[0]["path"], allow_pickle=False) as arrays:
        assert arrays.files == []


@pytest.mark.parametrize("bad", ["float_mask", "short_mask", "duplicate", "short_flags"])
def test_diagnostic_proposal_domain_rejects_invalid_arrays(env, bad):
    from chaoyang.inpainting_candidate.automatic_masks import RGBProposal
    masks = np.zeros(env.raw.shape[:3], bool)
    if bad == "float_mask":
        masks = masks.astype(np.float32)
    elif bad == "short_mask":
        masks = masks[:-1]
    flags = np.zeros(3, bool) if bad == "short_flags" else None
    item = RGBProposal("glove", 1, masks, "a" * 64, "b" * 64,
                       "c" * 64, "d" * 64, unsupported_frames=flags)
    proposals = [item, item] if bad == "duplicate" else [item]
    with pytest.raises(stages.ModelStageError, match="PROPOSAL_DIAGNOSTIC"):
        stages._seal_rgb_proposals(env.root, "TEXT", proposals, env.raw.shape[:3])
    assert not (env.root / "SAM_TEXT_PROPOSALS.npz").exists()


@pytest.mark.parametrize("bad", ["short", "duplicate", "reordered", "bool"])
def test_no_frame_reordering_or_implicit_missing_frame(env, bad):
    if bad == "short":
        env.mapping.pop()
    elif bad == "duplicate":
        env.mapping[2]["native_frame_index"] = 1
    elif bad == "reordered":
        env.mapping[1]["decoded_display_index"] = 2
    else:
        env.mapping[0]["native_frame_index"] = False
    with pytest.raises(stages.ModelStageError, match="FRAME_MAP_COUNT|ONE_TO_ONE"):
        mask_run(env)
    assert not (env.root / "mask").exists()


def test_historical_video_producer_rejected(env):
    old = env.root / "processed" / "old.mp4"
    old.parent.mkdir()
    old.write_bytes(b"old")
    env.source = file_ref(old)
    with pytest.raises(stages.ModelStageError, match="HISTORICAL_DERIVED"):
        mask_run(env)


def test_model_error_is_partial_not_complete(env):
    class Broken:
        def track_text(self, *args):
            raise RuntimeError("fixture model error")
    with pytest.raises(RuntimeError, match="fixture model error"):
        stages.run_automatic_mask_stage(env.raw, sam=Broken(), output_dir=env.root / "bad",
            raw_source_ref=env.source, frame_map=env.mapping, model_ref=env.weight,
            component_signature="b" * 64, authorization=env.authority)
    assert not (env.root / "bad" / "RECEIPT.json").exists()
    assert json.loads((env.root / "bad" / "FAILED.json").read_text())["status"] == "FAILED_PARTIAL_NOT_CONSUMABLE"


def test_project_cache_paths_and_temp_are_not_external(env, monkeypatch):
    monkeypatch.setattr(stages.tempfile, "tempdir", stages.tempfile.tempdir)
    for name in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
                 "TRITON_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
                 "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    caches = stages.configure_project_caches(env.root / "run")
    for key in ("TMPDIR", "TORCH_HOME", "TRITON_CACHE_DIR", "HF_HOME", "CUDA_CACHE_PATH"):
        assert Path(caches[key]).is_relative_to(env.root)
        assert os.environ[key] == caches[key]
    with pytest.raises(stages.ModelStageError, match="OUTSIDE_PROJECT"):
        stages.configure_project_caches(env.root.parent / "outside")


def test_frozen_model_load_rejects_incomplete_closure_before_import(env):
    with pytest.raises(stages.ModelStageError, match="SAM_CODE_CLOSURE"):
        stages.load_frozen_models(env.root, env.root / "run", authorization=env.authority,
                                  frozen_inputs=[env.weight], need_sam=True)


def test_model_loader_cpu_gate_precedes_cache_or_model_import(env):
    env.owner["algorithm_execution_allowed"] = False
    with pytest.raises(stages.ModelStageError, match="AUTHORITY_REQUIRED"):
        stages.load_frozen_models(env.root, env.root / "run", authorization=env.authority,
                                  frozen_inputs=[env.weight])
    assert not (env.root / "run").exists()


def test_noop_reference_hook_preserves_original_bytecode_and_frozen_neighbors():
    from chaoyang.pick_place_1008.visual import ProtectedProPainter
    parent = SimpleNamespace(settings=dict(SETTINGS))
    adapter = stages.ReferenceSelectingProtectedProPainter(parent,
        lambda **row: row["original_references"])
    assert adapter._hooked.__func__.__code__ is ProtectedProPainter.inpaint.__code__
    neighbors = list(range(15, 26))
    expected = _get_ref_index(20, neighbors, 40, 10, -1)
    assert adapter._select(20, neighbors, 40, 10, -1) == expected
    assert adapter.calls[-1]["neighbors"] == neighbors
    assert adapter.calls[-1]["original_references"] == adapter.calls[-1]["references"]
    assert ProtectedProPainter.inpaint.__globals__["_get_ref_index"] is _get_ref_index


@pytest.mark.parametrize("selected,code", [
    ([0], "COUNT_OR_NATIVE"), ([0, 0, 30], "COUNT_OR_NATIVE"),
    ([0, 20, 30], "COUNT_OR_NATIVE"), ([30, 10, 0], "COUNT_OR_NATIVE"),
])
def test_reference_hook_cannot_change_count_or_neighbors(selected, code):
    parent = SimpleNamespace(settings=dict(SETTINGS))
    adapter = stages.ReferenceSelectingProtectedProPainter(parent, lambda **_: selected)
    with pytest.raises(stages.ModelStageError, match=code):
        adapter._select(20, list(range(15, 26)), 40, 10, -1)


def test_reference_hook_cannot_enlarge_frozen_context():
    parent = SimpleNamespace(settings=dict(SETTINGS))
    adapter = stages.ReferenceSelectingProtectedProPainter(parent,
        lambda **_: [0, 10, 20, 30, 40, 60, 70])
    assert len(_get_ref_index(50, list(range(45, 56)), 120, 10, 8)) == 7
    with pytest.raises(stages.ModelStageError, match="CONTEXT_CHANGED"):
        adapter._select(50, list(range(45, 56)), 120, 10, 8)


def test_clean_wrapper_frozen_settings_drift_rejected():
    changed = dict(SETTINGS, raft_iter=21)
    with pytest.raises(ValueError, match="SETTINGS_DRIFT"):
        stages.ReferenceSelectingProtectedProPainter(SimpleNamespace(settings=changed))


class FakeCleanWrapper:
    def __init__(self, parent, selector, *, record_reference_selection=False):
        self.wrapped = SimpleNamespace(settings=dict(SETTINGS), effective_masks=None)
        self.selector, self.calls = selector, []
        self.original_function_sha256 = "f" * 64

    def inpaint(self, raw, removal, protected):
        from chaoyang.pick_place_1008.visual import effective_clean_masks
        effective, _ = effective_clean_masks(removal, protected)
        self.wrapped.effective_masks = effective
        clean = raw.copy()
        clean[effective] = 111
        return clean


def clean_run(env, monkeypatch, **kwargs):
    monkeypatch.setattr(stages, "ReferenceSelectingProtectedProPainter", FakeCleanWrapper)
    with np.load(env.root / "mask" / "MASK_ARRAYS.npz", allow_pickle=False) as packed:
        masks = {name: packed[name] for name in packed.files}
    masks = kwargs.pop("masks", masks)
    return stages.run_clean_stage(env.raw, masks=masks,
        mask_receipt_ref=file_ref(env.root / "mask" / "RECEIPT.json"),
        parent_propainter=SimpleNamespace(), output_dir=env.root / "clean",
        raw_source_ref=env.source, frame_map=env.mapping, model_refs=[env.weight],
        component_signature="c" * 64, authorization=env.authority, **kwargs)


def test_clean_seals_full_native_pixels_preserves_outside_and_protected(env, fake_tensors, monkeypatch):
    mask_run(env)
    receipt = clean_run(env, monkeypatch)
    assert receipt["status"] == "SEALED_COMPLETE"
    assert receipt["native_output_frames"] == 4
    assert receipt["validation"]["protected_changed_pixels"] == 0
    assert receipt["validation"]["outside_delete_changed_pixels"] == 0
    assert receipt["reference_mode"] == "FROZEN_BASELINE"
    assert receipt["causal_training_input"] is False
    assert receipt["visual_quality"] == "NOT_EVALUATED"
    verify_sealed_stage(receipt, "c" * 64, output_root=env.root / "clean")


def test_clean_capture_context_is_signed_actual_input_not_background_proof(env, fake_tensors, monkeypatch):
    mask_run(env)
    capture = env.root / "capture_config.json"
    capture.write_text(json.dumps({"stage": "RawRaftCaptureConfig", "background_registration": "ABSENT"}))
    reference = file_ref(capture)
    receipt = clean_run(env, monkeypatch, clean_context_refs={"capture_config": reference},
                        record_reference_selection=True)
    assert receipt["actual_inputs"]["clean_context_refs"] == {"capture_config": reference}
    assert receipt["reference_mode"] == "FROZEN_BASELINE"
    assert receipt["reference_selection_diagnostic_hook"] is True
    assert receipt["visual_quality"] == "NOT_EVALUATED"


def test_clean_reference_selector_without_signed_context_rejected_before_call(env, fake_tensors, monkeypatch):
    mask_run(env)
    with pytest.raises(stages.ModelStageError, match="SIGNED_FLOW_AND_REFERENCE_CONTEXT_REQUIRED"):
        clean_run(env, monkeypatch, reference_selector=lambda **row: row["original_references"])
    assert not (env.root / "clean").exists()


def test_clean_changed_capture_config_ref_is_rejected_before_call(env, fake_tensors, monkeypatch):
    mask_run(env)
    capture = env.root / "capture_config.json"
    capture.write_bytes(b"frozen capture spec")
    reference = file_ref(capture)
    capture.write_bytes(b"drifted capture spec")
    with pytest.raises(ValueError, match="SHA|sha|REF"):
        clean_run(env, monkeypatch, clean_context_refs={"capture_config": reference})
    assert not (env.root / "clean").exists()


def test_clean_opaque_context_cannot_be_used_as_unregistered_input(env, fake_tensors, monkeypatch):
    mask_run(env)
    with pytest.raises(stages.ModelStageError, match="CLEAN_CONTEXT_ROLE_UNSUPPORTED"):
        clean_run(env, monkeypatch, clean_context_refs={"old_manual_seed": env.weight})
    assert not (env.root / "clean").exists()


def test_clean_rejects_changed_in_memory_masks_even_if_old_receipt_sha_valid(env, fake_tensors, monkeypatch):
    mask_run(env)
    with np.load(env.root / "mask" / "MASK_ARRAYS.npz", allow_pickle=False) as packed:
        masks = {name: packed[name] for name in packed.files}
    masks["removal"] = ~masks["removal"]
    with pytest.raises(stages.ModelStageError, match="DIFFER_FROM_SEALED"):
        clean_run(env, monkeypatch, masks=masks)
    assert not (env.root / "clean").exists()


def test_clean_rejects_partial_upstream_even_with_valid_files(env, fake_tensors, monkeypatch):
    mask_run(env)
    receipt_path = env.root / "mask" / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "PARTIAL"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="PARTIAL_NOT_CONSUMABLE"):
        clean_run(env, monkeypatch)


def test_clean_protection_failure_does_not_seal(env, fake_tensors, monkeypatch):
    mask_run(env)
    class BrokenClean(FakeCleanWrapper):
        def inpaint(self, raw, removal, protected):
            clean = super().inpaint(raw, removal, protected)
            clean[:, 0, 0] = 250
            return clean
    monkeypatch.setattr(stages, "ReferenceSelectingProtectedProPainter", BrokenClean)
    with np.load(env.root / "mask" / "MASK_ARRAYS.npz", allow_pickle=False) as packed:
        masks = {name: packed[name] for name in packed.files}
    with pytest.raises(stages.ModelStageError, match="OUTSIDE_EFFECTIVE"):
        stages.run_clean_stage(env.raw, masks=masks,
            mask_receipt_ref=file_ref(env.root / "mask" / "RECEIPT.json"),
            parent_propainter=SimpleNamespace(), output_dir=env.root / "clean",
            raw_source_ref=env.source, frame_map=env.mapping, model_refs=[env.weight],
            component_signature="c" * 64, authorization=env.authority)
    assert not (env.root / "clean" / "RECEIPT.json").exists()
    assert (env.root / "clean" / "FAILED.json").exists()


def test_authority_identity_change_mid_stage_leaves_unsealed_partial(env, fake_tensors):
    calls = []
    def changing_authority():
        row = dict(env.owner)
        calls.append(row)
        if len(calls) >= 3:
            row["fencing_token"] = "different-writer"
        return row
    with pytest.raises(stages.ModelStageError, match="IDENTITY_DRIFT"):
        stages.run_automatic_mask_stage(env.raw, sam=FakeSam(env),
            output_dir=env.root / "mask", raw_source_ref=env.source, frame_map=env.mapping,
            model_ref=env.weight, component_signature="b" * 64, authorization=changing_authority)
    assert not (env.root / "mask" / "RECEIPT.json").exists()
    assert (env.root / "mask" / "FAILED.json").exists()


@pytest.mark.parametrize("diagnostic_only", [False, True])
def test_noop_reference_hook_matches_complete_original_cpu_inference_call_graph(diagnostic_only):
    """Real frozen PP algorithm with CPU fake NN providers; no CUDA/weights."""
    import torch
    from chaoyang.pick_place_1008.visual import ProtectedProPainter

    class Flow:
        def forward_bidirect_flow(self, flows, masks):
            return flows, None

        def combine_flow(self, original, predicted, masks):
            return predicted

    class Generator:
        def __init__(self):
            self.calls = []

        def img_propagation(self, masked, flows, masks, mode):
            assert mode == "nearest"
            return masked.flatten(0, 1), torch.zeros_like(masks).flatten(0, 1)

        def __call__(self, frames, flows, masks, updated, neighbors):
            self.calls.append({"frames": frames.clone(), "masks": masks.clone(),
                               "flow_f": flows[0].clone(), "flow_b": flows[1].clone(),
                               "neighbors": neighbors})
            return frames[:, :neighbors].clone()

    def parent():
        generator = Generator()
        return SimpleNamespace(settings=dict(SETTINGS), torch=torch,
            device=torch.device("cpu"), flow=Flow(), model=generator,
            raft=lambda frames, iters: (
                torch.zeros((1, frames.shape[1] - 1, 2, 48, 64)),
                torch.zeros((1, frames.shape[1] - 1, 2, 48, 64))))

    frames = np.zeros((24, 48, 64, 3), np.uint8)
    frames[:] = np.arange(24, dtype=np.uint8)[:, None, None, None] * 5
    removal = np.zeros(frames.shape[:3], bool)
    removal[:, 10:30, 20:35] = True
    protected = np.zeros_like(removal)
    protected[:, 15:20, 32:38] = True
    p0, p1 = parent(), parent()
    baseline = ProtectedProPainter(p0).inpaint(frames, removal, protected)
    noop = stages.ReferenceSelectingProtectedProPainter(p1,
        None if diagnostic_only else lambda **r: r["original_references"],
        record_reference_selection=diagnostic_only)
    candidate = noop.inpaint(frames, removal, protected)
    assert np.array_equal(baseline, candidate)
    assert noop.wrapped.settings == SETTINGS
    assert len(p0.model.calls) == len(p1.model.calls) == len(noop.calls)
    for a, b in zip(p0.model.calls, p1.model.calls):
        assert a["neighbors"] == b["neighbors"]
        assert all(torch.equal(a[key], b[key]) for key in ("frames", "masks", "flow_f", "flow_b"))
    assert all(row["references"] == row["original_references"] for row in noop.calls)


