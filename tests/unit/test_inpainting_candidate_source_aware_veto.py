"""Synthetic CPU origin tests, not new Mask production or visual approval."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automatic_masks as a
from chaoyang.inpainting_candidate import model_stages as stages
from chaoyang.inpainting_candidate.contracts import (
    SCHEMA, canonical_sha, file_ref, verify_sealed_stage)


SIG = {"raw_source_sha256": "a" * 64, "frame_map_sha256": "b" * 64,
       "model_sha256": "c" * 64}
POLICY = "SOURCE_AWARE_POINT_WEARER_V1"
BOUND_STATUS = "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE"


def rgb():
    return np.arange(4 * 48 * 64 * 3, dtype=np.uint8).reshape(4, 48, 64, 3)


def area(frames, bounds=(12, 12, 24, 24)):
    x0, y0, x1, y1 = bounds
    masks = np.zeros(frames.shape[:3], bool)
    masks[:, y0:y1, x0:x1] = True
    return masks


def proposal(frames, role="human", track=0, masks=None, **kwargs):
    return a.RGBProposal(role, track, area(frames) if masks is None else masks,
                         a.frames_sha256(frames), **SIG, **kwargs)


def point(frames, role="human", track=0, masks=None, **kwargs):
    options = {"source": "SAM31_RGB_POINT_AUTO",
               "refinement_receipt_payload_sha256": "d" * 64,
               "refinement_status": BOUND_STATUS}
    options.update(kwargs)
    return proposal(frames, role, track, masks, **options)


def union(frames, role="glove", track=9, point_mask=None, retained=None,
          conflict=None, **kwargs):
    empty = np.zeros(frames.shape[:3], bool)
    point_mask = empty.copy() if point_mask is None else point_mask
    retained = empty.copy() if retained is None else retained
    conflict = empty.copy() if conflict is None else conflict
    return proposal(frames, role, track, point_mask | retained,
                    source=a.TEXT_POINT_WEARER_UNION,
                    point_masks=point_mask, retained_text_masks=retained,
                    retention_conflict_masks=conflict,
                    refinement_receipt_payload_sha256="d" * 64,
                    refinement_status=BOUND_STATUS, **kwargs)


def conflict_track(frames, role="glove", track=9):
    return union(frames, role, track,
                 retained=area(frames, (36, 8, 44, 16)), conflict=area(frames))


def build(frames, proposals, policy=POLICY, **kwargs):
    return a.build_automatic_masks_with_source_aware_veto(
        frames, proposals, **SIG, source_aware_retention_veto=policy, **kwargs)


def assert_same(left, right):
    if isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    else:
        assert left == right


def test_old_build_abi_and_explicit_none_are_exact_default_noop():
    assert set(inspect.signature(a.build_automatic_masks).parameters) == {
        "frames_rgb", "proposals", "raw_source_sha256", "frame_map_sha256",
        "model_sha256", "projection_hints"}
    frames = rgb()
    proposals = [point(frames), conflict_track(frames)]
    old = a.build_automatic_masks(frames, proposals, **SIG)
    assert_same(old, build(frames, proposals, None))
    assert "accepted_point_wearer_support" not in old
    assert "retention_veto" not in old
    assert "source_aware_retention_veto" not in old["metadata"]


@pytest.mark.parametrize("role,other", [("human", "glove"), ("glove", "human")])
@pytest.mark.parametrize("origin", ["PLAIN_POINT", "UNION_POINT"])
def test_bound_selected_wearer_point_not_cancelled_by_other_role_text(role, other, origin):
    frames = rgb()
    p = (point(frames, role) if origin == "PLAIN_POINT" else
         union(frames, role, 0, point_mask=area(frames)))
    proposals = [p, conflict_track(frames, other)]
    old, new = build(frames, proposals, None), build(frames, proposals)
    target = area(frames)
    assert not old["effective_removal"][target].any()
    assert new["effective_removal"][target].all()
    assert new["removal"][target].all()
    assert new["accepted_point_wearer_support"][target].all()
    assert not new["retention_veto"][target].any()
    for key in ["human", "equipment", "wearer_support", "object_observed",
                "protected_visible", "semantic_conflict", "unknown", "held", "geometry_unknown"]:
        assert np.array_equal(old[key], new[key])
    assert new["unknown"][target].all() and new["semantic_conflict"][target].all()
    assert not new["protected_visible"][target].any()
    assert new["metadata"]["POINT_is_pixel_truth"] is False
    assert new["metadata"]["training_eligible"] is False
    assert new["metadata"]["human_review"] == "PENDING"
    assert new["metadata"]["source_aware_retention_veto"] == POLICY


def test_same_wearer_role_different_track_uses_same_source_aware_rule():
    frames = rgb()
    result = build(frames, [point(frames), conflict_track(frames, "human", 9)])
    assert result["effective_removal"][area(frames)].all()
    assert result["unknown"][area(frames)].all()


@pytest.mark.parametrize("role", ["human", "glove"])
def test_unsupported_point_native_frames_never_gain_accepted_origin(role):
    frames = rgb()
    p = point(frames, role, unsupported_frames=np.array([False, True, False, False]))
    # This separate TEXT support must not make rejected POINT pixels valid.
    anchor = proposal(frames, "human", 8)
    result = build(frames, [p, anchor, conflict_track(frames, "glove", 9)])
    assert not result["accepted_point_wearer_support"][1].any()
    assert result["retention_veto"][1, 16, 16]
    assert not result["effective_removal"][1, 16, 16]
    assert result["unknown"][1, 16, 16]
    assert result["accepted_point_wearer_support"][0, 16, 16]


@pytest.mark.parametrize("role", ["human", "glove"])
def test_missing_point_native_frame_is_not_filled_from_text_or_neighbor(role):
    frames = rgb()
    masks = area(frames)
    masks[2] = False
    result = build(frames, [point(frames, role, masks=masks),
        proposal(frames, "human", 8), conflict_track(frames, "glove", 9)])
    assert not result["accepted_point_wearer_support"][2].any()
    assert not result["effective_removal"][2, 16, 16]
    assert result["retention_veto"][2, 16, 16] and result["unknown"][2, 16, 16]


def test_rejected_human_association_cannot_borrow_another_track_selection(monkeypatch):
    frames = rgb()
    rejected = area(frames)
    rejected[:, 0, 0] = True
    original = a._projection_association
    def association(masks, hints):
        if masks[:, 0, 0].any():
            return False, "UNKNOWN", {"synthetic_rejected_association": True}
        return original(masks, hints)
    monkeypatch.setattr(a, "_projection_association", association)
    result = build(frames, [point(frames, masks=rejected),
        proposal(frames, "human", 8), conflict_track(frames)])
    assert result["human"][:, 16, 16].all()
    assert not result["accepted_point_wearer_support"].any()
    assert result["retention_veto"][:, 16, 16].all()
    assert not result["effective_removal"][:, 16, 16].any()


@pytest.mark.parametrize("role", ["controller", "attachment", "cable", "object"])
def test_nonwearer_point_role_never_resolves_text_retention_veto(role):
    frames = rgb()
    result = build(frames, [proposal(frames, "human", 8),
        point(frames, role), conflict_track(frames)])
    assert not result["accepted_point_wearer_support"].any()
    assert result["retention_veto"][:, 16, 16].all()
    assert not result["effective_removal"][:, 16, 16].any()


@pytest.mark.parametrize("origin", ["SAM31_RGB_TEXT", "UNION_TEXT_ONLY"])
def test_text_support_cannot_be_relabelled_as_accepted_point(origin):
    frames = rgb()
    p = (proposal(frames) if origin == "SAM31_RGB_TEXT" else
         union(frames, "human", 0, retained=area(frames)))
    result = build(frames, [p, conflict_track(frames)])
    assert not result["accepted_point_wearer_support"].any()
    assert result["retention_veto"][:, 16, 16].all()
    assert not result["effective_removal"][:, 16, 16].any()


@pytest.mark.parametrize("kwargs", [
    {"refinement_receipt_payload_sha256": None},
    {"refinement_status": None}, {"refinement_status": "UNBOUND"},
])
def test_unbound_point_is_local_unknown_not_veto_exception_or_session_exit(kwargs):
    frames = rgb()
    result = build(frames, [point(frames, **kwargs), conflict_track(frames)])
    assert not result["accepted_point_wearer_support"].any()
    assert result["retention_veto"][:, 16, 16].all()
    assert not result["effective_removal"][:, 16, 16].any()
    assert result["unknown"][:, 16, 16].all()
    assert result["metadata"]["upstream_point_refinement_status"] == "UNBOUND_POINT_SOURCE_DECLARATION"


@pytest.mark.parametrize("status", [BOUND_STATUS, "INCOMPLETE_RGB_PROMPT_SUPPORT", "PROPAGATION_ABORTED"])
def test_existing_bound_point_statuses_use_only_current_valid_selected_pixels(status):
    frames = rgb()
    result = build(frames, [point(frames, refinement_status=status,
        unsupported_frames=np.array([False, False, True, False])), conflict_track(frames)])
    assert result["accepted_point_wearer_support"].any(axis=(1, 2)).tolist() == [True, True, False, True]


def test_whole_side_without_any_point_cannot_generate_point_support():
    frames = rgb()
    result = build(frames, [union(frames, "human", 0), conflict_track(frames)])
    assert not result["accepted_point_wearer_support"].any()
    assert result["retention_veto"][:, 16, 16].all()


def test_enabled_without_retention_conflicts_is_pixel_noop_and_no_point_generation():
    frames = rgb()
    proposals = [point(frames)]
    old, new = build(frames, proposals, None), build(frames, proposals)
    for key, value in old.items():
        if key != "metadata":
            assert_same(value, new[key])
    assert not new["retention_veto"].any()


def test_nonempty_visible_object_protection_remains_exact_and_zero_overlap():
    frames = rgb()
    obj = area(frames, (24, 14, 28, 20))
    proposals = [point(frames), conflict_track(frames),
                 proposal(frames, "object", 3, masks=obj)]
    old, new = build(frames, proposals, None), build(frames, proposals)
    assert old["protected_visible"].any()
    assert np.array_equal(old["protected_visible"], new["protected_visible"])
    assert np.array_equal(old["object_observed"], new["object_observed"])
    assert not (new["effective_removal"] & new["protected_visible"]).any()
    assert not (new["accepted_point_wearer_support"] & obj).any()


def test_unsupported_union_point_cannot_borrow_valid_text_wearer_pixel():
    frames = rgb()
    unsupported = np.array([False, True, False, False])
    p = union(frames, "human", 0, point_mask=area(frames), unsupported_frames=unsupported)
    result = build(frames, [p, proposal(frames, "human", 8), conflict_track(frames)])
    assert result["human"][1, 16, 16]
    assert not result["accepted_point_wearer_support"][1, 16, 16]
    assert result["retention_veto"][1, 16, 16]
    assert not result["effective_removal"][1, 16, 16]


@pytest.mark.parametrize("field", ["point_masks", "retained_text_masks", "retention_conflict_masks"])
def test_invalid_union_origin_not_tolerated_by_source_aware_policy(field):
    frames = rgb()
    p = union(frames, "human", 0, point_mask=area(frames))
    with pytest.raises(a.AutomaticMaskContractError, match="ORIGIN_OR_UNION"):
        build(frames, [replace(p, **{field: None}), conflict_track(frames)])


@pytest.mark.parametrize("field", [*SIG, "frames_sha256"])
def test_point_signature_drift_remains_rejected(field):
    frames = rgb()
    with pytest.raises(a.AutomaticMaskContractError, match="SHA_MISMATCH"):
        build(frames, [replace(point(frames), **{field: "e" * 64}), conflict_track(frames)])


@pytest.mark.parametrize("value", ["", "SOURCE_AWARE", 0, False, [], {}])
def test_unknown_or_wrong_type_policy_rejected_before_raw_processing(value):
    with pytest.raises(a.AutomaticMaskContractError, match="SOURCE_AWARE_RETENTION_VETO"):
        a.build_automatic_masks_with_source_aware_veto(object(), [], **SIG,
            source_aware_retention_veto=value)


def test_policy_does_not_mutate_any_proposal_or_frozen_parameters():
    frames = rgb()
    proposals = [union(frames, "human", 0, point_mask=area(frames)), conflict_track(frames)]
    snapshots = [(p.masks.copy(), p.point_masks.copy(), p.retained_text_masks.copy(),
                  p.retention_conflict_masks.copy()) for p in proposals]
    prompts, settings = dict(a.PROMPTS), dict(a.SETTINGS)
    build(frames, proposals)
    assert a.PROMPTS == prompts and a.SETTINGS == settings
    for p, snapshot in zip(proposals, snapshots):
        for current, previous in zip((p.masks, p.point_masks, p.retained_text_masks,
                                     p.retention_conflict_masks), snapshot):
            assert np.array_equal(current, previous)


def stage_env(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic-cpu")
    source = tmp_path / "native" / "vst.h264"
    source.parent.mkdir()
    source.write_bytes(b"synthetic raw source, no video or model execution")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"synthetic weight not loaded")
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True,
        "task_id": "CPU_SYNTHETIC_NOT_MODEL_PRODUCTION", "run_signature": "a" * 64,
        "fencing_token": "synthetic", "gpu_uuid": "GPU-synthetic-cpu", "worker_logical_index": 0}
    calls = []
    def collect(sam, frames, **signatures):
        calls.append(("collect", signatures.copy()))
        p = a.RGBProposal("human", 0, area(frames), a.frames_sha256(frames), **signatures)
        return [p], {"calls": [{"role": r} for r in a.PROMPTS]}
    def refine(sam, frames, originals, **kwargs):
        calls.append(("refine", kwargs.copy()))
        sig = {k: kwargs[k] for k in SIG}
        p = replace(originals[0], source="SAM31_RGB_POINT_AUTO",
                    refinement_receipt_payload_sha256="d" * 64,
                    refinement_status=BOUND_STATUS)
        other = conflict_track(frames)
        other = replace(other, **sig)
        return [p, other], {"automatic_points_submitted": 0, "synthetic_mock_not_actual_model": True}
    monkeypatch.setattr(stages, "collect_sam_rgb_proposals", collect)
    monkeypatch.setattr(stages, "refine_sam_rgb_proposals", refine)
    mapping = [{"native_frame_index": i, "decoded_display_index": i,
                "timebase_status": "TIMEBASE_UNVERIFIED"} for i in range(4)]
    return {"sam": SimpleNamespace(), "raw_source_ref": file_ref(source),
        "frame_map": mapping, "model_ref": file_ref(weight), "authorization": lambda: owner}, calls


@pytest.mark.parametrize("policy", [None, POLICY])
def test_stage_dispatch_seals_actual_flag_and_arrays_without_changing_mock_calls(tmp_path, monkeypatch, policy):
    options, calls = stage_env(tmp_path, monkeypatch)
    root = tmp_path / "Mask"
    receipt = stages.run_automatic_mask_stage(rgb(), output_dir=root,
        component_signature="f" * 64, source_aware_retention_veto=policy, **options)
    verify_sealed_stage(receipt, "f" * 64, output_root=root)
    assert [r[0] for r in calls] == ["collect", "refine"]
    assert all("source_aware_retention_veto" not in kwargs for _, kwargs in calls)
    assert receipt["native_input_frames"] == receipt["native_output_frames"] == 4
    assert receipt["text_prompt_invocations"] == 6
    assert receipt["manual_points_submitted"] == receipt["producer_manual_seed_reads"] == 0
    metadata = json.loads((root / "MASK_METADATA.json").read_text())
    with np.load(root / "MASK_ARRAYS.npz", allow_pickle=False) as arrays:
        if policy is None:
            assert "source_aware_retention_veto" not in receipt["actual_inputs"]
            assert "source_aware_retention_veto" not in metadata
            assert "accepted_point_wearer_support" not in arrays
            assert "retention_veto" not in arrays
            assert not arrays["effective_removal"][:, 16, 16].any()
        else:
            assert receipt["actual_inputs"]["source_aware_retention_veto"] == POLICY
            assert metadata["source_aware_retention_veto"] == POLICY
            assert arrays["accepted_point_wearer_support"].dtype == np.bool_
            assert arrays["retention_veto"].dtype == np.bool_
            assert arrays["effective_removal"][:, 16, 16].all()
        assert arrays["unknown"][:, 16, 16].all()
        assert not arrays["protected_visible"][:, 16, 16].any()
    assert receipt["training_eligible"] is False and receipt["human_review"] == "PENDING"


@pytest.mark.parametrize("value", ["", False, {}, "OTHER"])
def test_stage_invalid_policy_is_refused_before_mock_model_or_outputs(tmp_path, monkeypatch, value):
    options, calls = stage_env(tmp_path, monkeypatch)
    with pytest.raises(a.AutomaticMaskContractError, match="SOURCE_AWARE_RETENTION_VETO"):
        stages.run_automatic_mask_stage(rgb(), output_dir=tmp_path / "Mask",
            component_signature="f" * 64, source_aware_retention_veto=value, **options)
    assert calls == [] and not (tmp_path / "Mask").exists()


def sealed_clean_fixture(tmp_path, monkeypatch, *, sealed_variant=None, count=4):
    """A signed synthetic Mask with a real, nonempty veto beside original P."""
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    assert tmp_path.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # This is test authority only; production GPU gating remains unchanged.
    monkeypatch.setattr(stages, "_gate", lambda authority: dict(authority()))
    owner = {"task_id": "CPU_FAKE_TENSORS_NO_MODEL_EXECUTION",
             "run_signature": "a" * 64, "fencing_token": "synthetic",
             "gpu_uuid": "GPU-synthetic-cpu", "worker_logical_index": 0}
    source = tmp_path / "native" / "vst.h264"
    source.parent.mkdir()
    source.write_bytes(b"synthetic raw source; never decoded")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"synthetic weight; never loaded")
    frames = np.broadcast_to((10 + np.arange(count, dtype=np.uint8) * 5)
        [:, None, None, None], (count, 16, 24, 3)).copy()
    mapping = [{"native_frame_index": i, "decoded_display_index": i,
                "timebase_status": "TIMEBASE_UNVERIFIED"} for i in range(count)]
    removal = np.zeros(frames.shape[:3], bool)
    removal[:, 6:10, 8:12] = True
    protected = np.zeros_like(removal)
    protected[:, 7:9, 14:18] = True
    baseline, _ = effective_clean_masks(removal, protected)
    veto = np.zeros_like(removal)
    veto[:, 5:7, 6:8] = True
    effective = baseline & ~veto
    if sealed_variant == "protected":
        effective[:, 7, 15] = True
    elif sealed_variant == "outside_dilation":
        effective[:, 0, 0] = True
    arrays = {"removal": removal, "protected_visible": protected,
              "retention_veto": veto, "effective_removal": effective}
    if sealed_variant == "missing":
        arrays.pop("effective_removal")
    elif sealed_variant == "dtype":
        arrays["effective_removal"] = effective.astype(np.uint8)
    elif sealed_variant == "shape":
        arrays["effective_removal"] = effective[:-1]
    mask_dir = tmp_path / "Mask"
    mask_dir.mkdir()
    mask_arrays = stages._npz(mask_dir / "MASK_ARRAYS.npz", arrays)
    source_ref = file_ref(source)
    parent_receipt = {"schema": SCHEMA, "stage": "Mask", "status": "SEALED_COMPLETE",
        "component_signature": "b" * 64, "outputs": [mask_arrays],
        "actual_inputs": {"raw_source": source_ref,
                          "frames_sha256": a.frames_sha256(frames),
                          "frame_map_sha256": canonical_sha(mapping)}}
    receipt_ref = stages._json(mask_dir / "RECEIPT.json", parent_receipt)
    return SimpleNamespace(raw=frames, masks=arrays, baseline=baseline, effective=effective,
        protected=protected, veto=veto, mask_arrays=mask_arrays, receipt_ref=receipt_ref,
        options={"masks": arrays, "mask_receipt_ref": receipt_ref,
                 "raw_source_ref": source_ref, "frame_map": mapping,
                 "model_refs": [file_ref(weight)], "component_signature": "c" * 64,
                 "authorization": lambda: dict(owner)})


def sealed_cpu_parent():
    """Real CPU tensors through frozen PP; every NN and CUDA API is a fake."""
    import torch
    from chaoyang.exact78_baseline.clean import SETTINGS
    calls = {name: [] for name in ("raft", "flow", "combine", "propagate", "generate")}
    def capture(mask):
        assert mask.device.type == "cpu"
        return mask.detach().cpu().numpy().copy()
    class Flow:
        def forward_bidirect_flow(self, flows, masks):
            calls["flow"].append(capture(masks))
            return flows, None
        def combine_flow(self, original, predicted, masks):
            calls["combine"].append(capture(masks))
            return predicted
    class Generator:
        def img_propagation(self, masked, flows, masks, mode):
            assert mode == "nearest" and masked.device.type == "cpu"
            calls["propagate"].append(capture(masks))
            return torch.ones_like(masked).flatten(0, 1), torch.zeros_like(masks).flatten(0, 1)
        def __call__(self, frames, flows, masks, updated, neighbors):
            calls["generate"].append({"masks": capture(masks), "neighbors": neighbors,
                "frames": frames.detach().cpu().numpy().copy()})
            return torch.ones_like(frames[:, :neighbors])
    def raft(frames, iters):
        assert frames.device.type == "cpu" and iters == SETTINGS["raft_iter"]
        calls["raft"].append(tuple(frames.shape))
        shape = (1, frames.shape[1] - 1, 2, *frames.shape[-2:])
        return torch.zeros(shape), torch.zeros(shape)
    tensor_api = SimpleNamespace(**{name: getattr(torch, name) for name in (
        "no_grad", "autocast", "from_numpy", "any", "cat", "equal")},
        cuda=SimpleNamespace(empty_cache=lambda: None, synchronize=lambda device: None))
    return SimpleNamespace(settings=dict(SETTINGS), torch=tensor_api, device=torch.device("cpu"),
        raft=raft, flow=Flow(), model=Generator()), calls


@pytest.mark.parametrize("hook", ["NONE", "DIAGNOSTIC", "NOOP_SELECTOR"])
def test_sealed_effective_real_pp_cpu_path_preserves_original_p_and_reference_graph(
        tmp_path, monkeypatch, hook):
    from chaoyang.pick_place_1008.visual import ProtectedProPainter, effective_clean_masks
    from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
    sample = sealed_clean_fixture(tmp_path, monkeypatch, count=24)
    old_parent, old_calls = sealed_cpu_parent()
    parent, calls = sealed_cpu_parent()
    original_globals = (ProtectedProPainter.inpaint.__globals__["effective_clean_masks"],
                        ProtectedProPainter.inpaint.__globals__["_get_ref_index"])
    old = ProtectedProPainter(old_parent).inpaint(sample.raw, sample.masks["removal"], sample.protected)
    wrapper = stages.ReferenceSelectingProtectedProPainter(parent,
        (lambda **row: row["original_references"]) if hook == "NOOP_SELECTOR" else None,
        record_reference_selection=hook == "DIAGNOSTIC", sealed_effective_removal=sample.effective)
    result = wrapper.inpaint(sample.raw, sample.masks["removal"], sample.protected)
    assert wrapper._hooked.__func__.__code__ is ProtectedProPainter.inpaint.__code__
    assert original_globals == (effective_clean_masks, _get_ref_index)
    assert (ProtectedProPainter.inpaint.__globals__["effective_clean_masks"],
            ProtectedProPainter.inpaint.__globals__["_get_ref_index"]) == original_globals
    assert wrapper.wrapped.settings == parent.settings == SETTINGS
    assert np.array_equal(wrapper.wrapped.effective_masks, sample.effective)
    assert not np.shares_memory(wrapper.wrapped.effective_masks, sample.effective)
    assert not (sample.effective & sample.protected).any()
    assert (sample.baseline & sample.veto).any() and not (sample.effective & sample.veto).any()
    assert np.array_equal(result[~sample.effective], sample.raw[~sample.effective])
    assert np.array_equal(result[sample.protected], sample.raw[sample.protected])
    assert (result[sample.effective] == 255).all()
    assert (old[sample.veto] == 255).all()  # The old recompute loses this veto.
    for name in ("flow", "combine", "propagate"):
        assert len(calls[name]) == len(old_calls[name]) == 1
        assert np.array_equal(calls[name][0][0, :, 0].astype(bool), sample.effective)
    assert calls["raft"] == old_calls["raft"]
    assert len(calls["generate"]) == len(old_calls["generate"])
    for index, (current, previous) in enumerate(zip(calls["generate"], old_calls["generate"])):
        mid = index * (SETTINGS["neighbor_length"] // 2)
        neighbors = list(range(max(0, mid - 5), min(len(sample.raw), mid + 6)))
        references = _get_ref_index(mid, neighbors, len(sample.raw), SETTINGS["ref_stride"], -1)
        assert current["neighbors"] == previous["neighbors"] == len(neighbors)
        assert np.array_equal(current["masks"][0, :, 0].astype(bool),
                              sample.effective[neighbors + references])
    if hook != "NONE":
        assert all(row["references"] == row["original_references"] for row in wrapper.calls)
    assert wrapper.wrapped.last_validation["sealed_effective_removal_consumed"] is True
    assert wrapper.wrapped.last_validation["protected_changed_pixels_per_frame"] == [0] * 24


def test_clean_stage_seals_exact_e_and_original_p_after_real_pp_cpu_fake(tmp_path, monkeypatch):
    sample = sealed_clean_fixture(tmp_path, monkeypatch)
    parent, calls = sealed_cpu_parent()
    receipt = stages.run_clean_stage(sample.raw, output_dir=tmp_path / "Clean",
        parent_propainter=parent, sealed_effective_removal=sample.effective, **sample.options)
    verify_sealed_stage(receipt, "c" * 64, output_root=tmp_path / "Clean")
    with np.load(tmp_path / "Clean" / "CLEAN_ARRAYS.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["effective_removal"], sample.effective)
        assert np.array_equal(arrays["protected_visible"], sample.protected)
        assert np.array_equal(arrays["clean_rgb"][~sample.effective], sample.raw[~sample.effective])
        assert (arrays["clean_rgb"][sample.effective] == 255).all()
    assert all(len(value) == 1 for value in calls.values())
    assert receipt["actual_inputs"]["effective_removal_source"] == {
        "mask_arrays": sample.mask_arrays, "array_key": "effective_removal",
        "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}
    assert receipt["validation"]["protected_changed_pixels"] == 0
    assert receipt["validation"]["outside_delete_changed_pixels"] == 0
    assert receipt["visual_quality"] == "NOT_EVALUATED" and receipt["training_eligible"] is False


@pytest.mark.parametrize("bad", ["list", "dtype", "short", "spatial", "changed",
    "sealed_missing", "sealed_dtype", "sealed_shape", "protected", "outside_dilation"])
def test_bad_sealed_e_is_rejected_before_wrapper_or_model(tmp_path, monkeypatch, bad):
    variant = bad.removeprefix("sealed_") if bad.startswith("sealed_") else (
        bad if bad in {"protected", "outside_dilation"} else None)
    sample = sealed_clean_fixture(tmp_path, monkeypatch, sealed_variant=variant)
    provided = sample.effective.copy()
    expected = "CLEAN_SEALED_EFFECTIVE_DIFFERS_FROM_UPSTREAM"
    if bad == "list":
        provided = provided.tolist()
        expected = "CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED"
    elif bad == "dtype":
        provided = provided.astype(np.uint8)
        expected = "CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED"
    elif bad == "short":
        provided = provided[:-1]
        expected = "CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED"
    elif bad == "spatial":
        provided = provided[:, :, :-1]
        expected = "CLEAN_SEALED_EFFECTIVE_DOMAIN_REQUIRED"
    elif bad == "changed":
        provided[:, 5, 6] = True
    elif bad in {"protected", "outside_dilation"}:
        expected = "CLEAN_SEALED_EFFECTIVE_PROTECTION_OR_DILATION_VIOLATION"
    constructed = []
    def forbidden_wrapper(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("BAD_E_REACHED_MODEL_WRAPPER")
    monkeypatch.setattr(stages, "ReferenceSelectingProtectedProPainter", forbidden_wrapper)
    parent, calls = sealed_cpu_parent()
    with pytest.raises(stages.ModelStageError, match=expected):
        stages.run_clean_stage(sample.raw, output_dir=tmp_path / "BadClean", parent_propainter=parent,
            sealed_effective_removal=provided, **sample.options)
    assert constructed == [] and all(value == [] for value in calls.values())
    assert not (tmp_path / "BadClean").exists()


def test_none_clean_keeps_legacy_wrapper_keywords_and_receipt(tmp_path, monkeypatch):
    from chaoyang.exact78_baseline.clean import SETTINGS
    sample = sealed_clean_fixture(tmp_path, monkeypatch)
    constructors = []
    class LegacyWrapper:
        # Intentionally has the exact old constructor ABI.
        def __init__(self, parent, selector, *, record_reference_selection=False):
            constructors.append((selector, record_reference_selection))
            self.wrapped = SimpleNamespace(settings=dict(SETTINGS), effective_masks=sample.baseline)
            self.calls, self.original_function_sha256 = [], "f" * 64
        def inpaint(self, raw, removal, protected):
            result = raw.copy()
            result[sample.baseline] = 255
            return result
    monkeypatch.setattr(stages, "ReferenceSelectingProtectedProPainter", LegacyWrapper)
    receipts = []
    for name, options in [("Omitted", {}), ("ExplicitNone", {"sealed_effective_removal": None})]:
        receipt = stages.run_clean_stage(sample.raw, output_dir=tmp_path / name,
            parent_propainter=SimpleNamespace(), **sample.options, **options)
        assert "effective_removal_source" not in receipt["actual_inputs"]
        normalized = {key: value for key, value in receipt.items() if key not in {"timings", "outputs"}}
        normalized["outputs"] = [{"name": Path(row["path"]).name, "bytes": row["bytes"],
                                  "sha256": row["sha256"]} for row in receipt["outputs"]]
        receipts.append(normalized)
    assert_same(*receipts)
    assert constructors == [(None, False), (None, False)]
    assert inspect.signature(stages.run_clean_stage).parameters["sealed_effective_removal"].default is None


@pytest.mark.parametrize("drift", ["expanded", "dtype"])
def test_clean_cannot_seal_if_model_effective_domain_drifts(tmp_path, monkeypatch, drift):
    from chaoyang.exact78_baseline.clean import SETTINGS
    sample = sealed_clean_fixture(tmp_path, monkeypatch)
    class DriftingWrapper:
        def __init__(self, parent, selector, **kwargs):
            effective = sample.baseline if drift == "expanded" else sample.effective.astype(np.uint8)
            self.wrapped = SimpleNamespace(settings=dict(SETTINGS), effective_masks=effective)
            self.calls, self.original_function_sha256 = [], "f" * 64
        def inpaint(self, raw, removal, protected):
            return raw.copy()
    monkeypatch.setattr(stages, "ReferenceSelectingProtectedProPainter", DriftingWrapper)
    with pytest.raises(stages.ModelStageError, match="CLEAN_SEALED_EFFECTIVE_MODEL_DOMAIN_CHANGED"):
        stages.run_clean_stage(sample.raw, output_dir=tmp_path / "DriftClean",
            parent_propainter=SimpleNamespace(), sealed_effective_removal=sample.effective, **sample.options)
    assert not (tmp_path / "DriftClean" / "RECEIPT.json").exists()
    assert (tmp_path / "DriftClean" / "FAILED.json").exists()
