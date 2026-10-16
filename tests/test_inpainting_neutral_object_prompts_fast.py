from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.inpainting_candidate.neutral_object_prompts import (
    BUNDLE_SCHEMA, POLICY_KEY, NeutralObjectPromptError,
    neutralize_generic_object_negatives,
)


def fixture():
    return {
        "schema": BUNDLE_SCHEMA, "anchor_frame": 3,
        "input_bindings": {"frames_sha256": "a" * 64},
        "instance_association": [{"role": "object", "side": "UNKNOWN"}],
        "objects": [{
            "obj_id": 9, "frame_idx": 3, "role": "glove", "side": "right",
            "points_px": [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]],
            "point_labels": [1, 0, 1, 0, 0],
            "negative_proposal_sources": [
                {"role": "object", "track_id": 4, "source": "CURRENT_RGB", "extra": [1]},
                {"role": "glove", "track_id": 2, "source": "CURRENT_RGB"},
                {"role": "object", "track_id": 8, "source": "CURRENT_RGB"},
            ],
            "extra": {"preserved": [7]},
        }],
    }


def test_only_object_negatives_removed_with_exact_alignment():
    original = fixture()
    out = neutralize_generic_object_negatives(original)
    obj = out["objects"][0]
    assert obj["points_px"] == [[1., 2.], [5., 6.], [7., 8.]]
    assert obj["point_labels"] == [1, 1, 0]
    assert obj["negative_proposal_sources"] == original["objects"][0]["negative_proposal_sources"][1:2]
    for key, value in original.items():
        if key != "objects":
            assert out[key] == value
    for key, value in original["objects"][0].items():
        if key not in ("points_px", "point_labels", "negative_proposal_sources"):
            assert obj[key] == value
    audit = out[POLICY_KEY]
    assert audit["generic_object_identity"] == "UNKNOWN_NOT_DEVICE"
    assert audit["device_promotion"] is audit["threshold_changes"] is False
    assert audit["counts"] == {"objects": 1, "positive_before": 2, "positive_after": 2,
                              "negative_before": 3, "negative_after": 1,
                              "generic_object_negatives_removed": 2}
    assert [r["point_index_original"] for r in audit["removed_negatives"]] == [1, 4]
    assert [r["negative_source_index_original"] for r in audit["removed_negatives"]] == [0, 2]
    assert [r["point_px"] for r in audit["removed_negatives"]] == [[3., 4.], [9., 10.]]
    assert [r["negative_proposal_source"] for r in audit["removed_negatives"]] == [
        original["objects"][0]["negative_proposal_sources"][i] for i in (0, 2)]


def test_deep_copy_leaves_input_and_nested_values_unchanged():
    original = fixture()
    before = deepcopy(original)
    out = neutralize_generic_object_negatives(original)
    out["objects"][0]["points_px"][0][0] = 999
    out["objects"][0]["extra"]["preserved"].append(8)
    out[POLICY_KEY]["removed_negatives"][0]["negative_proposal_source"]["extra"].append(2)
    out["input_bindings"]["frames_sha256"] = "b" * 64
    assert original == before


def test_instance_order_anchors_and_all_other_negative_roles_preserved():
    original = fixture()
    base = deepcopy(original["objects"][0])
    base["obj_id"] = 2
    base["frame_idx"] = 11
    for source, role in zip(base["negative_proposal_sources"], ("human", "attachment", "cable")):
        source["role"] = role
    original["objects"].insert(0, base)
    out = neutralize_generic_object_negatives(original)
    assert [o["obj_id"] for o in out["objects"]] == [2, 9]
    assert [o["frame_idx"] for o in out["objects"]] == [11, 3]
    assert out["objects"][0] == base


def test_no_generic_and_no_negative_and_empty_objects_are_supported():
    original = fixture()
    for source in original["objects"][0]["negative_proposal_sources"]:
        source["role"] = "controller"
    out = neutralize_generic_object_negatives(original)
    assert out["objects"] == original["objects"]
    assert out[POLICY_KEY]["counts"]["generic_object_negatives_removed"] == 0
    obj = original["objects"][0]
    obj.update(points_px=[[1., 2.]], point_labels=[1], negative_proposal_sources=[])
    assert neutralize_generic_object_negatives(original)["objects"] == original["objects"]
    original["objects"] = []
    assert neutralize_generic_object_negatives(original)[POLICY_KEY]["counts"]["objects"] == 0
    for invalid in (None, {}, {"schema": BUNDLE_SCHEMA, "objects": None}):
        with pytest.raises(NeutralObjectPromptError):
            neutralize_generic_object_negatives(invalid)
    with pytest.raises(NeutralObjectPromptError, match="ALREADY_APPLIED"):
        neutralize_generic_object_negatives(out)


def test_bad_point_labels_and_negative_alignment_fail_without_mutation():
    for field, value in (
        ("point_labels", [1, 0]),
        ("point_labels", [1, 0, True, 0, 0]),
        ("point_labels", [1, 0, 0.0, 0, 0]),
        ("point_labels", [1, 0, 2, 0, 0]),
        ("negative_proposal_sources", []),
    ):
        original = fixture()
        original["objects"][0][field] = value
        before = deepcopy(original)
        with pytest.raises(NeutralObjectPromptError):
            neutralize_generic_object_negatives(original)
        assert original == before


def test_missing_or_malformed_negative_identity_is_rejected():
    for source in (None, {}, {"role": "object", "track_id": True, "source": "CURRENT_RGB"},
                   {"role": "object", "track_id": 0},
                   {"role": "", "track_id": 0, "source": "CURRENT_RGB"}):
        original = fixture()
        original["objects"][0]["negative_proposal_sources"][0] = source
        with pytest.raises(NeutralObjectPromptError, match="SOURCE_IDENTITY"):
            neutralize_generic_object_negatives(original)


def test_malformed_coordinates_and_identity_are_rejected():
    for point in ([1], [True, 2], ["1", 2], [float("nan"), 2], [1, float("inf")]):
        original = fixture()
        original["objects"][0]["points_px"][0] = point
        with pytest.raises(NeutralObjectPromptError, match="FINITE_XY"):
            neutralize_generic_object_negatives(original)
    for field, value in (("obj_id", True), ("obj_id", 0), ("frame_idx", -1)):
        original = fixture()
        original["objects"][0][field] = value
        with pytest.raises(NeutralObjectPromptError):
            neutralize_generic_object_negatives(original)
    original = fixture()
    original["objects"].append(deepcopy(original["objects"][0]))
    with pytest.raises(NeutralObjectPromptError, match="UNIQUE"):
        neutralize_generic_object_negatives(original)


def test_sealed_diagnostic_bundles_have_exact_unchanged_positive_counts():
    # Evaluation-only archived bundles, never producer seed inputs.
    root = Path(__file__).resolve().parents[1] / "_run/inpainting_auto_v1_seed_mask_20261013/attempt_0001/sessions"
    expected = {
        "001": ("c3ff2e129ba8d981eb156453c8ac12289b8e8f40a5a96c103b4db70109d12d54", 7, 21, 34, 14),
        "050": ("57048f72c6611f153fab1200cb9caf396c8f951a9b42d879aa6aebcccd808d32", 10, 30, 30, 21),
    }
    if not all((root / key / "Mask/AUTOMATIC_SEEDS.json").is_file() for key in expected):
        pytest.skip("Sealed diagnostic-only bundles unavailable")
    totals = [0, 0, 0, 0]
    for key, (sha, objects, positive, before_negative, after_negative) in expected.items():
        raw = (root / key / "Mask/AUTOMATIC_SEEDS.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        original = json.loads(raw)
        before = deepcopy(original)
        out = neutralize_generic_object_negatives(original)
        assert original == before
        counts = out[POLICY_KEY]["counts"]
        assert counts == {
            "objects": objects, "positive_before": positive, "positive_after": positive,
            "negative_before": before_negative, "negative_after": after_negative,
            "generic_object_negatives_removed": before_negative - after_negative,
        }
        expected_bundle = deepcopy(original)
        for obj in expected_bundle["objects"]:
            source_iter = iter(obj["negative_proposal_sources"])
            keep_points, keep_labels, keep_sources = [], [], []
            for point, label in zip(obj["points_px"], obj["point_labels"]):
                source = next(source_iter) if label == 0 else None
                if source is not None and source["role"] == "object":
                    continue
                keep_points.append(point)
                keep_labels.append(label)
                if source is not None:
                    keep_sources.append(source)
            obj.update(points_px=keep_points, point_labels=keep_labels,
                       negative_proposal_sources=keep_sources)
        assert {key: value for key, value in out.items() if key != POLICY_KEY} == expected_bundle
        for index, value in enumerate((positive, positive, before_negative, after_negative)):
            totals[index] += value
    assert totals == [51, 51, 64, 35]
