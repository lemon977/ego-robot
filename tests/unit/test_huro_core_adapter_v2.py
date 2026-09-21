"""Known-answer adapter tests; not a claim of official core/FK execution."""
import hashlib
import importlib.metadata
import json
import math
import pytest
from chaoyang.pipeline.huro_core_adapter_v2 import (
    AdapterValidationError as Error, continuous_segments, core_execution_status,
    dependency_metadata, validate_frozen_placement, validate_samples, validate_robot_mapping,
)


def fixture(n=3):
    return (list(range(n)), [i*30_000_000 for i in range(n)],
            [[[[0., 0., 0.] for _ in range(21)] for _ in range(2)] for _ in range(n)],
            [[[True]*21, [True]*21] for _ in range(n)])


def test_dependency_metadata_missing_does_not_claim_runtime():
    def missing(_): raise importlib.metadata.PackageNotFoundError()
    data = dependency_metadata(missing)
    assert len(data["missing"]) == 10
    assert core_execution_status(data)["status"] == "BLOCKED_DEPENDENCY"
    assert not core_execution_status(data)["official_core_executed"]


def test_present_metadata_is_not_execution():
    data = dependency_metadata(lambda _: "0.0")
    assert core_execution_status(data)["status"] == "BLOCKED_CORE_ADAPTER_UNIMPLEMENTED"
    assert data["version_compatibility"] == "NOT_VERIFIED"


def test_partial_and_unknown_masks_stay_independent():
    ids, clock, points, masks = fixture()
    masks[1][0] = [False]*21
    points[1][0] = [[math.nan]*3 for _ in range(21)]
    masks[2][1][8] = False
    result = validate_samples(ids, clock, points, masks, ["left", "right"], "m")
    assert result["observed_point_counts"] == [42, 62]
    assert not result["inference_added"]


@pytest.mark.parametrize("clock", [[0, 1, 1], [0, 2, 1], [0., 1., 2.]])
def test_duplicate_reverse_float_timestamp_rejected(clock):
    ids, _, points, masks = fixture()
    with pytest.raises(Error): validate_samples(ids, clock, points, masks, ["left", "right"], "m")


@pytest.mark.parametrize("names,units", [(["right", "left"], "m"), (["left", "right"], "mm")])
def test_side_and_units_rejected(names, units):
    with pytest.raises(Error): validate_samples(*fixture(), names, units)


def test_observed_nan_rejected():
    ids, clock, points, masks = fixture()
    points[0][0][0][0] = math.nan
    with pytest.raises(Error): validate_samples(ids, clock, points, masks, ["left", "right"], "m")


def test_per_hand_gap_does_not_erase_other_hand():
    ids, clock, _, masks = fixture(5)
    masks[2][0] = [False]*21
    segments = continuous_segments(ids, clock, masks, max_gap_ns=50_000_000)
    assert [x["source_indices"] for x in segments if x["side"] == "left"] == [[0, 1], [3, 4]]
    assert [x["source_indices"] for x in segments if x["side"] == "right"] == [[0, 1, 2, 3, 4]]


def test_source_gap_not_compressed():
    _, clock, _, masks = fixture(3)
    result = continuous_segments([1, 2, 5], clock, masks, max_gap_ns=50_000_000)
    assert result[0]["source_frame_ids"] == [1, 2]
    assert result[1]["source_frame_ids"] == [5]


def test_nonuniform_dt_preserved_and_large_gap_split():
    ids, _, points, masks = fixture()
    clock = [0, 20_000_000, 100_000_000]
    validate_samples(ids, clock, points, masks, ["left", "right"], "m")
    out = continuous_segments(ids, clock, masks, max_gap_ns=40_000_000)
    assert out[0]["source_indices"] == [0, 1]
    assert out[1]["source_indices"] == [2]
    assert clock == [0, 20_000_000, 100_000_000]


def test_378_frames_explicit_chunks_not_silent_drop():
    ids, clock, _, masks = fixture(378)
    result = continuous_segments(ids, clock, masks, max_gap_ns=50_000_000)
    left = [r for r in result if r["side"] == "left"]
    assert [len(r["source_indices"]) for r in left] == [224, 154]
    assert [i for r in left for i in r["source_indices"]] == ids
    assert left[0]["boundary"] == "CHUNK_LIMIT"
    assert not any(r["solver_executed"] for r in result)


def test_empty_and_invalid_limits_rejected():
    with pytest.raises(Error): continuous_segments([], [], [], max_gap_ns=1)
    ids, clock, _, masks = fixture()
    with pytest.raises(Error): continuous_segments(ids, clock, masks, max_gap_ns=0)


def placement():
    value = [[1., 0., 0., .2], [0., 1., 0., .3], [0., 0., 1., .4], [0., 0., 0., 1.]]
    sha = hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
    return value, sha


def test_frozen_placement_no_estimation():
    matrix, sha = placement()
    assert validate_frozen_placement(matrix, sha) == {"sha256": sha, "placement_optimized": False}
    matrix[0][3] += .1
    with pytest.raises(Error): validate_frozen_placement(matrix, sha)


def test_reflection_and_scaling_rejected():
    matrix, sha = placement()
    matrix[0][0] = -1
    with pytest.raises(Error): validate_frozen_placement(matrix, sha)
    matrix[0][0] = 2
    with pytest.raises(Error): validate_frozen_placement(matrix, sha)


def mapping_fixture():
    names = ["base", "camera"] + [f"{s}{i}" for s in "lr" for i in range(21)]
    xml = '<robot name="fixture">' + ''.join(f'<link name="{n}"/>' for n in names)
    xml += '<joint name="active" type="revolute"/><joint name="unused" type="revolute"/></robot>'
    cfg = {"joints": {"lh": ["active"], "rh": []},
           "keypoint_mapping": {"left": {i: f"l{i}" for i in range(21)}, "right": {i: f"r{i}" for i in range(21)}},
           "eef_link_names": {"left": "l0", "right": "r0"},
           "eef_hand_joint_groups": {"left": "lh", "right": "rh"}, "camera_link": "camera"}
    return xml, cfg


def test_mapping_static_not_fk_proof():
    result = validate_robot_mapping(*mapping_fixture())
    assert result["locked_joints"] == ["unused"]
    assert result["fk_parity"] == "NOT_EXECUTED"


def test_mapping_bad_joint_rejected():
    xml, cfg = mapping_fixture()
    cfg["joints"]["lh"] = ["absent"]
    with pytest.raises(Error): validate_robot_mapping(xml, cfg)


def test_mapping_side_reuse_rejected():
    xml, cfg = mapping_fixture()
    cfg["keypoint_mapping"]["right"][8] = "l8"
    with pytest.raises(Error): validate_robot_mapping(xml, cfg)


def test_mapping_missing_tip_rejected():
    xml, cfg = mapping_fixture()
    del cfg["keypoint_mapping"]["left"][4]
    with pytest.raises(Error): validate_robot_mapping(xml, cfg)
