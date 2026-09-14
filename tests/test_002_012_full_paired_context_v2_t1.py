import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task29_context_v2_test",
    PROJECT / "tools/prepare_002_012_full_paired_context_v2_t1.py",
)
assert SPEC is not None and SPEC.loader is not None
context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = context
SPEC.loader.exec_module(context)


def synthetic(
    *,
    left_xy=(100.0, 200.0),
    right_xy=(1100.0, 980.0),
    left_slot=-1,
    right_slot=0,
    left_valid=True,
    right_valid=True,
    left_state=3,
    right_state=1,
):
    joints = np.zeros((2, 1, 21, 2), dtype=np.float64)
    joints[0, 0] = np.asarray(left_xy)
    joints[1, 0] = np.asarray(right_xy)
    arrays = {
        "joints": joints,
        "valid": np.asarray([[left_valid], [right_valid]], dtype=bool),
        "session_id": np.zeros((2, 1), dtype=np.int32),
        "state_code": np.asarray([[left_state], [right_state]], dtype=np.uint8),
        "state_names": np.asarray(
            [
                "invalid",
                "observed_gated_smoothed",
                "short_optical_flow_or_interpolation",
                "mid_bidirectional_interpolation",
                "reinitialized",
            ]
        ),
        "measurement_accepted": np.asarray([[False], [True]], dtype=bool),
        "source_slot": np.asarray([[left_slot], [right_slot]], dtype=np.int8),
        "label_override": np.zeros((2, 1), dtype=bool),
    }
    quality = {}
    for axis, side in enumerate(("left", "right")):
        quality[(side, 0)] = {
            "frame": "0",
            "hand": side,
            "valid": "1" if arrays["valid"][axis, 0] else "0",
            "state": str(arrays["state_names"][arrays["state_code"][axis, 0]]),
            "session_id": "0",
            "measurement_accepted": (
                "1" if arrays["measurement_accepted"][axis, 0] else "0"
            ),
            "identity_label_override": "0",
        }
    return arrays, quality


def test_corrected_expected_counts_are_frozen():
    assert context.EXPECTED_OPERATIONAL["grap_a_cap_002"]["right"] == {
        "AVAILABLE": 281,
        "OUTSIDE_IMAGE": 112,
        "MISSING": 0,
    }
    assert context.EXPECTED_OPERATIONAL["grap_a_cap_012"]["left"] == {
        "AVAILABLE": 364,
        "OUTSIDE_IMAGE": 0,
        "MISSING": 0,
    }
    assert context.EXPECTED_OPERATIONAL["grap_a_cap_012"]["right"] == {
        "AVAILABLE": 251,
        "OUTSIDE_IMAGE": 113,
        "MISSING": 0,
    }


def test_left_mid_interpolation_never_borrows_right_xy_or_track():
    arrays, quality = synthetic()
    left = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="left"
    )
    right = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="right"
    )
    assert left["physical_hand_axis_index"] == 0
    assert left["source_track_index"] == 0
    assert left["wrist_xy"] == [100.0, 200.0]
    assert left["source_state"] == "mid_bidirectional_interpolation"
    assert left["source_slot_observation"] == -1
    assert right["physical_hand_axis_index"] == 1
    assert right["wrist_xy"] == [1100.0, 980.0]


def test_source_slot_pair_swap_cannot_swap_physical_xy():
    arrays, quality = synthetic(left_slot=1, right_slot=0, left_state=3, right_state=2)
    left = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="left"
    )
    right = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="right"
    )
    assert (left["source_slot_observation"], right["source_slot_observation"]) == (1, 0)
    assert left["wrist_xy"] == [100.0, 200.0]
    assert right["wrist_xy"] == [1100.0, 980.0]
    assert left["lineage_id"] != right["lineage_id"]


def test_right_missing_does_not_change_left_lineage_or_xy():
    arrays, quality = synthetic(right_valid=False)
    left = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="left"
    )
    right = context.physical_side_record(
        arrays, quality, session="grap_a_cap_012", frame=0, side="right"
    )
    assert left["state"] == "AVAILABLE"
    assert left["wrist_xy"] == [100.0, 200.0]
    assert left["source_track_index"] == 0
    assert right["state"] == "MISSING"
    assert right["wrist_xy"] is None
    assert right["source_track_index"] == 1


def test_csv_npz_identity_mismatch_is_fail_closed():
    arrays, quality = synthetic()
    quality[("left", 0)]["hand"] = "right"
    with pytest.raises(context.ContextV2Error, match="CSV/NPZ"):
        context.physical_side_record(
            arrays, quality, session="grap_a_cap_012", frame=0, side="left"
        )


def test_quality_csv_requires_complete_physical_hand_axis_order(tmp_path: Path):
    path = tmp_path / "quality.csv"
    path.write_text(
        "frame,hand,valid,state,session_id,measurement_accepted,identity_label_override\n"
        "0,right,1,observed_gated_smoothed,0,1,0\n"
        "0,left,1,observed_gated_smoothed,0,1,0\n"
    )
    with pytest.raises(context.ContextV2Error, match="hand-axis order"):
        context.load_quality_rows(path, 1)


def test_operational_and_pixel_center_boundaries_explain_31_3():
    wrist = np.asarray([1150.844, 959.408447265625])
    assert context.operational_inside(wrist)
    assert not context.pixel_center_inside(wrist)
    distance, direction = context.outside_distance(wrist, pixel_center=True)
    assert distance == pytest.approx(0.408447265625)
    assert direction == "BOTTOM"
    wrist = np.asarray([1151.2223, 961.1878662109375])
    distance, direction = context.outside_distance(wrist, pixel_center=False)
    assert distance == pytest.approx(2.1878662109375)
    assert direction == "BOTTOM"


def test_real_012_fixed_axis_counts_denominators_and_alias_inventory():
    rows, summary = context.authority_context(PROJECT, "grap_a_cap_012", 364)
    assert len(rows) == 364
    left = summary["sides"]["left"]
    right = summary["sides"]["right"]
    assert left["state_counts"] == {"AVAILABLE": 364, "OUTSIDE_IMAGE": 0, "MISSING": 0}
    assert right["state_counts"] == {"AVAILABLE": 251, "OUTSIDE_IMAGE": 113, "MISSING": 0}
    assert right["operational_oob_intervals"] == [[251, 363]]
    assert right["pixel_center_oob_intervals"] == [[250, 363]]
    assert right["denominators"]["total_frames"]["rate"] == pytest.approx(113 / 364)
    assert right["denominators"]["valid_finite_geometry"]["rate"] == pytest.approx(113 / 364)
    observed = right["denominators"]["measurement_accepted_observed_only"]
    assert (observed["oob_numerator"], observed["denominator"]) == (112, 363)
    pixel = right["denominators"]["last_integer_pixel_center_audit_total"]
    assert (pixel["oob_numerator"], pixel["denominator"]) == (114, 364)
    assert left["source_slot_observation_counts"] == {"-1": 11, "0": 352, "1": 1}
    assert right["source_slot_observation_counts"] == {"0": 12, "1": 352}
    for frame in list(range(312, 318)) + list(range(335, 340)):
        assert rows[frame]["sides"]["left"]["source_track_index"] == 0
        assert rows[frame]["sides"]["right"]["source_track_index"] == 1
        assert rows[frame]["sides"]["left"]["source_state"] == "mid_bidirectional_interpolation"


def test_intervals_are_contiguous_and_lossless():
    assert context.intervals([1, 2, 3, 7, 9, 10]) == [[1, 3], [7, 7], [9, 10]]
    assert context.intervals([]) == []


def test_identity_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.json").write_text("{}")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        context.identity(tmp_path / "link/x.json")


def test_secure_create_direct_child_rejects_symlink_parent(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "audits"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        context.secure_create_direct_child(link, "attack")
    assert not (outside / "attack").exists()


def test_secure_create_direct_child_and_writer_are_oexcl(tmp_path: Path):
    parent = tmp_path / "audits"
    parent.mkdir()
    child, first = context.secure_create_direct_child(parent, "v2")
    assert child.is_dir()
    assert first["st_dev"] > 0 and first["st_ino"] > 0
    with pytest.raises(FileExistsError):
        context.secure_create_direct_child(parent, "v2")
    path = child / "x.bin"
    context.exclusive_bytes(path, b"first")
    with pytest.raises(FileExistsError):
        context.exclusive_bytes(path, b"second")
    assert path.read_bytes() == b"first"
