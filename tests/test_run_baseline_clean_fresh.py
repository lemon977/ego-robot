from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.depth_multiview_clean_successor import (  # noqa: E402
    SOURCE_DONOR,
    SOURCE_PROTECTED_OBJECT,
    SOURCE_UNSUPPORTED,
    audit_exact_provenance,
)
from tools import run_baseline_clean_fresh as fresh  # noqa: E402
from tools import run_static_scene_clean_v3 as geometry  # noqa: E402


def test_exact_real_donor_fusion_and_immutable_fallback() -> None:
    raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    removal = np.zeros((6, 8), bool)
    removal[2, 2:5] = True
    protected = np.zeros_like(removal)
    protected[2, 5] = True
    tracker = removal.copy()

    # Three donors agree at the first two removal pixels.  The third removal
    # pixel has only two donors and must therefore stay byte-identical raw.
    valid = np.array(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[10, 20, 30], [40, 50, 60], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    sx = np.array([[1, 2, 3]] * 3, dtype=np.int32)
    sy = np.array([[1, 1, 1]] * 3, dtype=np.int32)
    depths = np.ones((3, 3), dtype=np.float32)
    fused = geometry.fuse_exact_medoids(valid, rgb, sx, sy, depths, [1, 3, 5])
    result = geometry.compose(
        raw,
        removal,
        tracker,
        protected,
        np.array([2, 2, 2]),
        np.array([2, 3, 4]),
        fused,
    )

    donors = {}
    for frame_id in (1, 3, 5):
        donor = np.zeros_like(raw)
        donor[1, 1] = [10, 20, 30]
        donor[1, 2] = [40, 50, 60]
        donor[1, 3] = [70, 80, 90]
        donors[frame_id] = donor
    audit = audit_exact_provenance(
        raw,
        result["clean"],
        result["source_kind"],
        result["source_frame"],
        result["source_x"],
        result["source_y"],
        donors,
    )
    assert audit["verified"]
    assert np.all(result["source_kind"][2, 2:4] == SOURCE_DONOR)
    assert result["source_kind"][2, 4] == SOURCE_UNSUPPORTED
    assert result["source_kind"][2, 5] == SOURCE_PROTECTED_OBJECT
    assert np.array_equal(result["clean"][2, 4], raw[2, 4])
    assert np.array_equal(result["clean"][2, 5], raw[2, 5])
    assert result["changed_outside_removal_pixels"] == 0
    assert result["changed_protected_object_pixels"] == 0


def test_atomic_directory_publish_refuses_clobber(tmp_path: Path) -> None:
    staging = tmp_path / ".stage"
    staging.mkdir()
    (staging / "payload").write_bytes(b"fresh")
    final = tmp_path / "final"
    final.mkdir()
    (final / "payload").write_bytes(b"existing")

    with pytest.raises(fresh.CleanBaselineError, match="no-clobber"):
        fresh._publish_directory_no_replace(staging, final)
    assert (staging / "payload").read_bytes() == b"fresh"
    assert (final / "payload").read_bytes() == b"existing"


def test_atomic_directory_publish_success(tmp_path: Path) -> None:
    staging = tmp_path / ".stage"
    staging.mkdir()
    (staging / "payload").write_bytes(b"fresh")
    final = tmp_path / "final"

    fresh._publish_directory_no_replace(staging, final)
    assert not staging.exists()
    assert (final / "payload").read_bytes() == b"fresh"


def test_formal_failure_publishes_schema_valid_grade_c(tmp_path: Path) -> None:
    output = tmp_path / "clean"
    spec_path = tmp_path / "spec.json"
    spec = {
        "run_id": fresh.RUN_ID,
        "task": "chips",
        "session": "get_potato_chips_0902_034",
        "frame_count": 293,
        "output_root": str(output),
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = fresh.publish_grade_c(spec_path, spec, RuntimeError("synthetic hard gate"))

    assert result["grade"] == "C"
    review = json.loads((output / "AGENT_REVIEW.json").read_text(encoding="utf-8"))
    assert review["grade"] == "C"
    assert review["downstream_authorized"] is False
    assert review["hard_gates"]["formal_clean_execution"] == "FAIL"
    receipt = fresh.validate_review(output / "AGENT_REVIEW.json")
    assert receipt["status"] == "PASS_BASELINE_AGENT_REVIEW"


def test_input_schema_freezes_non_generative_contract() -> None:
    schema = json.loads(fresh.INPUT_SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]["method_contract"]["properties"]
    assert properties["donor_scope"]["const"] == "SAME_SESSION_SELECTED_RGB_ONLY"
    assert properties["donor_rgb"]["const"] == "EXACT_DECODED_SOURCE_PIXEL_NO_INTERPOLATION"
    assert properties["unsupported_policy"]["const"] == "KEEP_RAW_BYTE_EXACT"
    assert properties["object_policy"]["const"] == "BYTE_EXACT_PROTECT_PREENCODE"
    assert properties["generative_fill"]["const"] is False
    assert fresh.METHOD_CONTRACT["telea"] is False
    assert fresh.METHOD_CONTRACT["lama"] is False


def test_clean_grade_rejects_large_unsupported_region() -> None:
    grade, authorized, fraction = fresh.clean_grade({"MASK": "B"}, 100, 17)
    assert grade == "C"
    assert authorized is False
    assert fraction == 0.17

    grade, authorized, fraction = fresh.clean_grade({"MASK": "B"}, 100, 50)
    assert grade == "B"
    assert authorized is True
    assert fraction == 0.5


def _object6d_arrays(frame_count: int, onset: int) -> dict[str, np.ndarray]:
    valid = np.arange(frame_count) >= onset
    observed = valid.copy()
    direct = np.full((frame_count, 2), np.nan, dtype=np.float64)
    analytic = np.full((frame_count, 2), np.nan, dtype=np.float64)
    direct[valid] = [0.5, 0.6]
    analytic[valid] = [0.49, 0.61]
    return {
        "valid": valid,
        "observed": observed,
        "visibility": valid.astype(np.float64),
        "physical_instance_id": np.where(valid, 0, -1),
        "direct_near_far": direct,
        "analytic_near_far": analytic,
        "transforms": np.repeat(np.eye(4)[None], frame_count, axis=0),
        "world_transforms": np.repeat(np.eye(4)[None], frame_count, axis=0),
    }


def test_object6d_explicit_identity_onset_is_accepted_without_backfill() -> None:
    arrays = _object6d_arrays(12, 5)
    fresh.validate_object6d_arrays(
        frame_count=12,
        numeric={
            "leading_unobserved_policy": "INVALID_UNTIL_VISUAL_IDENTITY_ONSET",
            "active_start_local_frame": 5,
            "invalid_pre_identity_frames": 5,
        },
        summary={"active_start_local_frame": 5, "invalid_pre_identity_frames": 5},
        **arrays,
    )


def test_object6d_identity_onset_rejects_a_backfilled_prefix_pose() -> None:
    arrays = _object6d_arrays(12, 5)
    arrays["valid"][2] = True
    arrays["physical_instance_id"][2] = 0
    arrays["analytic_near_far"][2] = [0.5, 0.6]
    with pytest.raises(fresh.CleanBaselineError, match="identity-onset"):
        fresh.validate_object6d_arrays(
            frame_count=12,
            numeric={
                "leading_unobserved_policy": "INVALID_UNTIL_VISUAL_IDENTITY_ONSET",
                "active_start_local_frame": 5,
                "invalid_pre_identity_frames": 5,
            },
            summary={"active_start_local_frame": 5, "invalid_pre_identity_frames": 5},
            **arrays,
        )
