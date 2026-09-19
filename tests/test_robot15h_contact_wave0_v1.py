from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import run_0915_robot15h_contact_wave0_v1 as subject


def tactile_side(
    *, active_finger: int | None = None, offline_valid: bool = True,
    offset_ms: float = 4.0,
) -> dict:
    grid = np.zeros((5, 4, 8), np.int16)
    active = np.zeros_like(grid, bool)
    valid = np.ones_like(grid, bool)
    if active_finger is not None:
        grid[active_finger, 1, 2] = 7
        active[active_finger, 1, 2] = True
    return {
        "schema_version": subject.TACTILE_SCHEMA,
        "finger_grid_5x4x8": grid.tolist(),
        "active_mask_5x4x8": active.tolist(),
        "valid_mask_5x4x8": valid.tolist(),
        "offline_source_valid": offline_valid,
        "offline_source_offset_ms": offset_ms,
    }


def tactile_record(**right: object) -> dict:
    return {
        "left": tactile_side(),
        "right": tactile_side(**right),
    }


def interaction_row(
    *, frame: int = 92, hand: str = "right", finger: str = "index",
    object_id: str = "playing_card_02", distance_m: float = 0.004,
    inside_patch: bool = True, overlap: bool = True,
) -> dict:
    return {
        "frame_id": frame,
        "timestamp_s": frame / 30.0,
        "hand_id": hand,
        "finger_id": finger,
        "object_id": object_id,
        "pair_key": f"{hand}:{finger}:{object_id}",
        "finger_associated_visible_surface_point": {
            "status": "OBSERVED_VISIBLE_SURFACE",
            "surface_role": "skin",
            "source_pixel_uv": [640.0, 480.0],
            "association_semantics": (
                "DIRECT_OBSERVED_HAWOR_2D_PLUS_SAM31_HAND_ROLE_TO_LOCAL_STEREO_"
                "VISIBLE_SURFACE;NOT_ANATOMICAL_TIP_GROUND_TRUTH"
            ),
            "association_quality": {
                "local_depth_robust_sigma_m": 0.001,
                "lr_residual_median_px": 0.25,
            },
        },
        "object_visibility": "DIRECT_VISIBLE",
        "object_mask_state": "tracked",
        "two_d_adjacency": {
            "projected_finger_inside_object_mask": overlap,
            "adjacent_within_20px": overlap,
            "distance_to_visible_object_mask_px": 0.0 if overlap else 30.0,
        },
        "occlusion_evidence": "POSSIBLE_HAND_OBJECT_OCCLUSION_OR_CONTACT_BOUNDARY",
        "metric_interaction_state": "DIRECT_VISIBLE_SURFACE_RELATION",
        "metric": {
            "finite_patch_distance_m": distance_m,
            "inside_visible_patch": inside_patch,
            "object_plane_residual_p90_m": 0.001,
        },
        "short_gap_inferred": False,
        "contact_authority": "NONE",
        "relative_motion": {"status": "UNKNOWN"},
    }


def tactile_evidence(*, active: bool = True, offset_ms: float = 4.0) -> dict:
    return subject.tactile_finger_evidence(
        tactile_record(active_finger=1 if active else None, offset_ms=offset_ms),
        hand="right", finger="index",
    )


def test_specific_finger_grid_is_used_not_whole_hand_activity() -> None:
    thumb_only = tactile_record(active_finger=0)
    index = subject.tactile_finger_evidence(thumb_only, hand="right", finger="index")
    thumb = subject.tactile_finger_evidence(thumb_only, hand="right", finger="thumb")
    assert index["grid_index"] == 1
    assert index["finger_specific_tactile_active"] is False
    assert index["active_valid_nonzero_taxel_count"] == 0
    assert thumb["finger_specific_tactile_active"] is True
    assert thumb["active_valid_nonzero_taxel_count"] == 1


def test_invalid_or_out_of_window_tactile_cannot_support_evidence() -> None:
    invalid = subject.tactile_finger_evidence(
        tactile_record(active_finger=1, offline_valid=False),
        hand="right", finger="index",
    )
    late = subject.tactile_finger_evidence(
        tactile_record(active_finger=1, offset_ms=40.001),
        hand="right", finger="index",
    )
    assert invalid["finger_specific_tactile_active"] is False
    assert late["time_aligned_within_40ms"] is False
    assert late["finger_specific_tactile_active"] is False


def test_uncertainty_never_expands_fixed_five_mm_gate() -> None:
    row = interaction_row(distance_m=0.006)
    row["finger_associated_visible_surface_point"]["association_quality"][
        "local_depth_robust_sigma_m"
    ] = 0.100
    value = subject.evaluate_pair(
        row, tactile_evidence(), external_metric_authority=True,
    )
    assert value["geometric_proximity_within_fixed_5mm"] is False
    assert value["strict_contact_admitted"] is False
    assert "NOT_WITHIN_FIXED_5MM_FINITE_VISIBLE_PATCH" in value["blockers"]
    assert value["uncertainty"]["policy"] == (
        "UNCERTAINTY_NEVER_EXPANDS_THE_FIXED_5MM_DISTANCE_GATE"
    )


def test_inside_finite_patch_and_external_authority_are_mandatory() -> None:
    outside = subject.evaluate_pair(
        interaction_row(distance_m=0.001, inside_patch=False),
        tactile_evidence(), external_metric_authority=True,
    )
    assert outside["geometric_proximity_within_fixed_5mm"] is False
    closed = subject.evaluate_pair(
        interaction_row(distance_m=0.001), tactile_evidence(),
        external_metric_authority=False,
    )
    assert closed["strict_contact_admitted"] is False
    assert closed["r1_e_admitted"] is False
    assert "EXTERNAL_METRIC_AUTHORITY_UNVERIFIED" in closed["blockers"]
    assert closed["uncertainty"]["pixel_registration_uncertainty"] == "UNBOUND"


def test_authority_validator_accepts_only_explicitly_closed_inputs() -> None:
    value = subject.validate_external_authority(
        {"strict_metric_contact_authorized": False},
        {"contact_authority": "NONE"},
    )
    assert value["external_metric_authority"] is False
    assert value["r1_e_authorized"] is False
    with pytest.raises(RuntimeError, match="authority escalation"):
        subject.validate_external_authority(
            {"strict_metric_contact_authorized": True},
            {"contact_authority": "NONE"},
        )


def test_weak_hypothesis_requires_exact_scope_overlap_observability_and_tactile() -> None:
    row = interaction_row()
    supported = subject.weak_hypothesis_row(
        row, tactile_evidence(), object_plane_observed=True,
    )
    assert supported is not None
    assert supported["status"] == "HYPOTHESIS_ONLY"
    assert supported["training_eligible"] is False
    assert supported["contact_ground_truth"] is False
    assert supported["strict_contact_admitted"] is False
    assert supported["metric_contact"] is False

    assert subject.weak_hypothesis_row(
        interaction_row(frame=91), tactile_evidence(), object_plane_observed=True,
    ) is None
    assert subject.weak_hypothesis_row(
        interaction_row(finger="middle"), tactile_evidence(), object_plane_observed=True,
    ) is None
    assert subject.weak_hypothesis_row(
        interaction_row(overlap=False), tactile_evidence(), object_plane_observed=True,
    ) is None
    unknown = interaction_row()
    unknown["object_visibility"] = "UNKNOWN"
    assert subject.weak_hypothesis_row(
        unknown, tactile_evidence(), object_plane_observed=True,
    ) is None
    assert subject.weak_hypothesis_row(
        row, tactile_evidence(active=False), object_plane_observed=True,
    ) is None
    assert subject.weak_hypothesis_row(
        row, tactile_evidence(), object_plane_observed=False,
    ) is None

    occluded = interaction_row()
    occluded["finger_associated_visible_surface_point"].update({
        "status": "UNKNOWN",
        "reason": "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED",
    })
    supported = subject.weak_hypothesis_row(
        occluded, tactile_evidence(), object_plane_observed=True,
    )
    assert supported is not None
    assert supported["visual_evidence"]["finger_visible_surface"] == "ABSENT_OCCLUDED"


def test_no_contact_control_is_explicitly_not_ground_truth() -> None:
    row = interaction_row(frame=40, overlap=False)
    control = subject.no_contact_control_row(
        row, tactile_evidence(active=False), object_plane_observed=True,
    )
    assert control is not None
    assert control["status"] == "NO_CONTACT"
    assert control["training_eligible"] is False
    assert control["contact_ground_truth"] is False
    assert "NOT_CONTACT_GROUND_TRUTH" in control["control_semantics"]
    assert subject.no_contact_control_row(
        row, tactile_evidence(active=True), object_plane_observed=True,
    ) is None


def test_fixed_interaction_axes_reject_duplicates_and_gaps() -> None:
    rows = []
    for frame in range(2):
        for hand in subject.HANDS:
            for finger in subject.FINGERS:
                for object_id in subject.OBJECTS:
                    rows.append(interaction_row(
                        frame=frame, hand=hand, finger=finger, object_id=object_id,
                    ))
    doc = {
        "frame_count": 2,
        "pair_denominator": 60,
        "strict_metric_contact_authorized": False,
        "rows": rows,
    }
    assert len(subject.interaction_rows(doc, 2)) == 60
    doc["rows"][-1] = dict(doc["rows"][0])
    with pytest.raises(RuntimeError, match="duplicate or inferred-gap"):
        subject.interaction_rows(doc, 2)


def test_physical_left_is_second_sbs_eye_and_resize_only() -> None:
    frame = np.zeros((4, 8, 3), np.uint8)
    frame[:, :4] = (10, 20, 30)
    frame[:, 4:] = (90, 100, 110)
    left = subject.physical_left(frame)
    assert left.shape == (960, 1280, 3)
    np.testing.assert_array_equal(left[100, 100], [90, 100, 110])
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "cv2.remap" not in source
    assert "equiDis62" not in source


def test_review_video_full_decodes_native_frame_denominator(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 8),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.zeros((8, 16, 3), np.uint8)
        frame[:, :8] = 20
        frame[:, 8:] = 100 + index
        writer.write(frame)
    writer.release()
    destination = tmp_path / "review.mp4"
    review = subject.render_review(
        source, destination, 3, session_id="session", status="BLOCKED_UPSTREAM",
        evidence_rows=[], hypothesis_rows=[],
    )
    assert review["full_decode"] is True
    assert review["frame_count"] == 3
    assert review["input_operation"] == "SOURCEINDEX1_CROP_THEN_RESIZE_ONLY"


def test_writer_claim_binds_live_pid_epoch_fence_and_signature(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    fence_sha = hashlib.sha256(b"contact-w0-fencing-token").hexdigest()
    claim = {
        "schema_version": "0915-robot15h-contact-writer-claim-v1",
        "task_id": subject.TASK_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "unique_write_root": str(tmp_path.resolve()),
        "executor_epoch": 3,
        "run_signature_sha256": "a" * 64,
        "fencing_token_sha256": fence_sha,
        "pid": os.getpid(),
        "proc_start_ticks": subject.process_start_ticks(os.getpid()),
    }
    path = tmp_path / "CLAIM.json"
    path.write_text(json.dumps(claim), encoding="utf-8")
    assert subject.validate_writer_claim(
        path, signature_sha="a" * 64, executor_epoch=3, fencing_sha=fence_sha,
    )["pid"] == os.getpid()
    claim["executor_epoch"] = 4
    path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(RuntimeError, match="claim/fence mismatch"):
        subject.validate_writer_claim(
            path, signature_sha="a" * 64, executor_epoch=3, fencing_sha=fence_sha,
        )


def test_runner_declares_atomic_sessions_and_required_root_outputs() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    session_source = inspect.getsource(subject.run_session)
    main_source = inspect.getsource(subject.main)
    assert "session-staging" in session_source
    assert "os.replace(stage, final)" in session_source
    for name in (
        "CLAIM.json", "RUN_SIGNATURE.json", "CONTACT_EVIDENCE_LEDGER.json",
        "CONTACT_HYPOTHESIS_LEDGER.json", "BATCH_RESULT.json", "METRICS.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ):
        assert name in main_source
    assert "finger_grid_5x4x8" in source
    assert "wire_values_369" not in source
    assert '"strict_metric_contact_authorized": False' in source
    assert '"r1_e_authorized": False' in source
    assert '"training_eligible": False' in source
    assert '"gpu_used": False' in source
    assert "archive/" not in source


def test_expected_w0_terminal_accounting() -> None:
    rows = [
        {"status": "BLOCKED_UPSTREAM_HAND_ROLE"},
        {"status": "COMPLETED_DIAGNOSTIC_NO_STRICT_CONTACT", "hypothesis_rows": 3},
        {"status": "BLOCKED_UPSTREAM_OBJECT6D"},
        {"status": "BLOCKED_UPSTREAM_OBJECT6D"},
    ]
    counts = subject.summarize(rows)
    assert counts == {
        "total": 4,
        "diagnostic_completed": 1,
        "blocked_upstream_hand_role": 1,
        "blocked_upstream_object6d": 2,
        "failed_runtime": 0,
        "strict_contact_windows": 0,
        "r1_e_windows": 0,
        "hypothesis_rows": 3,
        "unrun": 0,
    }
