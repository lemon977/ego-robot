from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipeline.mask_producer_sam2 import (
    _anatomy_support_masks,
    _apply_anatomy_bounds,
    _clamp_px,
    _combine_human_support,
    _contact_band_radius_px,
    _assign_contact_ownership,
    _final_anatomy_evidence,
    _load_json_nofollow,
    _project_cylinder_mask,
    _ray_to_border,
    _read_regular_nofollow,
    _select_forearm_candidate,
    _verify_regular_sha256_nofollow,
    _select_hand_candidate,
)
from pipeline.forearm_v8_sam_diagnostic import (
    _validate_diagnostic_output_root,
    build_parser as build_v8_diagnostic_parser,
)
from pipeline.forearm_v9_short_window import (
    _task_ref_path,
    _validate_task_frozen_cli_bindings,
)


def test_regular_input_is_sha_bound_and_symlink_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"manifest-bound")
    digest = hashlib.sha256(b"manifest-bound").hexdigest()
    assert _read_regular_nofollow(source, digest) == b"manifest-bound"
    _verify_regular_sha256_nofollow(source, digest)
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _verify_regular_sha256_nofollow(source, "0" * 64)
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _read_regular_nofollow(source, "0" * 64)
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _read_regular_nofollow(link, digest)


def test_v9_task_implementation_ref_is_live_sha_bound_and_nofollow(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "pipeline" / "producer.py"
    implementation.parent.mkdir()
    implementation.write_bytes(b"frozen-code")
    digest = hashlib.sha256(b"frozen-code").hexdigest()
    assert _task_ref_path(
        tmp_path,
        {"path": "pipeline/producer.py", "sha256": digest},
    ) == implementation
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _task_ref_path(
            tmp_path,
            {"path": "pipeline/producer.py", "sha256": "0" * 64},
        )
    symlink = tmp_path / "pipeline" / "producer-link.py"
    symlink.symlink_to(implementation)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _task_ref_path(
            tmp_path,
            {"path": "pipeline/producer-link.py", "sha256": digest},
        )


def test_v9_frozen_cli_binding_understands_named_sam_and_object_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    checkpoint = tmp_path / "sam.pt"
    config = tmp_path / "sam.yaml"
    object_pose = tmp_path / "object.npz"
    sam_package = tmp_path / "submodules" / "sam2"
    sam_package.mkdir(parents=True)
    args = SimpleNamespace(
        source_manifest=str(source),
        source_manifest_sha256="1" * 64,
        sam2_checkpoint=str(checkpoint),
        sam2_checkpoint_sha256="2" * 64,
        sam2_config_file=str(config),
        sam2_config_sha256="3" * 64,
        object_pose_npz=str(object_pose),
        object_pose_sha256="4" * 64,
        sam2_root=str(sam_package.parent),
    )
    frozen = {
        "source_manifest": {"path": str(source), "sha256": "1" * 64},
        "sam2": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": "2" * 64,
            "config_file": str(config),
            "config_sha256": "3" * 64,
            "implementation_root": str(sam_package),
        },
        "object6d_read_only_evidence": {
            "pose_path": str(object_pose),
            "pose_sha256": "4" * 64,
        },
    }
    _validate_task_frozen_cli_bindings(frozen, args)

    malformed = dict(frozen)
    malformed["sam2"] = {
        "path": str(checkpoint),
        "sha256": "2" * 64,
        "config_file": str(config),
        "config_sha256": "3" * 64,
        "implementation_root": str(sam_package),
    }
    with pytest.raises(RuntimeError, match="CLI frozen input"):
        _validate_task_frozen_cli_bindings(malformed, args)


def test_v8_diagnostic_context_is_sha_bound_and_nofollow(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    context.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(b"{}\n").hexdigest()
    assert _load_json_nofollow(context, digest) == {}
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _load_json_nofollow(context, "0" * 64)
    link = tmp_path / "context-link.json"
    link.symlink_to(context)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _load_json_nofollow(link, digest)


def test_v8_diagnostic_output_is_exactly_task_scoped(tmp_path: Path) -> None:
    valid = (
        tmp_path
        / "_run"
        / "g2_004_mask_clean_candidate_v8"
        / "sessions"
        / "grap_a_cap_004"
        / "diagnostic"
        / "true_sam"
    )
    assert _validate_diagnostic_output_root(
        valid,
        project_root=tmp_path,
        task_id="g2_004_mask_clean_candidate_v8",
        session_id="grap_a_cap_004",
    ) == valid.resolve()
    with pytest.raises(RuntimeError, match="task-scoped true_sam root"):
        _validate_diagnostic_output_root(
            tmp_path / "outside",
            project_root=tmp_path,
            task_id="g2_004_mask_clean_candidate_v8",
            session_id="grap_a_cap_004",
        )


def test_v8_diagnostic_parser_requires_context_digest() -> None:
    action = next(
        item
        for item in build_v8_diagnostic_parser()._actions
        if item.dest == "session_context_sha256"
    )
    assert action.required is True


def test_forearm_ray_uses_measured_direction_and_image_boundary() -> None:
    end = _ray_to_border(
        np.asarray([50.0, 40.0], np.float32),
        np.asarray([-1.0, 2.0], np.float32),
        width=100,
        height=80,
    )
    assert np.allclose(end, [30.5, 79.0], atol=1e-4)


def test_forearm_ray_rejects_degenerate_direction() -> None:
    with pytest.raises(RuntimeError, match="degenerate"):
        _ray_to_border(
            np.asarray([50.0, 40.0], np.float32),
            np.asarray([0.0, 0.0], np.float32),
            width=100,
            height=80,
        )


def test_morphology_radius_scales_with_hand_and_resolution() -> None:
    radius = _clamp_px(0.035 * 200.0, 960.0, 0.0015, 0.0125)
    doubled = _clamp_px(0.035 * 400.0, 1920.0, 0.0015, 0.0125)
    assert doubled == 2 * radius


def test_contact_band_radius_scales_with_projected_object_not_session() -> None:
    small = _contact_band_radius_px(
        10_000,
        object_scale_ratio=0.08,
        image_min_dimension=1000,
        low_resolution_ratio=0.001,
        high_resolution_ratio=0.02,
    )
    large = _contact_band_radius_px(
        40_000,
        object_scale_ratio=0.08,
        image_min_dimension=2000,
        low_resolution_ratio=0.001,
        high_resolution_ratio=0.02,
    )
    assert large == 2 * small


def test_object_overlap_becomes_u_not_silent_human_subtraction() -> None:
    shape = (24, 24)
    human = np.zeros(shape, dtype=bool)
    human[8:16, 5:14] = True
    visible_object = np.zeros(shape, dtype=bool)
    visible_object[8:16, 12:19] = True
    analytic = visible_object.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    h_core, o_core, uncertain, metrics = _assign_contact_ownership(
        human, visible_object, analytic, kernel
    )
    raw_overlap = human & visible_object
    assert raw_overlap.any()
    assert np.all(uncertain[raw_overlap])
    assert not np.any(h_core & o_core)
    assert not np.any(h_core & uncertain)
    assert not np.any(o_core & uncertain)
    assert metrics["pre_contact_object_overlap_pixels"] == float(raw_overlap.sum())
    assert metrics["post_contact_object_overlap_pixels"] == 0.0


def test_hand_candidate_prefers_keypoint_complete_mask() -> None:
    masks = np.zeros((3, 32, 32), dtype=bool)
    masks[0, 8:14, 8:14] = True
    masks[1, 6:21, 6:21] = True
    masks[2, :, :] = True
    scores = np.asarray([0.99, 0.75, 0.95], np.float32)
    points = np.asarray([[8, 8], [12, 12], [18, 18]], np.float32)
    selected, metrics = _select_hand_candidate(
        masks,
        scores,
        points,
        np.asarray([5, 5, 22, 22], np.float32),
        min_recall=0.8,
        max_box_area_ratio=1.3,
    )
    assert selected[18, 18]
    assert metrics["point_recall"] == 1.0
    assert metrics["held"] == 0.0
    assert selected.sum() < masks[2].sum()


def test_hand_candidate_below_gate_is_explicit_hold_not_hidden_fallback() -> None:
    masks = np.zeros((1, 16, 16), dtype=bool)
    masks[0, 4:8, 4:8] = True
    selected, metrics = _select_hand_candidate(
        masks,
        np.asarray([0.9], np.float32),
        np.asarray([[4, 4], [7, 7], [12, 12]], np.float32),
        np.asarray([3, 3, 13, 13], np.float32),
        min_recall=0.8,
        max_box_area_ratio=1.3,
    )
    assert not selected.any()
    assert metrics["held"] == 1.0
    assert metrics["held_candidate_pixels"] == 16.0


def test_forearm_candidate_below_gate_returns_zero_pixels() -> None:
    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 10:31, 10:15] = True
    corridor = np.zeros((32, 32), dtype=bool)
    corridor[:, 8:18] = True
    selected, metrics = _select_forearm_candidate(
        masks,
        np.asarray([0.9], np.float32),
        np.asarray([[29, 29]], np.float32),
        np.asarray([12, 12], np.float32),
        corridor,
        min_recall=1.0,
    )
    assert not selected.any()
    assert metrics["held"] == 1.0
    assert metrics["held_candidate_pixels"] > 0


def test_analytic_cylinder_projection_is_finite_and_centered() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 1.0
    intrinsics = np.asarray([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])
    mask, center = _project_cylinder_mask(
        transform,
        intrinsics,
        radius=0.1,
        height_m=0.2,
        width=128,
        height=96,
    )
    assert mask.shape == (96, 128)
    assert mask.any()
    assert np.allclose(center, [64.0, 48.0])


def test_forearm_candidate_keeps_only_prompt_connected_component() -> None:
    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 10:31, 10:15] = True
    masks[0, 2:6, 24:29] = True
    corridor = np.zeros((32, 32), dtype=bool)
    corridor[:, 8:18] = True
    points = np.asarray([[12, 12], [12, 20], [12, 28]], np.float32)
    selected, metrics = _select_forearm_candidate(
        masks,
        np.asarray([0.9], np.float32),
        points[1:],
        points[0],
        corridor,
    )
    assert selected[20, 12]
    assert not selected[3, 26]
    assert metrics["raw_component_count_before_regularization"] == 2.0
    assert metrics["retained_prompt_component_count"] == 1.0
    assert metrics["rejected_disconnected_component_count"] == 1.0
    assert metrics["held"] == 0.0


def test_forearm_candidate_prior_corridor_never_hard_crops_raw_sam() -> None:
    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 8:, 8:17] = True
    corridor = np.zeros((32, 32), dtype=bool)
    corridor[:, 11:14] = True
    points = np.asarray([[12, 10], [12, 20], [12, 28]], np.float32)
    debug: dict[str, np.ndarray] = {}

    selected, metrics = _select_forearm_candidate(
        masks,
        np.asarray([0.9], np.float32),
        points[1:],
        points[0],
        corridor,
        debug_masks=debug,
    )

    assert selected[20, 8]
    assert selected[20, 16]
    assert np.array_equal(debug["raw_selected_forearm_sam"], masks[0])
    assert np.array_equal(debug["regularized_forearm_sam"], selected)
    assert metrics["boundary_connection"] == 1.0
    assert 0.0 < metrics["corridor_precision"] < 1.0
    assert metrics["held"] == 0.0


def test_anatomy_support_is_wrist_normalized_and_reaches_boundary() -> None:
    points = np.asarray(
        [[46, 40], [54, 40], [58, 32], [50, 24], [42, 32], [50, 50]],
        np.float32,
    )
    hand_envelope, corridor, metrics = _anatomy_support_masks(
        points,
        wrist=np.asarray([50, 50], np.float32),
        palm=np.asarray([50, 40], np.float32),
        wrist_width=8.0,
        forearm_width_ratio=2.0,
        width=100,
        height=80,
    )
    assert hand_envelope[40, 50]
    assert not hand_envelope[5, 95]
    assert corridor[-1].any()
    assert corridor[50, 50]
    assert metrics["forearm_half_width_px"] == 8.0
    assert metrics["envelope_radius_px"] == 8.0
    assert metrics["normalization"] == "measured_wrist_width_times_k_forearm"


def test_anatomy_bounds_reject_table_blob_and_hold_without_boundary_evidence() -> None:
    shape = (64, 64)
    hand = np.zeros(shape, dtype=bool)
    hand[20:30, 20:30] = True
    hand[2:8, 52:60] = True  # disconnected table/background false positive
    wrist = np.zeros(shape, dtype=bool)
    wrist[28:36, 22:28] = True
    wrist_region = np.zeros(shape, dtype=bool)
    wrist_region[24:38, 18:34] = True
    forearm = np.zeros(shape, dtype=bool)
    forearm[32:45, 21:29] = True
    envelope = np.zeros(shape, dtype=bool)
    envelope[14:38, 14:38] = True
    corridor = np.zeros(shape, dtype=bool)
    corridor[30:, 20:30] = True

    bounded, metrics = _apply_anatomy_bounds(
        hand,
        wrist,
        wrist_region,
        forearm,
        envelope,
        corridor,
        np.asarray([25, 30], np.float32),
        np.asarray([25, 22], np.float32),
        8.0,
    )

    assert bounded[24, 24]
    assert not bounded[4, 55]
    assert not bounded[-1, 25]
    assert metrics["rejected_non_anatomic_hand_pixels"] == 48.0
    assert metrics["analytic_corridor_completion_pixels"] == 0.0
    assert metrics["bridge_pixels"] == 0.0
    assert metrics["held"] == 1.0


def test_anatomy_bounds_keep_only_wrist_component_with_observed_boundary() -> None:
    shape = (64, 64)
    hand = np.zeros(shape, dtype=bool)
    hand[20:30, 20:30] = True
    wrist = np.zeros(shape, dtype=bool)
    wrist[28:36, 22:28] = True
    wrist_region = np.zeros(shape, dtype=bool)
    wrist_region[24:38, 18:34] = True
    forearm = np.zeros(shape, dtype=bool)
    forearm[32:, 21:29] = True
    forearm[42:, 33:38] = True  # separate table/background component
    envelope = np.zeros(shape, dtype=bool)
    envelope[14:38, 14:38] = True
    corridor = np.zeros(shape, dtype=bool)
    corridor[28:, 18:40] = True

    bounded, metrics = _apply_anatomy_bounds(
        hand,
        wrist,
        wrist_region,
        forearm,
        envelope,
        corridor,
        np.asarray([25, 30], np.float32),
        np.asarray([25, 22], np.float32),
        8.0,
    )

    assert bounded[-1, 25]
    assert not bounded[-1, 35]
    assert metrics["selected_component_boundary_connected"] == 1.0
    assert metrics["selected_component_touched_edges"] == ["bottom"]
    assert metrics["rejected_disconnected_forearm_pixels"] > 0
    assert metrics["analytic_corridor_completion_pixels"] == 0.0
    assert metrics["held"] == 0.0


@pytest.mark.parametrize(
    ("prompt_recall", "observed_boundary"),
    [(0.5, 1.0), (1.0, 0.0)],
)
def test_final_forearm_support_requires_recall_and_observed_boundary(
    prompt_recall: float, observed_boundary: float
) -> None:
    shape = (32, 32)
    human = np.zeros(shape, dtype=bool)
    human[14:, 4:8] = True
    human[14:, 24:28] = True
    empty = np.zeros(shape, dtype=bool)

    def hand(x: float) -> dict[str, object]:
        return {
            "joint_names": ["wrist", "palm_center", "index_proximal"],
            "keypoints_2d": [[x, 20.0], [x, 16.0], [x + 1.0, 15.0]],
            "joint_valid": [True, True, True],
            "joint_in_image": [True, True, True],
        }

    metadata = {
        "entities": {"hands": {"left": hand(6.0), "right": hand(26.0)}}
    }
    metrics = {
        "left": {
            "forearm_candidate": {"prompt_recall": prompt_recall},
            "anatomy_bounds": {
                "selected_component_boundary_connected": observed_boundary
            },
            "independent_hold": 0.0,
        },
        "right": {
            "forearm_candidate": {"prompt_recall": 1.0},
            "anatomy_bounds": {
                "selected_component_boundary_connected": 1.0
            },
            "independent_hold": 0.0,
        },
    }

    supported, evidence = _final_anatomy_evidence(
        metadata,
        human,
        empty,
        empty,
        metrics,
        hand_recall_gate=0.8,
        forearm_recall_gate=0.6,
    )

    assert evidence["left"]["wrist_component_crop_connected"] == 1.0
    assert evidence["left"]["forearm_supported"] == 0.0
    assert evidence["left"]["side_supported"] == 0.0
    assert evidence["right"]["side_supported"] == 1.0
    assert supported == 0.0


def test_no_evaluable_keypoints_is_nan_and_hold() -> None:
    shape = (32, 32)
    human = np.zeros(shape, dtype=bool)
    human[14:, 4:8] = True
    human[14:, 24:28] = True
    protected = np.ones(shape, dtype=bool)

    def hand(x: float) -> dict[str, object]:
        return {
            "joint_names": ["wrist", "palm_center", "index_proximal"],
            "keypoints_2d": [[x, 20.0], [x, 16.0], [x + 1.0, 15.0]],
            "joint_valid": [True, True, True],
            "joint_in_image": [True, True, True],
        }

    metadata = {"entities": {"hands": {"left": hand(6.0), "right": hand(26.0)}}}
    side_metrics = {
        side: {
            "forearm_candidate": {"prompt_recall": 1.0},
            "anatomy_bounds": {"selected_component_boundary_connected": 1.0},
            "independent_hold": 0.0,
        }
        for side in ("left", "right")
    }
    supported, evidence = _final_anatomy_evidence(
        metadata,
        human,
        protected,
        np.zeros(shape, dtype=bool),
        side_metrics,
        hand_recall_gate=0.8,
        forearm_recall_gate=0.6,
    )
    assert supported == 0.0
    assert np.isnan(evidence["left"]["contact_aware_keypoint_recall"])
    assert evidence["left"]["evaluable_keypoint_count"] == 0.0
    assert evidence["left"]["side_supported"] == 0.0


def test_human_support_cannot_be_rescued_by_temporal_only() -> None:
    assert _combine_human_support(0.0, 1.0) == 0.0
    assert _combine_human_support(1.0, 0.0) == 0.0
    assert _combine_human_support(1.0, 1.0) == 1.0


def test_twenty_pixel_shift_stays_unsupported_despite_temporal_support() -> None:
    shape = (96, 96)
    good = np.zeros(shape, dtype=bool)
    good[30:, 18:23] = True
    good[30:, 68:73] = True
    shifted = np.zeros_like(good)
    shifted[:, 20:] = good[:, :-20]
    empty = np.zeros(shape, dtype=bool)

    def hand(x: float) -> dict[str, object]:
        return {
            "joint_names": ["wrist", "palm_center", "index_proximal"],
            "keypoints_2d": [[x, 50.0], [x, 42.0], [x + 1.0, 38.0]],
            "joint_valid": [True, True, True],
            "joint_in_image": [True, True, True],
        }

    metadata = {"entities": {"hands": {"left": hand(20.0), "right": hand(70.0)}}}
    side_metrics = {
        side: {
            "forearm_candidate": {"prompt_recall": 1.0},
            "anatomy_bounds": {"selected_component_boundary_connected": 1.0},
            "independent_hold": 0.0,
        }
        for side in ("left", "right")
    }
    good_supported, _ = _final_anatomy_evidence(
        metadata,
        good,
        empty,
        empty,
        side_metrics,
        hand_recall_gate=0.8,
        forearm_recall_gate=0.6,
    )
    shifted_supported, shifted_evidence = _final_anatomy_evidence(
        metadata,
        shifted,
        empty,
        empty,
        side_metrics,
        hand_recall_gate=0.8,
        forearm_recall_gate=0.6,
    )
    assert good_supported == 1.0
    assert shifted_supported == 0.0
    assert shifted_evidence["left"]["contact_aware_keypoint_recall"] == 0.0
    assert _combine_human_support(shifted_supported, 1.0) == 0.0
