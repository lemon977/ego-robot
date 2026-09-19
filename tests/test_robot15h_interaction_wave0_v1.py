from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import run_0915_robot15h_interaction_wave0_v1 as subject


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_mask(path: Path, masks: np.ndarray) -> subject.PackedMask:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, height, width = masks.shape
    np.savez_compressed(
        path,
        packed=np.packbits(masks.reshape(frame_count, -1), axis=1, bitorder="big"),
        frame_count=np.asarray(frame_count, np.int32),
        height=np.asarray(height, np.int32),
        width=np.asarray(width, np.int32),
        bitorder=np.asarray("big"),
    )
    return subject.PackedMask(path, frame_count)


def depth_inputs() -> dict[str, np.ndarray]:
    return {
        "depth_m": np.full(subject.DEPTH_SHAPE, 0.5, np.float32),
        "depth_valid": np.ones(subject.DEPTH_SHAPE, bool),
        "lr_consistent": np.ones(subject.DEPTH_SHAPE, bool),
        "lr_residual_px": np.full(subject.DEPTH_SHAPE, 0.25, np.float32),
        "intrinsics": np.asarray([
            [300.0, 0.0, 319.5], [0.0, 300.0, 239.5], [0.0, 0.0, 1.0],
        ]),
        "hand_mask": np.ones(subject.MASK_SHAPE, bool),
        "hand_semantic_admitted": True,
        "hand_consumer_allowed": True,
        "object_union": np.zeros(subject.MASK_SHAPE, bool),
    }


def test_runner_is_w0_cpu_only_and_never_consumes_hawor_absolute_z() -> None:
    assert subject.TASK_ID == "0915_robot15h_interaction_occlusion_v1"
    assert subject.IMAGE_DOMAIN == "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
    assert subject.DEPTH_REFERENCE == "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
    assert subject.OBJECTS == ("playing_card_00", "playing_card_01", "playing_card_02")
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert 'hawor["joints_3d_camera"]' not in source
    assert 'hawor["joints_3d_world"]' not in source
    assert "48mm" in source
    assert '"constant_48mm_bias_subtracted": False' in source
    assert '"contact_fitting_alignment_used": False' in source
    assert '"short_gap_inferred_consumed": False' in source
    assert '"removal_consumed": False' in source
    assert '"gpu_used": False' in source
    assert "cv2.remap" not in source
    assert "equiDis62" not in source
    assert "archive/" not in source


def test_only_direct_observed_hawor_projection_can_seed_metric_sample() -> None:
    inputs = depth_inputs()
    blocked = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5],
        direct_observed=False,
        provenance="SHORT_GAP_INFERRED",
        associated_hand="left",
        associated_finger="index",
        **inputs,
    )
    assert blocked["status"] == "UNKNOWN"
    assert blocked["reason"] == "HAWOR_NOT_DIRECT_OBSERVED"
    assert blocked["fingertip_surface_observation"] is False

    observed = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5],
        direct_observed=True,
        provenance="OBSERVED",
        associated_hand="left",
        associated_finger="index",
        **inputs,
    )
    assert observed["status"] == "OBSERVED_VISIBLE_SURFACE"
    assert observed["surface_role"] == "unknown"
    assert observed["fingertip_surface_observation"] is False
    assert observed["depth_quality"]["external_metric_accuracy"] == "UNVERIFIED"
    np.testing.assert_allclose(observed["surface_point_xyz"][2], 0.5)


def test_object_pixels_and_depth_discontinuity_fail_closed() -> None:
    inputs = depth_inputs()
    inputs["object_union"][470:492, 630:652] = True
    value = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5], direct_observed=True, provenance="OBSERVED",
        associated_hand="right", associated_finger="thumb", **inputs,
    )
    assert value["status"] == "UNKNOWN"
    assert value["reason"] == "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED"

    inputs = depth_inputs()
    yy, xx = np.indices(subject.DEPTH_SHAPE)
    inputs["depth_m"] = np.where((xx + yy) % 2, 0.45, 0.58).astype(np.float32)
    value = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5], direct_observed=True, provenance="OBSERVED",
        associated_hand="right", associated_finger="thumb", **inputs,
    )
    assert value["status"] == "UNKNOWN"
    assert value["reason"] == "LOCAL_DEPTH_DISCONTINUITY"


def test_sam_hand_consumer_state_and_local_pixel_purity_are_hard_gates() -> None:
    inputs = depth_inputs()
    inputs["hand_consumer_allowed"] = False
    value = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5], direct_observed=True, provenance="OBSERVED",
        associated_hand="left", associated_finger="index", **inputs,
    )
    assert value["reason"] == "HAND_ROLE_NOT_CONSUMER_ADMITTED"

    inputs = depth_inputs()
    inputs["hand_semantic_admitted"] = False
    value = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5], direct_observed=True, provenance="OBSERVED",
        associated_hand="left", associated_finger="index", **inputs,
    )
    assert value["reason"] == "HAND_ROLE_FRAME_NOT_SEMANTIC_ADMITTED"

    inputs = depth_inputs()
    inputs["hand_mask"][:] = False
    value = subject.sample_finger_associated_visible_surface(
        source_pixel_uv=[640.5, 480.5], direct_observed=True, provenance="OBSERVED",
        associated_hand="left", associated_finger="index", **inputs,
    )
    assert value["reason"] == "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED"
    assert value["association_quality"]["hand_mask_purity"] == 0.0


def object_frame(*, offset_x: float = 0.0) -> dict:
    center = np.asarray([offset_x, 0.0, 0.5])
    boundary = [
        (center + np.asarray([x, y, 0.0])).tolist()
        for x, y in [(-0.03, -0.04), (0.03, -0.04), (0.03, 0.04), (-0.03, 0.04)]
    ]
    return {
        "center_xyz": {
            "observability": "OBSERVABLE_VISIBLE_SURFACE_CENTROID",
            "estimate": {"xyz_m": center.tolist()},
        },
        "plane_normal": {
            "observability": "OBSERVABLE_DIRECT_VISIBLE_PLANE",
            "estimate": {"unit_xyz": [0.0, 0.0, -1.0]},
            "residual": {"p90_plane_distance_m": 0.001},
        },
        "inplane_rotation": {
            "observability": "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS",
            "estimate": {"axis_unit_xyz": [1.0, 0.0, 0.0]},
        },
        "finite_visible_patch": {
            "observability": "OBSERVABLE_DIRECT_VISIBLE_FINITE_SUPPORT",
            "estimate": {"boundary_xyz_m": boundary},
            "residual": {
                "connected_component_count": 1,
                "largest_component_fraction": 1.0,
            },
        },
        "registered_valid_depth_fraction": 0.9,
        "mask_pixel_count": 100,
        "visibility_state": "DIRECT_VISIBLE",
        "mask_state": "tracked",
    }


def visible_sample(xyz: list[float], uv: list[float] | None = None) -> dict:
    return {
        "status": "OBSERVED_VISIBLE_SURFACE",
        "reason": None,
        "surface_point_xyz": xyz,
        "source_pixel_uv": uv or [640.0, 480.0],
        "associated_hand": "left",
        "associated_finger": "index",
        "surface_role": "unknown",
        "fingertip_surface_observation": False,
    }


def test_pair_uses_finite_patch_not_infinite_plane() -> None:
    mask = np.zeros(subject.MASK_SHAPE, bool)
    row = subject.pair_observation(
        frame_id=0,
        timestamp_s=0.0,
        hand="left",
        finger="index",
        object_id="playing_card_00",
        sample=visible_sample([0.10, 0.0, 0.5]),
        object_frame=object_frame(),
        object_mask=mask,
    )
    assert row["metric_interaction_state"] == "DIRECT_VISIBLE_SURFACE_RELATION"
    assert row["metric"]["inside_visible_patch"] is False
    assert row["metric"]["object_plane_signed_distance_m"] == pytest.approx(0.0)
    assert row["metric"]["finite_patch_distance_m"] > 0.05
    assert row["contact_authority"] == "NONE"


def test_unknown_object_patch_never_emits_metric_relation() -> None:
    frame = object_frame()
    frame["finite_visible_patch"] = {
        "observability": "UNOBSERVABLE", "estimate": None,
    }
    row = subject.pair_observation(
        frame_id=3, timestamp_s=0.1, hand="right", finger="middle",
        object_id="playing_card_02", sample=visible_sample([0.0, 0.0, 0.5]),
        object_frame=frame, object_mask=np.zeros(subject.MASK_SHAPE, bool),
    )
    assert row["metric"] is None
    assert row["metric_interaction_state"] == "UNKNOWN"
    assert row["metric_blocker"] == "OBJECT_FINITE_VISIBLE_PATCH_UNKNOWN"


def test_fragmented_object_support_is_not_convexified_into_metric_patch() -> None:
    frame = object_frame()
    frame["finite_visible_patch"]["residual"] = {
        "connected_component_count": 3,
        "largest_component_fraction": subject.MIN_LARGEST_COMPONENT_FRACTION - 0.01,
    }
    assert subject.finite_patch_from_frame(frame) is None
    row = subject.pair_observation(
        frame_id=2, timestamp_s=2 / 30, hand="left", finger="thumb",
        object_id="playing_card_00", sample=visible_sample([0.0, 0.0, 0.5]),
        object_frame=frame, object_mask=np.zeros(subject.MASK_SHAPE, bool),
    )
    assert row["metric"] is None
    assert row["metric_blocker"] == "OBJECT_FINITE_VISIBLE_PATCH_UNKNOWN"


def test_relative_motion_does_not_bridge_frame_gap() -> None:
    def row(frame: int, distance: float) -> dict:
        return {
            "pair_key": "left:index:playing_card_00",
            "frame_id": frame,
            "metric": {"finite_patch_distance_m": distance},
        }

    rows = [row(0, 0.02), row(1, 0.015), row(3, 0.001)]
    subject.add_relative_motion_diagnostics(rows)
    assert rows[1]["relative_motion"]["approach_supported"] is True
    assert rows[2]["relative_motion"]["status"] == "UNKNOWN"

    rows = [row(0, 0.02), {**row(1, 0.0), "metric": None}, row(2, 0.001)]
    subject.add_relative_motion_diagnostics(rows)
    assert rows[2]["relative_motion"]["status"] == "UNKNOWN"


def test_chips_propagate_first_object6d_blocker_without_artifact() -> None:
    item = {
        "session_id": "get_potato_chips_0915_007",
        "task": "potato_chips",
        "source_group": "group-a",
        "frame_count": 378,
    }
    row = subject.blocked_row(
        item, "NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK",
    )
    assert row["status"] == "BLOCKED_UPSTREAM"
    assert row["first_blocker"] == "NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK"
    assert row["interaction_attempted"] is False
    assert row["interaction_artifact_emitted"] is False


def test_031_like_session_without_any_consumer_hand_is_blocked() -> None:
    batch_row = {
        "side_results": [
            {"side": 0, "role": "left_hand", "consumer_allowed": False},
            {"side": 1, "role": "right_hand", "consumer_allowed": False},
        ],
    }
    assert subject.consumer_allowed_hand_sides(batch_row) == []
    item = {
        "session_id": "play_cards_0915_031", "task": "playing_cards",
        "source_group": "group-031", "frame_count": 149,
    }
    row = subject.blocked_row(
        item, "NO_CONSUMER_ADMITTED_SAM31_HAND_ROLE",
        status="BLOCKED_UPSTREAM_HAND_ROLE",
    )
    assert row["status"] == "BLOCKED_UPSTREAM_HAND_ROLE"
    assert row["interaction_attempted"] is False


def test_fragment_gate_is_frozen_at_098() -> None:
    assert subject.MIN_LARGEST_COMPONENT_FRACTION == 0.98


def test_writer_claim_binds_live_pid_epoch_fence_and_signature(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    fencing_sha = hashlib.sha256(b"bounded-interaction-fence").hexdigest()
    claim = {
        "schema_version": "0915-robot15h-interaction-writer-claim-v1",
        "task_id": subject.TASK_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "unique_write_root": str(tmp_path.resolve()),
        "executor_epoch": 2,
        "run_signature_sha256": "a" * 64,
        "fencing_token_sha256": fencing_sha,
        "pid": os.getpid(),
        "proc_start_ticks": subject.process_start_ticks(os.getpid()),
    }
    path = tmp_path / "CLAIM.json"
    write_json(path, claim)
    assert subject.validate_writer_claim(
        path, signature_sha="a" * 64, executor_epoch=2, fencing_sha=fencing_sha,
    )["pid"] == os.getpid()
    claim["executor_epoch"] = 3
    write_json(path, claim)
    with pytest.raises(RuntimeError, match="claim/fence mismatch"):
        subject.validate_writer_claim(
            path, signature_sha="a" * 64, executor_epoch=2, fencing_sha=fencing_sha,
        )


def test_review_video_fully_decodes_native_frame_denominator(tmp_path: Path) -> None:
    frame_count = 3
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 960),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((960, 1280, 3), 30 + 10 * index, np.uint8))
    writer.release()
    mask_data = np.zeros((frame_count, 960, 1280), bool)
    mask_data[:, 400:500, 500:650] = True
    masks = {
        object_id: write_mask(tmp_path / f"{object_id}.npz", mask_data)
        for object_id in subject.OBJECTS
    }
    rows = []
    for frame_id in range(frame_count):
        rows.append({
            "frame_id": frame_id,
            "hand_id": "left",
            "finger_id": "index",
            "finger_associated_visible_surface_point": visible_sample(
                [0.0, 0.0, 0.5], [600.0, 450.0],
            ),
            "metric": {"inside_visible_patch": True},
        })
    destination = tmp_path / "review.mp4"
    hand_mask = write_mask(tmp_path / "left_hand.npz", np.ones_like(mask_data))
    hand_roles = {
        "left": {
            "consumer_allowed": True,
            "mask": hand_mask,
            "states": [{"semantic_admitted": True} for _ in range(frame_count)],
        },
        "right": {"consumer_allowed": False, "mask": None, "states": []},
    }
    decoded = subject.render_review(
        source, destination, frame_count, masks, hand_roles, rows,
    )
    assert decoded["full_decode"] is True
    assert decoded["frame_count"] == frame_count
    assert subject.ref(destination)["bytes"] > 0


def test_main_declares_atomic_publication_and_all_required_root_outputs() -> None:
    run_source = inspect.getsource(subject.run_card_session)
    main_source = inspect.getsource(subject.main)
    assert "session-staging" in run_source
    assert "os.replace(stage, final)" in run_source
    assert 'stage / "INTERACTION_EVIDENCE.json"' in run_source
    for name in (
        "CLAIM.json", "RUN_SIGNATURE.json", "INTERACTION_EVIDENCE_LEDGER.json",
        "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ):
        assert name in main_source
    assert '"strict_metric_contact_authorized": False' in main_source
    assert '"physical_deployment_authorized": False' in main_source
