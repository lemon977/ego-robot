from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import run_0915_robot15h_object6d_wave0_v1 as subject


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
    return subject.PackedMask(path)


def test_runner_is_fixed_to_w0_encoded_physical_left_and_cpu_only() -> None:
    assert subject.TASK_ID == "0915_robot15h_geometry_object6d_wave0_v1"
    assert subject.INSTANCE_IDS == (
        "playing_card_00", "playing_card_01", "playing_card_02",
    )
    assert subject.DEPTH_REFERENCE == "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
    assert subject.IMAGE_DOMAIN == "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
    assert subject.PIXEL_MAP["lens_undistortion_applied"] is False
    assert subject.PIXEL_MAP["lens_remap_applied"] is False
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "cv2.remap" not in source
    assert "equiDis62" not in source
    assert '"weights": "ABSENT"' in source
    assert '"gpu_used": False' in source
    assert "archive/" not in source
    assert "removal" not in source.lower()
    assert "clean" not in source.lower()


def test_pixel_center_map_is_resize_only_two_x_plus_half() -> None:
    mapping, valid = subject.depth_to_mask_map()
    assert mapping.shape == (480, 640, 2)
    assert valid.all()
    np.testing.assert_array_equal(mapping[0, 0], [0.5, 0.5])
    np.testing.assert_array_equal(mapping[17, 23], [46.5, 34.5])
    np.testing.assert_array_equal(mapping[-1, -1], [1278.5, 958.5])


def test_registered_support_is_exact_intersection_not_hidden_completion() -> None:
    mask = np.zeros(subject.MASK_SHAPE, bool)
    mask[200:240, 300:380] = True
    depth = np.ones(subject.DEPTH_SHAPE, np.float32)
    valid = np.ones(subject.DEPTH_SHAPE, bool)
    valid[105:110, 160:170] = False
    intrinsic = np.asarray([
        [300.0, 0.0, 319.5], [0.0, 300.0, 239.5], [0.0, 0.0, 1.0],
    ])
    mapping, registration = subject.depth_to_mask_map()
    support, points = subject.registered_support(
        mask, depth, valid, intrinsic, mapping, registration,
    )
    assert support.shape == subject.DEPTH_SHAPE
    assert not np.any(support & ~valid)
    assert points.shape == (int(support.sum()), 3)
    assert 0 < support.sum() < mask.sum()


def test_finite_patch_keeps_exact_support_archive_row_and_never_full_extent() -> None:
    support = np.zeros(subject.DEPTH_SHAPE, bool)
    support[100:120, 200:240] = True
    support[105:110, 215:220] = False
    depth = np.ones(subject.DEPTH_SHAPE, np.float32)
    intrinsic = np.asarray([
        [300.0, 0.0, 319.5], [0.0, 300.0, 239.5], [0.0, 0.0, 1.0],
    ])
    record = subject.finite_patch_record(support, depth, intrinsic, archive_row=17)
    assert record["observability"] == "OBSERVABLE_DIRECT_VISIBLE_FINITE_SUPPORT"
    assert record["estimate"]["archive_row"] == 17
    assert record["estimate"]["support_pixel_count"] == int(support.sum())
    assert record["estimate"]["depth_pixel_bbox_xyxy_inclusive"] == [200, 100, 239, 119]
    assert "FULL_OBJECT" in record["semantics"]

    weak = np.zeros(subject.DEPTH_SHAPE, bool)
    weak[0, :23] = True
    assert subject.finite_patch_record(weak, depth, intrinsic, 0)["observability"] == "UNOBSERVABLE"


def test_normal_and_pi_axis_signs_are_temporally_unified() -> None:
    def row(normal: list[float], axis: list[float]) -> dict:
        return {
            "plane_normal": {
                "observability": "OBSERVABLE_DIRECT_VISIBLE_PLANE",
                "estimate": {"unit_xyz": normal},
            },
            "inplane_rotation": {
                "observability": "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS",
                "estimate": {"axis_unit_xyz": axis},
            },
        }

    rows = [row([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]), row([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0])]
    subject.temporally_unify_axes(rows)
    np.testing.assert_array_equal(rows[1]["plane_normal"]["estimate"]["unit_xyz"], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(rows[1]["inplane_rotation"]["estimate"]["axis_unit_xyz"], [1.0, 0.0, 0.0])
    assert rows[1]["plane_normal"]["estimate"]["temporal_sign_flipped"] is True
    assert rows[1]["inplane_rotation"]["estimate"]["temporal_axis_flipped"] is True


def test_chips_without_consumer_mask_is_blocked_without_object6d_artifact() -> None:
    item = {
        "session_id": "get_potato_chips_0915_007",
        "task": "potato_chips",
        "source_group": "recording-a",
        "frame_count": 378,
    }
    row = subject.blocked_row(
        item, depth_ok=True, mask_ok=False,
        blocker="NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK",
    )
    assert row["status"] == "BLOCKED_UPSTREAM_OBJECT_MASK"
    assert row["object6d_attempted"] is False
    assert row["object6d_artifact_emitted"] is False
    assert row["object6d_consumer_allowed"] is False


def test_mask_admission_rejects_non_card_and_requires_independent_instances() -> None:
    batch = {"object_mask_consumer_allowed": True}
    chips = {
        "status": "REJECTED_QUALITY",
        "task": "potato_chips",
        "object_mask_consumer_allowed": False,
        "frame_count": 10,
    }
    value, blocker = subject.validate_mask_session(
        batch, chips, {"frame_count": 10},
    )
    assert value is None
    assert blocker == "NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK"


def test_depth_gate_rejects_any_lens_remap(tmp_path: Path) -> None:
    result = {
        "status": "PASSED",
        "session_id": "play_cards_0915_031",
        "frame_count": 1,
        "consumption_authorized": True,
        "authorized_scopes": [subject.AUTHORIZED_SCOPE],
        "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False,
        "source_mutated": False,
    }
    result_path = tmp_path / "RESULT.json"
    write_json(result_path, result)
    write_json(tmp_path / "ADAPTER_CONTRACT.json", {
        "lens_remap_applied": True,
        "lens_undistortion_applied": False,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "output_spatial_unflip": True,
        "camera_swap": False,
        "physical_source_indices": {"left": 1, "right": 0},
    })
    write_json(tmp_path / "DEPTH_CONTRACT.json", {
        "depth_reference": subject.DEPTH_REFERENCE,
        "frame_geometry": [640, 480],
        "frame_count": 1,
        "consumption_authorized": True,
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
    })
    write_json(tmp_path / "RGB_ALIGNMENT_QA.json", {
        "depth_grid_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "maximum_absolute_channel_error": 0,
        "mismatched_pixels": 0,
    })
    write_json(tmp_path / "DEPTH_SUMMARY.json", {
        "frame_count": 1,
        "consumption_authorized": True,
        "frames": [{}],
    })
    path_ref = subject.ref(result_path)
    root, info, blocker = subject.validate_depth_session(
        {"status": "PASSED", "result": path_ref},
        {"session_id": "play_cards_0915_031", "frame_count": 1},
    )
    assert root is None and info is None
    assert blocker == "DEPTH_ENCODED_DOMAIN_QA_FAILED"


def test_writer_claim_binds_live_pid_epoch_fence_and_signature(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    fencing_sha = hashlib.sha256(b"bounded-fence-token").hexdigest()
    claim = {
        "schema_version": "0915-robot15h-object6d-writer-claim-v1",
        "task_id": subject.TASK_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "unique_write_root": str(tmp_path.resolve()),
        "executor_epoch": 4,
        "run_signature_sha256": "a" * 64,
        "fencing_token_sha256": fencing_sha,
        "pid": os.getpid(),
        "proc_start_ticks": subject.process_start_ticks(os.getpid()),
    }
    path = tmp_path / "CLAIM.json"
    write_json(path, claim)
    observed = subject.validate_writer_claim(
        path, signature_sha="a" * 64, executor_epoch=4, fencing_sha=fencing_sha,
    )
    assert observed["pid"] == os.getpid()
    claim["executor_epoch"] = 5
    write_json(path, claim)
    with pytest.raises(RuntimeError, match="claim/fence mismatch"):
        subject.validate_writer_claim(
            path, signature_sha="a" * 64, executor_epoch=4, fencing_sha=fencing_sha,
        )


def test_render_review_fully_decodes_every_frame(tmp_path: Path) -> None:
    frame_count = 3
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 960),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        image = np.full((960, 1280, 3), 30 + index * 20, np.uint8)
        writer.write(image)
    writer.release()
    mask_array = np.zeros((frame_count, 960, 1280), bool)
    mask_array[:, 400:500, 500:650] = True
    masks = {
        instance_id: write_mask(tmp_path / f"{instance_id}.npz", mask_array)
        for instance_id in subject.INSTANCE_IDS
    }
    frame = {
        "center_xyz": {
            "observability": "OBSERVABLE_VISIBLE_SURFACE_CENTROID",
            "estimate": {"xyz_m": [0.0, 0.0, 1.0]},
        },
        "plane_normal": {
            "observability": "OBSERVABLE_DIRECT_VISIBLE_PLANE",
            "estimate": {"unit_xyz": [0.0, 0.0, -1.0]},
        },
        "inplane_rotation": {
            "observability": "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS",
            "estimate": {"axis_unit_xyz": [1.0, 0.0, 0.0]},
        },
    }
    objects = [
        {"instance_id": instance_id, "frames": [frame for _ in range(frame_count)]}
        for instance_id in subject.INSTANCE_IDS
    ]
    intrinsic = np.asarray([
        [300.0, 0.0, 319.5], [0.0, 300.0, 239.5], [0.0, 0.0, 1.0],
    ])
    destination = tmp_path / "review.mp4"
    decoded = subject.render_review(
        source, destination, masks, objects, [intrinsic] * frame_count,
    )
    assert decoded["full_decode"] is True
    assert decoded["frame_count"] == frame_count
    assert subject.ref(destination)["bytes"] > 0


def test_main_declares_atomic_session_publication_and_required_terminals() -> None:
    run_source = inspect.getsource(subject.run_card_session)
    main_source = inspect.getsource(subject.main)
    assert "session-staging" in run_source
    assert "os.replace(stage, final)" in run_source
    assert 'output / "OBJECT6D_ELIGIBILITY_LEDGER.json"' in main_source
    assert 'output / "BATCH_RESULT.json"' in main_source
    assert 'output / "METRICS.json"' in main_source
    assert 'output / "RESULT.json"' in main_source
    assert 'output / "RUN_RECEIPT.json"' in main_source
    assert "FULL_OBJECT_BOUNDARY_AND_THICKNESS_UNPROVEN" in run_source

