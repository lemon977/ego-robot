from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.depth_multiview_clean_successor import (
    Camera,
    CleanGeometryError,
    SOURCE_PROTECTED_OBJECT,
    SOURCE_UNSUPPORTED,
    audit_exact_provenance,
    composite_evidence_only,
    deterministic_gate_frames,
    forward_warp_real_pixels,
    forward_warp_real_pixels_conservative_splat,
    fuse_visible_donors,
)
from tools import run_depth_multiview_clean_successor as runner


def camera(width: int, height: int, x: float = 0.0) -> Camera:
    intrinsic = np.asarray(
        [[20.0, 0.0, (width - 1) / 2], [0.0, 20.0, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = x
    return Camera(intrinsic, c2w)


def test_identity_depth_warp_and_fusion_use_exact_real_donor_pixels():
    height, width = 12, 16
    yy, xx = np.indices((height, width))
    donor_a = np.stack((xx + 10, yy + 20, xx + yy + 30), axis=2).astype(np.uint8)
    donor_b = donor_a.copy()
    donor_b[:, :, 0] += 2
    depth = np.full((height, width), 2.0, np.float32)
    removal = np.zeros((height, width), bool)
    removal[3:9, 5:12] = True
    protected = np.zeros_like(removal)
    excluded = np.zeros_like(removal)
    first = forward_warp_real_pixels(
        donor_a, depth, camera(width, height), depth, camera(width, height),
        excluded, removal, protected,
    )
    second = forward_warp_real_pixels(
        donor_b, depth, camera(width, height), depth, camera(width, height),
        excluded, removal, protected,
    )
    assert first.stable_depth_consistency == 1.0
    fused = fuse_visible_donors([(4, first), (9, second)])
    assert fused.valid[removal].all()
    assert np.all(np.isin(fused.source_frame[removal], (4, 9)))
    raw = np.full_like(donor_a, 220)
    tracker = removal.copy()
    composed = composite_evidence_only(raw, removal, tracker, protected, fused)
    assert composed["supported_fraction"] == 1.0
    assert composed["changed_outside_removal_pixels"] == 0
    donors = {4: donor_a, 9: donor_b}
    audit = audit_exact_provenance(
        raw,
        composed["clean"],
        composed["source_kind"],
        composed["source_frame"],
        composed["source_x"],
        composed["source_y"],
        donors,
    )
    assert audit == {"known_source_values": True, "mismatch_pixels": 0, "verified": True}


def test_no_evidence_leaves_raw_and_marks_unsupported_without_inpainting():
    height, width = 12, 16
    raw = np.full((height, width, 3), 91, np.uint8)
    depth = np.full((height, width), 2.0, np.float32)
    removal = np.zeros((height, width), bool)
    removal[3:8, 4:11] = True
    excluded = np.ones_like(removal)
    protected = np.zeros_like(removal)
    observation = forward_warp_real_pixels(
        raw, depth, camera(width, height), depth, camera(width, height),
        excluded, removal, protected,
    )
    with pytest.raises(CleanGeometryError, match="not enough donor"):
        fuse_visible_donors([(1, observation)])
    fused = fuse_visible_donors([(1, observation), (2, observation)])
    result = composite_evidence_only(raw, removal, removal, protected, fused)
    assert np.array_equal(result["clean"], raw)
    assert np.all(result["source_kind"][removal] == SOURCE_UNSUPPORTED)
    assert result["unsupported_pixels"] == int(removal.sum())
    assert result["supported_fraction"] == 0.0


def test_conservative_surface_splat_keeps_exact_source_provenance():
    height, width = 20, 24
    yy, xx = np.indices((height, width))
    donor = np.stack((xx + 10, yy + 20, xx + yy + 30), axis=2).astype(np.uint8)
    depth = np.full((height, width), 1.5, np.float32)
    removal = np.zeros((height, width), bool)
    removal[6:14, 7:17] = True
    protected = np.zeros_like(removal)
    observation = forward_warp_real_pixels_conservative_splat(
        donor,
        depth,
        camera(width, height),
        depth,
        camera(width, height),
        np.zeros_like(removal),
        removal,
        protected,
    )
    fused = fuse_visible_donors([(1, observation), (2, observation)])
    raw = np.full_like(donor, 220)
    composed = composite_evidence_only(raw, removal, removal, protected, fused)
    assert composed["supported_fraction"] == 1.0
    audit = audit_exact_provenance(
        raw,
        composed["clean"],
        composed["source_kind"],
        composed["source_frame"],
        composed["source_x"],
        composed["source_y"],
        {1: donor, 2: donor},
    )
    assert audit["verified"] is True


def test_object_protect_is_never_edited_and_overlap_fails_closed():
    height, width = 12, 16
    raw = np.full((height, width, 3), 100, np.uint8)
    depth = np.full((height, width), 2.0, np.float32)
    removal = np.zeros((height, width), bool)
    removal[3:8, 3:8] = True
    protected = np.zeros_like(removal)
    protected[8:10, 8:10] = True
    observation = forward_warp_real_pixels(
        raw - 20, depth, camera(width, height), depth, camera(width, height),
        np.zeros_like(removal), removal, protected,
    )
    fused = fuse_visible_donors([(1, observation), (2, observation)])
    result = composite_evidence_only(raw, removal, removal, protected, fused)
    assert np.array_equal(result["clean"][protected], raw[protected])
    assert np.all(result["source_kind"][protected] == SOURCE_PROTECTED_OBJECT)
    bad = removal.copy()
    bad[protected] = True
    with pytest.raises(CleanGeometryError, match="overlap"):
        composite_evidence_only(raw, bad, bad, protected, fused)


def test_gate_frame_contract_is_exact_12_plus_three_consecutive_12():
    spatial, windows = deterministic_gate_frames(206)
    assert len(spatial) == len(set(spatial)) == 12
    assert len(windows) == 3
    assert all(window == list(range(window[0], window[0] + 12)) for window in windows)
    with pytest.raises(CleanGeometryError):
        deterministic_gate_frames(35)


def ref(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def make_formal_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    session, frame_count, width, height = "synthetic_clean_session", 36, 16, 12
    video = root / "raw.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((height, width, 3), 30 + index, np.uint8))
    writer.release()

    dense_frames = root / "dense_frames"
    dense_frames.mkdir()
    dense_rows = []
    for index in range(frame_count):
        path = dense_frames / f"{index:06d}.npz"
        np.savez_compressed(path, depth_m=np.full((6, 8), 2.0, np.float32))
        row = ref(path)
        dense_rows.append({"frame_id": index, "relative_path": str(path.relative_to(root)), "bytes": row["bytes"], "sha256": row["sha256"]})
    dense_manifest = write_json(
        root / "DENSE_FRAME_MANIFEST.json",
        {
            "schema_version": "foundationstereo-dense-depth-frame-manifest-v1",
            "session": session,
            "expected_frames": frame_count,
            "actual_frames": frame_count,
            "frames": dense_rows,
        },
    )
    dense_result = write_json(
        root / "DENSE_RESULT.json",
        {
            "schema_version": "foundationstereo-dense-depth-result-v1",
            "status": "PASS_DENSE_METRIC_DEPTH_FULLSESSION",
            "metric_depth_complete": True,
            "session": session,
            "frame_count": frame_count,
            "frame_manifest": ref(dense_manifest),
        },
    )

    camera_path = root / "CAMERA.npz"
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    intrinsic = np.asarray([[20, 0, 7.5], [0, 20, 5.5], [0, 0, 1]], np.float64)
    np.savez_compressed(
        camera_path,
        c2w=c2w,
        intrinsics=np.repeat(intrinsic[None], frame_count, axis=0),
        original_frame_indices=np.arange(frame_count),
    )
    camera_result = write_json(root / "CAMERA_RESULT.json", {"status": "PASS", "session": session})

    trajectory = root / "OBJECT6D.npz"
    np.savez_compressed(trajectory, pose=np.repeat(np.eye(4)[None], frame_count, axis=0))
    object_result = write_json(
        root / "OBJECT6D_RESULT.json",
        {
            "status": "PASS_FORMAL_OBJECT6D_FULLSESSION",
            "session": session,
            "frame_count": frame_count,
            "consumption_authorized": True,
            "artifacts": {"trajectory": ref(trajectory)},
        },
    )

    registration_result = write_json(
        root / "REGISTRATION_RESULT.json",
        {
            "schema_version": "rgb-depth-registration-result-v1",
            "status": "PASS_RGB_REGISTERED_DENSE_DEPTH_FULLSESSION",
            "consumption_authorized": True,
            "session": session,
            "frame_count": frame_count,
        },
    )
    registered_root = root / "registered"
    registered_root.mkdir()
    registered_rows = []
    for index in range(frame_count):
        path = registered_root / f"{index:06d}.npz"
        depth = np.full((height, width), 2.0, np.float32)
        np.savez_compressed(path, depth_m=depth, valid=np.isfinite(depth))
        registered_rows.append({"frame_id": index, "depth": ref(path)})
    registered_manifest = write_json(
        root / "REGISTERED_MANIFEST.json",
        {
            "schema_version": "rgb-registered-dense-depth-frame-manifest-v1",
            "status": "PASS_RGB_REGISTERED_DENSE_DEPTH_FULLSESSION",
            "consumption_authorized": True,
            "session": session,
            "frame_count": frame_count,
            "resolution": {"width": width, "height": height},
            "depth_semantics": "RGB_OPTICAL_AXIS_CAMERA_Z_METRES_FLOAT32",
            "source_dense_stereo_result_sha256": ref(dense_result)["sha256"],
            "registration_result_sha256": ref(registration_result)["sha256"],
            "camera_geometry_sha256": ref(camera_path)["sha256"],
            "frames": registered_rows,
        },
    )

    mask_root = root / "masks"
    mask_root.mkdir()
    mask_rows = []
    for index in range(frame_count):
        row = {"frame_id": index}
        for key in ("removal_mask", "tracker_mask", "object_protect_mask"):
            image = np.zeros((height, width), np.uint8)
            if key != "object_protect_mask":
                image[4:8, 5:10] = 255
            path = mask_root / f"{index:06d}_{key}.png"
            assert cv2.imwrite(str(path), image)
            row[key] = ref(path)
        mask_rows.append(row)
    mask_manifest = write_json(
        root / "MASK_FRAME_MANIFEST.json",
        {
            "schema_version": "depth-multiview-clean-mask-frame-manifest-v1",
            "status": "PASS_FULLSESSION_OBJECT6D_CONSTRAINED_MASK",
            "consumption_authorized": True,
            "session": session,
            "frame_count": frame_count,
            "resolution": {"width": width, "height": height},
            "object6d_result_sha256": ref(object_result)["sha256"],
            "frames": mask_rows,
        },
    )
    mask_result = write_json(
        root / "MASK_RESULT.json",
        {
            "status": "PASS_FORMAL_OBJECT6D_CONSTRAINED_MASK",
            "session": session,
            "frame_count": frame_count,
            "automatic_gate_pass": True,
            "consumption_authorized": True,
        },
    )
    mask_review = write_json(
        root / "MASK_REVIEW.json",
        {"status": "PASS", "consumption_authorized": True},
    )
    spatial, windows = deterministic_gate_frames(frame_count)
    output = root / "must_not_exist"
    manifest = write_json(
        root / "INPUT.json",
        {
            "schema_version": "depth-multiview-clean-input-manifest-v1",
            "status": "READY_FOR_DEPTH_MULTIVIEW_CLEAN",
            "production_input": True,
            "phase": "CANARY_GATE",
            "task": "chips",
            "session": session,
            "frame_count": frame_count,
            "source_resolution": [width, height],
            "inputs": {
                "raw_video": ref(video),
                "dense_stereo_result": ref(dense_result),
                "dense_stereo_frame_manifest": ref(dense_manifest),
                "rgb_depth_registration_result": ref(registration_result),
                "rgb_registered_depth_manifest": ref(registered_manifest),
                "camera_geometry": ref(camera_path),
                "camera_geometry_result": ref(camera_result),
                "mask_result": ref(mask_result),
                "mask_visual_review": ref(mask_review),
                "mask_frame_manifest": ref(mask_manifest),
                "object6d_result": ref(object_result),
                "object6d_trajectory": ref(trajectory),
            },
            "gate_frames": {"spatial12": spatial, "temporal_3x12": windows},
            "method_contract": runner.METHOD_CONTRACT,
            "output_root": str(output.resolve()),
            "review_name": "SYNTHETIC_CLEAN_REVIEW.mp4",
            "no_clobber": True,
            "prior_canary_admission": None,
        },
    )
    return manifest, registered_root / "000000.npz"


def test_validate_only_accepts_complete_sha_bound_formal_fixture_and_rejects_drift(tmp_path):
    manifest, registered_frame = make_formal_fixture(tmp_path / "fixture")
    result = runner.validate_manifest(manifest)
    assert result["status"] == "PASS_VALIDATE_ONLY_CPU"
    assert result["execution_started"] is False
    with registered_frame.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(runner.ManifestError, match="byte count drift"):
        runner.validate_manifest(manifest)


def test_schemas_are_valid_and_method_contract_cannot_enable_fallbacks():
    input_schema = json.loads(runner.INPUT_SCHEMA.read_text())
    batch_schema = json.loads(runner.BATCH_SCHEMA.read_text())
    visual_schema = json.loads(runner.VISUAL_SCHEMA.read_text())
    for schema in (input_schema, batch_schema, visual_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
    assert runner.METHOD_CONTRACT["nearest_canvas_extrapolation"] is False
    assert runner.METHOD_CONTRACT["generative_or_inpaint_fallback"] is False


def test_current_plan_stays_fail_closed_until_registration_object6d_and_mask_are_formal():
    plan = runner.build_current_plan()
    assert plan["status"] == "CPU_READY_BLOCKED_UPSTREAM"
    assert plan["start_authorized"] is False
    for row in plan["rows"]:
        joined = "\n".join(row["blockers"])
        assert "RGB-depth" in joined
        assert "Object6D" in joined
        assert "Mask" in joined
        assert row["canary_start_authorized"] is False
        assert row["full_start_authorized"] is False
