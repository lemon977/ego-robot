import numpy as np

from pipeline.clean_medoid_light_successor import (
    audit_composite,
    donor_registration_support,
    localized_source_composite,
    robust_real_pixel_temporal_medoid,
)
from tools.run_clean_medoid_light_synthetic_fixture import run_fixture
from tools.evaluate_clean_canary_general_gate import (
    audit_provenance_maps,
    audit_source_map_geometry,
    encode_and_audit_lossless,
    validate_threshold_calibration,
    verify_asset_manifest,
)


def test_temporal_medoid_is_exact_real_pixel_and_excludes_declared_contamination():
    images = np.full((3, 20, 30, 3), 50, np.uint8)
    images[1] += 2
    images[2] -= 2
    images[0, 5:10, 7:14] = (200, 30, 20)
    valid = np.ones(images.shape[:3], bool)
    excluded = np.zeros_like(valid)
    excluded[0, 5:10, 7:14] = True
    result = robust_real_pixel_temporal_medoid(
        images, valid, excluded,
        foreground_bgr_l2_max=20,
        shadow_luma_delta_min=8,
        minimum_pure_observations=2,
    )
    row, column = np.indices(images.shape[1:3])
    assert result["valid"].all()
    assert np.array_equal(result["plate"], images[result["choice_index"], row, column])
    assert not np.any(result["plate"][5:10, 7:14] == np.asarray((200, 30, 20)))


def test_registration_support_excludes_pico_geometry_and_table_prefix():
    excluded = np.zeros((80, 100), dtype=bool)
    excluded[50:60, 40:50] = True
    support = donor_registration_support(
        excluded,
        minimum_y=30,
        extra_dilation_px=3,
    )
    assert support.dtype == bool
    assert not support[:30].any()
    assert not support[47:63, 37:53].all()
    assert not support[55, 45]
    assert support[70, 80]


def test_composite_changes_only_authorized_support_and_preserves_object():
    raw = np.full((30, 40, 3), 100, np.uint8)
    donor = np.full_like(raw, 80)
    human = np.zeros((30, 40), bool)
    human[10:20, 10:30] = True
    obj = np.zeros_like(human)
    obj[13:17, 18:22] = True
    clean, source, _ = localized_source_composite(raw, donor, np.ones_like(human), human, obj, 3)
    audit = audit_composite(raw, clean, source, obj)
    assert audit["changed_pixels_outside_authorized_domain"] == 0
    assert audit["changed_protected_object_pixels"] == 0
    assert np.array_equal(clean[obj], raw[obj])


def test_end_to_end_synthetic_fixture(tmp_path):
    report = run_fixture(tmp_path / "fixture")
    assert report["status"] == "PASS"
    assert report["real_pixel_medoid_exact"] is True


def test_ffv1_audit_master_is_pixel_exact(tmp_path):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(3)]
    assert encode_and_audit_lossless(tmp_path / "master.mkv", frames, 25.0) == 0


def test_source_map_geometry_rejects_self_authorized_background_edit():
    human = np.zeros((80, 100), bool)
    human[30:60, 35:65] = True
    protected = np.zeros_like(human)
    protected[42:50, 48:56] = True
    source = np.zeros(human.shape, np.uint8)
    source[human & ~protected] = 1
    source[protected] = 4
    good = audit_source_map_geometry(
        source, human, protected, seam_width_px=4, shadow_radius_px=12
    )
    assert good["semantics_verified"] is True
    source[3, 3] = 1
    bad = audit_source_map_geometry(
        source, human, protected, seam_width_px=4, shadow_radius_px=12
    )
    assert bad["semantics_verified"] is False
    assert bad["mismatch_pixels"] == 1


def test_source_map_geometry_limits_shadow_to_exterior_halo():
    human = np.zeros((80, 100), bool)
    human[30:60, 35:65] = True
    protected = np.zeros_like(human)
    source = np.zeros(human.shape, np.uint8)
    source[human] = 1
    source[25, 50] = 5
    assert audit_source_map_geometry(
        source, human, protected, seam_width_px=4, shadow_radius_px=12
    )["semantics_verified"] is True
    source[5, 5] = 5
    assert audit_source_map_geometry(
        source, human, protected, seam_width_px=4, shadow_radius_px=12
    )["semantics_verified"] is False


def test_runtime_thresholds_equal_frozen_sensitivity_replay():
    import json
    from pathlib import Path

    gate = json.loads(Path(
        "tasks/chips/runs/clean/20260903_chips001_clean_v3_quality_audit_v1/GENERAL_CLEAN_QUALITY_GATE_V1.json"
    ).read_text())
    replay_path, replay = validate_threshold_calibration(gate)
    assert replay_path.is_file()
    assert replay["status"] == "PASS_NO_MISCLASSIFICATION"


def test_provenance_maps_require_real_donor_id_and_exact_alpha_semantics():
    source = np.asarray([[0, 1, 2, 3, 4, 5]], np.uint8)
    donor = np.asarray([[0, 41, 81, 0, 0, 121]], np.uint16)
    alpha = np.asarray([[0, 65535, 32000, 65535, 0, 65535]], np.uint16)
    good = audit_provenance_maps(source, donor, alpha, [40, 80, 120])
    assert good["verified"] is True
    donor[0, 1] = 999
    bad = audit_provenance_maps(source, donor, alpha, [40, 80, 120])
    assert bad["verified"] is False
    assert bad["mismatch_pixels"] == 1


def test_producer_asset_manifest_verifies_sha_and_size(tmp_path):
    import hashlib

    root = tmp_path / "producer"
    root.mkdir()
    asset = root / "frames" / "00000.png"
    asset.parent.mkdir()
    asset.write_bytes(b"fixture")
    manifest = {
        "frames/00000.png": {
            "bytes": 7,
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
    }
    assert verify_asset_manifest(root, manifest) == 1
    asset.write_bytes(b"changed")
    try:
        verify_asset_manifest(root, manifest)
    except Exception as error:
        assert "drift" in str(error)
    else:
        raise AssertionError("tampered producer asset was accepted")
