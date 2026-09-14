from pathlib import Path

import cv2
import numpy as np

from tools.probe_baseline_clean_sources import orb_register, sha256, write_contact_sheet


def textured_frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_orb_register_recovers_translation_and_photometry() -> None:
    target = textured_frame()
    transform = np.float32([[1, 0, 7], [0, 1, -5]])
    donor = cv2.warpAffine(target, transform, (320, 240), borderValue=(0, 0, 0))
    donor = np.clip(donor.astype(np.float32) * 0.82 + 11.0, 0, 255).astype(np.uint8)
    result = orb_register(target, donor)
    assert result["inlier_count"] >= 40
    assert result["inlier_ratio"] >= 0.7
    assert result["median_reprojection_px_fullres"] <= 1.5
    assert result["coverage_ratio"] >= 0.9
    assert (
        result["median_absdiff_luma_after_affine"]
        < result["median_absdiff_luma"]
    )


def test_no_features_stays_unregistered() -> None:
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    result = orb_register(blank, blank)
    assert result["match_count"] == 0
    assert result["homography"] is None
    assert result["coverage_ratio"] == 0.0


def test_contact_sheet_and_sha(tmp_path: Path) -> None:
    output = tmp_path / "sheet.jpg"
    frames = [textured_frame(seed) for seed in range(4)]
    write_contact_sheet(output, frames, [0, 10, 20, 30])
    decoded = cv2.imread(str(output))
    assert decoded is not None
    assert decoded.shape == (540, 960, 3)
    assert len(sha256(output)) == 64
