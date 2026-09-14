from __future__ import annotations

import numpy as np

from tools.run_newtask_baseline_clean_l3 import (
    Case,
    estimate_registration,
    fixture_protection,
    select_real_donors,
)


def test_registration_recovers_diagnostic_translation() -> None:
    rng = np.random.default_rng(5)
    target = rng.integers(0, 256, size=(960, 540, 3), dtype=np.uint8)
    donor = np.zeros_like(target)
    donor[7:, 11:] = target[:-7, :-11]
    unsafe = np.zeros((960, 540), dtype=bool)
    homography, metrics = estimate_registration(target, donor, unsafe, unsafe)
    assert homography is not None
    mapped = homography @ np.asarray([200.0, 300.0, 1.0])
    mapped /= mapped[2]
    assert np.allclose(mapped[:2], [211.0, 307.0], atol=1.5)
    assert metrics["status"] == "PASS_DIAGNOSTIC_REGISTRATION"


def test_real_donor_selector_never_synthesizes_pixel_value() -> None:
    target_xy = np.asarray([[1, 1], [2, 2]], dtype=np.int32)
    rows = []
    for frame, value in ((3, 20), (7, 21), (9, 22)):
        rows.append(
            {
                "frame_index": frame,
                "valid": np.asarray([True, False]),
                "rgb": np.asarray([[value, value, value], [99, 99, 99]], dtype=np.uint8),
                "x": np.asarray([10, 11], dtype=np.int32),
                "y": np.asarray([12, 13], dtype=np.int32),
            }
        )
    result = select_real_donors(target_xy, rows)
    assert result["resolved"].tolist() == [True, False]
    assert tuple(result["rgb"][0]) in {(20, 20, 20), (21, 21, 21), (22, 22, 22)}
    assert result["frame"][0] in {3, 7, 9}
    assert result["frame"][1] == -1


def test_object_and_frozen_fixture_are_protected() -> None:
    case = Case(
        name="synthetic",
        source_mov=None,  # type: ignore[arg-type]
        source_sha256="0" * 64,
        frame_count=1,
        targets=(0,),
        donor_bank=(0,),
        frames_root=None,  # type: ignore[arg-type]
        human_root=None,  # type: ignore[arg-type]
        human_results=None,  # type: ignore[arg-type]
        object_root=None,  # type: ignore[arg-type]
        object_results=None,  # type: ignore[arg-type]
        table_y_min=0,
        fixture_xyxy=(10, 20, 30, 40),
        frame0_missing_semantic_points={},
    )
    object_mask = np.zeros((960, 540), dtype=bool)
    object_mask[100, 100] = True
    protected = fixture_protection(case, object_mask)
    assert protected[25, 15]
    assert protected[100, 100]
