from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from pipeline.clean_background import (
    FormalCleanBlocked,
    UNSUPPORTED_SENTINEL_RGB,
    assemble_clean_frame,
)
from pipeline.reveal_labels import EvidenceIntegrityError, RevealLabel, classify_reveal_pixels
from pipeline.tests.evidence_test_utils import (
    background_evidence,
    geometry_evidence,
    object_texture_evidence,
    policy_ref,
)


def _case(root: Path):
    object_mask = np.zeros((2, 3), bool)
    object_mask[0, 0] = True
    background_mask = np.zeros((2, 3), bool)
    background_mask[0, 1] = True
    policy = policy_ref(root)
    geometry = geometry_evidence(root, object_mask)
    texture = object_texture_evidence(
        root,
        np.full((2, 3, 3), 101, np.uint8),
        object_mask,
        coverage=1.0,
        flow=0.95,
        photometric=0.03,
    )
    background = background_evidence(
        root,
        np.full((2, 3, 3), 202, np.uint8),
        background_mask,
        background_mask,
        coverage=1.0,
        flow=0.96,
        photometric=0.02,
    )
    human = np.zeros((2, 3), bool)
    human[0, :] = True
    reveal = classify_reveal_pixels(
        h_core=human,
        o_visible_core=np.zeros_like(human),
        u_contact=np.zeros_like(human),
        policy_ref=policy,
        object_geometry=geometry,
        object_texture=texture,
        background_donor=background,
    )
    return policy, texture, background, reveal


def test_clean_uses_bound_donors_and_never_raw_fills_reveal(tmp_path: Path) -> None:
    policy, texture, background, reveal = _case(tmp_path)
    source = np.full((2, 3, 3), 7, np.uint8)
    clean = assemble_clean_frame(
        source_rgb=source,
        reveal=reveal,
        policy_ref=policy,
        object_texture=texture,
        background_donor=background,
        execution_mode="SYNTHETIC_DRY_RUN",
    )
    assert np.all(clean.clean_rgb[0, 0] == 101)
    assert np.all(clean.clean_rgb[0, 1] == 202)
    assert np.all(clean.clean_rgb[0, 2] == UNSUPPORTED_SENTINEL_RGB)
    assert not np.array_equal(clean.clean_rgb[0, 2], source[0, 2])
    assert np.all(clean.clean_rgb[1, :] == 7)
    assert clean.artifact_state == "SYNTHETIC_TEST_ONLY"
    assert clean.formal_clean_status == "FORMAL_CLEAN_BLOCKED"
    assert clean.hidden_fallback_used is False


def test_u_contact_is_not_written_and_keeps_raw_observation(tmp_path: Path) -> None:
    shape = (2, 3)
    uncertain = np.zeros(shape, dtype=bool)
    uncertain[1, 1] = True
    policy = policy_ref(tmp_path)
    reveal = classify_reveal_pixels(
        h_core=np.zeros(shape, dtype=bool),
        o_visible_core=np.zeros(shape, dtype=bool),
        u_contact=uncertain,
        policy_ref=policy,
        object_geometry=None,
        object_texture=None,
        background_donor=None,
    )
    source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    clean = assemble_clean_frame(
        source_rgb=source,
        reveal=reveal,
        policy_ref=policy,
        object_texture=None,
        background_donor=None,
        execution_mode="SYNTHETIC_DRY_RUN",
    )
    assert reveal.labels[1, 1] == RevealLabel.KEEP_SOURCE
    assert not reveal.target_mask[1, 1]
    assert np.array_equal(clean.clean_rgb[1, 1], source[1, 1])
    assert clean.coverage["unsupported_pixel_count"] == 0


def test_clean_rejects_donor_not_identical_to_reveal_donor(tmp_path: Path) -> None:
    policy, texture, background, reveal = _case(tmp_path / "first")
    replacement = object_texture_evidence(
        tmp_path / "second",
        texture.rgb.copy(),
        texture.support_mask.copy(),
        coverage=1.0,
        flow=0.95,
        photometric=0.03,
    )
    with pytest.raises(EvidenceIntegrityError, match="does not equal reveal object donor"):
        assemble_clean_frame(
            source_rgb=np.full((2, 3, 3), 9, np.uint8),
            reveal=reveal,
            policy_ref=policy,
            object_texture=replacement,
            background_donor=background,
            execution_mode="SYNTHETIC_DRY_RUN",
        )


def test_formal_clean_always_blocked_without_real_authorization(tmp_path: Path) -> None:
    policy, texture, background, reveal = _case(tmp_path)
    with pytest.raises(FormalCleanBlocked, match="FORMAL_CLEAN_BLOCKED"):
        assemble_clean_frame(
            source_rgb=np.full((2, 3, 3), 9, np.uint8),
            reveal=reveal,
            policy_ref=policy,
            object_texture=texture,
            background_donor=background,
            execution_mode="G2_CALIBRATION_CANDIDATE",
        )


def test_malformed_keep_source_inside_target_is_rejected(tmp_path: Path) -> None:
    policy, texture, background, reveal = _case(tmp_path)
    reveal.labels[0, 0] = RevealLabel.KEEP_SOURCE
    with pytest.raises(ValueError, match="may not use KEEP_SOURCE"):
        assemble_clean_frame(
            source_rgb=np.full((2, 3, 3), 9, np.uint8),
            reveal=reveal,
            policy_ref=policy,
            object_texture=texture,
            background_donor=background,
            execution_mode="SYNTHETIC_DRY_RUN",
        )


def test_clean_imports_no_opencv_or_inpainting_library() -> None:
    source_path = Path(__file__).parents[1] / "clean_background.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "cv2" not in imported
    assert "propainter" not in imported
