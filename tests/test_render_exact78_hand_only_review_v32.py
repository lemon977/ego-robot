from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import render_exact78_hand_only_review_v32 as review


SESSION = "play_cards_0901_042"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_video(path: Path, count: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1280, 960)
    )
    assert writer.isOpened()
    for frame in range(count):
        image = np.full((960, 1280, 3), frame * 30, dtype=np.uint8)
        cv2.circle(image, (180 + 30 * frame, 400), 60, (20, 120, 240), -1)
        writer.write(image)
    writer.release()


def link_names(side: str) -> np.ndarray:
    values = [f"hand_{side}_base_link"]
    for finger, count in (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4)):
        values.extend(f"hand_{side}_{finger}_link{index}" for index in range(1, count + 1))
    return np.asarray(values)


def make_inputs(root: Path, count: int = 3) -> tuple[Path, Path, Path, Path]:
    video = root / "physical_mono.mp4"
    write_video(video, count)
    rgb_result = root / "RGB_RESULT.json"
    write_json(
        rgb_result,
        {
            "schema_version": review.RGB_SCHEMA,
            "status": "PASS_DEVELOPMENT_MASKS",
            "session_id": SESSION,
            "frame_count": count,
            "image_domain": "CURRENT_MONO_1280X960_NO_ADDITIONAL_LENS_REMAP",
            "inputs": {"video": review.ref(video)},
            "claims": {"training_eligible": False},
        },
    )
    valid = np.ones((count, 2), dtype=bool)
    valid[-1, 1] = False
    transforms = np.tile(np.eye(4), (count, 2, 23, 1, 1)).astype(np.float64)
    for frame in range(count):
        for side in range(2):
            for link in range(23):
                transforms[frame, side, link, :3, 3] = (
                    (link % 5 - 2) * 0.015,
                    side * 0.005,
                    -(link // 5) * 0.032,
                )
    transforms[-1, 1] = np.nan
    fk_npz = root / "HAND_ONLY_Q22_FULL_FK.npz"
    np.savez_compressed(
        fk_npz,
        schema_version=np.asarray(review.FK_SCHEMA),
        session_id=np.asarray(SESSION),
        source_frame=np.arange(count),
        timestamp_s=np.arange(count) / 25.0,
        physical_robot_side=np.asarray(["left", "right"]),
        anatomical_side=np.asarray(["left", "right"]),
        joint_names=np.full((2, 22), "joint"),
        link_names=np.stack((link_names("l"), link_names("r"))),
        q22=np.where(valid[..., None], 0.0, np.nan) * np.ones((count, 2, 22)),
        q22_valid=valid,
        source_hawor_observed=valid,
        q22_inferred_not_observed=valid,
        projection_delta=np.where(valid[..., None], 0.0, np.nan)
        * np.ones((count, 2, 22)),
        hard_limit_pass=valid,
        fk_root_relative=transforms,
        fk_finite=valid,
        control_ground_truth=np.asarray(False),
        physical_deployable=np.asarray(False),
    )
    fk_result = root / "FK_RESULT.json"
    write_json(
        fk_result,
        {
            "schema_version": review.FK_SCHEMA,
            "status": "COMPLETED_HAND_ONLY_Q22_FK_ARM_BLOCKED",
            "session_id": SESSION,
            "temporal_authority": "OFFLINE_NONCAUSAL",
            "q22_fk": review.ref(fk_npz),
            "arm_admission": {"status": review.ARM_BLOCKER_STATUS},
            "control_ground_truth": False,
            "training_eligible": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
    )
    return rgb_result, video, fk_result, fk_npz


def test_complete_review_and_decode_receipt(tmp_path: Path) -> None:
    rgb_result, video, fk_result, fk_npz = make_inputs(tmp_path)
    output = tmp_path / "review"
    result = review.render_review(
        session_id=SESSION,
        rgb_result_path=rgb_result,
        rgb_video_path=video,
        fk_result_path=fk_result,
        fk_npz_path=fk_npz,
        output_root=output,
    )
    assert result["status"] == "COMPLETE_DEVELOPMENT_HAND_ONLY_REVIEW_ARM_BLOCKED"
    assert result["decode"]["full_decode_pass"] is True
    assert result["decode"]["review_frames"] == 3
    assert result["hand_only_coverage"] == {
        "valid_side_frames": 5,
        "unknown_side_frames": 1,
        "unknown_policy": "VISIBLE_UNKNOWN_PANEL_NO_FORWARD_FILL",
    }
    assert result["robotized_training_input"] is False
    assert result["training_eligible"] is False
    receipt = json.loads((output / "DECODE_RECEIPT.json").read_text())
    assert receipt["status"] == "PASS_FULL_FRAME_DECODE_TEST_FIXTURE"
    assert receipt["review"]["decoded_frames"] == 3
    frames = json.loads((output / "FRAME_SHA256.json").read_text())
    assert len(frames["source_decoded_bgr_sha256_by_frame"]) == 3
    assert len(frames["review_decoded_bgr_sha256_by_frame"]) == 3


def test_rejects_archive_input(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    evidence = archive / "evidence.json"
    evidence.write_text("{}")
    with pytest.raises(review.HandOnlyReviewError, match="archive input forbidden"):
        review.reject_archive_inputs((evidence,))


def test_finger_edges_cover_all_non_root_links() -> None:
    edges = review.finger_edges(link_names("l"))
    assert len(edges) == 22
    assert {child for _, child, _ in edges} == set(range(1, 23))


def test_refuses_rgb_video_lineage_drift(tmp_path: Path) -> None:
    rgb_result, video, fk_result, fk_npz = make_inputs(tmp_path)
    replacement = tmp_path / "replacement.mp4"
    write_video(replacement, 3)
    with pytest.raises(review.HandOnlyReviewError, match="exact source video"):
        review.validate_inputs(
            session_id=SESSION,
            rgb_result_path=rgb_result,
            rgb_video_path=replacement,
            fk_result_path=fk_result,
            fk_npz_path=fk_npz,
        )
