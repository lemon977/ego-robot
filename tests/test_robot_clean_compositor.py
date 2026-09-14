from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator

import cv2
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline.depth_occlusion_v3 import supersampled_intrinsics  # noqa: E402
from pipeline.robot_clean_compositor import (  # noqa: E402
    CompositorContractError,
    compose_frame,
    prepare_contract,
    sha256_file,
)
from tools.build_robot_clean_compositor_synthetic_fixture import build_fixture  # noqa: E402
from tools.compose_robot_clean_general import run  # noqa: E402


TEST_PARENT = PROJECT / "tasks/control/runs"


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve(strict=True).relative_to(PROJECT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@contextmanager
def _fixture(task_id: str = "chips") -> Iterator[tuple[Path, Path]]:
    root = Path(tempfile.mkdtemp(prefix=".test_robot_clean_", dir=TEST_PARENT))
    try:
        manifest = build_fixture(root / "fixture", task_id)
        yield root, manifest
    finally:
        shutil.rmtree(root)


def _rewrite_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("task_id", ["chips", "poker"])
def test_synthetic_fixture_preflights_both_tasks_without_frame_zero_assumption(
    task_id: str,
) -> None:
    with _fixture(task_id) as (_, manifest):
        prepared = prepare_contract(manifest, PROJECT)
        assert prepared.value["task_id"] == task_id
        assert prepared.session["frame_ids"] == [101, 104, 108, 113, 119, 126]
        assert prepared.value["frame_count"] == 6
        if task_id == "poker":
            geometry = prepared.object6d["geometry_sha256"].tolist()
            assert geometry[0] == geometry[1] == geometry[2]


def test_synthetic_smoke_emits_both_codecs_and_unpromotable_watermark() -> None:
    with _fixture("chips") as (root, manifest):
        result = run(manifest, root / "smoke")
        assert result["status"] == "DEV_SYNTHETIC_COMPLETE_NOT_FORMAL"
        assert result["consumption_authorized"] is False
        assert result["formal_candidate"] is False
        assert result["watermark"] == "DEV_SYNTHETIC"
        assert result["output"]["master_probe"]["codec_name"] == "ffv1"
        assert result["output"]["review_probe"]["codec_name"] == "h264"
        assert result["frame_ids"] == [101, 104, 108, 113, 119, 126]
        capture = cv2.VideoCapture(str(root / "smoke/ROBOT_ON_CLEAN_MASTER_FFV1.mkv"))
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            # The dark-red title bar and lower red mark are both baked into
            # every emitted synthetic frame, not merely into the review sheet.
            assert float(frame[:20, :, 2].mean()) > float(frame[:20, :, 0].mean()) + 20
            assert np.count_nonzero(frame[..., 2] > frame[..., 1] + 80) > 20
        capture.release()
        assert decoded == 6


def test_artifact_byte_tamper_is_rejected_before_output() -> None:
    with _fixture("chips") as (_, manifest):
        value = json.loads(manifest.read_text())
        object6d = PROJECT / value["artifacts"]["object6d"]["path"]
        with object6d.open("ab") as handle:
            handle.write(b"tamper")
        with pytest.raises(CompositorContractError, match="byte mismatch"):
            prepare_contract(manifest, PROJECT)


def test_session_object_identity_drift_is_rejected_even_with_fresh_sha() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        session_path = PROJECT / main["artifacts"]["session_manifest"]["path"]
        session = json.loads(session_path.read_text())
        session["object_ids"][0], session["object_ids"][1] = (
            session["object_ids"][1],
            session["object_ids"][0],
        )
        _rewrite_json(session_path, session)
        main["artifacts"]["session_manifest"] = _artifact(session_path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="task object contract"):
            prepare_contract(manifest, PROJECT)


def test_nonpenetration_sha_cannot_be_rebound_only_at_top_level() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        path = PROJECT / main["artifacts"]["nonpenetration_result"]["path"]
        value = json.loads(path.read_text())
        value["bindings"]["trajectory_sha256"] = "0" * 64
        _rewrite_json(path, value)
        main["artifacts"]["nonpenetration_result"] = _artifact(path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="nonpenetration_result_sha256"):
            prepare_contract(manifest, PROJECT)


def test_dev_cross_stage_cannot_claim_formal_readiness() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        path = PROJECT / main["artifacts"]["cross_stage_admission"]["path"]
        value = json.loads(path.read_text())
        value["routes"]["robot_formal_ready"] = True
        _rewrite_json(path, value)
        main["artifacts"]["cross_stage_admission"] = _artifact(path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="claims formal readiness"):
            prepare_contract(manifest, PROJECT)


def test_coordinate_threshold_cannot_be_overridden_in_manifest() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        main["coordinate_contract"]["occlusion_epsilon_m"] = 0.004
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="input schema failure"):
            prepare_contract(manifest, PROJECT)


def test_frame_manifest_truncation_is_rejected_before_composition() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        path = PROJECT / main["artifacts"]["robot_render_frames"]["path"]
        value = json.loads(path.read_text())
        value["frames"].pop()
        _rewrite_json(path, value)
        main["artifacts"]["robot_render_frames"] = _artifact(path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="frame list length mismatch"):
            prepare_contract(manifest, PROJECT)


def test_synthetic_authority_cannot_set_consumption_authorized() -> None:
    with _fixture("chips") as (_, manifest):
        main = json.loads(manifest.read_text())
        path = PROJECT / main["artifacts"]["mask_authority"]["path"]
        value = json.loads(path.read_text())
        value["consumption_authorized"] = True
        _rewrite_json(path, value)
        main["artifacts"]["mask_authority"] = _artifact(path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="consumption authority mismatch"):
            prepare_contract(manifest, PROJECT)


def test_clean_master_non_ffv1_codec_is_rejected_even_when_rebound() -> None:
    with _fixture("chips") as (root, manifest):
        main = json.loads(manifest.read_text())
        source = PROJECT / main["artifacts"]["clean_master"]["path"]
        h264 = root / "fixture/CLEAN_WRONG_CODEC.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(h264),
            ],
            check=True,
        )
        main["artifacts"]["clean_master"] = _artifact(h264)
        authority_path = PROJECT / main["artifacts"]["clean_authority"]["path"]
        authority = json.loads(authority_path.read_text())
        authority["bindings"]["clean_master_sha256"] = sha256_file(h264)
        _rewrite_json(authority_path, authority)
        main["artifacts"]["clean_authority"] = _artifact(authority_path)
        _rewrite_json(manifest, main)
        with pytest.raises(CompositorContractError, match="Clean master must be FFV1"):
            prepare_contract(manifest, PROJECT)


def test_depth_order_and_partial_alpha_are_composited_in_linear_space() -> None:
    clean = np.full((1, 2, 3), (20, 60, 100), dtype=np.uint8)
    robot = np.full((2, 4, 3), (200, 140, 80), dtype=np.uint8)
    alpha = np.ones((2, 4), dtype=np.float32)
    alpha[:, 2:] = 0.5
    near = np.full((2, 4), np.nan, dtype=np.float32)
    far = np.full((2, 4), np.nan, dtype=np.float32)
    near[:, :2] = 0.8
    far[:, :2] = 0.9
    index = np.zeros((2, 4), dtype=np.int16)
    index[:, :2] = 1
    visible = index > 0
    camera = np.asarray([[20.0, 0, 0.5], [0, 20.0, 0.0], [0, 0, 1.0]])
    camera_2x = supersampled_intrinsics(camera)
    u = np.arange(4)[None, :]
    v = np.arange(2)[:, None]
    norm = np.sqrt(
        1
        + ((u - camera_2x[0, 2]) / camera_2x[0, 0]) ** 2
        + ((v - camera_2x[1, 2]) / camera_2x[1, 1]) ** 2
    )
    robot_range = (np.full((2, 4), 1.0) * norm).astype(np.float32)
    result = compose_frame(
        clean_bgr=clean,
        robot_bgr_2x=robot,
        robot_range_m_2x=robot_range,
        robot_alpha_2x=alpha,
        object_depth_near_m_2x=near,
        object_depth_far_m_2x=far,
        object_index_2x=index,
        visible_object_mask_2x=visible,
        camera_intrinsics=camera,
    )
    assert result.bgr[0, 0].tolist() == clean[0, 0].tolist()
    low = np.minimum(clean[0, 1], robot[0, 2])
    high = np.maximum(clean[0, 1], robot[0, 2])
    assert np.all(result.bgr[0, 1] > low)
    assert np.all(result.bgr[0, 1] < high)
    assert result.object_front_subpixels == 4
    assert result.robot_visible_alpha_sum == pytest.approx(2.0)


@pytest.mark.parametrize(
    "bad_alpha",
    [np.nan, -0.01, 1.01],
)
def test_alpha_outside_frozen_contract_fails_closed(bad_alpha: float) -> None:
    clean = np.zeros((1, 1, 3), dtype=np.uint8)
    robot = np.zeros((2, 2, 3), dtype=np.uint8)
    alpha = np.zeros((2, 2), dtype=np.float32)
    alpha[0, 0] = bad_alpha
    near = np.full((2, 2), np.nan, dtype=np.float32)
    with pytest.raises(CompositorContractError, match="alpha"):
        compose_frame(
            clean_bgr=clean,
            robot_bgr_2x=robot,
            robot_range_m_2x=np.ones((2, 2), dtype=np.float32),
            robot_alpha_2x=alpha,
            object_depth_near_m_2x=near,
            object_depth_far_m_2x=near.copy(),
            object_index_2x=np.zeros((2, 2), dtype=np.int16),
            visible_object_mask_2x=np.zeros((2, 2), dtype=np.bool_),
            camera_intrinsics=np.eye(3),
        )


def test_cli_exposes_no_task_session_or_frame_override() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT / "tools/compose_robot_clean_general.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--task-id" not in completed.stdout
    assert "--session-id" not in completed.stdout
    assert "--start-frame" not in completed.stdout
    assert "--end-frame" not in completed.stdout
