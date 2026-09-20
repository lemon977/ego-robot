from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from chaoyang.ops import build_exact78_stereo_preflight_input_v31 as subject
from chaoyang.pipeline.exact78_stereo_preflight_v31 import mirrored_intrinsics


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_session(parent: Path) -> Path:
    session = parent / subject.SESSION_ID
    video = session / "source_stereo" / f"CameraRecord_{subject.SESSION_ID}_stereo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"encoded-sbs-placeholder")
    left = np.eye(4).tolist()
    right = np.eye(4)
    right[0, 3] = -0.064
    dump(
        session / "camera_params.json",
        {
            "reference_resolution": {"w": 2048, "h": 1536},
            "video_info": {"is_distorted": True},
            "left": {
                "intrinsics": {"fx": 935.0, "fy": 925.0, "cx": 1011.0, "cy": 773.0}
            },
            "right": {
                "intrinsics": {"fx": 936.0, "fy": 926.0, "cx": 1019.0, "cy": 775.0}
            },
            "extrinsics": {"left": left, "right": right.tolist()},
            "source_calibration": {
                "extrinsicConvention": "head_to_camera_3x4_row_major"
            },
        },
    )
    dump(
        session / "clip_manifest.json",
        {
            "clip_name": subject.SESSION_ID,
            "files": {"source_stereo_video": str(video.relative_to(session))},
            "source_stereo_video": {
                "width": 4096,
                "height": 1536,
                "fps": 30.0,
                "frame_count": 420,
            },
            "notes": ["The cropped distorted stereo source is retained."],
        },
    )
    return session


def make_authority(session: Path, path: Path) -> None:
    left_k = np.asarray(
        [[300.0, 0.0, 317.0], [0.0, 301.0, 241.0], [0.0, 0.0, 1.0]]
    )
    right_k = left_k.copy()
    p_left = np.concatenate((left_k, np.zeros((3, 1))), axis=1)
    p_right = np.concatenate((right_k, np.zeros((3, 1))), axis=1)
    p_right[0, 3] = -left_k[0, 0] * 0.064
    dump(
        path,
        {
            "schema_version": subject.AUTHORITY_SCHEMA,
            "session_id": subject.SESSION_ID,
            "source_bindings": {
                "camera_params_sha256": subject.sha256(session / "camera_params.json"),
                "source_stereo_sha256": subject.sha256(
                    session / "source_stereo" / f"CameraRecord_{subject.SESSION_ID}_stereo.mp4"
                ),
            },
            "image_domain": {
                "eye_width": 2048,
                "eye_height": 1536,
                "model_width": 640,
                "model_height": 480,
                "physical_left_source_index": 1,
                "physical_right_source_index": 0,
                "decoded_vst_already_undistorted": True,
                "lens_undistortion_applied": False,
                "horizontal_reflection_for_disparity_sign": True,
                "outputs_unflipped_to_physical_left": True,
            },
            "metric_conversion": {
                "physical_left_k": left_k.tolist(),
                "physical_right_k": right_k.tolist(),
                "model_left_k": mirrored_intrinsics(left_k, 640).tolist(),
                "model_right_k": mirrored_intrinsics(right_k, 640).tolist(),
                "projection_left": p_left.tolist(),
                "projection_right": p_right.tolist(),
                "baseline_m": 0.064,
                "calibration_pixel_domain": "PHYSICAL_LEFT_640x480",
            },
        },
    )


def test_frame_selector_is_fixed_deterministic_and_endpoint_inclusive() -> None:
    assert subject.deterministic_frame_indices(420) == [
        0,
        38,
        76,
        114,
        152,
        190,
        228,
        266,
        304,
        342,
        380,
        419,
    ]


def test_current_assets_without_encoded_projection_authority_fail_closed(tmp_path) -> None:
    session = make_session(tmp_path / "data")
    result = subject.build(session, tmp_path / "out", encoded_calibration=None)
    assert result["status"] == "BLOCKED_EXTERNAL_ASSET"
    assert result["execution_allowed"] is False
    assert result["blockers"] == ["MISSING_ENCODED_DOMAIN_K_P_BASELINE_AUTHORITY"]
    assert result["outputs"]["config"] is None
    blocker = json.loads((tmp_path / "out" / "BLOCKER.json").read_text())
    assert blocker["projection_matrices_synthesized"] is False
    provenance = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    raw = provenance["raw_camera_candidate"]
    assert np.isclose(raw["baseline_candidate_m"], 0.064)
    assert raw["encoded_projection_matrices_present"] is False
    assert raw["authorized_for_metric_conversion"] is False
    assert provenance["metadata_declarations"][
        "clip_notes_label_source_stereo_distorted"
    ] is True


def test_bound_encoded_authority_builds_runner_input_without_decoding(tmp_path) -> None:
    session = make_session(tmp_path / "data")
    authority = tmp_path / "encoded_calibration.json"
    make_authority(session, authority)
    output = tmp_path / "out"
    result = subject.build(session, output, encoded_calibration=authority)
    assert result["status"] == "READY_CPU_STEREO_PREFLIGHT"
    assert result["execution_allowed"] is True
    assert result["source_video_decoded"] is False
    config = json.loads(
        (output / "EXACT78_STEREO_PREFLIGHT_INPUT_V31.json").read_text()
    )
    assert config["frame_indices"] == subject.deterministic_frame_indices(420)
    assert config["image_domain"]["lens_undistortion_applied"] is False
    assert config["image_domain"]["physical_left_source_index"] == 1
    assert config["metric_conversion"]["baseline_m"] == 0.064
    assert "model_left_k" not in config["metric_conversion"]


def test_archive_source_is_rejected_without_opening_video(tmp_path) -> None:
    session = make_session(tmp_path / "archive" / "history")
    result = subject.build(session, tmp_path / "out", encoded_calibration=None)
    assert result["execution_allowed"] is False
    assert "ARCHIVE_SOURCE_FORBIDDEN" in result["blockers"]
    assert result["source_video_decoded"] is False


def test_builder_has_no_image_decoder_or_lens_transform_dependency() -> None:
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "cv2" not in imported
    assert called.isdisjoint({"remap", "undistort", "VideoCapture"})
