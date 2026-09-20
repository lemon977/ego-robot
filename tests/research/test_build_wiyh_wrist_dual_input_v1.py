from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.ops import build_wiyh_wrist_dual_input_v1 as builder
from chaoyang.ops import build_wiyh_wrist_dual_representation_v1 as producer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _transform(*, x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, frames: int = 2) -> tuple[Path, Path]:
    processed = tmp_path / "processed" / "chips_cards_handle_highview_0916" / "cleaned" / "playing_cards"
    experiments = tmp_path / "_run" / "current" / "experiments"
    nominal = {
        "left": _transform(x=0.08, y=0.04, z=-0.13),
        "right": _transform(x=0.08, y=-0.04, z=-0.13),
    }
    m0_attempt = experiments / builder.M0_EXPERIMENT / "attempts" / "attempt_0001"
    _write_json(
        m0_attempt / builder.M0_REFIT,
        {
            "authority_promoted": False,
            "sides": {
                side: {"nominal_controller_to_wrist": matrix.tolist()}
                for side, matrix in nominal.items()
            },
        },
    )
    _write_json(
        m0_attempt / "TERMINAL_RESULT.json",
        {"status": builder.EXPECTED_M0_TERMINAL, "authority_promoted": False},
    )

    for session_index, session_id in enumerate(builder.ALLOWED_SESSION_IDS):
        session_root = processed / session_id
        for frame in range(frames):
            hands: dict[str, object] = {}
            for side_index, side in enumerate(builder.SIDES):
                controller = _transform(
                    x=0.01 * frame,
                    y=0.1 * side_index,
                    z=0.4 + 0.01 * session_index,
                )
                hands[side] = {
                    "pose_source": "egodex_v1_hdf5_wrist_pose",
                    "T_wrist_to_camera": (controller @ nominal[side]).tolist(),
                    "controller6d": {
                        "pose_source": "egodex_v1_hdf5_controller_pose",
                        "T_controller_to_camera": controller.tolist(),
                    },
                }
            _write_json(
                session_root
                / "preprocess"
                / "all_data"
                / f"{frame:05d}"
                / "training_data.json",
                {
                    "metadata": {"idx": frame, "video_time_s": frame / 30.0},
                    "entities": {"hands": hands},
                },
            )

        spec = builder.SESSION_SPECS[session_id]
        attempt = experiments / spec["experiment"] / "attempts" / "attempt_0001"
        _write_json(
            attempt / "TERMINAL_RESULT.json",
            {"status": builder.EXPECTED_HAWOR_TERMINAL},
        )
        hawor_path = attempt / spec["hawor"]
        hawor_path.parent.mkdir(parents=True, exist_ok=True)
        joints_3d = np.zeros((2, frames, 21, 3), dtype=np.float32)
        joints_2d = np.zeros((2, frames, 21, 2), dtype=np.float32)
        roots = np.repeat(np.eye(3, dtype=np.float32)[None, None], 2, axis=0)
        roots = np.repeat(roots, frames, axis=1)
        observed = np.ones((2, frames), dtype=bool)
        observed[1, -1] = False
        for side in range(2):
            for frame in range(frames):
                joints_3d[side, frame, 0] = (
                    0.2 + 0.01 * frame,
                    0.1 * side,
                    0.5 + 0.01 * session_index,
                )
                joints_2d[side, frame, 0] = (100 + frame, 200 + side)
        with hawor_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                joints_3d_camera=joints_3d,
                joints_2d=joints_2d,
                observed=observed,
                root_orient_camera=roots,
                mano_joint_names=np.asarray(["wrist"] + [f"joint_{i}" for i in range(1, 21)]),
                anatomical_side_names=np.asarray(builder.SIDES),
            )

    # These intentionally malformed directories prove that the fixed builder
    # neither discovers nor fills adoption sessions without observations.
    for session_id in builder.EXCLUDED_MISSING_OBSERVATION_IDS:
        poison = processed / session_id / "DO_NOT_READ"
        poison.parent.mkdir(parents=True, exist_ok=True)
        poison.write_text("not an observation", encoding="utf-8")
    return processed, experiments


def test_builds_fixed_real_fields_and_fail_closed_surface(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    source = (
        processed
        / "play_cards_0916_097"
        / "preprocess"
        / "all_data"
        / "00000"
        / "training_data.json"
    )
    before = _sha256(source)
    output = tmp_path / "attempt_0001"
    provenance = builder.build(
        processed_root=processed,
        experiment_root=experiments,
        output_root=output,
    )

    assert _sha256(source) == before
    assert provenance["allowed_sessions"] == list(builder.ALLOWED_SESSION_IDS)
    assert provenance["source_policy"]["excluded_missing_observation_sessions"] == [
        "play_cards_0916_102",
        "play_cards_0916_103",
    ]
    assert [row["role"] for row in provenance["sessions"]] == [
        "DEVELOPMENT",
        "DEVELOPMENT",
        "REGRESSION_FORBIDDEN_FIT",
    ]
    assert provenance["outputs"]["frame_count"] == 6
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/wiyh_wrist_dual_input_provenance_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(provenance, schema)
    with np.load(output / "WRIST_DUAL_INPUT.npz", allow_pickle=False) as bundle:
        assert set(np.unique(bundle["recording_id"]).tolist()) == set(builder.ALLOWED_SESSION_IDS)
        assert bundle["T_camera_controller_raw"].shape == (6, 2, 4, 4)
        assert bundle["observed_T_camera_wrist"].shape == (6, 2, 4, 4)
        assert bundle["T_controller_wrist_M0"].shape == (2, 4, 4)
        assert not bundle["orientation_valid"].any()
        assert np.isnan(bundle["surface_source_pixel_uv"]).all()
        assert np.isnan(bundle["surface_patch_xyz_camera"]).all()
        assert not bundle["rgb_wrist_region_valid"].any()
        assert not bundle["source_pixel_traceable"].any()
        assert not bundle["mask_purity_valid"].any()
        assert not bundle["depth_rgb_registration_valid"].any()
        assert not bundle["local_depth_continuity_valid"].any()
        assert not bundle["contaminant_free"].any()
        assert set(bundle["surface_role"].tolist()[0]) == {"unknown"}
        invalid = ~bundle["observed_valid"]
        assert np.isnan(bundle["observed_T_camera_wrist"][invalid, :3, :]).all()
    on_disk = json.loads((output / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert on_disk == provenance


def test_refuses_to_clobber_complete_attempt(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    output = tmp_path / "attempt_0001"
    builder.build(processed_root=processed, experiment_root=experiments, output_root=output)
    npz_sha = _sha256(output / "WRIST_DUAL_INPUT.npz")
    with pytest.raises(builder.WiyhWristDualInputError, match="clobber"):
        builder.build(processed_root=processed, experiment_root=experiments, output_root=output)
    assert _sha256(output / "WRIST_DUAL_INPUT.npz") == npz_sha


def test_output_is_directly_accepted_by_dual_representation_preflight(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    output = tmp_path / "attempt_0001"
    builder.build(processed_root=processed, experiment_root=experiments, output_root=output)
    config = tmp_path / "producer_config.json"
    _write_json(
        config,
        {
            "schema_version": "chaoyang-wrist-dual-producer-config-v1",
            "fit_recording_ids": ["play_cards_0916_097", "play_cards_0916_098"],
            "forbidden_fit_recording_ids": ["play_cards_0916_101"],
            "selected_model": {"left": "M0_LEGACY", "right": "M0_LEGACY"},
            "fusion_heuristic_weight": 0.0,
            "development_numeric_correction_bound_mm": 30.0,
            "fps": 30.0,
            "review_image_domain": {
                "name": "PRECOMPUTED_HAWOR_SOURCE_UV",
                "width": 1280,
                "height": 960,
                "projection": "PRECOMPUTED_SOURCE_UV",
            },
            "temporal_authority": {
                "T_camera_controller_raw": "UNKNOWN_TEMPORAL_AUTHORITY",
                "observed_T_camera_wrist": "OFFLINE_NONCAUSAL",
                "visible_wrist_surface": "UNKNOWN_TEMPORAL_AUTHORITY",
            },
        },
    )
    result = producer.preflight(
        input_npz=output / "WRIST_DUAL_INPUT.npz",
        config_path=config,
    )
    assert result["status"] == "PASS_CPU_PREFLIGHT"
    assert result["frame_count"] == 6
    assert result["fit_recording_ids"] == ["play_cards_0916_097", "play_cards_0916_098"]
    assert result["forbidden_fit_recording_ids"] == ["play_cards_0916_101"]


def test_preflight_reads_fixed_sources_without_writing(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = builder.preflight(processed_root=processed, experiment_root=experiments)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["status"] == "PASS_CPU_SOURCE_PREFLIGHT"
    assert result["frame_count"] == 6
    assert result["orientation_valid_count"] == 0
    assert result["surface_point_candidate_count"] == 0
    assert result["writes_performed"] is False
    assert result["gpu_used"] is False
    assert before == after


def test_missing_pinned_observation_fails_without_output(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    spec = builder.SESSION_SPECS["play_cards_0916_101"]
    missing = (
        experiments
        / spec["experiment"]
        / "attempts"
        / "attempt_0001"
        / spec["hawor"]
    )
    missing.unlink()
    output = tmp_path / "attempt_0001"
    with pytest.raises(FileNotFoundError):
        builder.build(processed_root=processed, experiment_root=experiments, output_root=output)
    assert not output.exists()


def test_legacy_relation_drift_is_not_refit_or_hidden(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    record = (
        processed
        / "play_cards_0916_098"
        / "preprocess"
        / "all_data"
        / "00001"
        / "training_data.json"
    )
    value = json.loads(record.read_text(encoding="utf-8"))
    value["entities"]["hands"]["left"]["T_wrist_to_camera"][0][3] += 0.001
    _write_json(record, value)
    output = tmp_path / "attempt_0001"
    with pytest.raises(builder.WiyhWristDualInputError, match="legacy relation drift"):
        builder.build(processed_root=processed, experiment_root=experiments, output_root=output)
    assert not output.exists()


def test_terminal_status_drift_is_fail_closed(tmp_path: Path) -> None:
    processed, experiments = _fixture(tmp_path)
    spec = builder.SESSION_SPECS["play_cards_0916_097"]
    terminal = (
        experiments
        / spec["experiment"]
        / "attempts"
        / "attempt_0001"
        / "TERMINAL_RESULT.json"
    )
    _write_json(terminal, {"status": "PASS"})
    with pytest.raises(builder.WiyhWristDualInputError, match="terminal status drift"):
        builder.build(
            processed_root=processed,
            experiment_root=experiments,
            output_root=tmp_path / "attempt_0001",
        )
