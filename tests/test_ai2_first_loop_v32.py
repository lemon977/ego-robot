from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess

import jsonschema
import numpy as np
import pytest

from chaoyang.pipeline.ai2_independent_observability_v32 import (
    IndependentObservabilityError,
    compare_frozen_raw_bounded_v32,
    freeze_ai2_independent_observability,
    frozen_evaluation_mask,
)
from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import (
    CollisionDiagnostic,
    build_kai22_full_fk_sidecar,
    evaluate_kai22_r0_tiers_v32,
    reload_kai22_full_fk_sidecar,
    write_kai22_full_fk_sidecar,
)
from chaoyang.pipeline.kai22_full_fk_review_renderer_v1 import (
    Kai22ReviewError,
    frame_annotation_rows,
    render_kai22_full_fk_review,
)
from chaoyang.pipeline.kai22_clip_reconstruction_v1 import (
    Kai22ClipReconstructionError,
    reconstruct_kai22_clip_delta,
)
from chaoyang.pipeline.robot_renderer_cycles import parse_urdf
from chaoyang.pipeline.robot_visual_relative_v1 import HandLimits
from chaoyang.pipeline.temporal_authority_v1 import audit_suffix_invariance


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
ROOT = Path(__file__).resolve().parents[1]


def _validate(schema_name: str, value: dict) -> None:
    schema = json.loads((ROOT / "contracts" / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)


def _ref(path: str, sha: str) -> dict:
    return {"path": path, "bytes": 123, "sha256": sha}


def _selection_args() -> dict:
    frames = 160
    states = np.full(frames, "UNKNOWN", dtype="U32")
    states[:45] = "REGION_EVALUABLE"
    states[60:100] = "NOT_REGION_EVALUABLE"
    states[110:150] = "REGION_EVALUABLE"
    source_ids = [([] if state == "UNKNOWN" else ["rgb_review"]) for state in states]
    return {
        "session_id": "play_cards_0915_031",
        "frame_ids": np.arange(frames),
        "timestamps_s": np.arange(frames, dtype=np.float64) / 30.0,
        "region_states": states,
        "region_evidence_source_ids": source_ids,
        "evidence_sources": [
            {
                "source_id": "rgb_review",
                "family": "PHYSICAL_LEFT_RGB_REVIEW",
                "dependencies": [],
                "independent_from_evaluated_trajectory": True,
                "artifact_ref": _ref("not-opened-rgb-review.json", SHA_A),
            }
        ],
        "rgb_input_ref": _ref("not-opened-rgb.mp4", SHA_B),
        "raw_candidate_ref": _ref("not-opened-raw.npz", SHA_A),
        "bounded_candidate_ref": _ref("not-opened-bounded.npz", SHA_C),
        "producer_sha256": SHA_A,
        "config_sha256": SHA_B,
    }


def test_independent_selection_has_no_evaluated_trajectory_parameter() -> None:
    parameters = set(inspect.signature(freeze_ai2_independent_observability).parameters)
    assert not parameters.intersection({"hawor", "mano", "mano21", "q22", "pose_quality"})


def test_independent_selection_freezes_16_bins_two_bounded_windows_and_keeps_unknown() -> None:
    result = freeze_ai2_independent_observability(**_selection_args())
    assert result["status"] == "FROZEN_BEFORE_EVALUATED_TRAJECTORY_READ"
    assert len(result["review_frames"]) == 16
    assert len({row["source_frame"] for row in result["review_frames"]}) == 16
    assert [row["frame_count"] for row in result["continuous_windows"]] == [30, 30]
    assert result["continuous_windows"][0]["source_frame_start"] == 0
    assert result["continuous_windows"][1]["source_frame_start"] == 110
    assert len(result["frame_ledger"]) == 160
    assert result["state_counts"]["UNKNOWN"] == 35
    assert sum(row["region_state"] == "UNKNOWN" for row in result["frame_ledger"]) == 35
    assert result["evaluated_candidates"]["raw"]["disposition"] == "FROZEN_EXISTING_SHA"
    assert result["evaluated_candidates"]["bounded"]["disposition"] == "HOLD_NUMERIC_GATES"
    assert result["evaluated_candidates"]["candidate_content_read_during_selection"] is False
    assert result["evaluated_candidates"]["new_smoothed_candidate_created"] is False
    assert result["image_domain"]["source_index"] == 1
    assert result["image_domain"]["lens_remap_applied"] is False
    _validate("ai2_independent_observability_v32.schema.json", result)


@pytest.mark.parametrize("token", ["HaWoR", "MANO21", "q22_mask"])
def test_independent_selection_rejects_evaluated_data_dependencies(token: str) -> None:
    arguments = _selection_args()
    arguments["evidence_sources"][0]["dependencies"] = [token]
    with pytest.raises(IndependentObservabilityError, match="evaluated data"):
        freeze_ai2_independent_observability(**arguments)


def test_independent_selection_rejects_nonfixed_session() -> None:
    arguments = _selection_args()
    arguments["session_id"] = "play_cards_0915_999"
    with pytest.raises(IndependentObservabilityError, match="fixed first cohort"):
        freeze_ai2_independent_observability(**arguments)


def test_raw_and_bounded_comparison_reuses_sha_and_exact_frozen_set() -> None:
    selection = freeze_ai2_independent_observability(**_selection_args())
    frames = 160
    timestamps = np.arange(frames, dtype=np.float64) / 30.0
    template = np.zeros((frames, 21, 3), dtype=np.float64)
    time = np.arange(frames, dtype=np.float64)
    raw = template.copy()
    bounded = template.copy()
    raw[:, :, 1] = (0.0001 * time + 0.002 * ((-1.0) ** time))[:, None]
    bounded[:, :, 1] = (0.0001 * time + 0.0002 * ((-1.0) ** time))[:, None]
    result = compare_frozen_raw_bounded_v32(
        frozen_selection=selection,
        frame_ids=np.arange(frames),
        timestamps_s=timestamps,
        raw_candidate_ref=_ref("not-opened-raw.npz", SHA_A),
        bounded_candidate_ref=_ref("not-opened-bounded.npz", SHA_C),
        raw_joints=raw,
        bounded_joints=bounded,
        raw_valid=np.ones(frames, dtype=bool),
        bounded_valid=np.ones(frames, dtype=bool),
        raw_reprojection_error_px=np.full((frames, 21), 2.0),
        bounded_reprojection_error_px=np.full((frames, 21), 1.5),
        raw_observed=np.ones(frames, dtype=bool),
        bounded_observed=np.ones(frames, dtype=bool),
    )
    assert result["same_frozen_independent_frame_set"] is True
    assert result["bounded_disposition_before_comparison"] == "HOLD_NUMERIC_GATES"
    assert result["new_smoothed_candidate_created"] is False
    assert int(frozen_evaluation_mask(selection, np.arange(frames)).sum()) == 70
    with pytest.raises(IndependentObservabilityError, match="SHA lineage drift"):
        compare_frozen_raw_bounded_v32(
            frozen_selection=selection,
            frame_ids=np.arange(frames),
            timestamps_s=timestamps,
            raw_candidate_ref=_ref("wrong-raw.npz", SHA_B),
            bounded_candidate_ref=_ref("not-opened-bounded.npz", SHA_C),
            raw_joints=raw,
            bounded_joints=bounded,
            raw_valid=np.ones(frames, dtype=bool),
            bounded_valid=np.ones(frames, dtype=bool),
            raw_reprojection_error_px=np.full((frames, 21), 2.0),
            bounded_reprojection_error_px=np.full((frames, 21), 1.5),
            raw_observed=np.ones(frames, dtype=bool),
            bounded_observed=np.ones(frames, dtype=bool),
        )


def _write_chain_urdf(path: Path, side: str) -> None:
    links = "\n".join(f'  <link name="{side}_link_{index}"/>' for index in range(23))
    joints = []
    for index in range(22):
        joints.append(
            f'''  <joint name="{side}_joint_{index}" type="revolute">
    <parent link="{side}_link_{index}"/>
    <child link="{side}_link_{index + 1}"/>
    <origin xyz="0.01 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>'''
        )
    joint_xml = "\n".join(joints)
    path.write_text(
        f'<robot name="{side}">\n{links}\n{joint_xml}\n</robot>\n', encoding="utf-8"
    )


class _CollisionChecker:
    def check(self, physical_side: int, q22: np.ndarray) -> CollisionDiagnostic:
        collision = bool(q22[0] > 0.4 and physical_side == 1)
        return CollisionDiagnostic(
            known=True,
            non_adjacent_collision_pass=not collision,
            illegal_contact_count=int(collision),
            max_penetration_m=0.002 if collision else 0.0,
        )

    def close(self) -> None:
        pass


def _models(tmp_path: Path):
    paths = [tmp_path / "left.urdf", tmp_path / "right.urdf"]
    for side, path in zip(("left", "right"), paths, strict=True):
        _write_chain_urdf(path, side)
    return tuple(parse_urdf(path) for path in paths)


def _build_sidecar(tmp_path: Path):
    frames = 3
    q = np.zeros((frames, 2, 22), dtype=np.float64)
    q[1, 0, 3] = 1.2  # explicit hard-limit failure
    q[2, 1, 0] = 0.5  # scripted non-adjacent collision
    clip = np.zeros_like(q)
    clip[0, 0, 2] = 0.01
    return build_kai22_full_fk_sidecar(
        session_id="get_potato_chips_0915_007",
        frame_ids=np.arange(frames),
        timestamps_s=np.arange(frames, dtype=np.float64) / 30.0,
        q22=q,
        q22_valid=np.ones((frames, 2), dtype=bool),
        clip_delta=clip,
        models=_models(tmp_path),
        collision_checker=_CollisionChecker(),
        source_temporal_authority="OBSERVED_CURRENT",
        producer_sha256=SHA_A,
        config_sha256=SHA_B,
        input_sha256=SHA_C,
    )


def test_full_fk_sidecar_preserves_side_mapping_limits_clip_and_collision(tmp_path: Path) -> None:
    manifest, arrays = _build_sidecar(tmp_path)
    assert manifest["schema_version"] == "KAI22_FULL_FK_SIDECAR_V1"
    assert arrays["fk_root_relative"].shape == (3, 2, 23, 4, 4)
    assert arrays["joint_names"].shape == (2, 22)
    assert arrays["anatomical_side"].tolist() == ["right", "left"]
    assert arrays["physical_robot_side"].tolist() == ["left", "right"]
    assert arrays["clip_delta_linf"][0, 0] == pytest.approx(0.01)
    assert not arrays["hard_limit_pass"][1, 0]
    assert not arrays["fk_finite"][1, 0]
    assert arrays["collision_diagnostic_known"][2, 1]
    assert not arrays["non_adjacent_self_collision_pass"][2, 1]
    assert arrays["illegal_contact_count"][2, 1] == 1
    assert manifest["control_ground_truth"] is False
    assert manifest["physical_deployable"] is False
    _validate("kai22_full_fk_sidecar_v1.schema.json", manifest)


def test_full_fk_sidecar_npz_is_reloadable_and_sha_bound(tmp_path: Path) -> None:
    manifest, arrays = _build_sidecar(tmp_path)
    output = tmp_path / "output"
    manifest_path = output / "RESULT.json"
    arrays_path = output / "FULL_FK.npz"
    written = write_kai22_full_fk_sidecar(
        manifest_path=manifest_path,
        arrays_path=arrays_path,
        manifest=manifest,
        arrays=arrays,
    )
    reloaded_manifest, reloaded_arrays = reload_kai22_full_fk_sidecar(
        manifest_path, arrays_path
    )
    assert reloaded_manifest == written
    assert np.array_equal(reloaded_arrays["q22"], arrays["q22"])
    assert np.allclose(
        reloaded_arrays["fk_root_relative"], arrays["fk_root_relative"], equal_nan=True
    )


def _causal_audit() -> dict:
    current = {"rgb": np.zeros((2,), dtype=np.uint8)}
    return audit_suffix_invariance(
        full_current_inputs=current,
        truncated_current_inputs={"rgb": current["rgb"].copy()},
        required_fields=("rgb",),
    )


def _tier_args() -> dict:
    frames = 20
    shape = (frames, 2)
    return {
        "frame_ids": np.arange(frames),
        "q22_valid": np.ones(shape, dtype=bool),
        "fk_finite": np.ones(shape, dtype=bool),
        "hard_limit_pass": np.ones(shape, dtype=bool),
        "collision_known": np.ones(shape, dtype=bool),
        "non_adjacent_self_collision_pass": np.ones(shape, dtype=bool),
        "clip_delta_linf": np.zeros(shape, dtype=np.float64),
        "mapping_semantics_pass": True,
        "clip_masks_mapping_error": False,
        "independent_window_mask": np.ones(shape, dtype=bool),
        "stable_coverage_pass": True,
        "full_replay_reviewable": True,
        "development_threshold_authority": None,
        "temporal_authority_audit": _causal_audit(),
        "h50_current_input_fields": ("rgb",),
        "h50_validity_authority_pass": True,
        "h50_suffix_invariance_pass": True,
    }


def test_missing_threshold_authority_caps_result_at_kinematic_only() -> None:
    result = evaluate_kai22_r0_tiers_v32(**_tier_args())
    assert result["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert "MISSING_FROZEN_DEVELOPMENT_THRESHOLD_AUTHORITY" in result["levels"]["DEVELOPMENT_R0"]["reason_codes"]
    assert result["training_eligible"] is False
    _validate("kai22_r0_tiered_admission_v32.schema.json", result)


def test_collision_unknown_or_failure_blocks_kinematic_only() -> None:
    arguments = _tier_args()
    arguments["collision_known"][3, 0] = False
    arguments["non_adjacent_self_collision_pass"][3, 0] = False
    result = evaluate_kai22_r0_tiers_v32(**arguments)
    assert result["highest_admitted_level"] == "NONE"
    assert "SELF_COLLISION_DIAGNOSTIC_UNKNOWN" in result["levels"]["KINEMATIC_ONLY"]["reason_codes"]


def test_frozen_thresholds_allow_development_but_nonzero_clip_does_not() -> None:
    arguments = _tier_args()
    arguments["development_threshold_authority"] = {
        "status": "FROZEN_NUMERIC_THRESHOLDS",
        "minimum_contiguous_frames": 10,
        "minimum_coverage_fraction": 0.8,
        "authority_sha256": SHA_A,
    }
    arguments["h50_suffix_invariance_pass"] = False
    result = evaluate_kai22_r0_tiers_v32(**arguments)
    assert result["highest_admitted_level"] == "DEVELOPMENT_R0"
    arguments["clip_delta_linf"][5, 1] = 0.001
    result = evaluate_kai22_r0_tiers_v32(**arguments)
    assert result["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert "NONZERO_CLIP_DELTA" in result["levels"]["DEVELOPMENT_R0"]["reason_codes"]


def test_h50_requires_51_frame_consumer_window_and_temporal_authority() -> None:
    arguments = _tier_args()
    frames = 60
    shape = (frames, 2)
    arguments.update(
        frame_ids=np.arange(frames),
        q22_valid=np.ones(shape, dtype=bool),
        fk_finite=np.ones(shape, dtype=bool),
        hard_limit_pass=np.ones(shape, dtype=bool),
        collision_known=np.ones(shape, dtype=bool),
        non_adjacent_self_collision_pass=np.ones(shape, dtype=bool),
        clip_delta_linf=np.zeros(shape, dtype=np.float64),
        independent_window_mask=np.ones(shape, dtype=bool),
        development_threshold_authority={
            "status": "FROZEN_NUMERIC_THRESHOLDS",
            "minimum_contiguous_frames": 10,
            "minimum_coverage_fraction": 0.8,
            "authority_sha256": SHA_A,
        },
    )
    result = evaluate_kai22_r0_tiers_v32(**arguments)
    assert result["highest_admitted_level"] == "H50_READY"
    assert result["training_eligible"] is True


def test_clip_reconstruction_recovers_signed_delta_and_postclip_q() -> None:
    frames = 2
    joints = np.zeros((2, frames, 21, 3), dtype=np.float64)
    for side in range(2):
        for frame in range(frames):
            joints[side, frame, :, 0] = np.arange(21) * 0.01
            joints[side, frame, :, 1] = np.arange(21) ** 2 * 0.001
            joints[side, frame, :, 2] = side * 0.01 + frame * 0.001
    observed = np.ones((frames, 2), dtype=bool)
    valid = np.ones((frames, 2), dtype=bool)
    mapping = np.asarray([1, 0], dtype=np.int64)
    limits = tuple(
        HandLimits(
            lower=np.full(22, -0.05, dtype=np.float64),
            upper=np.full(22, 0.05, dtype=np.float64),
            neutral=np.zeros(22, dtype=np.float64),
        )
        for _ in range(2)
    )
    # First materialise the deterministic historical post-clip value.
    provisional = np.full((frames, 2, 22), np.nan, dtype=np.float64)
    from chaoyang.pipeline.kai22_clip_reconstruction_v1 import _desired_q22

    for anatomical, physical in enumerate(mapping.tolist()):
        for frame in range(frames):
            desired = _desired_q22(joints[anatomical, frame], limits[physical])
            provisional[frame, physical] = np.clip(
                desired, limits[physical].lower, limits[physical].upper
            )
    summary, arrays = reconstruct_kai22_clip_delta(
        joints_3d_camera=joints,
        source_observed_anatomical=observed,
        q22_postclip=provisional,
        q22_valid_physical=valid,
        human_to_physical=mapping,
        limits=limits,
    )
    assert summary["postclip_matches_frozen_q22_byte_exact"] is True
    assert summary["clip_delta_semantics"] == "POSTCLIP_Q22_MINUS_PRECLIP_DESIRED_Q22"
    assert summary["clipped_side_frames"] > 0
    assert np.array_equal(arrays["q22_reconstructed_postclip"], provisional)
    assert np.allclose(
        arrays["q22_preclip_desired"] + arrays["clip_delta"], provisional
    )


def test_clip_reconstruction_rejects_nonmatching_frozen_q() -> None:
    joints = np.zeros((2, 1, 21, 3), dtype=np.float64)
    joints[:, 0, :, 0] = np.arange(21) * 0.01
    joints[:, 0, :, 1] = np.arange(21) ** 2 * 0.001
    limits = tuple(
        HandLimits(-np.ones(22), np.ones(22), np.zeros(22)) for _ in range(2)
    )
    with pytest.raises(Kai22ClipReconstructionError, match="differs from frozen"):
        reconstruct_kai22_clip_delta(
            joints_3d_camera=joints,
            source_observed_anatomical=np.ones((1, 2), dtype=bool),
            q22_postclip=np.zeros((1, 2, 22), dtype=np.float64),
            q22_valid_physical=np.ones((1, 2), dtype=bool),
            human_to_physical=np.asarray([1, 0]),
            limits=limits,
        )


class _NoCollisionChecker:
    def check(self, physical_side: int, q22: np.ndarray) -> CollisionDiagnostic:
        return CollisionDiagnostic(
            known=True,
            non_adjacent_collision_pass=True,
            illegal_contact_count=0,
            max_penetration_m=0.0,
        )

    def close(self) -> None:
        pass


def test_review_annotation_keeps_invalid_clip_unknown_and_caps_tier() -> None:
    rows = frame_annotation_rows(
        frame_id=12,
        timestamp_s=0.4,
        highest_tier="KINEMATIC_ONLY",
        valid=np.asarray([True, False]),
        clip_delta=np.vstack((np.full(22, 0.01), np.full(22, np.nan))),
    )
    assert "CURRENT HIGHEST TIER: KINEMATIC_ONLY" in rows
    assert any("NOT DEVELOPMENT_R0" in row for row in rows)
    assert "physical left: valid=true | clipped=22/22" in rows[3]
    assert rows[4] == "physical right: valid=false | clip_delta=UNKNOWN"


def test_complete_frame_review_renders_and_fully_decodes(tmp_path: Path) -> None:
    models = _models(tmp_path)
    frames = 3
    q22 = np.zeros((frames, 2, 22), dtype=np.float64)
    clip_delta = np.zeros_like(q22)
    clip_delta[1, 0, 0] = 0.02
    manifest, arrays = build_kai22_full_fk_sidecar(
        session_id="get_potato_chips_0915_007",
        frame_ids=np.arange(frames),
        timestamps_s=np.arange(frames, dtype=np.float64) / 30.0,
        q22=q22,
        q22_valid=np.ones((frames, 2), dtype=bool),
        clip_delta=clip_delta,
        models=models,
        collision_checker=_NoCollisionChecker(),
        source_temporal_authority="OBSERVED_CURRENT",
        producer_sha256=SHA_A,
        config_sha256=SHA_B,
        input_sha256=SHA_C,
    )
    full_fk = tmp_path / "FULL_FK.npz"
    clip = tmp_path / "CLIP.npz"
    tier = tmp_path / "TIER.json"
    source = tmp_path / "source.mp4"
    destination = tmp_path / "review.mp4"
    np.savez_compressed(full_fk, **arrays)
    np.savez_compressed(clip, clip_delta=clip_delta)
    tier.write_text(
        json.dumps({"highest_admitted_level": "KINEMATIC_ONLY"}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=160x120:r=30",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    receipt = render_kai22_full_fk_review(
        session_id="get_potato_chips_0915_007",
        source_video=source,
        full_fk_npz=full_fk,
        clip_npz=clip,
        tier_result=tier,
        destination=destination,
        models=models,
    )
    assert manifest["schema_version"] == "KAI22_FULL_FK_SIDECAR_V1"
    assert receipt["frame_count"] == frames
    assert receipt["decode"]["status"] == "PASS_FULL_DECODE"
    assert receipt["decode"]["frames"] == frames
    assert receipt["displayed"]["highest_tier"] == "KINEMATIC_ONLY"
    assert receipt["displayed"]["not_development_r0"] is True
    assert receipt["new_thumb_candidate_created"] is False
    assert receipt["diagnostics"]["clipped_joint_values"] == 1
    _validate("kai22_full_fk_review_v1.schema.json", receipt)

    forbidden_tier = tmp_path / "FORBIDDEN_TIER.json"
    forbidden_tier.write_text(
        json.dumps({"highest_admitted_level": "DEVELOPMENT_R0"}),
        encoding="utf-8",
    )
    with pytest.raises(Kai22ReviewError, match="exactly KINEMATIC_ONLY"):
        render_kai22_full_fk_review(
            session_id="get_potato_chips_0915_007",
            source_video=source,
            full_fk_npz=full_fk,
            clip_npz=clip,
            tier_result=forbidden_tier,
            destination=tmp_path / "forbidden.mp4",
            models=models,
        )
