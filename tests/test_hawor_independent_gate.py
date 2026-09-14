from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import numpy as np

from pipeline.hawor_independent_gate import (
    evaluate_hawor_independent_gate,
    longest_missing_run_in_denominator,
    validate_independent_review,
)
from tools.evaluate_hawor_independent_gate import _schema_errors


ROOT = Path(__file__).resolve().parents[1]


def arrays(frame_count: int = 30) -> dict[str, np.ndarray]:
    joints_3d_camera = np.zeros((2, frame_count, 21, 3), dtype=np.float32)
    joints_3d_world = np.zeros_like(joints_3d_camera)
    joints_2d = np.zeros((2, frame_count, 21, 2), dtype=np.float32)
    for side in range(2):
        for frame in range(frame_count):
            for joint in range(21):
                xyz = np.array(
                    [side * 0.30 + joint * 0.002 + frame * 0.001, joint * 0.001, 0.55],
                    dtype=np.float32,
                )
                joints_3d_camera[side, frame, joint] = xyz
                joints_3d_world[side, frame, joint] = xyz
                joints_2d[side, frame, joint] = (
                    200 + side * 500 + joint * 2 + frame,
                    300 + joint,
                )
    observed = np.ones((2, frame_count), dtype=bool)
    return {
        "joints_3d_camera": joints_3d_camera,
        "joints_3d_world": joints_3d_world,
        "joints_2d": joints_2d,
        "root_orient_camera": np.broadcast_to(np.eye(3), (2, frame_count, 3, 3)).copy(),
        "observed": observed,
        "provenance": np.full((2, frame_count), "OBSERVED", dtype="U12"),
        "detector_confidence": np.full((2, frame_count), 0.9, dtype=np.float32),
        "original_frame_indices": np.arange(frame_count, dtype=np.int32),
        "fps": np.asarray(30.0),
        "mano_joint_names": np.asarray(
            [
                "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                "index_mcp", "index_pip", "index_dip", "index_tip",
                "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
            ],
            dtype="U20",
        ),
        "anatomical_side_names": np.asarray(["left", "right"], dtype="U8"),
        "mano_wrist_index": np.asarray(0, dtype=np.int32),
        "mano_tip_indices": np.asarray([4, 8, 12, 16, 20], dtype=np.int32),
        "mano_mcp_indices": np.asarray([2, 5, 9, 13, 17], dtype=np.int32),
    }


def hawor_result(frame_count: int = 30) -> dict:
    side = {"mask_numeric_gate": True, "robot_numeric_gate": True}
    return {
        "task_id": "chips",
        "session_id": "fixture_001",
        "frame_count": frame_count,
        "sides": {"left": deepcopy(side), "right": deepcopy(side)},
    }


def source_gate(g1: str = "PASS_NUMERIC_CAMERA_TIMELINE") -> dict:
    hand = {"all_21_valid_fraction": 1.0}
    return {
        "task_id": "chips",
        "session_id": "fixture_001",
        "gates": {
            "G0_SOURCE_MEDIA": "PASS",
            "G1_CAMERA_WORLD": g1,
            "G2_PICO21_PRELIMINARY": "PASS_NUMERIC_NOT_MANO21",
        },
        "hands": {"left": deepcopy(hand), "right": deepcopy(hand)},
    }


def valid_review(frame_count: int = 30) -> dict:
    return {
        "schema_version": "hawor-independent-review-v1",
        "task_id": "chips",
        "session_id": "fixture_001",
        "frame_count": frame_count,
        "provider": {
            "authority": "INDEPENDENT_QA",
            "family": "frozen-human-label-replay",
            "provider_id": "reviewer-and-tool-v1",
            "independent_from_hawor": True,
        },
        "evidence": [
            {"role": "FULL_TIMELINE_IDENTITY_REVIEW", "path": "identity.json", "sha256": "0" * 64},
            {"role": "HAND_SILHOUETTE_LABELS", "path": "silhouette.npz", "sha256": "1" * 64},
            {"role": "CONTOUR_METRICS", "path": "contour.json", "sha256": "2" * 64},
            {"role": "EXPECTED_ACTIVE_LABELS", "path": "active.json", "sha256": "3" * 64},
            {"role": "MISSING_VISIBILITY_LABELS", "path": "visibility.json", "sha256": "4" * 64},
        ],
        "expected_active": {
            "status": "PASS",
            "authority": "CAPTURE",
            "reason_codes": [],
            "per_side_intervals": {"left": [[0, frame_count]], "right": [[0, frame_count]]},
        },
        "anatomical_identity": {
            "status": "PASS",
            "reason_codes": [],
            "full_timeline_reviewed": True,
            "reviewed_frame_count": frame_count,
            "identity_switch_frames": [],
            "duplicate_track_frames": [],
            "collapse_frames": [],
        },
        "contour_projection": {
            "status": "PASS",
            "reason_codes": [],
            "method": "FROZEN_HUMAN_SILHOUETTE_LABELS",
            "mask_distance_threshold_px": 20,
            "robot_distance_threshold_px": 12,
            "sampled_frame_indices": list(range(24)),
            "per_side": {
                "left": {
                    "mask_coverage_fraction_p05_at_20px": 0.95,
                    "robot_coverage_fraction_p05_at_12px": 0.95,
                },
                "right": {
                    "mask_coverage_fraction_p05_at_20px": 0.95,
                    "robot_coverage_fraction_p05_at_12px": 0.95,
                },
            },
        },
        "missing_visibility": {
            "status": "PASS",
            "reason_codes": [],
            "all_missing_frames_classified": True,
            "classified_missing_frames": {"left": [], "right": []},
            "boundary_support_rule": "INDEPENDENT_SILHOUETTE_AT_LEAST_19_OF_21_JOINTS_IN_FRAME",
            "per_side_visibility_intervals": {
                "left": {
                    "visible_required": [[0, frame_count]],
                    "boundary_supported": [],
                    "occluded_or_out_of_frame": [],
                },
                "right": {
                    "visible_required": [[0, frame_count]],
                    "boundary_supported": [],
                    "occluded_or_out_of_frame": [],
                },
            },
        },
    }


def evaluate(**kwargs):
    return evaluate_hawor_independent_gate(
        hawor_result=kwargs.pop("hawor_result", hawor_result()),
        source_gate=kwargs.pop("source_gate", source_gate()),
        arrays=kwargs.pop("arrays", arrays()),
        **kwargs,
    )


def test_numeric_pass_cannot_self_approve_identity_or_contour():
    result = evaluate()
    assert result["routes"]["mask_numeric_candidate"] is True
    assert result["routes"]["mask_formal_seed_ready"] is False
    assert result["gates"]["anatomical_identity_continuity"]["status"] == "HOLD"
    assert result["gates"]["independent_contour_mask"]["status"] == "HOLD"
    assert result["routes"]["human_review_required"] is True


def test_valid_independent_full_timeline_and_contour_review_can_admit():
    result = evaluate(independent_review=valid_review())
    assert result["input_validation"]["valid"] is True
    assert result["routes"]["mask_formal_seed_ready"] is True
    assert result["routes"]["robot_formal_motion_seed_ready"] is True
    assert result["status"] == "PASS_FORMAL_HAWOR_SEED"


def test_hawor_family_cannot_masquerade_as_independent_provider():
    review = valid_review()
    review["provider"]["family"] = "HaWoR detector replay"
    errors = validate_independent_review(
        review, frame_count=30, task_id="chips", session_id="fixture_001"
    )
    assert "provider.family must be independent from HaWoR" in errors
    result = evaluate(independent_review=review)
    assert result["routes"]["mask_formal_seed_ready"] is False
    assert "INDEPENDENT_REVIEW_INVALID" in result["gates"]["independent_contour_mask"]["reason_codes"]


def test_self_projection_without_frozen_labels_is_rejected():
    review = valid_review()
    review["contour_projection"]["method"] = "NOT_EVALUATED"
    errors = validate_independent_review(
        review, frame_count=30, task_id="chips", session_id="fixture_001"
    )
    assert any("frozen independent silhouette" in error for error in errors)


def test_mask_20px_and_robot_12px_contour_contract_cannot_drift():
    review = valid_review()
    review["contour_projection"]["mask_distance_threshold_px"] = 12
    errors = validate_independent_review(
        review, frame_count=30, task_id="chips", session_id="fixture_001"
    )
    assert "contour PASS requires mask_distance_threshold_px=20" in errors


def test_world_g1_hold_does_not_block_image_mask_route():
    result = evaluate(
        source_gate=source_gate("HOLD_CAPTURE_MULTI_SOURCE_WORLD_EPOCH_UNPROVEN"),
        independent_review=valid_review(),
    )
    assert result["routes"]["mask_formal_seed_ready"] is True
    assert result["routes"]["robot_world_route_ready"] is False
    assert result["routes"]["robot_formal_motion_seed_ready"] is False
    assert result["attribution"]["world_route_only"]["owner"] == "CAPTURE_WORLD"


def test_pico_g2_failure_is_advisory_to_rgb_hawor():
    source = source_gate()
    source["gates"]["G2_PICO21_PRELIMINARY"] = "FAIL_CAPTURE_PICO_MOTION_OR_GAPS"
    result = evaluate(source_gate=source, independent_review=valid_review())
    assert result["routes"]["mask_formal_seed_ready"] is True
    assert result["gates"]["pico21_g2_advisory"]["status"] == "FAIL"


def test_interpolation_or_unknown_provenance_fails_integrity():
    data = arrays()
    data["provenance"][0, 5] = "INTERPOLATED"
    data["observed"][0, 5] = False
    result = evaluate(arrays=data)
    assert result["gates"]["direct_vs_interpolated_provenance"]["status"] == "FAIL"
    assert result["routes"]["mask_numeric_candidate"] is False


def test_long_missing_gap_is_authority_hold_without_visible_active_denominator():
    data = arrays()
    data["observed"][1, 4:14] = False
    data["provenance"][1, 4:14] = "MISSING"
    reported = hawor_result()
    reported["sides"]["right"]["mask_numeric_gate"] = False
    reported["sides"]["right"]["robot_numeric_gate"] = False
    result = evaluate(arrays=data, hawor_result=reported)
    assert result["gates"]["numeric_mask_seed"]["status"] == "FAIL"
    assert result["attribution"]["primary"][0] == {
        "owner": "AUTHORITY",
        "reason_code": "NUMERIC_FAILURE_DENOMINATOR_OR_VISIBILITY_UNRESOLVED",
    }
    assert result["status"] == "HOLD_HUMAN_REVIEW_REQUIRED"


def test_numeric_failure_on_authorized_visible_scope_is_hawor_failure():
    data = arrays()
    data["observed"][1, 4:14] = False
    data["provenance"][1, 4:14] = "MISSING"
    review = valid_review()
    review["missing_visibility"].update(
        {
            "status": "PASS",
            "reason_codes": [],
            "all_missing_frames_classified": True,
            "classified_missing_frames": {"left": [], "right": list(range(4, 14))},
        }
    )
    reported = hawor_result()
    reported["sides"]["right"]["mask_numeric_gate"] = False
    reported["sides"]["right"]["robot_numeric_gate"] = False
    result = evaluate(arrays=data, hawor_result=reported, independent_review=review)
    assert result["status"] == "FAIL_HAWOR_OR_CAPTURE"
    assert result["attribution"]["primary"][0]["owner"] == "HAWOR_ALGORITHM"


def test_out_of_frame_gap_is_excluded_by_independent_visibility_denominator():
    data = arrays()
    data["observed"][1, 4:14] = False
    data["provenance"][1, 4:14] = "MISSING"
    review = valid_review()
    review["missing_visibility"]["classified_missing_frames"]["right"] = list(range(4, 14))
    review["missing_visibility"]["per_side_visibility_intervals"]["right"] = {
        "visible_required": [[0, 4], [14, 30]],
        "boundary_supported": [],
        "occluded_or_out_of_frame": [[4, 14]],
    }
    reported = hawor_result()
    reported["sides"]["right"]["mask_numeric_gate"] = False
    reported["sides"]["right"]["robot_numeric_gate"] = False

    result = evaluate(arrays=data, hawor_result=reported, independent_review=review)

    assert result["gates"]["numeric_mask_seed"]["status"] == "PASS"
    assert result["gates"]["numeric_robot_seed"]["status"] == "PASS"
    assert result["gates"]["numeric_mask_seed"]["metrics"]["denominator"] == "EXPECTED_VISIBLE_ACTIVE"
    assert result["gates"]["occlusion_out_of_frame"]["metrics"]["per_side"]["right"][
        "excluded_missing_frames"
    ] == list(range(4, 14))
    assert not any(row["owner"] == "HAWOR_ALGORITHM" for row in result["attribution"]["primary"])
    assert result["status"] == "PASS_FORMAL_HAWOR_SEED"


def test_missing_gap_does_not_bridge_separate_visibility_intervals():
    observed = np.ones(12, dtype=bool)
    observed[[1, 2, 8, 9]] = False
    denominator = np.zeros(12, dtype=bool)
    denominator[0:4] = True
    denominator[7:11] = True
    assert longest_missing_run_in_denominator(observed, denominator) == 2


def test_missing_visibility_review_must_match_exact_missing_frames():
    data = arrays()
    data["observed"][1, 5:7] = False
    data["provenance"][1, 5:7] = "MISSING"
    review = valid_review()
    review["missing_visibility"]["classified_missing_frames"]["right"] = [5]
    reported = hawor_result()
    result = evaluate(arrays=data, hawor_result=reported, independent_review=review)
    assert result["gates"]["occlusion_out_of_frame"]["status"] == "HOLD"
    assert "MISSING_FRAME_CLASSIFICATION_NOT_EXACT" in result["gates"]["occlusion_out_of_frame"]["reason_codes"]


def test_mask_and_robot_contour_thresholds_stay_separate():
    review = valid_review()
    review["contour_projection"]["per_side"]["left"][
        "mask_coverage_fraction_p05_at_20px"
    ] = 0.87
    review["contour_projection"]["per_side"]["right"][
        "mask_coverage_fraction_p05_at_20px"
    ] = 0.89
    review["contour_projection"]["per_side"]["left"][
        "robot_coverage_fraction_p05_at_12px"
    ] = 0.87
    review["contour_projection"]["per_side"]["right"][
        "robot_coverage_fraction_p05_at_12px"
    ] = 0.89
    result = evaluate(independent_review=review)
    assert result["gates"]["independent_contour_mask"]["status"] == "PASS"
    assert result["gates"]["independent_contour_robot"]["status"] == "HOLD"
    assert result["routes"]["mask_formal_seed_ready"] is True
    assert result["routes"]["robot_formal_motion_seed_ready"] is False


def test_review_input_and_gate_output_match_json_schemas():
    input_schema = json.loads(
        (ROOT / "contracts/quality/hawor_independent_review_v1.schema.json").read_text()
    )
    output_schema = json.loads(
        (ROOT / "contracts/quality/hawor_independent_gate_output_v1.schema.json").read_text()
    )
    review = valid_review()
    jsonschema.validate(review, input_schema)
    jsonschema.validate(evaluate(independent_review=review), output_schema)


def test_cli_schema_validation_rejects_extra_review_fields():
    input_schema = json.loads(
        (ROOT / "contracts/quality/hawor_independent_review_v1.schema.json").read_text()
    )
    review = valid_review()
    review["uncontracted_claim"] = "PASS"
    errors = _schema_errors(review, input_schema)
    assert errors
    assert "Additional properties are not allowed" in errors[0]


def test_review_evidence_hash_errors_fail_closed():
    result = evaluate(
        independent_review=valid_review(),
        review_errors=["evidence[0] sha256 mismatch"],
    )
    assert result["input_validation"]["valid"] is False
    assert result["routes"]["mask_formal_seed_ready"] is False
