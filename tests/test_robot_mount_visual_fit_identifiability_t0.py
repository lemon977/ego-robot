from pathlib import Path

from tools.audit_robot_mount_visual_fit_identifiability_t0 import build_report


PROJECT = Path(__file__).resolve().parents[1]


def test_exact_458_frame_input_is_mount_invariant_under_legal_composition() -> None:
    report = build_report(PROJECT)
    assert report["status"] == "HOLD_A_CLASS_INPUT_SEMANTICS_MOUNT_NOT_IDENTIFIABLE"
    assert report["fit_executed"] is False
    assert report["mount_candidate_emitted"] is False
    assert report["inputs"]["frame_count"] == 458
    assert report["inputs"]["independent_tool_or_arm_fields_absent"] == [
        "q_arm",
        "T_camera_base",
        "T_camera_tool",
        "tool_T_camera",
    ]
    for side in ("left", "right"):
        evidence = report["exact_counterexample"]["by_side"][side]
        assert evidence["max_recomposed_hand_matrix_abs_difference"] < 1e-12
        assert evidence["max_recomposed_local_point_abs_difference_m"] < 1e-12


def test_frozen_wrist_is_already_hawor_wrist_not_an_independent_tool_pose() -> None:
    report = build_report(PROJECT)
    distributions = report["frozen_field_semantics_evidence"][
        "wrist_T_camera_origin_vs_hawor_joint0_residual_px"
    ]
    assert distributions["left"]["max"] < 1e-3
    assert distributions["right"]["max"] < 1e-3
    assert report["forbidden_reinterpretation_counterexample"]["by_side"]["left"][
        "constant_rotation_change_deg"
    ] == 37.0


def test_governance_and_tool_definition_remain_visual_only_and_shared() -> None:
    report = build_report(PROJECT)
    assert report["mount_provenance"] == "PROVISIONAL_MOUNT_VISUAL_ONLY"
    assert report["contact_infeasible"] == "UNMEASURED"
    assert report["advancement_authorized"] is False
    assert report["tool_definition_consistency"]["observed_z_m"] == [0.145, 0.145]
    assert report["execution_counters"]["gpu_calls"] == 0
    assert report["execution_counters"]["labels_blind_025_clean_processed_reads"] == 0
