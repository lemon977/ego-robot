from __future__ import annotations

from chaoyang.governance.robot15h_task_specs_v1 import (
    FOUNDATION_WEIGHT,
    HAWOR_WEIGHT,
    SAM31_WEIGHT,
    TASK_ORDER,
    WEIGHTS,
    build_packet,
    frozen_dag,
)
from chaoyang.ops.run_0915_robot15h_window_start_inventory_v1 import split_for_source_group


def test_first_packet_is_cpu_only_bounded_inventory() -> None:
    packet = build_packet(TASK_ORDER[0])
    assert packet["weights"] == "ABSENT"
    assert len(packet["read_set"]) == 8
    assert "0916_highview_FORBIDDEN" in packet["prerequisites"]
    assert "lens_undistortion_FORBIDDEN" in packet["prerequisites"]


def test_each_model_node_binds_exactly_one_expected_weight() -> None:
    for task_id, weights in WEIGHTS.items():
        if weights == "ABSENT":
            continue
        assert len(weights) == 1
        if "hawor" in task_id:
            assert weights == [HAWOR_WEIGHT]
        elif "sam31" in task_id:
            assert weights == [SAM31_WEIGHT]
        elif "foundationstereo" in task_id:
            assert weights == [FOUNDATION_WEIGHT]
        else:
            raise AssertionError(task_id)


def test_frozen_dag_keeps_strict_contact_and_release_dependencies() -> None:
    nodes = {row["task_id"]: row for row in frozen_dag()["nodes"]}
    contact = nodes["0915_robot15h_contact_dual_evidence_v1"]["dependencies"]
    assert "0915_robot15h_geometry_object6d_wave0_v1" in contact
    assert "0915_robot15h_interaction_occlusion_v1" in contact
    release = nodes["0915_robot15h_window_release_audit_v1"]["dependencies"]
    assert set(release) == set(TASK_ORDER[:-1])


def test_hawor_runtime_recovery_is_bounded_and_gates_r0() -> None:
    recovery_id = "0915_robot15h_hawor_wave0_recovery_v1"
    recovery = build_packet(recovery_id)
    assert recovery["dag_dependencies"] == ["0915_robot15h_hawor_wave0_v1"]
    assert recovery["weights"] == [HAWOR_WEIGHT]
    assert "same_T0_and_frozen_W0" in recovery["prerequisites"]
    r0 = build_packet("0915_robot15h_kai22_r0_wave0_v1")
    assert r0["dag_dependencies"] == [recovery_id]


def test_kai22_r0_is_cpu_only_and_preserves_missing_policy() -> None:
    packet = build_packet("0915_robot15h_kai22_r0_wave0_v1")
    assert packet["weights"] == "ABSENT"
    assert "gpu_FORBIDDEN" in packet["prerequisites"]
    assert "missing_frames_remain_invalid" in packet["prerequisites"]
    assert "bounded_v2/RESULT.json" in packet["required_outputs"]


def test_kai22_r0_runner_declares_frozen_human_to_physical_mapping() -> None:
    from chaoyang.ops.run_0915_robot15h_kai22_r0_wave0_v1 import HUMAN_TO_PHYSICAL

    assert HUMAN_TO_PHYSICAL.tolist() == [1, 0]


def test_source_group_split_is_stable() -> None:
    source_group = "0915:cards_120_0915:031:82abe0a14b424a66b0d83f6c16d460e4"
    assert split_for_source_group(source_group) == ("development", 6)


def test_virtual_r2_binds_development_motion_contract_and_review_manifest() -> None:
    packet = build_packet("0915_robot15h_robot_virtual_arm_v1")
    assert packet["weights"] == "ABSENT"
    assert "contracts/robot15h_virtual_installation_r0_motion_v1.schema.json" in packet["read_set"]
    assert "manifests/hardware/robot15h_virtual_installation_r0_motion_v1.json" in packet["read_set"]
    assert "REVIEW_MANIFEST.json" in packet["required_outputs"]
    assert "measured_TCP_mount_and_camera_base_calibration_ABSENT" in packet["prerequisites"]


def test_interaction_is_real_visible_surface_node_not_blocker_placeholder() -> None:
    packet = build_packet("0915_robot15h_interaction_occlusion_v1")
    assert packet["weights"] == "ABSENT"
    assert "src/chaoyang/ops/run_0915_robot15h_interaction_wave0_v1.py" in packet["read_set"]
    assert "INTERACTION_EVIDENCE_LEDGER.json" in packet["required_outputs"]
    assert "consumer_admitted_SAM_hand_role_required" in packet["prerequisites"]
    assert "fragmented_support_largest_component_fraction_at_least_0_98" in packet["prerequisites"]


def test_contact_separates_strict_evidence_from_hypothesis() -> None:
    packet = build_packet("0915_robot15h_contact_dual_evidence_v1")
    assert packet["weights"] == "ABSENT"
    assert "src/chaoyang/ops/run_0915_robot15h_contact_wave0_v1.py" in packet["read_set"]
    assert "CONTACT_EVIDENCE_LEDGER.json" in packet["required_outputs"]
    assert "CONTACT_HYPOTHESIS_LEDGER.json" in packet["required_outputs"]
    assert "uncertainty_must_not_expand_distance_gate" in packet["prerequisites"]
    assert "hypothesis_includes_no_contact_control" in packet["prerequisites"]


def test_r1_counterfactual_cannot_self_certify_or_move_wrist() -> None:
    packet = build_packet("0915_robot15h_robot_relative_refinement_v1")
    assert packet["weights"] == "ABSENT"
    assert "src/chaoyang/ops/run_0915_robot15h_r1_hypothesis_wave0_v1.py" in packet["read_set"]
    assert "contracts/kai22_r1_hypothesis_v1.schema.json" in packet["read_set"]
    assert "R1_ELIGIBILITY_LEDGER.json" in packet["required_outputs"]
    assert "wrist_translation_and_orientation_frozen_without_independent_metric_placement" in packet["prerequisites"]
    assert "optimization_metric_not_adoption_evidence" in packet["prerequisites"]


def test_release_control_nodes_are_not_placeholders() -> None:
    for task_id in (
        "0915_robot15h_release_candidate_v1",
        "0915_robot15h_window_release_audit_v1",
    ):
        packet = build_packet(task_id)
        assert packet["weights"] == "ABSENT"
        assert "src/chaoyang/ops/run_0915_robot15h_release_control_v1.py" in packet["read_set"]
        assert "CAPABILITY_MATRIX.json" in packet["required_outputs"]
        assert "DEADLINE_AUDIT.json" in packet["required_outputs"]
    audit = build_packet("0915_robot15h_window_release_audit_v1")
    assert "FINAL_AUDIT.json" in audit["required_outputs"]
    assert "REFERENCE_AUDIT.json" in audit["required_outputs"]
