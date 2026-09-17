from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / "chaoyang" / "ops" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contact = load_tool("build_contact_readiness_matrix_r3")
mask = load_tool("build_mask_failure_clusters_r3")
first_blocker = contact.first_blocker
classify = mask.classify
semantic_flags = mask.semantic_flags


def test_role_flags_keep_unmeasured_categories_unknown() -> None:
    payload = {
        "hard_gates": {
            "tracker_offscreen_or_unobserved_is_empty": True,
            "tracker_expected_visible_coverage_fraction_at_least_0p80": False,
        }
    }
    flags = semantic_flags("ROLE_MASK", payload)
    assert flags["OFFSCREEN_NONEMPTY"] == "FALSE"
    assert flags["REENTRY_IDENTITY_OR_COVERAGE_FAILURE"] == "TRUE"
    assert flags["LEFT_RIGHT_IDENTITY_SWAP"] == "UNKNOWN_NOT_MEASURED_BY_CURRENT_RECEIPT"
    assert flags["CONTACT_BOUNDARY_LEAKAGE"] == "UNKNOWN_NOT_MEASURED_BY_CURRENT_RECEIPT"


def test_role_classification_is_exclusive_and_evidence_bounded() -> None:
    payload = {
        "status": "TERMINAL_GRADE_C",
        "hard_gates": {
            "tracker_offscreen_or_unobserved_is_empty": True,
            "tracker_expected_visible_coverage_fraction_at_least_0p80": False,
        },
        "metrics": {
            "tracker_reentry_summary": {
                "right_tracker": {
                    "expected_visible_denominator": 20,
                    "present_on_expected_visible": 2,
                }
            }
        },
    }
    cluster, _, severity = classify("ROLE_MASK", {"lineage": "FRESH"}, payload)
    assert cluster == "REENTRY_IDENTITY_OR_COVERAGE_FAILURE"
    assert severity == 18


def test_object_union_precedes_reentry_when_both_are_explicit_failures() -> None:
    payload = {
        "frame_count": 100,
        "gates": {
            "poker_same_id_pre_and_post_action": False,
            "three_instances_disjoint": False,
            "no_union_or_identity_switch_fabricated": False,
        },
    }
    cluster, _, _ = classify("OBJECT_MASK", {"lineage": "FRESH"}, payload)
    # Precedence is explicit and stable; reentry is evaluated before union.
    assert cluster == "REENTRY_IDENTITY_OR_COVERAGE_FAILURE"


def test_upstream_and_runtime_are_not_mislabeled_as_mask_quality() -> None:
    assert classify(
        "ROLE_MASK",
        {"lineage": "NOT_RUN_UPSTREAM_C"},
        {"reason": "NOT_RUN_UPSTREAM_HAWOR_C"},
    )[0] == "UPSTREAM_HAWOR_C_NOT_RUN"
    assert classify(
        "ROLE_MASK",
        {"lineage": "FRESH_ROLE_EXCEPTION_C"},
        {"reason": "ROLE_RUNNER_EXCEPTION: video frame/fps mismatch"},
    )[0] == "INFRASTRUCTURE_INPUT_CONTRACT_FAILURE"


def test_contact_primary_blocker_is_unique_and_ordered() -> None:
    base = {
        "role_mask_ready": True,
        "object_mask_ready": True,
        "depth_ready": True,
        "object6d_ready": True,
        "object6d_direct_observed_only": True,
        "coordinate_domain_consistent": True,
        "direct_hand_object_overlap_frames": 10,
    }
    assert first_blocker(base) == "R3_PER_FINGER_CONTACT_PRODUCER_NOT_RUN"
    assert first_blocker({**base, "coordinate_domain_consistent": False}) == "HAND_OBJECT_CAMERA_COORDINATE_DOMAIN_NOT_CLOSED"
    assert first_blocker({**base, "role_mask_ready": False, "depth_ready": False}) == "ROLE_MASK_NOT_READY"
