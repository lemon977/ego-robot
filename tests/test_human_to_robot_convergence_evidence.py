import json
from pathlib import Path


REPO = Path("/mnt/workspace/code/chaoyang")
ROOT = REPO / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_rejected_direction_did_not_expand_or_hide_denominator():
    result = load("lanes/motion_product/partial_direction_031/wave0/RESULT.json")
    assert result["frozen_denominator"] == {"timeline": 149, "right_input": 102, "left_input": 0}
    assert result["window"]["attempted"] == 16
    assert result["window"]["gate_pass_count"] == 0
    assert result["full_expansion"]["executed"] is False


def test_contact_counts_distinguish_qualification_geometry_and_r1():
    result = load("lanes/scene/contact_screen_031/wave0/RESULT.json")
    assert result["counts"]["qualification_checked"] == 510
    assert result["counts"]["geometry_screened"] == 158
    assert result["counts"]["strict_contact_admissible"] == 0
    assert result["counts"]["r1_executed_windows"] == 0


def test_adapter_visual_mesh_does_not_imply_collision_pass():
    result = load("lanes/motion_product/adapter_collision/wave0/RESULT.json")
    assert result["visual_geometry"]["status"] == "REAL_STEP_DERIVED_VISUAL_MESH"
    assert result["collision_geometry"]["queries_executed"] == 0
    assert result["quality"] == "NOT_EVALUATED_NO_APPROVED_COLLISION_SEMANTICS"


def test_same_surface_metric_is_not_fabricated_from_screen_pixels():
    result = load("lanes/scene/occlusion_same_surface_031/wave0/RESULT.json")
    assert result["pixel_screen_diagnostic"]["legacy_consecutive_known_ownership_switches"] == 736
    assert result["same_surface_metric"]["correspondence_attempts"] == 0
    assert result["same_surface_metric"]["status"] == "NOT_EVALUATED"


def test_delivery_has_fifteen_decoded_slots_without_quality_promotion():
    result = load("delivery/DELIVERY_MANIFEST.json")
    assert result["slot_count"] == 15
    assert len({row["slot_id"] for row in result["slots"]}) == 15
    assert result["full_decode_pass"] is True
    assert result["quality_pass"] == 0
    assert result["adopted"] == 0
