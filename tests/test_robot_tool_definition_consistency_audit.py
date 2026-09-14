from pathlib import Path

from tools.audit_robot_tool_definition_consistency_t0 import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_paths_have_no_conflict_but_s7_is_not_yet_measurable() -> None:
    report = build(PROJECT_ROOT)
    assert report["a_class_p0"] is False
    assert report["actual_consumers"]["renderer"]["scope"] == "LIVE_KAIHAND_ONLY_SMOKE"
    assert report["actual_consumers"]["ik"]["observed_value_m"] == 0.145
    assert report["actual_consumers"]["visual_fit"]["observed_value_m"] == 0.145
    assert report["three_way_runtime_result"] == "UNMEASURED_S7_FULL_CHAIN_RENDERER_ABSENT"
    assert report["fail_closed_s7_admission"]["s7_rebuild_may_start_now"] is True
    assert report["fail_closed_s7_admission"]["full_chain_render_execution_admitted"] is False


def test_external_mink_path_is_not_in_project_inventory() -> None:
    report = build(PROJECT_ROOT)
    assert report["external_mjcf_inventory"] == {
        "marvin_pro_mink_xml_found": False,
        "matching_paths": [],
        "consumed_by_any_current_path": False,
    }
