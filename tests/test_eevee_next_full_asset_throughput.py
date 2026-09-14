from pathlib import Path

import pytest

from tools import probe_eevee_next_full_asset_throughput as probe


PROJECT = Path(__file__).resolve().parents[1]
PIN = PROJECT / "assets/robot/ROBOT_ASSET_PIN.json"


def test_scene_spec_is_exact_complete_d1_visual_closure() -> None:
    spec = probe.scene_spec(PROJECT, PIN)
    assert spec["scene_class"] == "VISUAL_ONLY_NOMINAL_ASSET_COMPLEXITY"
    assert spec["component_count"] == 3
    assert spec["mesh_count"] == 63
    assert spec["unique_mesh_count"] == 63
    assert spec["triangle_count_from_pin"] == 1_027_331
    assert spec["mount_applied"] is False
    assert spec["formal_pose_claim"] is False
    assert sum(len(item["visuals"]) for item in spec["components"]) == 63


def test_component_layout_is_explicit_and_not_a_mount_claim() -> None:
    spec = probe.scene_spec(PROJECT, PIN)
    layouts = {item["robot_name"]: item["display_translation"] for item in spec["components"]}
    assert layouts == {name: list(value) for name, value in probe.COMPONENT_LAYOUTS.items()}
    assert layouts["marvin_robot"] == [0.0, 0.0, 0.0]
    assert layouts["KaiBot-Dexhand shell-URDF-L-260624(1620)"] != [0.0, 0.0, 0.0]
    assert layouts["KaiBot-Dexhand shell-URDF-R-260424(1430)"] != [0.0, 0.0, 0.0]


def test_unknown_package_uri_fails_closed() -> None:
    urdf = PROJECT / "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf"
    with pytest.raises(probe.base.ProbeError, match="unsupported package URI"):
        probe.resolve_mesh_path(PROJECT, urdf, "package://not-pinned/mesh.STL")


def test_bad_vector_fails_closed() -> None:
    with pytest.raises(probe.base.ProbeError, match="invalid numeric vector"):
        probe._numbers("1 2", 3, (0.0, 0.0, 0.0))


def test_blender_triangle_ratio_gate_remains_strict() -> None:
    assert probe.MIN_BLENDER_TRIANGLE_RATIO == 0.99
    assert 1_024_400 / probe.EXPECTED_TRIANGLE_COUNT > probe.MIN_BLENDER_TRIANGLE_RATIO
