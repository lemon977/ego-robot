from pathlib import Path

import numpy as np

from tools import render_robot_eevee_fullchain_worker as worker


PROJECT = Path(__file__).resolve().parents[1]
WORKER = PROJECT / "tools/render_robot_eevee_fullchain_worker.py"


def test_worker_consumes_frozen_shared_palette_without_side_color_override() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "load_shared_robot_palette" in source
    assert 'records["ARM_SHELL_MATTE_WHITE"]' in source
    assert source.count('records["KAIHAND_SHELL_STEEL_BLUE"]') == 2
    assert "(0.10, 0.34, 0.78" not in source
    assert "(0.88, 0.34, 0.08" not in source


def test_worker_applies_frozen_color_management_and_pbr_scalars() -> None:
    source = WORKER.read_text(encoding="utf-8")
    for token in (
        'values["display_device"]',
        'values["view_transform"]',
        'values["look"]',
        'record["metallic"]',
        'record["roughness"]',
        'record["specular_ior_level"]',
    ):
        assert token in source


def test_naturalv2_transform_aims_plus_z_and_leaves_exact_12mm_root_gap() -> None:
    flange = np.eye(4, dtype=np.float64)
    flange[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    flange[:3, 3] = [0.1, -0.2, 0.3]
    hand = flange.copy()
    direction_flange = np.asarray([-0.07, 0.012, 0.025], dtype=np.float64)
    hand[:3, 3] += flange[:3, :3] @ direction_flange
    transform, metrics = worker._connector_transform_camera(flange, hand)
    raw_mating = np.asarray(
        [
            worker.CONNECTOR_CENTER_XY_RAW[0],
            worker.CONNECTOR_CENTER_XY_RAW[1],
            worker.CONNECTOR_MATING_Z_RAW,
            1.0,
        ]
    )
    raw_tip = raw_mating.copy()
    raw_tip[2] = worker.CONNECTOR_MAX_Z_RAW
    mapped_mating = transform @ raw_mating
    mapped_tip = transform @ raw_tip
    span = hand[:3, 3] - flange[:3, 3]
    assert np.allclose(mapped_mating[:3], flange[:3, 3], atol=1e-12)
    assert np.allclose(
        (mapped_tip[:3] - mapped_mating[:3]) / np.linalg.norm(mapped_tip[:3] - mapped_mating[:3]),
        span / np.linalg.norm(span),
        atol=1e-12,
    )
    assert np.isclose(
        np.linalg.norm(hand[:3, 3] - mapped_tip[:3]),
        worker.CONNECTOR_VISUAL_CLEARANCE_M,
        atol=1e-12,
    )
    assert np.isclose(metrics["connector_to_hand_root_clearance_m"], 0.012)
    assert metrics["native_plus_z_axis_error_deg"] < 1e-6


def test_worker_declares_65_object_closure_and_connector_palette_binding() -> None:
    source = WORKER.read_text(encoding="utf-8")
    for token in (
        "CONNECTOR_OBJECT_INDEX_BY_SIDE",
        'records["CONNECTOR_FLANGE_IVORY"]',
        'len(objects) + len(connector_objects) != 65',
        '"formal_appearance_gate": "HOLD_JOINT_FASTENER_TOPOLOGY"',
    ):
        assert token in source
