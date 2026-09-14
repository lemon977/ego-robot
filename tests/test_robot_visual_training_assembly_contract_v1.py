from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.build_robot_visual_training_assembly_contract_v1 import build_contract  # noqa: E402


def test_visual_contract_closes_without_claiming_physical_calibration() -> None:
    value = build_contract()
    assert value["status"] == "PASS_VISUAL_TRAINING_EXECUTION_ELIGIBLE_PHYSICAL_BLOCKED"
    assert value["authority_scope"]["visual_training_solver_execution"] is True
    assert value["authority_scope"]["physical_deployment"] is False
    assert value["tcp_contract"]["physical_tcp_calibrated"] is False
    assert value["assembly_contract"]["adapter_geometry_present"] is False
    world_base = np.asarray(value["coordinate_contract"]["T_world_base"])
    base_world = np.asarray(value["coordinate_contract"]["T_base_world"])
    assert np.allclose(world_base @ base_world, np.eye(4), atol=1e-8)
    assert all(value["gates"].values()) is False  # physical execution is deliberately false


def test_visual_contract_has_bilateral_proper_mounts() -> None:
    value = build_contract()
    for side in ("left", "right"):
        matrix = np.asarray(value["assembly_contract"][side])
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-8)
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
