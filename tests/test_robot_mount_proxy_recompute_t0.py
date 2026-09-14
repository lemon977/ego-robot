from pathlib import Path

import pytest

from tools.recompute_robot_mount_proxy_t0 import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_geometry_recomputes_but_does_not_invent_mount() -> None:
    report = build(PROJECT_ROOT)
    assert report["recomputed"]["geometry_matches_directive_values"] is True
    assert report["status"] == "GEOMETRY_MATCHES_BUT_RIGID_MOUNT_UNDERSPECIFIED"
    gate = report["rigid_mount_gate"]
    assert gate["status"] == "UNRESOLVED"
    assert gate["translation_defined_by_four_measurements"] is False
    assert gate["rotation_defined_by_four_measurements"] is False
    assert report["advancement_authorized"] is False


@pytest.mark.parametrize("side", ["left", "right"])
def test_recomputed_values_and_axis_definition(side: str) -> None:
    report = build(PROJECT_ROOT)
    measured = report["recomputed"]
    assert measured["link7"][side]["x_half_span_mm"] == pytest.approx(41.0000681877)
    assert measured["kai_base"][side]["min_z_mm"] == pytest.approx(-101.7175614834)
    assert measured["kai_base"][side]["rear_slice_xy_extent_mm"] == pytest.approx(
        [48.71340096, 7.308745291]
    )
    assert measured["flange_to_tool"][side]["xyz_m"][2] == pytest.approx(0.145)
    assert measured["proxy_span_mm_by_side"][side] == pytest.approx(43.2824385166)


def test_uncertainty_and_prohibited_counters_are_explicit() -> None:
    report = build(PROJECT_ROOT)
    assert report["uncertainty"]["proxy_span_worst_case_interval_mm"] == pytest.approx(
        [40.7324385166, 45.8324385166]
    )
    assert set(report["counters"].values()) == {0}

