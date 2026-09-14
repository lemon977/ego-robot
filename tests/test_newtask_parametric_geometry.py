from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.depth_occlusion_v3 import DepthV3Error
from pipeline.newtask_object_firewall import WrongGeometryError, validate_contract
from pipeline.newtask_parametric_geometry import (
    build_geometry_assets,
    downsample_protected_any,
    make_bowl_revolve_meshes,
    make_chip_saddle_mesh,
    make_chip_saddle_meshes,
    project_registry_frame,
    rasterize_mesh_depth,
    split_bowl_inner_outer_surfaces,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "_run" / "newtask_contract_v2_2"
K = np.asarray([[105.0, 0.0, 47.5], [0.0, 105.0, 31.5], [0.0, 0.0, 1.0]])


def _example(name: str) -> dict:
    value = json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))
    validate_contract(
        task=value["registry"]["task"],
        measurements=value["measurement"],
        registry=value["registry"],
        timeline=value["timeline"],
        schema_dir=CONTRACT_DIR,
        validation_mode="DRY_RUN",
    )
    return value


def _pose(x: float, y: float, z: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (x, y, z)
    return out


def _edge_multiplicity(mesh) -> np.ndarray:
    faces = mesh.faces
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def test_poker_schema_builds_three_cards_rack_and_nonrendered_support_roles():
    example = _example("example_poker.json")
    assets = build_geometry_assets(
        task="puke", measurements=example["measurement"], registry=example["registry"]
    )
    assert list(assets) == [
        "card_0", "card_1", "card_2", "rack", "mat_plane", "rack_top_plane"
    ]
    assert all(assets[f"card_{index}"].geometry_type == "CARD_THIN_BOX" for index in range(3))
    assert assets["rack"].role == "STATIC_FURNITURE"
    assert assets["rack"].clean_policy == "PRESERVE"
    assert assets["mat_plane"].visual_mesh is None
    assert assets["rack_top_plane"].visual_mesh is None
    assert np.allclose(assets["card_0"].box_size_xyz_m, [0.088, 0.063, 0.0003])


def test_all_generated_visual_and_collision_meshes_are_closed_two_manifolds():
    for example_name in ("example_poker.json", "example_chips.json"):
        example = _example(example_name)
        assets = build_geometry_assets(
            task=example["registry"]["task"],
            measurements=example["measurement"],
            registry=example["registry"],
        )
        for asset in assets.values():
            for mesh in (asset.visual_mesh, asset.collision_mesh):
                if mesh is not None:
                    assert np.all(_edge_multiplicity(mesh) == 2), asset.object_id


def test_chip_saddle_has_opposite_principal_curvature_and_separate_collision_mesh():
    measurement = _example("example_chips.json")["measurement"]["measurements"][0]
    visual, collision = make_chip_saddle_meshes(measurement)
    # Fit z = ax^2 + by^2 + c to the visual upper surface.  The schema example
    # has opposite-signed directional sag, so the fitted curvatures must oppose.
    layer_size = visual.vertices_m.shape[0] // 2
    top = visual.vertices_m[:layer_size]
    design = np.column_stack((top[:, 0] ** 2, top[:, 1] ** 2, np.ones(layer_size)))
    coefficients, *_ = np.linalg.lstsq(design, top[:, 2], rcond=None)
    assert coefficients[0] * coefficients[1] < 0.0
    assert collision.surface != visual.surface
    assert np.ptp(collision.vertices_m[:, 2]) > np.ptp(visual.vertices_m[:, 2])


def test_chip_saddle_rejects_non_saddle_same_signed_sag():
    with pytest.raises(DepthV3Error, match="opposite-signed"):
        make_chip_saddle_mesh(
            length_m=0.055,
            width_m=0.035,
            thickness_m=0.0015,
            center_sag_long_m=0.006,
            center_sag_short_m=0.004,
        )


def test_bowl_revolve_has_cavity_floor_rim_and_independent_collision_surface():
    measurement = _example("example_chips.json")["measurement"]["measurements"][1]
    visual, collision = make_bowl_revolve_meshes(measurement)
    visual_radius = np.linalg.norm(visual.vertices_m[:, :2], axis=1)
    collision_radius = np.linalg.norm(collision.vertices_m[:, :2], axis=1)
    assert np.isclose(visual_radius.max() * 2.0, measurement["outer_rim_diameter_m"])
    assert collision_radius.max() > visual_radius.max()
    assert visual.vertices_m[:, 2].min() == 0.0
    assert np.isclose(visual.vertices_m[:, 2].max(), measurement["outer_height_m"])
    assert "inner_outer" in visual.surface
    assert visual.surface != collision.surface


def test_bowl_inner_and_outer_depth_surfaces_are_explicit_and_nonempty():
    measurement = _example("example_chips.json")["measurement"]["measurements"][1]
    visual, _ = make_bowl_revolve_meshes(measurement)
    inner, outer = split_bowl_inner_outer_surfaces(visual)
    assert "inner" in inner.surface
    assert "outer" in outer.surface
    assert inner.faces.shape[0] > 0
    assert outer.faces.shape[0] > inner.faces.shape[0]
    inner_depth = rasterize_mesh_depth(inner, _pose(0.0, 0.0, 0.8), K, 64, 96)
    outer_depth = rasterize_mesh_depth(outer, _pose(0.0, 0.0, 0.8), K, 64, 96)
    assert inner_depth.amodal_mask.any()
    assert outer_depth.amodal_mask.any()
    assert not np.array_equal(inner_depth.amodal_mask, outer_depth.amodal_mask)


@pytest.mark.parametrize("example_name, object_id", [
    ("example_poker.json", "card_0"),
    ("example_chips.json", "chip_0"),
    ("example_chips.json", "bowl"),
])
def test_visual_geometry_projects_positive_ordered_cpu_depth(example_name, object_id):
    example = _example(example_name)
    assets = build_geometry_assets(
        task=example["registry"]["task"],
        measurements=example["measurement"],
        registry=example["registry"],
    )
    asset = assets[object_id]
    if asset.depth_adapter == "box_xyz":
        from pipeline.newtask_parametric_geometry import project_asset_depth

        depth = project_asset_depth(asset, _pose(0.0, 0.0, 0.8), K, 64, 96)
    else:
        depth = rasterize_mesh_depth(asset.visual_mesh, _pose(0.0, 0.0, 0.8), K, 64, 96)
    assert depth.amodal_mask.any()
    assert np.all(depth.near_m[depth.amodal_mask] > 0.0)
    assert np.all(depth.far_m[depth.amodal_mask] >= depth.near_m[depth.amodal_mask])


def test_registry_frame_applies_explicit_state_and_clean_protect_policies():
    example = _example("example_poker.json")
    assets = build_geometry_assets(
        task="POKER", measurements=example["measurement"], registry=example["registry"]
    )
    transforms = {
        "card_0": _pose(-0.09, 0.0, 0.8),
        "card_1": _pose(0.0, 0.0, 0.8),
        "card_2": _pose(0.09, 0.0, 0.8),
        "rack": _pose(0.0, 0.07, 0.95),
    }
    result = project_registry_frame(
        task="puke",
        assets=assets,
        registry=example["registry"],
        timeline_frame=example["timeline"]["frames"][0],
        transforms_object_to_camera=transforms,
        intrinsics=K,
        image_height=64,
        image_width=96,
    )
    assert set(result.depth_by_object) == {"card_0", "card_1", "card_2", "rack"}
    assert result.protected_union_2x.any()
    assert set(np.unique(result.object_index_2x)) >= {0, 1, 2, 3, 4}
    assert result.skipped_objects == {
        "mat_plane": "SUPPORT_AUTHORITY_NOT_RENDERED_AS_OPERATED_OBJECT",
        "rack_top_plane": "SUPPORT_AUTHORITY_NOT_RENDERED_AS_OPERATED_OBJECT",
    }
    assert downsample_protected_any(result.protected_union_2x).shape == (64, 96)


def test_out_of_view_state_suppresses_geometry_even_if_stale_pose_exists():
    example = _example("example_poker.json")
    assets = build_geometry_assets(
        task="POKER", measurements=example["measurement"], registry=example["registry"]
    )
    transforms = {
        "card_0": _pose(-0.09, 0.0, 0.8),
        "card_1": _pose(0.0, 0.0, 0.8),
        "card_2": _pose(0.09, 0.0, 0.8),
        "rack": _pose(0.0, 0.07, 0.95),
    }
    result = project_registry_frame(
        task="POKER",
        assets=assets,
        registry=example["registry"],
        timeline_frame=example["timeline"]["frames"][1],
        transforms_object_to_camera=transforms,
        intrinsics=K,
        image_height=64,
        image_width=96,
    )
    assert "card_2" not in result.depth_by_object
    assert result.skipped_objects["card_2"] == "STATE_OUT_OF_VIEW_POSE_NOT_RENDERED"


def test_chips_registry_renders_bowl_and_all_three_state_visible_chips():
    example = _example("example_chips.json")
    assets = build_geometry_assets(
        task="shupian", measurements=example["measurement"], registry=example["registry"]
    )
    transforms = {
        "chip_0": _pose(-0.09, 0.0, 0.75),
        "chip_1": _pose(0.0, 0.0, 0.75),
        "chip_2": _pose(0.09, 0.0, 0.75),
        "bowl": _pose(0.0, 0.06, 0.95),
    }
    result = project_registry_frame(
        task="CHIPS",
        assets=assets,
        registry=example["registry"],
        timeline_frame=example["timeline"]["frames"][0],
        transforms_object_to_camera=transforms,
        intrinsics=K,
        image_height=64,
        image_width=96,
    )
    assert set(result.depth_by_object) == {"chip_0", "chip_1", "chip_2", "bowl"}
    assert result.protect_mask_by_object_2x["bowl"].any()
    assert result.skipped_objects == {
        "mat_plane": "SUPPORT_AUTHORITY_NOT_RENDERED_AS_OPERATED_OBJECT"
    }


def test_build_assets_rejects_any_cylinder_injection_before_mesh_creation():
    example = _example("example_poker.json")
    poisoned = deepcopy(example["registry"])
    poisoned["objects"][0]["geometry_type"] = "CYLINDER"
    with pytest.raises(WrongGeometryError):
        build_geometry_assets(
            task="poker", measurements=example["measurement"], registry=poisoned
        )


def test_downsample_protect_mask_is_conservative_any_not_average():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 3] = True
    result = downsample_protected_any(mask)
    assert result.tolist() == [[False, True, False], [False, False, False]]
