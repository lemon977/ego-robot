"""Pure XML/topology/fake-native tests; never import a real geometry backend."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from chaoyang.humanego_ict.offline_mesh_collision_v1 import (
    MODE, REPRESENTATION, MESH_NAMES, ROBOT_BODIES, OfflineMeshCollisionAudit,
    assert_same_kinematics, build_mesh_vfs)
from chaoyang.humanego_ict.offline_mink_provider_v1 import (
    GeometryCallBudget, MESH_COLLISION_MODE, OfflineMinkError, OfflineMinkProvider,
    _collision_pairs)

SYNTHETIC_CALLS = []


@pytest.fixture(scope="module", autouse=True)
def report_synthetic_calls():
    yield
    print(f"MESH_AUDIT_SYNTHETIC_NATIVE_CALLS={sum(SYNTHETIC_CALLS)}")
    print("MESH_AUDIT_REAL_GEOMETRY_CALLS=0")


def sources_fixture():
    root = Path("/path/to/chaoyang/_run/synthetic_not_created/geometry")
    scene_path, model_path = root / "scene.xml", root / "robot.xml"
    scene = b'<mujoco model="fixture"><worldbody><geom name="floor" type="plane"/></worldbody><include file="robot.xml"/></mujoco>'
    model = ET.Element("mujoco", {"model": "robot"})
    asset, world = ET.SubElement(model, "asset"), ET.SubElement(model, "worldbody")
    sources = {scene_path: scene}
    for name in sorted(MESH_NAMES):
        ET.SubElement(asset, "mesh", {"name": name, "file": f"mesh/{name}.STL",
                                      "content_type": "model/stl"})
        sources[root / "mesh" / f"{name}.STL"] = f"fake uncompiled STL {name}".encode()
    for name in sorted(ROBOT_BODIES):
        body = ET.SubElement(world, "body", {"name": name})
        mesh = "ZJ_Robot_link" if name == "base_link" else name
        ET.SubElement(body, "geom", {"class": "visual", "mesh": mesh})
        attrs = {"class": "collision", "mesh": mesh}
        if name in {"base_link", "Link5_L", "Link5_R"}:
            attrs = {"class": "collision", "fromto": "0 0 0 0 0 -0.15", "size": ".06"}
        ET.SubElement(body, "geom", attrs)
        ET.SubElement(body, "joint", {"name": f"fixture_{name}"})
    sources[model_path] = ET.tostring(model)
    return sources, scene_path, model_path


def test_vfs_only_changes_collision_representation_not_sources_or_floor():
    sources, scene, model_path = sources_fixture()
    before = deepcopy(sources)
    xml, assets, receipt = build_mesh_vfs(sources, scene)
    assert sources == before and len(assets) == 18
    assert ET.fromstring(xml).find("worldbody/geom").attrib == {"name": "floor", "type": "plane"}
    clone = ET.fromstring(assets["frozen_collision_model.xml"])
    for body in clone.iter("body"):
        collision = body.findall("geom")[1]
        assert collision.attrib["type"] == "mesh" and "fromto" not in collision.attrib
        assert "size" not in collision.attrib
        assert collision.attrib["mesh"] == ("ZJ_Robot_link" if body.attrib["name"] == "base_link"
                                            else body.attrib["name"])
    assert len(list(clone.iter("joint"))) == len(list(ET.fromstring(sources[model_path]).iter("joint")))
    assert receipt["geometry_representation"] == REPRESENTATION
    assert receipt["physical_geometry_qualified"] is False


def test_vfs_does_not_read_unbound_assets():
    sources, scene, model = sources_fixture()
    del sources[next(path for path in sources if path.suffix == ".STL")]
    with pytest.raises(OfflineMinkError, match="UNBOUND_MESH_CLONE_DEPENDENCY"):
        build_mesh_vfs(sources, scene)


def test_vfs_requires_all_seventeen_meshes_and_bodies():
    sources, scene, model_path = sources_fixture()
    model = ET.fromstring(sources[model_path])
    model.find("asset").remove(model.find("asset/mesh"))
    sources[model_path] = ET.tostring(model)
    with pytest.raises(OfflineMinkError, match="SEVENTEEN"):
        build_mesh_vfs(sources, scene)


def model_fixture():
    # world, base, Base_L, links1..7L, Base_R, links1..7R; full body scope.
    bodies = ["world", "base_link", "Base_L", *[f"Link{i}_L" for i in range(1, 8)],
              "Base_R", *[f"Link{i}_R" for i in range(1, 8)]]
    parents = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16])
    welds = np.arange(18)
    welds[[1, 2, 10]] = 0
    model = SimpleNamespace(nq=14, nv=14, njnt=14, nbody=18, nsite=2, ngeom=18,
        nmesh=17, nkey=1, names=b"same synthetic fixture names", body_names=bodies,
        qpos0=np.zeros(14), key_qpos=np.zeros((1, 14)), jnt_type=np.full(14, 3),
        jnt_qposadr=np.arange(14), jnt_dofadr=np.arange(14), jnt_bodyid=np.arange(14),
        jnt_pos=np.zeros((14, 3)), jnt_axis=np.tile([0, 0, 1], (14, 1)),
        jnt_range=np.tile([-3., 3.], (14, 1)), jnt_limited=np.ones(14),
        body_parentid=parents, body_weldid=welds, body_pos=np.zeros((18, 3)),
        body_quat=np.tile([1, 0, 0, 0], (18, 1)), body_mass=np.ones(18), body_inertia=np.ones((18, 3)),
        site_bodyid=np.array([9, 17]), site_pos=np.zeros((2, 3)),
        site_quat=np.tile([1, 0, 0, 0], (2, 1)), geom_bodyid=np.arange(18),
        geom_contype=np.ones(18, int), geom_conaffinity=np.ones(18, int),
        geom_type=np.full(18, 3), exclude_signature=np.array([], int),
        opt=SimpleNamespace(disableflags=0))
    model.geom_type[0] = 0  # original floor remains a plane
    return model


def data_fixture(model):
    return SimpleNamespace(qpos=np.zeros(14), contact=[],
        site_xpos=np.zeros((2, 3)), site_xmat=np.tile(np.eye(3).ravel(), (2, 1)),
        xpos=np.zeros((18, 3)), xmat=np.tile(np.eye(3).ravel(), (18, 1)))


@pytest.fixture
def fake_native_provider():
    sources, scene, _ = sources_fixture()
    original = model_fixture()
    original.jnt_range[1, 1] = 2.094  # effective backend override, not raw XML
    counters = {"compile": 0, "forward": 0}
    control = {"contacts": [], "fk_shift": 0., "compile_error": False}

    def compile_clone(xml, *, assets):
        counters["compile"] += 1
        if control["compile_error"]:
            raise ValueError("synthetic compile failure")
        assert len(assets) == 18
        clone = deepcopy(original)
        clone.jnt_range[1, 1] = -.1  # clone must restore effective frozen limit
        clone.geom_type[1:] = 7
        return clone

    def forward(model, data):
        counters["forward"] += 1
        data.contact = deepcopy(control["contacts"])
        data.site_xpos[:] = control["fk_shift"]

    def name_to_id(model, object_type, name):
        return (["left_tool_site", "right_tool_site"].index(name) if object_type == "site"
                else model.body_names.index(name))

    mj = SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=compile_clone),
        MjData=data_fixture, mj_forward=forward, mj_name2id=name_to_id,
        mj_id2name=lambda model, kind, index: model.body_names[index],
        mjtObj=SimpleNamespace(mjOBJ_BODY="body", mjOBJ_SITE="site"),
        mjtGeom=SimpleNamespace(mjGEOM_MESH=7),
        mjtDisableBit=SimpleNamespace(mjDSBL_CONTACT=1))
    provider = SimpleNamespace(_assert_frozen=lambda: None, _sources=sources,
        scene_path=scene, _mujoco=mj, _backend=SimpleNamespace(model=original),
        budget=GeometryCallBudget(100), _collision_data=data_fixture(original))
    first_budget = provider.budget
    yield provider, control, counters
    total_used = first_budget.used + (provider.budget.used if provider.budget is not first_budget else 0)
    assert total_used >= sum(counters.values())
    SYNTHETIC_CALLS.append(sum(counters.values()))


def test_clone_effective_limits_same_fk_and_pair_policy(fake_native_provider):
    provider, _, counters = fake_native_provider
    before = deepcopy(provider._backend.model)
    layer = OfflineMeshCollisionAudit(provider)
    assert_same_kinematics(before, layer.model)
    assert np.array_equal(before.geom_type, provider._backend.model.geom_type)
    assert (7, 9) in layer.pair_set and (15, 17) in layer.pair_set  # Link5/7
    assert (9, 17) in layer.pair_set  # cross-arm
    checked, free, report = layer.check(np.zeros(14), provider._collision_data)
    assert checked and free and report["same_q_fk_parity_passed"]
    assert counters == {"compile": 1, "forward": 1} and provider.budget.used == 2
    assert report["physical_scene_collision_status"] == "UNKNOWN"


@pytest.mark.parametrize("pair", [(7, 9), (15, 17), (9, 17), (0, 9)])
def test_active_wrist_cross_arm_and_floor_penetration_remain_fail(fake_native_provider, pair):
    provider, control, _ = fake_native_provider
    control["contacts"] = [SimpleNamespace(geom1=pair[0], geom2=pair[1], dist=-.002)]
    layer = OfflineMeshCollisionAudit(provider)
    checked, free, report = layer.check(np.zeros(14), provider._collision_data)
    assert checked and not free and len(report["actual_nonadjacent_penetration_contacts"]) == 1


@pytest.mark.parametrize("invalid", ["nan_contact", "disabled_contacts"])
def test_invalid_native_evidence_cannot_pass(fake_native_provider, invalid):
    provider, control, _ = fake_native_provider
    layer = OfflineMeshCollisionAudit(provider)
    if invalid == "nan_contact":
        control["contacts"] = [SimpleNamespace(geom1=7, geom2=9, dist=np.nan)]
    else:
        layer.model.opt.disableflags = 1
    checked, free, _ = layer.check(np.zeros(14), provider._collision_data)
    assert not checked and not free


def test_fk_or_joint_semantics_drift_is_rejected(fake_native_provider):
    provider, control, _ = fake_native_provider
    layer = OfflineMeshCollisionAudit(provider)
    control["fk_shift"] = .00001
    with pytest.raises(OfflineMinkError, match="FK_PARITY"):
        layer.check(np.zeros(14), provider._collision_data)
    layer.model.jnt_range[0, 1] += .001
    with pytest.raises(OfflineMinkError, match="KINEMATICS_DRIFT"):
        assert_same_kinematics(provider._backend.model, layer.model)


def test_reset_window_budget_is_shared_by_new_layer(fake_native_provider):
    provider, _, _ = fake_native_provider
    layer = OfflineMeshCollisionAudit(provider)
    provider.budget = GeometryCallBudget(1)
    layer.check(np.zeros(14), provider._collision_data)
    with pytest.raises(OfflineMinkError, match="EXHAUSTED"):
        layer.check(np.zeros(14), provider._collision_data)
    assert provider.budget.used == 1


def test_failed_compile_still_consumes_attempted_budget(fake_native_provider):
    provider, control, counters = fake_native_provider
    control["compile_error"] = True
    with pytest.raises(ValueError, match="synthetic compile"):
        OfflineMeshCollisionAudit(provider)
    assert provider.budget.used == counters["compile"] == 1


@pytest.mark.parametrize("qp_passed,mesh_free,selected_free", [(True, True, True),
    (False, True, False), (True, False, False)])
def test_selected_mesh_does_not_hide_old_capsule_fail_or_drop_qp_margin(qp_passed, mesh_free, selected_free):
    provider = object.__new__(OfflineMinkProvider)
    provider.collision_validation = MESH_COLLISION_MODE
    provider._collision_data = object()
    old = {"original_qp_distances_m_capped_at_detection_horizon": [.06, .06 if qp_passed else 0.],
           "original_qp_minimum_distance_m": .001,
           "actual_nonadjacent_penetration_contacts": [{"signed_distance_m": -.08}]}
    provider._check_legacy_collisions = lambda q: (True, False, deepcopy(old))
    provider._ensure_selected_collision_layer = lambda: None
    mesh = {"collision_validation": MODE, "geometry_representation": REPRESENTATION,
            "collision_checked": True, "collision_free": mesh_free,
            "actual_nonadjacent_penetration_contacts": [] if mesh_free else [{"signed_distance_m": -.002}]}
    provider._mesh_audit = SimpleNamespace(check=lambda q, old_data: (True, mesh_free, deepcopy(mesh)))
    checked, free, report = provider._check_collisions(np.zeros(14))
    assert checked and free is selected_free
    assert report["legacy_capsule_diagnostics"]["collision_free"] is False
    assert report["mesh_diagnostics"]["collision_free"] is mesh_free
    assert report["original_qp_margin_passed"] is qp_passed


def test_missing_mesh_or_disabled_original_native_evidence_cannot_pass():
    provider = object.__new__(OfflineMinkProvider)
    provider.collision_validation = MESH_COLLISION_MODE
    provider._collision_data = object()
    provider._check_legacy_collisions = lambda q: (False, False, {
        "original_qp_distances_m_capped_at_detection_horizon": [.06, .06],
        "original_qp_minimum_distance_m": .001})
    provider._ensure_selected_collision_layer = lambda: None
    provider._mesh_audit = SimpleNamespace(check=lambda q, data: (True, True, {}))
    checked, free, _ = provider._check_collisions(np.zeros(14))
    assert not checked and not free
