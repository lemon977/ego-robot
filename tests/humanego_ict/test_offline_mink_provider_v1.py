"""CPU fixtures plus opt-in, bounded, actual frozen static-model mathematics.

Never a learned policy, measured feedback, hardware command, or safety test.
Real geometry fixtures share one attempted-call budget across the test session.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.humanego_ict.bridge import IKSolution

from chaoyang.humanego_ict.offline_mink_provider_v1 import (
    CLIENT_SHA256, FRAME, ENGINEERING_LIMIT_SOURCE, GeometryCallBudget,
    OfflineMinkError, OfflineMinkProvider, _arm_to_model, _client_step,
    _collision_pairs, _model_to_arm, _project_path, _read_verified,
    ORIGINAL_COLLISION_MODE, ORIGINAL_INITIALIZATION_MODE, FRAME_TASK_INITIALIZATION_MODE,
    STATIC_FRAME_TASK_PARAMETERS, STRICT_TRAINING_ADMISSION, EGO_TRAINING_ADMISSION,
    MESH_COLLISION_MODE)

STATIC_SYNTHETIC_CALLS = []


@pytest.fixture(scope="module", autouse=True)
def report_static_synthetic_calls():
    yield
    print(f"STATIC_SOLVER_SYNTHETIC_GEOMETRY_CLAIMS={sum(STATIC_SYNTHETIC_CALLS)}")
    print("STATIC_SOLVER_REAL_GEOMETRY_CALLS=0")


def transform(xyz, angle):
    value = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[:3, 3] = xyz
    return value


def test_metres_mm_base_direction_roundtrip():
    base = np.stack([transform([0, .026, 1.121], .7),
                     transform([0, -.026, 1.121], -.7)])
    arm = np.stack([transform([.25, -.1, .5], .9), transform([-.2, .1, .4], -.9)])
    model_mm = _arm_to_model(arm, base)
    np.testing.assert_allclose(model_mm[:, :3, 3], (base @ arm)[:, :3, 3] * 1000)
    np.testing.assert_allclose(_model_to_arm(model_mm, base), arm, atol=1e-12)
    assert not np.allclose(model_mm[:, :3, 3] / 1000, arm[:, :3, 3])


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.full((2, 4, 4), np.nan),
                                  np.zeros((2, 4, 4))])
def test_invalid_frame_arrays_rejected(bad):
    with pytest.raises(ValueError):
        _arm_to_model(bad, np.repeat(np.eye(4)[None], 2, axis=0))


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_budget_is_finite(limit):
    with pytest.raises(OfflineMinkError, match="FINITE_GEOMETRY"):
        GeometryCallBudget(limit)


@pytest.mark.parametrize("hard_limit", [0, -1, 5001, True, False, 1.5,
                                        np.int64(1000), None])
def test_explicit_budget_cap_is_strict_finite_integer(hard_limit):
    with pytest.raises(OfflineMinkError, match="FINITE_GEOMETRY"):
        GeometryCallBudget(1, hard_limit=hard_limit)


@pytest.mark.parametrize("limit", [0, -1, 5001, True, False, 1.5,
                                   np.int64(5000), None])
def test_explicit_5000_cap_does_not_relax_limit_validation(limit):
    with pytest.raises(OfflineMinkError, match="FINITE_GEOMETRY"):
        GeometryCallBudget(limit, hard_limit=5000)


def test_explicit_cap_cannot_be_smaller_than_selected_limit():
    with pytest.raises(OfflineMinkError, match="FINITE_GEOMETRY"):
        GeometryCallBudget(18, hard_limit=17)


def test_default_budget_preserves_1000_limit_and_exhaustion():
    budget = GeometryCallBudget()
    assert budget.limit == budget.hard_limit == 1000
    budget.claim(1000)
    with pytest.raises(OfflineMinkError, match="EXHAUSTED"):
        budget.claim()
    assert budget.used == 1000


def test_explicit_5000_budget_has_exact_boundary():
    budget = GeometryCallBudget(5000, hard_limit=5000)
    assert budget.limit == budget.hard_limit == 5000 and budget.used == 0
    budget.claim(4999)
    budget.claim()
    with pytest.raises(OfflineMinkError, match="EXHAUSTED"):
        budget.claim()
    assert budget.used == 5000


@pytest.mark.parametrize("attribute", ["limit", "hard_limit"])
def test_budget_public_caps_are_immutable(attribute):
    budget = GeometryCallBudget(2, hard_limit=3)
    with pytest.raises(AttributeError):
        setattr(budget, attribute, 5000)
    assert budget.limit == 2 and budget.hard_limit == 3 and budget.used == 0


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, np.int64(1), None, 3])
def test_invalid_claim_does_not_consume_or_expand_budget(count):
    budget = GeometryCallBudget(3)
    budget.claim()
    with pytest.raises(OfflineMinkError, match="EXHAUSTED"):
        budget.claim(count)
    assert budget.used == 1 and budget.limit == 3


def test_failed_claim_not_reset_or_overcount():
    budget = GeometryCallBudget(2)
    budget.claim(2)
    with pytest.raises(OfflineMinkError, match="EXHAUSTED"):
        budget.claim()
    assert budget.used == 2


def test_original_client_scalar_ast_no_execution():
    assert _client_step('p.add_argument("--tianji-arm-step-rad", default=0.01)') == .01
    with pytest.raises(OfflineMinkError):
        _client_step('p.add_argument("--tianji-arm-step-rad", default=float("nan"))')


def test_sha_drift_and_path_escape(tmp_path):
    # Test-generated local evidence, not production source mutation.
    file = tmp_path / "source.txt"
    file.write_bytes(b"test evidence")
    _read_verified(file, hashlib.sha256(b"test evidence").hexdigest())
    with pytest.raises(OfflineMinkError, match="SHA_DRIFT"):
        _read_verified(file, "0" * 64)
    with pytest.raises(ValueError):
        _project_path(tmp_path / "subproject", file)


def test_collision_enumeration_covers_cross_arm_and_nonadjacent():
    # body0=world/floor; body1=fixed base; body2/3 left links; body4 right link.
    model = SimpleNamespace(ngeom=5, geom_contype=np.ones(5, int),
        geom_conaffinity=np.ones(5, int), geom_bodyid=np.arange(5),
        body_weldid=np.array([0, 0, 2, 3, 4]), body_parentid=np.array([0, 0, 1, 2, 1]),
        exclude_signature=np.array([], int))
    geoms, pairs, excluded = _collision_pairs(model)
    assert geoms == [0, 1, 2, 3, 4]
    assert (2, 4) in pairs and (3, 4) in pairs  # full cross-arm checks
    assert (1, 3) in pairs  # nonadjacent arm/body
    assert (0, 2) in pairs  # world/floor is not a fictitious joint neighbor
    assert (2, 3, "ADJACENT_JOINT") in excluded
    assert (0, 1, "SAME_WELDED_BODY") in excluded


def test_collision_masks_and_explicit_model_excludes():
    model = SimpleNamespace(ngeom=3, geom_contype=np.array([1, 1, 0]),
        geom_conaffinity=np.array([1, 1, 0]), geom_bodyid=np.array([1, 3, 4]),
        body_weldid=np.arange(5), body_parentid=np.zeros(5, int),
        exclude_signature=np.array([(1 << 16) + 3]))
    geoms, pairs, excluded = _collision_pairs(model)
    assert geoms == [0, 1] and not pairs
    assert excluded == [(0, 1, "MODEL_EXPLICIT_EXCLUDE")]


def test_unchanged_wrist_chain_keeps_link5_link7_and_cross_arm_pairs():
    # Pure topology fixture: 0=floor, 1/2/3=left Link5/6/7,
    # 4/5/6=right Link5/6/7. No actual geometry/backend is constructed.
    model = SimpleNamespace(ngeom=7, geom_contype=np.ones(7, int),
        geom_conaffinity=np.ones(7, int), geom_bodyid=np.arange(7),
        body_weldid=np.arange(7), body_parentid=np.array([0, 0, 1, 2, 0, 4, 5]),
        exclude_signature=np.array([], int))
    geoms, pairs, excluded = _collision_pairs(model)
    assert geoms == list(range(7))
    assert (1, 3) in pairs and (4, 6) in pairs  # Link5/7 not suppressed
    assert (1, 4) in pairs and (3, 6) in pairs  # cross-arm remains checked
    assert (0, 1) in pairs and (0, 4) in pairs  # floor/world remains checked
    assert excluded == [(1, 2, "ADJACENT_JOINT"), (2, 3, "ADJACENT_JOINT"),
                        (4, 5, "ADJACENT_JOINT"), (5, 6, "ADJACENT_JOINT")]
    assert len(pairs) == 17


@pytest.mark.parametrize("field", ["collision_validation", "current_initialization_mode",
                                   "training_admission_policy"])
@pytest.mark.parametrize("value", [None, [], {}, True, "UNREGISTERED_MODE"])
def test_provider_mode_rejects_before_any_source_or_geometry_read(field, value):
    with pytest.raises(OfflineMinkError, match="EXPLICIT_SUPPORTED"):
        OfflineMinkProvider(project_root="/does_not_exist", source_index_path="absent",
            expected_index_sha256="0" * 64, client_source_path="absent", **{field: value})


def test_ego_admission_does_not_allow_mesh_training_dependency():
    with pytest.raises(OfflineMinkError, match="ORIGINAL_CAPSULE_DIAGNOSTICS_ONLY"):
        OfflineMinkProvider(project_root="/does_not_exist", source_index_path="absent",
            expected_index_sha256="0" * 64, client_source_path="absent",
            training_admission_policy=EGO_TRAINING_ADMISSION, collision_validation=MESH_COLLISION_MODE)


@pytest.fixture
def fake_static_provider(monkeypatch):
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    provider = object.__new__(OfflineMinkProvider)
    provider.current_initialization_mode = FRAME_TASK_INITIALIZATION_MODE
    provider.collision_validation = ORIGINAL_COLLISION_MODE
    provider.training_admission_policy = STRICT_TRAINING_ADMISSION
    provider.budget = GeometryCallBudget(100)
    provider.signature = "f" * 64
    provider._base = targets.copy()
    provider._static_hand_tasks = None
    provider._ensure_backend = lambda: None
    calls = []
    configuration = SimpleNamespace(q=np.zeros(14))
    configuration.integrate_inplace = lambda velocity, dt: setattr(configuration, "q", configuration.q + velocity * dt)
    future_tasks, limits = [object(), object(), object()], [object(), object(), object()]
    backend = SimpleNamespace(configuration=configuration, tasks=future_tasks, limits=limits,
        solver="frozen_solver", solve_damping=.1, kinetic_task=object(), hand_tasks={"A": object(), "B": object()},
        _joint_bounds_deg=[(side, joint, float(np.degrees(-3.)), float(np.degrees(3.)))
            for side in ("A", "B") for joint in range(1, 8)])

    def sync(q_deg, *, reset_posture_target):
        calls.append(("sync", reset_posture_target))
        configuration.q = np.radians(q_deg)

    backend.sync_config = sync
    backend.set_target = lambda side, target: calls.append(("future_target", side))
    backend.solve = lambda dt: (calls.append(("future_solve", dt)) or np.degrees(configuration.q))
    provider._backend = backend

    def check(q):
        provider.budget.claim(3)  # fake original native pass + two QP distances
        return True, True, {"fake_only": True}

    def fk(q):
        provider.budget.claim(2)  # two fake FK calls
        return targets.copy()

    provider._check_collisions, provider._check_legacy_collisions, provider.fk = check, check, fk

    class FrameTask:
        def __init__(self, **values):
            self.values = values

        def set_target(self, target):
            self.target = target

    def solve_ik(configuration, tasks, dt, solver, *, damping, limits):
        calls.append(("static_solve", tasks, dt, solver, damping, limits))
        return np.full(14, .02)

    fake_mink = SimpleNamespace(FrameTask=FrameTask, solve_ik=solve_ik,
        SO3=SimpleNamespace(from_matrix=lambda matrix: matrix.copy()),
        SE3=SimpleNamespace(from_rotation_and_translation=lambda matrix, xyz: (matrix, xyz)))
    monkeypatch.setitem(sys.modules, "mink", fake_mink)
    yield provider, calls, targets, future_tasks, limits
    STATIC_SYNTHETIC_CALLS.append(provider.budget.used)


def test_static_fit_uses_only_two_fixed_tasks_original_limits_and_frame(fake_static_provider):
    provider, calls, target, tasks, limits = fake_static_provider
    answer = provider.solve_static(target, np.zeros(14), 1 / 30)
    static = next(call for call in calls if call[0] == "static_solve")
    assert len(static[1]) == 2 and static[2] == 1 / 30 and static[-1] is limits
    assert static[4] == 1e-6
    assert calls[0] == ("sync", False)
    assert all(task.values["position_cost"] == 100. for task in static[1])
    assert all(task.values["orientation_cost"] == 11.459155902616466 for task in static[1])
    assert provider._backend.tasks is tasks and provider._backend.solve_damping == .1
    assert answer.collision_checked and answer.collision_free
    assert answer.diagnostics["physical_deployment_authorized"] is False
    assert answer.diagnostics["frame"] == FRAME and provider.budget.used == 9


def test_static_fit_then_future_keeps_original_solver_tasks_actual_dt(fake_static_provider):
    provider, calls, target, tasks, limits = fake_static_provider
    provider.solve_static(target, np.zeros(14), 1 / 30)
    provider.solve(target, np.zeros(14), .0173)
    assert calls[-1] == ("future_solve", .0173)
    assert ("sync", True) in calls and provider._backend.tasks is tasks
    assert provider._backend.limits is limits and provider._backend.solve_damping == .1
    assert provider.budget.used == 18


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_static_invalid_time_before_any_geometry(fake_static_provider, dt):
    provider, calls, target, _, _ = fake_static_provider
    with pytest.raises(OfflineMinkError, match="NATIVE_DT"):
        provider.solve_static(target, np.zeros(14), dt)
    assert provider.budget.used == 0 and calls == []


def test_static_solver_requires_explicit_opt_in(fake_static_provider):
    provider, calls, target, _, _ = fake_static_provider
    provider.current_initialization_mode = ORIGINAL_INITIALIZATION_MODE
    with pytest.raises(OfflineMinkError, match="CURRENT_MODE_REQUIRED"):
        provider.solve_static(target, np.zeros(14), 1 / 30)
    assert provider.budget.used == 0 and calls == []


@pytest.mark.parametrize("method", ["solve_retarget_current", "solve_retarget_future"])
def test_retarget_methods_require_signed_ego_policy(fake_static_provider, method):
    provider, calls, target, _, _ = fake_static_provider
    with pytest.raises(OfflineMinkError, match="EGO_DERIVED_TRAINING_ADMISSION_REQUIRED"):
        getattr(provider, method)(target, np.zeros(14), 1 / 30)
    assert provider.budget.used == 0 and calls == []


@pytest.mark.parametrize("method,source_solver", [("solve_retarget_current", "solve_static"),
                                                ("solve_retarget_future", "solve")])
def test_ego_projection_keeps_raw_q_and_recomputes_fk_collision(fake_static_provider, method, source_solver):
    provider, calls, target, _, _ = fake_static_provider
    provider.training_admission_policy = EGO_TRAINING_ADMISSION
    raw = np.zeros(14)
    raw[0], raw[7] = 3. + 1e-12, -4.
    original = raw.copy()
    provider_solver = lambda target, seed, dt: IKSolution(raw.copy(), True, False, {
        "position_error_m": [.5, .6], "rotation_error_rad": [.7, .8],
        "collision": {"actual_nonadjacent_penetration_contacts": [{"signed_distance_m": -.08}]},
        "physical_deployment_authorized": True})  # must not inherit forged physical permission
    setattr(provider, source_solver, provider_solver)
    real_fake_fk, real_fake_collision = provider.fk, provider._check_legacy_collisions

    def fk(q):
        calls.append(("post_fk", q.copy()))
        return real_fake_fk(q)

    def collision(q):
        calls.append(("post_collision", q.copy()))
        checked, _, _ = real_fake_collision(q)
        return checked, False, {"actual_nonadjacent_penetration_contacts": [{"signed_distance_m": -.02}]}

    provider.fk, provider._check_legacy_collisions = fk, collision
    answer = getattr(provider, method)(target, np.zeros(14), .0173)
    evidence = answer.diagnostics["retargeting"]
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(evidence["raw_q_rad"], original)
    np.testing.assert_array_equal(answer.q_rad, evidence["projected_q_rad"])
    assert answer.q_rad[0] <= 3. and answer.q_rad[7] >= -3.
    np.testing.assert_allclose(evidence["projection_correction_rad"], answer.q_rad - original, atol=0, rtol=0)
    np.testing.assert_array_equal(calls[0][1], answer.q_rad)
    np.testing.assert_array_equal(calls[1][1], answer.q_rad)
    assert evidence["projection_applied"] and evidence["source_solver"] == source_solver
    assert evidence["dt_s"] == .0173 and provider.budget.used == 5
    assert evidence["raw_fit_diagnostics"]["position_error_m"] == [.5, .6]
    assert evidence["post_projection_fit_diagnostics"]["position_error_m"] == [0., 0.]
    assert answer.collision_checked is True and answer.collision_free is False
    assert answer.diagnostics["physical_deployment_authorized"] is False
    assert answer.diagnostics["control_ground_truth"] is False


def test_ego_current_and_future_still_dispatch_original_methods_actual_dt(fake_static_provider):
    provider, calls, target, tasks, limits = fake_static_provider
    provider.training_admission_policy = EGO_TRAINING_ADMISSION
    current = provider.solve_retarget_current(target, np.zeros(14), 1 / 30)
    future = provider.solve_retarget_future(target, current.q_rad, .0257)
    assert any(call[0] == "static_solve" for call in calls)
    assert ("future_solve", .0257) in calls
    assert provider._backend.tasks is tasks and provider._backend.limits is limits
    assert not current.diagnostics["retargeting"]["projection_applied"]
    assert not future.diagnostics["retargeting"]["projection_applied"]
    assert provider.budget.used == 28


@pytest.mark.parametrize("bad", [np.full(14, np.nan), np.zeros(13)])
def test_nonfinite_or_wrong_shape_raw_q_never_becomes_valid_by_projection(fake_static_provider, bad):
    provider, _, target, _, _ = fake_static_provider
    provider.training_admission_policy = EGO_TRAINING_ADMISSION
    provider.solve_static = lambda target, seed, dt: IKSolution(bad, False, False, {})
    with pytest.raises(ValueError):
        provider.solve_retarget_current(target, np.zeros(14), 1 / 30)
    assert provider.budget.used == 0


def test_original_solver_failure_never_invents_a_retarget_q(fake_static_provider):
    provider, _, target, _, _ = fake_static_provider
    provider.training_admission_policy = EGO_TRAINING_ADMISSION

    def failure(target, seed, dt):
        raise RuntimeError("synthetic original solver failure")

    provider.solve_static = failure
    with pytest.raises(RuntimeError, match="original solver failure"):
        provider.solve_retarget_current(target, np.zeros(14), 1 / 30)
    assert provider.budget.used == 0


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_ego_invalid_time_rejected_before_any_geometry(fake_static_provider, dt):
    provider, calls, target, _, _ = fake_static_provider
    provider.training_admission_policy = EGO_TRAINING_ADMISSION
    with pytest.raises(OfflineMinkError, match="NATIVE_DT"):
        provider.solve_retarget_future(target, np.zeros(14), dt)
    assert provider.budget.used == 0 and calls == []


@pytest.fixture(scope="module")
def actual_provider():
    if os.environ.get("RUN_FROZEN_STATIC_GEOMETRY_TESTS") != "1":
        pytest.skip("Actual static geometry is explicit server-only CPU evidence")
    root = Path(os.environ["CHAoyang_PROJECT_ROOT"])
    index = Path(os.environ["FROZEN_TIANJI_SOURCE_INDEX"])
    client = Path(os.environ["FROZEN_TIANJI_CLIENT_SOURCE"])
    provider = OfflineMinkProvider(project_root=root, source_index_path=index,
        expected_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        client_source_path=client, budget=GeometryCallBudget(900))
    yield provider
    print(f"ACTUAL_STATIC_GEOMETRY_CALLS={provider.budget.used}")


def test_actual_lazy_frozen_source_and_dependency_binding(actual_provider):
    p = actual_provider
    assert p._backend is None
    assert len(p.signature) == 64 and len(p.source_receipts) == 22
    assert any(ref["sha256"] == CLIENT_SHA256 for ref in p.source_receipts)
    assert p.offline_only is True and p.ik_frame_id == FRAME


def test_actual_nominal_base_fk_direct_backend_parity(actual_provider):
    p = actual_provider
    q = p.synthetic_home_q_rad
    actual = p.fk(q)
    direct_mm = []
    for side in ("A", "B"):
        p.budget.claim()
        direct_mm.append(p._backend.fk_mm(side)(np.degrees(q)))
    model = np.stack(direct_mm)
    model[:, :3, 3] /= 1000
    np.testing.assert_allclose(p.model_world_from_arm_base @ actual, model, atol=1e-12)
    np.testing.assert_allclose(p.model_world_from_arm_base[:, :3, 3],
                               [[0, .026, 1.121], [0, -.026, 1.121]], atol=1e-12)
    assert np.isfinite(actual).all()


def test_actual_source_software_limits_and_new_explicit_fk_assumption(actual_provider):
    p = actual_provider
    root_profile = json.loads((p.root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/"
        "delivery_10am_20261012/ASSUMPTIONS.json").read_text())
    original = deepcopy(root_profile)
    profile = p.assumed_profile(root_profile)
    assert root_profile == original  # no protected profile mutation
    limits = profile["numerical_limits"]
    assert limits["joint_step_max_rad"] == [.01] * 14
    assert limits["joint_upper_rad"][1] == pytest.approx(np.deg2rad(120))
    assert limits["joint_upper_rad"][3] == pytest.approx(np.deg2rad(60))
    assert limits["joint_speed_max_rad_s"][0] == pytest.approx(1.5 * np.pi * .55)
    assert limits["position_tolerance_m"] == .01
    assert limits["rotation_tolerance_rad"] == pytest.approx(np.deg2rad(5))
    assert profile["numeric_limit_sources"]["absolute_fk_tolerances"] == ENGINEERING_LIMIT_SOURCE
    assert profile["rotation_convention"] == original["rotation_convention"]
    assert profile["physical_deployment_authorized"] is False


def test_actual_single_differential_step_and_full_collision_fail_preserved(actual_provider):
    p = actual_provider
    q = p.synthetic_home_q_rad
    target = p.fk(q)
    before = p.budget.used
    solution = p.solve(target, q, 1 / 30)
    assert p.budget.used - before == 9
    assert solution.diagnostics["differential_steps"] == 1
    assert max(solution.diagnostics["position_error_m"]) < 1e-8
    assert max(solution.diagnostics["rotation_error_rad"]) < 1e-6
    assert solution.collision_checked is True
    # Actual frozen model home has Link5/Link7 capsule penetration. Do not
    # promote the QP's two Link5/base distances into full-model PASS.
    assert solution.collision_free is False
    collision = solution.diagnostics["collision"]
    assert collision["eligible_pair_count"] > 2
    assert collision["original_qp_pair_count"] == 2
    assert min(collision["original_qp_distances_m_capped_at_detection_horizon"]) >= .001
    assert len(collision["actual_nonadjacent_penetration_contacts"]) >= 2
    assert collision["physical_scene_collision_status"] == "UNKNOWN"
    assert solution.diagnostics["robot_commands_sent"] == 0


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_actual_invalid_time_does_not_call_geometry(actual_provider, dt):
    p = actual_provider
    before = p.budget.used
    with pytest.raises(OfflineMinkError, match="NATIVE_DT"):
        p.solve(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros(14), dt)
    assert p.budget.used == before


def test_actual_source_drift_detected_before_any_geometry(actual_provider, monkeypatch):
    p = actual_provider
    before = p.budget.used
    identities = dict(p._identities)
    path = next(iter(identities))
    identities[path] = (0, 0, 0, 0, 0)
    monkeypatch.setattr(p, "_identities", identities)
    with pytest.raises(OfflineMinkError, match="IDENTITY_DRIFT"):
        p.fk(np.zeros(14))
    assert p.budget.used == before


def test_actual_disabled_native_collision_cannot_claim_checked(actual_provider):
    p = actual_provider
    old = int(p._backend.model.opt.disableflags)
    try:
        p._backend.model.opt.disableflags = old | int(p._mujoco.mjtDisableBit.mjDSBL_CONTACT)
        checked, free, diagnostics = p._check_collisions(p.synthetic_home_q_rad)
        assert checked is False and free is False
        assert diagnostics["native_contact_calculation_enabled"] is False
    finally:
        p._backend.model.opt.disableflags = old


def test_actual_profile_cannot_upgrade_measured_claim(actual_provider):
    p = actual_provider
    profile = json.loads((p.root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/"
        "delivery_10am_20261012/ASSUMPTIONS.json").read_text())
    profile["measured"] = True
    before = p.budget.used
    with pytest.raises(ValueError, match="CANNOT_DECLARE"):
        p.assumed_profile(profile)
    assert p.budget.used == before


def test_actual_assumed_ict_and_h50_reject_have_no_physical_claim(actual_provider):
    from chaoyang.humanego_ict.assumed_deployment_v1 import build_assumed_ict, bridge_assumed_action_chunk
    p = actual_provider
    root_profile = json.loads((p.root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/"
        "delivery_10am_20261012/ASSUMPTIONS.json").read_text())
    profile = p.assumed_profile(root_profile)
    q = p.synthetic_home_q_rad
    normalization = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
    ict = build_assumed_ict(q, x_rgb=np.zeros((1, 3, 240, 320), np.float32),
        assumption_profile=profile, fk_provider=p, normalization=normalization,
        anchor={"source_frame_id": 0, "anchor_ns": 0, "image_ns": 0, "joint_ns": 0,
                "time_source": "ASSUMED_OFFLINE_CLOCK"})
    assert ict["status"] == "BUILT_ASSUMED_ICT_OFFLINE_ONLY"
    pose = ict["current_input_npz"]["wrist_pose_world"]
    action = np.zeros((50, 30))
    action[:, :6] = pose[:, :3, 3].reshape(6)
    action[:, 6:18] = pose[:, :3, :2].reshape(12)
    result = bridge_assumed_action_chunk(action, action_mapping={
        "source_frame_ids": list(range(1, 51)), "timestamp_ns": [int(i * 1e9 / 30) for i in range(1, 51)],
        "anchor_ns": 0, "mapping_kind": "ASSUMED_OFFLINE_GRID"}, assumption_profile=profile,
        initial_q_rad=q, ik_provider=p, normalization=normalization)
    assert not result["numerical_candidate_accepted"]
    assert result["candidate_actions"] is None
    assert "IK_COLLISION_OR_UNCHECKED" in result["frame_errors"][0]
    np.testing.assert_array_equal(result["raw_actions"], action)
    assert result["physical_deployment_authorized"] is False
    assert result["control_ground_truth"] is False and result["measured"] is False
    assert result["robot_commands_sent"] == 0
