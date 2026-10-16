"""Synthetic CPU mathematics and cache tests; no real geometry or policy call."""
from copy import deepcopy
import io
import json
import sys

import numpy as np
from PIL import Image
import pytest

from chaoyang.humanego_ict.assumed_deployment_v1 import PROFILE_SCHEMA, MODE, ROT6_DECLARATION
from chaoyang.humanego_ict.bridge import IKSolution
from chaoyang.humanego_ict.inference import build_hand_ict, HAND_CHANNELS
from chaoyang.humanego_ict.joint26_data import (
    ARM_INDICES, HAND_INDICES, SCHEMA, LAYOUT, SOURCE_POLICY, SUPERVISION_CLASS,
    STRICT_ADMISSION_POLICY, EGO_ADMISSION_POLICY, EGO_ADMITTED_STATUS, EGO_VALIDITY_BASIS,
    Joint26Dataset, Joint26DataError, file_ref, load_joint26_cache,
)
from chaoyang.humanego_ict.joint26_prepare import convert_window, prepare_dataset, Joint26PreparationError

NORMALIZATION = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
SYNTHETIC_CALLS = {"solve": 0, "solve_static": 0, "solve_retarget_current": 0,
                   "solve_retarget_future": 0, "fk": 0}


def _count_synthetic_call(name):
    # The a09 lane is allotted at most 5000 fake solve/FK calls in one pass.
    if sum(SYNTHETIC_CALLS.values()) >= 5000:
        raise AssertionError("SYNTHETIC_GEOMETRY_BUDGET_EXHAUSTED")
    SYNTHETIC_CALLS[name] += 1


@pytest.fixture(scope="module", autouse=True)
def report_synthetic_call_budget():
    SYNTHETIC_CALLS.update({key: 0 for key in SYNTHETIC_CALLS})
    yield
    print("SYNTHETIC_PROVIDER_CALLS=" + json.dumps(SYNTHETIC_CALLS, sort_keys=True))
    assert sum(SYNTHETIC_CALLS.values()) <= 5000


def profile():
    p = {"schema": PROFILE_SCHEMA, "mode": MODE, "position_unit": "meters",
         "rotation_convention": ROT6_DECLARATION, "hand_channel_order": list(HAND_CHANNELS),
         "device_ack_status": "UNKNOWN", "initial_state_source": "FROZEN_XML_HOME_NOT_MEASURED_FEEDBACK",
         "camera_transform_policy": "FIXED_EXPLICIT_PROFILE", "settling_dt_s": 1 / 30,
         "assumptions": [],
         "numerical_limits": {"joint_lower_rad": [-2.] * 14, "joint_upper_rad": [2.] * 14,
             "joint_speed_max_rad_s": [1.] * 14, "joint_step_max_rad": [.1] * 14,
             "position_tolerance_m": .0001, "rotation_tolerance_rad": .0001}}
    for direction, identifier in (("T_world_from_camera", "SYNTHETIC_CAMERA_IDENTITY"),
            ("T_world_from_arm_base", "SYNTHETIC_BASE_IDENTITY"),
            ("T_model_tool_from_training_wrist_prior", "MODEL_TOOL_EQUALS_TRAINING_WRIST_PRIOR")):
        p["assumptions"].append({"id": identifier, "matrix_direction": direction,
            "default_matrix": np.eye(4).tolist(), "source_class": "ASSUMED_NOT_MEASURED", "physical_qualification": False})
    return p


class SyntheticProvider:
    offline_only = True
    signature = "f" * 64

    def __init__(self, fault=None):
        self.fault, self.solve_calls, self.fk_calls = fault, 0, 0

    def fk(self, q):
        _count_synthetic_call("fk")
        self.fk_calls += 1
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[0, :3, 3], poses[1, :3, 3] = q[:3], q[7:10]
        return poses

    def solve(self, poses, seed, dt):
        _count_synthetic_call("solve")
        self.solve_calls += 1
        q = seed.copy(); q[:3], q[7:10] = poses[0, :3, 3], poses[1, :3, 3]
        if self.fault == "residual":
            q[0] += .001
        elif self.fault == "step":
            q[0] += .5
        elif self.fault == "exception":
            raise RuntimeError("SYNTHETIC_CALL_BUDGET_EXHAUSTED")
        elif self.fault == "source_drift":
            raise ValueError("SOURCE_IDENTITY_DRIFT:frozen_xml")
        elif self.fault == "solver_exception":
            raise Exception("SYNTHETIC_NON_RUNTIME_SOLVER_FAILURE")
        free = self.fault != "collision" and not (self.fault == "first_collision" and self.solve_calls == 1)
        return IKSolution(q, self.fault != "unchecked", free)


def original_labels():
    anchor = 1_000_000_000
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    current = build_hand_ict(poses, np.ones(2, bool), np.eye(4), normalization=NORMALIZATION,
                           source_qpc_ns=np.array([anchor - 5_000_000] * 2), anchor_qpc_ns=anchor)
    a = np.zeros((50, 30), np.float64)
    a[:, 0] = np.arange(1, 51) * .0001
    a[:, 3] = -np.arange(1, 51) * .0001
    a[:, 6:12] = a[:, 12:18] = np.eye(3)[:, :2].reshape(6)
    a[:, 18:24], a[:, 24:30] = .25, .75
    return {"x_ict": current["x_ict"], "ict_mask": current["ict_mask"], "y_action": a,
        "action_valid_mask": np.ones((50, 30), bool), "anchor_qpc_ns": np.array(anchor, np.int64),
        "conditioning_source_qpc_ns": current["conditioning_source_qpc_ns"],
        "target_qpc_ns": anchor + np.arange(1, 51, dtype=np.int64) * 50_000_000,
        "target_source_frame_ids": np.arange(1, 51, dtype=np.int64), "reference_camera_to_world": np.eye(4)}


def convert(labels=None, provider=None, p=None, **options):
    return convert_window(original_labels() if labels is None else labels,
        assumption_profile=profile() if p is None else p,
        provider=SyntheticProvider() if provider is None else provider,
        initial_q_rad=np.zeros(14), normalization=NORMALIZATION, source_frame_id=0, **options)


def test_rebuilds_causal_ict_and_joint_state_not_future_hand():
    result = convert(); arrays = result["arrays"]
    assert result["status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert arrays["y_action"].shape == (30, 26)
    assert arrays["action_valid_mask"].all()
    assert arrays["state_mask"][ARM_INDICES].all()
    assert not arrays["state_mask"][HAND_INDICES].any()
    assert np.all(arrays["x_state26"][HAND_INDICES] == 0)
    assert np.all(arrays["y_action"][:, 7:13] == .25)
    assert np.all(arrays["y_action"][:, 20:26] == .75)
    assert result["control_ground_truth"] is result["physical_deployment_authorized"] is False
    assert result["robot_commands_sent"] == 0


def test_current_input_independent_of_future_targets_and_grid():
    labels = original_labels(); before = convert(labels)["arrays"]
    labels["y_action"][:, :6] += .001
    labels["target_qpc_ns"] += 1_000_000
    after = convert(labels)["arrays"]
    for name in ("x_ict", "ict_mask", "x_state26", "state_mask", "current_joint_q_rad"):
        np.testing.assert_array_equal(before[name], after[name])


def test_intermediate_optimizer_collision_is_diagnostic_but_final_must_pass():
    result = convert(provider=SyntheticProvider("first_collision"))
    assert result["status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert result["settling_steps_used"] == 2
    assert result["current_optimization_diagnostics"][0]["collision_free"] is False
    assert result["current_optimization_diagnostics"][1]["collision_free"] is True
    assert result["arrays"]["action_valid_mask"].all()
    assert all(row["collision_gate_enforced"] for row in result["frame_diagnostics"])
    assert all(row["transition_gate_enforced"] for row in result["frame_diagnostics"])


def test_static_initialization_step_is_not_a_client_execution_transition():
    labels = original_labels()
    # Explicit current/past synthetic wrist, not y[0] copied into state.
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[0, 0, 3] = .15
    poses[1, 0, 3] = -.15
    anchor = int(labels["anchor_qpc_ns"])
    ict = build_hand_ict(poses, np.ones(2, bool), np.eye(4), normalization=NORMALIZATION,
        source_qpc_ns=labels["conditioning_source_qpc_ns"][:2], anchor_qpc_ns=anchor)
    labels["x_ict"] = ict["x_ict"]
    labels["y_action"][:, 0] += .15
    labels["y_action"][:, 3] -= .15
    result = convert(labels)
    assert result["status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    first = result["current_optimization_diagnostics"][0]
    assert first["transition_gate_enforced"] is False
    assert first["actual_max_step_rad"] > profile()["numerical_limits"]["joint_step_max_rad"][0]
    assert first["transition_limits_passed"] is False
    assert result["arrays"]["state_mask"][ARM_INDICES].all()
    assert all(row["transition_gate_enforced"] for row in result["frame_diagnostics"])


@pytest.mark.parametrize("case", ["step", "speed"])
def test_future_transition_gates_not_relaxed_by_static_initializer(case):
    labels = original_labels()
    labels["y_action"][2:, 0] += .15 if case == "step" else .02
    result = convert(labels)
    assert result["current_status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert result["arrays"]["action_valid_mask"][:2].all()
    assert not result["arrays"]["action_valid_mask"][2:].any()
    assert result["frame_errors"][2] == ["IK_STEP_OR_SPEED_LIMIT"]
    diagnostic = result["frame_diagnostics"][2]
    assert diagnostic["transition_gate_enforced"] is True
    assert diagnostic["transition_limits_passed"] is False


def test_final_current_joint_limit_still_required():
    labels = original_labels()
    labels["x_ict"][0, 1] = 3
    result = convert(labels)
    assert result["current_error"] == "IK_JOINT_LIMIT"
    assert not result["arrays"]["state_mask"].any()


def test_final_current_fk_residual_still_required():
    result = convert(provider=SyntheticProvider("residual"))
    assert result["current_error"] == "CURRENT_IK_NOT_CONVERGED_WITHIN_BUDGET"
    assert result["settling_steps_used"] == result["current_settle_steps_max"] == 60
    assert result["current_stop_reason"] == "SETTLE_ITERATION_BUDGET_EXHAUSTED"
    assert result["current_kinematics_converged"] is False
    assert not result["arrays"]["state_mask"].any()


def test_final_optimizer_collision_never_admitted_even_at_zero_fk_error():
    result = convert(provider=SyntheticProvider("collision"))
    assert result["settling_steps_used"] == 60
    assert result["current_error"] == "CURRENT_IK_COLLISION_OR_UNCHECKED"
    assert result["current_kinematics_converged"] is True
    assert result["current_stop_reason"] == "COLLISION_OR_UNCHECKED_AFTER_SETTLE_BUDGET"
    assert not result["arrays"]["state_mask"].any()
    assert not result["arrays"]["action_valid_mask"].any()


def test_source_integrity_drift_is_a_task_stop_not_masked_ik_failure():
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_DRIFT"):
        convert(provider=SyntheticProvider("source_drift"))


def test_library_exception_is_recorded_without_hiding_interrupts():
    result = convert(provider=SyntheticProvider("solver_exception"))
    assert "SYNTHETIC_NON_RUNTIME_SOLVER_FAILURE" in result["current_error"]


@pytest.mark.parametrize("fault,code", [("collision", "IK_COLLISION_OR_UNCHECKED"),
    ("unchecked", "IK_COLLISION_OR_UNCHECKED"), ("step", "CURRENT_IK_NOT_CONVERGED_WITHIN_BUDGET"),
    ("exception", "SYNTHETIC_CALL_BUDGET_EXHAUSTED")])
def test_failed_current_never_fabricates_complete_actions(fault, code):
    result = convert(provider=SyntheticProvider(fault))
    assert result["status"] == "FAILED_ASSUMED_MODEL_NUMERICS"
    assert code in result["current_error"]
    assert not result["arrays"]["action_valid_mask"].any()
    assert not result["arrays"]["state_mask"].any()


@pytest.mark.parametrize("case", ["future_current", "duplicate_target", "missing_settling_dt", "missing_matrix"])
def test_bad_mapping_and_assumptions_reject_before_geometry(case):
    labels, p, provider = original_labels(), profile(), SyntheticProvider()
    if case == "future_current":
        labels["conditioning_source_qpc_ns"][0] = int(labels["anchor_qpc_ns"]) + 1
    elif case == "duplicate_target":
        labels["target_source_frame_ids"][1] = labels["target_source_frame_ids"][0]
    elif case == "missing_settling_dt":
        p.pop("settling_dt_s")
    else:
        p["assumptions"].pop()
    with pytest.raises(ValueError):
        convert(labels, provider, p)
    assert provider.solve_calls == provider.fk_calls == 0


@pytest.mark.parametrize("case", ["hand_outside", "invalid_source"])
def test_future_failure_keeps_prefix_and_invalid_predecessor(case):
    labels = original_labels()
    if case == "hand_outside":
        labels["y_action"][3, 18] = -0.01
    else:
        labels["action_valid_mask"][3, 0] = False
    result = convert(labels)
    assert result["arrays"]["action_valid_mask"][:3].all()
    assert not result["arrays"]["action_valid_mask"][3:].any()
    assert "INVALID_PREDECESSOR" in result["frame_errors"][4]
    assert result["status"] == "FAILED_ASSUMED_MODEL_NUMERICS"


def test_expanded_static_budget_converges_after_sixty_without_future_leak():
    class DelayedProvider(SyntheticProvider):
        def __init__(self):
            super().__init__(); self.seen_targets = []
        def solve(self, poses, seed, dt):
            self.seen_targets.append(poses.copy())
            answer = super().solve(poses, seed, dt)
            if self.solve_calls <= 60:
                q = answer.q_rad.copy(); q[0] += .001
                return IKSolution(q, answer.collision_checked, answer.collision_free)
            return answer
    labels, provider = original_labels(), DelayedProvider()
    # Future is intentionally incompatible with current. It must not help the
    # static solve; its unchanged real-time transition gate rejects it later.
    labels["y_action"][:, 0] += .3
    result = convert_window(labels, assumption_profile=profile(), provider=provider,
        initial_q_rad=np.zeros(14), normalization=NORMALIZATION, source_frame_id=0,
        settle_steps=61)
    assert result["settling_steps_used"] == result["current_settle_steps_max"] == 61
    assert result["current_kinematics_converged"] is True
    assert result["current_stop_reason"] == "CONVERGED_FK_AND_COLLISION"
    assert result["current_status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert all(np.all(poses[:, :3, 3] == 0) for poses in provider.seen_targets[:61])
    assert np.all(result["arrays"]["current_joint_q_rad"] == 0)
    assert result["frame_errors"][0] == ["IK_STEP_OR_SPEED_LIMIT"]
    assert not result["arrays"]["action_valid_mask"].any()
    assert result["status"] == "FAILED_ASSUMED_MODEL_NUMERICS"


def test_three_hundred_static_steps_still_reject_unconverged_current():
    provider = SyntheticProvider("residual")
    result = convert_window(original_labels(), assumption_profile=profile(), provider=provider,
        initial_q_rad=np.zeros(14), normalization=NORMALIZATION, source_frame_id=0,
        settle_steps=300)
    assert provider.solve_calls == result["settling_steps_used"] == 300
    assert result["current_error"] == "CURRENT_IK_NOT_CONVERGED_WITHIN_BUDGET"
    assert result["current_kinematics_converged"] is False
    assert result["current_stop_reason"] == "SETTLE_ITERATION_BUDGET_EXHAUSTED"
    assert not result["arrays"]["state_mask"].any()
    assert not result["arrays"]["action_valid_mask"].any()
    assert all("INVALID_CURRENT_PREDECESSOR" in errors[0] for errors in result["frame_errors"])


@pytest.mark.parametrize("steps", [0, -1, 301, True, 60.5, "60"])
def test_invalid_static_budget_rejects_before_synthetic_geometry(steps):
    provider = SyntheticProvider()
    with pytest.raises(Joint26PreparationError, match="FINITE_FROZEN_SETTLING_LIMIT"):
        convert_window(original_labels(), assumption_profile=profile(), provider=provider,
            initial_q_rad=np.zeros(14), normalization=NORMALIZATION, source_frame_id=0,
            settle_steps=steps)
    assert provider.solve_calls == provider.fk_calls == 0


def _preparation_fixture(tmp_path, policy=None):
    """One frozen fake source; no mathematical provider is called by this helper."""
    labels = tmp_path / "original.npz"
    np.savez_compressed(labels, **original_labels())
    image = tmp_path / "image.png"; Image.new("RGB", (2, 2)).save(image)
    receipt = tmp_path / "causal.json"
    receipt.write_text(json.dumps({"input_class": "CAUSAL_TRAINING_INPUT", "sample_id": "synthetic_a",
        "source_frame_id": 0, "control_ground_truth": False}))
    sample = {"sample_id": "synthetic_a", "session_id": "synthetic",
        "source_frame_id": 0, "source_group_id": "synthetic_group", "split": "train",
        "labels": file_ref(labels), "branch_admission": {"raw": True, "robotized": True},
        "rgb": {branch: file_ref(image) for branch in ("raw", "robotized")},
        "causal_receipts": {branch: file_ref(receipt) for branch in ("raw", "robotized")}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"control_ground_truth": False, "physical_deployment_authorized": False,
        "normalization": NORMALIZATION, "samples": [sample]}))
    mapping = tmp_path / "source_map.json"
    mapping.write_text(json.dumps({"production_manifest_ref": file_ref(source), "branch": "raw",
        "split": "train", "sample_ids": ["synthetic_a"]}))
    frozen_profile = profile()
    if policy is not None:
        frozen_profile["current_initialization"] = policy
    profile_path = tmp_path / "profile.json"; profile_path.write_text(json.dumps(frozen_profile))
    index = tmp_path / "geometry.json"; index.write_text("{}")
    client = tmp_path / "client.py"; client.write_text("# synthetic-only source\n")
    kwargs = {"source_manifest": source, "source_map": mapping, "profile_path": profile_path,
        "geometry_index": index, "client_source": client, "output": tmp_path / "output",
        "project_root": tmp_path}
    packet = {"inputs": {name: file_ref(path) for name, path in {
        "source_manifest": source, "source_map": mapping, "profile": profile_path,
        "geometry_index": index, "client_source": client}.items()}, "budgets": {"samples_max": 1}}
    return kwargs, packet


@pytest.mark.parametrize("case", ["legacy", "expanded_legacy", "static_mesh"])
def test_prepare_consumes_signed_static_budget_and_legacy_defaults(tmp_path, monkeypatch, case):
    from chaoyang.humanego_ict import joint26_prepare as preparation
    from chaoyang.humanego_ict import offline_mink_provider_v1 as mathematics
    expanded = case != "legacy"
    policy = {"mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY", "settle_steps_max": 300,
              "geometry_calls_per_window_max": 5000} if expanded else None
    if case == "static_mesh":
        policy.update(mode="STATIC_FRAME_TASK_CURRENT_TARGET_ONLY", static_position_cost=100.0,
            static_orientation_cost=11.459155902616466, static_qp_damping=1e-6)
    kwargs, packet = _preparation_fixture(tmp_path, policy)
    if case == "static_mesh":
        frozen = json.loads(kwargs["profile_path"].read_text())
        frozen["collision_validation"] = "FROZEN_STL_MESH_AUDIT_V1"
        kwargs["profile_path"].write_text(json.dumps(frozen))
        packet["inputs"]["profile"] = file_ref(kwargs["profile_path"])
    if expanded:
        packet["budgets"].update(current_settle_steps_max=300, geometry_calls_per_window_max=5000)
    expected = {"current_settle_steps_max": 300 if expanded else 60,
                "geometry_calls_per_window_max": 5000 if expanded else 1000}
    expected_options = {"current_initialization_mode": "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"
        if case == "static_mesh" else "STATIC_NUMERICAL_CURRENT_TARGET_ONLY",
        "collision_validation": "FROZEN_STL_MESH_AUDIT_V1" if case == "static_mesh" else "ORIGINAL_CAPSULE_V1",
        "training_admission_policy": STRICT_ADMISSION_POLICY}
    seen = []
    class NoGeometryProvider:
        synthetic_home_q_rad = np.zeros(14)
        def __init__(self, **arguments):
            self.budget = arguments["budget"]
            assert self.budget.limit == expected["geometry_calls_per_window_max"]
            assert all(arguments[key] == value for key, value in expected_options.items())
        def assumed_profile(self, value): return deepcopy(value)
    def no_geometry_conversion(*args, **arguments):
        seen.append(arguments["settle_steps"])
        return {"status": "FAILED_SYNTHETIC_NO_GEOMETRY", "current_error": "SYNTHETIC_ONLY",
            "frame_errors": [["SYNTHETIC_ONLY"] for _ in range(30)], "arrays": None}
    monkeypatch.setattr(mathematics, "OfflineMinkProvider", NoGeometryProvider)
    monkeypatch.setattr(preparation, "convert_window", no_geometry_conversion)
    before = dict(SYNTHETIC_CALLS)
    result = prepare_dataset(**kwargs, packet=packet)
    assert seen == [expected["current_settle_steps_max"]]
    assert result["preparation_budgets"] == expected
    assert result["preparation_options"] == expected_options
    assert result["converted_windows"] == 0
    assert SYNTHETIC_CALLS == before


@pytest.mark.parametrize("field,value", [("current_settle_steps_max", 60),
    ("geometry_calls_per_window_max", 1000), ("current_settle_steps_max", True)])
def test_prepare_static_budget_drift_rejects_before_provider(tmp_path, field, value):
    policy = {"mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY", "settle_steps_max": 300,
              "geometry_calls_per_window_max": 5000}
    kwargs, packet = _preparation_fixture(tmp_path, policy)
    packet["budgets"].update(current_settle_steps_max=300, geometry_calls_per_window_max=5000)
    packet["budgets"][field] = value
    with pytest.raises(Joint26PreparationError, match="SIGNED_PREPARATION_BUDGET_MISMATCH"):
        prepare_dataset(**kwargs, packet=packet)
    assert not kwargs["output"].exists()


def mesh_static_profile():
    p = profile()
    p["current_initialization"] = {"mode": "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY",
        "settle_steps_max": 300, "geometry_calls_per_window_max": 5000,
        "static_position_cost": 100.0, "static_orientation_cost": 11.459155902616466,
        "static_qp_damping": 1e-6}
    p["collision_validation"] = "FROZEN_STL_MESH_AUDIT_V1"
    return p


class SyntheticStaticMeshProvider(SyntheticProvider):
    """Dispatch/report fixture only; no MuJoCo, meshes or actual static solver."""
    def __init__(self, fault=None):
        super().__init__(fault); self.static_calls = 0; self.static_targets = []
    def _answer(self, poses, seed):
        q = seed.copy(); q[:3], q[7:10] = poses[0, :3, 3], poses[1, :3, 3]
        checked = self.fault != "mesh_unknown"
        free = checked and self.fault not in {"mesh_failed", "qp_margin"}
        mesh_free = checked and self.fault != "mesh_failed"
        collision = {"collision_validation": "FROZEN_STL_MESH_AUDIT_V1",
            "collision_checked": checked, "collision_free": free,
            "legacy_capsule_diagnostics": {"collision_checked": True, "collision_free": False},
            "mesh_diagnostics": {"collision_checked": checked, "collision_free": mesh_free},
            "physical_scene_collision_status": "UNKNOWN"}
        if self.fault == "missing_mesh":
            collision.pop("mesh_diagnostics")
        if self.fault == "selected_drift":
            collision["collision_free"] = not free
        return IKSolution(q, checked, free, {"collision": collision,
            "geometry_qualified": False, "physical_deployment_authorized": False})
    def solve(self, poses, seed, dt):
        _count_synthetic_call("solve"); self.solve_calls += 1
        return self._answer(poses, seed)
    def solve_static(self, poses, seed, dt):
        _count_synthetic_call("solve_static"); self.static_calls += 1
        self.static_targets.append(poses.copy())
        return self._answer(poses, seed)


def test_static_current_dispatch_preserves_future_solver_and_dual_collision_evidence():
    provider, p = SyntheticStaticMeshProvider(), mesh_static_profile()
    result = convert(provider=provider, p=p, settle_steps=1)
    assert result["status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert provider.static_calls == 1 and provider.solve_calls == 30
    assert result["current_diagnostics"]["solver_mode"] == "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"
    assert all(row["solver_mode"] == "ORIGINAL_TELEOP_DIFFERENTIAL_IK"
               for row in result["frame_diagnostics"])
    assert all(row["transition_gate_enforced"] and row["collision_gate_enforced"]
               for row in result["frame_diagnostics"])
    reports = result["current_diagnostics"]["collision_reports"]
    assert reports["legacy_capsule"]["status"] == "FAILED"
    assert reports["mesh"]["status"] == reports["selected"]["status"] == "PASSED"
    assert reports["physical_scene_collision_status"] == "UNKNOWN"
    assert reports["physical_deployment_authorized"] is False
    assert result["numerical_limits"] == p["numerical_limits"]
    assert result["physical_deployment_authorized"] is result["control_ground_truth"] is False


def test_static_current_never_borrows_future_targets():
    labels, provider = original_labels(), SyntheticStaticMeshProvider()
    labels["y_action"][:, 0] += .3
    result = convert(labels, provider, mesh_static_profile(), settle_steps=1)
    assert provider.static_calls == 1 and provider.solve_calls == 1
    assert np.all(provider.static_targets[0][:, :3, 3] == 0)
    assert np.all(result["arrays"]["current_joint_q_rad"] == 0)
    assert result["current_status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert result["frame_errors"][0] == ["IK_STEP_OR_SPEED_LIMIT"]
    assert not result["arrays"]["action_valid_mask"].any()


def test_missing_static_method_rejects_without_legacy_fallback():
    provider = SyntheticProvider()
    with pytest.raises(Joint26PreparationError, match="STATIC_CURRENT_SOLVER_REQUIRED"):
        convert(provider=provider, p=mesh_static_profile(), settle_steps=1)
    assert provider.solve_calls == provider.fk_calls == 0


@pytest.mark.parametrize("fault", ["missing_mesh", "selected_drift"])
def test_mesh_collision_missing_or_drifting_evidence_never_admitted(fault):
    provider = SyntheticStaticMeshProvider(fault)
    result = convert(provider=provider, p=mesh_static_profile(), settle_steps=1)
    assert provider.static_calls == 1 and provider.solve_calls == provider.fk_calls == 0
    assert result["current_error"] == "SIGNED_MESH_COLLISION_DIAGNOSTICS_REQUIRED"
    assert result["current_diagnostics"]["selected_collision_status"] == "UNKNOWN"
    assert not result["arrays"]["state_mask"].any()
    assert not result["arrays"]["action_valid_mask"].any()


@pytest.mark.parametrize("fault,status", [("mesh_failed", "FAILED"), ("mesh_unknown", "UNKNOWN"),
                                        ("qp_margin", "FAILED")])
def test_static_fk_convergence_does_not_override_selected_collision(fault, status):
    provider = SyntheticStaticMeshProvider(fault)
    result = convert(provider=provider, p=mesh_static_profile(), settle_steps=1)
    assert result["current_kinematics_converged"] is True
    assert result["current_error"] == "CURRENT_IK_COLLISION_OR_UNCHECKED"
    reports = result["current_diagnostics"]["collision_reports"]
    assert reports["selected"]["status"] == status
    if fault == "qp_margin":
        assert reports["mesh"]["status"] == "PASSED"
    assert reports["legacy_capsule"]["status"] == "FAILED"
    assert not result["arrays"]["state_mask"].any()
    assert not result["arrays"]["action_valid_mask"].any()


@pytest.mark.parametrize("case", ["step", "speed"])
def test_static_solver_does_not_relax_future_transition_gates(case):
    labels, provider = original_labels(), SyntheticStaticMeshProvider()
    labels["y_action"][2:, 0] += .15 if case == "step" else .02
    result = convert(labels, provider, mesh_static_profile(), settle_steps=1)
    assert provider.static_calls == 1
    assert result["current_status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    assert result["arrays"]["action_valid_mask"][:2].all()
    assert not result["arrays"]["action_valid_mask"][2:].any()
    assert result["frame_errors"][2] == ["IK_STEP_OR_SPEED_LIMIT"]
    assert result["frame_diagnostics"][2]["solver_mode"] == "ORIGINAL_TELEOP_DIFFERENTIAL_IK"


def fixture_cache(tmp_path, n_train=16, n_val=3):
    data = convert()["arrays"]
    stream = io.BytesIO(); np.savez_compressed(stream, **data)
    (tmp_path / "labels.npz").write_bytes(stream.getvalue())
    image = Image.new("RGB", (320, 240), (1, 2, 3)); image.save(tmp_path / "rgb.png")
    diagnostic = {"status": "PASSED_ASSUMED_MODEL_NUMERICS_ONLY", "control_ground_truth": False,
        "physical_deployment_authorized": False, "current_status": "PASSED_ASSUMED_MODEL_NUMERICS_ONLY",
        "frame_errors": [[] for _ in range(30)], "valid_frames": 30, "robot_commands_sent": 0,
        "numerical_limits": profile()["numerical_limits"]}
    (tmp_path / "diag.json").write_text(json.dumps(diagnostic))
    samples = []
    for index in range(n_train + n_val):
        samples.append({"sample_id": "synthetic_" + str(index), "session_id": str(index),
            "source_frame_id": 0, "source_group_id": "synthetic_capture_" + str(index),
            "split": "train" if index < n_train else "validation", "sample_weight": 1,
            "labels": file_ref(tmp_path / "labels.npz"), "diagnostics": file_ref(tmp_path / "diag.json"),
            "conversion_status": "PASSED_ASSUMED_MODEL_NUMERICS_ONLY",
            "branches": {b: {"admitted": True, "rgb": file_ref(tmp_path / "rgb.png")}
                         for b in ("raw", "robotized")}})
    manifest = {"schema": SCHEMA, "action_dim": 26, "horizon": 30, "action_layout": LAYOUT,
        "ict_dim": 29, "max_ict": 8, "image_size": [240, 320], "source_policy": SOURCE_POLICY,
        "supervision_class": SUPERVISION_CLASS, "parameter_mode": "ASSUMED", "samples": samples,
        "normalization": {"method": "IDENTITY_NO_FIT"}, "control_ground_truth": False,
        "physical_deployment_authorized": False, "numerical_limits": profile()["numerical_limits"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "manifest.json", manifest


def test_preloaded_reader_preserves_16_3_and_defensive_arrays(tmp_path):
    path, _ = fixture_cache(tmp_path)
    dataset, qualification = load_joint26_cache(path, "raw", "train", project_root=tmp_path)
    assert len(dataset) == 16
    assert qualification["data_gate"]["status"] == "PASSED_RESEARCH_LABEL_GATE"
    assert qualification["data_gate"]["train_groups"] == 16
    assert qualification["data_gate"]["validation_groups"] == 3
    row = dataset[0]; row["y_action"][:] = 100
    assert (dataset[0]["y_action"] < 100).all()
    assert dataset[0]["x_state26"].shape == (26,)
    assert dataset[0]["x_rgb"].shape == (3, 240, 320)
    dataset.verify_all_dependencies()


def test_insufficient_groups_remain_blocked(tmp_path):
    path, _ = fixture_cache(tmp_path, 15, 2)
    assert Joint26Dataset(path, "robotized", "train", project_root=tmp_path).qualification["data_gate"]["status"] == "BLOCKED_DATA_VOLUME"


def test_group_leakage_and_sha_drift_reject(tmp_path):
    path, manifest = fixture_cache(tmp_path)
    manifest["samples"][-1]["source_group_id"] = manifest["samples"][0]["source_group_id"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(Joint26DataError, match="GROUP_LEAKAGE"):
        Joint26Dataset(path, "raw", project_root=tmp_path)
    manifest["samples"][-1]["source_group_id"] = "distinct"
    path.write_text(json.dumps(manifest)); dataset = Joint26Dataset(path, "raw", project_root=tmp_path)
    with (tmp_path / "rgb.png").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(Joint26DataError, match="SHA_OR_IDENTITY_DRIFT"):
        dataset.verify_all_dependencies()


def test_partial_window_never_counts_as_joint_training_group(tmp_path):
    path, _ = fixture_cache(tmp_path)
    with np.load(tmp_path / "labels.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["action_valid_mask"][0, 0] = False
    np.savez_compressed(tmp_path / "partial.npz", **arrays)
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["labels"] = file_ref(tmp_path / "partial.npz")
    path.write_text(json.dumps(manifest))
    with pytest.raises(Joint26DataError, match="INCOMPLETE_JOINT26"):
        Joint26Dataset(path, "raw", project_root=tmp_path)


def ego_profile():
    p = mesh_static_profile()
    p["training_admission_policy"] = EGO_ADMISSION_POLICY
    p["collision_validation"] = "ORIGINAL_CAPSULE_V1"
    p["current_initialization"].update(settle_steps_max=60, geometry_calls_per_window_max=5000)
    return p


class SyntheticRetargetProvider(SyntheticProvider):
    """Recorded projection fixture, never a physical IK implementation."""
    def __init__(self, fault=None):
        super().__init__(fault)
        self.current_calls, self.future_calls, self.current_targets, self.future_dts = 0, 0, [], []

    def fk(self, q):
        poses = super().fk(q)
        if self.fault == "all_quality":
            angle = .3
            poses[:, :3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        return poses

    def _retarget(self, poses, seed, *, current):
        raw = seed.copy(); raw[:3], raw[7:10] = poses[0, :3, 3], poses[1, :3, 3]
        if self.fault == "source_drift":
            raise ValueError("SOURCE_IDENTITY_DRIFT:frozen_xml")
        if self.fault == "solver_exception_future" and not current and self.future_calls == 4:
            raise RuntimeError("SYNTHETIC_RETARGET_SOLVER_FAILURE")
        if self.fault == "all_quality":
            raw[0] += .25 if current else .5
        elif self.fault in {"projection", "projection_drift"}:
            raw[0] = 3
        elif (self.fault == "nonfinite_current" and current
              or self.fault == "nonfinite_future" and not current):
            raw[0] = np.nan
        lower, upper = np.full(14, -2.), np.full(14, 2.)
        q = np.clip(raw, lower, upper)
        evidence = {"raw_q_rad": raw.tolist(), "projected_q_rad": q.tolist(),
            "projection_correction_rad": (q - raw).tolist(),
            "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
            "projection_applied": bool(np.any(q != raw)),
            "max_abs_projection_correction_rad": float(np.max(np.abs(q - raw)))}
        if self.fault == "projection_drift":
            evidence["projected_q_rad"][0] = 1.9
        return IKSolution(q, True, self.fault != "all_quality", {"retargeting": evidence,
            "collision": {"collision_validation": "ORIGINAL_CAPSULE_V1"}})

    def solve_retarget_current(self, poses, seed, dt):
        _count_synthetic_call("solve_retarget_current"); self.current_calls += 1
        self.current_targets.append(poses.copy())
        return self._retarget(poses, seed, current=True)

    def solve_retarget_future(self, poses, seed, dt):
        _count_synthetic_call("solve_retarget_future"); self.future_calls += 1
        self.future_dts.append(float(dt))
        return self._retarget(poses, seed, current=False)


def test_ego_structural_labels_keep_failed_quality_without_training_block():
    provider = SyntheticRetargetProvider("all_quality")
    result = convert(provider=provider, p=ego_profile())
    assert result["status"] == result["current_status"] == EGO_ADMITTED_STATUS
    assert result["training_admission_policy"] == EGO_ADMISSION_POLICY
    assert result["label_validity_basis"] == EGO_VALIDITY_BASIS
    assert result["training_quality_gates_enforced"] is result["robot_quality_passed"] is False
    assert result["current_kinematics_converged"] is False
    assert result["current_stop_reason"] == "FIXED_RESEARCH_COMPUTE_BUDGET_COMPLETED"
    assert provider.current_calls == result["settling_steps_used"] == 60
    assert provider.future_calls == 30
    assert provider.solve_calls == 0
    np.testing.assert_array_equal(provider.future_dts, [.05] * 30)
    assert result["frame_errors"] == [[] for _ in range(30)]
    assert result["valid_frames"] == 30 and result["arrays"]["action_valid_mask"].all()
    evidence = result["quality_diagnostics"]["frames"][0]
    assert evidence["collision_free"] is evidence["transition_limits_passed"] is False
    assert "FK_POSITION_TOLERANCE_EXCEEDED" in evidence["quality_errors"]
    assert "FK_ROTATION_TOLERANCE_EXCEEDED" in evidence["quality_errors"]
    assert "ASSUMED_MODEL_COLLISION_OR_UNCHECKED" in evidence["quality_errors"]
    assert "TELEOP_STEP_OR_SPEED_LIMIT_EXCEEDED" in evidence["quality_errors"]
    assert result["physical_deployment_authorized"] is result["control_ground_truth"] is False
    assert not result["arrays"]["state_mask"][HAND_INDICES].any()


def test_ego_projection_is_recorded_and_fk_uses_actual_projected_q():
    result = convert(provider=SyntheticRetargetProvider("projection"), p=ego_profile())
    assert result["status"] == EGO_ADMITTED_STATUS
    assert result["arrays"]["current_joint_q_rad"][0] == 2
    assert np.all(result["arrays"]["y_action"][:, 0] == 2)
    report = result["current_diagnostics"]["provider_diagnostics"]["retargeting"]
    assert report["raw_q_rad"][0] == 3 and report["projected_q_rad"][0] == 2
    assert report["projection_correction_rad"][0] == -1
    assert result["current_diagnostics"]["position_error_m"][0] == 2
    assert result["current_kinematics_converged"] is False


def test_ego_current_sixty_steps_never_reads_future_targets_or_time_grid():
    labels, provider = original_labels(), SyntheticRetargetProvider()
    before = convert(labels, provider, ego_profile())["arrays"]
    labels["y_action"][:, :6] += .8
    labels["target_qpc_ns"] += 1_000_000
    second = SyntheticRetargetProvider()
    after = convert(labels, second, ego_profile())["arrays"]
    assert provider.current_calls == second.current_calls == 60
    assert all(np.all(target[:, :3, 3] == 0) for target in second.current_targets)
    for key in ("x_ict", "ict_mask", "x_state26", "state_mask", "current_joint_q_rad"):
        np.testing.assert_array_equal(before[key], after[key])


def test_ego_keeps_invalid_original_source_mask_and_invalid_predecessor():
    labels, provider = original_labels(), SyntheticRetargetProvider()
    labels["action_valid_mask"][3, 0] = False
    result = convert(labels, provider, ego_profile())
    assert provider.current_calls == 60 and provider.future_calls == 3
    assert result["arrays"]["action_valid_mask"][:3].all()
    assert not result["arrays"]["action_valid_mask"][3:].any()
    assert result["frame_errors"][3] == ["SOURCE_ACTION_LABEL_INVALID"]
    assert result["frame_errors"][4] == ["INVALID_PREDECESSOR"]
    assert result["status"] != EGO_ADMITTED_STATUS


@pytest.mark.parametrize("fault", ["nonfinite_current", "nonfinite_future", "solver_exception_future", "projection_drift"])
def test_ego_invalid_solver_or_projection_never_fabricates_complete_labels(fault):
    result = convert(provider=SyntheticRetargetProvider(fault), p=ego_profile())
    assert result["status"] != EGO_ADMITTED_STATUS
    assert not result["arrays"]["action_valid_mask"].all()
    if fault in {"nonfinite_current", "projection_drift"}:
        assert not result["arrays"]["state_mask"].any()
    if fault == "solver_exception_future":
        assert result["arrays"]["action_valid_mask"][:3].all()
        assert "SYNTHETIC_RETARGET_SOLVER_FAILURE" in result["frame_errors"][3]


def test_ego_missing_explicit_provider_or_wrong_budget_rejects_before_geometry():
    provider = SyntheticProvider()
    with pytest.raises(Joint26PreparationError, match="EXPLICIT_RETARGET_PROVIDER"):
        convert(provider=provider, p=ego_profile())
    provider = SyntheticRetargetProvider()
    with pytest.raises(Joint26PreparationError, match="FIXED_SIXTY"):
        convert(provider=provider, p=ego_profile(), settle_steps=61)
    assert provider.current_calls == provider.future_calls == provider.fk_calls == 0


def fixture_ego_cache(tmp_path, *, n_train=16, n_val=3):
    path, manifest = fixture_cache(tmp_path, n_train, n_val)
    with np.load(tmp_path / "labels.npz") as handle:
        arrays = {key: handle[key].copy() for key in handle.files}
    # Structurally legal but explicitly violates old software speed/step gates.
    arrays["y_action"][:, 0] = np.where(np.arange(30) % 2, -1.5, 1.5)
    np.savez_compressed(tmp_path / "ego_labels.npz", **arrays)
    diagnostic = json.loads((tmp_path / "diag.json").read_text())
    diagnostic.update(status=EGO_ADMITTED_STATUS, current_status=EGO_ADMITTED_STATUS,
        training_admission_policy=EGO_ADMISSION_POLICY, label_validity_basis=EGO_VALIDITY_BASIS,
        training_quality_gates_enforced=False, robot_quality_passed=False, settling_steps_used=60,
        quality_diagnostics={"current": {"collision_free": False},
            "frames": [{"quality_errors": ["TELEOP_STEP_OR_SPEED_LIMIT_EXCEEDED"]} for _ in range(30)]})
    (tmp_path / "ego_diag.json").write_text(json.dumps(diagnostic))
    manifest.update(training_admission_policy=EGO_ADMISSION_POLICY,
        label_validity_basis=EGO_VALIDITY_BASIS, robot_quality_passed=False)
    for sample in manifest["samples"]:
        sample.update(conversion_status=EGO_ADMITTED_STATUS,
            labels=file_ref(tmp_path / "ego_labels.npz"), diagnostics=file_ref(tmp_path / "ego_diag.json"))
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_ego_cache_both_rgb_branches_keep_16_3_groups_and_no_quality_claim(tmp_path):
    path, _ = fixture_ego_cache(tmp_path)
    for branch in ("raw", "robotized"):
        for split in ("train", "validation"):
            dataset = Joint26Dataset(path, branch, split, project_root=tmp_path)
            assert dataset.qualification["data_gate"]["status"] == "PASSED_RESEARCH_LABEL_GATE"
            assert dataset.qualification["training_admission_policy"] == EGO_ADMISSION_POLICY
            assert dataset.qualification["robot_quality_passed"] is False
            assert dataset[0]["action_valid_mask"].all()
            assert dataset[0]["physical_deployment_authorized"] is False


@pytest.mark.parametrize("policy", [None, STRICT_ADMISSION_POLICY, "UNREGISTERED_RELAXED_POLICY"])
def test_ego_cache_never_implicitly_weakens_default_strict_mode(tmp_path, policy):
    path, manifest = fixture_ego_cache(tmp_path)
    if policy is None:
        manifest.pop("training_admission_policy")
    else:
        manifest["training_admission_policy"] = policy
    path.write_text(json.dumps(manifest))
    with pytest.raises(Joint26DataError):
        Joint26Dataset(path, "raw", project_root=tmp_path)


@pytest.mark.parametrize("fault", ["bounds", "nonfinite", "mask", "time", "group", "sha"])
def test_ego_structural_source_and_format_gates_are_not_removed(tmp_path, fault):
    path, manifest = fixture_ego_cache(tmp_path)
    if fault == "group":
        manifest["samples"][-1]["source_group_id"] = manifest["samples"][0]["source_group_id"]
    elif fault == "sha":
        with (tmp_path / "ego_labels.npz").open("ab") as stream:
            stream.write(b"drift")
    else:
        with np.load(tmp_path / "ego_labels.npz") as handle:
            arrays = {key: handle[key].copy() for key in handle.files}
        if fault == "bounds": arrays["y_action"][0, 0] = 3
        elif fault == "nonfinite": arrays["y_action"][0, 0] = np.nan
        elif fault == "mask": arrays["action_valid_mask"][0, 0] = False
        else: arrays["target_qpc_ns"][0] = arrays["anchor_qpc_ns"]
        np.savez_compressed(tmp_path / "broken.npz", **arrays)
        manifest["samples"][0]["labels"] = file_ref(tmp_path / "broken.npz")
    path.write_text(json.dumps(manifest))
    with pytest.raises(Joint26DataError):
        Joint26Dataset(path, "raw", project_root=tmp_path)


def test_ego_does_not_remove_independent_group_volume_gate(tmp_path):
    path, _ = fixture_ego_cache(tmp_path, n_train=15, n_val=2)
    dataset = Joint26Dataset(path, "raw", project_root=tmp_path)
    assert dataset.qualification["data_gate"]["status"] == "BLOCKED_DATA_VOLUME"


def test_prepare_publishes_explicit_ego_structural_admission_not_quality_pass(tmp_path):
    kwargs, packet = _preparation_fixture(tmp_path)
    kwargs["profile_path"].write_text(json.dumps(ego_profile()))
    packet["inputs"]["profile"] = file_ref(kwargs["profile_path"])
    packet["budgets"].update(current_settle_steps_max=60, geometry_calls_per_window_max=5000)
    provider = SyntheticRetargetProvider("all_quality")
    provider.synthetic_home_q_rad = np.zeros(14)
    provider.assumed_profile = lambda value: deepcopy(value)
    result = prepare_dataset(**kwargs, packet=packet, provider_factory=lambda sample: provider)
    assert result["converted_windows"] == 1
    manifest = json.loads((kwargs["output"] / "JOINT26_MANIFEST.json").read_text())
    assert manifest["training_admission_policy"] == EGO_ADMISSION_POLICY
    assert manifest["samples"][0]["conversion_status"] == EGO_ADMITTED_STATUS
    assert all(manifest["samples"][0]["branches"][b]["admitted"] for b in ("raw", "robotized"))
    assert manifest["robot_quality_passed"] is manifest["complete_product_quality_passed"] is False
    dataset = Joint26Dataset(kwargs["output"] / "JOINT26_MANIFEST.json", "raw", project_root=tmp_path)
    assert dataset.qualification["data_gate"]["status"] == "BLOCKED_DATA_VOLUME"
    assert dataset.qualification["training_admission_policy"] == EGO_ADMISSION_POLICY
