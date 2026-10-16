"""Bounded source-derived joint26 preparation, never a real-control producer.

The CLI checks its finite task before importing NumPy or geometry. Numerical
conversion can be tested with an explicitly synthetic offline provider. Missing
geometry/causal inputs never become identity transforms or fabricated states.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path


class Joint26PreparationError(ValueError):
    pass


def _plain_json(path):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise Joint26PreparationError("DUPLICATE_JSON_KEY:" + k)
            d[k] = v
        return d
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(Joint26PreparationError("NONFINITE_JSON:" + s)))


def _plain_ref(path):
    path = Path(path); raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_new(path, data):
    """Exclusive immutable publication; no overwrite, including on retry."""
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(data)
    return _plain_ref(path)


def _write_json(path, value):
    return _write_new(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      indent=2, allow_nan=False) + "\n").encode())


def _current_poses(labels, normalization):
    import numpy as np
    from .inference import _normalization, rot6_to_matrix
    mean, std = _normalization(normalization)
    tokens = np.asarray(labels["x_ict"])
    valid = np.asarray(labels["ict_mask"])
    if (tokens.shape != (8, 29) or valid.shape != (8,) or valid.dtype != np.dtype(bool)
            or not valid[:2].all() or valid[2:].any() or not np.isfinite(tokens).all()
            or np.any(tokens[~valid] != 0)):
        raise Joint26PreparationError("BILATERAL_CAUSAL_CURRENT_ICT_REQUIRED")
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    for side in range(2):
        if tokens[side, 0] != side + 1 or tokens[side, -1] != -1:
            raise Joint26PreparationError("CURRENT_ICT_TYPE_OR_UNKNOWN_GRASP_DRIFT")
        poses[side, :3, 3] = tokens[side, 1:4] * std + mean
        poses[side, :3, :3] = rot6_to_matrix(tokens[side, 4:10])
    return poses


def _error(provider, q, targets):
    import numpy as np
    from .bridge import _se3
    poses = _se3(provider.fk(q.copy()), (2, 4, 4), "joint26_fk")
    position = np.linalg.norm(poses[:, :3, 3] - targets[:, :3, 3], axis=1)
    relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ poses[:, :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    return position, rotation


def _raise_if_integrity(error):
    # Immutable-source drift is a task-wide stop, never a maskable IK failure.
    code = str(error)
    if any(marker in code for marker in ("SOURCE_SHA_DRIFT", "SOURCE_IDENTITY_DRIFT",
            "SOURCE_SHA_OR_IDENTITY_DRIFT", "SOURCE_BYTE_COUNT_DRIFT",
            "SOURCE_SIGNATURE_DRIFT", "SOURCE_REFERENCE_CHANGED")):
        raise error


def _collision_reports(answer, collision_validation):
    """Keep representation evidence separate from the selected numerical gate."""
    raw = deepcopy(answer.diagnostics)
    collision = raw.get("collision", {}) if isinstance(raw, dict) else {}
    if (isinstance(collision, dict) and "collision_validation" in collision
            and collision["collision_validation"] != collision_validation):
        raise Joint26PreparationError("SIGNED_COLLISION_REPRESENTATION_MISMATCH")
    mesh_mode = collision_validation == "FROZEN_STL_MESH_AUDIT_V1"
    if mesh_mode:
        required = ("legacy_capsule_diagnostics", "mesh_diagnostics")
        valid = (isinstance(collision, dict)
            and collision.get("collision_validation") == collision_validation
            and type(collision.get("collision_checked")) is bool
            and type(collision.get("collision_free")) is bool
            and collision["collision_checked"] is answer.collision_checked
            and collision["collision_free"] is answer.collision_free)
        if valid:
            valid = all(isinstance(collision.get(name), dict)
                and type(collision[name].get("collision_checked")) is bool
                and type(collision[name].get("collision_free")) is bool for name in required)
        if not valid:
            error = Joint26PreparationError("SIGNED_MESH_COLLISION_DIAGNOSTICS_REQUIRED")
            error.diagnostics = {"provider_diagnostics": raw,
                "collision_validation": collision_validation, "selected_collision_status": "UNKNOWN"}
            raise error
        legacy, mesh = (collision[name] for name in required)
    else:
        legacy = dict(collision) if isinstance(collision, dict) else {}
        legacy.update(collision_checked=answer.collision_checked is True,
                      collision_free=answer.collision_free is True)
        mesh = {"collision_checked": False, "collision_free": False,
                "reason": "MESH_AUDIT_NOT_SELECTED"}
    def with_status(report):
        report = deepcopy(report)
        report["status"] = ("UNKNOWN" if report.get("collision_checked") is not True
            else "PASSED" if report.get("collision_free") is True else "FAILED")
        return report
    return {"collision_validation": collision_validation,
        "selected": with_status({"collision_checked": answer.collision_checked is True,
                                 "collision_free": answer.collision_free is True}),
        "legacy_capsule": with_status(legacy), "mesh": with_status(mesh),
        "physical_scene_collision_status": "UNKNOWN", "physical_deployment_authorized": False}


def _step(provider, targets, seed, dt, limits, *, enforce_collision=True, enforce_transition=True,
          static_current=False, collision_validation="ORIGINAL_CAPSULE_V1"):
    import numpy as np
    from .bridge import IKSolution, _array
    lower, upper, speed, step, tolerances = limits
    # Only the causal current-pose caller can opt in. Future stages leave this
    # false and retain the original teleoperation solver and actual time step.
    solve = provider.solve_static if static_current else provider.solve
    answer = solve(targets.copy(), seed.copy(), float(dt))
    if not isinstance(answer, IKSolution):
        raise Joint26PreparationError("INVALID_IK_SOLUTION_TYPE")
    collision_reports = _collision_reports(answer, collision_validation)
    solver_mode = "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY" if static_current else "ORIGINAL_TELEOP_DIFFERENTIAL_IK"
    evidence = {"solver_mode": solver_mode, "collision_reports": collision_reports,
                "provider_diagnostics": deepcopy(answer.diagnostics), "dt_s": float(dt)}
    q = _array(answer.q_rad, (14,), "joint26_ik_q_rad")
    if np.any(q < lower) or np.any(q > upper):
        error = Joint26PreparationError("IK_JOINT_LIMIT"); error.diagnostics = evidence
        raise error
    delta = np.abs(q - seed)
    transition = {"transition_gate_enforced": enforce_transition,
        "actual_max_step_rad": float(np.max(delta)),
        "actual_max_speed_rad_s": float(np.max(delta / dt)),
        "transition_limits_passed": bool(np.all(delta <= step) and np.all(delta <= speed * dt)),
        "dt_s": float(dt)}
    if enforce_transition and not transition["transition_limits_passed"]:
        error = Joint26PreparationError("IK_STEP_OR_SPEED_LIMIT")
        error.diagnostics = {**evidence, **transition}
        raise error
    if enforce_collision and (answer.collision_checked is not True or answer.collision_free is not True):
        error = Joint26PreparationError("IK_COLLISION_OR_UNCHECKED"); error.diagnostics = evidence
        raise error
    position, rotation = _error(provider, q, targets)
    return q, bool(np.all(position <= tolerances[0]) and np.all(rotation <= tolerances[1])), {
        "position_error_m": position.tolist(), "rotation_error_rad": rotation.tolist(), "dt_s": float(dt),
        "provider_sha256": provider.signature, "collision_checked": answer.collision_checked is True,
        "collision_free": answer.collision_free is True, "collision_gate_enforced": enforce_collision,
        "solver_mode": solver_mode,
        "collision_reports": collision_reports,
        "provider_diagnostics": deepcopy(answer.diagnostics), **transition}


def _retarget_step(provider, targets, seed, dt, limits, *, current):
    """Finite projected research labels; fit/collision/transition are evidence.

    The provider owns the explicitly recorded projection and recomputes its
    diagnostics there. No failed precision/safety check is rewritten as PASS.
    """
    import numpy as np
    from .bridge import IKSolution, _array
    lower, upper, speed, step, tolerances = limits
    solve = provider.solve_retarget_current if current else provider.solve_retarget_future
    answer = solve(targets.copy(), seed.copy(), float(dt))
    if not isinstance(answer, IKSolution):
        raise Joint26PreparationError("INVALID_IK_SOLUTION_TYPE")
    q = _array(answer.q_rad, (14,), "ego_derived_q_rad")
    raw_report = answer.diagnostics.get("retargeting") if isinstance(answer.diagnostics, dict) else None
    if not isinstance(raw_report, dict):
        raise Joint26PreparationError("EGO_RECORDED_BOUNDS_PROJECTION_REQUIRED")
    raw = _array(raw_report.get("raw_q_rad"), (14,), "ego_raw_q_rad")
    projected = _array(raw_report.get("projected_q_rad"), (14,), "ego_projected_q_rad")
    correction = _array(raw_report.get("projection_correction_rad"), (14,), "ego_projection_correction_rad")
    supplied_lower = _array(raw_report.get("joint_lower_rad"), (14,), "ego_projection_lower_rad")
    supplied_upper = _array(raw_report.get("joint_upper_rad"), (14,), "ego_projection_upper_rad")
    if (not np.array_equal(lower, supplied_lower) or not np.array_equal(upper, supplied_upper)
            or not np.array_equal(projected, np.clip(raw, lower, upper))
            or not np.array_equal(q, projected) or not np.array_equal(correction, projected - raw)
            or np.any(q < lower) or np.any(q > upper)):
        raise Joint26PreparationError("EGO_PROJECTION_EVIDENCE_OR_BOUNDS_DRIFT")
    collision_reports = _collision_reports(answer, "ORIGINAL_CAPSULE_V1")
    position, rotation = _error(provider, q, targets)
    delta = np.abs(q - seed)
    position_passed, rotation_passed = (bool(np.all(position <= tolerances[0])),
                                      bool(np.all(rotation <= tolerances[1])))
    transition_passed = bool(np.all(delta <= step) and np.all(delta <= speed * dt))
    quality_errors = []
    if not position_passed:
        quality_errors.append("FK_POSITION_TOLERANCE_EXCEEDED")
    if not rotation_passed:
        quality_errors.append("FK_ROTATION_TOLERANCE_EXCEEDED")
    if answer.collision_checked is not True or answer.collision_free is not True:
        quality_errors.append("ASSUMED_MODEL_COLLISION_OR_UNCHECKED")
    if not transition_passed:
        quality_errors.append("TELEOP_STEP_OR_SPEED_LIMIT_EXCEEDED")
    return q, position_passed and rotation_passed, {
        "solver_mode": "EGO_CURRENT_STATIC_DERIVATION" if current else "EGO_FUTURE_ORIGINAL_DIFFERENTIAL_IK",
        "provider_sha256": provider.signature, "provider_diagnostics": deepcopy(answer.diagnostics),
        "position_error_m": position.tolist(), "rotation_error_rad": rotation.tolist(),
        "position_tolerance_passed": position_passed, "rotation_tolerance_passed": rotation_passed,
        "collision_checked": answer.collision_checked is True,
        "collision_free": answer.collision_free is True, "collision_reports": collision_reports,
        "collision_gate_enforced": False, "transition_gate_enforced": False,
        "transition_limits_passed": transition_passed,
        "actual_max_step_rad": float(np.max(delta)), "actual_max_speed_rad_s": float(np.max(delta / dt)),
        "dt_s": float(dt), "quality_errors": quality_errors,
        "quality_gates_used_for_training_admission": False,
        "physical_deployment_authorized": False}


def convert_window(labels, *, assumption_profile, provider, initial_q_rad,
                   normalization, source_frame_id, settle_steps=60, substeps=5):
    """Current pose + first30 native targets -> explicitly assumed joint labels.

    Strict mode retains FK/collision and actual-time transition qualification.
    Explicit Ego mode uses sixty causal current iterations and one future solve
    per native target; recorded bounded projection supplies structural labels,
    while fit/collision/transition checks remain diagnostics, never quality PASS.
    Failed inputs/solvers stay invalid. Unobserved current hands stay UNKNOWN.
    """
    import numpy as np
    from .assumed_deployment_v1 import _limits, _profile, _provider
    from .bridge import _array, _se3
    from .inference import build_hand_ict, rot6_to_matrix
    from .joint26_data import (ARM_INDICES, HAND_INDICES, STRICT_ADMISSION_POLICY,
                              EGO_ADMISSION_POLICY, EGO_ADMITTED_STATUS, EGO_VALIDITY_BASIS)
    from .joint26_runtime import preparation_options

    if (type(settle_steps) is not int or not 1 <= settle_steps <= 300
            or type(substeps) is not int or not 1 <= substeps <= 5):
        raise Joint26PreparationError("FINITE_FROZEN_SETTLING_LIMIT_REQUIRED")
    profile = deepcopy(assumption_profile)
    options = preparation_options(profile)
    admission_policy = options.get("training_admission_policy", STRICT_ADMISSION_POLICY)
    ego_derived = admission_policy == EGO_ADMISSION_POLICY
    if ego_derived and (settle_steps != 60 or options["collision_validation"] != "ORIGINAL_CAPSULE_V1"):
        raise Joint26PreparationError("EGO_FIXED_SIXTY_CURRENT_STEPS_ORIGINAL_CAPSULE_REQUIRED")
    if ego_derived and any(not callable(getattr(provider, name, None)) for name in
                           ("solve_retarget_current", "solve_retarget_future")):
        raise Joint26PreparationError("EGO_EXPLICIT_RETARGET_PROVIDER_REQUIRED")
    static_current = options["current_initialization_mode"] == "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"
    if not ego_derived and static_current and not callable(getattr(provider, "solve_static", None)):
        raise Joint26PreparationError("STATIC_CURRENT_SOLVER_REQUIRED")
    effective_substeps = 1 if ego_derived else substeps
    admitted_status = EGO_ADMITTED_STATUS if ego_derived else "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    if (profile.get("initial_state_source") != "FROZEN_XML_HOME_NOT_MEASURED_FEEDBACK"
            or profile.get("camera_transform_policy") not in {
                "FIXED_EXPLICIT_PROFILE", "SOURCE_LABEL_REFERENCE_CAMERA_TO_WORLD"}):
        raise Joint26PreparationError("EXPLICIT_CURRENT_STATE_AND_CAMERA_RULE_REQUIRED")
    settle_dt = profile.get("settling_dt_s")
    if (isinstance(settle_dt, bool) or not isinstance(settle_dt, (int, float))
            or not np.isfinite(settle_dt) or not 0 < settle_dt <= 1):
        raise Joint26PreparationError("EXPLICIT_CAUSAL_NUMERICAL_SETTLING_DT_REQUIRED")
    if profile["camera_transform_policy"] == "SOURCE_LABEL_REFERENCE_CAMERA_TO_WORLD":
        profile.setdefault("transforms", {})["T_world_from_camera"] = _se3(
            labels["reference_camera_to_world"], (4, 4), "source_reference_camera_to_world").tolist()
    matrices = _profile(profile)
    if not _provider(provider, "solve") or not callable(getattr(provider, "fk", None)):
        raise Joint26PreparationError("SIGNED_OFFLINE_IK_FK_PROVIDER_REQUIRED")
    limits = _limits(profile)
    seed = _array(initial_q_rad, (14,), "explicit_xml_home_initial_q_rad")
    if np.any(seed < limits[0]) or np.any(seed > limits[1]):
        raise Joint26PreparationError("INITIAL_JOINT_LIMIT")
    if type(source_frame_id) is not int or source_frame_id < 0:
        raise Joint26PreparationError("INTEGER_SOURCE_FRAME_ID_REQUIRED")
    action = np.asarray(labels["y_action"])
    source_mask = np.asarray(labels["action_valid_mask"])
    if (action.shape != (50, 30) or source_mask.shape != (50, 30)
            or source_mask.dtype != np.dtype(bool) or not np.isfinite(action).all()):
        raise Joint26PreparationError("EXACT_FROZEN_H50_WRIST_LABEL_REQUIRED")
    anchor_array = np.asarray(labels["anchor_qpc_ns"])
    target_times = np.asarray(labels["target_qpc_ns"])
    target_ids = np.asarray(labels["target_source_frame_ids"])
    current_times = np.asarray(labels["conditioning_source_qpc_ns"])
    if (anchor_array.shape != () or anchor_array.dtype.kind not in "iu"
            or target_times.shape != (50,) or target_times.dtype.kind not in "iu"
            or target_ids.shape != (50,) or target_ids.dtype.kind not in "iu"
            or current_times.shape != (8,) or current_times.dtype.kind not in "iu"):
        raise Joint26PreparationError("EXACT_NATIVE_INTEGER_FRAME_TIME_MAPPING_REQUIRED")
    anchor = int(anchor_array)
    times, ids = target_times[:30].astype(np.int64), target_ids[:30].astype(np.int64)
    if (anchor < 0 or np.any(times <= anchor) or np.any(np.diff(times) <= 0)
            or np.any(ids <= source_frame_id) or np.any(np.diff(ids) <= 0)
            or np.any(current_times[:2] < 0) or np.any(current_times[:2] > anchor)):
        raise Joint26PreparationError("NONCAUSAL_CURRENT_OR_INVALID_FUTURE_MAPPING")
    native_dt = np.diff(np.concatenate(([anchor], times))) / 1e9
    result = {"status": "FAILED_EGO_INPUT_OR_DERIVATION" if ego_derived else "FAILED_ASSUMED_MODEL_NUMERICS",
        "parameter_mode": "ASSUMED", "training_admission_policy": admission_policy,
        "label_validity_basis": EGO_VALIDITY_BASIS if ego_derived else "STRICT_ASSUMED_MODEL_NUMERICS_ONLY",
        "training_quality_gates_enforced": not ego_derived, "robot_quality_passed": False,
        "initial_state_source": profile["initial_state_source"], "current_hand_status": "UNKNOWN_MASKED",
        "provider_sha256": provider.signature, "control_ground_truth": False,
        "physical_deployment_authorized": False, "robot_commands_sent": 0,
        "frame_errors": [[] for _ in range(30)], "frame_diagnostics": [{} for _ in range(30)],
        "current_status": "FAILED", "current_error": None, "settling_steps_used": 0,
        "current_settle_steps_max": settle_steps, "current_kinematics_converged": False,
        "current_stop_reason": "NOT_STARTED",
        "preparation_options": deepcopy(options),
        "current_optimization_diagnostics": [],
        "quality_diagnostics": {"current": {}, "frames": [{} for _ in range(30)]},
        "future_substeps_max": effective_substeps,
        "current_intermediate_state_policy": "NUMERICAL_OPTIMIZER_DIAGNOSTICS_ONLY_NEVER_LABEL_ICT_OR_EXECUTION",
        "geometry_calls_used": None, "numerical_limits": deepcopy(profile["numerical_limits"]),
        "camera_transform_policy": profile["camera_transform_policy"],
        "used_profile": deepcopy(profile),
        "used_profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "effective_transforms": {key: value.tolist() for key, value in matrices.items()},
        "source_reference_camera_to_world": np.asarray(labels["reference_camera_to_world"]).tolist()}
    arrays = {"x_ict": np.zeros((8, 29), np.float32), "ict_mask": np.zeros(8, bool),
        "x_state26": np.zeros(26, np.float64), "state_mask": np.zeros(26, bool),
        "y_action": np.zeros((30, 26), np.float64), "action_valid_mask": np.zeros((30, 26), bool),
        "anchor_qpc_ns": np.array(anchor, np.int64), "target_qpc_ns": times.copy(),
        "target_source_frame_ids": ids.copy(), "reference_camera_to_world": np.asarray(labels["reference_camera_to_world"]).copy(),
        "effective_camera_to_world": matrices["T_world_from_camera"].copy(),
        "current_joint_q_rad": np.zeros(14, np.float64), "current_joint_valid": np.zeros(14, bool),
        "conditioning_source_qpc_ns": current_times.copy(), "native_target_dt_s": native_dt.copy()}
    result["arrays"] = arrays
    world_base_inv = np.linalg.inv(matrices["T_world_from_arm_base"])
    tool_prior_inv = np.linalg.inv(matrices["T_model_tool_from_training_wrist_prior"])
    to_tool = lambda poses: world_base_inv @ matrices["T_world_from_camera"] @ poses @ tool_prior_inv
    try:
        current_target = to_tool(_current_poses(labels, normalization))
        converged = False
        # Frozen numerical scale is independent of future target values/times;
        # it is not a claim that synthetic home historically moved this way.
        for iteration in range(settle_steps):
            if ego_derived:
                seed, converged, diagnostics = _retarget_step(provider, current_target, seed,
                    settle_dt, limits, current=True)
            else:
                seed, converged, diagnostics = _step(provider, current_target, seed, settle_dt, limits,
                    enforce_collision=False, enforce_transition=False, static_current=static_current,
                    collision_validation=options["collision_validation"])
            result["settling_steps_used"] = iteration + 1
            result["current_diagnostics"] = diagnostics
            result["current_optimization_diagnostics"].append(deepcopy(diagnostics))
            result["current_kinematics_converged"] = converged
            result["quality_diagnostics"]["current"] = deepcopy(diagnostics)
            result["current_stop_reason"] = "IN_PROGRESS"
            # Intermediate optimization is not a control path. The legacy
            # strict mode still requires final FK/collision qualification;
            # Ego mode executes its fixed resource recipe without this gate.
            if not ego_derived and converged and diagnostics["collision_checked"] and diagnostics["collision_free"]:
                result["current_stop_reason"] = "CONVERGED_FK_AND_COLLISION"
                break
        if ego_derived:
            result["current_stop_reason"] = "FIXED_RESEARCH_COMPUTE_BUDGET_COMPLETED"
        elif not converged:
            result["current_stop_reason"] = "SETTLE_ITERATION_BUDGET_EXHAUSTED"
            raise Joint26PreparationError("CURRENT_IK_NOT_CONVERGED_WITHIN_BUDGET")
        if not ego_derived and (not diagnostics["collision_checked"] or not diagnostics["collision_free"]):
            result["current_stop_reason"] = "COLLISION_OR_UNCHECKED_AFTER_SETTLE_BUDGET"
            raise Joint26PreparationError("CURRENT_IK_COLLISION_OR_UNCHECKED")
        arrays["current_joint_q_rad"] = seed.copy()
        arrays["current_joint_valid"][:] = True
        arrays["x_state26"][ARM_INDICES] = seed.copy()
        arrays["state_mask"][ARM_INDICES] = True
        # Only current/past input contributes these rebuilt ICT tokens.
        current_tool = provider.fk(seed.copy())
        world_prior = matrices["T_world_from_arm_base"] @ current_tool @ matrices["T_model_tool_from_training_wrist_prior"]
        rebuilt = build_hand_ict(world_prior, np.ones(2, bool), matrices["T_world_from_camera"],
            normalization=normalization, source_qpc_ns=current_times[:2].astype(np.int64), anchor_qpc_ns=anchor)
        arrays["x_ict"], arrays["ict_mask"] = rebuilt["x_ict"].copy(), rebuilt["ict_mask"].copy()
        result["current_status"] = admitted_status
    except Exception as error:
        _raise_if_integrity(error)
        if result["current_stop_reason"] in {"NOT_STARTED", "IN_PROGRESS"}:
            result["current_stop_reason"] = "ERROR:" + str(error)
        if hasattr(error, "diagnostics"):
            result["current_diagnostics"] = deepcopy(error.diagnostics)
        result["current_error"] = str(error)
        result["frame_errors"] = [["INVALID_CURRENT_PREDECESSOR:" + str(error)] for _ in range(30)]
        if getattr(provider, "budget", None) is not None:
            result["geometry_calls_used"] = provider.budget.used
        return result
    predecessor_valid = True
    for frame in range(30):
        try:
            if not predecessor_valid:
                raise Joint26PreparationError("INVALID_PREDECESSOR")
            if not source_mask[frame].all():
                raise Joint26PreparationError("SOURCE_ACTION_LABEL_INVALID")
            hand = action[frame, 18:].reshape(2, 6)
            if np.any(hand < 0) or np.any(hand > 1):
                raise Joint26PreparationError("SOURCE_HAND_OUTSIDE_NORMALIZED_RANGE")
            camera_poses = np.repeat(np.eye(4)[None], 2, axis=0)
            camera_poses[:, :3, 3] = action[frame, :6].reshape(2, 3)
            camera_poses[:, :3, :3] = rot6_to_matrix(action[frame, 6:18].reshape(2, 6))
            targets, previous = to_tool(camera_poses), seed.copy()
            converged = False
            # Every allowed substep receives dt/N, never N times the real dt.
            for iteration in range(effective_substeps):
                if ego_derived:
                    seed, converged, diagnostics = _retarget_step(provider, targets, seed,
                        native_dt[frame], limits, current=False)
                else:
                    seed, converged, diagnostics = _step(provider, targets, seed,
                        native_dt[frame] / effective_substeps, limits,
                        collision_validation=options["collision_validation"])
                if not ego_derived and converged:
                    break
            if not ego_derived and not converged:
                raise Joint26PreparationError("IK_UNREACHABLE_FK_RESIDUAL")
            if not ego_derived and (np.any(np.abs(seed - previous) > limits[3]) or np.any(np.abs(seed - previous) > limits[2] * native_dt[frame])):
                raise Joint26PreparationError("IK_FINAL_STEP_OR_SPEED_LIMIT")
            arrays["y_action"][frame] = np.concatenate((seed[:7], hand[0], seed[7:], hand[1]))
            arrays["action_valid_mask"][frame] = True
            result["frame_diagnostics"][frame] = {**diagnostics, "substeps_used": iteration + 1,
                                                   "native_dt_s": float(native_dt[frame])}
            result["quality_diagnostics"]["frames"][frame] = deepcopy(result["frame_diagnostics"][frame])
        except Exception as error:
            _raise_if_integrity(error)
            predecessor_valid = False
            result["frame_errors"][frame] = [str(error)]
            result["frame_diagnostics"][frame] = deepcopy(getattr(error, "diagnostics", {}))
    if arrays["action_valid_mask"].all():
        result["status"] = admitted_status
    if getattr(provider, "budget", None) is not None:
        result["geometry_calls_used"] = provider.budget.used
    result["valid_frames"] = int(arrays["action_valid_mask"].all(axis=1).sum())
    result["input_seed_sha256"] = hashlib.sha256(np.asarray(initial_q_rad, np.float64).tobytes()).hexdigest()
    return result


def prepare_dataset(*, source_manifest, source_map, profile_path, geometry_index,
                    client_source, output, packet, project_root=None, provider_factory=None):
    """Called only after the root-owned finite runtime authorization succeeds."""
    import numpy as np
    from .joint26_data import (SCHEMA, LAYOUT, SOURCE_POLICY, SUPERVISION_CLASS,
                               STRICT_ADMISSION_POLICY, EGO_ADMISSION_POLICY,
                               EGO_ADMITTED_STATUS, EGO_VALIDITY_BASIS,
                               file_ref, reference_bytes, _json, _ordinary)
    from .offline_mink_provider_v1 import GeometryCallBudget, OfflineMinkProvider
    from .assumed_deployment_v1 import _profile
    from .joint26_runtime import preparation_budgets, preparation_options
    root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    if not output.is_relative_to(root) or ".." in output.parts or any(p.is_symlink() for p in (*reversed(output.parents), output)):
        raise Joint26PreparationError("PROJECT_ONLY_EXCLUSIVE_OUTPUT_REQUIRED")
    if output.exists():
        raise Joint26PreparationError("NEW_ATTEMPT_OUTPUT_REQUIRED")
    inputs = {name: file_ref(_ordinary(value, root)) for name, value in {
        "source_manifest": source_manifest, "source_map": source_map, "profile": profile_path,
        "geometry_index": geometry_index, "client_source": client_source}.items()}
    for name, reference in inputs.items():
        if packet.get("inputs", {}).get(name) != reference:
            raise Joint26PreparationError("PREPARE_PACKET_INPUT_SIGNATURE_REQUIRED:" + name)
    source = _json(reference_bytes(inputs["source_manifest"], root))
    mapping = _json(reference_bytes(inputs["source_map"], root))
    explicit_profile = _json(reference_bytes(inputs["profile"], root)); _profile(explicit_profile)
    preparation = preparation_budgets(explicit_profile)
    options = preparation_options(explicit_profile)
    admission_policy = options.get("training_admission_policy", STRICT_ADMISSION_POLICY)
    ego_derived = admission_policy == EGO_ADMISSION_POLICY
    admitted_status = EGO_ADMITTED_STATUS if ego_derived else "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
    defaults = {"current_settle_steps_max": 60, "geometry_calls_per_window_max": 1000}
    for name, value in preparation.items():
        signed_value = packet.get("budgets", {}).get(name, defaults[name])
        if type(signed_value) is not int or signed_value != value:
            raise Joint26PreparationError("SIGNED_PREPARATION_BUDGET_MISMATCH:" + name)
    if (mapping.get("production_manifest_ref") != inputs["source_manifest"]
            or mapping.get("branch") != "raw" or mapping.get("split") != "train"
            or source.get("control_ground_truth") is not False
            or source.get("physical_deployment_authorized") is not False):
        raise Joint26PreparationError("EXACT_HISTORIC_ADMITTED_RESEARCH_SOURCE_REQUIRED")
    train_ids = set(mapping["sample_ids"])
    max_samples = packet.get("budgets", {}).get("samples_max")
    if type(max_samples) is not int or not 1 <= max_samples <= 693:
        raise Joint26PreparationError("FINITE_PREPARE_SAMPLE_BUDGET_REQUIRED")
    selected_ids = packet.get("selected_sample_ids")
    samples = source["samples"]
    if selected_ids is not None:
        if not isinstance(selected_ids, list) or len(set(selected_ids)) != len(selected_ids):
            raise Joint26PreparationError("EXACT_SELECTED_SAMPLE_IDS_REQUIRED")
        selected = set(selected_ids)
        samples = [s for s in samples if s["sample_id"] in selected]
        if len(samples) != len(selected):
            raise Joint26PreparationError("SELECTED_SAMPLE_ID_ABSENT")
    if len(samples) > max_samples:
        raise Joint26PreparationError("PREPARE_SAMPLE_BUDGET_EXCEEDED")
    # Fully verify local frozen payload before the first mathematical call.
    for sample in samples:
        if sample["split"] == "train" and sample["sample_id"] not in train_ids:
            if sample["branch_admission"]["raw"]:
                raise Joint26PreparationError("RAW_TRAIN_ANCHOR_NOT_IN_FROZEN_MAP")
        reference_bytes(sample["labels"], root)
        for branch in ("raw", "robotized"):
            if sample["branch_admission"][branch]:
                reference_bytes(sample["rgb"][branch], root)
                receipt = _json(reference_bytes(sample["causal_receipts"][branch], root))
                if (receipt.get("input_class") != "CAUSAL_TRAINING_INPUT"
                        or receipt.get("sample_id") != sample["sample_id"]
                        or receipt.get("source_frame_id") != sample["source_frame_id"]
                        or receipt.get("control_ground_truth") is not False):
                    raise Joint26PreparationError("FROZEN_CAUSAL_RGB_RECEIPT_REQUIRED")
    output.mkdir(parents=True)
    rows, errors, limits = [], {}, None
    shared_provider = None
    for sample in samples:
        if provider_factory is None:
            if shared_provider is None:
                shared_provider = OfflineMinkProvider(project_root=root, source_index_path=geometry_index,
                    expected_index_sha256=inputs["geometry_index"]["sha256"], client_source_path=client_source,
                    budget=GeometryCallBudget(preparation["geometry_calls_per_window_max"], hard_limit=5000),
                    **options)
            else:
                # One frozen backend/source closure, independent explicit home
                # seed per window, and a fresh finite budget. sync_config in
                # every solve prevents the previous window becoming a seed.
                shared_provider.budget = GeometryCallBudget(
                    preparation["geometry_calls_per_window_max"], hard_limit=5000)
            provider = shared_provider
        else:
            provider = provider_factory(sample)
        profile = provider.assumed_profile(explicit_profile)
        limits = deepcopy(profile["numerical_limits"])
        with np.load(io.BytesIO(reference_bytes(sample["labels"], root)), allow_pickle=False) as z:
            old = {key: z[key].copy() for key in z.files}
        try:
            converted = convert_window(old, assumption_profile=profile, provider=provider,
                initial_q_rad=provider.synthetic_home_q_rad, normalization=source["normalization"],
                source_frame_id=sample["source_frame_id"],
                settle_steps=preparation["current_settle_steps_max"])
        except Exception as error:
            _raise_if_integrity(error)
            converted = {"status": "FAILED_INPUT_OR_GEOMETRY", "current_error": str(error),
                "frame_errors": [[str(error)] for _ in range(30)], "control_ground_truth": False,
                "physical_deployment_authorized": False, "arrays": None}
        arrays = converted.pop("arrays")
        converted.update(profile_source_ref=deepcopy(inputs["profile"]),
                         preparation_budgets=deepcopy(preparation),
                         preparation_options=deepcopy(options),
                         source_labels_ref=deepcopy(sample["labels"]),
                         source_manifest_ref=deepcopy(inputs["source_manifest"]))
        identifier = sample["sample_id"]
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier):
            raise Joint26PreparationError("SAFE_SAMPLE_IDENTIFIER_REQUIRED")
        label_ref = None
        if arrays is not None:
            stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
            label_ref = _write_new(output / (identifier + ".labels.npz"), stream.getvalue())
        diagnostic_ref = _write_json(output / (identifier + ".diagnostics.json"), converted)
        accepted = converted["status"] == admitted_status
        for code in {e for row in converted["frame_errors"] for e in row}:
            errors[code] = errors.get(code, 0) + 1
        rows.append({"sample_id": identifier, "session_id": sample["session_id"],
            "source_frame_id": sample["source_frame_id"], "source_group_id": sample["source_group_id"],
            "split": sample["split"], "sample_weight": sample.get("sample_weight", 1),
            "source_labels": deepcopy(sample["labels"]), "labels": label_ref,
            "diagnostics": diagnostic_ref, "conversion_status": converted["status"],
            "branches": {b: {"admitted": bool(accepted and sample["branch_admission"][b]),
                "historical_rgb_admitted": bool(sample["branch_admission"][b]),
                "rgb": deepcopy(sample["rgb"][b]), "causal_receipt": deepcopy(sample["causal_receipts"][b])}
                for b in ("raw", "robotized")},
            "time_support": {"source_clock": source.get("pose_clock"), "timebase_verified": False,
                "mode": "SOURCE_DECLARED_NATIVE_MAPPING_PRESERVED"}})
    matrix_profile = _write_json(output / "EXPLICIT_INPUT_PROFILE.json", explicit_profile)
    manifest = {"schema": SCHEMA, "action_dim": 26, "horizon": 30, "action_layout": LAYOUT,
        "ict_dim": 29, "max_ict": 8, "image_size": [240, 320], "source_policy": SOURCE_POLICY,
        "supervision_class": SUPERVISION_CLASS, "parameter_mode": "ASSUMED",
        "training_admission_policy": admission_policy,
        "label_validity_basis": EGO_VALIDITY_BASIS if ego_derived else "STRICT_ASSUMED_MODEL_NUMERICS_ONLY",
        "training_quality_gates_enforced": not ego_derived, "robot_quality_passed": False,
        "normalization": {"method": "IDENTITY_NO_FIT"}, "control_ground_truth": False,
        "physical_deployment_authorized": False, "inputs": inputs, "samples": rows,
        "preparation_budgets": deepcopy(preparation),
        "preparation_options": deepcopy(options),
        "explicit_profile": matrix_profile, "numerical_limits": limits,
        "current_hand_state_policy": "UNKNOWN_ZERO_WITH_FALSE_MASK_NO_FUTURE_COPY",
        "timebase_qualification": "SOURCE_DECLARED_NOT_PHYSICALLY_VERIFIED",
        "physical_eye_identity_verified": source.get("physical_eye_identity_verified", False),
        "complete_product_quality_passed": False, "historical_visual_fail_unknown_preserved": True}
    manifest_ref = _write_json(output / "JOINT26_MANIFEST.json", manifest)
    counts = {b: {s: {"samples": sum(r["branches"][b]["admitted"] and r["split"] == s for r in rows),
                          "source_groups": len({r["source_group_id"] for r in rows if r["branches"][b]["admitted"] and r["split"] == s})}
                   for s in ("train", "validation")} for b in ("raw", "robotized")}
    result = {"schema": "humanego-ict-joint26-preparation-result-v1", "manifest": manifest_ref,
        "candidate_samples": len(rows), "converted_windows": sum(r["conversion_status"] == admitted_status for r in rows),
        "training_admission_policy": admission_policy,
        "robot_quality_passed": False,
        "branches": counts, "failure_clusters": errors, "control_ground_truth": False,
        "preparation_budgets": deepcopy(preparation),
        "preparation_options": deepcopy(options),
        "physical_deployment_authorized": False, "robot_commands_sent": 0,
        "qualification": "DERIVED_RESEARCH_LABELS_NOT_REAL_DEMONSTRATION",
        "data_gates": {b: "PASSED_RESEARCH_LABEL_GATE" if counts[b]["train"]["source_groups"] >= 16 and counts[b]["validation"]["source_groups"] >= 3 else "BLOCKED_DATA_VOLUME" for b in counts}}
    result_ref = _write_json(output / "PREPARATION_RESULT.json", result)
    return {"result": result_ref, "manifest": manifest_ref, **result}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for option in ("task-packet", "source-manifest", "source-map", "profile", "geometry-index", "client-source", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    # Standard-library-only check, then the sole root-owned task/writer gate.
    packet = _plain_json(args.task_packet)
    from .joint26_runtime import authorize_joint26_task
    paths = {"source_manifest": args.source_manifest, "source_map": args.source_map,
             "profile": args.profile, "geometry_index": args.geometry_index, "client_source": args.client_source}
    authorize_joint26_task(args.task_packet, phase="prepare", input_paths=paths, output_dir=args.output)
    result = prepare_dataset(source_manifest=args.source_manifest, source_map=args.source_map,
        profile_path=args.profile, geometry_index=args.geometry_index, client_source=args.client_source,
        output=args.output, packet=packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
