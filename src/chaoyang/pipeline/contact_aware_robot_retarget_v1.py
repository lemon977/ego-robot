"""Fail-bounded contract for contact-aware Robot retargeting.

The solver owns neither human-contact inference nor final occlusion.  It
consumes a frozen contact-hypothesis sidecar and executes the six V1 stages in
a fixed order.  Concrete optimizers implement :class:`StageSolver`; this module
enforces budgets, hard geometry gates, handedness, and best-diagnostic output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np


SCHEMA_VERSION = "CONTACT_AWARE_ROBOT_RETARGET_V1"


class RetargetContractError(ValueError):
    """Raised for malformed inputs or a solver that violates its contract."""


class Stage(str, Enum):
    WRIST_ARM_IK = "WRIST_ARM_IK"
    HUMAN_POSE_HAND_RETARGET = "HUMAN_POSE_HAND_RETARGET"
    CONTACT_FINGER_REFINEMENT = "CONTACT_FINGER_REFINEMENT"
    COLLISION_CLEANUP = "COLLISION_CLEANUP"
    TEMPORAL_REFINEMENT = "TEMPORAL_REFINEMENT"
    FINAL_AUDIT = "FINAL_AUDIT"


STAGE_ORDER = tuple(Stage)


@dataclass(frozen=True)
class SolverBudget:
    max_initializations: int = 2
    max_iterations: int = 200
    frame_timeout_s: float = 30.0
    session_canary_timeout_s: float = 1800.0


@dataclass(frozen=True)
class GeometryThresholds:
    contact_target_min_m: float
    contact_target_max_m: float
    contact_penetration_p95_max_m: float
    gross_penetration_max_m: float
    noncontact_penetration_threshold_m: float = 0.001
    noncontact_penetration_count_max: int = 0


TASK_THRESHOLDS: dict[str, GeometryThresholds] = {
    "poker": GeometryThresholds(0.0, 0.002, 0.0015, 0.003),
    "chips": GeometryThresholds(0.0, 0.003, 0.003, 0.006),
}


@dataclass(frozen=True)
class RetargetCandidate:
    joint_positions: np.ndarray
    wrist_to_world: np.ndarray
    chirality: tuple[str, ...]
    objective: float
    iterations: int
    contact_distance_m: np.ndarray
    contact_penetration_m: np.ndarray
    gross_penetration_m: float
    noncontact_penetration_m: np.ndarray
    arm_reachable: bool
    structure_closed: bool
    coordinate_chain_complete: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageContext:
    task_id: str
    expected_chirality: tuple[str, ...]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    contact_hypothesis: Any
    previous: RetargetCandidate | None = None


@dataclass(frozen=True)
class StageReceipt:
    stage: Stage
    initialization_count: int
    best_objective: float | None
    elapsed_s: float
    status: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RetargetRunResult:
    status: str
    authority_promotable: bool
    best_diagnostic: RetargetCandidate | None
    receipts: tuple[StageReceipt, ...]
    failed_stage: Stage | None
    failure_reasons: tuple[str, ...]


class StageSolver(Protocol):
    def __call__(
        self,
        context: StageContext,
        *,
        initialization: int,
        max_iterations: int,
        frame_timeout_s: float,
    ) -> RetargetCandidate: ...


def _validate_budget(budget: SolverBudget) -> None:
    if budget.max_initializations < 1 or budget.max_initializations > 2:
        raise RetargetContractError("max_initializations must be in [1,2]")
    if budget.max_iterations < 1 or budget.max_iterations > 200:
        raise RetargetContractError("max_iterations must be in [1,200]")
    if not 0.0 < budget.frame_timeout_s <= 30.0:
        raise RetargetContractError("frame_timeout_s must be in (0,30]")
    if not 0.0 < budget.session_canary_timeout_s <= 1800.0:
        raise RetargetContractError("session_canary_timeout_s must be in (0,1800]")


def validate_candidate(candidate: RetargetCandidate, context: StageContext) -> tuple[str, ...]:
    """Return all hard-constraint violations for one candidate."""

    failures: list[str] = []
    q = np.asarray(candidate.joint_positions, dtype=np.float64)
    lower = np.asarray(context.joint_lower, dtype=np.float64)
    upper = np.asarray(context.joint_upper, dtype=np.float64)
    wrists = np.asarray(candidate.wrist_to_world, dtype=np.float64)
    if q.shape != lower.shape or q.shape != upper.shape or q.ndim != 2:
        failures.append("JOINT_SHAPE_MISMATCH")
    elif not np.isfinite(q).all():
        failures.append("NONFINITE_JOINTS")
    elif np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
        failures.append("JOINT_LIMIT")
    if wrists.shape != (len(context.expected_chirality), 4, 4):
        failures.append("WRIST_SHAPE_MISMATCH")
    elif not np.isfinite(wrists).all():
        failures.append("NONFINITE_WRIST")
    else:
        for matrix in wrists:
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
                failures.append("WRIST_HOMOGENEOUS_ROW")
                break
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                failures.append("IMPROPER_WRIST_ROTATION")
                break
    if tuple(candidate.chirality) != tuple(context.expected_chirality):
        failures.append("CHIRALITY_SWAP")
    if not candidate.arm_reachable:
        failures.append("ARM_UNREACHABLE")
    if not candidate.structure_closed:
        failures.append("ROBOT_STRUCTURE_OPEN")
    if not candidate.coordinate_chain_complete:
        failures.append("COORDINATE_CHAIN_INCOMPLETE")
    threshold = TASK_THRESHOLDS.get(context.task_id)
    if threshold is not None:
        if not np.isfinite(candidate.gross_penetration_m) or (
            candidate.gross_penetration_m > threshold.gross_penetration_max_m
        ):
            failures.append("GROSS_PENETRATION")
        noncontact = np.asarray(candidate.noncontact_penetration_m, dtype=np.float64)
        if not np.isfinite(noncontact).all() or int(
            np.count_nonzero(noncontact > threshold.noncontact_penetration_threshold_m)
        ) > threshold.noncontact_penetration_count_max:
            failures.append("NONCONTACT_PENETRATION")
    if not np.isfinite(float(candidate.objective)):
        failures.append("NONFINITE_OBJECTIVE")
    if candidate.iterations < 0 or candidate.iterations > 200:
        failures.append("ITERATION_BUDGET_VIOLATION")
    return tuple(dict.fromkeys(failures))


def audit_geometry(candidate: RetargetCandidate, task_id: str) -> tuple[str, ...]:
    """Apply task-specific digital-geometry gates (never physical accuracy)."""

    if task_id not in TASK_THRESHOLDS:
        raise RetargetContractError(f"unsupported task_id {task_id!r}")
    threshold = TASK_THRESHOLDS[task_id]
    failures: list[str] = []
    distance = np.asarray(candidate.contact_distance_m, dtype=np.float64)
    penetration = np.asarray(candidate.contact_penetration_m, dtype=np.float64)
    noncontact = np.asarray(candidate.noncontact_penetration_m, dtype=np.float64)
    if distance.size == 0 or not np.isfinite(distance).all():
        failures.append("CONTACT_DISTANCE_UNAVAILABLE")
    elif np.any(distance < threshold.contact_target_min_m) or np.any(
        distance > threshold.contact_target_max_m
    ):
        failures.append("CONTACT_TARGET_DISTANCE")
    if penetration.size == 0 or not np.isfinite(penetration).all():
        failures.append("CONTACT_PENETRATION_UNAVAILABLE")
    elif float(np.percentile(penetration, 95)) > threshold.contact_penetration_p95_max_m:
        failures.append("CONTACT_PENETRATION_P95")
    if not np.isfinite(candidate.gross_penetration_m) or (
        candidate.gross_penetration_m > threshold.gross_penetration_max_m
    ):
        failures.append("GROSS_PENETRATION")
    if not np.isfinite(noncontact).all():
        failures.append("NONCONTACT_PENETRATION_UNAVAILABLE")
    elif int(np.count_nonzero(noncontact > threshold.noncontact_penetration_threshold_m)) > (
        threshold.noncontact_penetration_count_max
    ):
        failures.append("NONCONTACT_PENETRATION")
    return tuple(failures)


def run_staged_retarget(
    context: StageContext,
    solvers: Mapping[Stage, StageSolver],
    *,
    budget: SolverBudget = SolverBudget(),
    clock: Callable[[], float] = time.monotonic,
) -> RetargetRunResult:
    """Execute exactly six stages with finite retries and fail-closed receipts."""

    _validate_budget(budget)
    if set(solvers) != set(STAGE_ORDER):
        raise RetargetContractError("solvers must provide exactly the six V1 stages")
    if context.task_id not in TASK_THRESHOLDS:
        raise RetargetContractError("task_id must be poker or chips")
    if not isinstance(context.contact_hypothesis, Mapping) or (
        context.contact_hypothesis.get("schema_version") != "HUMAN_CONTACT_HYPOTHESIS_V1"
    ):
        raise RetargetContractError(
            "contact_hypothesis must be an independent HUMAN_CONTACT_HYPOTHESIS_V1 sidecar"
        )
    if not context.expected_chirality or any(side not in ("left", "right") for side in context.expected_chirality):
        raise RetargetContractError("expected_chirality must contain left/right identities")
    lower = np.asarray(context.joint_lower, dtype=np.float64)
    upper = np.asarray(context.joint_upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 2 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RetargetContractError("joint limits must be finite matching 2D arrays")
    if np.any(lower > upper):
        raise RetargetContractError("joint lower limit exceeds upper limit")

    started = clock()
    current = context.previous
    best_diagnostic = current
    receipts: list[StageReceipt] = []
    for stage in STAGE_ORDER:
        if clock() - started >= budget.session_canary_timeout_s:
            return RetargetRunResult(
                "FAILED_RUNTIME_FINAL", False, best_diagnostic, tuple(receipts), stage,
                ("SESSION_CANARY_TIMEOUT",),
            )
        stage_started = clock()
        candidates: list[RetargetCandidate] = []
        failures: list[str] = []
        for initialization in range(budget.max_initializations):
            if clock() - started >= budget.session_canary_timeout_s:
                failures.append("SESSION_CANARY_TIMEOUT")
                break
            attempt_started = clock()
            try:
                candidate = solvers[stage](
                    replace(context, previous=current),
                    initialization=initialization,
                    max_iterations=budget.max_iterations,
                    frame_timeout_s=budget.frame_timeout_s,
                )
            except Exception as exc:  # optimizer errors become bounded diagnostic failures
                failures.append(f"SOLVER_EXCEPTION:{type(exc).__name__}")
                continue
            elapsed = clock() - attempt_started
            if elapsed > budget.frame_timeout_s:
                failures.append("FRAME_TIMEOUT")
                continue
            violations = validate_candidate(candidate, context)
            if violations:
                failures.extend(violations)
                if best_diagnostic is None or candidate.objective < best_diagnostic.objective:
                    best_diagnostic = candidate
                continue
            candidates.append(candidate)
        if not candidates:
            receipts.append(StageReceipt(stage, budget.max_initializations, None, clock() - stage_started, "FAILED", tuple(dict.fromkeys(failures))))
            return RetargetRunResult(
                "FAILED_QUALITY_C", False, best_diagnostic, tuple(receipts), stage,
                tuple(dict.fromkeys(failures)) or ("NO_FEASIBLE_CANDIDATE",),
            )
        current = min(candidates, key=lambda item: item.objective)
        if best_diagnostic is None or current.objective < best_diagnostic.objective:
            best_diagnostic = current
        receipts.append(StageReceipt(stage, budget.max_initializations, current.objective, clock() - stage_started, "PASSED", tuple(dict.fromkeys(failures))))

    assert current is not None
    geometry_failures = audit_geometry(current, context.task_id)
    if geometry_failures:
        return RetargetRunResult(
            "FAILED_QUALITY_C", False, best_diagnostic, tuple(receipts), Stage.FINAL_AUDIT,
            geometry_failures,
        )
    return RetargetRunResult("PASSED_DIAGNOSTIC", False, current, tuple(receipts), None, ())
