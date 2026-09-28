"""Candidate 2: projected solver retraction, never final-output clipping.

All VarValues contain physical q throughout FK/rest/smoothness and solve.
Non-smooth bound transitions may impair convergence; quality gates stay separate.
"""
from __future__ import annotations
import ast
import numpy as np


def intersect_affine_full_limits(lower, upper, full_lower, full_upper, matrix, offset):
    """Propagate independent/fixed/mimic full-joint limits into actuator box.

    Reject coupled (multi-actuator) full joints, which cannot be represented by
    this projected box. Caller supplies the pinned robot's verified affine map.
    """
    lo, hi = np.array(lower, float), np.array(upper, float)
    a, b = np.asarray(matrix, float), np.asarray(offset, float)
    flo, fhi = np.asarray(full_lower, float), np.asarray(full_upper, float)
    if lo.ndim != 1 or hi.shape != lo.shape or a.shape != (len(b), len(lo)) or flo.shape != b.shape or fhi.shape != b.shape:
        raise ValueError('FULL_LIMIT_SHAPE')
    if not all(np.isfinite(v).all() for v in [lo, hi, a, b, flo, fhi]) or np.any(lo > hi) or np.any(flo > fhi):
        raise ValueError('FINITE_ORDERED_LIMITS_REQUIRED')
    for row, shift, low, high in zip(a, b, flo, fhi):
        indices = np.flatnonzero(row != 0)
        if len(indices) > 1:
            raise ValueError('COUPLED_FULL_JOINT_UNSUPPORTED')
        if len(indices) == 0:
            if not low <= shift <= high:
                raise ValueError('FIXED_JOINT_OUT_OF_BOUNDS')
            continue
        j = indices[0]
        x, y = (low - shift) / row[j], (high - shift) / row[j]
        lo[j], hi[j] = max(lo[j], min(x, y)), min(hi[j], max(x, y))
    if np.any(lo > hi):
        raise ValueError('INFEASIBLE_MIMIC_BOUNDS')
    return lo, hi


def make_projected_joint_type(lower, upper, home, joint_mask):
    """Physical-q variables; every trial update is projected before evaluation."""
    import jax.numpy as jnp
    import jaxls
    lo, hi, home, mask = (np.asarray(x, float) for x in (lower, upper, home, joint_mask))
    if home.ndim != 1 or any(x.shape != home.shape for x in [lo, hi, mask]):
        raise ValueError('JOINT_SHAPE')
    if not all(np.isfinite(x).all() for x in [lo, hi, home, mask]) or np.any(lo > hi):
        raise ValueError('FINITE_ORDERED_LIMITS_REQUIRED')
    if not np.isin(mask, [0, 1]).all() or np.any(home < lo) or np.any(home > hi):
        raise ValueError('INVALID_HOME_OR_MASK')
    jh, jm = jnp.asarray(home), jnp.asarray(mask)
    dtype = np.asarray(jh).dtype
    # Round computational bounds inward, never outside the declared float64 box.
    lcast, ucast = lo.astype(dtype), hi.astype(dtype)
    lcast = np.where(lcast.astype(float) < lo, np.nextafter(lcast, np.array(np.inf, dtype=dtype)), lcast)
    ucast = np.where(ucast.astype(float) > hi, np.nextafter(ucast, np.array(-np.inf, dtype=dtype)), ucast)
    if np.any(lcast > ucast) or np.any(np.asarray(jh) < lo) or np.any(np.asarray(jh) > hi):
        raise ValueError('HOME_OR_BOX_NOT_REPRESENTABLE_IN_SOLVER_DTYPE')
    jl, ju = jnp.asarray(lcast), jnp.asarray(ucast)
    def retract(q, delta):
        projected = jnp.minimum(ju, jnp.maximum(jl, q + delta))
        return jnp.where(jm, projected, jh)
    class ProjectedPhysicalJoint(jaxls.Var, default_factory=lambda: jh,
                                 retract_fn=retract, tangent_dim=len(home)):
        pass
    return ProjectedPhysicalJoint


def validate_physical_seed(seed, home, mask, lower, upper, frames):
    from chaoyang.pipeline.huro_constrained_core_v1 import prepare_initial_guess, check_raw_joint_limits
    import jax.numpy as jnp
    initial = prepare_initial_guess(home, seed, frames, mask)
    initial = np.asarray(jnp.asarray(initial))
    if not check_raw_joint_limits(initial, lower, upper)['pass']:
        raise ValueError('INITIAL_GUESS_OUT_OF_BOUNDS')
    return initial


def projected_core_source(wrist_adapted_source):
    """Opt-in pinned source transformer; preserve every physical-q consumer."""
    source = wrist_adapted_source
    def once(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('PROJECTED_SOURCE_DRIFT: ' + before[:60])
        source = source.replace(before, after)
    once('def solve_retargeting_with_wrist(', 'def solve_retargeting_with_projected_joints(')
    once('    wrist_link_indices: jnp.ndarray,\n', '    wrist_link_indices: jnp.ndarray,\n    initial_guess: jnp.ndarray | None = None,\n')
    once('    init_guess = jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))',
         '    init_guess = (jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))\n'
         '                  if initial_guess is None else initial_guess)')
    # The entire variable family, including both smoothness endpoints, must change.
    if source.count('robot.joint_var_cls(') != 3:
        raise ValueError('VARIABLE_CONSUMER_COUNT_DRIFT')
    source = source.replace('robot.joint_var_cls(', '_completion_projected_joint_cls(')
    ast.parse(source)
    return source
