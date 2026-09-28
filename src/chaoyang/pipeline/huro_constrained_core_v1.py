"""Opt-in HuRo constraint adapter; not a registered trajectory runner.

Uses installed jaxls inequality constraints, not post-solve clipping. The
publisher must bind the returned source before a real trajectory invocation.
Numerical convergence is not a substitute for checking the original output q.
"""
from __future__ import annotations
import ast
import numpy as np


def joint_limit_inequalities(full_q, lower, upper):
    """Signed g(q)<=0, including mimic/full joint limits when used by HuRo."""
    import jax.numpy as jnp
    return jnp.concatenate((full_q - upper, lower - full_q), axis=-1)


def prepare_initial_guess(home, seed, frames, joint_mask):
    """Keep immutable home/rest separate from the optimizer seed; no clipping."""
    home = np.asarray(home, dtype=float)
    mask = np.asarray(joint_mask, dtype=float)
    if home.ndim != 1 or mask.shape != home.shape or not np.isfinite(home).all():
        raise ValueError('HOME_OR_MASK_SHAPE')
    if not np.isin(mask, [0., 1.]).all() or frames < 1:
        raise ValueError('INVALID_MASK_OR_FRAMES')
    initial = np.broadcast_to(home if seed is None else seed, (frames, len(home))).copy()
    if not np.isfinite(initial).all():
        raise ValueError('NONFINITE_INITIAL_GUESS')
    # Locked degrees never inherit the previous block's optimizer state.
    initial[:, mask == 0] = home[mask == 0]
    return initial


def check_raw_joint_limits(q, lower, upper, *, tolerance=0.):
    """Diagnostic only. Never returns modified/clipped joint values."""
    q, lower, upper = (np.asarray(v, dtype=float) for v in (q, lower, upper))
    if lower.shape != upper.shape or q.shape[-1:] != lower.shape:
        raise ValueError('LIMIT_SHAPE')
    if tolerance < 0 or not np.isfinite([tolerance]).all():
        raise ValueError('INVALID_TOLERANCE')
    if not np.isfinite(q).all() or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError('INVALID_LIMIT_INPUT')
    excess = np.maximum(np.maximum(q - upper, lower - q), 0.)
    return {'pass': bool(np.all(excess <= tolerance)), 'maximum_violation': float(excess.max(initial=0.)),
            'posthoc_clip_used': False, 'tolerance': float(tolerance)}


def constrained_core_source(wrist_adapted_source):
    """Fail closed on source drift; transform only the supplied pinned function.

    Keep every FK, rest/home, wrist, and temporal term unchanged. Existing
    official source and existing runner are not edited or auto-invoked.
    """
    source = wrist_adapted_source
    def replace_once(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('CONSTRAINED_CORE_SOURCE_DRIFT: ' + before[:60])
        source = source.replace(before, after)
    replace_once('def solve_retargeting_with_wrist(', 'def solve_retargeting_with_constraints(')
    replace_once('    wrist_link_indices: jnp.ndarray,\n',
                 '    wrist_link_indices: jnp.ndarray,\n    initial_guess: jnp.ndarray | None = None,\n')
    replace_once('    init_guess = jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))',
                 '    init_guess = (jnp.broadcast_to(initial_cfg, (timesteps, initial_cfg.shape[0]))\n'
                 '                  if initial_guess is None else initial_guess)\n'
                 '    if init_guess.shape != (timesteps, initial_cfg.shape[0]):\n'
                 '        raise ValueError("INITIAL_GUESS_SHAPE")\n'
                 '    init_guess = jnp.where(joint_mask, init_guess, initial_cfg)')
    replace_once('    @jaxls.Cost.factory\n    def joint_limit_cost(',
                 '    @jaxls.Cost.factory(kind="constraint_leq_zero")\n    def joint_limit_cost(')
    replace_once('        upper_viol = jnp.maximum(0.0, cfg_full - robot.joints.upper_limits_all)\n'
                 '        lower_viol = jnp.maximum(0.0, robot.joints.lower_limits_all - cfg_full)\n'
                 '        return jnp.concatenate([upper_viol, lower_viol]).flatten() * weight_limit',
                 '        return _completion_joint_limits(cfg_full, robot.joints.lower_limits_all,\n'
                 '                                        robot.joints.upper_limits_all).flatten()')
    # AL constraint tolerances are convergence controls, not relaxed acceptance gates.
    replace_once('            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),',
                 '            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),\n'
                 '            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(\n'
                 '                tolerance_absolute=1e-7, tolerance_relative=1e-7),')
    ast.parse(source)
    return source
