#!/usr/bin/env python3
"""Bounded target-session IK with viable limits and collapsed-axis-safe solver.

The mounted hand transform is an explicit numerical proxy; placement and IK
initialization use only the target session's prefix and its own warm starts.
HaWoR itself remains offline-smoothed, so outputs are OFFLINE_VISUAL only.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.ops import run_robot_pose_only_independent_placement_pilot_v1 as pilot  # noqa: E402


class Heartbeat:
    def __init__(self, path: Path):
        self.path = path
        self.phase = 'STARTING'
        self.frame = -1
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            atomic_json(self.path, {'pid': os.getpid(), 'phase': self.phase,
                                    'last_finished_frame': self.frame, 'heartbeat_at': now_iso()})
            self.stop_event.wait(30)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.phase = 'TERMINAL'
        self.stop_event.set()
        self.thread.join(timeout=5)
        atomic_json(self.path, {'pid': os.getpid(), 'phase': self.phase,
                                'last_finished_frame': self.frame, 'heartbeat_at': now_iso()})


def _closed(ref: dict) -> Path:
    path = Path(ref['path'])
    actual = artifact_ref(path)
    if actual['bytes'] != ref['bytes'] or actual['sha256'] != ref['sha256']:
        raise RuntimeError(f'input bytes/SHA conflict: {path}')
    return path


def _one_step_viable_limits(lower: np.ndarray, upper: np.ndarray,
                            previous: np.ndarray, previous_previous: np.ndarray | None,
                            step_limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the next acceleration interval intersecting immutable URDF limits.

    For current q with prior p, some next q' can satisfy |q'-2q+p|<=a and
    lower<=q'<=upper iff (lower+p-a)/2 <= q <= (upper+p+a)/2.
    This uses only past q and fixed limits; it neither predicts nor reads a
    future observation and does not loosen the existing velocity/accel gates.
    """
    lo, hi = arm.temporal.bounded_limits(lower, upper, previous, previous_previous,
                                         step_limit)
    acceleration = arm.temporal.ACCELERATION_LIMIT
    lo = np.maximum(lo, .5 * (lower + previous - acceleration))
    hi = np.minimum(hi, .5 * (upper + previous + acceleration))
    if np.any(lo > hi + 1e-10):
        raise RuntimeError('one-step viable IK bounds empty without relaxing hard gates')
    midpoint = .5 * (lo + hi)
    collapsed = lo >= hi - 1e-12
    lo[collapsed] = midpoint[collapsed]
    hi[collapsed] = midpoint[collapsed]
    return lo, hi


def _fill_free(template: np.ndarray, free: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Expand scipy's free-coordinate vector without broadcasting it as 7D."""
    expanded = template.copy()
    expanded[free] = values
    return expanded


def _solve_one_closed(assets, side, target_base, lower, upper, seeds,
                      previous=None, lo=None, hi=None):
    if lo is None:
        lo, hi = lower.copy(), upper.copy()
    free = hi - lo > 1e-12
    if np.all(free):
        return arm.solve_one(assets, side, target_base, lower, upper, seeds,
                             previous=previous, lo=lo, hi=hi)
    attempts = []
    for seed_index, seed in enumerate(seeds):
        x = np.clip(np.asarray(seed, np.float64), lo, hi)

        def residual(candidate):
            pose = arm.old.official._pose_residual(
                arm.old.official._tool_fk(assets, side, candidate), target_base)
            branch = np.minimum(arm.arm.margins(assets, side, candidate) - .003, 0.0) / .001
            continuity = np.empty(0) if previous is None else .01 * (candidate - previous)
            return np.concatenate((pose, branch, continuity))

        if np.any(free):
            sol = least_squares(
                lambda values: residual(_fill_free(x, free, values)), x[free],
                bounds=(lo[free], hi[free]), max_nfev=500,
                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            x = _fill_free(x, free, sol.x)
        pos, rot = arm.arm.error(assets, side, x, target_base)
        margins = arm.arm.margins(assets, side, x)
        failed = not (pos <= 10.0 and rot <= 5.0 and float(np.min(margins)) >= -1e-8)
        score = (failed, max(0.0, -float(np.min(margins))),
                 max(0.0, pos - 10.0) + max(0.0, rot - 5.0), pos + rot)
        attempts.append((*score, x, seed_index, pos, rot, margins))
    return min(attempts, key=lambda value: value[:4])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--frames', default='24', help='24 or all')
    parser.add_argument('--wall-cap-s', type=int, default=5400)
    args = parser.parse_args()
    if args.output_root.exists() or args.frames not in ('24', 'all') or not 0 < args.wall_cap_s <= 7200:
        raise RuntimeError('fresh bounded 24/all attempt required')
    seed = json.loads(args.seed_result.read_text(encoding='utf-8'))
    if seed['status'] != 'PASSED_DEVELOPMENT_NUMERIC' or seed['summary']['frames'] != 4:
        raise RuntimeError('four-frame independent anchor has not passed')
    state_path = _closed(seed['outputs']['states'])
    hawor_path = _closed(seed['inputs']['hawor_npz'])
    preset_path = _closed(seed['inputs']['preset'])
    preset = json.loads(preset_path.read_text(encoding='utf-8'))
    with np.load(state_path, allow_pickle=False) as loaded:
        frozen = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(hawor_path, allow_pickle=False) as loaded:
        hawor = {key: np.asarray(loaded[key]) for key in (
            'joints_3d_world', 'c2w', 'original_frame_indices')}
    human = np.asarray(hawor['joints_3d_world'], np.float64)
    total = len(hawor['c2w'])
    count = total if args.frames == 'all' else min(24, total)
    if not np.array_equal(hawor['original_frame_indices'][:count], np.arange(count)):
        raise RuntimeError('HaWoR frame identity not contiguous')
    valid = np.isfinite(human).all(axis=(2, 3))
    world_base = np.asarray(frozen['T_world_base'], np.float64)
    mounts = np.asarray(frozen['T_tool_hand_root'], np.float64)
    pilot._proper(world_base, 'T_world_base')
    for side in range(2):
        pilot._proper(mounts[side], f'mount {side}')
    base_world = np.linalg.inv(world_base)
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    neutral = np.clip(np.zeros_like(lower), lower, upper)
    anchor = seed['anchor_frame']
    if anchor != 0 or not valid[:, anchor].all():
        raise RuntimeError('this successor requires frame-0 bilateral anchor')
    anchor_rotation = [frozen['T_target_hand_root_world'][anchor, side, :3, :3] for side in range(2)]
    palm_anchor = [arm.shared.wrist_adapter.final_v3_mano_palm_basis(
        human[side, anchor], handedness=arm.shared.SIDES[side]) for side in range(2)]
    q = np.full((count, 2, 7), np.nan)
    target = np.full((count, 2, 4, 4), np.nan)
    actual = np.full_like(target, np.nan)
    rows = []
    args.output_root.mkdir(parents=True)
    heartbeat = Heartbeat(args.output_root / 'HEARTBEAT.json')
    started = time.monotonic()
    heartbeat.start()
    try:
        for side in range(2):
            previous = None
            previous_previous = None
            for frame in range(count):
                heartbeat.phase = f'ARM_SIDE_{side}'
                if time.monotonic() - started > args.wall_cap_s:
                    raise TimeoutError('arm session wall cap')
                if not valid[side, frame]:
                    rows.append({'frame': frame, 'side': arm.SIDES[side],
                                 'status': 'UNKNOWN_MISSING_HAWOR', 'pass': False})
                    previous = previous_previous = None
                    continue
                palm = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
                    human[side, frame], handedness=arm.shared.SIDES[side])
                rotation = palm @ palm_anchor[side].T @ anchor_rotation[side]
                root_target = pilot._root_target(human[side, frame, 0], rotation)
                target[frame, side] = root_target
                tool_target = base_world @ root_target @ np.linalg.inv(mounts[side])
                if frame < 4:
                    if not np.allclose(root_target, frozen['T_target_hand_root_world'][frame, side], atol=1e-8):
                        raise RuntimeError('frozen pilot target changed')
                    fitted = np.asarray(frozen['q_arm'][frame, side], np.float64)
                    if not np.isfinite(fitted).all() or np.any(fitted < lower[side] - 1e-8) or np.any(fitted > upper[side] + 1e-8):
                        raise RuntimeError('frozen pilot q invalid or outside URDF limits')
                    pos, rot = arm.arm.error(assets, side, fitted, tool_target)
                    margins = arm.arm.margins(assets, side, fitted)
                    seed_index = 'FROZEN_TARGET_SESSION_PILOT_PREFIX'
                elif previous is None:
                    lo, hi = lower[side], upper[side]
                    initial = frozen['q_arm'][anchor, side]
                    seeds = [initial, neutral[side], .5 * (lower[side] + upper[side])]
                    *_, fitted, seed_index, pos, rot, margins = _solve_one_closed(
                        assets, side, tool_target, lower[side], upper[side], seeds,
                        previous=previous, lo=lo, hi=hi)
                else:
                    lo, hi = _one_step_viable_limits(
                        lower[side], upper[side], previous, previous_previous,
                        arm.temporal.ARM_SOLVER_STEP_LIMIT)
                    seeds = [np.clip(previous, lo, hi), np.clip(frozen['q_arm'][anchor, side], lo, hi)]
                    *_, fitted, seed_index, pos, rot, margins = _solve_one_closed(
                        assets, side, tool_target, lower[side], upper[side], seeds,
                        previous=previous, lo=lo, hi=hi)
                q[frame, side] = fitted
                actual[frame, side] = world_base @ arm.old.official._tool_fk(assets, side, fitted) @ mounts[side]
                next_viable = (previous is None or bool(np.all(
                    2 * fitted - previous - arm.temporal.ACCELERATION_LIMIT <= upper[side] + 1e-9) and np.all(
                    2 * fitted - previous + arm.temporal.ACCELERATION_LIMIT >= lower[side] - 1e-9)))
                passed = next_viable and pos <= preset['hard_gates']['arm_ik_position_mm_max'] and (
                    rot <= preset['hard_gates']['arm_ik_rotation_deg_max']) and (
                    float(np.min(margins)) >= preset['hard_gates']['joint_limit_margin_min'])
                rows.append({'frame': frame, 'side': arm.SIDES[side], 'status': 'SOLVED' if passed else 'HOLD_NUMERIC_GATE',
                             'position_mm': float(pos), 'rotation_deg': float(rot),
                             'branch_margins_m': margins.tolist(), 'seed_index': seed_index,
                             'next_step_viable': next_viable, 'pass': bool(passed)})
                previous_previous, previous = previous, fitted
                heartbeat.frame = frame
        if not np.allclose(q[:4], frozen['q_arm'], atol=1e-8, equal_nan=True):
            raise RuntimeError('first four frames do not reproduce frozen pilot q')
        if not np.allclose(target[:4], frozen['T_target_hand_root_world'], atol=1e-8, equal_nan=True):
            raise RuntimeError('first four frames do not reproduce frozen pilot target')
        if not np.isfinite(q[valid[:, :count].T]).all() or not np.isnan(q[~valid[:, :count].T]).all():
            raise RuntimeError('valid/unknown q structural mismatch')
        solved = [row for row in rows if row['status'] != 'UNKNOWN_MISSING_HAWOR']
        failed = [row for row in solved if not row['pass']]
        velocity, acceleration = [], []
        for side in range(2):
            for frame in range(1, count):
                if valid[side, frame] and valid[side, frame - 1]:
                    velocity.extend(np.abs(q[frame, side] - q[frame - 1, side]).tolist())
            for frame in range(2, count):
                if all(valid[side, frame - lag] for lag in (0, 1, 2)):
                    acceleration.extend(np.abs(q[frame, side] - 2 * q[frame - 1, side] + q[frame - 2, side]).tolist())
        velocity_max = float(max(velocity, default=0.0))
        acceleration_max = float(max(acceleration, default=0.0))
        gates = {
            'all_observed_rows_pose_branch': len(failed) == 0,
            'one_step_viability': all(row.get('next_step_viable', True) for row in solved),
            'arm_velocity': velocity_max <= 0.12 + 1e-9,
            'arm_acceleration': acceleration_max <= 0.06 + 1e-9,
            'base_fixed': True, 'motion_gain_exact_one': True,
            'missing_frames_unknown_not_filled': bool(np.isnan(q[~valid[:, :count].T]).all()),
            'per_side_anchor_is_observed': True,
        }
        states_path = args.output_root / 'ARM_CANARY_STATES.npz'
        np.savez_compressed(states_path, q_arm=q, T_world_base=world_base,
                            T_tool_hand_root=mounts, T_target_hand_root_world=target,
                            T_actual_hand_root_world=actual, valid_side_frame=valid[:, :count],
                            source_frames=hawor['original_frame_indices'][:count],
                            anchor_frame_per_side=np.asarray([anchor, anchor]))
        metrics = {
            'observed_rows': len(solved), 'unknown_rows': int((~valid[:, :count]).sum()),
            'failed_rows': len(failed),
            'position_mm_max': max((row['position_mm'] for row in solved), default=None),
            'rotation_deg_max': max((row['rotation_deg'] for row in solved), default=None),
            'velocity_rad_per_frame_max_contiguous': velocity_max,
            'acceleration_rad_per_frame2_max_contiguous': acceleration_max,
        }
        result = {
            'schema_version': 'rc1-independent-pose-only-arm-v6', 'created_at': now_iso(),
            'session': seed['session_id'], 'frame_count': count,
            'status': 'PASS_NUMERIC_CANARY_NO_AUTHORITY' if all(gates.values()) else 'HOLD_NUMERIC_CANARY',
            'input_mode': 'OFFLINE_VISUAL', 'training_eligible': False,
            'authority': False, 'control_ground_truth': False,
            'inputs': {'seed_result': artifact_ref(args.seed_result), 'hawor_npz': artifact_ref(hawor_path),
                       'preset': artifact_ref(preset_path), 'code': artifact_ref(Path(__file__))},
            'metrics': metrics, 'gates': gates, 'rows': rows,
            'output_states': artifact_ref(states_path),
            'claim_limit': 'Target-session fixed base; first four q states from same-session pilot, then same-session IK with past-only one-step URDF/acceleration viability and collapsed-axis-safe scipy mapping. Hard gates unchanged. Static mount unmeasured; HaWoR offline smoothed. No causal training or physical authority.',
        }
        atomic_json(args.output_root / 'RESULT.json', result)
        print(json.dumps({'status': result['status'], 'metrics': metrics,
                          'result': str(args.output_root / 'RESULT.json')}))
        return 0
    except Exception as exc:
        atomic_json(args.output_root / 'FAILED_RUNTIME_FINAL.json', {
            'session_id': seed['session_id'], 'status': 'FAILED_RUNTIME_FINAL',
            'error': f'{type(exc).__name__}: {exc}', 'created_at': now_iso(),
            'seed_result': artifact_ref(args.seed_result), 'code': artifact_ref(Path(__file__))})
        raise
    finally:
        heartbeat.stop()


if __name__ == '__main__':
    raise SystemExit(main())
