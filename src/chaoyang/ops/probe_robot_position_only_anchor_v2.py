#!/usr/bin/env python3
"""Ablate independent first-prefix position-only anchor orientation.

This never changes the historical v1 attempt, nor does it weaken its 10 mm / 5
degree gates. It checks whether the v1 orientation choice caused its IK HOLD.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402


def solve_position_only(assets, side: int, target_xyz: np.ndarray, mount: np.ndarray,
                        lower: np.ndarray, upper: np.ndarray, seeds: list[np.ndarray]) -> dict:
    options = []
    for seed_index, seed in enumerate(seeds):
        def residual(q):
            root = arm.old.official._tool_fk(assets, side, q) @ mount
            position = (root[:3, 3] - target_xyz) / 0.01
            branch = np.minimum(arm.arm.margins(assets, side, q) - 0.003, 0.0) / 0.001
            regularizer = 0.01 * (q - seed)
            return np.r_[position, branch, regularizer]
        fitted = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                               max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = fitted.x
        root = arm.old.official._tool_fk(assets, side, q) @ mount
        error_mm = float(np.linalg.norm(root[:3, 3] - target_xyz) * 1000)
        margin = float(np.min(arm.arm.margins(assets, side, q)))
        score = (error_mm > 10 or margin < -1e-8, error_mm + max(0, -margin * 1000), seed_index)
        options.append({'q': q, 'error_mm': error_mm, 'min_margin': margin,
                        'seed_index': seed_index, 'score': score, 'root': root})
    return min(options, key=lambda item: item['score'])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError('fresh ablation output required')
    prior = json.loads(args.v1_result.read_text(encoding='utf-8'))
    if prior['scope'] != 'OFFLINE_VISUAL_4FRAME_PILOT':
        raise RuntimeError('prior is not frozen four-frame placement pilot')
    hawor_path = Path(prior['inputs']['hawor_npz']['path'])
    preset_path = Path(prior['inputs']['preset']['path'])
    for ref in (prior['inputs']['hawor_npz'], prior['inputs']['preset']):
        actual = artifact_ref(Path(ref['path']))
        if actual['bytes'] != ref['bytes'] or actual['sha256'] != ref['sha256']:
            raise RuntimeError('v1 input SHA changed')
    preset = json.loads(preset_path.read_text(encoding='utf-8'))
    with np.load(hawor_path, allow_pickle=False) as loaded:
        human = np.asarray(loaded['joints_3d_world'], np.float64)
        c2w = np.asarray(loaded['c2w'], np.float64)
    valid = np.isfinite(human).all(axis=(2, 3))
    anchor = prior['first_prefix_anchor_frame']
    if not valid[:, anchor].all():
        raise RuntimeError('frozen anchor is no longer observed')
    midpoint = .5 * (human[0, anchor, 0] + human[1, anchor, 0])
    camera_base = np.asarray(preset['camera_to_robot_base_rotation'], np.float64)
    mounts = np.asarray(preset['T_tool_hand_root'], np.float64)
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    neutral, middle = np.clip(np.zeros_like(lower), lower, upper), .5 * (lower + upper)
    rows = []
    for index, base_target in enumerate(preset['base_midpoint_target_candidates_m']):
        base = np.eye(4)
        base[:3, :3] = c2w[anchor, :3, :3] @ camera_base
        base[:3, 3] = midpoint - base[:3, :3] @ np.asarray(base_target)
        base_world = np.linalg.inv(base)
        side_rows = []
        for side in range(2):
            target_xyz = (base_world @ np.r_[human[side, anchor, 0], 1])[:3]
            fit = solve_position_only(assets, side, target_xyz, mounts[side],
                                      lower[side], upper[side], [neutral[side], middle[side]])
            root_target = np.eye(4)
            root_target[:3, :3] = fit['root'][:3, :3]
            root_target[:3, 3] = target_xyz
            tool_target = root_target @ np.linalg.inv(mounts[side])
            *_, q_full, _, pos, rot, margin = arm.solve_one(
                assets, side, tool_target, lower[side], upper[side],
                [fit['q'], neutral[side]])
            side_rows.append({
                'side': side, 'position_only_mm': fit['error_mm'],
                'position_only_joint_margin': fit['min_margin'],
                'full_pose_position_mm': float(pos), 'full_pose_rotation_deg': float(rot),
                'full_pose_joint_margin': float(np.min(margin)),
                'full_pose_pass': bool(pos <= 10 and rot <= 5 and float(np.min(margin)) >= -1e-8),
                'q_position_only': fit['q'].tolist(), 'q_full_pose': q_full.tolist(),
            })
        rows.append({'candidate_index': index, 'base_target_m': base_target,
                     'side_rows': side_rows,
                     'both_hard_pass': all(r['full_pose_pass'] for r in side_rows),
                     'sum_position_only_mm': sum(r['position_only_mm'] for r in side_rows)})
    args.output_root.mkdir(parents=True)
    output = {
        'schema_version': 'rc1-independent-placement-position-only-anchor-ablation-v2',
        'created_at': now_iso(), 'session_id': prior['session_id'],
        'status': 'PASSED_DEVELOPMENT_NUMERIC' if any(r['both_hard_pass'] for r in rows) else 'FAILED_QUALITY_C',
        'input_mode': 'OFFLINE_VISUAL', 'training_eligible': False,
        'input_closure': {'v1_result': artifact_ref(args.v1_result), 'code': artifact_ref(Path(__file__))},
        'rows': rows,
        'claim_limit': 'Position-only anchor orientation ablation on the same four frozen base candidates; no full-session IK, digital collision, rendered Robot, causal HaWoR, or physical authority.',
    }
    atomic_json(args.output_root / 'RESULT.json', output)
    print(json.dumps({'status': output['status'], 'both_pass_candidates': [r['candidate_index'] for r in rows if r['both_hard_pass']],
                      'min_position_only_mm': min(r['sum_position_only_mm'] for r in rows)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
