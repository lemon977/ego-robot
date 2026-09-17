#!/usr/bin/env python3
"""Four-frame independent placement with first-prefix position-only anchor.

V1 fixed hand-root orientation from neutral FK and failed. V2 obtains a
reachable anchor orientation from position-only IK on the same target-session
first frame, then uses bounded same-session temporal IK. No foreign q/base.
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
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.ops import run_robot_pose_only_independent_placement_pilot_v1 as base_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1-result', type=Path, required=True)
    parser.add_argument('--ablation-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--wall-cap-s', type=int, default=300)
    args = parser.parse_args()
    if args.output_root.exists() or not 0 < args.wall_cap_s <= 1800:
        raise RuntimeError('fresh bounded attempt required')
    started = time.monotonic()
    v1 = json.loads(args.v1_result.read_text(encoding='utf-8'))
    ablation = json.loads(args.ablation_result.read_text(encoding='utf-8'))
    if v1['session_id'] != ablation['session_id'] or ablation['status'] != 'PASSED_DEVELOPMENT_NUMERIC':
        raise RuntimeError('frozen source/ablation mismatch')
    prior_ref = ablation['input_closure']['v1_result']
    if artifact_ref(args.v1_result)['sha256'] != prior_ref['sha256']:
        raise RuntimeError('ablation did not bind current v1 attempt')
    hawor_ref = v1['inputs']['hawor_npz']
    raw_ref = v1['inputs']['raw_video']
    preset_ref = v1['inputs']['preset']
    hawor_path, raw_path, preset_path = (Path(ref['path']) for ref in (hawor_ref, raw_ref, preset_ref))
    for ref in (hawor_ref, raw_ref, preset_ref):
        current = artifact_ref(Path(ref['path']))
        if current['bytes'] != ref['bytes'] or current['sha256'] != ref['sha256']:
            raise RuntimeError('v1 input closure conflict')
    preset = json.loads(preset_path.read_text(encoding='utf-8'))
    with np.load(hawor_path, allow_pickle=False) as source:
        hawor = {key: np.asarray(source[key]) for key in (
            'joints_3d_world', 'c2w', 'intrinsics', 'original_frame_indices')}
    human = np.asarray(hawor['joints_3d_world'], np.float64)
    count = int(v1['summary']['frames'])
    if count != 4 or not np.array_equal(hawor['original_frame_indices'][:count], np.arange(count)):
        raise RuntimeError('four-frame identity changed')
    valid = np.isfinite(human).all(axis=(2, 3))
    anchor = int(v1['first_prefix_anchor_frame'])
    if not valid[:, anchor].all():
        raise RuntimeError('anchor changed')
    # Candidate index is frozen before inspecting future-frame quality. All
    # four passed anchor ablation, so use the first preset item deterministically.
    chosen = next(r for r in ablation['rows'] if r['candidate_index'] == 0)
    if not chosen['both_hard_pass']:
        raise RuntimeError('candidate 0 did not pass anchor gate')
    midpoint = .5 * (human[0, anchor, 0] + human[1, anchor, 0])
    world_base = base_pilot._world_base(
        hawor['c2w'][anchor], midpoint,
        np.asarray(preset['camera_to_robot_base_rotation'], np.float64),
        np.asarray(chosen['base_target_m'], np.float64))
    base_world = np.linalg.inv(world_base)
    mounts = np.asarray(preset['T_tool_hand_root'], np.float64)
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    neutral = np.clip(np.zeros_like(lower), lower, upper)
    middle = .5 * (lower + upper)
    target = np.full((count, 2, 4, 4), np.nan)
    actual = np.full_like(target, np.nan)
    q_arm = np.full((count, 2, 7), np.nan)
    rows = []
    for side in range(2):
        q_anchor = np.asarray(chosen['side_rows'][side]['q_position_only'], np.float64)
        if q_anchor.shape != (7,) or not np.isfinite(q_anchor).all():
            raise RuntimeError('anchor q invalid')
        root_at_anchor = world_base @ arm.old.official._tool_fk(assets, side, q_anchor) @ mounts[side]
        anchor_rotation = root_at_anchor[:3, :3]
        palm_anchor = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
            human[side, anchor], handedness=arm.shared.SIDES[side])
        previous = None
        previous_previous = None
        for frame_id in range(count):
            if time.monotonic() - started > args.wall_cap_s:
                raise TimeoutError('v2 four-frame wall cap exceeded')
            if frame_id < anchor or not valid[side, frame_id]:
                rows.append({'frame': frame_id, 'side_index': side, 'status': 'UNKNOWN_MISSING_HAWOR'})
                previous = previous_previous = None
                continue
            palm = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
                human[side, frame_id], handedness=arm.shared.SIDES[side])
            rotation = palm @ palm_anchor.T @ anchor_rotation
            root_target = base_pilot._root_target(human[side, frame_id, 0], rotation)
            target[frame_id, side] = root_target
            tool_target = base_world @ root_target @ np.linalg.inv(mounts[side])
            if previous is None:
                lo, hi = lower[side], upper[side]
                seeds = [q_anchor, neutral[side], middle[side]]
            else:
                lo, hi = arm.temporal.bounded_limits(
                    lower[side], upper[side], previous, previous_previous,
                    arm.temporal.ARM_SOLVER_STEP_LIMIT)
                seeds = [np.clip(previous, lo, hi), np.clip(q_anchor, lo, hi)]
            *_, q, seed_index, pos, rot, margins = arm.solve_one(
                assets, side, tool_target, lower[side], upper[side], seeds,
                previous=previous, lo=lo, hi=hi)
            q_arm[frame_id, side] = q
            actual[frame_id, side] = world_base @ arm.old.official._tool_fk(assets, side, q) @ mounts[side]
            passed = pos <= preset['hard_gates']['arm_ik_position_mm_max'] and (
                rot <= preset['hard_gates']['arm_ik_rotation_deg_max']) and (
                float(np.min(margins)) >= preset['hard_gates']['joint_limit_margin_min'])
            rows.append({'frame': frame_id, 'side_index': side,
                         'status': 'PASS_NUMERIC' if passed else 'HOLD_NUMERIC',
                         'position_mm': float(pos), 'rotation_deg': float(rot),
                         'min_joint_margin': float(np.min(margins)), 'seed_index': seed_index})
            previous_previous, previous = previous, q
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / 'ARM_POSITION_ANCHOR_STATES.npz'
    np.savez_compressed(state_path, q_arm=q_arm, T_world_base=world_base,
                        T_tool_hand_root=mounts, T_target_hand_root_world=target,
                        T_actual_hand_root_world=actual, valid_side_frame=valid[:, :count],
                        source_frames=hawor['original_frame_indices'][:count])
    video_path = args.output_root / f'{v1["session_id"]}_POSITION_ANCHOR_4FRAME_ENDPOINT_DIAGNOSTIC.mp4'
    base_pilot._review(raw_path, video_path, hawor, target, actual, rows, count)
    passed_count = sum(row['status'] == 'PASS_NUMERIC' for row in rows)
    output = {
        'schema_version': 'rc1-independent-placement-position-anchor-pilot-v2',
        'created_at': now_iso(), 'session_id': v1['session_id'],
        'status': 'PASSED_DEVELOPMENT_NUMERIC' if passed_count == count * 2 else 'FAILED_QUALITY_C',
        'input_mode': 'OFFLINE_VISUAL', 'training_eligible': False,
        'control_ground_truth': False, 'physical_deployment_authorized': False,
        'inputs': {'v1_result': artifact_ref(args.v1_result),
                   'ablation_result': artifact_ref(args.ablation_result),
                   'preset': artifact_ref(preset_path), 'hawor_npz': artifact_ref(hawor_path),
                   'raw_video': artifact_ref(raw_path), 'code': artifact_ref(Path(__file__))},
        'anchor_frame': anchor, 'frozen_candidate_index': 0,
        'rows': rows,
        'summary': {'frames': count, 'passed_side_rows': passed_count,
                    'total_side_rows': count * 2, 'elapsed_wall_s': round(time.monotonic() - started, 2)},
        'outputs': {'states': artifact_ref(state_path), 'review_video': artifact_ref(video_path)},
        'claim_limit': 'Four-frame position-only anchor numerical pilot. Anchor orientation is chosen from a reachable robot pose by design; no full-session, digital collision, Robot mesh render, causal HaWoR or physical authority.',
    }
    atomic_json(args.output_root / 'RESULT.json', output)
    print(json.dumps({'status': output['status'], 'summary': output['summary'],
                      'result': str(args.output_root / 'RESULT.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
