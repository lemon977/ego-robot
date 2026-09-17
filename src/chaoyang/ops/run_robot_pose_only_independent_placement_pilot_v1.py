#!/usr/bin/env python3
"""Bounded first-prefix, world-fixed Robot placement experiment.

Only the static T_tool_hand_root numerical proxy is inherited. Foreign-session
q, base, wrist offsets, anchor rotation and trajectory are never read. The
HaWoR input remains offline smoothed, so this is NOT a causal-training result.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402


def _proper(transform: np.ndarray, name: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f'{name}: nonfinite or wrong shape')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise RuntimeError(f'{name}: non-homogeneous')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0.999:
        raise RuntimeError(f'{name}: improper rotation')


def _verify(reference: dict) -> Path:
    path = Path(reference['path'])
    actual = artifact_ref(path)
    if actual['bytes'] != reference['bytes'] or actual['sha256'] != reference['sha256']:
        raise RuntimeError(f'bytes/SHA conflict: {path}')
    return path


def _world_base(c2w: np.ndarray, midpoint: np.ndarray,
                cam_base_rotation: np.ndarray, target_in_base: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = c2w[:3, :3] @ cam_base_rotation
    out[:3, 3] = midpoint - out[:3, :3] @ target_in_base
    _proper(out, 'T_world_base')
    return out


def _root_target(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = position
    return out


def _project(point_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int] | None:
    camera = np.linalg.inv(c2w) @ np.r_[point_world, 1.0]
    if not np.isfinite(camera).all() or camera[2] <= 1e-6:
        return None
    point = intrinsic @ camera[:3]
    return int(round(point[0] / point[2])), int(round(point[1] / point[2]))


def _review(raw_path: Path, output: Path, hawor: dict,
            targets: np.ndarray, actual: np.ndarray, rows: list[dict], count: int) -> None:
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        raise RuntimeError('raw video decode failed')
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '10', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output),
    ], stdin=subprocess.PIPE)
    lookup = {(row['frame'], row['side_index']): row for row in rows}
    for frame_id in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'raw video ended before frame {frame_id}')
        cv2.rectangle(frame, (0, 0), (width, 108), (0, 0, 0), -1)
        cv2.putText(frame, 'DEVELOPMENT 4-FRAME NUMERIC PLACEMENT - NOT ROBOT RENDER / NOT CAUSAL',
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
        for side, color in enumerate(((255, 80, 40), (40, 40, 255))):
            target = targets[frame_id, side]
            measured = actual[frame_id, side]
            if not np.isfinite(target).all() or not np.isfinite(measured).all():
                continue
            human_xy = _project(target[:3, 3], hawor['c2w'][frame_id], hawor['intrinsics'][frame_id])
            robot_xy = _project(measured[:3, 3], hawor['c2w'][frame_id], hawor['intrinsics'][frame_id])
            if human_xy:
                cv2.circle(frame, human_xy, 10, color, 3)
            if robot_xy:
                cv2.drawMarker(frame, robot_xy, color, cv2.MARKER_CROSS, 22, 3)
            if human_xy and robot_xy:
                cv2.line(frame, human_xy, robot_xy, color, 2)
            item = lookup[(frame_id, side)]
            line = f'{"LEFT" if side == 0 else "RIGHT"}: {item["position_mm"]:.1f}mm {item["rotation_deg"]:.1f}deg {item["status"]}'
            cv2.putText(frame, line, (12, 61 + 34 * side), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
        for _ in range(10):
            proc.stdin.write(frame.tobytes())
    capture.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('review encode failed')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'],
                             capture_output=True, text=True)
    if decoded.returncode:
        raise RuntimeError(f'review full decode failed: {decoded.stderr[-300:]}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--hawor-result', type=Path, required=True)
    parser.add_argument('--preset', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--wall-cap-s', type=int, default=300)
    args = parser.parse_args()
    if args.output_root.exists() or not 1 <= args.frames <= 24 or not 0 < args.wall_cap_s <= 1800:
        raise RuntimeError('fresh bounded pilot required')
    start = time.monotonic()
    result = json.loads(args.hawor_result.read_text(encoding='utf-8'))
    preset = json.loads(args.preset.read_text(encoding='utf-8'))
    if result.get('session_id') != args.session_id or preset.get('status') != 'FROZEN_DEVELOPMENT_PRESET_NOT_CALIBRATION':
        raise RuntimeError('session/preset identity mismatch')
    hawor_path = _verify(result['outputs']['npz'])
    raw_path = _verify(result['inputs']['source_video'])
    mount_source = _verify(preset['mount_source'])
    if preset['mount_source']['extracted_field_only'] != 'T_tool_hand_root':
        raise RuntimeError('mount source field restriction changed')
    with np.load(mount_source, allow_pickle=False) as source:
        mounts_from_source = np.asarray(source['T_tool_hand_root'], dtype=np.float64)
    mounts = np.asarray(preset['T_tool_hand_root'], dtype=np.float64)
    if mounts.shape != (2, 4, 4) or not np.allclose(mounts, mounts_from_source, atol=1e-8):
        raise RuntimeError('static mount assumption does not match source field')
    for side in range(2):
        _proper(mounts[side], f'mount side {side}')
    with np.load(hawor_path, allow_pickle=False) as source:
        hawor = {key: np.asarray(source[key]) for key in (
            'joints_3d_world', 'c2w', 'intrinsics', 'original_frame_indices')}
    total = len(hawor['c2w'])
    if args.frames > total or not np.array_equal(hawor['original_frame_indices'][:args.frames], np.arange(args.frames)):
        raise RuntimeError('frame identity mismatch')
    human = np.asarray(hawor['joints_3d_world'], np.float64)
    valid = np.isfinite(human).all(axis=(2, 3))
    bilateral = np.flatnonzero(valid[0] & valid[1])
    if len(bilateral) == 0 or bilateral[0] > min(14, args.frames - 1):
        raise RuntimeError('no bilateral observed prefix anchor')
    anchor = int(bilateral[0])
    _proper(hawor['c2w'][anchor], 'anchor c2w')
    rotation_cam_base = np.asarray(preset['camera_to_robot_base_rotation'], np.float64)
    if not np.allclose(rotation_cam_base.T @ rotation_cam_base, np.eye(3), atol=1e-8) or np.linalg.det(rotation_cam_base) < .999:
        raise RuntimeError('camera-base orientation improper')
    midpoint = .5 * (human[0, anchor, 0] + human[1, anchor, 0])
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    neutral = np.clip(np.zeros_like(lower), lower, upper)
    middle = .5 * (lower + upper)
    palm_anchor = [arm.shared.wrist_adapter.final_v3_mano_palm_basis(
        human[side, anchor], handedness=arm.shared.SIDES[side]) for side in range(2)]
    candidate_rows = []
    for candidate_index, base_target in enumerate(preset['base_midpoint_target_candidates_m']):
        if time.monotonic() - start > args.wall_cap_s:
            raise TimeoutError('pilot wall cap exceeded during candidate placement')
        world_base = _world_base(hawor['c2w'][anchor], midpoint,
                                 rotation_cam_base, np.asarray(base_target, np.float64))
        initial_q = []
        errors = []
        for side in range(2):
            neutral_root = world_base @ arm.old.official._tool_fk(assets, side, neutral[side]) @ mounts[side]
            target = _root_target(human[side, anchor, 0], neutral_root[:3, :3])
            tool_target = np.linalg.inv(world_base) @ target @ np.linalg.inv(mounts[side])
            *_, q, seed_index, pos, rot, margins = arm.solve_one(
                assets, side, tool_target, lower[side], upper[side], [neutral[side], middle[side]])
            initial_q.append(q)
            errors.append({'side': side, 'seed_index': seed_index, 'position_mm': float(pos),
                           'rotation_deg': float(rot), 'min_joint_margin': float(np.min(margins))})
        failures = sum(item['position_mm'] > 10 or item['rotation_deg'] > 5 or item['min_joint_margin'] < -1e-8
                       for item in errors)
        score = (failures, sum(item['position_mm'] + item['rotation_deg'] for item in errors), candidate_index)
        candidate_rows.append({'candidate_index': candidate_index, 'base_target_m': base_target,
                               'T_world_base': world_base, 'initial_q': initial_q,
                               'anchor_errors': errors, 'score': score})
    chosen = min(candidate_rows, key=lambda item: item['score'])
    world_base = chosen['T_world_base']
    base_world = np.linalg.inv(world_base)
    count = args.frames
    targets = np.full((count, 2, 4, 4), np.nan)
    actual = np.full_like(targets, np.nan)
    q_arm = np.full((count, 2, 7), np.nan)
    rows = []
    for side in range(2):
        neutral_root = world_base @ arm.old.official._tool_fk(assets, side, neutral[side]) @ mounts[side]
        anchor_rotation = neutral_root[:3, :3]
        previous = None
        previous_previous = None
        for frame_id in range(count):
            if time.monotonic() - start > args.wall_cap_s:
                raise TimeoutError('pilot wall cap exceeded during frame solve')
            if frame_id < anchor or not valid[side, frame_id]:
                rows.append({'frame': frame_id, 'side_index': side, 'status': 'UNKNOWN_MISSING_HAWOR'})
                previous = previous_previous = None
                continue
            palm = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
                human[side, frame_id], handedness=arm.shared.SIDES[side])
            rotation = palm @ palm_anchor[side].T @ anchor_rotation
            target = _root_target(human[side, frame_id, 0], rotation)
            targets[frame_id, side] = target
            tool_target = base_world @ target @ np.linalg.inv(mounts[side])
            if previous is None:
                seeds = [neutral[side], middle[side]]
                lo, hi = lower[side], upper[side]
            else:
                lo, hi = arm.temporal.bounded_limits(
                    lower[side], upper[side], previous, previous_previous,
                    arm.temporal.ARM_SOLVER_STEP_LIMIT)
                seeds = [np.clip(previous, lo, hi), np.clip(neutral[side], lo, hi)]
            *_, q, seed_index, pos, rot, margins = arm.solve_one(
                assets, side, tool_target, lower[side], upper[side], seeds,
                previous=previous, lo=lo, hi=hi)
            q_arm[frame_id, side] = q
            actual[frame_id, side] = world_base @ arm.old.official._tool_fk(assets, side, q) @ mounts[side]
            passed = pos <= preset['hard_gates']['arm_ik_position_mm_max'] and (
                rot <= preset['hard_gates']['arm_ik_rotation_deg_max']) and (
                float(np.min(margins)) >= preset['hard_gates']['joint_limit_margin_min'])
            rows.append({'frame': frame_id, 'side_index': side, 'status': 'PASS_NUMERIC' if passed else 'HOLD_NUMERIC',
                         'position_mm': float(pos), 'rotation_deg': float(rot),
                         'min_joint_margin': float(np.min(margins)), 'seed_index': seed_index})
            previous_previous, previous = previous, q
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / 'ARM_INDEPENDENT_PREFIX_STATES.npz'
    np.savez_compressed(state_path, q_arm=q_arm, T_world_base=world_base,
                        T_tool_hand_root=mounts, T_target_hand_root_world=targets,
                        T_actual_hand_root_world=actual, valid_side_frame=valid[:, :count],
                        source_frames=hawor['original_frame_indices'][:count])
    video_path = args.output_root / f'{args.session_id}_INDEPENDENT_PLACEMENT_4FRAME_ENDPOINT_DIAGNOSTIC.mp4'
    _review(raw_path, video_path, hawor, targets, actual, rows, count)
    passed_rows = sum(item['status'] == 'PASS_NUMERIC' for item in rows)
    result_out = {
        'schema_version': 'rc1-independent-pose-only-placement-pilot-v1',
        'created_at': now_iso(), 'session_id': args.session_id,
        'status': 'PASSED_DEVELOPMENT_NUMERIC' if passed_rows == count * 2 else 'FAILED_QUALITY_C',
        'scope': 'OFFLINE_VISUAL_4FRAME_PILOT', 'authority_promoted': False,
        'training_eligible': False, 'control_ground_truth': False, 'physical_deployment_authorized': False,
        'inputs': {'hawor_result': artifact_ref(args.hawor_result), 'hawor_npz': artifact_ref(hawor_path),
                   'raw_video': artifact_ref(raw_path), 'preset': artifact_ref(args.preset),
                   'robot_asset_pin': artifact_ref(ROOT / 'assets/robot/ROBOT_ASSET_PIN.json'),
                   'code': artifact_ref(Path(__file__))},
        'first_prefix_anchor_frame': anchor,
        'chosen_candidate_index': chosen['candidate_index'],
        'candidate_evidence': [
            {'candidate_index': item['candidate_index'], 'base_target_m': item['base_target_m'],
             'anchor_errors': item['anchor_errors'], 'score': list(item['score'])} for item in candidate_rows],
        'rows': rows,
        'summary': {'frames': count, 'passed_side_rows': passed_rows,
                    'total_side_rows': count * 2, 'elapsed_wall_s': round(time.monotonic() - start, 2)},
        'outputs': {'states': artifact_ref(state_path), 'review_video': artifact_ref(video_path)},
        'claim_limit': 'First-prefix numerical placement only. HaWoR input may use offline temporal smoothing; mount/base are unmeasured proxies. This is not a rendered full Robot, contact, causal-training or physical result.',
    }
    atomic_json(args.output_root / 'RESULT.json', result_out)
    print(json.dumps({'status': result_out['status'], 'summary': result_out['summary'],
                      'result': str(args.output_root / 'RESULT.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
