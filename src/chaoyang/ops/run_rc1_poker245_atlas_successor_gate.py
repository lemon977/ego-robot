#!/usr/bin/env python3
"""Close pixel lineage for the frozen Poker245 V4 occlusion event.

This is deliberately a fail-closed development canary. V4's optical-flow and
homography checks can propose pixels, but the current evidence does not bind a
physical card and its face across the hidden interval independently. Proposed
pixels are therefore recorded, not written into the strict output.
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
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops import run_poker_real_occlusion_plane_donor_v3 as v3
from chaoyang.ops import run_poker_real_occlusion_plane_donor_v4 as v4

FONT = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
RAW_VISIBLE = 0
UNKNOWN = 5
WARPED_PROPOSAL = 6


def _homography(source_keypoints, source_desc, target_keypoints, target_desc):
    if source_desc is None or target_desc is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(source_desc, target_desc, k=2)
    backward = matcher.match(target_desc, source_desc)
    reverse = {item.queryIdx: item.trainIdx for item in backward}
    matches = []
    for pair in forward:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            match = pair[0]
            if reverse.get(match.trainIdx) == match.queryIdx:
                matches.append(match)
    if len(matches) < 4:
        return None
    src = np.float32([source_keypoints[item.queryIdx].pt for item in matches])
    dst = np.float32([target_keypoints[item.trainIdx].pt for item in matches])
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    return matrix


def _panel(frame, label, line):
    image = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, 640, 72), fill=(0, 0, 0))
    font = ImageFont.truetype(str(FONT), 17)
    draw.text((8, 5), label, font=font, fill=(255, 255, 255))
    draw.text((8, 39), line, font=font, fill=(255, 230, 80))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v4-result', type=Path, required=True)
    parser.add_argument('--closure-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f'fresh immutable attempt required: {args.output_root}')
    v4_result = json.loads(args.v4_result.read_text(encoding='utf-8'))
    closure = json.loads(args.closure_result.read_text(encoding='utf-8'))
    if v4_result.get('session') != 'play_cards_0903_245' or closure.get('case') != 'Poker245':
        raise RuntimeError('full session identity mismatch')
    if v4_result.get('summary', {}).get('runtime_event_frames') != 56:
        raise RuntimeError('V4 event is not the frozen 56-frame interval')
    object_path = Path(v4_result['inputs']['object_manifest']['path'])
    role_path = Path(v4_result['inputs']['role_manifest']['path'])
    old_clean_path = Path(v4_result['inputs']['old_clean_video']['path'])
    for name, reference in v4_result['inputs'].items():
        path = Path(reference['path'])
        actual = artifact_ref(path)
        if actual['bytes'] != reference['bytes'] or actual['sha256'] != reference['sha256']:
            raise RuntimeError(f'V4 input closure mismatch: {name}')
    frozen_decision_path = Path(v4_result['outputs']['frame_decisions']['path'])
    if artifact_ref(frozen_decision_path)['sha256'] != v4_result['outputs']['frame_decisions']['sha256']:
        raise RuntimeError('V4 decision closure mismatch')
    objects = json.loads(object_path.read_text(encoding='utf-8'))
    roles = json.loads(role_path.read_text(encoding='utf-8'))
    decisions = json.loads(frozen_decision_path.read_text(encoding='utf-8'))['rows']
    rows = [row for row in decisions if row['event_phase'] == 'REAL_OCCLUSION_RUNTIME']
    if [row['target_frame'] for row in rows] != list(range(34, 90)):
        raise RuntimeError('frozen V4 frame identity mismatch')
    raw_path = Path(objects['input']['selected_rgb']['path'])
    frames = v3._decode(raw_path, 90)
    old_frames = v3._decode(old_clean_path, 90)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=5, edgeThreshold=10, patchSize=21)
    sources = v3._source_features(frames, objects, orb)
    state_mask = sources[33][0]
    previous_frame = frames[33]
    previous_human = cv2.imread(roles['frames'][33]['hand_tracker_union']['path'], cv2.IMREAD_GRAYSCALE) > 0
    args.output_root.mkdir(parents=True)
    map_root = args.output_root / 'pixel_maps'
    map_root.mkdir()
    video_path = args.output_root / 'Poker245_V4候选_同牌面严格拒绝_真实遮挡56帧.mp4'
    process = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '2560x480', '-r', '10', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt',
        'yuv420p', '-movflags', '+faststart', str(video_path),
    ], stdin=subprocess.PIPE)
    output_rows = []
    proposed_total = 0
    unknown_total = 0
    for row in rows:
        target_id = int(row['target_frame'])
        target = frames[target_id]
        old = old_frames[target_id]
        human_ref = roles['frames'][target_id]['hand_tracker_union']
        human = cv2.imread(human_ref['path'], cv2.IMREAD_GRAYSCALE)
        if human is None or human.shape != target.shape[:2]:
            raise RuntimeError(f'human mask unavailable/mismatched at {target_id}')
        human = human > 0
        predicted_mask, flow_evidence = v4._flow_prediction(previous_frame, target, state_mask, previous_human)
        flow_pass = bool(flow_evidence['flow_runtime_pass'])
        search_seed = predicted_mask if flow_pass else state_mask
        search_roi = v3._expanded_roi(search_seed, search_seed.shape,
                                      factor=2.0 if flow_pass else 2.5)
        human_expanded = cv2.dilate(human.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        allowed = (search_roi > 0) & ~human_expanded
        target_kp, target_desc = orb.detectAndCompute(target, allowed.astype(np.uint8) * 255)
        candidates = []
        for candidate_source, (candidate_mask, candidate_kp, candidate_desc) in sources.items():
            candidate_metrics, candidate_visual = v3._candidate(
                source_frame=candidate_source, target_frame=target_id,
                source=frames[candidate_source], target=target,
                source_mask=candidate_mask, source_keypoints=candidate_kp,
                source_desc=candidate_desc, target_keypoints=target_kp,
                target_desc=target_desc, target_allowed=allowed,
                search_roi=search_roi > 0,
            )
            candidates.append((candidate_metrics, candidate_visual))
        replay_row, visual = max(candidates, key=lambda item: v3._score(item[0]))
        identity = v4._identity_checks(visual['projected_mask'], predicted_mask, flow_evidence)
        replay_pass = bool(replay_row['runtime_pass'] and all(identity.values()))
        if replay_row['source_frame'] != row['source_frame'] or replay_pass != row['runtime_pass']:
            raise RuntimeError(f'V4 deterministic replay mismatch at {target_id}')
        # Strict output writes no hidden card pixels without independent face-ID.
        strict = target.copy()
        strict[human] = (80, 80, 80)
        source_class = np.full(human.shape, RAW_VISIBLE, np.uint8)
        source_class[human] = UNKNOWN
        proposal_class = np.zeros(human.shape, np.uint8)
        source_frame = np.full(human.shape, -1, np.int16)
        source_xy = np.full((*human.shape, 2), -1, np.int16)
        proposed = target.copy()
        proposal = np.zeros(human.shape, bool)
        reason = 'V4_RUNTIME_REJECTED'
        matrix_values = None
        source_id = int(row['source_frame'])
        if row['runtime_pass']:
            if source_id >= target_id or source_id not in sources:
                raise RuntimeError(f'invalid causal source at {target_id}')
            source_mask, source_kp, source_desc = sources[source_id]
            if objects['frames'][source_id]['physical_instances']['0']['physical_instance_id'] != 0:
                raise RuntimeError('donor source physical instance changed')
            matrix = _homography(source_kp, source_desc, target_kp, target_desc)
            if matrix is None:
                raise RuntimeError(f'cannot reproduce V4 homography at {target_id}')
            projected = cv2.warpPerspective(source_mask.astype(np.uint8), matrix,
                                            (target.shape[1], target.shape[0]),
                                            flags=cv2.INTER_NEAREST) > 0
            proposal = projected & human
            if int(proposal.sum()) != int(row['write_pixels']):
                raise RuntimeError(f'V4 write-pixel mismatch at {target_id}: {proposal.sum()} != {row["write_pixels"]}')
            inverse = np.linalg.inv(matrix)
            yy, xx = np.where(proposal)
            target_xy = np.stack([xx, yy], axis=1).astype(np.float32).reshape(-1, 1, 2)
            source_float = cv2.perspectiveTransform(target_xy, inverse).reshape(-1, 2)
            sx = np.rint(source_float[:, 0]).astype(np.int32)
            sy = np.rint(source_float[:, 1]).astype(np.int32)
            in_bounds = (sx >= 0) & (sx < target.shape[1]) & (sy >= 0) & (sy < target.shape[0])
            valid = np.zeros_like(proposal)
            valid[yy[in_bounds], xx[in_bounds]] = source_mask[sy[in_bounds], sx[in_bounds]]
            if int(valid.sum()) != len(xx):
                # Do not keep a partial mapping while claiming V4 exact replay.
                raise RuntimeError(f'source-xy closure mismatch at {target_id}')
            proposal_class[proposal] = WARPED_PROPOSAL
            source_frame[proposal] = source_id
            source_xy[yy, xx, 0] = sx.astype(np.int16)
            source_xy[yy, xx, 1] = sy.astype(np.int16)
            proposed[proposal] = frames[source_id][sy, sx]
            matrix_values = matrix.tolist()
            reason = 'WITHHELD_PHYSICAL_FACE_ID_NOT_INDEPENDENTLY_PROVEN'
            state_mask = visual['projected_mask']
        elif flow_pass:
            state_mask = predicted_mask
        previous_frame = target
        previous_human = human
        # A proposal is not promoted to legal pixels; the final map is UNKNOWN.
        np.savez_compressed(map_root / f'{target_id:06d}.npz',
                            final_pixel_source=source_class, proposal_source=proposal_class,
                            donor_frame_id=source_frame, donor_xy=source_xy,
                            m_write=human.astype(np.uint8))
        unknown_total += int(human.sum())
        proposed_total += int(proposal.sum())
        overlay = target.copy()
        overlay[human] = (0, 120, 255)
        overlay[proposal] = (255, 0, 255)
        line = f'frame {target_id} | V4 proposal={int(proposal.sum())} | strict=UNKNOWN'
        cells = [
            _panel(target, 'Raw', line),
            _panel(old, '旧Clean可视基线（非真值）', line),
            _panel(proposed, 'V4开发候选（不可训练）', line),
            _panel(overlay, '严格牌面门：橙UNKNOWN／紫候选', reason),
        ]
        assert process.stdin is not None
        process.stdin.write(np.concatenate(cells, axis=1).tobytes())
        output_rows.append({
            'target_frame': target_id,
            'source_frame': source_id,
            'v4_runtime_pass': bool(row['runtime_pass']),
            'physical_instance_id': 0,
            'source_face_appearance': 'RIGHTMOST_PURPLE_BACK',
            'target_face_id': 'UNKNOWN',
            'independent_physical_face_proof': False,
            'strict_runtime_pass': False,
            'proposal_pixels': int(proposal.sum()),
            'strict_written_pixels': 0,
            'unknown_pixels': int(human.sum()),
            'reason': reason,
            'homography_source_to_target': matrix_values,
            'pixel_map': artifact_ref(map_root / f'{target_id:06d}.npz'),
        })
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('video encoding failed')
    decision_path = args.output_root / 'FRAME_DECISIONS.json'
    atomic_json(decision_path, {'schema_version': 'rc1-poker245-atlas-successor-gate-v1', 'rows': output_rows})
    result = {
        'schema_version': 'rc1-poker245-atlas-successor-gate-v1',
        'created_at': now_iso(), 'session_id': 'play_cards_0903_245',
        'status': 'FAILED_QUALITY_C', 'input_mode': 'OFFLINE_VISUAL',
        'causal_dependency_claim': 'HOLD_UPSTREAM_MASK_IDENTITY_NOT_PROVEN',
        'training_eligible': False, 'authority_promoted': False,
        'summary': {'frames': len(output_rows), 'v4_candidate_frames': sum(x['v4_runtime_pass'] for x in output_rows),
                    'v4_proposed_pixels': proposed_total, 'strict_written_pixels': 0,
                    'strict_unknown_pixels': unknown_total,
                    'reason': 'Independent same-physical-card and same-face proof unavailable across hidden interval.'},
        'inputs': {'v4_result': artifact_ref(args.v4_result), 'closure_result': artifact_ref(args.closure_result),
                   'raw_video': artifact_ref(raw_path), 'object_manifest': artifact_ref(object_path),
                   'role_manifest': artifact_ref(role_path), 'old_clean_video': artifact_ref(old_clean_path),
                   'code': artifact_ref(Path(__file__).resolve())},
        'outputs': {'review_video': artifact_ref(video_path), 'frame_decisions': artifact_ref(decision_path)},
        'claim_limit': 'Development pixel-source proposal only. No Clean improvement, hidden-pixel accuracy, causal training, Object6D/contact, or physical truth claim.',
    }
    atomic_json(args.output_root / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'summary': result['summary'],
                      'video': str(video_path)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
