#!/usr/bin/env python3
"""Append a C-watermarked Chips182 alternate while preserving prior 60 rows."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
import subprocess
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--render-result', type=Path, required=True)
    parser.add_argument('--collision-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('immutable matrix revision required')
    matrix = json.loads(args.base.read_text(encoding='utf-8'))
    rows = matrix['rows']
    if len(rows) != 60 or len({(r['task'], r['session_id']) for r in rows}) != 60:
        raise RuntimeError('frozen 60-row denominator mismatch')
    render = json.loads(args.render_result.read_text(encoding='utf-8'))
    collision = json.loads(args.collision_result.read_text(encoding='utf-8'))
    if (render['session'], render['frame_count'], render['status']) != (
        'get_potato_chips_0903_182', 351, 'HOLD_HAND_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW'
    ) or render['authority'] or not render['numeric']['arm_pass'] or render['numeric']['hand_pass']:
        raise RuntimeError('Chips182 C-grade render status mismatch')
    if (collision['session_id'], collision['status'], collision['illegal_contact_count'],
            len(collision['selected_frames'])) != (
        'get_potato_chips_0903_182', 'PASS_DEVELOPMENT_COLLISION_AUDIT', 0, 351
    ):
        raise RuntimeError('full collision audit mismatch')
    video = render['outputs']['video']
    video_ref = artifact_ref(Path(video['path']))
    if (video_ref['sha256'], video_ref['bytes'], video['decoded_frames']) != (
        video['sha256'], video['bytes'], 351
    ):
        raise RuntimeError('video SHA/frame closure mismatch')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,avg_frame_rate,width,height', '-of', 'json', video_ref['path'],
    ], text=True, timeout=30))['streams'][0]
    if (int(probe['nb_frames']), probe['avg_frame_rate'], probe['width'], probe['height']) != (
        351, '30/1', 1920, 480
    ):
        raise RuntimeError('video stream mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', video_ref['path'],
                    '-f', 'null', '-'], check=True, timeout=120)
    match = [r for r in rows if r['session_id'] == 'get_potato_chips_0903_182']
    if len(match) != 1 or match[0]['task'] != 'chips' or not match[0]['delivery_full_video']:
        raise RuntimeError('Chips182 primary video row missing')
    row = match[0]
    if row.get('independent_placement_successor'):
        raise RuntimeError('alternate already published')
    row['independent_placement_successor'] = {
        'mode': 'OFFLINE_VISUAL', 'video': video_ref, 'video_frames': 351,
        'full_decode': 'PASS_FFMPEG_XERROR', 'arm_numeric_pass': True,
        'hand_soft_similarity_pass': False, 'hand_failed_rows': 114,
        'digital_collision_pass': True, 'visual_grade': 'C_WATERMARKED_DIAGNOSTIC',
        'render_result': artifact_ref(args.render_result),
        'collision_result': artifact_ref(args.collision_result),
        'training_eligible': False, 'robotized_clean_background': False,
        'claim_limit': 'Raw-background C-grade review with 114 hand soft-anatomy HOLD rows; no authority or causal training.',
    }
    counts = {task: sum(bool(r['delivery_full_video']) for r in rows if r['task'] == task)
              for task in ('chips', 'poker')}
    if counts != {'chips': 30, 'poker': 20}:
        raise RuntimeError(f'primary video count changed: {counts}')
    matrix.update({
        'schema_version': 'rc1-robot30-research-delivery-matrix-rev5',
        'supersedes': artifact_ref(args.base), 'generator': artifact_ref(Path(__file__)),
        'normalized_watchable_counts': counts,
        'claim_limit': 'Primary 60-row counts/terminals unchanged; Chips182 C-grade alternate added separately from Poker001 numeric alternate.',
    })
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0005.json', matrix)
    with (args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0005.csv').open(
        'x', newline='', encoding='utf-8'
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'task', 'rank', 'session_id', 'primary_video_exists', 'primary_hard_geometry_pass',
            'independent_alternate_video_exists', 'independent_alternate_full_decode',
            'independent_alternate_arm_pass', 'independent_alternate_hand_pass',
            'independent_alternate_visual_grade', 'causal_training_eligible',
        ])
        writer.writeheader()
        for item in rows:
            alt = item.get('independent_placement_successor') or {}
            writer.writerow({
                'task': item['task'], 'rank': item['rank'], 'session_id': item['session_id'],
                'primary_video_exists': bool(item['delivery_full_video']),
                'primary_hard_geometry_pass': item.get('robot30_hard_geometry_pass', False),
                'independent_alternate_video_exists': bool(alt.get('video')),
                'independent_alternate_full_decode': alt.get('full_decode') == 'PASS_FFMPEG_XERROR',
                'independent_alternate_arm_pass': alt.get('arm_numeric_pass', alt.get('arm_hand_numeric_pass', False)),
                'independent_alternate_hand_pass': alt.get('hand_soft_similarity_pass', alt.get('arm_hand_numeric_pass', False)),
                'independent_alternate_visual_grade': alt.get('visual_grade', 'NUMERIC_DEVELOPMENT' if alt else ''),
                'causal_training_eligible': False,
            })
    print(json.dumps({'primary_counts': counts, 'alternate_video_rows': 2,
                      'output': str(args.output_root)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
