#!/usr/bin/env python3
"""Append a verified independent-placement alternate video to the 60-row matrix."""
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
        raise FileExistsError('immutable revision required')
    matrix = json.loads(args.base.read_text(encoding='utf-8'))
    rows = matrix['rows']
    if len(rows) != 60 or len({(row['task'], row['session_id']) for row in rows}) != 60:
        raise RuntimeError('60-row denominator conflict')
    render = json.loads(args.render_result.read_text(encoding='utf-8'))
    collision = json.loads(args.collision_result.read_text(encoding='utf-8'))
    if (render['session'], render['status'], render['frame_count']) != (
        'play_cards_0901_001', 'PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW', 645
    ) or render['authority'] or not render['numeric']['arm_pass'] or not render['numeric']['hand_pass']:
        raise RuntimeError('render is not the frozen numeric Poker001 review')
    if (collision['session_id'], collision['status'], collision['illegal_contact_count']) != (
        'play_cards_0901_001', 'PASS_DEVELOPMENT_COLLISION_AUDIT', 0
    ) or len(collision['selected_frames']) != 643:
        raise RuntimeError('full collision audit mismatch')
    video = render['outputs']['video']
    video_ref = artifact_ref(Path(video['path']))
    if (video_ref['bytes'], video_ref['sha256'], video['decoded_frames']) != (
        video['bytes'], video['sha256'], 645
    ):
        raise RuntimeError('video receipt/SHA/frame mismatch')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,avg_frame_rate,width,height', '-of', 'json', video_ref['path'],
    ], text=True, timeout=30))['streams'][0]
    if (int(probe['nb_frames']), probe['avg_frame_rate'], probe['width'], probe['height']) != (
        645, '30/1', 1920, 480
    ):
        raise RuntimeError('video stream geometry conflict')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', video_ref['path'],
                    '-f', 'null', '-'], check=True, timeout=120)
    selected = [row for row in rows if row['session_id'] == 'play_cards_0901_001']
    if len(selected) != 1 or selected[0]['task'] != 'poker' or not selected[0]['delivery_full_video']:
        raise RuntimeError('Poker001 historical video row missing')
    row = selected[0]
    if row.get('independent_placement_successor'):
        raise RuntimeError('alternate successor already exists')
    row['independent_placement_successor'] = {
        'mode': 'OFFLINE_VISUAL', 'video': video_ref,
        'video_frames': 645, 'video_fps': 30, 'full_decode': 'PASS_FFMPEG_XERROR',
        'arm_hand_numeric_pass': True, 'digital_collision_pass': True,
        'render_result': artifact_ref(args.render_result),
        'collision_result': artifact_ref(args.collision_result),
        'training_eligible': False, 'robotized_clean_background': False,
        'claim_limit': 'Additional raw-background full Robot review. Static mount unmeasured; HaWoR offline-smoothed; no causal or physical authority.',
    }
    counts = {task: sum(bool(item['delivery_full_video']) for item in rows if item['task'] == task)
              for task in ('chips', 'poker')}
    if counts != {'chips': 30, 'poker': 20}:
        raise RuntimeError(f'primary delivery count changed: {counts}')
    matrix.update({
        'schema_version': 'rc1-robot30-research-delivery-matrix-rev4',
        'supersedes': artifact_ref(args.base),
        'generator': artifact_ref(Path(__file__)),
        'normalized_watchable_counts': counts,
        'claim_limit': 'Primary 60-row terminal and 30/20 video counts unchanged; one separately verified Poker001 alternate offline review added.',
    })
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0004.json', matrix)
    csv_path = args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0004.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'task', 'rank', 'session_id', 'primary_video_exists', 'primary_hard_geometry_pass',
            'independent_alternate_video_exists', 'independent_alternate_video_full_decode',
            'independent_alternate_numeric_pass', 'causal_training_eligible',
        ])
        writer.writeheader()
        for item in rows:
            alt = item.get('independent_placement_successor') or {}
            writer.writerow({
                'task': item['task'], 'rank': item['rank'], 'session_id': item['session_id'],
                'primary_video_exists': bool(item['delivery_full_video']),
                'primary_hard_geometry_pass': item.get('robot30_hard_geometry_pass', False),
                'independent_alternate_video_exists': bool(alt.get('video')),
                'independent_alternate_video_full_decode': alt.get('full_decode') == 'PASS_FFMPEG_XERROR',
                'independent_alternate_numeric_pass': alt.get('arm_hand_numeric_pass', False) and alt.get('digital_collision_pass', False),
                'causal_training_eligible': False,
            })
    print(json.dumps({'primary_counts': counts, 'new_alternate_videos': 1,
                      'output': str(args.output_root)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
