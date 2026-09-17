#!/usr/bin/env python3
"""Append the verified Poker005 C video to the immutable research 60-row index."""
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json  # noqa: E402


def _probe(path: Path) -> tuple[int, str]:
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,avg_frame_rate', '-of', 'json', str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    return int(stream['nb_frames']), stream['avg_frame_rate']


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--poker005-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f'no-clobber: {args.output_root}')
    base = json.loads(args.base.read_text(encoding='utf-8'))
    result = json.loads(args.poker005_result.read_text(encoding='utf-8'))
    if len(base['rows']) != 60 or result['session_id'] != 'play_cards_0901_005':
        raise RuntimeError('fixed 60-row / full session identity mismatch')
    if result['terminal_status'] != 'FAILED_QUALITY_C_DIAGNOSTIC_VIDEO':
        raise RuntimeError('Poker005 quality terminal changed')
    video = result['full_video']
    path = Path(video['path'])
    actual = artifact_ref(path)
    if actual['bytes'] != video['bytes'] or actual['sha256'] != video['sha256']:
        raise RuntimeError('Poker005 video SHA mismatch')
    frames, fps = _probe(path)
    if frames != 520 or fps != '30/1' or video['decode'] != 'PASS_FFMPEG_XERROR':
        raise RuntimeError('Poker005 review frame/fps/decode closure mismatch')
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                            capture_output=True, text=True)
    if decode.returncode:
        raise RuntimeError(f'Poker005 video full decode failed: {decode.stderr[-300:]}')
    rows = base['rows']
    matches = [row for row in rows if row['session_id'] == result['session_id']]
    if len(matches) != 1 or matches[0]['verified_video'] is not None:
        raise RuntimeError('Poker005 row missing or already has video')
    row = matches[0]
    row.update({
        'robot30_terminal_status': result['terminal_status'],
        'robot30_hard_geometry_pass': False,
        'robot30_result_ref': artifact_ref(args.poker005_result),
        'robot30_result_closure': 'PASS',
        'input_mode': result['input_mode'],
        'training_eligible': False,
        'delivery_class': 'C_NEW_C_DIAGNOSTIC_FULL_VIDEO',
        'verified_video': actual,
        'verified_frame_count': frames,
        'review_ref_state': 'PASS',
        'claim_limit': 'Verified complete C-grade offline video only; not hard geometry pass, causal training, control or physical authority.',
    })
    counts = base['counts']
    counts['poker']['new_c_diagnostic_video'] += 1
    counts['poker']['all_watchable_full_video'] += 1
    counts['poker']['missing_full_video'] -= 1
    if counts['poker']['all_watchable_full_video'] != 20 or counts['poker']['missing_full_video'] != 10:
        raise RuntimeError('Poker research count did not close to 20/30')
    if len({(r['task'], r['session_id']) for r in rows}) != 60:
        raise RuntimeError('duplicate session in 60-row matrix')
    base.update({
        'schema_version': 'rc1-robot30-research-delivery-matrix-rev2',
        'mode': 'OFFLINE_VISUAL_RESEARCH',
        'authority_promoted': False,
        'supersedes': artifact_ref(args.base),
        'generator': artifact_ref(Path(__file__)),
        'successor_results': [*base['successor_results'], artifact_ref(args.poker005_result)],
        'claim_limit': 'Research video closure only: video existence, full decode, hard geometry and causal training remain separate.',
    })
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0002.json', base)
    csv_path = args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0002.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'task', 'rank', 'session_id', 'robot30_terminal_status',
            'video_exists', 'full_decode', 'hard_geometry_pass', 'causal_training_eligible',
            'verified_frame_count', 'video_path', 'video_sha256', 'delivery_class',
        ])
        writer.writeheader()
        for item in rows:
            ref = item.get('verified_video') or {}
            writer.writerow({
                'task': item['task'], 'rank': item['rank'], 'session_id': item['session_id'],
                'robot30_terminal_status': item['robot30_terminal_status'],
                'video_exists': bool(ref), 'full_decode': item.get('review_ref_state') == 'PASS',
                'hard_geometry_pass': item.get('robot30_hard_geometry_pass', False),
                'causal_training_eligible': item.get('training_eligible', False),
                'verified_frame_count': item.get('verified_frame_count'),
                'video_path': ref.get('path'), 'video_sha256': ref.get('sha256'),
                'delivery_class': item['delivery_class'],
            })
    print(json.dumps({'chips_watchable': counts['chips']['all_watchable_full_video'],
                      'poker_watchable': counts['poker']['all_watchable_full_video'],
                      'poker_missing': counts['poker']['missing_full_video']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
