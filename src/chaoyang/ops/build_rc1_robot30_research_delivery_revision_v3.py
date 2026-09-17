#!/usr/bin/env python3
"""Normalize prior/research video references without changing authority status."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json  # noqa: E402


def _closed(ref: dict) -> bool:
    if not ref:
        return False
    actual = artifact_ref(Path(ref['path']))
    return actual['bytes'] == ref['bytes'] and actual['sha256'] == ref['sha256']


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError('new immutable revision required')
    data = json.loads(args.base.read_text(encoding='utf-8'))
    rows = data['rows']
    if len(rows) != 60 or len({(r['task'], r['session_id']) for r in rows}) != 60:
        raise RuntimeError('fixed 60-row selection mismatch')
    for row in rows:
        prior = row.get('verified_video')
        successor = row.get('research_successor') or {}
        research = successor.get('video')
        if prior and research:
            raise RuntimeError(f'duplicate video sources: {row["session_id"]}')
        ref = prior or research
        if ref and not _closed(ref):
            raise RuntimeError(f'video bytes/SHA mismatch: {row["session_id"]}')
        if ref:
            source_kind = ('RESEARCH_C_DIAGNOSTIC' if research or row.get('delivery_class', '').startswith('C_NEW_C_')
                           else 'PRIOR_VERIFIED')
            candidate_decode = any(
                candidate.get('artifact', {}).get('sha256') == ref['sha256']
                and candidate.get('full_decode', {}).get('status') == 'PASS'
                and candidate.get('frame_match') is True
                for candidate in row.get('candidate_videos', [])
            )
            decode = ((row.get('review_ref_state') == 'PASS' or candidate_decode)
                      if prior else successor.get('video_qa', {}).get('full_decode') == 'PASS_FFMPEG_XERROR')
            if not decode:
                raise RuntimeError(f'video decode receipt missing: {row["session_id"]}')
            frames = row.get('verified_frame_count') if prior else successor['video_qa']['frames']
            if frames is None or frames <= 0:
                raise RuntimeError(f'video frame count missing: {row["session_id"]}')
            row['delivery_full_video'] = {
                'artifact': {'path': ref['path'], 'bytes': ref['bytes'], 'sha256': ref['sha256']},
                'source_kind': source_kind,
                'frames': frames,
                'full_decode_receipt_pass': True,
            }
        else:
            row['delivery_full_video'] = None
        row['research_video_terminal_status'] = successor.get(
            'terminal_status', row['robot30_terminal_status'])
        row['causal_training_eligible'] = False
    counts = {task: sum(bool(r['delivery_full_video']) for r in rows if r['task'] == task)
              for task in ('chips', 'poker')}
    if counts != {'chips': 30, 'poker': 20}:
        raise RuntimeError(f'video total mismatch: {counts}')
    data.update({
        'schema_version': 'rc1-robot30-research-delivery-matrix-rev3',
        'supersedes': artifact_ref(args.base),
        'generator': artifact_ref(Path(__file__)),
        'normalized_watchable_counts': counts,
        'claim_limit': 'delivery_full_video is byte/SHA-closed with existing decode receipt; independent of hard geometry, causality, Clean and training.',
    })
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0003.json', data)
    with (args.output_root / 'ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0003.csv').open(
        'x', newline='', encoding='utf-8'
    ) as handle:
        fields = ['task', 'rank', 'session_id', 'historical_robot30_terminal_status',
                  'research_video_terminal_status', 'video_exists', 'full_decode',
                  'hard_geometry_pass', 'causal_training_eligible', 'video_frames',
                  'video_path', 'video_sha256', 'video_source_kind']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            video = row['delivery_full_video'] or {}
            ref = video.get('artifact') or {}
            writer.writerow({
                'task': row['task'], 'rank': row['rank'], 'session_id': row['session_id'],
                'historical_robot30_terminal_status': row['robot30_terminal_status'],
                'research_video_terminal_status': row['research_video_terminal_status'],
                'video_exists': bool(video), 'full_decode': video.get('full_decode_receipt_pass', False),
                'hard_geometry_pass': row.get('robot30_hard_geometry_pass', False),
                'causal_training_eligible': False, 'video_frames': video.get('frames'),
                'video_path': ref.get('path'), 'video_sha256': ref.get('sha256'),
                'video_source_kind': video.get('source_kind'),
            })
    print(json.dumps(counts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
