#!/usr/bin/env python3
"""Freeze, but do not launch, nine exact Poker pose-only research packets."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def checked(reference: dict) -> dict:
    current = artifact_ref(Path(reference['path']))
    if (current['bytes'], current['sha256']) != (reference['bytes'], reference['sha256']):
        raise RuntimeError(f"reference changed: {reference['path']}")
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--packet-root', type=Path, required=True)
    parser.add_argument('--preset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
    poker = sorted((row for row in matrix['rows'] if row['task'] == 'poker'
                    and row['delivery_class'] == 'C_NEW_SOLVE_REQUIRED'
                    and row.get('delivery_full_video') is None), key=lambda row: row['rank'])
    if len(matrix['rows']) != 60 or len(poker) != 9:
        raise RuntimeError('frozen 60-row / nine-gap denominator changed')
    preset_ref = artifact_ref(args.preset)
    entries = []
    for row in poker:
        session_id = row['session_id']
        packet_path = args.packet_root / session_id / 'TASK_PACKET.json'
        packet = json.loads(packet_path.read_text(encoding='utf-8'))
        if packet['session_id'] != session_id or packet['task'] != 'poker':
            raise RuntimeError(f'packet identity conflict: {session_id}')
        inputs = {name: checked(packet['inputs'][name]) for name in
                  ('hawor_result', 'hawor_npz', 'raw_video', 'robot_asset_pin')}
        hawor_result = json.loads(Path(inputs['hawor_result']['path']).read_text(encoding='utf-8'))
        if hawor_result.get('session_id') != session_id:
            raise RuntimeError(f'HaWoR result identity conflict: {session_id}')
        if checked(hawor_result['outputs']['npz'])['sha256'] != inputs['hawor_npz']['sha256']:
            raise RuntimeError(f'HaWoR NPZ closure conflict: {session_id}')
        if checked(hawor_result['inputs']['source_video'])['sha256'] != inputs['raw_video']['sha256']:
            raise RuntimeError(f'Raw closure conflict: {session_id}')
        entries.append({
            'queue_order': len(entries) + 1, 'rank': row['rank'], 'session_id': session_id,
            'frame_count': packet['frame_count'], 'status': 'PENDING_FIXED_REGRESSION_REVIEW',
            'packet': artifact_ref(packet_path), 'inputs': inputs,
            'execution_mode': 'OFFLINE_VISUAL', 'training_eligible': False,
            'attempt_policy': 'fresh immutable directory; quality C not automatically retried',
        })
    atomic_json(args.output, {
        'schema_version': 'rc1-independent-pose-only-nine-poker-queue-v1',
        'created_at': now_iso(), 'status': 'FROZEN_NOT_LAUNCHED',
        'source_matrix': artifact_ref(args.matrix), 'preset': preset_ref,
        'generator': artifact_ref(Path(__file__)), 'entries': entries,
        'prerequisites': ['three fixed 24-frame numeric regressions',
                          'Poker001 complete review and full decode',
                          'one-session-at-a-time resource budget and no-clobber attempts'],
        'excluded': {'play_cards_0902_049': 'upstream FAILED_RUNTIME_FINAL; independent diagnosis only'},
        'claim_limit': 'Queue input closure only. No new video, quality pass, causal training or authority.',
    })
    print(json.dumps({'status': 'FROZEN_NOT_LAUNCHED', 'entries': len(entries),
                      'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
