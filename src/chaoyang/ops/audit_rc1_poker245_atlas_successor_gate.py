#!/usr/bin/env python3
"""Verify the frozen Poker245 fail-closed atlas canary and its pixel maps."""
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'no-clobber audit required: {args.output}')
    result = json.loads(args.result.read_text(encoding='utf-8'))
    if result['session_id'] != 'play_cards_0903_245' or result['status'] != 'FAILED_QUALITY_C':
        raise RuntimeError('result identity/status mismatch')
    for name, reference in {**result['inputs'], **result['outputs']}.items():
        actual = artifact_ref(Path(reference['path']))
        if actual['bytes'] != reference['bytes'] or actual['sha256'] != reference['sha256']:
            raise RuntimeError(f'artifact closure mismatch: {name}')
    decisions = json.loads(Path(result['outputs']['frame_decisions']['path']).read_text(encoding='utf-8'))
    rows = decisions['rows']
    if [row['target_frame'] for row in rows] != list(range(34, 90)):
        raise RuntimeError('not the frozen 56-frame event')
    objects = json.loads(Path(result['inputs']['object_manifest']['path']).read_text(encoding='utf-8'))
    proposal_pixels = 0
    unknown_pixels = 0
    for row in rows:
        ref = row['pixel_map']
        actual = artifact_ref(Path(ref['path']))
        if actual['bytes'] != ref['bytes'] or actual['sha256'] != ref['sha256']:
            raise RuntimeError(f'pixel map SHA mismatch: {row["target_frame"]}')
        with np.load(ref['path']) as data:
            final = data['final_pixel_source']
            proposal = data['proposal_source']
            donor_frame = data['donor_frame_id']
            donor_xy = data['donor_xy']
            write = data['m_write'].astype(bool)
        if final.shape != write.shape or donor_xy.shape != (*write.shape, 2):
            raise RuntimeError('pixel-map shape mismatch')
        if not np.all(final[write] == 5) or not np.all(final[~write] == 0):
            raise RuntimeError('strict UNKNOWN / RAW provenance violated')
        proposed = proposal == 6
        if int(proposed.sum()) != row['proposal_pixels'] or np.any(proposed & ~write):
            raise RuntimeError('proposal count/write-domain mismatch')
        if np.any(proposal[~proposed] != 0) or np.any(donor_frame[~proposed] != -1):
            raise RuntimeError('nonproposal pixels gained donor provenance')
        if row['strict_written_pixels'] != 0 or row['strict_runtime_pass']:
            raise RuntimeError('unproven face was promoted')
        if proposed.any():
            sources = np.unique(donor_frame[proposed]).tolist()
            if sources != [row['source_frame']] or sources[0] >= row['target_frame']:
                raise RuntimeError('future or mismatched donor frame')
            source_entry = objects['frames'][sources[0]]['physical_instances']['0']
            if not source_entry['observed'] or source_entry['physical_instance_id'] != 0:
                raise RuntimeError('donor not a directly observed physical instance')
            mask = cv2.imread(source_entry['mask']['path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError('source mask missing')
            sx = donor_xy[proposed, 0].astype(np.int32)
            sy = donor_xy[proposed, 1].astype(np.int32)
            if np.any(sx < 0) or np.any(sx >= mask.shape[1]) or np.any(sy < 0) or np.any(sy >= mask.shape[0]):
                raise RuntimeError('source xy out of bounds')
            if not np.all(mask[sy, sx] > 0):
                raise RuntimeError('donor xy outside directly observed card mask')
        proposal_pixels += int(proposed.sum())
        unknown_pixels += int(write.sum())
    video = Path(result['outputs']['review_video']['path'])
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,avg_frame_rate,width,height',
        '-of', 'json', str(video),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    if (int(stream['nb_frames']), stream['avg_frame_rate'], int(stream['width']), int(stream['height'])) != (56, '10/1', 2560, 480):
        raise RuntimeError('review-video frame/time/shape mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'],
                   check=True, capture_output=True)
    if proposal_pixels != result['summary']['v4_proposed_pixels'] or unknown_pixels != result['summary']['strict_unknown_pixels']:
        raise RuntimeError('summary counts do not close')
    audit = {
        'schema_version': 'rc1-poker245-atlas-successor-audit-v1',
        'created_at': now_iso(), 'status': 'PASS',
        'session_id': 'play_cards_0903_245', 'frames': 56,
        'proposal_pixels': proposal_pixels, 'strict_unknown_pixels': unknown_pixels,
        'strict_written_pixels': 0, 'video_full_decode': True,
        'result': artifact_ref(args.result), 'auditor': artifact_ref(Path(__file__).resolve()),
        'claim_limit': 'Pixel lineage and fail-closed behavior only; no card-face truth or hidden-pixel accuracy.',
    }
    atomic_json(args.output, audit)
    print(json.dumps({'status': audit['status'], 'frames': 56,
                      'proposal_pixels': proposal_pixels, 'strict_written_pixels': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
