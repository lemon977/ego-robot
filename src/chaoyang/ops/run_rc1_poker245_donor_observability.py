#!/usr/bin/env python3
"""Factorized, fail-closed observability replay for the frozen Poker245 event.

This reads immutable V4/strict receipts. It never writes donor pixels or infers
physical-card identity from matching purple backs. Videos are review only.
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402


EVENTS = {
    'PARTIAL_OCCLUSION': list(range(34, 43)),
    'AMBIGUOUS_SAME_BACK': list(range(45, 61)),
}


def evaluate_observability(
    *, source_observed: bool, source_precedes_target: bool,
    identity_proof: bool, face_proof: bool, geometry_pass: bool,
) -> dict:
    """Four independent requirements; runtime cannot substitute appearance for ID."""
    checks = {
        'A_PAST_SURFACE_DIRECTLY_OBSERVED': bool(source_observed and source_precedes_target),
        'B_SAME_PHYSICAL_CARD_INDEPENDENTLY_PROVEN': bool(identity_proof),
        'C_SAME_FACE_INDEPENDENTLY_PROVEN': bool(face_proof),
        'D_CURRENT_GEOMETRY_OBSERVED': bool(geometry_pass),
    }
    return {
        'checks': checks,
        'accept_donor_pixels': all(checks.values()),
        'missing': [name for name, passed in checks.items() if not passed],
    }


def _verify(reference: dict) -> Path:
    path = Path(reference['path'])
    actual = artifact_ref(path)
    if actual['bytes'] != reference['bytes'] or actual['sha256'] != reference['sha256']:
        raise RuntimeError(f'artifact closure mismatch: {path}')
    return path


def _write_event_video(source: Path, output: Path, frames: list[int], decisions: dict[int, dict]) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'cannot decode review source: {source}')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != 2560 or height != 480:
        raise RuntimeError(f'original review layout changed: {width}x{height}')
    proc = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height + 90}', '-r', '6', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output),
    ], stdin=subprocess.PIPE)
    wanted = set(frames)
    seen = set()
    for offset in range(56):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'old 56-frame review decode stopped at {offset}')
        target = offset + 34
        if target not in wanted:
            continue
        seen.add(target)
        decision = decisions[target]
        checks = decision['checks']
        board = np.zeros((height + 90, width, 3), np.uint8)
        board[90:] = frame
        line1 = f"play_cards_0903_245 frame {target} | donor accepted: NO | UNKNOWN retained"
        line2 = 'A past observed={}  B same card={}  C same face={}  D geometry={}'.format(
            *('YES' if value else 'NO' for value in checks.values()))
        cv2.putText(board, line1, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        cv2.putText(board, line2, (18, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2)
        for _ in range(4):
            proc.stdin.write(board.tobytes())
    if seen != wanted:
        raise RuntimeError(f'event frame mismatch: {sorted(wanted - seen)}')
    if cap.read()[0]:
        raise RuntimeError('old review has extra frames')
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed: {output}')
    check = subprocess.run([
        'ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'
    ], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f'event video decode failed: {output}: {check.stderr[-500:]}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f'fresh output required: {args.output_root}')
    strict = json.loads(args.strict_result.read_text(encoding='utf-8'))
    if strict.get('session_id') != 'play_cards_0903_245' or strict['summary']['frames'] != 56:
        raise RuntimeError('frozen session/event identity mismatch')
    decisions_path = _verify(strict['outputs']['frame_decisions'])
    review_path = _verify(strict['outputs']['review_video'])
    object_path = _verify(strict['inputs']['object_manifest'])
    _verify(strict['inputs']['raw_video'])
    decisions = json.loads(decisions_path.read_text(encoding='utf-8'))['rows']
    objects = json.loads(object_path.read_text(encoding='utf-8'))
    if [r['target_frame'] for r in decisions] != list(range(34, 90)):
        raise RuntimeError('event frame sequence changed')
    args.output_root.mkdir(parents=True)
    rows = []
    stats = Counter()
    for old in decisions:
        target = old['target_frame']
        source = old['source_frame']
        source_object = objects['frames'][source]['physical_instances']['0']
        # Object authority becomes invalid at frame 34. V4 optical flow and
        # purple-back appearance are not independent identity/face proof.
        source_seen = bool(source_object['observed'] and source_object['valid']
                           and source_object['physical_instance_id'] == 0)
        current_object = objects['frames'][target]['physical_instances']['0']
        independently_identified = bool(current_object['observed'] and current_object['valid']
                                        and current_object['physical_instance_id'] == 0)
        # The current object manifest has no face-ID field. Never equate its
        # physical_instance_id or a same-looking back with face continuity.
        face_proof = False
        geometry_pass = bool(old['v4_runtime_pass'] and old['homography_source_to_target'])
        assessment = evaluate_observability(
            source_observed=source_seen, source_precedes_target=source < target,
            identity_proof=independently_identified, face_proof=face_proof,
            geometry_pass=geometry_pass,
        )
        if assessment['accept_donor_pixels']:
            raise RuntimeError('unexpected donor promotion: independent face proof absent')
        pixel_map = _verify(old['pixel_map'])
        with np.load(pixel_map) as data:
            proposed = int(np.count_nonzero(data['proposal_source']))
            if proposed != old['proposal_pixels']:
                raise RuntimeError(f'proposal source map mismatch at frame {target}')
            if proposed and not np.all(data['donor_frame_id'][data['proposal_source'] != 0] < target):
                raise RuntimeError(f'future donor at frame {target}')
        row = {
            'session_id': 'play_cards_0903_245', 'target_frame': target,
            'source_frame': source, 'proposal_pixels': proposed,
            'strict_written_pixels': 0, 'pixel_source_map': old['pixel_map'],
            'source_object_direct_observed': source_seen,
            'target_object_direct_observed': independently_identified,
            'target_face_id': 'UNKNOWN', 'source_face_appearance': old['source_face_appearance'],
            'geometry_evidence': 'V4_RUNTIME_PASS' if geometry_pass else 'V4_RUNTIME_REJECTED',
            **assessment,
        }
        rows.append(row)
        stats.update(assessment['missing'])
    decision_path = args.output_root / 'OBSERVABILITY_DECISIONS.json'
    atomic_json(decision_path, {'session_id': 'play_cards_0903_245',
                                'input_mode': 'OFFLINE_VISUAL_DEVELOPMENT', 'rows': rows})
    lookup = {r['target_frame']: r for r in rows}
    videos = {}
    for name, frames in EVENTS.items():
        output = args.output_root / f'Poker245_{name}_同时间轴证据门对照.mp4'
        _write_event_video(review_path, output, frames, lookup)
        videos[name] = artifact_ref(output)
    result = {
        'schema_version': 'rc1-poker245-donor-observability-v1',
        'created_at': now_iso(), 'session_id': 'play_cards_0903_245',
        'status': 'FAILED_QUALITY_C', 'authority_promoted': False,
        'input_mode': 'OFFLINE_VISUAL_DEVELOPMENT', 'training_eligible': False,
        'input_closure': {'strict_result': artifact_ref(args.strict_result),
                          'code': artifact_ref(Path(__file__)),
                          'object_manifest': strict['inputs']['object_manifest'],
                          'raw_video': strict['inputs']['raw_video']},
        'events': {name: {'frames': frames, 'video': videos[name]} for name, frames in EVENTS.items()},
        'outputs': {'decisions': artifact_ref(decision_path)},
        'summary': {'frames': len(rows), 'v4_candidate_frames': sum(r['proposal_pixels'] > 0 for r in rows),
                    'proposed_pixels': sum(r['proposal_pixels'] for r in rows),
                    'strict_written_pixels': 0,
                    'missing_gate_frame_counts': dict(stats),
                    'minimum_missing_evidence': 'Independent physical-card continuity and face identity after Object Mask becomes invalid at frame 34.'},
        'claim_limit': 'A-D runtime observability diagnosis and reviewed UNKNOWN; no hidden-pixel accuracy or Clean improvement claim.',
    }
    atomic_json(args.output_root / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'summary': result['summary'],
                      'result': str(args.output_root / 'RESULT.json')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
