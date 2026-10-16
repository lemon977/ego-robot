#!/usr/bin/env python
"""Summary JSON of a takeover full baseline run (per-session metrics aggregated); stdout.

Usage: /usr/local/bin/python -B scripts/takeover/baseline_summary.py <task_id>
"""
import os
os.chdir('/path/to/chaoyang')
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
task = sys.argv[1]
cand = Path(f'_run/current/{task}/attempts/attempt_0001/candidate')
progress = json.loads((cand / 'PROGRESS.json').read_text())
rows = []
for row in progress['sessions']:
    folder = Path(row['folder']) if row.get('folder') else cand / 'sessions' / row['session']
    path = folder / 'SESSION.json'
    if row['status'] != 'OK' or not path.is_file():
        continue
    s = json.loads(path.read_text())
    masks = json.loads((folder / 'masks.json').read_text())
    sides = s['motion']['metrics']['per_side']
    hs = s['motion']['hand_source_counts']
    prov = masks['provider_counts']
    n = s['frame_count']
    fore = [v for k, v in (masks['human'].get('sam_forearm_hit_rate') or {}).items()
            if v is not None and masks['human']['sam_forearm_probe_points'][k] >= 40]
    rows.append({
        'session': s['session'], 'task': s['task'], 'day': '0901' if '_0901_' in s['session'] else '0902',
        'frames': n, 'fps': s['fps'],
        'observed': {k: hs[k].get('OBSERVED', 0) / n for k in ('left', 'right')},
        'filled': {k: n - hs[k].get('OBSERVED', 0) - hs[k].get('NONE', 0) for k in ('left', 'right')},
        'none': {k: hs[k].get('NONE', 0) for k in ('left', 'right')},
        'arm_pass': min(sides[k]['tracking_pass_rate_observed'] for k in ('left', 'right')),
        'pos_p95_mm': max(sides[k]['position_mm_p95'] for k in ('left', 'right')),
        'bone_med': max(sides[k]['hand_bone_deg_median'] for k in ('left', 'right')),
        'fallback': sum(c for k in ('left', 'right') for name, c in prov[k].items()
                        if name.startswith('PICO_HULL_FALLBACK')) / (2 * n),
        'registered': sum(prov[k].get('PICO_HULL_FALLBACK_SAM_REGISTERED', 0) + prov[k].get('PICO_HULL_FALLBACK_SHIFT_CARRIED', 0)
                          for k in ('left', 'right')) / (2 * n),
        'forearm': min(fore) if fore else 1.0,
        'removal': masks['removal_fraction_mean'],
        'held_frames': (masks['objects'] or {}).get('held_frames'),
        'folder': str(folder.resolve()),
        'bracket_code': 'removal_bracket_frames' in masks['human'],
        'forearm_retry': masks['human'].get('forearm_retry') or [],
        'uncovered': s['render']['removal_uncovered_fraction'],
        'gpu_s': sum(v for k, v in s['masks']['timings'].items()),
        'render_s': s['render_seconds'],
        'final_bytes': s['outputs']['final']['bytes'], 'review_bytes': s['outputs']['review']['bytes'],
    })
def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {'min': round(values[0], 4), 'median': round(statistics.median(values), 4),
            'p90': round(values[int(0.9 * (len(values) - 1))], 4), 'max': round(values[-1], 4)}
stage = json.loads((cand / 'RESULT.json').read_text()) if (cand / 'RESULT.json').is_file() else {}
out = {'task': task, 'progress_counts': progress['counts'], 'updated_at': progress['updated_at'],
       'has_index': (cand / 'index.html').is_file(),
       'stage_status': stage.get('status'), 'recomputed': stage.get('recomputed', []),
       'sessions_with_bracket_code': sum(r['bracket_code'] for r in rows),
       'forearm_retries': {r['session']: r['forearm_retry'] for r in rows if r['forearm_retry']},
       'ok_sessions': len(rows), 'frames': sum(r['frames'] for r in rows),
       'by_group': {f'{d}/{t}': {'sessions': sum(1 for r in rows if r['day'] == d and r['task'] == t),
                                 'frames': sum(r['frames'] for r in rows if r['day'] == d and r['task'] == t)}
                    for d in ('0901', '0902') for t in ('playing_cards', 'potato_chips')},
       'fps': dict(Counter(r['fps'] for r in rows)),
       'observed_left': stats([r['observed']['left'] for r in rows]),
       'observed_right': stats([r['observed']['right'] for r in rows]),
       'unobserved_fraction_by_day': {d: {k: round(1 - sum(r['observed'][k] * r['frames'] for r in rows if r['day'] == d)
                                                  / max(1, sum(r['frames'] for r in rows if r['day'] == d)), 4)
                                          for k in ('left', 'right')} for d in ('0901', '0902')},
       'filled_frames_total': {k: sum(r['filled'][k] for r in rows) for k in ('left', 'right')},
       'none_frames_total': {k: sum(r['none'][k] for r in rows) for k in ('left', 'right')},
       'arm_pass_min_side': stats([r['arm_pass'] for r in rows]),
       'pos_p95_mm': stats([r['pos_p95_mm'] for r in rows]),
       'hand_bone_median_deg': stats([r['bone_med'] for r in rows]),
       'mask_fallback_fraction': stats([r['fallback'] for r in rows]),
       'mask_fallback_registered_fraction': stats([r['registered'] for r in rows]),
       'sam_forearm_hit_rate_min_side': stats([r['forearm'] for r in rows]),
       'folders_outside_candidate': sum(1 for r in rows if not r['folder'].startswith(str(cand.resolve()) + '/')),
       'removal_fraction': stats([r['removal'] for r in rows]),
       'removal_uncovered_fraction': stats([r['uncovered'] for r in rows]),
       'gpu_seconds_total': round(sum(r['gpu_s'] for r in rows)), 'render_seconds_total': round(sum(r['render_s'] for r in rows)),
       'video_bytes_total': sum(r['final_bytes'] + r['review_bytes'] for r in rows),
       'worst': {key: [(r['session'], round(r[key] if not isinstance(r[key], dict) else min(r[key].values()), 4))
                       for r in sorted(rows, key=lambda r: (r[key] if not isinstance(r[key], dict) else min(r[key].values())), reverse=rev)[:6]]
                 for key, rev in (('fallback', True), ('uncovered', True), ('arm_pass', False), ('observed', False), ('bone_med', True),
                                  ('forearm', False))},
       'failed': {r['session']: (r.get('error') or '')[-300:] for r in progress['sessions'] if r['status'].startswith('FAILED')}}
print(json.dumps(out, ensure_ascii=False, indent=1))
