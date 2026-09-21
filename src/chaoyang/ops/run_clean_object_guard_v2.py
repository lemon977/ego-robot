"""Registered CPU adapter: guarded reuse of two frozen exact78 Clean references."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from chaoyang.pipeline.clean_object_guard_v2 import guard_frame, SOURCE_KIND

SESSION_FRAMES = {'get_potato_chips_0902_103': 284, 'play_cards_0902_042': 171}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(path):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


class Reader:
    def __init__(self, mappings, allowed_roots):
        self.mappings = mappings
        self.roots = [Path(p).resolve(strict=True) for p in allowed_roots]
        self.bytes = 0
        self.count = 0

    def path(self, value):
        p = Path(value)
        # Only explicit per-session mappings frozen by the input spec are allowed.
        matches = [(a, b) for a, b in self.mappings.items()
                   if str(p) == a or str(p).startswith(a.rstrip('/') + '/')]
        if matches:
            a, b = max(matches, key=lambda x: len(x[0]))
            p = Path(b) / str(p)[len(a):].lstrip('/')
        p = p.resolve(strict=True)
        if not p.is_file() or not any(p.is_relative_to(r) for r in self.roots):
            raise ValueError(f'READ_SCOPE_ESCAPE: {p}')
        return p

    def read(self, ref):
        p = self.path(ref['path'])
        s = p.stat()
        data = p.read_bytes()
        after = p.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(s, k) != getattr(after, k) for k in fields):
            raise RuntimeError(f'UNSTABLE_READ: {p}')
        if len(data) != ref['bytes'] or digest(data) != ref['sha256']:
            raise ValueError(f'REFERENCE_DRIFT: {p}')
        self.bytes += len(data); self.count += 1
        return data

    def json(self, ref):
        return json.loads(self.read(ref))

    def image(self, ref, mask=False):
        data = cv2.imdecode(np.frombuffer(self.read(ref), np.uint8),
                            cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
        if data is None: raise ValueError('IMAGE_DECODE_FAILED')
        if mask:
            if not set(np.unique(data)).issubset({0, 255}):
                raise ValueError('NONBINARY_MASK')
            return data > 0
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def _array_sha(value):
    return digest(value.dtype.str.encode() + np.asarray(value.shape, '<i8').tobytes()
                  + np.ascontiguousarray(value).tobytes())


def run_session(session, reader, output):
    sid = session['session_id']; expected = SESSION_FRAMES[sid]
    root = output / sid
    root.mkdir(); (root/'candidate_frames').mkdir(); (root/'pixel_sources').mkdir()
    (root/'keyframes').mkdir()
    role_manifest = reader.json(session['expanded_role_manifest'])
    object_manifest = reader.json(session['object_manifest'])
    clean_manifest = reader.json(session['legacy_clean_manifest'])
    if role_manifest.get('schema_version') != 'exact78-expanded-role-clean-frame-manifest-v3':
        raise ValueError('UNKNOWN_2D_ROLE_SCHEMA')
    if object_manifest.get('schema_version') != 'exact78-task-object-identity-batch-manifest-v1':
        raise ValueError('UNKNOWN_2D_OBJECT_SCHEMA')
    for doc in [role_manifest, object_manifest, clean_manifest]:
        if doc.get('session') != sid or len(doc['frames']) != expected:
            raise ValueError('SESSION_OR_FRAME_COUNT_MISMATCH')
    if clean_manifest.get('schema_version') != 'clean-synthetic-pixel-source-manifest-v1':
        raise ValueError('UNKNOWN_LEGACY_SOURCE_SCHEMA')
    legacy_codes = clean_manifest.get('source_kind_codes', {})
    if set(legacy_codes) != {'0', '1', '2', '3'}:
        raise ValueError('UNKNOWN_LEGACY_SOURCE_CODES')
    width, height = 480, 360
    video = root / f'{sid}_RAW_LEGACY_GUARDED_FULL_REVIEW.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '2',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width*3}x{height+44}',
               '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
               '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(video)]
    rows = []; counts = Counter(); timestamps = []; producer = None
    try:
        producer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for index, (roles, objects, clean) in enumerate(zip(role_manifest['frames'],
                object_manifest['frames'], clean_manifest['frames'], strict=True)):
            if any(v.get('source_frame') != index for v in [roles, objects, clean]):
                raise ValueError('FRAME_ORDER_MISMATCH')
            raw = reader.image(roles['source_rgb'])
            legacy = reader.image(clean['clean_rgb'])
            with np.load(io.BytesIO(reader.read(clean['pixel_source_map'])), allow_pickle=False) as z:
                legacy_kind = z['source_kind']
                if int(z['source_frame']) != index: raise ValueError('SOURCE_MAP_FRAME_MISMATCH')
            shape = raw.shape[:2]
            if shape != (960, 1280): raise ValueError('UNEXPECTED_PIXEL_DOMAIN')
            role_union = np.zeros(shape, bool)
            role_refs = roles['role_masks']
            for name in ['left_human', 'right_human', 'left_tracker', 'right_tracker']:
                role_union |= reader.image(role_refs[name], mask=True)
            protect = np.zeros(shape, bool); identities = []
            expected_objects = ['0', '1', '2'] if sid.startswith('get_potato') else ['0']
            physical = objects['physical_instances']
            if sorted(physical) != expected_objects: raise ValueError('OBJECT_SET_MISMATCH')
            for key in expected_objects:
                obj = physical[key]; m = reader.image(obj['mask'], mask=True)
                if int(m.sum()) != obj['area_px']: raise ValueError('MASK_AREA_MISMATCH')
                protect |= m
                identities.append({'observed': obj.get('observed'), 'valid': obj.get('valid'),
                    'physical_instance_id': obj.get('physical_instance_id'),
                    'expected_instance_id': int(key), 'mask_area_px': int(m.sum())})
            result, source, write, stats = guard_frame(raw, legacy, legacy_kind,
                                                       role_union, protect, identities)
            # Acquisition timestamps are explicit input references, never index/fps estimates.
            meta_ref = session['timestamps'][index]
            meta = reader.json(meta_ref)['metadata']
            if meta['idx'] != index or type(meta['ts']) is not int:
                raise ValueError('TIMESTAMP_IDENTITY_INVALID')
            timestamps.append(meta['ts'])
            frame_path = root/'candidate_frames'/f'{index:06d}.png'
            ok = cv2.imwrite(str(frame_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok: raise RuntimeError('CANDIDATE_PNG_WRITE_FAILED')
            map_path = root/'pixel_sources'/f'{index:06d}.npz'
            np.savez_compressed(map_path, source_kind=source, M_write=write,
                protected_object=protect, original_frame_id=np.int32(index),
                acquisition_timestamp_ns=np.int64(meta['ts']))
            panels = []
            titles = ['RAW reference', 'LEGACY Clean (unverified semantics)',
                      'GUARDED: '+('RAW ABSTAIN' if not stats['identity_all_known'] else 'restricted reuse')]
            for img, title in zip([raw, legacy, result], titles, strict=True):
                panel = np.zeros((height+44, width, 3), np.uint8)
                panel[44:] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(panel, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240), 1)
                cv2.putText(panel, f'frame {index} | OFFLINE_VISUAL / NOT_FOR_TRAINING',
                            (8, 36), cv2.FONT_HERSHEY_SIMPLEX, .35, (230,230,230), 1)
                panels.append(panel)
            review = np.concatenate(panels, axis=1)
            producer.stdin.write(review.tobytes())
            if index in {0, expected//2, expected-1}:
                cv2.imwrite(str(root/'keyframes'/f'{index:06d}.png'), cv2.cvtColor(review, cv2.COLOR_RGB2BGR))
                # Exact same pixel domain crops; no fabricated annotation or object labels.
                delta = np.any(legacy != result, axis=-1)
                ys, xs = np.nonzero(delta)
                if len(xs):
                    x0,x1=max(0,int(xs.min())-8),min(1280,int(xs.max())+9)
                    y0,y1=max(0,int(ys.min())-8),min(960,int(ys.max())+9)
                    triplet = np.concatenate([raw[y0:y1,x0:x1],legacy[y0:y1,x0:x1],result[y0:y1,x0:x1]],axis=1)
                    cv2.imwrite(str(root/'keyframes'/f'{index:06d}_changed_region_triplet.png'),cv2.cvtColor(triplet,cv2.COLOR_RGB2BGR))
            rows.append({'frame_id': index, 'acquisition_timestamp_ns': meta['ts'],
                'display_time_s': index/30., 'source_video_time_s': meta.get('video_time_s'),
                'object_provider': objects.get('provider'), 'identities': identities,
                'source_rgb': roles['source_rgb'], 'timestamp_source': meta_ref,
                'candidate_rgb': reference(frame_path), 'source_map': reference(map_path),
                'raw_array_sha256': _array_sha(raw), 'candidate_array_sha256': _array_sha(result), **stats})
            counts[stats['status']] += 1
            for key in ['legacy_write_pixels','candidate_write_pixels','withheld_legacy_write_pixels',
                        'outside_write_changed_pixels','protected_changed_pixels']:
                counts[key] += stats[key]
        producer.stdin.close()
        errors = producer.stderr.read().decode(errors='replace')
        if producer.wait(timeout=120): raise RuntimeError('ENCODER_FAILED: '+errors[-2000:])
    finally:
        if producer is not None and producer.poll() is None:
            producer.terminate(); producer.wait(timeout=15)
    decode = subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-threads','2',
        '-i',str(video),'-f','null','-'],capture_output=True,text=True,timeout=180)
    probe = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate','-of','json',str(video)],
        capture_output=True,text=True,check=True,timeout=180)
    probe_json = json.loads(probe.stdout)
    if decode.returncode or int(probe_json['streams'][0]['nb_read_frames']) != expected:
        raise RuntimeError('FULL_DECODE_OR_FRAME_COUNT_FAILED')
    ts = np.asarray(timestamps, np.int64)
    write_json(root/'FRAME_MANIFEST.json', {'schema_version':'clean-object-guard-frames-v2',
        'session_id':sid, 'source_kind_codes':SOURCE_KIND, 'frames':rows})
    result = {'session_id':sid, 'status':'ENGINEERING_GUARD_EXECUTED_VISUAL_QUALITY_UNPROVEN',
        'frame_count':expected, 'counts':dict(counts), 'full_decode':True,
        'video':reference(video),'ffprobe':probe_json,
        'frame_manifest':reference(root/'FRAME_MANIFEST.json'),
        'timestamps_strictly_increasing':bool(np.all(np.diff(ts)>0)),
        'nonuniform_capture_time_preserved_in_manifest':True,
        'video_time_policy':'CFR_30_DISPLAY_ONLY_NOT_ACQUISITION_CLOCK',
        'new_inpainting_performed':False,'legacy_donor_revalidated':False,
        'visual_quality_adopted':False,'numeric_quality_pass':False,
        'training_eligible':False,'control_ground_truth':False,
        'depth_or_object6d_consumed':False,
        'limitation':'Unknown object identity restores raw including human/Tracker; known frames inherit unvalidated legacy Clean semantics. Not a new reconstruction or contact claim.'}
    write_json(root/'RESULT.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    cv2.setNumThreads(2)
    spec_ref = reference(args.spec); spec = json.loads(args.spec.read_text())
    if spec.get('schema_version') != 'clean-object-guard-input-v2':
        raise ValueError('UNKNOWN_INPUT_SCHEMA')
    if {x['session_id'] for x in spec['sessions']} != set(SESSION_FRAMES) or len(spec['sessions']) != 2:
        raise ValueError('FIXED_COHORT_MISMATCH')
    root = args.output.resolve()
    allowed = Path(spec['allowed_output_root']).resolve(strict=True)
    if root == allowed or not root.is_relative_to(allowed): raise ValueError('OUTPUT_SCOPE_ESCAPE')
    root.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    reader = Reader(spec['explicit_path_mappings'],spec['allowed_read_roots'])
    try:
        sessions = [run_session(s,reader,root) for s in spec['sessions']]
        result = {'schema_version':'clean-object-guard-result-v2',
            'created_at':datetime.now(timezone.utc).isoformat(),
            'status':'ENGINEERING_GUARD_EXECUTED_VISUAL_QUALITY_UNPROVEN',
            'spec':spec_ref,'sessions':sessions,'GPU_USED':False,'SOURCE_DATA_MODIFIED':False,
            'read_file_count':reader.count,'read_verified_bytes':reader.bytes,
            'wall_seconds':time.monotonic()-started,'candidate_count':1,
            'PIPELINE_COMPLETE':True,'NUMERIC_QUALITY_PASS':False,
            'TRAINING_ELIGIBLE':False,'CONTROL_AUTHORIZED':False,
            'code': [reference(Path(__file__)), reference(Path(__file__).parents[1]/'pipeline/clean_object_guard_v2.py')],
            'adoption':'ENGINEERING_SAFETY_GUARD_ONLY_NOT_CLEAN_VISUAL_BASELINE'}
        write_json(root/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'result':str(root/'RESULT.json')}))
        return 0
    except Exception as exc:
        write_json(root/'FAILURE.json',{'status':'FAILED_RUNTIME','error':str(exc),
            'type':type(exc).__name__,'spec':spec_ref,'partial_not_successor':True})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
