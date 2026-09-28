"""Bounded CPU diagnostic; no retargeting, model, GPU, or quality adoption."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from chaoyang.pipeline.shared_local_hand_target_v1 import source_provenance_masks

PARENT = 'four_stream_completion_20260928'
TASK = PARENT + '_sensor'


def checked_path(value, root):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if '..' in path.parts or not path.is_relative_to(root):
        raise ValueError('reference outside project')
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError('symlink path component')
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError('reference outside project')
    return resolved


def read_pinned(ref, root):
    path = checked_path(ref['path'], root)
    before = path.stat()
    if before.st_size > 32 * 1024 * 1024:
        raise ValueError('diagnostic reference exceeds 32 MiB')
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError('unstable input')
    if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('reference bytes/SHA mismatch')
    return raw


def source_summary(source):
    valid = np.asarray(source['manus_hand_valid'])
    # HAND_MOTION explicitly excludes virtual Hand_Invalid from direct evidence.
    joint_direct = source.get('joint_observed_local_25')
    if joint_direct is not None:
        joint_direct = np.asarray(joint_direct)
        if joint_direct.dtype != np.bool_ or joint_direct.shape != (*valid.shape,25):
            raise ValueError('direct joint observation must be boolean (T,2,25)')
    direct = None if joint_direct is None else np.asarray(joint_direct)[:, :, 1:].all(axis=-1).T
    valid, direct = source_provenance_masks(valid.T, direct)
    points = np.asarray(source['manus_local_25_m'])
    names = source['manus25_joint_names'].tolist()
    if points.shape != (valid.shape[1], 2, 25, 3) or len(names) != 25:
        raise ValueError('MANUS25 shape mismatch')
    edges = [(i, i + 1) for a, b in [(1, 4), (5, 9), (10, 14), (15, 19), (20, 24)] for i in range(a, b)]
    rows = []
    for side in range(2):
        for a, b in edges:
            length = np.linalg.norm(points[:, side, b] - points[:, side, a], axis=-1)
            values = length[valid[side] & np.isfinite(length)]
            rows.append(dict(side=['left','right'][side], start=names[a], end=names[b],
                             count=int(values.size), median_m=float(np.median(values)) if values.size else None))
    return dict(source_valid_side_frames=int(valid.sum()), source_observed_side_frames=int(direct.sum()),
                source_valid_not_directly_observed=int((valid & ~direct).sum()), bone_lengths=rows,
                semantics='SOURCE_RECORDED_NAMES_NOT_INDEPENDENT_ANATOMICAL_VERIFICATION')


def collision_pairs(models, states, frames):
    import pybullet as p
    rows = []
    for side, model in enumerate(models):
        client = p.connect(p.DIRECT)
        if client < 0:
            raise RuntimeError('CPU collision client unavailable')
        try:
            body = p.loadURDF(str(model.path), useFixedBase=True,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                              physicsClientId=client)
            info = [p.getJointInfo(body, i, physicsClientId=client) for i in range(p.getNumJoints(body, physicsClientId=client))]
            moving = [j for j in model.joints if j.joint_type != 'fixed']
            expected = [j.name for j in moving]
            if expected != states['joint_names'][side].tolist():
                raise ValueError('frozen q joint order mismatch')
            indices = {x[1].decode(): i for i, x in enumerate(info) if x[3] >= 0}
            links = {-1: p.getBodyInfo(body, physicsClientId=client)[0].decode()}
            links.update({i:x[12].decode() for i,x in enumerate(info)})
            cases = [('mid_limits', np.array([(j.lower+j.upper)*.5 for j in moving]))]
            cases += [(f'frozen_frame_{frame}', states['q22'][frame,side]) for frame in frames]
            for label, q in cases:
                if q.shape != (22,) or not np.isfinite(q).all():
                    raise ValueError('nonfinite frozen q')
                for name, value in zip(expected, q, strict=True):
                    p.resetJointState(body, indices[name], float(value), physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                pairs = {}
                for point in p.getContactPoints(body, body, physicsClientId=client):
                    if point[8] >= 0:
                        continue
                    key = tuple(sorted((links[point[3]], links[point[4]])))
                    pairs[key] = max(pairs.get(key,0.), -float(point[8]))
                rows.append(dict(side=['left','right'][side], case=label,
                                 pairs=[dict(links=list(k), penetration_m=v) for k,v in sorted(pairs.items())]))
        finally:
            p.disconnect(client)
    return rows


def run(config):
    root = Path(config['repo_root']).resolve(strict=True)
    packet = json.loads(read_pinned(config['packet'], root))
    if packet['task_id'] != TASK:
        raise ValueError('wrong sensor task')
    parent = json.loads(read_pinned(config['parent_packet'], root))
    if parent['task_id'] != PARENT or parent['writer'] != packet['writer']:
        raise ValueError('parent and delegated writer mismatch')
    index = json.loads((root/'tasks/current/INDEX.json').read_text())
    entries = {x['task_id']:x for x in index['task_packets']}
    if not entries[PARENT]['execution_allowed'] or entries[TASK]['packet_sha256'] != config['packet']['sha256']:
        raise ValueError('parent routing or delegated packet changed')
    if entries[PARENT]['packet_sha256'] != config['parent_packet']['sha256']:
        raise ValueError('parent packet SHA drift')
    for task, key in ((PARENT, 'parent_packet'), (TASK, 'packet')):
        if checked_path(entries[task]['packet_path'],root) != checked_path(config[key]['path'],root):
            raise ValueError('packet path differs from routable index')
    writer = packet['writer']
    ticks = Path(f"/proc/{writer['pid']}/stat").read_text().split(') ',1)[1].split()[19]
    if str(ticks) != str(writer['proc_start_ticks']):
        raise ValueError('publisher identity changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA_VISIBLE_DEVICES must be empty')
    output_root = checked_path(packet['output_root'], root)
    expected_output = root/'_run/current'/PARENT/'attempts/attempt_0001/lanes/sensor'
    if output_root != expected_output:
        raise ValueError('output must be exact sensor lane')
    allowed = [checked_path(p,root) for p in packet['write_set']]
    if output_root not in allowed:
        raise ValueError('sensor lane missing from delegated write-set')
    kind=config.get('diagnostic_kind','source_collision_pairs_v1')
    if kind not in ('source_collision_pairs_v1','asset_geometry_v1','triangle_pairs_v1',
                    'triangle_pairs_common_margin_v1','compound_pairs_v1'):
        raise ValueError('unsupported diagnostic kind')
    names={'source_collision_pairs_v1':'DIAGNOSTIC_V1.json','asset_geometry_v1':'ASSET_GEOMETRY_V1.json','triangle_pairs_v1':'TRIANGLE_PAIRS_V1.json'}
    names['triangle_pairs_common_margin_v1']='TRIANGLE_PAIRS_COMMON_MARGIN_V1.json'
    names['compound_pairs_v1']='COMPOUND_PAIRS_V1.json'
    output = output_root/names[kind]
    if output.exists():
        raise FileExistsError(output)
    config_frames = config['frame_indices']
    if not 1 <= len(config_frames) <= 8 or any(type(f) is not int or f < 0 for f in config_frames):
        raise ValueError('one to eight explicit frame indices required')
    def npz(ref):
        with np.load(io.BytesIO(read_pinned(ref,root)),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    source, states = npz(config['source']), npz(config['states'])
    if str(source['session_id']) != 'play_cards_0916_097':
        raise ValueError('diagnostic restricted to 097')
    if max(config_frames) >= len(states['q22']):
        raise ValueError('frame out of bounds')
    np.testing.assert_array_equal(source['frame_id'], states['source_frame_id'])
    np.testing.assert_array_equal(source['anatomical_side_names'], ['left','right'])
    np.testing.assert_array_equal(states['physical_robot_side_names'], ['left','right'])
    np.testing.assert_array_equal(states['anatomical_side_names'], ['left','right'])
    if states['q22'].shape != (165,2,22) or states['joint_names'].shape != (2,22):
        raise ValueError('frozen states must use (T,physical_side,joint) axes')
    if not np.array_equal(states['source_frame_id'],np.arange(165)):
        raise ValueError('frozen 097 timeline changed')
    from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import load_pinned_kaihand_models
    read_pinned(config['asset_pin'], root)
    if Path(config['asset_pin']['path']).resolve() != (root/'assets/robot/ROBOT_ASSET_PIN.json').resolve():
        raise ValueError('wrong asset pin')
    models, asset_refs = load_pinned_kaihand_models(root)
    if kind=='asset_geometry_v1':
        from chaoyang.pipeline.sensor_collision_asset_audit_v1 import audit_assets
        diagnostic=dict(asset_geometry=audit_assets(models,root,json.loads(read_pinned(config['asset_pin'],root))))
    elif kind in ('triangle_pairs_v1','triangle_pairs_common_margin_v1'):
        from chaoyang.pipeline.sensor_collision_pair_probe_v1 import fixed_fk_pair_probe
        diagnostic=dict(triangle_comparison=fixed_fk_pair_probe(models,root,json.loads(read_pinned(config['asset_pin'],root)),same_margin=kind=='triangle_pairs_common_margin_v1',control_root=output_root/'TRIANGLE_COMMON_MARGIN_CONTROLS_V1'))
    elif kind=='compound_pairs_v1':
        from chaoyang.pipeline.sensor_collision_pair_probe_v1 import fixed_fk_compound_probe
        diagnostic=dict(compound_comparison=fixed_fk_compound_probe(
            models,root,json.loads(read_pinned(config['asset_pin'],root)),
            control_root=output_root/'COMPOUND_PAIRS_CONTROLS_V1',
            derived_root=output_root/'COMPOUND_PAIRS_DERIVED_V1'))
    else:
        diagnostic=dict(collision_pairs=collision_pairs(models,states,config_frames))
    result = dict(schema_version='FOUR_STREAM_SENSOR_DIAGNOSTIC_V1', task_id=TASK,
                  execution='EXECUTED', quality='NOT_EVALUATED', adoption='NOT_ADOPTED',
                  source=source_summary(source), diagnostic_kind=kind, **diagnostic,
                  inputs=config, assets=asset_refs, gpu_used=False, optimizer_executed=False,
                  limitations=['Collision pair diagnosis, not checker calibration or physical collision proof.',
                               'Source naming audit does not certify anatomical correspondences.'])
    # Recheck routing, owner and all input pins before immutable publication.
    if json.loads((root/'tasks/current/INDEX.json').read_text())['task_packets'] != index['task_packets']:
        raise ValueError('routing changed during diagnostic')
    for key in ('packet','parent_packet','source','states','asset_pin'):
        read_pinned(config[key],root)
    if Path(f"/proc/{writer['pid']}/stat").read_text().split(') ',1)[1].split()[19] != ticks:
        raise ValueError('publisher identity changed during diagnostic')
    with output.open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args = parser.parse_args(argv)
    result = run(json.loads(args.config.read_text(encoding='utf8')))
    print(json.dumps(dict(execution=result['execution'],quality=result['quality'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
