"""Finite HuRo-only palm-frame continuation. No models, training or GPU."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from chaoyang.pipeline.huro_hand_frame_v2 import METHOD_ID, solve_aligned_sequence, evaluate_q, temporal_metrics
from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import (
    load_pinned_kaihand_models, PyBulletNonAdjacentSelfCollisionChecker,
)
from chaoyang.ops.run_huro_derived_hand_only_v1 import evidence, load_local_q22_comparison
from chaoyang.ops.render_huro_hand_only_common_review_v1 import render


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_ref(ref):
    p = Path(ref['path'])
    if p.is_symlink() or not p.is_file():
        raise RuntimeError(f'ordinary reference required: {p}')
    before = p.stat()
    raw = p.read_bytes()
    after = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f'unstable reference: {p}')
    if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise RuntimeError(f'reference mismatch: {p}')
    return raw


def npz_ref(ref):
    import io
    with np.load(io.BytesIO(read_ref(ref)), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class ExplicitSelfCollision(PyBulletNonAdjacentSelfCollisionChecker):
    """Same geometry and check for all methods; explicitly enable self collision."""
    def __init__(self, models):
        import pybullet as bullet
        self._bullet = bullet
        self._tolerance = 0.
        self._client = bullet.connect(bullet.DIRECT)
        self._bodies, self._joint_indices = [], []
        if self._client < 0:
            raise RuntimeError('CPU collision connection failed')
        try:
            for model in models:
                body = bullet.loadURDF(str(model.path), useFixedBase=True,
                    flags=bullet.URDF_USE_SELF_COLLISION | bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                    physicsClientId=self._client)
                expected = tuple(j.name for j in model.joints if j.joint_type != 'fixed')
                found = {}
                for index in range(bullet.getNumJoints(body,physicsClientId=self._client)):
                    info = bullet.getJointInfo(body,index,physicsClientId=self._client)
                    if info[3] >= 0: found[info[1].decode()] = index
                if set(found) != set(expected): raise RuntimeError('collision joint order mismatch')
                self._bodies.append(body)
                self._joint_indices.append(tuple(found[name] for name in expected))
        except Exception:
            self.close()
            raise


def stats(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return dict(n=int(finite.size), p50=float(np.median(finite)) if finite.size else None,
                p95=float(np.percentile(finite,95)) if finite.size else None,
                max=float(np.max(finite)) if finite.size else None)


def verify_fence(config):
    state = json.loads(Path(config['lane_state']).read_text())
    expected = config['writer']
    if state['writer'] != expected:
        raise RuntimeError('HuRo writer fencing changed; stop without touching other lanes')
    pid = expected['pid']
    actual = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    if str(actual) != str(expected['proc_start_ticks']):
        raise RuntimeError('HuRo owner identity changed')


def run_case(config, item, root, hands, collision):
    verify_fence(config)
    old_result = json.loads(read_ref(item['previous_result']))
    raw = npz_ref(old_result['inputs']['hawor'])
    old = npz_ref(old_result['outputs']['states'])
    local = npz_ref(old_result['comparison']['local_r0'])
    clock = npz_ref(item['clock_source'])
    frames = raw['original_frame_indices']
    if not np.array_equal(frames, old['source_frame_id']) or not np.array_equal(frames,clock['frame_id']):
        raise RuntimeError('source frame axis mismatch')
    if not np.array_equal(raw['anatomical_side_names'], ['left','right']):
        raise RuntimeError(f"unregistered anatomical axis: {raw['anatomical_side_names']}")
    if 'human_to_physical' in local and not np.array_equal(local['human_to_physical'],[1,0]):
        raise RuntimeError('local side mapping mismatch')
    ns = np.asarray(clock['timestamp_ns'])
    if ns.shape != frames.shape or not np.issubdtype(ns.dtype, np.integer):
        raise RuntimeError('integer capture timestamp axis required')
    times = (ns - ns[0]).astype(np.float64) * 1e-9
    if np.any(np.diff(times)<=0): raise RuntimeError('invalid capture time')
    case = root / item['name']
    case.mkdir(exist_ok=False)
    def progress(t, n):
        verify_fence(config)
        if time.monotonic() - STARTED > config['max_wall_seconds']:
            raise RuntimeError('finite wall budget reached; no auto retry')
        print(json.dumps(dict(session=item['name'],solved=t,total=n)),flush=True)
    states = solve_aligned_sequence(hands,raw['joints_3d_camera'],raw['observed'],frames,
                                    max_evaluations=config['max_evaluations'],progress=progress)
    states['timestamp_s'] = times
    states['timestamp_ns'] = ns
    states['clip_delta'] = np.where(states['valid'][...,None],np.zeros_like(states['q22']),np.nan)
    states['postprocessing'] = np.asarray('NONE_RAW_BOUNDED_SOLVER')
    states['temporal_authority'] = np.asarray('FROZEN_CAPTURE_TIMESTAMP_NS_NOT_FRAME_OVER_FPS')
    local_q, local_valid, qkey, vkey = load_local_q22_comparison(local,states['q22'].shape)
    sources = dict(local=(local_q,local_valid),huro_v1=(old['q22'],old['valid']),huro_v2=(states['q22'],states['valid']))
    common = states['target_valid'] & local_valid & old['valid'] & states['valid']
    metrics, arrays = {}, {'common_all_three':common}
    for name,(q,valid) in sources.items():
        evaluation = evaluate_q(hands,q,valid,states['target21_robot_frame'],states['target_valid'],collision)
        temporal = temporal_metrics(q,valid,times,frames)
        for key,value in {**evaluation,**temporal}.items(): arrays[name+'__'+key]=value
        observed = states['source_observed_physical']
        metrics[name] = dict(valid_side_frames=int(valid.sum()),
            observed_side_frames=int(observed.sum()), observed_without_output=int((observed & ~valid).sum()),
            output_without_source_observation=int((valid & ~observed).sum()),
            total_timeline_side_frames=int(valid.size),
            per_physical_side_valid=valid.sum(axis=0).tolist(),
            tip_rms_mm_all_own_valid=stats(evaluation['tip_rms_mm']),
            tip_rms_mm_common_all_three=stats(evaluation['tip_rms_mm'][common]),
            direction_rms_common_all_three=stats(evaluation['direction_rms'][common]),
            hard_limit_violation_side_frames=int((evaluation['limit_violation_rad']>1e-12).sum()),
            nonadjacent_collision_side_frames=int((evaluation['collision_count']>0).sum()),
            collision_evaluated_side_frames=int(evaluation['collision_known'].sum()),
            max_penetration_m=stats(evaluation['penetration_m']),
            abs_velocity_rad_s=stats(np.abs(temporal['velocity_rad_s'])),
            abs_acceleration_rad_s2=stats(np.abs(temporal['acceleration_rad_s2'])),
            abs_jerk_rad_s3=stats(np.abs(temporal['jerk_rad_s3'])),
            posthoc_clip=('NONE_RAW_BOUNDED_SOLVER' if name=='huro_v2' else
                          'FROZEN_CLIP_DELTA_AVAILABLE' if name=='local' and 'clip_delta' in local else 'NOT_REVERIFIED'))
    state_path = case/'HURO_PALM_FRAME_V2_STATES.npz'
    np.savez_compressed(state_path,**states)
    np.savez_compressed(case/'COMMON_EVALUATION_ARRAYS.npz',**arrays)
    comparison = dict(status='COMMON_TARGET_INTERNAL_DIAGNOSTIC_NOT_EXTERNAL_ACCURACY',
        local_r0=old_result['comparison']['local_r0'],local_q22_field=qkey,local_valid_field=vkey,
        common_side_frames=int((local_valid & states['valid']).sum()))
    result = {**old_result,'method_id':METHOD_ID,'outputs':{'states':evidence(state_path)},
        'comparison':comparison,'solver_metrics':metrics['huro_v2'],
        'valid_side_frames':int(states['valid'].sum()),
        'physical_left_valid':int(states['valid'][:,0].sum()),'physical_right_valid':int(states['valid'][:,1].sum()),
        'continuation':dict(config=evidence(root/'RUN_CONFIG.json'),previous_result=item['previous_result'],
            clock_source=item['clock_source'],target_frame='INFERRED_PALM_FRAME_MATCHED_TO_NEUTRAL_KAIHAND',
            claims=dict(numeric_quality_pass=False,training_eligible=False,visual_review='NOT_REVIEWED'))}
    write_json(case/'RESULT.json',result)
    report=dict(schema_version='HURO_COMMON_EVALUATION_V2',session_id=old_result['session_id'],
        status='COMPLETED_DIAGNOSTICS_NOT_QUALITY_ADOPTION',common_all_three_side_frames=int(common.sum()),
        target_definition='Fixed wrist/MCP palm basis; median MCP scale; same target for ALL three methods',
        clock='timestamp_ns from frozen source; gaps excluded, no frame/fps substitute',
        clock_source=item['clock_source'],metrics=metrics,arrays=evidence(case/'COMMON_EVALUATION_ARRAYS.npz'),
        collision_policy='CPU Bullet DIRECT; SELF_COLLISION | EXCLUDE_PARENT; zero penetration tolerance; no threshold tuning',
        missing_external_ground_truth=True,postprocessing='V2 NONE; saved raw solver output',
        authority='DEVELOPMENT_ONLY',config=evidence(root/'RUN_CONFIG.json'))
    write_json(case/'EVALUATION.json',report)
    read_ref(item['rgb'])
    if not np.array_equal(frames,np.arange(len(frames))):
        raise RuntimeError('renderer requires contiguous encoded frame axis from zero')
    rendered = render(result_path=case/'RESULT.json',rgb_video=Path(item['rgb']['path']),output_root=case)
    print(json.dumps(dict(session=item['name'],phase='COMPLETE',comparison=metrics),ensure_ascii=False),flush=True)
    return dict(name=item['name'],result=evidence(case/'RESULT.json'),evaluation=evidence(case/'EVALUATION.json'),
                render=rendered)


STARTED = 0.


def main():
    global STARTED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    root=args.config.resolve().parent
    config=json.loads(args.config.read_text())
    if config['schema_version'] != 'HURO_HAND_FRAME_RUN_CONFIG_V2' or config['gpu_used'] is not False:
        raise RuntimeError('unregistered configuration')
    if root != Path(config['output_root']).resolve() or args.config.name != 'RUN_CONFIG.json':
        raise RuntimeError('output binding mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('GPU must be disabled')
    verify_fence(config)
    for ref in config['code_closure']: read_ref(ref)
    lock=(root/'execution.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(root/'EXECUTION_STARTED.json',dict(pid=os.getpid(),started_unix=time.time(),config=evidence(args.config)))
    STARTED=time.monotonic()
    models,asset_refs=load_pinned_kaihand_models(Path(config['repo_root']))
    hands=tuple(load_hand_model(m.path,side) for m,side in zip(models,['left','right']))
    collision=ExplicitSelfCollision(models)
    results=[]
    try:
        for item in config['sessions']:
            results.append(run_case(config,item,root,hands,collision))
        for ref in config['code_closure']: read_ref(ref)
        verify_fence(config)
        write_json(root/'RESULT.json',dict(schema_version='HURO_HAND_FRAME_FOLLOWUP_V2',
            status='COMPLETED_DEVELOPMENT_DIAGNOSTICS',sessions=results,
            config=evidence(args.config),asset_refs=asset_refs,elapsed_seconds=time.monotonic()-STARTED,
            PIPELINE_COMPLETE=True,NUMERIC_QUALITY_PASS=False,TRAINING_ELIGIBLE=False,
            VISUAL_REVIEW_STATUS='NOT_REVIEWED',TRAINING_COMPLETE=False,
            CONTROL_GROUND_TRUTH=False,PHYSICAL_DEPLOYABLE=False,GPU_USED=False,
            official_full_huro_reproduction=False,stage9='BLOCKED_HARDWARE_LICENSE',wuji20='BLOCKED_ROBOT_ASSET'))
    except Exception as exc:
        write_json(root/'FAILED_EXECUTION.json',dict(status='FAILED_NO_AUTO_RETRY',error=repr(exc),
                                                   completed_sessions=results,elapsed_seconds=time.monotonic()-STARTED))
        raise
    finally:
        collision.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
