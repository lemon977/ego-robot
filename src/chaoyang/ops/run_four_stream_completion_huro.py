"""Fixed 007/181:196 projected HuRo candidate; no full-session adoption."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import threading
from pathlib import Path
import sys
import time
import types
import numpy as np

PARENT = 'four_stream_completion_20260928'
TASK = PARENT + '_huro'
SESSION = 'get_potato_chips_0915_007'
OLD_RESULT = '_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/full_0001/' + SESSION + '/RESULT.json'
ENV = '_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/env'
CODE = ['ops/run_four_stream_completion_huro.py', 'ops/run_v5_huro.py', 'ops/run_huro_fixed_placement_core_v2.py', 'ops/run_huro_common_review_v2.py', 'pipeline/huro_projected_joints_v1.py', 'pipeline/huro_constrained_core_v1.py']

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    path = Path(path).resolve(strict=True)
    a = path.stat(); content = path.read_bytes(); b = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(a) != identity(b): raise RuntimeError('REFERENCE_UNSTABLE')
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())

def verify(item):
    if ref(item['path']) != item: raise RuntimeError('REFERENCE_DRIFT:' + item['path'])
    return Path(item['path'])

def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)

def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])

def guard(repo, attempt):
    repo = Path(repo).resolve(strict=True)
    if not attempt or '/' in attempt or attempt in ['.', '..']: raise ValueError('ATTEMPT')
    rows = read(repo / 'tasks/current/INDEX.json')['task_packets']
    packets = {}
    for task in [PARENT, TASK]:
        found = [r for r in rows if r['task_id'] == task]
        if len(found) != 1 or (task == PARENT and not found[0].get('execution_allowed')): raise RuntimeError('ROUTING')
        path = repo / 'tasks/current' / task / 'TASK_PACKET.json'
        if (repo / found[0]['packet_path']).resolve() != path.resolve() or ref(path)['sha256'] != found[0]['packet_sha256']: raise RuntimeError('PACKET_SHA')
        packets[task] = read(path)
    writer = packets[TASK]['writer']
    if writer != packets[PARENT]['writer'] or ticks(writer['pid']) != writer['proc_start_ticks']: raise RuntimeError('WRITER_FENCE')
    lane = repo / '_run/current' / PARENT / 'attempts' / attempt / 'lanes/huro'
    if lane.resolve() != Path(packets[TASK]['output_root']).resolve() or str(lane.relative_to(repo)) not in packets[TASK]['write_set']: raise RuntimeError('WRITE_SET')
    for p in [lane, *lane.parents]:
        if p == repo: break
        if p.is_symlink(): raise RuntimeError('OUTPUT_SYMLINK')
    if not lane.resolve().is_relative_to(repo): raise RuntimeError('OUTPUT_ESCAPE')
    if Path(sys.prefix).resolve() != (repo / ENV).resolve(): raise RuntimeError('PINNED_ENV_REQUIRED')
    return repo, lane, writer

def environment():
    return dict(prefix=sys.prefix, executable=sys.executable, python=sys.version,
                packages={n: importlib.metadata.version(n) for n in ['jax', 'jaxlib', 'jaxlie', 'jaxls', 'pyroki', 'yourdfpy', 'numpy']})

def slice_case(case):
    def sliced(d):
        return {k: v[181:197].copy() if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 378 else v for k, v in d.items()}
    return {**case, 'hand': sliced(case['hand']), 'r0': sliced(case['r0']), 'valid': case['valid'][181:197].copy(), 'rotation_valid': case['rotation_valid'][181:197].copy()}

def inspect_robot(robot):
    import jax.numpy as jnp
    from chaoyang.pipeline.huro_projected_joints_v1 import intersect_affine_full_limits, make_projected_joint_type, validate_physical_seed
    joints = robot.robot.joints
    home = np.asarray(robot.get_neutral_config()); n = len(home)
    offset = np.asarray(joints.get_full_config(jnp.zeros(n)))
    matrix = np.stack([np.asarray(joints.get_full_config(jnp.eye(n)[i])) - offset for i in range(n)], axis=1)
    for q in [home, home * .37, np.linspace(-.2, .2, n)]:
        if not np.allclose(np.asarray(joints.get_full_config(jnp.asarray(q))), matrix @ q + offset, rtol=0, atol=1e-6): raise RuntimeError('NONAFFINE_MIMIC_MAP')
    lo, hi = intersect_affine_full_limits(np.asarray(joints.lower_limits), np.asarray(joints.upper_limits), np.asarray(joints.lower_limits_all), np.asarray(joints.upper_limits_all), matrix, offset)
    cls = make_projected_joint_type(lo, hi, home, robot.joint_mask)
    seed = validate_physical_seed(None, home, robot.joint_mask, lo, hi, 32)
    return dict(home=home.tolist(), joint_mask=np.asarray(robot.joint_mask).tolist(), lower=lo.tolist(), upper=hi.tolist(), full_lower=np.asarray(joints.lower_limits_all).tolist(), full_upper=np.asarray(joints.upper_limits_all).tolist(), affine_matrix=matrix.tolist(), affine_offset=offset.tolist(), names=list(joints.actuated_names), seed_policy='FROZEN_HOME_NO_WARM_START'), cls, seed

def r0_arm_seed(mapping, case):
    """Only the physical arm initialization changes; hand/home/rest stay frozen."""
    r0 = case['r0']; names = mapping['names']; n = len(names)
    if len(set(names)) != n: raise ValueError('DUPLICATE_JOINT_NAMES')
    if not np.array_equal(r0['human_to_physical'], [0,1]) or tuple(r0['anatomical_side_names']) != ('left','right'):
        raise ValueError('SEED_SIDE_MAPPING')
    for key in ['frame_id','timestamp_ns']:
        if not np.array_equal(r0[key], case['hand'][key]): raise ValueError('SEED_TIME_MAPPING')
    if not np.array_equal(r0['frame_id'], np.arange(181,197)): raise ValueError('SEED_FRAME_SCOPE')
    for value in [case['valid'],case['rotation_valid'],r0['target_valid'],r0['wrist_valid']]:
        value = np.asarray(value)
        if value.shape != (16,2) or value.dtype != np.bool_ or not value.all():
            raise ValueError('SEED_VALID_REQUIRED')
    q = np.asarray(r0['q_arm'])
    if q.shape != (16,2,7) or not np.isfinite(q).all(): raise ValueError('SEED_ARM_SHAPE_OR_FINITE')
    actual = np.asarray(r0['T_actual_root_cam']); target = np.asarray(r0['T_target_root_cam'])
    if actual.shape != (16,2,4,4) or target.shape != actual.shape or not np.isfinite([actual,target]).all():
        raise ValueError('SEED_POSE_SHAPE_OR_FINITE')
    for pose in [actual,target]:
        rotation = pose[...,:3,:3]
        if not np.allclose(np.swapaxes(rotation,-1,-2) @ rotation,np.eye(3),atol=1e-5,rtol=0) or not np.allclose(np.linalg.det(rotation),1,atol=1e-5,rtol=0):
            raise ValueError('SEED_POSE_NOT_SO3')
        if not np.allclose(pose[...,3,:], [0,0,0,1], atol=1e-8,rtol=0): raise ValueError('SEED_HOMOGENEOUS')
    position = np.linalg.norm(actual[...,:3,3]-target[...,:3,3],axis=-1)*1000
    relative = np.swapaxes(target[...,:3,:3],-1,-2) @ actual[...,:3,:3]
    angle = np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
    if np.any(position>20) or np.any(angle>15): raise ValueError('SEED_WRIST_GATE')
    home = np.asarray(mapping['home'],dtype=np.float32)
    seed = np.broadcast_to(home,(32,n)).copy()
    mask = np.asarray(mapping['joint_mask'],bool)
    for side,suffix in enumerate(['L','R']):
        expected = [f'Joint{i}_{suffix}' for i in range(1,8)]
        if not all(name in names for name in expected): raise ValueError('SEED_ARM_NAMES')
        indices = [names.index(name) for name in expected]
        if not mask[indices].all(): raise ValueError('SEED_ARM_LOCKED')
        # Match unchanged upstream block padding: repeat last physical arm seed.
        seed[:16,indices] = q[:,side,:]
        seed[16:,indices] = q[-1,side,:]
    lower,upper = np.asarray(mapping['lower']),np.asarray(mapping['upper'])
    if not np.isfinite(seed).all() or np.any(seed<lower) or np.any(seed>upper): raise ValueError('SEED_LIMITS_NO_CLIP')
    full = seed @ np.asarray(mapping['affine_matrix']).T + np.asarray(mapping['affine_offset'])
    if np.any(full<np.asarray(mapping['full_lower'])) or np.any(full>np.asarray(mapping['full_upper'])):
        raise ValueError('SEED_FULL_LIMITS_NO_CLIP')
    return seed

def candidate_suffix(candidate, backend):
    if candidate not in ['c2','c3'] or backend not in ['cpu','cuda']: raise ValueError('CANDIDATE_BACKEND')
    return ('_C3_R0_ARM' if candidate=='c3' else '') + ('_CPU_V1' if backend=='cpu' else '_CUDA_V1')

def verify_c3_predecessor(previous, mapping, source, motion, r0):
    if previous['frames'] != [181,196] or previous['position_gate_mm'] != 20. or previous['rotation_gate_deg'] != 15. or previous['solver_block_size'] != 32 or previous['dt_policy'] != 'UNCHANGED_UPSTREAM_ADJACENT_Q_DIFFERENCE':
        raise RuntimeError('C3_PREDECESSOR_CONTRACT_DRIFT')
    if environment() != previous['environment']: raise RuntimeError('C3_ENV_DRIFT')
    if ref(motion) != previous['motion'] or ref(r0) != previous['r0']: raise RuntimeError('C3_INPUT_DRIFT')
    if mapping != read(verify(previous['mapping'])): raise RuntimeError('C3_HOME_OR_LIMIT_OR_MAPPING_DRIFT')
    if source != verify(previous['candidate_source']).read_text(): raise RuntimeError('C3_LOSS_OR_CONSTRAINT_SOURCE_DRIFT')
    verify(previous['core'])
    for item in previous['assets']: verify(item)
    for item in previous['code']:
        if not item['path'].endswith('/ops/run_four_stream_completion_huro.py'): verify(item)

def cpu_guard():
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU_ENV_REQUIRED')
    if len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('CPU_AFFINITY_EXCEEDS_TWO')

def freeze(repo, lane, backend, candidate_id='c2'):
    cpu_guard()
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU_PREFLIGHT_REQUIRED')
    from chaoyang.ops.run_v5_huro import load_case, adapted_core
    from chaoyang.ops.run_huro_fixed_placement_core_v2 import compose_with_mounts
    from chaoyang.pipeline.huro_projected_joints_v1 import projected_core_source
    old = read(repo / OLD_RESULT)
    motion, r0 = verify(old['hand_motion']), verify(old['R0'])
    case = load_case(SESSION, motion, r0)
    selected = slice_case(case)
    if not np.array_equal(selected['hand']['frame_id'], np.arange(181, 197)): raise RuntimeError('FRAME_AXIS')
    if not selected['valid'].any(): raise RuntimeError('EMPTY_VALID_WINDOW')
    suffix = candidate_suffix(candidate_id, backend)
    root = lane / ('H01_007_181_196_FROZEN' + suffix); root.mkdir()
    core_path = repo / 'vendor/HuRo/pipeline/retargeting/retargeter.py'
    core, _, _, source = adapted_core(core_path)
    config, assets, _ = compose_with_mounts(repo, root, case['r0']['T_flange_hand'])
    robot = core.Retargeter(config)
    mapping, _, _ = inspect_robot(robot)
    candidate = projected_core_source(source)
    if candidate_id == 'c3':
        verify_c3_predecessor(read(lane / 'H01_007_181_196_FROZEN_CPU_V1/FROZEN_INPUT.json'), mapping, candidate, motion, r0)
    with (root / 'PROJECTED_CORE.py').open('x') as f: f.write(candidate)
    write(root / 'MIMIC_AND_SEED.json', mapping)
    if candidate_id == 'c3':
        seed = r0_arm_seed(mapping, selected)
        with (root / 'INITIAL_GUESS.npy').open('xb') as f: np.save(f,seed,allow_pickle=False)
    predecessor_name = 'H01_007_181_196_FROZEN_CPU_V1' if candidate_id=='c3' else 'H01_007_181_196_FROZEN'
    asset_manifest = read(root / 'COMBINED_ASSET_MANIFEST.json')
    payload = dict(schema_version='COMPLETION_H01_WINDOW_V1', task_id=TASK, session_id=SESSION, frames=[181,196], source_frame_count=378,
       solve_backend=backend, candidate_id=candidate_id, seed_policy='R0_ARM_ONLY_HOME_HAND_REST_UNCHANGED' if candidate_id=='c3' else 'FROZEN_HOME_NO_WARM_START',
       initial_guess=ref(root / 'INITIAL_GUESS.npy') if candidate_id=='c3' else None,
       predecessor_freeze=ref(lane / predecessor_name / 'FROZEN_INPUT.json'),
       wall_seconds=900,
       predecessor=ref(repo / OLD_RESULT), motion=ref(motion), r0=ref(r0), core=ref(core_path), environment=environment(),
       code=[ref(repo / 'src/chaoyang' / name) for name in CODE], assets=asset_manifest['sources'],
       combined=ref(root / 'TIANJI_KAI_KINEMATIC_COMBINED.urdf'), config=ref(root / 'ROBOT_CONFIG.json'), mapping=ref(root / 'MIMIC_AND_SEED.json'), candidate_source=ref(root / 'PROJECTED_CORE.py'),
       valid_side_frames=int(selected['valid'].sum()), mask=selected['valid'].tolist(), position_gate_mm=20., rotation_gate_deg=15., dt_policy='UNCHANGED_UPSTREAM_ADJACENT_Q_DIFFERENCE', solver_block_size=32,
       claim_limit='Historical frozen 16-frame candidate; not final Motion/HuRo shared-input baseline or full-session result.')
    write(root / 'FROZEN_INPUT.json', payload)
    return dict(status='PREFLIGHT_COMPLETE_NO_SOLVE', frozen=ref(root / 'FROZEN_INPUT.json'))

def lease_guard(repo, writer):
    lease = read(repo / '_run/current/GPU_LEASE.json')
    if lease.get('status') not in ['ACQUIRED', 'RUNNING'] or lease.get('task_id') not in [PARENT, TASK]: raise RuntimeError('GPU_LEASE_SCOPE')
    if lease.get('pid') != os.getppid() or ticks(lease['pid']) != lease['process_startticks']: raise RuntimeError('GPU_LEASE_OWNER')
    if lease.get('gpu_process_pid') not in [None, os.getpid()]: raise RuntimeError('GPU_LEASE_WORKER')
    if lease.get('executor_epoch') != writer['executor_epoch'] or datetime.fromisoformat(lease['expires_at']) <= datetime.now(timezone.utc): raise RuntimeError('GPU_LEASE_STALE')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(lease['gpu_id']) or os.environ.get('JAX_PLATFORMS') != 'cuda': raise RuntimeError('GPU_ENV')
    return lease

def solve(repo, lane, writer, frozen_path, frozen_sha, backend, candidate_id='c2'):
    lease = lease_guard(repo, writer) if backend == 'cuda' else None
    if backend == 'cpu': cpu_guard()
    if ref(frozen_path)['sha256'] != frozen_sha: raise RuntimeError('FROZEN_SHA')
    frozen = read(frozen_path)
    if frozen.get('candidate_id','c2') != candidate_id: raise RuntimeError('CANDIDATE_DRIFT')
    if frozen.get('solve_backend') != backend or frozen.get('wall_seconds') != 900: raise RuntimeError('EXECUTION_BACKEND_DRIFT')
    if frozen['task_id'] != TASK or frozen['session_id'] != SESSION or frozen['frames'] != [181,196]: raise RuntimeError('WINDOW_SCOPE')
    for key in ['motion','r0','core','combined','config','mapping','candidate_source','predecessor']: verify(frozen[key])
    for item in frozen['code'] + frozen['assets']: verify(item)
    if environment() != frozen['environment']: raise RuntimeError('ENV_DRIFT')
    from chaoyang.ops.run_v5_huro import load_case, adapted_core, solve_case, load_npz
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
    from chaoyang.pipeline.huro_constrained_core_v1 import check_raw_joint_limits
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != ('gpu' if backend == 'cuda' else 'cpu'): raise RuntimeError('EXPECTED_BACKEND')
    core, _, _, _ = adapted_core(Path(frozen['core']['path']))
    robot = core.Retargeter(types.SimpleNamespace(config=read(frozen['config']['path']), urdf_path=frozen['combined']['path']))
    mapping, cls, seed = inspect_robot(robot)
    expected = read(frozen['mapping']['path'])
    if mapping != expected: raise RuntimeError('ROBOT_MAP_DRIFT')
    core.__dict__['_completion_projected_joint_cls'] = cls
    exec(compile(Path(frozen['candidate_source']['path']).read_text(), '<frozen-projected-core>', 'exec'), core.__dict__)
    solver = core.solve_retargeting_with_projected_joints
    case = slice_case(load_case(SESSION, Path(frozen['motion']['path']), Path(frozen['r0']['path'])))
    if not case['valid'].any() or not np.array_equal(case['valid'], frozen['mask']): raise RuntimeError('EMPTY_OR_DRIFTED_VALID_WINDOW')
    if candidate_id == 'c3':
        if frozen.get('seed_policy') != 'R0_ARM_ONLY_HOME_HAND_REST_UNCHANGED': raise RuntimeError('SEED_POLICY_DRIFT')
        seed = r0_arm_seed(mapping,case)
        if not np.array_equal(seed,np.load(verify(frozen['initial_guess']),allow_pickle=False)): raise RuntimeError('SEED_DRIFT')
        from chaoyang.pipeline.huro_projected_joints_v1 import validate_physical_seed
        seed = validate_physical_seed(seed, mapping['home'], mapping['joint_mask'], mapping['lower'], mapping['upper'], 32)
    raw_blocks = []
    def checked_solver(**kwargs):
        lease_guard(repo, writer) if backend == 'cuda' else cpu_guard()
        if not np.array_equal(np.asarray(kwargs['initial_cfg']), np.asarray(mapping['home'], dtype=np.float32)): raise RuntimeError('HOME_DRIFT')
        q, cost = solver(**kwargs, initial_guess=jnp.asarray(seed))
        raw_blocks.append(np.asarray(q))
        return q, cost
    suffix = candidate_suffix(candidate_id, backend)
    root = lane / (('H01_007_181_196_PROJECTED' if candidate_id=='c3' else 'H01_007_181_196_PROJECTED_C2') + suffix); root.mkdir()
    started = time.monotonic()
    write(root / 'RUNNING.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), wall_seconds=900, backend=backend, frozen=ref(frozen_path)))
    # Hard watchdog terminates only this owned worker, never other tasks.
    watchdog = threading.Timer(900, lambda: os.kill(os.getpid(), signal.SIGTERM))
    watchdog.daemon = True; watchdog.start()
    try:
        product = solve_case(core, checked_solver, robot, load_pinned_robot_assets(repo), case, root)
    except Exception as exc:
        write(root / 'FAILED_RUNTIME.json', dict(error=repr(exc), elapsed_seconds=time.monotonic()-started, backend=backend))
        raise
    finally:
        watchdog.cancel()
    raw = np.concatenate(raw_blocks)[:16]
    full = np.asarray(jax.vmap(robot.robot.joints.get_full_config)(jnp.asarray(raw)))
    with (root / 'RAW_SOLVER_Q.npz').open('xb') as f: np.savez_compressed(f, raw_q=raw, full_q=full, frame_id=np.arange(181,197), timestamp_ns=case['hand']['timestamp_ns'], valid=case['valid'])
    arrays = load_npz(root / 'HURO_CORE_V1.npz')
    valid = arrays['target_valid']
    rotation_valid = valid & case['rotation_valid']
    position_pass = valid & (arrays['position_residual_mm'] <= 20.)
    pose_pass = position_pass & rotation_valid & (arrays['rotation_residual_deg'] <= 15.)
    collision = dict(status='NOT_EVALUATED', scope='HAND_NON_ADJACENT_SELF_ONLY', arms='NOT_EVALUATED', environment='NOT_EVALUATED')
    try:
        from chaoyang.ops.run_huro_hand_frame_v2 import ExplicitSelfCollision
        from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import load_pinned_kaihand_models
        models, _ = load_pinned_kaihand_models(repo)
        checker = ExplicitSelfCollision(models)
        try:
            collision['rows'] = [dict(frame=int(181+i), side=int(s), **asdict(checker.check(int(s), arrays['q_hand22'][i,s]))) for i,s in np.argwhere(valid)]
            collision['status'] = 'EXECUTED'
        finally: checker.close()
    except ImportError as exc: collision.update(status='MISSING_CPU_COLLISION_DEPENDENCY', error=str(exc))
    result = dict(task_id=TASK, execution='REAL_OFFICIAL_CORE_ADAPTED_WINDOW_EXECUTED', original_q=ref(root / 'RAW_SOLVER_Q.npz'), candidate=ref(root / 'HURO_CORE_V1.npz'), frozen=ref(frozen_path),
      actuator_limits=check_raw_joint_limits(raw,mapping['lower'],mapping['upper']), full_limits=check_raw_joint_limits(full,mapping['full_lower'],mapping['full_upper']),
      side_frames=int(valid.sum()), rotation_valid_side_frames=int(rotation_valid.sum()), position_pass_side_frames=int(position_pass.sum()), pose_pass_side_frames=int(pose_pass.sum()), pose_gate_pass=bool(valid.any() and np.all(pose_pass[valid])), collision=collision,
      elapsed_seconds=time.monotonic()-started, max_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), backend=backend, gpu_used=backend=='cuda', lease_fencing=lease['fencing_token'] if lease else None, posthoc_clip_used=False,
      quality='REQUIRES_FULL_REVIEW_NO_ADOPTION', adoption='NOT_ADOPTED', full_session_executed=False)
    write(root / 'WINDOW_RESULT.json', result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True); p.add_argument('--attempt',required=True)
    p.add_argument('--mode',choices=['freeze','solve'],required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],required=True)
    p.add_argument('--candidate',choices=['c2','c3'],default='c2')
    p.add_argument('--frozen',type=Path); p.add_argument('--frozen-sha')
    args=p.parse_args(); repo,lane,writer=guard(args.repo,args.attempt)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    if args.mode=='solve' and (not args.frozen or not args.frozen_sha): p.error('solve requires frozen path/SHA')
    print(json.dumps(freeze(repo,lane,args.backend,args.candidate) if args.mode=='freeze' else solve(repo,lane,writer,args.frozen,args.frozen_sha,args.backend,args.candidate), ensure_ascii=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
