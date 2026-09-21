"""Finite two-session official HuRo-core adaptation, never a training claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def read_json_ref(reference):
    path=Path(reference['path'])
    before=path.stat();raw=path.read_bytes();after=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or len(raw)!=reference['bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('JSON reference mismatch or unstable read')
    return json.loads(raw)


def verify_writer(config):
    state=json.loads((Path(config['lane_root'])/'STATE.json').read_text())
    expected=config['writer']
    for key in ('pid','proc_start_ticks','executor_epoch','fencing_token'):
        if state[key]!=expected[key]: raise RuntimeError('writer fence changed')
    ticks=Path(f"/proc/{expected['pid']}/stat").read_text().split(') ',1)[1].split()[19]
    if str(ticks)!=str(expected['proc_start_ticks']): raise RuntimeError('writer process identity changed')


def solve_block(core, robot, points, mask, *, block_size):
    import numpy as np
    import jax.numpy as jnp
    import jaxlie
    count = len(points)
    if not 0 < count <= block_size:
        raise ValueError("block length outside frozen budget")
    points = np.where(mask[...,None],points,0.)
    if not np.isfinite(points).all():
        raise ValueError("valid target contains nonfinite coordinate")
    points = np.pad(points,((0,block_size-count),(0,0),(0,0)))
    mask = np.pad(mask,((0,block_size-count),(0,0)))
    local_links=np.concatenate([robot.left_local_link_indices,robot.right_local_link_indices])
    global_indices=np.array([0,4,8,12,16,20,21,25,29,33,37,41])
    weights=dict(core.DEFAULT_WEIGHTS)
    weights['ego_view_rot']=weights['ego_view_pos']=0.
    rest=np.where(robot.hand_joint_mask>.5,weights['rest_weight_default']*weights['hand_rest_scale'],weights['rest_weight_default'])
    q,cost=core.solve_retargeting(robot=robot.robot,
        local_keypoints=jnp.asarray(points,dtype=jnp.float32),local_link_indices=jnp.asarray(local_links),
        local_conn_mask=jnp.asarray(robot.conn_mask),local_kpt_mask=jnp.asarray(mask,dtype=jnp.float32),
        global_keypoints=jnp.asarray(points[:,global_indices],dtype=jnp.float32),
        global_link_indices=jnp.asarray(local_links[global_indices]),global_kpt_mask=jnp.asarray(mask[:,global_indices],dtype=jnp.float32),
        joint_mask=jnp.asarray(robot.joint_mask),initial_cfg=jnp.asarray(robot.get_neutral_config(),dtype=jnp.float32),
        weights=weights,rest_weight_per_joint=jnp.asarray(rest,dtype=jnp.float32),
        padding_mask=jnp.asarray(np.arange(block_size)<count,dtype=jnp.float32),camera_link_index=robot.camera_link_index,
        target_cam_se3=jaxlie.SE3.from_matrix(jnp.tile(jnp.eye(4),(block_size,1,1))),cam_mask=jnp.zeros(block_size),
        hand_joint_mask=jnp.asarray(robot.hand_joint_mask))
    return np.asarray(q)[:count],float(cost)


def run_case(core, robot, assets, neutral_roots, case, config, output):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jaxlie
    from chaoyang.ops.run_huro_fixed_placement_core_v2 import verified_npz,dump,ref
    from chaoyang.pipeline.huro_frozen_targets_v2 import build_targets,segmented_chunks
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    from chaoyang.pipeline.huro_hand_only_retarget_v1 import _keypoint_links
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
    source = read_json_ref(case['clip_receipt'])
    if source['session_id'] != case['session_id']:
        raise ValueError('session mismatch')
    if source['source_frozen_q22']!=case['r0'] or source['source_hawor']!=case['raw_hawor']:
        raise ValueError('shared input freeze and predecessor refs differ')
    r0=verified_npz(source['source_frozen_q22'])
    raw=verified_npz(source['source_hawor'])
    neutral=[]
    home=robot.config['home_config']
    for side,model in zip(('left','right'),(assets.left_hand,assets.right_hand)):
        fk=forward_kinematics(model,{j.name:home[j.name] for j in model.joints if j.joint_type!='fixed'})
        neutral.append(np.stack([fk[n][:3,3] for n in _keypoint_links(side)]))
    targets=build_targets(r0,raw,neutral,neutral_roots)
    ids=targets['source_frame_id']; clocks=targets['timestamp_ns']; target_valid=targets['target_valid']
    chunks=segmented_chunks(ids,clocks,target_valid,max_frames=config['block_size'],max_gap_ns=config['max_gap_ns'])
    n=len(ids); total_joints=len(robot.get_neutral_config())
    solved=np.full((n,total_joints),np.nan)
    block_records=[]
    for number,indices in enumerate(chunks):
        verify_writer(config)
        start=time.monotonic()
        pts=targets['target21_base'][indices].reshape(len(indices),42,3)
        mask=np.repeat(target_valid[indices],21,axis=1)
        q,cost=solve_block(core,robot,pts,mask,block_size=config['block_size'])
        if q.shape!=(len(indices),total_joints) or not np.isfinite(q).all() or not np.isfinite(cost):
            raise RuntimeError('nonfinite official solve output')
        solved[indices]=q
        block_records.append(dict(block=number,source_indices=indices,cost=cost,seconds=time.monotonic()-start))
        print(json.dumps({'session':case['session_id'],'block':number+1,'blocks':len(chunks),'seconds':block_records[-1]['seconds']}),flush=True)
    names=list(robot.robot.joints.actuated_names)
    link_names=list(robot.robot.links.names)
    qarm=np.full((n,2,7),np.nan); qhand=np.full((n,2,22),np.nan)
    actual_roots=np.full((n,2,4,4),np.nan); points=np.full((n,2,21,3),np.nan)
    limit_violation=np.full((n,2),np.nan)
    valid=np.zeros((n,2),bool)
    for t in np.flatnonzero(target_valid.any(axis=1)):
        fk=np.asarray(jaxlie.SE3(robot.robot.forward_kinematics(jnp.asarray(solved[t]))).as_matrix())
        for side,model in enumerate((assets.left_hand,assets.right_hand)):
            if not target_valid[t,side]: continue
            s=('left','right')[side]
            arm_idx=[names.index(x) for x in ARM_JOINT_NAMES[side]]
            hand_joints=[j for j in model.joints if j.joint_type!='fixed']
            hand_idx=[names.index(j.name) for j in hand_joints]
            qarm[t,side]=solved[t,arm_idx];qhand[t,side]=solved[t,hand_idx]
            actual_roots[t,side]=fk[link_names.index(robot.config['eef_link_names'][s])]
            points[t,side]=fk[[link_names.index(x) for x in _keypoint_links(s)],:3,3]
            arm_joints={j.name:j for j in assets.tianji.joints}
            selected=[arm_joints[x] for x in ARM_JOINT_NAMES[side]]+hand_joints
            qpart=np.r_[qarm[t,side],qhand[t,side]]
            lo=np.array([j.lower for j in selected]);hi=np.array([j.upper for j in selected])
            limit_violation[t,side]=max(0.,float(np.max(lo-qpart)),float(np.max(qpart-hi)))
            valid[t,side]=limit_violation[t,side]<=1e-5
    arrays=dict(q_arm=qarm,q_hand=qhand,q22=qhand,valid_side_frame=valid,
                source_valid_side_frame=target_valid,source_observed_physical=targets['source_observed_physical'],
                actual_hand_root_T=actual_roots,fk21_base=points,raw_solver_q_actuated=solved,
                limit_violation_rad=limit_violation,source_frame_id=ids,timestamp_ns=clocks,
                human_to_physical=np.array([1,0]),joint_names=np.asarray(names),
                control_ground_truth=np.asarray(False),training_eligible=np.asarray(False))
    np.savez_compressed(output/'HURO_OFFICIAL_CORE_STATES.npz',**arrays)
    np.savez_compressed(output/'FROZEN_COMMON_TARGETS.npz',**targets)
    distances=np.linalg.norm(points-targets['target21_base'],axis=-1)
    finite=distances[np.isfinite(distances)]
    summary=dict(status='OFFICIAL_CORE_FIXED_PLACEMENT_EXECUTED',session_id=case['session_id'],
        official_core_executed=True,frames=n,source_observed_sides=targets['source_observed_physical'].sum(axis=0).tolist(),
        target_valid_sides=target_valid.sum(axis=0).tolist(),within_limit_export_sides=valid.sum(axis=0).tolist(),
        fixed_scale_by_physical=[float(x) if np.isfinite(x) else None for x in targets['fixed_scale_by_physical']],
        input_refs={'r0':source['source_frozen_q22'],'raw':source['source_hawor'],'clip_diagnostic':case['clip_receipt']},
        historical_r0_clipped_side_frames=source['clipped_side_frames'],historical_r0_was_postclipped=True,
        blocks=block_records,solver_convergence='NOT_EXPOSED_BY_UPSTREAM_WRAPPER_FINITE_COST_ONLY',
        temporal_regularizer='OFFICIAL_INDEX_FIRST_DIFFERENCE_NOT_DT_NORMALIZED_SEGMENTED_AT_GAPS_AND_VALIDITY_CHANGES',
        temporal_evaluation='USE_EXPORTED_REAL_TIMESTAMP_NS_NO_CROSS_GAP_DIFFERENCES',
        fixed_shape_scale_authority='OFFLINE_NONCAUSAL_NOT_PHYSICAL_CALIBRATION',
        target_residual_all_joint_median_m=float(np.median(finite)) if len(finite) else None,
        outputs={'states':ref(output/'HURO_OFFICIAL_CORE_STATES.npz'),'targets':ref(output/'FROZEN_COMMON_TARGETS.npz')},
        numeric_quality_pass=False,training_eligible=False,control_ground_truth=False,
        visual_review='PENDING_COMMON_RENDERER',collision='NOT_EVALUATED_HERE',gpu_used=False)
    dump(output/'RESULT.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv);config=json.loads(args.config.read_text())
    verify_writer(config)
    if os.environ.get('JAX_PLATFORMS')!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU_ONLY_ENV_REQUIRED')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    from chaoyang.ops.run_huro_fixed_placement_core_v2 import compose_with_mounts,load_core,dump,ref
    from chaoyang.pipeline.huro_core_adapter_v2 import CORE_SOURCE_SHA
    root=Path(config['output']);lane=Path(config['lane_root']).resolve()
    if not root.resolve().is_relative_to(lane) or root.exists(): raise ValueError('new own output required')
    if config['block_size']!=32 or config['max_gap_ns']!=250_000_000: raise ValueError('frozen chunk policy differs')
    root.mkdir(parents=True)
    repo=Path(config['repo'])
    cfg,assets,neutral_roots=compose_with_mounts(repo,root,config['T_flange_hand'])
    core=load_core(repo/'vendor/HuRo/pipeline/retargeting/retargeter.py');robot=core.Retargeter(cfg)
    freeze=read_json_ref(config['shared_input_freeze'])
    if freeze['change_after_freeze_allowed'] or freeze['relative_motion_scale']!=1.:
        raise ValueError('shared fixed input contract violated')
    cases=[]
    for case in freeze['source_sessions']:
        target=root/case['session_id'];target.mkdir()
        cases.append(run_case(core,robot,assets,neutral_roots,case,config,target))
    dump(root/'RESULT.json',dict(status='OFFICIAL_CORE_TWO_SESSION_EXECUTED',official_core_executed=True,
        cases=[ref(root/c['session_id']/'RESULT.json') for c in cases],config=ref(args.config),
        numeric_quality_pass=False,training_eligible=False,control_ground_truth=False,gpu_used=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
