"""Render frozen official HuRo q without re-solving; compare on frozen targets."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import numpy as np
from chaoyang.pipeline.full_robot_review_v2 import (
    read_frozen, reference, write_json, render_review, time_edges, motion_derivatives,
)


def load(ref, roots):
    with np.load(io.BytesIO(read_frozen(ref, roots)), allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def stats(values):
    x=np.asarray(values); x=x[np.isfinite(x)]
    return dict(count=int(x.size), median=float(np.median(x)) if x.size else None,
                p95=float(np.percentile(x,95)) if x.size else None,
                maximum=float(x.max()) if x.size else None)


def validate_inputs(core, target, baseline):
    for key,other in [('source_frame_id','frame_id'),('timestamp_ns','timestamp_ns')]:
        if not np.array_equal(core[key],baseline[other]) or not np.array_equal(core[key],target[key]):
            raise ValueError('TIME_OR_FRAME_ID_MISMATCH')
    if not np.array_equal(core['human_to_physical'],[1,0]):
        raise ValueError('SIDE_MISMATCH')
    valid=core['source_valid_side_frame']
    if valid.dtype!=bool or not np.array_equal(valid,target['target_valid']):
        raise ValueError('TARGET_VALIDITY_MISMATCH')
    # Limit violations remain evaluated/rendered, not dropped to improve residuals.
    if np.any(valid & ~np.isfinite(core['q_arm']).all(axis=-1)) or np.any(valid & ~np.isfinite(core['q22']).all(axis=-1)):
        raise ValueError('MISSING_SOLVER_OUTPUT_ON_VALID_TARGET')
    if not np.isfinite(target['target21_base'][valid]).all():
        raise ValueError('NONFINITE_VALID_TARGET')
    if not np.allclose(target['target_root_T'][valid],baseline['T_target_root'][valid],atol=2e-8,rtol=0):
        raise ValueError('PLACEMENT_OR_TARGET_DRIFT')
    return valid.copy()


def diagnostic_fk_unbounded(model, q):
    """Mathematical FK of RAW invalid states; NEVER an admissibility decision.

    Original immutable models and their actual limits remain unchanged. Limit
    violations are separately exported, never clipped or accepted as valid.
    """
    from dataclasses import replace
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    view=replace(model,joints=tuple(replace(j,lower=None,upper=None) for j in model.joints))
    return forward_kinematics(view,q)


def forward_points(assets, q_arm, q22, valid, mounts):
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
    from chaoyang.pipeline.huro_hand_only_retarget_v1 import _keypoint_links
    roots=np.full((*valid.shape,4,4),np.nan)
    points=np.full((*valid.shape,21,3),np.nan)
    for t,side in np.argwhere(valid):
        # Complete joint identity is required by production FK. The opposite
        # independent branch has a neutral display-only placeholder, not an observation.
        state={j.name:float((j.lower+j.upper)/2) for j in assets.tianji.joints if j.joint_type!='fixed'}
        state.update(dict(zip(ARM_JOINT_NAMES[side],q_arm[t,side])))
        arm_fk=diagnostic_fk_unbounded(assets.tianji,state)
        roots[t,side]=arm_fk[('flange_L','flange_R')[side]]@mounts[side]
        model=(assets.left_hand,assets.right_hand)[side]
        names=[j.name for j in model.joints if j.joint_type!='fixed']
        hand_fk=diagnostic_fk_unbounded(model,dict(zip(names,q22[t,side])))
        points[t,side]=np.stack([(roots[t,side]@hand_fk[n])[:3,3] for n in _keypoint_links(('left','right')[side])])
    return roots,points


def compare_points(reference_points, candidate_points, target, reference_valid, candidate_valid):
    common=target['target_valid'] & reference_valid & candidate_valid
    for points in (reference_points,candidate_points,target['target21_base']):
        if not np.isfinite(points[common]).all():raise ValueError('NONFINITE_COMMON_COMPARISON')
    output={'comparison_scope':'SAME_FROZEN_OFFLINE_TARGETS_NOT_PHYSICAL_GROUND_TRUTH',
            'total_timeline_frames':len(common),'common_side_frames':common.sum(axis=0).tolist(),
            'reference_coverage':reference_valid.sum(axis=0).tolist(),
            'candidate_coverage':candidate_valid.sum(axis=0).tolist()}
    for name,points in [('reference',reference_points),('official_huro',candidate_points)]:
        errors=np.linalg.norm(points-target['target21_base'],axis=-1)*1000
        output[name]={'all_joint_error_mm_common':stats(errors[common]),
                      'wrist_position_error_mm_common':stats(errors[...,0][common]),
                      'fingertip_error_mm_common':stats(errors[...,[4,8,12,16,20]][common])}
        # Thumb-index gap is a link-origin distance, not measured pad contact.
        gap=np.linalg.norm(points[...,4,:]-points[...,8,:],axis=-1)*1000
        target_gap=np.linalg.norm(target['target21_base'][...,4,:]-target['target21_base'][...,8,:],axis=-1)*1000
        output[name]['thumb_index_link_gap_error_mm_common']=stats(np.abs(gap-target_gap)[common])
    output['adoption']='NOT_AUTOMATIC_NO_INDEPENDENT_REAL_WORLD_ACCURACY_OR_COLLISION_PASS'
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    repo=Path(os.environ['CHAOYANG_REPO_ROOT']).resolve(strict=True)
    spec=json.loads(args.spec.read_text());out=args.output.resolve()
    allowed=repo/'_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro'
    if out.exists() or not out.is_relative_to(allowed):raise ValueError('NEW_OWN_OUTPUT_REQUIRED')
    if spec['schema_version']!='HURO_COMMON_REVIEW_SPEC_V2':raise ValueError('SPEC_SCHEMA')
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
    from chaoyang.pipeline.robot_scene_state_cpu import _rotation_vector
    assets=load_pinned_robot_assets(repo);out.mkdir(parents=True)
    write_json(out/'INPUT_SPEC.json',spec);results=[]
    for case in spec['sessions']:
        dest=out/case['session_id'];dest.mkdir()
        core=load(case['states'],[repo]);target=load(case['targets'],[repo]);base=load(case['reference_motion'],[repo])
        valid=validate_inputs(core,target,base)
        roots,points=forward_points(assets,core['q_arm'],core['q22'],valid,base['T_flange_hand'])
        parity=float(np.max(np.abs(roots[valid]-core['actual_hand_root_T'][valid])))
        point_parity=float(np.max(np.abs(points[valid]-core['fk21_base'][valid])))
        if max(parity,point_parity)>3e-5:raise ValueError('OFFICIAL_FK_PROJECT_FK_PARITY')
        refvalid=base['wrist_valid']&base['finger_valid']
        _,refpoints=forward_points(assets,base['q_arm'],base['q22'],refvalid,base['T_flange_hand'])
        position=np.full(valid.shape,np.nan);rotation=position.copy()
        for t,side in np.argwhere(valid):
            delta=np.linalg.inv(target['target_root_T'][t,side])@roots[t,side]
            position[t,side]=np.linalg.norm(delta[:3,3])*1000
            rotation[t,side]=np.degrees(np.linalg.norm(_rotation_vector(delta[:3,:3])))
        edge,times=time_edges(core['timestamp_ns'],core['source_frame_id'])
        motion=dict(q_arm=core['q_arm'],q22=core['q22'],wrist_valid=valid,finger_valid=valid,
                    T_actual_root=roots,T_target_root=target['target_root_T'],
                    T_flange_hand=base['T_flange_hand'],neutral_q_arm=base['neutral_q_arm'],neutral_roots=base['neutral_roots'],
                    position_residual_mm=position,rotation_residual_deg=rotation,
                    # Shared display says solver/tolerance; unknown convergence cannot pass.
                    tolerance_pass=np.zeros_like(valid),solver_success_known=np.zeros_like(valid),
                    geometric_tolerance_only=valid&(position<=20)&(rotation<=15),
                    timestamp_ns=core['timestamp_ns'],frame_id=core['source_frame_id'],
                    human_to_physical=core['human_to_physical'],limit_violation_rad=core['limit_violation_rad'],
                    **{'arm_'+k:v for k,v in motion_derivatives(core['q_arm'],valid,times,edge).items()},
                    **{'finger_'+k:v for k,v in motion_derivatives(core['q22'],valid,times,edge).items()})
        np.savez_compressed(dest/'MOTION_RESULT.npz',**motion)
        comparison=compare_points(refpoints,points,target,refvalid,valid)
        comparison.update(project_vs_official_fk_max_abs=parity,point_fk_max_abs=point_parity,
                          raw_limit_violation_side_frames=int(np.count_nonzero(valid&(core['limit_violation_rad']>1e-5))),
                          official_solver_convergence='UNKNOWN_NOT_EXPOSED',
                          geometric_tolerance_only_by_side=motion['geometric_tolerance_only'].sum(axis=0).tolist(),
                          candidate_q_passthrough_bit_exact=True,
                          wrist_rotation_error_deg=stats(rotation[valid]))
        write_json(dest/'COMPARISON.json',comparison)
        video=reference(case['reference_video']['path'])
        if video!=case['reference_video']:raise ValueError('REFERENCE_VIDEO_DRIFT')
        warning='RAW LIMIT VIOLATIONS PRESENT' if comparison['raw_limit_violation_side_frames'] else 'RAW LIMITS CHECKED'
        display=render_review(motion,assets,video['path'],dest/'HURO_FULL_ROBOT_COMPARISON_V2.mp4',
                              case['session_id']+' | HURO '+warning,
                              preview_frames=(0,len(valid)//2,len(valid)-1))
        write_json(dest/'COLLISION_AND_DISPLAY.json',display)
        result=dict(schema_version='HURO_COMMON_REVIEW_RESULT_V2',session_id=case['session_id'],
                    source=case,motion=reference(dest/'MOTION_RESULT.npz'),comparison=reference(dest/'COMPARISON.json'),
                    review_video=reference(dest/'HURO_FULL_ROBOT_COMPARISON_V2.mp4'),decoded_frames=display['decoded_frames'],
                    actual_q_source='OFFICIAL_CORE_NO_LOCAL_IK_NO_CLIPPING',
                    display_solver_gate='FALSE_BECAUSE_UPSTREAM_CONVERGENCE_UNKNOWN; geometric_tolerance_only separate',
                    numeric_quality_pass=False,training_eligible=False,gpu_used=False,
                    visual_review_status='PENDING_HUMAN_REVIEW')
        write_json(dest/'RESULT.json',result);results.append(result)
    write_json(out/'RESULT.json',dict(schema_version='HURO_COMMON_REVIEW_BATCH_V2',sessions=results,gpu_used=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
