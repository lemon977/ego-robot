"""Explicit SHA-bound CPU full Robot operation for the four-stream V2 release."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from chaoyang.pipeline.full_robot_review_v2 import (
    HUMAN_TO_PHYSICAL, read_frozen, reference, relative_wrist, validate_motion,
    solve_full_chain, render_review, write_json,
)


def prepare_arrays(case, roots, assets):
    with np.load(io.BytesIO(read_frozen(case['motion'], roots)), allow_pickle=False) as z:
        source = {k:z[k] for k in z.files}
    kind = case['kind']
    if kind == 'FROZEN_R0':
        if not np.array_equal(source['human_to_physical'], HUMAN_TO_PHYSICAL):
            raise ValueError('R0_SIDE_MAPPING_DRIFT')
        arrays = dict(q22=source['q22_init'], relative_wrist_T=source['relative_wrist_T'],
                      finger_valid=source['valid_side_frame'], wrist_valid=source['valid_side_frame'],
                      timestamp_ns=source['timestamp_ns'], frame_id=source['frame_id'],
                      human_to_physical=source['human_to_physical'],
                      source_observed_valid_anatomical=source['source_observed_valid_anatomical'])
        source_method = 'FROZEN_R0_HISTORICALLY_POSTCLIPPED_Q22_UNMODIFIED_BY_THIS_OPERATION'
    elif kind in {'PICO_MANUS', 'FROZEN_MANO21'}:
        if kind == 'PICO_MANUS' and 'provenance_v2_direct_observation_unverified' not in source:
            raise ValueError('PICO_SOURCE_PROVENANCE_CORRECTION_REQUIRED')
        from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
        from chaoyang.pipeline.huro_hand_frame_v2 import solve_aligned_sequence, palm_basis
        hands = [load_hand_model(m.path, side) for m, side in
                 zip((assets.left_hand,assets.right_hand),('left','right'))]
        if kind == 'PICO_MANUS':
            joints = source['joints_world_m'].transpose(1,0,2,3)
            observed = source['joint_world_valid'].all(axis=-1).T
            ns = source['timestamp_ns']; ids = source['frame_id']
            world_wrist = source['T_world_wrist'][:,::-1].copy()
            wrist_valid = source['wrist_world_valid'][:,::-1].copy()
        else:
            joints = source['joints_3d_world']; observed = source['observed'].astype(bool)
            ids = source['original_frame_indices'].astype(np.int64)
            timestamps = []
            for i, ref in enumerate(case['timestamps']):
                metadata = json.loads(read_frozen(ref, roots))['metadata']
                if metadata['idx'] != int(ids[i]) or type(metadata['ts']) is not int:
                    raise ValueError('MANO_TIME_MAPPING_MISMATCH')
                timestamps.append(metadata['ts'])
            ns = np.asarray(timestamps,dtype=np.int64)
            world_wrist = np.full((len(ids),2,4,4),np.nan); wrist_valid = observed[::-1].T.copy()
            for human, physical in enumerate(HUMAN_TO_PHYSICAL):
                for t in np.flatnonzero(observed[human]):
                    world_wrist[t,physical] = np.eye(4)
                    world_wrist[t,physical,:3,:3] = palm_basis(joints[human,t])
                    world_wrist[t,physical,:3,3] = joints[human,t,0]
        # Existing frozen hand-only method is only a local reference, NOT official HuRo.
        fit = solve_aligned_sequence(hands,joints,observed,ids,max_evaluations=80,
                                     progress=lambda a,b:print(f'finger {a}/{b}',flush=True))
        relative, origins = relative_wrist(world_wrist,wrist_valid)
        arrays = dict(q22=fit['q22'],finger_valid=fit['valid'],wrist_valid=wrist_valid,
                      relative_wrist_T=relative,timestamp_ns=ns,frame_id=ids,
                      human_to_physical=HUMAN_TO_PHYSICAL,world_wrist_origin=origins,
                      finger_solver_success=fit['solver_success'],source_observed_physical=fit['source_observed_physical'])
        if kind == 'PICO_MANUS':
            # The hand-only solver consumes computable geometry, not direct authority.
            arrays['source_observed_physical'] = np.zeros_like(fit['valid'])
            arrays['source_computable_physical'] = observed[::-1].T.copy()
            arrays['provenance_v2_upstream_resampled'] = source['provenance_v2_upstream_resampled'].any(axis=-1)[:,::-1].copy()
            arrays['provenance_v2_direct_observation_unverified'] = source['provenance_v2_direct_observation_unverified'].any(axis=-1)[:,::-1].copy()
            arrays['provenance_v2_exact_interpolation_support_known'] = np.zeros_like(fit['valid'])
            arrays['segment_id'] = source['segment_id']
            for key in ('wrist_position_observed','wrist_rotation_observed','surface_valid'):
                arrays[key] = source[key][:,::-1].copy()
        source_method = 'EXISTING_HURO_DERIVED_PALM_FRAME_V2_REFERENCE_NOT_OFFICIAL_HURO_CORE'
    else:
        raise ValueError('UNKNOWN_INPUT_KIND')
    if len(arrays['frame_id']) != case['expected_frames']:
        raise ValueError('SESSION_FRAME_COUNT_DRIFT')
    validate_motion(arrays)
    return arrays, source_method


def clipping_claim(kind):
    if kind == 'FROZEN_R0':
        return 'No new clipping. Frozen R0 input was historically clipped; input limits do not certify preclip validity.'
    if kind in {'PICO_MANUS', 'FROZEN_MANO21'}:
        return 'No posthoc clipping in this operation. Finger q comes from the explicit hand-only reference solver; no inherited R0 historical clipping claim.'
    raise ValueError('UNKNOWN_INPUT_KIND')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interface-smoke',action='store_true',help='Only first two frames, explicitly not a full experiment')
    args=parser.parse_args(argv)
    root=Path(os.environ['CHAOYANG_REPO_ROOT']).resolve(strict=True)
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    if spec['schema_version'] != 'FOUR_STREAM_FULL_ROBOT_REVIEW_SPEC_V2': raise ValueError('UNKNOWN_SPEC')
    output=args.output.resolve()
    allowed=root/'_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001'
    if not output.is_relative_to(allowed) or output.exists(): raise ValueError('OUTPUT_SCOPE_OR_EXISTS')
    if spec['mount_policy'] != 'FLANGE_TO_HAND_CONVERTED_TO_TOOL_V2': raise ValueError('MOUNT_POLICY')
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
    from chaoyang.ops.run_0915_robot15h_virtual_arm_wave0_v1 import verify_static_assets, verify_robot_asset_closure
    assets=load_pinned_robot_assets(root); refs,flange_mounts=verify_static_assets()
    closure=verify_robot_asset_closure()
    output.mkdir(parents=True)
    write_json(output/'INPUT_SPEC.json',spec)
    all_results=[]; start=time.monotonic()
    for case in spec['sessions']:
        dest=output/case['session_id'];dest.mkdir()
        arrays,method=prepare_arrays(case,spec['allowed_read_roots'],assets)
        if args.interface_smoke:
            n=len(arrays['frame_id'])
            arrays={k:(v[:2] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in arrays.items()}
        np.savez_compressed(dest/'MOTION_INPUT.npz',**arrays)
        solved=solve_full_chain(arrays,assets,flange_mounts,progress=lambda s:print(s,flush=True))
        np.savez_compressed(dest/'MOTION_RESULT.npz',**solved)
        # Source review/video is SHA bound once, not inferred from a directory date.
        video_ref=reference(case['review_video']['path'])
        if video_ref != case['review_video']: raise ValueError('VIDEO_REFERENCE_DRIFT')
        review=render_review(solved,assets,video_ref['path'],dest/'FULL_ROBOT_REVIEW_V2.mp4',case['session_id'],
                             preview_frames=(0,len(arrays['frame_id'])//2,len(arrays['frame_id'])-1))
        write_json(dest/'COLLISION_AND_DISPLAY.json',review)
        valid=solved['wrist_valid'];good=solved['tolerance_pass']
        hand_limits=[]
        for side,model in enumerate((assets.left_hand,assets.right_hand)):
            joints=[j for j in model.joints if j.joint_type!='fixed'];q=solved['q22'][:,side]
            mask=solved['finger_valid'][:,side];lo=np.asarray([j.lower for j in joints]);hi=np.asarray([j.upper for j in joints])
            hand_limits.append(int(np.count_nonzero(mask & np.any((q<lo)|(q>hi),axis=-1))))
        result=dict(schema_version='FULL_ROBOT_REVIEW_RESULT_V2',session_id=case['session_id'],
                    status='INTERFACE_SMOKE_ONLY' if args.interface_smoke else 'EXECUTED_PENDING_QUALITY_AND_VISUAL_REVIEW',
                    source_method=method,input=case,motion_input=reference(dest/'MOTION_INPUT.npz'),
                    motion_result=reference(dest/'MOTION_RESULT.npz'),review_video=reference(dest/'FULL_ROBOT_REVIEW_V2.mp4'),
                    decoded_frames=review['decoded_frames'],full_timeline_frames=len(valid),
                    wrist_valid_by_physical_side=valid.sum(axis=0).tolist(),
                    finger_valid_by_physical_side=solved['finger_valid'].sum(axis=0).tolist(),
                    arm_tolerance_pass_by_physical_side=good.sum(axis=0).tolist(),
                    arm_failure_by_physical_side=(valid & ~good).sum(axis=0).tolist(),
                    input_q_finger_limit_violation_frames_by_side=hand_limits,
                    historical_preclip_evidence=case.get('upstream_clip_receipt'),
                    clipping_claim=clipping_claim(case['kind']),
                    mount_correction='T_tool_hand = inverse(T_flange_tool) @ T_flange_hand',
                    arm_gate_scope='Inherited virtual-development 20mm/15deg plus optimizer success, not physical accuracy',
                    input_mask_policy='Invalid source rows remain invalid; gray neutral mesh is display-only',
                    surface_reference='URDF link origins/meshes only; not calibrated physical fingertip contacts',
                    gpu_used=False,training_eligible=False,control_ground_truth=False,
                    visual_review_status='PENDING_HUMAN_REVIEW',algorithm_adoption='NOT_AUTOMATIC',assets=refs)
        write_json(dest/'RESULT.json',result);all_results.append(result)
    write_json(output/'RESULT.json',dict(schema_version='FULL_ROBOT_REVIEW_BATCH_V2',
        interface_smoke=args.interface_smoke,sessions=all_results,elapsed_seconds=time.monotonic()-start,
        asset_closure=closure,spec=reference(args.spec),gpu_used=False,training_eligible=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
