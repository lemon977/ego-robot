"""Point-semantics-aware historical overlay, explicitly not calibrated output."""
import numpy as np
from chaoyang.pipeline.s01_point_semantics_v1 import validate_contract,metadata_errors

FRAMES = (14,15,16,44,45,46,74,75,76,104,105,106)


def consume(contract, config, fit, motion, mode):
    validate_contract(contract)
    errors=metadata_errors(config,fit)
    if errors != contract['metadata_validation']['errors']:
        raise ValueError('metadata evidence differs from semantics contract')
    if mode != 'REJECTED_PROVENANCE_DIAGNOSTIC':
        raise ValueError('legacy inconsistent provenance not eligible for normal consumer')
    if str(motion['session_id']) != 'play_cards_0916_097':
        raise ValueError('short overlay restricted to097')
    np.testing.assert_array_equal(motion['frame_id'],np.arange(165))
    np.testing.assert_array_equal(motion['source_video_frame_id'],np.arange(165))
    np.testing.assert_array_equal(motion['anatomical_side_names'],['left','right'])
    np.testing.assert_allclose(motion['T_controller_wrist_estimate'],config['T_controller_wrist_estimate'],atol=1e-10,rtol=0)
    if tuple(motion['camera_image_size']) != (1280,960) or int(motion['camera_source_index'])!=1:
        raise ValueError('encoded image domain mismatch')
    if bool(motion['physical_calibration_verified']):
        raise ValueError('historical evidence cannot certify physical calibration')
    return {'mode':mode,'adoption':'NOT_ADOPTED','metadata_errors':errors,
            'wrist_accuracy':'NOT_EVALUABLE_NO_SAME_POINT_LABEL',
            'housing_to_controller_origin_error':'FORBIDDEN_DIFFERENT_POINT_SEMANTICS'}


def project(points,K):
    points=np.asarray(points);valid=np.isfinite(points).all(-1)&(points[...,2]>1e-6)
    uv=np.full(points.shape[:-1]+(2,),np.nan)
    uv[valid]=points[valid,:2]/points[valid,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    return uv,valid


def projected_points(motion,frame):
    if frame not in FRAMES:raise ValueError('frame outside preregistered windows')
    T=motion['T_camera_wrist'][frame];local=motion['manus_local_25_m'][frame]
    xyz=np.einsum('sij,skj->ski',T[:,:3,:3],local)+T[:,None,:3,3]
    uv,good=project(xyz,motion['camera_K'])
    controller,cgood=project(motion['T_camera_controller'][frame,:,:3,3],motion['camera_K'])
    wrist,wgood=project(T[:,:3,3],motion['camera_K'])
    head=np.asarray(motion['head_valid'])
    wrist_mask=np.asarray(motion['wrist_camera_valid'])
    joint_mask=np.asarray(motion['joint_camera_valid_25'])
    # Legacy compose_motion explicitly stores wrist_world_valid=cv.copy(),
    # where cv is controller pose computability. It is NOT direct observation.
    controller_mask=np.asarray(motion['provenance_v2_controller_pose_computable'] if 'provenance_v2_controller_pose_computable' in motion else motion['wrist_world_valid'])
    for name,mask,shape in [('head',head,(165,)),('controller',controller_mask,(165,2)),
                            ('wrist',wrist_mask,(165,2)),('joint',joint_mask,(165,2,25))]:
        if mask.dtype!=np.bool_ or mask.shape!=shape:
            raise ValueError('invalid independent validity mask: '+name)
    controller_finite=np.isfinite(motion['T_world_controller'][frame]).all(axis=(-1,-2))
    cgood &= head[frame] & controller_mask[frame] & controller_finite
    wgood &= head[frame] & wrist_mask[frame] & controller_mask[frame] & controller_finite
    good &= wgood[:,None] & joint_mask[frame]
    controller[~cgood]=np.nan;wrist[~wgood]=np.nan;uv[~good]=np.nan
    return dict(skeleton_uv=uv,skeleton_valid=good,controller_uv=controller,controller_valid=cgood,
                wrist_uv=wrist,wrist_valid=wgood,manus_root_uv=uv[:,0],manus_root_valid=good[:,0])


def proxy_errors(points,annotations,frame):
    rows=[]
    for annotation in annotations['points']:
        if annotation['session']!='play_cards_0916_097' or annotation['frame']!=frame:continue
        side={'left':0,'right':1}[annotation['side']];node=annotation['node_id']
        valid=bool(annotation['visible'] and points['skeleton_valid'][side,node])
        error=float(np.linalg.norm(points['skeleton_uv'][side,node]-annotation['uv_1280'])) if valid else None
        rows.append(dict(frame_id=frame,side=annotation['side'],node_id=node,
          discrepancy_px=error,annotation_sigma_px=annotation['sigma_px'],valid=valid,
          interpretation='BONE_NODE_VS_GLOVE_SURFACE_PROXY_NOT_SAME_PHYSICAL_POINT',
          independent_accuracy=False,quality_pass=None))
    return rows
