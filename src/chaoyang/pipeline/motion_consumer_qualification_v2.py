"""Bound HaWoR consumer eligibility, not model inference or quality adoption."""
import numpy as np
from chaoyang.pipeline.motion_invalid_propagation_v1 import Policy,propagate

SCHEMA='MOTION_CONSUMER_QUALIFICATION_V2'
def qualify(hand,robot,observability,qualification,source_sha256,session_id):
    if qualification.get('schema_version')!=SCHEMA or qualification.get('session_id')!=session_id or qualification.get('source_sha256')!=source_sha256:raise ValueError('QUALIFICATION_BINDING')
    frames=np.asarray(hand['original_frame_indices']);n=len(frames)
    if not np.array_equal(frames,robot['frame_id']) or hand['anatomical_side_names'].tolist()!=['left','right']:raise ValueError('CONSUMER_FRAME_SIDE')
    if not np.array_equal(hand['timestamp_ns'],robot['timestamp_ns']) or not np.array_equal(robot['human_to_physical'],[0,1]):raise ValueError('CONSUMER_TIME_OR_SIDE_MAPPING')
    lookup={int(frame):i for i,frame in enumerate(frames)}
    rows=observability['diagnostic']['frame_rows']
    if observability.get('session_id')!=session_id or observability['inputs']['source']['sha256']!=source_sha256:raise ValueError('OBSERVABILITY_BINDING')
    decisions=propagate(rows,Policy())['rows'];diagnostic={}
    for row,decision in zip(rows,decisions):
        key=(row['frame_id'],row['side'])
        if key in diagnostic or key[0] not in lookup:raise ValueError('DUPLICATE_OR_UNKNOWN_OBSERVABILITY_ROW')
        i=lookup[key[0]];s=['left','right'].index(key[1])
        for field in ['observed','inferred','predicted_valid']:
            if type(row[field]) is not bool or row[field]!=bool(hand[field][s,i]):raise ValueError('SOURCE_PROVENANCE_DRIFT')
        diagnostic[key]=decision
    certificates={}
    for cert in qualification.get('rows',[]):
        key=(cert['frame_id'],cert['side'])
        if key in certificates or key[0] not in lookup or key[1] not in ['left','right']:raise ValueError('DUPLICATE_OR_UNKNOWN_QUALIFICATION_ROW')
        certificates[key]=cert
    masks={name:np.zeros((n,2),bool) for name in ['position','rotation','fingers','robot_wrist','robot_fingers']}
    ledger=[]
    for i,frame in enumerate(frames):
        for s,side in enumerate(['left','right']):
            key=(int(frame),side);d=diagnostic.get(key);cert=certificates.get(key);reasons=[]
            predicted=bool(hand['predicted_valid'][s,i]);kind=cert.get('estimate_kind','UNKNOWN') if cert else qualification.get('producer_kind','UNKNOWN')
            for flag,flag_kind in [('short_gap_inferred','INTERPOLATION'),('held','HOLD'),('copied_from_other_side','COPY_OTHER_SIDE')]:
                if flag in hand:
                    value=np.asarray(hand[flag])
                    if value.shape!=(2,n) or value.dtype!=np.bool_:raise ValueError('SOURCE_ESTIMATE_FLAG_SHAPE')
                    if value[s,i]:kind=flag_kind
            if not predicted:reasons.append('SOURCE_MISSING_NO_SUBSTITUTION')
            if d is None:reasons.append('NO_INDEPENDENT_OBSERVABILITY_ROW')
            elif not d['consumer_masks']['diagnostic_overlay']:reasons.extend(d['reasons'])
            if not cert:reasons.append('QUALITY_QUALIFICATION_MISSING')
            elif kind in ['HOLD','COPY_OTHER_SIDE']:reasons.append('FORBIDDEN_'+kind)
            elif kind=='INTERPOLATION':reasons.append('INTERPOLATION_NOT_OBSERVED_SEPARATE_AUTHORITY_REQUIRED')
            elif kind!='MODEL_INFERENCE':reasons.append('UNKNOWN_ESTIMATE_KIND')
            approved=bool(cert and cert.get('review_status')=='APPROVED_FOR_CONSUMPTION' and cert.get('independent_of_model_prediction') is True and cert.get('evidence_refs'))
            if cert and not approved:reasons.append('INDEPENDENT_QUALITY_EVIDENCE_REQUIRED')
            if approved:
                for ref in cert['evidence_refs']:
                    if ref.get('sha256')==source_sha256 or not ref.get('path') or len(ref.get('sha256',''))!=64:raise ValueError('SELF_EVIDENCE_OR_MALFORMED_REF')
            allowed=predicted and d is not None and d['consumer_masks']['diagnostic_overlay'] and approved and kind=='MODEL_INFERENCE'
            for field in ['position','rotation','fingers']:
                masks[field][i,s]=bool(allowed and cert.get('capabilities',{}).get(field)=='PASS')
            position=float(robot['position_residual_mm'][i,s]);rotation=float(robot['rotation_residual_deg'][i,s])
            pose_pass=np.isfinite(position) and np.isfinite(rotation) and position<=20 and rotation<=15
            masks['robot_wrist'][i,s]=bool(masks['position'][i,s] and masks['rotation'][i,s] and robot['wrist_valid'][i,s] and pose_pass)
            masks['robot_fingers'][i,s]=bool(masks['fingers'][i,s] and robot['finger_valid'][i,s])
            if allowed and not pose_pass:reasons.append('EXISTING_20MM_15DEG_WRIST_GATE_FAILED')
            accepted=bool(masks['robot_wrist'][i,s] and masks['robot_fingers'][i,s])
            ledger.append(dict(frame_id=int(frame),side=side,estimate_kind=kind,source_observed=bool(hand['observed'][s,i]),source_inferred=bool(hand['inferred'][s,i]),
                physical_observation=False,qualified_model_inference=bool(allowed),robot_accepted=accepted,reasons=reasons,
                action='ROBOT_ELIGIBLE_NOT_QUALITY_ADOPTED' if accepted else 'RAW_IMAGE_FALLBACK_NO_NEUTRAL_NO_HOLD_NO_SIDE_COPY'))
    return dict(masks=masks,ledger=ledger)

def robot_payload(robot,qualified):
    """Pass through accepted values; unknown consumer data are NaN, never home."""
    result={k:np.array(v,copy=True) for k,v in robot.items()}
    result['target_valid']=np.asarray(robot['target_valid'],bool)&qualified['masks']['position']
    result['wrist_valid']=qualified['masks']['robot_wrist'].copy()
    result['finger_valid']=qualified['masks']['robot_fingers'].copy()
    for key,mask in [('q_arm',result['wrist_valid']),('q22',result['finger_valid']),('q_hand22',result['finger_valid']),('T_actual_root_cam',result['wrist_valid']),('T_target_root_cam',result['target_valid']),('actual21_camera',result['finger_valid']),('actual21_camera_m',result['finger_valid'])]:
        if key in result:
            result[key]=np.asarray(result[key],float).copy();result[key][~mask]=np.nan
    return result

def fallback_image(raw_image,frame_ledger):
    """Conservative raw fallback until all declared sides have a valid composite."""
    if not frame_ledger or any(not row['robot_accepted'] for row in frame_ledger):return np.array(raw_image,copy=True)
    return None  # Eligible for downstream compositor, not a fabricated Robot image.
