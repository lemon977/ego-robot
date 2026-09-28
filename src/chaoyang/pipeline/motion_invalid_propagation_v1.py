"""Candidate consumer policy; never edits source arrays or estimates missing motion."""
from dataclasses import dataclass,asdict
from copy import deepcopy
import math

VERSION='MOTION_INVALID_PROPAGATION_V1'
@dataclass(frozen=True)
class Policy:
    min_roi_side_px: float | None = None
    depth_ratio_floor: float | None = None
    threshold_basis: str = 'NO_NUMERIC_HEURISTICS'
    allow_inferred_robot_target: bool = False
    def __post_init__(self):
        if self.min_roi_side_px is not None and (not math.isfinite(self.min_roi_side_px) or self.min_roi_side_px<=0):raise ValueError('ROI_THRESHOLD')
        if self.depth_ratio_floor is not None and (not math.isfinite(self.depth_ratio_floor) or not 0<self.depth_ratio_floor<1):raise ValueError('DEPTH_RATIO_THRESHOLD')
        if (self.min_roi_side_px is not None or self.depth_ratio_floor is not None) and self.threshold_basis=='NO_NUMERIC_HEURISTICS':raise ValueError('THRESHOLD_BASIS_REQUIRED')
        if type(self.allow_inferred_robot_target) is not bool:raise ValueError('BOOLEAN_POLICY_REQUIRED')

def finite_number(x):return type(x) in (int,float) and math.isfinite(x)

def decide(row,policy=Policy(),previous=None):
    """Classify support independently per hand; flags do not assert physical truth."""
    if row.get('side') not in ('left','right') or type(row.get('frame_id')) is not int:raise ValueError('FRAME_SIDE_REQUIRED')
    for key in ['observed','inferred','predicted_valid','roi_valid']:
        if type(row.get(key)) is not bool:raise ValueError('BOOLEAN_SOURCE_REQUIRED:'+key)
    reasons=[];flags=[];invalid=False;unknown=False
    observed,inferred=row['observed'],row['inferred']
    if observed and inferred:invalid=True;reasons.append('CONFLICTING_PROVENANCE')
    if not row['predicted_valid']:unknown=True;reasons.append('SOURCE_OUTPUT_MISSING')
    if not observed and not inferred:unknown=True;reasons.append('SOURCE_PROVENANCE_UNKNOWN')
    inside,total=row.get('joints_inside_image'),row.get('joints_total')
    if type(inside) is not int or type(total) is not int or not 0<=inside<=total or total<=0:raise ValueError('PROJECTION_COUNTS')
    if inside==0:unknown=True;reasons.append('NO_PROJECTED_JOINT_SUPPORT')
    elif inside<total:flags.append('PARTIAL_IMAGE_SUPPORT_NOT_PER_JOINT_MASK')
    if not row['roi_valid']:unknown=True;reasons.append('ROI_MISSING')
    else:
        width,height=row.get('roi_width_px'),row.get('roi_height_px')
        if not finite_number(width) or not finite_number(height) or min(width,height)<=0:invalid=True;reasons.append('ROI_NONPOSITIVE_OR_NONFINITE')
        elif policy.min_roi_side_px is not None and min(width,height)<policy.min_roi_side_px:unknown=True;reasons.append('ROI_BELOW_REVIEW_ONLY_MIN_SIDE')
        if any(row.get('touches_'+edge) is True for edge in ['left','right','top','bottom']):flags.append('ROI_TOUCHES_BORDER_NOT_AUTOMATIC_INVALID')
    depth=row.get('root_depth_m');root=row.get('root_camera_m')
    if row['predicted_valid']:
        if not finite_number(depth) or depth<=0:invalid=True;reasons.append('NONPOSITIVE_OR_NONFINITE_CAMERA_DEPTH')
        if not isinstance(root,(list,tuple)) or len(root)!=3 or not all(finite_number(v) for v in root):invalid=True;reasons.append('NONFINITE_OR_MISSING_ROOT')
        elif finite_number(depth) and not math.isclose(root[2],depth,rel_tol=1e-6,abs_tol=1e-9):invalid=True;reasons.append('ROOT_DEPTH_FIELD_MISMATCH')
    if previous is not None:
        if previous.get('side')!=row['side']:raise ValueError('CROSS_SIDE_PREVIOUS_FORBIDDEN')
        if row['frame_id']<=previous['frame_id']:raise ValueError('NON_MONOTONIC_FRAME')
        adjacent=row['frame_id']==previous['frame_id']+1
        if not adjacent:flags.append('GAP_NO_DEPTH_COMPARISON')
        if row.get('tracking_reset') or previous.get('tracking_reset'):adjacent=False;flags.append('RESET_NO_DEPTH_COMPARISON')
        old=previous.get('root_depth_m')
        if adjacent and previous.get('predicted_valid') is True and row['predicted_valid'] and policy.depth_ratio_floor is not None and finite_number(depth) and finite_number(old) and min(depth,old)>0 and min(depth,old)/max(depth,old)<policy.depth_ratio_floor:
            unknown=True;reasons.append('DEPTH_DISCONTINUITY_REQUIRES_REVIEW_NOT_PROVEN_COLLAPSE')
    support=not invalid and not unknown
    status='INVALID' if invalid else 'UNKNOWN' if unknown else 'OBSERVED_SUPPORTED' if observed else 'INFERRED_SUPPORTED'
    return dict(schema_version=VERSION,frame_id=row['frame_id'],side=row['side'],status=status,
        source_observed=observed,source_inferred=inferred,source_predicted_valid=row['predicted_valid'],
        effective_observed=bool(support and observed),effective_inferred=bool(support and inferred),
        consumer_masks=dict(diagnostic_overlay=support,robot_position_target=bool(support and (observed or policy.allow_inferred_robot_target)),
                            wrist_rotation=False,fingers=False,surface=False),
        reasons=reasons,flags=flags,missing_display_action='EXPLICIT_MISSING_MARKER_KEEP_FRAME',
        note='Position/image-support policy cannot certify rotation, individual joints, surface, or physical accuracy.')

def propagate(rows,policy=Policy()):
    result=[];previous={}
    for row in rows:
        side=row.get('side');item=decide(row,policy,previous.get(side));result.append(item);previous[side]=row
    return dict(schema_version=VERSION,policy=asdict(policy),rows=result,input_row_count=len(rows),output_row_count=len(result),
                source_modified=False,missing_frames_removed=False,interpolation_performed=False,hold_performed=False)

def consumer_payload(value,decision,consumer):
    """No previous state argument: missing values cannot become neutral/held motion."""
    if consumer not in decision['consumer_masks']:raise ValueError('UNKNOWN_CONSUMER')
    if not decision['consumer_masks'][consumer]:return dict(frame_id=decision['frame_id'],side=decision['side'],value=None,valid=False,action=decision['missing_display_action'])
    if not isinstance(value,(list,tuple)) or not value or not all(finite_number(x) for x in value):raise ValueError('FINITE_EXPLICIT_VECTOR_REQUIRED')
    return dict(frame_id=decision['frame_id'],side=decision['side'],value=deepcopy(list(value)),valid=True,
                source_observed=decision['source_observed'],source_inferred=decision['source_inferred'])
