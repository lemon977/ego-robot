from __future__ import annotations
import copy
import cv2
import numpy as np
TAGS=frozenset({'NEAR_HUMAN_ROLE','NEAR_EQUIPMENT_ROLE','NEAR_BOTH','NO_SUPPORT','UNKNOWN'})
NEAR_PX=12.0
def need(v,s):
 if not v:raise ValueError(s)
def classify(human,equipment,protected,cup,image_evidence_valid):
 arrays=[np.asarray(x) for x in [human,equipment,protected,cup]]
 need(all(x.dtype==np.bool_ and x.shape==(480,640) for x in arrays),'ROLE_CANONICAL_BOOL_MASK_DOMAIN')
 need(type(image_evidence_valid) is bool,'ROLE_IMAGE_VALIDITY_BOOL')
 h,e,p,c=arrays
 need(not np.any(h&e),'ROLE_ORIGINAL_HUMAN_EQUIPMENT_NOT_DISJOINT')
 need(np.array_equal(p,c&~(h|e)),'ROLE_ONLY_ORIGINAL_V9_P_POLICY')
 visible=bool(image_evidence_valid and c.any() and p.any())
 nh=ne=False
 if visible:
  near=cv2.distanceTransform((~p).astype(np.uint8),cv2.DIST_L2,5)<=NEAR_PX
  nh=bool((h&near).any());ne=bool((e&near).any())
  tag='NEAR_BOTH' if nh and ne else 'NEAR_HUMAN_ROLE' if nh else 'NEAR_EQUIPMENT_ROLE' if ne else 'NO_SUPPORT'
  why=None
 else:
  tag='UNKNOWN'
  why='IMAGE_MASK_EVIDENCE_UNAVAILABLE' if not image_evidence_valid else 'CUP_OBSERVED_MASK_EMPTY' if not c.any() else 'PROTECTED_VISIBLE_MASK_EMPTY'
 return dict(role_2D_mask_evidence=tag,near_human_role=nh,near_equipment_role=ne,unknown_reason=why,
             mask_area_pixels=dict(sam_core=int(h.sum()),equipment=int(e.sum()),protected_visible=int(p.sum()),cup_observed=int(c.sum())),
             physical_contact='UNKNOWN',occlusion_direction='UNKNOWN',
             equipment_only_false_contact_claim=False,projected_fingertips_used=False)
def validate_qpc(values,n):
 need(isinstance(values,list) and len(values)==n and all(type(x) is int and 0<=x<=np.iinfo(np.int64).max for x in values),'ROLE_ORIGINAL_INTEGER_QPC_NO_SDK_CAST')
 need(all(b>a for a,b in zip(values,values[1:])),'ROLE_QPC_NOT_STRICT')
 return values
def windows(qpc,triggers):
 validate_qpc(qpc,len(qpc));need(bool(qpc),'ROLE_EMPTY_QPC')
 need(all(type(x) is int and 0<=x<len(qpc) for x in triggers),'ROLE_TRIGGER_NATIVE_DOMAIN')
 times=np.array([x-qpc[0] for x in qpc],dtype=np.float64)/1e9;out=[]
 for f in sorted(set(triggers)):
  a=max(0.,float(times[f])-.5);b=min(float(times[-1]),float(times[f])+.5)
  if out and a<=out[-1]['end_time_s']+.1+1e-12:
   out[-1]['end_time_s']=max(out[-1]['end_time_s'],b);out[-1]['role_trigger_frames'].append(f)
  else:out.append(dict(start_time_s=a,end_time_s=b,role_trigger_frames=[f]))
 for row in out:
  row['start_frame']=int(np.searchsorted(times,row['start_time_s'],side='left'))
  row['end_frame_inclusive']=int(np.searchsorted(times,row['end_time_s'],side='right')-1)
 return out
def preserve_frame(frame,qpc,index):
 need(type(frame.get('frame_index')) is int and frame['frame_index']==index and type(frame.get('raw_frame_index')) is int and frame['raw_frame_index']==index and type(frame.get('raw_qpc_ns')) is int and frame['raw_qpc_ns']==qpc,'ROLE_SOURCE_FRAME_NATIVE_QPC')
 need(set(frame.get('hands',{}))=={'left','right'},'ROLE_SOURCE_SIDES_REQUIRED')
 for side in ['left','right']:
  for key in ['raw_observed','display_valid','diagnostic_display','tracking_gate_applicable','qpc_velocity_violation','qpc_acceleration_violation']:
   need(type(frame['hands'][side].get(key)) is bool,'ROLE_SOURCE_OR_NUM_FLAG_BOOL:'+side+':'+key)
 need(type(frame.get('camera_raw_valid')) is bool,'ROLE_CAMERA_VALIDITY_FLAG')
 return copy.deepcopy(frame)
