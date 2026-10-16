"""Explicit new-policy adapter; no fit, model or geometry change."""
import numpy as np
from .contract import need,read,pin
FIELDS=('core','sam_core','equipment','removal','cup_observed','semantic_conflict','protected_visible','held','unknown','effective_removal')
def arrays(v):
 import hashlib,os
 from .contract import path
 p=path(v['path']);before=p.stat();ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 with p.open('rb') as fd:
  need(before.st_size==v['bytes'],'FR_NPZ_SIZE');h=hashlib.sha256()
  for b in iter(lambda:fd.read(1<<20),b''):h.update(b)
  need(h.hexdigest()==v['sha256'],'FR_NPZ_SAME_FD_SHA');fd.seek(0)
  with np.load(fd,allow_pickle=False) as z:result={k:z[k] for k in z.files}
  need(ident(before)==ident(os.fstat(fd.fileno()))==ident(p.stat()),'FR_NPZ_CHANGED_DURING_LOAD')
 return result
def require_session_scope(session,n):
 need(type(n) is int and session.frame_count==n and session.qpc_ns.dtype.kind in 'iu' and session.qpc_ns.shape==(n,),'FR_SOURCE_INTEGER_QPC')
 need(n>1 and all(int(session.qpc_ns[i+1])>int(session.qpc_ns[i]) for i in range(n-1)),'FR_SOURCE_QPC_STRICT_MONOTONIC')
 return True
def load_case(c,tick):
 from chaoyang.pick_place_1008 import source
 tick();session=source.load_prepared(__import__('pathlib').Path(c['frozen_identity_refs']['source_session_input']['path']).parent)
 n=c['native_frames'];require_session_scope(session,n)
 motion=arrays(c['frozen_identity_refs']['actions']);need(motion['qpc_ns'].dtype.kind in 'iu' and np.array_equal(motion['qpc_ns'],session.qpc_ns),'FR_SOURCE_MOTION_INTEGER_QPC_EXACT');raw=arrays(c['old_GPU_inputs']['raw_rgb.npz'])['rgb'];clean=arrays(c['actual_clean_files']['candidate_clean_rgb.npz'])['rgb']
 oldpacked=arrays(c['old_GPU_inputs']['masks.npz']);packed=arrays(c['actual_clean_files']['candidate_masks.npz']);n=c['native_frames']
 need(raw.dtype==clean.dtype==np.uint8 and raw.shape==clean.shape==(n,480,640,3),'FR_RGB_UINT8_N')
 need(set(oldpacked)==set(packed)==set(FIELDS),'FR_TEN_FIELDS')
 masks={}
 for k in FIELDS:
  tick();need(packed[k].dtype==oldpacked[k].dtype==np.uint8 and packed[k].shape==oldpacked[k].shape==(n,38400),'FR_PACKED_BIG_UINT8')
  if k not in ['protected_visible','effective_removal']:need(np.array_equal(packed[k],oldpacked[k]),'FR_OLD8_BITEXACT:'+k)
  masks[k]=np.unpackbits(packed[k],axis=1,count=307200,bitorder='big').reshape(n,480,640).astype(bool)
 need(np.array_equal(masks['protected_visible'],masks['cup_observed']) and np.array_equal(masks['semantic_conflict'],masks['cup_observed']&masks['core']),'FR_NEW_P_SEMANTIC_SCHEMA')
 need(np.array_equal(masks['unknown'],masks['cup_observed']&~masks['core']) and not np.any(masks['held']&~masks['unknown']),'FR_LEGACY_UNKNOWN_HELD_NOT_NEW_TRUTH')
 oldE=np.unpackbits(oldpacked['effective_removal'],axis=1,count=307200,bitorder='big').reshape(n,480,640).astype(bool)
 need(np.array_equal(masks['effective_removal'],oldE&~masks['cup_observed']),'FR_NEW_E_SCHEMA')
 masks['raw_rgb']=raw;masks['metadata']=read(c['actual_clean_files']['candidate_masks.json']);masks['motion_info']=read(c['frozen_identity_refs']['motion_info']);masks['baseline_quality_problems']=c['quality_problems']
 masks['upstream_refs']={'actions':c['frozen_identity_refs']['actions'],'frame_mapping':c['frozen_identity_refs']['frame_mapping'],'motion_info':c['frozen_identity_refs']['motion_info'],'raw_rgb':c['old_GPU_inputs']['raw_rgb.npz'],'clean_rgb':c['actual_clean_files']['candidate_clean_rgb.npz'],'masks':c['actual_clean_files']['candidate_masks.npz'],'mask_info':c['actual_clean_files']['candidate_masks.json']}
 need(session.frame_count==n and session.qpc_ns.dtype.kind in 'iu' and session.qpc_ns.shape==(n,),'FR_SOURCE_INTEGER_QPC')
 return session,motion,clean,masks
def annotate_contact_roles(contact,masks):
 from chaoyang.pick_place_1008 import visual
 note='Legacy HUMAN_CUP_IMAGE_* is wearer union(sam_core OR equipment) 2D evidence; not human-only or physical contact truth.'
 contact['legacy_alias_semantics']=note
 contact['evidence_role_schema']='model_mask_contact_role_evidence_newP_sidecar_v1'
 for f,row in enumerate(contact['frames']):
  cup=masks['protected_visible'][f]
  if not cup.any():role='UNKNOWN'
  else:
   near=visual._near(cup,visual.CONTACT_PX*visual._contact_scale(cup.shape));human=bool((masks['sam_core'][f]&near).any());eq=bool((masks['equipment'][f]&near).any())
   role='NEAR_BOTH' if human and eq else 'NEAR_HUMAN_ROLE' if human else 'NEAR_EQUIPMENT_ROLE' if eq else 'NO_SUPPORT'
  row['evidence_role']=role;row['legacy_alias_semantics']=note;row['evidence_role_domain']='2D_MODEL_MASK_TO_NEW_MODEL_CUP_PROTECTION_NOT_PHYSICAL_CONTACT'
 return contact
