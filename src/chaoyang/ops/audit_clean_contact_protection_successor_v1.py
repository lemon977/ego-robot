#!/usr/bin/env python3
"""Audit and visualize a bounded contact-aware Clean removal-mask successor.

This does not publish a new Clean authority. It compares the current broad
dilation with a smaller object-aware proposal and verifies what the existing
donor provenance can and cannot reject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font_path():
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file(): return path
    return None


def text(image, rows):
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil); cache={}
    for x,y,value,color,size in rows:
        cache.setdefault(size,ImageFont.truetype(font_path(),size) if font_path() else ImageFont.load_default())
        draw.text((x,y),value,font=cache[size],fill=(color[2],color[1],color[0]))
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def artifact(path):
    p=Path(path).resolve(strict=True); return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)}


def mask(path):
    if isinstance(path, dict):
        path = path['path']
    value=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if value is None: raise RuntimeError(f'cannot read {path}')
    return value>0


def dilate(value, radius):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    return cv2.dilate(value.astype(np.uint8),kernel)>0


def unions(row):
    human=None; tracker=None
    for key,ref in row['role_masks'].items():
        current=mask(ref['path'])
        if key.endswith('human'): human=current if human is None else human|current
        else: tracker=current if tracker is None else tracker|current
    obj=np.zeros_like(human)
    for key,ref in row.items():
        if key.startswith('physical_object_') and isinstance(ref,dict): obj|=mask(ref['path'])
    return human,tracker,obj


def proposed(human,tracker,obj):
    near=dilate(obj,20)
    small=dilate(human,4)|dilate(tracker,8)
    far=dilate(human,8)|dilate(tracker,20)
    result=np.where(near,small,far)
    return result & ~obj


def select_frames(rows,count):
    scores=[]
    for i,row in enumerate(rows):
        human,tracker,obj=unions(row); current=mask(row['clean_removal_object_protected'])
        contact=int((current & dilate(obj,16)).sum())
        expansion=max(0,int(current.sum())-int((human|tracker).sum()))
        scores.append((contact+0.15*expansion,i))
    picked=[]
    for _,i in sorted(scores,reverse=True):
        if all(abs(i-j)>=8 for j in picked): picked.append(i)
        if len(picked)>=count: break
    return sorted(picked)


def overlay(raw,current,proposal,obj):
    result=raw.copy()
    current_edge=current & ~cv2.erode(current.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    proposal_edge=proposal & ~cv2.erode(proposal.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    object_edge=obj & ~cv2.erode(obj.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    result[current_edge]=(0,0,255); result[proposal_edge]=(0,255,255); result[object_edge]=(0,255,0)
    return result


def resize(image,w=480,h=360): return cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)


def audit_one(name,manifest_path,donor_manifest_path,clean_result_path,out,count):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8')); donor=json.loads(Path(donor_manifest_path).read_text(encoding='utf-8'))
    rows=manifest['frames']; selected=select_frames(rows,count); tiles=[]; metrics=[]
    valid_object_frames=0
    for row in rows:
        _,_,obj=unions(row); valid_object_frames += int(obj.any())
    for index in selected:
        row=rows[index]; human,tracker,obj=unions(row); current=mask(row['clean_removal_object_protected']); prop=proposed(human,tracker,obj)
        raw=cv2.imread(row['source_rgb']['path']); clean=cv2.imread(donor['frames'][index]['clean_rgb']['path'])
        if raw is None or clean is None: raise RuntimeError('missing image')
        curr_vis=raw.copy(); curr_vis[current]=(.55*curr_vis[current]+.45*np.array([0,0,255])).astype(np.uint8)
        prop_vis=raw.copy(); prop_vis[prop]=(.55*prop_vis[prop]+.45*np.array([0,255,255])).astype(np.uint8)
        diagnostic=overlay(raw,current,prop,obj)
        strip=np.hstack([resize(raw),resize(curr_vis),resize(prop_vis),resize(diagnostic)])
        strip=text(strip,[(8,8,f'{name} 帧 {index}',(255,255,255),20),(490,8,'当前大膨胀/红',(255,255,255),20),(970,8,'建议接触保护/黄',(255,255,255),20),(1450,8,'边界：红当前 黄建议 绿物体',(255,255,255),18)])
        tiles.append(strip)
        metrics.append({'frame':index,'original_role_pixels':int((human|tracker).sum()),'current_removal_pixels':int(current.sum()),'proposed_removal_pixels':int(prop.sum()),'current_added_pixels':int((current&~(human|tracker)).sum()),'proposed_added_pixels':int((prop&~(human|tracker)).sum()),'object_pixels':int(obj.sum()),'current_object_overlap':int((current&obj).sum()),'proposed_object_overlap':int((prop&obj).sum())})
    montage=np.vstack(tiles)
    montage_path=out/f'{name}_Clean接触保护Mask_successor_canary.png'; cv2.imwrite(str(montage_path),montage)
    current_added=sum(m['current_added_pixels'] for m in metrics); proposed_added=sum(m['proposed_added_pixels'] for m in metrics)
    return {
        'session':manifest['session'],'task':manifest['task'],'selected_frames':selected,
        'object_observed_frame_coverage':valid_object_frames/len(rows),
        'selected_frame_metrics':metrics,
        'selected_current_added_pixels':current_added,
        'selected_proposed_added_pixels':proposed_added,
        'selected_added_pixel_reduction_ratio':1-proposed_added/max(current_added,1),
        'montage':artifact(montage_path),'inputs':{'expanded_manifest':artifact(manifest_path),'donor_manifest':artifact(donor_manifest_path),'clean_result':artifact(clean_result_path)}
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,required=True); args=p.parse_args(); out=args.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    roots={
      'Poker245':(
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json',
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/real_donor_v1/play_cards_0903_245/SOURCE_MAP_MANIFEST.json',
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/propainter_v1/play_cards_0903_245/RESULT.json'),
      'Chips039':(
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json',
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/real_donor_v1/get_potato_chips_0902_039/SOURCE_MAP_MANIFEST.json',
       'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_039/RESULT.json')}
    reports=[audit_one(name,*values,out,6) for name,values in roots.items()]
    result={
      'schema_version':'clean-contact-protection-successor-canary-v1','status':'GO_FOR_BOUNDED_CLEAN_SUCCESSOR_NOT_AUTHORITY','method':{'contact_band_px':20,'human_dilation_near_px':4,'tracker_dilation_near_px':8,'human_dilation_far_px':8,'tracker_dilation_far_px':20,'visible_object_byte_exact_protection':True},
      'reports':reports,
      'diagnosis':[
       'The current 18-24px human and 60px tracker dilation removes too much context around contacts.',
       'Visible object protection only works on frames where the object identity mask is observed; it cannot reconstruct hidden object appearance.',
       'The current identical-coordinate temporal donor has no semantic class for plates or other non-task distractors, so its two-donor RGB consensus can copy plate pixels while still passing its declared gates.',
       'A publishable successor must add causal geometric/flow warp and semantic support-surface rejection, plus a temporal object atlas for hidden card/chip appearance.'
      ],
      'next_gate':'Run fresh Poker245 and Chips039 Clean successor with the proposed mask, semantic donor rejection, and byte-exact visible-object gate; compare full-session reviews before expansion.',
      'claim_limit':'Mask and provenance diagnostic only; no new Clean, Robotized RGB, occlusion, contact, or physical authority.'
    }
    path=out/'RESULT.json'; path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'result':str(path),'reports':reports},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
