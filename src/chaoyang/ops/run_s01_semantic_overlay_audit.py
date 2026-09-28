"""Registered 12-frame semantic overlay of frozen C3; never fits any parameter."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from chaoyang.ops.run_four_stream_completion_assembly_canary import authority
from chaoyang.ops.run_four_stream_completion_sensor import read_pinned,checked_path
from chaoyang.pipeline.s01_overlay_consumer_v1 import FRAMES,consume,projected_points,proxy_errors


def draw_frame(raw,points,motion,frame,housing,glove,font):
    import cv2
    from PIL import Image,ImageDraw
    overlay=raw.copy();colors=[(255,120,0),(0,80,255)]
    def xy(uv):return tuple(np.rint(uv).astype(int))
    def onscreen(uv):return np.isfinite(uv).all() and 0<=uv[0]<1280 and 0<=uv[1]<960
    for side,color in enumerate(colors):
        for child,parent in enumerate(motion['manus_parent_ids']):
            if child==parent:continue
            if not(points['skeleton_valid'][side,child] and points['skeleton_valid'][side,parent]):continue
            a,b=points['skeleton_uv'][side,[child,parent]]
            if max(np.abs(a).max(),np.abs(b).max())>1e6:continue
            ok,a,b=cv2.clipLine((0,0,1280,960),xy(a),xy(b))
            if ok:cv2.line(overlay,a,b,color,2,cv2.LINE_AA)
        for key,marker in [('controller_uv',cv2.MARKER_CROSS),('wrist_uv',cv2.MARKER_SQUARE),('manus_root_uv',cv2.MARKER_DIAMOND)]:
            uv=points[key][side]
            if onscreen(uv):cv2.drawMarker(overlay,xy(uv),color,marker,18,2)
    for row in housing['sessions']['play_cards_0916_097']:
        if row['frame']!=frame:continue
        for side,visible in enumerate(row['visible']):
            if visible:cv2.circle(overlay,xy(np.array(row['housing_center_uv_640'][side])*2),16,(0,255,255),2)
    for row in glove['points']:
        if row['session']=='play_cards_0916_097' and row['frame']==frame and row['visible']:
            cv2.circle(overlay,xy(row['uv_1280']),int(row['sigma_px']),(50,220,50),1)
    canvas=Image.new('RGB',(1280,650),(20,20,20))
    for x,panel in [(0,raw),(640,overlay)]:
        canvas.paste(Image.fromarray(panel[:,:,::-1]).resize((640,480)),(x,72))
    d=ImageDraw.Draw(canvas)
    text=[(12,8,'左：原图；右：旧C3语义诊断（非已标定腕、非采用结果）'),
          (12,38,f'源帧 {frame}  采集时间 {float(motion["timestamp_s"][frame]):.6f}秒；左手蓝，右手红'),
          (12,560,'十字=追踪原点；黄圈=外壳参考；方框=人体腕估计；菱形=MANUS虚拟根'),
          (12,590,'绿圈=旧手套指尖代理±20像素；不是骨骼/腕真值；出画点不强行挪回'),
          (12,620,'旧拟合元数据矛盾：拒绝正常采用；仅保留诊断；未新增拟合')]
    for x,y,value in text:d.text((x,y),value,font=font,fill=(245,245,245))
    return canvas


def run(config):
    root,lane,index,ticks=authority(config)
    if config['frame_indices']!=list(FRAMES):raise ValueError('frame scope changed')
    refs=config['references'];data={k:read_pinned(v,root) for k,v in refs.items()}
    contract,fit_config,fit,housing,glove=[json.loads(data[k]) for k in ('contract','fit_config','fit','housing','glove')]
    registered={e['path']:e for e in contract['evidence']}
    for key in ('fit_config','fit','housing','glove'):
        if refs[key] != registered.get(refs[key]['path']):raise ValueError('contract evidence drift')
    predecessor=json.loads(data['predecessor'])
    if predecessor['hand_motion']!=refs['motion'] or predecessor['source_video']!=config['video']:
        raise ValueError('motion/video predecessor drift')
    with np.load(io.BytesIO(data['motion']),allow_pickle=False) as z:motion={k:z[k] for k in z.files}
    state=consume(contract,fit_config,fit,motion,config['mode'])
    prereg=json.loads(data['preregistration'])
    if prereg['frame_indices']!=list(FRAMES) or prereg['new_fit_allowed'] is not False:raise ValueError('preregistration mismatch')
    code={str(checked_path(ref['path'],root)) for ref in config['code_closure']}
    required=['ops/run_s01_semantic_overlay_audit.py','pipeline/s01_overlay_consumer_v1.py',
              'pipeline/s01_point_semantics_v1.py','ops/run_four_stream_completion_assembly_canary.py',
              'ops/run_four_stream_completion_sensor.py']
    if not {str(root/'src/chaoyang'/p) for p in required}.issubset(code):raise ValueError('incomplete code closure')
    for ref in config['code_closure']:read_pinned(ref,root)
    read_pinned(config['video'],Path('/mnt/data/egodata/datasets/ego'))
    read_pinned(config['font'],Path('/usr/share/fonts'))
    out=lane/'S01_SEMANTIC_OVERLAY_V1'
    if out.exists() or out.is_symlink():raise FileExistsError(out)
    import cv2
    from PIL import ImageFont
    cv2.setNumThreads(2);font=ImageFont.truetype(config['font']['path'],20)
    cap=cv2.VideoCapture(config['video']['path'])
    if not cap.isOpened():raise ValueError('source video unavailable')
    if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))!=(4096,1536,165):
        cap.release();raise ValueError('source video domain or timeline mismatch')
    out.mkdir();rows=[];artifacts=[]
    try:
        for frame in FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,stereo=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))!=frame+1:raise ValueError('decode frame mapping failed')
            raw=cv2.resize(stereo[:,2048:],(1280,960));points=projected_points(motion,frame)
            panel=draw_frame(raw,points,motion,frame,housing,glove,font);p=out/f'FRAME_{frame:06d}.png';panel.save(p)
            b=p.read_bytes();artifacts.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
            rows.append(dict(frame_id=frame,timestamp_ns=int(motion['timestamp_ns'][frame]),
                points={k:np.where(np.isfinite(v),v,np.nan).tolist() for k,v in points.items() if k.endswith('_uv')},
                proxy_discrepancies=proxy_errors(points,glove,frame)))
    finally:cap.release()
    if authority(config)[2:]!=(index,ticks):raise ValueError('authority changed')
    for ref in list(refs.values())+config['code_closure']:read_pinned(ref,root)
    read_pinned(config['video'],Path('/mnt/data/egodata/datasets/ego'))
    result=dict(schema_version='S01_SEMANTIC_OVERLAY_AUDIT_V1',execution='EXECUTED',consumer=state,
       quality='NOT_EVALUATED',adoption='NOT_ADOPTED',new_fit=False,collision_adoption=False,
       inputs=config,frames=rows,artifacts=artifacts,independent_accuracy=False,gpu_used=False)
    # JSON null preserves invalid projections; never replace unknowns with zero.
    def finite(x):
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,list):return [finite(v) for v in x]
        if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
        return x
    result=finite(result)
    with (out/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    args=p.parse_args(argv);result=run(json.loads(args.config.read_text()))
    print(json.dumps({'execution':result['execution'],'adoption':result['adoption']}));return 0


if __name__=='__main__':raise SystemExit(main())
