"""Fixed-parameter sensor-chain regression, not robot IK or physical accuracy certification."""
import json
from pathlib import Path
import subprocess
import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.pico_manus_motion_v2 import pose_matrices, installation_matrices, require_transform

TASK='sensor0916_generalization_visuals_retry1_20260930'
ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
DATA=Path('/path/to/private_data/egodata/datasets/ego')
RAW=DATA/'chips_cards_handle_highview_0916/cards_130_0916'
PROC=DATA/'processed/chips_cards_handle_highview_0916/cleaned/playing_cards'
IDS=('025','097','098','101')
EDGES=[(a,a+1) for start,end in ((1,4),(5,9),(10,14),(15,19),(20,24)) for a in range(start,end)]
EDGES += [(1,5),(5,10),(10,15),(15,20)]
COLORS=((235,130,30),(40,50,235))
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',23)

def text(img,lines):
    p=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p)
    for s,xy in lines:d.text(xy,s,font=FONT,fill=(20,20,20))
    return cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)

def project(p,k):
    valid=np.isfinite(p).all(-1)&(p[...,2]>1e-4)
    uv=np.full(p.shape[:-1]+(2,),np.nan)
    uv[valid]=(p[valid]@k.T)[:,:2]/p[valid,2,None]
    return uv,valid

def encoder(path,size):
    return subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)

def pt(p):return tuple(np.rint(p).astype(int))
def skeleton(img,uv,valid,color):
    for a,b in EDGES:
        if valid[a] and valid[b]:cv2.line(img,pt(uv[a]),pt(uv[b]),color,2,cv2.LINE_AA)
    for j in range(1,25):
        if valid[j]:cv2.circle(img,pt(uv[j]),3,color,-1,cv2.LINE_AA)

def main():
    manifest=load_json(ROOT/'INPUTS.json')
    for ref in manifest['files']:
        assert artifact_ref(Path(ref['path']))==ref,'SOURCE_DRIFT'
    with h5py.File(RAW/'025/dataset.hdf5','r') as f:prior=json.loads(f.attrs['controller_to_wrist_calibration'])
    install=installation_matrices(prior)
    basecam=load_json(PROC/'play_cards_0916_025/camera_params.json')
    baseclip=load_json(PROC/'play_cards_0916_025/clip_manifest.json')
    fixedextr=require_transform(basecam['selected_effective_rep103_head_to_camera'])
    fixedk=np.asarray(baseclip['video_rectification']['intrinsics'])
    allresults=[]
    for sid in IDS:
        session='play_cards_0916_'+sid;out=ROOT/session;out.mkdir(exist_ok=False)
        folder=PROC/session;clip=load_json(folder/'clip_manifest.json');cam=load_json(folder/'camera_params.json')
        with h5py.File(RAW/sid/'dataset.hdf5','r') as f:
            head,hv=pose_matrices(f['head_pose'][:]);ctrl,cv=pose_matrices(np.stack([f[s+'_controller_pose'][:] for s in ('left','right')],1))
            stored,sv=pose_matrices(np.stack([f[s+'_wrist_pose'][:] for s in ('left','right')],1))
            joints=np.stack([f[s+'_hand_joints'][:] for s in ('left','right')],1)
            hand=np.stack([f[s+'_hand_valid'][:] for s in ('left','right')],1).astype(bool)
            ts=f['timestamp_ns'][:];fid=f['video_frame_idx'][:]
            native=installation_matrices(json.loads(f.attrs['controller_to_wrist_calibration']))
            assert f.attrs['hand_frame']=='wrist-local (relative to MANUS wrist root)'
            source_quality={k:str(f.attrs[k]) for k in ('pre_filter_frame_count','dropped_incomplete_frame_count','alignment_quality','max_skew_ms','n_segments') if k in f.attrs}
        n=len(ts);dt=np.diff(ts.astype(np.int64))*1e-9
        assert np.all(dt>0) and np.median(dt)>0.001,'TIME_INVALID'
        np.testing.assert_array_equal(fid,np.arange(n))
        assert clip['video']['frame_count']==n
        assert clip['selection']['source_calibration_key']==cam['selected_calibration_key']=='right'
        np.testing.assert_allclose(native,install,atol=1e-12,rtol=0)
        np.testing.assert_allclose(cam['selected_effective_rep103_head_to_camera'],fixedextr,atol=1e-12,rtol=0)
        np.testing.assert_allclose(clip['video_rectification']['intrinsics'],fixedk,atol=1e-12,rtol=0)
        assert clip['video_rectification']['image_domain_mode']=='passthrough_scaled_source_domain'
        wrist=ctrl@install[None]
        np.testing.assert_allclose(wrist[sv&cv],stored[sv&cv],atol=1e-6,rtol=0)
        world=np.einsum('tsij,tskj->tski',wrist[:,:,:3,:3],joints)+wrist[:,:,None,:3,3]
        inv=np.full_like(head,np.nan);inv[hv]=np.linalg.inv(head[hv]);tcw=fixedextr@inv
        points=np.einsum('tij,tskj->tski',tcw[:,:3,:3],world)+tcw[:,None,None,:3,3]
        uv,pv=project(points,fixedk)
        cp=np.einsum('tij,tsj->tsi',tcw[:,:3,:3],ctrl[:,:,:3,3])+tcw[:,None,:3,3]
        cuv,cpv=project(cp,fixedk)
        wp=np.einsum('tij,tsj->tsi',tcw[:,:3,:3],wrist[:,:,:3,3])+tcw[:,None,:3,3]
        wuv,wpv=project(wp,fixedk)
        jvalid=hand[:,:,None]&cv[:,:,None]&np.isfinite(joints).all(-1);jvalid[:,:,0]=False
        valid=jvalid&hv[:,None,None]&pv
        width,height=clip['video']['width'],clip['video']['height']
        inside=valid&(uv[...,0]>=0)&(uv[...,0]<width)&(uv[...,1]>=0)&(uv[...,1]<height)
        gap=dt>3*np.median(dt);pathvalid=cv[1:]&cv[:-1]&~gap[:,None]
        delta=np.linalg.norm(np.diff(wrist[:,:,:3,3],axis=0),axis=-1)
        speed=delta/dt[:,None];speed[~pathvalid]=np.nan
        # A single static world origin and scale per sequence; never frame-wise recentering.
        center=np.nanmean(wrist[:,:,:3,3][cv],axis=0)
        rotation=np.array([[.7071,-.7071,0],[.35355,.35355,-.8660254]])
        fixed_scale=440.
        def view(p,mode):
            q=np.asarray(p)-center
            xy=q@rotation.T if mode==0 else np.stack([q[...,0],-q[...,2]],-1)
            return xy*fixed_scale+np.array([320,350])
        cap=cv2.VideoCapture(str(folder/clip['files']['video']))
        files=[out/'2D_原图腕骨架贴合.mp4',out/'3D_骨架与腕轨迹.mp4']
        encs=[encoder(p,'1280x720') for p in files]
        try:
            for i in range(n):
                ok,raw=cap.read()
                if not ok:raise RuntimeError('SOURCE_TRUNCATED')
                overlay=raw.copy()
                for s,color in enumerate(COLORS):
                    skeleton(overlay,uv[i,s],inside[i,s],color)
                    if wpv[i,s] and hv[i] and cv[i,s] and np.all(np.abs(wuv[i,s])<20000):cv2.circle(overlay,pt(wuv[i,s]),11,color,3)
                    if cpv[i,s] and hv[i] and cv[i,s] and np.all(np.abs(cuv[i,s])<20000):cv2.drawMarker(overlay,pt(cuv[i,s]),(0,210,210),cv2.MARKER_CROSS,22,3)
                image=np.full((720,1280,3),248,np.uint8)
                image[100:580,:640]=cv2.resize(raw,(640,480));image[100:580,640:]=cv2.resize(overlay,(640,480))
                image=text(image,[(session+'｜原图',(12,15)),('固定参数硬件投影｜蓝左 红右',(652,15)),('圆环：安装估计腕点；黄十字：手柄原点；骨架根与腕重合是定义，不是准确度证明',(12,605)),(f'源帧 {int(fid[i])} / {n}  采集时间 {(int(ts[i])-int(ts[0]))*1e-9:.3f}s｜30FPS逐帧展示，非重采样',(12,639)),('编码图像域适用性未验证；不拟合，不换手，不隐藏错位',(12,675))])
                encs[0].stdin.write(image.tobytes())
                canvas=np.full((720,1280,3),248,np.uint8)
                for mode in (0,1):
                    offset=np.array([640*mode,0])
                    for axis,color,label in ((0,(80,80,220),'X'),(1,(80,180,80),'Y'),(2,(220,80,80),'Z')):
                        p=center.copy();p[axis]+=.2
                        a=view(center,mode)+offset;b=view(p,mode)+offset
                        cv2.arrowedLine(canvas,pt(a),pt(b),color,2);cv2.putText(canvas,label,pt(b),0,.5,color,1)
                    for s,color in enumerate(COLORS):
                        trace=view(wrist[:i+1,s,:3,3],mode)+offset
                        for j in range(1,i+1):
                            if pathvalid[j-1,s] and np.isfinite(trace[j-1:j+1]).all():cv2.line(canvas,pt(trace[j-1]),pt(trace[j]),color,1,cv2.LINE_AA)
                        if cv[i,s]:
                            skeleton(canvas,view(world[i,s],mode)+offset,jvalid[i,s],color)
                            origin=view(wrist[i,s,:3,3],mode)+offset
                            cv2.circle(canvas,pt(origin),7,color,2)
                            for axis in range(3):
                                endpoint=wrist[i,s,:3,3]+wrist[i,s,:3,axis]*.065
                                cv2.line(canvas,pt(origin),pt(view(endpoint,mode)+offset),((50,50,200),(50,180,50),(200,80,50))[axis],2)
                # Enlarged wrist-local fingers: translation removed for readability, explicitly labeled.
                for s,color in enumerate(COLORS):
                    localuv=joints[i,s]@rotation.T*1100+np.array([170+300*s,135])
                    skeleton(canvas,localuv,jvalid[i,s],color)
                canvas=text(canvas,[(session+'｜三维骨架与历史腕轨迹',(12,10)),('上方：腕局部手指放大（不显示腕平移）',(12,45)),('世界固定斜视图｜1米=440像素',(12,605)),('世界侧视图｜横X向前，纵Z向上',(652,605)),(f'蓝：解剖左手  红：解剖右手｜采集时间 {(int(ts[i])-int(ts[0]))*1e-9:.3f}s',(12,640)),('腕局部RGB短轴表示旋转；轨迹跨时间缺口不连接；不是视觉估计或机器人FK',(12,678))])
                encs[1].stdin.write(canvas.tobytes())
                if i in (0,n//2,n-1):
                    cv2.imwrite(str(out/f'2D_{i:04d}.jpg'),image);cv2.imwrite(str(out/f'3D_{i:04d}.jpg'),canvas)
            for enc in encs:enc.stdin.close();assert enc.wait()==0
        finally:
            cap.release()
            for enc in encs:
                if enc.poll() is None:enc.kill();enc.wait()
        probes=[]
        for file in files:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(file),'-f','null','-'],check=True)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(file)]))
            assert int(probe['streams'][0]['nb_read_frames'])==n;probes.append(probe)
        np.savez_compressed(out/'MOTION_AND_PROJECTION.npz',timestamp_ns=ts,frame_id=fid,joints_world=world,wrist_world=wrist,controller_world=ctrl,joint_uv=uv,wrist_uv=wuv,controller_uv=cuv,joint_valid=jvalid,joint_in_image=inside,head_valid=hv,wrist_valid=cv,gap_after=gap)
        metrics={'session':session,'frames':n,'same_installation_as_025':True,'same_camera_parameters_as_025':True,'no_fitting':True,'valid_wrist_frames_left_right':cv.sum(0).tolist(),'valid_hand_frames_left_right':hand.sum(0).tolist(),'joint_in_image_fraction_left_right':(inside.sum((0,2))/np.maximum(valid.sum((0,2)),1)).tolist(),'wrist_range_xyz_m':np.ptp(wrist[:,:,:3,3],axis=0).tolist(),'speed_p95_mps_left_right':np.nanpercentile(speed,95,axis=0).tolist(),'max_consecutive_displacement_m_left_right':np.nanmax(np.where(pathvalid,delta,np.nan),axis=0).tolist(),'gap_indices':np.where(gap)[0].tolist(),'source_quality':source_quality,'image_accuracy_px':None,'independent_ground_truth':False,'robot_solver_generalization_tested':False,'videos':[artifact_ref(p) for p in files],'decode':probes,'observation':'In-image fraction is coverage, not accuracy. Same exported calibration is not independent physical validation.'}
        atomic_json(out/'RESULT.json',metrics);allresults.append(metrics)
        print(session,'DONE',flush=True)
    atomic_json(ROOT/'RESULT.json',{'status':'COMPLETE_DIAGNOSTIC_NOT_ACCURACY_CERTIFIED','scope':'Frozen PICO/MANUS upstream chain only; V16 robot IK not rerun','sessions':allresults,'baseline_changed':False,'parameter_fitting':False,'blind_holdout':False,'physical_accuracy_verified':False})
    return 0

if __name__=='__main__':raise SystemExit(main())
