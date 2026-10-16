"""Freeze the user-selected V16 development baseline; independent raw-image diagnostic."""
import ast
import json
from pathlib import Path
import subprocess
import cv2
import h5py
import numpy as np
from chaoyang.ops import run_robot100_constraint_review_v16_retry1 as v16
from chaoyang.pipeline.pico_manus_motion_v2 import pose_matrices, require_transform

old=v16.old
ROOT=old.REPO_ROOT/'_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001'


def code_closure(start):
    pending=list(start); visited=set()
    while pending:
        path=pending.pop()
        if path in visited: continue
        visited.add(path)
        if len(visited)>160: raise RuntimeError('CLOSURE_BUDGET')
        tree=ast.parse(path.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                if not name.startswith('chaoyang.'): continue
                candidate=old.REPO_ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate not in visited: pending.append(candidate)
    return [old.artifact_ref(p) for p in sorted(visited)]


def project(points,k):
    points=np.asarray(points,float)
    good=np.isfinite(points).all(axis=-1)&(points[...,2]>1e-4)
    uv=np.full(points.shape[:-1]+(2,),np.nan)
    uv[good]=(points[good]@k.T)[:,:2]/points[good,2,None]
    return uv,good


def main():
    result=old.load_json(v16.ROOT/'RESULT.json')
    # This approval is scoped to the exact reviewed artifact, not physical accuracy.
    assert result['session_id']=='play_cards_0916_025'
    for key in ('video','motion','constraint_review'):
        assert old.artifact_ref(Path(result[key]['path']))==result[key]
    inputs=result['source']
    h5path=old.baseline._stable_ref(Path(inputs['hdf5']['path']),inputs['hdf5'])
    video=old.baseline._stable_ref(Path(inputs['video']['path']),inputs['video'])
    camera_path=video.parent/'camera_params.json'; clip_path=video.parent/'clip_manifest.json'
    camera=old.load_json(camera_path); clip=old.load_json(clip_path)
    assert clip['clip_name']=='play_cards_0916_025'
    assert clip['acquisition_aligned_hdf5_contract']['sha256']==inputs['hdf5']['sha256']
    refs={'camera':old.artifact_ref(camera_path),'clip':old.artifact_ref(clip_path),'hdf5':inputs['hdf5'],'video':inputs['video']}
    closure=code_closure([Path(v16.__file__),Path(v16.prior.__file__),Path(__file__)])
    old.atomic_json(ROOT/'DEPENDENCY_KEEP_SET.json',{'reason':'Old version names in this transitive source closure are implementation dependencies, not alternative baselines. Do not delete. Relative/dynamic imports remain protected by the governance contract as well.','files':closure})
    baseline={'schema_version':'CHAOYANG_0916_HARDWARE_BASELINE_V16_V1','baseline_id':'0916_HARDWARE_WORLD_WRIST_V16',
        'baseline_selected':True,'selection_scope':'USER_ACCEPTED_OFFLINE_VISUAL_DEVELOPMENT_BASELINE',
        'user_approval':'2026-09-29: user said the current result looks reasonable and selected this branch chain as baseline.',
        'reviewed_video':result['video'],'frozen_result':old.artifact_ref(v16.ROOT/'RESULT.json'),'motion':result['motion'],
        'input':inputs,'code_closure':closure,'chain':['same-session raw controller/head/MANUS HDF5 + synchronized original video','controller-to-wrist transform stored in HDF5','anatomical left/right identity, WORLD_FIRST_UNIT_GAIN','frozen V15 targets and q22','V16 explicit inward inequality with unchanged 20mm/15deg gates','fixed-view robot + adapter + hand closeups'],
        'verified_session':'play_cards_0916_025','batch_100_validated':False,'physical_accuracy_verified':False,'image_alignment_verified':False,'collision_certified':False,
        'excluded_branch':'0902 Exact78/Inpainting','do_not_use':['V8.1','V9','V10','V11','V12','V13','V14','V15 as independent current baseline'],
        'retention':'Keep transitive implementation and sealed predecessor evidence; remove historical execution instructions from current navigation.'}
    old.atomic_json(ROOT/'BASELINE.json',baseline)
    with h5py.File(h5path,'r') as f:
        head,hv=pose_matrices(f['head_pose'][:])
        wrist,wv=pose_matrices(np.stack([f[s+'_wrist_pose'][:] for s in ('left','right')],axis=1))
        ctrl,cv=pose_matrices(np.stack([f[s+'_controller_pose'][:] for s in ('left','right')],axis=1))
        joints=np.stack([f[s+'_hand_joints'][:] for s in ('left','right')],axis=1)
        hand_valid=np.stack([f[s+'_hand_valid'][:] for s in ('left','right')],axis=1).astype(bool)
        ts=f['timestamp_ns'][:]; frame_id=f['video_frame_idx'][:]
        prior=json.loads(f.attrs['controller_to_wrist_calibration'])
        assert str(f.attrs['hand_frame'])=='wrist-local (relative to MANUS wrist root)'
    np.testing.assert_array_equal(frame_id,np.arange(len(head)))
    # Use this encoded stream's selected physical camera, not the misleading legacy left label.
    key=clip['selection']['source_calibration_key']
    assert key==camera['selected_calibration_key']=='right'
    extr=require_transform(camera['selected_effective_rep103_head_to_camera'])
    rect=clip['video_rectification']
    assert rect['rectified'] is False and rect['image_domain_mode']=='passthrough_scaled_source_domain'
    k=np.asarray(rect['intrinsics'],float)
    world=np.einsum('tsij,tskj->tski',wrist[:,:,:3,:3],joints)+wrist[:,:,None,:3,3]
    transform=extr@np.linalg.inv(head)
    camera_points=np.einsum('tij,tskj->tski',transform[:,:3,:3],world)+transform[:,None,None,:3,3]
    cp=np.einsum('tij,tsj->tsi',transform[:,:3,:3],ctrl[:,:,:3,3])+transform[:,None,:3,3]
    wp=np.einsum('tij,tsj->tsi',transform[:,:3,:3],wrist[:,:,:3,3])+transform[:,None,:3,3]
    uv,jvalid=project(camera_points,k); cuv,cvalid=project(cp,k); wuv,wvalid=project(wp,k)
    jvalid &= hand_valid[:,:,None]&wv[:,:,None]&hv[:,None,None]
    cvalid &= cv&hv[:,None]; wvalid &= wv&hv[:,None]
    # Hand_Invalid (node 0) is only a virtual root, never an observed anatomical joint.
    jvalid[:,:,0]=False
    width,height=int(rect['width']),int(rect['height'])
    inside=jvalid&(uv[...,0]>=0)&(uv[...,0]<width)&(uv[...,1]>=0)&(uv[...,1]<height)
    origin,ok=project(np.array([[0.,0.,1.],[.1,0.,1.],[0.,.1,1.],[0.,0.,-1.]]),k)
    np.testing.assert_allclose(origin[0],k[:2,2]); assert not ok[-1]
    assert origin[1,0]>origin[0,0] and origin[2,1]>origin[0,1]
    np.testing.assert_allclose(wp,camera_points[:,:,0],atol=1e-9)
    np.savez_compressed(ROOT/'PROJECTED_POINTS.npz',joint_uv=uv,controller_uv=cuv,wrist_uv=wuv,joint_valid=jvalid,joint_in_frame=inside,wrist_valid=wvalid,controller_valid=cvalid,timestamp_ns=ts,frame_id=frame_id,K=k,T_camera_world=transform)
    edges=[(i,i+1) for a,b in ((1,4),(5,9),(10,14),(15,19),(20,24)) for i in range(a,b)]
    cap=cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==width and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==height
    out=ROOT/'play_cards_0916_025_硬件腕骨架_原图投影诊断.mp4'
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-s','1920x840','-r',str(inputs['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    def point(image,p,color,radius,kind='circle'):
        if not np.isfinite(p).all() or max(abs(p))>20000: return
        x,y=np.rint(p).astype(int)
        if kind=='cross': cv2.drawMarker(image,(x,y),color,cv2.MARKER_CROSS,20,2)
        else: cv2.circle(image,(x,y),radius,color,2,cv2.LINE_AA)
    try:
        for i in range(len(head)):
            ok,raw=cap.read()
            if not ok: raise RuntimeError('SOURCE_TRUNCATED')
            overlay=raw.copy()
            for side,color in ((0,(245,130,30)),(1,(40,50,245))):
                for a,b in edges:
                    if inside[i,side,a] and inside[i,side,b]: cv2.line(overlay,tuple(np.rint(uv[i,side,a]).astype(int)),tuple(np.rint(uv[i,side,b]).astype(int)),color,3,cv2.LINE_AA)
                for j in range(1,25):
                    if inside[i,side,j]: point(overlay,uv[i,side,j],color,5)
                if wvalid[i,side]: point(overlay,wuv[i,side],color,12)
                if cvalid[i,side]: point(overlay,cuv[i,side],(0,220,220),10,'cross')
            canvas=np.full((840,1920,3),245,np.uint8)
            canvas[48:768,:960]=cv2.resize(raw,(960,720))
            canvas[48:768,960:]=cv2.resize(overlay,(960,720))
            canvas=old.baseline._put_chinese(canvas,[('原始视频｜025，同步同帧',(18,5),28),('硬件投影诊断｜蓝左、红右、黄十字手柄原点',(978,5),25),
                (f'帧{i:03d}｜圆环：安装参数估计腕点；小点：MANUS关节；虚拟根不当作实测关节',(18,772),24),
                ('同会话物理右相机｜按现有编码域K直投，不新增畸变；域适用性未验证，不代表准确度通过',(18,806),22)])
            encoder.stdin.write(canvas.tobytes())
            if i in (0,50,75,100,175): cv2.imwrite(str(ROOT/f'OVERLAY_{i:04d}.jpg'),canvas)
        encoder.stdin.close()
        assert encoder.wait()==0
    finally:
        cap.release()
        if encoder.poll() is None: encoder.kill();encoder.wait()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(out),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(out)]))
    assert int(probe['streams'][0]['nb_read_frames'])==len(head)
    old.atomic_json(ROOT/'RESULT.json',{'status':'COMPLETE_DIAGNOSTIC_NOT_ACCURACY_CERTIFIED','baseline':old.artifact_ref(ROOT/'BASELINE.json'),'video':old.artifact_ref(out),'points':old.artifact_ref(ROOT/'PROJECTED_POINTS.npz'),'inputs':refs,'decode':probe,'joint_in_frame_count_left_right':inside.sum(axis=(0,2)).tolist(),'projectable_joint_count_left_right':jvalid.sum(axis=(0,2)).tolist(),'frame_count':len(head),'encoded_domain_authority':camera['encoded_domain_intrinsics_authority'],'physical_camera':key,'controller_to_wrist':prior,'independent_2d_reference_available':False,'accuracy_pixels':None,'hardware_better_than_vision':'NOT_ESTABLISHED','no_parameter_fitting':True,'physical_accuracy_verified':False})
    return 0


if __name__=='__main__': raise SystemExit(main())
