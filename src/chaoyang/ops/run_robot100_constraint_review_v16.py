"""Frozen V15 targets: enforce the existing inward bound without relaxing gates."""
from pathlib import Path
import json
import subprocess
import numpy as np
import cv2
from scipy.optimize import minimize
from chaoyang.ops import run_robot100_side_review_v15 as prior
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.exact78_style_inward_solver_v1 import bounded_joint_limits, _pose_error
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives
from chaoyang.pipeline.v5_product import ProductRobotRenderer
from chaoyang.ops.render_tianji_kai_mount_proxy_audit import body_bounds

old = prior.old
ROOT = old.REPO_ROOT / '_run/current/robot100_constraint_review_v16_20260929/attempts/attempt_0001'


def objective(q, assets, side, target_tool, reference):
    residual = arm._pose_residual(arm._tool_fk(assets, side, q), target_tool)
    return float(residual @ residual + 1e-4 * np.sum((q-reference)**2))


def refine(source, assets, mounts):
    result = {k: np.array(v, copy=True) for k, v in source.items()}
    lower, upper = arm._arm_limits(assets)
    minimum = float(source['posture_minimum_m'])
    valid, edge = source['wrist_valid'], source['time_edge_valid']
    changes = []
    for side in range(2):
        previous = previous_previous = None
        for frame in range(len(valid)):
            if not valid[frame, side]:
                previous = previous_previous = None
                continue
            continuous = frame > 0 and edge[frame] and valid[frame-1, side]
            if continuous and previous is not None:
                lo, hi, accel = bounded_joint_limits(lower[side], upper[side], previous, previous_previous,
                    step_limit=float(source['joint_step_limit_rad']), acceleration_limit=float(source['joint_acceleration_limit_rad']))
            else:
                lo, hi, accel = lower[side], upper[side], False
            q = source['q_arm'][frame, side].copy()
            margin = lambda value: float(old.v6.inward_margin_m(assets, mounts, source['neutral_q_arm'], side, value))
            needs = margin(q) < minimum or np.any(q < lo-1e-10) or np.any(q > hi+1e-10)
            if needs:
                target_tool = source['T_target_root'][frame, side] @ np.linalg.inv(source['T_tool_hand'][side])
                reference = q.copy() if previous is None else previous
                # The 10-nanometre interior buffer is numerical protection, not a relaxed gate.
                candidate = minimize(objective, np.clip(q, lo, hi), args=(assets, side, target_tool, reference),
                    method='SLSQP', bounds=list(zip(lo, hi)),
                    constraints=[{'type':'ineq', 'fun':lambda value: margin(value)-minimum-1e-8}],
                    options={'maxiter':300, 'ftol':1e-11})
                q = candidate.x
                result['solver_success'][frame, side] = candidate.success
                result['solver_status'][frame, side] = candidate.status
                result['solver_nfev'][frame, side] = candidate.nfev
                changes.append({'frame':frame, 'side':['left','right'][side], 'success':bool(candidate.success),
                    'message':str(candidate.message), 'margin_before_m':margin(source['q_arm'][frame,side]),
                    'margin_after_m':margin(q), 'maximum_joint_change_rad':float(np.max(np.abs(q-source['q_arm'][frame,side])))})
            actual, position, rotation = _pose_error(assets, source['T_tool_hand'], side, q, source['T_target_root'][frame,side])
            result['q_arm'][frame,side] = q
            result['T_actual_root'][frame,side] = actual
            result['position_residual_mm'][frame,side] = position
            result['rotation_residual_deg'][frame,side] = rotation
            result['posture_margin_m'][frame,side] = margin(q)
            result['acceleration_bound_applied'][frame,side] = accel
            assert np.all(q >= lo-1e-8) and np.all(q <= hi+1e-8), 'JOINT_BOUND'
            previous_previous, previous = previous, q.copy()
    result['tolerance_pass'] = (valid & result['solver_success'] & (result['position_residual_mm']<=20)
        & (result['rotation_residual_deg']<=15) & (result['posture_margin_m']>=minimum-1e-9))
    result['quality_pass'] = result['tolerance_pass'].copy()
    result['solver_policy'] = np.asarray('V15_FROZEN_TARGETS_EXPLICIT_INWARD_CONSTRAINT')
    times = (source['timestamp_ns']-source['timestamp_ns'][0]).astype(float)*1e-9
    for name, values in motion_derivatives(result['q_arm'], valid, times, edge).items():
        result['arm_'+name] = values
    for key in ('T_target_root','world_wrist_T','q22','frame_id','timestamp_ns','wrist_valid','finger_valid'):
        np.testing.assert_array_equal(result[key], source[key])
    assert np.all(result['tolerance_pass'][source['tolerance_pass']]), 'PREVIOUS_PASS_REGRESSION'
    return result, changes


def render(row, solved, destination):
    motion = dict(solved, q_hand22=solved['q22'], T_cam_base=np.eye(4))
    r = ProductRobotRenderer(old.REPO_ROOT, motion, {'width':960,'height':480,'K':[[580,0,480],[0,580,240],[0,0,1]]}, include_adapter=True)
    b = r.b
    # Whole-sequence framing from FK, no extra full-sequence rendering pass.
    points = solved['T_actual_root'][:,:,:3,3][solved['wrist_valid']]
    lo = np.minimum(points.min(axis=0)-.12, [-.12,-.4,1.02])
    hi = np.maximum(points.max(axis=0)+.12, [.12,.4,1.5])
    center = (lo+hi)/2
    full_view = b.computeViewMatrixFromYawPitchRoll(center.tolist(),max(1.25,float(np.linalg.norm(hi-lo))*1.05),40,-12,0,2)
    projection = b.computeProjectionMatrixFOV(48,2,.02,20)
    r.projection = projection
    pv = np.asarray(projection).reshape(4,4,order='F') @ np.asarray(full_view).reshape(4,4,order='F')
    def project(point):
        clip = pv @ np.r_[point,1.]
        if clip[3]<=0: return None
        ndc = clip[:3]/clip[3]
        return (int((ndc[0]+1)*480), int((1-ndc[1])*240))
    cap = cv2.VideoCapture(str(old.baseline._stable_ref(Path(row['video']['path']),row['video'])))
    if not cap.isOpened(): raise RuntimeError('SOURCE_OPEN')
    encoder = subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r',str(row['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
    try:
        for i in range(len(solved['frame_id'])):
            ok, raw = cap.read()
            if not ok: raise RuntimeError('SOURCE_TRUNCATED')
            canvas = np.full((1080,1920,3),244,np.uint8)
            canvas[44:524,:960] = old.baseline._letterbox_bgr(raw,960,480)
            r.view = full_view
            layers = r.frame_layers(i)
            full = cv2.cvtColor(layers.rgb,cv2.COLOR_RGB2BGR)
            full[~layers.alpha] = 244
            # Display-only projected wrist trail: no scene depth or measured table is invented.
            for side, color in ((0,(220,100,20)),(1,(35,45,220))):
                for j in range(max(1,i-30),i+1):
                    if not solved['time_edge_valid'][j] or not solved['wrist_valid'][j-1:j+1,side].all(): continue
                    a,c = [project(solved['T_actual_root'][k,side,:3,3]) for k in (j-1,j)]
                    if a is not None and c is not None: cv2.line(full,a,c,color,2,cv2.LINE_AA)
            origin = center+np.array([0,.28,-.3])
            for delta,color in (([.18,0,0],(80,160,80)),([0,0,.18],(150,90,50))):
                a,c = project(origin),project(origin+delta)
                if a is not None and c is not None: cv2.arrowedLine(full,a,c,color,3,cv2.LINE_AA,tipLength=.15)
            full = old.baseline._put_chinese(full,[('轨迹线为显示辅助｜竖直方向：世界Z向上',(12,450),21)])
            canvas[44:524,960:] = full
            for side, hand in enumerate(r.hands):
                lower, upper = body_bounds(r.client,(hand,))
                r.view = b.computeViewMatrixFromYawPitchRoll(((lower+upper)/2).tolist(),.42,40,-24,0,2)
                layer = r.frame_layers(i)
                close = cv2.cvtColor(layer.rgb,cv2.COLOR_RGB2BGR)
                close[~layer.alpha] = 244
                canvas[564:1044,side*960:(side+1)*960] = close
            delta = (solved['world_wrist_T'][i,1,:3,3]-solved['world_wrist_T'][0,1,:3,3])*1000
            canvas = old.baseline._put_chinese(canvas,[
                ('原始采集｜人体右手为主要活动侧',(18,4),28),
                ('机器人固定前侧视角｜蓝：左手　红：右手',(978,4),27),
                ('解剖左手近景｜固定尺度，镜头随手平移',(18,528),27),
                ('解剖右手近景｜固定尺度，镜头随手平移',(978,528),27),
                (f'025 帧{i:03d}｜右腕较首帧：前向{delta[0]:+.0f}毫米，上向{delta[2]:+.0f}毫米｜V16 虚拟安装，非物理标定；待审阅',(18,1048),22)])
            if i in (0,50,75,100,175): cv2.imwrite(str(ROOT/f'REVIEW_{i:04d}.jpg'),canvas)
            encoder.stdin.write(canvas.tobytes())
            if i%40==0: print('RENDER_FRAME',i,flush=True)
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError('ENCODE_FAILED')
    finally:
        cap.release(); r.close()
        if encoder.poll() is None: encoder.kill(); encoder.wait()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(destination),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(destination)]))
    assert int(probe['streams'][0]['nb_read_frames'])==len(solved['frame_id'])
    return probe


def main():
    predecessor = old.load_json(prior.ROOT/'RESULT.json')
    with np.load(old.baseline._stable_ref(Path(predecessor['motion']['path']),predecessor['motion']),allow_pickle=False) as frozen:
        source = {k:frozen[k] for k in frozen.files}
    assets = old.load_pinned_robot_assets(old.REPO_ROOT)
    mounts,_ = old.baseline._mounts()
    checks = prior.side_tests()
    solved, changes = refine(source,assets,mounts)
    old.atomic_json(ROOT/'CONSTRAINT_REVIEW.json',{'changes':changes,'before':source['tolerance_pass'].sum(axis=0).tolist(),'after':solved['tolerance_pass'].sum(axis=0).tolist(),'input_unchanged':True,'gate_unchanged':True})
    motion = ROOT/'ROBOT_MOTION_V16.npz'
    if motion.exists(): raise RuntimeError('FRESH_OUTPUT_REQUIRED')
    np.savez_compressed(motion,**solved)
    print('REFINE_DONE',solved['tolerance_pass'].sum(axis=0).tolist(),flush=True)
    video = ROOT/'play_cards_0916_025_V16_内收硬约束_腕轨迹与手指.mp4'
    probe = render(predecessor['source'],solved,video)
    result = {'schema_version':'ROBOT100_CONSTRAINT_REVIEW_V16','status':'COMPLETE_PENDING_HUMAN_REVIEW','session_id':'play_cards_0916_025',
        'source':predecessor['source'],'predecessor':old.artifact_ref(prior.ROOT/'RESULT.json'),'checks':checks,
        'tracking_pass_left_right':solved['tolerance_pass'].sum(axis=0).tolist(),
        'position_max_mm_left_right':np.nanmax(solved['position_residual_mm'],axis=0).tolist(),
        'rotation_max_deg_left_right':np.nanmax(solved['rotation_residual_deg'],axis=0).tolist(),
        'minimum_inward_margin_m_left_right':np.nanmin(solved['posture_margin_m'],axis=0).tolist(),
        'maximum_joint_step_rad':float(np.max(np.abs(np.diff(solved['q_arm'],axis=0)))),
        'constraint_review':old.artifact_ref(ROOT/'CONSTRAINT_REVIEW.json'),'motion':old.artifact_ref(motion),'video':old.artifact_ref(video),'decode':probe,
        'baseline_selected':False,'quality_adopted':False,'physical_accuracy_verified':False,'collision_verified':False,'review':'PENDING'}
    old.atomic_json(ROOT/'RESULT.json',result)
    return 0


if __name__=='__main__': raise SystemExit(main())
