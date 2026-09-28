"""Frozen-input full Tianji/KaiHand CPU review, with explicit virtual authority.

No camera calibration, contact, control or training authority is created here.
Flange mounts are converted to tool mounts, never silently relabelled.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = 'FULL_ROBOT_MOTION_INPUT_V2'
HUMAN_TO_PHYSICAL = np.array([1, 0], dtype=np.int64)


def reference(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def read_frozen(ref, roots):
    p = Path(ref['path']).resolve(strict=True)
    if not any(p.is_relative_to(Path(r).resolve(strict=True)) for r in roots):
        raise ValueError('READ_SCOPE_ESCAPE')
    before = p.stat()
    data = p.read_bytes()
    after = p.stat()
    if any(getattr(before, k) != getattr(after, k) for k in
           ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('UNSTABLE_INPUT')
    if len(data) != ref['bytes'] or hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('FROZEN_REFERENCE_DRIFT')
    return data


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_se3(values):
    values = np.asarray(values, float)
    if values.shape[-2:] != (4, 4) or not np.isfinite(values).all():
        raise ValueError('INVALID_SE3_SHAPE_OR_FINITE')
    rotation = values[..., :3, :3]
    if not np.allclose(values[..., 3, :], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('INVALID_HOMOGENEOUS_ROW')
    if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6):
        raise ValueError('INVALID_ROTATION')
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('LEFT_HANDED_ROTATION')


def tool_mount_from_flange(flange_tool, flange_hand):
    validate_se3(flange_tool); validate_se3(flange_hand)
    return np.linalg.inv(flange_tool) @ flange_hand


def time_edges(timestamp_ns, frame_id, segment_id=None):
    ns = np.asarray(timestamp_ns)
    ids = np.asarray(frame_id)
    if ns.dtype != np.dtype('int64') or ns.ndim != 1 or ids.shape != ns.shape or len(ns) < 2:
        raise ValueError('INT64_ACQUISITION_TIME_REQUIRED')
    delta = np.diff(ns)
    if np.any(delta <= 0) or np.any(np.diff(ids) <= 0):
        raise ValueError('NONMONOTONIC_TIME_OR_ID')
    edge = (np.diff(ids) == 1) & (delta <= 2.5 * np.median(delta))
    if segment_id is not None:
        seg = np.asarray(segment_id)
        if seg.shape != ns.shape:
            raise ValueError('SEGMENT_SHAPE')
        edge &= seg[1:] == seg[:-1]
    return np.r_[False, edge], (ns - ns[0]).astype(float) * 1e-9


def relative_wrist(world_wrist, valid):
    """One origin per side for the entire sequence; never recenter per frame/gap."""
    transforms = np.asarray(world_wrist, float)
    valid = np.asarray(valid, bool)
    if transforms.shape != (*valid.shape, 4, 4) or valid.ndim != 2 or valid.shape[1] != 2:
        raise ValueError('WRIST_SIDE_SHAPE')
    validate_se3(transforms[valid])
    relative = np.full_like(transforms, np.nan)
    origins = np.full((2, 4, 4), np.nan)
    for side in range(2):
        indices = np.flatnonzero(valid[:, side])
        if len(indices):
            origins[side] = transforms[indices[0], side]
            relative[indices, side] = np.linalg.inv(origins[side]) @ transforms[indices, side]
    return relative, origins


def validate_motion(arrays):
    q = np.asarray(arrays['q22'])
    finger = np.asarray(arrays['finger_valid'])
    wrist = np.asarray(arrays['wrist_valid'])
    relative = np.asarray(arrays['relative_wrist_T'])
    if q.ndim != 3 or q.shape[1:] != (2, 22):
        raise ValueError('Q22_PHYSICAL_SIDE_SHAPE')
    if finger.shape != q.shape[:2] or wrist.shape != finger.shape or finger.dtype != bool or wrist.dtype != bool:
        raise ValueError('INDEPENDENT_BOOLEAN_MASKS_REQUIRED')
    if relative.shape != (*wrist.shape, 4, 4):
        raise ValueError('RELATIVE_WRIST_SHAPE')
    if not np.isfinite(q[finger]).all():
        raise ValueError('VALID_FINGERS_NONFINITE')
    validate_se3(relative[wrist])
    if not np.array_equal(arrays['human_to_physical'], HUMAN_TO_PHYSICAL):
        raise ValueError('SIDE_MAPPING_MISMATCH')
    return time_edges(arrays['timestamp_ns'], arrays['frame_id'], arrays.get('segment_id'))


def motion_derivatives(values, valid, times, edge):
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    v = np.full_like(values, np.nan); a = v.copy(); j = v.copy()
    dt = np.diff(times)
    eligible = valid[1:] & valid[:-1] & edge[1:, None]
    v[1:] = np.where(eligible[..., None], np.diff(values, axis=0) / dt[:, None, None], np.nan)
    if len(times) > 2:
        a[2:] = np.diff(v[1:], axis=0) / (.5 * (dt[1:] + dt[:-1]))[:, None, None]
    if len(times) > 3:
        vt = .5 * (times[1:] + times[:-1]); at = .5 * (vt[1:] + vt[:-1])
        j[3:] = np.diff(a[2:], axis=0) / np.diff(at)[:, None, None]
    return dict(velocity=v, acceleration=a, jerk=j, edge_valid=eligible)


def seed_usable_solution(q, lower, upper, *, solver_success, solver_status,
                         position_mm, rotation_deg, independent_fk):
    """Eligibility for the *next* frame, never the current frame quality gate."""
    q, lower, upper = (np.asarray(value, dtype=float) for value in (q, lower, upper))
    fk = np.asarray(independent_fk, dtype=float)
    return bool(
        (solver_success or solver_status == 0)
        and q.shape == lower.shape == upper.shape == (7,)
        and np.isfinite(q).all()
        and np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)
        and fk.shape == (4, 4) and np.isfinite(fk).all()
        and np.isfinite(position_mm) and position_mm <= 20.
        and np.isfinite(rotation_deg) and rotation_deg <= 15.
    )


def solve_full_chain(arrays, assets, flange_mounts, *, progress=None, max_nfev=80,
                     continuity_weight=0.0, experimental_seed_recovery_frames=None):
    """Fixed placement, actual-dt gaps reset the seed, target scale remains one."""
    from scipy.optimize import least_squares
    from chaoyang.pipeline import robot_scene_state_cpu as arm
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
    edge, times = validate_motion(arrays)
    if not np.isfinite(continuity_weight) or continuity_weight < 0:
        raise ValueError('INVALID_CONTINUITY_WEIGHT')
    lower, upper = arm._arm_limits(assets)
    neutral = .5 * (lower + upper)
    neutral_fk = forward_kinematics(assets.tianji, {
        name: float(neutral[side, k]) for side in range(2)
        for k, name in enumerate(ARM_JOINT_NAMES[side])})
    flange_names = ('flange_L', 'flange_R'); tool_names = ('left_tool', 'right_tool')
    flange_tools = np.stack([np.linalg.inv(neutral_fk[f]) @ neutral_fk[t]
                            for f, t in zip(flange_names, tool_names)])
    tool_mounts = tool_mount_from_flange(flange_tools, flange_mounts)
    neutral_roots = np.stack([neutral_fk[f] for f in flange_names]) @ flange_mounts
    check = np.stack([neutral_fk[t] for t in tool_names]) @ tool_mounts
    if not np.allclose(check, neutral_roots, atol=1e-10):
        raise ValueError('FLANGE_TOOL_MOUNT_PARITY_FAILURE')
    valid = arrays['wrist_valid']; count = len(valid)
    target = np.full((count, 2, 4, 4), np.nan)
    actual = target.copy(); q_arm = np.full((count, 2, 7), np.nan)
    position = np.full((count, 2), np.nan); rotation = position.copy()
    converged = np.zeros((count, 2), bool); nfev = np.zeros((count, 2), int)
    solver_status = np.full((count, 2), -999, dtype=np.int32)
    seed_usable = np.zeros((count, 2), bool)
    seed_in = np.full((count, 2, 7), np.nan)
    seed_source_frame = np.full((count, 2), -1, dtype=np.int64)
    reset_reason = np.full((count, 2), 'UNSET', dtype='<U32')
    for side in range(2):
        target[valid[:, side], side] = neutral_roots[side] @ arrays['relative_wrist_T'][valid[:, side], side]
        seed = neutral[side].copy()
        previous_usable = False
        for frame in range(count):
            if not valid[frame, side]:
                seed = neutral[side].copy()
                previous_usable = False
                reset_reason[frame, side] = 'INVALID_INPUT'
                continue
            continuous = bool(frame > 0 and edge[frame] and valid[frame-1, side])
            if not continuous:
                seed = neutral[side].copy()
                reset_reason[frame, side] = 'FIRST_OR_TIME_GAP'
            elif not previous_usable:
                seed = neutral[side].copy()
                reset_reason[frame, side] = 'PRIOR_SOLUTION_UNUSABLE'
            else:
                seed_source_frame[frame, side] = int(arrays['frame_id'][frame - 1])
                reset_reason[frame, side] = 'INHERITED_PRIOR_SOLUTION'
            seed_in[frame, side] = seed
            target_tool = target[frame, side] @ np.linalg.inv(tool_mounts[side])
            prior = seed.copy()
            use_continuity = bool(continuity_weight and edge[frame] and frame > 0
                                  and valid[frame-1, side])
            def residual(q):
                pose = arm._pose_residual(arm._tool_fk(assets, side, q), target_tool)
                if use_continuity:
                    return np.concatenate((pose, continuity_weight * (q - prior)))
                if continuity_weight:
                    return np.concatenate((pose, np.zeros(7)))
                return pose
            result = least_squares(
                residual,
                seed, bounds=(lower[side], upper[side]), max_nfev=max_nfev,
                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            q_arm[frame, side] = result.x; nfev[frame, side] = result.nfev
            solver_status[frame, side] = result.status
            actual[frame, side] = arm._tool_fk(assets, side, result.x) @ tool_mounts[side]
            delta = np.linalg.inv(target[frame, side]) @ actual[frame, side]
            position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000
            rotation[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
            converged[frame, side] = bool(result.success)
            independent_joint_map = {
                name: float(neutral[other_side, index])
                for other_side in range(2)
                for index, name in enumerate(ARM_JOINT_NAMES[other_side])
            }
            independent_joint_map.update({
                name: float(value) for name, value in zip(
                    ARM_JOINT_NAMES[side], result.x, strict=True)
            })
            independent_fk = forward_kinematics(
                assets.tianji, independent_joint_map)[flange_names[side]] @ flange_mounts[side]
            validated_seed = np.allclose(independent_fk, actual[frame, side], atol=3e-6) and seed_usable_solution(
                result.x, lower[side], upper[side], solver_success=bool(result.success),
                solver_status=int(result.status), position_mm=position[frame, side],
                rotation_deg=rotation[frame, side], independent_fk=independent_fk)
            # A diagnostic can change exactly one predeclared frame; all other
            # frames retain the frozen old seeding rule for causal attribution.
            previous_usable = (validated_seed if experimental_seed_recovery_frames is not None
                               and frame in experimental_seed_recovery_frames
                               else bool(result.success))
            seed_usable[frame, side] = previous_usable
            seed = result.x.copy() if previous_usable else neutral[side].copy()
            if progress and (frame % 25 == 0 or frame == count-1):
                progress(f'arm side={side} frame={frame+1}/{count}')
    tolerance_pass = valid & converged & (position <= 20.) & (rotation <= 15.)
    return dict(q_arm=q_arm, q22=arrays['q22'].copy(), wrist_valid=valid.copy(),
                finger_valid=arrays['finger_valid'].copy(), T_target_root=target,
                T_actual_root=actual, T_flange_hand=flange_mounts, T_tool_hand=tool_mounts,
                T_flange_tool=flange_tools, neutral_q_arm=neutral, neutral_roots=neutral_roots,
                position_residual_mm=position, rotation_residual_deg=rotation,
                continuity_weight=np.asarray(continuity_weight, dtype=float),
                solver_success=converged, solver_status=solver_status,
                solver_nfev=nfev, seed_usable=seed_usable, seed_in=seed_in,
                seed_source_frame=seed_source_frame, reset_reason=reset_reason,
                tolerance_pass=tolerance_pass, quality_pass=tolerance_pass.copy(),
                timestamp_ns=arrays['timestamp_ns'], frame_id=arrays['frame_id'],
                human_to_physical=HUMAN_TO_PHYSICAL, time_edge_valid=edge,
                **{'arm_'+k: v for k, v in motion_derivatives(q_arm, valid, times, edge).items()},
                **{'finger_'+k: v for k, v in motion_derivatives(arrays['q22'], arrays['finger_valid'], times, edge).items()})


class MeshScene:
    """Actual URDF visual meshes, fixed cameras, explicit collision diagnostics."""
    def __init__(self, assets, motion):
        import pybullet as bullet
        from chaoyang.ops.render_tianji_kai_mount_proxy_audit import set_robot_neutral, place_hand, body_bounds
        from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
        self.b = bullet; self.client = bullet.connect(bullet.DIRECT)
        self.assets = assets; self.motion = motion; self.arm_names = ARM_JOINT_NAMES
        self.place_hand = place_hand
        flags = bullet.URDF_USE_SELF_COLLISION | bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.robot = bullet.loadURDF(str(assets.tianji.path), useFixedBase=True, flags=flags, physicsClientId=self.client)
        set_robot_neutral(self.client, self.robot)
        self.hands = [bullet.loadURDF(str(m.path), useFixedBase=True, flags=flags, physicsClientId=self.client)
                      for m in (assets.left_hand, assets.right_hand)]
        self.hand_names = [tuple(j.name for j in m.joints if j.joint_type != 'fixed')
                           for m in (assets.left_hand, assets.right_hand)]
        self.maps = {}
        for body in [self.robot, *self.hands]:
            self.maps[body] = {bullet.getJointInfo(body, k, physicsClientId=self.client)[1].decode(): k
                              for k in range(bullet.getNumJoints(body, physicsClientId=self.client))}
        self.set_joints(self.robot, [n for side in ARM_JOINT_NAMES for n in side], motion['neutral_q_arm'].reshape(-1))
        for side, body in enumerate(self.hands):
            place_hand(self.client, body, motion['neutral_roots'][side])
        lo, hi = body_bounds(self.client, (self.robot, *self.hands))
        self.camera_center = .5 * (lo + hi)
        self.camera_distance = max(1.8, float(np.linalg.norm(hi-lo)) * 1.05)
        self.main_view = bullet.computeViewMatrixFromYawPitchRoll(self.camera_center, self.camera_distance, 40, -22, 0, 2)
        # All sessions use this asset-derived neutral camera, never target/frame fitting.
        self.projection = bullet.computeProjectionMatrixFOV(48, 640/480, .02, 20.)
        self.collision_flags = flags
        self.detail_bodies = []; self.detail_views = []
        for side, model in enumerate((assets.left_hand, assets.right_hand)):
            body = bullet.loadURDF(str(model.path), useFixedBase=True, physicsClientId=self.client)
            origin = np.eye(4); origin[0,3] = 10. * (side+1)
            place_hand(self.client, body, origin)
            self.maps[body] = {bullet.getJointInfo(body,k,physicsClientId=self.client)[1].decode(): k
                              for k in range(bullet.getNumJoints(body,physicsClientId=self.client))}
            a, z = body_bounds(self.client, (body,))
            self.detail_bodies.append(body)
            self.detail_views.append(bullet.computeViewMatrixFromYawPitchRoll(.5*(a+z), .48, 35, -35, 0, 2))

    def set_joints(self, body, names, q):
        for name, value in zip(names, q, strict=True):
            self.b.resetJointState(body, self.maps[body][name], float(value), physicsClientId=self.client)

    def _color(self, body, color):
        for link in range(-1, self.b.getNumJoints(body, physicsClientId=self.client)):
            self.b.changeVisualShape(body, link, rgbaColor=color, physicsClientId=self.client)

    def frame(self, index):
        from chaoyang.ops.render_tianji_kai_mount_proxy_audit import link_frames
        b = self.b; d = self.motion; valid = d['wrist_valid'][index]
        for side in range(2):
            q = d['q_arm'][index, side] if valid[side] else d['neutral_q_arm'][side]
            self.set_joints(self.robot, self.arm_names[side], q)
        frames = link_frames(self.client, self.robot)
        for side, body in enumerate(self.hands):
            root = frames[('flange_L', 'flange_R')[side]] @ d['T_flange_hand'][side]
            if valid[side] and not np.allclose(root, d['T_actual_root'][index, side], atol=3e-6):
                raise ValueError('BULLET_NUMPY_FK_PARITY_FAILURE')
            self.place_hand(self.client, body, root)
            finger_ok = bool(valid[side] and d['finger_valid'][index, side])
            if finger_ok:
                self.set_joints(body, self.hand_names[side], d['q22'][index, side])
            # Robot physical right is anatomical left: semantic coloring is explicit.
            color = ((.9, .2, .18, 1), (.15, .4, .95, 1))[side] if finger_ok else (.55, .55, .55, .30)
            self._color(body, color)
        b.performCollisionDetection(physicsClientId=self.client)
        counts = dict(hand_self=[None, None], arm_hand=[None, None], dual_hand=None, robot_self=None)
        for side, body in enumerate(self.hands):
            if valid[side] and d['finger_valid'][index, side]:
                counts['hand_self'][side] = sum(p[8] < 0 for p in b.getContactPoints(body, body, physicsClientId=self.client))
                counts['arm_hand'][side] = sum(p[8] < 0 for p in b.getContactPoints(self.robot, body, physicsClientId=self.client))
        if valid.all():
            counts['robot_self'] = sum(p[8] < 0 for p in b.getContactPoints(self.robot, self.robot, physicsClientId=self.client))
        if valid.all() and d['finger_valid'][index].all():
            counts['dual_hand'] = sum(p[8] < 0 for p in b.getContactPoints(*self.hands, physicsClientId=self.client))
        pixels = b.getCameraImage(640, 480, self.main_view, self.projection,
                                 renderer=b.ER_TINY_RENDERER, shadow=0, physicsClientId=self.client)[2]
        detail = []
        for side, body in enumerate(self.detail_bodies):
            good = bool(valid[side] and d['finger_valid'][index,side])
            q = d['q22'][index,side] if good else np.zeros(22)
            self.set_joints(body, self.hand_names[side], q)
            self._color(body, ((.9,.2,.18,1),(.15,.4,.95,1))[side] if good else (.6,.6,.6,.35))
            pix = b.getCameraImage(320,240,self.detail_views[side],self.projection,
                                  renderer=b.ER_TINY_RENDERER,shadow=0,physicsClientId=self.client)[2]
            detail.append(np.asarray(pix,np.uint8).reshape(240,320,4)[...,:3].copy())
        return np.asarray(pixels, np.uint8).reshape(480, 640, 4)[..., :3].copy(), counts, detail

    def close(self):
        self.b.disconnect(self.client)


def render_review(motion, assets, video_path, destination, session_id, *, preview_frames=()):
    import cv2
    import subprocess
    scene = MeshScene(assets, motion)
    capture = cv2.VideoCapture(str(video_path)); count = len(motion['frame_id'])
    if not capture.isOpened():
        scene.close(); raise ValueError('SOURCE_VIDEO_OPEN_FAILED')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '2', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1280x788', '-r', '30', '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    collision_rows = []; source_index = -1
    try:
        for t in range(count):
            requested = int(motion['frame_id'][t])
            while source_index < requested:
                ok, raw = capture.read(); source_index += 1
                if not ok: raise ValueError('SOURCE_VIDEO_TRUNCATED')
            mesh, collisions, detail = scene.frame(t); collision_rows.append(collisions)
            canvas = np.full((788, 1280, 3), 245, np.uint8)
            # Letterbox the upstream diagnostic; no image-domain anisotropic warp.
            h,w = raw.shape[:2]; scale = min(640/w,480/h)
            rw,rh = int(round(w*scale)),int(round(h*scale))
            x,y = (640-rw)//2,68+(480-rh)//2
            canvas[y:y+rh,x:x+rw] = cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),(rw,rh),interpolation=cv2.INTER_AREA)
            canvas[68:548, 640:] = mesh
            canvas[548:,640:960] = detail[1]  # anatomical left / physical right
            canvas[548:,960:] = detail[0]
            titles = [session_id+' | source / reference at left | full Tianji + KaiHand at right',
                      'VIRTUAL FIXED MOUNT, UNIT MOTION GAIN | NOT CALIBRATED / NOT FOR TRAINING',
                      f'frame {requested} | wrist valid {motion["wrist_valid"][t].astype(int).tolist()} | gray = no observed motion']
            for k, title in enumerate(titles):
                cv2.putText(canvas, title, (8,18+k*21), cv2.FONT_HERSHEY_SIMPLEX, .47, (20,20,20), 1)
            lines = ['LOCAL HAND VIEWS: fixed asset-frame cameras (no wrist motion)',
                     'BLUE anatomical LEFT -> robot physical RIGHT',
                     'RED anatomical RIGHT -> robot physical LEFT',
                     'Gray neutral geometry is display-only, never an observation.',
                     'Arm target residual mm: '+str(np.round(motion['position_residual_mm'][t],1).tolist()),
                     'Arm solver / tolerance: '+str(motion['tolerance_pass'][t].astype(int).tolist()),
                     'No object geometry; fingertip origins are not contact points.']
            for k,line in enumerate(lines):
                cv2.putText(canvas,line,(8,570+28*k),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
            writer.stdin.write(canvas.tobytes())
            if t in preview_frames:
                cv2.imwrite(str(Path(destination).with_name(f'preview_{t:06d}.png')), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        writer.stdin.close(); error = writer.stderr.read().decode(); rc = writer.wait()
        if rc: raise RuntimeError('FFMPEG_FAILED:'+error[-2000:])
    finally:
        capture.release(); scene.close()
        if writer.poll() is None:
            writer.terminate(); writer.wait()
    decoded = 0; check = cv2.VideoCapture(str(destination))
    while check.read()[0]: decoded += 1
    check.release()
    if decoded != count: raise ValueError('FULL_OUTPUT_DECODE_COUNT')
    return dict(decoded_frames=decoded, collision_rows=collision_rows,
                collision_flags=int(scene.collision_flags), camera_center=scene.camera_center.tolist(),
                camera_distance=scene.camera_distance, camera_policy='ASSET_NEUTRAL_FIXED_ALL_FRAMES',
                collision_limitations=['URDF collision geometry only', 'No object/environment geometry',
                                      'Non-adjacent self contacts only; directly adjacent links excluded',
                                      'Arm-hand mount contacts reported without suppression',
                                      'Full robot/dual-hand contacts unknown unless both wrists valid'],
                display_fps=30, acquisition_time='timestamp_ns in MOTION_RESULT.npz')
