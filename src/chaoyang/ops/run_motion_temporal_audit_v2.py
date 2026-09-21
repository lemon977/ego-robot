"""Independent CPU temporal/amplitude accounting; never adopts an algorithm."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np

DEFINITIONS={
    'time':'int64 acquisition ns, original frame_id; display time is separate',
    'edge':'consecutive frame IDs, increasing ns, dt <= 2.5 * median positive dt, and optional unchanged segment_id',
    'velocity':'(q[t]-q[t-1])/(time[t]-time[t-1]), located at time midpoint',
    'acceleration':'difference of adjacent velocities divided by their midpoint-time difference',
    'jerk':'difference of adjacent accelerations divided by their midpoint-time difference',
    'gap':'NaN derivatives propagate; no derivative bridges an invalid side or disconnected edge',
    'angular_amplitude':'per-joint max-minus-min of actual exported raw joint angle radians (may violate limits; no clipping); no unwrap; global and segment-local ranges separate',
    'wrist_amplitude':'per-axis max-minus-min in the declared common virtual frame, metres; no alignment/lag/scale fitting',
    'following':'same frame_id/time target minus actual translation, Euclidean mm; never searches lag',
    'adoption':'lower jerk alone is not improvement; zero-motion actual with moving target is a failure control, not success',
}

def edges(timestamp_ns,frame_id,segment_id=None):
    ns=np.asarray(timestamp_ns);ids=np.asarray(frame_id)
    if ns.dtype!=np.int64 or ids.dtype!=np.int64 or ns.ndim!=1 or ids.shape!=ns.shape or len(ns)<2:
        raise ValueError('INT64_TIME_AND_FRAME_REQUIRED')
    dt=np.diff(ns)
    if np.any(dt<=0) or np.any(np.diff(ids)<=0):raise ValueError('NONMONOTONIC_TIME_OR_FRAME')
    good=(np.diff(ids)==1)&(dt<=2.5*np.median(dt))
    if segment_id is not None:
        seg=np.asarray(segment_id)
        if seg.shape!=ns.shape:raise ValueError('SEGMENT_SHAPE')
        good &= seg[1:]==seg[:-1]
    return np.r_[False,good],(ns-ns[0]).astype(float)*1e-9

def derivatives(values,valid,times,edge):
    q=np.asarray(values,float);mask=np.asarray(valid)
    if q.ndim!=3 or mask.dtype!=bool or q.shape[:2]!=mask.shape or len(q)!=len(times):
        raise ValueError('DERIVATIVE_SHAPE_OR_MASK')
    if not np.isfinite(q[mask]).all():raise ValueError('NONFINITE_VALID_VALUES')
    v=np.full_like(q,np.nan);a=v.copy();j=v.copy();dt=np.diff(times)
    if np.any(dt<=0):raise ValueError('NONPOSITIVE_DT')
    eligible=mask[1:]&mask[:-1]&edge[1:,None]
    v[1:]=np.where(eligible[...,None],np.diff(q,axis=0)/dt[:,None,None],np.nan)
    vt=(times[1:]+times[:-1])/2
    if len(q)>2:a[2:]=np.diff(v[1:],axis=0)/np.diff(vt)[:,None,None]
    at=(vt[1:]+vt[:-1])/2
    if len(q)>3:j[3:]=np.diff(a[2:],axis=0)/np.diff(at)[:,None,None]
    return dict(velocity=v,acceleration=a,jerk=j)

def segments(valid,edge):
    result=[];start=None
    for t,good in enumerate(valid):
        if start is not None and (not good or not edge[t]):result.append((start,t));start=None
        if good and start is None:start=t
    if start is not None:result.append((start,len(valid)))
    return result

def finite_stats(values):
    a=np.asarray(values,float);v=a[np.isfinite(a)]
    if not v.size:return dict(count=0,p50_abs=None,p95_abs=None,max_abs=None,rms=None)
    return dict(count=int(v.size),p50_abs=float(np.percentile(abs(v),50)),p95_abs=float(np.percentile(abs(v),95)),
        max_abs=float(abs(v).max()),rms=float(np.sqrt(np.mean(v*v))))

def angular_metrics(q,valid,times,edge):
    deriv=derivatives(q,valid,times,edge);sides=[]
    for side in range(2):
        v=valid[:,side];spans=segments(v,edge);n=int(v.sum())
        global_range=np.ptp(q[v,side],axis=0) if n else np.full(q.shape[-1],np.nan)
        local=[np.ptp(q[a:b,side],axis=0) for a,b in spans]
        local_max=np.max(local,axis=0) if local else np.full(q.shape[-1],np.nan)
        encode=lambda arr:[float(x) if np.isfinite(x) else None for x in arr]
        sides.append(dict(physical_side=side,full_timeline_frames=len(q),valid_frames=n,missing_frames=len(q)-n,
            segment_count=len(spans),segments_half_open=[[a,b] for a,b in spans],
            global_joint_range_rad=encode(global_range),max_within_segment_joint_range_rad=encode(local_max),
            global_max_joint_range_rad=float(global_range.max()) if n else None,
            derivative_stats={k:finite_stats(a[:,side]) for k,a in deriv.items()},
            derivative_valid_frames={k:int(np.isfinite(a[:,side]).all(-1).sum()) for k,a in deriv.items()}))
    return dict(units=dict(angle='rad',velocity='rad/s',acceleration='rad/s^2',jerk='rad/s^3'),sides=sides),deriv

def wrist_metrics(target,actual,valid,edge):
    rows=[]
    for side in range(2):
        mask=valid[:,side];n=int(mask.sum());spans=segments(mask,edge)
        if not n:
            rows.append(dict(physical_side=side,valid_frames=0,missing_frames=len(mask),target_range_m=None,actual_range_m=None,
                zero_lag_following_error_mm=finite_stats([]),motion_state='NO_VALID_TARGET'));continue
        t=target[mask,side,:3,3];a=actual[mask,side,:3,3]
        if not np.isfinite(t).all() or not np.isfinite(a).all():raise ValueError('NONFINITE_VALID_WRIST')
        tr=np.ptp(t,axis=0);ar=np.ptp(a,axis=0)
        target_span=float(np.linalg.norm(tr));actual_span=float(np.linalg.norm(ar))
        state='FROZEN_OUTPUT_WITH_MOVING_TARGET' if target_span>1e-9 and actual_span<=1e-9 else 'DESCRIPTIVE_ONLY_NOT_ADOPTION'
        segment_rows=[]
        for start,end in spans:
            segment_rows.append(dict(start=start,end=end,target_range_m=np.ptp(target[start:end,side,:3,3],axis=0).tolist(),
                actual_range_m=np.ptp(actual[start:end,side,:3,3],axis=0).tolist()))
        rows.append(dict(physical_side=side,full_timeline_frames=len(mask),valid_frames=n,missing_frames=len(mask)-n,
            target_range_m=tr.tolist(),actual_range_m=ar.tolist(),target_range_norm_m=target_span,actual_range_norm_m=actual_span,
            amplitude_norm_ratio=actual_span/target_span if target_span>1e-9 else None,
            zero_lag_following_error_mm=finite_stats(np.linalg.norm(a-t,axis=-1)*1000),
            within_segment_ranges=segment_rows,motion_state=state))
    return rows

def evaluate(arrays):
    qarm=np.asarray(arrays['q_arm']);qfinger=np.asarray(arrays['q22'])
    w=np.asarray(arrays['wrist_valid']);f=np.asarray(arrays['finger_valid']);ns=arrays['timestamp_ns'];ids=arrays['frame_id']
    count=len(ns)
    if qarm.shape!=(count,2,7) or qfinger.shape!=(count,2,22) or w.shape!=(count,2) or f.shape!=w.shape or w.dtype!=bool or f.dtype!=bool:
        raise ValueError('MOTION_SHAPE_OR_MASK')
    edge,times=edges(ns,ids,arrays.get('segment_id'))
    if 'time_edge_valid' in arrays and not np.array_equal(edge,arrays['time_edge_valid']):raise ValueError('RECORDED_EDGE_MISMATCH')
    if not np.array_equal(arrays['human_to_physical'],[1,0]):raise ValueError('SIDE_MAPPING_MISMATCH')
    target=np.asarray(arrays['T_target_root']);actual=np.asarray(arrays['T_actual_root'])
    if target.shape!=(count,2,4,4) or actual.shape!=target.shape:raise ValueError('WRIST_TRANSFORM_SHAPE')
    arm,ad=angular_metrics(qarm,w,times,edge);finger,fd=angular_metrics(qfinger,f,times,edge)
    parity={}
    for prefix,derived in [('arm',ad),('finger',fd)]:
        for name,value in derived.items():
            key=prefix+'_'+name
            if key in arrays:
                if not np.allclose(value,arrays[key],atol=1e-7,rtol=1e-7,equal_nan=True):raise ValueError('DERIVATIVE_PARITY_'+key)
                parity[key]='MATCH'
            else:parity[key]='NOT_STORED_RECOMPUTED'
    return dict(full_timeline_frames=count,acquisition_duration_seconds=float(times[-1]),
        dt_seconds=finite_stats(np.diff(times)),connected_edges=int(edge.sum()),disconnected_edges=int(count-1-edge.sum()),
        arm=arm,finger=finger,wrist=wrist_metrics(target,actual,w,edge),stored_derivative_parity=parity,
        NUMERIC_QUALITY_PASS=False,TRAINING_ELIGIBLE=False,comparison='NO_AUTOMATIC_CROSS_INPUT_OR_METHOD_COMPARISON')

def frozen_read(ref,allowed):
    p=Path(ref['path']).resolve(strict=True)
    if not p.is_relative_to(allowed):raise ValueError('INPUT_SCOPE_ESCAPE')
    before=p.stat()
    if before.st_size>32*1024*1024:raise ValueError('INPUT_SIZE_LIMIT')
    data=p.read_bytes();after=p.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('UNSTABLE_INPUT')
    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('REFERENCE_DRIFT')
    return data

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    root=Path(os.environ['CHAOYANG_REPO_ROOT']).resolve(strict=True)
    attempt=root/'_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001'
    expected=attempt/'lanes/exact78/temporal_audit_v2'
    if args.output.resolve()!=expected.resolve() or args.output.exists():raise ValueError('OUTPUT_SCOPE_OR_EXISTS')
    if not args.spec.resolve(strict=True).is_relative_to(attempt/'lanes/exact78'):raise ValueError('SPEC_SCOPE')
    spec_bytes=args.spec.read_bytes();spec=json.loads(spec_bytes)
    if spec['schema_version']!='FOUR_STREAM_TEMPORAL_AUDIT_SPEC_V2' or len(spec['cases'])!=9:raise ValueError('SPEC_SCHEMA_OR_COUNT')
    known=set();rows=[];total=0;start=time.monotonic()
    for case in spec['cases']:
        key=(case['lane'],case['session_id'])
        if key in known:raise ValueError('DUPLICATE_CASE')
        known.add(key);data=frozen_read(case['motion_result'],attempt);total+=len(data)
        if total>64*1024*1024:raise ValueError('TOTAL_READ_BUDGET')
        with np.load(io.BytesIO(data),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        row=evaluate(arrays)
        if row['full_timeline_frames']!=case['expected_frames']:raise ValueError('EXPECTED_FRAME_COUNT')
        fps=float(case['display_fps'])
        if not np.isfinite(fps) or fps<=0:raise ValueError('DISPLAY_FPS')
        row.update(lane=case['lane'],session_id=case['session_id'],source=case['motion_result'],display_fps=fps,
            display_duration_seconds=row['full_timeline_frames']/fps,display_fps_source='FROZEN_SPEC_NOT_VIDEO_REDECODE')
        rows.append(row);print(json.dumps(dict(case=key,status='EVALUATED',frames=row['full_timeline_frames'])),flush=True)
    args.output.mkdir(parents=True)
    result=dict(schema_version='FOUR_STREAM_TEMPORAL_AUDIT_RESULT_V2',status='DESCRIPTIVE_AUDIT_COMPLETE_NOT_ALGORITHM_ADOPTION',
        definitions=DEFINITIONS,cases=rows,input_bytes=total,elapsed_seconds=time.monotonic()-start,gpu_used=False,
        source_data_modified=False,spec=dict(path=str(args.spec.resolve()),bytes=len(spec_bytes),sha256=hashlib.sha256(spec_bytes).hexdigest()))
    with (args.output/'RESULT.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    fmt=lambda value:'—' if value is None else f'{value:.6g}'
    lines=['# 四线时间、幅度与跟随误差：只读评价', '',
        '完整机器结果：[RESULT.json](RESULT.json)。仅描述工程数值，不产生算法采用、训练或真实精度授权。', '',
        '物理 side0 对应解剖右手，side1 对应解剖左手。所有分母保留完整原时间轴；没有搜索时移、缩放或重设目标。', '',
        '## 腕部与覆盖', '',
        '| 支线/会话/物理侧 | 有效/全帧 | 目标范围范数 m | 实际范围范数 m | 同帧误差 p95 mm | 状态 |',
        '|---|---:|---:|---:|---:|---|']
    for row in rows:
        for side,wrist in enumerate(row['wrist']):
            lines.append(f"| {row['lane']}/{row['session_id']}/{side} | {wrist['valid_frames']}/{row['full_timeline_frames']} | {fmt(wrist.get('target_range_norm_m'))} | {fmt(wrist.get('actual_range_norm_m'))} | {fmt(wrist['zero_lag_following_error_mm']['p95_abs'])} | {wrist['motion_state']} |")
    lines += ['', '## 关节幅度与离散 jerk', '',
        '幅度为全有效帧逐关节 max-minus-min 中的最大值；跨断点总范围不等于段内运动，逐段范围另存 JSON。jerk 只在连续有效链上计算。', '',
        '| 支线/会话/物理侧 | 臂幅度 rad | 指幅度 rad | 臂 jerk p95 rad/s³ | 指 jerk p95 rad/s³ |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        for side in range(2):
            a=row['arm']['sides'][side];f=row['finger']['sides'][side]
            lines.append(f"| {row['lane']}/{row['session_id']}/{side} | {fmt(a['global_max_joint_range_rad'])} | {fmt(f['global_max_joint_range_rad'])} | {fmt(a['derivative_stats']['jerk']['p95_abs'])} | {fmt(f['derivative_stats']['jerk']['p95_abs'])} |")
    lines += ['', '## 时间与解释', '',
        '| 支线/会话 | 采集跨度 s | 展示时长 s | 断边数 |', '|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['lane']}/{row['session_id']} | {fmt(row['acquisition_duration_seconds'])} | {fmt(row['display_duration_seconds'])} | {row['disconnected_edges']} |")
    lines += ['', '速度位于原采样间隔中点，加速度位于相邻速度时刻中点，jerk 再取相邻加速度之差；全部除以相应真实时间差。缺手、frame gap、超过 2.5×median dt 的时间缺口和声明的重置均断开导数。', '',
        '展示时长来自冻结 spec 的 FPS 与帧数，本入口没有重解码视频；视频完整解码证据在各原结果及独立复核。', '',
        '冻结输出也可有零 jerk：只要目标移动而实际严格不动就标 FROZEN_OUTPUT_WITH_MOVING_TARGET。幅度比例接近1、jerk更低或同帧残差更小也都不自动证明真实贴合。不同输入的结果不直接横向排序，缺覆盖不能因缺少 jerk 样本被视为更优。']
    with (args.output/'README_ZH.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
