"""Frozen-config CPU admission ledger; no model, IK, renderer or discovery."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import numpy as np
from chaoyang.pipeline.motion_four_certificate_ledger_v1 import generate, KINDS

CANONICAL = Path('/mnt/workspace/code/chaoyang')
PARENT = 'four_stream_completion_20260928'
CHILD = PARENT + '_motion'
SESSIONS = {'get_potato_chips_0915_007':378, 'play_cards_0915_031':149}

def confined(repo, path):
    p=Path(path)
    if not p.is_absolute(): p=repo/p
    if '..' in p.parts or not p.is_relative_to(repo): raise ValueError('PATH_ESCAPE')
    for part in [p,*p.parents]:
        if part==repo: break
        if part.is_symlink(): raise ValueError('SYMLINK_REFUSED')
    if not p.resolve().is_relative_to(repo): raise ValueError('PATH_ESCAPE')
    return p

def stable_bytes(repo, path):
    p=confined(repo,path)
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>32*1024*1024: raise ValueError('REFERENCE_SIZE_OR_TYPE')
        with os.fdopen(fd,'rb',closefd=False) as f: b=f.read(32*1024*1024+1)
        after=os.fstat(fd); now=p.stat(follow_symlinks=False)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(before)!=identity(after) or identity(after)!=identity(now) or len(b)!=after.st_size: raise ValueError('UNSTABLE_READ')
        return b
    finally: os.close(fd)

def verified(repo, ref):
    b=stable_bytes(repo,ref['path'])
    if len(b)!=ref['bytes'] or hashlib.sha256(b).hexdigest()!=ref['sha256']: raise ValueError('SHA_OR_BYTES_DRIFT')
    return b

def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])

def guard(repo,cfg):
    if repo!=CANONICAL or repo.resolve()!=repo: raise ValueError('CANONICAL_REPO_REQUIRED')
    index=json.loads(stable_bytes(repo,'tasks/current/INDEX.json'))
    packets={}
    for task in (PARENT,CHILD):
        rows=[r for r in index['task_packets'] if r['task_id']==task]
        if len(rows)!=1 or (task==PARENT and rows[0].get('execution_allowed') is not True): raise ValueError('ROUTING')
        r=rows[0]; ref=cfg['packets'][task]; expected=repo/'tasks/current'/task/'TASK_PACKET.json'
        if confined(repo,r['packet_path'])!=expected or confined(repo,ref['path'])!=expected or r['packet_sha256']!=ref['sha256']: raise ValueError('PACKET_BINDING')
        packets[task]=json.loads(verified(repo,ref))
    writer=packets[CHILD]['writer']
    if writer!=packets[PARENT]['writer'] or writer!=cfg['writer'] or ticks(writer['pid'])!=writer['proc_start_ticks']: raise ValueError('WRITER_FENCE')
    attempt=cfg['attempt']
    if not re.fullmatch('attempt_[0-9]+',attempt): raise ValueError('ATTEMPT')
    lane=repo/'_run/current'/PARENT/'attempts'/attempt/'lanes/motion'
    if str(lane)!=packets[CHILD]['output_root'] or str(lane.relative_to(repo)) not in packets[CHILD]['write_set']: raise ValueError('WRITE_SET')
    confined(repo,lane)
    return lane

def run(config, config_sha256):
    repo=CANONICAL
    raw=stable_bytes(repo,config)
    if hashlib.sha256(raw).hexdigest()!=config_sha256: raise ValueError('CONFIG_SHA_DRIFT')
    cfg=json.loads(raw)
    if cfg.get('schema_version')!='MOTION_LEDGER_FROZEN_CONFIG_V1': raise ValueError('CONFIG_SCHEMA')
    session=cfg['session_id']
    if session not in SESSIONS: raise ValueError('SESSION_SCOPE')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU_ENV_REQUIRED')
    lane=guard(repo,cfg)
    name=cfg['output_name']
    if not re.fullmatch('[A-Za-z0-9_]+[.]json',name): raise ValueError('OUTPUT_NAME')
    target=confined(repo,lane/name)
    if target.exists(): raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    ref=cfg['source']; data=verified(repo,ref)
    with np.load(io.BytesIO(data),allow_pickle=False) as h:
        frames=h['original_frame_indices'].tolist(); valid=h['predicted_valid']
        if frames!=list(range(SESSIONS[session])) or valid.shape!=(2,len(frames)) or valid.dtype!=np.bool_: raise ValueError('FULL_FRAME_MAPPING')
        if tuple(h['anatomical_side_names'].tolist())!=('left','right'): raise ValueError('SOURCE_SIDE_MAPPING')
        rows=[{'frame_id':f,'side':s,'valid':bool(valid[j,i]),'kind':'MODEL_INFERENCE' if valid[j,i] else 'INVALID'} for i,f in enumerate(frames) for j,s in enumerate(('left','right'))]
        for flag,kind in [('short_gap_inferred','TIME_INTERPOLATION'),('held','HOLD'),('copied_from_other_side','COPIED_OTHER_SIDE')]:
            if flag in h:
                mask=h[flag]
                if mask.shape!=valid.shape or mask.dtype!=np.bool_: raise ValueError('PROVENANCE_FLAG_SHAPE')
                for i in range(len(frames)):
                    for j in range(2):
                        if mask[j,i]: rows[2*i+j]['kind']=kind
        # Source availability never becomes physical observation or independent quality.
    # Producer receipt binds session to the source: filenames alone are insufficient.
    receipt=json.loads(verified(repo,cfg['source_receipt']))
    if receipt.get('session_id')!=session or receipt.get('inputs',{}).get('hawor_source')!=ref: raise ValueError('SOURCE_SESSION_BINDING')
    certs={}; checked=[]
    if set(cfg.get('certificates',{}))-set(KINDS): raise ValueError('CERTIFICATE_KIND')
    for kind, cref in cfg.get('certificates',{}).items():
        certs[kind]=json.loads(verified(repo,cref))['rows']
    for eref in cfg.get('evidence',[]):
        verified(repo,eref); checked.append(dict(eref,bytes_verified=True,stable_read=True))
    result=generate(session_id=session,source_sha256=ref['sha256'],frame_ids=frames,source_rows=rows,certificates=certs,verified_refs=checked)
    result['frozen_config']={'path':str(confined(repo,config)),'bytes':len(raw),'sha256':config_sha256}
    result['source_receipt']=cfg['source_receipt']; result['source']=ref
    result['certificate_read_scope']='Only explicitly frozen certificates; missing does not prove global absence'
    result['source_valid_meaning']='Producer availability, not independent quality'
    # Fence again immediately before exclusive creation; no overwrite/resume guess.
    guard(repo,cfg)
    with target.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return {'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'product_completed':False}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--config-sha256',required=True)
    a=p.parse_args(argv); print(json.dumps(run(a.config,a.config_sha256)))

if __name__=='__main__': main()
