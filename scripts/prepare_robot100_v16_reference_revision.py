"""Prepare an isolated, non-authoritative reference revision; no algorithm or shared write."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO=Path('/path/to/chaoyang')
ROOT=REPO/'_run/current/robot100_v16_reference_repair_20260929/attempts/attempt_0001'
SELECTED=REPO/'_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/BASELINE.json'

def ref(path):
    path=Path(path);before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns): raise RuntimeError('UNSTABLE:'+str(path))
    return {'path':str(path),'bytes':after.st_size,'sha256':h.hexdigest()}

def write(path,value):
    with path.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    b=json.loads(SELECTED.read_text()); runtime=[];snapshots=[];changes=[]
    for item in b['code_closure']:
        path=Path(item['path']);current=ref(path)
        governance='/src/chaoyang/governance/' in str(path)
        if current!=item:
            if not governance: raise RuntimeError('NUMERICAL_SOURCE_CHANGED:'+str(path))
            changes.append({'old_reference':item,'current_reference':current,'scope':'MUTABLE_GOVERNANCE_NOT_NUMERICAL_REPLAY','historical_bytes_recovered':False})
        target=ROOT/'source_snapshot'/path.relative_to(REPO)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(path.read_bytes())
        assert ref(target)['sha256']==current['sha256']
        snapshots.append(ref(target))
        if not governance: runtime.append(current)
    dynamic=REPO/'src/chaoyang/ops/immutable_artifact_io.py'
    runtime.append(ref(dynamic))
    target=ROOT/'source_snapshot'/dynamic.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f: f.write(dynamic.read_bytes())
    snapshots.append(ref(target))
    # Static import closure alone is not claimed to contain assets or dynamic imports.
    pin_path=REPO/'assets/robot/ROBOT_ASSET_PIN.json';pin=json.loads(pin_path.read_text())
    assets=[ref(pin_path)]
    for item in pin['files']:
        current=ref(REPO/item['path'])
        assert current['sha256']==item['sha256'] and current['bytes']==item['bytes'], 'ASSET_PIN_MISMATCH'
        assets.append(current)
    mount=REPO/'_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json'
    value=json.loads(mount.read_text());assets.extend([ref(mount),ref(Path(value['decoded_review_mesh']['path']))])
    for font in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/wqy/wqy-microhei.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
        if Path(font).is_file(): assets.append(ref(font));break
    v16=json.loads(Path(b['frozen_result']['path']).read_text());v15=json.loads(Path(v16['predecessor']['path']).read_text())
    inputs=[ref(SELECTED),b['frozen_result'],b['motion'],b['reviewed_video'],v16['predecessor'],v15['motion']]+[b['input'][k] for k in ('hdf5','video','clip_manifest')]
    for item in inputs: assert ref(Path(item['path']))==item,'FROZEN_INPUT_CHANGED'
    environment={'python_executable':sys.executable,'python':sys.version,'platform':platform.platform(),'packages':{},'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]}
    for name in ('numpy','scipy','h5py','opencv-python','Pillow','pybullet'):
        try: environment['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: environment['packages'][name]='NOT_DISTRIBUTION_REGISTERED'
    write(ROOT/'ENVIRONMENT.json',environment)
    manifest={'schema_version':'ROBOT100_V16_REPRODUCTION_REVISION_2','status':'PREPARED_NOT_PUBLISHED','baseline_id':b['baseline_id'],'selected_video':b['reviewed_video'],'selected_motion':b['motion'],'v15_motion':v15['motion'],'source_row':b['input'],'predecessor':ref(SELECTED),'runtime_code':runtime,'frozen_inputs':inputs,'assets':assets,'snapshot_refs':snapshots,'control_plane_changes':changes,'environment':ref(ROOT/'ENVIRONMENT.json'),'claim':'Reference repair only. Accepted video and motion unchanged. Mutable governance snapshot is provenance, not numerical algorithm identity. No retrospective replacement of old SHA.'}
    write(ROOT/'REPRODUCTION_MANIFEST.json',manifest)
    write(ROOT/'PREPARATION_RESULT.json',{'status':'PREPARED_NOT_PUBLISHED','manifest':ref(ROOT/'REPRODUCTION_MANIFEST.json'),'runtime_code_count':len(runtime),'asset_refs':len(assets),'source_snapshots':len(snapshots),'drift_explicitly_recorded':changes,'shared_files_modified':False,'algorithms_executed':False})
    print(json.dumps({'prepared':str(ROOT),'runtime':len(runtime),'assets':len(assets),'drifts':len(changes)}))

if __name__=='__main__':main()
