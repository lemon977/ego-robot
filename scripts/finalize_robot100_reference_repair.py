"""Isolated handoff packaging only: no shared publication or algorithm execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

REPO=Path('/path/to/chaoyang')
ROOT=REPO/'_run/current/robot100_v16_reference_repair_20260929/attempts/attempt_0001'

def ref(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def write(name,value):
    with (ROOT/name).open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    for path in ('src/chaoyang/pipeline/robot100_v16_replay_safety.py','src/chaoyang/governance/operation_retirement.py'):
        target=ROOT/Path(path).name
        with target.open('xb') as f:f.write((REPO/path).read_bytes())
    spec=importlib.util.spec_from_file_location('guard_tests',ROOT/'test_robot100_v16_replay_safety.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    with (ROOT/'CPU_TEST_LOG.txt').open('x') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    write('CPU_TEST_RESULTS.json',{'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':result.skipped,'passed':result.wasSuccessful(),'scope':'Guard functions only; not production replay or image accuracy'})
    if not result.wasSuccessful():raise RuntimeError('GUARD_TEST_FAILURE')
    manifest=json.loads((ROOT/'REPRODUCTION_MANIFEST.json').read_text())
    new_paths=['src/chaoyang/pipeline/robot100_v16_replay_safety.py','src/chaoyang/ops/run_robot100_v16_safe_replay.py','src/chaoyang/governance/operation_retirement.py']
    for path in new_paths:
        p=REPO/path;manifest['runtime_code'].append(ref(p))
        snapshot=ROOT/'source_snapshot'/path;snapshot.parent.mkdir(parents=True,exist_ok=True)
        with snapshot.open('xb') as f:f.write(p.read_bytes())
        manifest['snapshot_refs'].append(ref(snapshot))
    verified=[]
    for category in ('runtime_code','frozen_inputs','assets','snapshot_refs'):
        for item in manifest[category]:
            tests.SAFETY.read_verified(item);verified.append({'category':category,**item})
    tests.SAFETY.read_verified(manifest['environment'])
    manifest['status']='PATCH_READY_NOT_INTEGRATED'
    manifest['preparation_predecessor']=ref(ROOT/'REPRODUCTION_MANIFEST.json')
    write('REPRODUCTION_MANIFEST_READY.json',manifest)
    write('REFERENCE_VERIFICATION.json',{'status':'PASS','reference_count':len(verified)+1,'references':verified,'environment':manifest['environment'],'selected_video_unchanged':True,'historical_manifest_rewritten':False,'governance_drift_resolved_by_new_scope_not_old_sha_edit':True})
    retired={}
    for item in manifest['runtime_code']:
        name=Path(item['path']).stem
        if name.startswith('run_robot100_') and name!='run_robot100_v16_safe_replay':
            retired[name]={'replacement':'run_robot100_v16_safe_replay','reason':'Historical fixed-output entry. Retained as import dependency, not a current replay command.'}
    write('RETIRED_OPERATION_ENTRYPOINTS.proposed.json',{'schema_version':'CHAOYANG_RETIRED_ENTRYPOINTS_V1','operations':retired,'scope':'0916 only. Merge with any existing registry, never replace other branches entries.'})
    patches=[]
    for path in new_paths:
        proc=subprocess.run(['git','diff','--no-index','--','/dev/null',path],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if proc.returncode not in (0,1):raise RuntimeError(proc.stderr.decode())
        patches.append(proc.stdout)
    patches.append((REPO/'scripts/robot100_cli_retirement.patch').read_bytes())
    with (ROOT/'maintenance.patch').open('xb') as f:f.write(b'\n'.join(patches))
    write('RESULT_REFERENCE_REPAIR.json',{'status':'PATCH_READY_NOT_INTEGRATED','manifest':ref(ROOT/'REPRODUCTION_MANIFEST_READY.json'),'tests':ref(ROOT/'CPU_TEST_RESULTS.json'),'references':ref(ROOT/'REFERENCE_VERIFICATION.json'),'patch':ref(ROOT/'maintenance.patch'),'production_replay_executed':False,'shared_entrypoint_modified':False,'old_entrypoints_disabled':False,'historical_files_deleted':False,'git_committed':False,'other_ai_untouched':True,'remaining':['Serial publisher integration and registered fresh-output replay','Archive stale navigation with references and recovery map','Commit only owned files without staging concurrent Exact78 edits']})
    print(json.dumps({'status':'PATCH_READY_NOT_INTEGRATED','tests':result.testsRun,'references':len(verified)+1,'root':str(ROOT)}))

if __name__=='__main__':main()
