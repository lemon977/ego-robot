"""Delete exactly one obsolete untracked navigation backup after safety checks."""
from pathlib import Path
import os
import subprocess
from chaoyang.governance.common import REPO_ROOT,artifact_ref,atomic_json,load_json
ROOT=REPO_ROOT/'_run/current/robot100_retire_stale_navigation_20260929/attempts/attempt_0001'
TARGET=REPO_ROOT/'docs/current/ROBOT100_BATCH_ZH.md.orig'

def main():
    assert TARGET.resolve()==TARGET and TARGET.is_file() and not TARGET.is_symlink()
    stat=TARGET.stat(); assert stat.st_nlink==1
    assert not subprocess.check_output(['git','ls-files','--',str(TARGET.relative_to(REPO_ROOT))],cwd=REPO_ROOT).strip(),'TRACKED_FILE'
    refs=[]
    for folder in ('docs/current','tasks/current'):
        for path in (REPO_ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.md','.json') and path.stat().st_size<4*1024*1024:
                # The cleanup task packet itself is an authorized target record, not a consumer.
                if 'robot100_retire_stale_navigation_20260929' in str(path): continue
                if TARGET.name in path.read_text(encoding='utf8',errors='replace'): refs.append(str(path))
    assert not refs, 'CURRENT_REFERENCE_FOUND'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if fd.resolve()==TARGET: raise RuntimeError('ACTIVE_FD')
                except FileNotFoundError: pass
        except (FileNotFoundError,ProcessLookupError): pass
        except PermissionError: raise RuntimeError('PROCESS_PROOF_INCOMPLETE')
    ref=artifact_ref(TARGET); data=TARGET.read_bytes()
    recovery=ROOT/'RETIRED_NAVIGATION_BACKUP.bin'
    with recovery.open('xb') as f: f.write(data)
    assert artifact_ref(recovery)['sha256']==ref['sha256']
    current=TARGET.stat()
    assert (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)
    assert artifact_ref(TARGET)==ref
    TARGET.unlink()
    atomic_json(ROOT/'RESULT.json',{'status':'COMPLETE','deleted':[ref],'recovery':artifact_ref(recovery),'logical_deleted_bytes':len(data),'space_reclaimed_claimed':False,'scope':'one untracked stale navigation backup; old required code and sealed evidence retained','source_data_modified':False,'other_branch_modified':False})
    return 0

if __name__=='__main__': raise SystemExit(main())
