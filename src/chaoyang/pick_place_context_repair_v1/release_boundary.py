"""Readonly release boundary. Never stops actors, releases leases, or starts work."""
from pathlib import Path
import json,hashlib
from chaoyang.governance.common import REPO_ROOT,process_identity
from chaoyang.pick_place_context_repair_v1.approval import exact,ref,read,OPERATION
from chaoyang.pick_place_1008.stage_reuse import TERMINAL_STATUSES
STATUS='ROOT_EXPLICIT_CONTEXT_PAD4_NEW_FAMILY_PUBLICATION_APPROVED'
ACTIVE={'RUNNING','CLAIMED','WAIT_GPU_RESOURCE'}
def full_partition(queue):
    records=queue['records'];gates=queue['gates'];parts=gates+records
    if len(records)!=20 or len(gates)!=2 or queue['goal_total_sessions']!=199 or queue['goal_total_native_frames']!=29322:
        raise RuntimeError('A_ORIGINAL_FULL_PARTITION_SCOPE')
    flat=[s for r in parts for s in r['sessions']]
    source_ids=queue['source']['sessions']
    if len(flat)!=199 or len(set(flat))!=199 or len(source_ids)!=199 or set(flat)!=set(source_ids) or sum(r['native_frames'] for r in parts)!=29322 or queue['source']['native_frames']!=29322:
        raise RuntimeError('A_ORIGINAL22_UNIQUE_FULL_PARTITION')
    if len({r['task_id'] for r in parts})!=22:raise RuntimeError('A_FULL_PARTITION_DUPLICATE_PRODUCER')
    return {'full_partitions':len(parts),'production_partitions':len(records),'production_sessions':sum(len(r['sessions']) for r in records),'production_native_frames':sum(r['native_frames'] for r in records),'gate_partitions':len(gates),'gate_sessions':sum(len(r['sessions']) for r in gates),'gate_native_frames':sum(r['native_frames'] for r in gates),'all_unique_sessions':len(flat),'all_native_frames':sum(r['native_frames'] for r in parts)}
def repair_terminal_identity(ledger_status,outer_status,receipt_status):
    if ledger_status not in TERMINAL_STATUSES or outer_status not in TERMINAL_STATUSES or receipt_status not in TERMINAL_STATUSES:
        raise RuntimeError('A_REPAIR_TERMINAL_STATUS_NOT_FROZEN_LEGAL_SET')
    if ledger_status!=outer_status or outer_status!=receipt_status:
        raise RuntimeError('A_REPAIR_LEDGER_OUTER_RECEIPT_ACTUAL_STATUS_DRIFT')
    return {'ledger_actual_status':ledger_status,'outer_actual_status':outer_status,'terminal_actual_status':receipt_status,'legal_frozen_terminal_statuses':sorted(TERMINAL_STATUSES)}

def approved_idle(path):
    a=read(path)
    if a.get('schema')!='root_context_pad4_publication_approval_v1' or a.get('status')!=STATUS or a.get('operation')!=OPERATION:
        raise RuntimeError('A_PUBLICATION_EXPLICIT_ROOT_APPROVAL_PENDING')
    for x in a['exact_new_runtime_code']:exact(x)
    queue=read(exact(a['original_queue']))
    partition=full_partition(queue)
    records=queue['records']
    state=read(REPO_ROOT/'docs/governance/LONG_HORIZON_TASK_STATE.json')
    ledger={r['task_id']:r for r in state['tasks']}
    registry=read(REPO_ROOT/'tasks/current/INDEX.json')
    registered={r['task_id']:r for r in registry['task_packets']}
    if state.get('next_task') or any(r['status'] in ACTIVE for r in state['tasks']) or any(r.get('execution_allowed') for r in registry['task_packets']):
        raise RuntimeError('A_QUEUE_OR_REPAIR_STILL_ACTIVE')
    ids=[queue['source']['task_id']]+[r['task_id'] for r in queue['gates']]+[r['task_id'] for r in records]
    coverage=[]
    def require_dead(pid,ticks,label):
        if not isinstance(pid,int) or not isinstance(ticks,int) or ticks<=0:raise RuntimeError('A_FROZEN_ACTOR_IDENTITY_REQUIRED:'+label)
        try:actual=process_identity(pid)
        except (FileNotFoundError,ProcessLookupError,OSError):return
        if actual.get('start_ticks')==ticks:raise RuntimeError('A_IDENTICAL_ACTOR_STILL_LIVE:'+label)
    def check_task_actors(tid,outer):
        packet=read(exact(outer['task_packet']))
        writer=packet['writer'];require_dead(writer['pid'],writer['proc_start_ticks'],tid+':writer')
        workerpath=REPO_ROOT/'_run/current'/tid/'attempts/attempt_0001/WORKER.json'
        worker=read(workerpath)
        if worker.get('task_id')!=tid or worker.get('writer')!=writer or worker.get('run_signature')!=packet['run_signature'] or worker.get('packet_command')!=packet['commands'][0]:
            raise RuntimeError('A_WORKER_PACKET_IDENTITY:'+tid)
        require_dead(worker['pid'],worker['proc_start_ticks'],tid+':worker')
    for tid in ids:
        r=ledger.get(tid); rr=registered.get(tid)
        if not r or not rr or r.get('status')!='PASSED' or r.get('pid') is not None or r.get('proc_start_ticks') is not None or rr.get('execution_allowed') is not False:
            raise RuntimeError('A_ORIGINAL_GLOBAL_NOT_CLOSED:'+tid)
        outer=read(exact(r['result']));terminal=read(exact(r['terminal_receipt']))
        if outer.get('task_id')!=tid or outer.get('status')!='PASSED' or outer.get('worker_returncode')!=0 or outer.get('validation_errors')!=[] or terminal.get('status')!='PASSED' or terminal.get('result')!=r['result']:
            raise RuntimeError('A_ORIGINAL_OUTER_NOT_GLOBALLY_PASSED:'+tid)
        if rr.get('result')!=r['result'] or rr.get('terminal_receipt')!=r['terminal_receipt']:raise RuntimeError('A_REGISTRY_TERMINAL_DRIFT')
        ix=read(exact(outer['required_outputs']['candidate/INDEX.json']))
        if ix.get('status')!='COMPLETE' or ix.get('sessions_requested')!=ix.get('sessions_complete'):raise RuntimeError('A_ORIGINAL_INDEX_INCOMPLETE:'+tid)
        check_task_actors(tid,outer)
        coverage.append({'task_id':tid,'result':r['result'],'index':outer['required_outputs']['candidate/INDEX.json']})
    inventory=read(exact(a['release_inventory']))
    if inventory.get('schema')!='root_original_and_repair_actor_inventory_v1' or inventory.get('original_queue')!=a['original_queue'] or inventory.get('repair_queue_dispatch_complete') is not True:
        raise RuntimeError('A_REPAIR_COORDINATORS_NOT_EXPLICITLY_FINISHED')
    if len(inventory.get('actors',[]))<3 or not {'original_dispatcher','p5_024_028_waiter','p5_029_039_coordinator'}.issubset({x['role'] for x in inventory['actors']}):
        raise RuntimeError('A_ALL_ACTOR_INVENTORY_REQUIRED')
    for actor in inventory['actors']:
        pid=actor['pid'];ticks=actor['proc_start_ticks']
        if not isinstance(pid,int) or not isinstance(ticks,int):raise RuntimeError('A_ACTOR_IDENTITY_REQUIRED')
        try:actual=process_identity(pid)
        except (FileNotFoundError,ProcessLookupError,OSError):continue
        if actual.get('start_ticks')==ticks:raise RuntimeError('A_IDENTICAL_ACTOR_STILL_LIVE:'+actor['role'])
    repairs=inventory.get('repair_task_ids',[])
    if len(repairs)!=4 or len(set(repairs))!=4 or inventory.get('repaired_session_ids')!=['024','028','029','039']:
        raise RuntimeError('A_FOUR_BOUNDED_REPAIR_QUEUE_INVENTORY_REQUIRED')
    for tid in repairs:
        r=ledger.get(tid);rr=registered.get(tid)
        if not r or not rr or r.get('status') not in TERMINAL_STATUSES or r.get('pid') is not None or r.get('proc_start_ticks') is not None or rr.get('execution_allowed') is not False:
            raise RuntimeError('A_REPAIR_NOT_TERMINALLY_CLOSED:'+tid)
        outer=read(exact(r['result']));terminal=read(exact(r['terminal_receipt']))
        if outer.get('task_id')!=tid or terminal.get('task_id')!=tid or terminal.get('result')!=r['result'] or rr.get('result')!=r['result'] or rr.get('terminal_receipt')!=r['terminal_receipt']:
            raise RuntimeError('A_REPAIR_TERMINAL_CLOSURE')
        repair_terminal_identity(r['status'],outer.get('status'),terminal.get('status'))
        check_task_actors(tid,outer)
    lease=read(REPO_ROOT/'_run/current/GPU_LEASE.json')
    if lease.get('status')!='RELEASED':raise RuntimeError('A_GPU_LEASE_NOT_EXPLICITLY_RELEASED')
    return {'status':'ACTUAL_APPROVED_RELEASE_BOUNDARY_IDLE','original_globally_passed_tasks':len(ids),'original_sessions':199,'original_native_frames':29322,'repair_terminal_tasks':len(repairs),'actor_identities_checked':len(inventory['actors']),'actual_full_partition':partition,'original_global_refs':coverage,'GPU_started':False}
