"""Serial publisher: select user-reviewed 0916 baseline and retire stale navigation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
from datetime import datetime,timedelta
from chaoyang.governance.common import REPO_ROOT,RECEIPT_PATH,TASK_STATE_PATH,AUTHORITY_PATH,artifact_ref,atomic_json,canonical_bytes,load_json,now_iso,publish_bundle
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK='robot100_v16_selection_navigation_20260929'
ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
SOURCE=REPO_ROOT/'_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'
PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'

def main():
    state=copy.deepcopy(load_json(TASK_STATE_PATH)); index=copy.deepcopy(load_json(INDEX))
    if state.get('next_task') or any(x.get('execution_allowed') for x in index['task_packets']): raise RuntimeError('OTHER_PUBLISHER_ACTIVE')
    if any(x.get('status') in ('RUNNING','PENDING','CLAIMED','WAIT_GPU_RESOURCE') for x in state['tasks']): raise RuntimeError('ACTIVE_TASK')
    if ROOT.exists() or PACKET.exists(): raise RuntimeError('FRESH_TASK_REQUIRED')
    revision=int(load_json(RECEIPT_PATH)['governance_revision'])
    selected=load_json(SOURCE/'BASELINE.json'); assert selected['baseline_selected'] is True
    result=load_json(SOURCE/'RESULT.json'); assert result['status']=='COMPLETE_DIAGNOSTIC_NOT_ACCURACY_CERTIFIED'
    pid=os.getpid(); ticks=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
    writer={'pid':pid,'proc_start_ticks':ticks,'executor_epoch':1,'fencing_token':secrets.token_hex(24),'scope':TASK,'created_at':now_iso()}
    doc=REPO_ROOT/'docs/current/ROBOT100_BATCH_ZH.md'; entry=REPO_ROOT/'docs/current/AI_WORK_ENTRY_ZH.md'
    pointer=REPO_ROOT/'docs/current/ROBOT100_HARDWARE_BASELINE_V16.json'
    previous={p:artifact_ref(p) for p in (doc,entry)}
    signature=hashlib.sha256(canonical_bytes({'baseline':artifact_ref(SOURCE/'BASELINE.json'),'docs':list(previous.values()),'publisher':artifact_ref(Path(__file__))})).hexdigest()
    packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK,'objective':'Select user-reviewed 0916 V16, replace obsolete current instructions, preserve 0902 and sealed dependencies','phase':'DOCUMENT_BASELINE_PUBLICATION','plan_revision':'USER_V16_ACCEPTANCE','execution_revision':TASK.upper(),'deadline_at':(datetime.now().astimezone()+timedelta(minutes=10)).isoformat(timespec='seconds'),'read_set':['AGENTS.md',str(SOURCE/'BASELINE.json'),str(SOURCE/'RESULT.json'),str(doc),str(entry)],'write_set':[str(ROOT),str(doc),str(entry),str(pointer)],'prerequisites':['governance_PASS','no_other_publisher','user_selected_V16'],'required_outputs':['RESULT.json','current branch navigation','selected baseline pointer'],'budgets':{'cpu_threads_total':1,'workers_max':1,'gpu_count':0,'training_steps':0,'session_attempts_max':1,'wall_seconds_max':600},'stop_conditions':['writer_fence_drift','publication_complete'],'claim_limit':'Visual development baseline only; overlay has visible mismatch, physical accuracy unverified. No deletion of implementation or sealed data.','writer':writer,'run_signature':signature,'commands':['publisher_document_finalize_only']}
    errors=_validate_packet(packet)
    if errors: raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True); PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET,packet)
    state['tasks'].append({'task_id':TASK,'phase':packet['phase'],'attempt':1,'status':'RUNNING','session':'play_cards_0916_025','updated_at':now_iso(),'heartbeat_at':now_iso(),'pid':pid,'proc_start_ticks':ticks,'gpu_id':None,'task_packet':artifact_ref(PACKET),'run_signature':signature,'quality':'NOT_APPLICABLE_DOCUMENT_PUBLICATION','adoption':'VISUAL_DEVELOPMENT_BASELINE','review':'USER_ACCEPTED_V16_VIDEO'})
    state['next_task']={'task_id':TASK,'session':'play_cards_0916_025','prerequisites':packet['prerequisites'],'expected_resource':'ONE_CPU_ZERO_GPU','stop_condition':'document publication complete'}
    index['task_packets'].append({'task_id':TASK,'packet_path':str(PACKET.relative_to(REPO_ROOT)),'packet_sha256':artifact_ref(PACKET)['sha256'],'execution_class':'CURRENT_LEDGER_ROUTABLE','execution_allowed':True})
    def publish(event):
        nonlocal revision
        index['packet_revision']=event;index['status']='PASS' if state.get('next_task') else 'PASS_NO_ACTIVE_TASKS'
        value=publish_bundle(load_json(AUTHORITY_PATH),state,event_type=event,expected_revision=revision,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
        revision=int(value['governance_revision'])
    publish('ROBOT100_V16_SELECTION_REGISTERED')
    for path,ref in previous.items():
        assert artifact_ref(path)==ref,'DOCUMENT_DRIFT'
        (ROOT/('BEFORE_'+path.name)).write_bytes(path.read_bytes())
    assert int(load_json(RECEIPT_PATH)['governance_revision'])==revision,'FENCE_DRIFT'
    atomic_json(pointer,{'schema_version':'CHAOYANG_BRANCH_BASELINE_POINTER_V1','baseline_id':selected['baseline_id'],'baseline_selected':True,'selection_scope':selected['selection_scope'],'baseline':artifact_ref(SOURCE/'BASELINE.json'),'reviewed_video':selected['reviewed_video'],'diagnostic_result':artifact_ref(SOURCE/'RESULT.json'),'image_alignment_status':'VISIBLE_MISMATCH_NOT_ACCEPTED','batch_100_validated':False,'physical_accuracy_verified':False,'excluded_branch':'0902 Exact78/Inpainting'})
    document='''# 0916硬件分支：唯一当前基线V16

用户于2026-09-29确认V16当前视频合理，选择该链路作为本分支后续开发基线。这个选择只覆盖离线运动展示，不证明原图贴合、物理标定、碰撞安全或100条批量质量通过。

## 唯一入口

- [机器基线指针](ROBOT100_HARDWARE_BASELINE_V16.json)
- [冻结链路、输入、代码与用户选择](../../_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/BASELINE.json)
- [用户确认的视频及V16运动](../../_run/current/robot100_constraint_review_v16_retry1_20260929/attempts/attempt_0001/RESULT.json)
- [原图腕骨架叠加诊断](../../_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/RESULT.json)

## 链路和用途

同会话原始PICO/MANUS HDF5与同步原片 → HDF5保存的controller-to-wrist安装关系 → 解剖左右手恢复、世界首帧1:1平移目标 → MANUS到KaiHand手部映射 → V16显式内收约束 → 固定视角Robot、连接件与双手近景。原片是同步观察参考，不是腕轨迹的视觉估计输入；这不是去手补背景成片。

已核验025会话176帧：左右各176/176通过既定20毫米/15度/10毫米内收门，原始q、目标和FK可追溯。手指映射未因本次确认获得新的物理精度认证。其他会话须从这个冻结链路建立新任务验证，不继承025的通过数。

## 准确度检查发现

本次叠加使用同会话camera_params与clip_manifest：旧名left槽实际对应物理right相机，使用登记的编码域K和外参，不套用已拒绝的HDF5旧video_cam，不新增畸变、不拟合画面。样图中腕和骨架明显错位，**原图贴合尚未通过**。相机编码域/外参适用性与腕安装关系须独立验证；无独立二维参考点，不给出虚假的像素准确率，不能断言优于视觉基线。

## 旧入口处置与边界

V8.1、V9、V10、V11、V12、V13、V14、V15不再是可选基线。本页已删除它们的运行命令及“当前待确认”入口，只保留一个V16指针。旧attempt中的CURRENT、README和终态结果都是历史上下文，不得按mtime或名称重新采用。

V16仍导入部分旧版本命名的实现，且输入绑定V15冻结目标；这些不是多套当前基线，不能删除。保留范围见[依赖保留清单](../../_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/DEPENDENCY_KEEP_SET.json)。旧导航正文在本次发布attempt保存，用于恢复，不在默认执行入口展示。

0902 Exact78/Inpainting是另一位AI的独立分支，不读取或复用其数据、安装数值、基线选择和任务状态。后续准确度修正使用新任务、新版本，禁止覆盖已选择的V16。
'''
    doc.write_text(document,encoding='utf8')
    original=entry.read_text(encoding='utf8')
    prefix='''# 先按分支路由

- **0916 PICO＋MANUS硬件Robot：**只从[0916当前V16基线](ROBOT100_BATCH_ZH.md)和[机器指针](ROBOT100_HARDWARE_BASELINE_V16.json)开始。用户已选择V16作为运动展示开发基线；原图投影准确度尚未通过。不要把下方0902任务当作0916入口。
- **0902 Exact78/Inpainting：**继续下方已有入口；本次没有改变该分支基线或任务。

---

'''
    assert '# 先按分支路由' not in original,'ALREADY_ROUTED'
    entry.write_text(prefix+original,encoding='utf8')
    final={'status':'COMPLETE','baseline_pointer':artifact_ref(pointer),'current_branch_doc':artifact_ref(doc),'routing_entry':artifact_ref(entry),'baseline_selected':True,'user_review':'ACCEPTED_OFFLINE_VISUAL_DEVELOPMENT_V16','image_alignment':'VISIBLE_MISMATCH_NOT_ACCEPTED','stale_current_instructions_removed':['V9 direct module command','V14 pending current claim','V15 latest default','old V8.1 CURRENT pointer as authority'],'physical_files_deleted':[],'reason_no_dependency_deletion':'V16 uses predecessor code and frozen inputs; source data, sealed results and 0902 untouched','before':list(previous.values())}
    atomic_json(ROOT/'RESULT.json',final)
    state['tasks'][-1].update(status='PASSED',pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(ROOT/'RESULT.json'))
    state['next_task']=None
    for row in index['task_packets']:
        if row['task_id']==TASK: row['execution_allowed']=False
    publish('ROBOT100_V16_USER_SELECTED_NAVIGATION_FINAL')
    atomic_json(ROOT/'TERMINAL.json',{'status':'PASSED','revision':revision})
    print('FINALIZED',revision,flush=True)

if __name__=='__main__': main()
