"""Finite, close-only publisher. No model execution or source-data writes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, atomic_write, load_json, now_iso, publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.four_stream_completion import ATTEMPT, TASK, ACTIVE, ticks

OUT = ATTEMPT / 'final_closeout_20260928'
INDEX = REPO_ROOT / 'tasks/current/INDEX.json'
S2 = REPO_ROOT / '_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001'
SENSOR = REPO_ROOT / '_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/lanes/sensor/review_metric_v2'
SESSIONS = [
    ('scene', 'get_potato_chips_0902_103', 284, S2 / 'lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/robot.mp4'),
    ('scene', 'play_cards_0902_042', 171, S2 / 'lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/robot.mp4'),
    ('sensor', 'play_cards_0916_097', 165, SENSOR / 'play_cards_0916_097/SENSOR_METRIC_REVIEW_V2.mp4'),
    ('sensor', 'play_cards_0916_098', 179, SENSOR / 'play_cards_0916_098/SENSOR_METRIC_REVIEW_V2.mp4'),
    ('sensor', 'play_cards_0916_101', 122, SENSOR / 'play_cards_0916_101/SENSOR_METRIC_REVIEW_V2.mp4'),
    ('motion', 'get_potato_chips_0915_007', 378, S2 / 'lanes/motion_product/formal_product_007/attempt_0002/robot.mp4'),
    ('motion', 'play_cards_0915_031', 149, S2 / 'lanes/motion_product/formal_product_031/attempt_0004/robot.mp4'),
]


def close_state(state, refs, created):
    """Only terminalize our parent/three children; preserve HuRo and other owners."""
    updated = deepcopy(state)
    wanted = {TASK, *(TASK + '_' + x for x in ('scene', 'sensor', 'motion'))}
    if any(sum(r['task_id'] == tid for r in state['tasks']) != 1 for tid in wanted):
        raise RuntimeError('CLOSE_TASK_SET_INCOMPLETE')
    for row in updated['tasks']:
        tid = row['task_id']
        if tid not in wanted:
            continue
        if row.get('status') not in ACTIVE:
            raise RuntimeError('ALREADY_TERMINAL_OR_UNEXPECTED:' + tid)
        status = 'CANCELLED' if tid in (TASK, TASK + '_sensor') else 'REJECTED_QUALITY'
        row.update(status=status, phase='USER_CLOSED_WITH_UNMET_QUALITY',
                   phase_detail='NO_AUTOMATIC_RESUME_NEW_TASK_REQUIRED',
                   updated_at=created, heartbeat_at=None, pid=None, proc_start_ticks=None,
                   result=refs[tid], adoption='NOT_ADOPTED', review='NOT_REVIEWED',
                   quality='UNMET_OR_UNVERIFIED', last_attempt_terminal=status,
                   last_attempt_reason='USER_ENDED_REMAINING_QUALITY_RESEARCH')
    next_task = updated.get('next_task')
    if isinstance(next_task, dict) and next_task.get('task_id') in wanted:
        updated['next_task'] = None
    elif isinstance(next_task, str) and next_task in wanted:
        updated['next_task'] = None
    before = [r for r in state['tasks'] if r['task_id'] not in wanted]
    after = [r for r in updated['tasks'] if r['task_id'] not in wanted]
    if before != after:
        raise RuntimeError('OTHER_OWNER_CHANGED')
    return updated


def closed_index(index):
    result = deepcopy(index)
    ids = {TASK, *(TASK + '_' + x for x in ('scene', 'sensor', 'motion', 'huro'))}
    result['task_packets'] = [p for p in result['task_packets'] if p['task_id'] not in ids]
    # Physical packets remain immutable; only remove terminal work from routing.
    return result


def run(args, timeout=600):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)


def media(source, name, expected=None, kind='HISTORICAL_NOT_ADOPTED'):
    source = source.resolve(strict=True)
    if not source.is_relative_to(REPO_ROOT) or source.is_symlink():
        raise RuntimeError('MEDIA_PATH_OUTSIDE_PROJECT')
    ref = artifact_ref(source)
    probe = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                           '-count_frames', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
                           '-of', 'json', str(source)]).stdout)
    stream = probe['streams'][0]
    frames = int(stream['nb_read_frames'])
    if expected is not None and frames != expected:
        raise RuntimeError(f'FRAME_COUNT_MISMATCH:{source}:{frames}:{expected}')
    run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2', '-i', str(source),
         '-map', '0:v:0', '-f', 'null', '-'])
    if artifact_ref(source) != ref:
        raise RuntimeError('MEDIA_CHANGED_DURING_READ')
    dest = OUT / 'review' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and artifact_ref(dest)['sha256'] != ref['sha256']:
        raise RuntimeError('REVIEW_DEST_CONFLICT')
    if not dest.exists():
        shutil.copyfile(source, dest)
    copied = artifact_ref(dest)
    if (copied['sha256'], copied['bytes']) != (ref['sha256'], ref['bytes']):
        raise RuntimeError('COPY_MISMATCH')
    print(json.dumps({'decoded': name, 'frames': frames}, ensure_ascii=False), flush=True)
    return dict(source=ref, delivered=copied, kind=kind, frames=frames,
                full_decode='PASS', probe=stream, new_algorithm_execution=False,
                quality_adopted=False, human_review='NOT_REVIEWED')


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'RESULT.json').exists():
        raise RuntimeError('FINAL_RESULT_ALREADY_SEALED')
    records = []
    for lane, session, frames, path in SESSIONS:
        item = dict(lane=lane, session_id=session, quality='NOT_ADOPTED',
                    latest_full_pipeline='CANCELLED_NOT_IMPLEMENTED',
                    final_quality_video='CANCELLED_NO_QUALIFIED_RESULT',
                    new_stage_videos='CANCELLED_NOT_IMPLEMENTED',
                    human_review='NOT_REVIEWED')
        item['existing_media'] = media(path, session + '_历史未采用.mp4', frames)
        item['explanation'] = ('历史显示修正版，仅诊断，不证明腕骨架真实贴合。' if lane == 'sensor'
                               else '历史原场景候选，质量拒绝；不含本轮修复，不是新基线。')
        records.append(item)
    extras = []
    for folder, name in [('C01_REVIEW_ZH', 'C01_数值修复_16帧诊断'),
                         ('C02_REVIEW_ZH', 'C02_候选1_未采用'),
                         ('C02_ROLE_FIX2_REVIEW_ZH', 'C02_候选2_未采用')]:
        matches = list((ATTEMPT / 'lanes/scene' / folder).glob('*.mp4'))
        if len(matches) != 1:
            raise RuntimeError('EXPECTED_SINGLE_WINDOW_VIDEO:' + folder)
        extras.append(media(matches[0], name + '.mp4', 16, 'FIXED_4FPS_WINDOW_DIAGNOSTIC'))
    progress = load_json(ATTEMPT / 'checkpoints/PROGRESS_0008.json')
    evidence = []
    def collect(value):
        if isinstance(value, dict):
            if {'path', 'bytes', 'sha256'} <= value.keys():
                path = Path(value['path'])
                if not path.is_absolute():
                    path = REPO_ROOT / path
                actual = artifact_ref(path)
                if (actual['bytes'], actual['sha256']) != (value['bytes'], value['sha256']):
                    raise RuntimeError('CHECKPOINT_REF_INVALID:' + str(value['path']))
                evidence.append(actual)
            else:
                for v in value.values(): collect(v)
        elif isinstance(value, list):
            for v in value: collect(v)
    collect(progress)
    evidence.append(artifact_ref(ATTEMPT / 'checkpoints/PROGRESS_0008.json'))
    semantic = ATTEMPT / 'lanes/sensor/S01_SEMANTIC_OVERLAY_V1'
    for path in sorted(semantic.glob('FRAME_*.png')):
        dest = OUT / 'review' / '097_语义诊断图片' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        evidence.append(artifact_ref(path))
    atomic_json(OUT / 'DELIVERY_INVENTORY.json', dict(schema_version='COMPLETION_CLOSEOUT_INVENTORY_V1',
        created_at=now_iso(), sessions=records, diagnostic_windows=extras,
        frozen_evidence=evidence, quality_adopted_products=0,
        huro='TERMINAL_REJECTED_QUALITY_NO_NEW_COMPARISON',
        cancellation_authority='User approved final closeout and cancellation of unresolved quality research.'))
    lines = ['# 先看这里：本轮结束交付', '',
        '本轮已按用户要求结束未解决的质量研究。这里交付已有成果，不是质量通过的新算法基线。', '',
        '## 观看顺序与真实结论', '',
        '1. 先看 C01 数值修复短窗：工程数值适配已修复，不代表 Clean 语义质量通过。',
        '2. C02 两条短窗显示已有失败候选；候选3的最终拒绝与方法分析见证据清单，不能把候选2视频称为候选3。',
        '3. 七条会话视频全部为已存在的历史候选/诊断。每条已完整解码，未重新求解。',
        '4. 097 图片区分手柄、腕与 MANUS 根的语义；不证明实际物理贴合。', '',
        '## 每会话交付', '', '| 会话 | 可观看文件 | 结论 |', '|---|---|---|']
    for item in records:
        name = Path(item['existing_media']['delivered']['path']).name
        lines.append(f"| {item['session_id']} | [历史视频](review/{name}) | {item['explanation']} |")
    lines += ['', '## 已取消／未实现', '',
        '上述七个会话的新质量合格全片、新阶段视频与新同步对照均未完成，本轮取消；旧视频不补数。',
        '支线1：C01 工程修复保留；C02 三候选耗尽，残留/伪影未解决，不采用。',
        '支线2：097/098/101 保留坐标链诊断；缺独立物理同名点与安装证据，真实腕骨架贴合未达成。',
        '支线3：失效传播修复及007(378帧)/031(149帧)完整账本保留；当前证据不足导致全帧原图回退，不算Robot成功。',
        'HuRo：已有官方核心实验与失败分析保留，终止且不采用；不再求解或制作新对比。',
        'Depth、Object6D、Contact 与完整原场景产品质量均无新增采用。', '',
        '## 复现和证据', '',
        '`DELIVERY_INVENTORY.json` 绑定原始结果路径、bytes/SHA、帧数、完整解码与取消项。',
        '`RESULT.json` 记录本轮终态；旧检查点保持原样，不再作为活动指令。',
        '历史代码入口及冻结配置保留供查证；重新研究必须获得新授权并登记新任务，不能续跑本轮。',
        '人工视觉审阅：未完成；用户同意结束研究不等于接受视频质量。',
        '没有训练、修改原始/processed数据，未做物理安全认证。', '']
    atomic_write(OUT / '先看这里.md', '\n'.join(lines).encode('utf-8'))
    print(json.dumps({'prepared': str(OUT), 'videos': len(records) + len(extras)}), flush=True)


def publish(expected):
    if (OUT / 'RESULT.json').exists():
        raise RuntimeError('ALREADY_SEALED_CHECK_EXISTING_RESULT')
    receipt = load_json(RECEIPT_PATH)
    if receipt['governance_revision'] != expected:
        raise RuntimeError('CAS_MISMATCH')
    old_writer = load_json(ATTEMPT / 'WRITER.json')
    try:
        if ticks(old_writer['pid']) == old_writer['proc_start_ticks']:
            raise RuntimeError('PREDECESSOR_WRITER_STILL_ALIVE')
    except FileNotFoundError:
        pass
    state = load_json(TASK_STATE_PATH)
    ids = {TASK, *(TASK + '_' + x for x in ('scene', 'sensor', 'motion', 'huro'))}
    for row in state['tasks']:
        if row['task_id'] in ids and row.get('pid'):
            try:
                if ticks(row['pid']) == row.get('proc_start_ticks'):
                    raise RuntimeError('CAMPAIGN_PROCESS_STILL_ALIVE')
            except FileNotFoundError:
                pass
    inv = load_json(OUT / 'DELIVERY_INVENTORY.json')
    checks = load_json(OUT / 'ENGINEERING_CHECKS.json')
    if checks.get('exit_code') != 0:
        raise RuntimeError('ENGINEERING_CHECKS_FAILED')
    for item in inv['sessions']:
        for ref in (item['existing_media']['source'], item['existing_media']['delivered']):
            if validate_artifact_ref(ref): raise RuntimeError('MEDIA_REF_INVALID')
    for item in inv['diagnostic_windows']:
        if validate_artifact_ref(item['delivered']): raise RuntimeError('WINDOW_REF_INVALID')
    for ref in inv['frozen_evidence']:
        if validate_artifact_ref(ref): raise RuntimeError('EVIDENCE_REF_INVALID')
    created = now_iso()
    writer = dict(scope=TASK, capability='CLOSE_ONLY_NO_ALGORITHM', pid=os.getpid(),
                  proc_start_ticks=ticks(os.getpid()), executor_epoch=old_writer['executor_epoch'] + 1,
                  fencing_token=secrets.token_hex(24), predecessor=artifact_ref(ATTEMPT / 'WRITER.json'),
                  authority='User: implement final closeout; finish all tasks; stop unmet quality research.',
                  created_at=created)
    atomic_json(OUT / 'CLOSE_WRITER.json', writer)
    refs = {}
    for lane in ('scene', 'sensor', 'motion'):
        tid = TASK + '_' + lane
        value = dict(schema_version='COMPLETION_LANE_CLOSED_V1', task_id=tid,
                     status='CANCELLED' if lane == 'sensor' else 'REJECTED_QUALITY',
                     quality='UNMET_OR_UNVERIFIED', adoption='NOT_ADOPTED', human_review='NOT_REVIEWED',
                     next_action='NONE_NEW_AUTHORIZATION_AND_TASK_REQUIRED',
                     evidence=artifact_ref(ATTEMPT / 'checkpoints/PROGRESS_0008.json'),
                     delivery_inventory=artifact_ref(OUT / 'DELIVERY_INVENTORY.json'), created_at=created)
        path = OUT / (lane.upper() + '_TERMINAL.json')
        atomic_json(path, value)
        refs[tid] = artifact_ref(path)
    final = dict(schema_version='COMPLETION_FINAL_CLOSEOUT_V1', task_id=TASK,
                 status='CLOSED_WITH_UNMET_QUALITY', task_terminal_status='CANCELLED',
                 campaign_closed=True, project_quality_complete=False, quality_adopted_products=0,
                 human_review='NOT_REVIEWED', adoption='NOT_ADOPTED', new_algorithm_executions=0,
                 raw_or_processed_modified=False, training_executed=False, gpu_used=False,
                 automatic_resume=False, next_task=None, created_at=created,
                 governance_revision_before=expected, governance_revision_target=expected + 1,
                 predecessor=artifact_ref(ATTEMPT / 'checkpoints/PROGRESS_0008.json'),
                 user_scope_decision=writer['authority'], writer=artifact_ref(OUT / 'CLOSE_WRITER.json'),
                 lanes=refs.copy(), huro=artifact_ref(ATTEMPT / 'lanes/huro/TERMINAL_NOT_ADOPTED.json'),
                 inventory=artifact_ref(OUT / 'DELIVERY_INVENTORY.json'),
                 engineering=artifact_ref(OUT / 'ENGINEERING_CHECKS.json'),
                 git_push_and_windows_delivery='SEPARATE_POST_PUBLICATION_RECEIPT',
                 conclusion='本轮任务已结束；质量目标未达成并停止推进，已有成果按清单交付。')
    atomic_json(OUT / 'RESULT.json', final)
    refs[TASK] = artifact_ref(OUT / 'RESULT.json')
    updated = close_state(state, refs, created)
    updated['recent_events'] = (updated.get('recent_events', []) + [dict(
        task_id=TASK, status='CANCELLED', created_at=created, result=refs[TASK],
        message='User-approved final closeout, unmet quality, no automatic continuation.')])[-100:]
    for lane in ('scene', 'sensor', 'motion'):
        p = ATTEMPT / 'lanes' / lane / 'STATE.json'
        old = load_json(p)
        atomic_json(OUT / (lane.upper() + '_PREDECESSOR_STATE.json'), old)
        old.update(status='TERMINAL_USER_CLOSED', next_action='NONE_NEW_TASK_REQUIRED',
                   quality='UNMET_OR_UNVERIFIED', adoption='NOT_ADOPTED', review='NOT_REVIEWED',
                   result=refs[TASK + '_' + lane], updated_at=created)
        atomic_json(p, old)
    publication = publish_bundle(load_json(AUTHORITY_PATH), updated,
        event_type='FOUR_STREAM_COMPLETION_USER_FINAL_CLOSEOUT', expected_revision=expected,
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=closed_index(load_json(INDEX)))
    atomic_json(OUT / 'PUBLICATION.json', dict(revision=publication['governance_revision'],
        result=refs[TASK], close_writer_released=True, no_algorithm_started=True, created_at=now_iso()))
    print(json.dumps({'closed': True, 'revision': publication['governance_revision'],
                      'result': refs[TASK]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'publish'])
    parser.add_argument('--expected-revision', type=int)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT / 'close.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare': prepare()
        else: publish(args.expected_revision)


if __name__ == '__main__':
    main()
