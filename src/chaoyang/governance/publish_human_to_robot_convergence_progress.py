#!/usr/bin/env python3
"""Publish the first evidence-backed convergence wave and current navigation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from chaoyang.governance.common import (AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
                                        artifact_ref, atomic_json, load_json, now_iso, publish_bundle)

TASK="human_to_robot_baseline_v1_convergence_20260923";ROOT=REPO_ROOT/f"_run/current/{TASK}/attempts/attempt_0001"
PLAN=REPO_ROOT/"docs/current/PLAN.md";ENTRY=REPO_ROOT/"docs/current/AI_WORK_ENTRY_ZH.md";DOC=REPO_ROOT/"docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
VIS=REPO_ROOT/"docs/current/visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md"
E={"motion":ROOT/"lanes/motion_product/partial_direction_031/wave0/RESULT.json",
   "motion_review":ROOT/"lanes/motion_product/partial_direction_031/review_v1/RESULT.json",
   "cable":ROOT/"lanes/scene/cable_007/wave0/RESULT.json","contact":ROOT/"lanes/scene/contact_screen_031/wave0/RESULT.json",
   "sensor":ROOT/"lanes/sensor/review_v1/RESULT.json","compare":ROOT/"lanes/compare/numeric_v1/RESULT.json"}
def lane(name,execution,structure,quality,improvement,adoption,evidence,blocker):
 return {"schema_version":"HUMAN_TO_ROBOT_CONVERGENCE_LANE_STATE_V1","task_id":TASK,"lane":name,"status":"READY",
         "execution":execution,"structure":structure,"quality":quality,"improvement":improvement,"adoption":adoption,
         "evidence":[artifact_ref(p) for p in evidence],"blocker":blocker,"updated_at":now_iso(),
         "writer":{"pid":None,"proc_start_ticks":None,"executor_epoch":9},"training_eligible":False,
         "control_ground_truth":False,"physical_deployable":False,"external_metric_authority":False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--expected-revision',type=int,required=True);a=ap.parse_args()
 if int(load_json(RECEIPT_PATH)['governance_revision'])!=a.expected_revision:raise RuntimeError('GOVERNANCE_CAS_MISMATCH')
 state=load_json(TASK_STATE_PATH)
 if state.get('next_task',{}).get('task_id')!=TASK:raise RuntimeError('TASK_NOT_CURRENT')
 for p in E.values():
  if not p.is_file():raise FileNotFoundError(p)
 m,c,s,cmp=load_json(E['motion']),load_json(E['contact']),load_json(E['sensor']),load_json(E['compare'])
 states={
 'motion_product':lane('motion_product','EXECUTED','PASS',m['quality'],'HARD_POSITION_16_OF_16_PALM_NORMAL_PASS_LONGITUDINAL_FAIL_NO_FULL_EXPANSION','NOT_ADOPTED',[E['motion'],E['motion_review']],
  {"missing":"031 supported longitudinal direction <=15deg and any independent left-hand model output","consumer":"partial-direction full expansion and complete product","owner":"motion_product","unblock_action":"do not retune rejected candidate; audit independent RGB/ROI left evidence separately","affected":["031 motion quality","new 031 product"],"unaffected":["old R0","007","Sensor","Contact diagnostics"]}),
 'scene':lane('scene','EXECUTED','PASS','INCONCLUSIVE_MIXED','CABLE_TOP1_16_OF_16_AND_CONTACT_GEOMETRY_SCREEN_158_OF_510','NOT_ADOPTED',[E['cable'],E['contact']],
  {"missing":"independent human/device association for 007 cable removal and strict observed hand authority for Contact","consumer":"ProPainter cable invocation and Robot R1","owner":"scene","unblock_action":"obtain independent role support; keep current ProPainter and R1 unexecuted","affected":["007 Clean improvement","strict Contact","Robot R1"],"unaffected":["031 sampled proximity","existing Clean candidates","R0"]}),
 'sensor':lane('sensor','EXECUTED_REUSED_ARRAYS_NEW_REVIEW','PASS','PENDING_USER_VISUAL_REVIEW','466_FRAME_COMPLETE_REVIEW_WITH_CLIPPED_SEGMENTS_WORLD_ORIENTATION_LOCAL_AND_SAVED_FK','CANDIDATE_ONLY',[E['sensor']],
  {"missing":"user visual acceptance and independent external wrist truth","consumer":"Sensor visual adoption","owner":"sensor","unblock_action":"review three new full-session videos; do not refit 101 or call 098 labelled","affected":["visual adoption"],"unaffected":["native arrays","kinematic-only backend"]}),
 'compare':lane('compare','EXECUTED_REUSED_ARRAYS','PASS',cmp['quality'],'COMMON_OLD_CONTRACT_NUMERIC_COMPARISON_FROZEN_NO_WINNER','NOT_ADOPTED',[E['compare']],
  {"missing":"independent truth and accepted same-session Clean","consumer":"winner and final visual method claim","owner":"compare","unblock_action":"retain finite old-contract result; do not mix rejected partial-direction local candidate with old HuRo contract","affected":["winner claim"],"unaffected":["bounded numeric differences","old review videos"]})}
 for name,value in states.items():atomic_json(ROOT/f"lanes/{name}/STATE.json",value)
 progress={"schema_version":"HUMAN_TO_ROBOT_CONVERGENCE_PROGRESS_V1","task_id":TASK,"created_at":now_iso(),"checkpoint":"WAVE0_BEFORE_H3",
           "summary":{"031_direction_qualified":"102/102_MODEL_DERIVED","031_ik_window_position_pass":"16/16","031_ik_window_supported_direction_pass":"0/16","031_full_expansion":False,
                      "007_cable_top1_unique":"16/16_AI_PROXY","007_propainter_invoked":False,
                      "contact_qualification_checked":c['counts']['qualification_checked'],"contact_geometry_screened":c['counts']['geometry_screened'],
                      "contact_strict_admissible":0,"robot_r1_executed":0,"sensor_new_reviews":"3/3_466_FRAMES","local_huro_winner":None},
           "evidence":{k:artifact_ref(v) for k,v in E.items()},"lane_states":{k:artifact_ref(ROOT/f"lanes/{k}/STATE.json") for k in states},
           "authority":{"training_eligible":False,"control_ground_truth":False,"physical_deployable":False,"external_metric_authority":False}}
 progress_path=ROOT/"PROGRESS_WAVE0.json"
 if not progress_path.is_file():atomic_json(progress_path,progress)
 now=now_iso();row=next(r for r in state['tasks'] if r.get('task_id')==TASK);row.update(status='PENDING',attempt=1,updated_at=now,heartbeat_at=None,pid=None,proc_start_ticks=None)
 state['recent_events']=(state.get('recent_events',[])+[{"task_id":TASK,"attempt":1,"status":"PENDING","created_at":now,"message":"Wave0 executed: 031 IK rejected without expansion; cable/contact/Sensor/compare advanced independently; remaining finite work queued."}])[-100:]
 PLAN.write_text("# 当前计划：Human→Robot Baseline v1 四支线收敛执行中\n\n"
  f"当前唯一任务：`{TASK}`，12小时截止时间见任务状态。S2保持封存。\n\n"
  "首波已执行：031方向资格102/102；固定窗位置16/16通过，但支持方向0/16，因此没有扩片；007线缆16/16形成局部top-1代理，但独立角色支持缺失，未调用ProPainter；031 Contact完成510条资格检查和158条实际表面采样，严格Contact/R1仍为0；Sensor三条466帧新回放已生成；Local/HuRo仅冻结旧共同合同数值结论。\n\n"
  "继续项：独立左手证据、007角色支持、031同表面遮挡、连接件碰撞语义、四片产品与15槽位收敛。局部失败不阻塞其他READY项。\n\n"
  f"机器进度：[{ROOT/'PROGRESS_WAVE0.json'}]({ROOT/'PROGRESS_WAVE0.json'})；[视频索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。\n",encoding='utf-8')
 ENTRY.write_text("# 当前执行入口：Human→Robot Baseline v1 收敛任务\n\n"
  f"先读 [当前计划](PLAN.md)、[Wave0机器进度]({ROOT/'PROGRESS_WAVE0.json'}) 与 [视频索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。\n\n"
  "不要重试已拒绝的031数学候选，不降低20mm/15°门；不要把007局部颜色形状代理直接变成删除权限；不要把158条模型条件几何采样写成严格Contact；Sensor回放只待视觉审阅，不重复求解。\n",encoding='utf-8')
 VIS.parent.mkdir(parents=True,exist_ok=True)
 sensor_rows=s['sessions'];VIS.write_text("# Human→Robot Baseline v1 收敛任务视频索引\n\n> 仅导航到项目内唯一实体；视觉通过需用户审阅。\n\n"
  f"- 031 IK固定窗（NEW，REJECTED）：[{artifact_ref(Path(load_json(E['motion_review'])['video']['path']))['path']}]({load_json(E['motion_review'])['video']['path']})\n"
  f"- 007线缆top-1证据（NEW，INCONCLUSIVE）：[{load_json(E['cable'])['review_video']['path']}]({load_json(E['cable'])['review_video']['path']})\n"
  +"\n".join(f"- Sensor {r['session_id']}（NEW，PENDING_REVIEW）：[{r['video']['path']}]({r['video']['path']})" for r in sensor_rows)+"\n\n"
  "旧四产品和Local/HuRo视频保持原实体，通过本任务最终15槽位索引引用；未产生的新产品不得以旧文件改名。\n",encoding='utf-8')
 marker='## 当前收敛任务（2026-09-23）'
 text=DOC.read_text(encoding='utf-8')
 if marker not in text:
  first=text.find('\n');insert=f"\n{marker}\n\n已登记 `{TASK}`。首波真实结果：031硬位置16/16但支持方向0/16，未扩片；Contact几何筛选158/510但严格资格0；007线缆局部top-1为16/16但角色支持仍不足；Sensor三片466帧新回放待审。详见[当前视频索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。\n\n"
  DOC.write_text(text[:first+1]+insert+text[first+1:],encoding='utf-8')
 pub=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='HUMAN_TO_ROBOT_CONVERGENCE_WAVE0_PUBLISHED',expected_revision=a.expected_revision,generator_path=Path(__file__))
 print(json.dumps({"status":"PASS","revision":pub['governance_revision'],"checkpoint":str(ROOT/'PROGRESS_WAVE0.json')},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
