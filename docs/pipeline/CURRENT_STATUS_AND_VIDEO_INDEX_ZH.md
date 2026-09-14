# 当前状态与视频索引

更新时间：2026-09-11 15:21（Asia/Shanghai）

## 当前正式主线：0909 `rgb30_v1`

当前唯一状态入口是 [0909 `rgb30_v1` 当前状态](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS_ZH.md)，机读版本为 [`CURRENT_STATUS.json`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS.json)。

| 阶段 | 当前状态 | 可看/可核验入口 |
|---|---|---|
| RAW | `001..104` 共104个连续目录，源只读 | `/mnt/data/egodata/datasets/ego/chips_cards_handle_0909` |
| QA / RGB30 / tracker | current-code final-contract canary已提交；正式104条双worker运行中 | [机读快照](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS.json)、[控制STATE](../../tasks/control/runs/20260911_handle0909_rgb30_v1/STATE.json) |
| 010 | resumed canary 404帧，`PASS / ELIGIBLE / PUBLISHED` | [QA全片视频](/mnt/data/egodata/datasets/ego/processed_archive/chips_cards_handle_0909_debug_20260911/.canary_chips_cards_handle_0909_rgb30_v1_contract_final_resumed/cleaned/potato_chips/get_potato_chips_0909_010/qa/qa_review_full.mp4)、[发布RGB](/mnt/data/egodata/datasets/ego/processed_archive/chips_cards_handle_0909_debug_20260911/.canary_chips_cards_handle_0909_rgb30_v1_contract_final_resumed/cleaned/potato_chips/get_potato_chips_0909_010/CameraRecord_get_potato_chips_0909_010.mp4) |
| HaWoR / Mask / Clean | 尚未在新 release 批次运行 | 无 current 0909 视频 |
| Stereo Depth / Object6D | 尚无0909 metric authority，未运行 | [深度证据边界](../../DEPTH_ACCURACY_PACKAGE_20260911/README_ZH.md) |
| Robot | 无0909 authority、无action sidecar | 下方六会话视频仅作开发参考 |
| HumanEgo | 0909 未训练 | 无checkpoint |

final-contract canary分流为：001策略排除且有长缺口/补值超限；038共同支持率低于99%并有长缺口/补值超限；104缺VST QPC sidecar。它已生成总`DATASET_RESULT.json`并绑定current代码。正式104条现已fresh恢复；正式完成仍只认正式根的`DATASET_RESULT.json.state=COMMITTED`。

旧 exact78 的 Mask/Clean、Robot 和 HumanEgo 已按用户指令暂停；本页下方两任务78与六会话内容是冻结的历史/开发证据，不具有0909执行权。

## 2026-09-11 六会话完整任务复核（开发参考，非0909 current）

当前人工视频入口是 [`六会话完整任务四阶段视频索引 V2`](../../tasks/control/runs/20260911_six_session_full_pipeline_video_review_v2/INDEX_ZH.md)，远端媒体根为 `tasks/control/runs/20260911_six_session_full_pipeline_video_review_v2/fullsession_review_media_v2/`。六条均保持原始 30 fps、完整 `frame_id=0..N-1`，没有 loop、hold-last、pad、重复或时长裁切；旧 `20260911_six_session_pipeline_video_review_v1` 只允许标记为 `SUPERSEDED_REVIEW_ONLY` 的 12 秒循环预览。

| 任务 | 会话 | 完整帧数 / 时长 | 当前复核状态 |
|---|---|---:|---|
| Chips | `get_potato_chips_0903_050` | 302 / 10.07 s | HaWoR V3；Robot arm/hand 数值 PASS，待视觉确认 |
| Chips | `get_potato_chips_0903_140` | 415 / 13.83 s | HaWoR V3；Robot arm/hand 数值 PASS，待视觉确认 |
| Chips | `get_potato_chips_0903_182` | 351 / 11.70 s | Robot arm PASS；右小指 joint lower-bound，hand 115 rows HOLD，视频已水印 |
| Poker | `play_cards_0903_203` | 151 / 5.03 s | HaWoR V3；Robot arm/hand 数值 PASS，待视觉确认 |
| Poker | `play_cards_0903_224` | 144 / 4.80 s | HaWoR V3；Robot arm/hand 数值 PASS，待视觉确认 |
| Poker | `play_cards_0903_243` | 106 / 3.53 s | HaWoR V3；Robot arm/hand 数值 PASS；右侧51–53帧真实 UNKNOWN 空白 |

本轮修正的两个根因已经固定进管线：旧 HaWoR review 实际叠加了 RAW 细暗线和 bounded 亮线，current 已换成 bounded-only V3；旧 Robot renderer 把 world wrist 位移硬缩为 0.35，current 已改为 `motion_gain=1.0`，并由统一 selector 为每条完整轨迹求一个整段固定、不随帧移动的 base placement。Robot 六条 arm 6/6 通过，KaiHand 5/6 通过；所有 Robot 仍为 `authority=false`，不能生成训练 action sidecar。

数值与 lineage 入口：[`HaWoR CURRENT`](../../tasks/control/runs/20260911_six_session_hawor_temporal_successor_v1/CURRENT_SUCCESSOR_MANIFEST.json)、[`Robot CURRENT`](../../tasks/control/runs/20260911_six_session_robot_motion_transfer_successor_v2/CURRENT_MANIFEST.json)、[`完整视频 RESULT`](../../tasks/control/runs/20260911_six_session_full_pipeline_video_review_v2/RESULT.json)。

<!-- BEGIN TWO_TASK_78_NINE_HOUR_EVIDENCE -->

## 两任务78流程九小时证据状态

> 证据刷新：`2026-09-11T19:40:08+08:00`；总状态：`IN_PROGRESS_EVIDENCE_GATED`。这里不使用模型记忆；仅消费已落盘、SHA-256复算通过的`RESULT.json`、`AGENT_REVIEW.json`或`STATE.json`。九小时directive SHA：`a2640f5df02d362937246e5e11b6114b0d99652cd8dd5f167f396dd471a2bdc7`。

| 必交付项 | Chips | Poker |
|---|---|---|
| Mask 78交付 | `COMPLETE_TERMINAL_MIXED`：Fresh successor V3 sealed both Chips Mask lanes: role-removal B53/C25 and task-object B72/C6; all 78 sessions in each lane have honest terminals and Grade C is not promoted. | `COMPLETE_TERMINAL_MIXED`：Fresh successor V3 sealed both Poker Mask lanes: role-removal B71/C7 and task-object B49/C29; all 78 sessions in each lane have honest terminals and Grade C is not promoted. |
| Clean 78交付 | `HOLD`：PAUSED_BY_USER after in-flight Clean203 reached terminal B. Authorized Clean subbatch is 4/4 B and explicitly synthetic; no further Clean work or 78x2 terminal-mixed index started. | `HOLD`：PAUSED_BY_USER after in-flight Clean203 reached terminal B. Authorized Clean subbatch is 4/4 B and explicitly synthetic; no further Clean work or 78x2 terminal-mixed index started. |
| Robot 78交付 | `HOLD`：PAUSED_BY_USER at safe boundary: two-task numeric preparation and four hand A/B PNGs preserved; zero Robot authority/sidecars, no batch launched, no matching process remains; resume requires new user instruction and hand visual approval. | `HOLD`：PAUSED_BY_USER at safe boundary: two-task numeric preparation and four hand A/B PNGs preserved; zero Robot authority/sidecars, no batch launched, no matching process remains; resume requires new user instruction and hand visual approval. |
| HUMAN_RAW_RGB checkpoint | `HOLD`：PAUSED_BY_USER: HumanEgo watcher stopped at WAIT_ROBOT_ACTION_SIDECAR; zero training/checkpoints; resume requires a new instruction. | `HOLD`：PAUSED_BY_USER: HumanEgo watcher stopped at WAIT_ROBOT_ACTION_SIDECAR; zero training/checkpoints; resume requires a new instruction. |
| ROBOT_VIEW_RGB checkpoint | `HOLD`：PAUSED_BY_USER: four-checkpoint visual A/B branch stopped before bundle/training; exact watcher count is zero. | `HOLD`：PAUSED_BY_USER: four-checkpoint visual A/B branch stopped before bundle/training; exact watcher count is zero. |
| 同帧同action推理对比 | `HOLD`：PAUSED_BY_USER: real comparison remains ungenerated; tested CPU tools are preserved as preparation only. | `HOLD`：PAUSED_BY_USER: real comparison remains ungenerated; tested CPU tools are preserved as preparation only. |

机读状态：[`CURRENT_9H_STATUS.json`](../../tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/evidence_governance/CURRENT_9H_STATUS.json)；追加式事件账本：[`9H_EVENT_LEDGER.jsonl`](../../tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/evidence_governance/9H_EVENT_LEDGER.jsonl)。

完成口径：Mask、Clean、Robot均需两任务各78条诚实终态；含C时记`COMPLETE_TERMINAL_MIXED`且C不得晋升，真正78条全过才记`COMPLETE_PASS`。训练需四个checkpoint（两任务×两RGB域），只消费明确A/B并满足16/3会话与256/48配对H50门；随后需逐任务同帧、同action推理可视化对比。

<!-- END TWO_TASK_78_NINE_HOUR_EVIDENCE -->
更新时间：`2026-09-10T22:48:46+08:00`

当前仅以 [两任务 Baseline 复用指南](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_REUSE_GUIDE_ZH.md)、[机器 authority](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_AUTHORITY.json) 和 [最新视频索引](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/VIDEO_INDEX_ZH.md) 为入口。旧 fallback、旧对象身份、旧 Clean 和被否决 Robot 视频均不在本页列出。

从零搭建、各阶段技术原理、质量门、失败修复与理想管线图见 [端到端复现文档](RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)。

## 旧 010 compatibility 证据（已被 `rgb30_v1` current 取代）

> 本节记录早期403帧HDF时间轴兼容性审计，不再是0909运行状态。当前404帧final-contract canary与正式104条运行状态以本页顶部链接为准。

0909批次已只读枚举001..104共104个会话；当前010唯一权威 run 是 [`20260911_handle0909_010_pipeline_compatibility_v1`](../../tasks/control/runs/20260911_handle0909_010_pipeline_compatibility_v1/README_ZH.md)。`USER_SWITCH_001_TO_010`只终止旧001单条转换请求，不将001排除出后续104会话inventory。源 `chips_cards_handle_0909/010` 保持不变；已原子发布 tracker-style 010 目标，HDF5 403 行、视频 403 帧严格 1:1，采用户确认的 sourceIndex0 无径向 warp 直通图像。早期 equiDis62/FOV90 二次去畸变产物均已标为 `NONCANONICAL_INVALIDATED`。

兼容目标的 K 仅声称为本会话 factory/source K 按像素缩放，未经独立标定确认为编码 VST 域的 metric ground truth；D 不用于 passthrough 编码域。目标已补齐同会话 controller 6DoF sidecar 及逐帧 metadata：原始PICO 1377/1377行左右controller均存在，HDF5双侧403/403行全finite；wrist是controller经同会话标定组合，MANUS25两侧有效403/403。PICO optical Hand却在双侧1377/1377行都`isActive=0`，不得用controller/MANUS25伪装PICO21。

因此目标仅授权 visual/development compatibility，不授权 metric depth、Object6D 绝对尺度、robot-base 外参或 contact/collision。当前 CPU preflight v4 中 G0 媒体与 G1 数值 camera timeline PASS；源格式无有效 PICO21，故 G2 严格 fail closed，G3 HaWoR 待中央 GPU lease 安全后运行，不得写成全管线 PASS。

0909 canonical 执行顺序现已固定增加 `HaWoR–Stereo visible-surface QA`：仅在同 session 的 current bounded MANO、Depth frames、registration authority 和 current role/hand mask 齐备时，于 **Depth registration 之后、Object6D/Robot 之前** 计算 selected-left camera 同像素表面 `delta-Z`，并产出左右曲线、热图、MAE/P50/P95、coverage 及 RGB overlay MP4。其门只检查 support 是否充足，不是外部 GT 精度门，不授权也不自动纠正 HaWoR/Depth/Object6D/Robot。完整合同与基准结果见 [RESULT](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/RESULT.json)、[8-session INVENTORY](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/INVENTORY.json)和 [0909 QA CHECKLIST](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/HAWOR_STEREO_SURFACE_QA_CHECKLIST.json)。

exact78 的封账计数不变：role-removal 156/156（B54/C102），task-object 156/156（B121/C35）；两lane同会话A/B交集36，其中公制标定ready 26。显式扩批已经完成：Chips050 canary与successor21均为`COMPLETE_PHASE`，合计22/22个校正Depth为Grade B、36/36个独立physical-object Object6D child为Grade B，Chips未使用union，中央GPU lease已释放。它们只授权Clean视觉基线，不授权Robot接触或训练。Clean四条/552帧CPU prepare已原子发布，object overlap=0，但GPU plan为`NOT_RUNNABLE_WAIT_REAL_DONOR_AND_AUTHORITY`，GPU未启动。

“每任务78条终态”不能写成“每任务78条已通过/已Clean”。分任务实际数量如下：

| 阶段 | Chips | Poker | 含义 |
|---|---:|---:|---|
| HaWoR | 78终态；72 A/B、6 C | 78终态；72 A/B、6 C | 推理批量结束，但C不可下游 |
| role-removal Mask | 78终态；7 B、71 C | 78终态；47 B、31 C | Chips主要瓶颈 |
| task-object Mask | 78终态；72 B、6 C | 78终态；49 B、29 C | 与role lane独立审核 |
| 两Mask lane同时A/B且有公制标定 | 7 | 19 | 合计26条，已完成Depth/Object6D |
| Clean | 034用户接受成片1条；exact78新prepare 0条 | 042用户接受成片1条；另4条仅prepare | prepare/spec不等于Clean成片 |

Robot 的两段48帧 world-first V3 已获用户目视确认；这只授权全片开发审核。Poker171帧与Chips293帧 fullsession V1 都诚实终止为 `FULLSESSION_HOLD_NUMERIC_REVIEW`。Poker fullsession V2同样HOLD且arm位姿退化；所有authority字段仍为false，不得生成训练sidecar。最新机读并行状态见 [`PARALLEL_CONTINUATION_STATE.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/PARALLEL_CONTINUATION_STATE.json)。

<!-- BASELINE_STATUS_BEGIN -->
状态快照：`2026-09-09T15:22:00+08:00`（来源：baseline `STATE.json`）

| 阶段 | Chips | Poker |
|---|---|---|
| Raw QA | GRADE_B_CURRENT | GRADE_A_CURRENT |
| HaWoR | BOUNDED_V2_GRADE_B_CURRENT | BOUNDED_V2_GRADE_A_CURRENT |
| Depth | CORRECTED_STEREO_CURRENT_INPUT | CORRECTED_STEREO_CURRENT_INPUT |
| Mask | ROLE_REMOVAL_PLUS_THREE_INSTANCE_OBJECT_PROTECTION_GRADE_B_CURRENT | ROLE_REMOVAL_PLUS_ACTION_CONDITIONED_SAME_CARD_GRADE_B_CURRENT |
| Object6D | THREE_FIXED_INSTANCE_GRADE_B_CURRENT_NO_CONTACT_TRUTH | ACTION_CONDITIONED_SAME_CARD_GRADE_B_CURRENT_NO_CONTACT_TRUTH |
| Clean | EXPANDED_ROLE_V3_SYNTHETIC_GRADE_B_CURRENT | EXPANDED_ROLE_V3_SYNTHETIC_GRADE_B_CURRENT |
| Robot | NO_AUTHORITY_V3_ACCEPTED_FULLSESSION_V1_HOLD | NO_AUTHORITY_V3_ACCEPTED_FULLSESSION_V1_V2_HOLD |
| Training | WAIT_CURRENT_ROBOT_ACTION_SIDECAR | WAIT_CURRENT_ROBOT_ACTION_SIDECAR |
<!-- BASELINE_STATUS_END -->

## 当前 Clean

- [Chips 293帧 Clean V3](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/get_potato_chips_0902_034_expanded_role_v3/CLEAN_SYNTHETIC_中文全片复核.mp4)：右腕 Tracker/绑带已纳入独立时序删除域；源帧90–292共203帧均非空且原Tracker像素漏删最大值为0；三个薯片独立保护。
- [Poker 171帧 Clean V3](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/play_cards_0902_042_expanded_role_v3/CLEAN_SYNTHETIC_中文全片复核.mp4)：扩大左手、腕带和Tracker删除域，A♦保护保持。
- [Poker与Chips困难窗接受凭证](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/CLEAN_EXPANDED_V3_USER_ACCEPTANCE_20260909.json)；[Chips全片追加接受凭证](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/CHIPS_CLEAN_EXPANDED_V3_FULL_USER_ACCEPTANCE_20260909.json)。两任务当前全片均已由用户确认通过。

Clean 是快速视觉背景基线：先消费同会话真实 donor，其余 UNKNOWN 使用 ProPainter，像素来源在 source-map 中区分。`SYNTHETIC_PROPAINTER` 不是物理观测，不得作为深度、对象姿态、接触或标定证据。

## 当前 Robot

当前仍没有可复用的任务 Robot authority。[双侧静态装配闭包四视图](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/static_interface_closure_audit_v1/TIANJI_NATURALV2_KAIHAND_静态接口闭包_双侧四视图.png) 已通过静态开发审核。

上一条Poker 24帧共享平移视频已被用户否决，不再列为current视频。用户指出physical-left KaiHand应“掌心朝右手、拇指向上”，并与HaWoR人手保持真实thumb-index虎口。复核发现旧代码异手性叉乘符号错误：旧候选4个关键帧真实palmar normal误差约153°，旧门却误报约25°；只看拇指端点在腕点上方也不足以验证虎口。

后继24帧虽然通过旧有数值门，但已被用户否决：代码把`physical-left → human-right`、`physical-right → human-left`交叉映射；第一视角朝外，而整机栏从机器人正面观看，造成颜色、手形和左右臂任务角色不统一。否决收据位于该候选目录的`USER_VISUAL_DECISION.json`；不得继续派生。

用户已否决第0帧V1；V4也因肉眼改动不足而撤回。V5把左拇指归一化可见间隙由0.1401增至0.2122（人体参考0.2306），用户已明确确认“姿态没有问题”。V7钢蓝灰手已被用户的白手要求取代；[当前Poker第0帧白手V8](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_outward_frame0_pose_v8/POKER_042_第0帧_同侧左右手_同向整机姿态确认.png)的全部状态字段与V5/V7 bit-exact，只把双手改为暖白，机械臂和NaturalV2法兰仍为暖象牙白，蓝/红只标左右。

[当前Chips034第0帧 world-first 审核图](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_outward_frame0_world_v2/CHIPS_034_第0帧_同侧左右手_同向整机姿态确认.png)从Chips自身`joints_3d_world`重算，不复制Poker姿态。用户已确认Poker V8与Chips V2单帧的姿态、左右手和颜色无问题；随后也确认两段48帧V3时序无问题，并授权进入全片开发审核。

两条当前短片都取原始连续第0–47帧，以12 fps慢放为4秒三联画；第0帧`q_arm/q_hand/T_camera_base`与已接受静帧bit-exact，physical-left始终对应human-left，physical-right始终对应human-right。两条都实现`T_world_hand(t)=c2w(t)@T_camera_hand(t)`、固定task-level `T_world_base`、`T_world_base@FK@T_tool_hand≈T_world_hand(t)`，并仅在渲染时使用`T_camera_base(t)=inv(c2w(t))@T_world_base`：

| 任务 | 当前短片 / RESULT | 相机相对第0帧运动 | world base运动 | 机械臂末端最大误差 | 机械臂速度 / 加速度 | 数值门 |
|---|---|---:|---:|---:|---:|---:|
| Chips034 | [48帧world-first同侧白手慢放](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/CHIPS_034_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4) / [RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/RESULT.json) | 18.471 mm / 5.535° | 0 | 0.4203 mm / 0.1877° | 0.10813 rad/帧 / 0.05898 rad/帧² | 20/20 |
| Poker042 | [48帧world-first同侧白手慢放](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/POKER_042_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4) / [RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/RESULT.json) | 8.317 mm / 1.152° | 0 | 0.01295 mm / 0.00340° | 0.05455 rad/帧 / 0.01577 rad/帧² | 20/20 |

Poker V3/V5/V7/V8静态审核代码本身仍是只用`joints_3d_camera`的相机链；现在的Poker短片是另一个显式world-first时序后继，不能混称同一条代码路径。非零`c2w`运动、固定world base和闭包门证明短片没有直接把头戴相机位移当作机器人底座或腕部运动。V3还把首帧相机门诚实写成绝对误差≤1e-12（Poker实测6.94e-18，Chips为0），只有`q_arm/q_hand`声明bit-exact。两段短片已获用户视觉确认，但仍不构成Robot current authority、正式action sidecar或训练授权。

全片开发审核的当前结果：

| 任务 | 帧数 | 通过门 | 失败重点 | 结论 |
|---|---:|---:|---|---|
| Poker042 | 171 | 14/18 | arm world位置11.0162 mm；bone 104.886°；tip 29.827°；右臂frame98–170分支失败 | `HOLD`，困难窗90–102 |
| Chips034 | 293 | 16/18 | tip 15.6319°；右臂最后4帧分支失败 | `HOLD`，等待经Poker验证的新后继方法 |
| Poker042 V2 | 171 | 13/18 | arm位置49.0604 mm；arm旋转17.9956°；bone 92.9914°；tip 25.2499°；右臂frame103–170分支失败 | `HOLD`；局部branch buffer只推迟5帧并造成arm退化 |

手形算法必须把拇指 `q[0:6]` 独立计算，并将 index/middle/ring/pinky 按 MANO `MCP→PIP→DIP→TIP` 与机器人骨段逐段对齐。禁止把腕点到指尖整条弧长重采样后让掌段冒充第一节指骨。V2表明逐帧局部branch penalty不能同时保证branch和腕部pose；后继必须重新设计跨时域可行路径。既有10 mm、60°、15°和`Link5.x≥0.04 m`门限不放宽。

## 下一步

1. Poker V1/V2保持HOLD；下一版先做跨时域branch可行性与指链语义的CPU验证，不继续沿用V2局部penalty，也不修改门限。
2. Exact78 Depth/Object6D扩批已完成且lease释放；Clean仍须先补齐并授权fresh same-session real-donor producer及`SOURCE_MAP_MANIFEST.json`，之后才可启动GPU。
3. 只有全片Robot A/B authority及正式action sidecar达到最低数量后，才为每个任务分别训练 `HUMAN_RAW_RGB` 与 `ROBOT_VIEW_RGB`。

机器状态：[STATE.json](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/STATE.json)。技术来源：[TECHNICAL_PROVENANCE_ZH.md](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/TECHNICAL_PROVENANCE_ZH.md)。
