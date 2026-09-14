# 两任务批量、Robot 与 HumanEgo 自动续跑交接

> **2026-09-11 15:34 当前交接覆盖：** 0909 `rgb30_v1`仍由独立程序处理；用户已另行明确恢复exact78缺口链。
> 唯一状态入口为 [`CURRENT_STATUS_ZH.md`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS_ZH.md)
> 和 [`CURRENT_STATUS.json`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS.json)。
> exact78当前状态必须先读[`EXACT78_RESUME_CURRENT_STATUS_ZH.md`](../../tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/EXACT78_RESUME_CURRENT_STATUS_ZH.md)；旧PID与旧暂停文字只是历史恢复证据。

## 0909 自动化恢复边界

current-code final-contract resumed canary已提交为010发布、001/038/104拒收，
`DATASET_RESULT.json.state=CANARY_COMMITTED`，SHA为`138ab87a…`。正式104条随后已fresh恢复；
14:58父PID `1757739`及两个worker存活，正式根已有10条拒收marker但尚无总结果。

当前已有正式writer，**不要再次运行下面的启动命令**。它只供未来确认当前writer不存在、
代码与合同SHA一致且目标允许no-clobber恢复时参考：

```bash
cd /mnt/workspace/code/chaoyang
python3 tools/batch_convert_handle_egodex_to_tracker.py --dry-run
python3 tools/batch_convert_handle_egodex_to_tracker.py --workers 2 --render-review
```

若未来正式批量中断，只能在确认没有同目标活进程且已提交 marker 的 receipt 与当前 source/policy/mapping/code/calibration/dependency 完全一致后执行：

```bash
cd /mnt/workspace/code/chaoyang
python3 tools/batch_convert_handle_egodex_to_tracker.py --resume --workers 2 --render-review
```

控制入口：[`STATE.json`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/STATE.json)、[`RESULT.json`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/RESULT.json)、`PID_FINAL.txt`和`BATCH_RESUMED_FINAL.log`。旧`PID.txt=STOPPED_BY_USER`只对应上一停止尝试。实时判断还必须核对进程身份、终态marker和目标存在性。禁止同时启动两个正式writer；禁止删除staging来伪造成功；禁止将`.canary_*`复制成正式根；禁止让`rejected/`会话进入HaWoR。正式`DATASET_RESULT.json`尚未出现前，后继全部保持WAIT。

早期 `run_handle0909_fov_detector_canary.py --watch` 仍可能以独立开发 watcher 存活，但它绑定旧 pinhole 候选，既不是 `rgb30_v1` writer，也不授权后续；不得把它的 PID 当作正式批量运行证据。

exact78旧SIGSTOP进程已在核验已完成会话后退出；fresh no-clobber role-mask metadata-recovery guardian PID `1765429`已启动，15:34为`6/41`。HumanEgo watcher仍未启动，Robot action sidecar仍为0。0909程序与exact78输出根彼此独立，禁止互相覆盖。

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
更新时间：2026-09-10 22:36 +08:00

本页用于额度耗尽、终端断开或下一位执行者接手时恢复工作。它只描述允许自动发生的动作和明确人工门，不把 `RUNNING/WAIT/QUEUED` 写成 `COMPLETE`。

> 当前恢复入口：先读 [`PARALLEL_CONTINUATION_STATE.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/PARALLEL_CONTINUATION_STATE.json)。2026-09-10 18:00 以前写成“48帧待用户确认 / 26条尚未授权 / Clean仅queue”的文字已被本次状态取代；不得据此重启旧controller。

当前真实状态为：两任务48帧world-first V3已获用户确认；Poker/Chips全片V1分别14/18和16/18门，均为HOLD；Poker fresh V2也以13/18门HOLD结束且未晋升。Exact78 Chips050 canary与successor21均已完成，22个Depth和36个独立Object6D child全部Grade B，中央GPU租约已释放。Clean四条/552帧prepare已原子发布并验证object overlap=0；GPU plan明确`NOT_RUNNABLE_WAIT_REAL_DONOR_AND_AUTHORITY`，GPU未启动。HumanEgo仍只有PID 1047418 watcher，`WAIT_ROBOT_ACTION_SIDECAR`、`training_started=false`。

## 1. 恢复时先读什么

1. [端到端技术与复现说明](RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)
2. [exact78 中文交接](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/CONTINUATION_HANDOFF_ZH.md) 与 [机读交接](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/CONTINUATION_HANDOFF.json)
3. [exact78 实时续跑状态](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/CONTINUATION_STATE.json)
4. [HumanEgo A/B 训练恢复说明](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/TRAINING_CONTINUATION_HANDOFF_ZH_V1.md) 与 [机读收据](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/TRAINING_CONTINUATION_HANDOFF_RECEIPT_V1.json)
5. [Robot 正式化准备结果](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_task_translation_formalization_prepare_v1/RESULT.json)、[技术说明](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_task_translation_formalization_prepare_v1/TECHNICAL_NOTE_ZH.md) 和 [左手用户视觉约束](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_task_translation_formalization_prepare_v1/poker/PHYSICAL_LEFT_USER_VISUAL_CONSTRAINT_RECEIPT.json)
6. [只读健康状态](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/AUTOMATION_CONTINUATION_HEALTH.json)、[并行续跑状态](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/PARALLEL_CONTINUATION_STATE.json) 与中央 [`GPU_LEASE.json`](../../_run/GPU_LEASE.json)

目录名、mtime、聊天摘要或孤立视频都不是 authority。恢复前必须核 `RESULT + AGENT_REVIEW + bytes/SHA + session/frame closure`。

## 2. 当前自动化边界

| 进程 | 快照 PID | 当前行为 | 允许自动做什么 | 明确禁止 |
|---|---:|---|---|---|
| role-removal Mask guardian | 1503585（已退出） | 14:06完成156/156并封账：B54/C102，remaining=0，lease released | 无需恢复；只读总收据与终态索引 | 不重跑、不覆盖、不改质量门 |
| exact78 continuation controller | 1503583（已退出） | 14:13完成授权自动范围：固定4条join-ready的Depth/Object6D均B，Clean queued=4但未启动 | 无需恢复；`COMPLETE_AUTHORIZED_AUTOMATIC_SCOPE`是终态 | 不启动Clean、Robot、训练、新cohort或扩大26条calibrated-ready范围 |
| exact78 expansion v2 | 已退出 | Chips050 canary与successor21均`COMPLETE_PHASE`；22 Depth B、36独立Object6D child B；lease released | 只读复用冻结结果和逐session SHA | 不处理缺标定10条；Chips不得用union冒充实例；不授权Robot接触或训练 |
| Clean expanded-role v3 prepare | 已退出 | 4条/552帧CPU prepare已原子发布；`PREPARE_RESULT`通过；GPU未启动 | 等待fresh real-donor producer及authority | 不用只冻结034/042的旧donor runner冒充通用runner；不在Depth持lease时启动GPU |
| HumanEgo A/B launcher | 已退出；旧快照PID 1047418 | 当前无等待器、无优化器；准入仍为`WAIT_ROBOT_ACTION_SIDECAR` | 新授权后也只能先补Robot A/B sidecar、正式handoff、成对bundle与epoch-0 | 不用Robot C/撤回视频/fallback补数 |
| continuation health watcher | 1503528（动态PID，仍以`AUTOMATION_CONTINUATION_HEALTH.json.pid.json`为准） | 2026-09-10 11:37已重新启动；每60秒只读检查状态/PID/lease | 报告假RUNNING、过期lease、进度分母和训练越权 | 不重启、不释放lease、不晋升authority |

PID是本页更新时间的快照，不能只因PID变化就启动第二份进程。先检查对应状态文件中的 `pid + proc_start_ticks` 和 `/proc/<pid>/stat`，再按各自交接文档中的防重复命令恢复。

## 3. 已完成、进行中、等待

- HaWoR exact78：156/156 已封，A1/B143/C12。
- task-object identity Mask：156/156 已封，B121/C35。
- role-removal Mask bounded-v2.1：156/156 已封，B54/C102；总结果 `COMPLETE_156_TERMINALS`，总审阅 `PASS_156_HONEST_TERMINALS`。
- 双Mask join：两lane均156终态；同会话A/B交集36，其中公制标定ready 26、缺标定诚实终止10。此计数不授权扩大Depth/Object6D执行范围。
- 固定4条 Poker join-ready 的 Depth 与 observed-only Object6D：均4/4 Grade B；Poker224原48帧partial已由同closure no-clobber恢复并完成。
- Exact78扩批：Chips050 canary与successor21均为`COMPLETE_PHASE`；22/22个Depth Grade B，36/36个独立physical-object Object6D child Grade B、`KEEP_INVALID`，未使用union；中央GPU lease已释放。
- Clean批量：4条/552帧prepare已原子发布，object overlap=0；仅生成handoff/spec，不是Clean视觉终态。缺fresh same-session real-donor producer、`SOURCE_MAP_MANIFEST.json`验证和authority，因此GPU不得启动。
- Robot：仍是 `NO_CURRENT_TASK_ROBOT_AUTHORITY`。两段48帧world-first V3已由用户确认并只授权全片开发审核；Poker171帧和Chips293帧fullsession V1均HOLD。Poker V2在不改阈值时仍失败5门：arm 49.0604 mm / 17.9956°、bone 92.9914°、tip 25.2499°、右branch frame103–170；不能晋升或写训练sidecar。
- HumanEgo：旧launcher已退出；准入仍为`WAIT_ROBOT_ACTION_SIDECAR`，`training_started=false`，没有本轮checkpoint。详见[四checkpoint当前准入](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/FOUR_CHECKPOINT_CURRENT_READINESS_ZH.md)。

## 4. Robot 必须保留的人工门

当前没有可晋升的Robot authority。旧[Poker共享task平移24帧视频](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_static_closure_task_translation_24frame_v1/POKER_042_正确闭包_task共享平移_24帧开发复核.mp4)和后继physical-left 24帧均只作已否决问题证据，禁止进入current、训练或继续派生。

用户进一步明确：人类左手掌心朝右且抬起，人类右手掌心朝下且靠桌；机器必须同侧对应，第一视角和整机图必须同向朝外。V1因双臂穿身/手指错误被否决，V4因视觉改动不足被否决。V5只重算左拇指6-DoF，左侧归一化可见间隙从0.1401增至0.2122，用户已确认姿态没有问题。[当前Poker白手V8](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_outward_frame0_pose_v8/POKER_042_第0帧_同侧左右手_同向整机姿态确认.png)与V5/V7全部`FRAME0_STATES.npz`字段bit-exact，只把双手改为暖白。[当前Chips034 world-first第0帧](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_outward_frame0_world_v2/CHIPS_034_第0帧_同侧左右手_同向整机姿态确认.png)从Chips自身`joints_3d_world`重算，使用一个固定`T_world_base`，不复制Poker姿态。用户在2026-09-10明确接受两张静帧，并只授权生成短视频。

用户已确认的视频为：[Poker042 48帧慢放](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/POKER_042_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4)（[RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/RESULT.json)，20/20门）与[Chips034 48帧慢放](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/CHIPS_034_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4)（[RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/RESULT.json)，20/20门）。两者均为连续源帧0–47、12 fps慢放、4秒、1920×480三联画；第0帧`q_arm/q_hand`与已接受静帧bit-exact，`T_camera_base`按绝对误差≤1e-12闭合。该确认只允许派生fresh全片审核。

失败过程必须保留但不得晋升：Poker temporal V1把逐帧拇指细化污染到手根朝向，首帧偏离已接受根旋转9.109°；V2改为“已接受首帧根朝向 + human-world掌基相对旋转”。Chips temporal V1把机械臂求解单步误设为0.06 rad，低于既定0.12 rad/帧审核速度包络，快速右手区间积累44.21 mm / 9.21°误差；V2让IK使用完整0.12 rad包络但仍保留0.06 rad/帧²加速度门，最终最大误差降至0.4203 mm / 0.1877°。当前V3统一两个任务的最终runner，并把旧名`frame0_camera_base_bit_exact`纠正为`frame0_camera_base_abs_error_le_1e_12`；Poker实测6.94e-18，Chips为0。不要通过放宽最终误差门来掩盖失败。

即使两段短片已经确认，以下动作仍必须等待全片数值、接触、碰撞、遮挡和action-sidecar门：

- 修改已接受首帧状态、同侧映射或task placement；
- 从HOLD的V1继续派生或晋升；
- 晋升Robot authority；
- 写正式action sidecar训练handoff；
- 启动HumanEgo优化器。

Robot正式化代码已经闭合Object6D多实例、五指接触、固定分母碰撞、3 mm z-buffer与action-sidecar合同，并通过25项CPU测试；这属于“准备就绪”，不是执行或质量通过。当前短片只验证坐标链、姿态与短时序连续性，尚未通过Object6D接触、全片碰撞/遮挡或action-sidecar质量门。

坐标链必须分清：Poker V3/V5/V7/V8静态审核代码直接用`joints_3d_camera`，其`T_camera_base`按`T_A_B`表示B到A的约定是base→camera，IK内部使用`inv(T_camera_base) @ T_camera_hand @ inv(T_tool_hand)`。该静态链不读`joints_3d_world/c2w`，不得直接外推为全片轨迹。当前两条48帧短片则显式实现`T_world_hand(t)=c2w(t)@T_camera_hand(t)`、`T_world_base@FK@T_tool_hand≈T_world_hand(t)`和`T_camera_base(t)=inv(c2w(t))@T_world_base`。Poker窗内相机移动8.317 mm / 1.152°，Chips为18.471 mm / 5.535°，而两者`robot_world_base_motion_max=0`、camera/world base闭包最大绝对误差均为4.44e-16，所以这两条时序确实使用了去头动后的world-first链。旧`run_newtask_robot_kinematic_canary.py`虽也是world-first，但仍硬编码已被否决的交叉映射`physical-left→human-right`，所以不是current authority。正式通用化必须同时保留world-first和同侧映射，不能只修其中一项。

## 5. HumanEgo A/B 不得改变的问题定义

实验固定比较：同一任务、同一Robot action sidecar下，`HUMAN_RAW_RGB` 与 `ROBOT_VIEW_RGB` 对HumanEgo的影响。两支的action、HaWoR/Object ICT、metadata、帧集合、H50窗口、60/8/5/5 split、seed和训练recipe必须byte-identical，唯一变量是RGB。

两任务同时达到每任务至少16个train、3个validation current Robot A/B会话，并形成至少256/48个配对H50窗口前，训练launcher保持WAIT是正确状态。

## 6. 总健康检查

一次性检查：

```bash
cd /mnt/workspace/code/chaoyang
/usr/local/bin/python3 tools/check_two_task_pipeline_continuation.py \
  --stale-seconds 600 \
  --strict \
  --output tasks/control/runs/20260908_two_task_e2e_baseline_v1/AUTOMATION_CONTINUATION_HEALTH.json
```

常驻检查只能在确认没有同命令进程后启动：

```bash
cd /mnt/workspace/code/chaoyang
pgrep -af '^/usr/local/bin/python3 tools/check_two_task_pipeline_continuation.py --watch' || \
  /usr/local/bin/python3 tools/check_two_task_pipeline_continuation.py \
    --watch --poll-seconds 60 --stale-seconds 600 \
    --output tasks/control/runs/20260908_two_task_e2e_baseline_v1/AUTOMATION_CONTINUATION_HEALTH.json
```

健康报告为 `HEALTHY` 只表示状态、PID和authority边界自洽，不表示所有批量完成，也不替代视频目视审核。

## 7. 接手者的最短执行顺序

1. 运行总健康检查；若有alert，先核状态与PID，禁止盲目释放GPU lease。
2. 核 `MASK_ROLE_STATE=COMPLETE`、156条总收据与双Mask join；不要重启已退出的role guardian。
3. 核 expansion v2 的两个`COMPLETE_PHASE`、22 Depth B、36 Object6D child B及released lease；不要重启旧controller或successor21。
4. Clean prepare完成只表示handoff/spec可消费；虽Depth已释放lease，仍须先有fresh donor producer、`SOURCE_MAP_MANIFEST.json`和authority，不能直接启动GPU。
5. Robot两条48帧V3已确认；fullsession V1/V2都禁止晋升。下一版须保持world-first、同侧映射、拇指独立、非拇指`MCP→PIP→DIP→TIP`和原门限，并避免V2逐帧局部branch penalty。
6. 只有Robot A/B sidecar正式handoff达到训练门，才允许现有HumanEgo launcher从WAIT继续。

## 8. 不得自动做的事情

- 不从 `archive/legacy/fallback/fail_forward_prod` 取数据补齐数量。
- 不覆盖final目录，不把partial/staging当终态。
- 不把Grade C、用户否决或development-only Robot视频写入current authority。
- 不修改质量阈值来提高通过率；算法或合同改变必须fresh版本、canary、全片复核。
- 不在另一个进程持有中央GPU lease时抢卡。
- 不永久删除历史；已确认过时项只做可恢复归档并留下tree SHA与恢复命令。
