# exact78 完成执行路线

本文件只定义执行顺序和边界，不手填实时计数。实时状态必须读取：

1. `CURRENT_STATUS_RECEIPT.json`
2. `CURRENT_PROJECT_STATUS_MIN.json`
3. 当前 Task Packet

## 有限收敛顺序

1. Clean runtime recovery：重跑被缺失 import 破坏的 19 条，发布 successor 58 行矩阵。
2. Robot：只消费 Clean PASSED；每条按 Metric Contact、Pose-only Visual、质量 C、运行失败或前置阻塞封账。
3. Visual Aux eligibility：先用轻量投影预检计算每条会话真实 H50 窗口，再核 Raw/Robotized 同窗；Poker 固定补足 6 train、2 validation 的 Visual Wave1。
4. 四个 epoch-0：Chips/Poker × Raw/Robotized，全部通过后才进入正式训练。
5. 四支 Visual Aux：串行取得 GPU lease，按共同 recipe 训练，发布 checkpoint、loss、ADE/FDE/PCK。

H50 预检必须先于逐帧 RGB bundle 渲染。一个 Robot 数值 PASS 不自动等于可训练：若任一手根长期在画外，`future_2d_valid` 会使整条会话得到零窗口，此时应记 `BLOCKED_PREREQ_ZERO_H50_WINDOWS`，而不是生成空 checkpoint 或降低窗口门。

当前训练接口已有两类真实数据闭环证据，但都不能代替正式 train/validation ledger：

- Poker245（heldout）得到 18 个 H50 窗口，证明数值通过轨迹可以进入配对 bundle。
- Chips050（test，Pose-only Visual）得到 131 个 H50 窗口；Human Raw/Robotized 两支均完成一次真实 forward/backward smoke，证明 `control_ground_truth=false` 的视觉轨迹可以合法生成 future-2D 监督。

二者都保持原冻结 split，不得移入训练集。四支 checkpoint 仍必须等待各任务的 16/3 个 train/validation 会话和 256/48 个窗口门真实满足。

## 本轮完成口径

- `EXACT78_TERMINAL_COMPLETE`：156 条 Raw 会话在每个适用阶段都有唯一、可追溯终态。
- `EXACT78_WAVE0_CLEAN_COMPLETE`：冻结的 58 条 Wave0 会话均由当前 successor 给出 Clean `PASSED`、质量 C、最终运行失败或前置阻塞；旧 import 失败不能冒充算法质量失败。
- `EXACT78_ROBOT_TERMINAL_COMPLETE`：156 行 Robot 矩阵无遗漏；每行只能是 Metric Contact、Pose-only Visual、质量 C、最终运行失败或前置阻塞。该口径不等于 156 条 Robot authority。
- `FOUR_VISUAL_AUX_CHECKPOINTS_COMPLETE`：四支 future-2D Visual Aux 均完成真实 epoch-0 和正式训练，并发布 checkpoint、loss、ADE/FDE/PCK。数字 retarget 轨迹不属于 real Robot action。
- `EXACT78_END_TO_END_COMPLETE` 只有以上里程碑均封账时才成立；Physical Deployment 与最终 Robot Policy 仍可因外部真值/真实动作缺失保持阻塞。

## 转换率整改顺序

转换率按互斥的“首个阻塞阶段”统计，不能把各阶段 C 数直接相加。当前整改必须按以下优先级执行：

1. **消除假损失**：先恢复 19 条因当前代码被清理而出现的 Clean import 运行失败；不改数据、不改模型阈值。
2. **拆分视觉与公制通路**：缺公制标定的三路 A/B 会话不能做 Metric Contact，但可在固定 Visual Wave1 选择内做 Clean 与 pose-only 非接触视觉；不得借邻会话标定。
3. **Mask 失败簇修复**：HaWoR、Role Mask、Object Identity 分簇，各自只允许代表 canary、两条旧 A/B regression 和最多两轮 successor。
4. **Robot 失败簇修复**：arm 可达性与 hand fingertip-direction 分开优化。arm 使用全片统一 base placement 与最差帧关节余量；hand 只增加 distal/tip-direction refinement 和逐指时序 warm-start，不降低既有角度门。
5. **训练不消费错误终态**：只有 `clean_join_ready`、合法 Robotized RGB、共享 `training_valid_mask` 与完全同窗 future-2D 标签可进入四支 Visual Aux。

每轮优化必须同时发布：失败 canary 结果、固定 regression、转换率变化、计算预算、停止原因。提升数量但使旧 A/B 退化的 successor 不得成为 current baseline。

## 禁止捷径

- 不把 Clean 合成像素反喂给 Depth/Object6D。
- 不把 pose-only/contact UNKNOWN 帧当 metric contact 数据。
- 不把 visual trajectory 称为 real robot action。
- 不把数字残差、视觉审核或内部 registration 写成物理真实精度。
- 不为提高转换率而取消失败门；必须按失败簇修算法并保留 regression。

转换率的最新证据报告由 `src/chaoyang/ops/build_exact78_conversion_funnel_v1.py` 生成，禁止从聊天记忆复制计数。

## 当前执行包

- Clean 运行时恢复：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/task_packets/exact78_wave0_clean_runtime_recovery_v53/TASK_PACKET.json`
- Robot 当前可运行批次：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/task_packets/exact78_v52_lane_c_contact_robot/TASK_PACKET.json`
- 四支 Visual Aux：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/task_packets/exact78_visual_aux_four_checkpoints_v55/TASK_PACKET.json`
- 转换漏斗证据：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_conversion_funnel_v2/CONVERSION_FUNNEL.json`
- Robot 失败簇：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_robot_failure_clusters_v1/ROBOT_FAILURE_CLUSTERS.json`
- Robot→H50 首轮实测：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_h50_preflight_v55/PREFLIGHT.json`
- Pose-only H50 轻量预检：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_h50_pose_only_preflight_v55/PREFLIGHT.json`
- 当前8条Pose-only全量H50预检：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_batch_preflight_v55/PREFLIGHT.json`
- 固定split eligibility索引生成器：`src/chaoyang/human_ego/tools/build_visual_aux_eligibility_index_v56.py`
- 当前split eligibility索引：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_eligibility_index_v56/ELIGIBILITY_INDEX.json`
- Pose-only 配对 canary：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_bundle_canary_v55_2/get_potato_chips_0903_050/RESULT.json`
- Pose-only 两分支真实反向传播 smoke：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_bundle_canary_v55_2/REAL_BUNDLE_SMOKE_RESULT.json`
- Wave0 自动续跑器：`src/chaoyang/ops/run_exact78_wave0_completion_supervisor_v55.py`
- future-2D 模型：`src/chaoyang/human_ego/training/VisualAuxFuture2DModel.py`
- future-2D 训练器：`src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py`
- future-2D 数据账本 schema：`contracts/visual_aux_dataset_ledger_v53.schema.json`

执行代理不得从本路线中的叙述推断实时数量；每次仍须先读 current receipt、最小状态页和对应 Task Packet。
