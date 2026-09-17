# V7.1 自动化执行交接

本页只说明执行方式，不手填实时数量。实时状态只读：

- `CURRENT_STATUS_RECEIPT.json`
- `CURRENT_PROJECT_STATUS_MIN.json`
- `CURRENT_PROJECT_STATUS_ZH.md`

## 四小时优化冲刺边界

本轮人工工程与研究优化预算固定为：

```text
开始：2026-09-15 00:16:18 +08:00
人工截止：2026-09-15 04:16:18 +08:00
之后模式：AUTOMATION_ONLY
```

机器合同位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/OPTIMIZATION_SPRINT_V71.json
```

四小时只用于冻结可执行 successor 命令、接触/遮挡 canary、证据 DAG 和必要回归；不以持续阅读长日志或无界调参填满预算。到期后仅允许已经冻结的自动监督、有限重试和明确终态封账继续运行。

## 自动执行器链

```text
src/chaoyang/ops/run_v71_finite_convergence_supervisor.py
```

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/STATE.json
```

该早期控制面监督器已经完成它能够执行的恢复工作；它不是当前进度事实，是否仍存活必须以事实账本和 PID/startticks 为准，不能根据本页判断。恢复旧监督器前必须先确认没有新执行器负责同一写集。

其设计职责是：

1. 采用 H3/H4/Object Hypothesis 的 immutable worker receipt。
2. 等现有 Clean worker 自己完成，不启动第二个 Clean。
3. Clean pending 归零后重建 156 行 exact78 矩阵。
4. 通过治理 CAS 更新任务终态并生成会议快照。

它明确不会：

- 启动任何 GPU 任务；
- 启动旧 Robot 批处理器；
- 覆盖 R7_0 或任意 final；
- 移动 train/validation/test split；
- 把候选视频晋升 Robot authority；
- 在没有 goldset 时计算遮挡 accuracy。

仅在其 Task Packet 仍为 current 且零冲突时，恢复命令为：

```bash
cd /mnt/workspace/code/chaoyang
python src/chaoyang/ops/run_v71_finite_convergence_supervisor.py \
  --poll-seconds 30 \
  --max-wall-seconds 172800
```

执行器使用 `archive/baseline-20260917-0aa69e9/content/history/runtime/_run/v71_finite_convergence_supervisor.lock` 防止双主。若进程中断，重新执行上述命令即可；所有业务结果来自 immutable receipt，而不是执行器内存。

### 四小时后的 S1 自动 canary

已启动第二个窄作用域监督器：

```text
src/chaoyang/ops/run_post_clean_s1_automation_v71.py
```

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_clean_s1/AUTOMATION_STATE.json
```

它必须同时观察到 `2026-09-15 04:16:18 +08:00` 已到、治理 `PASS/FRESH`、Wave0 Clean pending 为零且不存在活跃 Clean 任务，才会继续。它先完成 Chips023 全片 Robot-only 统一 z-buffer 开发导出，再对 `play_cards_0901_042` 执行一次已验证的 `fps 30→25` Role Mask 输入合同修复 canary；该 canary 通过后才顺序运行冻结的 `play_cards_0903_189`、`play_cards_0903_202` 两条旧 A/B regression，并生成独立的 `1 failure + 2 A/B` bundle 收据。任何一条失败都会有限停止，不扩批。最后在同一冻结失败会话上依次执行 Cutie 和 SAM2.1 **任务物体 modal-mask** 开发 canary。Robot z-buffer 只验证同一场景内 Robot optical-Z 与 instance/part/link/triangle provenance；没有对象层，不能证明手物遮挡已经解决。全片导出固定使用 R7_2：源图像尺寸优先从 NPZ 读取，否则由 K 反推，不得回退到 2048×1536 硬编码。R7_1 的旧 z-buffer 不得用于 selected-camera ownership。Role 修复与物体跟踪结果分开发布，不能互相证明。每个物体后端最多 3 次运行时 attempt、单次最多 2 GPU 小时；真实执行前强制重新计算 Cutie/SAM2.1 大权重 SHA，完整日志写入 automation 目录。两个后端结束后还会自动生成开发比较收据，只报告 known/UNKNOWN 覆盖、最长 UNKNOWN 连续段、面积时序变化、旧基线未观测区覆盖和后端间 IoU；冻结重入帧不是人工真值，因此该收据明确不计算 accuracy，也不晋升 Mask/Gold authority。

语义审计确认现有 S1 prompt 全部来自 `TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK`，所以该 canary 只能研究物体遮挡/重入，不能声称修复 Role Mask。Role Mask 的四角色输入尚未冻结，继续保持独立阻塞；不得为了利用现有脚本而混淆两个阶段。

该监督器不更新 current ledger、不覆盖任何 final、不扩批，也不授予 Mask、Contact、Object6D、Gold 或物理 authority。

### S1 后的自动封账

第二层监督器已经启动：

```text
src/chaoyang/ops/run_v71_post_s1_finalize.py
```

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_s1_finalize/AUTOMATION_STATE.json
```

它等待前述 S1 状态成为 `TERMINAL`，随后按固定顺序执行：治理校验、receipt 绑定的当前回归集、Clean 零 pending/零活跃任务双门、156 行 exact78 终态矩阵发布。最后只通过治理 CAS writer 聚合 `exact78_clean_r70_v71`、Role 修复 canary 和 Object Identity 双后端比较的任务终态。`COMPARISON_COMPLETE_REVIEW_REQUIRED` 只表示开发比较执行完成，不表示 Mask authority；脚本中的 claim limit 明确禁止自动晋升 Mask、Robot、Contact、Gold 或物理 authority。

恢复前先确认没有同名进程；安全恢复命令：

```bash
cd /mnt/workspace/code/chaoyang
python src/chaoyang/ops/run_v71_post_s1_finalize.py \
  --poll-seconds 30 \
  --max-wait-seconds 172800
```

### 封账后的 Robot 扩批与 H50 资格刷新

第三层监督器已经启动：

```text
src/chaoyang/ops/run_v71_post_finalize_robot_expansion.py
```

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_finalize_robot_expansion/AUTOMATION_STATE.json
```

它同时等待人工截止时间和 S1 封账成为 `TERMINAL`，随后重建 current cross-stage matrix，只消费 `READY_FOR_ROBOT_CURRENT_DRAFT` 的 Clean-joined 会话。旧的 Robot 任务已经封为运行失败，因此本执行器使用 `robot_geometry_v1` 任务心跳和新的 immutable output root，不复活旧 executor。每个子阶段最多 2 小时，整批最多 48 小时；进程超时会终止整个子进程组，重启时沿用冻结 selection，并允许已经完成的 candidate/C 终态继续存在。

Robot 扩批结束后，执行器刷新 pose-only candidate index、cross-stage matrix 和 H50 eligibility。它只产生 V5.2 pose-only 开发候选；所有候选仍需 V7.1 unified z-buffer、自穿透和 causal compositor 审计，因此执行器最终将 `robot_geometry_v1` 保持在明确的 `BLOCKED_PREREQ`，不会自动授予 Robot authority。

H50 扩批选择固定为：30 条 metric 会话，加最多 13 条 Poker Visual Tier 会话。Poker 的 metric Wave0 在冻结 split 中只有 10 条 train、1 条 validation，所以无论 GPU 是否可用，单跑 Wave0 都不可能满足 16/3 会话门。Visual Tier 先补 6 train/2 validation 的确定性缺口，再在同一冻结 split 内增加 4 train/1 validation reserve，避免一条 zero-window 或 Robot quality-C 就让四 checkpoint 路线必然失败。这些会话只缺同合同 metric calibration，可走 Visual Clean/pose-only 路线，但不能产生 Metric Contact authority。机器选择见：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/expansion_plan_R7_1/VISUAL_AUX_ROBOT_EXPANSION_PLAN.json
```

安全恢复前必须确认没有同名进程：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONPATH=. python src/chaoyang/ops/run_v71_post_finalize_robot_expansion.py \
  --poll-seconds 30 \
  --max-wait-seconds 259200 \
  --robot-timeout-seconds 172800
```

## 研究与优化队列

### Robot 质量门已分层

Robot 不再要求逐帧严格复制人手姿态才进入视觉候选。固定策略见：

```text
docs/governance/ROBOT_QUALITY_GATE_POLICY_V71_ZH.md
```

有限值、真实 URDF 限位、左右手性与安装闭包、UNKNOWN provenance、非相邻 link
自穿透和统一 z-buffer 保持硬门；速度/加速度只作为按 FPS 解释的视觉连续性门；arm
位置/旋转残差、hand bone/tip 残差以及全片逐行严格相似度属于软诊断。首批 6 条 Chips
均为 hard geometry pass、strict pose match fail，已经进入 pose-only review、增量
Occlusion 和 causal bundle。该分层不授予 Robot authority，也不把数字轨迹改称真实 action。

Clean 性能审计位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/optimization/clean_runtime/
```

当前 9 条、2491 帧样本显示：排除资源等待后约 87.9% 活动时间位于 CPU/I/O、donor、逐帧校验、编码和完整解码，ProPainter vendor 推理约占 11.3%；GPU lease 约 69.7% 时间被 CPU 前后处理持有。当前 worker 不热修改；它结束后优先改成 `CPU_PREPARED → GPU_INFERENCE → CPU_FINALIZE`，并仅在中段持有 GPU lease，再以一条 Chips 和一条 Poker canary 做回归。

GPU 空闲后只允许按 Task Packet 启动有界 canary：

```text
successor_hawor_bounded_v71
successor_role_mask_v71
successor_object_identity_v71
object_pose_hypothesis_v2
robot_geometry_v1
```

每个 successor 必须先固定 1 条失败代表和 2 条旧 A/B regression；最多两轮、每轮 4 小时墙钟或 2 GPU 小时。Task Packet 没有冻结 concrete command/input signature 时，不允许自动“猜参数”启动。

HaWoR 两个已冻结 child contract 的 CPU-only 诊断复现由以下有限监督器执行：

```text
src/chaoyang/ops/run_hawor_diagnostic_reproduction_pair_v71.py
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/hawor/diagnostic_reproduction_R7_4/
```

它只处理既有 HaWoR track 的 temporal bounded 参数，不重跑模型、不需要 GPU，也不能补回未知的
推理权重 SHA。结果只用于判断时序后处理是否值得形成真正 successor；原
`successor_hawor_bounded_v71=BLOCKED_REFERENCE_PROOF` 不会被诊断运行伪装成 authority 或 Wave delta。

R7_2 曾因监督器读取 `rows` 而不是实际 `canaries` 键误报零 canary 通过；R7_3 修复计数并切换到
项目固定 HaWoR Python 后，又把质量 HOLD 的退出码 2 误当运行失败。两份旧收据保留用于审计，
不得作为当前结论。R7_4 是正确不可变 successor：6 条均真实执行且无 runtime exception，4 条
`PASS_NUMERIC_NEEDS_HUMAN_REVIEW`，2 条因左手 `BONE_CV_REGRESSED` 保持
`HOLD_NUMERIC_GATES`。这只是 CPU 时序后处理开发证据，仍不解除 HaWoR 原推理权重来源证明债务。

Robot 扩批的 hard/soft 终态由以下轻量 watcher 自动转换为逐 session 归因表：

```text
src/chaoyang/ops/run_robot_conversion_diagnosis_watcher_v71.py
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_conversion_diagnosis_watcher_v71/AUTOMATION_STATE.json
```

它严格区分 `HARD_GEOMETRY_FAIL`、`HARD_PASS_SOFT_ARM_MISMATCH`、`HARD_PASS_SOFT_HAND_MISMATCH` 和严格姿态通过，不再把 morphology/workspace 的软残差统一写成“Robot 不可行”。输出仍是开发归因，不授予 Robot、Contact、控制或物理 authority。

右小指代表 canary 的可达性下界诊断位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_kaihand_pinky_reachability_v71/get_potato_chips_0902_087/RESULT.json
```

在 087 第75帧固定 wrist、当前 KaiHand URDF/关节限位下，44 个确定性多起点的 tip-only 优化最小误差仍为 18.050°，高于 15°门，且最优解四个小指关节均落在边界。它证明的是该数字模型、该帧、该固定 wrist 下的采样可达下界，不是 KaiHand 物理真值或全局不可达证明；后续应比较“投影到 Robot 可达流形后的软残差”，不能谎称原始人手姿态被精确复现。

Robot placement 的回放缩减审计位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_placement_reduction_audit_v71/RESULT.json
```

已完成的前 6 条 Chips 会话中，`0.20 m` 与 `0.258 m` 两个 backoff 候选均复现原 7 候选搜索的同一 winner（6/6），候选评估数由 42 降至 12，理论候选数减少 71.4%。这不是实测墙钟加速，也没有覆盖 Poker 或未见会话，因此不热改正在运行的 R7.3；下一不可变 Robot revision 必须先用 Poker canary 和旧 Chips regression 验证，才可采用缩减候选集。

该判断已交给独立 watcher 在 hard/soft 扩批终结后自动重算全部已完成 sweep：

```text
src/chaoyang/ops/run_robot_placement_reduction_watcher_v71.py
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_placement_reduction_watcher_v71/AUTOMATION_STATE.json
```

只有同时满足至少 12 条、Chips/Poker 均覆盖且 winner agreement 为 100%，watcher 才会输出 `READY_FOR_IMMUTABLE_SUCCESSOR_CANARY`；它仍不会改写 R7.3 或直接启动新 revision。否则自动保持原 7 候选集合。

同一 watcher 也会在终局自动复算 hand round-1/round-2 的汇总指标与记录墙钟。只有至少 12 条、跨 Chips/Poker 且 round-2 全部无汇总改善时，才会输出 `READY_FOR_CONDITIONAL_SKIP_CANARY`；实际跳过仍要求下一 revision 冻结逐行 reachability/saturation predicate。

Occlusion visible-surface watcher 已改为增量消费 immutable hard/soft Robot candidate，不再等 34 条全部结束后才开始。每条候选一经发布即可生成 sparse z-buffer、24 帧重点窗口和 Silver 数值证据；只有最终 index 封账仍需等待 Robot 上游终态。该并行化不改变候选、Object6D 或 Robot authority。

终局还会额外发布 `OCCLUSION_SILVER_REPORT.json`。当前实现未覆盖全片 temporal consistency、
authorized-band 外 byte-exact、双向 closure 和合法隐藏外观来源，所以即使 visible-surface 数值门
通过，完整 Silver 任务仍会有限收敛为 `FAILED_QUALITY_C`，并由
`run_occlusion_silver_governance_finalizer_v71.py` 自动写回事实账本；不会留下永远 RUNNING，也不会
把局部 canary 冒充 Silver authority。

Hand round-2 的首批价值审计位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hand_round2_value_audit_v71/RESULT.json
```

前 6 条 Chips 的 round-2 在 `failed_rows`、最大 fingertip direction error 和最大 bone error 三项汇总上均没有改善，额外记录墙钟合计 239 秒、均值约 39.8 秒/会话。它说明 round-2 是可优化项，但不是当前最大耗时源；不得据此直接删除 round-2。未来只有在冻结逐行 reachability/saturation predicate，并通过 Poker 与旧通过 regression 后，才能有条件跳过。

## 四支 checkpoint

### Task Packet R7_2 修复

当前 Task Packet 指针已迁移到：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_task_packets_R2/TASK_PACKET_INDEX.json
```

旧 Chips/Poker Visual Aux packet 曾把尚未生成的 `robotized_causal/ROBOTIZED_RGB_LEDGER.json`
列入初始 `read_set`。R7_2 successor 不改旧包，只让两支任务先读取已存在的
`visual_tier_robot_R7_2/AUTOMATION_STATE.json`，上游终态后再沿 receipt 引用读取冻结 matrix、
eligibility 和 ledger。当前 index 的全部 Task Packet `read_set` 已核验为零缺失。

### Robot 扩批后的 Visual Tier 补量与因果训练自动接力

Poker 的 metric cohort 在冻结 split 中最多只有 train 10 条、validation 1 条，因此增加一个独立 Visual Tier 监督器，而不是让四支训练在一个已知不可能通过的门上等待：

```text
src/chaoyang/ops/run_v71_post_robot_visual_tier.py
```

它等待 metric Robot 扩批终结，随后只处理 expansion plan R7_2 冻结的 calibration-missing、三路 A/B Poker（当前不可变选择为 10 train/3 validation，包含最低补量和 reserve）。Clean 只运行 causal same-session real donor，不运行 ProPainter，不需要也不伪造 Depth/Object6D；随后仅运行 pose-only Robot。输出使用独立 Visual Tier matrix/eligibility，不写回 metric current matrix。所有结果固定 `metric_geometry=false`，不得进入 Object6D、Metric Contact 或物理部署。

构建最终 combined matrix 前还必须等待 hard/soft Robot watcher 进入 `TERMINAL`。这只约束最终 eligibility 冻结，不阻塞前面的 Visual Tier Clean/Robot；目的是避免主 Robot 已结束而审计 watcher 尚未追平时，把缺最后一批候选的不完整 matrix 永久冻结。

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2/AUTOMATION_STATE.json
```

第四层监督器已经启动：

```text
src/chaoyang/ops/run_v71_post_robot_visual_aux.py
```

状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2/AUTOMATION_STATE.json
```

它等待 Visual Tier 监督器进入 `TERMINAL`，然后消费 metric 与 Visual Tier 合并但不提权的 eligibility 视图中，同时具备 Grade-B visual Clean、pose-only Robot 开发候选和非零 H50 窗口的会话。每个 session 使用 `--causal-training` 建 bundle：时刻 `t` 只允许 `source_frame <= t` 的真实 donor；未来 donor 与 ProPainter 像素从 Raw/Robotized 两支共享的 `rgb_training_valid_mask` 排除。每个任务分别检查 train 16 条/256 窗口、validation 3 条/48 窗口；不足即封 `BLOCKED_DATA_VOLUME`，绝不移动 split、复用 test/heldout 或生成空 checkpoint。

资格通过时，Chips 与 Poker 两个 pair 独立执行 Raw/Robotized epoch-0 和正式训练。GPU 通过 `src/chaoyang/ops/run_gpu_command_with_v71_lease.py` 获取 V7.1 TTL/fencing lease，等待 30 分钟不消耗算法 attempt，每支最多 3 次运行 attempt、单次 12 GPU 小时。正式输出固定包含 `best.pt`、`last.pt`、兼容 checkpoint、loss/ADE/FDE 曲线与 RESULT；所有结果均为 single-seed future-2D 工程 checkpoint，`control_ground_truth=false`、`policy_checkpoint=false`。

当前因果 bundle canary 已在 Chips050 保留 131 个 H50 窗口，同时排除 10,530,223 个未来 donor 像素和 57,963,382 个 ProPainter/不支持像素。它只证明因果过滤可运行，不授予 Occlusion、Contact 或 Robot authority。

安全恢复前必须确认没有同名进程：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONPATH=. python src/chaoyang/ops/run_v71_post_robot_visual_aux.py \
  --poll-seconds 30 \
  --max-wait-seconds 604800
```

当前四个分支都由 `VISUAL_AUX_CHECKPOINT_INDEX.json` 独立记录。只有对应任务达到冻结的 train/validation session 与 H50 window 门，且 causal Robotized 输入通过泄漏证明后，自动化才允许进入 epoch-0。阻塞状态是有限终态，不允许用 test/heldout 数据补 train，也不允许输出空 checkpoint 或假 loss 曲线。

V7.1 当前训练资格明确使用 `ANY_ENDPOINT_40_OF_50`：每个 endpoint 的 valid mask 独立保留，只有实际有效的一侧贡献 loss，不能把单手有效伪装成双手有效。旧的 `BOTH_ENDPOINTS_40_OF_50` 预检仍保留为历史负面证据，但不再作为当前门。重新按当前合同核验后，Chips023/039 分别提供 293/217 个因果 train H50 窗口。

机器容量预检位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_visual_aux_capacity_preflight_v71/RESULT.json
```

它只证明路由容量存在：Chips 潜在 17 train/4 validation，Poker 潜在 20 train/4 validation；其中尚未通过的 Robot、Clean 和 bundle 只能计入 potential，不能计入实际训练 ledger。当前正式 checkpoint 数仍以事实账本为准，容量预检不得被写成“训练数据已就绪”。

### Visual Aux 后的传感器 H3/H4 有界 canary

新传感器 H0/H1/H2 已形成独立终态；H3/H4 的 CPU preflight 对 331 条可读取会话均已通过。旧 `BLOCKED_RESOURCE` 收据记录的是当时 Clean 占用 GPU，不再被解释成输入失败。为避免 106,868 帧在未测吞吐前盲目扩批，增加最后一层自动监督器：

```text
src/chaoyang/ops/run_v71_post_visual_aux_sensor_canaries.py
```

它等待四支 Visual Aux 自动器进入任一明确终态，然后按 CANARY 优先级串行执行：

1. `play_cards_0910_053` 第 90 帧的 FoundationStereo canary；rectification rotation 只从同 session SBS 与 `camera_params.json` 估计，禁止使用 exact78 固定旋转。
2. 同一会话 12 个冻结采样帧的 `SENSOR_ROLE_MASK_V1` canary；白手套、前臂、左右 Controller 和一张目标牌分开，另外两张牌作为负 seed，禁止 union。
3. 根据 canary 实测墙钟计算 106,868 帧的保守全批预算。超过每阶段 12 GPU 小时即发布新的 `BLOCKED_RESOURCE` 收据；自动门失败写 `FAILED_QUALITY_C`；运行故障写 `FAILED_RUNTIME_FINAL`。

无论 canary 结果如何，都不把 1 帧/12 帧证据冒充 331 条 H3/H4 authority。状态文件：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/automatic_canaries_R7_1/AUTOMATION_STATE.json
```

安全恢复命令：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONPATH=. python src/chaoyang/ops/run_v71_post_visual_aux_sensor_canaries.py \
  --poll-seconds 30 \
  --max-wait-seconds 1209600
```
