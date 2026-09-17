# exact78 转换率问题与通用优化任务

本文件是整改任务说明，不维护实时进度。实时数量只读 `CURRENT_STATUS_RECEIPT.json` 和 `CURRENT_PROJECT_STATUS_MIN.json`。

## 已证实的损失来源

第一份互斥首阻塞漏斗位于：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_conversion_funnel_v2/CONVERSION_FUNNEL.json`

在其冻结快照中，156 条 Raw 首次被阻塞于：HaWoR 12、Role Mask 20、Object Identity Mask 23、三路已过但同会话标定缺失 43；只有 58 条进入公制几何层。换算为每 100 条 Raw，约有 92 条 HaWoR A/B、79 条同时通过 Role、65 条三路共同 A/B、37 条 metric-ready。它说明低转换率来自串联硬门，而不是某一个 Robot 模型单独丢掉九成数据。

Clean 曾出现的 19 条失败已经证明是当前代码 import 被清理破坏造成的运行失败，不是数据或 ProPainter 质量 C。它们正在由 fresh/no-clobber successor 恢复，不能计入算法自然淘汰率。

Robot 已封账 C 的聚类证据位于：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_robot_failure_clusters_v1/ROBOT_FAILURE_CLUSTERS.json`

当前失败同时包含 arm 可达性和 hand fingertip direction。两者必须分开修复。Chips066 的右小指失败集中在帧 40–59，最大方向误差约 15.96°，且相关关节已经打到冻结限位；全局继续加优化权重不能解决这个形态/基坐标可行性问题。

Robot 数值通过也不等于能贡献训练窗口。首轮 H50 实测位于：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_h50_preflight_v55/PREFLIGHT.json`

该冻结三条样本中，Chips023、Chips039 的 physical-left hand root 全片投影在 1280×960 画外，均为 0 个 H50 窗口；Poker245 有 18 个窗口，但属于 heldout，不能移动到 train。这个结果只说明三条候选的投影可用性，不外推成全部 Robot 的最终通过率。

Pose-only Visual 通路也已用真实会话验证：Chips050（固定 test split）生成 131 个 H50 窗口，两个 RGB 分支均通过有限梯度的 forward/backward smoke。第一次构建曾因把源图最后一个像素按 `WIDTH/source_width` 映射而得到略大于 1 的 normalized coordinate，被 validator 正确拒绝；当前实现改用闭区间映射 `(WIDTH-1)/(source_width-1)` 并新增边界单测。这个修复消除的是标签实现假损失，不改变 Robot、H50 或 split 的质量门。

因此训练前损失必须继续拆成三类：Robot 几何失败、投影/双手 H50 覆盖不足、固定 split 中 train/validation 数量不足。只有第三类满足后才能训练，不能用 test/heldout canary 补分母。

对当前 8 条 Pose-only 候选的全量轻量预检位于：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_batch_preflight_v55/PREFLIGHT.json`

结果为 6 条 READY、2 条零窗口，共 520 个 H50 窗口；但 6 条 READY 全部属于 test/heldout。仅有的两条 train（Chips023/039）恰好都因一侧手根全片画外而为零窗口。因此“已有 8 条 Robot candidate”对四支训练的当前 train/validation 贡献仍是 0，不能只看 Robot candidate 总数估算 checkpoint 就绪率。当前 Robot 扩批必须优先按冻结 split 产生 train/validation 会话，并在昂贵的 RGB 渲染前先跑 H50 预检。

## 不破坏质量门的优化顺序

1. 先恢复基础设施假损失：完成 19 条 Clean successor，运行失败与质量 C 分账。
2. 分开视觉和公制资格：58 条可做 metric geometry；另外 43 条三路共同 A/B 即使缺标定，也可做 Visual Clean 与 pose-only Robot，但 contact 帧必须 invalid，不能借邻会话标定。
3. HaWoR、Role Mask、Object Mask 分簇 successor；每簇固定一条失败 canary、两条旧 A/B regression、最多两轮，不做逐会话手工补丁。
4. Robot arm 先优化全片固定 base placement、双侧可达余量和观察域覆盖；hand 再做机器人形态归一化、逐指关节基映射和限位可行性。任何新映射必须跨 Chips/Poker regression，不降低 15° 等既有门。
5. 每条 Robot 数值结果先运行 `preflight_visual_aux_h50_candidates_v55.py`；零窗口会话停止在训练前，避免生成数百张无用 PNG。
6. Robotized RGB 使用合法 Clean 与 Robot render；未解决的 Object/Robot overlap 在 Raw/Robotized 两支使用同一 `training_valid_mask=0`，不伪造遮挡像素。
7. 固定 split 下分别统计 train/validation 的 session 与窗口。Poker Wave0 理论 session 不足时，只能使用预先冻结的 Visual Wave1 6 train + 2 validation，不能根据训练结果事后挑选。
8. 四支 checkpoint 仅在四个真实 epoch-0 和每任务 16/3 session、256/48 H50 窗口门都通过后串行训练；训练目标仅 future-2D，数字 `q_arm/q_hand` 不是控制真值。

## 需要实现和验收的通用 successor

| 子问题 | 通用修改 | 通过条件 | 失败退路 |
|---|---|---|---|
| Clean 假失败 | 恢复固定 import 与运行签名 | 19 条全进明确终态 | runtime final，不冒充质量 C |
| Role Mask | HaWoR 几何实例提示、离屏重捕获 | canary 通过且两条旧 A/B 不退化 | 两轮后冻结 C |
| Object Mask | action-conditioned identity、遮挡 UNKNOWN | 目标实例不漂移、不 union | 两轮后冻结 C |
| Arm | 全片固定 placement 的双侧可达/画面覆盖联合预检 | 原数值门通过且 H50 覆盖增加 | 保持 C 或 pose-only blocked |
| Hand | 固定机器人形态映射、逐指限位可行性 | Chips/Poker regression 均不退化 | 保持 C，不放宽角度门 |
| Occlusion | goldset + known accuracy/coverage 双门 | accuracy≥95% 且 coverage≥70% | UNKNOWN 并双支排除 |

## Robot 当前扩批暴露的新结构性问题

当前 train 批次 Chips083/087/090 的 method-1 placement 扫描进一步说明：只扫描一个 `task_base_backoff_m` 标量不够通用。已完成的候选中，083 仍有约 200 以上 failed side-frame rows，090 最佳候选仍有 21 行失败；087 最接近，但最佳已见候选仍有 17 行失败。多条候选的速度与加速度恰好触及冻结上限，而位置/旋转残差仍超门，说明问题不是简单把机器人整体向前或向后平移即可解决。

下一版通用 arm placement successor 应冻结同一 session 的单个刚体 base，联合搜索有限的 XYZ 平移与 yaw（不能逐帧移动 base），目标按以下顺序字典序优化：

1. 所有硬约束可行且 failed rows 为 0；
2. 最差帧关节余量与 position/rotation residual；
3. 左右手根的画面内覆盖与 H50 窗口数；
4. 与当前 accepted task template 的最小偏移。

必须在 Chips/Poker 固定 regression 上验证，并保持 world-first、base session-constant 和既有速度/加速度门。若二维/三维小范围刚体搜索仍不可行，就应冻结为 Robot C，而不是逐帧移动 base 或降低阈值。

## 本轮禁止的“提转换率”方式

- 不把旧失败改名为 PASS。
- 不降低 Mask、IK、hand 或 H50 门。
- 不把一只手永久画外的会话计作双手 H50。
- 不把 test/heldout 移进 train/validation。
- 不把 Clean 补画像素用于 Depth/Object6D/contact 几何。
- 不把 pose-only trajectory 称作 real Robot action。
- 不把当前小样本候选比例写成模型总体精度。

最终应同时发布“每阶段留存率”和“首阻塞原因”。只有这样，100 条到 Robot 少于 10 条时，才能知道损失来自数据资格、运行故障、几何质量还是训练窗口，而不是把所有原因混成一个通过率。
