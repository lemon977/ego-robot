# Human→Robot S2 H3：给外部 AI 的现状与问题

更新时间：2026-09-23 15:12（Asia/Shanghai）。本文用于征求技术意见，不是新的运行收据，不授权修改 H3、质量门或当前 S2 attempt。

## 一、已经确认的事实

### 总体

- 当前任务：`human_to_robot_evidence_unlock_s2_20260923`，仍处于 H3；H6 时间门为 16:45:07。
- 四条正式产品候选均完整解码并通过结构检查；质量通过 `0/4`，正式采用 `0/4`。
- 所有结果仍是 `OFFLINE_VISUAL`，且 `training_eligible=false`、`control_ground_truth=false`、`physical_deployable=false`、`external_metric_authority=false`。

### 装配与显示

- 007/031 当前产品绑定 68.4 mm 修正版完整开发安装合同，不使用旧 75.2 mm 版本。
- STEP 派生的真实连接件网格已进入正式 renderer 的 RGB、光学轴深度和 component ID。
- 031 已验证正式画面由实际 q/FK 驱动；目标腕只作诊断标记，没有被用来单独移动手根。
- 数值 FK 与 renderer 消费的 FK 在浮点误差范围一致。这只证明显示忠实，不证明目标、安装或动作物理正确。
- 连接件只有视觉几何；没有批准且完成覆盖证明的碰撞几何，当前必须保持 `UNVERIFIED_VISUAL_GEOMETRY_ONLY`。

### 031 动作与来源

- 时间轴 149 帧。右侧有 102 帧模型输出（frame 47–148）；左侧没有 ROI、没有模型输出，也没有补值。
- 右侧 102 帧来自非直接 ROI 的 HaWoR 模型预测，属于 `OFFLINE_NONCAUSAL`；这里的 `inferred` 是来源分类，不等同 motion infiller。
- 当前 full-pose 腕目标残差约为 P50 `77.5 mm`、P95 `104.4 mm`。
- position-only 候选能把大多数位置残差压低到约 P95 `4.56 mm`，但旋转很差，且没有完成其所需碰撞资格，因此被拒绝；这个结果不是位置真值证明。
- 旧 two-stage 第二阶段没有把第一阶段位置保留为硬约束，最终退回位置/旋转折中；102 帧中 85 帧出现限位触及。它被拒绝，但这不是全局数学无解证明。
- 目标旋转来自 HaWoR MANO21 掌基的模型派生方向，没有外部姿态真值。位置目标同样来自模型，不自动具有真值权限。
- `P031_WRIST_TARGET_TRACKING` 已使用本轮允许的 2/2 次证据驱动修复；新的部分方向目标不能在 S2 内改名形成第三次重试。

### Scene、Clean 与遮挡

- 031 有 149 帧当前域 Depth 和直接可见物体表面证据；其 H3 固定窗为 frame 66–81。
- 该窗 known decision coverage 为 `82.87%`、UNKNOWN 为 `17.13%`；原始归属切换为 `736/3269`。后者只是原始切换计数，不是 22.51% 错误率，也不能推出 77.49% 正确率。
- 当前没有合法的跨帧同表面对应和冻结时序质量门，因此 031 遮挡质量仍为 `INCONCLUSIVE`。
- 007 Stereo 固定统计只有 `1021` 个 robust matches，低于冻结的 `1500` 门；007及两条0902没有合格 Depth，不能获得实片遮挡结论。
- 007 的 181–196 帧局部 ProPainter 确实被调用，写入区外和保护区改动均为 0；但现有支持是 broad `capture_device` mask，不是一条持续的 cable instance。长细线仍在写入支持外，因此此次结果证明的是支持/身份缺口，不是“正确线缆Mask输入后模型仍失败”。
- 当前没有冻结的、唯一且可拒绝的 top-1 线缆实例证据，所以 S2 不再重跑 Clean。

### Contact / Robot R1

- 当前为 `0 screened / 0 executed / 0 adopted`。
- 这表示筛选没有完成，尚未建立任何可用窗口；不能写成“已经筛遍并证明零可用窗口”。
- 没有合法窗口和固定腕可行性证据时，不运行 Robot R1，也不能用 full-pose 失败反向证明所有局部接触窗口不存在。

### Sensor 与 Local/HuRo

- Sensor 三会话 097/098/101 共 466 帧的原生 HandMotion、共同 target-builder/solver/FK 数组和回放已重载复核；只冻结到相应 `KINEMATIC_ONLY / mixed visual` 范围，没有本轮新拟合。
- Local R0 与 HuRo-derived 已在 007/031 使用相同冻结目标、真实连接件和共同评价集合刷新。
- 没有独立外部真值；可以报告同目标残差、覆盖、限位和失败类型，但不能宣布真实人体精度、实机效果或总体赢家。

### 工程闭合

- 当前签名正式入口 resume：4/4 复用、0 变更；四个重新绑定的视频与 H3 前序视频字节一致，因此不是算法改善。
- S2 收尾测试 `37/37` 通过；H3 前项目本地回归为 1490 项（1489 pass、1 skip），独立碰撞 fixture `6/6`。
- 14 项 hardlink transaction 测试在 CPFS 项目内 TMP 语义下未评价，不能计入通过，也不抹掉其他测试。

## 二、希望外部 AI 回答的问题

请只做技术分析和后继任务建议，不把建议写成已经运行的事实。

### Q1：031 后继 IK 的最小、可证伪合同是什么？

在保持原腕位置为硬约束、固定 68.4 mm 完整安装合同、原限位、已有碰撞范围和手指 q 的前提下，怎样按证据选择掌法线、掌纵向或完整方向？请明确：

- 方向资格怎样在求解前冻结，避免按“哪个轴更容易达到”事后选轴；
- 位置门 `tau_p` 应从哪个现有合同读取，如何确认 5 mm 是否只是损失尺度而非质量门；
- 怎样逐帧验证位置硬约束，而不是只报 P50/P95；
- 哪些结果只能叫 `POSITION_FEASIBILITY_DIAGNOSTIC`，哪些才足以形成新的动作候选；
- 如何证明新方法与旧 two-stage 确实不同，且值得登记一个新的有限 successor。

### Q2：怎样区分目标映射问题、局部求解失败和固定数字模型不可行？

请基于现有 position-only q、full-pose q、限位和当前碰撞范围提出有限检查，不新增全局 IK 平台。需要明确：

- 找到具体 q 能证明什么；
- 局部搜索没找到解不能证明什么；
- 什么证据才允许 `TARGET_FRAME_MAPPING_INVALID`；
- 什么情况下才可能使用 `MODEL_INFEASIBLE_CERTIFIED`，而不是笼统“机器人不可达”。

### Q3：031 遮挡时序应怎样在不引入新模型的前提下评价？

目前只有实际 q/FK、component ID、Robot depth、同域 Scene depth 和相机数据。请给出：

- Robot 表面与 Scene 表面各自的合法跨帧对应方式；
- 如何把 736 次切换拆成真实穿越、边界/显露、几何不确定和不可解释翻转；
- 最小可复验指标及分母；
- 在 H3 已经暴露的开发窗上，怎样避免事后发明通过阈值。

### Q4：007 是否存在值得登记的 top-1 cable successor？

请说明仅凭现有原始 RGB、设备关联、局部颜色/边缘和现有运动证据，能否冻结“一条持续的物理线缆实例”。需要给出：

- 候选生成、唯一性门、领先间隔和拒绝条件；
- 遮挡、出画、自交和保护区冲突的处理；
- 怎样证明是支持/传递缺口，而不是 ProPainter 能力不足；
- 若无法形成可拒绝的唯一实例，是否应正式停止此路线。

### Q5：Contact/R1 应如何完成第一次诚实筛选？

当前是“未筛选”，不是“零窗口”。请给出基于当前有限可见 patch 的漏斗：时间/手有效、Depth 注册、对象身份、可见表面、距离与不确定度、固定腕可行性、求解准入。尤其请说明：

- 031 右手的非直接、离线模型来源能否用于任何开发级几何筛选，允许到什么程度；
- full-pose IK 失败是否应阻塞所有 finger/object 局部候选；
- 如何避免用贴牌拟合或同一接触距离自证对齐；
- 何时应终态为 `NOT_SCREENED`、`NO_ADMISSIBLE_WINDOW_ESTABLISHED`、`INFEASIBLE` 或 `SOLVER_REJECTED`。

### Q6：连接件碰撞最小可行范围是什么？

请先判断现有碰撞后端是否能直接消费真实 STEP 派生网格；若只能使用批准的保守包络，请说明：

- 如何证明包络覆盖原网格；
- 转接环孔洞造成凸包假阳性时怎样报告；
- 哪些非邻接对必须检查；
- 怎样避免把逐帧无碰撞、受控 fixture 或视觉网格误写成整机碰撞通过。

### Q7：下一个有限 successor 应优先解决哪一个问题？

请在以下方向中排序，并说明信息增益、最小输入、运行成本、停止条件及对产品的真实影响：

1. 031 位置硬约束＋分级方向 IK；
2. 007 top-1 cable instance 与一次小窗 Clean；
3. 031 遮挡时序对应与归因；
4. 031 Contact/R1 首次筛选；
5. 连接件碰撞覆盖。

不要建议同时启动全部方向，也不要通过降低已有质量门来制造成功。

## 三、不可更改的边界

- H3 机器结果、旧 attempt、分母和质量门保持不变。
- S2 H6 前不新增全片 HaWoR、Clean、IK、Depth 或 HuRo；当前 S2 不再启动第三个 031 IK 或无 top-1 证据的 007 Clean。
- 不下载新模型、不训练、不使用 Clean 作为几何、不补造外部标定/碰撞资产/Contact 真值。
- 新建议必须说明是假设、需要什么新证据，以及若失败如何有限终止。

## 四、建议优先阅读的项目证据

- [H3_RESULT](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/checkpoints/H3_RESULT.json)
- [H6_READINESS](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/h6_readiness/attempt_0001/RESULT.json)
- [H3 外部复核后的执行决定](../../HUMAN_TO_ROBOT_S2_H3_REVIEW_DECISION_ZH.md)
- [S2 视频索引](INDEX_ZH.md)
- [031 来源旁车](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/source_provenance_031/attempt_0001/RESULT.json)
- [031 two-stage 结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/two_stage_031/attempt_0001/RESULT.json)
- [031 腕目标权限](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/target_authority_031/attempt_0001/RESULT.json)
- [031 遮挡结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json)
- [007 Clean canary](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json)

返回建议时，请把“已由当前证据证明”“合理推断”“需要新增实验才能判断”三类结论分开。
