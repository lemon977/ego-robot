# 启动入口：0915 Robot 质量恢复与跨会话验证，第二轮 15 小时

版本：v2.1。编制日期：2026-09-19。
本文件是交给本地执行 AI 的启动指令，不表示 ChatGPT 已在远端注册任务。
配套：[执行任务书](01_EXECUTION_PLAN_15H_ZH.md) · [算法审计视图](02_ALGORITHM_AUDIT_LIVE_ZH.md) · [任务字段模板](03_TASK_MANIFEST.template.json) · [验收测试](04_ACCEPTANCE_TESTS_ZH.md)。

## 可直接交给本地 AI 的授权文字

按本包启动新一轮 15 小时实际执行，不续跑上一轮已封账任务，不仅生成计划后退出。

1. 范围仅为 `/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`。固定分组为 W0 四条、W1-DIAG 两条、W1-ADOPTION 两条、H9 FINAL HOLDOUT 四条；另在 H0 封存 EXTRA FINAL 四条，只有 H11 扩批门通过后才能打开。0916 仅清洗，完全不进入本轮下游，不能自动全量处理 220 条。
2. 用户最新报告起点为治理 `12362 / gov-012362-8d04bae5bef9`，分支 `task/0915-full-funnel-0916-clean-v1`，47 节点终态、无活动任务。启动时核实实际 HEAD、receipt、资源与任务索引；最新 HEAD 尚未在聊天中提供，不得沿用旧 `4228645` 充当当前提交。
3. 先登记一个父协调任务和有限工作包。三条 CPU lane 隔离写入，唯一 publisher 更新共享合同/任务指针；GPU 单租约串行。沿用现有调度和状态枚举，不搭第二套治理框架。
4. 第一小时内并行启动 P0、A0、B0、C0。P0 最多45分钟，仅阻塞新R2；旧合同找不到就发布tombstone，不能阻塞HaWoR、SAM、Depth、Object、R0或Contact audit。
5. 本轮区分模型运行许可、字段/窗口消费许可、整会话严格质量、物理部署资格。旧拒绝不抹除；不得放宽失败门凑 PASS，也不得把整会话严格失败机械地传播到所有独立消费者。
6. 允许 W1-DIAG 的 `play_cards_0915_044` 与 `get_potato_chips_0915_097` 运行 HaWoR 诊断推理；其运行不再以 W0 整会话 4/4 通过为前提。打开 W1-ADOPTION 的 `play_cards_0915_106` 与 `get_potato_chips_0915_029` 前，必须冻结候选代码、配置、阈值和输入 SHA；打开后只能 ADOPT、REJECT 或同签名故障重试，不得按结果返修。
7. SAM 任务物体从合法 RGB、任务元数据和合法 text/visual prompt 独立初始化。手部可使用同帧合格 2D 提示或 SAM 自身候选，不以 HaWoR 整会话严格 3D 通过为统一前置条件。身份不确定必须保留。
8. 复用已完成的 12 会话、3,390 帧 encoded Depth 缓存。先把已有 Depth 用到 Object/Interaction，不优先给剩余库存再跑深度。只允许有明确局部失败依据的固定模型小型对照重算，原缓存不覆盖。
9. 对 Contact 增加“可观测性与门是否可满足”审计。严格 5 mm 门不改；可见表面不等于隐藏接触面。只有手关联可见表面与物体表面来自同帧同一 encoded Stereo depth、同一 physical-left/K/P/baseline/Depth SHA，且身份、有限patch、LR一致性、注册与不确定度闭合时，才可写 `local_stereo_metric_dev=true` 并允许开发级 R1-E；`external_metric_authority/control_ground_truth/physical_deployment_authorized` 始终为 false。不得用 Contact 窗口反向拟合 alignment 自证。
10. R0 主线优先恢复真实质量；R2 只对通过自身用途准入的 R0 窗口做限量虚拟 IK，不等待 R1，但不能再次对不合格整片盲批。实测 TCP/mount/world→base 仍缺失，全部 NON_CONTROL / NON_DEPLOYABLE。
11. 不等待用户提供尺寸、标定、新录制或人工标注。优先用现有来源文件建立 source_group；独立性不明时明确限制泛化声明。曾用于调参/审计的样本不再作为未见测试。
12. 不换模型、权重、环境或驱动；不启动训练/RL；冻结机器人几何；不重开 Removal V1/V2、Clean/inpaint、线缆或皮套专项。正常 RGB 与独立机器人场景并排展示即可。
13. T0 在本地收到本次启动指令并发生首次执行性写入时记录一次，15 小时包含文档修订、注册、实验与收尾，不因失败或 successor 重置。H9 冻结所有候选；只对已采用能力打开四条 H9 FINAL HOLDOUT。H11 同时满足扩批门才打开 H0 封存的 EXTRA FINAL；H13.5 停止新会话，H15 前收尾。
14. 每十分钟在 `_run/current` 写事实快照；受治理 Markdown 只在里程碑更新。技术合同按本次运行冻结 exact bytes/SHA，治理版本刷新不得改动已经运行的技术合同。
15. 本地提交，不 push；只终止本轮所属进程。不得 reset/clean 用户改动或删除失败证据来换 Git clean。最后分别报告推理、导出、严格通过、局部可用窗口、R1-E/R1-H/R2 和实际算法改善。
16. 来源独立性只写 `METADATA_VERIFIED_NO_USER_ATTESTATION`：唯一 VST/QPC SHA、不重叠 QPC、唯一采集时间和无跨片段合并是当前依据；旧文件中的用户逐会话确认声明不再作为 V2.1 authority。

除发现不可消解的当前任务冲突、源数据损坏/越权、范围外破坏性操作、付费或真机控制外，无需再逐项询问。遇到局部 blocker，按任务书转入独立就绪工作。

## 本地执行 AI 第一条回执应包含

真实父任务 ID、实际 HEAD/治理 revision、T0/截止时间、冻结 12 会话 manifest 的路径/SHA、三个 lane 的实际状态、GPU 当前所属者、下一条真实已注册 operation 的调用。不能以“模板仍 dispatch=false”为理由停留在规划：模板本就不能直接调度，应转换为现有合法任务包后执行。

## 范围冲突优先级

用户本轮最新状态与明确限制 > 本包注册后的正式边界 > 旧 15 小时计划。
旧 v1 中任何可能纳入 0916、必须整波全部严格 PASS 才允许诊断推理、每十分钟改治理 Markdown 的表述，本轮均不沿用。历史结果和失败收据本身保持不变。
