# 0915 裸手全链与 0916 清洗当前任务

## 当前决定

- 0915 Mask 模型已经由用户确定为 **SAM3.1**，这是本任务唯一可执行的 Mask 权重。
- 不创建 SAM2.1 或 Cutie challenger，不执行胜者选择，也不因历史对比材料改变当前路线。
- 固定执行顺序为：processed-only 审计与物理左目准备 → HaWoR → SAM3.1 → FoundationStereo → Object6D/Clean/Contact/Robot 与 220 会话账本。
- 0916 只做 240 会话数据清洗，不进入 HaWoR、Mask、Depth、Contact 或 Robot。

## 输入与禁止项

- 0915：`/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915`，固定 220 会话、58,686 帧。
- 0916：`/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916`，固定 240 会话。
- 0915 只以物理左目 RGB 上的 HaWoR 作为手部视觉来源；PICO26、controller pose 和 `trackingData` 手部字段保留但不消费。
- 两个源树及现有 0915 processed 发布根均只读；0916 发布到独立的 `processed/chips_cards_handle_highview_0916`。
- `archive/` 不是当前事实源；`_run/current/` 是运行证据，不是算法规范。

## 模型与权重边界

每个算法任务只绑定一个逻辑权重：

1. HaWoR 任务绑定 HaWoR 推理 bundle（模型与必需 detector 作为一个逻辑闭包）。
2. Mask 任务只绑定 `assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt`。
3. Depth 任务只绑定 FoundationStereo checkpoint。
4. 输入准备、CAD、0916 清洗和后处理/Robot 均为 `weights=ABSENT`。

SAM3.1 角色合同区分左右皮肤/前臂、左右手指皮套、左右线缆以及任务物体实例；禁止创建 tracker/controller 角色。任务物体按可见物理实例分离，离屏后保持未知，只有具备重识别证据才恢复旧 ID。

## Robot 与标定边界

- `KAI_HAND固定件.STEP` 是候选几何，不能推出实测安装变换。
- Robot TCP、tool→KaiHand 实测安装变换、camera/world→base 标定仍为 `ABSENT`。
- Robot Visual 只运行会话内相对运动和有限的开发级静态工作空间搜索，使用真实 pinned URDF 做 IK、关节限位及速度/加速度损失统计。
- 所有 Robot 结果必须保持 `control_ground_truth=false`、`physical_deployment_authorized=false`、`calibration_authority=DEVELOPMENT_ONLY`。
- 严格 Contact-aware Robot 在实测标定缺失时 fail-closed，不得用单位阵代替。

## 状态与验收

当前任务在 `task/0915-full-funnel-0916-clean-v1` 分支执行，未推送远端。0916 双 CPU worker 清洗已经提交：240/240 会话终态，222 个 `CLEANED`、18 个 `REJECTED`、0 个运行失败。分任务为 playing_cards 128/2、potato_chips 94/16；拒绝原因为 17 个 `TACTILE_QUALITY` 和 1 个 `VISUAL_OR_TRACKING_CONTENT`。浅层复核见 [`visuals/0916_CLEANING_V1/README_ZH.md`](visuals/0916_CLEANING_V1/README_ZH.md)。

组合输入任务因旧版 v1 自包含审计被终止而如实封为 `FAILED_RUNTIME_FINAL`；这不改变独立 0916 数据集的 `COMMITTED` 状态。0915 将在后继的 v2 processed-only 审计中重新核验，不能沿用或粉饰旧审计结果。

首次六小时不可变进度收据见 [`../../tasks/receipts/0915_0916_FULL_FUNNEL_6H_PROGRESS.json`](../../tasks/receipts/0915_0916_FULL_FUNNEL_6H_PROGRESS.json)。它是时间点快照，不替代最终全阶段账本。

最终必须满足：0915 的 220 个会话在全阶段账本中全部终态；0916 的 240 个会话全部终态且运行失败为 0 才可发布完成状态；浅层可视化、测试、治理、结构、Markdown 链接、SHA、视频完整解码和源树不变性全部通过。
