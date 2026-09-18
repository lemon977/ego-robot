# 0915 裸手全链与 0916 清洗当前任务

## 当前决定

- **0915 全批仍停止，单会话 Mask 可继续。** 单样本复核暴露 VST 图像域疑点：旧输入
  对 SBS 物理左目额外执行了 `equiDis62 → pinhole` 重映射。用户已经确认
  `sourceIndex=1 + resize-only`，并接受该输入上的 HaWoR 单样本结果；当前只解除同一
  会话 SAM3.1 Mask canary，不恢复 220 会话任务，详见
  [`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。
- 单会话 A/B 已以 `BLOCKED_EXTERNAL` 封账：当前 remap 的输出位移 P50 为
  102.24 px、P95 为 241.02 px；legacy processed 单目来自 SBS `sourceIndex=0`
  物理右目。浅层证据见
  [`visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。
- 用户已确认物理左目 `sourceIndex=1 + resize-only` 为正确单目画面，并确认单会话
  HaWoR 视觉结果无明显问题。双目 Depth 的矫正方式继续为 `NOT_EVALUATED`；当前只
  授权 `play_cards_0915_001` 的 SAM3.1 Mask canary。
- resize-only raw HaWoR canary 的直接观测为左手 148/150、右手 145/150；后继
  `hawor_bounded_v2` 通过冻结数值门，短缺口连续性层通过单样本人工复核。相关临时
  运行与可视化已经带收据清理，未自动注册 220 会话后继。
- 0915 Mask 模型已经由用户确定为 **SAM3.1**，这是本任务唯一可执行的 Mask 权重。
- 不创建 SAM2.1 或 Cutie challenger，不执行胜者选择，也不因历史对比材料改变当前路线。
- 原定全批执行顺序仍失效，不得从已取消的 HaWoR 任务续跑。Mask 必须从新的、独立的
  `play_cards_0915_001` SAM3.1 单会话 canary 任务开始。
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

0915 `0915_hawor_full_v1` 已按用户要求取消，不是失败后可自动恢复的任务。停止时旧
重映射输入上记录 120 个已触达会话（119 个旧质量 C、1 个停止造成的运行失败）；
这些数字不评价正确 VST 图像域上的 HaWoR。终止收据见
[`0915_HAWOR_FULL_V1_USER_STOP.json`](../../tasks/receipts/0915_HAWOR_FULL_V1_USER_STOP.json)。

现有 `play_cards_0915_001` HaWoR/SAM3.1/Depth/Robot 可视化已经降级为问题复现证据，
不再是 0915 基线，也不能支持此前“相机参数没有问题”的结论。

组合输入任务因旧版 v1 自包含审计被终止而如实封为 `FAILED_RUNTIME_FINAL`；这不改变独立 0916 数据集的 `COMMITTED` 状态。0915 将在后继的 v2 processed-only 审计中重新核验，不能沿用或粉饰旧审计结果。

首次六小时不可变进度收据见 [`../../tasks/receipts/0915_0916_FULL_FUNNEL_6H_PROGRESS.json`](../../tasks/receipts/0915_0916_FULL_FUNNEL_6H_PROGRESS.json)。它是时间点快照，不替代最终全阶段账本。

未来若重启 0915 全批，仍必须满足 220 个会话在全阶段账本中全部终态；当前单会话
SAM3.1 Mask canary 不构成全批授权。0916 的 240 个会话已满足全部终态且运行失败为 0。
