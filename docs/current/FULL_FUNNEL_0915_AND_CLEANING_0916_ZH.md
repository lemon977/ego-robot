# 0915 裸手全链与 0916 清洗当前任务

## 当前决定

- **0915 全批仍停止，单会话证据链已进入下一轮 bounded 优化。** 单样本复核暴露 VST 图像域疑点：旧输入
  对 SBS 物理左目额外执行了 `equiDis62 → pinhole` 重映射。用户已经确认
  `sourceIndex=1 + resize-only`，并接受该输入上的 HaWoR 单样本结果；同一会话
  SAM3.1 严格角色 canary 已完成且用户接受其作为下一轮起点，但弱角色没有质量通过；这不恢复 220 会话任务，详见
  [`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。
- 单会话 A/B 已以 `BLOCKED_EXTERNAL` 封账：当前 remap 的输出位移 P50 为
  102.24 px、P95 为 241.02 px；legacy processed 单目来自 SBS `sourceIndex=0`
  物理右目。浅层证据见
  [`visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。
- 用户已确认物理左目 `sourceIndex=1 + resize-only` 为正确单目画面，并确认单会话
  HaWoR 视觉结果无明显问题。双目 Depth 的矫正方式继续为 `NOT_EVALUATED`；
  `play_cards_0915_001` 的 SAM3.1 Mask canary 已运行并完成本轮人工判断。
- resize-only raw HaWoR canary 的直接观测为左手 148/150、右手 145/150；后继
  `hawor_bounded_v2` 通过冻结数值门，短缺口连续性层通过单样本人工复核。相关临时
  运行与可视化已经带收据清理，未自动注册 220 会话后继。
- 0915 Mask 模型已经由用户确定为 **SAM3.1**，这是本任务唯一可执行的 Mask 权重。
- 不创建 SAM2.1 或 Cutie challenger，不执行胜者选择，也不因历史对比材料改变当前路线。
- 原定全批执行顺序仍失效，不得从已取消的 HaWoR 任务续跑。Mask 已从新的、独立的
  `play_cards_0915_001` SAM3.1 单会话 canary 任务开始并封账；浅层复核见
  [`visuals/0915_SAM31_STRICT_ROLE_CANARY_V1/README_ZH.md`](visuals/0915_SAM31_STRICT_ROLE_CANARY_V1/README_ZH.md)。
- 0916 只做 240 会话数据清洗，不进入 HaWoR、Mask、Depth、Contact 或 Robot。

## 当前 bounded 执行 DAG

当前治理只允许一个 `next_task`，因此执行路由保持串行；这不等于算法依赖串行：

1. `0915_stereo_interaction_cpu_canary_v1`：同一 coordinator 下并行运行两个独立 writer
   lane。Stereo lane 做 raw-resize 与 rectified 域预检；Interaction lane 只做 2D
   adjacency/approach/co-motion 和可选触觉支持。
2. `0915_sam31_weak_role_canary_v1`：只绑定 SAM3.1，算法上不消费 Stereo/Interaction
   结果；目标仅为双前臂、逐个可见 sleeve、黄色线缆和 `playing_card_02`，v5 双手及
   `card00/card01` 作为 SHA 守卫的只读回归输入。
3. `0915_removal_envelope_single_session_canary_v1`：`weights=ABSENT` 的 CPU-only
   单样本任务；只读封存 SAM semantic、bounded HaWoR 与 resize-only RGB，独立产生
   MANO capsule、palm/wrist/forearm corridor、可替换 appearance profile 的 cable
   region、逐像素来源位图和 feather alpha。不重跑 SAM，不运行 inpaint；结果必须独立
   人工验收，并禁止反哺几何证据链。
4. `0915_foundationstereo_single_session_canary_v1`：只有 Stereo lane 内部 admission
   通过才可注册。输出为 rectified-left optical-Z，单位米，尺度源为 rectified `fx ×`
   same-session factory baseline；外部精度保持 `UNVERIFIED`。
5. `0915_planar_object6d_single_session_canary_v1`：先只处理稳定的 `card00/card01`，分别
   输出 translation、plane normal、in-plane rotation 的可观测性及平面残差；不生成统一
   置信度，不补遮挡姿态。

SAM 弱角色与 FoundationStereo 是独立 GPU 任务并通过中央租约串行；不通过扩展多
`next_task` 来制造伪并行。任何 bounded 结果都不自动授权全批。

第 1 个任务已经完成：两个 CPU lane 都有独立 writer fence、运行签名与终态，合并结果
为 `PASSED`。Stereo lane 为 `PASS_GPU_DEPTH_ADMISSION`；raw resize-only 垂直误差
median/P90/P95 为 1.955/2.804/3.121 px，单会话 rectified 候选为
0.628/1.901/3.881 px。该 PASS 是池化 admission；frame 81/94 仍有逐帧尾部超门，
后继 Depth 必须单独置 invalid。Interaction v0a 在 4500 个指尖—物体配对中记录 59 个 2D
邻接、1378 个 2D 接近、804 个 2D 共动与 16 个触觉支持假设；这些均为开发级弱证据。
浅层复核见
[`visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1/README_ZH.md`](visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1/README_ZH.md)。

第 2 个任务也已完成运行并发布完整 150 帧审阅视频。v5 左右手与前两张牌只读回归门
保持字节一致；新前臂有证据 92/150、90/150，第三张牌 115/150，右线缆 62/150，
各皮套与左线缆仅 6–8/150。运行终态为 `PASSED`，但用户人工视觉验收已因 mask 闪烁、
黄色线缆和皮套未被可靠选中而判为 `REJECTED_QUALITY_AS_CLEAN_BASELINE`；弱角色不能
据此进入自动 Clean 或全批。后继设计改为分离 Raw Candidate / Semantic / Removal /
Feather：当前封存运行未保存 raw candidate 像素，必须显式记为缺失；动态 MANO 各向异性
包络和 cable appearance tracker 只可补齐 Clean removal，不得改写 SAM 原始证据或进入
Depth/Object6D/Contact/Robot geometry。实现与独立验收边界见
[`REMOVAL_ENVELOPE_V1_ZH.md`](REMOVAL_ENVELOPE_V1_ZH.md)。浅层复核见
[`visuals/0915_SAM31_WEAK_ROLE_CANARY_V1/README_ZH.md`](visuals/0915_SAM31_WEAK_ROLE_CANARY_V1/README_ZH.md)。

第 3 个任务已经完成 CPU 执行并生成 150 帧四面板视频。自动门通过：293 个直接观测
side-frame、7 个双端验证内部 hold、0 个 geometry unknown，direct MANO21 关节覆盖 1.0，
source bits 逐像素闭合，protected visible-object core 与最终 removal 零交集。Semantic
质量仍保持 `REJECTED_QUALITY_AS_CLEAN_BASELINE`；Removal 独立为
`AWAITING_USER_VISUAL_REVIEW`。当前 cable appearance profile 在 150 帧均有候选，靠近
卡牌黄色图案的少量区域可能是假阳性，所以未运行 inpaint，也未授权扩批。复核见
[`visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md`](visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md)。

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

单会话执行结果为：左右手 145/150、148/150 帧有证据；三张牌 143/150、146/150、
95/150；前臂、皮套和线缆没有稳定过门。每帧保存 `seeded/tracked/reseeded/unknown`，
当前任务 `PASSED` 只表示执行与证据发布完成，不表示所有角色质量通过。

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
