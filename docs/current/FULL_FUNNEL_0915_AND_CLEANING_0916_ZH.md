# 0915 裸手全链与 0916 清洗当前任务

## 当前决定

- **0915 全批仍停止。** 用户已确认所有 VST 编码视频本来没有畸变；旧输入对 SBS
  物理眼额外执行的 `equiDis62 → pinhole` 是重复 lens-undistortion。当前统一边界为
  sourceIndex-aware crop + resize-only。用户已接受该输入上的 HaWoR 单样本结果；同一会话
  SAM3.1 严格角色 canary 已完成且用户接受其作为下一轮起点，但弱角色没有质量通过；这不恢复 220 会话任务，详见
  [`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。
- 单会话 A/B 已以 `BLOCKED_EXTERNAL` 封账：当前 remap 的输出位移 P50 为
  102.24 px、P95 为 241.02 px；legacy processed 单目来自 SBS `sourceIndex=0`
  物理右目。浅层证据见
  [`visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。
- 用户已确认所有 VST 编码视频已经无畸变，并确认单会话 HaWoR 视觉结果无明显问题。
  双目 Depth 的编码域标定继续为 `NOT_EVALUATED`；
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

## 已封账 bounded 执行 DAG

当前治理只允许一个 `next_task`，因此执行路由保持串行；这不等于算法依赖串行：

1. `0915_stereo_interaction_cpu_canary_v1`：同一 coordinator 下并行运行两个独立 writer
   lane。Stereo lane 曾做 raw-resize 与错误的 `equiDis62` rectified 域预检；Interaction lane 只做 2D
   adjacency/approach/co-motion 和可选触觉支持。
2. `0915_sam31_weak_role_canary_v1`：只绑定 SAM3.1，算法上不消费 Stereo/Interaction
   结果；目标仅为双前臂、逐个可见 sleeve、黄色线缆和 `playing_card_02`，v5 双手及
   `card00/card01` 作为 SHA 守卫的只读回归输入。
3. `0915_removal_envelope_single_session_canary_v1`：`weights=ABSENT` 的 CPU-only
   单样本任务；只读封存 SAM semantic、bounded HaWoR 与 resize-only RGB，独立产生
   MANO capsule、palm/wrist/forearm corridor、可替换 appearance profile 的 cable
   region、逐像素来源位图和 feather alpha。不重跑 SAM，不运行 inpaint；结果必须独立
   人工验收，并禁止反哺几何证据链。
4. `0915_foundationstereo_single_session_canary_v1`：历史上由随后撤销的 Stereo admission
   注册；其 rectified-left optical-Z 来自重复 lens-undistortion，当前为错误图像域证据，
   不得注册或复活。
5. `0915_planar_object6d_single_session_canary_v1`：先只处理稳定的 `card00/card01`，分别
   输出 translation、plane normal、in-plane rotation 的可观测性及平面残差；不生成统一
   置信度，不补遮挡姿态。

SAM 弱角色与 FoundationStereo 是独立 GPU 任务并通过中央租约串行；不通过扩展多
`next_task` 来制造伪并行。任何 bounded 结果都不自动授权全批。

第 1 个任务已经完成：两个 CPU lane 都有独立 writer fence、运行签名与终态，合并执行
结果为 `PASSED`。Stereo lane 曾发布 `PASS_GPU_DEPTH_ADMISSION`，该 admission 因重复
lens-undistortion 已撤销；raw resize-only 垂直误差
median/P90/P95 为 1.955/2.804/3.121 px，单会话 rectified 候选为
0.628/1.901/3.881 px，但后者只作为失败实验数值保留；frame 81/94 还有逐帧尾部超门。
Interaction v0a 在 4500 个指尖—物体配对中记录 59 个 2D
邻接、1378 个 2D 接近、804 个 2D 共动与 16 个触觉支持假设；这些均为开发级弱证据。
浅层复核见
[`visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1/README_ZH.md`](visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1/README_ZH.md)。

第 2 个任务也已完成运行并发布完整 150 帧审阅视频。v5 左右手与前两张牌只读回归门
保持字节一致；新前臂有证据 92/150、90/150，第三张牌 115/150，右线缆 62/150，
各皮套与左线缆仅 6–8/150。运行终态为 `PASSED`，但用户人工视觉验收已因 mask 闪烁、
黄色线缆和皮套未被可靠选中而判为 `REJECTED_QUALITY_AS_CLEAN_BASELINE`；弱角色不能
据此进入自动 Clean 或全批。后继设计分离 Raw Candidate / Semantic / Removal /
Feather。当前封存运行未保存 raw candidate 像素，必须显式记为缺失。V1 的动态 MANO
包络、wrist→边界 corridor 和宽松 appearance union 已被人工复核否决；不能继续调半径。
V2 以 SAM foreground 为主体，弱先验只允许局部 repair，不得改写 SAM 原始证据或进入
Depth/Object6D/Contact/Robot geometry。V1 失败边界见
[`REMOVAL_ENVELOPE_V1_ZH.md`](REMOVAL_ENVELOPE_V1_ZH.md)，V2 合同见
[`REMOVAL_ENVELOPE_V2_ZH.md`](REMOVAL_ENVELOPE_V2_ZH.md)。浅层复核见
[`visuals/0915_SAM31_WEAK_ROLE_CANARY_V1/README_ZH.md`](visuals/0915_SAM31_WEAK_ROLE_CANARY_V1/README_ZH.md)。

第 3 个任务已经完成 CPU 执行并生成 150 帧四面板视频。自动门通过：293 个直接观测
side-frame、7 个双端验证内部 hold、0 个 geometry unknown，direct MANO21 关节覆盖 1.0，
source bits 逐像素闭合，protected visible-object core 与最终 removal 零交集。Semantic
质量仍保持 `REJECTED_QUALITY_AS_CLEAN_BASELINE`；Removal 人工质量终态也已判为
`REJECTED_QUALITY`。V1 将 MANO 中心线、wrist→边界前臂猜测和宽松黄色 appearance 候选
直接 union 为强擦除区域；150/150 帧都有 cable 候选及第 60 帧 34 个候选连通域证明该
分支没有形成可靠实例身份。V1 只作为失败实验封存，不调参重试、运行 inpaint 或扩批。
V2 核心、schema、配置与 16 项合成合同测试已实现，真实视频 Canary 尚未运行；它需要
真实 foreground proposal、stable-background QA 与 cable reverse-pass，缺失时 fail-closed。
FoundationStereo/Object6D 与 Clean 解耦。V1 复核见
[`visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md`](visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md)。

第 4 个任务曾完成 150 帧 FoundationStereo GPU canary。模型单次加载、300 次双向
推理、SBS 与审阅视频完整解码均通过，GPU 租约正常释放；不可变执行终态为
`REJECTED_QUALITY`。用户随后确认全部 VST 编码视频已经无畸变，而该任务再次应用了
`equiDis62`，所以当前证据 authority 为 `WITHDRAWN_WRONG_IMAGE_DOMAIN`。其失败门为
左右一致性残差跨帧 P90 `17.8653 px > 5 px`，以及帧级
深度中位数步长 P90 `0.4290 m > 0.35 m`；其余 6 个质量门通过。该结果不授权
`VISUAL_OBJECT6D_CANDIDATE_INPUT`，因此第 5 个 Planar Object6D 任务没有启动。浅层
视频与数值见
[`visuals/0915_FOUNDATIONSTEREO_CANARY_V1/README_ZH.md`](visuals/0915_FOUNDATIONSTEREO_CANARY_V1/README_ZH.md)。

新的 Depth 任务必须以
[`VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json`](../../tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json)
为硬前置：解码视频不得使用任何 lens-undistortion。若需 stereo epipolar alignment，必须
另行固定编码域标定并先做 raw resize-only 对照；旧 FoundationStereo 任务不得复活。
同一复核已覆盖旧 exact78 worker：它也对解码 VST 像素执行 `make_map/remap_pair`。当前
Depth 与 Object6D 的原 `58 passed` 均已撤回，账本各记为 `0 passed / 58 blocked`；相关
Contact/Occlusion 假设不再拥有当前消费资格。历史产物未删除，也不能作为新任务输入。

后继 `0915_stereo_encoded_domain_preflight_v1` 已按正确图像域完成全部 150 帧 CPU
复核：只按 `sourceIndex` 裁切左右眼再 resize，不做镜头去畸变或 remap。34,280 个
robust matches 的 `|dy|` median/P90/P95 为 `1.9432/2.8167/3.0964 px`，frame 81/94
无单帧异常，结论为 `PASS_DIRECT_FOUNDATION_INPUT`。这只解除“是否需要额外极线 remap”
的问题，不产生 Depth。其 GPU successor 随后固定物理左右眼身份，对两眼同时水平镜像
以适配正视差模型，并在推理后把输出反镜像回原物理左目。该 FoundationStereo canary
已完成 150 帧、单次加载和 300 次双向推理；全部内部门通过，RGB/Depth 像素回域误差
为 0。浅层入口见
[`visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1/README_ZH.md`](visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1/README_ZH.md)。
外部精度仍为 `UNVERIFIED`，只允许同会话新 Object6D canary 消费。

新的 Object6D 合同固定三张牌为三个独立物理实例，`black_card_tray` 为独立 support
entity，`card_set` 只表达语义成员关系而不拥有共同刚体姿态。`center_xyz`、plane
normal、in-plane rotation 和 full extent 分别报告可观测性；牌尺寸尚未实测时
full extent 保持 `UNOBSERVABLE`，不得补造隐藏中心或完整 6DoF。
该 canary 已完成：三张牌的 visible-surface center 分别为 `143/146/95` 帧，plane
normal 分别为 `139/113/48` 帧；第三张牌遮挡较多时保持 unknown。时间轴见
[`visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2/README_ZH.md`](visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2/README_ZH.md)。
这只是字段级内部可观测性，不是外部 pose accuracy。后继单样本开发链已完成：
Object6D QA 通过，但非接触 Human/Stereo 对齐的 hold-out P90 为 `36.79 mm`，未过
`30 mm` 门；最近有限 object patch 距离为 `6.60 mm`，没有同一 hand/finger/object
连续五帧的 Contact 窗口。因此 Kai22 R0 已交付，R1 为 `BLOCKED_LOCAL_EVIDENCE`，R2
未运行。完整口径见
[`INTERACTION_CONTACT_ROBOT_DEV_V1_ZH.md`](INTERACTION_CONTACT_ROBOT_DEV_V1_ZH.md)。

`0915_removal_envelope_v2_real_canary_v1` 也已在 150 帧实片上运行，终态
`REJECTED_QUALITY`。V2 的 repair contribution P95 为 `0.00293`、area inflation P95
为 `1.00294`、物体保护核心损伤为 `0 px`，证明保守结构没有重演 V1 的大片误擦；但
temporal derivative P95 仍为 `0.8424`，与 SAM semantic base 的 `0.8437` 几乎相同，
没有解决闪烁。不得进入 inpaint 或批量 Clean。

## 输入与禁止项

- 0915 逻辑发布身份仍为 `chips_cards_hands__0915`，固定 220 会话、58,686 帧；当前挂载
  实体根为 `/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`。旧/新路径下
  数据集收据、预检、单样本相机参数与 SBS 的 bytes/SHA 完全一致，消费者只能按
  [`0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json`](../../tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json)
  枚举文件逐项验证后解析路径；没有修改 `/mnt/data/egodata`。
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
