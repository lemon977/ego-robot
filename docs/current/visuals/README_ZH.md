# 当前可直接复核的可视化

更新时间：2026-09-18（Asia/Shanghai）

本目录是浅层导航入口。视频、图片和数值文件大多是指向不可变证据的符号链接；删除链接不会删除原始证据。除非另有说明，以下产物均为开发复核证据，不是外部真值、Robot 控制真值或实体部署 authority。

## 0915 物理左目 resize-only HaWoR canary（当前优先复核）

- [`0915_HAWOR_RESIZE_ONLY_CANARY_V1/README_ZH.md`](0915_HAWOR_RESIZE_ONLY_CANARY_V1/README_ZH.md)
- [`0915_HAWOR_RESIZE_ONLY_CANARY_V1/0915_HAWOR_RESIZE_ONLY_REVIEW.mp4`](0915_HAWOR_RESIZE_ONLY_CANARY_V1/0915_HAWOR_RESIZE_ONLY_REVIEW.mp4)
- [`0915_HAWOR_RESIZE_ONLY_CANARY_V1/0915_HAWOR_RESIZE_ONLY_CONTACT_SHEET.jpg`](0915_HAWOR_RESIZE_ONLY_CANARY_V1/0915_HAWOR_RESIZE_ONLY_CONTACT_SHEET.jpg)

固定 150 帧输入未执行 remap。HaWoR 观测左手 148 帧、右手 145 帧；左手通过冻结
数值门，右手因画面内关节比例 81.58% 及骨长 CV 0.08272 未通过，整体为
`FAILED_QUALITY_C`。等待用户复核，不自动进入 SAM3.1。

## 0915 VST 图像域 A/B（当前优先复核）

- [`0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md`](0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)
- [`0915_VST_IMAGE_DOMAIN_AB_V1/0915_VST_IMAGE_DOMAIN_AB_REVIEW.mp4`](0915_VST_IMAGE_DOMAIN_AB_V1/0915_VST_IMAGE_DOMAIN_AB_REVIEW.mp4)
- [`0915_VST_IMAGE_DOMAIN_AB_V1/0915_VST_IMAGE_DOMAIN_AB_CONTACT_SHEET.jpg`](0915_VST_IMAGE_DOMAIN_AB_V1/0915_VST_IMAGE_DOMAIN_AB_CONTACT_SHEET.jpg)

A 为 SBS `sourceIndex=1` 物理左目 resize-only；B 为已停止的
`equiDis62 → FOV90 pinhole`。当前 remap 位移 P50 102.24 px、P95 241.02 px。
legacy processed 单目已确认来自 `sourceIndex=0` 物理右目，不能代替左目。该任务已
以 `BLOCKED_EXTERNAL` 封账，等待用户确认 A；不授权任何模型或批处理。

## 0915 HaWoR 当前批次诊断

- [`0915_HAWOR_DIAGNOSTIC_V1/README_ZH.md`](0915_HAWOR_DIAGNOSTIC_V1/README_ZH.md)
- [`0915_ONE_SESSION_CANARY_V1/README_ZH.md`](0915_ONE_SESSION_CANARY_V1/README_ZH.md)

**停止说明：** 两个 0915 目录消费了经 `equiDis62 → pinhole` 显式重映射的物理左目。
用户观察到不应有的画面弯曲，而 VST 编码像素是否已经校正尚未确认。因此这些文件只
保留为问题复现证据，不是 0915 基线，也不再支持相机、HaWoR、SAM3.1 或 Depth 正确性。
当前停止线见 [`../VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](../VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。

## 0. 0916 独立清洗

- [`0916_CLEANING_V1/README_ZH.md`](0916_CLEANING_V1/README_ZH.md)
- [`0916_CLEANING_V1/COUNTS.png`](0916_CLEANING_V1/COUNTS.png)

固定 240 个会话全部终态：222 个清洗通过、18 个质量拒绝、0 个运行失败。该目录只提供数据清洗计数、拒绝原因和代表性审阅图，不包含 HaWoR、Mask、Depth、Contact 或 Robot 结论。

## 1. 三路手腕

### 裸手旧数据：Chips023

- `裸手_Chips023_PICO_HaWoR_Stereo三路手腕_全片.mp4`
- `裸手_Chips023_三路手腕_Z方向分离全片.mp4`（右侧是逐帧相对差值尺，不是绝对 Z 轨迹）
- `../../../_run/current/media/current-visuals/裸手_Chips023_三路手腕_绝对3D轨迹全片.mp4`（推荐：相机绝对 3D、c2w 稳定后的 world 3D、左右腕绝对 Z 曲线）
- `裸手_Chips023_三路手腕_绝对3D轨迹关键帧.png`
- `裸手_Chips023_三路手腕_绝对3D轨迹数值.json`
- `裸手_Chips023_三路手腕关键帧.png`
- `裸手_Chips023_三路手腕数值.json`

三路均覆盖 420/420 帧。颜色为：绿色 PICO/OpenXR wrist、橙色 HaWoR MANO wrist、紫色 FoundationStereo 可见表面代理。

| 全片欧氏距离均值 | 左手 | 右手 |
|---|---:|---:|
| HaWoR − PICO | 90.3 mm | 105.4 mm |
| Stereo surface proxy − PICO | 103.9 mm | 96.6 mm |
| Stereo surface proxy − HaWoR | 32.5 mm | 38.7 mm |

PICO 是工程追踪参考，HaWoR 是单目学习估计，Stereo 是可见表面，不是解剖手腕中心；三者都不是外部物理真值。紫色 Stereo 点被代码明确放回 **HaWoR wrist 的同一条成像射线**，所以紫色与橙色在 RGB 上重合是构造结果。它们的 3D 均值仍相差左 32.5 mm、右 38.7 mm；不能从二维点重合推出深度相同。

`Z方向分离全片` 的绿色 PICO 点被**有意定义为每帧 0 mm**，橙/紫点显示的是 `HaWoR−PICO` 和 `Stereo surface−PICO`；因此绿色点固定不动不表示 PICO 的相机 Z 没有变化。直接读取 `T_wrist_to_camera` 得到：左腕绝对 optical-Z 为 277.8–307.9 mm（范围 30.1 mm），右腕为 277.5–499.4 mm（范围 221.9 mm）。该视频回答“另外两路相对 PICO 差多少”，不回答“PICO 自己沿 Z 走了多少”。

PICO/OpenXR wrist 目前只能作为这条裸手会话中覆盖完整的**工程锚点**，不能称为三路中已验证最准确者。项目尚未用刚性 wrist 标记、MoCap 或测量治具对 PICO、HaWoR、Stereo proxy 任一路做外部手腕位置真值测试。

已核对该会话的索引与投影：`T_wrist_to_camera` 与源 PICO wrist（`joint_names[5] = wrist`）在420帧、左右手上均为 0 mm 差异，投影到保存的 wrist 2D 也为 0 px。因此绿色点偏离肉眼认为的腕部，不是可视化误用了 fingertip 索引；它来自 PICO/OpenXR wrist 的关节定义或追踪估计偏差，也仍可能包含图像域/外部标定误差。稳定不等于无偏：PICO 可以提供连续轨迹先验，但在估计 `PICO local wrist → canonical wrist` 的左右手局部坐标偏移前，不应直接强制 HaWoR 对齐到绿色点。

绝对 3D 版没有逐帧减去 PICO：左上 RGB 显示投影位置；右侧分别显示相机坐标三维轨迹、经逐帧 `c2w` 稳定到首帧相机锚点的 world 三维轨迹，以及左右手三路绝对 optical-Z 曲线。相机面板包含头部运动，world 面板用于区分头动与手动；world 仍依赖 PICO SLAM/c2w，不是外部真值。

### 手套新数据：PlayCards0910_001

- `手套_PlayCards0910_001_Controller_HaWoR_Stereo三路手腕_全片.mp4`
- `手套_PlayCards0910_001_三路手腕_Z方向分离全片.mp4`（推荐：Controller=0 的逐帧相对 Z 尺）
- `手套_PlayCards0910_001_三路手腕数值.json`

Controller wrist 覆盖 191/191 帧；HaWoR 仅左 3 帧、右 2 帧；Controller wrist 射线上的 Stereo surface proxy 为左 189 帧、右 184 帧。因此该会话适合说明 HaWoR 对白色手套域失配，也适合检查 Controller 与表面深度的长期差异，但不能用极少的 HaWoR 共同帧评估总体精度。

紫色 Stereo surface 点是沿 **Controller wrist 的同一条成像射线**采样/反投影的，因此它和绿色 Controller 点在 RGB 上应当重合；差异主要在射线方向的距离，必须看右侧毫米曲线或 JSON，不能看二维标记间距。它们并不相等：Stereo−Controller 全片平均欧氏距离为左 80.0 mm、右 186.0 mm。

| 有效共同帧上的欧氏距离均值 | 左手 | 右手 |
|---|---:|---:|
| HaWoR − Controller | 108.4 mm，N=3 | 357.0 mm，N=2 |
| Stereo surface − Controller | 80.0 mm，N=189 | 186.0 mm，N=184 |
| Stereo surface − HaWoR | 120.0 mm，N=3 | 307.5 mm，N=2 |

## 2. 扑克牌遮挡关系

- `Poker245_扑克牌与Robot可见表面遮挡_全片151帧.mp4`
- `Poker245_扑克牌与Robot可见表面遮挡_均匀预览.png`
- `Poker245_扑克牌与Robot可见表面遮挡_数值.json`

视频是源会话全部 151 帧、30 FPS，因此总时长约 5.03 秒，不是只抽 24 帧。四栏依次为：Raw 与物体可见 Mask、统一 Robot z-buffer、前后关系、诊断合成。蓝色表示物体可见面，橙色表示 Robot 在前，紫色/棋盘表示 UNKNOWN。

当前内部指标为 known decision coverage 91.83%、unknown pixel ratio 8.17%、受保护可见物体像素保留率 99.9747%。`accuracy_reported=false`：没有独立人工 Gold 标注，不能把这些数值写成遮挡准确率。

## 3. SAM3.1 手套与 Controller 提示探针

- `SAM31_手套与PICO手柄_文本提示探针.png`
- `SAM31_手套与PICO手柄_文本提示探针_数值.json`
- `新数据_PlayCards0910_053_12帧Mask角色复核_带图例.png`
- `新数据_PlayCards0910_053_12帧Mask角色复核数值.json`

这是 PlayCards0910_053 第 90 帧的 16 项文本提示探针。四种 glove 提示对左右手均返回实例并覆盖冻结锚点；四种 PICO/VR Controller 文本提示均返回零实例。

结论是：白色手套并非在 SAM3.1 中完全不可分；Controller 不能依赖当前文本提示单独识别，应使用记录的 Controller 6D/投影作为身份先验，再配合 SAM3.1 点或框提示。旧 `MASK_CANARY12_RAW_VS_CLASSES.png` 实际执行的是点提示加光流，未执行 preflight 文档中的 glove/Controller 文本，因此其失败不能证明整套戴手套采集方案不可行。

同时必须避免把“文本提示返回手套实例”误写成“完整 removal mask 已通过”。单帧探针主要覆盖白色手套布面；黄色 MANUS 线缆/附件和手套上的 Controller 有明显漏分。它证明 SAM3.1 能识别 glove 主体，不证明现有 mask 已足以 Clean。正式 removal 必须显式合并 glove/forearm、MANUS 线缆/附件和 Controller 四类像素，其中 Controller 应由记录的6DoF投影提供点/框先验，细线缆还需要独立提示/外观分支与时序重播门。

12帧带图例版本的颜色为：紫色左手套/前臂、橙色右手套/前臂、青色左 Controller、蓝色右 Controller、绿色受保护扑克牌。该 canary 确实在12个抽样帧都产生了五类候选，但状态为 `HOLD_AUTOMATIC_GATE`：右手初始负点排除失败，Controller 的双向光流循环一致性在多帧失败。

随后执行的最后一次有界 24 帧测试已把“当前自动 Mask → Clean 路线”封为 `FAILED_QUALITY_C`，不再对相同方案做提示词重试：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_sensor_glove_final_decision_v1/attempts/attempt_0001/play_cards_0910_053/play_cards_0910_053_最终测试_手套Controller混合Mask_24帧.mp4`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_sensor_glove_final_decision_v1/attempts/attempt_0001/get_potato_chips_0910_050/get_potato_chips_0910_050_最终测试_手套Controller混合Mask_24帧.mp4`
- R3 决策收据：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/sensor_h4_bounded_decision/attempts/attempt_0001/RESULT.json`

冻结样本上两任务的 glove role pass ratio 都是 0%，Controller role pass ratio 都是 39.58%；Poker object gate 通过而 Chips object gate 为 0%。所以当前精确口径是：**拒绝现有 SAM3.1 glove 文本 + Controller 投影提示直接进入自动 Clean 的实现，不拒绝新数据本身。** 正式 H4 机器任务仍保持 `BLOCKED_RESOURCE / NOT_EVALUATED / POLICY_DEFERRED`，开发 canary 的质量 C 不覆盖正式像素 authority；Controller+MANUS、Tactile、修复标定后的 Stereo 和 Raw 视觉支路继续独立可用。

MANUS 原始输出是 wrist-local MANUS25 骨架，不是 MANO 参数或 MANO Mesh。当前新数据合同应称为 `HAND21_FROM_MANUS_CONTROLLER`：Controller 6DoF 与同会话 `T_controller_to_wrist` 给全局 wrist/root，MANUS25 给手指关节；若下游必须消费 MANO，需要再做 MANUS25→MANO 的显式 retarget/拟合，并保留来源和拟合残差，不能直接把 MANUS25 改名为 MANO。

对视觉 Robotized 数据，“Mask + Clean 正确消除手套/前臂/Controller”是必要条件，但仍不充分：还必须保护手指接触区的物体像素、为被遮挡物体提供合法外观来源、正确执行 Object/Robot z-buffer，并保证训练版 donor 因果。Contact、Depth 和 Object6D 必须继续从 Raw/传感器数据产生，不能从 Clean 反推。

## 4. Clean 底图上的 Robot 相机视角

下列 13 条均为完整会话、30 FPS。它们只修正旧 Visual-Aux 合成把 Robot 直接画在 Raw 上、导致 Robot 链接之间仍露出真人手的问题。当前版本以 Clean master 为底图，只恢复 Robot renderer 改变的像素。

**用户复核已确认这批视频不能作为正确 Robotized 结果：**`play_cards_0903_245` 的牌在接触区被 Clean 破坏且 Robot/物体前后关系仍错；`get_potato_chips_0902_039` 的同坐标 temporal donor 带入了盘子像素。它们应视为“Clean/遮挡失败复核集”，不是通过样例。

| 任务 | 会话 | 帧数 |
|---|---|---:|
| Chips | `get_potato_chips_0902_023` | 420 |
| Chips | `get_potato_chips_0902_039` | 306 |
| Chips | `get_potato_chips_0902_083` | 264 |
| Chips | `get_potato_chips_0902_087` | 312 |
| Chips | `get_potato_chips_0902_090` | 288 |
| Chips | `get_potato_chips_0902_091` | 339 |
| Chips | `get_potato_chips_0902_093` | 321 |
| Chips | `get_potato_chips_0902_095` | 363 |
| Chips | `get_potato_chips_0902_096` | 348 |
| Chips | `get_potato_chips_0902_098` | 413 |
| Chips | `get_potato_chips_0902_100` | 330 |
| Chips | `get_potato_chips_0903_050` | 302 |
| Poker | `play_cards_0903_245` | 151 |

文件名统一为 `<session>_Clean底图_Robot相机视角_全片.mp4`。这一版仅解决“复核画面里不要残留真人手”和“使用头戴相机正视角”；它没有授予物体与 Robot 遮挡顺序 authority，也没有通过 contact-preservation/semantic-donor 质量门。Clean 自身的补洞伪影、Robot 轨迹抖动和接触关系仍需分别审核。

## 5. 30 FPS 合同

Clean 输入合同保持严格 30 FPS。25 FPS 输入不能作为 30 FPS 直接放行；若需要消费历史 25 FPS Poker，必须先生成带确定性帧映射和时间戳更新的 canonical 30 FPS 输入，再进入 Clean。

## 6. 三路手腕融合 canary

- `裸手_Chips023_三路手腕融合Canary_全片.mp4`
- `裸手_Chips023_三路手腕融合Canary_关键帧.png`

该版本没有把三路强行设为同一个点。它先用前30个有效帧估计左右手各自的 `PICO local wrist → HaWoR anatomical wrist` 固定局部偏移，再将校准后的 PICO 作为运动先验、HaWoR 作为 RGB/解剖位置观测，并给经过 surface-to-wrist 前缀校准的 Stereo optical-Z 15%稳健权重。最后使用因果低通，只消费当前和过去帧。

在 Chips023 上，融合结果的逐帧 3D step P95 为左 1.89 mm、右 18.31 mm；对应 HaWoR 为左 5.00 mm、右 19.91 mm。左手抖动明显下降，右手只小幅下降，说明右手的高速真实运动/共同异常不能靠平滑器消除。融合以 HaWoR 前缀为锚，不是外部真值，也不能证明 absolute-Z 准确。

对手套新数据，应将同一设计中的 PICO optical wrist 替换成已有的 Controller 6DoF wrist anchor，MANUS25 adapter 替换 HaWoR 的解剖/手指角色；Stereo 仍只作表面 Z 小权重约束。

- `手套_PlayCards0910_001_Controller_MANUS_Stereo相对Z修正Canary_全片.mp4`
- `手套_PlayCards0910_001_Controller_MANUS_Stereo相对Z修正Canary_关键帧.png`

这条新数据 canary 进一步确认不能逐帧追随 Stereo：表面 Z-step P95 为左67.73 mm、右219.68 mm。采用15帧因果滚动中位数、限幅和慢更新后，实际 wrist 修正最大左4.29 mm、右4.50 mm；左腕 step P95 从1.57降到1.44 mm，右腕11.37 mm基本不变。当前应把它定义为保守的慢漂移修正，而不是绝对深度校正。

## 7. Clean 接触保护 successor canary

- `Poker245_Clean接触保护Mask_successor_canary.png`
- `Chips039_Clean接触保护Mask_successor_canary.png`

四栏依次为 Raw、当前大膨胀 removal、建议的接触保护 removal、两者边界与可见物体边界。当前基线使用人手18–24 px、Tracker 60 px 的膨胀；canary 改为接触窄带内人手4 px/Tracker 8 px，远离物体处人手8 px/Tracker 20 px，并继续从 removal 中扣除当前可见物体像素。

在各自选出的6个困难帧中，新增消除边缘像素减少：Poker245 为64.14%，Chips039 为65.62%。这验证了“当前膨胀过大”可以直接修正，但尚未形成新 Clean authority。Poker245 的物体 Mask 只有28.48%帧有可见对象，Chips039 为59.48%；物体 Mask 无效的帧无法保护牌/薯片。另一个独立缺陷是同整数坐标 temporal donor 没有“盘子/非任务干扰物”语义，两个 donor 像素一致也可能把盘子复制进洞内。因此下一版必须同时补：Object重入/遮挡 atlas、因果几何或光流 warp、非任务支持面拒绝。仅缩小 dilation 不能单独解决完整 Clean。

### 全片候选

- `Poker245_Clean接触保护前后_全片.mp4`（151 帧，30 FPS）
- `Chips039_Clean接触保护前后_全片.mp4`（306 帧，30 FPS）

四栏依次为 Raw、现行 Clean、接触保护 successor 候选和删除区域诊断。候选把“旧固定大膨胀删除、但新接触保护 Mask 不再要求删除”的像素从 Raw 原样恢复；当前可见的任务物体像素也从 Raw 恢复。它没有重新生成或伪造被手完全遮挡的物体外观。

| 会话 | 全片 removal 减少 | 恢复的过删像素 | 物体 Mask 有效帧覆盖 |
|---|---:|---:|---:|
| Poker245 | 21.74% | 8,666,686 px | 28.48% |
| Chips039 | 21.25% | 12,079,241 px | 59.48% |

这两条全片已通过 `ffmpeg -xerror` 完整解码。它们证明“缩小接触窄带膨胀＋恢复过删 Raw 像素”能够直接落到全片，不再只是六帧示意图；但由于物体 Mask 覆盖不足，Mask 无效帧仍明确显示“隐藏外观 UNKNOWN”。因此它们是下一轮 temporal object atlas 的输入证据，不是新的 Clean B 或训练 RGB authority。

## 8. R3 Robot 硬几何与可见表面遮挡增量

v75 的 9 条精确复用会话已完成独立数字硬几何审计：9/9 通过 finite、URDF 限位、时序合同和当前 URDF/mesh 自碰撞门；严格人手姿态相似度为 0/9，但该项按 R3 仅为软诊断。6 条会话具备冻结矩阵要求的 Clean 前置并生成 24 帧可见表面 z-buffer 视频，3 条因缺 Clean 绑定明确 `BLOCKED_PREREQ`。

视频目录：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_occlusion_visible_surface_v75_r3/sessions/<session>/focus_visible_surface_ordering/`

当前包含：`get_potato_chips_0902_103`、`112`、`114`、`119`、`121`、`122`。每条均为 24 帧、1280×960、30 FPS，已完整解码。总收据为：

`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_occlusion_visible_surface_v75_r3/RESULT.json`

这些结果只比较当前可见的 Robot/Object optical-Z 并显式输出 UNKNOWN；`accuracy_reported=false`，没有隐藏物体外观、独立 Gold 标注、正式 Silver、Contact、Robot 控制或物理部署 authority。它们不能替代用户要求的 Poker 全片正面遮挡复核；Poker canary 仍需等待 Robot runtime successor 和合法 causal compositor。

## 9. Clean-20/21 因果语义前置审计

- `Poker245_Clean20_21语义前置审计_全片.mp4`
- `Chips039_Clean20_21语义前置审计_全片.mp4`

两条视频均为完整会话、1280×720、30 FPS。它们检查 `M_remove / M_flow / M_write` 分离、接触窄带、可见物体保护、donor 时间方向和 `M_write` 外变化，不是新的 Clean 成片。

| 会话 | 接触窄带额外删除减少 | 旧 donor 来自未来帧 | 因果过滤后 UNKNOWN/write | 可见物体与 `M_write` 重叠 |
|---|---:|---:|---:|---:|
| Poker245 | 69.62% | 43.79% | 82.36% | 0 |
| Chips039 | 75.88% | 41.96% | 69.90% | 0 |

写域缩小和当前可见物体保护方向已得到全片像素证据，但正式训练仍被阻塞：旧 donor 是双向的、没有 support-surface 语义拒绝，也没有无损帧/source-map 闭包；Poker verified atlas 尚不存在。因而这轮没有申请 GPU、没有运行新 ProPainter，终态是 `BLOCKED_PREREQ` 而不是质量 C。视频中的有损 MP4 编码漂移不能被解释成物体被算法删除，正式 byte-exact 门必须在无损帧上计算。

## 10. 历史 Poker Object Mask 有界对比（非当前路线）

- `Poker015_SAM31_vs_SAM21_全片开发复核.mp4`（428 帧，1280×480，30 FPS）

这条视频只用于比较当前 SAM3.1 证据与 SAM2.1 causal challenger，不代表 SAM2.1 已晋升。SAM2.1 虽完成 428 帧推理，但 369 帧为 `UNKNOWN/低面积`，known coverage 只有 13.79%，且冻结重入窗口没有共同有效帧，因此结论为 `NO_GO / FAILED_QUALITY_C`。Cutie 在同一冻结 canary 上发生 CUDA OOM，按运行时失败封账且没有重试；Poker001/005 回归因 canary 未通过而没有启动。

这段内容仅保留历史失败证据，不能据此注册后续 challenger。当前 0915 全链已经由用户锁定为 SAM3.1 唯一可执行 Mask 模型，不再创建 SAM2.1/Cutie 任务或执行胜者选择。历史 Role challenger 因 Chips010 缺少合法四角色 seed、004/009 缺少 reviewed 四角色 adapter 而为 `BLOCKED_PREREQ`，没有把旧 object prompt 错当成人/左右手/前臂/Tracker 提示。
