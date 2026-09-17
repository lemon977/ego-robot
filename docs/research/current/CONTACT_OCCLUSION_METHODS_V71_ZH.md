# V7.1 手指—物体接触遮挡方法调研与有界 Canary

更新时间：2026-09-15

## 结论先行

大规模数据处理并不依赖“每一帧都有真实 Object6D”。当前公开方法更常见的工程组合是：

```text
少量可见关键帧 / 对象 onboarding
        ↓
视频实例记忆与点跟踪
        ↓
直接观测锚点 + 短 gap 传播
        ↓
手—物接触或刚体先验作低 authority 约束
        ↓
不确定帧显式 UNKNOWN / valid=false
```

这能提高覆盖率，但不会自动产生物理真值。对 Chaoyang，最值得先跑的不是把 HOLD、
MagicHOI 或 4D Gaussian 直接换成批量主流程，而是：

1. 用 SAM 2.1 / Cutie 改善遮挡后重入和同实例保持；
2. 对刚体对象采用“对象 onboarding + 可见帧 pose anchor + 短 gap track”；
3. 只在少量困难片段上用 joint hand-object reconstruction 判断它是否真的比当前组合更好。

正式 authority 仍应保持 V7.1 的单向证据规则：隐藏区域的 amodal 结果、attachment
传播、扩散补全和 joint reconstruction 都是 hypothesis，不能反向晋升正式 Object6D、
Contact Gold 或物理接触精度。

## 1. 别人的大规模处理如何绕开逐帧真实 Object6D

### 1.1 视频记忆负责“身份连续”，不是恢复隐藏三维

[SAM 2](https://ai.meta.com/research/sam2/) 用流式视频记忆和 occlusion head 处理目标暂时
消失、重入和交互修正；官方说明其数据引擎把模型放进标注循环，约 5.1 万视频和 60 万
masklet 支撑规模化，而不是逐帧从零标注。代码、checkpoint 和训练代码为
[Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/README.md)。

[Cutie](https://github.com/hkchengrex/Cutie) 用对象级记忆减少遮挡和相似干扰物造成的身份漂移，
提供脚本化推理，仓库为 MIT。两者输出的仍是**可见区域 modal mask**；目标完全被手挡住时，
模型的“仍记得这个实例”不等于知道其真实像素、深度或 6D pose。

### 1.2 Amodal 方法补的是“可能完整形状/外观”

[Diffusion-VAS（CVPR 2025）](https://github.com/Kaihua-Chen/diffusion-vas) 以 SAM 2 可见 mask
和约 8 FPS 视频为输入，利用扩散先验生成 amodal segmentation 与内容补全；仓库为 MIT，
提供 demo 和 checkpoints，同时其评测/微调发布状态仍需在 canary 前按具体 commit 核查。
这种路线适合生成 temporal donor 或离线审核候选，但隐藏部分是生成式先验，不能作为 Raw
真实物体纹理、Gold 遮挡标签或 Object6D 真值。

### 1.3 对象 pose 通常是“一次初始化，随后跟踪”

[FoundationPose（CVPR 2024）](https://github.com/NVlabs/FoundationPose) 支持 CAD 模式和少量
reference-view 的 model-free 模式；官方 demo 在首帧估计 pose，之后切换到 tracking，而不是
逐帧重新注册。输入仍需要对象 mesh 或参考视图、对象区域、相机信息和可用深度。其官方
[许可证](https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE) 限制为非商业
研究/评估，不能未经审查直接成为可发布产品依赖。

[NVIDIA Video-to-Data](https://nvidia-isaac.github.io/video_to_data/reconstruction/) 把上述思路
做成容器化长流程：Grounding DINO 定位、SAM 2 跟 mask、SAM3D/BundleSDF 建 mesh、
FoundationPose 跟 pose，再由深度和 EKF 对齐/平滑。它证明了“onboarding + 多模型流水线”
是可扩展方向，也同时说明依赖很重；官方界面文档给出的完整重建环境约需 110 GB 容器，
不是零成本的每帧真值生成器。

### 1.4 点跟踪和 3D 跟踪负责传播，不负责语义真值

[CoTracker3](https://github.com/facebookresearch/co-tracker) 有 online（因果）和 offline 两种
版本，能联合跟踪准稠密点并输出可见性，适合将对象表面锚点跨过短遮挡。官方仓库明确大部分
代码为 CC-BY-NC，批量部署前必须保留许可边界。它没有对象实例语义、真实尺度或刚体 pose
保证，因此只能与实例 mask、Stereo 和刚体 RANSAC 联合使用。

[SpatialTrackerV2（ICCV 2025）](https://github.com/henry123-boy/SpaTrackerV2) 可从 RGB 或
RGB-D 预测世界空间 3D 点轨迹，但仓库的 online、训练和更多 depth backend 仍列为 TODO，
且[许可为 CC-BY-NC 4.0](https://raw.githubusercontent.com/henry123-boy/SpaTrackerV2/main/LICENSE.txt)。
它适合离线 challenger，不适合现在直接替换 FoundationStereo/Object6D authority。

### 1.5 Joint hand-object reconstruction 是“逐片段优化”，不是廉价批处理

- [HOLD（CVPR 2024）](https://zc-alexfan.github.io/hold) 从单目视频联合优化 MANO 手和未知
  对象隐式表面，不要求预扫描模板；[代码](https://github.com/zc-alexfan/hold)为 MIT，但要
  对每条 sequence 做预处理和训练。
- [MagicHOI（ICCV 2025）](https://byran-wang.github.io/MagicHOI/) 用 novel-view diffusion
  prior 约束不可见物体区域，在有限视角下优于纯几何方案；[官方代码](https://github.com/byran-wang/MagicHOI)
  为 Apache 2.0，仍是逐 sequence 训练、对齐和验证流程。
- [BIGS（CVPR 2025）](https://github.com/On-JungWoan/BIGS) 面向双手、未知刚体和严重遮挡，
  用 MANO、3D Gaussian 与 diffusion SDS 重建不可见部分；仓库为 MIT，但作者明确提示 SDS
  拟合可能耗时很长。
- [Do as I Do（2026）](https://www.do-as-i-do.com/) 将 SAM3/SAM3D、MoGe、HaWoR、
  TAPIR、guided diffusion 与采样式 retarget 连接起来；[代码](https://github.com/malik-group/do-as-i-do)
  为 MIT。它值得借鉴模块边界和困难 clip 修复，但仍是很新的研究系统，不能用其输出替代本地
  外部真值。

因此这些方法适合 1–2 个困难片段的 challenger；若没有独立几何/人工证据，render residual
变小也只能说明跨模型闭环变好。

### 1.6 合成数据解决鲁棒性，不生成本片段真值

[AnyHand（2026）](https://chen-si-cs.github.io/projects/AnyHand/) 发布了包含 RGB、Depth、Mask、
相机和 3D 手姿的百万级合成手—物数据，覆盖多种抓取、遮挡和视角；其结果表明用合成数据
联合训练能改善 HaMeR/WiLoR 的交互场景拟合。它适合训练“遮挡下仍找对手指/左右手”的
successor，但不能证明某个真实 Poker/Chips 帧的关节、接触或隐藏对象位置。

[HOT3D（CVPR 2025）](https://facebookresearch.github.io/hot3d/) 则代表另一条规模化路线：用
多视角设备、对象 onboarding 和质量 mask 获得大规模 3D 手物数据。官方实验显示 multi-view
优于单目，StereoMatch 优于 MonoDepth；仅用 hand proxy 的对象定位仍很粗。其
[API 为 Apache 2.0](https://github.com/facebookresearch/hot3d)，数据与 MANO 各有独立许可。
这对 Controller+MANUS+SBS 新线尤其有参考价值。

### 1.7 Contact prior 只能约束解，不能创造接触观测

HOLD 用手—物 interaction constraint 联合优化两者；[EasyHOI（CVPR 2025）](https://lym29.github.io/EasyHOI-page/)
则组合分割、inpainting、单图 3D foundation model，再以 3D physical constraints 调整手姿。
[PICO（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Cseke_PICO_Reconstructing_3D_People_In_Contact_with_Objects_CVPR_2025_paper.html)
展示了另一种扩展方式：先建立带手工点击的接触对应数据库，再用检索到的 mesh/contact 约束
拟合。共同规律是把 contact 当作 SDF 距离、非穿透、接触区域或检索先验来缩小解空间；如果
contact seed 本身来自同一 attachment 传播，就会自证循环。因此 V7.1 必须坚持
`direct/tracked → contact seed → attachment`，且 attachment 不可反向提权。

## 2. 可用性与 authority 边界

| 方法 | 主要输入 | 批处理性 | 代码/许可 | 在本项目可承担 | 不能承担 |
|---|---|---|---|---|---|
| SAM 2.1 | RGB视频、点/框/首帧mask | 高；支持streaming | Apache 2.0 | modal mask、遮挡/重入状态 | 隐藏像素、Object6D、接触真值 |
| Cutie | RGB视频、首帧实例mask | 高；脚本化 | MIT | 对象身份连续 challenger | 隐藏表面/深度 |
| Diffusion-VAS | 视频、SAM 2 mask、depth prior | 中；约8 FPS输入 | MIT；commit需冻结 | amodal候选、离线 donor | Raw真实纹理、Gold、物理几何 |
| FoundationPose | RGB-D、mask、K、CAD或参考视图 | 中高；首帧注册后跟踪 | NVIDIA非商业研究许可 | 刚体可见帧pose候选 | 形变Chips正式pose、遮挡真值 |
| CoTracker3 online | RGB、对象内query points | 高；因果长视频 | CC-BY-NC | 短gap点传播、重入一致性 | 实例语义、尺度、6D真值 |
| SpatialTrackerV2 | RGB或RGB-D | 中；工程成熟度偏低 | CC-BY-NC | 离线3D track challenger | 当前正式Depth/Object6D替换 |
| HOLD | 单目视频、手/物初值与mask | 低；逐片段训练 | MIT | 困难clip联合重建诊断 | 大规模默认baseline、外部真值 |
| MagicHOI | 短视频、预处理与NVS prior | 低；逐片段训练 | Apache 2.0 | 严重遮挡刚体clip诊断 | 未见区域authority |
| BIGS | 双手视频、MANO初值 | 很低；SDS耗时 | MIT | 双手刚体研究canary | 156条批量主线 |
| Do as I Do | 单目RGB及多模型资产 | 中低；模块化但新 | MIT | reconstruction/retarget参考实现 | 本项目物理部署authority |
| HOT3D | 多视角、对象模型/参考、标定 | 数据/基线参考 | API Apache；数据另审 | 新传感器line设计与回归 | 直接迁移其GT到本地 |
| AnyHand | 合成RGB-D及完整标签 | 训练规模高 | 数据/权重许可需单审 | 手估计遮挡增强 | 本地真实帧的真值 |

## 3. 标准 exact78 线：最多三个 Canary

### S1：SAM 2.1 + Cutie 遮挡重入对照（最高优先级）

输入固定为 R7_0 Raw、首个可靠对象 mask、任务实例 ID；Poker 每张牌和 Chips 三实例分别跑，
不允许 union。SAM 2.1 使用 causal streaming；Cutie 作独立 challenger。只在检测到重入、
面积突变或两个模型分歧时重新 prompt，禁止无限 carry-forward。

Go 门：冻结困难帧上 identity switch、错误非空、重入延迟和边界泄漏均优于当前 baseline，且
两条旧 A/B regression 不退化。No-Go：任一物理实例交换、完全遮挡期间持续输出伪 mask，或
GPU 预算 2 小时内不能完成两条 full-session。输出最多晋升 Role/Object **modal mask**，不能
晋升 amodal、Contact 或 Object6D。

### S2：对象 onboarding + pose anchor + 短 gap track

Poker 先建立薄双面 mesh/纹理 atlas；Chips 只在低形变片段测试刚体假设。以
FoundationStereo、K、实例 mask 和 FoundationPose 得到可见 pose anchor；对象内点再用
CoTracker3 online 传播，短 gap 后必须被下一个直接 anchor 双向闭合验证。

Go 门：直接 anchor 的重投影、Stereo surface residual、前后端 pose 差和实例 ID 均通过冻结
阈值，且短 gap 覆盖率提高。No-Go：Poker 正反面翻转错误、Chips 发生形变仍强制单一 SE(3)、
或闭合失败。只有直接 anchor 可进入正式 observed-only Object6D；track gap 始终进入
`BIDIRECTIONAL_TRACKED` hypothesis。

### S3：MagicHOI/HOLD 困难 clip A/B

只选一个 Poker 重遮挡 clip 和一个低形变 Chips clip，每段 5–10 秒；以同一 mask、相机与手
初值分别跑 HOLD、MagicHOI，BIGS 只在双手遮挡且前两者失败时进入下一轮。每个方法最多
2 GPU 小时/4 小时墙钟，超预算直接 No-Go，不扩到 exact78。

Go 门：留出帧上的 silhouette、可见 Stereo surface、时序重投影和手物穿透同时改善，且未见
区域在不同初始化间稳定。即便 Go，也只发布 `DEVELOPMENT_EVIDENCE/HYPOTHESIS_ONLY`；没有
外部真值时不得写成真实手形、对象完整几何或接触 truth。

## 4. Controller + MANUS + PICO/SBS 新线：最多三个 Canary

### N1：Controller/MANUS 几何 + tactile 事件门

Controller 只锚 wrist，MANUS25 只提供手指姿态；同 session 标定把两者变换到同一相机/世界
坐标。Tactile 只产生左右手、指别和时间上的 contact event，不能产生接触力、对象 ID 或掌心
接触。事件前后用 Object6D direct anchor 建 `CONTACT_SEED`，之后才允许 attachment 传播。

Go 门：跨传感器时间残差、左右/指别一致、接触前后距离变化符合事件时序；No-Go：缺外参、
时间不同步或 tactile 与对象实例无法消歧。输出可成为 tactile-supported **时序证据**，仍不是
物理距离 truth。类似的视觉—触觉 neural field 路线可参考
[NeuralFeels](https://github.com/facebookresearch/neuralfeels)，但其公开系统依赖校准视觉、触觉、
proprioception 和对象 mesh，且报告的优化速度约 1–5 Hz，不应直接套用为本线 baseline。

### N2：HOT3D 风格 SBS 多视角对象 lifting

使用同 session 4096×1536 SBS、实际 K/R/t、左右一致的实例 mask 和对象 onboarding 参考；
先三角化/robust aggregate 对象表面对应，再与 FoundationStereo optical-Z 和 Controller/MANUS
手位置做独立对照。1280×960 mono 只用于显示和 prompt，不参与该几何闭环。

Go 门：左右循环一致、registration、三角化 baseline、直接可见帧重投影通过；No-Go：借用
exact78 标定、图像域与 K 不一致、或遮挡像素参与三角化。输出是本 session 的多视角几何候选，
通过本地外部标定前仍不能声称毫米级真实精度。

### N3：Video-to-Data / Do as I Do 单 clip shadow pipeline

选择一条 rigid Poker 和一条 Chips，仅验证对象 onboarding、mask memory、pose smoothing、
inspection render 与本地数据接口；优先复用本地真实 SBS 深度和 Controller/MANUS，不以 MoGe/
HaWoR 覆盖它们。执行上作为 shadow runner，不写 current baseline。

Go 门：能在 2 GPU 小时内产出可追溯 mesh/pose/hand alignment，且直接证据与 hypothesis 分离；
No-Go：依赖缺失、模型许可未闭合、对象 identity 交换或 learned depth 覆盖 measured SBS。
输出只能是跨系统 development comparison。

## 5. Go / No-Go 总矩阵

| Canary | 主要解决 | 启动前硬依赖 | 预算 | Go 后进入 | No-Go 终态 |
|---|---|---|---:|---|---|
| S1 SAM2.1/Cutie | mask重入、实例保持 | R7_0、首帧mask、实例表 | ≤2 GPUh | modal mask successor候选 | FAILED_QUALITY_C / BLOCKED_LICENSE |
| S2 anchor+track | 遮挡前后pose连续 | K、Stereo、mask、mesh/refs | ≤2 GPUh | direct Object6D + separate hypothesis | FAILED_QUALITY_C / BLOCKED_PREREQ |
| S3 joint HO A/B | 困难片段隐藏几何 | 短clip、mask、手初值 | 每方法≤2 GPUh | development evidence | NO_GO_RESEARCH_BUDGET |
| N1 sensor contact | 接触时间和指别 | Controller/MANUS/tactile同步 | CPU优先，≤4h | tactile timing evidence | BLOCKED_PREREQ |
| N2 SBS lifting | 新线对象3D anchor | 同session标定、SBS、mask | ≤2 GPUh | visual geometry candidate | FAILED_QUALITY_C |
| N3 shadow pipeline | 模块化全链对照 | 资产和许可闭包 | ≤2 GPUh | development comparison | BLOCKED_PREREQ / BUDGET |

共同停止规则：一条 canary + 两条冻结 regression；失败最多两轮；worker 只写 immutable receipt；
任何隐藏区域或传播结果不得覆盖 direct observation；任何未来帧 donor 只可进入
`OFFLINE_BIDIRECTIONAL_VISUALIZATION`，训练输入保持 causal。

## 6. Chips023 坐标域与 ownership 实测结果

24 帧 Robot–Object ownership canary 暴露了两个不能靠调阈值解决的坐标合同问题：

1. HaWoR NPZ 未显式保存 `image_size` 时，旧导出器错用 2048×1536；本 session 的
   RGB、Mask 和 K 实际是 1280×960。现改为优先读显式尺寸，否则由中心化 K 的
   `cx/cy` 反推图像域，不再硬编码默认分辨率。
2. Object6D 的 `T_object_to_camera` 属于 rectified-depth camera，而 Robot 和最终
   ownership 属于 selected-left RGB camera。必须先计算
   `T_object_to_selected = T_stereo_rectified_camera_to_selected_camera @ T_object_to_rectified`，
   才能和 selected-left K 共同投影。

修正后 R7_4 的直接观测帧中，对象几何与 Raw 实例 Mask 的 IoU 中位数从修正前的
0 提高到 0.540（均值 0.464，最大 0.576）。24 帧里 7 帧出现 Robot–Object 投影重叠，
共 4350 个重叠像素，其中 353 个因缺少合法对象外观证据保持 `UNKNOWN`。证据为：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/
get_potato_chips_0902_023/robot_object_ownership_24frame_R7_4/RESULT.json
```

这仅证明坐标链和投影重叠已从“完全错域”恢复到可诊断状态。Chips 仍用刚性椭圆近似，
形变、隐藏表面外观和独立 Gold 标注都未闭合，因此不授予 Occlusion Gold、Contact、
Robot 或物理部署 authority。旧 R7_1 z-buffer 和 R7_2 ownership 不得作为 selected-camera
ownership 证据。

对旧 Object6D NPZ 的进一步审计确认：生产器曾直接计算
`T_object_to_world = selected_camera_c2w @ T_object_to_rectified_camera`。这会遗漏约
5.108° 的 rectified→selected 旋转；在 Chips023/physical_object_0 的 189 个有效帧上，
旧/修正 world translation 差的中位数为 39.85 mm，最大 43.64 mm。新适配器只发布
开发 sidecar，不原地修改旧 Object6D：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/selected_camera_adapter_R7_2/
get_potato_chips_0902_023/physical_object_0/RESULT.json
```

在完成更多 session 回归前，旧 `T_object_to_world` 禁止作为 Robot/Contact 坐标输入；
它在既有 Clean-only pinned revision 中的历史用途不因此被静默重写。

全部 58 条 metric-ready 会话的三个历史 Object6D root 已完成扫描：共 136 个物理实例
（Chips 117，Poker 19），无读取失败。136/136 的旧/修正 world pose 中位平移差都
超过 10 mm，108/136 超过 30 mm；实例中位数为 39.67 mm，P95 为 48.11 mm，
最大 56.74 mm。这是代码路径坐标差异，不是外部真值误差；但它已足以解释为什么
旧 Object6D world pose 不应直接进入 Robot/Contact。完整表格在：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/coordinate_domain_audit_R7_3/
OBJECT6D_COORDINATE_DOMAIN_AUDIT.csv
```

该 136 个实例的 selected-camera/world 开发 sidecar 也已全部生成，无失败：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/
selected_camera_adapter_batch_R7_3/RESULT.json
```

后续 Robot/Contact 开发必须消费这个明确坐标域 sidecar 或同等受测修正，不得再直接
读取 legacy `T_object_to_world`。

## 7. 2026-09-15 定向复核后的基线取舍

### 7.1 Masquerade 解决的是视觉 embodiment gap，不是物体遮挡真值

[Masquerade 项目页](https://masquerade-robot.github.io/)和其
[公开代码说明](https://github.com/MarionLepert/phantom)确认，公开流程的核心仍是手姿估计、
真人手臂 inpainting 和双臂 Robot overlay，再用未来 2D Robot keypoint 预训练视觉编码器。
代码说明还明确写出 Epic-Kitchens RGB 路线缺少深度，overlay 的深度方向约有 3–4 cm 误差；
其辅助标签是 2D 投影 waypoint，不是完整 3D action。

因此本项目可以复用它的三个设计：Raw/Robotized 严格配对、future-2D 辅助目标、视觉预训练后
与真实 Robot 数据联合训练；但不能从 Masquerade 推导出“手指和物体谁遮挡谁已经解决”。
Chaoyang 当前的统一 Robot z-buffer、合法对象像素来源、显式 UNKNOWN 和因果训练输入仍是必要
的增量，不能退回简单图层覆盖。

### 7.2 Video-to-Data 是可扩展的对象 onboarding 参考，不是全遮挡恢复器

[NVIDIA Video-to-Data v0.2 文档](https://nvidia-isaac.github.io/video_to_data/reconstruction/)
当前公开组合为 Grounding DINO + SAM2 + SAM3D/BundleSDF + FoundationPose，并加入 EKF、重力
对齐和 Gaussian-splat 全局轨迹优化。这个组合值得作为 S2/N3 shadow pipeline，因为它把对象
mesh、pose anchor、手轨迹和检查渲染做成可恢复的模块化产物。

但其[对象重建采集指南](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/modules/v2d_hoi_object_reconstruction/README.md)
仍要求多视角覆盖、尽量保持对象和背景可见，并指出反光、低纹理、透明和形变对象明显更难。
这说明大规模方案依赖“先看到并建模，再短时跟踪”，并没有从单个完全遮挡帧凭空得到真实
物体外观。Poker 可优先进入刚体 onboarding；明显形变 Chips 必须继续降级为 visible surface
或 UNKNOWN。

### 7.3 OmniHands 与 OpenTouch 分别进入手部和传感器 challenger

[OmniHands](https://omnihand.github.io/)面向交互手、双手和严重自遮挡的 4D hand mesh，适合在
一条 Poker 和一条 Chips 困难片段上与 HaWoR 做 hand-only challenger。它不估计合法对象纹理，
也不能单独决定 Robot/Object ownership，因此不直接替换批量 Object6D/Compositor。

[OpenTouch](https://opentouch-tactile.github.io/)的公开结果显示，触觉与硬件手姿在视觉离屏、
透明物体和接触强弱难辨时提供互补证据。这支持新传感器线保持 H1 Controller/MANUS、H2
Tactile、H3 Stereo 并行，再在时间同步后合并；其系统报告的约 2 ms 同步性能不能迁移为本地
事实，本项目仍必须从本地时间戳测量同步残差。

### 7.4 已落地的可见表面 successor 与下一门

Chips087 的连续 24 帧真实 canary 已把 Robot 所有 link 放入统一 z-buffer，并只使用 Raw 中
直接可见的三实例对象像素和 FoundationStereo optical-Z。两处已修复的工程错误是：

1. Robot 未实际覆盖物体时，不再要求 Stereo 深度质量才能保留 Raw 对象像素；
2. Session 条件保留率按像素加权，不再用最差单帧值冒充全片指标。

对真实重叠边缘，只允许“其余质量代理均通过、深度 margin 为 object-front、且 3×3 邻域存在
可靠内区 object-front”时恢复一像素边缘。最终结果为：known coverage 84.12%、UNKNOWN
15.88%、条件物体像素保留率 99.72%。证据：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_occlusion_silver_canary_v71/get_potato_chips_0902_087/
visible_surface_ordering_24frame_92_115_v8/RESULT.json
```

这仍不是完整 Silver：隐藏物体外观、全片 temporal consistency、authorized band 外 byte-exact
和独立 Gold 均未闭合。watcher 已改成增量消费 immutable hard/soft candidate：每条 Robot 候选
发布后立即以 64 帧稀疏 z-buffer 搜索最大 Robot/Object overlap，再发布局部 24 帧诊断；最终
index 才等待 Robot 上游终态。这样既避免人工挑容易帧，也不把所有遮挡计算推迟到扩批结束。

增量链路首两条已经完成：Chips083 的 known coverage 为 87.70%、UNKNOWN 为 12.30%、条件
对象像素保留率为 100%；Chips087 分别为 84.01%、15.99% 和 99.67%。两条均只授予
`PASSED_DEVELOPMENT_VISIBLE_SURFACE_CANARY`，不报告人工 accuracy，也不代表 Silver authority。
机器结果位于：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_occlusion_visible_surface_watcher_v71/sessions/
```

## 8. 建议的执行顺序

```text
S1：先解决最主要的实例/重入转换率
  ↓
S2：只给刚体对象建立 direct anchor + hypothesis
  ↓
N1/N2：利用新线真实传感器做独立证据
  ↓
S3/N3：只分析 S1/S2 仍失败的困难 clip
```

不建议现在把 BIGS、MagicHOI、HOLD、SpatialTrackerV2 或生成式 amodal completion 直接扩到
156 条。它们最有价值的用途是定位“当前主线到底缺 mask、pose anchor、隐藏外观还是手物联合
几何”，而不是以新的生成结果掩盖 `UNKNOWN`。

## 9. 本报告的 claim limit

本报告基于论文、官方项目页和官方代码仓库的公开材料，给出工程可行性判断；未在 Chaoyang
数据上执行这些 challenger，也未核验每个依赖权重的本地 SHA。所有 Go/No-Go 指标必须在实际
canary 前写入 immutable Task Packet。本报告不授予 Mask、Object6D、Contact、Occlusion、
Robot 或 Physical Deployment authority。
