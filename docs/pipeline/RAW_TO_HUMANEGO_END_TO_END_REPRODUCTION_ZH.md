# Raw → HaWoR / Stereo Depth → Mask → Object6D → Clean → Contact → Robot → HumanEgo

> 本文只描述当前稳定算法、接口、质量门和已知边界。实时数量、PID和终态必须读取
> [`CURRENT_STATUS_RECEIPT.json`](../governance/CURRENT_STATUS_RECEIPT.json) 绑定的事实账本；
> 当前算法身份必须读取 [`CURRENT_BASELINE_REGISTRY_V2.json`](../governance/CURRENT_BASELINE_REGISTRY_V2.json)。

## 1. 总体数据流与禁止的回路

```text
Raw RGB/stereo/calibration/frame identity
├── HaWoR：单目MANO手形、姿态、21关节、mesh
├── Role Mask：人手/Tracker四角色删除域
└── Task-object Mask：任务物体物理身份与保护域
          │
          ├── FoundationStereo optical-Z ──→ Object6D可见表面位姿
          └── Role removal - object protection ──→ Clean视觉背景

HaWoR + Object Mask + formal Object6D
          ↓
Human Contact Hypothesis（不依赖Robot render）
          ↓
Contact-aware Robot Retarget
          ↓
Robot RGBA / z-buffer / part-id
          ↓
Occlusion Compositor（只合成，不反馈solver）
          ↓
Raw vs Robotized严格配对的H50 future-2D Visual Aux
```

Clean不得反喂Depth、Object6D或Contact真值；Compositor不得反向决定Robot接触。这样避免“Robot等Contact、Contact又等Robot”的循环依赖。

## 2. 图像分辨率与坐标域

同一会话会出现多种分辨率，因为它们承担不同职责，不是数据互相矛盾：

| 域 | 典型尺寸 | 用途 | 是否公制几何主域 |
|---|---:|---|---|
| 原始SBS | 4096×1536 | 两眼原始采集，每眼2048×1536 | 标定来源 |
| selected-left | 1280×960 | HaWoR、Mask、Clean、Robotized、HumanEgo统一RGB网格 | 否 |
| rectified stereo | 1280×960/eye | 双目极线校正后的匹配域 | 是 |
| FoundationStereo推理 | 640×480/eye | 降采样后视差推理 | 是，需同步缩放K |
| ProPainter工作域 | 960×720 | 仅补Clean剩余UNKNOWN区 | 否 |
| 中文review | 任意拼图尺寸 | 人工审阅 | 绝不是数据输入 |

1280×960 的 handle MP4 是 acquisition-aligned 视觉消费域；不能仅因尺寸相同就称为双目metric rectification。任何resize、undistort、rectify都必须同步记录像素中心约定、K和registration。

统一变换记号 `T_A_B` 表示把B坐标表示到A。重要坐标包括 raw eye、rectified-left、selected-left、camera、world、robot-base、tool/flange和hand-root。

## 3. Raw与帧身份

Raw阶段冻结会话ID、任务、源视频、相机参数、tracking/SLAM、帧数、fps和SHA。下游每一帧必须保存`source_frame`，禁止通过时长、重复尾帧或循环补帧伪造对齐。

exact78固定分母与0909/0910 acquisition-aligned v2是不同release；不得混合计数或借用不同会话标定。0909/0910数据侧release与可视化位于`/mnt/data/egodata`，本仓库只读消费。

## 4. HaWoR bounded-v2

HaWoR以单目RGB预测MANO参数：root平移/旋转、手部关节旋转、shape，再经MANO forward得到21个三维关节和mesh。`joints_3d_camera`是单目学习估计，不是Stereo测量；`joints_3d_world = c2w(t) @ joints_3d_camera`用于消除头戴相机运动。

bounded-v2只在连续observed段内进行置信加权时序平滑，对SO(3)使用合法旋转插值/回溯；missing段保持missing，不跨遮挡伪造手。质量门包括finite、proper rotation、身份稳定、骨长变化、2D重投影和更新幅度。

已知边界：HaWoR absolute-Z没有外部真值。HaWoR与Stereo表面差只能称跨系统一致性，不可直接判定谁的物理误差是多少。

## 5. 两条SAM3.1 Mask lane

### 5.1 Role Mask

四个角色固定为`left_human/right_human/left_tracker/right_tracker`。SAM3.1使用当帧点/框提示、短程memory、周期refresh和有界光流承接；HaWoR腕/掌位置用于当前帧重新锚定。出画或不可见必须empty；重入必须从当前RGB重捕获；warp只能留在当前腕ROI。

### 5.2 Task-object Identity Mask

它回答“正在操作的是哪个物理实例”，不回答“所有同类物体在哪里”。Chips三个实例独立；接触、相连或身份歧义时对应实例`observed=false + empty`，禁止用union冒充身份。Poker用动作条件保持同一张目标牌。

Clean删除域为：

```text
role_union = human_left ∪ human_right ∪ tracker_left ∪ tracker_right
clean_removal = role_union - object_protection_union
```

两条lane分别评级；任一C都不能靠另一条Mask绕过。

## 6. FoundationStereo Depth

输入是同帧rectified左右图、同会话内参/外参和冻结checkpoint。网络输出视差`d`，公制optical-Z按：

```text
Z = fx * baseline / disparity
```

其中`fx`必须属于当前推理分辨率，baseline来自同设备且身份可验证的双目标定。每帧保存`disparity_px`、`depth_m`、`depth_valid`及到selected-left的registration。

当前接口没有FoundationStereo原生confidence。下游只能使用`depth_quality_evidence`，例如finite disparity、registration、遮挡边缘、低纹理、局部一致性和有效范围；这些不能改名为模型confidence。

可信边界：公式重算、SIFT registration和内部几何闭环证明代码/图像域一致，不证明30/50/70/100cm真实距离误差。外部平面标定完成前，不得宣称毫米级物理精度。

## 7. Object6D observed-only

Object6D读取原始RGB对应的物体身份Mask、有效Depth、K、registration和c2w。对Mask内有效点反投影，使用稳健中心、SVD/PCA轴与法向生成camera/world SE(3)。

正式规则是`DIRECT_OBSERVED_ONLY + KEEP_INVALID`：遮挡帧不插值、不传播、不用Clean补出的像素。Chips三个physical instance分别输出，禁止union。平面/点云residual只是内部拟合，不是物体真实pose误差或接触真值。

## 8. Clean

Clean的像素来源优先级：

1. 同会话、几何和时序验证通过的真实temporal/stereo donor；
2. 剩余UNKNOWN删除区送ProPainter；
3. 物体保护像素必须保持Raw byte-exact。

每像素source map至少区分raw、real donor、synthetic和invalid。发布PASSED终态必须有精确帧序列、master、中文review、source map、manifest、RESULT/AGENT_REVIEW和完整解码证明。ProPainter像素是视觉合成，不是被遮挡世界的物理真值。

## 9. Human Contact Hypothesis

该模块在Robot之前运行，仅基于HaWoR、Object Mask、formal Object6D和动作时序。formal Object6D保持不变，假设sidecar单独标记：

```text
DIRECT_OBJECT6D
BIDIRECTIONAL_RIGID_HYPOTHESIS
HAND_OBJECT_ATTACHMENT_HYPOTHESIS
UNKNOWN
```

gap、端点差、相对变换漂移和实例身份均有有界门；超限或缺后端观测必须UNKNOWN。输出只是数字几何接触假设，不是物理接触真值。

当前仅有确定性几何fixture和开发诊断；真实双人复核goldset未闭合，因此Contact没有current authority。

## 10. Robot Visual

正式轨迹采用world-first：

```text
joints_world(t) = c2w(t) @ joints_camera(t)
T_base_hand_target(t) = T_base_world @ T_world_hand_target(t)
```

`T_world_base`在一个session内固定，逐帧`inv(c2w(t))`只用于把固定Robot场景渲染回相机。求解顺序：arm/wrist IK → hand pose retarget → contact finger refinement → collision cleanup → temporal refinement → final audit。

硬门包括finite/proper SE3、左右手性、KaiHand root与法兰闭包、关节限位、可达性、gross penetration、非接触指穿透、坐标与帧身份。晋升顺序固定为4关键帧→24帧→全片。

当前world-first v5.2与per-side first-observed anchor只构成开发候选；没有Robot authority。未知adapter、TCP、安装和world→base外部标定不得伪造。所有数字轨迹必须命名`visual_robot_trajectory_sidecar`并标记`control_ground_truth=false`。

## 11. Occlusion Compositor

Compositor只决定最终像素ownership：`BACKGROUND/HUMAN_FRONT/OBJECT_FRONT/ROBOT_FRONT/TIE_UNKNOWN`。对象外观来源顺序是当前Raw可见像素、验证过的temporal donor、可信纹理renderer、否则UNKNOWN；禁止拿Clean桌面像素冒充被手挡住的物体。

训练中UNKNOWN区域`training_valid_mask=0`。准确率必须与known coverage、unknown ratio、unknown contact-frame ratio和最长UNKNOWN run同时报告。没有冻结goldset时不能声称手指—物体遮挡关系已经解决。

## 12. HumanEgo Visual Aux与Policy

Visual Aux只学习未来2D双手端点：

```text
future_2d_xy_original:   [T, 50, 2, 2]
future_2d_xy_normalized: [T, 50, 2, 2]
future_2d_valid:         [T, 50, 2]
current_frame_valid:     [T]
rgb_training_valid_mask: [T, H, W]
```

Raw与Robotized分支共享session、frame、标签、valid、split、seed、语言和配置；唯一变量是RGB域。主指标是ADE_2D、FDE_2D、PCK、有效窗口覆盖、左右身份错误和时序平滑，并输出loss曲线。

数字`q_arm/q_hand`只用于渲染QA，不作为真实控制监督。真实Policy训练需要同步的`real_robot_action_sidecar`；缺失时保持`BLOCKED_EXTERNAL`，不得把Visual Aux checkpoint改名为Policy checkpoint。

## 13. 终态、authority与复现顺序

任务终态区分PASSED、FAILED_QUALITY_C、FAILED_RUNTIME_FINAL、BLOCKED_PREREQ、BLOCKED_RESOURCE、BLOCKED_EXTERNAL和CANCELLED。GPU等待不是attempt。每次attempt独立，final不同字节拒绝覆盖。

单会话复现顺序：

1. 校验current receipt、Task Packet和run signature；
2. Raw/frame/K/c2w闭包；
3. HaWoR与Stereo可并行；
4. 两条Mask独立运行与评级；
5. 三路A/B且标定通过后运行Depth/Object6D；
6. Clean独立运行并保留source map；
7. Contact hypothesis与Robot分级canary；
8. Robot render后运行Occlusion并按goldset评级；
9. 构建Raw/Robotized严格配对Visual Aux bundle；
10. 只有真实动作到位才进入Policy。

所有current代码、权重、schema、质量门、证据和缺口见同revision的基线注册表；本文不复制易过期数量。
