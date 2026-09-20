# 三线稳定基线执行计划 V3.1

状态：CURRENT EXECUTION PLAN  
执行授权：仅来自 `tasks/current/INDEX.json` 中 `execution_allowed=true` 的当前 Task Packet。

## 总原则

三条支线为 Exact78、AI1 和 AI2。任何新增诊断只约束其直接消费者，不得扩大为跨支线前置。
评价口径修复、算法质量改善、训练完成、人工验收、训练资格、控制真值和真机资格分别记账。

历史产物和收据不可覆盖；新候选失败不影响其他有效基线。若新证据证明旧结果存在坐标、标签或
时间泄漏错误，则追加限权收据，只限制受影响用途，不修改旧字节，也不全局清零。

Canonical 运行产物固定在 `_run/current/<task_id>/attempts/attempt_NNNN/`。权重、NPZ、视频和账本
实体各保存一份；浅层文档只提供导航。禁止新增根级 `out_*` 或 `worktree_*` 业务目录。

## 公共接口

变换统一采用 `T_A_B`，含义为把 B 系点变换到 A 系：

```text
T_camera_wrist = T_camera_controller @ T_controller_wrist
```

`WRIST_DUAL_REPRESENTATION_V1` 必须保存完整 `T_camera_wrist [T,2,4,4]`，并严格区分：

- `anatomical_wrist_center`：指定骨架模型的运动学腕中心；
- `visible_wrist_surface_observation`：带 source pixel、surface patch、Depth 来源和可见性的当前表面观测；
- `fused_T_camera_wrist`：开发级融合结果，不是真值。

没有可靠 surface correspondence 时只允许 `region_registration_only=true`，不得伪造三维表面点。
未经校准的不确定度写 `UNKNOWN`；普通 confidence 只能叫 heuristic weight。

所有 Human、Robotized RGB、crop、state 和 confidence 字段必须标记：

```text
CAUSAL_CURRENT | OFFLINE_NONCAUSAL | INFERRED | UNKNOWN_TEMPORAL_AUTHORITY
```

训练 current input 需要 suffix-invariance：完整序列与截断至 t 的 t 时刻输入中，uint8/离散字段逐字节
一致，浮点字段满足 `atol=1e-6, rtol=1e-5`。失败字段只能作为离线结果，不能进入在线 current input。

## Exact78

### E0 训练输入闭环

1. 复核冻结 156 清单、source group、切片重叠和 DEVELOPMENT split。
2. Chips/Poker 各生成一个合法 Raw/Robotized pair。
3. 完成 loader、loss、backward、真实 optimizer update、save/reload 和固定样本 inference。
4. 验证 H50 原时间轴、逐手 valid、端点、UNKNOWN 和 suffix-invariance。
5. E0 完成即使 E1 未完成，E2 也进入训练 ready。

### E1 Chips023 多源 3D 一致性对照

E1 不是外部精度验证。Stereo preflight 分别终态化：

- `IMAGE_DOMAIN_CHECK`：物理眼、crop/resize、镜像/回域、零重复 lens-undistortion、K 缩放；
- `STEREO_GEOMETRY_CHECK`：纵向残差、中心/边缘、encoded 图像极线适用性；
- `METRIC_CONVERSION_CHECK`：当前域 K/P、主点、baseline、视差符号、镜像 `cx'` 与 unflip。

三项均通过才允许 `LOCAL_STEREO_METRIC_DEV=true`；否则只保存 disparity/诊断。先运行固定 12 帧
canary，最多一个正常 attempt 和一个同签名 runtime retry。质量失败不重试，也不阻塞 E2。

冻结比较为 legacy、static calibration only、static calibration + fusion，显示 PICO controller、
tracker wrist、HaWoR anatomical wrist、Stereo visible surface 和 fused wrist 的原生 2D、PICO-world、
head-locked 3D、XYZ/Z/速度/残差。融合与其输入变近不构成准确性证明。

### E2 四模型训练

固定 `seed=7, batch=16, AdamW, lr=3e-4`，每模型目标为 10,000 optimizer updates 或 100 epochs
先到者，GPU 上限 12 小时。Raw/Robotized 使用相同清单、初始化、采样、非 RGB 输入和目标更新数。
每 1,000 updates 保存共同 checkpoint；预算到而目标未完成时保存完整恢复状态并写
`TRAINING_PAUSED_BUDGET`。临时 A/B 只比较最大共同千步 checkpoint。必须同时评价 learned、
stay-put 和 constant-velocity；loss 下降不等于方法有效。

## AI1

数据角色固定为 097/098 development、101 regression only、102/103 frozen adoption。

先核对 MANUS 25 节点、parent、local/global、raw/retargeted、单位、node0、左右手和重复 root，
再闭合 `MANUS local → wrist → controller → PICO world → headset/camera`。

残差同时在 camera 与 controller local 系分析：

```text
r_controller(t) = R_camera_controller(t)^T r_camera(t)
```

camera-space 残差随姿态变化首先检查固定 lever arm；速度相关性只提示时序候选。只有时间、语义和
可辨识性闭合，且静态模型在未拟合片段仍失败，才能写 `STATIC_CALIBRATION_INSUFFICIENT`。

模型顺序固定为 M0 legacy、M1 每侧 controller-local translation、M2 每侧 static SE(3)。只在
097/098 拟合并做 leave-one-recording-out；采用最简单通过者。禁止 per-frame fit，101 不得参与选择。

三维 visible surface 仅在独立腕部 RGB 证据、可追溯 pixel、mask 纯度、RGB/Depth 注册、局部 Depth
连续且无对象/线缆/附件污染时发布；否则降级 region-only。动态修正相对 static prior，30 mm 只是
`development_numeric_correction_bound`，不能累积替代标定；必须报告总偏移和触界比例。

AI1 依次交付原生骨架、M0/M1/M2 数值与回放、097/098 local baseline、101 regression、102/103
adoption；融合失败不能删除合格原生/静态结果。

## AI2

在现有状态体系中新增 wrist、palm、thumb、index、middle、ring、little 部位可观测 mask，状态为：

```text
FULLY_VISIBLE | PARTIALLY_VISIBLE_EVALUABLE | OCCLUDED | OUT_OF_FRAME | UNKNOWN
```

可见性裁判不得直接或间接依赖同一次待评估 HaWoR。HaWoR-dependent SAM 若无独立 RGB/冻结提示或
审阅证据，只能给 UNKNOWN。独立证据表明可见而 HaWoR 无输出，必须计 observable failure；部分可见
按部位计数。必须同时报告 observable pass、timeline pass、unknown 和 out-of-frame 分母。

先审计 detector/tracker、temporal attention、pose attention、motion infiller、bounded successor 和
c2w 的 temporal authority。抖动只在冻结的同一 observable frame set 上比较 wrist/fingertip step、
骨段旋转、骨长、真实 timestamp acceleration/jerk、reprojection、correction 和 latency。raw observed
不覆盖、inferred 不增加 observed、不跨 gap、不删困难帧；没有足够窗口写
`INCONCLUSIVE_OBSERVABILITY`。

Kai22 R0 先核对 joint order、side、axis、sign、unit、zero、palm base、thumb4 clip，再修确定错误；
仍有问题才比较一个有界 thumb candidate。分别发布 `KINEMATIC_ONLY`、`DEVELOPMENT_R0`、
`H50_READY`；Contact/Object 不作为 R0 总开关。

## 调度与完成

三线 CPU 可并行，GPU 只允许一个租约。E0 ready 后 E2 获得训练主优先级；E1 仅先获得 12 帧 canary。
AI1/AI2 冻结验证在 Exact 每 1,000-update 安全 checkpoint 间调度。任一线失败不清零其他线。

最终分别报告：

```text
PIPELINE_COMPLETE
NUMERIC_QUALITY_PASS
VISUAL_REVIEW_STATUS
TRAINING_COMPLETE
TRAINING_ELIGIBLE
CONTROL_GROUND_TRUTH
PHYSICAL_DEPLOYABLE
```

默认 `control_ground_truth=false`、`physical_deployable=false`、`external_metric_authority=false`。

## 本轮执行终态

V3.1 已完成代码闭环与首轮 CPU-only 输入审计；机器汇总见 [STATUS.json](STATUS.json)。三线保持
独立，没有一条线的失败被扩大成另一条线的阻塞，也没有把运行时修复记成算法质量提升。

- Exact78：E0/E1 均按证据门 fail-closed，未启动模型或 GPU；E2 未获训练准入。
- AI1：CPFS 原子发布兼容问题已修复，466 帧位置观测完成开发级 M0/M1 对照；M1 未采用，M2 和
  102/103 adoption 仍被独立证据阻塞。旧 attempt 的陈旧 staging 引用已通过独立 successor 重绑定，
  旧字节未修改。
- AI2：Poker 逻辑任务到 `playing_cards` 资产目录的映射错误已修复，8/8 当前资产路径闭合；8 个会话
  均因独立可观测性、重投影、suffix pair 或 R0 局部质量证据不足而阻塞，不再是路径误拒。

这些终态不授予训练、外部公制、控制或真机部署权限。下一轮只能针对 [STATUS.json](STATUS.json) 中
列明的直接 blocker 注册有限 successor，不能复活本轮父任务或覆盖其收据。
