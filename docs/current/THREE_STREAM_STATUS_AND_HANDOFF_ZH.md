# Chaoyang 三条数据链现状与后续 AI 交接

更新时间：2026-09-20（Asia/Shanghai）

这份文档回答三个问题：旧 exact78 到底是什么数据、两个 Robot30 在哪里；手套数据研究到底走到哪；0915 裸手全链目前真正卡在哪里。`archive/` 仅在本页用于定位已经封存的历史证据，不是当前算法规范。

## 一页结论

| 数据链 | 当前可信结果 | Robot/ML 状态 |
|---|---|---|
| 旧 exact78（裸手+tracker） | 156 会话历史漏斗已闭合；47/60 Robot30 全片视频可看，24/60 有离线 hard-geometry pass evidence | 0 条 training eligible，0 条控制/部署 authority |
| 0916 手套数据（AI1） | PICO wrist、MANUS、SAM mask、HaWoR 在 101 真盲测全部跑通；冻结手指几何双侧失败 | 不可作监督标签；只可保留为开发传感器模态 |
| 0915 裸手数据（AI2） | encoded-domain Depth 可用作内部开发证据；HaWoR/R0/Contact/Robot 严格转换尚未恢复 | 当前 strict Robot 转换仍为 0；不可作控制或训练真值 |

## 1. 旧 exact78：原始数据、Robot30 与转化率

### 原始数据

`exact78` 不是“100 条”。冻结分母是两类任务各 78，共 156 条：

- Chips 78：
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips`：16
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/potato_chips`：52
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0903/potato_chips`：10
- Poker 78：
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards`：43
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards`：25
  - `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0903/playing_cards`：10

逐会话冻结清单：

`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json`

### 两个各 30 的 Robot 数据在哪里

完整 60 行索引及每条视频历史绝对路径：

`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1/lane_e_robot30/attempts/attempt_0002/ROBOT30_VIDEO_DELIVERY_MATRIX.json`

矩阵是在归档前生成的，`verified_video.path` 仍以旧
`/mnt/workspace/code/chaoyang/tasks/` 开头。实际当前路径要把该前缀替换为：

`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/`

已逐条核验：47 个 `verified_video` 按此重定向后 47/47 都存在，并于 2026-09-20
重新完成当前文件的全片解码，结果仍为 47/47 PASS；其余 13 行本来就没有 verified full
video，不能按目录名臆造。矩阵同时给出每条视频的 bytes、SHA、帧数和完整解码状态。

可直接打开的两类代表视频：

- Chips：`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v74/batch_001/render/chips/get_potato_chips_0902_103/fullsession_review/get_potato_chips_0902_103_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4`
- Poker：`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/robot_v77_batch_001_r22/attempts/attempt_0002/work/render/poker/play_cards_0903_227/fullsession_review/play_cards_0903_227_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4`

其余逐条路径必须从上述 60 行矩阵读取并只替换固定前缀；不要用 `find` 结果替代冻结选择顺序。

终态与 hard-geometry 证据：

`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/robot30_causal_budget_closure/attempts/attempt_0001/ROBOT30_FINAL_TERMINAL_INDEX.json`

真实状态：

- Chips30：29 条 verified full video，10 条 hard-geometry pass evidence。
- Poker30：18 条 verified full video，14 条 hard-geometry pass evidence。
- 合计：47/60 有完整审阅视频；24/60 有离线 hard-geometry pass evidence；35 条预算未评估；1 条运行失败。
- 60/60 均 `training_eligible=false`、`control_ground_truth=false`、`physical_deployment_authorized=false`。

### 从 raw 到 Robot 的历史漏斗

权威历史漏斗：

`/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_conversion_funnel_v2/CONVERSION_FUNNEL.json`

156 raw → 144 HaWoR A/B → 124 加 role mask → 101 triple-ready → 58 metric/depth/object → 39 Clean → 8 Robot visual candidates → 0 Robot authority。

按 100 条 raw 等比例解释：约 `5.1` 条能到旧版离线 Robot visual candidate，`0` 条能成为训练/控制 authority。首要瓶颈不是单一算法：12 条止于 HaWoR C，20 条止于 role-mask C，43 条缺后续 metric calibration，23 条止于 object-mask C；而旧 exact78 Depth 还因重复 lens-undistortion 已撤权，不能复用为当前深度结论。

### wrist 与 PICO 参考的误差

代表会话：`get_potato_chips_0902_023`，420/420 帧。按用户明确授权，把 PICO/OpenXR wrist 当作这次工程比较参考：

| HaWoR − PICO | 左手 | 右手 |
|---|---:|---:|
| 2D mean / P95 | 237.7 / 282.1 px | 175.8 / 334.6 px |
| 3D mean / P95 | 90.3 / 101.8 mm | 105.4 / 140.2 mm |
| signed mean ΔXYZ | +52.1, +46.2, -54.1 mm | +58.8, +10.5, -83.1 mm |

定义：2D 是同一 selected-camera 图像域内 HaWoR wrist 与 PICO wrist 的像素欧氏距离；
3D 是同一 camera-optical XYZ 中 `HaWoR − PICO` 的欧氏距离；signed `ΔZ` 只是光轴
深度分量。该数值没有用 Stereo 补深度，也没有做事后刚体/平移贴合。

- [完整 420 帧参考视频](visuals/EXACT78_WRIST_PICO_REFERENCE_V1/CHIPS023_PICO_ASSUMED_REFERENCE_FULL_REVIEW.mp4)
- [数值收据](visuals/EXACT78_WRIST_PICO_REFERENCE_V1/METRICS.json)

确定结论：当前 exact78 HaWoR wrist 的绝对 2D/3D 定位不够支撑毫米级 Robot/Contact 标签，主要是大幅图像偏移与系统性深度偏浅；“轨迹平滑”不能消除这个偏差。PICO 是用户指定工程参考，不自动升级为实验室 MoCap 真值。

## 2. AI1：0916 手套数据

### 数据

- 总原始根：`/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916`
- 本轮牌任务根：`/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916/cards_130_0916`
- 当前真盲测：`.../cards_130_0916/101`
- 只读 processed：`/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned/playing_cards/play_cards_0916_101`
- 隔离代码分支：`research/wiyh-hand-depth-v1`

### 已经做完

- 097/098 用于开发与已暴露诊断；099 因左腕仅 5 个可见帧按预登记门拒绝。
- 101 在查看 RGB 前冻结并执行 V53 SAM3.1 → V54 scale/PICO visibility → V55 HaWoR + frozen V45 geometry。
- 101 的 mask/visibility/HaWoR 运行全部完成，左/右合格观测 114/77；完整视频 122/122 帧。
- 几何质量仍失败：左侧 landmark P50 `150.38 px` 且形状门失败；右侧 joint-in-mask P50 `0.4167`、joint→mask P95 `107.19 px`。
- 右侧 PICO-derived wrist→mask P95 为 `2.98 px`，说明动态腕锚点本身可落在手区；这不能替代整套手指骨架验收。
- 本轮没有发布 3D/Depth 精度数值：冻结 2D 独立门已经失败，`depth_unlocked=false`。因此当前只能定性为“PICO wrist 有开发价值、MANUS 跨会话手指几何不合格”，不能声称组合后的 3D 更准。

- [完整 122 帧视频](visuals/WIYH_SESSION101_BLIND_V55/PICO_WRIST_MANUS_SESSION101_BLIND_FULL_REVIEW.mp4)
- [完整研究结论](../research/current/world_in_your_hands/SESSION101_BLIND_VALIDATION_V53_V55_ZH.md)

### ML 判定与下一任务

当前 PICO wrist + MANUS fingers 的组合 **不是 training-ready label**。下一 AI 必须把 101 固定为失败 holdout，不得继续在 101 调参后宣称盲测。合理后继是：

1. 明确每个硬件安装实例的 Controller→wrist 与 MANUS local-basis provenance，检查是否存在会话/佩戴实例变化。
2. 候选只允许 session-static、物理可解释的安装/基座参数；禁止逐帧贴 mask。
3. 在开发会话上形成候选后，重新选一个从未打开的会话作 adoption holdout。
4. 只有独立 RGB 2D 几何与完整视频通过后，才启动 Depth；通过仍不等于外部毫米真值。

## 3. AI2：0915 裸手全链

### 数据

- 原始根：`/mnt/data/egodata/datasets/ego/chips_cards_hands_0915`
- Poker：`cards_120_0915`，120 会话。
- Chips：`chips_100_0915`，100 会话。
- 当前 processed：`/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`
- 合计 220 会话、58,686 帧。

### 当前实质结果

- 正确图像域已经固定为 encoded VST 物理眼 crop + resize-only，不做 lens undistortion/remap。
- FoundationStereo 在 12 个 W0/W1 会话上 12/12 通过内部 encoded-domain Depth 门；外部毫米精度仍未验证。
- HaWoR W0 4/4、W1-DIAG 2/2、W1-ADOPTION 2/2 都没有恢复整会话 strict PASS。
- W1-DIAG Chips097 只有右手逐侧 strict 通过；其余只能作局部/结构诊断。
- Contact 审计：5040 行 → 483 direct visible-surface rows → 161 unique samples → 251 metric rows → 0 inside finite patch → 0 严格 ≤5 mm；`LOCAL_STEREO_METRIC_DEV=false`。
- Poker044 Object6D 只有三个 visible candidates，身份未绑定；R1-E 未获授权。
- R1-H 可以生成小范围开发候选，但 0 adopted，不能冒充成功 Robot。

当前恢复任务的浅层入口：

`/mnt/workspace/code/chaoyang/docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/`

当前长程任务尚在 canonical 主工作树封账；最终数字必须以其 F0/最终提交为准。本页不覆盖其活动工作树。

### 下一任务优先级

1. 先完成当前恢复任务封账，固定真实失败矩阵；不要扩到 220。
2. HaWoR：定位跨会话 strict 失败是观察窗口、错侧/裁切、绝对 wrist 偏差还是 articulation；消费者局部窗口不能反向改写 session strict。
3. Contact：先修复 `251 metric → 0 finite patch` 的像素注册/可见 patch/遮挡关系，不要把 5 mm 放宽来制造 Contact。
4. Object6D：必须绑定真实物体实例；visible candidate 不等于 card identity 或完整 6DoF。
5. Robot：先争取多个严格 R0，再研究局部 R1-E；R1-H 继续 `training_eligible=false`。

## 给下一个 AI 的执行边界

- 先读 `docs/current/AI_WORK_ENTRY_ZH.md`、本页及对应分支研究文档。
- 不把 `archive/` 当算法规范；这里只允许按明确路径读取历史收据。
- 不修改 `/mnt/data/egodata`；不覆盖 sealed 输出。
- 任何 GPU 推理走中央租约；不能停止其他实验。
- 不同时在 canonical 活动工作树和隔离研究分支写同一文件。
- 结果必须区分：程序完成、数值质量通过、人工视频通过、training/control/deployment authority；四者不能互相替代。
