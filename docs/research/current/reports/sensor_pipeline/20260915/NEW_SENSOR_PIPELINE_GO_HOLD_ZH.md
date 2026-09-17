# 手套＋Controller＋MANUS＋PICO 新传感器线结论（2026-09-15）

## 结论

这套方案可以继续走，但当前结论是 `GO_WITH_SUCCESSOR`，不是“现有基线可直接扩批”。

- `H1 Hand`：可以用。331/332 条已经形成 `HAND21_FROM_MANUS_CONTROLLER`；1 条源数据阻塞。
- `H2 Tactile`：可以用。331/332 条可用于五指接触事件时序；它不是标定力或掌心触觉。
- `H3 Stereo`：当前 canary 为 `FAILED_QUALITY_C`，不能按当前实现扩到 106868 帧。需要同 session 的 SBS 标定、分辨率合同和更快推理 successor。
- `H4 Mask`：当前 12 帧 canary 为 `FAILED_QUALITY_C`。SAM3.1 能找到白色手套主体，但单纯文本提示没有完整覆盖黄色 MANUS 线、固定附件和 PICO Controller；不能直接进入 Clean 扩批。

因此，新数据不是不可用；可用的主架构应当是：

```text
Controller/PICO 6DoF wrist anchor
        +
MANUS25 finger articulation
        ↓
canonical hand adapter
        ↓
role/accessory/controller/object masks（successor）
        ↓
causal Clean
        ↓
Robot Geometry
```

## 三路手腕如何融合

三路并不应等权平均，也不应把三个点强行重合：

| 来源 | 主要职责 | 不能假设什么 |
|---|---|---|
| PICO/Controller 6DoF | 稳定、连续的手腕运动和朝向先验 | 原始 tracker 原点不等于解剖手腕中心 |
| HaWoR 或 MANUS adapter | 解剖手腕、手指形状与 RGB 对齐 | 单目 absolute-Z 不是外部真值 |
| FoundationStereo | 可见表面 optical-Z 约束 | 表面点不是手腕关节中心 |

推荐的因果融合步骤：

1. 在每个 session 的短校准前缀中，估计 tracker 局部坐标到解剖手腕的固定偏移：
   `delta_local = median(R_tracker^T (p_anatomical - p_tracker))`。
2. 用 `p_tracker + R_tracker delta_local` 得到经解剖偏置校准的稳定手腕轨迹。
3. HaWoR 数据用 HaWoR wrist 约束 RGB/解剖位置；手套数据用 Controller 锚定的 MANUS wrist adapter，不伪称 MANO。
4. Stereo 只对 Z 提供稳健小权重约束。先在校准前缀估计 surface-to-wrist 偏置，再做 Huber 限幅；不得直接用表面 Z 覆盖 wrist Z。
5. 对整只手施加同一个 wrist translation；不修改手形和手指姿态。
6. 输出 source residual、时序步长和 2D 重投影；没有外部标定时只授予 development/visual authority。

当前机器可读合同已经冻结为 `contracts/wrist_fusion_v1.schema.json`，新传感器默认参数在 `contracts/wrist_fusion_sensor_defaults_v1.json`。Controller 是必须存在的主锚；Stereo 权重被 schema 限制为不超过 0.2，任何帧都不能以 Stereo 表面代理直接覆盖 wrist。若 Controller 缺失，该侧 wrist/Robot 进入无效或阻塞，不允许偷偷退化成 Stereo wrist。

裸手 Chips023 已生成第一版 canary，验证“固定局部偏移＋稳健三路融合”的代码路径。这个 canary 以 HaWoR 校准前缀为锚，因此只能说明融合是否更稳、是否保持 RGB 对齐，不能说明真实手腕误差。

手套 `play_cards_0910_001` 也已完成 Controller/MANUS＋Stereo 相对 Z canary。逐帧直接修正会放大抖动，因此最终采用15帧因果滚动中位数、±30 mm innovation 限幅、0.15 Stereo 权重和0.15慢更新。实际施加修正最大只有左4.29 mm、右4.50 mm：左腕 Z-step P95 从1.57 mm降到1.44 mm；右腕从11.37 mm基本不变。Stereo surface 自身分别为67.73和219.68 mm，不能逐帧直接覆盖 wrist。该结果支持“Controller 为主、Stereo 只修慢漂移”，不支持“Stereo 比 Controller 更准”。

## Mask 与 Clean 的必要 successor

新数据不能继续只依赖一个“glove”文本提示。H4 successor 必须拆开：

1. SAM3.1 负责左/右白色手套和前臂主体，并使用视频记忆和重入 reseed。
2. Controller 使用 6DoF 投影、CAD/包围盒或点提示形成独立 mask；文本提示只作辅证。
3. 黄色 MANUS 线、固定带和附件使用独立 accessory 类，不能期待 glove mask 自动包含。
4. 任务物体实例独立保护；Chips 三实例禁止 union。
5. Clean 在接触窄带缩小 dilation，并对当前可见物体像素 byte-exact 保护。
6. 隐藏物体外观来自经过 pose/时序验证的 causal object atlas；没有合法像素时标 `UNKNOWN`，不能用桌面或盘子填充。

当前 Clean 的同像素双 donor 共识无法识别“盘子”等非任务干扰物，因此即使 provenance 门通过，也可能把盘子复制进消除区。必须加入几何/光流 warp 和语义支持面拒绝，之后先在 Poker245、Chips039 做全片 canary。

## 证据

- H0：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/RESULT.json`
- H1：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/RESULT.json`
- H2：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h2_tactile/RESULT.json`
- H3：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0004/TERMINAL_RESULT.json`
- H4：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/attempts/attempt_0004/TERMINAL_RESULT.json`
- SAM3.1 提示探针：`archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/SAM31_手套与PICO手柄_文本提示探针.png`
- 12帧角色复核：`docs/current/visuals/新数据_PlayCards0910_053_12帧Mask角色复核_带图例.png`
- 裸手融合：`archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/裸手_Chips023_三路手腕融合Canary_全片.mp4`
- 手套相对Z修正：`archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/手套_PlayCards0910_001_Controller_MANUS_Stereo相对Z修正Canary_全片.mp4`

## Authority 边界

`HAND21_FROM_MANUS_CONTROLLER`、三路 wrist fusion、Stereo 表面 Z 和数字 Robot trajectory 都不是外部手腕真值或真实 Robot action。当前可以授权的是数据适配和视觉开发路线；Metric Contact、物理部署和真实策略训练仍需各自独立证据门。
