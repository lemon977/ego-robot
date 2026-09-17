# Play Cards 0910_001 全链路开发基线问题总结

更新时间：2026-09-14（Asia/Shanghai）

## 结论

`play_cards_0910_001` 的 191 帧产物只能证明开发流水线能够执行，不能证明各阶段质量合格。八宫格混用了正式模型、工程代理和 fallback；其中 HaWoR 仅在左手 3 帧、右手 2 帧有检测，导致后续 Mask、Stereo 手腕代理、Robot 输入均缺少稳定共同分母。该套结果不得晋升为当前 authority。

## 各阶段的问题

| 阶段 | 当前产物的问题 | 影响 |
|---|---|---|
| Raw | 191 帧视频本身可解码；主要问题不是 Raw 损坏，而是白色数据手套、手柄和该视角与现有 HaWoR 训练域差异明显。 | 上游视觉手部检测失效。 |
| HaWoR | 左手仅 3/191、右手仅 2/191 帧检出；这不是完整手部轨迹。 | 无法支撑全片 MANO、三维手腕、接触或稳定 Robot retarget。 |
| Mask | 使用 Controller+MANUS 几何、颜色和光流的开发基线，不是 exact78 当前 SAM3.1 role/object authority。 | 轮廓、左右身份、离屏重入和手物遮挡关系可能错误。 |
| FoundationStereo | Depth 是双目可见表面 optical-Z；所谓“Stereo 手腕”又依赖 HaWoR 手腕射线，只剩左 N=2、右 N=1 的共同样本。 | 不能把它解释为解剖手腕，也不能由 N=1–2 推断总体精度。 |
| Object6D | 只有 51/191 帧直接观测；输入又来自开发 Mask。 | 位姿不连续，接触遮挡时尤其缺测；不是全片物体真值。 |
| Clean | 使用逐帧 Telea 基线，不是 exact78 当前“同像素 temporal donor + ProPainter”的结构性基线。 | 易产生涂抹、边缘残留和帧间闪烁，不能代表 exact78 已发布 Clean；而 exact78 Clean 本身也尚未通过接触边界/语义 donor 门。 |
| Robot | HaWoR 检出时用 HaWoR，其余帧由 Controller+MANUS 补位，输入来源在时间上切换；contact/occlusion 未形成 authority。 | 可能出现手腕跳变、手形不一致、穿透和错误遮挡。正式状态仍为 HOLD。 |
| 八宫格 | 同一画面并列不等于同一正式 authority 链，开发 fallback 容易被误认为正式算法。 | 只能作为失败诊断，不应作为当前基线演示。 |

## 原 V1 三路手腕数字为何不可作为精度结论（已被 V2 替代）

| 对比 | 左手 | 右手 | 核心限制 |
|---|---:|---:|---|
| HaWoR − Controller | 108.4 mm，N=3 | 357.0 mm，N=2 | 共同样本过少；Controller+固定外参不是外部真值。 |
| Stereo − Controller | 94.4 mm，N=2 | 370.1 mm，N=1 | Stereo 是可见表面代理，不是关节中心。 |
| Stereo − HaWoR | 105.5 mm，N=2 | 30.3 mm，N=1 | 只能说明极少数帧的跨系统差异。 |

以上 94.4/370.1/105.5/30.3 mm 是旧 V1 的 HaWoR 射线共同帧统计，不得与后文 Controller 射线 V2 的 80.0/186.0/120.0/307.5 mm 混用。当前浅层视频和 JSON 使用 V2。

因此，0910_001 的主要故障链是：

```text
HaWoR 严重漏检
  → Mask 使用开发代理
  → Stereo 手腕共同样本几乎为空
  → Object6D 仅部分帧可见
  → Clean 使用逐帧 Telea
  → Robot 混用两类轨迹来源
```

## 替代复核：Chips023

选择 `get_potato_chips_0902_023`，因为 PICO、HaWoR 和 Stereo surface-Z proxy 均有 420/420 帧。视频明确显示三者语义不同：PICO 是工程手部追踪 wrist；HaWoR 是单目学习得到的 MANO wrist；Stereo 是在可见 HaWoR 关节射线上统计表面 Z 偏移后投到 wrist 射线的代理，不是解剖关节真值。

| 双手全片均值 | 左手 | 右手 |
|---|---:|---:|
| HaWoR − PICO 欧氏距离 | 90.3 mm | 105.4 mm |
| Stereo proxy − PICO 欧氏距离 | 103.9 mm | 96.6 mm |
| Stereo proxy − HaWoR 欧氏距离 | 32.5 mm | 38.7 mm |

这些数值的共同分母是 420 帧，适合观察时序变化，但仍只属于跨系统内部比较；三路都不是外部测量真值。

产物位于原数据会话的 `visualizations/wrist_comparison_v2/`，避免再次散落到临时运行目录。

## 0910_001 Controller 射线重试

为直接回答“Controller 与视觉 tracker 相差多少”，V2 改为三路独立语义：

1. Controller 6D 与同会话固定 `controller_to_wrist` 外参组成的 wrist，191/191 帧；
2. HaWoR 单目视觉 tracker 的 MANO wrist，左 3/191、右 2/191；
3. 在 Controller wrist 投影射线附近取 FoundationStereo 11×11 有效深度中位数得到的可见表面代理，左 189/191、右 184/191。

第3路反投影仍沿 Controller wrist 的同一条射线，所以绿色 Controller 与紫色 Stereo surface 在 RGB 上重合是预期构造，不是“两种三维估计相等”。深度/三维差异应读取毫米曲线与 JSON。

MANUS25 只给 wrist-local 手形，wrist root 仍来自 Controller，因此没有把它伪装成独立第四路 tracker。

| V2 全片/有效共同帧 | 左手 | 右手 |
|---|---:|---:|
| HaWoR − Controller 平均欧氏距离 | 108.4 mm，N=3 | 357.0 mm，N=2 |
| Stereo surface − Controller 平均欧氏距离 | 80.0 mm，N=189 | 186.0 mm，N=184 |
| Stereo surface − HaWoR 平均欧氏距离 | 120.0 mm，N=3 | 307.5 mm，N=2 |

Controller 轨迹连续并不自动代表准确。尤其右手 Stereo−Controller P95 为 483.5 mm，说明 Controller 射线有时落到手套、手柄、背景或遮挡边缘表面；因此 Stereo surface 不能直接覆盖 Controller wrist。V2 产物在数据会话的 `visualizations/controller_tracker_wrist_v2/`。

## 2026-09-15 追加复核

### SAM3.1 手套与 Controller 文本探针

旧 H4 `MASK_CANARY12_RAW_VS_CLASSES.png` 使用的是 SAM3.1 点提示加光流，并未实际执行 preflight 中声明的 glove/Controller 文本。为避免据此误判采集方案，已在 `play_cards_0910_053` 第 90 帧独立测试 16 个文本提示：

- 四种 glove 提示在左右手位置均返回实例并覆盖冻结锚点；单帧白色手套分割可行。
- 四种 PICO/VR Controller 文本提示均无实例；Controller 文本单独不可用。
- 下一版应使用 Controller 记录位姿/投影确定身份，以点或框提示细化轮廓，而不是等待文本识别手柄。

该结论仅来自一帧 prompt sweep，不授予时序 Mask authority，也不能证明整个新传感器方案已经可用。浅层入口为 `archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/SAM31_手套与PICO手柄_文本提示探针.png`。

### 扑克牌全片遮挡复核

`play_cards_0903_245` 已生成全部 151 帧的四栏可见表面 z-buffer 视频，不再使用 24 帧 Chips 片段代替。当前 known coverage 为 91.83%、UNKNOWN 为 8.17%、受保护可见物体像素保留率为 99.9747%，但没有独立 Gold 标注，结果明确记录 `accuracy_reported=false`。

### Clean 底图 Robot 相机视角

审计发现旧 Visual-Aux `robotized_rgb` 尽管绑定了 Clean RESULT，实际视觉合成仍以 Raw 分支为画布，因此 Robot 链接之间会露出真人手。现已输出 13 条修正复核片：以 Clean master 为底图，只恢复 Robot renderer 改变的像素。它们用于暴露问题，不是通过结果。

用户全片复核进一步确认：`play_cards_0903_245` 的牌在指尖接触区被 Clean 破坏，且物体/Robot 前后关系仍错；`get_potato_chips_0902_039` 的 Clean 带入盘子像素。原因是当前手部固定膨胀 18–24 px、Tracker 膨胀 60 px，而物体 byte-exact 保护只覆盖当前帧可见 mask；同坐标 temporal donor 又没有 Stereo/光流/平面几何对齐，真实来源不等于语义正确。统一浅层入口为 `docs/current/visuals/`。
