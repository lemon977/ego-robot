# 手套＋Controller 新数据 Robotized RGB 路线拒绝说明

## 结论

截至 2026-09-15，拒绝将以下组合直接作为 0910 新传感器数据的批量 Clean / Robotized RGB 基线：

```text
SAM3.1 白色数据手套文本实例
+ 同帧 Controller 6D 原点投影点提示手柄
+ SAM3.1 可见任务物体文本保护
→ 去除手套和手柄
→ ProPainter
```

终态为 `FAILED_QUALITY_C`。本轮没有启动 ProPainter，因为其输入 Mask 已经不满足最低条件；继续补图只会把错误区域变成更难识别的合成图像。

这个拒绝只针对当前自动生成 Robotized RGB 的视觉路线，不否定以下已独立发布的数据：

- `HAND21_FROM_MANUS_CONTROLLER`：可继续作为数字手部/手腕观测，但不是 MANO 或解剖真值。
- Tactile sidecar：可继续用于事件时序，但不是力或接触真值。
- Raw RGB、Controller、MANUS 与 Tactile 的研究用途。

## 冻结测试

测试在推理前冻结，未按结果替换帧：

| 任务 | 会话 | 帧数 | 取样 |
|---|---|---:|---|
| Poker | `play_cards_0910_053` | 180 | 24 帧，均匀帧＋tactile 活跃/困难帧 |
| Chips | `get_potato_chips_0910_050` | 294 | 24 帧，均匀帧＋tactile 活跃/困难帧 |

使用固定的 SAM3.1 multiplex checkpoint：

```text
SHA256 = 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
```

不是用旧 SAM2，也不是退回较弱模型。测试脚本为：

```text
src/chaoyang/ops/run_sensor_glove_controller_final_canary_v1.py
```

## 量化结果

| 质量门 | Poker053 | Chips050 | 最低门 |
|---|---:|---:|---:|
| 左右手套角色通过率 | 0.0% | 0.0% | ≥90% |
| 左右 Controller 角色通过率 | 39.58% | 39.58% | ≥90% |
| 可见任务物体帧覆盖率 | 100.0% | 0.0% | ≥80% |
| 消除区面积门通过率 | 95.83% | 100.0% | 100% |
| 最大消除面积 | 46.23% | 43.25% | ≤45% |

`protected_overlap_pixels_after_subtraction=0` 只说明布尔相减代码正确，不能挽救错误的输入实例。

## 为什么失败

### 1. 白手套没有形成可消费的左右实例

提示词 `a white instrumented glove` 在两条会话的锚帧都只返回 1 个实例。合同要求左右手独立，以免两只手交叉或离屏重入后交换身份；单一实例不能映射为左右两个角色，因此系统按 fail-closed 输出空手套流。最终 48 个“会话帧×手侧”样本的手套角色通过率为 0%。

之前的单点文本探针只证明某些措辞能够返回“一个包含锚点的区域”，没有证明它能稳定覆盖完整手套、区分左右角色、排除 Controller 并跨时序保持身份。本次冻结帧测试否定了把探针结果升级成批量 Mask authority 的做法。

### 2. Controller 6D 投影不能直接作为手柄像素锚点

本轮没有依赖文本理解“PICO 手柄”，而是把每帧记录的 Controller 6D 原点通过该 session 的鱼眼相机模型投影到 1280×960 RGB，再作为 SAM3.1 正点提示。可视化显示这些投影点经常位于真实手柄之外；SAM3.1 因而分到背景、柜体、桌面或碗。

Poker 中右侧提示虽然 24/24 被某个 Mask 包含，但这些 Mask 的面积和语义明显错误；左侧仅极少帧命中。Chips 中代表帧直接把碗当成 Controller 消除区。故“点在 Mask 内”不是充分质量门，必须同时满足物体身份和合理面积。

这说明当前 `Controller pose → camera → image` 合同或 Controller 原点到可见外壳的几何关系不足以自动定位手柄像素。没有经核实的 Controller CAD、可见外壳采样点和投影闭环前，不能把这一点当成通用 Mask 提示。

### 3. 任务物体保护不具备跨任务通用性

`playing cards` 在 Poker053 的 24 帧均返回可见牌实例；`potato chips` 在 Chips050 返回 0 个实例。Chips 代表帧中手柄提示反而选择了碗。若继续 Clean，会删除/污染碗、薯片或接触附近像素，复现此前“盘子像素被搬过来”和“指尖接触处物体被消除”的问题。

### 4. Stereo 不能在当前合同下补救该视觉路线

独立 H3 canary 已使用同 session 标定且没有硬编码 exact78 rotation，但 rectification 垂直误差为 median 6.56 px、P90 30.56 px，终态为 `FAILED_QUALITY_C`。因此当前 Stereo 不能作为手套/手柄 Mask 的可靠纠错源。这个结论只限制该新传感器线，不改变 exact78 已有 FoundationStereo authority。

## 为什么不启动第二个同签名 retry

这次运行本身正常结束，用时约 90 秒，不是 CUDA、OOM、解码或随机崩溃。Poker 与 Chips 同时出现三个独立的结构性失败：左右手套实例不足、Controller 投影语义错误、Chips 物体保护缺失。相同代码、权重、输入和提示重复一次不会改变这些前提，因此按照有界 successor 规则直接封为质量 C，而不是消耗第二次运行时 retry。

如果未来重开，必须作为新版本路线，至少新增以下证据，而不是重跑本 canary：

1. Controller 外壳 CAD 或经像素验证的 `T_controller_visible_shell`，并通过多帧投影闭环。
2. 面向白手套＋外置设备的实例分割训练/标注或可验证的点/框提示生成器。
3. Poker 与 Chips 分开的对象保护合同，Chips 必须保持多实例且不得把碗/盘子当成消除对象。
4. 困难帧人工冻结审计集，并验证接触窄带不会删除物体。

在这些前提出现前，不扩批、不运行 fresh ProPainter、不生成误导性的 Robotized RGB。

## 证据

机器结果：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_sensor_glove_final_decision_v1/attempts/attempt_0001/RESULT.json
bytes = 22596
sha256 = 5a6cb9a497c2c6443d942cca1c58a421d26c24c62cc0dda33a83a14aa8959ed5
```

可视化：

```text
archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/最终测试_Poker053_手套Controller混合Mask_24帧.mp4
archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/最终测试_Chips050_手套Controller混合Mask_24帧.mp4
```

视频均为完整冻结的 24 帧、3 FPS、1920×960。六栏依次是：原图＋投影、手套 Mask、Controller Mask、任务物体保护、最终候选消除区、Mask 棋盘预览。棋盘只表示将被删除的区域，不是 Clean 结果。

## 当前推荐用法

新数据保留为一条独立的传感器研究线：

```text
Controller wrist anchor + MANUS21 fingers + Tactile timing + Raw RGB
```

可以研究轨迹、手指状态和触觉时序；当前不得宣称已有可靠 Clean、Robotized RGB、接触遮挡或物理深度。标准 exact78 裸手线继续使用自己的冻结基线，不受本次拒绝影响。
