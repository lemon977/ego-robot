# Removal Envelope V2 保守修补合同

状态：核心实现与合同测试 `PASS`；真实 150 帧 Canary 为 `REJECTED_QUALITY`。V1 已被
人工复核判为 `REJECTED_QUALITY`，V2 是独立后继，不修改或覆盖 V1 证据。

## 真实视频结论（2026-09-18）

`play_cards_0915_001` 的 150 帧真实 canary 已执行。V2 的“SAM 主体 + 有条件局部
repair”成功避免了 V1 的无约束膨胀：area inflation P95 为 `1.00294`，repair
contribution P95 为 `0.00293`，repair-majority 帧为 `0`，protected visible-object
core damage 为 `0 px`。但 temporal area derivative P95 仍为 `0.8424`，和封存
semantic base 的 `0.8437` 基本相同；V2 没有新增闪烁，也没有消除 SAM 原有闪烁。
另一个失败门是 whole-video stable-background hook 仅覆盖 `0.00163` 的像素，不能作为
充分背景误擦审计区域。

结论：V2 保守结构保留，但本 canary 不成为 Clean authority，不进入 inpaint 或批量。
下一任务不能继续调 MANO/cable/forearm repair 半径；应先改善 SAM semantic temporal
admission，并把背景 QA 改成局部、分段或相机运动补偿后的稳定背景证据。浅层复核见
[`0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1`](visuals/0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1/README_ZH.md)。

V2 固定采用：

```text
Removal V2 = admitted SAM foreground + validated local repairs
```

SAM semantic mask 是唯一主体。MANO、foreground proposal 与 appearance 只能修补已经有
局部支持的缺失；证据不足时输出 `UNKNOWN` 或空 repair，不能主动生成大片擦除区。

## 四类有限来源

1. MANO 只连接 SAM 邻域内、两端都有 SAM 支持且长度受限的短缺口；禁止完整手指 capsule。
2. Sleeve 是紧贴 hand boundary、靠近 MANO 方向并具备时序共动的小块 repair；不是自由目标。
3. Forearm 只能从真实 foreground proposal 中选择 wrist-connected、形状和时序均合格的
   连通分量；禁止 wrist→图像边界生成。
4. Cable 每帧最多接受配置允许的 top-k 实例，评分绑定 anchor、细长度、路径、端点和双向
   时序；黄色只是当前设备 appearance profile，不是 cable 类别定义。

## 独立质量门

V2 逐帧发布 semantic base、各 repair、最终 removal、source bits 和 feather alpha，并检查：

- background spill；
- removal/semantic area inflation；
- temporal area derivative；
- repair contribution ratio；
- 每类 repair 的来源与状态。

缺 stable-background hook、真实 foreground proposal 或 cable reverse-pass support 时，相应门
保持 `UNKNOWN_FAIL_CLOSED`，不得宣称质量通过或启动扩批。

## 权限边界

`removal_envelope_v2` 与 `feather_alpha_v2` 只允许进入 `CLEAN_MASK_QA` 和
`VISUAL_INPAINT`。它们禁止进入 Depth、Object6D、Contact、Robot geometry 或 control
ground truth。FoundationStereo 与 Object6D 从 Raw/Stereo、相机参数和 admitted semantic
object evidence 独立推进，不依赖 Clean。

实现入口：

- `src/chaoyang/pipeline/removal_envelope_v2.py`
- `contracts/removal_envelope_v2.schema.json`
- `configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json`
- `tests/pipeline/test_removal_envelope_v2.py`

真实 runner、proposal、background hook 与 cable reverse-pass 已接入并运行，但质量未过门。
不得把结构门或完整执行误报为视觉 PASS。
