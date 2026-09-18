# Removal Envelope V1 失败实验封存

状态：单会话执行与证据发布 `PASSED`，人工视觉质量
`REJECTED_QUALITY`。V1 只作为失败实验与回归反例封存，不允许调半径/HSV 后重试，不得
进入 inpaint 或扩批。裁决见
[`0915_REMOVAL_ENVELOPE_V1_USER_VISUAL_REVIEW.json`](../../tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_VISUAL_REVIEW.json)，浅层复核见
[`visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md`](visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md)。

失败原因是结构性证据越权：MANO 中心线、黄色 appearance 候选和 wrist→边界前臂猜测
被直接 union 成“必须擦除”。自动门只证明来源闭合与骨架覆盖，没有测量背景误擦、面积
膨胀、repair 主导比例或 cable 身份精度。第 60 帧的 34 个黄色候选连通域以及全片
150/150 `APPEARANCE_TRACKED` 是失败证据，不是跟踪成功。

## 目的

本模块只回答“为了视觉 Clean，哪些像素必须擦除”，不回答“这些像素在物理上是什么”。
第一版固定在 `play_cards_0915_001`、物理左目 `sourceIndex=1`、resize-only/no-remap、
1280×960、150 帧上运行。它是 `weights=ABSENT` 的 CPU canary，不调用 SAM、不运行
inpaint、不启动 220 会话扩批。

四层严格分工：

1. `raw_candidate_mask`：SAM 原始候选。当前封存的弱角色运行没有保存该像素产物，必须
   记录 `ABSENT_UPSTREAM_NOT_PERSISTED`；不得从 semantic mask 反造。
2. `semantic_role_mask`：封存、只读的 admitted SAM hand/forearm/sleeve/cable 与状态证据。
3. `removal_envelope`：允许包含几何扩张与明确标记的短内部缺口推断，只供 Clean。
4. `feather_alpha`：removal 外的有限软边界，只供后续独立的视觉 inpaint 任务。

`raw_candidate_mask` 和 `semantic_role_mask` 可在各自质量门下参与开发级 QA、遮挡和接触
证据判断；`removal_envelope` 与 `feather_alpha` 明确禁止作为 Depth、Object6D、Contact、
Robot geometry 或 control ground truth 的输入。

## 已拒绝的 V1 来源

```text
Removal Envelope V1
= admitted SAM hand / forearm
+ optional admitted SAM sleeve
+ observed MANO finger capsules
+ bounded-internal-hold MANO finger capsules
+ observed/held palm, wrist and forearm corridors
+ admitted SAM cable
+ appearance-tracked cable region
- protected visible object core outside the interaction band
```

每个最终 removal 像素用独立 source bit 记录来源；裁掉 protected core 后必须满足
`source_bits != 0` 与最终 `removal_envelope` 逐像素相等。旧的
`HUMAN_EQUIPMENT_UNION.npz` 已丢失分来源信息，不能作为本模块主输入。

Sleeve semantic 是 optional evidence。即使它连续为 `UNKNOWN`，只要独立 Removal 质量门
通过，Clean 仍可通过；反之 SAM sleeve 有输出也不能替代 Clean 验收。

## MANO 包络

手指 capsule 半径采用：

```text
projected finger-width estimate × accessory scale × confidence scale
```

投影 finger width 当前由 MANO MCP 间距估计，不宣称为 mesh-derived 真实手指宽度。半径
有显式上下限；高置信帧的增长也有上限，低置信帧只能保持最近可信尺度或收缩，不能自行
扩大。

短缺口只允许两端都有同侧直接观测、长度不超过 2 帧、端点置信度和最大关节步长均过门
的内部插值。leading/trailing、过长、低置信或高运动缺口保持 `UNKNOWN`。该状态不会改写
HaWoR 的 `observed`，也不是 Contact 或 Robot online 的观测。

前臂 corridor 从 wrist 沿 palm→wrist 反方向延伸到图像边界，宽度随 palm 投影尺度变化并
受上下限约束；不使用整手固定像素的各向同性大膨胀。

## Cable appearance profile

Cable 是独立 Removal 来源，MANO 不兜底。系统类别始终为 cable；当前实例的黄色只存在于
可替换配置 `device_yellow_cable_0915_play_cards_001_v1` 中。跟踪使用当前帧 appearance、
近手几何 anchor、上一帧区域、连通分量和最多 2 帧显式 hold；它只能获得
`VISUAL_REMOVAL_INFERENCE_ONLY` 权限，不能升级为 cable semantic evidence。

## Object protection

V1 只保护 admitted task-object mask 腐蚀后的明确可见 interior，并从保护区排除 MANO
interaction band。禁止直接计算 `human_envelope - whole_object_mask`：真实 hand/object
重叠仍允许 erase，隐藏物体保持 unknown，后续 inpaint 的观感不能提升 Object6D 或
Contact 权限。

## 人工验收终态

Semantic 现状保持 `REJECTED_QUALITY_AS_CLEAN_BASELINE`。Removal 单独报告：

- 真人手、皮套和附件是否仍有明显残留；
- 桌面/背景是否被大量误擦；
- 明确可见任务物体是否被误吞；
- envelope 是否仍闪烁或尺寸抽动；
- capsule/corridor 是否跟随正确的手；
- cable region 是否覆盖可见线缆且没有明显漂移。

自动门只验证来源闭合、MANO 关节覆盖、protected core 不相交、帧数、完整视频解码、输入
SHA 不变及基础时序统计。完整视频人工复核已经否决质量：V1 将弱 prior 主动生成的大块
区域升级为强 removal，背景/任务物体误伤风险不可接受。运行 `PASSED` 仅表示证据生成
成功，不能代替视觉质量结论。

V2 必须改为 `admitted SAM foreground + validated local repairs`：MANO 只在 SAM 邻域内补
局部断口；forearm 只能从真实 foreground proposal 中选择 wrist-connected 区域；cable
只能接受带 anchor、细长度、路径、端点和双向时序约束的极少实例。无法验证时保持
`UNKNOWN`。真实 inpaint 仍是另一个绑定自身权重的任务。
