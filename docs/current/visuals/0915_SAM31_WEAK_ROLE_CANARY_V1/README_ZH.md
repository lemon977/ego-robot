# 0915 SAM3.1 弱角色单会话 canary

状态：运行 `PASSED`；人工视觉验收 `REJECTED_QUALITY_AS_CLEAN_BASELINE`

会话：`play_cards_0915_001`，物理左目 `sourceIndex=1`，resize-only，无额外 remap，
150/150 帧完整解码。

- [150 帧完整审阅视频](0915_SAM31_WEAK_ROLE_REVIEW.mp4)
- [5 帧接触表](0915_SAM31_WEAK_ROLE_CONTACT_SHEET.jpg)
- 完整机器产物：
  `_run/current/0915_sam31_weak_role_canary_v1/attempts/attempt_0001/`

## 本轮做了什么

只对 v5 的弱角色重新推理：左右前臂、每侧 4 个按可见顺序编号的独立手指皮套、
左右黄色线缆，以及 `playing_card_02`。左右手和 `playing_card_00/01` 没有重算，
只作为 SHA 守卫的只读回归证据；回归门为 `PASS` 且运行前后字节完全一致。

没有创建 `tracker` 或 `controller` 角色，没有读取 PICO26、controller pose 或
`trackingData` 手部结果。模型只绑定固定的 SAM3.1 权重。

## 结果

| 实例组 | 有证据帧 | UNKNOWN 帧 |
|---|---:|---:|
| 左前臂 | 92 | 58 |
| 右前臂 | 90 | 60 |
| 左侧 4 个皮套 | 每实例 6–8 | 每实例 142–144 |
| 右侧 4 个皮套 | 每实例 8 | 每实例 142 |
| 左黄色线缆 | 8 | 142 |
| 右黄色线缆 | 62 | 88 |
| `playing_card_02` | 115 | 35 |

每帧都显式记录 `seeded/tracked/reseeded/unknown`。空 mask 表示 `UNKNOWN`，不是角色
不存在。质量触发后最多做一次有限 reseed，没有固定周期重锚。

质量账本进一步区分了两类问题。8 个皮套与左线缆的传播 API 虽返回完整方向帧序列，
但 primary/fallback 各自都只有 4/150 帧的 raw mask 非空；其余帧是 raw track 没有产出，
质量门随后以 `AREA_BELOW_MINIMUM / AREA_RATIO_LOW` 显式记为 `UNKNOWN`，不是把大量有效
raw mask 误删。不能把 6–8 个合并有效帧误写成这些附件只可见 6–8 帧。左右前臂的
反向路由则出现
`UNKNOWN_DIRECTION_TRACKER_HAS_NO_CONFIRMED_INSTANCE`，需要单独修正反向初始化，不能靠
放宽面积门或增加固定周期 reseed 混在一起处理。

## 结论边界

任务 `PASSED` 表示固定输入上的运行、证据发布、回归保护和资源治理完成，不表示所有
弱角色质量通过。用户复核完整视频后明确指出选中区域闪烁，黄色线缆与手指皮套没有被
可靠选中，因此本结果已被拒绝作为 Clean mask 基线，不能进入自动 Clean、Contact 或
220 会话扩批。人工验收收据见
[`0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json`](../../../../tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json)。

下一版的收敛方向不是继续要求 SAM 对附件做完美语义识别，而是保持四层证据隔离：

1. Raw candidate 只表示模型原始候选。但本次封存运行没有保存其逐帧像素，只保存 raw
   面积等账本，必须记为 `ABSENT_UPSTREAM_NOT_PERSISTED`，禁止从 semantic 反造。
2. Semantic mask 原样保留 SAM 的 hand/forearm/sleeve/cable 及其 `UNKNOWN`，供 QA、
   遮挡和证据审计使用。
3. Removal mask 允许扩出 semantic mask；候选方案以 HaWoR/MANO 投影骨段构造随尺度
   变化的各向异性 finger capsule、palm 与 wrist/forearm corridor，再并入可靠的 SAM
   residual。每个像素必须保留来源，不把几何包络伪称为 sleeve 语义。
4. Feather mask 只为视觉 inpaint 对 removal 边界再扩有限像素；不得反向进入 Depth、
   Object6D、Contact 或几何真值。

线缆可能伸出 MANO finger envelope，因此仍需独立的近手 ROI、黄色外观、连通性与时序
corridor；黄色仅是当前设备实例的可替换 appearance profile，不能写成 cable 类定义。
未覆盖的线缆必须保留为 `UNKNOWN_CABLE_RESIDUAL`，不能声称已擦净。Task
object 只保护明确可见的高置信 core；hand-object overlap/遮挡带仍可 erase，但隐藏物体
像素保持 invalid，不能因 inpaint 视觉效果提升为 Object6D 或 Contact 证据。

视频 SHA256：`067d940db2107cc499f03623c9a4ea8b1b4f0b4e574cbf2ae67d1e993f630520`。
接触表从该视频固定抽取帧 0/30/60/90/120，仅用于浅层浏览。
