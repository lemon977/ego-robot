# 0915 HaWoR 短缺口连续性 canary

状态：`PASS_CONTINUITY_CANARY_NEEDS_HUMAN_REVIEW`

这是一条尚未升格为全量基线的单会话 successor canary。输入是
`play_cards_0915_001` 的物理左目 `sourceIndex=1` resize-only 图像与已经通过数值门的
`hawor_bounded_v2`。它只处理检测/跟踪中有同侧前后观测夹住的 1–2 帧短缺口，不读取
PICO、Controller 或 `trackingData` 手部结果，也不外推首尾缺失。

## 直接复核

- [observed-only 与短缺口连续性全片对比](0915_HAWOR_BOUNDED_V2_VS_SHORT_GAP_CONTINUITY.mp4)：
  左栏为原始 `bounded_v2` 观测，右栏为连续性输出；150 帧、30 FPS、2560×960，完整
  解码通过。
- [全部缺口帧总览](0915_HAWOR_SHORT_GAP_FRAMES_CONTACT_SHEET.jpg)：黄色/橙色骨架为
  `SHORT_GAP_MANO_INTERP`，青色/紫色骨架仍是 detector-backed `observed`。
- [机器结果收据](../../../../tasks/receipts/0915_HAWOR_SHORT_GAP_CONTINUITY_V1_RESULT.json)

## 原因与结果

闪烁不是 `hawor_bounded_v2` 的骨长优化删掉了骨架。上游 `model_tracks.npy` 在相应帧
本身没有该侧 track，检测置信度为 0、检测框和 MANO 参数为 NaN；画面中手通常仍在，
但接近下边界、互相靠近或局部遮挡。缺口均为内部短缺口：

| 侧别 | 原始观测 | 补入帧（零基） | 连续性有效 | 剩余缺失 |
|---|---:|---|---:|---:|
| 左手 | 148/150 | 45、141 | 150/150 | 0 |
| 右手 | 145/150 | 23、24、35、39、44 | 150/150 | 0 |
| 双手同帧 | 143/150 | — | 150/150 | 0 |

所有 6 个 gap（共 7 帧）均同时通过：端点检测置信度不低于 0.4、手腕 world step 不高于
20 mm、任一关节 world step 不高于 25 mm、任一关节 2D step 不高于 40 px。补入后
骨长 CV 为左 0.02044、右 0.06246；原有观测几何、`observed` mask 和 detector evidence
均逐元素保持不变。

## 消费边界

- `observed=true` 仍只表示真实 detector-backed HaWoR 观测。
- `short_gap_inferred=true` 只表示离线、非因果、同侧 MANO 参数插值；它不是新的 HaWoR
  观测、接触真值或在线估计。
- 可视化和明确声明接受离线补帧的运动连续性消费者可以使用
  `visual_continuity_valid = observed | short_gap_inferred`。
- Contact、严格质量覆盖率和任何“直接观测”统计必须继续只用 `observed`。Robot online
  控制不得消费这条非因果结果。
- 超过 2 帧、无两端观测、运动门不通过或首尾缺失时保持 UNKNOWN，禁止凭空补手。

本结果需要用户观看全片后才能决定是否注册为 0915 全批 successor；当前没有授权
SAM3.1、Robot 或 220 会话扩批。
