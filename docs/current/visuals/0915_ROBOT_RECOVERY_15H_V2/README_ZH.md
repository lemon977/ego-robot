# 0915 Robot Recovery V2.1 浅层复核入口

本目录只存可快速打开的完整视频和摘要图。所有内容均为
`OFFLINE DEVELOPMENT EVIDENCE / NOT FOR TRAINING / NOT CONTROL GROUND TRUTH`。
详细状态以同一父任务 `packages/` 内不可变 `RESULT.json` 为准。

父任务终态为 `REJECTED_QUALITY / REJECTED_NO_RECOVERY`。全部已启动 package 已有终态，
协调器按 `ALL_STARTED_PACKAGES_TERMINAL` 于 `2026-09-20T10:57:51+08:00` 正常提前退出；
原 `2026-09-20T14:44:12+08:00` 只是 15 小时最晚截止点。W0/W1 strict、R0 quality、
strict Contact 与 Robot adoption 均为 0；浅层视频是失败诊断证据，不是成功转换样本。

## A1 · W1-DIAG Raw HaWoR

- [Poker044 全片](play_cards_0915_044_W1_DIAG_HAWOR_RAW_FULL.mp4)：166帧；左右
  side strict均拒绝。
- [Chips097 全片](get_potato_chips_0915_097_W1_DIAG_HAWOR_RAW_FULL.mp4)：394帧；
  左侧拒绝、右侧 development side strict通过；整会话仍拒绝。

MISS 不补帧。这两条证明合法输入能够执行，不是 HaWoR 精度提升。

## A2 · W1-ADOPTION Raw HaWoR

- [Poker106 全片](A2_W1_ADOPTION/play_cards_0915_106_W1_ADOPTION_HAWOR_RAW_FULL.mp4)：
  170帧，左右均拒绝。
- [Chips029 全片](A2_W1_ADOPTION/get_potato_chips_0915_029_W1_ADOPTION_HAWOR_RAW_FULL.mp4)：
  234帧，右侧有233个 structural候选帧，但未获得R0质量准入。

最终 `adoption_decision=REJECT`；A1父候选签名与A2运行签名严格分开。

## B1 / B1R · SAM3.1

- [Chips097 Hand](B1_SAM31/hand/get_potato_chips_0915_097_SAM31_HAND_TEMPORAL_REVIEW.mp4)：
  右手393/394帧为 direct-observed proxy；左手无合法锚点，保持UNKNOWN。
- [Chips097 Object](B1_SAM31/object/get_potato_chips_0915_097_SAM31_TASK_OBJECT_REVIEW.mp4)：
  三个物理袋槽均UNKNOWN，禁止union。
- [Poker044 Object](B1_SAM31/object/play_cards_0915_044_SAM31_TASK_OBJECT_REVIEW.mp4)：
  三个可见候选的物理牌/牌面身份均UNKNOWN，不可消费。
- [Poker044 Hand B1R](B1R_POKER044_HAND/hand/play_cards_0915_044_SAM31_HAND_TEMPORAL_REVIEW.mp4)：
  166帧完整；旧 `KeyError(164)` 已被有界转换为反向UNKNOWN。左右手仍质量拒绝，
  `consumer_allowed=false`。

B1R 改善的是运行终态与可追溯性，不是 Mask 质量。以上视频均不报告 accuracy。

## A3 / C1 · 数字诊断摘要

- [A3 Kai22 R0 摘要](A3_R0/A3_R0_DIAGNOSTIC_SUMMARY.svg)：A1有341、A2有233个
  structural side-frame 完成限位/FK/手内碰撞检查；正式运动门缺失，质量准入0。
- [C1 Contact 自检摘要](C1_SELF_CHECK/C1_SELF_CHECK_SUMMARY.svg)：内部投影/深度回环通过，
  但 registration未绑定、5 mm内候选0，Contact保持UNKNOWN。
- [A5 Kai22拇指饱和摘要](A5_KAI22_SATURATION/KAI22_SATURATION_SUMMARY.svg)：
  A3两条Chips的边界命中全部集中在 physical-left thumb_joint4 上界；这是内部映射
  clip机制诊断，不是越界或质量通过。
- [A6 拇指有界 IK 拒绝摘要](A6_THUMB_BOUNDED_IK/A6_THUMB_BOUNDED_IK_REJECTION_SUMMARY.svg)：
  唯一冻结候选虽降低指尖/方向残差，但两条 Chips 的 thumb4 饱和数与最长连续段
  都恶化，且饱和迁移到其他拇指关节，因此严格拒绝。

## B2 · Depth → Object6D 可见面诊断

- [Poker044 全片 review-only Object6D](B2_DEPTH_TO_OBJECT_POKER044/play_cards_0915_044_OBJECT6D_REVIEW_ONLY.mp4)：
  166帧、30 FPS、1280×960，完整解码通过。3条几何track的有限可见patch数分别为
  130/151/80帧。物理牌与牌面身份一直为 `UNKNOWN_UNBOUND`，
  `consumer_allowed=false`，仅可用于人工复核，不得进入 Contact/Robot/Clean/训练。

## D1 · Clean 零写入准备复核

- [Poker044 全片](D1_CLEAN_PREP/play_cards_0915_044_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4)：
  166帧、30 FPS、1440×506。左右 Hand 都是 UNKNOWN，所以没有删除/待写域；
  三张牌只作同帧 Raw 保护，物理牌/牌面身份未解决。
- [Chips097 全片](D1_CLEAN_PREP/get_potato_chips_0915_097_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4)：
  394帧、30 FPS、1440×506。青色是 `M_write=UNKNOWN`，品红是 `M_remove`，
  黄色是 `M_flow` 边界；只有右手可用，左手保持 UNKNOWN。

两条视频的 candidate 仍逐像素等于 Raw，固定显示
`CLEAN NOT MATERIALIZED / UNKNOWN / OFFLINE REVIEW`。早期 `D1_CLEAN_PREP/` 视频字节无误，
但 receipt 记录了提交前 staging 路径，已用不改画面的 V2 封装替代，不得再引用 V1 receipt。

不得从这些图片或视频推导外部公制精度、隐藏表面、Contact、Robot控制或物理部署结论。
