# 0915 Robot15h W0 Kai22 R0 审阅

原 RGB 使用物理左目 `sourceIndex=1 + crop + resize-only`；右栏是 pinned Kai22 URDF 的 wrist-local FK，不是相机叠加、真机控制或物理部署结果。

- [play_cards_0915_031](play_cards_0915_031_KAI22_R0_REVIEW.mp4)：已导出 `True`，质量准入 `False`，有效 side-frame `98`，prior `RAW_DIRECT_OBSERVED_FALLBACK_BOUNDED_NOT_ADMITTED`。
- [play_cards_0915_119](play_cards_0915_119_KAI22_R0_REVIEW.mp4)：已导出 `True`，质量准入 `False`，有效 side-frame `134`，prior `HAWOR_BOUNDED_V2_NUMERIC_PASS_HUMAN_REVIEW_PENDING`。
- [get_potato_chips_0915_007](get_potato_chips_0915_007_KAI22_R0_REVIEW.mp4)：已导出 `True`，质量准入 `False`，有效 side-frame `378`，prior `RAW_DIRECT_OBSERVED_FALLBACK_BOUNDED_NOT_ADMITTED`。
- [get_potato_chips_0915_042](get_potato_chips_0915_042_KAI22_R0_REVIEW.mp4)：已导出 `True`，质量准入 `False`，有效 side-frame `710`，prior `HAWOR_BOUNDED_V2_NUMERIC_PASS_HUMAN_REVIEW_PENDING`。

4 个 HaWoR 会话的严格双手门均拒绝，因此 R0 可审计导出不计作质量成功；缺失手没有补帧。人工视觉验收仍为 PENDING。
