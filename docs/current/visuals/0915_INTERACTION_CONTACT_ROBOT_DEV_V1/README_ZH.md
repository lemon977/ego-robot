# 0915 Interaction → Contact → Kai22 开发级单样本复核

固定输入为 `play_cards_0915_001` 的 150 帧 encoded physical-left resize-only 数据。没有重跑 HaWoR、SAM3.1、FoundationStereo 或 Object6D；没有读取 Removal/Clean，也没有使用短缺口补帧作为公制证据。

- Object6D QA：`PASS`。visible surface center 不是物体固定中心；跨帧变化只作诊断。
- 牌尺寸：`UNKNOWN`，未使用标准牌尺寸，也未补造隐藏边界。
- Human/Stereo alignment：`REJECTED_HELDOUT_ALIGNMENT`；公制 wrist translation 授权为 `False`。
- Contact 状态计数：`{"APPROACH": 2, "NO_EVIDENCE": 1032, "UNKNOWN": 3466}`。
- 合格的固定 hand/finger/object 连续窗口：`0`。
- Kai22 R0：`COMPLETED_DEVELOPMENT_BASELINE`，使用 direct-observed HaWoR；无 Contact 时仍独立交付。
- Kai22 R1：`BLOCKED_LOCAL_EVIDENCE`；首阻塞：`HUMAN_STEREO_ALIGNMENT_NOT_ADMITTED`。
- Kai22 R2：`NOT_RUN_OPTIONAL_R1_NOT_CLOSED`，不会反向阻塞 R1。

三段视频均完整解码 150 帧。`KAI22_R0_VS_R1_REVIEW.mp4` 在 R1 阻塞时明确显示阻塞，不把未修改的 R0 冒充 refinement。

所有结果仅为 `DEVELOPMENT_RELATIVE / NON_CONTROL / NON_DEPLOYABLE`。自动门不代表视觉验收、接触真值、物理碰撞完整性或真机部署授权；完整牌体、隐藏背面、卡托及未建模环境保持 `UNVERIFIED`。
