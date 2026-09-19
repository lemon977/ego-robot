# 0915 Robot15h W1 FoundationStereo 审阅入口

状态：`8/8 PASSED_INTERNAL_DEPTH_GATES`；共 2,332 帧。

本批在 H9 开封后运行，8 个独立录制会话的源视频 SHA 全部在调度时绑定。模型只加载一次，
完成 4,664 次左右一致性所需的双向推理；8 条视频均按原会话帧数完整解码。

输入与输出约束：

- 输入为 encoded SBS 的物理左右眼 `sourceIndex=1/0`，只 crop/resize；
- 不做 lens undistortion/remap，不交换左右相机；
- 两眼同时水平反射只作为 FoundationStereo 视差符号 adapter；
- 输出反射回原物理左目 640×480 像素域；
- 结果只通过内部一致性门，外部毫米精度仍为 `UNVERIFIED`，不授权 strict Contact。

审阅视频：

- [Poker 044](play_cards_0915_044_FOUNDATIONSTEREO_REVIEW.mp4)（166 帧）
- [Poker 106](play_cards_0915_106_FOUNDATIONSTEREO_REVIEW.mp4)（170 帧）
- [Chips 097](get_potato_chips_0915_097_FOUNDATIONSTEREO_REVIEW.mp4)（394 帧）
- [Chips 029](get_potato_chips_0915_029_FOUNDATIONSTEREO_REVIEW.mp4)（234 帧）
- [Poker 054 final holdout](play_cards_0915_054_FOUNDATIONSTEREO_REVIEW.mp4)（171 帧）
- [Poker 003 final holdout](play_cards_0915_003_FOUNDATIONSTEREO_REVIEW.mp4)（144 帧）
- [Chips 068 final holdout](get_potato_chips_0915_068_FOUNDATIONSTEREO_REVIEW.mp4)（478 帧）
- [Chips 056 final holdout](get_potato_chips_0915_056_FOUNDATIONSTEREO_REVIEW.mp4)（575 帧）

人工抽查 Poker054 第 85 帧与 Chips056 第 287 帧：左栏保持正确的 VST encoded 画面，未出现旧
equiDis62 重复去畸变导致的弯曲；右栏 optical-Z 与主要桌面/容器结构视觉一致，手部与物体边缘
仍有无效洞和边界不连续。因此这里是开发级 Depth 证据，不是外部深度真值或 Robot 成功。

机器证据位于
`_run/current/0915_robot15h_foundationstereo_waves_v1/attempts/attempt_0001/`；源数据和冻结
`BATCH_MANIFEST.json` 未修改。
