# 0915 HaWoR 当前批次诊断

本目录只把已经冻结的 HaWoR NPZ 叠加回同会话物理左目 RGB，便于检查
检测缺失、左右手身份和投影。它不修改检测、不插值缺失手，也不授予解剖或物理真值权威。

- `play_cards_0915_001_HAWOR_PHYSICAL_LEFT_REVIEW.mp4`：150 帧完整视频；青色为左手，橙色为右手，`OBS/MISS` 是原始 HaWoR 状态。
- `play_cards_0915_001_HAWOR_CONTACT_SHEET.jpg`：均匀抽取 12 帧的接触表。
- `RESULT.json`：输入与输出 SHA、解码状态和覆盖计数。

该会话左手观测 94/150、右手 95/150、双手同时观测 39/150。OBS 帧中的投影总体贴合可见手部；
缺失既包含手未进入画面，也包含局部遮挡、指套和交互造成的 detector/tracklet 召回不足。

当前全批实现曾直接以完整 150 帧作为左右手共同可见分母，并要求左右手各自覆盖至少 95%。
这与既有 `get_potato_chips_0915_001` 开发证据不一致：后者左手只观测 62/379，仍被记录为
`PASS_DEVELOPMENT_HAWOR`。因此本目录不能支持“当前已完成 playing_cards 全部为真实 HaWoR 质量 C”的结论；
必须先恢复独立的 expected-active/visibility denominator，或把原始 HaWoR 完成状态与下游 admission 分开。
