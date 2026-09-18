# 0915 FoundationStereo encoded-domain 单会话

状态：`PASSED`（仅开发级 Object6D 候选输入）

- 会话：`play_cards_0915_001`
- 视频：[`0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4`](0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4)
- 帧数：150/150；30 FPS；完整解码
- 输入：SBS 中物理左目 `sourceIndex=1`、物理右目 `sourceIndex=0`，只 crop + resize
- 禁止项：未交换左右相机，未做 lens undistortion，未做 lens remap
- 模型 adapter：两眼同时水平镜像以适配 FoundationStereo 的正视差方向；模型输出随后水平反镜像回原物理左目 640×480
- 模型域主点：`cx_mirror = width - 1 - cx_physical`
- 最终 Depth：使用原物理左目 K，坐标域为 `PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z`
- RGB/Depth 回域证明：150 帧最大通道误差 0、错位像素 0、坐标往返最大误差 0 px

内部质量门全部通过：最终有效覆盖门 150/150，几何有效比例中位数 0.95983，
左右一致比例中位数 0.94666，Depth 中位数帧间变化 P90 为 0.00445 m。

这些指标只证明同一 encoded pixel domain 内的算法一致性，不是外部毫米精度验证。
30/50/70/100 cm 实测板尚未提供，因此 `external_accuracy=UNVERIFIED`。本结果仅授权给
同一会话的新 Object6D 可观测性 canary，不授权 Contact、Robot、批处理或部署。

旧 exact78 批量 Depth 因曾对已经无畸变的 VST 编码像素重复去畸变而撤权，不能作为
Depth 数值真值。旧批量只在原始 RGB、Mask、Clean 与 Robot 离线视觉层按
[`0915_OLD_BATCH_BASELINE_COMPARISON_ZH.md`](../../0915_OLD_BATCH_BASELINE_COMPARISON_ZH.md)
所列 receipt 口径进行对照。
