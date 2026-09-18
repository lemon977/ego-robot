# 0915 VST 图像域单会话 A/B

状态：`USER_CONFIRMED_A / MONOCULAR_ONLY`

本目录只回答一个问题：`play_cards_0915_001` 的物理左目在进入单目
HaWoR/SAM3.1 前，是否应沿用当前 `equiDis62 → 90° pinhole` 重映射。它不评价
HaWoR、SAM3.1、Depth 或 Robot 的质量，也不授权恢复批处理。

## 直接复核

- [全片 A/B 视频](0915_VST_IMAGE_DOMAIN_AB_REVIEW.mp4)：150 帧，30 FPS，完整解码通过。
- [六帧对照表](0915_VST_IMAGE_DOMAIN_AB_CONTACT_SHEET.jpg)
- [当前 remap 位移场](0915_VST_IMAGE_DOMAIN_WARP_FIELD.png)

对照表和视频从左到右为：

1. **A：物理左目原像素**。SBS `sourceIndex=1`，只做 2048×1536 到
   1280×960 的等比例面积缩放，不应用畸变参数。
2. **B：已停止的 remap**。同一物理左目应用当前项目实现的
   `equiDis62 → 1280×960 FOV90 pinhole`。
3. **差分增强**。A 与 B 的像素差乘 2.5，仅用于暴露几何变化。
4. **legacy 参考**。现有 processed 单目；证据表明它来自 `sourceIndex=0`
   物理右目，所以不能代替物理左目。

## 数字证据

| 检查 | 全片中位数或结果 | 解释 |
|---|---:|---|
| 当前 remap 位移 P50 | 102.24 px | 不是微小校正 |
| 当前 remap 位移 P95 | 241.02 px | 边缘几何变化很大 |
| 当前 remap 最大位移 | 332.88 px | 当前实现不能未经验证直接作为基线 |
| legacy vs SBS sourceIndex=0 resize-only | PSNR 41.16 dB；相关 0.99898 | legacy 确认来自物理右目 |
| legacy vs SBS sourceIndex=1 resize-only | PSNR 15.81 dB；相关 0.42180 | legacy 不是物理左目 |
| 已存 remap vs 本次重算 remap | PSNR 39.16 dB；相关 0.99901 | 复现了被停止的实际输入路径 |
| 已存 remap vs 左目 resize-only | PSNR 14.15 dB；相关 0.13909 | 两个候选域存在实质差异 |

完整数值位于运行证据
`_run/current/0915_vst_image_domain_ab_v1/attempts/attempt_0001/IMAGE_DOMAIN_METRICS.json`；
任务终态收据为
[`tasks/receipts/0915_VST_IMAGE_DOMAIN_AB_V1_RESULT.json`](../../../../tasks/receipts/0915_VST_IMAGE_DOMAIN_AB_V1_RESULT.json)。

## 目前可以和不可以下的结论

可以确认：

- 被停止的输入确实执行了当前项目自己的非线性 remap；用户看到的弯曲有明确实现来源。
- 该 remap 对 1280×960 图像造成百像素量级位移，不能继续作为默认单目基线。
- legacy processed 单目是 `sourceIndex=0` 物理右目，只能作不同眼参考。
- 下一轮单目 canary 的唯一合理候选是 `sourceIndex=1 + resize-only`，但仍需用户先对
  A/B 画面作视觉确认。

还不能确认：

- 原始 SBS 在物理意义上已经无畸变。随采集保存的本地元数据明确写着
  `pxrcapture_rawfisheye_sbs_hevc` 和 `RAW (distorted) fisheye`。
- 当前 `equiDis62` 投影公式、系数顺序、正反映射方向和 PICO 编码像素域是一致的。
  目前没有随 `.h264` 保存的机器可读 `isRgbDistorted` 标记，也没有项目内固定的官方
  模型约定来证明这一点。
- 双目 Depth 可以直接消费 A。双目矫正需要另做左右目极线、尺度和标定板验证，不能
  从单目视觉确认外推。

PICO 官方 SpatialMP4 工具公开了 `IsRgbDistorted()` 和 RGB distortion model 查询，说明
编码流是否畸变应当由容器/采集证据显式判定；本样本现有输入是裸 `.h264`，该标记不可读。
参考：[PICO SpatialMP4](https://github.com/Pico-Developer/SpatialMP4)。

## 下一步停止线

用户已确认“A明显是对的”。当前只新建一个单会话、物理左目 resize-only 的 HaWoR
canary；SAM3.1 及后续阶段仍需等待该 canary 复核。Depth 保持独立
`NOT_EVALUATED`，不得恢复 220 会话批处理。
