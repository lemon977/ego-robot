# 0915 VST 图像域停止线

状态：`MONOCULAR_A_CONFIRMED / HAWOR_CANARY_ACCEPTED / SAM31_SINGLE_CANARY_ALLOWED`

本页记录 0915 裸手视觉链曾经的图像域停止线及其解除范围。用户在单样本复核中指出，
现有画面出现不应有的弯曲，怀疑原始 VST 像素被再次去畸变；随后确认物理左目
`sourceIndex=1 + resize-only` 为正确单目输入，并接受该输入上的 HaWoR 单样本视觉结果。
这只解除同一会话 SAM3.1 Mask canary 的停止线，不授权 Depth 或 220 会话扩批。

## 已确认的实现事实

- 0915 processed 会话元数据把已发布 1280×960 单目视频声明为
  `rectified=false`、`image_domain_mode=passthrough_scaled_source_domain`。
- 同会话 SBS 为 4096×1536，即每眼 2048×1536；物理左目在
  `camera_params.json` 中为 `sourceIndex=1`。
- 当前 `prepare_0915_physical_left_batch_v1.py` 并非直接使用该眼原像素：它先裁出
  `sourceIndex=1`，再以 `equiDis62` 参数生成映射，通过 `cv2.remap` 转成
  1280×960、90° pinhole，并把产物标为
  `EQUIDIS62_TO_PINHOLE_1280X960_FOV90`。

因此，现有单样本和批量结果确实经过了一次显式重映射。用户观察到的弯曲足以使
这一路径停止，但目前也不能反向断言某组内参本身错误。

单会话 A/B 已完成，直接复核见
[`visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。
当前 remap 在 1280×960 输出上的位移为 P50 102.24 px、P95 241.02 px、最大
332.88 px，属于实质几何变换而非微小修正。现有 processed legacy 单目与
SBS `sourceIndex=0` resize-only 的全片中位 PSNR 为 41.16 dB、亮度相关为
0.99898，确认它是物理右目，不能替代 `sourceIndex=1` 物理左目。

同时保留相反证据：原始会话随附元数据把录制 API 和格式写为
`pxrcapture_rawfisheye_sbs_hevc`、`RAW (distorted) fisheye`。这阻止我们把
resize-only 画面称为物理 pinhole 或已完成外部标定。当前真正缺失的是 PICO
`equiDis62` 的官方系数/方向约定，以及裸 `.h264` 编码像素的独立 distortion flag；
现有公式不能仅凭“文件里有畸变系数”获得 authority。

## 权威影响

- 0915 源数据和现有 processed 发布根均未修改。
- 0916 的 240 会话清洗终态不受影响。
- 由上述重映射视频派生的 HaWoR、SAM3.1、FoundationStereo、Object6D、Clean、
  Contact 和 Robot 结果全部降级为“被隔离的开发证据”，不得称为 0915 基线，
  不得作为扩批或质量结论。
- 已停止 `0915_hawor_full_v1`。停止时状态文件记录 120 个已触达会话，其中
  119 个为旧输入域上的 `FAILED_QUALITY_C`，1 个因停止产生 `FAILED_RUNTIME`；
  这些计数不用于判断 HaWoR 在正确 VST 图像域上的质量。
- 单样本 `play_cards_0915_001` 的现有视频继续保留用于展示问题，不再证明相机、
  HaWoR、SAM3.1 或 Depth 的正确性。

## 后续推进边界

1. A/B 已由用户确认：单目固定为 `sourceIndex=1 + resize-only`。
2. HaWoR 单会话已经人工接受；下一步只授权同一会话的 SAM3.1 Mask canary，不自动
   授权 220 会话扩批。
3. 双目 Depth 继续独立停止，等待左右目极线、尺度及标定板证据；不能沿用单目结论。
4. 若未来恢复矫正，必须固定 PICO 模型约定、系数顺序、正反映射方向和编码域证据，
   禁止再以项目内假设替代采集 SDK/container 的 distortion state。

用户已明确确认“A 明显是对的”。因此 0915 单目候选正式锁定为物理左目
`sourceIndex=1 + resize-only`，禁止沿用已停止的 `equiDis62 → FOV90 pinhole` 路径。
该确认只授权 `play_cards_0915_001` 的单会话 HaWoR canary；SAM3.1、Depth 和 220
会话扩批仍未授权。确认收据见
[`0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json`](../../tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json)。

raw canary 的直接观测为左手 148/150、右手 145/150；随后 `hawor_bounded_v2` 通过
冻结数值门，并以独立短缺口层验证 150/150 的离线视觉连续性。用户已确认视觉无明显
问题；短缺口运行与可视化产物按要求清理，代码、测试、结果及清理收据保留。当前允许
进入一个 `play_cards_0915_001` SAM3.1 Mask canary；Depth、Robot 和全批继续停止。
