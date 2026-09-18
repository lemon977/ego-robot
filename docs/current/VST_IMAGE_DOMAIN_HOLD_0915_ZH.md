# 0915 VST 图像域停止线

状态：`RESEARCHED_PENDING_USER_CONFIRMATION`

本页是 0915 裸手视觉链当前最高优先级的停止说明。用户在单样本复核中指出，
现有画面出现不应有的弯曲，怀疑原始 VST 像素被再次去畸变。该问题解决前，
不得继续 HaWoR、SAM3.1、Depth、Object6D、Clean、Contact 或 Robot 扩批。

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

## 重新执行前必须共同确认

1. 用户复核 A/B，确认 `sourceIndex=1 + resize-only` 是否与采集时看到的物理左目一致。
2. 单目确认只授权一个单会话 HaWoR canary，不自动授权 SAM3.1 或 220 会话扩批。
3. 双目 Depth 继续独立停止，等待左右目极线、尺度及标定板证据；不能沿用单目结论。
4. 若未来恢复矫正，必须固定 PICO 模型约定、系数顺序、正反映射方向和编码域证据，
   禁止再以项目内假设替代采集 SDK/container 的 distortion state。

单会话 A/B 任务已以 `BLOCKED_EXTERNAL` 封账，当前没有可执行的 0915 算法任务。
下一步只等待用户对浅层 A/B 的确认；确认后必须新建有限 canary 任务，不能复活已
取消的批量任务。
