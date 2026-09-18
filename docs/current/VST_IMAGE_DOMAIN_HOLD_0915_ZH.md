# 0915 VST 图像域停止线

状态：`ALL_ENCODED_VST_VIDEO_ALREADY_UNDISTORTED_CONFIRMED / NO_LENS_REMAP`

本页记录 0915 裸手视觉链曾经的图像域停止线及其解除范围。用户在单样本复核中指出，
现有画面出现不应有的弯曲，原因是已经无畸变的 VST 编码视频像素又被应用了一次
镜头去畸变。项目所有当前 VST 视频统一按“编码像素已经无畸变”处理：按 `sourceIndex`
裁出物理眼后只允许 resize，不得把 `camera_params.json` 中保留的 `equiDis62` 字段再次
应用到解码帧。用户确认收据见
[`VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json`](../../tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json)。

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

原始会话随附元数据中的 `pxrcapture_rawfisheye_sbs_hevc`、`RAW (distorted) fisheye`
以及 `equiDis62` 系数保留为采集侧 provenance；它们不授权对当前编码视频再次执行镜头
去畸变。这里仍不把 resize-only 画面提升为外部标定 pinhole，也不由此获得 metric Depth
精度。若 Depth 需要极线对齐，必须另建“编码域、零镜头畸变”的双目标定任务；极线变换
不能夹带 `equiDis62` lens-undistortion。

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
- `0915_foundationstereo_single_session_canary_v1` 又消费了 `equiDis62` remap；其不可变
  执行终态仍为 `REJECTED_QUALITY`，但当前证据 authority 进一步降为
  `WITHDRAWN_WRONG_IMAGE_DOMAIN`，不得作为 Depth/Object6D 起点。
- 复核还确认旧 exact78 FoundationStereo worker 同样调用 `make_map/remap_pair`。因此原
  Depth `58/58` 的当前授权计数改为 `0 passed / 58 blocked`，依赖该 Depth 的 Object6D
  也从 `58/58` 改为 `0 passed / 58 blocked`。文件作为历史证据保留，但不能再进入
  Contact、Occlusion 或 Robot 的当前消费链；依赖它们的 Contact/Occlusion 假设声明已撤回。

## 后续推进边界

1. A/B 已由用户确认：单目固定为 `sourceIndex=1 + resize-only`。
2. HaWoR 单会话已经人工接受；下一步只授权同一会话的 SAM3.1 Mask canary，不自动
   授权 220 会话扩批。
3. 双目 Depth 继续独立停止，等待编码域左右目极线、尺度及标定板证据；任何候选都不得
   对解码视频应用 `equiDis62` 或其他镜头去畸变。
4. 将来若需要 stereo epipolar alignment，它必须与 lens-undistortion 分开建模、使用
   编码视频域标定，并先发布 raw resize-only 对照与人工复核；禁止恢复旧 remap。
5. 只有新的 encoded-domain Depth 先通过后，才能新建 Object6D successor；旧 58 条结果
   不允许作为 warm start、回归真值或当前质量基线。

用户先确认“A 明显是对的”，随后进一步确认所有 VST 编码视频本来就没有畸变。因此
当前图像域不再只是一条单目显示偏好，而是所有视频消费者的硬边界：物理眼裁切加
resize-only，lens-undistortion 一律禁止。早期单目确认收据继续保留为过程证据：
[`0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json`](../../tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json)。

raw canary 的直接观测为左手 148/150、右手 145/150；随后 `hawor_bounded_v2` 通过
冻结数值门，并以独立短缺口层验证 150/150 的离线视觉连续性。用户已确认视觉无明显
问题；短缺口运行与可视化产物按要求清理，代码、测试、结果及清理收据保留。当前允许
进入一个 `play_cards_0915_001` SAM3.1 Mask canary；Depth、Robot 和全批继续停止。
后续 Depth 必须新建编码域 successor，不得复活已终态的 FoundationStereo remap 任务。
