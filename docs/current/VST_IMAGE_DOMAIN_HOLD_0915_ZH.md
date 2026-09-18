# 0915 VST 图像域停止线

状态：`USER_HOLD_IMAGE_DOMAIN_UNRESOLVED`

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

因此，现有单样本和批量结果确实经过了一次显式重映射。现有文件证据只能说明工厂
标定参数随数据保留，不能证明 VST 编码像素仍处于需要应用该参数的原始镜头域。
用户观察到的弯曲足以使这一路径停止，但目前也不能反向断言某组内参本身错误。

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

1. “原始 VST RGB”的权威文件与眼别：SBS 中物理左目 `sourceIndex=1`，还是
   processed 中已有的 1280×960 legacy 单目。
2. 编码像素是否已由 VST/采集 SDK 做过镜头校正；不能只依据随附工厂
   `equiDis62` 参数推断。
3. 固定同一帧并排比较原像素路径与现有重映射路径，检查直线、手部比例、视场和
   边缘形变；在用户确认前不运行模型。
4. 新任务只能选择一个明确图像域，禁止先去畸变再对已校正像素重复映射。

当前没有可执行的 0915 算法任务。下一步只围绕上述单一图像域问题讨论和取证；
确认后必须新建有限 canary 任务，不能复活已取消的批量任务。
