# 0915 与旧批量基线的比较口径

状态：CURRENT COMPARISON CONTRACT

## 比较对象

用户所称“以前确认的基线”固定指 receipt 绑定的 frozen exact78 批量基线，不指近期
`play_cards_0915_001` 的单样本 SAM/Removal 实验。

- frozen cohort：156 个会话，78 Chips + 78 Poker，共 56,663 帧；
- 0915 Poker 单样本的旧任务对照：`play_cards_0902_042`；
- 0915 Chips 单样本的旧任务对照：`get_potato_chips_0902_103`；
- 当前浅层入口：[`TWO_TASK_BASELINE_20260917`](visuals/TWO_TASK_BASELINE_20260917/README_ZH.md)；
- 视频闭包：[`VIDEO_RECEIPT.json`](visuals/TWO_TASK_BASELINE_20260917/VIDEO_RECEIPT.json)；
- 帧映射闭包：[`FRAME_MAP_RECEIPT.json`](visuals/TWO_TASK_BASELINE_20260917/FRAME_MAP_RECEIPT.json)；
- 冻结复现收据：[`BASELINE_E2E_PAIR_V1_COMPLETION.json`](../../tasks/receipts/BASELINE_E2E_PAIR_V1_COMPLETION.json)。

`archive/` 仍不是当前事实源。上述历史大视频只因 current registry 和浅层 receipt 对路径、
字节、SHA、帧数及顺序作了绑定，才可作为冻结视觉回归证据。该收据明确
`fresh_model_inference_old_baseline=false`：它证明旧产物可复核，不证明当前代码重新推理可
逐字节复现。

## 允许比较的项目

新 0915 结果必须与相同任务的旧对照比较：

1. 原始 RGB 是否保持 VST encoded resize-only 域，没有重复 lens undistortion；
2. Hand/Role Mask 的覆盖、身份稳定、消失/重现和逐帧闪烁；
3. Clean 的人物/设备残留、背景损伤、可见任务物体保护和时序连续性；
4. Robot 离线可视化中的相对任务动作与可见碰撞诊断。

旧 Role Mask 含 hand/tracker，而 0915 为裸手且没有 tracker，因此角色类别不能逐项等同；
只能比较共同的轮廓覆盖、时序稳定和任务物体保护。

## 禁止继承的旧结论

- 旧 FoundationStereo Depth 和依赖它的 Object6D 曾为 58/58，但因对无畸变 VST 像素
  重复执行 lens undistortion，当前 authority 已撤回为 0 PASS / 58 BLOCKED；
- 旧 Clean 58 条只证明结构、修改域、帧数和解码闭包，不证明接触边界或隐藏物体恢复正确；
- 旧 Robot 仅为离线可视化，`control_ground_truth=false`、
  `physical_deployment_authorized=false`、`training_eligible=false`；
- 不得用旧 Depth/Object6D 数值替代新的 encoded-domain Canary 质量门。

## 本轮停止线

新的 FoundationStereo 先只跑 `play_cards_0915_001`。只有 Depth 在原物理左目坐标完成
逐像素 RGB 对齐、内部一致性和完整解码质量门，才允许三张牌的字段级 Planar Object6D。
任何视觉改善均不能单独提升毫米精度、Contact 或 Robot 控制 authority。
