# 0915 物理左目 resize-only HaWoR 单会话 canary

状态：`EXECUTION_PASSED / SESSION_FAILED_QUALITY_C / USER_REVIEW_REQUIRED`

本 canary 只运行 `play_cards_0915_001`。输入从 SBS 裁出物理左目
`sourceIndex=1`，仅由 2048×1536 缩放为 1280×960；`rectified=false`、
`remap_applied=false`。未启动 SAM3.1、Depth 或任何批处理。

## 直接复核

- [全片 HaWoR 叠加视频](0915_HAWOR_RESIZE_ONLY_REVIEW.mp4)：150 帧、30 FPS，完整解码通过。
- [六帧关键帧](0915_HAWOR_RESIZE_ONLY_CONTACT_SHEET.jpg)
- [机器结果收据](../../../../tasks/receipts/0915_HAWOR_RESIZE_ONLY_CANARY_V1_RESULT.json)

蓝色为左手、红色为右手；只绘制 HaWoR 直接观测帧，缺失帧没有补造。

## 结果

| 指标 | 左手 | 右手 |
|---|---:|---:|
| 直接观测帧 | 148/150 | 145/150 |
| 观测覆盖 | 98.67% | 96.67% |
| 最长缺失 | 1 帧 | 2 帧 |
| 关节在画面内 | 93.85% | 81.58% |
| 骨长 CV 最大值 | 0.02860 | 0.08272 |
| wrist step P99 | 27.71 mm | 31.04 mm |
| 单侧严格门 | PASS | FAIL |

整体 `numeric_mask_gate_pass=false`。右手首个明确问题是关节在画面内比例低于 0.90，
同时骨长 CV 0.08272 略高于 0.08；这与画面下缘/右缘附近的手部裁切有关。左手通过
当前冻结数值门。高检出率不能覆盖右手的几何失败，因此该会话保持
`FAILED_QUALITY_C`，不得自动进入 SAM3.1。

## 边界

- 本结果证明采用 A 图像域后 HaWoR 单会话推理约 86 秒即可完成，并恢复双手高覆盖；
  它不证明所有关节位置准确。
- scaled factory K 仅作为 HaWoR 开发输入；用户对 A 的确认不把裸像素提升为物理
  pinhole authority。
- 下一步必须由用户复核全片骨架。没有新的明确确认，不登记 SAM3.1 或全批任务。
