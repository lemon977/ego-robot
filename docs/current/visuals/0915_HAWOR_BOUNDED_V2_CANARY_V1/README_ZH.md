# 0915 resize-only HaWoR bounded_v2 单会话复核

状态：`PASS_NUMERIC_NEEDS_HUMAN_REVIEW`

这是 `play_cards_0915_001` 的完整 150 帧 HaWoR 基线审阅结果。输入仍是用户确认的
物理左目 `sourceIndex=1` resize-only 图像，没有执行去畸变/remap。先运行 raw HaWoR，
再运行当前登记基线 `hawor_bounded_v2`；未启动 SAM3.1、Depth、Robot 或批处理。

## 直接复核

- [raw 与 bounded_v2 全片叠加视频](0915_HAWOR_RAW_VS_BOUNDED_V2_REVIEW.mp4)：
  150 帧、30 FPS、1280×960，完整解码通过。
- [六帧总览](0915_HAWOR_RAW_VS_BOUNDED_V2_CONTACT_SHEET.jpg)
- [机器结果收据](../../../../tasks/receipts/0915_HAWOR_BOUNDED_V2_CANARY_V1_RESULT.json)

视频中细暗线是 raw HaWoR，亮线是 bounded_v2；青色为解剖左手，紫色为解剖右手。
两者使用同一像素坐标和固定尺度。缺失帧仍为缺失，没有插值补手。

## 数值变化

| 指标 | 左手 raw → bounded_v2 | 右手 raw → bounded_v2 |
|---|---:|---:|
| 观测帧 | 148 → 148 | 145 → 145 |
| wrist step P95 | 18.88 → 5.07 mm | 21.76 → 14.25 mm |
| 全关节二阶差分 P95 | 30.29 → 3.09 mm | 31.74 → 5.59 mm |
| 骨长 CV 最大值 | 0.02860 → 0.02051 | 0.08272 → 0.06344 |
| 2D 重投影 P95 | 8.75 px | 8.99 px |

采用完整更新 `alpha=1.0`，身份切换计数为 0。最大参数更新为：root 平移
19.57 mm、root 旋转 3.08°、pose 旋转 8.36°、beta L2 0.75，均在冻结上限内。
所有数值门通过，但仍需人工观看全片；本结果不自动授权扩批或 Robot 消费。

## 运行环境

使用仓库发布的 `_run/current/environments/hawor-py310-v1`，并由
`src/chaoyang/ops/hawor_python.sh` 启动。在会话标签相同的复跑中，环境入口修复前后的
NPZ 和视频均逐字节一致；随后只把复用 runner 遗留的 `0910` 视频标题改正为真实的
`0915` 会话标签，算法 NPZ 仍逐字节一致。GPU 调用为 0。
