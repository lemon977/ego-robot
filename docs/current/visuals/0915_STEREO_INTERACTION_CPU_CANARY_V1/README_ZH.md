# 0915 Stereo 与 Interaction v0a 单会话 CPU 证据

状态：`RAW_RESIZE_DIAGNOSTIC_RETAINED / RECTIFIED_BRANCH_WITHDRAWN`

会话：`play_cards_0915_001`，150 帧。该任务由同一 coordinator 启动两个互不相交的
CPU writer lane；两条 lane 都已独立封账，未使用 GPU，也未修改源数据。

- [合并审阅图](0915_STEREO_INTERACTION_CPU_REVIEW.png)
- 完整机器产物：
  `_run/current/0915_stereo_interaction_cpu_canary_v1/attempts/attempt_0001/`

## Stereo 预检

- 物理眼顺序：左目 `sourceIndex=1`，右目 `sourceIndex=0`。
- 原始 SBS：4096×1536；每眼：2048×1536；150/150 帧可解码。
- raw resize-only 垂直匹配误差：median 1.955 px、P90 2.804 px、P95 3.121 px。
- 本会话 rectified 候选的池化误差：median 0.628 px、P90 1.901 px、P95 3.881 px。
  12/12 抽样帧只表示每帧至少有 15 个 robust matches，不表示逐帧误差都通过：frame 81
  的 P90/P95 为 7.376/18.999 px，frame 94 的 P90 为 6.422 px，超过当前 5/8 px
  诊断阈值。
- 同会话 factory baseline 为 0.0637716504 m。
- 原运行终态为 `PASS_GPU_DEPTH_ADMISSION`；该 admission 现已撤销。

用户已确认 VST 编码视频本来没有畸变，因此这里消费 `equiDis62` 的 rectified 分支属于
重复去畸变，连同 `PASS_GPU_DEPTH_ADMISSION` 一并撤销。raw resize-only 的匹配统计仍可
保留为诊断数据，但不单独提供 metric stereo 标定。新的 Depth 前置必须是编码视频域、
零 lens-undistortion 的双目标定；当前任务不得复活。

## Interaction v0a

Interaction v0a 只消费已接受的 HaWoR、SAM3.1 v5 角色证据，以及 processed 内
`entities.tactile`。没有读取 PICO26、controller pose 或 `trackingData` 手部结果。

150 帧 × 10 个指尖 × 3 张牌共 4500 个配对状态：

| 证据 | 正状态 | 负状态 | UNKNOWN |
|---|---:|---:|---:|
| 2D 邻接 | 59 | 3416 | 1025 |
| 2D 接近 | 1378 | 1896 | 1226 |
| 2D 共动 | 804 | 2470 | 1226 |
| 触觉支持假设 | 16 | 3459 | 1025 |

图中绿色为 2D 邻接、蓝色为非邻接、灰色为 `UNKNOWN`，黄色菱形仅表示触觉支持假设。
`UNKNOWN` 包括 mask 未知及 HaWoR 非直接观测证据，不能解释为“手或物体不存在”。

本结果的 authority 为 `DEVELOPMENT_WEAK_EVIDENCE_ONLY / IMAGE_2D_ONLY`。它没有
relative-Z、遮挡顺序、接触真值、力、Object6D 或 Robot authority，也不授权全批处理。
