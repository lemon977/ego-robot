# 0915 Stereo 与 Interaction v0a 单会话 CPU 证据

状态：`PASSED`

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
- 终态：`PASS_GPU_DEPTH_ADMISSION`。

这里的 rectification 使用同会话 `equiDis62` 内参和由 10 个估计帧得到、再由 10 个
不重叠 held-out 帧验收的图像匹配旋转；不是把 factory 外参直接宣称为已验证矫正。
它只允许 FoundationStereo 在同一单样本上继续验证。后续 Depth 必须保留逐帧 invalid
与 LR/时序门，不能用池化 PASS 覆盖上述坏帧。外部毫米精度仍为 `UNVERIFIED`；P95 与
空间覆盖也没有全面优于 raw，因此不能把预检写成 Depth 质量通过。

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
