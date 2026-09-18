# 0915 Encoded-domain Stereo Preflight V1

本目录只显示物理左/右目按 `sourceIndex` 裁切后 resize 的原始编码像素；未使用镜头去畸变、相机模型 remap、Depth warm-start 或 GPU。

- 决策：`PASS_DIRECT_FOUNDATION_INPUT`
- 150 帧完整解码：是
- frame 81/94：已单独诊断
- 本结果只决定后续 stereo 输入路径，不构成深度或毫米精度证明。
