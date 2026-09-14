# 深度与三维精度独立汇报包

这是2026-09-11深度专项的项目根目录便利副本。原始authority、运行产物和正式文档均保留在原位置；本目录采用复制而不是移动，避免破坏已有路径与SHA引用。

## 建议阅读顺序

1. [一页会议摘要](documents/MEETING_DEPTH_SUMMARY_ZH.md)
2. [当前深度与三维精度状态](documents/DEPTH_ACCURACY_CURRENT_STATUS_ZH.md)
3. [Chips034右手三栏3D空间对比视频](videos/Chips034_Right_3D_Depth_Comparison.mp4)
4. [整手Z平移前后对比视频](videos/Chips034_Right_Before_After_Z_Alignment.mp4)
5. [四手和时间分段表](images/HAWOR_STEREO_DIFFERENCE_TABLE.png)
6. [c2w→Robot只读审计](documents/C2W_ROBOT_COORDINATE_AUDIT.md)
7. [外部深度标定实验计划](documents/EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md)

## 目录内容

- `videos/`：三条正式293帧视频，以及一条median-Z对齐辅助视频。
- `images/`：proxy skeleton关键帧、四手表、五类深度图、内部/外部精度图。
- `tables/`：CSV、三重点窗区域指标JSON及表格结果。
- `documents/`：本专项四份正式文档及端到端复现总文档。
- `evidence/`：最初的定量精度边界报告和HaWoR–Stereo QA报告/RESULT。
- `receipts/`：任务状态、最终收据、媒体收据与原始SHA清单。
- `scripts/`：视频、表格、会议图和最终验证的生成脚本。

## 视频颜色

- 青色：FoundationStereo在当前视角可见的表面深度proxy。
- 橙色：原始HaWoR/MANO表面或关节。
- 绿色：只沿相机Z应用逐帧median correction后的HaWoR/MANO。

绿色不是第三个传感器，也不是Ground Truth。Stereo proxy skeleton是HaWoR 2D关节射线附近的可见表面采样，不是Stereo测得的解剖关节。

## 当前最重要结论

- Chips034右手HaWoR与Stereo的signed bias为`-58.90 mm`，不是少数边缘点造成的瞬时异常。
- 只做整手median-Z correction后，surface MAE由`59.15 mm`降到`13.99 mm`，说明absolute-Z placement解释了大部分平均差异。
- 对齐后P95仍为`43.43 mm`，局部手形、姿态、遮挡边缘或Stereo表面误差仍存在。
- FoundationStereo当前只授权为`VISUAL_OBJECT6D_CANDIDATE_INPUT`；尚无外部真值，不能宣称毫米级物理精度或Robot接触精度。
- Robot当前development代码是world-first坐标链，但仍无current Robot authority或HumanEgo可消费action sidecar。

## 完整性验证

运行：

```bash
python scripts/verify_package.py
```

它会用`PACKAGE_MANIFEST.json`重新校验本目录所有已登记文件的字节数和SHA256。正式运行产物的原始收据仍以`receipts/FINAL_RECEIPT.json`为准。

