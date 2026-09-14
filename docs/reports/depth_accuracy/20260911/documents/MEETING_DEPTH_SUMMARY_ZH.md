# 深度与三维精度会议摘要

日期：2026-09-11  
范围：Raw → HaWoR / FoundationStereo → Object6D → Robot  
结论：视觉链内部闭环已建立；真实物理精度尚未标定

## 已确认

1. **FoundationStereo链路可用于视觉Object6D候选输入。** Raw左右眼、calibration、rectification、640×480视差、`Z=fB/d`、有效域和1280×960 selected-left registration均有可追溯闭包。当前授权仍是`VISUAL_OBJECT6D_CANDIDATE_INPUT`。
2. **各类“深度”含义不同。** HaWoR Z是单目MANO三维估计；Stereo Z是双目可见表面optical-Z；Object6D是Depth+Mask+几何得到的物体位置；Robot z-buffer是CAD渲染深度；contact signed distance是两个估计几何体间的距离。
3. **Chips034右手存在持续大负偏。** `ΔZ=Z_MANO-Z_Stereo`下，signed bias=`-58.90 mm`、MAE=`59.15 mm`、P95=`100.76 mm`。293帧中288帧负偏，204帧`<-50 mm`，最长连续39–213帧共175帧。
4. **整手absolute-Z placement是主要因素，但不是全部。** 每帧只平移HaWoR整手的Z，不改手形和姿态，MAE由`59.15`降到`13.99 mm`，下降约`76.4%`；对齐后P95仍`43.43 mm`，后两个重点窗的部分手指proxy残差仍大。
5. **当前Robot开发代码是world-first。** 每帧使用`joints_3d_world`，固定`T_world_base`做IK，渲染时用`inv(c2w(t))@T_world_base`回到相机域；不是把固定`T_camera_base`与camera joints直接扩展到全片。

## 四手对比

| 会话/手 | signed bias | abs MAE | P50 | P95 |
|---|---:|---:|---:|---:|
| Chips034 左 | +6.87 mm | 14.12 mm | 11.46 mm | 23.97 mm |
| Chips034 右 | **-58.90 mm** | **59.15 mm** | **56.71 mm** | **100.76 mm** |
| Poker042 左 | -16.94 mm | 23.47 mm | 21.09 mm | 46.39 mm |
| Poker042 右 | +4.32 mm | 24.71 mm | 21.24 mm | 55.76 mm |

这些数值是HaWoR表面与Stereo表面的系统间差异，不是相对真实世界的误差。

## 尚未确认

- FoundationStereo在30/50/70/100 cm真实距离下的bias、MAE、P95和时序抖动。
- Chips034右手59 mm分歧应由HaWoR absolute-Z还是Stereo局部/系统误差承担多少。
- HaWoR wrist、MCP、tip相对MoCap/标记点的真实误差。
- Object6D相对已知刚体/AprilTag pose的真实平移和旋转误差。
- c2w相对真实头部/相机运动的尺度、同步、漂移和跳变。
- 相机到Robot base、Robot TCP、法兰/KaiHand安装和真实接触的误差。
- 当前Robot只有development review，`authority=false`、`action_sidecar_published=false`；不能解除HumanEgo Robot分支等待。

## 不能写进汇报的结论

- “FoundationStereo达到毫米级真实精度”。
- “Stereo proxy skeleton是真实手骨架”。
- “59 mm全部是HaWoR错”或“全部是Stereo错”。
- “Object6D plane residual等于真实pose误差”。
- “Robot IK residual等于真实TCP或接触误差”。
- “最终Robot接触精度为X mm”。

## 会议优先观看产物

1. [Chips034右手三栏3D空间对比视频](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_3D_Depth_Comparison.mp4)
2. [只做整手Z平移前后对比视频](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Before_After_Z_Alignment.mp4)
3. [HaWoR joint与Stereo surface-depth proxy视频](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Proxy_Skeleton.mp4) / [关键帧PNG](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Proxy_Skeleton.png)
4. [四手与Chips右手时间分段表](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/tables_v1/HAWOR_STEREO_DIFFERENCE_TABLE.png)
5. [五类深度总览图](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/diagrams_v1/DEPTH_TYPES_OVERVIEW_ZH.png)
6. [内部一致性与外部真值边界图](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/diagrams_v1/INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png)

颜色统一：青色=Stereo可见表面proxy；橙色=原始HaWoR/MANO；绿色=只施加median-Z correction后的MANO。绿色不是第三个传感器，也不是真值。

## 下一步

1. 按[外部深度标定实验计划](EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md)采集30/50/70/100 cm×中心/左/右共12个平面片段，第一次得到真实Stereo Z误差表。
2. 对c2w做静态世界参考测试，验证world-first链是否真实抵消头动，而不仅是矩阵闭环。
3. 用已知刚体/AprilTag建立Object6D pose真值。
4. 完成Robot相机外参、TCP、法兰/KaiHand安装标定，再建立接触误差预算。

## 详细证据

- [当前深度与三维精度状态](DEPTH_ACCURACY_CURRENT_STATUS_ZH.md)
- [c2w→Robot只读代码链审计](C2W_ROBOT_COORDINATE_AUDIT.md)
- [外部深度标定实验计划](EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md)
- [原始定量证据边界报告](../../tasks/control/runs/20260911_quantitative_accuracy_evidence_audit_v1/REPORT_ZH.md)
- [HaWoR与Stereo表面QA报告](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/REPORT_ZH.md)

