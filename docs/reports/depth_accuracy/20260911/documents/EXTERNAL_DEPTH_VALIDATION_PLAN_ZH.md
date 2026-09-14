# FoundationStereo 外部深度标定实验计划

状态：`PLANNED_NOT_EXECUTED`  
目的：测量当前相机、标定、图像域和FoundationStereo组合在实际工作距离下的外部物理误差，而不是继续做算法内部自洽比较。

## 1. 要回答的问题

在当前硬件和当前冻结处理链下，平面真实距离为30、50、70、100 cm时，FoundationStereo输出的可见表面 optical-Z 的bias、MAE、P95和时序抖动分别是多少？它在中心、左侧和右侧视野是否有系统差异？

实验完成前，不使用“5 mm级”“10 mm级”或“几十毫米级”描述真实精度。

## 2. 保持冻结的算法链

实验必须复用当前正式链，不另换图像域或临时K：

```text
Raw SBS 4096×1536
→ left/right原始鱼眼2048×1536
→ 冻结calibration + rectification
→ 640×480 FoundationStereo disparity
→ Z=320×0.0637716504026918/d
→ valid门
→ registration到selected-left 1280×960
```

每次实验必须记录以下文件的绝对路径、字节数和SHA256：相机合同、左右相机内外参、rectification/remap、FoundationStereo代码与权重、Depth runner、registration authority、Raw视频和最终Depth manifest。

## 3. 所需设备

- 当前双目相机与固定支架；实验期间禁止手持。
- 尺寸足够覆盖目标ROI的硬质平面标定板，优先哑光棋盘格/AprilTag板；避免纯白、反光或柔性纸张。
- 外部距离参考：优先经校验的激光测距仪；次选刚性钢尺/卷尺。必须记录工具型号、读数分辨率和测量不确定度。
- 水平仪或可重复的平面支架，用于减少板面倾斜。
- 环境温度与照明记录；相机开机预热至少10分钟。

## 4. 距离与视野组合

固定相机，移动标定板。建议以相机坐标原点/厂家定义的光学基准为距离起点；若实际只能从相机外壳测量，必须同时测量并记录外壳基准到光学中心的offset。

| 距离 | 中心 | 左侧视野 | 右侧视野 | 每个位置静止时长 |
|---:|---:|---:|---:|---:|
| 30 cm | 必做 | 必做 | 必做 | 5–10秒 |
| 50 cm | 必做 | 必做 | 必做 | 5–10秒 |
| 70 cm | 必做 | 必做 | 必做 | 5–10秒 |
| 100 cm | 必做 | 必做 | 必做 | 5–10秒 |

共12个静态片段。每个片段建议至少150帧；前30帧作为曝光/自动参数稳定期，不纳入主统计，但仍保留原始数据。

视野位置定义应使用selected-left 1280×960像素坐标并在记录中固化，例如：中心`u≈640`、左侧`u≈320`、右侧`u≈960`；板中心v尽量保持`≈480`。不要只写“左边/右边”。

## 5. 采集步骤

1. 固定相机和支架，记录序列号、焦距/模式、分辨率、fps、曝光、增益、白平衡和是否自动。
2. 记录相机图像到底是原始鱼眼、设备端去畸变还是设备端warp；禁止只凭文件名推断。
3. 相机预热，采一段空场Raw用于检查同步与掉帧。
4. 将标定板放到第一个真实距离，用外部测距工具从固定基准测量至少三次；保存原始读数，不只保存平均值。
5. 调整板面使其近似正对相机。若允许更严谨，使用PnP估计板面法向并报告倾角；主ROI的真值应是沿相机optical-Z的平面交点，而不只是斜距。
6. 分别在中心、左侧、右侧采集5–10秒；期间板和相机均不得移动。
7. 对30、50、70、100 cm重复。
8. 完整保存Raw SBS，不只保存解码PNG或最终Depth。
9. 用当前冻结FoundationStereo链离线运行；不按结果调阈值或筛掉困难距离。
10. 生成全量机读表、中文图和异常帧索引。

## 6. ROI与真值定义

不能直接把整张图都当平面。先在selected-left域检测棋盘格/AprilTag板边界，再内缩至少20像素或板短边5%，排除深度边缘和背景混合区。将ROI反映射到Depth网格时遵循冻结像素中心约定。

真实距离建议定义为标定板平面在相机光轴方向上的optical-Z。若测距仪给的是斜距，需用已测板姿态换算，或将这一偏差计入真值不确定度。报告必须同时保存：

- `truth_distance_raw_readings_mm`
- `truth_distance_mean_mm`
- `truth_instrument_resolution_mm`
- `truth_reference_point`
- `board_normal_camera`
- `board_tilt_deg`
- `truth_optical_z_mm`

## 7. 每个片段输出指标

对每一有效像素`i`和每一帧`t`：

```text
error_i,t = Z_stereo_i,t - Z_truth_plane_i,t
bias        = mean(error)
MAE         = mean(abs(error))
P50 / P95   = percentile(abs(error), 50 / 95)
valid_rate  = valid pixels / fixed ROI pixels
```

时序指标不能把所有像素混在一起掩盖抖动。先计算每帧ROI median Z，再对该时间序列报告：

- `temporal_std_mm`
- frame-to-frame median absolute difference
- P95 frame-to-frame jump
- 无效帧数和最长连续无效区间

同时按ROI中心和边缘环分别统计，以暴露视差边缘误差。

## 8. 最终汇报表

| 距离 | 位置 | 真值optical-Z | Stereo median Z | Bias | MAE | P95 | temporal std | valid rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 300 mm | 中/左/右 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 500 mm | 中/左/右 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 700 mm | 中/左/右 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 1000 mm | 中/左/右 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

建议额外输出：误差随距离曲线、视野位置对比、每段时间序列、ROI valid热图、误差直方图和最差P95帧。

## 9. 数据目录与机读合同建议

```text
external_depth_validation_YYYYMMDD/
├── CAPTURE_CONTRACT.json
├── calibration/
├── distance_0300mm/{center,left,right}/raw/
├── distance_0500mm/{center,left,right}/raw/
├── distance_0700mm/{center,left,right}/raw/
├── distance_1000mm/{center,left,right}/raw/
├── depth_outputs/
├── per_frame_metrics.csv
├── aggregate_metrics.csv
├── RESULT.json
├── AGENT_REVIEW.json
└── REPORT_ZH.md
```

`CAPTURE_CONTRACT.json`至少包含相机身份、时间、温度、曝光参数、真值工具、三次读数、板身份、距离基准、图像域、K/R/t/baseline SHA和操作者备注。

## 10. 预注册质量门

本计划不提前发明“必须5 mm”之类的成功阈值。实验运行前只冻结数据完整性门：

- 12个位置全部采集且Raw可完整解码。
- 左右眼帧数、时间戳和相机身份一致。
- calibration、runner、weights和真值记录都有SHA。
- ROI在全片可重现，分母固定，不能按Depth好坏动态缩小。
- 每个位置都报告invalid率，不能只统计有效且表现好的像素。
- 算法参数对12个位置保持一致。
- 失败位置保留并进入总表，不补拍覆盖原结果。

获取真实数据后，再根据Object6D和Robot任务容差制定可接受阈值。视觉Object6D、粗抓取和精细接触可能需要不同门，不能共用一个未经验证的“毫米级”标签。

## 11. 后续扩展

完成平面Z实验后，按优先级继续：

1. 倾斜平面与不同纹理，检查边缘和视差梯度。
2. 已知尺寸三维刚体/AprilTag，测Object6D平移和旋转真值。
3. MoCap或标记点手部采集，测HaWoR wrist/MCP/tip真值。
4. Robot TCP、法兰和KaiHand安装标定，并用实体接触事件校验signed distance。

只有这些外部误差项都完成，才可以建立Robot contact的真实误差预算。

