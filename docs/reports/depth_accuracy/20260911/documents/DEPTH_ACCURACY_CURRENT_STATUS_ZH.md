# 当前深度与三维精度状态

更新时间：2026-09-11  
适用范围：`Raw → HaWoR / FoundationStereo → Object6D → Robot` 视觉链  
结论等级：当前状态与证据边界说明；不是外部标定证书

## 1. 一句话结论

当前 FoundationStereo 的公式、单位、左右帧身份、深度网格和 selected-left registration 已完成内部闭环，可以作为 `VISUAL_OBJECT6D_CANDIDATE_INPUT`。但项目尚未完成已知真实距离、手部3D真值、物体pose真值、Robot TCP/安装真值测量，因此不能宣称 Stereo、HaWoR、Object6D 或最终接触达到毫米级物理精度。

Chips034 右手存在持续、同方向的 HaWoR MANO 表面与 Stereo 可见表面 Z 分歧：293帧中288帧为负偏，204帧小于`-50 mm`，最长连续区间为39–213帧。每帧只给 HaWoR 整手施加一个 median-Z 平移后，surface MAE 从`59.15 mm`降到`13.99 mm`，说明 absolute-Z placement 是主要因素；对齐后P95仍为`43.43 mm`，说明局部手形、姿态、遮挡边缘或Stereo表面仍有不可忽略的残差。

## 2. “深度”不是一个量

| 名称 | 来源 | 数学/物理含义 | 当前可用范围 |
|---|---|---|---|
| HaWoR Z | 单目RGB预测MANO参数，再由MANO forward得到mesh和21点 | 学习模型估计的手部三维；包含整体placement与手形/姿态 | 人手运动、MANO几何、Robot retarget候选；不是测距真值 |
| FoundationStereo Z | rectified左右图视差 | 相机坐标中的可见表面 optical-Z，`Z=f_xB/d` | 视觉Object6D候选输入；不是遮挡后表面或解剖关节 |
| Object6D Z / near-far | Depth + task-object Mask + 几何估计 | 物体可见点、平面/几何模型和pose的空间位置 | 视觉物体状态候选；内部拟合残差不是真实pose误差 |
| Robot z-buffer | Robot CAD、FK、相机与光栅化 | 数字Robot模型在渲染相机中的最近表面深度 | 遮挡合成；不是传感器测得的真实Robot表面 |
| contact signed distance | Robot fingertip pad几何与Object6D几何 | 两个估计几何体间的有符号距离 | 接触候选/诊断；没有TCP、安装和物体真值时不能称真实接触距离 |

会议图：[`DEPTH_TYPES_OVERVIEW_ZH.png`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/diagrams_v1/DEPTH_TYPES_OVERVIEW_ZH.png)。

## 3. 为什么各阶段分辨率不同

| 图像域 | 分辨率 | 用途 | 不能混用的原因 |
|---|---:|---|---|
| 原始SBS视频 | 4096×1536 | 同步左右眼原始容器 | 一帧横向包含两眼，不是单眼相机平面 |
| 原始单眼鱼眼 | 2048×1536 | 标定与去畸变/rectification输入 | 有鱼眼畸变，不能直接配selected-left的针孔K |
| selected-left针孔RGB | 1280×960 | HaWoR、Mask、Object6D、Robot overlay的主视觉域 | 是设备/项目选定的左眼针孔域，K为`fx=fy=640,cx=639.5,cy=479.5` |
| Stereo rectified临时图 | 1280×960/eye | 先把左右鱼眼映射到共极线针孔域 | rectified域与selected-left域旋转/采样不同，仍需registration |
| FoundationStereo / Depth网格 | 640×480 | 模型推理、视差和`depth_m`存储 | 为半分辨率推理；K同步缩放为`fx=fy=320,cx=319.5,cy=239.5` |
| ProPainter Clean推理 | 960×720 | 视频补全的算力/显存折中 | 属于合成视觉域，不是Depth几何authority |

Depth半分辨率像素中心与selected-left全分辨率像素中心的约定为：

```text
u_full = 2 * u_depth + 0.5
v_full = 2 * v_depth + 0.5
```

因此不能把640×480 depth用普通`resize`后直接当1280×960深度真值，也不能把不同图像域的K混用。registration必须显式包含半分辨率到全分辨率的像素中心变换，以及rectified-left到selected-left的相机旋转。

## 4. 当前 FoundationStereo 深度如何得到

```text
Raw SBS 4096×1536
  → 拆成 left/right 2048×1536
  → 使用冻结相机模型去畸变与双目rectification
  → 得到共极线对齐的左右针孔图
  → 缩放到640×480并同步缩放K
  → FoundationStereo预测正水平视差 d=u_left-u_right
  → valid门：finite、d>0.25 px、对应点在右图域内、0.1m<Z<3m
  → Z=f_x*B/d，其中f_x=320 px，B=0.0637716504026918 m
  → 保存 disparity_px/depth_m/valid/scaled_intrinsics
  → registration到1280×960 selected-left视觉域
  → 与task-object Mask相交后供Object6D使用
```

公式重算与保存的`depth_m`误差约`2.5e-7 m`，只证明代码和文件中的公式一致，不能解释为传感器有`0.00025 mm`物理精度。

Stereo只观测当前左眼可见表面。细手指边缘、左右遮挡边界、反光、低纹理、重复纹理和运动模糊都可能产生局部视差错误；被遮挡的物体背面和解剖关节中心没有被Stereo直接测量。

## 5. registration当前结论

Depth网格先通过固定像素中心变换升到selected-left尺度，再通过冻结相机几何映射到selected-left针孔域。当前会话内部SIFT检查的最坏P90约为：

- Chips034：`0.5221 px`
- Poker042：`0.6078 px`

这说明当前Depth图像域与selected-left图像域的内部对齐没有出现大幅像素错位。它不等于外部标定板控制点误差，也不能单独证明Z方向物理准确。

## 6. HaWoR表面与Stereo表面的全片比较

定义：

```text
ΔZ = Z_MANO_visible_surface - Z_registered_Stereo_surface
```

负值表示HaWoR MANO表面比Stereo可见表面更靠近相机。P50/P95为`|ΔZ|`的分位数；这些是两个估计系统的差异，不是任一系统相对真实世界的MAE。

| 会话/手 | signed bias (mm) | abs MAE (mm) | P50 (mm) | P95 (mm) |
|---|---:|---:|---:|---:|
| Chips034 左 | +6.87 | 14.12 | 11.46 | 23.97 |
| Chips034 右 | **-58.90** | **59.15** | **56.71** | **100.76** |
| Poker042 左 | -16.94 | 23.47 | 21.09 | 46.39 |
| Poker042 右 | +4.32 | 24.71 | 21.24 | 55.76 |

完整表格：[`HAWOR_STEREO_DIFFERENCE_TABLE.csv`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/tables_v1/HAWOR_STEREO_DIFFERENCE_TABLE.csv) / [`PNG`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/tables_v1/HAWOR_STEREO_DIFFERENCE_TABLE.png)。

## 7. Chips034右手：整手Z偏移还是局部结构错误

### 7.1 全片事实

- 负偏帧：`288/293`
- `<-30 mm`：`270/293`
- `<-50 mm`：`204/293`
- 最长连续`<-50 mm`：第`39–213`帧，共`175`帧
- 原始frame-balanced surface MAE：`59.15 mm`
- 只做per-frame median-Z correction后MAE：`13.99 mm`
- 对齐后P95：`43.43 mm`

所以它不是少量边缘离群点。整只手沿Z持续同方向分离是主要问题；但单一Z平移不能消除手掌/手指所有局部误差。

### 7.2 三个重点窗

| 帧窗 | 整手median offset | Z对齐后whole MAE/P95 | palm MAE/P95 | Stereo support |
|---|---:|---:|---:|---:|
| 20–60 | -49.79 mm | 7.81 / 24.07 mm | 6.32 / 16.25 mm | 约100% |
| 90–120 | -68.94 mm | 18.46 / 60.20 mm | 16.69 / 59.92 mm | 约100% |
| 180–210 | -71.23 mm | 16.89 / 57.04 mm | 17.14 / 64.23 mm | 约100% |

后两个窗口中，中指/无名指等proxy区域残差明显大于手掌；180–210窗口小指没有满足固定定义的可见支持，因此保留为N/A而不是插值。区域由“MANO表面顶点到最近MANO关节proxy”划分，用于工程分区，不是解剖真值分割。

机读区域结果：[`CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/tables_v1/CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json)。

## 8. 三条直观视频如何阅读

统一颜色：青色=`Stereo visible-surface proxy`；橙色=`原始HaWoR/MANO`；绿色=`仅沿相机Z做median correction后的MANO`。绿色不是第三个传感器、不是Ground Truth，也不会写回任何上游产物。

1. [`Chips034_Right_3D_Depth_Comparison.mp4`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_3D_Depth_Comparison.mp4)：左栏RGB+MANO；中栏X-Z/Y-Z侧视；右栏逐帧median/P50/P95/coverage和历史曲线。
2. [`Chips034_Right_Before_After_Z_Alignment.mp4`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Before_After_Z_Alignment.mp4)：同一帧原始分离与只做整体Z平移后的表面残差。
3. [`Chips034_Right_Proxy_Skeleton.mp4`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Proxy_Skeleton.mp4) / [`关键帧PNG`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/media_v1/Chips034_Right_Proxy_Skeleton.png)：在HaWoR joint 2D射线附近取Stereo有效深度中位数并反投影。连线表示joint到可见表面的Z差，不表示Stereo测到了关节。

三条视频均为293/293帧、1920×720、30 fps，已全片解码。

## 9. Object6D当前能说明什么

当前Depth与task-object Mask可用于估计可见点云、近远深度、平面或受限几何pose。已有plane residual、Mask内有效率等指标只回答“点云和所选模型内部是否自洽”，不能回答物体在真实世界中的绝对位置误差。

当前exact78扩展状态中，26个会话具备校正双目标定且两类Mask可join；已完成22个Depth Grade B结果和36个独立物理Object6D child Grade B结果。另有59个会话缺标定，不能用猜测K/baseline生成metric Depth。数量状态应以批次实时STATE为准，本文数字是本专项冻结快照。

## 10. 内部一致性与外部真实精度

### 已完成的内部证据

- `Z=fB/d`文件重算闭环
- 左右帧、标定、分辨率与单位闭包
- selected-left registration的内部SIFT像素对齐
- Object6D点云/平面/Mask内几何残差
- Robot数字模型中的IK、FK、关节限位、速度与加速度残差
- HaWoR camera/world字段的`joints_world ≈ c2w × joints_camera`代码闭包

### 尚未完成的外部真值

- 30/50/70/100 cm已知距离下的Stereo Z bias/MAE/P95
- HaWoR wrist、MCP、tip相对MoCap/标记点/已知几何的误差
- registration相对外部控制点的误差
- Object6D相对AprilTag/刚体真值的平移和旋转误差
- Robot TCP、法兰、KaiHand安装和相机到Robot基座的实测标定误差
- Robot pad到真实物体表面的接触误差与时间同步误差

会议图：[`INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png`](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/diagrams_v1/INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png)。

## 11. 当前允许与禁止的结论

允许：

- FoundationStereo链路内部闭环，可作为视觉Object6D候选输入。
- Chips034右手HaWoR与Stereo存在持续大负偏，整体Z placement解释了大部分平均差异。
- median-Z correction后仍有局部结构残差，需要分别检查HaWoR手形/姿态和Stereo遮挡/边缘。

禁止：

- “FoundationStereo真实精度是X mm”。
- “Stereo proxy skeleton是真实21点手骨架”。
- “Chips034右手的59 mm全部是HaWoR错误”或“全部是Stereo错误”。
- 把Stereo Z直接覆盖HaWoR Z并称为校正真值。
- 由内部plane residual或IK residual推导“Robot最终接触精度X mm”。

## 12. 证据入口

- [深度真实精度与误差边界](../../tasks/control/runs/20260911_quantitative_accuracy_evidence_audit_v1/REPORT_ZH.md)
- [HaWoR MANO表面与Stereo深度QA](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/REPORT_ZH.md)
- [端到端复现总文档](RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)
- [本专项任务状态](../../tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1/TASK_STATUS_ZH.md)

