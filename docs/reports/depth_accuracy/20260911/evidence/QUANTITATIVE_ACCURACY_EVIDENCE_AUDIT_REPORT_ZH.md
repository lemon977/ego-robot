# 深度、位姿与 Robot 接触精度证据审计

日期：2026-09-11（Asia/Shanghai）  
结论：当前项目有较完整的数值闭环、时序门和视觉 Grade 证据，但没有外部物理真值，因此不能给出所问的真实 MAE/P95 或可加和 Robot 接触误差预算。

## 结论表

| 项目 | 外部真值结论 | 当前能报告的数 |
|---|---|---|
| HaWoR wrist Z MAE/P95/drift | **NOT MEASURED** | 只有历史 HaWoR–PICO 估计器差异和内部平滑/坐标闭环 |
| HaWoR fingertip Z MAE/P95/drift | **NOT MEASURED** | 只有历史 HaWoR–PICO 五指尖 pooled 差异；没有 fingertip GT |
| FoundationStereo 0.3/0.5/0.7 m 绝对深度误差 | **NOT MEASURED** | 只有公式重算误差、视差敏感度和无真值的图像/局部表面诊断 |
| depth→selected-left 注册像素误差 | **NOT MEASURED（对外部控制点）** | 有 3 帧 SIFT 内部一致性：Chips034 max P90=0.5221 px；Poker042 max P90=0.6078 px |
| Object6D 平移/旋转误差 | **NOT MEASURED** | 有投影、平面、时序限速、直接观测比例；均不是 pose GT error |
| Robot 物理接触误差 | **NOT MEASURED** | 旧 CAD SDF/碰撞诊断为 HOLD；没有力/触觉/外部 tracker 真值 |
| Robot 接触误差预算 | **NOT AVAILABLE** | 输入数值跨单位、相关且缺少不确定性分布/Jacobian，不能相加或 RSS |

## HaWoR Z：外部误差没有测，历史代理差异可以复现

权威只读审计 `hawor_actual_processing_audit_v1/RESULT.json`（SHA `1f540f...`）明确写出：真实 2D、真实 3D、真实 fingertip depth、contact/occlusion accuracy 全为 `null`。`joints_2d` 是估计 3D 通过 K 自投影；`joints_3d_world` 是相同估计通过逐帧 c2w 换坐标，因此不能拿二者互相验证真实精度。

为界定已有历史证据，本次只读复算了 20260905 legacy optimized HaWoR 与 PICO/OpenXR `keypoints_3d_camera` 的 optical-Z 差异。它们是两个不同估计器、关节定义也不同；capture-world 还存在 `notAccurate` 证据。因此下表是 **proxy disagreement**，不是 HaWoR MAE/P95：

| Session / side | wrist abs mean / P95 (mm) | 五指尖 pooled abs mean / P95 (mm) | wrist residual drift (mm) |
|---|---:|---:|---:|
| Chips027 L | 80.814 / 89.557 | 62.118 / 79.534 | -22.679 |
| Chips027 R | 67.508 / 106.849 | 76.411 / 126.984 | -6.826 |
| Poker017 L | 77.592 / 107.272 | 91.785 / 115.337 | +46.823 |
| Poker017 R | 84.098 / 128.104 | 64.172 / 117.844 | +11.840 |

这里 drift 的固定定义是 `末10帧 wrist residual 均值 - 首10帧 wrist residual 均值`；两段首末时间分别相隔 11.667 s 与 6.833 s。它不是相对静止靶标 drift，也不是 current exact78 算法真误差。

HaWoR 的 `world_recompute_max=3.3e-5 mm`、投影重算 `6.7e-5 px` 等只是实现闭合；bounded-v2 的 wrist step、acceleration 和 candidate-vs-raw 也只是时序/偏离门。

## FoundationStereo：0.3/0.5/0.7 m 绝对误差均未测

当前校正公式是 `Z=fx*B/d`，其中 `fx=320 px`、`B=0.0637716504026918 m`。现有 `2.47e-7–2.50e-7 m` 是由冻结 disparity 重算 Z 的浮点闭合，不是物理深度误差。

项目没有 0.3、0.5、0.7 m 的已知距离靶标或同步验证深度。因此三个距离的 MAE/P95 均为 **NOT MEASURED**。可计算的一阶视差敏感度如下，但不能替代误差条：

| Z | disparity | 1 px 的局部 Z 变化 | 0.25 px 的局部 Z 变化 |
|---:|---:|---:|---:|
| 0.3 m | 68.023 px | 4.410 mm | 1.103 mm |
| 0.5 m | 40.814 px | 12.251 mm | 3.063 mm |
| 0.7 m | 29.153 px | 24.011 mm | 6.003 mm |

这只说明误差传播斜率。当前没有证据表明 FoundationStereo 的实际 disparity error 是 1 px 或 0.25 px。

## 注册：必须区分 current baseline P90 与 historical P95

034/042 current baseline 的 `REGISTRATION_RESULT.json` 是各 3 帧 SIFT/feature 内部一致性：

- Chips034（SHA `ed745c...`）：frames 0/146/292，median 0.1308–0.1490 px，P90 0.4642–0.5221 px。
- Poker042（SHA `bf6225...`）：frames 0/85/170，median 0.1303–0.1742 px，P90 0.4902–0.6078 px。

另有 20260907 historical corrected-v2 两帧审计报告 P95：Chips027 为 0.8248/0.8493 px，Poker017 为 1.0564/0.8863 px。它与上面的 3 帧 P90 不是同一会话、版本或统计量，不能混写。

这些数来自 remapped stereo-left 与 selected-left 图像的匹配特征，不是独立标注控制点，所以只能证明 sampled image-registration consistency。对外部真值的 pixel MAE/P95 仍是 **NOT MEASURED**。

## Object6D：Grade B 与限速不等于 pose accuracy

Chips034 与 Poker042 的 Object6D baseline 都是 Grade B，并显式 `robot_contact_authorized=false`。没有 MoCap、测量治具或独立物体 pose 真值，所以 translation/rotation MAE/P95 均为 **NOT MEASURED**。

例如 Poker042 observed-only 结果记录 raw 最大逐帧平移 0.03356 m、旋转 179.276°，随后被限制到 0.03 m/12°。这个“正好达到门值”的数证明 rate limiter 工作，不证明姿态误差只有 30 mm/12°。Chips027 historical HOLD 中的 surface MAD P95=12.13 mm、world speed P99=1.995 m/s 也只是拟合/时序诊断。

## Robot 接触：没有可合法合成的 error budget

逐项状态如下：

- HaWoR：物理贡献 **NOT MEASURED**；只有上面的历史 PICO 代理差异。
- Depth：物理贡献 **NOT MEASURED**；公式闭合和视差导数不是实际误差。
- Object6D：平移/旋转贡献 **NOT MEASURED**；视觉 Grade B 不授予 contact authority。
- Registration：对外部真值的 px 误差与换算后的接触 mm 均 **NOT MEASURED**；缺少控制点、工作距离传播和相关性模型。
- FK/IK：物理端点误差 **NOT MEASURED**。六会话 development review 的最坏 target residual 为 2.842 mm/1.927°，只是解算器/FK 对自身目标的内部闭合，`authority=false`，不是接触结果。
- Mount：物理平移/旋转 **NOT MEASURED**。固定文件状态是 `PINNED_DEVELOPMENT_ONLY_NOT_FORMAL_CALIBRATION_AUTHORITY`；43.282439 mm 是视觉几何推导的 connector span，不是测得的 `T_tool_hand`。代码中的 ±2.55 mm 也明确是 visual-only bookkeeping envelope，不是硬件计量。

历史 contact 诊断同样不能替代误差预算：Chips027 named-pad contact+nonpenetration 为 0/24，active-pad SDF min/median/max=-1.499/17.177/159.798 mm；Poker017 bounded attempt 为 0/24，index SDF min/median/max=2.277/5.557/11.869 mm。两者都是 HOLD 下的 CAD/Object6D 几何结果，不是现实接触观测。

所以不能把 HaWoR 的 estimator disagreement、registration px、Object6D rate limit、FK residual 和 mount proxy 直接相加，也不能做 RSS。缺少的不是算术，而是每一项对外部真值的标定不确定性、相关性和共同坐标下的传播模型。

## 要获得真实预算，最小新增测量

1. 用 0.3/0.5/0.7 m 的已知平面/深度靶标测 FoundationStereo bias、MAE、P95，并保存温度/曝光/纹理条件。
2. 用独立控制点或标定板测 depth→selected-left pixel residual，不再只用自动匹配特征。
3. 用 MoCap/AprilTag 多视角或测量治具获取手腕、指尖与物体 pose 真值。
4. 实测 flange→KaiHand rigid transform 及重复装配分布；用外部 tracker 验证 FK endpoint。
5. 用触觉/力或导电接触标签同步标注 pad→object signed distance，再估计各项相关性并传播到 contact error。

完整数值、定义、路径和 SHA 见同目录 `RESULT.json`。本审计未修改任何六会话视频、Robot sidecar、current/canonical 指针或模型产物。
