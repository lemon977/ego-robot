# HaWoR MANO 可见表面 vs FoundationStereo optical-Z 固定 QA

日期：2026-09-11  
范围：Chips034 / Poker042 正式 baseline canary；六会话只做同 session 输入 inventory，未执行 QA。

## 结论

已实现并验收一个 fail-closed 的固定 QA：在 selected-left camera 像素网格上，重建 MANO 三角形可见表面，将 FoundationStereo rectified optical-Z 按同 session `REGISTRATION_AUTHORITY.npz` 转到同一相机域，计算：

`delta-Z = Z_mano_visible_surface - Z_stereo_registered`

Chips034 和 Poker042 均通过支持量 admission 门。这只说明有足够有效像素计算该代理指标，**不是精度 PASS**。MANO 与 FoundationStereo 都是估计值，该结果不提升 Depth、Contact、Object6D、Robot 或训练 authority。

## 实现合同

- 由 bounded NPZ 的 `root_orient + hand_pose + betas + root_translation` 调用 `run_mano/run_mano_left` 重建 778 顶点网格；固定 MANO faces。重建 joints 与 NPZ joints 闭合最大误差为 0.0 mm。
- 主表面使用双手合并的 nearest-triangle z-buffer，不用投影顶点抽样。关节中心不进入主指标。
- MANO 轮廓和 role mask 各侵蚀 3 px。Stereo 支持为 `valid & finite & 0.10<=Z<=3.0m`，再要求中心+四邻域至少 3 个有效样本且中心与中位数相差 <=20 mm。
- Depth frame schema 无 confidence，因此明确记为 `ABSENT_IN_DEPTH_FRAME_SCHEMA_NOT_FABRICATED`。未按 delta-Z 做异常值剔除。
- 遮挡处理只能声明为“两手 nearest surface + current visible role mask + 侵蚀”；剩余物体/桌面遮挡为 `UNKNOWN`。
- Chips034 使用 current 左/右 per-side role mask。Poker042 当前 authority 只有一张 combined visible role-removal mask：同一张 mask 分别与 MANO 左/右 z-buffer label 求交，不冒充 per-side observed mask；其 RESULT SHA 为 `2380c2aa...e982`，manifest SHA 为 `cabef669...327`。

## 数值（mm）

| session / side | frame-balanced signed bias | frame-balanced abs MAE | abs P50 | abs P95 | pixel-weighted abs MAE / P95 |
|---|---:|---:|---:|---:|---:|
| Chips034 left | +6.87 | 14.12 | 11.46 | 23.97 | 14.19 / 30.94 |
| Chips034 right | -58.90 | 59.15 | 56.71 | 100.76 | 52.37 / 106.61 |
| Poker042 left | -16.94 | 23.47 | 21.09 | 46.39 | 24.18 / 46.52 |
| Poker042 right | +4.32 | 24.71 | 21.24 | 55.76 | 24.39 / 55.91 |

Chips034 右手差异显著更大，因此应作为回归排查窗；但没有外部 GT 时不能归因为 HaWoR 或 Stereo 单方错误。

## Admission 门（不是精度门）

每侧至少 `max(10, ceil(20%*N))` 帧包含 >=100 样本，全会话 >=5000 样本，平均 eroded-support coverage >=0.05。不足时终态必须为 `HOLD_INSUFFICIENT_SUPPORT`。

## 产物和验证

每会话产出 `FRAME_METRICS.json`、`SURFACE_DELTA_METRICS.npz`、左右 delta-Z 曲线、surface 差异热图、`RGB_OVERLAY.mp4`和不可变 `RESULT.json`。两条 MP4 均通过 `ffmpeg -xerror`，且 ffprobe 帧数分别精确为 293/171，尺寸 1920x480。产物 SHA 已逐项重算闭合；pytest 4/4 PASS。

六会话的 current same-session HaWoR/Depth/registration/role-mask 输入已完成只读 inventory，但本次没有扩大执行，也不干扰已冻结的六会话视频审阅包。

机读入口：`RESULT.json`、`INVENTORY.json`、`AGENT_REVIEW.json`、`SHA256SUMS.txt`。
