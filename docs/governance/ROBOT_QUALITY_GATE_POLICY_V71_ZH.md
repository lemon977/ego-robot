# Robot 质量门分层策略 V7.1

本页冻结 Robot Geometry、Robotized RGB 与训练候选的门语义。实时数量仍以
`CURRENT_PROJECT_STATUS_ZH.md` 为准。

## 结论

“逐帧严格复刻人手姿态”不是 Robot 几何硬门。人手与 KaiHand 的骨长、关节数、
活动范围和腕臂可达域不同，因此允许机器人在保持动作语义和连续性的前提下采用不同轨迹。

当前 R7.3 的处理方式是：保留硬几何门；把 arm endpoint 残差、finger bone/tip
方向残差和 `all observed rows pass` 降为软诊断。软门失败的轨迹可进入
`POSE_ONLY_VISUAL_ROBOT_REVIEW_READY`，但不能因此自动获得 Robot authority。

## A. 必须保留的硬门

| 门 | 原因 | 适用范围 |
|---|---|---|
| finite state / proper SE(3) | 防止 NaN、反射矩阵和损坏轨迹 | 全部 Robot 产物 |
| 真实 URDF 关节限位 | 防止生成模型本身不可能实现的姿态 | Geometry、Visual、训练 |
| 左右手性、link/法兰/root 闭包 | 防止装反手、装错坐标链 | Geometry、Visual、训练 |
| 缺失观测保持 UNKNOWN | 禁止用插值伪造未观测的 hand/wrist | Geometry、训练 ledger |
| 非相邻 link 自穿透检查 | 防止手指穿掌、左右机器人互穿 | Geometry、Visual |
| 统一 z-buffer 和 part/link provenance | 保证手臂、手掌、手指、物体按深度遮挡 | Robotized RGB |
| 帧身份、因果输入和 SHA 闭包 | 防止错帧、未来信息泄漏和版本混用 | 训练 |

这些门不能因“训练只需要视频”而删除；否则会把结构错误、穿模或未来泄漏写进训练集。

## B. 视觉连续性门

速度和加速度上限继续作为 Robotized 视频的连续性门，目的是防止单帧跳变和明显闪烁，
不是物理速度安全认证。阈值必须按 FPS 和轨迹采样合同解释；后续可用真实分布分位数校准，
但在完成跨 Chips/Poker regression 前不热改当前 R7.3。

当前数字合同为：arm 每帧最大速度 `0.12 rad`、最大加速度 `0.06 rad`；hand
每帧最大速度 `0.08 rad`、最大加速度 `0.06 rad`。它们只约束生成视频连续性，
不能写成真机控制安全限值。

## C. 只作软目标的门

- Robot wrist/tool 与人手 wrist 的逐帧位置和旋转残差。
- KaiHand 指骨方向与 MANO 指骨方向的逐帧差。
- 指尖方向误差。
- `pose_branch_all_observed` 与 `anatomy_all_observed` 的全片零失败要求。
- 对人手轨迹的逐帧精确重合。

这些指标用于候选排序、困难帧定位和训练有效区选择。单独失败不得写成
`FAILED_HARD_GEOMETRY`，也不应阻止 pose-only 视觉候选。

## D. Contact 与物理部署另设门

Robot 几何可行不等于手物接触正确。`METRIC_CONTACT_ROBOT` 仍需独立 Object6D、
合法 Contact evidence、penetration/contact distance 和对象实例门。物理部署还需真实
adapter CAD、TCP、安装以及 world-to-base 标定。数字 retarget 轨迹始终标记
`control_ground_truth=false`。

## 当前证据边界

首批 6 条 Chips 候选均通过 hard geometry，自穿透、有限值、关节限位、时序和
UNKNOWN provenance 门；6/6 均未达到逐帧严格人手姿态匹配。因此这 6 条可以继续做
统一 z-buffer、Occlusion 和 causal Visual Aux 候选，但仍是 development evidence，
不是 Robot authority。该结果支持“严格模仿门不应阻塞视觉训练候选”，不支持删除硬门。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/batch_001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/batch_002/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/candidates/`
