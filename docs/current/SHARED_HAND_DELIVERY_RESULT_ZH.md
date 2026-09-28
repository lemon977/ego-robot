# Human→Robot 共享手映射、Robot 与 Clean 有限任务结果

`产品技术质量：0/4；采用：0/4；Robot合格范围：无新增全片，固定窗4块中3块数值可行但未采用；Clean合格范围：无，LaMa方法质量未评估；开发Kai22合格H50窗口：0`

这是本任务的唯一结果入口。状态轴分别记录执行、结构、质量、改善与采用；文件存在或可解码不代表质量通过。

## 直接查看

- Poker 既有171帧Robot第三人称全片（复用、质量拒绝）：[视频](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/lanes/robot/POKER_171_FIXED_THIRD_PERSON_CANDIDATE_V3.mp4)
- 本轮Robot 80–111联合轨迹结果（新增、固定窗拒绝）：[数值收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/POKER_080_111_JOINT_TRAJECTORY_FIX1.json)
- Poker Clean 76–91旧结果与V3实现无效结果（新增、16帧）：[对照视频](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3/POKER_076_091_OLD_VS_LAMA_V3_REJECTED.mp4)
- Clean固定帧总览：[contact sheet](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3/POKER_076_091_OLD_VS_LAMA_V3_CONTACT_SHEET.png)
- Sensor097共享局部手映射结果（新增、质量拒绝）：[结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/hand_data/RESULT_REPAIR2.json)

## 实际结论

| 分支 | 实际执行 | 真实增量 | 当前硬失败 | 采用 |
|---|---|---|---|---|
| Robot | 80–111帧、64个侧帧，4个联合块 | 3个块数值可行 | 右侧80–95搜索未获满足20 mm/15°硬门的窗口；93–95仍失败；未扩171 | 否 |
| Clean | ProPainter内部追踪；LaMa V2/V3各16帧真实推理 | 定位到内部模型后又定位到ONNX输出域适配错误 | 两轮实现修复耗尽；V3错误地对已处于0..255域的输出再次乘255，白洞饱和；方法质量未评估；未扩171 | 否 |
| Sensor/Data | Sensor097全165帧、330个侧项，共享局部目标与Kai22求解 | 方向误差P50约118.58°降至13.48°；物理左侧方向165/165通过 | 夹合0/330、声明碰撞0/330，数值合格帧与H50窗口均为0；未运行学习消费者 | 否 |

Clean的后续动作已经精确到一项，但不属于本任务：只移除错误的输出`×255`，其他输入、mask、传播、模型、门和分母全部保持不变。上游示例对输入除以255，而对输出直接转为`uint8`，没有再次乘255。[实际示例](https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX/blob/main/app.py)

## 权威收据

- Robot：[RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/RESULT.json)
- Clean：[TERMINAL_RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3/TERMINAL_RESULT.json)
- Data：[RESULT_REPAIR2.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/hand_data/RESULT_REPAIR2.json)
- 总结果：任务终态发布后由 `_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/RESULT.json` 提供。

全部结果仅为 `OFFLINE_VISUAL`/离线开发证据：`training_eligible=false`、`control_ground_truth=false`、`physical_deployable=false`、`external_metric_authority=false`。
