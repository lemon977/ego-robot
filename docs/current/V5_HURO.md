# 支线4：HuRo核心公平比较

本轮已登记[四支线收敛任务](COMPLETION_20260928_ZH.md)。以下为前驱冻结证据；新进展以[机器状态](STATUS.json)及本轮lane结果为准，不能将旧数值当新验收。

前驱2026-09-24终态：无活动任务；最新共享手任务未新跑HuRo。下述冻结比较不能用于后来更换输入的代表会话。

## 已执行与未采用

官方核心适配和腕目标测试、007/031全片求解已经执行。后续用同一固定URDF独立重算Local/HuRo FK；不是首次接入任务。

原HuRo硬限位失败007为230/756侧帧、031为84/102；共同运动学有效526与18侧帧用于残差统计，原分母不得删去。007该交集内HuRo完整旋转P50约32°，仍不合格。

固定16帧窗口派生有界q消除了该窗口的限位问题，但属于后处理研究，不替换原HuRo，不代表全片、碰撞、旋转或产品通过。没有方法胜者。

## 后续任务边界

新授权后在官方核心适配层检查原始求解限位与完整腕旋转，保存原q和派生结果。固定输入、资产、安装、相机、时间轴、掩码和评价器；优先失败短窗，不直接重跑全片。Local新增输入不得与旧HuRo直接比较。

- [同输入独立FK结果](../../_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json)
- [窗口派生修复](../../_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/compare/huro_limit_repair_window_v1/RESULT.json)
- [历史与视频索引](../plans/INDEX_ZH.md)
