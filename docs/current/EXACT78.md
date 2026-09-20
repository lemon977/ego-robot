# Exact78 当前执行入口

计划：[PLAN.md](PLAN.md)  
机器状态：[STATUS.json](STATUS.json)

## 当前稳定事实

- 冻结分母为 Chips 78 + Poker 78，共 156 个历史会话。
- 旧 Exact78 FoundationStereo/Depth 因重复 lens-undistortion 已撤权。
- Chips023 旧 PICO 对照保留为历史工程参考，不是外部 wrist 真值。
- 本轮只训练 Chips/Poker × Raw/Robotized 四个 H50 future-2D Visual Aux，不扩成 Policy。

## 当前任务

1. E0：合法 Raw/Robotized pair、真实更新、保存重载、H50/suffix-invariance。
2. E1：Chips023 encoded-domain Stereo 三段 preflight 和多源 2D/3D/Depth 对照。
3. E2：四模型固定预算训练、共同 checkpoint、公平 A/B、stay-put/constant-velocity 基线。

E1 不阻塞 E2。实际 task ID、attempt、结果与 blocker 只从 [STATUS.json](STATUS.json) 读取。

