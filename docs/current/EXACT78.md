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

## V3.1 实际终态

- E0：`BLOCKED_INPUTS`。六个非归档候选根共发现 483 个 raw candidate，但没有当前、非归档且
  已授权的冻结 156 cohort，也没有合法 Raw/Robotized pair manifest；不得把 483 自动缩成 156，
  也不得从 `archive/` 静默恢复权威。
- E1：`BLOCKED_EXTERNAL_ASSET`。Chips023 encoded SBS/camera 资产存在，但当前输出域的 K/P、主点、
  baseline 和视差符号权威未闭合，因此不允许 `LOCAL_STEREO_METRIC_DEV`。
- E2：未启动。GPU 使用 0、模型调用 0、训练更新 0。

下一步需要显式提供或授权当前冻结 156 清单和合法 pair manifest；Stereo 公制消费还需要当前
encoded 输出域的 K/P/baseline 证据。两个 blocker 相互独立。
