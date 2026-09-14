# HumanEgo 当前入口

本目录保留当前项目需要的 HumanEgo 模型、dataloader、H50 future-2D
辅助监督接口、推理逻辑与测试。当前算法、代码闭包和限制以
[`CURRENT_BASELINE_REGISTRY_V2.json`](../docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)
中的 `humanego_aux`、`humanego_policy` 两项为准，运行数量只读
[`CURRENT_PROJECT_STATUS_MIN.json`](../docs/governance/CURRENT_PROJECT_STATUS_MIN.json)。

当前边界：

- Visual Aux 目标是 `future_2d_xy` / `future_2d_valid`，不是机器人控制动作。
- `visual_robot_trajectory_sidecar` 的 `control_ground_truth=false`，不得冒充
  `real_robot_action_sidecar`。
- 目前没有已授权 Visual Aux checkpoint；最终 Policy 仍因缺真实 Robot action
  而 `BLOCKED_EXTERNAL`。
- 历史 `NOW/` 训练、daemon 和可视化入口均已退休，不得作为当前复现入口。

当前训练只能由事实账本选择的 Task Packet 启动；不得从旧 README、历史 checkpoint
目录名或聊天记录推断 readiness。

上游许可证见 [`LICENSE`](LICENSE)，全项目第三方说明见
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。
