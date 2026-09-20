# B2 Poker044 Depth → Object：当前部署入口

状态：`DEPLOYED_EXECUTED_AND_INDEPENDENTLY_AUDITED_REVIEW_ONLY`。

本目录的唯一当前机器入口是 `DEPLOYED_CLOSURE.json`。它精确绑定已登记算法、执行结果和独立审计。

当前结论：

- Poker044 可见 finite patch 的数值与来源闭包已完成。
- `physical_card_identity=UNKNOWN_UNBOUND`。
- `face_identity=UNKNOWN_UNBOUND`。
- `review_only=true`，`consumer_allowed=false`。
- 不具有 Contact、Robot、Clean、训练、控制或部署 authority。

`PREPARATION_RESULT.json` 仅是历史准备收据，不是当前终态。
`history/local_preparation/HISTORY_PREDEPLOYMENT_INSTRUCTIONS_DO_NOT_RUN.md` 仅保留注册前的历史命令，不得再执行。

已封存结果路径：

```text
_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/
  packages/B2_DEPTH_TO_OBJECT_POKER044/RESULT.json
  packages/B2_DEPTH_TO_OBJECT_POKER044/INDEPENDENT_AUDIT/RESULT.json
docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/B2_DEPTH_TO_OBJECT_POKER044/INDEX.json
```

禁止用时间新旧或 `latest` 目录重选结果；必须通过 closure 中的完整路径和 SHA 消费。
