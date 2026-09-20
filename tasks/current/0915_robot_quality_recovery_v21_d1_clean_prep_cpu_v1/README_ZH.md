# D1 Clean 准备：当前部署入口

状态：`DEPLOYED_AND_EXECUTED_CLEAN_REMAINS_BLOCKED`。

本目录的唯一当前机器入口是 `DEPLOYED_CLOSURE.json`。深层终态由下列不可变收据给出：

```text
_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/
  packages/D1_CLEAN_PREP/D1_TERMINAL_INDEX.json
```

当前浅层可视化只允许消费：

```text
docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP_V2/INDEX.json
```

`D1_CLEAN_PREP/` 是已标记 wrapper receipt 路径缺陷的 V1 历史证据，不是当前入口。

## 当前结论

- 已验证 560 帧 `M_remove ⊆ M_write ⊆ M_flow`、`UNKNOWN == M_write`和任务物体保护域。
- Candidate 与 Raw 逐像素相同；改写像素数为 0。
- 没有生成 fresh inpainting，所以不是 Clean 质量通过，也没有 Contact、训练、控制或部署 authority。

## 历史准备文件

顶层 `RESULT.json` 和 `TEST_RESULT.json` 仅是
`HISTORICAL_LOCAL_PREPARATION_ONLY`，不是远端部署终态。准备期 checksum 已隔离到
`history/local_preparation/`，不得在当前任务目录就地执行或用来否定 deployed closure。

禁止重跑已封存的 V1/V2 命令；新的 fresh fill 必须使用新候选、新输出目录和新收据。
