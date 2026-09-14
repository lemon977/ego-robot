# AI 当前状态读取协议

回答任何当前进度、数量、精度、authority、运行状态或下一任务前，必须：

1. 读取并验证 `CURRENT_STATUS_RECEIPT.json`。
2. 验证 receipt 所绑定的 authority、task state、中文状态页 bytes/SHA。
3. 检查 governance revision、generation id 与 freshness。
4. 阅读 `CURRENT_PROJECT_STATUS_ZH.md`；需要细节时再读取对应 RESULT/REPORT。
5. 不从历史对话、目录名、mtime、`latest/final/current` 名称或单独视频补全事实。
6. 没有证据时写“当前事实账本未提供”；所有推断必须标为推断。

验证命令：

```bash
cd /mnt/workspace/code/chaoyang
python -m tools.governance.validate_governance_state
```

若结果为 `STATUS_CONFLICT`、`STALE` 或 `SUSPECTED_DEAD_WORKER`，停止状态推断和 authority 晋升，先运行恢复审计。

## 长任务的动态计数

Stage-0 recovery 只冻结启动时快照。其后 Wave0 Clean guardian 每次 heartbeat 必须从冻结的
58 行 selection 与已经原子发布的逐会话 `RESULT.json` 重新得到轻量终态闭包，并同步更新：

- `waves.wave0_clean_passed`；
- `waves.wave0_clean_pending`；
- Clean stage 的 `passed / grade_c / running / blocked`；
- Clean stage 对应的 selection 与逐会话终态证据引用。

不得从可变 `STATE.json` 的一个裸计数直接晋升 authority，也不得手工修改生成的 Markdown 或
`CURRENT_PROJECT_STATUS_MIN.json`。当前 guardian row 只计一次 `running=1`；未完成且不是当前 row
的会话计入 `blocked`。如果 selection SHA、唯一性、会话身份或终态合同不闭合，heartbeat 继续
保持进程活性，但跳过计数晋升并输出 fail-closed warning，等待下一次审计修复。
