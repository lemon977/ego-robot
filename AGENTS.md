# Chaoyang agent execution protocol

1. 先运行 `python -m tools.governance.validate_governance_state`；再读 `CURRENT_RC1_STATUS_MIN.json` 和当前 `TASK_PACKET.json`。只有非 RC1 任务才回退到 `CURRENT_PROJECT_STATUS_MIN.json`。
2. 默认只读 Task Packet 的 `read_set`；初始最多8个文件、20条搜索结果和80行日志。失败时才按原因扩展。
3. 当前文档权威先从 receipt 绑定的 `DOC_AUTHORITY_MAP.json` 读取；当前算法和质量门只从同一 revision 的 `ALGORITHM_CONTRACT.json` 与 `CURRENT_BASELINE_REGISTRY_V2.json` 读取，不从旧 README、聊天或目录名推断。
4. Writer 必须持有 PID、`/proc` startticks、executor epoch、run signature 与 fencing token；双writer立即终止。
5. GPU等待使用 `WAIT_GPU_RESOURCE`，不消耗算法attempt；质量C不自动重试，所有任务必须有限收敛。
6. 每次attempt使用新目录；partial不能作为successor输入。final同字节可幂等，不同字节拒绝覆盖。
7. run signature覆盖 input manifest、code、config、weights、calibration-or-ABSENT 和 schema。
8. 禁止伪造标定、FoundationStereo confidence、adapter CAD、接触真值、真实Robot action或物理精度。
9. Clean像素只用于视觉；不得进入Depth、Object6D、Contact几何或控制真值。
10. `visual_robot_trajectory_sidecar` 必须标记 `control_ground_truth=false`；Visual Aux与Policy严格分名。
11. current计数只由事实账本生成；禁止手改 `CURRENT_PROJECT_STATUS_ZH.md`、authority index和current registry。
12. 清理不得触碰 `/mnt/data/egodata` 数据与数据侧可视化，也不得触碰 `/nas/chenxianchi` 中其他项目。
13. 删除前读取清理Task Packet，校验 current引用、活动PID/FD/CWD、Git dirty/untracked和许可证；证明不足即`BLOCKED_REFERENCE_PROOF`。
14. 回复只报告本次状态、变化、验证、阻塞和下一任务，不重复整套历史。
15. V7.1 worker只发布immutable receipt；不得原地覆盖revision，也不得因latest successor自动重跑pinned结果。
16. `HAND_OBJECT_ATTACHMENT`不得反向支持Contact/Object6D authority、gold accuracy或tactile升级；发布前验证evidence DAG无环。
17. 正式训练RGB必须为`CAUSAL_TRAINING_INPUT`，任何future donor或双向补图只可标为离线可视化。
18. Robot Geometry不以Clean为前置；Metric Contact和Robotized compositor分别执行各自证据门。
19. RC1 worker不得默认读取完整计划；当前交付语义来自receipt绑定的RC1 contract/release spec，具体执行只来自Task Packet。
20. source group按原始独立采集证据计数；同一merged recording切出的多个session不得重复计入训练/验证容量。
21. RC1 checkpoint数据门失败时必须写`BLOCKED_DATA_VOLUME`，不得降低16 train + 3 validation独立source group门，也不得用session数替代。
