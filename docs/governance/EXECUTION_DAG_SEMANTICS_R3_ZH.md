# R3 执行依赖语义

> 状态：`CURRENT`。本文只解释调度语义，不代表任何阶段已经通过。

`Depth → Mask → Clean → Contact → Robot → Occlusion → Visual Aux` 表示最终消费者看到的语义链，不是 worker 必须严格串行执行的顺序。

正式执行依赖只来自：

1. `CURRENT_STATUS_RECEIPT.json` 绑定的当前任务状态；
2. `CURRENT_V71_TASK_PACKET_INDEX.json` 指向的当前 Task Packet；
3. 每个 `TASK_PACKET.json` 的 `prerequisites`、`read_set` 和 `write_set`。

只要 Task Packet 的前置条件与写集隔离成立，下列工作可以并行：

```text
Depth 输入域与内部 QA
Role/Object Mask 失败簇审计
Atlas 与因果 donor 构建
Robot Geometry
Contact readiness 账本
current-only 清理的零引用批次
```

禁止根据技术文档的章节顺序自行添加依赖。例如：

- Clean 失败不阻塞 Robot Geometry；
- Object6D 缺失不阻塞 pose-only Robot；
- Contact/Attachment 不得形成反向自证；
- 双向离线可视化不得进入因果训练输入；
- Visual Aux 只在对应任务 pair 的正式 eligibility 通过后启动。

当前调度优先级：保留唯一 Robot v76 及其 hard/soft、Occlusion watcher；并行处理 Depth 输入域、两条 Clean 语义 canary、Mask 失败簇和 Contact readiness。不得增加同类 watcher。
