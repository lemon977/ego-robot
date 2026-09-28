# Chaoyang agent execution protocol

1. 先运行 `PYTHONPATH=src python -m chaoyang.cli validate-governance`；再按 `docs/current/AI_WORK_ENTRY_ZH.md` 读取 RC1 或项目最小状态和当前 Task Packet。只有非 RC1 任务才使用 `CURRENT_PROJECT_STATUS_MIN.json`。
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
22. `tasks/current/INDEX.json` 无 `execution_allowed=true` 行时不得执行算法任务；新目标必须发布新的有限 Task Packet，不得复活历史包。
23. 0911/0914/0915 清洗终态只从 `tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json` 与数据根收据读取；不得重启已提交队列或覆盖 processed 发布根。
24. 当前脚本只能通过算法合同登记后由 `chaoyang run <operation>` 调用；旧实现、临时脚本和未登记模块不得作为隐式入口。
25. 用户于2026-09-23明确要求：此后所有新建或写入的项目文件都必须位于 `/mnt/workspace/code/chaoyang/` 内，包括代码、文档、交接文件夹、下载压缩包、可视化、日志、测试产物、缓存、临时文件和工作树。显式把 TMPDIR、模型/编译缓存及导出路径设置到项目内适用目录，检查解析后的真实路径，不得通过符号链接写到项目外。不得再向 `/mnt/data/`、`/tmp/` 或其他项目外目录输出文件；既有外部源数据仅按授权只读。用户已删除的项目外交接包不重建。历史脚本和旧指令中的外部输出路径必须先修正才能再次运行。
26. 当前任务先读实际 Task Packet、四张短卡及本线最新 STATE/RESULT；旧任务正文只在具体失败码或 SHA 冲突时追溯。已证实的模型调用、几何消费者与 Sensor 后端不登记为“首次实现”。
27. 每一阶段写明实际消费者、结果引用、具体 `next_action`、真实依赖及局部阻塞；同权限且 READY 的下游不因另一支线质量拒绝而等待。派生输入可由既有生产者合法生成时记实现／生产任务，不写外部阻塞。
28. 运行中消费者绑定不可变代码、输入和组件签名，不能热切 `current`。未变签名复用；真正变化只重评受影响阶段。单个候选质量拒绝限制采用及其条件扩片，不取消其他 READY 审阅和交付。
29. 来源、执行、结构、质量、审阅、改善与采用分别报告。视频存在、完整解码、状态 `PASSED` 或槽位填满都不自动证明内容质量；新审阅必须绑定真实数组、帧、消费者和 SHA，AI 抽样不冒充人工全片验收。
