# 脚本用途与执行入口

`scripts/` 保存任务包装、维护检查和历史发布工具。执行算法前，先从[当前任务索引](../tasks/current/INDEX.json)找到允许执行的 Task Packet，并核对收据绑定的 [ALGORITHM_CONTRACT](../docs/governance/ALGORITHM_CONTRACT.json)；具体操作、代码 SHA、输入、输出和预算以该包为准。登记后的算法通过 `chaoyang run <operation>` 调用。

## 目录用途

| 位置 | 用途与查阅方式 |
| --- | --- |
| `humanego_ict/` | HumanEgo ICT 预检、任务登记和训练包装；根据当前候选路线选择。 |
| `pick_place_1008/` | pick_place 专用包装；先查当前任务及其实现绑定。 |
| `takeover/` | Exact78 接管相关工具；目录名不构成执行授权。 |
| `maintenance/` | 只读交接、有限 CPU 检查及特定维护任务。 |
| `migration/` | 历史迁移工具；按明确的现用引用追溯。 |
| 本目录根部 | 仍有引用的历史发布、收尾工具及 `.patch`、`.transfer`；退役无消费者工具按清理证明归档。 |

历史版本号、文件名或补丁内容不能替代当前合同和 Task Packet。不要根据旧 `.patch`、`.transfer` 自动应用补丁或重放发布命令。

2026-10-16 已将两份无消费者的退役文档发布工具与获证的生成测试目录、字节码缓存合入[单一恢复归档](../archive/project_files_cleanup_20261016/INDEX.json)。具体移出和保留项见[最终清理报告](../_run/maintenance/project_files_cleanup_20261016/attempt_a06/REPORT_ZH.md)及[可搜索维护记忆](../inpainting_optimization/agent_memory/FILESYSTEM_CLEANUP_20261016_ZH.md)。有引用、冻结绑定、许可证或共享 inode 的项继续保留；归档工具不作为执行入口，恢复须另建有限任务。

## H30x26 训练路线

HumanEgo ICT joint26 使用 H30x26 动作接口。查阅[同事命令说明](../docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/COLLEAGUE_COMMANDS_ZH.md)与[训练 Bash 包装](humanego_ict/train_joint26_pick_place.sh)，确认本次输入与 Task Packet 后使用对应入口。相同接口不证明旧腕位姿 checkpoint 可用于 joint26，也不证明物理部署资格。

## 维护工具的写入范围

| 工具 | 行为 |
| --- | --- |
| [project_handoff_check_v1.py](maintenance/project_handoff_check_v1.py) | 读取 JSON、源码 SHA 和软件元数据；输出只读交接视图，不启动算法或写文件。 |
| [run_project_checks_v1.py](maintenance/run_project_checks_v1.py) | 执行有限 CPU 测试；在项目内建立独立 attempt，写测试日志、缓存和结果。 |
| [project_navigation_cleanup_v1.py](maintenance/project_navigation_cleanup_v1.py) | 固定于 20261009 的维护写入工具；使用旧锁与输出目录，不能作为通用只读检查命令复用。 |

从项目根运行 `make handoff` 可读取交接视图；`make check` 的测试范围与写入位置见[测试说明](../tests/README.md)。需要修改、迁移或清理时使用本次有限维护包，按其 write_set 操作；旧维护脚本的存在不授权再次写入。

只读交接检查按当前收据明确绑定的 Inpainting 发布补充识别原版、R02 和 R03。冻结 V1 清单仍含 94 项运行源码；既有 R03 合法登记三份治理包装覆盖和一个 helper，现用闭包为 95 项。检查器逐项核验 19 核心与运行源码 SHA，不根据目录中的最高版本选补充，不启动算法，也不把结构通过升级为质量或人工审阅通过。

源码开发位于 `src/chaoyang/`。[冻结 baseline](../baseline_v1/MANIFEST.json)用于 SHA 与角色核对，不加入 `PYTHONPATH`。所有新增日志、缓存、临时文件和输出必须解析到项目目录内。
