# 文档入口

日常工作从[当前总入口](current/README_ZH.md)进入，AI 接手按 [AGENTS](../AGENTS.md)和 [AI 工作入口](current/AI_WORK_ENTRY_ZH.md)先完成治理校验。当前事实以 [receipt](governance/CURRENT_STATUS_RECEIPT.json)绑定的同 revision 文档权威、算法合同和基线登记为准。

| 工作域 | 文档入口 |
|---|---|
| Inpainting 唯一生产 V1 | [说明](../inpainting_v1/README.md)、[机器清单](../inpainting_v1/MANIFEST.json)、[链路](../inpainting_v1/PIPELINE_ZH.md)、[运行](../inpainting_v1/RUN_ZH.md) |
| Inpainting 候选与经验 | [本线接手说明](../inpainting_optimization/incremental_v1_8h/HANDOFF_ZH.md)、[阶段导航](../inpainting_optimization/CONTINUE_HERE_ZH.md)、[记忆更新](../inpainting_optimization/agent_memory/UPDATES_ZH.md) |
| 视频审阅 | [统一导航](../review/NAVIGATION_ZH.md) |
| MANUS／0916 | [独立支线说明](current/ROBOT100_BATCH_ZH.md)，保留另一维护者的原入口 |
| ICT 关节训练 | [Joint26 交接](plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/README_ZH.md)、[状态](plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json)、[命令](plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/COLLEAGUE_COMMANDS_ZH.md) |
| 目录、环境与维护 | [目录导览](current/PROJECT_LAYOUT_ZH.md)、[维护导航](maintenance/README.md)、[项目约定](maintenance/PROJECT_CONVENTIONS_V1_ZH.md)、[环境交接](maintenance/ENVIRONMENT_HANDOFF_V1_ZH.md) |

本 Inpainting 支线唯一默认为 **INPAINTING_RAW_TO_ROBOT_V1**。旧使用快照、双基线合同及历史计划仅按明确依赖或失败码追溯；`current/` 是混合物理容器，不能凭所在目录或文档中的旧“当前”选版。执行、改善、视觉审阅与训练资格分别读取对应证据。

## 文档职责

| 分区 | 内容与维护方式 |
|---|---|
| `current/` | 浅层入口与工作卡；阅读范围查 [DOC_AUTHORITY_MAP](governance/DOC_AUTHORITY_MAP.json) |
| `guides/` | 采集、运行、复现指南 |
| `reference/` | 接口、架构与稳定参考 |
| `research/current/` | 研究报告；结论不自动成为生产 authority |
| `plans/` | 各线计划、规格和续接；由明确链接定位，不按日期猜版本 |
| `governance/` | receipt、合同、状态与索引；生成文件由现有 publisher 发布 |
| `maintenance/` | 维护规则与环境说明；完整运行证明放项目内 `_run/maintenance/` |

源码在 [src](../src/chaoyang/)，操作与测试入口分别看 [scripts](../scripts/README.md)、[tests](../tests/README.md)。日志、测试工作树、缓存和临时脚本写项目内 `_run/`，不向说明目录继续堆放运行产物。

## 历史与更新

[历史隔离索引](../archive/inpainting_baselines/README.md)及[计划索引](plans/INDEX_ZH.md)用于追溯。冻结证据、Task Packet、原媒体和引用路径保持原位；旧 2026-10-08 使用快照不再是新修改的首读入口。

导航页只维护入口和必要边界，详细算法、命令与状态各链接一处。更新时核验目标文件、来源身份及链接，再由现有 publisher 绑定新的导航 SHA；不复制第二套权威状态，不回写历史结果。
