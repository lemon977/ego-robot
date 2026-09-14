# 文档入口

> 当前正式任务是 0909 `rgb30_v1`。先读
> [`governance/CURRENT_TASK.md`](governance/CURRENT_TASK.md) 和
> [`0909 当前状态`](../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS_ZH.md)。
> 旧 exact78 与 HumanEgo/Robot 自动线保持 `PAUSED_BY_USER`，下面旧日期内容仅作证据追溯。

扑克/薯片任务只从下面五个活动入口开始。算法和逐阶段输入输出已经合并到第1份，避免同一流程存在多个互相冲突的版本：

1. [`pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md`](pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)：Raw→HaWoR→Depth→双语义 Mask→Object6D→Clean→Robot→HumanEgo A/B 的从零复现、技术细节、质量门、常见失败修复与理想管线图。
2. [`pipeline/AUTOMATION_CONTINUATION_HANDOFF_ZH.md`](pipeline/AUTOMATION_CONTINUATION_HANDOFF_ZH.md)：批量、Robot与HumanEgo的自动续跑边界、实时状态入口、安全恢复和人工门。
3. [`pipeline/CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md`](pipeline/CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md)：当前状态与人工查看视频的统一入口。
4. [`acquisition/OBJECT6D_POKER_CHIPS_CAPTURE_ZH.md`](acquisition/OBJECT6D_POKER_CHIPS_CAPTURE_ZH.md)：Object6D 输入说明；当前不要求用户补拍。
5. [`governance/CURRENT_STAGE_BASELINES_ZH.md`](governance/CURRENT_STAGE_BASELINES_ZH.md)：当前12阶段算法、边界与下一整改需求。

exact78的四个HumanEgo checkpoint当前是否可启动，单独查看[`四checkpoint准入状态`](../tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/FOUR_CHECKPOINT_CURRENT_READINESS_ZH.md)：截至2026-09-11 15:00仍因Robot authority/action sidecar为0而不具备优化器准入。

2026-09-10移出的9份重复或过时文档保存在`archive/20260910_obsolete_docs_after_world_first_v3/`，其`ARCHIVE_MANIFEST.json`记录原路径、SHA256和恢复命令。归档文件不具有当前执行权。

端到端复现文档的不可变字节与编写时 authority 快照见 [`pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.receipt.json`](pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.receipt.json)。运行状态仍以实时 STATE/authority 为准。

2026-09-07 的直接机器证据入口：

- [`corrected FoundationStereo 557 帧 RESULT`](../tasks/control/runs/20260907_foundationstereo_corrected_metric_v2/RESULT.json)
- [`旧深度内参 P0 禁用审计`](../tasks/control/runs/20260907_foundationstereo_depth_intrinsics_contract_audit_v1/RESULT.json)
- [`Object ICT v3 两会话预检`](../tasks/control/runs/20260907_object_ict_v3_two_session_preflight/REPORT_ZH.md)
- [`错误训练停机收据`](../tasks/control/runs/20260907_current_ict_audit_v1/STOP_RECEIPT.json)
- [`被取代的新任务计划映射`](history/2026-09-07/superseded_newtask/README.md)

其他目录的职责：

- `governance/`：项目级约束和任务治理。
- `architecture/`：组件所有权、引用图和接口设计。
- `data/`：通用数据规格与 catalog；若与扑克/薯片现行流程冲突，以主流程文档为准并安排修订。
- `reproducibility/`：环境、依赖和权重复现证据。
- `robot/`：Robot compositor 的专门说明。
- `organization/`：目录迁移、校验和、回滚及存储审计证据。
- `newtask/`：尚未合并的通用质量门/输入指南；旧执行计划逐步移入 `history/`。
- `history/`：已被取代但为追溯保留的历史文档，不具有当前执行权。

禁止依据历史文档、目录名或旧 `current` 猜测阶段是否可用。机器判断始终读取对应
`RESULT.json`、`consumption_authorized`、上游输入 SHA 和质量门。
