# 文档入口

本页只负责导航，不承载项目状态。唯一正式 current 入口是 [`docs/current/README_ZH.md`](current/README_ZH.md)；机器事实由 [`CURRENT_STATUS_RECEIPT.json`](governance/CURRENT_STATUS_RECEIPT.json)、[`DOC_AUTHORITY_MAP.json`](governance/DOC_AUTHORITY_MAP.json) 和 [`ALGORITHM_CONTRACT.json`](governance/ALGORITHM_CONTRACT.json) 共同约束。

## 活动文档分区

- `current/`：唯一当前导航与少量浅层复核索引。
- `guides/`：采集、运行与复现指南；不覆盖机器 receipt。
- `reference/`：接口、架构和稳定复现参考。
- `research/current/`：仍保留研究价值的当前报告；研究结论不自动授予生产 authority。
- `governance/`：状态、合同、schema、任务与文档权威登记。

历史状态页、终态任务、旧实验报告和失效链接文档位于 [`archive/baseline-20260917-0aa69e9/`](../archive/baseline-20260917-0aa69e9/)，不参与当前导航、构建或测试。不得依据历史文件名中的 `current`、`latest` 或 `final` 重启任务。

代码统一位于 `src/chaoyang/`，当前任务仅从 [`tasks/current/INDEX.json`](../tasks/current/INDEX.json) 读取；当前无可执行任务时，`freshness.reason=no_active_tasks` 仅表示没有活动心跳，并不表示所有研究材料刚更新。
