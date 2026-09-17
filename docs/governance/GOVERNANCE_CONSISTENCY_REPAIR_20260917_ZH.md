# 治理一致性修复交接（2026-09-17）

## 目的与边界

本轮只修治理、文档与机器索引的一致性，不改变历史实验结论，不提升任何 Robot、Object Mask、Clean、Contact、触觉或训练 authority，也不重跑算法。当前事实仍以 `CURRENT_STATUS_RECEIPT.json` 绑定的文件为准。

## 已闭合项目

1. `validate_governance_state.py` 现在递归检查算法合同、基线注册表和回归清单中的 artifact ref；每次校验对不超过 64 MiB 的引用复算 SHA，对更大的模型只检查路径与 bytes，并在输出中单列跳过的 SHA 数。这样可以发现“合同文件自身 SHA 正确、内部代码 SHA 已漂移”的情况，同时避免每次重哈希多 GB 模型。
2. 基线与合同通过正常 publish 重新生成，绑定当前 Robot renderer、GPU lease runner 和 lease test 的实际 bytes/SHA。不得手工改生成 JSON。
3. 旧 `archive/baseline-20260917-0aa69e9/content/history/docs/stale-linked/docs/reference/pipeline/CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md` 已明确历史化，失效的 13 个链接不再伪装成可点击入口；当前导航只从根 README、receipt 和 `DOC_AUTHORITY_MAP.json` 进入。
4. donor 研究结论改为“保守弃权”，不再把 54 次 UNKNOWN 称为已证明的正确拒绝。`HAND_OBJECT_ATTACHMENT` 只能延续已由独立证据建立的 pose，不能证明实例身份、Contact、Gold 或触觉。
5. 浅层视频索引删除不存在的 CSV 入口，只保留实际存在的 rev9 JSON。
6. 新增不可变 `TASK_MATRIX_REV_0014.json`，汇总 rev13 之后的 RGB 图像域复核、Poker245 donor 可观测性、独立 Robot 初始化失败与 Robot30 rev9 索引修正；全部保持 research-only，正式 RC1 不变。
7. 当前任务包索引发布后继版本，并为每条记录写入 `execution_class` 与 `execution_allowed`。引用 SUPERSEDED V7.1 计划的 20 条记录必须是 `HISTORICAL_SUPERSEDED_READ_SET` 且不可执行；索引存在本身不再等于可重启。
8. `DOC_AUTHORITY_MAP.json` 补登记根 README 已推荐的端到端复现、Robot 质量门、浅层可视化、Clean/Contact 审计和本交接页。校验器会检查所有登记为 CURRENT 的 Markdown 本地链接；历史文档坏链不阻塞 current，但必须显式历史化。

## 字段语义

- `CURRENT_STATUS_RECEIPT.governance_revision`：当前发布 revision。
- `PLAN_REVISION.governance_revision`：计划批准时冻结的创建/生效 revision，不要求追平每次发布。
- `CURRENT_V71_TASK_PACKET_INDEX.governance_revision`：当前 pointer 的发布 revision；指向的不可变历史索引可保留自身创建 revision，后继索引由新的发布 revision 固定。
- `freshness.reason=no_active_tasks`：只表示没有需要心跳的活动治理任务，因此 `age_seconds=0`；不表示研究文档刚更新。

## 后续 AI 的最小复核

先运行：

```bash
conda run -n humanego chaoyang validate-governance
```

预期 `status=PASS`，并检查输出中的 `transitive_refs`、`current_markdown_local_links_checked` 和 `freshness_scope=active_task_heartbeat_only`。大型模型 SHA 没有在此快速门中重算；需要重验时应单独安排资产校验，不能把 `large_sha_skipped` 解读为已复算。

全仓历史 Markdown 仍可能包含迁移后失效的旧链接。它们属于分批历史整理债务，不得为消除数量而伪造文件、复制大型媒体或把历史文档重新标成 CURRENT；优先保证 `DOC_AUTHORITY_MAP` 中 CURRENT 文档零坏链。
