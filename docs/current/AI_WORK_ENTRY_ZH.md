# 后续 AI 工作入口与优化边界

状态：CURRENT
基线：`clean-baseline-v1`
发布标签：`clean-baseline-v1-final`

本页是交给后续 AI 的最小执行入口。事实仍以当前状态 receipt、文档权威表、算法合同和任务索引为准；聊天、历史目录及文件名中的 `current/latest/final` 均不授予执行权。

## 必须按顺序读取

1. [`CURRENT_STATUS_RECEIPT.json`](../governance/CURRENT_STATUS_RECEIPT.json)
2. RC1 工作读 [`CURRENT_RC1_STATUS_MIN.json`](../governance/CURRENT_RC1_STATUS_MIN.json)，其他工作读 [`CURRENT_PROJECT_STATUS_MIN.json`](../governance/CURRENT_PROJECT_STATUS_MIN.json)
3. [`DOC_AUTHORITY_MAP.json`](../governance/DOC_AUTHORITY_MAP.json)
4. [`ALGORITHM_CONTRACT.json`](../governance/ALGORITHM_CONTRACT.json)
5. [`tasks/current/INDEX.json`](../../tasks/current/INDEX.json)
6. 与任务直接相关的合同、代码和 fixture；不要默认加载整个归档

开始前执行：

```bash
PYTHONPATH=src python -m chaoyang.cli validate-governance
PYTHONPATH=src python scripts/migration/validate_structure.py --allow-dirty
```

## 当前可执行状态

- `tasks/current/INDEX.json` 当前为 `PASS_NO_ACTIVE_TASKS`，没有可执行任务。
- 只有索引中 `execution_allowed=true` 且与账本 `next_task` 一致的任务包可以调度。
- 空索引内的 revision 11169 是创建 revision；当前生效 revision 读取 `CURRENT_V71_TASK_PACKET_INDEX.json` 和 `CURRENT_STATUS_RECEIPT.json`。冻结 payload 的创建 revision 不要求被原地改写。
- `tasks/receipts/HISTORICAL_TASK_CATALOG.json` 和 `archive/` 只用于查询/恢复，任何历史终态任务都不得直接重启。
- 0911/0914/0915 数据清洗已经完成，见 [`DATA_CLEANING_0911_0915_ZH.md`](DATA_CLEANING_0911_0915_ZH.md)；它不是待办任务。
- 0915 裸手全链已经按用户要求停止。现有物理左目准备显式执行了
  `equiDis62 → 1280×960 FOV90 pinhole` 重映射，而 VST 编码像素是否已经校正尚未
  建立权威；现有派生结果不得作为基线。先读
  [`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)，再复核
  [`单会话 A/B`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。A/B 已证明当前
  remap 是百像素级变换，且 legacy 单目是物理右目。用户已经确认物理左目
  `sourceIndex=1 + resize-only` 为正确单目画面；当前最多只允许一个单会话 HaWoR
  canary。该 canary 已完成但因右手画面边界/骨长门保持 `FAILED_QUALITY_C`，等待
  用户复核；Depth 仍为独立未决问题，SAM3.1 和全批均未授权。复核入口见
  [`0915_HAWOR_RESIZE_ONLY_CANARY_V1`](visuals/0915_HAWOR_RESIZE_ONLY_CANARY_V1/README_ZH.md)。
- 0916 独立清洗已经完成；0915/0916 状态见
  [`FULL_FUNNEL_0915_AND_CLEANING_0916_ZH.md`](FULL_FUNNEL_0915_AND_CLEANING_0916_ZH.md)。
- 用户仍将未来 Mask 模型锁定为 SAM3.1，但图像域问题未解决前不得运行它，也不得
  创建 SAM2.1/Cutie 候选或胜者选择任务。

如果用户提出新目标，应建立新的、有限收敛的任务包并发布新的治理 revision；不要把旧任务包改回 `PENDING`。任务包至少固定输入、代码、配置、权重或 `ABSENT`、标定或 `ABSENT`、输出 schema、质量门、预算、终止条件和回滚路径。

## 脚本和任务命名

- 当前 Python 操作名采用 `<verb>_<subject>[_<scope>]_vN.py`，CLI operation 等于文件 stem。
- 名称必须表达动作和对象，例如 `tactile_quality_gate_v1`、`batch_clean_handle_content_v3`；禁止使用含糊的 `run.py`、`new.py`、`latest.py`、`final.py`。
- 版本升级必须在算法合同中只保留一个当前实现；旧版本移入 `archive/`，不得长期维护双实现。
- 新任务 ID 采用 `<domain>_<bounded_deliverable>_vN`；日期只用于不可变采集批次或运行实例，不代替语义版本。
- 运行产物放在 `_run/current/<task_id>/attempts/attempt_NNNN/`，不得散落到源码、文档或仓库外路径。
- 当前维护操作统一使用 `PYTHONPATH=src python -m chaoyang.cli run <operation> ...`；CLI 会拒绝未登记到算法合同的模块。

## 后续优化顺序

以下是边界清晰的候选工作，不代表已授权执行；每项都需要新的 current 任务包。

1. **P0：解除 RC1 数据量门。** Chips 与 Poker 当前均为 `BLOCKED_DATA_VOLUME`。先满足合同要求的独立 source group 数量与 train/validation 隔离；不得用同一 merged recording 切出的多个 session 重复计数。
2. **P1：RC1 checkpoint 与比率验收。** 只有 P0 通过后，才按 [`CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md`](../governance/CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md) 训练和比较四个 checkpoint；GPU 必须走租约，不与现有实验抢占。
3. **P2：算法 bounded canary。** HaWoR、Mask、Depth、Object6D、Contact、Robot 或 HumanEgo 的改动先固定失败样本、输入签名和质量门，运行小 canary，再决定是否扩批。不得依据视觉观感提升 authority。
4. **P3：数据清洗增强。** V3 已完成；后续可增加标定力/独立接触真值、更多损坏 fixture 和性能 profiling，但不得改变三批已发布终态或把缺失 MANUS 伪造成数据。

H4 自动 glove/Controller Mask-to-Clean 路线已经被当前策略拒绝，除非出现新的独立证据和新合同，不得自动恢复。Attachment 不得反向证明对象身份、Contact 或物理真值。

## 路径和资源边界

- 原始数据根：`/mnt/data/egodata/datasets/ego`
- 处理数据根：`/mnt/data/egodata/datasets/ego/processed`
- 运行根：`<repo>/_run/current`
- 模型根：`<repo>/assets/models`
- FoundationStereo 环境：`<repo>/_run/current/environments/foundationstereo-py311-v1`
- HaWoR 环境：`<repo>/_run/current/environments/hawor-py310-v1`

不得移动、删除或改写原始数据；不得把临时结果写到约定路径之外。GPU 工作必须经过租约，治理、哈希、文档、归档和普通测试使用 CPU。

## 完成定义

每个后续任务结束时必须同时满足：

- 有不可变结果/失败/阻塞收据，且计数来自机器文件；
- 代码、合同、文档和任务指针由同一发布 receipt 绑定；冻结 payload 必须显式区分 created/effective/published revision；
- 当前 Markdown 链接、artifact bytes/SHA、导入和配置引用均通过校验；
- `pytest`、治理校验、结构校验和相关 canary 通过；
- `git status --porcelain` 为空；
- 归档或删除动作有 inventory、Merkle、PATH_REDIRECTS 和恢复说明。

若任何门不满足，应明确写 `BLOCKED_*` 或 `FAILED_*`，不得用聊天描述替代终态收据。
