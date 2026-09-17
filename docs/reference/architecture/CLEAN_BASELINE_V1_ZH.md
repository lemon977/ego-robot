# Clean baseline v1：目录迁移、约束与恢复

## 范围

本次迁移把活动代码统一到 src/chaoyang，把系统配置、manifest、测试和文档按职责拆分；历史文档、终态任务、旧运行目录、第三方附属样例及未进入当前合同的版本链隔离到 archive/baseline-20260917-0aa69e9/。数据集 /mnt/data/egodata/datasets/ego 未参与移动或内容修改。

最终活动基线以 Git 标签 `clean-baseline-v1-final`、当前状态 receipt 和
`DOC_AUTHORITY_MAP.json` 共同识别。标签固定代码树，receipt 固定治理 revision，
文档权威表固定允许指导当前工作的页面；三者用途不可互相替代。

## 权威关系

- docs/current/ 是浅层导航。
- docs/governance/CURRENT_STATUS_RECEIPT.json 绑定当前机器事实。
- tasks/current/INDEX.json 只容纳可执行项；当前基线为零活动任务。
- tasks/receipts/HISTORICAL_TASK_CATALOG.json 只用于查询，不能调度。
- tasks/receipts/RETIRED_REFERENCES.json 如实记录基线建立时已缺失的旧证据；original_* 字段不构成当前 artifact reference。
- archive/ 是物理保留区，不属于活动树。

主要路径映射写入归档的 PATH_REDIRECTS.json。程序不得硬编码仓库绝对路径，应使用 chaoyang.paths.get_paths() 或显式 CLI 参数。数据路径可配置，但必须保持在仓库之外。

## 运行环境与归档边界

当前运行所需的两个第三方环境位于忽略目录：

- `_run/current/environments/foundationstereo-py311-v1`
- `_run/current/environments/hawor-py310-v1`

它们由当前系统配置引用，不再从历史基线归档执行。Clean、Mask 和 LaMa 的旧环境
未被当前算法合同引用，按 `PURGE_RECEIPT.json` 从主归档中进行可审计清除；清除收据
绑定清除前 Merkle、精确前缀、文件数和字节数。主归档剩余条目仍由原始
`INVENTORY.jsonl` 加清除收据联合解释，不能只用目录现状替代清单。

辅助归档分别保存代码版本链、HumanEgo 旧工具、治理历史、退役运行配置和文档残留。
每个归档都包含自己的 `INVENTORY.jsonl`、`MERKLE_ROOT.json`、
`PATH_REDIRECTS.json` 和 `RESTORE.md`；归档被 Git、构建、测试和 current 导航排除。

0911/0914/0915 数据清洗不在 archive 内，终态位于约定的 processed 根；参见
[`DATA_CLEANING_0911_0915_ZH.md`](../../current/DATA_CLEANING_0911_0915_ZH.md)。

归档根中的 RESTORE.md 给出 Git bundle、工作区 patch、移动反转和抽样恢复步骤。任何恢复都应先写入 archive/.staging/，通过 SHA 校验后再由人工决定是否重新进入活动树。

## 验证

    PYTHONPATH=src python -m chaoyang.cli validate-governance
    PYTHONPATH=src python scripts/migration/validate_structure.py --allow-dirty
    PYTHONPATH=src python -m pytest -q

结构验证器禁止旧顶层目录、活动代码中的固定仓库根、未登记的 ignored 内容和遗留导入重新进入基线。
