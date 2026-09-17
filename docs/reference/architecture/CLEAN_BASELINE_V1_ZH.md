# Clean baseline v1：目录迁移、约束与恢复

## 范围

本次迁移把活动代码统一到 src/chaoyang，把系统配置、manifest、测试和文档按职责拆分；历史文档、终态任务、旧运行目录、第三方附属样例及未进入当前合同的版本链隔离到 archive/baseline-20260917-0aa69e9/。数据集 /mnt/data/egodata/datasets/ego 未参与移动或内容修改。

## 权威关系

- docs/current/ 是浅层导航。
- docs/governance/CURRENT_STATUS_RECEIPT.json 绑定当前机器事实。
- tasks/current/INDEX.json 只容纳可执行项；当前基线为零活动任务。
- tasks/receipts/HISTORICAL_TASK_CATALOG.json 只用于查询，不能调度。
- tasks/receipts/RETIRED_REFERENCES.json 如实记录基线建立时已缺失的旧证据；original_* 字段不构成当前 artifact reference。
- archive/ 是物理保留区，不属于活动树。

主要路径映射写入归档的 PATH_REDIRECTS.json。程序不得硬编码仓库绝对路径，应使用 chaoyang.paths.get_paths() 或显式 CLI 参数。数据路径可配置，但必须保持在仓库之外。

归档根中的 RESTORE.md 给出 Git bundle、工作区 patch、移动反转和抽样恢复步骤。任何恢复都应先写入 archive/.staging/，通过 SHA 校验后再由人工决定是否重新进入活动树。

## 验证

    PYTHONPATH=src python -m chaoyang.cli validate-governance
    PYTHONPATH=src python scripts/migration/validate_structure.py --allow-dirty
    PYTHONPATH=src python -m pytest -q

结构验证器禁止旧顶层目录、活动代码中的固定仓库根、未登记的 ignored 内容和遗留导入重新进入基线。
