# Chaoyang Ego-to-Robot Pipeline

2026-09-28已整理当前入口、四线状态与历史计划。请先读[当前总入口](docs/current/README_ZH.md)，不要从历史任务书续跑。[后续建议](docs/current/NEXT_ACTIONS_ZH.md)尚未登记执行；[Git与服务器证据边界](docs/current/GIT_DELIVERY_ZH.md)说明哪些产物不在GitHub。

Clean是独立视觉分支，不向Depth/Object6D/Contact提供几何真值；下面流水线名称仅列能力，不表示所有阶段已合格。

本仓库实现第一视角数据的 Raw → HaWoR / Mask → Stereo Depth → Object6D → Clean → Contact → Robot Visual → HumanEgo 流水线。2026-09-17 起，活动代码统一使用 src/chaoyang 包布局；旧路径只可通过归档清单查询，不再是执行入口。

## 当前唯一入口

执行或回答状态问题前，按顺序读取：

1. [当前入口](docs/current/README_ZH.md)
2. [后续 AI 工作入口](docs/current/AI_WORK_ENTRY_ZH.md)
3. [当前状态 receipt](docs/governance/CURRENT_STATUS_RECEIPT.json)
4. [当前Human→Robot机器状态](docs/current/STATUS.json)；仅核查历史RC1时读取[RC1最小事实页](docs/governance/CURRENT_RC1_STATUS_MIN.json)
5. [文档权威索引](docs/governance/DOC_AUTHORITY_MAP.json)与[算法合同](docs/governance/ALGORITHM_CONTRACT.json)
6. [当前任务索引](tasks/current/INDEX.json)；只有 execution_allowed=true 的行可执行
7. [三批数据清洗验收基线](docs/current/DATA_CLEANING_0911_0915_ZH.md)
8. [端到端复现说明](docs/reference/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)
9. [Clean baseline v1 迁移与恢复说明](docs/reference/architecture/CLEAN_BASELINE_V1_ZH.md)

freshness.reason=no_active_tasks 只表示没有需要心跳的活动治理任务，不表示所有研究文档刚更新。实时事实由 receipt、SHA 与 DOC_AUTHORITY_MAP 共同约束。历史文件名中的 current/latest/final 不具有当前权威性。

## 目录职责

    chaoyang/
    ├── src/chaoyang/       # pipeline、human_ego、governance、ops、cli
    ├── configs/            # pipeline、training、systems 配置
    ├── contracts/          # schema 与机器合同
    ├── manifests/          # 数据、模型、实验与 vendor 清单
    ├── scripts/            # 少量迁移和人工入口
    ├── tests/              # 当前回归
    ├── docs/               # 当前入口、指南、参考、研究与治理
    ├── tasks/current/      # 保留任务包路径；仅INDEX.json路由项可执行
    ├── tasks/receipts/     # 紧凑证明与历史目录
    ├── assets/models/      # 权重本体忽略，清单受控
    ├── vendor/             # 运行必需的最小第三方源码
    ├── _run/current/       # 忽略的锁、缓存、日志、媒体和实验结果
    └── archive/            # 隔离且被 Git/构建/测试排除的可恢复历史

## 统一命令

    PYTHONPATH=src python -m chaoyang.cli paths
    PYTHONPATH=src python -m chaoyang.cli validate-governance
    PYTHONPATH=src python -m chaoyang.cli run <operation> [args...]

默认路径：原始数据 /mnt/data/egodata/datasets/ego；处理数据 /mnt/data/egodata/datasets/ego/processed；运行产物 <repo>/_run/current；模型权重 <repo>/assets/models。分别使用 EGO_DATA_ROOT、EGO_PROCESSED_ROOT、CHAOYANG_RUN_ROOT、CHAOYANG_MODEL_ROOT 覆盖。

## 不可破坏规则

- 不移动、删除或改写 /mnt/data/egodata。
- current authority 只由 receipt 绑定的机器文件解释。
- 已有 final 不覆盖；输入、代码、权重、标定或 schema 签名变化不得静默复用。
- HAND_OBJECT_ATTACHMENT 只可延续遮挡期 pose，不得反向证明身份、Contact 或正式 Object6D。
- Clean 合成像素不得反喂 Depth、Object6D、Contact 几何或动作真值。
- GPU 任务必须走租约；迁移、治理、哈希和测试默认 CPU。
- 永久删除不可再生内容前，必须有归档清单、零 current 引用、零活动 FD/CWD、Git 保护快照和删除收据。

第三方来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [vendor 清单](manifests/vendor.json)。
