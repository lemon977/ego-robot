# Chaoyang 项目入口

本 Inpainting 支线唯一生产基线为 **INPAINTING_RAW_TO_ROBOT_V1**，对应 pick_place_1008 原版 199 会话、29,322 原生帧的 V9 生产链。当前选择及依赖由 [V1 清单](inpainting_v1/MANIFEST.json)和同 revision 治理合同确定。

| 要做的工作 | 从这里进入 |
|---|---|
| 使用原 V1 | [说明](inpainting_v1/README.md)、[完整链路](inpainting_v1/PIPELINE_ZH.md)、[新任务运行与恢复](inpainting_v1/RUN_ZH.md) |
| 审阅视频 | [统一审阅导航](review/NAVIGATION_ZH.md)，区分原 V1、保留成果与执行诊断片 |
| 继续 Inpainting 优化 | [接手说明](inpainting_optimization/incremental_v1_8h/HANDOFF_ZH.md)、[各线续接](inpainting_optimization/CONTINUE_HERE_ZH.md) |
| 检索已核对的经验与误读 | 先读[记忆更新](inpainting_optimization/agent_memory/UPDATES_ZH.md)，再检索[执行记忆](inpainting_optimization/agent_memory/README_ZH.md) |
| MANUS／0916 独立支线 | [原支线说明](docs/current/ROBOT100_BATCH_ZH.md)及[原 v16 指针](docs/current/ROBOT100_HARDWARE_BASELINE_V16.json)，由另一维护者负责 |
| Raw／Robotized ICT 关节训练 | [Joint26 交接](docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/README_ZH.md)、[两条命令](docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/COLLEAGUE_COMMANDS_ZH.md)、[该线状态](docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json) |
| Tianji／Revo2 资产、通道与上位机资料 | [专项交接](revo2_ict_handoff/README.md)，来源与 SHA 见其机器索引 |

## 开始工作

先读 [AGENTS](AGENTS.md)，运行治理校验，再按[当前总入口](docs/current/README_ZH.md)和 [AI 工作入口](docs/current/AI_WORK_ENTRY_ZH.md)读取本次任务范围：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m chaoyang.cli validate-governance
```

核对 [receipt](docs/governance/CURRENT_STATUS_RECEIPT.json)绑定的[文档权威](docs/governance/DOC_AUTHORITY_MAP.json)、[算法合同](docs/governance/ALGORITHM_CONTRACT.json)及[基线登记](docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)。活动任务只看[实时状态](docs/current/STATUS.json)和[任务索引](tasks/current/INDEX.json)；新的执行目标须登记有限 Task、写集、预算、截止及真实 writer。

`CURRENT` 表示文档在其登记范围内可读。目录名、最高版本、历史文件中的 current/latest/final，以及 `freshness=no_active_tasks` 均不能代替选版、执行许可或内容审阅。可搜索记忆帮助定位证据，每次执行仍须核对当前权威与来源 SHA。

## 目录与维护

| 位置 | 职责 |
|---|---|
| `src/chaoyang/` | 现用源码；V1 依赖按清单原路径读取 |
| `inpainting_v1/` | 唯一生产 V1 的说明、链路、运行与机器清单 |
| `inpainting_optimization/` | 候选实验、交接、结果索引与可搜索记忆 |
| `configs/`、`contracts/`、`manifests/` | 配置、schema、数据／模型／来源清单 |
| `docs/` | [当前入口、指南、参考、计划与治理](docs/README.md) |
| `tasks/` | 有限任务包与不可变收据；仅 INDEX 中唯一授权行可执行 |
| `review/` | 观看导航及审阅证据 |
| `_run/` | 按任务／attempt 隔离的结果、日志、缓存和维护证明 |
| `archive/`、`baseline_v1/` | 历史证据与冻结依赖；不作为第二套默认源码 |

详细位置见[目录导览](docs/current/PROJECT_LAYOUT_ZH.md)，新增文件与封存规则见[项目约定](docs/maintenance/PROJECT_CONVENTIONS_V1_ZH.md)，环境与恢复见[交接说明](docs/maintenance/ENVIRONMENT_HANDOFF_V1_ZH.md)。[scripts](scripts/README.md)和 [tests](tests/README.md)各自说明可维护入口；`make check`、`make check-server`分别用于便携及服务器 CPU 检查，其通过不代表视觉或训练质量通过。

所有新文件、临时输出和缓存写入本项目真实路径。外部原始数据、NAS、0916 与 ICT 独立工作域按各自授权处理。清理须证明精确目标的引用、活动 FD/CWD、Git、许可证及恢复来源；不按旧日期或目录名删除，也不丢弃既有 Git 工作。

## 结果与历史的阅读边界

V1 实际动作为同会话 Controller/Head＋MANUS；SAM3.1、ProPainter 与 Tianji／KaiHand22 负责视觉生产链。199 会话执行完成、用户采用、视觉审阅、产品质量、训练和控制资格分别记录。原 FAIL、隐藏 UNKNOWN 和人工未审状态继续保留；Clean 生成像素只用于离线视觉，不作 Depth、Object6D、Contact 或控制真值。

旧 v13、V2、P1、0915／0918 实验及优化候选沿[历史隔离索引](archive/inpainting_baselines/README.md)追溯。被 V1 明确引用的旧名共享代码仍是必要依赖，保留原路径及 SHA；原 199 V1、旧 v13 的 234 会话与旧 156 会话资格账本分开统计。旧 H50×30／20k 腕模型与 H30×26 关节模型保持各自身份，状态只从对应训练线读取。

Git 与服务器产物范围见[交付边界](docs/current/GIT_DELIVERY_ZH.md)；第三方来源与许可证见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)及 [vendor 清单](manifests/vendor.json)。
