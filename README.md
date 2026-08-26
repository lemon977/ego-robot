# Ego Robot Visual Replacement

更新时间：2026-08-26（Asia/Shanghai）

## 当前状态

`STOPPED_AWAITING_USER_TASK_REVIEW`

仓库清理和任务文档定稿已获授权；RAW 恢复、MASK/CLEAN、Object6D、retarget、渲染、合成、训练和评测均未获开工授权。用户审核 `CURRENT_TASK.md` 并明确开始后，才创建 G0 任务卡。

## 项目目标

用一个代码主干、一个项目级 profile、自动 session context 和有限标准路由，将 ego RAW 中的真人手、手腕和前臂替换为 Tianji 机械臂与 KaiHand 灵巧手，同时保持物体身份、轨迹、纹理、正确遮挡和时序稳定性。

项目有两条严格隔离的产物线：

- `004_CONTACT_GOLD`：允许在新 sidecar 中 refinement Object6D、物体 SDF、接触/非穿透和 `q_hand`，但绝不覆盖 R2。
- `EXACT78_R2_VISUAL_DOMAIN`：冻结 R2 的 `q_hand / wrist / validity / frame order / split / 非图像输入`，只替换最终图像。

`HAND_ONLY_DIAGNOSTIC` 与 `HAND_ARM_FINAL` 是诊断/渲染轨道，不是额外训练产物线。正式 RobotRGB 只能来自后者。

## 权威入口

按以下顺序阅读，禁止根据旧目录名中的 `final/pass/latest` 猜测状态：

1. `README.md`：状态、目标和仓库边界。
2. `CURRENT_TASK.md`：最终任务、阶段门和审核项。
3. `TECHNICAL.md`：组件所有权、算法边界和质量门。
4. `DATA_SPEC.md`：现存冻结数据和计划 I/O 合同。
5. `DATA_CATALOG.json`：机器可读状态与路径。
6. `CLEANUP_REPORT.md`：本轮删除、保留和校验证据。

## 当前目录

```text
chaoyang/
├── HumanEgo/          # 保留的模型/训练主干与冻结 R2 数值资产
├── assets/robot/      # 本地机器人资产；大型 mesh/CAD 不提交 Git
├── hand_benchmark/    # exact78 紧凑冻结证据；整体不提交 Git
├── _run/              # 新流水线可删除 staging（当前为空）
├── processed/         # 原子晋级的正式结果（当前为空）
└── *.md / *.json      # 权威项目文档与机器索引
```

HaWoR 与 ProPainter 不再以内嵌、修改过的平行仓库存在；后续按审核后的锁定版本作为外部依赖接入。详见 `THIRD_PARTY_NOTICES.md`。

## 不可破坏的 I/O 规则

- `/mnt/data/egodata` 是唯一权威 RAW，只读引用，不复制进仓库；当前不可读是正式出图的硬阻塞。
- 所有新状态和产物都写在本项目下。
- `_run/` 不是正式数据源；正式结果只能校验后原子晋级到 `processed/<session>_vN/`。
- 正式 consumer 只读冻结 manifest 与 `processed/`，找不到即失败，禁止从 legacy、RAW 或 staging 静默补齐。
- 每个组件只有一个 producer；QA 只写归因和建议，不能覆盖上游正式结果。

## Git 边界

首次仓库只提交代码、配置、文档和小型 provenance。RAW、运行产物、exact78 帧级数据、checkpoint、视频、机器人运行资产、密钥与缓存由 `.gitignore` 排除。仓库级身份为 `lemonwu977 <1850764377@qq.com>`。
