# 四支线 V4 修正版 V2 当前交接

更新时间：2026-09-22。唯一当前路由为 [`four_stream_full_pipeline_v4_takeover_v2`](../../tasks/current/four_stream_full_pipeline_v4_takeover_v2/TASK_PACKET.json)；具体状态以[当前治理收据](../governance/CURRENT_STATUS_RECEIPT.json)、[状态](STATUS.json)和[输入审计](../../_run/current/four_stream_full_pipeline_v4_takeover_v2/attempts/attempt_0001/ROUTE_AND_INPUT_AUDIT.json)为准。旧 V4 和 Recovery 协调任务均因没有最终结果封为运行失败，**不是四条算法质量失败**。

## 当前唯一执行路由

- 任务：`four_stream_full_pipeline_v4_takeover_v2`
- 计划：`FOUR_STREAM_V4_TAKEOVER_V2`
- 任务包：[TASK_PACKET.json](../../tasks/current/four_stream_full_pipeline_v4_takeover_v2/TASK_PACKET.json)
- 运行根：`_run/current/four_stream_full_pipeline_v4_takeover_v2/attempts/attempt_0001/`
- 当前状态：[STATUS.json](STATUS.json)

首轮输入审计已经由登记入口实际执行并释放 writer；任务目前 `PENDING`。Exact78 仍缺合法配对候选索引；Controller/MANUS 三会话数据存在，旧“0916 缺失”由错误目录根造成，但 M0/M1 对照输入仍未完整绑定；HaWoR 缺 007/031 目标 producer；HuRo 缺真实会话同输入 runner。没有新算法质量通过或新 Robot 对照视频。

下一执行顺序：先为 Controller/MANUS 找到并 SHA 绑定冻结 HaWoR experiment，运行 097/098 开发、101 保留回归；同时为 HaWoR 与 HuRo 登记目标会话执行器。Exact78 必须先有真实因果 Raw/Robotized candidate index，否则保持 blocked。所有候选单轮有界；GPU 单租约；旧视频不得改名冒充新结果。

## 历史 V4 / Recovery 记录（不可执行）

## 历史 V4 执行路由（已终态，不可运行）

- 任务：`four_stream_full_pipeline_v4`
- 计划：`FOUR_STREAM_FULL_PIPELINE_V4`
- 任务包：[TASK_PACKET.json](../../tasks/current/four_stream_full_pipeline_v4/TASK_PACKET.json)
- 运行根：`_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/`
- 当前状态：[STATUS.json](STATUS.json)
- 当前进度：[PROGRESS_2H.json](../../_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/PROGRESS_2H.json)
- 活动 claim：[CLAIM_WINDOW.json](../../_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/CLAIM_WINDOW.json)

旧 V4 已因 coordinator-only runtime 终态化，仅保留运行合同审计；V3 也仅可通过 SHA 收据复用。不要把旧 V4/V3 状态或视频当作 recovery 已通过结果。源数据、processed、archive、sealed 和旧基线均只读。

## 已完成事实

1. V4 路由已登记，四条 lane 已使用独立 writer root、缓存和状态文件初始化。
2. V3 证据已按 SHA 生成复用清单，没有复制或覆盖旧产物。
3. Exact78 固定 cohort 已核对：156/156 身份交集，无 unresolved；train 59、validation 77、development_final 20。`development_final` 已暴露，不得称盲测。
4. WiLoR 权重目标当前不存在；DiffuEraser vendor 当前不存在。两项只能保持 blocker，不得伪造权重、`.done` 或结果。

## Recovery 当前 lane 状态

| lane | 当前状态 | 实证 | 下一步 |
|---|---|---|---|
| Exact78 | `BLOCKED_PREREQ` | 156/156 cohort rebind 成功；Chips/Poker 均无合法 V32 pair，candidate index 缺失 | 修复/登记真实 pair producer，不能用 Raw 复制替代 Robotized |
| Controller/MANUS | `BLOCKED_PREREQ` | 固定入口已执行；`play_cards_0916_097/098/101` 在指定 processed root 不存在 | 找到已批准的 0916 processed 输入或明确外部阻塞 |
| HaWoR/Retarget | `READY_CPU` | 合同 preflight 通过；新 R0 尚未运行 | 核对 prepared manifest、输入 SHA 和 GPU 权重后再运行 |
| HuRo | `READY_CPU` | 合同 preflight 通过；新 canary 尚未运行 | 固定共同输入与 wrist-objective spec 后运行开发 canary |

所有新结果仍为开发级；没有训练完成、质量通过、控制真值或部署资格。

## 旧 V4 四条 lane 封存时状态（非当前可执行状态）

| lane | 当前状态 | 当前动作/结论 | 下一步 |
|---|---|---|---|
| Exact78 | `READY_CPU` | 固定 cohort 已核验，配对生产尚未完成 | 按冻结清单生成真实 Raw/Robotized pair，完成一次 update/save/reload |
| Controller/MANUS | `READY_CPU` | 静态腕比较可开始；没有采用 M1/M2 的结论 | 在固定开发片段运行 M0/M1 对照，保留完整变换和数值 |
| HaWoR/Retarget | `BLOCKED_PREREQ` | 持久化 HaWoR 输入生产器缺失或未验证；V3 的 007/031 仅可复用 | 先闭合合法输入 producer/签名；不得借旧视频冒充新 R0 |
| HuRo | `READY_REUSE_AUDIT` | 当前只完成 V3 复用审计；没有新的 HuRo 质量结论 | 在共享冻结输入上做显式 wrist-objective canary；保持开发级、非部署 |

所有 lane 当前均为 `DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE`。目前没有 `NUMERIC_QUALITY_PASS`、`TRAINING_COMPLETE`、`TRAINING_ELIGIBLE` 或视觉验收通过结论；`control_ground_truth=false`、`physical_deployable=false`、`external_metric_authority=false`。

## Recovery 交接给下一位 AI 的顺序

```text
先读取 Recovery STATUS.json / lane STATE.json / Recovery RUN_SIGNATURE.json
        ↓
确认没有第二个 writer，完成逐会话 image-domain、期限、输入 producer 和算法合同 preflight
        ↓
Exact78：pair producer → ledger → 一次真实训练更新 → save/reload
Controller/MANUS：固定片段 M0/M1 数值对照
HaWoR：解决输入 producer blocker，或明确 BLOCKED_EXTERNAL
HuRo：只运行已授权的共同输入开发 canary
        ↓
更新各自 lane STATE、progress receipt 和当前文档
```

推荐入口检查：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli validate-governance
cat docs/current/STATUS.json
cat _run/current/four_stream_full_pipeline_v4_recovery_v1/attempts/attempt_0001/RUN_SIGNATURE.json
```

任何新结果都必须保留输入、代码、配置、环境和 SHA；失败实验与 blocker 不删除。GPU 仍然单租约，真实模型结果不能用旧视频或文字收据代替。
