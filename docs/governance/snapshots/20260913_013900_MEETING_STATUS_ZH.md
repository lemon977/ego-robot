# 当前项目实时事实页

> 本文档由机器状态自动生成。关键计数不得手工修改。

## A. 快照身份

- 状态生成时间：`2026-09-13T01:37:48+08:00`
- governance revision：`1`
- generation id：`gov-000001-9abc9c81aa87`
- freshness：`FRESH`（age=0s）
- generator code SHA：`04068a63aa571d0982d25a8cb17515bd08bb9c9d5456b37f9fe191d1398bab63`
- repository：`1f97e25f99723ec3a6c73dfe6f3fb963ca515db5` / `main`
- host：`dsw-1019706-57c5b8df6-4vg6j`
- data root：`/mnt/data/egodata/datasets/ego`

## B. exact78 固定分母

- Raw：`156`
- Wave0 metric-ready：`58`
- Wave0 已有 Clean：`4`
- Wave0 待 Clean：`54`
- Wave1 新增：`0`
- Wave2 新增：`0`
- 整个 exact78 缺标定：`59/156`
- 三路 A/B 中缺标定：`43/101`

## C. 各阶段实时矩阵

| 阶段 | 总数 | PASSED | C | RUNNING | BLOCKED | 当前 authority |
|---|---:|---:|---:|---:|---:|---|
| Raw | 156 | 156 | 0 | 0 | 0 | `FROZEN_EXACT78_COHORT` |
| HaWoR | 156 | 144 | 12 | 0 | 0 | `BOUNDED_V2_TERMINALS` |
| Role Mask | 156 | 124 | 32 | 0 | 0 | `SAM31_ROLE_SUCCESSOR_V3` |
| Object Mask | 156 | 121 | 35 | 0 | 0 | `TASK_OBJECT_IDENTITY` |
| Depth | 58 | 58 | 0 | 0 | 0 | `VISUAL_OBJECT6D_CANDIDATE_INPUT` |
| Object6D | 58 | 58 | 0 | 0 | 0 | `OBSERVED_ONLY_KEEP_INVALID` |
| Clean | 58 | 4 | 0 | 0 | 54 | `POKER4_EXPANDED_ROLE_V3_ONLY` |
| Contact | 2 | 0 | 0 | 0 | 2 | `NO_CURRENT_CONTACT_AUTHORITY` |
| Robot Visual | 156 | 0 | 0 | 0 | 156 | `NO_CURRENT_TASK_ROBOT_AUTHORITY` |
| HumanEgo Aux | 4 | 0 | 0 | 0 | 4 | `NOT_STARTED` |
| HumanEgo Policy | 4 | 0 | 0 | 0 | 4 | `BLOCKED_REAL_ROBOT_ACTION` |

## D. 当前运行任务

当前无活跃任务。

## E. 当前阻塞

| 阻塞项 | 状态 | 影响范围 | 解除条件 |
|---|---|---|---|
| KaiHand adapter CAD | `BLOCKED_EXTERNAL` | Physical Robot authority | Verified adapter CAD and measured installation transform. |
| Robot TCP and installation calibration | `BLOCKED_EXTERNAL` | Physical Robot authority | Measured TCP, mount and camera/world-to-base calibration. |
| Real Robot action demonstrations | `BLOCKED_EXTERNAL` | HumanEgo policy checkpoints | Synchronized action/state/RGB data satisfying the formal schema. |

## F. 当前可支持的结论

### 可以支持

- FoundationStereo disparity-to-depth formula and registration chain are internally closed for the current visual Depth products.（`SUPPORTED_INTERNAL_CONSISTENCY`；边界：Not an external millimetre-accuracy measurement.）
- Chips034 right hand shows a persistent negative HaWoR-versus-Stereo Z discrepancy.（`SUPPORTED_INTERNAL_CONSISTENCY`；边界：Cross-system surface difference; it does not establish which system is physically correct.）

### 不能支持或仅为假设

- The Chips034 discrepancy is primarily a HaWoR absolute-Z placement error.（`HYPOTHESIS_ONLY`；边界：Median-Z correction is diagnostic and no external hand-depth truth exists.）
- Current Object6D poses are physical ground truth.（`UNSUPPORTED`；边界：Observed visible-surface geometry only; occluded frames stay invalid.）
- Current visual Robot results are deployable or training-authorized.（`UNSUPPORTED`；边界：Central Robot authority and action sidecars are both absent.）

## G. 最新状态变化

### PASSED

- `governance_fact_ledger_v1` / `-` / `PASSED` / no-result / `2026-09-13T01:37:47+08:00`

### FAILED_QUALITY_C

- 无。

### FAILED_RUNTIME

- 无。

### BLOCKED

- 无。

## H. 下一任务

- next_task_id：`exact78_wave0_freeze_v1`
- next_session：`-`
- prerequisites：`governance_fact_ledger_v1`
- expected_resource：`CPU/read-only evidence hashing`
- stop_condition：`Any source artifact bytes/SHA mismatch or denominator conflict.`

## 固定读取协议

回答进度、精度或 authority 前，必须先校验 `CURRENT_STATUS_RECEIPT.json` 绑定的文件SHA；若状态为 `STALE`、`SUSPECTED_DEAD_WORKER` 或 `STATUS_CONFLICT`，停止推断并先做恢复审计。
