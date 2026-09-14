# 当前项目实时事实页

> 本文档由机器状态自动生成。关键计数不得手工修改。

## A. 快照身份

- 状态生成时间：`2026-09-13T23:45:53+08:00`
- governance revision：`1064`
- generation id：`gov-001064-74f7b0fe690f`
- freshness：`FRESH`（age=0s）
- generator code SHA：`8c18566fbc9f548da23705f8d5cfc08e889c8319a1005daef6576ecc4772a519`
- repository：`1f97e25f99723ec3a6c73dfe6f3fb963ca515db5` / `main`
- host：`dsw-1019706-57c5b8df6-4vg6j`
- data root：`/mnt/data/egodata/datasets/ego`

## B. exact78 固定分母

- Raw：`156`
- Wave0 metric-ready：`58`
- Wave0 已有 Clean：`5`
- Wave0 待 Clean：`53`
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
| Clean | 58 | 5 | 0 | 0 | 53 | `EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_FINALS` |
| Contact | 2 | 0 | 0 | 0 | 2 | `POKER042_HYPOTHESIS_ONLY_NO_CONTACT_AUTHORITY` |
| Robot Visual | 156 | 0 | 1 | 0 | 155 | `NO_CURRENT_TASK_ROBOT_AUTHORITY_POKER042_4FRAME_FINAL_C` |
| HumanEgo Aux | 4 | 0 | 0 | 0 | 4 | `SCHEMA_READY_BUNDLES_BLOCKED_ROBOT_VISUAL` |
| HumanEgo Policy | 4 | 0 | 0 | 0 | 4 | `BLOCKED_EXTERNAL_REAL_ROBOT_ACTION` |

## D. 当前运行任务

当前无活跃任务。

## E. 当前阻塞

| 阻塞项 | 状态 | 影响范围 | 解除条件 |
|---|---|---|---|
| KaiHand adapter CAD | `BLOCKED_EXTERNAL` | Physical Robot authority | Verified adapter CAD and measured installation transform. |
| Robot TCP and installation calibration | `BLOCKED_EXTERNAL` | Physical Robot authority | Measured TCP, mount and camera/world-to-base calibration. |
| Real Robot action demonstrations | `BLOCKED_EXTERNAL` | HumanEgo policy checkpoints | Synchronized action/state/RGB data satisfying the formal schema. |
| Shared H20 occupied by external egotouch training | `BLOCKED_RESOURCE` | Wave0 ProPainter Clean GPU batch | Wait for the external egotouch process to exit, then re-run physical GPU, lease and disk preflight without preempting it. |
| Contact-aware Robot execution closure | `BLOCKED_EXTERNAL` | Poker042 contact-aware Robot retarget, Robot render, occlusion compositor and any Robot authority; Chips034 additionally remains BLOCKED_PREREQ upstream. Evidence: /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json. Poker042 human-contact output is HYPOTHESIS_ONLY. | Provide a verified NaturalV2 flange-to-KaiHand assembly transform or adapter CAD/measured installation transform, world-to-Tianji-base calibration, and Robot TCP/installation calibration. Then rerun only the frozen four-frame retarget gate before any 24-frame or full-session render; do not guess missing transforms from a prior visual proxy. |
| Contact/Robot 034/042 formal prerequisites | `BLOCKED_PREREQ` | Chips034 Human Contact remains blocked because its current Role Mask is Grade C and no formal Wave0 Depth/Object6D exists. Poker042 Human Contact is complete as HYPOTHESIS_ONLY; only its Robot retarget and compositor remain blocked. | For Chips034, pass the frozen Role Mask successor and publish a fresh formal Depth/Object6D delta before contact inference. For Poker042, do not rerun contact inference; close the separately recorded Robot assembly/world-to-base/TCP prerequisites, then run only the frozen four-frame retarget gate. Clean and Robot render are required later for the compositor, not for contact inference. |

## F. 当前可支持的结论

### 可以支持

- FoundationStereo disparity-to-depth formula and registration chain are internally closed for the current visual Depth products.（`SUPPORTED_INTERNAL_CONSISTENCY`；边界：Not an external millimetre-accuracy measurement.）
- Chips034 right hand shows a persistent negative HaWoR-versus-Stereo Z discrepancy.（`SUPPORTED_INTERNAL_CONSISTENCY`；边界：Cross-system surface difference; it does not establish which system is physically correct.）

### 不能支持或仅为假设

- The Chips034 discrepancy is primarily a HaWoR absolute-Z placement error.（`HYPOTHESIS_ONLY`；边界：Median-Z correction is diagnostic and no external hand-depth truth exists.）
- Current Object6D poses are physical ground truth.（`UNSUPPORTED`；边界：Observed visible-surface geometry only; occluded frames stay invalid.）
- Current visual Robot results are deployable or training-authorized.（`UNSUPPORTED`；边界：Central Robot authority and action sidecars are both absent.）
- Masquerade supports edited-human visual pretraining but does not supply contact truth or a complete contact-aware compositor for exact78.（`DEVELOPMENT_EVIDENCE`；边界：Method and limitation evidence only; it grants no local Contact, Robot, compositor, training or physical authority.）
- Poker042 can produce a hypothesis-only human-contact sidecar without waiting for Clean or Robot rendering.（`DEVELOPMENT_EVIDENCE`；边界：The HYPOTHESIS_ONLY artifact has now been produced; this claim still grants no physical contact truth, Robot solve, compositor result or authority.）
- Poker042 has a full-session human-contact hypothesis sidecar while formal Object6D remains unchanged.（`HYPOTHESIS_ONLY`；边界：Digital HaWoR fingertip-to-observed Object6D geometry only: 73 direct frames, 4 bounded inferred frames and 94 UNKNOWN; not physical contact truth or Robot authority.）

## G. 最新状态变化

### PASSED

- `exact78_v52_stage0` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v52/stage0/STAGE0_RESULT.json / `2026-09-13T23:45:53+08:00`
- `governance_recovery_v5` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v52_governance_recovery_v5/GOVERNANCE_RECOVERY_V5.json / `2026-09-13T23:45:42+08:00`
- `robot_historical_closure_recovery_v1` / `play_cards_0902_042` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_robot_visual_training_closure_recovery_v1/VISUAL_TRAINING_ASSEMBLY_CONTRACT.json / `2026-09-13T02:36:34+08:00`
- `workspace_cleanup_v3` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_root_cleanup_v3_lane_a/CLEANUP_RECEIPT.json / `2026-09-13T01:51:25+08:00`

### FAILED_QUALITY_C

- `robot_contact_retarget_poker042_visual_v1` / `play_cards_0902_042` / `FAILED_QUALITY_C` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_poker042_contact_retarget_fourframe_visual_v3/RESULT.json / `2026-09-13T03:18:35+08:00`
- `robot_contact_retarget_poker042_visual_v1` / `play_cards_0902_042` / `FAILED_QUALITY_C` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_poker042_contact_retarget_fourframe_visual_v2/RESULT.json / `2026-09-13T03:08:06+08:00`

### FAILED_RUNTIME

- `exact78_upstream_c_successors_v1` / `get_potato_chips_0902_034` / `FAILED_RUNTIME` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v52_governance_recovery_v5/GOVERNANCE_RECOVERY_V5.json / `2026-09-13T23:45:42+08:00`
- `exact78_wave0_clean_v1` / `get_potato_chips_0903_052` / `FAILED_RUNTIME` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v52_governance_recovery_v5/GOVERNANCE_RECOVERY_V5.json / `2026-09-13T23:45:42+08:00`
- `exact78_upstream_c_successors_v1` / `get_potato_chips_0902_034` / `FAILED_RUNTIME_RETRYABLE` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v3_lane_c_successors_v1/WAIT_GUARDIAN_RUNTIME_RECOVERY_20260913T0222.json / `2026-09-13T02:26:49+08:00`

### BLOCKED

- `contact_occlusion_canary_034_042_v1` / `get_potato_chips_0902_034,play_cards_0902_042` / `BLOCKED_EXTERNAL` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary_v3_visual_v1/poker/play_cards_0902_042/RESULT.json / `2026-09-13T02:22:58+08:00`
- `contact_occlusion_canary_034_042_v1` / `get_potato_chips_0902_034,play_cards_0902_042` / `BLOCKED_EXTERNAL` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json / `2026-09-13T02:17:29+08:00`
- `contact_occlusion_canary_034_042_v1` / `get_potato_chips_0902_034,play_cards_0902_042` / `BLOCKED_EXTERNAL` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json / `2026-09-13T02:13:54+08:00`
- `contact_occlusion_canary_034_042_v1` / `get_potato_chips_0902_034,play_cards_0902_042` / `BLOCKED_PREREQ` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary_v2/INPUT_CLOSURE_AND_4FRAME_GO_NO_GO.json / `2026-09-13T02:01:25+08:00`
- `contact_occlusion_canary_034_042_v1` / `get_potato_chips_0902_034,play_cards_0902_042` / `BLOCKED_PREREQ` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_contact_robot_v1_canary/INPUT_CLOSURE_AND_4FRAME_GO_NO_GO.json / `2026-09-13T01:48:50+08:00`

## H. 下一任务

- next_task_id：`exact78_v52_lane_a_clean`
- next_session：`-`
- prerequisites：`exact78_v52_stage0`
- expected_resource：`CPU preflight first; GPU only after 3x10s gate. Lane C fixtures, E and F may run CPU-only.`
- stop_condition：`Wave0 SHA/signature conflict, prerequisite failure or GPU wait exceeding 30 minutes.`

## 固定读取协议

回答进度、精度或 authority 前，必须先校验 `CURRENT_STATUS_RECEIPT.json` 绑定的文件SHA；若状态为 `STALE`、`SUSPECTED_DEAD_WORKER` 或 `STATUS_CONFLICT`，停止推断并先做恢复审计。
