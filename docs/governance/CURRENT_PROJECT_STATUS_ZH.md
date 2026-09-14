# 当前项目实时事实页

> 本文档由机器状态自动生成。关键计数不得手工修改。

## A. 快照身份

- 状态生成时间：`2026-09-14T19:12:16+08:00`
- governance revision：`5869`
- generation id：`gov-005869-1eff2016fad2`
- freshness：`FRESH`（age=0s）
- generator code SHA：`feadae102fb1e784711daa615fd7475fbaa95d6eddd5cc0ebc1506e5694afc70`
- repository：`1f97e25f99723ec3a6c73dfe6f3fb963ca515db5` / `main`
- host：`dsw-1019706-57c5b8df6-4vg6j`
- data root：`/mnt/data/egodata/datasets/ego`

## B. exact78 固定分母

- Raw：`156`
- Wave0 metric-ready：`58`
- Wave0 已有 Clean：`39`
- Wave0 待 Clean：`0`
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
| Clean | 58 | 39 | 0 | 0 | 19 | `EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_SESSION_TERMINALS` |
| Contact | 2 | 0 | 0 | 0 | 2 | `POKER042_HYPOTHESIS_ONLY_NO_CONTACT_AUTHORITY` |
| Robot Visual | 156 | 0 | 1 | 0 | 155 | `NO_CURRENT_TASK_ROBOT_AUTHORITY_POKER042_4FRAME_FINAL_C` |
| HumanEgo Aux | 4 | 0 | 0 | 0 | 4 | `SCHEMA_READY_BUNDLES_BLOCKED_ROBOT_VISUAL` |
| HumanEgo Policy | 4 | 0 | 0 | 0 | 4 | `BLOCKED_EXTERNAL_REAL_ROBOT_ACTION` |

## D. 当前运行任务

| task_id | session | phase | attempt | PID | GPU | heartbeat | 状态 |
|---|---|---|---:|---:|---:|---|---|
| `exact78_v52_lane_c_contact_robot` | `get_potato_chips_0902_067` | `robot_067_finite_continuation` | 2 | 2593657 | - | `2026-09-14T19:12:16+08:00` | `RUNNING` |

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

- `chaoyang_current_only_cleanup_v6` / `workspace` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260914_chaoyang_cleanup_v6/FINAL_RESULT.json / `2026-09-14T18:37:33+08:00`
- `chaoyang_current_only_cleanup_v6` / `workspace` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260914_chaoyang_cleanup_v6/FINAL_RESULT.json / `2026-09-14T18:36:45+08:00`
- `exact78_wave0_clean_v1` / `play_cards_0902_042` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/TERMINAL_AUDIT_V52_1.json / `2026-09-14T18:36:45+08:00`
- `exact78_v52_lane_a_clean_successor_v521` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v52/lane_a_clean_successor_v521/RUNNER_RESULT.json / `2026-09-14T16:24:01+08:00`
- `exact78_wave0_clean_v1` / `play_cards_0902_042` / `PASSED` / /mnt/workspace/code/chaoyang/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/TERMINAL_AUDIT_V52_1.json / `2026-09-14T16:24:01+08:00`

### FAILED_QUALITY_C

- 无。

### FAILED_RUNTIME

- 无。

### BLOCKED

- 无。

## H. 下一任务

- next_task_id：`exact78_v52_lane_c_contact_robot`
- next_session：`-`
- prerequisites：`exact78_wave0_clean_v1, contact_geometry_v2_fixture`
- expected_resource：`CPU Robot retarget plus serialized GPU render when needed`
- stop_condition：`All 156 sessions receive METRIC_CONTACT_ROBOT, POSE_ONLY_VISUAL_ROBOT, FAILED_QUALITY_C, FAILED_RUNTIME_FINAL, or BLOCKED_PREREQ terminal; no physical authority promotion.`

## 固定读取协议

回答进度、精度或 authority 前，必须先校验 `CURRENT_STATUS_RECEIPT.json` 绑定的文件SHA；若状态为 `STALE`、`SUSPECTED_DEAD_WORKER` 或 `STATUS_CONFLICT`，停止推断并先做恢复审计。
