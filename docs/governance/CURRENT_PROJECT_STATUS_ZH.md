# 当前项目实时事实页

> 本文档由机器状态自动生成。关键计数不得手工修改。

## A. 快照身份

- 状态生成时间：`2026-09-18T17:24:53+08:00`
- governance revision：`12032`
- generation id：`gov-012032-27509b9465df`
- freshness：`FRESH`（age=0s）
- generator code SHA：`aa4a5b939dbdd5c0212301f8b12ec0766287017a3f477f39953db9e00d715120`
- repository：`b4918fe4817c432914be7079e3874f918ea40351` / `task/0915-full-funnel-0916-clean-v1`
- host：`dsw-1019706-57c5b8df6-4vg6j`
- data root：`/mnt/data/egodata/datasets/ego`

## B. exact78 固定分母

- Raw：`156`
- Wave0 metric-ready：`58`
- Wave0 已有 Clean：`58`
- Wave0 待 Clean：`0`
- Wave0 Clean 质量 C：`0`
- Wave0 Clean 运行失败终态：`0`
- Wave0 Clean 前置阻塞：`0`
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
| Depth | 58 | 0 | 0 | 0 | 58 | `NO_CURRENT_DEPTH_AUTHORITY_WRONG_VST_IMAGE_DOMAIN` |
| Object6D | 58 | 0 | 0 | 0 | 58 | `NO_CURRENT_OBJECT6D_AUTHORITY_BLOCKED_UPSTREAM_DEPTH_WRONG_VST_IMAGE_DOMAIN` |
| Clean | 58 | 58 | 0 | 0 | 0 | `EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_SESSION_TERMINALS` |
| Contact | 2 | 0 | 0 | 0 | 2 | `NO_CURRENT_CONTACT_AUTHORITY_UPSTREAM_OBJECT6D_WITHDRAWN` |
| Robot Visual | 156 | 0 | 1 | 0 | 155 | `NO_CURRENT_TASK_ROBOT_AUTHORITY_POKER042_4FRAME_FINAL_C` |
| HumanEgo Aux | 4 | 0 | 0 | 0 | 4 | `SCHEMA_READY_BUNDLES_BLOCKED_ROBOT_VISUAL` |
| HumanEgo Policy | 4 | 0 | 0 | 0 | 4 | `BLOCKED_EXTERNAL_REAL_ROBOT_ACTION` |

> Clean 的 PASSED 数量只表示当前冻结合同下的结构、来源和解码终态；不表示接触边界、隐藏物体外观或 donor 语义已经通过。Robotized/Visual Aux 还必须单独通过 contact-preservation、pixel-source legality 和 Occlusion 门。

## D. 当前运行任务

当前无活跃任务。

## E. 当前阻塞

| 阻塞项 | 状态 | 影响范围 | 解除条件 |
|---|---|---|---|
| KaiHand adapter CAD | `BLOCKED_EXTERNAL` | Physical Robot authority | Verified adapter CAD and measured installation transform. |
| Robot TCP and installation calibration | `BLOCKED_EXTERNAL` | Physical Robot authority | Measured TCP, mount and camera/world-to-base calibration. |
| Real Robot action demonstrations | `BLOCKED_EXTERNAL` | HumanEgo policy checkpoints | Synchronized action/state/RGB data satisfying the formal schema. |
| Contact-aware Robot execution closure | `BLOCKED_EXTERNAL` | Poker042 contact-aware Robot retarget, Robot render, occlusion compositor and any Robot authority; Chips034 additionally remains BLOCKED_PREREQ upstream. Evidence: archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json. Poker042 human-contact output is HYPOTHESIS_ONLY. | Provide a verified NaturalV2 flange-to-KaiHand assembly transform or adapter CAD/measured installation transform, world-to-Tianji-base calibration, and Robot TCP/installation calibration. Then rerun only the frozen four-frame retarget gate before any 24-frame or full-session render; do not guess missing transforms from a prior visual proxy. |
| Contact/Robot 034/042 formal prerequisites | `BLOCKED_PREREQ` | Chips034 Human Contact remains blocked because its current Role Mask is Grade C and no formal Wave0 Depth/Object6D exists. Poker042 Human Contact is complete as HYPOTHESIS_ONLY; only its Robot retarget and compositor remain blocked. | For Chips034, pass the frozen Role Mask successor and publish a fresh formal Depth/Object6D delta before contact inference. For Poker042, do not rerun contact inference; close the separately recorded Robot assembly/world-to-base/TCP prerequisites, then run only the frozen four-frame retarget gate. Clean and Robot render are required later for the compositor, not for contact inference. |
| Sensor-line DEPTH-10/20 play_cards_0910_001 prerequisites | `BLOCKED_PREREQ` | New sensor-line play_cards_0910_001 Stereo/DEPTH-20 only; sourceIndex code is fixed, exact78 Depth and Controller/MANUS branches are unchanged. | Recover or verify same-session rectification that passes median <=2 px and P90 <=5 px on held-out frames. Current corrected-source P90 is 47.98/35.09/14.31 px. Do not request GPU or run FoundationStereo until the CPU gate passes. |
| Clean-20/21 causal semantic prerequisites | `BLOCKED_PREREQ` | Poker245 and Chips039 Clean successor only; Wave0 R7_0 structural terminals remain immutable. | Build causal semantic donor rejection and lossless frame/source-map closure; build a verified Poker atlas; only then request GPU for fresh ProPainter. Current causal UNKNOWN/write is 82.36% Poker and 69.90% Chips. |
| Mask challenger R7_3 bounded NO_GO | `BLOCKED_PREREQ` | Role C cluster and Poker Object Mask successor only; SAM3.1 current authority and existing A/B results unchanged. | Provide a reviewed four-role prompt/adapter closure for Role Mask. For Poker object re-entry, improve causal memory/re-detection so the frozen failed canary passes known coverage and post-reentry overlap before running Poker001/005 regression. SAM2.1 current run achieved only 13.785% known coverage; Cutie OOM is runtime evidence, not quality evidence. |
| R2.2 Mask canary not executed | `BLOCKED_RESOURCE` | Poker and Chips independent SAM3.1 canaries | Acquire the serialized GPU lease and run the frozen candidate/reference canaries; absence of pixel gold limits results to development metrics. |
| R2.2 causal Clean prerequisites incomplete | `BLOCKED_PREREQ` | Poker245 and Chips039 prefix-only Clean successor | Publish support-surface semantics and lossless successor RGB/source-map pairs; Poker also requires a pose-verified causal atlas warp. |
| R2.2 same-session rectification gate failed | `BLOCKED_PREREQ` | play_cards_0910_001 FoundationStereo and Stereo wrist proxy | Provide a verified same-session selected-eye/SBS mapping and rectification whose vertical epipolar residual P90 is at most 5 px with the frozen coverage gate. |
| 0915 left-mono chips instance identity propagation | `BLOCKED_PREREQ` | get_potato_chips_0915_001 Object Mask and dependent Object6D/Clean/Contact/Robot/HumanEgo only. | Supply reviewed task-specific three-instance temporal supervision or a pinned object tracker/segmentation weight that propagates the three correct frame-0 chips seeds. Do not synthesize missing tracker roles. |

## F. 当前可支持的结论

### 可以支持

- 当前无已发布支持结论。

### 不能支持或仅为假设

- FoundationStereo disparity-to-depth formula and registration chain are internally closed for the current visual Depth products.（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Chips034 right hand shows a persistent negative HaWoR-versus-Stereo Z discrepancy.（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- The Chips034 discrepancy is primarily a HaWoR absolute-Z placement error.（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Current Object6D poses are physical ground truth.（`UNSUPPORTED`；边界：Observed visible-surface geometry only; occluded frames stay invalid.）
- Current visual Robot results are deployable or training-authorized.（`UNSUPPORTED`；边界：Central Robot authority and action sidecars are both absent.）
- Masquerade supports edited-human visual pretraining but does not supply contact truth or a complete contact-aware compositor for exact78.（`DEVELOPMENT_EVIDENCE`；边界：Method and limitation evidence only; it grants no local Contact, Robot, compositor, training or physical authority.）
- Poker042 can produce a hypothesis-only human-contact sidecar without waiting for Clean or Robot rendering.（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Poker042 has a full-session human-contact hypothesis sidecar while formal Object6D remains unchanged.（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Robot v5.2首个v73批次的3条严格C由软姿态门触发；三条轨迹均通过全片数字URDF碰撞及硬时序/限位审计。（`DEVELOPMENT_EVIDENCE`；边界：数字URDF和跨系统软姿态诊断；不授予Robot、Contact、控制或物理部署authority。）
- 双手同时可见的H50硬门会拒绝单侧有效的第一视角会话；087在逐侧valid掩码下由0个恢复为233个因果H50窗口。（`DEVELOPMENT_EVIDENCE`；边界：单会话工程诊断；只支持masked future-2D辅助训练资格，不支持Robot、动作、Contact或物理authority。）
- 087硬几何候选已形成严格因果的Raw/Robotized Visual Aux bundle；逐侧valid掩码保留233个H50窗口。（`DEVELOPMENT_EVIDENCE`；边界：Development Visual Aux bundle only; unresolved overlap is excluded, not solved. No occlusion, contact, Robot action, policy or deployment authority.）
- 087真实会话已闭合Robot optical-Z与Stereo可见物体表面的4帧前后关系canary；接触窄带已知覆盖81.47%，UNKNOWN 18.53%。（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Occlusion compositor只在Robot/物体真实重叠区要求Stereo排序后，087连续接触窗的物体条件保留率中位数由77.48%提高到95.92%，但仍未达到99% Silver门。（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- 087连续24帧可见表面z-buffer successor的物体条件保留率按像素加权为97.01%，已消除非重叠物体被深度门误降背景的问题，但仍未达到99% Silver门。（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- Chips087连续24帧可见表面Robot/Object光学Z诊断在边缘邻域一致性约束后达到99.72%条件物体像素保留率，known coverage 84.12%，UNKNOWN 15.88%。（`WITHDRAWN`；边界：Withdrawn from current consumption: the supporting Stereo/Object6D evidence was derived after applying forbidden lens remapping to already-undistorted VST pixels.）
- V7.1 Visual Aux has bounded routing capacity for both Chips and Poker pairs（`DEVELOPMENT_EVIDENCE`；边界：Capacity forecast only: Chips 17 train/4 validation potential and Poker 20 train/4 validation potential; pending Robot/Clean/bundle receipts are not completed datasets or checkpoints.）
- Chips087 right-pinky frame75 exceeds the current sampled KaiHand tip-direction reachable floor（`DEVELOPMENT_EVIDENCE`；边界：One-frame deterministic sampled digital reachability diagnosis under fixed wrist and current URDF limits; not global physical reachability, human truth, Robot authority or calibration.）
- A two-candidate Robot base-backoff subset reproduced the full-sweep winner on the first six completed Chips sessions.（`DEVELOPMENT_EVIDENCE`；边界：Retrospective six-session Chips replay only; 71.4% fewer candidate evaluations is not measured wall-clock speedup, Poker generality, Robot authority, or evidence for unseen sessions.）
- Hand round-2 produced no frozen summary-metric improvement on the first six completed Chips sessions.（`DEVELOPMENT_EVIDENCE`；边界：Six-session Chips summary and recorded-duration audit only; not proof that round-2 is globally redundant and not permission to mutate R7.3.）
- Frozen HaWoR temporal diagnostic contracts have executable CPU reproductions（`DEVELOPMENT_EVIDENCE`；边界：CPU post-processing diagnostics over existing HaWoR tracks only; not HaWoR re-inference, weight provenance, successor authority, external 3D truth or Wave delta.）

## G. 最新状态变化

### PASSED

- `0915_stereo_encoded_domain_preflight_v1` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/_run/current/0915_stereo_encoded_domain_preflight_v1/attempts/attempt_0001/RESULT.json / `2026-09-18T17:08:21+08:00`
- `0915_removal_envelope_single_session_canary_v1` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/_run/current/0915_removal_envelope_single_session_canary_v1/attempts/attempt_0001/RESULT.json / `2026-09-18T14:56:58+08:00`
- `0915_sam31_weak_role_canary_v1` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/_run/current/0915_sam31_weak_role_canary_v1/attempts/attempt_0001/RESULT.json / `2026-09-18T14:11:17+08:00`
- `0915_stereo_interaction_cpu_canary_v1` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/_run/current/0915_stereo_interaction_cpu_canary_v1/attempts/attempt_0001/RESULT.json / `2026-09-18T13:56:44+08:00`
- `0915_sam31_strict_role_canary_v5` / `-` / `PASSED` / /mnt/workspace/code/chaoyang/_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001/RESULT.json / `2026-09-18T12:47:38+08:00`

### FAILED_QUALITY_C

- 无。

### FAILED_RUNTIME

- `0915_sam31_strict_role_canary_v4` / `-` / `FAILED_RUNTIME_FINAL` / /mnt/workspace/code/chaoyang/_run/current/0915_sam31_strict_role_canary_v4/attempts/attempt_0001/RESULT.json / `2026-09-18T12:35:30+08:00`
- `0915_sam31_strict_role_canary_v2` / `-` / `FAILED_RUNTIME_FINAL` / /mnt/workspace/code/chaoyang/_run/current/0915_sam31_strict_role_canary_v2/attempts/attempt_0001/RESULT.json / `2026-09-18T12:23:33+08:00`

### BLOCKED

- 无。

## H. 下一任务

状态机当前未选择下一任务。

## 固定读取协议

回答进度、精度或 authority 前，必须先校验 `CURRENT_STATUS_RECEIPT.json` 绑定的文件SHA；若状态为 `STALE`、`SUSPECTED_DEAD_WORKER` 或 `STATUS_CONFLICT`，停止推断并先做恢复审计。
