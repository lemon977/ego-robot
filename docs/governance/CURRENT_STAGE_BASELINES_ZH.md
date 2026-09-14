# 当前各阶段算法与整改需求

> 本页随事实账本原子生成；运行数量以 `CURRENT_PROJECT_STATUS_ZH.md` 为准。

| 阶段 | 当前算法 | 当前授权 | 主要边界 | 下一整改 |
|---|---|---|---|---|
| Raw | `exact78_frozen_cohort_v1` | `FROZEN_EXACT78_COHORT` | 0909/0910 acquisition-aligned v2 is a separate release and denominator. | No successor changes the frozen exact78 denominator. |
| HaWoR | `hawor_bounded_v2` | `BOUNDED_V2_TERMINALS` | Monocular MANO 3D and Z are not external metric truth. | Only bounded canary plus fixed regression may replace a C cluster. |
| Role Mask | `sam31_role_successor_v3` | `SAM31_ROLE_SUCCESSOR_V3` | Four role masks are independent; C sessions are not downstream-authorized. | Fresh representative canary and two A/B regressions. |
| Object Mask | `sam31_task_object_identity_v1` | `TASK_OBJECT_IDENTITY` | Chips physical instances may never be unioned to pass an identity gate. | Action-conditioned identity reacquisition with fail-closed ambiguity. |
| Depth | `foundationstereo_corrected_metric_v1` | `VISUAL_OBJECT6D_CANDIDATE_INPUT` | FoundationStereo exposes no native confidence in this contract.；Internal metric closure is not external millimetre accuracy. | External 30/50/70/100 cm validation is required for physical accuracy claims. |
| Object6D | `object6d_observed_visible_surface_v1` | `OBSERVED_ONLY_KEEP_INVALID` | Only visible-surface geometry is measured; occluded frames remain invalid.；Plane residual is not external pose truth. | A separate hypothesis sidecar may bridge bounded gaps but cannot overwrite formal Object6D. |
| Clean | `real_temporal_stereo_donor_then_propainter_v1` | `EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_SESSION_TERMINALS` | Generated pixels are visual completion, not physical background truth.；Clean never feeds Depth/Object6D/contact truth. | Finish frozen Wave0 terminals without changing selection. |
| Contact | `human_contact_hypothesis_v1_geometry_v2` | `POKER042_HYPOTHESIS_ONLY_NO_CONTACT_AUTHORITY` | Geometry fixtures pass, but the real-session goldset pack is explicitly blocked pending two independent human labels and a Robot candidate.；Hypotheses are not physical contact truth. | Freeze and double-review the real contact/occlusion goldset before authority. |
| Robot Visual | `world_first_retarget_v5_2_first_observed_anchor` | `NO_CURRENT_TASK_ROBOT_AUTHORITY_POKER042_4FRAME_FINAL_C` | No Robot authority is currently published; 23 review videos are development candidates only (3 strict-numeric-pass, 5 pose-only, 15 quality-C diagnostics).；control_ground_truth=false.；Adapter/TCP/install/world-to-base physical calibration is absent. | Four keyframes, then 24 frames, then full session; at most two bounded successors. |
| Occlusion | `occlusion_compositor_v1_development_interface` | `NO_CURRENT_AUTHORITY` | The review pack remains BLOCKED_EXTERNAL_PENDING_TWO_INDEPENDENT_HUMAN_LABELS_AND_ROBOT_CANDIDATE; occlusion is not yet solved. | Goldset first, then task canaries; compositor never feeds back into retarget. |
| HumanEgo Aux | `h50_future_2d_visual_aux_v5_2` | `SCHEMA_READY_BUNDLES_BLOCKED_ROBOT_VISUAL` | Zero current checkpoints.；Visual retarget labels are not Robot control actions. | Train only after Robotized visual authority and eligibility gates pass. |
| HumanEgo Policy | `blocked_external_real_robot_action` | `BLOCKED_EXTERNAL_REAL_ROBOT_ACTION` | Real synchronized Robot action supervision is absent; visual trajectories cannot replace it. | Remain BLOCKED_EXTERNAL until real actions exist. |

## 固定解释边界

- HaWoR 单目三维、Stereo/Object6D 内部残差与 Robot 数字接触距离都不是物理真值。
- `visual_robot_trajectory_sidecar` 始终标记 `control_ground_truth=false`，不能冒充真实 Robot action。
- Contact 与 Occlusion 只有接口/几何开发证据；真实 goldset 未闭合，因此不能声称遮挡关系已经解决。
- 当前基线只由本页同 revision 的机器注册表与 receipt 解释；旧 README、目录名和聊天不构成 authority。
