# 2026-09-02 可复现证据索引

状态：`SEALED_WITH_EXPLICIT_HOLDS`。这是本轮 Mask / Clean / Robot / 新任务合同与几何准备的统一入口。历史 `_run` 产物没有被修改；所有 PASS 都只在表中声明的 scope 内成立。

## 先看这些

- [系统复现指南](SYSTEM_REPRODUCTION_GUIDE_ZH.md)：环境、命令、generic/per-session/session-patch 分类和恢复顺序。
- [质量门与理由](QUALITY_GATES_AND_RATIONALE_ZH.md)：每个 PASS/HOLD 的判定条件。
- [失败谱系与修复](FAILURES_AND_FIXES_ZH.md)：已发生故障、后继修复和未关闭项。
- [算法与权重 manifest](ALGORITHM_AND_WEIGHT_MANIFEST.json)：精确路径、bytes、SHA-256、来源、许可证与 claim scope。
- [机器索引](ARTIFACT_INDEX.json)：本页的机器可读映射；该文件按惯例 self-excluded。

## 当前终态

| 系统 | 终态 | Authority | Claim 边界 |
|---|---|---|---|
| Mask | `CHIPS_ASSISTED_FULL_PASS / POKER_TRACKER_HOLD` | [v4 通用性矩阵](../../../_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1/GENERALIZATION_MATRIX_V4.md) · [chips RESULT](../../../_run/gpt_mask_assisted_bilateral_flow_refresh_chips_20260902_v4/RESULT.json) · [poker RESULT](../../../_run/gpt_mask_assisted_bilateral_flow_refresh_poker_20260902_v4/RESULT.json) | chips 360/360 可消费；同参数 poker tracker HOLD、无 full union；不是零人工/盲测/正式 PICO/跨任务通用 PASS |
| Clean L2 | `SOURCE_READINESS_COMPLETE` | [L2 REPORT](../../../_run/newtask_baseline_clean_20260901_233238/REPORT.md) | 路线与源准备；没有正式 CLEAN_RGB |
| Clean L3 | `HOLD_HUMAN_SEMANTIC_CLOSURE_UNRESOLVED` | [L3 REPORT](../../../_run/newtask_baseline_clean_l3_20260902_v1/REPORT.md) · [VALIDATION](../../../_run/newtask_baseline_clean_l3_20260902_v1/VALIDATION.json) | 10 个手机帧；61,202 copied pixels byte-exact；cross-video donor=0；ORB 非 K/c2w |
| Clean chips frame0 | `HOLD_CROSS_CAMERA_STATIC_INSUFFICIENT` | [RESULTS](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/RESULTS.json) · [protocol](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/PROTOCOL.json) | bilateral Mask 可消费且手已替换；跨机位 static 重复对象/扭曲；未启动 10 帧/全片 |
| Robot 004 | `PASS_F345_F402_CPU_READY_FOR_WINDOW_EXPANSION / FULL460_HOLD` | [MANO/Kai RESULTS](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/RESULTS.json) · [ORDER authority](../../../_run/robot_final_v3_mano_order_authority_cpu_20260902_v1/ORDER_AUTHORITY_AUDIT.json) · [timeline](../../../_run/robot_004_mano21_contact_timeline_cpu_20260902_v1/CONTACT_TIMELINE.json) | 正确 authority 下仅 f345/f402 单侧 CPU PASS；未做 10-frame CAD、新 full460/render 或 cross-session；旧 227/224 抓取语义撤销 |
| Robot 跨会话 | `PASS_DOWNSTREAM_EVENT_RESCUE_TRACKER_HOLDS_PRESERVED` | [rescue INDEX](../../../_run/robot_cross_session_event_rescue_cpu_20260902_v1/INDEX.md) · [独立 QA](../../../_run/robot_cross_session_event_rescue_independent_qa_20260902_v1/INDEX.md) | 002/035/065 protection 不变；068 双侧与 139 side0 downstream rescue PASS；035/065/068/139 数据门仍 HOLD；contact_required=0，无抓取 claim |
| 新任务合同 | `COMPLETE_FULL` | [v2.2 QA](../../../_run/newtask_contract_v2_2/QA_REPORT.md) | 文件解码权威；同帧尺子/对象内容仍需人工 |
| 新任务几何 | `COMPLETE_APPROX_GEOMETRY` | [Geometry INDEX](../../../_run/newtask_geometry_prep_v2/INDEX.md) · [REPRODUCE](../../../_run/newtask_geometry_prep_v2/REPRODUCE.md) | synthetic/APPROX；formal PICO metric/pose pending |

Robot window successor 的历史实际命令（脚本绑定历史 root，不得原地重跑）：

```bash
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_window_initializer_dp_cpu_20260902_v1/run_window_init_dp.py
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_temporal_blocker_audit_cpu_20260902_v1/audit_temporal.py
```

initializer runner SHA-256 `7145eede6e9a4fea8af72f77e9886bfd14782c1b4cf0b8095e428a54458f4c68`；initializer `RESULTS.json` SHA-256 `5d96bd8aebfcfe42c86b1d6ec2742420587bbf4d33b45b3f90d55a67ddac82f5`；temporal runner SHA-256 `93d3fb7933b5296877ca9507cf1ee2ad83dae589c184605f217599a22232899e`；temporal `RESULTS.json` SHA-256 `e0b130e9724de4ebe04c89d6ad2cd4a11a28d51f35c963eacc33fde46f1d245a`。完整 path/bytes/provenance/license/claim scope 见机器 manifest。

Event rescue 历史命令为 `/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_event_rescue_cpu_20260902_v1/run_event_rescue.py`；独立 QA 命令为 `/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_event_rescue_independent_qa_20260902_v1/run_independent_qa.py`。rescue `RESULTS.json` SHA-256 `ac49051ab010bd967dcb6f842ad65831a165ae9017e170ea5971cd86682b3d85`；QA `RESULTS.json` SHA-256 `8646c775c4af16d01b4514aaded7393b9e0f331429d251456b3a5261b873108b`。

## 直接看视频和图

### Robot MANO21 / KaiHand（当前手部 authority）

- [当前 Robot 入口与撤销说明](../../../NOW/robot/README.md)。旧 full460 的 `227/224`
  只能保留错误标签下的遮挡统计；所有旧视频均非当前 hand contact/nonpenetration/grasp
  authority。
- [ORDER_AUTHORITY_AUDIT](../../../_run/robot_final_v3_mano_order_authority_cpu_20260902_v1/ORDER_AUTHORITY_AUDIT.json)：SHA-256 `ab6a280589a26c34e3c1c3e495911e00e26c66055b8199fb47fa1e128a7c29c1`，40,130 B；MANO21 wrist0、tips 4/8/12/16/20、MCP 2/5/9/13/17，864 个 root 对照点。
- [f345/f402 RESULTS](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/RESULTS.json)：SHA-256 `01f792ccae11fd379ae6cb0379b7ce107f0b6e6500ea360938e92c7a59fd98c7`，31,909 B；[PROTOCOL](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/PROTOCOL.json)：`894d4bd1112fe859a7f9cfc2e2193ebf65b112bfe5c467034c084a26ac9dacae`，2,366 B。
- [artifact/process 撤销审计](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/ARTIFACT_PROCESS_AUDIT.json)：SHA-256 `1cd0628891cadc6301b44cc2cb6e42f9cfba75766a5b0ebe20517857ea70256a`，6,235 B；[contact intent 审计](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/CONTACT_INTENT_AUDIT.json)：`0faf84c9477670a9b4011f4174aaf9bcfa09d7c4b63b7b223034d39ffca036e8`，5,818 B。
- [CPU keyframes NPZ](../../../archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/KEYFRAMES_CPU.npz)：SHA-256 `d6f944d4af8c987591f2441dac87596f60b1510675304a185c33310b24c40554`，2,493 B；当前状态 `PASS_F345_F402_CPU_READY_FOR_WINDOW_EXPANSION`，不是 timeline/full460 PASS。
- [正确 human-intent timeline](../../../_run/robot_004_mano21_contact_timeline_cpu_20260902_v1/CONTACT_TIMELINE.json)：SHA-256 `258f32b5fd6e8a3cf804633bdbcba4ee3eb4d9d2776678cc72079b335a7c7ae2`，1,468,150 B；[区间图](../../../_run/robot_004_mano21_contact_timeline_cpu_20260902_v1/CORRECT_LABEL_TIMELINE.png)：`3b1fb1590cd3096b39df0858f526bc7c3a4f27c904b6330fbf2a9d7d58cc6a9f`，27,665 B；[10 帧原始证据图](../../../_run/robot_004_mano21_contact_timeline_cpu_20260902_v1/TEN_FRAME_CPU_CONTACT_SHEET.png)：`ae94fdb33b6b97113d821af1d7511793b7af788f5c687389126ba6d805eb6119`，2,245,691 B。251 个 active frame 只表示 human required-finger intent。
- 代码 pins：`pipeline/robot_wrist_kai_adapter.py` `14f8b189…3f97b0`；`pipeline/robot_contact_geometry.py` `a4b86485…81bcfb1`；两测试 `218337d1…dc85`、`5aff2494…cc29`；gate runner `3bd0bd56…1c513e`。完整 64 位 SHA/bytes 在机器 manifest；回归 30/30 PASS。

### Robot（当前用户优先）

- [004 窗口 overlay 视频](../../../NOW/robot/robot_window_pass_overlay.mp4)：SHA-256 `17dcd5e6d95ceb7883e61ed9ca860d8a529ba0e9a9112cc6b7554a7687c61cc5`，280,266 B；只保留 arm/posture 视觉追溯，不是当前手指接触 authority。
- [004 窗口 opaque 视频](../../../NOW/robot/robot_window_pass_opaque.mp4)：SHA-256 `5d6276491ef6f3f70e098d3987819a1212afd3c61d54fd3e75825c049e17e20e`，101,640 B。
- [004 六帧图](../../../NOW/robot/robot_window_pass_6frame.png)：SHA-256 `e8b89d51f3890d58a65469c3cfb641a08cc0f046d8f4947c430c9095c993ee4a`，2,139,794 B。
- [跨会话 27 帧图](../../../NOW/robot/cross_session_contact.png)：SHA-256 `9321938638ef48d410bff0380c12ea416517c1867085011b0905eb816fcaf7b7`，5,434,254 B。
- [跨会话终态报告](../../../NOW/robot/cross_session_contact_report.md)：以 corrected contact addendum 为准。
- [窗口 initializer 状态](../../../NOW/robot/CROSS_SESSION_WINDOW_INITIALIZER_STATUS.md) · [窗口 contact sheet](../../../_run/robot_cross_session_window_initializer_dp_cpu_20260902_v1/CROSS_SESSION_WINDOW_INIT_CONTACT_SHEET.png)：SHA-256 `c2ede6205d93546ab48cbe816c65489a457428bd9396676fe4e4b379c5c43561`，2,517,469 B；1440×1800 完整解码。
- [Event rescue contact sheet](../../../_run/robot_cross_session_event_rescue_cpu_20260902_v1/CROSS_SESSION_EVENT_RESCUE_CONTACT_SHEET.png)：SHA-256 `89b620b461c2f638328d46ac25a127357d5cfcda9805d165239ce7b30a202964`，2,558,590 B；1440×1800 完整解码。

### Clean L3

- [扑克/薯片联合图](../../../_run/newtask_baseline_clean_l3_20260902_v1/L3_POKER_CHIPS_CONTACT.png)：SHA-256 `87470a8271e6b8aae78ecd5aad56d31bf7f7d61700d855f670880eb13a03030e`，1,666,569 B。
- [chips 短视频](../../../_run/newtask_baseline_clean_l3_20260902_v1/chips/CHIPS_L3_DIAGNOSTIC.mp4)：SHA-256 `f1b9a29d7e45a077339854fe17eda74644440796109201dc376759dbbc8c09a0`，282,300 B。
- [poker 短视频](../../../_run/newtask_baseline_clean_l3_20260902_v1/poker/POKER_L3_DIAGNOSTIC.mp4)：SHA-256 `b4479235039ebcb66cf2d04e019f26dd6df53382bc37a5b99841df28ceb9222e`，268,950 B。
- [NOW 状态页](../../../NOW/newtask/BASELINE_CLEAN_L3_STATUS.md)：chips fill 77.45%；poker 0.98%；两条仍 HOLD。

### Clean chips PHONE_DEV frame0 terminal HOLD

- [一页结论](../../../NOW/clean/cross_camera_final_hold.md) · [Clean 短入口](../../../NOW/clean/README.md)。
- [最终 bounded whole-table A/B](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/FRAME0_BOUNDED_WHOLE_TABLE_A_B.png)：SHA-256 `9d8076cfa8010ebd68b0af8f6a0cad7b202276a31a49635ec1f8ee35ef4752dc`，375,275 B。
- [六个独立锚点](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/INDEPENDENT_ANCHORS.png)：SHA-256 `49cb7f65ca1f9abdba528ae62e20572c27b46720980712ce159af18718623d2f`，225,027 B。
- [RESULTS](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/RESULTS.json)：SHA-256 `81f4656978b911e3ae76029f0e07d5e013b046f93c84115d423c251f6af86f52`，8,424 B；[冻结 protocol](../../../_run/newtask_clean_chips_frame0_whole_table_bounded_final_20260902_v1/PROTOCOL.json)：SHA-256 `bd5652644f09246ad1cf51b1592fb9f761aded3911c8b7d8b830fb7b88060888`，2,529 B。
- runner `tools/build_clean_frame0_whole_table_bounded_final.py`：SHA-256 `4a4382cc2e9833755b46f579500d7e8db1294381d10bafebca12f8193fa9247e`，19,959 B。

失败演进全部是 frame0-only、non-formal 诊断：

| 阶段 | Review | SHA-256 |
|---|---|---|
| narrow donor / multiband / Poisson | [FRAME0_THREE_CANDIDATES](../../../_run/newtask_clean_chips_frame0_narrow_seam_candidates_20260902_v1/FRAME0_THREE_CANDIDATES.png) | `78ab4c6250e1e668beb999551641fc57e290ea020b17313ab484c0867ee32738` |
| pure donor 证明 Poisson boundary reinjection | [FRAME0_PURE_DONOR_A_B](../../../_run/newtask_clean_chips_frame0_pure_donor_ab_20260902_v2/FRAME0_PURE_DONOR_A_B.png) | `34b207a0b4d42fed50200c7fa62bf8a5b3150886036b041323359655b7abc104` |
| source-driven graph-cut | [FRAME0_GRAPHCUT_A_B](../../../_run/newtask_clean_chips_frame0_graphcut_seam_ab_20260902_v1/FRAME0_GRAPHCUT_A_B.png) | `3a87220b609460f313c3f4a9f577a99f37059153f6bded1ad5792cc1a4d14a56` |
| target-frame PatchMatch | [FRAME0_PATCHMATCH_A_B](../../../_run/newtask_clean_chips_frame0_patchmatch_graphcut_ab_20260902_v1/FRAME0_PATCHMATCH_A_B.png) | `3c85f6167f901cdbc61f5266fb7fbb45c67d03ebf1de07f56dd32ba866151900` |
| real-pixel quilting vs generated controls | [FRAME0_QUILTING_VS_INPAINT](../../../_run/newtask_clean_chips_frame0_quilting_inpaint_ab_20260902_v2/FRAME0_QUILTING_VS_INPAINT.png) | `5ed7964e0b7e498ed545825d4e837af138702edfafe62a8332a48b1047a2cc94` |
| unconstrained whole-table H/ECC | [FRAME0_WHOLE_TABLE_A_B](../../../_run/newtask_clean_chips_frame0_whole_table_plate_ab_20260902_v2/FRAME0_WHOLE_TABLE_A_B.png) | `682946eaf30a458c6db31078ca1d25103e6718367701375e31a2e4a4197f6dae` |

最小缺口：同机位/同姿态 2--3 秒无人手 clean plate；若做不到，至少四个分布于 table replacement polygon 外围、两图独立测量的同名点。当前没有 10-frame/full Clean 视频。

### Mask

- [v4 chips PASS / poker HOLD 统一图](../../../NOW/mask/bilateral_pass_vs_hold.png)
- [chips 10 帧总览](../../../NOW/mask/bilateral_chips_10frame.png) · [分 class 视频](../../../NOW/mask/bilateral_chips_classes.mp4) · [binary union 视频](../../../NOW/mask/bilateral_chips_union.mp4)
- [chips 最终 QA](../../../NOW/mask/bilateral_chips_qa.md) · [360 帧 union](../../../NOW/mask/bilateral_chips_union_masks/) · [poker HOLD 图](../../../NOW/mask/bilateral_poker_hold.png)
- [v4 通用性矩阵](../../../NOW/mask/bilateral_generalization.md) · [poker 最小标注计划](../../../NOW/mask/poker_tracker_label_plan.md)
- v4 checkpoint SHA-256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；runner SHA-256：`65cb00ffaa0d6c36fd8fea6a7c5448abcf3083f738fe5c1d5d3ca37bb969a774`；protocol SHA-256：`ab52ed204cb5221a72df62db8b0cbf5759f71399c6d9191efac9639a734d57bd`。

以下为被 v4 successor 保留、但未改写的历史 generic/neutral-prompt 诊断：

- [四动作流总览](../../../NOW/mask/baseline_action_overview.png)
- [chips object 视频](../../../NOW/mask/baseline_chips_object.mp4) · [chips human HOLD 视频](../../../NOW/mask/baseline_chips_human_hold.mp4)
- [poker object 视频](../../../NOW/mask/baseline_poker_object.mp4) · [poker human HOLD 视频](../../../NOW/mask/baseline_poker_human_hold.mp4)
- [Wearable semantic rescue 状态](../../../NOW/mask/WEARABLE_SEMANTIC_RESCUE_STATUS.md) · [chips sleeve 0/12 图](../../../_run/gpt_mask_wearable_semantic_rescue_20260902_v3/baseline/chips/sleeve_cuff/CHIPS_SLEEVE_CUFF_BEST_FAILED_CANARY12.png) · [poker tracker class-only 12/12 图](../../../_run/gpt_mask_wearable_semantic_rescue_20260902_v3/baseline/poker/wrist_wearable/POKER_WRIST_WEARABLE_CANARY12.png)

### 新任务几何

- [Poker synthetic geometry/depth/protect](../../../_run/newtask_geometry_prep_v2/SYNTHETIC_POKER_GEOMETRY_DEPTH_PROTECT.png)
- [Chips synthetic geometry/depth/protect](../../../_run/newtask_geometry_prep_v2/SYNTHETIC_CHIPS_GEOMETRY_DEPTH_PROTECT.png)
- [手机 2D APPROX 布局图](../../../_run/newtask_geometry_prep_v2/BASELINE_2D_APPROX_CONTACT_SHEET.jpg)

## 权重、资产与许可证

- 唯一实际加载的学习权重是本地 pinned `sam3.1_multiplex.pt`：3,502,755,717 B，SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。Clean、Robot、合同和几何的 `learned_weights` 均为空。
- SAM code/weights 的本地许可证文件 SHA-256 为 `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be`；分发须遵守并携带该许可。
- Tianji/KaiHand/法兰/URDF/CAD/mesh 再分发权未确认；手机和 PICO/egodata 也没有在本证据集中冻结对外分发许可。
- 因此本索引只支持工作区内复现，不构成外发授权。

## 完整性规则

`ALGORITHM_AND_WEIGHT_MANIFEST.json` 中 29 个算法文件、50 个 authority 文件和 1 个权重均已按实际文件复核 path/bytes/SHA。Clean L3 自带 108 项 `SHA256SUMS` 且 `sha256sum -c` 全通过。新增 Clean frame0 与 Robot MANO21/KaiHand runner/protocol/result/order/timeline 也逐项复核；Robot adapter/geometry 回归 30/30 PASS。生成文档的最终 bytes/SHA 在 `ARTIFACT_INDEX.json` 中记录；机器索引自身为避免循环哈希而 `self_excluded_from_hash_list=true`。
