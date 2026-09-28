# Human→Robot Baseline v1：十小时交付视频导航

权威逐项帧数、完整解码、SHA、来源和状态见[15 槽位收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/delivery/RESULT.json)。这里的“新”仅指本轮审阅排版，**不是新求解或质量通过**。全部为离线开发视频，用户视觉验收待定；没有声称已复制到用户设备。四片产品结构 4/4，质量 0/4，采用 0/4。

## 先看主会话与唯一新实验

| 内容 | 视频 | 范围与结论 |
|---|---|---|
| 007 Raw／Motion／正式 Robot 全链 | [378 帧审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_007/get_potato_chips_0915_007_RAW_MOTION_FORMAL_PRODUCT_REVIEW.mp4) | 新排版、旧 q；产品质量拒绝 |
| 031 Raw／Motion／正式 Robot 全链 | [149 帧审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_031/play_cards_0915_031_RAW_MOTION_FORMAL_PRODUCT_REVIEW.mp4) | 新排版、旧 q；原右手 102、左手 0，质量拒绝 |
| 007 扩展上下文唯一候选 | [181–196 旧新 Clean 同帧对照](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/scene/context_007_v1/007_CONTEXT_OLD_NEW_CLEAN_16FRAME_REVIEW.mp4) | 57 帧输入；仍有残臂、线缆、碗边残影；不扩全片；[独立抽样复核](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/scene/context_007_v1/AI_VISUAL_REVIEW.json) |
| 007／031 同目标 Local-HuRo | [007：378 帧](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/get_potato_chips_0915_007_SAME_TARGET_FK_REVIEW.mp4)；[031：149 帧](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/play_cards_0915_031_SAME_TARGET_FK_REVIEW.mp4) | 新独立 FK 评价及同步排版、旧求解；HuRo 部分帧硬限位失败，不宣布胜者；[数值](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json) |

## 15 个固定全片槽位

下面每项均由本轮收据重载、全解码并核对 SHA。`REUSED`与质量拒绝可以同时成立；`NEW_REVIEW`不表示算法改善。

| 槽位 | 视频 | 帧数 | 来源／质量 |
|---|---|---:|---|
| Scene 007 | [Raw／支持／Clean](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_007_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4) | 378 | REUSED／REJECTED |
| Scene 031 | [Raw／支持／Clean](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_031_wave5/SCENE_CLEAN_CANDIDATE_REVIEW.mp4) | 149 | REUSED／REJECTED |
| Scene 0902_103 | [Raw／支持／Clean](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_103_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4) | 284 | REUSED／REJECTED |
| Scene 0902_042 | [Raw／支持／Clean](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_042_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4) | 171 | REUSED／REJECTED |
| Sensor 097 | [米制回放](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/lanes/sensor/review_metric_v2/play_cards_0916_097/SENSOR_METRIC_REVIEW_V2.mp4) | 165 | REUSED／显示通过；贴合未定 |
| Sensor 098 | [米制回放](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/lanes/sensor/review_metric_v2/play_cards_0916_098/SENSOR_METRIC_REVIEW_V2.mp4) | 179 | REUSED／显示通过；贴合未定 |
| Sensor 101 | [米制回放](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/lanes/sensor/review_metric_v2/play_cards_0916_101/SENSOR_METRIC_REVIEW_V2.mp4) | 122 | REUSED／显示通过；贴合未定 |
| Motion 007 | [全链](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_007/get_potato_chips_0915_007_RAW_MOTION_FORMAL_PRODUCT_REVIEW.mp4) | 378 | NEW_REVIEW／REJECTED |
| Motion 031 | [全链](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_031/play_cards_0915_031_RAW_MOTION_FORMAL_PRODUCT_REVIEW.mp4) | 149 | NEW_REVIEW／REJECTED |
| Compare 007 | [同目标](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/get_potato_chips_0915_007_SAME_TARGET_FK_REVIEW.mp4) | 378 | NEW_REVIEW／限位范围不通过 |
| Compare 031 | [同目标](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/play_cards_0915_031_SAME_TARGET_FK_REVIEW.mp4) | 149 | NEW_REVIEW／限位范围不通过 |
| Product 007 | [正式 Robot 候选](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/motion_product/formal_007_current/attempt_0001/robot.mp4) | 378 | REUSED／REJECTED |
| Product 031 | [正式 Robot 候选](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_031/attempt_0004/robot.mp4) | 149 | REUSED／REJECTED |
| Product 0902_103 | [正式 Robot 候选](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/robot.mp4) | 284 | REUSED／REJECTED |
| Product 0902_042 | [正式 Robot 候选](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/robot.mp4) | 171 | REUSED／REJECTED |

## 不占全片槽位的限定证据

Sensor 本轮的[独立固定／异常样本复核](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/sensor/independent_review_v1/RESULT.json)仅在声明范围内有效。031 的[同物体 patch](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/scene/object_patch_031/window_v1/RESULT.json)、[Robot 刚性对应](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/motion_product/robot_correspondence_031/window_v1/RESULT.json)及[连接件碰撞查询](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/motion_product/adapter_collision/window_v1/RESULT.json)为旧固定窗真实调用。本轮新增[66–81 帧连续并排审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/geometry_reuse_review_v1_runtime_fix1/031_PATCH_AND_ROBOT_CORRESPONDENCE_16FRAME_REVIEW.mp4)及[逐帧输入 SHA](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/geometry_reuse_review_v1_runtime_fix1/RESULT.json)：81 帧没有 t+1，Robot 对应明确标为不适用，不伪造画面。它们不证明全片几何真值或完整碰撞通过；Contact 严格准入与 Robot R1 仍为 0。
