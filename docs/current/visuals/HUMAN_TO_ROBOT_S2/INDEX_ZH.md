# Human→Robot S2 当前视频索引

> 实体只保存在对应 attempt；这里仅导航。当前 4/4 是结构完整候选，质量通过仍为 0/4。

给外部 AI 的当前结论与问题：[H3_EXTERNAL_AI_HANDOFF_ZH](H3_EXTERNAL_AI_HANDOFF_ZH.md)

## S2 正式产品候选

- `get_potato_chips_0915_007`：[378帧 current-signature robot.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_007/attempt_0002/robot.mp4)；quality=`REJECTED_QUALITY`，geometry=`UNKNOWN_NO_QUALIFIED_DEPTH`。
- `play_cards_0915_031`：[149帧 current-signature robot.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_031/attempt_0004/robot.mp4)；quality=`REJECTED_QUALITY`，geometry=`DIRECT_VISIBLE_OBJECT_SURFACE_ONLY`。
- `get_potato_chips_0902_103`：[284帧 current-signature robot.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/robot.mp4)；quality=`REJECTED_QUALITY`，geometry=`UNKNOWN_NO_QUALIFIED_DEPTH`。
- `play_cards_0902_042`：[171帧 current-signature robot.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/robot.mp4)；quality=`REJECTED_QUALITY`，geometry=`UNKNOWN_NO_QUALIFIED_DEPTH`。

以上四个 current-signature 视频与各自 H3 前序 attempt 字节一致，只闭合当前正式入口签名；不表示出现了视觉或算法改善。历史 attempt 保持不可变，不再作为当前导航入口。

## 固定证据视频

- 031 H3 连接件/遮挡 A-B-C：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003/H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003/H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4)
- 007 局部附件 Clean canary：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/ATTACHMENT_CLEAN_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/ATTACHMENT_CLEAN_REVIEW.mp4)

## Sensor 共同后端回放（冻结复用）

- `play_cards_0916_097`：[165帧 SENSOR_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_097_wave0/SENSOR_REVIEW.mp4)
- `play_cards_0916_098`：[179帧 SENSOR_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_098_wave1/SENSOR_REVIEW.mp4)
- `play_cards_0916_101`：[122帧 SENSOR_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_101_wave1/SENSOR_REVIEW.mp4)

以上三条由 S2 对原实体、NPZ 和 SHA 重新验证后复用；不是 S2 新拟合，也没有复制第二份视频。

## Local R0 / HuRo 同目标诊断

- `get_potato_chips_0915_007`：[378帧 LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_007/attempt_0001/LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4)
- `play_cards_0915_031`：[149帧 LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_031/attempt_0001/LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4)

两条均为 `INCONCLUSIVE_REJECTED_BACKGROUND`，只支持同目标数字诊断；无独立真值、无总体赢家。

## S2 H3 检查点

- 证据快照：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/checkpoints/H3_RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/checkpoints/H3_RESULT.json)
- 外部复核后的执行决定：[H3_REVIEW_DECISION](../../HUMAN_TO_ROBOT_S2_H3_REVIEW_DECISION_ZH.md)
- 031 来源旁车：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/source_provenance_031/attempt_0001/RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/source_provenance_031/attempt_0001/RESULT.json)
- 031 两阶段 IK 诊断：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/two_stage_031/attempt_0001/RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/two_stage_031/attempt_0001/RESULT.json)
- 031 腕目标权限：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/target_authority_031/attempt_0001/RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/target_authority_031/attempt_0001/RESULT.json)
- 007 Local/HuRo 真实连接件刷新：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_007/attempt_0001/RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_007/attempt_0001/RESULT.json)
- 031 Local/HuRo 真实连接件刷新：[/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_031/attempt_0001/RESULT.json](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/adapter_refresh_031/attempt_0001/RESULT.json)
