# Human→Robot R2 当前可视化索引

本目录只做导航，不复制第二份视频真值。所有条目均为开发诊断；`training_eligible=false`、
`control_ground_truth=false`、`physical_deployable=false`。

## Sensor 共同后端完整回放

- [0916_097，165 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_097_wave0/SENSOR_REVIEW.mp4)
- [0916_098，179 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_098_wave1/SENSOR_REVIEW.mp4)
- [0916_101，122 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/run_101_wave1/SENSOR_REVIEW.mp4)

三条视频均从保存的 HandMotion 与 Kai22 后端数组生成；它们证明结构消费和回放完整，不证明外部
腕部真值或真机可执行性。

## Scene 与 renderer 诊断

- [031 实际 ProPainter 候选，149 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_031_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)
- [007 实际 ProPainter 候选，378 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_007_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)
- [0902_103 同配方候选，284 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_103_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)
- [0902_042 同配方候选，171 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_042_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)
- [031 当前帧物体保护第二轮，149 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_031_wave5/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)
- [真实机器人资产固定输入隔离图](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane4_compare/renderer_four_way_wave3/RENDERER_FOUR_WAY_DIAGNOSTIC.png)
- [真实 STEP 解码转接环 scene-object 受控图](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/assembly_scene_canary_wave11/ADAPTER_SCENE_OBJECT_REVIEW.png)

Wave4 中红色为 `M_write`、黄色为 `UNKNOWN`；Wave5 另以绿色显示当前帧物体保护。全部视频均是
真实模型候选，但四条质量均拒绝、正式 Clean 仍为0。Wave5 保护域内改写像素降为0，但没有消除
手—物体交界模糊或设备遗漏，不能据此采用。

## 按会话 Robot 候选

- [031 单侧有效 R0 候选，149 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/product_candidate_031_wave6/robot_candidate.mp4)
- [007 双侧 R0 候选，378 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/product_candidate_007_wave6/robot_candidate.mp4)
- [0902_103 同配方回归候选，284 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/product_candidate_103_wave6/robot_candidate.mp4)
- [0902_042 同配方回归候选，171 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/product_candidate_042_wave6/robot_candidate.mp4)

031 对无观测侧采用 `NOT_RENDERED_NO_FILL_NO_HOLD`，没有默认姿态、镜像或永久保持。四条均完成
同字节 resume，但消费的是质量拒绝的 Scene 候选，且没有深度遮挡，因此只证明逐会话结构闭环。

## Local R0／HuRo 同一候选背景诊断

- [031，149 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane4_compare/same_rejected_background_031_wave10/LOCAL_R0_VS_HURO_SAME_REJECTED_BACKGROUND.mp4)
- [007，378 帧](../../../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane4_compare/same_rejected_background_007_wave10/LOCAL_R0_VS_HURO_SAME_REJECTED_BACKGROUND.mp4)

两边使用同一冻结 HandMotion 版本、相机域、Robot 资产、帧分母和同一条 Scene 候选。031 共同有效侧
为右侧102帧；007为两侧378/378帧。背景Clean已被质量拒绝且遮挡UNKNOWN，因此这些视频用于确认
两种求解结果确实被分别消费，不是最终视觉质量对照，也不产生赢家。

## 当前未交付

- 正式采用的 Clean：0/4；实际模型候选：4/4。
- 007／031 Robot 候选：2/2；质量通过/正式采用：0/2。
- Robot R1：0 个执行／采用窗口。
- Local／HuRo 同候选背景诊断：2/2；正式采用 Clean 的最终对照仍未完成。
