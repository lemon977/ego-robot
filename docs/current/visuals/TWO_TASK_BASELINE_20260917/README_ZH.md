# 旧 exact78 批量基线：两任务视觉对照

状态：`PASS_FROZEN_ARTIFACT_REPRODUCTION_CLOSURE`

这是用户所指“之前确认并完成批量处理的旧数据基线”的浅层入口。它来自 frozen exact78：
78 个 Chips + 78 个 Poker，共 156 会话、56,663 帧。此处固定两个代表会话：

- Poker：`play_cards_0902_042`，171 帧；
- Chips：`get_potato_chips_0902_103`，284 帧。

整链视频：

- [`play_cards_0902_042_RAW_TO_ROBOT_FULLSESSION.mp4`](../../../../archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/visuals/attempt_0002/play_cards_0902_042_RAW_TO_ROBOT_FULLSESSION.mp4)
- [`get_potato_chips_0902_103_RAW_TO_ROBOT_FULLSESSION.mp4`](../../../../archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/visuals/attempt_0002/get_potato_chips_0902_103_RAW_TO_ROBOT_FULLSESSION.mp4)

关键帧：

- Poker：[0](play_cards_0902_042_KEYFRAME_000000.png)、[85](play_cards_0902_042_KEYFRAME_000085.png)、[170](play_cards_0902_042_KEYFRAME_000170.png)
- Chips：[0](get_potato_chips_0902_103_KEYFRAME_000000.png)、[142](get_potato_chips_0902_103_KEYFRAME_000142.png)、[283](get_potato_chips_0902_103_KEYFRAME_000283.png)

证据闭包：[`VIDEO_RECEIPT.json`](VIDEO_RECEIPT.json)、
[`FRAME_MAP_RECEIPT.json`](FRAME_MAP_RECEIPT.json)。两条视频均为 30 FPS、完整解码，逐帧
消费，无补帧、插值、裁剪或循环。`fresh_model_inference_old_baseline=false`：这是冻结
产物的 SHA/解码复核，不是当前代码重新推理。

比较时允许看 Raw 画面域、HaWoR 连贯性、共同 Mask 轮廓、Clean 背景损伤和 Robot 离线
动作；不允许继承旧 Depth/Object6D 数值，因为那 58 条结果已因重复 lens undistortion
撤权。Robot 仍为离线可视化，`control_ground_truth=false`、
`physical_deployment_authorized=false`、`training_eligible=false`。
