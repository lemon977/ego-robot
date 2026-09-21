# 支线1：Exact78

当前有效范围：chips0902_103 284帧、cards0902_042 171帧，全部执行及完整解码。工程保护修复可采用，Clean语义与完整Robot质量没有通过。

- [Clean保护、来源和像素验证](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/exact78/HANDOFF_ZH.md)
- [完整Robot及独立复核](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/exact78/HANDOFF_FULL_ROBOT_ZH.md)
- [共同基线与有效元数据更正](BASELINE_RELEASE.json)

保守guard的M_write外和受保护像素保持不变；333帧Raw回退会带回手/Tracker，历史donor未重新取得身份授权，不宣称完整去手成功。旧撤权Depth/Object6D未被消费。

完整Robot固定底座、安装、尺度和视角；实际目标与FK都导出。物理侧0腕423次失败，碰撞检查也有实测检出，不能仅凭全片视频升级为质量通过。历史hand-only保留原用途，不改称“从未有效”。

复现入口：`chaoyang run run_clean_object_guard_v2` 与 `chaoyang run run_full_robot_review_v2`，精确spec/解释器/代码版本见交接及发布清单。下一步见[有限后续任务](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/NEXT_TASKS_ZH.md)；不做Attachment身份自证、不借其他会话几何、不删除困难帧。
