# 支线3：HaWoR → Kai22 → 完整Robot

007 378帧、031 149帧全部执行、解码并独立验证；冻结R0的q、时间和有效性原样保留。与HuRo使用[同一冻结输入](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/SHARED_AI2_AI4_INPUT_FREEZE.json)。

- [完整结果](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai2/full_robot_v2_run1/RESULT.json)
- [独立数值验证与有限碰撞归因](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/exact78/AI2_INDEPENDENT_VERIFICATION.md)
- [九片真实dt、幅度与跟随评价](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/exact78/temporal_audit_v2/README_ZH.md)
- [按能力采用状态](BASELINE_RELEASE.json)

采用工程修复：`T_tool_hand = inverse(T_flange_tool) @ T_flange_hand`；旧消费者混淆法兰与tool，包含约145mm固定位移。新实现保存目标、实际FK、失败帧、原始q和独立掩码，固定虚拟安装和尺度。

不采用为算法质量：007只有一侧378帧有效，腕臂门377帧通过；031两物理侧94/4帧有效，仅1/1帧通过，最大腕残差约4.78m。上游缺手和米级腕运动未被裁剪、复制或删除。历史R0已经postclip，限位内不能回推原始求解合法。neutral资产已有碰撞，不能声称全自碰撞通过。

复现：`chaoyang run run_full_robot_review_v2 --spec <本lane/ROBOT_INPUT_SPEC.json> --output <新登记目录>`。下一任务先定位RAW/侧别/腕坐标首次异常，新输入须与HuRo同步重新冻结，不在末端平滑掩盖。
