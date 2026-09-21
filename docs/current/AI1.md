# 支线2：PICO Controller＋MANUS

097/098/101共466帧全部执行；3条完整Robot和3条额外腕手诊断均完整解码。102/103未读，101不是盲测。

- [唯一有效输入：来源更正链](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai1/provenance_correction_v2_run1/RESULT.json)
- [方法、复现与结论](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai1/HANDOFF_ZH.md)
- [编码图像域诊断](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai1/CAMERA_DOMAIN_AUDIT_ZH.md)
- [publisher实际集成记录](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/PRODUCER_PATCH_INTEGRATION.json)

变换链内部闭合；手柄到解剖腕仍是未实测安装prior。MANUS及Controller在采集导出端已重采样，不能把可计算值或最近源时刻称直接观测。追加更正只降低observed权限，所有几何/时间/求解输入逐字节不变。新producer也已回归并集成，旧RESULT不覆写；历史交接中“补丁未集成”是其封存时状态，以publisher集成收据为准。

虚拟腕臂门857/932个有效侧帧通过、75失败未删。真实图像贴合未达成；保存外参近竖直双目基线与编码画面主要水平视差不一致。下一步核实设备到编码域的明确映射，不作无约束外参/时移搜索，不用HaWoR拟合腕或补手。

复现入口：`chaoyang run run_pico_manus_motion_v2`、`run_pico_manus_provenance_correction_v2`、`run_full_robot_review_v2`。新Robot消费必须使用更正后motion。训练/控制/真实标定授权均为false；人工审阅待完成。
