# 历史瓶罐抓取 Robot 004

该任务保存 `grap_a_cap_004` 的 Robot 纠错、物理门和视频验证。它与扑克、薯片完全隔离；
历史 `_run/robot_004_*` 仅作为只读 authority，新运行写入本目录的 `runs/robot/`。

当前目标：在正确 MANO21、有限 Y 轴物体、KaiHand 全网格和 exact distinct-finger SAT
约束下修复 f367，先通过冻结十帧门，再生成明确标注的验证视频。
