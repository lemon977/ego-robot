# 薯片任务

唯一配置入口：[task.manifest.json](task.manifest.json)。新运行写
`tasks/chips/runs/<system>/<run_id>`；不要再写 `_run/newtask_shupian_*`，后者只保留历史证据。

## 当前状态

- HaWoR：`get_potato_chips_0901_001` 已用仓内 HaWoR 完成 799 帧；左手 799 帧直接观测，
  右手 797 帧直接观测，f238/f746 显式 infiller，输出是正确 MANO21（wrist=0）。
- Mask：新数据 799/799 帧 assisted bilateral mask 已授权；包含双手、完整前臂、双 tracker
  与腕带上下文，物体保护重叠为 0。两任务共享 SAM3.1 基座权重，任务配置独立。
- Clean：已消费上传中的 `base_001` 内 CRC 完整 clean-plate 视频与相机参数，生成 799 帧
  Clean v3；手区覆盖 100%，保护物体和桌外区域改动均为 0。它是视觉 Clean，不是 metric GT。
- Robot：已产 799 帧 Clean 背景完整视频。全任务 81 个双臂实际 FK 锚点全部通过 10 mm/5°；
  平滑全片通过关节速度 0.12、加速度 0.06 门。尚缺 metric Object6D，不能声称真实接触/碰撞。
- 几何：三片 `CHIP_SADDLE`、一个 `BOWL_REVOLVE` 和 `mat_plane` 已有近似 dry-run；
  真实尺寸和 PICO 位姿仍未提供。

完整结果与边界：[reports/CHIPS001_PIPELINE_STATUS_20260902.md](reports/CHIPS001_PIPELINE_STATUS_20260902.md)。

本任务与扑克共享系统算法/权重，但配置和 runs 完全隔离。不能复制扑克的 tracker
标注计划、牌/木架几何或 Clean 保留对象。
