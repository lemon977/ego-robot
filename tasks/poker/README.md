# 扑克牌任务

唯一配置入口：[task.manifest.json](task.manifest.json)。新运行写
`tasks/poker/runs/<system>/<run_id>`；不要再写 `_run/newtask_puke_*`，后者只保留历史证据。

## 当前状态

- Mask：同一 SAM3.1 权重下，双 human 和 lower cuff 通过；upper/lower tracker 时序门
  未闭合，`consumption_authorized=false`，因此不能交给 Clean。
- 下一步标注：最低 8 帧、双 tracker 共 16 个实例；稳妥版 11 帧、22 个实例。只补
  tracker，当前不要求重标已经通过的人体和 lower cuff。
- Clean：算法无专用权重；需要先获得可消费 Mask，并补同相机/同姿态/同曝光 clean
  plate。当前手机空桌不同机位，只能诊断。
- 几何：三张 `CARD_THIN_BOX`、一个 `RACK_BOX`、`mat_plane` 与
  `rack_top_plane` 已有近似 dry-run；真实尺寸和 PICO 位姿仍未提供。

本任务与薯片共享系统算法/权重，但配置和 runs 完全隔离。不能复制薯片的 frame-0
anchor、tracker 结论、碗/薯片几何或 Clean 保留对象。
