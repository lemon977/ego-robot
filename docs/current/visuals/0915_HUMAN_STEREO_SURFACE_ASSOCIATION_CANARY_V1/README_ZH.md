# 0915 Human/Stereo 表面关联诊断

固定样本为 `play_cards_0915_001` 的 150 帧。HaWoR、SAM3.1、FoundationStereo 和 Object6D v2 均未重跑；MANO 只从冻结参数在固定 CPU 环境中重建表面。

- 旧 joint-centre ↔ visible-surface hold-out P90：`36.91 mm`。
- MANO-surface ↔ Stereo-surface hold-out P90：`19.33 mm`，状态 `PASS_DEVELOPMENT_ALIGNMENT`。
- 中心连通指尖表面资格：`436/1500`；旧方法为 `406`。
- 不变 `5 mm`、有限可见牌面门下最近距离：`6.60 mm`；五帧窗口 `0`。
- R1：`BLOCKED_LOCAL_EVIDENCE`；首阻塞 `NO_FIXED_PAIR_FIVE_FRAME_CONTACT_WINDOW_AT_UNCHANGED_5MM_GATE`。本任务没有修改 q22 或 wrist。

该结果只说明开发级几何关联是否自洽，不是接触真值、外部毫米精度、控制真值或部署授权。没有使用统一 48 mm 偏置，没有使用物体/Contact 数据拟合对齐，也没有读取 Removal/Clean。
