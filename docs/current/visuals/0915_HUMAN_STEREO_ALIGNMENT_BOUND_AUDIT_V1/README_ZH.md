# 0915 Human/Stereo 尺度边界审计 V1

本页只审计已封存的 MANO-surface ↔ Stereo-surface 对齐，不重跑模型，不消费 Removal/Clean，
也不改变 Contact 的 `5 mm` 有限可见面门。

- 有界拟合尺度：`0.800000`（冻结下界 `0.800`）
- 无约束拟合尺度：`0.783730`
- 有界 hold-out P90：`19.33 mm`
- 审计结论：`REJECTED_BOUNDED_FIT_SATURATION`
- 公制 wrist-object 平移授权：`False`
- 最近有限可见牌面距离：`6.60 mm`
- 固定配对连续 Contact 窗口：`0`
- Robot：R0 保留；R1 为 `BLOCKED_LOCAL_EVIDENCE`；`q22/wrist` 未修改。

数值 hold-out 改善仍是有用诊断，但最优尺度落在冻结区间外，裁剪到边界后的“通过”不能升级为
公制对齐权威。Contact 结果原样继承，没有把 6.60 mm 放宽成接触，也没有使用 48 mm 统一偏置。
