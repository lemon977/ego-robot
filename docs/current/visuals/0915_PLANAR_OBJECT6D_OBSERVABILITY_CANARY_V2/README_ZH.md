# 0915 三牌 Planar Object6D encoded-domain canary

状态：`PASSED`（仅开发级可见表面可观测性）

- 会话：`play_cards_0915_001`
- 时间轴：[`0915_PLANAR_OBJECT6D_OBSERVABILITY_TIMELINE_V2.png`](0915_PLANAR_OBJECT6D_OBSERVABILITY_TIMELINE_V2.png)
- Depth 输入：通过质量门的 encoded-domain FoundationStereo，原物理左目 640×480
- Mask 输入：同一物理左目 resize-only SAM3.1，1280×960
- Join：解析像素中心映射 `x_sam=2*x_depth+0.5`、`y_sam=2*y_depth+0.5`
- GPU：未使用

| 物理实例 | center_xyz | plane_normal | inplane_rotation | full_extent |
|---|---:|---:|---:|---:|
| `playing_card_00` | 143/150 | 139/150 | 139/150 | 0/150 |
| `playing_card_01` | 146/150 | 113/150 | 113/150 | 0/150 |
| `playing_card_02` | 95/150 | 48/150 | 48/150 | 0/150 |

绿色只表示该帧该字段满足内部“直接可见”可观测门；灰色表示 unknown、遮挡、Mask
未知或可见点不够平面。`center_xyz` 是直接可见表面点的中位中心，不是被遮挡后的完整物体
中心。牌尺寸尚未实测，所有 `full_extent` 都保持 `UNOBSERVABLE`。

`black_card_tray` 是独立 support entity，但当前没有独立 mask，因此保持 `UNKNOWN` 且
无 geometry authority。`card_set` 仅表达三张牌的语义成员关系，不拥有共同 mask、pose
或刚体几何。

本任务不补隐藏形状、不输出统一 `valid/confidence`，也不授予外部毫米精度、Contact、
Robot 或部署 authority。下一步若要进入 Interaction/Contact，必须先审阅逐帧几何叠加，
并实测牌宽/高；不能仅凭本时间轴自动放行。
