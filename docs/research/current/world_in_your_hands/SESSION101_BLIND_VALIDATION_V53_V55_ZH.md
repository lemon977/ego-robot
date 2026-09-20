# WIYH Session 101 真盲测：V53–V55

终态：`FAIL_CROSS_SESSION_BLIND_MANUS_ORIGIN_ALIGNMENT`。这是质量拒绝，不是运行失败。

## 输入与独立性

- 原始采集：`/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916/cards_130_0916/101`
- 原始 HDF5：上述目录的 `dataset.hdf5`
- 清洗后只读会话：`/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned/playing_cards/play_cards_0916_101`
- 帧数：122
- 输入 Merkle：`b19b0daed25fdd33dbd5157da41f0678005f0e4dd5c15b777a89a51660df1fa8`
- V45 几何参数在 101 RGB 被查看前已经冻结；101 没有参与参数、阈值或候选选择。
- 运行前后 Merkle 相同，`/mnt/data/egodata` 未改写。

## 实际执行链

1. V53 用冻结的多锚点 SAM3.1 方案生成独立 RGB-only hand mask。
2. V54 不查看 RGB，依次应用冻结的尺度自适应面积门和 PICO 腕点画内门。
3. V55 只把 V54 合格 mask 框送入冻结 V29 HaWoR；随后用冻结 V45 的每侧常量 MANUS origin/basis 做一次性盲测。
4. PICO Controller→wrist 动态锚点逐帧保持不变；没有在 101 上拟合任何参数。

## 可见性与 HaWoR

| 项目 | 左手 | 右手 |
|---|---:|---:|
| V53 HOLDOUT 原始可观测 mask | 114 | 110 |
| 尺度门合格 | 114 | 77 |
| PICO 腕点画内后合格 | 114 | 77 |
| HaWoR 实际观测 | 114 | 77 |

HaWoR 推理本身正常完成，双手同时观测 72 帧，总 side-frame 191；GPU 工作段约 27.84 秒。

## 冻结几何验收

左手：

- 独立 mask 重合门通过；joint-in-mask P50 `0.8333`，bone-in-mask P50 `0.8733`。
- 但 HaWoR 独立 landmark 误差 P50 `150.38 px`，略高于冻结 `150 px` 门。
- joint centroid→mask centroid P95 `181.49 px`、extent log error P95 `1.6173`，均失败。

右手：

- 腕点距离 mask P95 仅 `2.98 px`，说明动态 PICO wrist 本身贴近手区。
- 但 joint-in-mask P50 只有 `0.4167`，joint→mask P95 `107.19 px`，独立 mask 门失败。
- HaWoR 独立 landmark 误差 P50/P95 为 `117.57/232.99 px`，但不能抵消区域重合失败。

因此，失败集中在“冻结 MANUS 手指局部几何跨会话不能稳定贴合真实手”，不是 SAM 可见性不足、GPU 崩溃或 PICO 腕点消失。

## 可视化

- [122 帧完整视频](../../../current/visuals/WIYH_SESSION101_BLIND_V55/PICO_WRIST_MANUS_SESSION101_BLIND_FULL_REVIEW.mp4)
- [6 帧接触表](../../../current/visuals/WIYH_SESSION101_BLIND_V55/CONTACT_SHEET.jpg)
- [逐侧冻结门审计](../../../current/visuals/WIYH_SESSION101_BLIND_V55/CROSS_SESSION_BLIND_ALIGNMENT_AUDIT.json)

左栏是 `PICO wrist + native MANUS`，右栏是同一 PICO wrist 加冻结 V45 origin/basis。洋红叉是 PICO Controller，圆圈是 PICO-derived wrist；mask 仅作独立 RGB 验收。视频完整解码 122/122 帧，SHA-256：

`e25965c7a546331b3e61644cb3e649ce6bfcfa125a20d21556676903d34c112c`

## 是否可用于 ML

当前结论是 **不能作为监督训练标签、接触真值或 Robot 控制真值**：

- `training_eligible=false`
- `authority_promoted=false`
- `depth_unlocked=false`

原始 PICO、MANUS、触觉和 RGB 仍可作为保留的传感器模态，适合后续研究“会话/安装实例条件化的传感器融合”或自监督学习；但当前 V45 固定几何不能作为跨会话标签生成器。下一轮不得在 101 上继续调参并把同一 101 重命名为盲测，应把 101 固定为失败 holdout，另选未见会话验证新候选。
