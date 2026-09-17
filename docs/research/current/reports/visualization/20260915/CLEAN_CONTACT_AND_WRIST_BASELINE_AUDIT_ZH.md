# Clean 接触边界与三路手腕基线复核

更新时间：2026-09-15（Asia/Shanghai）

## 结论

本次用户全片复核确认了两个独立问题：当前 Clean 的结构性 Grade B 不能代表接触区视觉正确；三路手腕视频中的二维重合不能代表三维或深度相等。

## Clean 当前到底做了什么

当前 exact78 Clean 先对 Role Mask 做固定膨胀：Chips 左/右手为 24/20 px，Poker 左/右手为 24/18 px，左右 Tracker 都为 60 px。删除域随后减去当前帧可见 task-object mask。

`get_potato_chips_0902_039` 全片原始 role union 为 38,122,344 pixels，膨胀后为 57,078,813，增加 18,956,469（相对原始约 49.7%）。`play_cards_0903_245` 从 26,550,491 增至 39,914,809，增加 13,364,318（约 50.3%）。这与“消除边缘过多”的视觉判断一致。

所谓 `protected_object_byte_exact` 只检查当前帧可见物体 mask 内像素没有被改。被指尖挡住的牌面/薯片不在可见 mask 内，因此不受保护，也没有被该门恢复。

当前 real donor 实际算法不是 Stereo donor：它只在同会话的 ±1/2/4/8/16/32/64 帧中，搜索相同整数像素坐标；两个 donor RGB 在 L∞≤12 时复制第一帧像素。它不使用 Stereo 重投影、光流或背景平面几何。于是当盘子在别帧占据同一坐标时，盘子像素可以合法通过现有来源门，但语义仍然错误。

因此现有 Clean Grade B 只支持：帧数、修改域、当前可见物体 byte-exact、逐像素来源和视频解码闭环。它不支持：接触边界正确、隐含物体恢复正确、donor 表面语义正确或可直接进入 Robotized 训练。

## 两个已确认失败样例

- `archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/play_cards_0903_245_Clean底图_Robot相机视角_全片.mp4`：牌在指尖接触区被 Clean 影响，物体/Robot ownership 也未正确闭合。
- `archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/get_potato_chips_0902_039_Clean底图_Robot相机视角_全片.mp4`：Clean donor 带入盘子像素。

这两条视频是失败复核证据，不是 current Robot/Occlusion authority。

## 下一版 Clean 必须增加的门

1. 用按类别/局部尺度自适应的 removal support 代替 18–60 px 一刀切膨胀。
2. 把 visible object、contact band、amodal/atlas support 和未知区分开。
3. 扑克牌隐藏外观只允许来自因果、pose 验证的 object atlas 或可信 renderer；没有来源时输出 UNKNOWN。
4. temporal donor 必须有光流/单应/深度平面之一的几何对应，并检查来源表面语义；只验证 byte-exact 不够。
5. 增加 contact-boundary retention、wrong-surface donor、object silhouette continuity 和全片人工难例门。
6. 只有 `training_valid_mask=1` 的像素才能进入 Visual Aux；失败区不能用桌面或盘子伪装。

## 为什么三路手腕看起来重合

裸手 Chips023 的 Stereo proxy 是把 HaWoR 多个可见关节的 Stereo 表面 Z 差取中位数，再把结果放回 HaWoR wrist 射线；所以紫色 Stereo 与橙色 HaWoR 在 RGB 上重合。它们三维欧氏距离均值仍为左 32.5 mm、右 38.7 mm。

手套 PlayCards0910_001 的 Stereo surface 是在 Controller wrist 射线上取 11×11 Stereo depth 中位数，再沿同一射线反投影；所以紫色 Stereo 与绿色 Controller 在 RGB 上重合。它们全片三维欧氏距离均值仍为左 80.0 mm、右 186.0 mm。

这类视频的二维点只能说明“使用哪条射线”，不能显示沿射线的 Z 分离。深度差必须看毫米曲线、侧视 X-Z/Y-Z 图或逐帧 JSON。Controller/PICO、HaWoR 和 Stereo surface 都不是外部手腕真值。

补充说明：浅层 `三路手腕_Z方向分离全片.mp4` 使用逐帧相对坐标，裸手版每帧令 PICO Z=0，手套版每帧令 Controller Z=0；参考点固定只是坐标原点定义。Chips023 原始 PICO `T_wrist_to_camera` 的绝对 optical-Z 实际变化范围为左腕 30.1 mm、右腕 221.9 mm。该相对尺不得被解释为 PICO/Controller 的绝对轨迹。

已增加两条不再依赖二维点间距的全片复核：`archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/裸手_Chips023_三路手腕_Z方向分离全片.mp4` 与 `archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/手套_PlayCards0910_001_三路手腕_Z方向分离全片.mp4`。右侧以 PICO/Controller 为 0，逐帧显示 HaWoR 和 Stereo surface 的 signed camera-Z 毫米差。

## 当前基线边界

| 阶段 | 当前基线 | 本次修正后的解释 |
|---|---|---|
| HaWoR | 裸手 monocular MANO bounded-v2 | 手套域严重失配；单目 Z 非外部真值 |
| 新传感器 Hand | Controller wrist + MANUS25 | 工程观测，不叫 HaWoR/MANO 真值 |
| Role/Object Mask | exact78 为 SAM3.1 两条独立 lane | 新手套线尚无时序 authority |
| Depth | FoundationStereo corrected metric v1 | 可见表面 optical-Z，不是 wrist joint |
| Object6D | observed-only + KEEP_INVALID | 遮挡帧不补成正式 pose |
| Clean | same-pixel temporal donor + ProPainter | 结构闭环，不保证接触/语义正确 |
| Robot Geometry | world-first v7.1 over v5.2 | 可独立于 Clean；非控制真值 |
| Occlusion | Silver 开发态 | 未解决 hidden appearance，不能称遮挡已处理好 |

权威的当前算法、代码/权重 SHA 和 scope 仍以 receipt 绑定的 `docs/governance/CURRENT_BASELINE_REGISTRY_V2.json` 为准。
