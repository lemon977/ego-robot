# Clean SAM3.1 分层路线 T0 运行结果

T0 已通过中央 GPU lease 在不可变 `attempt_0004` 中完成。这里的“通过”只表示真实
SAM3.1 推理、视频生成和结构校验完成，不表示新 Clean 已通过质量门。

| 会话 | 输入帧 | 复核视频 | 当前删除面积相对变化 | 可见物体被候选修改 | 结论 |
|---|---:|---|---:|---:|---|
| Poker245 | 151 | 151 帧、15 FPS、慢放 0.5× | -21.72% | 0 px | SAM3.1 轮廓明显更贴手，但未达到预设“减少至少 50%”的首轮门 |
| Chips039 | 306 | 306 帧、15 FPS、慢放 0.5× | -21.46% | 0 px | SAM3.1 轮廓明显更贴手；旧 Clean 中已有的盘子/补图污染仍可能保留 |

浅层视频：

```text
archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/Clean_SAM31分层路线_Poker245_全片慢放.mp4
archive/baseline-20260917-0aa69e9/content/history/docs/current-visual-shortcuts/Clean_SAM31分层路线_Chips039_全片慢放.mp4
```

六栏含义：

```text
原始 RGB | 当前冻结删除区 | 真实 SAM3.1 左右角色
当前 Clean | 自适应边界候选 | 像素来源/UNKNOWN
```

颜色固定为：绿色=左手，紫色=右手，橙色=Tracker，青色=当前可见任务物体；
来源图中绿色=从 Raw 恢复，紫色=沿用旧 Clean，粉色棋盘=没有合法来源的 UNKNOWN。

SAM3.1 两侧均在全片产生非空 mask，左右 mask 总交叠为 Poker `0 px`、Chips
`854 px`。这只能说明推理输出覆盖和身份分离代理，不能当 segmentation accuracy。
可见任务物体 `0 px` 变化来自显式的 `RAW_CURRENT_VISIBLE_OBJECT` 恢复规则，是代码
合同闭环，不是隐藏物体纹理恢复正确的证据。

## 已回答的问题

- 当前固定删除区确实比 SAM3.1 手/前臂轮廓更宽，T0 全片总删除面积减少约 21%。
- SAM3.1 视频 predictor 可以在两个裸手会话上做左右独立、锚帧启动、前后双向传播；
  本次没有回退到 SAM2/SAM2.1。
- 仅收紧 mask 并恢复 Raw 可见物体，不能修复旧 Clean 已经产生的错误 donor、拖影或
  盘子污染；因此不能把 T0 候选当作完整 Clean successor。

## T0 后的决定

```text
新路线研究：GO（进入有界的 fresh donor / ProPainter 对照）
Clean authority 晋升：NO-GO
接触遮挡已解决：NO
```

下一轮必须 fresh 生成背景，禁止复用旧 Clean 像素作为最终候选；并将
`BACKGROUND / CURRENT_VISIBLE_OBJECT / TEMPORAL_OBJECT_DONOR / UNKNOWN` 分层保存。
只有 fresh causal donor + ProPainter 与当前基线在同一冻结帧集上比较后，才能判断
盘子污染、牌边损伤和接触区是否真正改善。

完整机器结果：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/
  attempts/attempt_0004/RESULT.json
```
