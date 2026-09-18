# 0915 SAM3.1 严格角色单会话 Canary

状态：`EXECUTION_PASSED / AWAITING_USER_VISUAL_REVIEW / NO_BATCH_AUTHORITY`

本 canary 只运行 `play_cards_0915_001` 的 150 帧。输入为用户确认的物理左目
`sourceIndex=1 + resize-only`，分辨率 1280×960；没有执行去畸变或 remap，也没有读取
PICO26、Controller pose 或 `trackingData` 手部结果。唯一 Mask 权重为已固定的
`sam3.1_multiplex.pt`。

## 直接复核

- [150 帧并排审阅视频](0915_SAM31_STRICT_ROLE_REVIEW.mp4)：左侧原始 RGB，右侧角色实例；
  150 帧、30 FPS、1280×480，完整解码通过。
- [六帧总览](0915_SAM31_STRICT_ROLE_CONTACT_SHEET.jpg)
- [机器结果收据](../../../../tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V4_RESULT.json)

颜色图例：左右手为青/红，前臂为黄，手指皮套为紫，线缆为橙，三张牌分别为绿/蓝/洋红。
每个角色独立持久化；Clean 所需的 `human_equipment_union` 仅由手、前臂、皮套和线缆派生，
不吞并任务物体。

## 当前实现

每个角色使用短文本和人工复核的首帧 box 初始化；手部还使用 HaWoR 投影区域做候选选择与
逐帧质量检查。当前固定 runtime 是 `Sam3MultiplexTrackingWithInteractivity`，所以首次 box
在本地实现中的准确语义是 `MULTIPLEX_GEOMETRIC_BOX`，不是另建的 exemplar 模型。正负点
当前用于候选选择和质量证据；没有启用会切换到不连续 partial-SAM2 路径的逐点 refinement。

全片采用前向/反向传播。只有面积、连通域、质心跳变、HaWoR 重叠、腕部连通或任务 ROI 等
质量条件触发时才允许至多一次 fallback reseed；没有固定每 N 帧重置。每帧显式记录
`seeded`、`tracked`、`reseeded` 或 `unknown`，空 mask 不等于角色不在画面。

## 单样本结果

| 实例 | 有证据帧 | unknown | reseed | 当前判断 |
|---|---:|---:|---:|---|
| left hand | 145/150 | 5 | 1 | 可作为当前单样本基线 |
| right hand | 148/150 | 2 | 0 | 可作为当前单样本基线 |
| left forearm | 7/150 | 143 | 1 | 不可用 |
| right forearm | 90/150 | 60 | 0 | 部分可用，未过全片门 |
| left finger sleeve | 8/150 | 142 | 1 | 不可用 |
| right finger sleeve | 8/150 | 142 | 1 | 不可用 |
| left cable | 8/150 | 142 | 1 | 不可用 |
| right cable | 5/150 | 145 | 1 | 不可用 |
| playing card 00 | 143/150 | 7 | 0 | 单样本稳定 |
| playing card 01 | 146/150 | 4 | 0 | 单样本稳定 |
| playing card 02 | 95/150 | 55 | 0 | 遮挡/移动阶段仍不足 |

因此，当前 canary 证明“角色实例 + 时序状态账本”的工程链路已经打通，也证明手和前两张牌
明显优于长关键词方案；它同时否定“只靠一个宽 box 即可稳定分出前臂、皮套和线缆”。这些弱
角色保持 `unknown`，不得进入 Clean、遮挡或 Contact 几何。运行任务 `PASSED` 只表示单会话
执行和证据发布成功，不表示 11 个角色全部通过质量验收。

## 边界

- 没有创建 `tracker`、`controller` 或 exemplar 子系统。
- 没有自动扩到 220 会话；是否继续优化由本视频的用户复核决定。
- 本结果不是像素 Gold、Contact 真值、Robot authority 或物理部署依据。
- 单次运行约 500 秒，峰值 CUDA allocated 约 31.2 GB；模型只加载一次。
