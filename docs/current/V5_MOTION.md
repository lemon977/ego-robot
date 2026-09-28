# 支线3：HaWoR / Robot / 原场景产品

当前：无活动任务，质量未通过。原始q/FK与历史失败保持不变。

## 已完成

四会话原场景候选、连接件颜色/深度/部件ID、正式入口与同签名resume已经执行。007双侧378帧；031全149帧时间轴、右102有效、左无合法输出。后续Chips0915_042和Poker0902_042有363/171帧第三人称回放，但不是新的合格原场景产品。

## 需要分开处理的问题

- 031第47→48帧约1.286m跳变在HaWoR源腕中已存在；第47帧贴底22×12px ROI。不能归咎renderer或删帧掩盖；没有证明物理不可达。
- Poker与后续Chips还有求解分支/初始化导致的跳变。旧平滑和连续性修复曾恶化腕目标门，未采用。
- 最新Poker80–111帧联合轨迹四块中三块可行，右93–95仍超过20mm/15°门；搜索未找到合格窗，不证明不存在可行解；没有扩全171帧或把三块拼成成功全片。

## 后续任务边界

Poker冻结目标、装配与评价器，只针对失败块的约束求解及边界连续性提出新候选。031作为独立上游证据任务，不能混入Poker的求解改善率。通过固定窗后才能扩片，并同时检查幅度、延迟、限位、碰撞和原场景合成。

- [最新Robot状态及失败帧](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/STATE.json)
- [最新固定窗数值](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/POKER_080_111_JOINT_TRAJECTORY_FIX1.json)
- [最新视频导航](SHARED_HAND_DELIVERY_RESULT_ZH.md)
- [四产品正式候选历史](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)
