# 支线1：Scene / Clean / 几何

本轮已登记[四支线收敛任务](COMPLETION_20260928_ZH.md)。以下为前驱冻结证据；新进展以[机器状态](STATUS.json)及本轮lane结果为准，不能将旧数值当新验收。

前驱2026-09-24终态：无活动任务；没有合格Clean产品。本卡汇总至2026-09-24共享手终态，文件名V5仅为链接兼容。

## 已完成

- R2真实执行007/031/0902_103/0902_042全片ProPainter候选；不是未调用模型。
- S1完成031全149帧FoundationStereo及三张牌的可见表面/中心/平面方向，不等于隐藏完整位姿。
- 后续核验写入区、保护区、模型内部、传播与回贴；局部保护修复不等于视觉改善。
- 最新Poker76–91帧完成LaMa V2/V3真实推理，定位到输出域适配错误。

## 最新失败与下一步

已处于0..255的LaMa输出被再次乘255，出现饱和白洞。这是实现无效，LaMa方法质量尚未评估；不是已经证明该方法质量失败。现任务修复预算已耗尽，未扩171帧。

新授权后仅修输出×255一项，冻结其他条件，重跑同16帧并与Raw/旧候选同帧比较。必须核验mask外逐像素不变、物体保护、残留与时序；未过固定窗不得扩全片。007残留与供体覆盖问题另立任务，不与Poker混算改善。

## 权威证据

- [最新Clean终态](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3/TERMINAL_RESULT.json)
- [最新状态](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/STATE.json)
- [031几何与Contact历史分会话终态](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/SESSION_TERMINALS.json)
- [视频与结果](SHARED_HAND_DELIVERY_RESULT_ZH.md) · [历史轮次](../plans/INDEX_ZH.md)

严格Contact/R1未取得可用成果；同会话不同图像域和短窗/全片结论不得互相继承。新实验不得阻断其他无此依赖的支线。
