# 支线4：HuRo 核心同输入诊断

本轮质量任务复用旧同目标比较，不新增HuRo求解；有限比较已收束，不给胜者或产品质量PASS。

当前任务仅复用[旧共同目标 Local/HuRo 限定比较](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json)和既有视频，不为本地 Poker/Chips 的变化重跑 HuRo 求解或后处理。`next_action`：把旧比较的会话、共同有效分母、硬限位失败和可看视频接入本轮导航，然后本线终态；不宣布赢家或训练真值。新增 HuRo 求解预算为 0。

## 最新代表会话任务终态（2026-09-24）

本轮仅完成既有共同目标比较的冻结结论与[同步视频导航](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json)：007／031 旧共同目标下 HuRo 原 q 存在硬限位失败，不能宣布方法胜者或用于本地代表会话的训练真值。本轮 HuRo 新求解、新平滑、新代表片实验均为0；不因为本地 Chips／Poker 代表改变而重做第三方方法。下文旧“本轮”均为历史结果，派生有界 q 不热换原 HuRo q。

本轮四线CPU增量任务已终态；以下“当前”均指本轮最新实物，不表示有活动进程。

新增[固定窗可读代价图索引](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/compare/huro_limit_tradeoff_dashboard_v1/RESULT.json)：原q越限侧帧007为3/32、031为8/16；派生q均为0，但这只解除窗口内机械臂硬限位，不解决完整旋转、碰撞或产品质量。

当前四线增量实例保留原HuRo q，固定007 181–196与031 66–81各16帧生成派生有界后处理，并以相同固定URDF独立重算FK：[候选数值](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/compare/huro_limit_repair_window_v1/RESULT.json)、[FK及代价审阅图](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/compare/huro_limit_independent_review_v1/RESULT.json)。窗口内源有效32/32与16/16侧帧均有有限、独立FK硬限位合格的派生q；007修3侧帧，031修8侧帧。031相对冻结目标的最大位置代价增量约0.139 mm、完整旋转约0.037°。这是派生运动学窗口证据，碰撞、全片与官方HuRo方法质量仍未通过。`next_action`：只在固定质量/碰撞合同与资源具备时评价扩展，不把本侧车热换旧共同比较。以下十小时段落为历史结果。

当前10小时实例已复用旧共同目标和两方法q，完成[007/031同一固定URDF独立FK评价及同步视频](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json)，并在[15槽位收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/delivery/RESULT.json)核对全解码与SHA。HuRo部分q触硬限位：007输入共同有效756侧帧中运动学共同有效526，031为102中18；不删原分母、不裁剪q造通过。`next_action`：限定结论已交付；若未来要修HuRo限位或同背景质量，须新授权和共同合同，不称当前有方法胜者。Local独享R1不混入。以下旧段落仅为历史快照。

当前：R2两会话共同目标数值比较和拒绝背景诊断已完成，S1只读复用。见[S1比较继承收据](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/compare/WAVE0_READINESS.json)及[R2视频](visuals/HUMAN_TO_ROBOT_R2/README_ZH.md)。合格Clean背景对照未完成，无方法胜者。S1没有新增HuRo求解、修复或质量采用，当前无活动任务。

## V5历史与继续适用的技术边界

**终态：`FAILED_QUALITY_LIMITS`。** 核心腕目标测试通过，但两条全片结果触发关节限位硬门，仅交失败对照视频；见[终态收据](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/HURO_TERMINAL.json)。本卡不授权重跑。

本卡仅针对[唯一当前路线](HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)，不阻塞主产品。唯一算法点是在 HuRo 核心加有效性屏蔽的腕位置／旋转残差（5mm／2°），先做目标扰动方向性、有限差分与 invalid 不施约束测试，再跑固定007／031全片，不扫权重。

公平比较必须锁同一个 HandMotion 快照、资产、mount、placement、有效帧、Clean与合成器；本地 R0 对 HuRo 核心，支线1的可选 R1另列。旧“HuRo手指＋机械臂后处理”只作历史对照，不算核心方法改进。031 左侧当前无合法输入时，该侧不可评估，不能填0或夸称双手成功。碰撞按有效帧分母、事件帧率、持续时间和穿透程度统计，不把 contact 记录数当事件数。
