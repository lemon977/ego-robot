# Human→Robot Baseline v1：007 支持覆盖与 031 目标来源恢复

这是旧 `human_to_robot_product_first_cleanup_20260923` 的有限后继，预算 3 小时；不复活旧 attempt。四支线产品架构不变，本段只处理 Scene 和 Motion/Product 的两个已定位缺口。所有新文件、缓存和临时文件留在项目内。旧结果、raw、processed、archive、sealed 只读。

## 目标与边界

1. 对 007 固定帧 181–196，以原图投诉区域和左右手可见区域为分母，逐帧量化既有角色、实际模型 mask、写入区与最终 Clean 的对应关系。特别检查 184–193 帧左手是否漏入 model mask；不得以“角色非空”代替覆盖。保存叠图和实际数组。只有找到可验证的新输入支持并通过物体保护、reference 消费测试，才允许一次改变签名的 ProPainter 小窗；没有这些证据则终态为输入支持缺口，不刷 GPU。
2. 对 031 固定 149 帧、原右侧 102 输入、左侧 0 输入，追踪 HaWoR ROI、模型输出、target builder、Robot FK。第 47 帧及相邻帧单列；`inferred` 按本地 writer 的“非直接 ROI 模型预测”语义解释，不冒充 motion infiller 或无图像证据。区分 target 数值有效、来源资格、时间连续性和 IK 适用性；不覆盖旧 NPZ，不运行新 IK。
3. 若上述两项生成新的合法消费者输入，下一阶段需按本任务 packet 的条件分支做真实消费与固定窗质量比较；否则保留明确拒绝与解除动作。不能把诊断通过称产品质量改善。

## 停止与验收

- 固定两处旧失败反例、逐帧数值、同帧可视化、SHA、输入/代码签名和真实命令。旧 007 Clean、031 R0 仅作为只读参照。
- 007 的模型输入全区域覆盖、Clean 残留、物体保护分别记账；031 的 149/102/0 分母不改。
- 不改变 20 mm 位置、15° 方向/完整旋转、碰撞和 Contact 门；不将模型预测升级为外部三维真值。
- 合法输入缺失或 3 小时预算到点时终态化本段，保留具体下一项；其他支线已有合法结果不撤权。
- 产品仍为 `OFFLINE_VISUAL`，training/control/deployment/external metric 权限均为 false。
