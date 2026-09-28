# 支线1：共享 Scene／Clean／遮挡

本轮质量任务已按有限候选终态；以下新固定窗及同域G0均为真实证据，但Clean和遮挡未获产品质量PASS，不再原配方重试。

当前任务：Poker `play_cards_0902_042` 76–91 帧已用现有 SAM3 产生逐帧可见牌面支持（16/16帧、三个实例ID持续），并运行唯一一次[牌面保护 ProPainter 固定窗](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/VISUAL_REVIEW.json)。约6,609个原写入区内的可见牌面像素得到保护，但手持牌边仍涂抹、腕带与绿边残留；[同帧视频](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/POKER_076_091_RAW_OLD_SUPPORT_NEW_REVIEW.mp4)已解码16帧，质量拒绝，不扩171帧。[171帧旧新支持差异](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_ACCEPTED_MASK_VS_CURRENT_AUDIT.json)还显示现行旧候选在投诉窗对腕带/前臂覆盖明显不足；历史获验收 Clean 的[当前域重绑定](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE/RESULT.json)也不能继承旧版PASS。`next_action`：本次冻结配方停止扩片；同会话几何或其他不依赖合格Clean的工作继续。0902 sourceIndex0 矫正域不可与0915 sourceIndex1混用；尚无新的Clean质量PASS。

[Poker 同域 Stereo G0](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_0902_STEREO_G0.json)确认本会话 sourceIndex0 矫正域与旧3帧 canary 一致，但当前完整171帧深度不足、实际重叠 UNKNOWN 843,458 像素。完整 Stereo GPU 批次在 Clean 与 Motion 必需门均拒绝时不启动；这不是遮挡质量通过，也不把3帧推广为171帧。

## 最新代表会话任务终态（2026-09-24）

代表为 Chips 0915_042 与 Poker 0902_042。Poker 的[171 帧旧 Clean 审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_042_wave4/SCENE_CLEAN_CANDIDATE_REVIEW.mp4)结构合法但质量拒绝；本轮固定样本仍见牌面／手部补图污迹，未获采用。Chips 0915_042 有原 RGB，但本任务未找到本会话独立合法的模型删除 Mask、保护、reference 支持及 Clean；没有把旧 HaWoR 投影转成 Scene 真值或生成假 Clean。后续若取得可信独立支持，需要新授权任务才可生产候选；这一缺口未阻断本轮 Robot／学习接口诊断。详细入口见[本轮导航](visuals/HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE/INDEX_ZH.md)。下文“本轮四线CPU增量”仅指已结束的旧任务。

本轮四线CPU增量任务已终态；以下“当前”均指本轮最新实物，不表示有活动进程。

当前四线增量实例已完成007固定181–196帧真实model mask、write/protect与Clean的逐帧消费核对：[16帧结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/scene/full_window_consumption_review_v1/RESULT.json)、[投诉点分解](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/scene/complaint_consumption_v1_runtime_fix1/RESULT.json)。写入区外改写0；帧184六个设备点已进入模型及写入区却仍有残留，左白线身份未证。不再只修这些点的Mask或原样重试模型。`next_action`：核验真实16帧旧新审阅视频，针对背景重建/投诉身份提出可证伪的新输入证据；本轮不启动第二模型候选。以下十小时段落为历史终态，不是当前任务状态。

当前10小时实例 `human_to_robot_10h_delivery_20260924`：007 唯一57帧上下文候选已实际运行并[复核拒绝](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/scene/context_007_v1/AI_VISUAL_REVIEW.json)；固定181–196帧仍有残臂、附件/线缆及碗边残影，不扩378帧、不重试同参数。四会话旧Clean已在[15槽位收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/delivery/RESULT.json)中按真实会话、全片帧数与SHA复核。已有031 patch、Robot对应、连接件查询保留原短窗适用范围，不能称Contact/R1或整机碰撞通过。`next_action`：本配方已无合法扩片动作；若未来获得新的牌边消费错误或场景证据，应另获授权，不能原样重试。当前产品质量不因本线局部执行自动升级。

以下旧段落仅为历史任务快照。

最新007跨眼来源诊断：[四帧终态收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_007_cross_eye_source_probe_20260923/attempts/attempt_0001/RESULT.json)。physical-right 到封存 physical-left Raw 在同一 SBS 帧上尝试 181／184／192／196；左眼域逐像素闭合，四帧桌面平面匹配未过预定门，合法写入支持均为0。它只拒绝此跨眼配方，不证明所有跨眼方法失败。旧 Clean 质量仍拒绝，007 不扩378帧、不取得产品采用。

最新 007 供体诊断已收敛：[全会话两帧搜索](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_007_late_donor_probe_20260923/attempts/attempt_0001/RESULT.json)在192／196帧找到8／4个通过原匹配门的供体；[空间内插复核](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_007_spatial_donor_check_20260923/attempts/attempt_0001/RESULT.json)要求供体及目标均落在 SIFT fit 内点凸包内，三供体支持随之降为11,410／274,299（约4.2%）和0／292,685。该差异说明旧门允许大范围未验证外推；两项 `PASSED` 均只针对诊断结构，不能升级为真实隐藏桌面、Clean 或产品质量。现有同会话平面供体法停止扩片，保留失败实物；获得新的同场景无遮挡证据或另获授权的合成候选后，再开独立有限任务。当前正式Clean采用仍0。

当前（S1）：R2四会话全片ProPainter已执行；S1又完成031/007各16帧真实Clean。设备支持进入模型，但031边界模糊/补图错误、007黄线残留/背景污迹仍使两条质量拒绝。042附件证据64/171帧，103无独立seed。正式Clean采用0，当前无活动任务。见[S1逐会话结果](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/SESSION_TERMINALS.json)及[最新审阅](visuals/HUMAN_TO_ROBOT_S1/README_ZH.md)。

## V5历史与继续适用的技术边界

以下“ProPainter未运行”仅描述V5，不描述R2/S1。

**终态：`FAILED_QUALITY_C`。** 四会话全片 Mask 已执行，但设备漏检和物体保护未过门；ProPainter 未运行、正式 Clean 为0。详见[Scene终态收据](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/SCENE_FINAL_ASSESSMENT.json)。本卡下列是已执行的合同，不授权重跑。

本卡仅针对[唯一当前路线](HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)。输入为四会话各自[冻结原帧域](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/ROUTE_MANIFEST.json)。Stage 为 `role_masks → prepare → inpaint → 独立QA → compositor`；真实角色不足、物体实例未知、参考帧设备污染或 GPU 资源不足均须留明确结果，不可用旧域 Mask 代替。

Mask 独立存人手／前臂、佩戴设备、任务物体；Chips 实例不合并，Poker 未证同牌同面不继承身份。可信可见物体 `M_protect` 与时间安全带分开；所有参考帧均给模型实际消费的 `M_context_exclude`。ProPainter 生成像素标为合成，不做几何证据。最终统一输出像素域硬回贴，`M_write` 外与可信保护区逐像素等于 Raw，Clean 单独审查残留、误删、闪烁、子视频接缝。机器人遮住问题不算 Clean 通过。

Depth/Object/ContactHints 只从 Raw 与同域几何产生。Depth 注册未通过则 UNKNOWN；R1 只能在独立有效连续窗口内小幅手指修正，失败整段回 R0。缺 R1 不妨碍基础产品候选，不能冒充接触改善。
