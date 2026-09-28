# 支线2：Controller＋MANUS 独立运动入口

本轮质量任务已终态：真实原RGB＋q诊断前向可执行，但局部Kai22标签未获训练资格；以下结构有效数不等于标签质量通过。

当前任务已完成三会话[局部 q/FK/碰撞收据复用与目标尺度审查](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/sensor/SENSOR_LOCAL_KAI22_QUALITY_AUDIT.json)：097/098/101 有效侧帧330/358/244，独立FK与保存值一致；旧碰撞检查仅在原声明范围内可复用。MANUS根到MCP中位约33mm、Robot约91mm；旧缩放目标指尖半径约566–569mm，Robot FK约154mm，现有q不获局部形状训练标签资格。新增[097无根骨段审查](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/sensor/SENSOR_097_ROOT_FREE_BONE_QUALITY.json)、[真实q-only H50诊断包](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/sensor/SENSOR_097_QONLY_H50_DIAGNOSTIC.json)及[physical-left原RGB＋q实际模型前向](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/sensor/SENSOR_097_QONLY_REAL_RGB_FORWARD.json)：未来50帧／2,200个结构有效q元素可重载，缺失腕安全屏蔽、前向和loss通过；当前状态token被屏蔽、非完整loader，训练准入元素0、优化0步。`next_action`：定位局部左右/尺度与腕独立性，确认表示错误才修并重评；不能通过关闭腕mask冒充局部形状合格。098不拟合，101不调参。

[097原生骨段语义核对](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/sensor/SENSOR_097_NATIVE_BONE_SEMANTICS.json)显示显式MANUS25→21映射下，两侧骨段尺度不一致：physical-left 的 Index 虚拟腕根→MCP 原生／Robot 中位34.3／96.0 mm，MCP→PIP 82.7／18.6 mm。单一根尺度修正不能使已保存q成为合格Kai22局部标签；这是供应商数据与机器人几何的内部比较，不是外部解剖真值。

## 最新代表会话任务终态（2026-09-24）

097／098／101 既有三条完整修正回放与共同后端继续作为可复用实体；[466 帧视频与等比例显示收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/lanes/sensor/review_metric_v2/RESULT.json)和[独立固定／异常样本复核](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/sensor/independent_review_v1/RESULT.json)本轮未重新求解。现有 q 的 H50 结构窗口分别为 115／129／72（两侧相同），完整62维窗口均为0；原生 MANUS 和 Kai22 标签分级，腕贴合无独立真值，训练资格不晋升。技术审阅仅覆盖既有固定及异常样本，不称全片逐帧视觉通过；后续独立贴合真值与新任务授权方可扩大结论，不重拟合098或调101。下文旧“本轮”是历史快照。

本轮四线CPU增量任务已终态；以下“当前”均指本轮最新实物，不表示有活动进程。

当前四线增量实例已对097/098/101全部466帧原生关节重投影并保存逐帧frustum状态：[数值](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/sensor/projection_frustum_v1_runtime_fix1/RESULT.json)、[全时间轴审阅图](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/sensor/projection_frustum_review_v1/RESULT.json)。正Z画外关节槽分别2385、4448、1429；它们不能算源无效或绘制遗漏，也不证明腕贴合。旧466帧q/FK和视频不重算。`next_action`：只凭独立RGB解剖证据审阅贴合；确认确定性绘制错误才修。以下十小时段落为历史结果。

当前10小时实例复用三会话466帧HandMotion、共同q/FK和已修正等比例视频，不重拟合。[本轮固定/异常样本与投影有效性复核](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/sensor/independent_review_v1/RESULT.json)已落盘，三条全片视频完整解码并绑定[交付槽位](visuals/HUMAN_TO_ROBOT_10H_DELIVERY/INDEX_ZH.md)；可见区外投影按3D有效与2D画内分开统计。`next_action`：技术审阅已按声明范围完成，等待用户视觉反馈或新的独立关键点真值；不重跑旧求解。Scene/Clean/Depth不是本线前置。以下旧段落只作历史快照。

Sensor显示修正任务已按限定用途终态 `PASSED`：[结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/RESULT.json)和[合同](../plans/HUMAN_TO_ROBOT_SENSOR_DISPLAY_CORRECTION_20260923/00_EXECUTION.md)。三条完整视频复用旧HandMotion与Kai22 FK；世界X/Z现为固定等比例，MANUS与Robot局部图共用整会话米制比例，既有合法骨段裁切不重复修改。九个固定样本可读，但没有独立腕中心真值；这不是安装标定、外部SLAM精度或训练资格通过。

当前：沿用R2三会话共同后端，权限为 `KINEMATIC_ONLY`。097/098/101共466帧已进入共同target-builder、solver、Kai22 FK；有效side-frame分别为330、358、244。见[R2 Sensor状态](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/STATE.json)和[完整回放](visuals/HUMAN_TO_ROBOT_R2/README_ZH.md)。S1未新增Sensor求解或标定，当前无活动任务；贴合质量与训练资格未晋升。

## V5历史与继续适用的技术边界

**终态：三条回放与运动包已交付，贴合质量未晋升。** 097／098／101共466帧，完整解码与来源检查见[结果收据](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/sensor/run_0001/RESULT.json)。本卡不授权重跑。

本卡仅针对[唯一当前路线](HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)。固定 097／098／101 三条 0916 Poker，165／179／122 帧。无需 HaWoR。`HAND_MOTION_V1.npz` 以解剖左右分别保存相机腕、21点、逐项有效性与来源；原始 MANUS25、节点名和显式25→21映射单独保留，不能将21点误认机器人关节角。

先验证时间、左右、四元数、MANUS local/global 与 root 应用，再比较 M0／历史 M1／固定安装候选。安装平移是 Controller 坐标系下每轴绝对±0.16m；0.05m 是相对原安装先验的残差尺度，绝不对 M1 累加。旧拟合器只用097冻结代理标注，101为回归而非全新盲测；无安装组证据不跨会话传播。世界视图若依赖开发固定相机假设，应与 authoritative `world_valid` 严格区分。

三个全片视频和运动包可作独立入口开发证据；投影错位时 `visual_quality_pass=false`，不能因全片解码通过称标定准确或产品闭环。
