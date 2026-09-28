# 支线3：HaWoR／Robot 与最终产品

本轮质量任务已终态：初始化反事实及右臂后缀候选均未通过冻结质量门，旧正式产品q保持不变，技术质量仍拒绝。

当前任务的 Chips 156–160 帧[初值单变量反事实](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/motion_product/CHIPS_SEED_COUNTERFACTUAL_0_160.json)保持冻结前缀一致，但最大步长反而2.152→2.197 rad、159帧收敛退化，故不扩全片或替换产品。Poker 0902_042 [旧求解器精确复现](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/motion_product/POKER_096_111_SOLVER_CAUSAL_AUDIT.json)确认第99帧`status=0`后被neutral重置；[右臂100–170帧完整依赖重算](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.json)虽把99→100步长1.968→0.114 rad，右臂原质量门却152→147/171，候选拒绝，旧产品q不变。[104→105小指首错层](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/motion_product/POKER_104_105_FINGER_FIRST_LAYER.json)显示源目标骨方向最多只变约3.4°，但求解q在侧摆±0.26 rad两端跳变；当前不先滤波或改权重。当前[产品独立质量门](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/motion_product/POKER_171_TECHNICAL_QUALITY.json)仍拒绝，技术PASS前不邀用户采用。031不运行新IK。

## 最新代表会话任务终态（2026-09-24）

Chips 0915_042 与 Poker 0902_042 已各生成原 RGB＋固定第三人称全臂[完整回放导航](visuals/HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE/INDEX_ZH.md)，分别363／171帧，真实消费保存 q/FK 与连接件视觉。Chips 使用68.4 mm完整开发安装合同的新虚拟机械臂；Poker 复用旧 R0。两者并未取得稳定质量通过：[同真实 dt 数值](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_representative_baseline_20260924/attempts/attempt_0001/MOTION_DISCONTINUITY_DIAGNOSTIC_V1.json)指出相邻目标转动很小时仍有机械臂约2弧度解跳变。唯一 Chips 连续性修复候选虽降低左臂最大步长，却使左／右目标门通过帧351／295→281／166，已[拒绝](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_representative_baseline_20260924/attempts/attempt_0001/CHIPS_ARM_CONTINUITY_CANDIDATE_ASSESSMENT_V1.json)，旧 q 与视频不变。031只维持149／102／0困难回归，不重做第47帧定位或开新IK。下文旧“本轮”均为历史快照。

本轮四线CPU增量任务已终态；以下“当前”均指本轮最新实物，不表示有活动进程。

[031第47–49帧原图、HaWoR框和源腕位置](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/motion_product/frame47_upstream_evidence_v1/RESULT.json)已落盘：47帧为首次有效，框22×12px贴底；48帧63×27px，源腕位置跳变约1.286m。102个有效右手帧的原HaWoR腕点与HandMotion保存值逐项相同。这是输入/模型起始证据缺弱的定位，不是模型唯一因果证明或新质量门；不删异常帧。

当前四线增量实例从冻结target与保存Robot输出生成007/031全时间轴连续性数组和[审阅图](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/motion_product/target_continuity_review_v1/RESULT.json)。[149帧HandMotion→target位置绑定](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/motion_product/target_source_binding_v1/RESULT.json)定位031第47→48帧约1285.65 mm跳变在保存HandMotion中已存在，不由target-builder的位置映射制造；旋转另有合同变换，不称完整姿态逐项一致。保存Robot该步约255.4 mm、48帧目标跟踪差约225.54 mm；显示/FK一致不代表跟踪通过。149总帧、原右102、左0保持。`next_action`：若要修质量，应在新授权的producer范围追溯冻结目标突变，不用删帧、改画面或启动本轮未授权的新IK。以下十小时段落为历史结果。

当前10小时实例不运行新IK，复用031第47帧冻结目标突变结论、旧149总帧／右102／左0分母和正式入口q/FK。已新增两片[007全链审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_007/RESULT.json)／[031全链审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/full_chain_031/RESULT.json)，均为Raw、保存HandMotion投影及正式失败产品的同步展示，不称求解改善。四片纯Robot按正确会话分槽并全解码；正式入口与同签名resume沿用[S2四片收据](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/formal_entry_resume/attempt_0002/RESULT.json)，本轮未改其求解或产品代码。031既有几何短窗新增[连续审阅](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/motion_product/geometry_reuse_review_v1_runtime_fix1/RESULT.json)。`next_action`：本轮审阅已完成；后续运动质量或左手恢复需新的输入证据和授权，不能把全片解码当双手恢复。以下旧段落仅为历史快照。

031 同像素MANO前表面／Stereo光学Z已扩到全149帧时间轴并加入三张可见纸牌的UNKNOWN与边界排除：[逐帧账本](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_surface_depth_temporal_audit_20260923/attempts/attempt_0001/lanes/motion/surface_depth_temporal_v1/RESULT.json)及[149帧诊断回放](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_surface_depth_temporal_audit_20260923/attempts/attempt_0001/lanes/motion/surface_depth_temporal_v1/031_SURFACE_Z_OBJECT_GUARD_FULL_REVIEW.mp4)。严格纸牌准入的54帧中，物体排除前后逐帧深度差中位数的中位数约−51.7／−51.6 mm，纸牌边界剔除量460/73,345点。晚段物体实例UNKNOWN，不能推广到原102帧全体；也不能将该差值作为腕中心真值或直接静态标定。旧Motion/R0不变。

031固定16帧MANO可见表面—Stereo同像素光学Z诊断已落盘：[结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_surface_depth_diagnostic_20260923/attempts/attempt_0001/lanes/motion/surface_depth_v1/RESULT.json)、[真实对照](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_surface_depth_diagnostic_20260923/attempts/attempt_0001/lanes/motion/surface_depth_v1/031_MANO_STEREO_SURFACE_Z_16FRAME_REVIEW.mp4)。15/16帧有37,621个经手区内部、LR一致和局部连续筛选的同像素样本，逐帧差异不稳定；不把可见表面当关节中心，也不从本诊断直接修改腕目标或固定偏置。物体遮挡和独立可见对应是下一步限制。四产品质量／采用仍0/4。

031 可观测性分母已追加修正：[149帧区域结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_observability_dtype_fix_20260923/attempts/attempt_0001/lanes/motion/observability_dtype_fix_v1/RESULT.json)及[完整同帧回放](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_031_observability_dtype_fix_20260923/attempts/attempt_0001/lanes/motion/observability_dtype_fix_v1/031_REGION_PROXY_FULL_REVIEW.mp4)。旧 R2 账本因 `uint16` 前景1按8位灰度读取，将区域可见误计为0；修正为102/149（47–148）。这是 SAM 手区代理，不是独立左右手／腕中心真值；SAM又是HaWoR ROI上游，故102帧模型预测不能用它自证精度。旧HandMotion和R0数值未修改，质量仍未通过。

当前：R2四会话Robot候选已生成并通过内容签名resume，质量/采用0。007双侧378/378帧；031绑定右侧102/149帧、左侧缺失。见[R2 Motion状态](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/STATE.json)。S1核验031源包 `observed=[0,0]`、`inferred=[0,102]`，据此阻塞Contact；该标记不等于独立RGB已经证明手不可见，生产者语义仍需核查。S1未新增动作恢复或减抖改善，当前无活动任务。

## V5历史与继续适用的技术边界

**终态：四会话 HandMotion/R0 来源闭合，但质量未过；去人产品0。** 031左侧无合法模型输入、103腕容差0/0。见[四会话交接收据](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/MOTION_HANDOFF_V5.json)。本卡不授权重跑。

本卡仅针对[唯一当前路线](HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)。固定 007／031，不因失败换会话。逐帧逐侧追踪 `可见代理→ROI→模型输入→原始预测→导出有效→retarget→R0`，先找第一个从有变无环节，不能对空输入调平滑或镜像另一手。

007 已有双侧旧候选全片，031 当前左侧无合法 ROI，旧少数所谓左检测与右手大面积重叠；未获独立第二手证据前 `NOT_VISIBLE_OR_UNKNOWN`，不伪造左侧。bounded 修正的 6mm／4°／6°是相对输入参数限制，不是帧间速度上限；评价抖动同时必须保留动作幅度、拇指食指夹合与真实dt。输出统一 HandMotion、Robot R0 状态、数值硬门与来源收据。

机器人求解不等 Clean，但最终 `robot.mp4` 必须消费支线1同版本 Scene 和同一合成器。旧 R0 或 Raw overlay 可诊断，不能叫产品通过。无可靠 `c2w` 时相机相对视觉、没有世界去头漂移结论；无可靠物体深度时遮挡 UNKNOWN。
