# Human→Robot Baseline v1：唯一当前使用与验收文档

> 当前清理增量：三批精确目标实删，累计逻辑删除 5,219,025,177 字节，实际物理回收未知。056/068历史Stereo完整帧树已裁剪，旧收据字节不变但不再授予完整Depth直接消费；原视频、每段3个NPZ样本、原输入、代码、权重和环境保留。[清理进度](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/PROGRESS_CLEANUP_B2.json)、[限权说明](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/cleanup/batch2/HISTORICAL_PAYLOAD_TRIM_NOTICE.json)为当前口径。下方首波段落的762,165,394字节只描述先前ProPainter批次，不是累计值。

## 产品优先与安全清理任务（2026-09-23，已终态，质量未通过）

本轮任务 `human_to_robot_product_first_cleanup_20260923` 已终态 `REJECTED_QUALITY`，起点与完整约束见[当前计划](PLAN.md)。007 帧181–196已真正运行新ProPainter并生成旧／新Clean与同q/FK Robot对照；投诉线缆局部改善，但手与另一白线残留，整幅Clean和全片产品仍拒绝。031第47帧独立FK证明冻结目标异常跳变，不是renderer偷偷按目标放手；本轮不启动新IK。031牌边旧候选依然模糊，不进行无证据重跑。007当前代码的正式入口与同签名resume已完成378帧，但视频与旧S2字节相同，仅是集成复测。031同物体patch、Robot刚性对应、连接件碰撞三个消费者已实际调用，均只具有限定开发诊断资格。三批安全对象累计实删5,219,025,177逻辑字节，不代表物理回收量。四产品结构4/4、质量0/4、采用0/4不变。见[本轮视频与数值索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_PRODUCT_FIRST/INDEX_ZH.md)和[最终机器结果](/mnt/workspace/code/chaoyang/_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/RESULT.json)。以下各段仅记录对应历史轮次，不再是当前任务指令。

## 收敛任务终态（2026-09-23）

`human_to_robot_baseline_v1_convergence_20260923` 已终态：031固定窗位置16/16、支持方向0/16，因此未扩片；Contact完成510条资格检查、158条实际几何采样，但严格Contact/R1仍为0；Sensor新增466帧完整回放；15个交付槽位全部解码，质量通过与产品采用仍为0。详见[交付索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。


## 当前收敛任务（2026-09-23）

已登记 `human_to_robot_baseline_v1_convergence_20260923`。首波真实结果：031硬位置16/16但支持方向0/16，未扩片；Contact几何筛选158/510但严格资格0；007线缆局部top-1为16/16但角色支持仍不足；Sensor三片466帧新回放待审。详见[当前视频索引](visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。


## S2 进行中（2026-09-23）

S2 已完成真实连接件 renderer、深度/部件 ID 接口、visible-surface compositor 与正式 CLI 接入。007、031、0902_103、0902_042 已各生成一条完整可解码正式候选并通过同签名 resume；当前均因 Scene／运动／几何证据未满足而未采用。031 已证明画面忠实消费实际 q/FK，但原 full-pose 腕目标残差仍为 P50 77.5 mm、P95 104.4 mm；position-only 诊断改善位置却破坏旋转，因此被拒绝。031 固定窗可见表面遮挡 known coverage 约82.87%，其余17.13%保持 UNKNOWN；007及0902没有合格Depth，未制造排序。Contact/Robot R1 当前是 `0 screened / 0 executed / 0 adopted`，表示筛选尚未完成、尚未建立可用窗口结论，不是筛选后证明零窗口。Sensor三片与Local/HuRo证据采用冻结复核，不重复推理，也不宣布外部精度或方法胜者。 [当前S2视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)


> **2026-09-23 S1 质量闭环终态**：后继
> `human_to_robot_quality_closure_s1_20260923` 已终态化为
> `REJECTED_QUALITY / TERMINAL_WITH_QUALITY_GAPS`，当前无活动任务。S1 不是对 R2
> 候选的升级采用：它证明了局部附件 evidence 能真实进入 ProPainter，并闭合了 031 的正确
> encoded-domain Depth/Object6D；但 031/007 两条 16 帧 Clean 仍因边界模糊、幻觉、线缆残留或背景污迹被拒绝，正式 Clean 仍为 0。031 的直接观测手表面为 0，故严格 Contact 和 Robot R1 仍为 0。
> 机器事实见 [S1 最终结果](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/RESULT.json)、
> [最终验证](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/FINAL_VALIDATION.json)、
> [Scene 终态](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/SESSION_TERMINALS.json)
> 与 [Geometry/Contact 终态](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/SESSION_TERMINALS.json)。

## S1 新增实质证据

- 附件追踪：031、007 局部窗各 7 帧 seed + 9 帧 track；0902_042 为 64/171 已知、107/171 unknown；0902_103 无独立 seed，未补造。
- Clean：031 与 007 各真实执行 16 帧，附件改写覆盖分别为 100% 与 99.9995%；结构通过、视觉质量拒绝、未采用。
- 031 几何：Stereo preflight 通过，FoundationStereo 149/149 帧完成；三张牌分别发布 observed-only center/plane/axis，不补隐藏轮廓。
- Contact：右手 102 帧均为 inferred，direct-observed 双手均为 0；未把推断投影拿去采样 Stereo，因此合法 Contact 行和 R1 窗口均为 0。
- 所有 15 条 S1 视频完整解码；源数据、processed、archive、sealed 未写入。

S1封存验证记录1493 passed；上次对话中的终态后1494 passed是另一次测试，不能倒写到旧收据。本次同步后的代码回归另有JUnit/日志，并由[STATUS](STATUS.json)引用。最新视觉见[S1索引](visuals/HUMAN_TO_ROBOT_S1/README_ZH.md)，下轮提议见[三小时评审稿](../plans/HUMAN_TO_ROBOT_NEXT_3H_REVIEW_20260923/00_REVIEW.md)。

> **2026-09-23 R2 最终状态**：旧 V5 与 R1 下述结论继续只读保留；
> `human_to_robot_root_cause_gated_r2_20260923` 已终态化为
> `REJECTED_QUALITY / TERMINAL_WITH_QUALITY_GAPS`，当前无活动任务。四会话均完成真实 ProPainter
> 候选、逐会话 Robot 候选、完整解码和内容签名 resume；两条主会话另完成 Local/HuRo 同一拒绝背景诊断。
> 最终是 **Scene 候选 4/4、Scene 质量通过 0/4、正式 Clean 0/4、主产品候选 2/2、0902 回归
> 候选 2/2、全部产品质量/采用 0/4、Robot R1 执行窗口 0**。结构闭环不等于产品成功。
> 机器事实见 [R2 最终结果](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/RESULT.json)
> 与 [最终验证](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/FINAL_VALIDATION.json)。

## R2 已发生的实质进展

- Scene：031、007、0902_103、0902_042 已分别真实调用 ProPainter 并完整解码 149、378、284、171 帧。
  四条均为 `EXECUTED + structure PASS + REJECTED_QUALITY + CANDIDATE_ONLY`。第二轮只在031尝试
  “当前帧可见物体内部保护”，把该保护域内被改写像素从 41,028 降到 0，但固定帧没有实质视觉改善，
  手—物体交界模糊和设备漏擦仍在，因此拒绝该候选，并停止对007的无证据扩跑。
  [Wave4 四会话收据](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/PROGRESS_WAVE4.json)；
  [Wave5 质量结论](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/PROGRESS_WAVE5.json)
- Motion／Product：007 两侧 378/378 帧闭合；031 只有一侧 102/149 帧有合法输出，另一侧保持
  `UNKNOWN/invalid`，不镜像、不补零、不显示默认机器人姿态。消费者现已允许合法单侧窗口，因此031
  不再被“双手全局闸门”阻断；031和007分别生成149/378帧 Robot候选并通过同字节resume。由于其
  Clean仍被质量拒绝且遮挡未知，两条产品仍是 `CANDIDATE_ONLY / REJECTED_QUALITY`。
  0902_103 有效侧帧 284/281，0902_042 为171/171。
  四条产品均新增内容寻址 `CACHE_BINDING_V2`；只有 Scene、Motion、camera domain、robot asset、
  renderer 和配置签名以及现有输出 SHA 全部一致时才允许 resume。
  [Motion 状态](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/STATE.json)
- Sensor：097、098、101 已真实进入共同 target-builder／solver／Kai22 FK；有效 side-frame 分别为
  330、358、244，自碰撞检查覆盖全部有效项且本次数值为 0。权限只到 `KINEMATIC_ONLY`，不是
  `DEVELOPMENT_R0`、训练标签或控制真值。[Sensor 状态](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor/STATE.json)
- Contact／Robot R1：已对四视觉会话逐级终态化。031 左侧独立可观测证据不足只限制双手质量结论，
  不撤销右侧102帧R0；031真正的首个R1阻塞是严格公制Contact authority未获准。007缺Object6D
  可见表面，两条0902缺本轮可消费的Depth注册；没有任何R1窗口获准、执行或采用。
  [范围修正收据](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json)
- Local／HuRo：同 target 数值比较已重算，但没有独立真值所以不宣布胜者。真实 pinned 机器人资产的
  固定输入复现 byte-exact；只变 q、只变 camera、只变 background 均只在对应路径产生变化，背景
  变化侵入 Robot mask 为 0 像素。两主会话又各生成一条同背景 Local/HuRo 完整回放，但背景来自
  `REJECTED_QUALITY` Clean 且遮挡为UNKNOWN，所以只是诊断，仍不宣布胜者。
  [Wave10 收据](../../_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/PROGRESS_WAVE10.json)
- 装配：收到的 STEP 已可解码为候选几何；实测安装变换、Robot TCP、camera/world→base 仍为
  `ABSENT`，不得据此升级完整装配采用或真机权限。

当前审阅导航见 [R2 可视化索引](visuals/HUMAN_TO_ROBOT_R2/README_ZH.md)。13 条最终验收视频均完整解码并与
SHA 收据一致；全量回归为 1489 passed、1 skipped。R2 已封账，本段不撤销下文 V5 的历史失败证据。

状态：本轮已有限封账为 `FAILED_QUALITY_C / INCOMPLETE`，**没有达成去人产品基线**。[最终结果](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/RESULT.json)与[治理收据](../governance/CURRENT_STATUS_RECEIPT.json)是机器事实；[源域路由](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/ROUTE_MANIFEST.json)及各 lane 不可变结果负责来源。计划原截至时间为 2026-09-23 04:33（北京时间）；质量C无授权重试，故提前终态化。

## V5历史：实际结果与当时门槛

已生成并完整核对 11 条全片审阅 MP4（4 Mask、3 传感器、2 R0、2 HuRo），以及4张问题拼图；见[浅层实际视频索引](visuals/FOUR_STREAM_V5/INDEX_ZH.md)。**纯 `robot.mp4` 产品 0/4**，不能用 Raw overlay 替代。

- [Scene四会话终态](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/SCENE_FINAL_ASSESSMENT.json)：设备 Mask 空帧为371/378、142/149、284/284、112/171；缺可信同帧物体保护。ProPainter 未运行，Clean 0帧。旧设备遗漏与时间窗误保护代码已在V5候选修正，但源 Mask 质量未通过。
- [Motion四会话交接](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/MOTION_HANDOFF_V5.json)：四条 HandMotion/R0 来源闭合，数值质量未过；031左侧无合法独立模型输入，103腕容差0/0。
- [支线2结果](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/sensor/run_0001/RESULT.json)：097/098/101共466帧独立回放已解码，视觉贴合待审，不作为HuRo/HaWoR生成依赖。
- [HuRo终态](../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/HURO_TERMINAL.json)：腕目标代码测试通过，但全片候选触发冻结关节限位，未晋升。

下一次执行应先注册**新的有限任务包**，明确新增可信 wearable/cable 删除来源和同帧物体保护证据，再重开 Scene；不能沿用本轮已耗尽的单候选预算扫提示词。新结果通过 Scene 门后，才允许一次冻结 ProPainter、独立 Clean 复核、同版本 R0 合成。031左手要有真实可见代理/独立ROI输入才可修，不从右手镜像或补零。

## 产品合同

唯一 route ID：`HUMAN_TO_ROBOT_BASELINE_V1`。四会话：0915 Chips007 378帧、Poker031 149帧为主产品；0902 Chips103 284帧、Poker042 171帧为同配方回归。四会话源帧、相机域与来源 SHA 各自绑定，0902 sourceIndex0 的矫正域不得混用 0915 sourceIndex1 的 encoded 域。

单入口待各阶段绑定完毕后使用：

```bash
PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run run_human_to_robot_baseline_v1 \
  --input <真实会话目录或其CameraRecord视频> \
  --motion-source hawor \
  --output <全新目录> \
  --config <同会话V5绑定JSON> --dry-run
```

本轮四会话均缺合格 Scene 绑定，旧通用入口实际 `--dry-run` 返回码2／`BLOCKED_PREREQ`，**不得去掉 `--dry-run` 强行执行**。`--config` 必须按 route ID、完整 session ID 和 SHA 绑定 `SCENE_CLEAN_MANIFEST.json` 与 `ROBOT_R0_V1.npz`；缺任一阶段或相机域不符时不能拿 Raw overlay 或黑底骨架冒充 `robot.mp4`。旧通用入口仍未完成安全缓存验收；R2 受限候选入口则已增加内容签名与输出 SHA 双重校验，并在四会话上通过严格 resume。该结论不自动升级旧通用入口。Controller＋MANUS 已形成独立 HandMotion；上述四视觉会话目前不允许静默切换成传感器来源。

数据边界：

```text
Raw ──→ Scene Mask ──→ Clean ───────────┐
  ├──→ Depth/Object/ContactHints ───────┤→ 同一合成器 → robot.mp4
  └──→ HandMotion ──→ Robot R0 [→可选R1]┘
```

Clean 合成像素不得回流到 Depth、Object、Contact、标定或 IK。R0 不等待 Clean；但**产品视频必须消费 Clean**。缺可靠深度时 R1 与深度遮挡保持 UNKNOWN，不阻断有条件的 Clean＋R0 离线视觉候选；UNKNOWN 不能写成未接触或准确遮挡。有效 `c2w` 才能声明 camera→world→固定 robot placement；没有则只称相机相对视觉。0922 的0915 HandMotion `world_valid=false`，不能声称去头漂移。

## 四张短任务卡

- [Scene／Clean／遮挡](V5_SCENE.md)：四会话同一配方；角色与设备独立，可信可见物体只在证据足够时保护；`M_write` 外编码前 byte-exact；质量独立验收。
- [Controller＋MANUS](V5_SENSOR.md)：097／098／101 共466帧；保留 MANUS25 和显式21点映射；安装平移绝对每轴 ±0.16m、原先验残差尺度0.05m；不叠加旧 M1。
- [HaWoR／Robot 产品](V5_MOTION.md)：007／031 首个失效环节、左右真实性、bounded 时序、同资产 R0、同 Scene 合成；旧无效侧不补零冒充观测。
- [HuRo 公平诊断](V5_HURO.md)：有效性屏蔽的核心腕目标；本地 R0 对 HuRo，同 HandMotion／资产／有效帧／Clean；R1另列。

## 验收与标签

`PASSED_INPUT_AUDIT` 仅证明982帧源域冻结；`EXECUTED_PENDING_VISUAL_REVIEW` 仅证明程序与全片视频运行，不证明 Clean、接触或 Robot 姿态正确。最终只有 007／031 **实际消费同版本 Scene**、四会话同配方回归并通过适用质量门，才可称“支持范围内第一版端到端基线”。所有结果 `OFFLINE_VISUAL`、`training_eligible=false`、`control_ground_truth=false`；不外推156条转化率，不训练四支正式 checkpoint。

每个阶段决定立即写不可变结果，最终统一 publisher 更新 `CURRENT_*` 和文档权威映射。浅层可视化目标为 `docs/current/visuals/FOUR_STREAM_V5/` 中实际 MP4 与中文索引；仅完成拷贝并复核 SHA 后才报告本地交付。失败按资源、运行、质量、预算未评估分开终态化，绝不选择更新的文件夹自动替换冻结结果。

## S2 H3 检查点

H3 已完成装配、显示、动作跟踪与遮挡的分开验收。连接件已进入正式 renderer 的 RGB／深度／部件 ID；031 视频忠实消费实际 q/FK，但 full-pose 腕目标仍未跟上。position-only 与两阶段候选均被拒绝：前者牺牲旋转，后者回到原 full-pose 折衷且 85/102 帧触及手臂限位。来源链确认右手 102 帧是非直接 detector ROI 上的 HaWoR 模型输出，`inferred` 不等于 motion infiller；左手仍无 ROI、无模型输出。031 遮挡在固定窗有约82.87%已判归属，17.13% UNKNOWN，连续切换尚无冻结质量门，因此不采用。四条正式候选结构完成，质量通过仍为0/4。

## S2 终态

S2 已完成当前签名入口、完整解码和证据绑定，但四条产品质量通过为0/4。连接件只有视觉几何，031遮挡仍为inconclusive，Contact/Robot R1筛选、执行和采用均为0，尚未建立可用窗口结论；Local/HuRo无独立真值。终态不授予训练、控制或部署权限。
