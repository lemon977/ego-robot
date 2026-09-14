# MASQUERADE 对接触、遮挡与 HumanEgo 训练的可复用结论

状态：`DEVELOPMENT_EVIDENCE`  
研究日期：2026-09-13  
适用范围：exact78 V3 的研究与工程设计说明，不是数据 authority、接触真值、Robot authority 或物理部署授权。

## 1. 原始来源与版本边界

本文只使用作者发布的原始来源：

- [Masquerade 官方项目页](https://masquerade-robot.github.io/)
- [Masquerade 论文 arXiv:2508.09976v1](https://arxiv.org/html/2508.09976v1)，论文页面标注后续发表于 ICRA 2026
- [作者官方代码仓库 MarionLepert/phantom](https://github.com/MarionLepert/phantom)
- 本次逐行核对的官方代码 commit：[`a8bb81c1bbe6ade129a1f6f0906482f510354a5e`](https://github.com/MarionLepert/phantom/tree/a8bb81c1bbe6ade129a1f6f0906482f510354a5e)

未用二手博客、复述文章或搜索摘要补齐论文未说明的参数。尤其需要注意：公开仓库包含视频处理流水线，但没有论文中的完整 Masquerade policy/co-training 实现；因此不能从该仓库反推出论文没有明确给出的训练 batch 配比或 future horizon 数值。

## 2. 原始方法实际做了什么

Masquerade 的核心不是接触真值恢复，而是缩小视觉 embodiment gap：从野外第一视角 RGB 视频估计双手、去除人体、叠加跟随估计轨迹的双臂 Robot，然后用 edited human video 预训练视觉编码器，再和真实 Robot demonstration 联合训练 policy。官方项目页对流水线和训练关系有简洁说明，[论文 Method](https://arxiv.org/html/2508.09976v1#S3)给出完整定义。

### 2.1 RGB-only 是 Masquerade 的正式 EPIC 配置

官方 README 明确区分 Phantom 与 Masquerade：Masquerade 输入是可能遮挡的一手或双手 RGB 视频，单目深度方向约有 3–4 cm 误差，human video 只提供 2D projected waypoint 辅助监督，[不是完整 3D Robot action](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/README.md#L182-L192)。

官方 EPIC 配置将 [`depth_for_overlay: false`](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/configs/epic.yaml#L18-L31) 固定为关闭。RGB-only overlay 实现直接以 Robot/gripper mask 覆盖输入图像，[没有物体深度或前后关系判断](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/robotinpaint_processor.py#L478-L512)。

因此，论文中展示的 Masquerade 结果能够证明“不完美的视觉 robotization 对其训练设定有帮助”，不能证明 RGB-only overlay 在接触区具有正确遮挡关系。

### 2.2 官方代码存在可选 depth overlay，但不是 contact-aware compositor

代码根据 `depth_for_overlay` 在 RGB-only 和 depth-aware 两条路径之间切换，[开关位置见官方实现](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/robotinpaint_processor.py#L269-L280)。可选 depth 路径比较真实场景 depth 与 simulated Robot depth；当真实表面更近时隐藏 Robot，但显式排除了膨胀后的 human-hand mask 区域，[接触手区不会由同一规则解决](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/robotinpaint_processor.py#L514-L564)，[最终 overlay mask 条件见此](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/robotinpaint_processor.py#L597-L620)。

这条可选分支提供了一个可复用的最小思想：Robot z-buffer 必须与场景深度比较。但它没有提供：

- 人手接触意图或遮挡期物体 pose hypothesis；
- 接触区 Robot/Object 谁在前的独立决策；
- 深度 valid、质量证据和不确定性合同；
- 被人手挡住的物体 RGB 从哪里恢复；
- `UNKNOWN`、coverage、连续未知帧或 paired-frame drop；
- 物体 identity、Raw pixel protection 或 temporal donor 验证。

论文自己将这一点列为限制：没有 depth 时无法可靠判断 Robot 应在物体前还是后，可能错误覆盖场景物体；快速运动、严重遮挡和不适配的灵巧抓取也会导致帧被丢弃，[见 Limitations](https://arxiv.org/html/2508.09976v1#S5)。

### 2.3 遮挡手的 carry-forward 不能当接触推断

论文附录说明：单手消失后复用最后可见 action 到 episode 结束；整段不可见则给固定 out-of-frame action，两手都缺失则丢帧，[见 Additional data-editing details](https://arxiv.org/html/2508.09976v1#A1.SS3)。官方代码也确实实现了无最大 gap 的 last-valid carry-forward 和整段 neutral action，[见 `_refine_actions`](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/action_processor.py#L286-L354)。

该策略是在大规模 noisy video 中维持 overlay 连续性的工程折衷。它没有观测物体、不输出接触状态、不验证手物相对变换，也不区分三个薯片实例，所以不能复用为 exact78 的 contact truth 或 formal Object6D 补帧。

### 2.4 Edited-human future 2D 是辅助目标，不是真实 Robot action

Masquerade 将平滑后的 end-effector 位置投影到图像，预测未来 `H` 个 2D waypoint；为补偿头戴相机运动，将未来帧的点通过 homography warp 回当前帧视角，[见标签定义](https://arxiv.org/html/2508.09976v1#S3.SS2.SSS2)。论文明确说这些 2D 标签只用于视觉编码器 auxiliary loss，不直接进入 policy action learning。

联合训练时：

- edited human batch 继续优化 `L_2D`；
- 真实 Robot demonstration 以真实 Cartesian end-effector action 优化 `L_policy`；
- 总损失为 `L_2D + λ L_policy`，[见 Policy learning](https://arxiv.org/html/2508.09976v1#S3.SS3)；
- 论文报告每任务 50 条真实双臂 Robot demonstrations，且在候选值中 `λ=10` 最好，[见训练附录](https://arxiv.org/html/2508.09976v1#A1.SS1)。

论文的 overlay ablation 和 co-training ablation 都显示移除对应部分会明显退化；其结论支持“edited visual input 与真实 Robot action 联合训练值得验证”，但不允许使用 human 2D waypoint 冒充真实 Robot action。

本项目 V3 的 `H=50`、`[T,50,2,2]` 和 human:robot batch `1:1` 是本地冻结合同。论文正文只把 horizon 写成符号 `H`，公开代码也没有完整 policy loader，因此这几个具体值不能标为“复现 Masquerade 官方参数”。`λ=10` 则有论文附录直接支持，但仍需在本项目数据域独立验证。

## 3. 对三个 V1 模块的逐项映射

| Masquerade 原始做法或限制 | exact78 可复用部分 | 不可直接复用部分 | V1 落点 |
|---|---|---|---|
| 先估计手，再 retarget，再 render | 模块顺序与责任分离 | render 结果不能反过来生成接触标签 | `HUMAN_CONTACT_HYPOTHESIS_V1` 独立于 Robot；retarget 只接受该 sidecar |
| 单手暂时消失时 carry-forward | 需要显式处理 tracking gap | 无上限 carry-forward 会长期伪造 pose/contact | hypothesis 只允许 ≤15 帧且 ≤0.5 s，超限 `UNKNOWN` |
| 单目估计可用于粗略视觉 overlay | 视觉辅助信号可以不完美 | 不能升级为外部 3D/contact truth | 每项输出标 `pose_source`、`hypothesis=true`、衰减 confidence；formal Object6D 不改写 |
| hand keypoint → Robot EE，之后平滑 | 分阶段 retarget 和 tracking failure gate | 论文使用 parallel-jaw gripper，不能直接证明 KaiHand 灵巧接触可解 | 固定六阶段 solver、chirality/关节/结构/坐标链硬门、task-specific 数字几何门 |
| RGB-only Robot mask 直接覆盖 | 可作为低保真 development baseline | 无法用于需要正确牌/薯片遮挡的 authority | `OCCLUSION_COMPOSITOR_V1` 不接受“只看 mask 即覆盖”作为已知判定 |
| 可选 real depth 与 Robot depth 比较 | z-buffer 前后比较的基本形式 | human-hand 区被排除；无 valid/quality/provenance/UNKNOWN | 显式 `depth_valid`、quality evidence、ownership、合法 object RGB provenance |
| imperfect overlay 仍可提升训练 | 可先做可控视觉 A/B | 训练收益不能倒推单帧遮挡正确或物理精度 | compositor accuracy 与 HumanEgo policy 指标分开报告 |
| future 2D + homography | edited-human visual auxiliary 目标 | 不能当真实 Robot action | HumanEgo Aux 与 Policy checkpoint 分名，invalid future step 有 valid mask |
| edited human 与真实 Robot co-training | 保留 auxiliary loss、防止视觉表征遗忘 | 无真实 action 时不能训练最终 policy | Real Robot action readiness 是正式 policy 的外部门 |

### 3.1 `HUMAN_CONTACT_HYPOTHESIS_V1`

已实现文件：

- `pipeline/human_contact_hypothesis_v1.py`
- `contracts/human_contact_hypothesis_v1.schema.json`

Masquerade 没有独立的人—物接触推断模块。本项目只能借鉴“遮挡需要显式时序策略”，不能复制其无限 carry-forward。V1 使用两类旁路假设：双向刚体短 gap 与手物 attachment；正式 Object6D 始终 `DIRECT_OBSERVED_ONLY + KEEP_INVALID`，输入/输出 SHA 检查保证不被修改。

重要 authority 边界：Grade A/B、`KEEP_INVALID`、身份分离的 formal Object6D 即使不具备 `robot_contact_authorized`，仍可作为 `HYPOTHESIS_ONLY` 诊断输入；它不能因此变成 contact truth，也不能反写 Object6D authority。

### 3.2 `CONTACT_AWARE_ROBOT_RETARGET_V1`

已实现文件：

- `pipeline/contact_aware_robot_retarget_v1.py`
- `contracts/contact_aware_robot_retarget_v1.schema.json`

可复用的是“手估计与 Robot 求解分开”“tracking error 失败就跳过”“时序平滑在初始 retarget 后执行”。不可复用的是 Masquerade 对 parallel-jaw gripper 的直接映射和缺失帧无限复用。KaiHand 的接触、碰撞、结构闭包与左右手性需要本项目自己的硬门。

六阶段顺序固定为 wrist/arm IK、human-pose hand retarget、contact finger refinement、collision cleanup、temporal refinement、final audit。每阶段最多 2 次初始化、200 次迭代、单帧 30 秒；全 canary 最长 1800 秒。失败仅保留最佳诊断解，`authority_promotable=false`。

### 3.3 `OCCLUSION_COMPOSITOR_V1`

已实现文件：

- `pipeline/occlusion_compositor_v1.py`
- `contracts/occlusion_compositor_v1.schema.json`

Masquerade 可选 depth overlay 的 z 比较是这一模块的起点，不是完整实现。V1 进一步要求：

- FoundationStereo 只有 `depth_m`、`depth_valid` 和质量证据；`depth_confidence_present=false`；
- object pixel 只能来自当前 Raw 可见像素、验证过的 temporal donor 或可信纹理 renderer；
- 没有外观证据时必须 `TIE_UNKNOWN`，不得用 Clean 桌面冒充物体；
- `accuracy_on_known` 必须与 coverage/UNKNOWN 指标同时过门；
- compositor 不向 contact hypothesis 或 retarget 反馈控制信号。

## 4. 通用性与失败边界

Masquerade 的实验证据来自 EPIC-KITCHENS、双 Kinova Gen3、Robotiq 2F-85、固定 Robot 相机和三类双臂厨房任务，[硬件与数据域见论文附录](https://arxiv.org/html/2508.09976v1#A1.SS4)。从该证据可以合理迁移的是流程假设：视觉 embodiment editing、未来 2D auxiliary 和真实 Robot co-training 值得实验。

不能直接迁移的结论包括：

- 对扑克牌薄几何或三袋薯片实例仍能正确恢复接触；
- KaiHand 指腹接触、非穿透或 wrist/adapter 闭包已解决；
- RGB-only overlay 的物体遮挡正确；
- 数字接触距离等于真实物理误差；
- Masquerade 的 OOD 成功率可预测 exact78 或本项目真机成功率；
- `λ=10`、Horizon 或 batch ratio 在本项目仍最优。

以下情况必须 fail closed：

1. formal Object6D 不是 A/B、不是 `KEEP_INVALID`、身份有 union/switch，或者长 gap/前后不一致；
2. Robot asset 只有 identity pin，没有执行授权、真实装配闭包或 camera/world→base 坐标链；
3. 只有 object amodal geometry，没有合法 object RGB；
4. 深度无效、注册失败、遮挡边缘/低纹理/局部一致性证据不通过；
5. 通过大量 `UNKNOWN` 获得虚高 known accuracy；
6. 只有 edited-human 2D 标签，没有同步真实 Robot action。

## 5. 四关键帧 Go/No-Go 合同

四关键帧是诊断升级门，不是“必须凑出四帧”的产量目标。frame ID 必须由显式 selection artifact 冻结；缺失时不得按经验或目录顺序编造。建议覆盖 approach、touch、stable grasp/manipulation、release 四类相位，但某类没有可观测证据时应 No-Go，不得用另一相位替代。

三道门独立判断：

```text
human_contact_gate
  只检查 HaWoR + Object Identity Mask + formal Object6D

retarget_gate
  检查 hypothesis sidecar + Robot执行资产/装配闭包 + 坐标链 + 显式4帧selection
  不依赖 Clean 或 Robot render

compositor_gate
  检查 Clean + Depth + Mask + Object pose/hypothesis + Robot render + object RGB provenance
  不反控前两道门
```

历史的独立门闭包诊断保留在：

`tasks/control/runs/20260913_contact_robot_v1_canary_v2/INPUT_CLOSURE_AND_4FRAME_GO_NO_GO.json`

Poker042 当前开发结果已经推进到 fresh v3：

- 全片 `HUMAN_CONTACT_HYPOTHESIS_V1`：`tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json`。171帧中 formal Object6D 直接观测73帧，128–131四帧满足短gap双向刚体门，剩余94帧保持 `UNKNOWN`；正式Object6D文件和数组的前后SHA一致。数字SDF只支持 `HYPOTHESIS_ONLY`，不支持物理接触真值。
- 四帧selection已经显式冻结为99、100、104、114；这四帧都来自 direct observed Object6D。选择覆盖两段数字 `TOUCH` evidence，不伪装成完整的 approach/grasp/release 四相位真值。
- Robot资产中Tianji CAD和左右KaiHand真实URDF/Mesh的SHA闭包通过，HaWoR/Object6D内部camera→world关系闭合；但Robot asset pin仍只是 `T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION`，且缺NaturalV2法兰→KaiHand确认装配变换、adapter CAD或安装测量、world→Tianji base、TCP/安装标定。因此retarget仍是精确No-Go，solver未运行。
- 中文四关键帧诊断：`tasks/control/runs/20260913_contact_robot_v1_canary_v3_visual_v1/poker/play_cards_0902_042/RESULT.json`。画面只含Raw、HaWoR、observed对象Mask/Object6D和数字状态，逐帧标注 `HYPOTHESIS_ONLY`、`NOT_CONTACT_TRUTH`、`ROBOT_SOLVER_NOT_RUN` 与五个外部硬阻塞；没有绘制Robot或Robot proxy。人工目视检查了2560×1800的2×2原图，并从80帧短视频解码抽查frame 114面板：中文、单位、颜色图例和五项阻塞均可读，叠加层没有发现明显错帧或裁切；这仍只是视觉诊断通过，不是接触或Robot质量门通过。
- Clean缺失和Robot render缺失仍只阻挡compositor，不反向阻挡已完成的contact hypothesis，也不是retarget No-Go的理由。

当前其余结果：

- `get_potato_chips_0902_034`：不在 Wave0，当前没有满足本门的 formal Object6D，因此 human contact 本身即 `BLOCKED_PREREQ`；Role Mask C 同时阻挡后续 Clean/compositor，但不是用来反向否决一个已经闭包的 contact hypothesis。

本次没有选择虚构frame ID、没有执行solver、没有生成Robot render，也没有晋升任何authority。

## 6. 实现与测试映射

| 边界 | 回归测试 |
|---|---|
| 正式 Object6D 不变，短双向 gap 才能成为旁路 hypothesis | `test_contact_hypothesis_fills_short_two_sided_gap_without_mutating_object6d` |
| 长/单边 gap 保持 UNKNOWN | `test_contact_hypothesis_keeps_long_or_one_sided_gaps_unknown` |
| Chips 不允许 union identity | `test_contact_hypothesis_rejects_chips_union_identity` |
| attachment 检查手物相对漂移而非全局移动 | `test_attachment_hypothesis_uses_relative_drift_not_global_object_travel` |
| Retarget 固定六阶段且绝不自动晋升 | `test_retarget_runs_exact_stage_order_and_never_auto_promotes` |
| 左右手交换、预算越界、错误 sidecar fail closed | `test_retarget_preserves_best_diagnostic_but_fails_chirality_swap`、`test_retarget_budget_cannot_exceed_frozen_limits`、`test_retarget_rejects_compositor_or_untyped_contact_input` |
| Poker/Chips 使用不同数字 penetration 门 | `test_task_specific_penetration_gate_is_stricter_for_poker` |
| object RGB provenance 优先级固定 | `test_object_pixel_provenance_has_frozen_precedence` |
| 无合法外观的 OBJECT_FRONT 转 UNKNOWN 且训练 invalid | `test_object_front_without_appearance_becomes_unknown_and_training_invalid` |
| 不伪造 FoundationStereo confidence | `test_depth_quality_is_evidence_not_native_confidence`、`test_occlusion_schema_rejects_invented_foundationstereo_confidence` |
| 全 UNKNOWN 不能靠准确率绕过门 | `test_unknown_coverage_cannot_pass_by_abstaining_everywhere` |
| known accuracy 与 Raw protected retention 联合过门 | `test_known_accurate_frames_pass_only_with_protected_raw_retention` |
| Clean/render 只阻挡 compositor，不循环阻挡 hypothesis/retarget 合同 | `test_clean_and_render_only_block_compositor_not_upstream_modules` |
| Canary no-clobber 且永不晋升 | `test_real_canary_closure_is_fail_closed_and_never_promotes`、`test_canary_diagnostic_is_no_clobber` |

## 7. 可支持与不可支持的表述

可以支持：

- Masquerade 的主要贡献是 edited-human visual pretraining 与真实 Robot action co-training，而不是接触真值恢复。
- 官方 Masquerade EPIC 配置是 RGB-only；代码中的 depth overlay 是可选路径。
- 官方论文明确承认单目深度、重遮挡、快速运动、前后关系和 dexterous retarget 的局限。
- exact78 的三模块分离是对这些限制的工程化扩展，不是对官方代码的逐行复刻。

不能支持：

- Masquerade 已解决接触区物体遮挡。
- 可选 depth overlay 等价于本项目 contact-aware compositor。
- carry-forward action 可以成为 formal Object6D 或 contact truth。
- human future 2D label 可以替代真实 Robot action。
- 当前 Chips034/Poker042 已通过四帧 Robot canary。
- 当前任一 V1 数字残差具有真实物理精度 authority。
