# Human→Robot S2 H3 外部复核交接

生成时间：2026-09-23 H3  
任务：`human_to_robot_evidence_unlock_s2_20260923`  
权威边界：本文只汇总 H3 已有证据，不使新算法或产品获得采用权。

## 一、当前确定结论

1. **执行链已经接通，但产品质量没有通过。**
   - 007：378/378 帧，结构通过，质量拒绝。
   - 031：149/149 帧，结构通过，质量拒绝。
   - 0902_103：284/284 帧，结构通过，质量拒绝。
   - 0902_042：171/171 帧，结构通过，质量拒绝。
   - 因此当前是 `4/4 structural candidates, 0/4 quality pass, 0/4 adopted`。

2. **renderer 没有偷偷把手放到目标腕位置。**
   - 031 中 renderer FK 与保存的实际 FK 差异 P95 约 `3.74e-13 mm`。
   - producer FK 与保存的实际 FK 差异 P95 约 `2.45e-13 mm`。
   - 画面中的 Robot 忠实消费实际 `q/FK`；当前大残差是求解/目标问题，不是显示器伪造对齐。

3. **031 full-pose 腕目标没有被 Robot 有效跟上。**
   - 位置残差 P50/P95 约 `77.50/104.39 mm`。
   - position-only 可以把位置 P50/P95 降至近 `0/4.56 mm`，但旋转 P50/P95 恶化为 `63.86/104.73°`。
   - 两阶段候选位置仍为 `77.41/104.39 mm`，旋转为 `30.04/39.83°`，`0/102` 帧通过 pose gate，`85/102` 帧触及 arm limit。
   - 两个候选均被拒绝；不能用 position-only 的小位置误差冒充完整动作恢复。

4. **031 腕目标旋转不具备外部真值权限。**
   - 旋转来自 HaWoR 预测 MANO21 手掌基构造，并与 KaiHand neutral palm 对齐。
   - 它不是直接解剖 wrist orientation 测量，也不是外部 GT。
   - placement 是固定虚拟相机/基座锚点，不是实测 world→robot 标定。
   - 所以当前证据只能说“冻结 full-pose 目标与关节约束冲突或目标方向不可靠”，不能说“真实物理不可达”。

5. **031 的 `inferred` 语义已经追溯清楚。**
   - 右手在 frame 47–148 有 102 帧 HaWoR 模型输出，但 ROI 不是 direct detector 支持。
   - `inferred = predicted_valid AND NOT ROI det_direct`，不代表 motion infiller 生成。
   - 该输出是 `OFFLINE_NONCAUSAL`，不是直接 3D 观测。
   - 左手没有 ROI，没有模型输出，也没有被镜像或填充。

6. **连接件已进入真实 renderer，但连接件碰撞尚未验证。**
   - STEP 派生网格已进入 RGB、Robot depth 和 component ID，并按实际 q/FK 运动。
   - 007 中 Local/HuRo 分别有 314/318 帧可见连接件。
   - 031 冻结视角中为 0 可见像素，证据表明是视外/遮挡，不是资产没有消费。
   - 当前只有视觉几何；`collision_scope=UNVERIFIED_VISUAL_GEOMETRY_ONLY`。

7. **031 遮挡已经真实执行，但尚不能通过质量门。**
   - 固定窗中已判定重叠像素：`SCENE_FRONT=1890`，`ROBOT_FRONT=4009`，`UNKNOWN=1219`。
   - known coverage 约 `82.87%`，UNKNOWN 约 `17.13%`。
   - 连续可比较像素 3269 中出现 736 次归属切换；当前没有冻结的时序切换质量门，因此必须保持 `INCONCLUSIVE`。
   - 3 mm 只是排序容差，不是 3 mm 配准精度证明。

8. **007 Scene 只完成了有界局部改善。**
   - 16 帧局部修复是真实 ProPainter 消费，写入区外及保护区修改为 0。
   - 大块手/设备删除有改善，但长细线缆仍残留，全片设备/保护语义仍未闭合。
   - 不支持直接扩成全片采用。

9. **007 Depth 门保持拒绝。**
   - 固定 12 帧只有 1021 个 robust matches，冻结门为 1500。
   - 不降门、不换好帧，不授权 FoundationStereo/Depth successor。
   - 因此 007 遮挡与 Contact 保持 UNKNOWN。

10. **Contact / Robot R1 当前没有合法窗口。**
    - `0 screened / 0 executed / 0 adopted`。
    - 031 受 full-pose/目标权限限制；007 和 0902 没有合格 Depth。
    - 本轮不应伪造 Contact 或通过移动腕来做出贴合。

11. **Sensor 和 Local/HuRo 只能保留限定结论。**
    - Sensor 097/098/101 共 466 帧共同后端和视频已重新核对；没有重新拟合。
    - 098/101 最多为 kinematic-only；097 视觉贴合仍 inconclusive。
    - Local/HuRo 同 target 数值证据已复核，但没有独立真值，不能宣布胜者。

## 二、已排除或禁止继续试的方向

- 不把 68.4 mm 与 75.2 mm 的 6.8 mm 差异当成 77–104 mm 腕残差的唯一根因。
- 不用 position-only 小误差覆盖大旋转错误。
- 不把 renderer 视觉放置错误当成当前 031 主因。
- 不将 `inferred` 等同 motion infiller，也不把左手填出来。
- 不把连接件视觉加载升级成连接件碰撞 PASS。
- 不把 031 的 UNKNOWN 遮挡像素算成正确。
- 不降低 007 Stereo 1500 match 门，不临时换帧。
- 不扩大 007 Clean canary，除非冻结窗口的独立视觉复核支持。
- 不用 Local/HuRo 各自的输入证明自己更准。
- 不为了触发 Robot R1 而伪造 Depth、Contact 或实测 mount。

## 三、请其他 AI 重点回答的问题

### Q1：031 full-pose 目标应该如何重构？

现有 rotation 是从单目 HaWoR MANO21 手掌基构造，无外部 wrist orientation 真值。请判断下一个有界实验应优先是：

1. 只使用可观测的 wrist position，将 orientation 降为软先验；
2. 改用 palm normal / finger directions 的部分方向约束；
3. 先做明确的 fixed-base 可达集投影，再解手指；
4. 其他更可证伪的有界方案。

请给出明确的目标函数、冻结量、可变量和拒绝门，避免再次通过调权重凑画面。

### Q2：如何区分“目标 orientation 错”和“fixed-base 真不可达”？

希望得到一个不依赖外部标定的开发级讨论，包括：

- 用什么可行性问题判定 position 是否可达；
- orientation 仅部分可观测时如何表达可观测子空间；
- 如何在不移动冻结 base/mount 的条件下生成有证据的拒绝码。

### Q3：031 遮挡的 736/3269 连续归属切换该怎样判读？

请先区分真实穿越、物体边界移动、Robot 动作和深度/注册抖动，再建议一个固定的连续窗质量指标。不希望用“永久保持上一帧”掩盖问题。

### Q4：007 长细线缆还值得做有界修复吗？

已知大块手/设备删除有改善，但长细线缆留下。请给出：

- 如何用局部锨点、细长度、端点和时序轨迹定义 top-1 实例；
- 什么拒绝条件表明应该停止，而不是继续扩 mask；
- 这项修复是否值得在本 S2 剩余时间内执行。

### Q5：连接件碰撞覆盖如何最小化落地？

当前只有真实视觉网格。请说明在不伪造完整动力学模型、不新增碰撞忽略对的情况下，是否应该：

- 将批准的简化几何只用于非邻接碰撞诊断；
- 或者保持 `UNVERIFIED_VISUAL_GEOMETRY_ONLY` 并将它留到下一个有资产授权的任务。

### Q6：当前是否还有证据支持 H6 前开新的全片算法运行？

当前倾向是“没有”，原因是：

- 031 两个有界 IK 候选均被拒绝；
- 007 Clean 仍有长线缆，且全片语义未闭合；
- 007 Stereo 仍是 1021/1500；
- Contact/R1 合法窗口为 0。

如果认为还有一个值得运行的有界实验，请指明它会新增什么可区分证据，而不是只生成另一个画面。

### Q7：Local/HuRo 在没有外部真值时，最少应该保留哪些结论？

请明确区分：

- 求解器对同一 target 的残差；
- 各自覆盖率与失败模式；
- 限位/碰撞/时序平滑度；
- 哪些数字不能被解读为人手或真实机器人精度。

### Q8：H6/H9 应该冻结哪些组件，哪些只应该保留为失败候选？

请按以下四类回答：

```text
模块技术冻结
会话结构候选
会话质量通过
产品采用
```

不要因为结构完成就把 0/4 质量通过改写成基线成功。

## 四、核心证据路径

- H3 机器快照：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/checkpoints/H3_RESULT.json`
- 当前视频索引：`/mnt/workspace/code/chaoyang/docs/current/visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md`
- 031 腕残差：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/h3_motion_audit_031/attempt_0002/RESULT.json`
- 031 来源追溯：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/source_provenance_031/attempt_0001/RESULT.json`
- 031 目标权限：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/target_authority_031/attempt_0001/RESULT.json`
- 031 两阶段候选：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/two_stage_031/attempt_0001/RESULT.json`
- 031 遮挡：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json`
- 007 Stereo：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/encoded_stereo_preflight_v1/get_potato_chips_0915_007/RESULT.json`
- 007 Clean canary：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json`
- Sensor 复核：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/sensor/reuse_audit_v1/RESULT.json`
- Local/HuRo 复核：`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/compare/reuse_audit_v1/RESULT.json`

## 五、当前权限和测试边界

```text
execution = EXECUTED
structure = PASS
quality = REJECTED_QUALITY_MIXED_INCONCLUSIVE
adoption = NOT_ADOPTED

training_eligible = false
control_ground_truth = false
physical_deployable = false
external_metric_authority = false
```

治理状态：`PASS / FRESH`，revision `13939`。  
项目内合法测试组：1490 tests，1489 passed，1 skipped；专用碰撞 fixture 6/6 passed。  
14 个 hardlink transaction tests 因 CPFS 项目内 TMPDIR 不具备所需本地 hardlink 语义而未评估；因此不宣称“完整测试全通过”。
