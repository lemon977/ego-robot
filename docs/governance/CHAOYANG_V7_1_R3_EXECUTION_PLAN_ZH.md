# Chaoyang 全链优化执行计划 V7.1-R3

> 状态：`CURRENT`。本计划是执行合同，不是完成证明；实时数量、任务状态和 authority 只从 `CURRENT_STATUS_RECEIPT.json` 绑定的机器文件读取。

## 1. 目标与事实边界

执行链：

```text
Depth → Mask → Clean → Contact → Robot → Occlusion → Visual Aux
```

并行覆盖 exact78、新传感器线、四支 future-2D Visual Aux、治理和 current-only 清理。阶段完成定义为分母内任务进入唯一终态，不要求全部通过。

固定事实边界：

- 正式 H4 为 `BLOCKED_RESOURCE / NOT_EVALUATED / POLICY_DEFERRED`，没有像素 Mask authority；attempt_0003/0004 的质量 C 只属于开发 canary。
- 当前 Mask 基线为 SAM3.1；SAM2.1/Cutie 只可作为 challenger。
- FoundationStereo、HaWoR、Controller、Object6D 和数字 Robot 碰撞均不是外部物理真值。
- `visual_robot_trajectory_sidecar.control_ground_truth=false`；Visual Aux 不是 Robot policy。

## 2. 治理和不可变发布

`MIGRATE-00` 必须通过 CAS 发布文档权威、算法合同、迁移收据、计划 revision 和 current receipt。Worker 只写 immutable attempt，只有 aggregator 可写 current。

质量门分为：

```text
HARD_STRUCTURAL
DOWNSTREAM_ELIGIBILITY
SOFT_DIAGNOSTIC
EXTERNAL_AUTHORITY
```

严格模仿人手姿态属于 `SOFT_DIAGNOSTIC`，不得单独阻止 Robot Geometry。所有任务固定输出 `RESULT.json`、`ARTIFACT_MANIFEST.json`、`METRICS.json`、`RUN_RECEIPT.json`、`DECISION.md` 和 `NEXT_ACTION.json`。

## 3. 执行 DAG

```text
MIGRATE-00 + RECOVER-V74
        ├─ DEPTH-00 → DEPTH-10 → DEPTH-20
        ├─ MASK-ROLE + MASK-OBJECT → CLEAN-B/C
        ├─ Raw/Object evidence → ATLAS
        ├─ Causal history → DONOR
        ├─ CLEAN-C + DONOR → CLEAN-20
        ├─ CLEAN-20 + Poker Atlas → CLEAN-21
        ├─ Depth/Object/Mask → CONTACT-10
        ├─ ROBOT-V75 → ROBOT-TARGET-10 → ROBOT-REACH-20 → ROBOT-PROFILE-30
        └─ Robot + Clean + Contact → OCCLUSION → Robotized RGB → Visual Aux
```

### 3.1 Depth

- `DEPTH-00` 审计 selected-left、rectified-left、鱼眼/MP4 图像域、K、畸变、rectification、baseline、R/t、单位、optical-Z、registration 和 camera/world/robot 坐标方向。同设备同采集合同才能复用标定。
- `DEPTH-10` 输出 disparity/valid、左右一致性、registration、静态漂移、边缘/低纹理/反光/运动模糊风险和 `depth_quality_evidence`；明确 `depth_confidence_present=false`、`external_metric_accuracy=UNKNOWN`。
- `DEPTH-20` 以 Controller 为 wrist SE(3) 主锚、MANUS25 为手指、HaWoR 为可见性诊断，Stereo 仅作可见表面有界 optical-Z 修正。固定最近15帧中位数、30 mm innovation 拒绝、Stereo权重0.15、EMA 0.15、单帧修正±5 mm；无效时保持 Controller wrist。

### 3.2 Mask

- `MASK-ROLE` 使用 SAM3.1 维护左右手/前臂/Tracker/Controller 身份；离屏为空、重入恢复、左右不交换。Robot link Mask 来自 renderer part-ID。
- `MASK-OBJECT` 对 Poker 物理牌和 Chips 三实例独立维护；盘子/桌面不得混入，禁止 union 掩盖身份错误。
- H4 只再允许一次有界 SAM3.1 多模态提示 canary；失败只关闭该 Mask→Clean 路线，不阻塞 Controller/MANUS/Tactile/Depth。

### 3.3 Clean

```text
Raw/Object → Atlas
Causal history → Donor
Mask/Flow → Clean-B/C
Clean-C + Donor → Clean-20
Clean-20 + Poker Atlas → Clean-21
```

`M_remove`、`M_flow`、`M_write` 必须分开；保护当前可见物体；donor 必须 `donor_frame_id<=target_frame_id`；Atlas 只用身份和 pose 验证通过的真实像素；ProPainter 只补剩余洞。正式输出30 FPS并保存 source-frame 映射。

门限：contact-band 额外删除相对旧基线减少至少50%，人手残留和非人泄漏不退化，可见物体保留≥99.9%，`M_write` 外 byte-exact。

### 3.4 Contact

`CONTACT-10` 按手/指腹/物体实例输出 `APPROACH|TOUCH_CANDIDATE|SLIDE_CANDIDATE|RELEASE|UNKNOWN`、表面距离、相对速度、slip、uncertainty、valid 和证据父节点。

```text
DIRECT_OBJECT6D / independent tracking
→ CONTACT_SEED
→ stable hypothesis
→ HAND_OBJECT_ATTACHMENT
→ occluded pose propagation
```

Attachment 不得反向证明 Contact、Object6D 或 tactile-supported contact。无独立真值时保持 `HYPOTHESIS_ONLY / external_accuracy=UNKNOWN`。

### 3.5 Robot

- `RECOVER-V74` 依据死亡 PID、attempt 日志和失败收据封为 `FAILED_RUNTIME_FINAL`；已完整结果只按 SHA adopt。
- `ROBOT-V75` 修复 finalizer schema 与任意 cwd import，不覆盖 v74。
- `ROBOT-TARGET-10` 冻结 human camera/world→object-relative wrist→robot-base→flange→KaiHand root。
- `ROBOT-REACH-20` 保留相对物体或首帧的手腕位移，默认尺度1.0；仅因 workspace/限位投影最近可行点，不为姿态相似度压缩轨迹。
- `ROBOT-PROFILE-30` 输出各阶段耗时、P50/P95、合格帧/小时、clipping 和 fallback。

Robot terminal status 与 tier 分开；硬门只包含 finite、proper rotation、坐标闭包、IK、限位和基于当前 URDF/mesh 的数字碰撞。姿态相似度仅软诊断。

### 3.6 Occlusion 与 Visual Aux

全部 Robot link 和物体进入统一 z-buffer，每像素保存 depth、instance/part/link/triangle id 和 pixel source。对象外观只来自当前 Raw、合法 causal donor、pose 验证 atlas 或可信 renderer；否则 `UNKNOWN`。

Silver 只检查 provenance/coverage/unknown/z-buffer/保护率，不报告 accuracy；Gold 需要冻结独立标注。

四支 checkpoint 按 Chips、Poker 两个 pair 独立训练，仅监督 H50 `future_2d_xy/future_2d_valid`。Raw/Robotized 必须帧、标签、split、seed 和 valid mask 完全一致且因果。每支输出 best/last、loss CSV/PNG、ADE/FDE/PCK、identity、coverage、smoothness、配置/代码/数据 SHA 和推理视频。

## 4. 调度、终态和验收

GPU 优先级：恢复/短 canary > regression > Clean successor > Robot batch > checkpoint。每任务最多3次运行时 attempt；质量失败不自动重试。研究失败必须有限收敛到明确终态。

最终至少发布 Depth 审计/QA/手腕融合、Role/Object Mask 矩阵、Clean 矩阵、Contact index、Robot 矩阵与 reach/profile、Occlusion Silver、四支 checkpoint index、exact78 156行矩阵、文档权威/算法合同/迁移收据、current 状态和会议快照。

任何内部一致性、视觉审核或数字几何残差不得升级成外部物理精度、真实动作或部署 authority。
