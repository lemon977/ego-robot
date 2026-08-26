# 最终项目整改任务

状态：`STOPPED_AWAITING_USER_TASK_REVIEW`

版本：`task-plan-v3-final-review`

日期：2026-08-26

## 1. 最终判断

讨论方案总体合理且可行，能够针对现有问题建立真正的项目级约束：004 特调、平行 producer、上下游越权、临时目录被当正式数据、R2 定义漂移以及失败后跨层修图。

落地时冻结以下修正：

1. RAW 恢复包含“外部挂载可用”和“项目内逐项验证”两步；需要平台权限时应 HOLD，不能用本地副本冒充权威 RAW。
2. 004 的旧 MASK/CLEAN 只能作为冻结诊断基线；若存在真人残留或物体破坏，必须先归因 `MASK` 或 `CLEAN_BACKGROUND`。
3. `EXACT78_R2_VISUAL_DOMAIN` 与“78/78 均通过”可能真实冲突。R2 自身造成不可接受穿模时必须 HOLD，不能为凑数修改 R2。
4. exact78 的 78 sessions / 32,318 frames 是固定 source scope；主 matched 对照使用 RAW/Robot 共同的 `PAIRED_KEPT` frame/window 交集，并公开覆盖率与剔除原因。
5. 004、025、149 是验收角色，不是硬编码参数来源。004 建金标准；025 和未参与调参的 149 用同一代码/profile 盲跑。
6. harmonization 只能在几何 PASS 后修改 robot RGB 与窄边界带，不能改 alpha、depth、object、background 或几何。
7. 速度对齐属于主对照之后的独立 matched ablation，不能混入图像域主变量。

## 2. 冻结目标与禁止项

### `004_CONTACT_GOLD`

允许在新版本 namespace 中增加 Object6D refinement、metric geometry/SDF、contact-aware `q_hand`、接触/非穿透约束和 compositor 修正。原 R2 及其 SHA 永久保留，不允许覆盖或回写 exact78。

### `EXACT78_R2_VISUAL_DOMAIN`

逐数组冻结 `q_hand`、`wrist_T_camera`、validity、confidence、failure reason、frame/timestamp、左右手顺序、单位、split、语言、state 与全部非图像输入。唯一计划变量是：

```text
RAW image selector -> manifest-authenticated FINAL RobotRGB selector
```

禁止 `if session_id/frame_id`、逐帧阈值、隐藏 fallback、移动物体/手腕、render-time IK、隐藏 mesh/link/手指、从 legacy 或 RAW 补齐正式结果，以及让 QA 修改上游数据。

## 3. 固定执行阶段

### G0 — RAW 与冻结基线

- 恢复并只读校验 `/mnt/data/egodata`，至少覆盖 004/025/149：路径、普通文件/链接、bytes、SHA、分辨率、FPS、帧序、时间戳和 RAW 身份。
- 重验 exact78 R2、split 和 checkpoint SHA；任何漂移立即 STOP。
- 输出唯一任务卡；不运行正式 producer。

通过条件：权威 RAW 可逐帧读取且 source manifest 完整。否则 `HOLD_RAW_UNAVAILABLE`。

### G1 — 架构与合同

建立并审核：

```text
architecture/COMPONENT_OWNERSHIP_v1.md
architecture/DATA_REFERENCE_GRAPH_v1.json
contracts/pipeline_contract_v1.yaml
contracts/project_profile_v1.yaml
contracts/session_context.schema.json
contracts/route_decision.schema.json
contracts/failure_attribution.schema.json
```

明确 MASK、CLEAN、Object6D、retarget、base/IK、renderer、compositor、harmonizer、QA 的唯一读写边界。profile 生命周期为 `DRAFT -> CALIBRATION_LOCKED -> FROZEN`。

### G2 — 004 接触金标准

1. 冻结并验证 004 RAW/MASK/CLEAN。
2. 自动加人工复核 stress frames：接触前、首次接触、稳定抓握、最大遮挡、释放、快速运动，合计 12–24 帧。
3. 先输出 `HAND_ONLY_DIAGNOSTIC`：完整手 mesh、物体层、逐 link depth/ID、物体 SDF、接触/非穿透指标、推荐合成和 thumb-only 对比。
4. 根据证据，每轮只选择 `OBJECT_GEOMETRY / RETARGET_QHAND / COMPOSITOR` 一个分支修改；上游 MASK/CLEAN 失败则退回相应 owner。
5. 再完成 session-constant Tianji base search、q_arm/adapter、`HAND_ARM_FINAL` 与四栏视频。

停止门：真人无残留、物体无破坏、拇指和其他手指遮挡正确、无可见穿模/闪烁、机器人未覆盖区 CLEAN 自然。通过后冻结 004 金标准；未通过则 HOLD，不降低门槛。

### G3 — 通用主干与盲测

- 一个代码主干、一个项目 profile、自动 session context、有限 route registry。
- 所有尺寸、膨胀、接触带和时间窗口按分辨率、手尺度、前臂宽度、物体投影尺度与 FPS 归一化。
- SAM3 只提供 `H_core / O_visible_core / U_contact` 证据；手后应露出物体的区域走 Object6D/CAD/解析几何与 texture donor，露出背景的区域才走真实时序 donor/atlas/ProPainter。
- 每条 session 生成 12–24 个 canary，并根据手臂面积、接触比例、运动、模糊、Object6D 有效率和 donor 覆盖输出：`PASS_FULL_SESSION / RETRY_STANDARD_ROUTE / HOLD_UNSUPPORTED`。
- 完全相同代码/profile 依次盲跑 025、149 和保留 train canary。失败只允许修改通用算法语义；禁止单 session 补丁。

全部通过后才冻结项目 profile。

### G4 — exact78 生产

固定顺序：`train62 -> validation8 -> 冻结 pipeline/profile/checkpoint -> test8 一次性生产和评测`。

每条通过的 session 必须从 `_run` 原子晋级，并包含唯一 `SESSION_MANIFEST.json`、文件 SHA、质量报告与 `PREVIEW.mp4`。失败进入 HOLD；不得静默补齐。

### G5 — matched 对照

- 生成唯一 `PAIRED_KEPT_MANIFEST.json`，RAW 与 Robot 使用相同 split、R2、非图像输入、初始化、seed、batch、epoch、validation 选择规则和 frame/window 集。
- 报告五组：RAW 主对照、RobotRGB 主对照、paired-kept 覆盖、physical-valid 子集、可选 speed-matched ablation。
- 验证 checkpoint 谱系与所有冻结 digest；test8 不参与调参或 checkpoint 选择。

## 4. 分层质量门

- L0 数据完整性：文件、维度、帧序、SHA、selector、manifest。
- L1 像素/区域：真人残留、物体破坏、边界、机器人覆盖、颜色异常。
- L2 几何/遮挡：per-link depth/ID、contact、SDF penetration、物体前后关系。
- L3 时序/session：闪烁、跳变、base 稳定、donor 覆盖、失败聚类。
- L4 人工/可选语义筛查：固定 stress frames 和四栏视频；VLM 只能辅助，不能替代 L0–L3。

## 5. 每轮任务卡合同

每次交给 AI 的任务卡必须写明：唯一假设、冻结输入及 SHA、允许修改目录、禁止修改项、唯一 owner/分支、预期输出证据、通过/HOLD 条件和回滚条件。

每轮 QA 只能生成 `failure_attribution.json` 与下一版建议。未经下一张显式任务卡，不能重跑 producer、自动修复或覆盖上游正式结果。

## 6. 审核后的第一张任务卡

用户明确开工后，仅执行 G0：检查 RAW 挂载和权威性、重验冻结 SHA、建立组件/数据引用图与六份合同草案。不得在同一任务卡启动 MASK/CLEAN/render/train。
