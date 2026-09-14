# 组件所有权合同 v1

状态：`DRAFT`

合同版本：`pipeline-contract-v1`

本文件定义组件边界。v9 独立 QA 已判 `DIAGNOSTIC_HOLD`；当前 MASK 启发式路线止于 v9，不授权 v10 heuristic。新的 MASK 运行必须等待 H/O/U/B benchmark 与 `auditor_a1` 冻结，因而 G2 calibration、正式生产、CLEAN、晋级、训练和评测全部 fail-closed。

## 1. 全局不变量

1. `/mnt/data/egodata` 是唯一权威 RAW，永远只读；本地副本、legacy、`_run` 和旧 adapter 路径均不能替代它。
2. 每种 artifact 只有一个 producer。consumer 只能读取 manifest 显式引用且 SHA256 已验证的普通文件。
3. 所有实验写入 `_run/<run_id>/`；只有 `promotion_manager` 能将完整 session 以同文件系统原子 rename 晋级为 `processed/<session>_vN/`。
4. `004_CONTACT_GOLD` 与 `EXACT78_R2_VISUAL_DOMAIN` 使用隔离 namespace，互相不能读取另一条产物线的可变输出。
5. `EXACT78_R2_VISUAL_DOMAIN` 冻结 R2 的 `q_hand / wrist_T_camera / validity / confidence / failure_reason / frame_order / timestamp / split / non_image_inputs`，只允许最终图像 selector 变化。
6. `HAND_ONLY_DIAGNOSTIC` 仅供诊断；正式 RobotRGB 只能来自 `HAND_ARM_FINAL`。
7. route 只能取合同登记的有限集合，禁止 `session_id/frame_id` 分支、逐帧阈值、未登记 fallback 和跨 owner 自动修复。
8. QA 只能写 QA namespace 中的归因、报告和建议，不能写 producer artifact，也不能触发重跑或晋级。
9. 任一必需引用、digest、模型、权重或证据缺失即 HOLD；不得猜测、插值或继承 004 常数。G2 任务卡可携带 SHA 绑定的无量纲 calibration overlay，但未审核值只能生成 `G2_CALIBRATION_CANDIDATE`，不能产生正式 PASS。
10. G0/G1 只执行机器一致性、schema、digest 与合同闭环校验，不设置常规人工视觉审核门。首次常规人工审核只能发生在 G2 的 004 RAW/MASK/CLEAN stress frames 与完整预览完成后。
11. 校准权限与正式权限独立：`calibration_execution_authorized=true` 只覆盖 004 MASK/CLEAN review staging；`formal_production_allowed=false`，不得写 `processed/`。
12. `SYNTHETIC_TEST_ONLY` 只能用于单元测试，不能作为 G2 review、正式 CLEAN、晋级或正式 consumer 的输入；execution mode、artifact state 与 authorization ref 必须相互一致。
13. reveal identity 必须同时记录路由前 object/background/protected-object overlap 和路由后互斥分配、coverage、HOLD accounting；无法解释的像素一律 `HOLD_UNSUPPORTED`。
14. MASK producer 与 MASK auditor 分别版本化。producer 产物必须绑定 `producer_pN`、像素语义 SHA 和实现 SHA；auditor 产物必须绑定 `auditor_a1`、阈值/profile SHA、实现 SHA 和冻结 benchmark SHA。
15. 历史 v3、v8、v9 的覆盖范围和 auditor 不同，只能作法证证据，禁止按历史“通过帧数”排名。统一比较必须在同一冻结 `auditor_a1`、同一 H/O/U/B benchmark、同一 object protection 和同一指标定义下重评。
16. blind benchmark 标签禁止进入 MASK producer。几何/PCA/Object6D/时序可以提供提示或审计，但不得 fill/union/hard-crop 生成最终人体像素。

## 2. 唯一组件边界

| 组件 ID | 唯一职责与输入 | 唯一可写 artifact | 禁止行为 |
|---|---|---|---|
| `source_resolver` | 只读权威 RAW；验证 source manifest、普通文件、bytes、SHA、帧序、时间戳、分辨率和 FPS | `verified_source_manifest` | 写 RAW；从 legacy/缓存补齐；把链接或 staging 当正式帧 |
| `session_profiler` | 读取 `verified_source_manifest` 与 RAW metadata | `session_context`、`canary_set` | 写像素；填造缺失测量；按 session ID 特调 |
| `mask_benchmark_curator` | 读取权威 RAW、冻结 source manifest、自动分层选择和人工 H/O/U/B 标注 | `mask_benchmark` | 生成候选 MASK；调 producer/auditor；向 producer 暴露 blind 标签；无人审即冻结 |
| `mask_producer` | 读取已验证 RAW 与 `session_context` | `mask_evidence`（`H_core`、arm/forearm evidence、`U_contact` 的 mask 侧证据） | 修物体；写 CLEAN；直接决定 donor；覆盖 RAW |
| `object_geometry_producer` | 读取 RAW、锁定的 Object6D/CAD/解析几何引用与 context | `object_evidence`、`object_geometry`、`object_depth_sdf`、`object_texture_donor` | 删手；写背景；修改 q_hand/wrist；伪造 metric scale；在 004 MASK/CLEAN 人审批准前 refinement |
| `reveal_policy_owner` | 读取 project profile、显式任务卡 overlay 与未来 formal authorization | `verified_reveal_policy` | 启用未登记 route；接受绝对像素阈值；把 synthetic 标成 formal；在 profile 未冻结或授权缺失时开启正式 CLEAN |
| `temporal_donor_atlas` | 读取已验证 RAW、context、MASK、object identity 与 `verified_reveal_policy`，按实测 FPS 建时序背景 atlas | `temporal_atlas_manifest` | 写 CLEAN；纳入被 MASK 的 source；覆盖受保护物体；固定帧窗口；隐藏 inpaint fallback |
| `independent_mask_qa` | 只读 source、canary、冻结 MASK 与冻结 H/O/U/B benchmark；使用 SHA 绑定的 `auditor_a1` | `mask_independent_qa` | 写/修 MASK；调用 MASK producer；修改阈值；跨 auditor/覆盖范围排名；benchmark 未冻结时 PASS |
| `reveal_router` | 读取 H/O/U、几何、donor coverage 与项目 profile | `route_decision`、`reveal_labels` | 直接补图；调用未登记 route；静默 fallback |
| `clean_background_producer` | 读取 RAW、`reveal_labels`、已验证 object/background donor | `clean_layers`、`clean_coverage` | 渲染机器人；用背景覆盖应露出的物体；修改 object geometry |
| `mask_clean_review_packager` | 只读 004 已冻结 RAW/MASK/CLEAN、canary/stress frames、`mask_independent_qa=PASS` 与全部 digest | `mask_clean_review_package`（stress 对比和完整预览视频） | 信任未绑定布尔值；修 MASK/CLEAN；调用 Object6D refinement、retarget、renderer、compositor 或 final robot |
| `retarget_producer` | CONTACT_GOLD：读取冻结 human motion、robot model、contact/SDF；EXACT78：只验证并解析冻结 R2 | `resolved_retarget`；CONTACT_GOLD 独占 `contact_gold_sidecar` | 覆盖 R2；写图像；EXACT78 refinement；render-time 改 q_hand |
| `base_ik_producer` | 读取冻结 wrist、robot model、场景几何 | `base_ik_solution`（session-constant base、q_arm、adapter） | 移动 wrist；按帧移动 base；不可达时隐藏 link |
| `renderer` | 读取冻结 pose/camera、robot/object geometry、资产 digest | `render_layers`（完整 robot/object RGBA、per-link ID/depth、object depth） | 写 CLEAN；改变 pose；隐藏失败手指/link；用 thumb-only 作正式输入 |
| `compositor` | 读取通过 L0 前置检查的 CLEAN、render depth/ID 与 reveal labels；执行确定性合成前几何门 | `geometry_composite`、`compositor_diagnostics`、`geometry_gate_result` | 改 geometry/q；按 link 名单作弊遮挡；写 harmonized 外观；把未校准门判为 PASS |
| `harmonizer` | 仅读取 compositor 产生且可审计的 `geometry_gate_result=PASS` 与 composite；由 alpha/profile 推导窄带审计 mask | `final_robot_rgb`、`harmonization_audit` | 改 alpha/depth/object/background；扩大到未授权区域；掩盖几何失败 |
| `qa_evaluator` | 只读全部 manifest、证据、diagnostics 与 final image | `quality_report`、`failure_attribution`、`next_revision_advice` | 修改上游；自动重跑；降低阈值；批准自己的未审人工门 |
| `promotion_manager` | 只读 staging、合同/schema、QA 结论及全部 digest | `session_manifest`、原子晋级目录、`cohort_manifest` | 复制式发布；覆盖版本；晋级 HOLD/不完整目录；从 RAW/legacy 补文件 |

`qa_evaluator` 对 L0–L4 只拥有判定文件，不拥有任何被判定数据。`promotion_manager` 只拥有发布动作与发布 manifest，不拥有内容生成。

## 3. Artifact 唯一所有权

| Artifact type | 唯一 producer | 正式 consumer |
|---|---|---|
| `verified_source_manifest` | `source_resolver` | profiler、MASK、Object、CLEAN、release audit |
| `session_context`, `canary_set` | `session_profiler` | 所有下游 producer、QA |
| `mask_benchmark` | `mask_benchmark_curator` | `independent_mask_qa`（只读；blind 标签不得进入 producer） |
| `mask_evidence` | `mask_producer` | reveal router、QA |
| `object_evidence`, `object_geometry`, `object_depth_sdf`, `object_texture_donor` | `object_geometry_producer` | reveal router、retarget、renderer、compositor、QA |
| `verified_reveal_policy` | `reveal_policy_owner` | reveal router、temporal atlas、CLEAN、review packager、QA |
| `temporal_atlas_manifest` | `temporal_donor_atlas` | reveal router、CLEAN、QA |
| `mask_independent_qa` | `independent_mask_qa` | review packager、QA（只读） |
| `route_decision`, `reveal_labels` | `reveal_router` | CLEAN、QA |
| `clean_layers`, `clean_coverage` | `clean_background_producer` | compositor、QA |
| `mask_clean_review_package` | `mask_clean_review_packager` | 人工审核者、QA（只读） |
| `resolved_retarget`, `contact_gold_sidecar` | `retarget_producer` | base/IK、renderer、QA |
| `base_ik_solution` | `base_ik_producer` | renderer、QA |
| `render_layers` | `renderer` | compositor、QA |
| `geometry_composite`, `compositor_diagnostics`, `geometry_gate_result` | `compositor` | harmonizer、QA |
| `final_robot_rgb`, `harmonization_audit` | `harmonizer` | QA、promotion、formal selector |
| `quality_report`, `failure_attribution`, `next_revision_advice` | `qa_evaluator` | human review、promotion（仅 PASS 结论） |
| `session_manifest`, `cohort_manifest` | `promotion_manager` | formal selector、training/evaluation |

任何新增 artifact type 必须先修改合同并经人工审核；禁止用新文件名绕开现有 owner。

## 3.1 MASK 实验身份与统一比较

当前证据覆盖不等价：v3 是 004 全 460 帧 MASK 候选，v8 是 4 帧 forearm diagnostic，v9 是 228–242 的 15 帧 diagnostic。三者还使用不同阶段的门和 auditor，因此历史数字不得横向排序。

后续唯一允许的比较协议为：

```text
producer_pN + pixel-semantics SHA
  -> frozen H/O/U/B benchmark (45–60 frames)
  -> frozen auditor_a1 + threshold/profile SHA
  -> human miss / object corruption / background over-mask /
     boundary excluding U / temporal flip
```

benchmark 必须包含 004 development、004 同 session blind、跨 session blind 三部分，各 15–20 帧；blind 标签对 MASK producer 不可见。benchmark 或 `auditor_a1` 任一尚未冻结，结论只能是 `NOT_COMPARABLE/HOLD`。v9 是当前启发式路线终点；若统一 benchmark 仍失败，下一步只能是受控 backbone A/B 或 `HOLD_UNSUPPORTED`，不能增加 v10 heuristic。

G2 首个人审前所需 artifact 已分别绑定唯一 schema，包括 `verified_source_manifest`、`canary_set`、`mask_evidence`、`verified_reveal_policy`、`object_texture_donor`、`temporal_atlas_manifest`、`route_decision`、`reveal_labels`、`clean_layers`、`clean_coverage`、`mask_independent_qa` 与 `mask_clean_review_package`。其余 `UNDEFINED_DRAFT` schema 继续阻止正式生产；顶层 `formal_production_allowed=false` 与 `formal_clean_enabled=false` 不因 schema 就绪而改变。

这段前置图无循环依赖：policy 先于 donor；object donor 与 temporal atlas 只进入 reveal router；reveal labels 再进入 CLEAN；CLEAN/coverage 与独立 MASK QA 只进入 review packager。CLEAN 输出不回流 donor producer，review/QA 也不能回写任一上游 producer。

## 4. 产物线和诊断轨道隔离

| 维度 | `004_CONTACT_GOLD` | `EXACT78_R2_VISUAL_DOMAIN` |
|---|---|---|
| Retarget 输入 | 原始证据的独立 refinement namespace | SHA 锁定的 R2，只读 |
| 可写 sidecar | `contact_gold_sidecar/vN` | 无；只生成解析/验证视图 |
| 可改 q_hand | 仅显式 `RETARGET_QHAND` 单分支任务卡 | 禁止 |
| 可改 Object6D/SDF | 仅显式 `OBJECT_GEOMETRY` 单分支任务卡 | 可生成图像所需证据，但不得改变冻结非图像输入 |
| 正式输出 namespace | `_run/.../004_CONTACT_GOLD/...` | `_run/.../EXACT78_R2_VISUAL_DOMAIN/...` |
| 跨线读取 | 禁止读取 exact78 可变产物 | 禁止读取 004 refinement/gold 输出 |

两条轨道均可输出：

- `HAND_ONLY_DIAGNOSTIC`：完整手 mesh、物体层、per-link ID/depth、SDF/contact、thumb-only 对比；永不进入 RobotRGB selector。
- `HAND_ARM_FINAL`：完整 Tianji 机械臂与 KaiHand；只有此轨道的 `final_robot_rgb` 可晋级和进入 selector。

## 5. 有限路由与单分支修改

合同登记的 route ID 只有：

```text
OBJECT6D_GEOMETRY_TEXTURE
CAD_GEOMETRY_TEXTURE
ANALYTIC_GEOMETRY_TEXTURE
TEMPORAL_DONOR_ATLAS
PROPAINTER_BACKGROUND
```

前三者只服务 `REVEAL_OBJECT`，后两者只服务 `REVEAL_BACKGROUND`。route 的模型、权重、参数、阈值或必要证据未锁定时，该 route 必须保持 disabled/HOLD。一次 route decision 必须显式列出选择依据和未选原因，不能按顺序捕获异常后自动切换。

G2 校准只允许显式启用 `ANALYTIC_GEOMETRY_TEXTURE` 与 `TEMPORAL_DONOR_ATLAS`；它们的未审核无量纲阈值必须来自当前任务卡 overlay，并且不能进入正式 consumer。其他 route 均保持关闭。

MASK 模型身份锁定为 `SAM2_1_HIERA_LARGE`，代码来源是 Grounded-SAM-2 固定提交，权重和配置均由 SHA256 绑定。SAM3 当前 `UNAVAILABLE`，任何产物不得把 SAM2 输出标记为 SAM3。

004 的迭代任务卡每轮只能指定一个修改 owner：`OBJECT_GEOMETRY`、`RETARGET_QHAND` 或 `COMPOSITOR`。如果证据落在 `MASK` 或 `CLEAN_BACKGROUND`，必须结束当前轮并创建新任务卡，不能跨层顺手修复。

## 6. L0–L4 门和人工等待

G0/G1 是机器一致性门，不是常规人工视觉审核门。合同文件仍需项目所有者批准，但这不等于视觉质量审核，也不得伪造视觉 PASS。

G2 的首个人工视觉门固定为：

```text
004 RAW/MASK/CLEAN 已冻结并通过机器校验
-> 自动 stress frames + 完整 RAW/MASK/CLEAN 预览打包
-> AWAITING_HUMAN_MASK_CLEAN_REVIEW
-> 人工 APPROVED 后才允许后续几何/机器人阶段
```

处于 `AWAITING_HUMAN_MASK_CLEAN_REVIEW` 或 `REJECTED` 时，禁止启动 004 Object6D refinement、retarget、base/IK、renderer、compositor、harmonizer 或任何 final robot 输出。此时可以推进不依赖该人工结论的合同校验、代码逻辑审计和其他只读机器任务。

| Gate | 最小职责 | 失败处置 |
|---|---|---|
| L0 | 文件、schema、bytes、SHA、维度、帧序、selector、manifest、RAW authority | HOLD；禁止任何高层掩盖 |
| L1 | 真人残留、物体破坏、边界、机器人覆盖、颜色异常 | 单 owner 归因；不自动回写 |
| L2 | per-link depth/ID、contact、SDF penetration、物体/机器人前后关系 | 几何未 PASS 禁止 harmonizer |
| L3 | 闪烁、跳变、session-constant base、donor 覆盖、失败聚类 | 标准 route RETRY 或 HOLD |
| L4 | 固定 stress frames、四栏视频与必要人工审核；VLM 仅辅助 | 标记 `manual_review_required` 并等待人工；可推进不依赖该结论的其他任务 |

Gate 只能按 L0 → L4 前进。人工门未完成时，相关 session 和依赖它的冻结/晋级任务必须等待，不得将“未审核”解释成 PASS。

## 7. 晋级协议

`promotion_manager` 必须按以下顺序执行，任一步失败均保留 `_run` 并 HOLD：

1. staging 关闭所有写句柄；验证任务卡、product line、track 和 run identity。
2. 验证所有 schema、普通文件、bytes、SHA256、输入引用、producer 和 frame count。
3. 验证 L0–L4；人工门要求为真时必须存在独立人工批准引用。
4. fsync 文件与 staging 目录。
5. 确认目标 `processed/<session>_vN/` 不存在且在同一文件系统。
6. 单次原子 rename；fsync `processed/` 父目录。
7. 生成/更新由 digest 绑定的 cohort manifest。不得修改已经发布的 session 版本。

正式 consumer 只读 `cohort_manifest -> session_manifest -> artifact` 引用链。链上任一节点缺失或 digest 不符即失败，不回退 `_run`、RAW、legacy 或其他版本。
