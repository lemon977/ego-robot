# 实时使用算法审计：0915 第二轮质量恢复

版本：v2.1。状态：TERMINAL_REJECTED_NO_RECOVERY。
报告运行：`0915-robot-quality-recovery-v21-20260919T234412+0800`；唯一起始时间
`2026-09-19T23:44:12+08:00`，最晚截止时间 `2026-09-20T14:44:12+08:00`。
全部已启动 package 已先行终态，因此协调器依据任务包的
`all_started_packages_terminal` 条件于 `2026-09-20T10:57:51+08:00` 正常退出；
没有空转到最晚截止时间，也没有降低质量门。
当前治理状态必须由 `CURRENT_STATUS_RECEIPT.json` 及校验器读取；本文只记录
已落盘的里程碑，不替代实时心跳。
配套：[任务书](01_EXECUTION_PLAN_15H_ZH.md) · [启动指令](00_START_HERE_ZH.md)。

## 1. 本文只记录“真正运行了什么”

事实来源固定为现有治理receipt、算法合同、不可变RESULT、输入/技术合同快照、实际PID/租约。运行快照放 `_run/current`，每十分钟更新；此Markdown只在启动、算法采用/拒绝、波次结束和最终发布时更新。

不新增一套authority注册系统，不允许worker各自编辑current文档。publisher从机器文件生成聚合视图，未知为null/UNKNOWN，不从计划或聊天推断PASS。

## 2. 顶部状态卡

| 字段 | 初始化值 | 真实来源 |
|---|---|---|
| snapshot_source | GOVERNED_PARENT_RECEIPT_AND_IMMUTABLE_CHILD_RESULTS | 本地receipt读取 |
| reported_governance | 12362 / gov-012362-8d04bae5bef9 | 启动前用户报告，仅保留为起点 |
| effective_governance / actual_head | 12435 PASS/FRESH / a4f371ab… | 2026-09-20 00:49 核验；后续以机器receipt为准 |
| reported_branch | task/0915-full-funnel-0916-clean-v1 | 用户报告 |
| active_parent / active_children | 0915_robot_quality_recovery_15h_v2 / B1（已注册，启动前独立审计） | 当前任务索引 |
| started_at / deadline_at | 2026-09-19 23:44:12 +08:00 / 2026-09-20 14:44:12 +08:00 | 本轮真实启动事件 |
| elapsed_s / remaining_s | null / null | monotonic计时 |
| GPU PID / owner / lease | null / null / null | 进程与租约交叉检查 |
| current_model_entrypoint | null | 真实进程命令及代码SHA |
| last_actual_progress_at | null | 处理帧/任务结果 |
| manifest_sha | null | 冻结12条及最多4替补清单 |
| last_adopted_algorithm | null | adoption receipt |
| state | TERMINAL_REJECTED_NO_RECOVERY | 全部已启动 package 终态；候选 adoption REJECT |

V2.1 还必须显示：W0/W1-DIAG/W1-ADOPTION/H9/EXTRA FINAL 各自的访问状态，W1-ADOPTION 首次打开前的 candidate freeze SHA，H9 每项能力是否 `ADOPTED` 或 `NO_ADOPTED_FIX`，以及 EXTRA FINAL 是否 `SEALED`、`OPENED` 或 `NOT_OPENED_EXPANSION_GATE_FAILED`。不得把“文件已存在”推断成已打开数据。

如果十分钟快照无法读取，标STALE，不从上一次绿色状态推断仍在运行。操作记录中必须有真实任务ID/PID，而非“后台处理中”的自然语言承诺。

## 3. 本轮算法身份表

| 能力 | 起点为用户报告 | 本轮必须补的实际字段 |
|---|---|---|
| encoded adapter | 已固定裁切/resize/镜像回域 | 输入尺寸/eye index/K/P来源、代码与配置SHA |
| HaWoR raw/bounded | W0严格0/4 | raw与bounded各入口、失败阶段、拟采用修复 |
| HaWoR consumer admission | 待新审计 | 旧门结果、同口径新结果、用途窗口结果三列 |
| SAM3.1 | 部分可用，W1未跑 | 权重SHA、prompt来源、state/reseed/物理ID映射 |
| FoundationStereo | 12/12内部门通过 | 复用缓存实际SHA、本轮重算次数与原因 |
| Object geometry | W0两个Poker可用 | 每实例/字段合法范围，Chips表面路由 |
| Human/Stereo alignment | 原尺度触边拒绝 | s具体作用对象、对应语义、独立split、未决原因 |
| Strict Contact | 无连续窗 | 每道门候选数量，表面/投影足点语义 |
| Local Stereo metric | 待新审计 | `local_stereo_metric_dev` 与 `external_metric_authority` 分列；R1-E 开发许可 |
| Temporal/event hypothesis | 上轮R1-H未采用 | 旧拒绝原因、本轮新证据、反证/留出验证 |
| R0 | 四导出，严格0 | 自身质量vs继承阻塞、真实q/wrist有效范围 |
| R1-E / R1-H | 无采用 | 真实solver尝试、相同样本before/after、采用原因 |
| Virtual R2 | 四质量拒绝+合同漂移 | 技术snapshot SHA、变换链、IK/碰撞失败分类 |
| Batch consumer dispatcher | 旧W1除Depth外阻塞 | 运行门与消费门解耦测试、逐能力就绪情况 |

同一能力只选一个当前采用版本。候选、拒绝、旧版本可审计，但不是多个“current/latest/final”并行执行入口。

## 4. 每项算法的最少必填字段

```text
capability_id
selected_version / candidate_version
entrypoint / real_command
code_sha / git_commit / config_sha / environment_sha
weights_path / weights_sha （纯非学习后处理才为ABSENT）
input_manifest_sha / upstream_artifact_shas
technical_contract_snapshot_sha / asset_shas
coordinate_frame / units / dt_semantics / evidence_type
old_strict_gate_result / same_gate_new_result / consumer_window_result
implementation_status / actual_execution_status / quality_status
allowed_consumers / disallowed_consumers / authority_scope
fit_split / validation_split / source_group_independence
thresholds_with_origin / fixed_denominators
raw_measurements / coverage / first_blocker / all_rejection_bits
adoption_or_rejection_reason / receipt_path / rollback_target
actual_cpu_s / actual_gpu_s / peak_memory / processed_frames
```

`code_done`不是`real_video_run`；`exported`不是`quality_pass`；`quality_pass`不是`physical_deployable`。

## 5. W0失败矩阵：禁止只写0/4

每个会话至少两手分别列出：

| 项目 | 必填内容 |
|---|---|
| 固定分母 | 总帧数、时长、expected side-frames、实际obs/未知 |
| 原始检测 | 非空数、crop/索引错、身份不明、视野外 |
| raw MANO | 数值/骨段/左右手/坐标异常 |
| bounded | 更新饱和、重投影回退、抖动改善与真实动作损失 |
| 质量聚合 | 哪些规则是frame/window/session，是否跨gap计算 |
| R0自身 | joint顺序、限位、FK、碰撞、时序、与human方向差 |
| 整片拒绝 | 原strict逐项原因，不能因新scope覆盖 |
| 局部可用 | 合法窗口起止、hand_id、消费者、有效秒数 |
| 修复类型 | ALGORITHM / IMPLEMENTATION / GATE_SEMANTICS / DEPENDENCY_ONLY |
| 实验结果 | 固定样本前后数值/视频；没有真值不写精度提高百分比 |

原因位图可以重叠；first_blocker仅供排序，不能隐藏后续错误。展示应同时有“只要任一门失败”和“每道门独立失败”统计。

## 6. 依赖审计表

```text
consumer
old_dependency
actual_minimum_input_requirement
proof_or_unit_test
new_run_permission
new_consumption_scope
unchanged_quality_constraints
negative_controls_passed
registered_contract_version
```

必查四条：

- W0严格0/4是否错误阻止W1合法输入上的诊断推理。
- Object SAM是否错误要求全会话HaWoR strict 3D。
- R0 q22是否误继承world scale/Contact失败。
- 虚拟R2是否误要求R1成功；同时是否忽略R0窗口自身质量。

依赖修复不会自动把原严格拒绝转成通过。新增用途准入必须有独立字段、测试和用途声明。

## 7. Contact可满足性与来源漏斗

按 `(session, hand_id, finger_id, object_id)` 列出每一门；同时保存unique surface sample数与pair-row数，防重复计量。

```text
all_pair_rows
observed_human_rows
finger_identity_known
valid_hand_semantic
stereo_surface_sample_valid
surface_role_known
object_identity_known
object_patch_valid
projection_footpoint_inside_patch
uncertainty_admitted
distance_under_existing_threshold
fixed_pair_temporal_window_admitted
```

给每个被拒候选显示：采样uv、hand mask、object mask、对应surface XYZ、正交足点、有限patch、plane_normal、distance、不确定度/未知项。

特别记录 `same_pixel_object_membership_required` 是否真实存在；不存在则不能宣称找到了布尔矛盾。真实接触面不可见引起的unknown，和代码错误引起的空集，要分别计数。

严格阈值仍读当前5mm合同；不得用10mm、倍增uncertainty或估计皮套厚度代替。hypothesis/event可用性另列，既不冒充概率也不加到strict_contact_count。

每个窗口另存 `LOCAL_STEREO_METRIC_DEV_V1`：同帧同 encoded Depth、physical-left 回域、K/P/baseline/镜像/Depth SHA、直接可见 hand surface、object identity、finite patch、LR consistency、局部深度、注册与不确定度逐项布尔值。只有全部为真才允许 `local_stereo_metric_dev=true` 和 `r1_e_development_allowed=true`；`external_metric_authority` 永远不因该字段升级。记录 `alignment_consumed_contact_window=false`，防止循环自证。

## 8. 对齐专项

每一版必须回答：scale缩放何物；offset在哪个方向/坐标；K/P/B实际来源；是否处理主点/镜像/resize；可见表面与MANO是否同一物理表面。

按左右手、skin/equipment、图像半径、距离、视角、时间块输出残差、触边比例、参数稳定性。跨源组验证不成立时明写，不把同一001中的相邻20帧称20次独立验证。

旧0.783730、0.8、19.33mm保留为旧拟合结果。新候选即使更小残差，也要报告覆盖变化及独立误差，不能把拟合目标下降当公制真值认证。

## 9. 遮挡与时序专项

记录 direct/derived/noncausal/assumed_proxy 四类依赖。重播种模型状态、历史surface搬运、短gap插值、任务约束恢复各自分开。

回放验证必须声明屏蔽了哪些输入及其全部派生缓存，哪些未来信息合法允许、哪些被留作评估。时序HaWoR或SLAM已消费隐藏帧而未重新隔离时，标 `LEAKAGE_UNRESOLVED`，只能做诊断。

至少含no-contact替代解释。共动可能来自相机运动，触觉可能无法确定对象；不能仅凭两者之一认证card身份/接触点。真实遮挡重现只验证重现可见状态。

上轮R1-H拒绝原因、本轮修改、独立验证差异必须并排。没有差异则不重跑。

## 10. Robot专项

每层分别写 R0 / R1-E / R1-H / Virtual R2，必须绑定：

- q22关节顺序、左右URDF、原生mesh、pad/link与安装snapshot SHA。
- wrist坐标、变换方向、camera/world用途、时间戳。
- q22/wrist/arm各自validity，不默认绑定成一个valid布尔值。
- 限位、非邻接自碰撞、可见有限面相交、速度/加速度及边界连续性。
- R1相同样本before/after、切向逃逸、prior偏离、valid coverage。
- R2前向复核法兰与手根是否同一个目标定义、虚拟base是否会话固定。
- 真实隐藏物体/环境碰撞是否未验证，物理部署一律false。

恢复原合同bytes只修复证据闭包，不修复IK。IK误差降低也不自动修复技术快照丢失。

## 11. 同一统计口径的实时批次表

从真实RESULT聚合，不从目录数量聚合：

```text
session_id, original_wave, task, source_group, group_independence
access_tier, access_state, first_opened_at, candidate_freeze_sha
frame_count, duration_s, input_signature
hawor_attempted, hawor_completed, hawor_original_strict_quality
sam_object_attempted, sam_object_admitted, identity_known_frames
depth_cache_reused, depth_new_attempted, object_fields_valid
r0_exported, r0_strict_quality_admitted
r0_local_windows, r0_usable_side_frames, r0_usable_seconds
r1_e_attempted, r1_e_adopted, r1_h_attempted, r1_h_adopted
r2_attempted, r2_admitted
first_blocker, all_failure_bits, budget_stop_reason
actual_cpu_s, actual_gpu_s, result_sha, review_path
```

`group_independence` 的 V2.1 唯一允许值为 `METADATA_VERIFIED_NO_USER_ATTESTATION` 或 `UNKNOWN`；不得从旧 manifest 复制用户逐会话确认声明。

原主体12条与H0封存的EXTRA FINAL 4条分开列；其余204条单独报。不按每根手指/视频片段放大session数，不把局部窗口准入当whole-session准入。

## 12. 单张实验卡

```text
experiment_id / capability / parent_version
observed_failure / evidence_paths
specific_hypothesis / proposed_change
unchanged_inputs_and_thresholds
fit_dataset / validation_dataset / independence
negative_controls / fixed_evaluation_denominator
registered_before_run / maximum_attempts / budget
actual_command / code_config_asset_shas
actual_results / coverage_change / limitations
decision = ADOPT | REJECT | INCONCLUSIVE
allowed_scope / forbidden_claims / rollback
```

阈值来自原合同就注明；时间、最多两次修复、最多四条扩展属于本轮预算，不写成科研标准。单位/范围错误纠正是可验证工程修复，不是偷偷降低质量门。

## 13. 每次里程碑必须先回答的四个问题

1. 本轮比起点新增了几个真实推理会话、几个严格合格Robot会话、多少局部可用秒数？
2. 哪个真实错误被修复？若只有调度/审计修复，请明确不是算法质量提升。
3. 哪个依赖当前仍在阻塞？它是输入缺失、实现未完成、质量不通过、还是错误范围的门？
4. 哪项下一任务已就绪而尚未执行？是否有GPU/CPU/时间原因，还是scheduler错误地全局停止？

## 14. 封账模板

```text
实际T0/结束/耗时：
实际HEAD/治理：
主清单12条SHA/额外验证清单：
W0四会话原strict -> 同口径新strict：
W1-DIAG / W1-ADOPTION / H9 HOLDOUT各层实际推理/完成/strict/R0结果：
R0严格合格会话数：
R0仅局部可用会话/窗口/秒数：
R1-E采用 / R1-H采用 / R2采用：
Object新增字段覆盖 / 复用Depth数 / 新Depth推理数：
HaWoR/SAM实际质量改进：
Contact可满足性结论 / 隐藏面不可观测范围：
尺度原因已验证 / 未决：
独立验证或数据泄漏限制：
旧R2合同恢复或撤权；算法质量仍如何：
未达到目标与原因：
真实批处理命令 / resume命令：
本轮进程/租约已排空：
Git干净且未push：
physical_deployable=0；用户视觉验收=PENDING。
```

封账必须追加算法分级：`TARGET_MET | PARTIAL_MATERIAL_PROGRESS | REJECTED_NO_RECOVERY | INCONCLUSIVE_RUNTIME`。其判定依据是 W0 同口径恢复、W1-ADOPTION、冻结后 holdout 和多会话 R0 质量，不是任务节点数或导出数。

## 15. H0–H1 实际里程碑（2026-09-20 00:35 +08:00）

本节是已落盘证据的阶段记录，不代替最终封账。父任务仍沿用唯一时钟
`0915-robot-quality-recovery-v21-20260919T234412+0800`；不因 successor 或运行成功重置。

### 15.1 P0 / A0 / B0 / C0 / C1

- P0 已从本地 session log 恢复旧 R2 合同 exact 4438 bytes，SHA
  `fb412c90...e8ac6`。这只修复 lineage；旧 R2 质量仍为 0/4。
- A0 已重建 4 会话、1058 帧、2116 side-frame 的完整失败矩阵。raw observed
  与旧 R0 valid 均为 1320 side-frame；R0 导出 4/4，但整会话质量准入 0/4。
  q22 限位违规和时间轴失败均为 0，但旧 R0 没有全帧 FK/碰撞与 q22
  运动门，因此新准入窗口仍为 0。
- B0 证明 Task Object 不应受 HaWoR 整会话 strict 阻塞；Poker Object 2/2
  只获得 visible-surface 开发准入，Chips 0/2 仍 fail-closed。没有 Mask
  accuracy 声称。
- C0 首次审计遗漏 observed-surface status 过滤，已由 `INVALIDATION.json`
  撤销；有效 successor 是 `packages/C0_v2`，不得以父协调器中旧 C0
  `RESULT` 作当前结论。
- 纠正后 Contact 漏斗为 5040 pair rows、251 metric rows、0 inside finite patch、
  0 within fixed 5 mm、0 pixel-registration-bound。C1 进一步确认 146 个 patch
  全部有限，最近有限距离仍为 24.30 mm；真实重叠候选帧7/7没有独立
  可观测手指表面。决策为 `NO_GO_CONTACT_SUCCESSOR`，Contact 保持 UNKNOWN，
  R1-E 保持关闭，5 mm 门未改。

### 15.2 A1 W1-DIAG 真实运行

固定 Poker044（166帧）和 Chips097（394帧）已在 physical-left `sourceIndex=1`
crop + resize-only 域运行。实际模型加载1次、detector加载1次、tracker reset 2次，
运行失败0。候选签名为 `9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b`，
所有原 strict 阈值不变，bounded successor 未消费。

- Poker044：session strict 拒绝；left/right 逐侧 strict 均拒绝；raw observed 16/74。
- Chips097：session strict 拒绝；left 逐侧拒绝，right 逐侧 strict 通过；raw observed
  0/394。
- 这证明的是“W0 0/4 不应阻止 W1 合法输入推理”，不是 HaWoR 精度提升。
- 局部 structural 区间只作候选；R0 质量仍须独立通过最小窗口、FK、限位、
  碰撞和时序门，当前不升级任何 R0 质量结果。
- 两条全片可视化已完整解码：
  `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/`。

### 15.3 当前决策与下一步

1. A1 dependency-split candidate 已在 W1-ADOPTION 解封前冻结；下一步只能用
   版本化、可审计的 cohort-fence adapter 运行 Poker106/Chips029，不得根据
   044/097 结果改阈值、预处理或模型。
2. B1 Mask 必须使用 A1 新生成的同图像域 2D direct-observed prompt。旧
   `EQUIDIS62 → FOV90` RGB/HaWoR 缓存不可复用。Object 与 Hand terminal 解耦；
   Hand 按侧 fail-closed。
3. Poker 身份歧义保持 UNKNOWN；Chips 三袋分开，禁止 union。
4. C 线只保留 CPU 内部实现自检；没有新的独立手指表面与物理 registration
   证据前，不再启动 Contact successor。

阶段判定：`PARTIAL_MATERIAL_PROGRESS`。实际修复了执行/依赖范围，并获得一侧
W1 strict 证据；但 whole-session HaWoR strict、R0 质量、Contact、R1/R2 和部署权威均未恢复。

### 15.4 A2 / B1 执行状态（2026-09-20 00:49 +08:00）

- A2 W1-ADOPTION 已完成 106/029 两条真实 GPU 推理，运行失败0，租约等待
  400.5秒，模型/检测器各加载1次。`adoption_adapter_or_run_signature_sha256`
  为 `e3ca5185…`；A1 `9ccf7ff5…` 只作 parent candidate lineage。Poker106
  左右观测为 29/100，两侧 strict 拒绝；Chips029 左右观测为 223/233，
  两侧仍 strict 拒绝。Chips029 右侧仅产生 233 个 structural 候选帧
  （0–183、185–233），R0 自身门仍待评估。A2 是
  `COMPLETED_ADOPTION_ALL_TERMINAL`，但 `adoption_decision=NOT_AUTOMATED`，不等于候选已采用。
  两条全片视频已放入 `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/A2_W1_ADOPTION/`。
- B1 SAM3.1 successor 已通过 19/19 CPU 合同测试并进入当前算法合同。
  W0 四条只做 8 个 Hand/Object 旧 `RESULT` SHA 回归，不重跑 SAM；GPU 候选
  严格只有 Poker044/Chips097。Hand 使用 A1 同图像域 direct-observed 2D 并
  按侧准入，Object 不依赖 Hand/HaWoR 整段终态。B1 将在 A2 释放同一 GPU
  租约后顺序执行；当前仅为已注册/已预检，不是 Mask quality pass。
- C1 依旧是 `NO_GO_CONTACT_SUCCESSOR`。CPU 自检可继续验证内部几何与
  fail-closed 路径，但不得修改 5 mm、不得生成 Contact authority。

### 15.5 B1 Mask 和 C1 CPU 自检终态（2026-09-20 01:08 +08:00）

- B1 只对 Poker044/Chips097 做了真实 GPU 推理，W0 四条的 8 个旧结果均仅做
  SHA/合同回归。Chips097 右手输出 393/394 帧 `PASS_HAND_DIRECT_OBSERVED_PROXY`；
  左手由于0个 direct-observed anchor 保持 UNKNOWN。Chips Object 未选到可消费实例，
  三个物理槽均 UNKNOWN。Poker044 Object 有3个可见候选，但无独立的同一
  物理牌/牌面身份证据，因此全部 UNKNOWN。Poker044 Hand 在 tracker 传播时出现
  `KeyError:164`，失败 staging 保留；整个 B1 终态为 `FAILED_RUNTIME_FINAL`，
  不能写成 Mask quality pass。运行前的独立审计已修复 terminal 误报，因此本次
  正确保留了 runtime failure 与其余 capability 证据。
- C1 CPU-only 自检已在冻结 C0_v2/C1 缓存上重放。27个顶层文件与168个
  Depth 帧 SHA 全部通过；504个 packed support row 、23,202个边界点和251个 metric row
  内部回环一致。但 inside finite patch 仍为0，5 mm内仍为0，最小距离
  24.304 mm；92–98 无独立 finger surface，registration 仍 `UNBOUND`。结论严格保持
  `Contact=UNKNOWN / authority=NONE / R1-E=CLOSED`。浅层图在
  `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/C1_SELF_CHECK/`。

### 15.6 A3 Kai22 R0 自身门诊断（2026-09-20 01:24 +08:00）

A3 已作为 `kai22_r0_quality_successor_v1` 登记，并分别对 A1 W1-DIAG 与
A2 W1-ADOPTION 的冻结 structural side-frame 执行 CPU 全帧检查。它不以
whole-session HaWoR strict 作为自身求解门，也没有修改 HaWoR 候选、数值阈值或
父候选签名。

- A1：560 timeline frames、341 structural side-frames；341/341 完成 q22、
  URDF 限位、完整 FK 与 pinned KaiHand 非父子手内碰撞检查，最终质量准入 0。
- A2：404 timeline frames、233 structural side-frames；233/233 完成相同检查，
  最终质量准入 0。
- Poker044 只有右侧1个 structural frame，缺运动上下文；Poker106 为0。
- Chips097 右侧340帧有334帧运动上下文，但280/340帧至少一个关节触及限位边界；
  q22 速度 P95 约1.035 rad/s、加速度 P95 约36.16 rad/s²。
- Chips029 右侧233帧有229帧运动上下文，112/233帧触及限位边界；速度 P95
  约0.999 rad/s、加速度 P95 约33.91 rad/s²。

V2.1 当前没有冻结可信的正 q22 速度阈值、加速度阈值和最小连续质量窗口；KaiHand
URDF 的 velocity 字段又全部是0，不能当成有效阈值。因此 A3 状态固定为
`COMPLETED_DIAGNOSTIC_NO_R0_QUALITY_UPGRADE`，first blocker 是
`MISSING_FROZEN_Q22_MOTION_THRESHOLDS_AND_MINIMUM_WINDOW`。cross-hand、Tianji arm、
object、environment collision 仍为 `UNVERIFIED`，不得把静态门通过写成 Robot
质量通过、控制真值或训练资格。

收据：

- `packages/A3_R0/A1/RESULT.json`，SHA256 `46561139…`
- `packages/A3_R0/A2/RESULT.json`，SHA256 `d63c7fdb…`
- 浅层摘要：`docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/A3_R0/`

### 15.7 B1R Poker044 Hand 与 A4 动态门审计（2026-09-20 01:30 +08:00）

B1R 只处理原 B1 唯一的运行失败，不改写原 B1 `FAILED_RUNTIME_FINAL`。根因定位为：
Poker044 右手 fallback 锚点165完成正向传播后，反向传播在尚未 yield 任何帧时访问
frame164并抛出精确 `KeyError(164)`。successor 只在 anchor=165、reverse、0 yield、
166帧及 `KeyError(164)` 全部同时成立时转换一次；未返回方向保持
`UNKNOWN_NOT_ABSENT`，其余异常仍 fail-closed。

- B1R GPU 执行成功，终态 `COMPLETED_WITH_QUALITY_REJECTION`，不是 Mask pass。
- Poker044 左手为 `REJECTED_DIRECT_OBSERVED_ADMISSION`；右手为
  `REJECTED_TRACKER_DIRECTION_INCOMPLETE`；两侧 `consumer_allowed=false`。
- 完整复核视频为166帧、30 FPS、1280×420，full decode通过：
  `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/B1R_POKER044_HAND/hand/`
- B1R `RESULT.json` SHA256 `7be40d38…`，`B1R_HAND_ADMISSION.json` SHA256
  `8dcd3041…`。它只能与原 B1 已完成 capability refs 组成新的 composite closure，
  不能冒充原 B1 aggregate 被原地修复。

A4 对项目内 KaiHand URDF、旧 R0、R7.3 policy、历史 hard-geometry 收据和 A1/A2/W0
分布做了只读 authority 审计。结论是目前没有可直接复用到 A3 raw direct-retarget q22 的
正 velocity、acceleration 或 minimum-window authority：

- R7.3 的0.08 rad/frame与0.06 rad/frame²只约束 post-solver Robotized 视频连续性；
  严格30 Hz下影子换算2.4 rad/s与54 rad/s²也只能作 comparator。
- KaiHand 左右各22个运动关节的 URDF velocity 均为0，不能作为正阈值。
- A1/A2 与 W0 的 raw q22 分布和 post-solver limiter 输出语义不同，不能在同一批数据上
  选择阈值再宣布通过。

因此 A3 继续固定 `DIAGNOSTIC_ONLY_NO_QUALITY_UPGRADE`。若要晋升，至少需要先冻结
artifact/算子/单位/阈值/最短窗口，再使用未参与阈值选择的4条新独立来源会话
（Chips2、Poker2）和预冻结连续性标签验证。A4 收据位于
`packages/A4_MOTION_GATE_AUTHORITY_AUDIT/`，`RESULT.json` SHA256 `401d182d…`。

### 15.8 A5 Kai22 限位饱和根因（2026-09-20 01:43 +08:00）

A5 对 A3 四条会话和冻结 W0 四条会话逐帧重算 retarget 公式，全部有效 q22 的最大
闭包误差为 `2.22e-16 rad`。A3 两条 Chips 的限位边界命中均只发生在
physical-left `thumb_joint4` 上界：

- Chips097：280/340，82.35%，最长连续98帧。
- Chips029：112/233，48.07%，最长连续26帧。
- A3 没有任何非拇指或下界命中。Poker044只有1个有效side-frame、Poker106为0，
  不足以估计Poker饱和率。

冻结映射先计算无符号 MANO 拇指 distal bend，再令 `thumb_joint4 = 0.5 × bend2`，
而 pinned URDF 上界只有0.175 rad，最后在 retarget 内部执行 `np.clip`。因此 raw bend
达到0.350 rad（约20.05°）时就精确停在上界。它解释了系统性拇指饱和，不支持“全手
轴翻转/全局scale错误”作为当前主因。

合同必须区分：`internal_mapping_clip=true` 是现行 retarget 的组成部分；
`posthoc_gate_clip_allowed=false` 表示质量门之后禁止再补救性裁剪。后者不能被读成
q22 从未被内部 clamp。当前命中仍位于合法范围，既不是越界，也不是质量通过。

唯一允许准备的 challenger 是不扫参的冻结几何有界 IK：输入、左右映射、URDF、
structural mask和所有质量门保持不变，只替换直接角度复制；必须在 Chips097/029 分别降低
thumb4占界率与最长连续段，同时不恶化拇指方向/指尖目标、FK、碰撞或时序。任何单会话
退化、饱和迁移、求解失败或需改阈值即拒绝。A5 收据位于
`packages/A5_KAI22_SATURATION_AUDIT/`，`RESULT.json` SHA256 `f7bfcb5e…`；浅层图在
`docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/A5_KAI22_SATURATION/`。

### 15.9 B2 既有 Depth → Object6D 可执行性审计（2026-09-20 01:45 +08:00）

本节是执行前可行性审计；Poker044 canary 已于 15.12 完成，下文的
“建议执行”不再是待办项。

B2 本轮只做读审计，没有重跑 FoundationStereo，也没有生成新 Object6D。
W0 4条与 W1 8条共 12 会话的 Depth 都通过原合同；本次重算并核对
3,390/3,390 个 frame NPZ SHA256，帧轴、解码帧数、数组和 `Z=fB/d`
回环一致。这只证明内部闭包，`external_millimeter_accuracy=false`，不是外部
毫米精度认证。

- Poker031/119 的已有 Object6D 只作不可变 CPU 回归；它们可发布直接可见
  finite plane/patch，但物理牌与牌面 identity 没有独立绑定，必须保持 UNKNOWN。
- Poker044 有 3 条 B1 可见 track，但实例消费准入为0。它只能在新的 fail-closed
  合同下执行一次 CPU、`REVIEW_DIAGNOSTIC_ONLY` 可见面 canary；
  `consumer_allowed=false`，不得进入 Contact/Robot/Clean/训练。
- 剩余9条都先被合法 Task Object Mask 阻塞，不开无效 Object6D，也不重跑 Depth GPU。
- 现有 W0 runner 是 Poker 平面语义，不能用于 Chips。Chips successor 必须三袋分离，
  只发布每实例直接可见 non-rigid local patch，禁止 union、单平面或整袋刚体6DoF。

完整收据位于 `packages/B2_DEPTH_OBJECT_AUDIT/`。当前唯一建议执行项是注册后的
Poker044 review-only CPU canary；还没有 Object identity、hidden geometry、Contact、
训练、控制或部署权威。

### 15.10 A6 拇指单因素有界 IK 候选拒绝（2026-09-20 01:55 +08:00）

A6 只复用已登记 `run_robot_hand_fullsession_v2.py` 的第一阶段拇指目标和数值配置，
每帧以 A3 same-frame q22 为唯一 seed，只优化 `q[0:6]`，`q[6:22]` bitwise 冻结。
没有使用多 seed、没有使用历史 8/16/32 tip-weight refinement，没有扫参。
对 Chips097/029 的远端全量 CPU 只读 preflight 虽然显著降低了指尖/方向残差，
但必须按预冻结的核心门拒绝：

- Chips097：`thumb_joint4` 上界命中 `280/340 → 304/340`，最长连续段 `98 → 113`。
- Chips029：`112/233 → 233/233`，最长连续段 `26 → 184`。
- 饱和还迁移到 q0/1/2/4/5 中的多个拇指关节；这证明现有 objective 在没有
  authority-bound joint-centering/barrier 时会用大面积顶限位换取几何残差改善。
- URDF limit、FK、pinned 非父子手内碰撞、solver 和 non-thumb 冻结均通过，
  但不能抵消上述退化。

候选终态为 `COMPLETED_READ_ONLY_PREFLIGHT_REJECTED_STRICT_COMPARATOR`，未注册、
未 adoption、未使用 GPU。证据位于 `packages/A6_THUMB_BOUNDED_IK_REJECTED/`。
除非治理先冻结已有 authority 的 joint-centering/barrier objective，否则不应继续调权重或开 A7。

### 15.11 D1 Clean 零写入准备终态（2026-09-20 02:04 +08:00）

`robot_quality_recovery_v21_d1_clean_prep_cpu_v1` 已登记并对 Poker044/Chips097
完成真实 CPU 准备。它只从冻结 B1+B1R capability closure 构建 lossless Raw、
Hand、Task Object、`M_remove/M_write/M_flow/UNKNOWN` 和 source map，没有运行补洞模型。

- 独立 validator 重算 560/560 帧，全部 SHA、`M_remove ⊆ M_write ⊆ M_flow`、
  `UNKNOWN == M_write`、Task Object veto、source-map 和 decoded Raw byte-exact 硬门全部通过。
- Poker044 左右手均 UNKNOWN，因此 `M_remove=M_write=0`；仅保留 2,999,647 个
  同帧可见对象保护像素，物理牌/牌面身份仍 UNKNOWN。
- Chips097 仅右手393/394帧可用，左手 UNKNOWN；`M_write` 共 14,909,727 像素，
  目前同样全部为 UNKNOWN，候选像素修改数为0。Chips三袋依然三槽分离，没有 union。
- 全局修改像素0，`M_write`外修改0，保护对象修改0；未读取旧 Clean、
  future donor 或 hidden truth。

终态是 `BLOCKED_PREREQ_FRESH_INPAINTING_OR_UNKNOWN`，`clean_terminal=false`，
不是 Clean 质量通过。深层证据位于 `packages/D1_CLEAN_PREP_INPUT_V1/` 与
`packages/D1_CLEAN_PREP/`；独立验证收据 SHA256 为 `c043eb96…`。
另一次独立只读审计重算6,027个唯一 SHA-bound 文件（共737,431,440 bytes），
560/560 Raw 帧与重新 CPU 解码一致，560/560 candidate PNG 与 Raw 文件字节及解码像素
都一致，892/892 Hand/Object frame-instance mask 与原 B1 packed semantic/state ledger 一致。
独立决策为 `GO_D1_CPU_PREPARATION_ONLY_CLEAN_REMAINS_BLOCKED`，收据位于
`packages/D1_CLEAN_PREP/INDEPENDENT_AUDIT/`。

两条零写入全片视频已完整解码，只用于复核域和 UNKNOWN：
`docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP_V2/`。初次 V1 视频字节无误，
但 receipt 记录了原子提交前的 staging path；已保留缺陷收据并用不改画面的
V2 包装重新发布，不得引用 V1 receipt。

### 15.12 B2 Poker044 可见 Object6D review canary（2026-09-20 02:02 +08:00）

`b2_depth_to_visible_object_poker044_v1` 已登记并完成唯一允许的 CPU canary。
运行重用 166 帧冻结 FoundationStereo cache，没有重跑 Depth/SAM，并对
Poker031/119 的不可变回归重新核对 SHA。

- 三条 geometry track 有直接可见 finite patch 的帧数分别为 130/151/80。
- `physical_card_identity=UNKNOWN_UNBOUND`、`face_identity=UNKNOWN_UNBOUND`；
  geometry track ID 绝不代表已证实的物理实例。
- `consumer_allowed=false`、`review_only=true`、`external_metric_accuracy=false`，
  无 hidden geometry/Contact/Clean/Robot/训练/控制权威。
- 浅层视频 166帧、30 FPS、1280×960，full decode 通过：
  `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/B2_DEPTH_TO_OBJECT_POKER044/`。
- 独立只读审计核对22个钉住引用、3×166行 JSON/NPZ 和逐帧 support：
  361个非空 support row、746,746个直接可见 support 像素一致，未发现缺陷。

`RESULT.json` SHA256 为 `4aa63612…`，深层证据位于
`packages/B2_DEPTH_TO_OBJECT_POKER044/`。这是可见表面的开发复核证据，不是 Object6D 真值。
