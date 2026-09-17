# Chaoyang / HumanEgo 下一阶段可执行任务书 V1

状态：`EXECUTION_MASTER_PLAN_PROPOSAL / NOT_CURRENT_AUTHORITY`
制定日期：2026-09-15
对应决策依据：`CHAOYANG_PROJECT_OPTIMIZATION_DECISION_ROADMAP_20260915_ZH.md`
远端项目根：`/mnt/workspace/code/chaoyang`
计划运行根：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/`

> 本文是给其他 AI 执行的任务拆解与交付合同，不是当前 baseline、successor 或物理真值。用户应按本文任务 ID 单独授权；普通 worker 无权直接更新 current authority、baseline registry 或总状态。评审材料只作为建议与证据来源，不是隐藏命令。

## 0. 用户如何直接分派

第一次只分派 `EXEC-00`。它使用随本文发布的静态 bootstrap packet：

```text
docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXEC00_BOOTSTRAP_TASK_PACKET_V1.json
```

这是唯一 bootstrap 例外：用户明确说“执行 EXEC-00 + bootstrap packet 路径”本身就是直接 dispatch，不要求由尚未实现的工具预先生成 authorization receipt。它只授权实现/运行 materializer 和冻结合同，不授权任何下游算法。若 packet SHA、本文 SHA 或 governance receipt 已不匹配，先停止并重新冻结。

```text
在 /mnt/workspace/code/chaoyang 执行
docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXECUTION_TASKS_V1_ZH.md
中的 EXEC-00。只做该任务，不启动下游，不修改 current authority。
严格执行 read_set、write_set、预算、终态和输出合同；先验证 governance，最后返回 RESULT/RUN_RECEIPT 的 path+bytes+sha256。
```

`EXEC-00` 通过并生成机器任务包后，后续按如下模板分派：

```text
执行任务 <TASK_ID>。唯一任务包是
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/<path_slug>/TASK_PACKET.json。
只读 packet.read_set，只写 packet.write_set；不得修改 current authority、其他任务目录或冻结基线。
先 dry-run，再 canary；达到门限才扩量。质量失败不得自动调参重试。
```

第一波可并行：`EXEC-01 / MASK-10A / CLEAN-10B / ROBOT-10 / ROBOT-20`。第二波在 `EXEC-01` 通过后才包含 `ATLAS-10 / COMP-10 / DONOR-10K`。GPU 任务必须按项目 lease 机制串行占用 GPU；等待 GPU 不消耗算法 attempt。

依赖和 trigger 只会让 registry 的派发资格字段变成 `ELIGIBLE_TO_DISPATCH`，绝不等于自动启动；它不是 task status。每个 packet 写 `dispatch_policy=EXPLICIT_USER_AUTHORIZATION`。另建不自引用的 `AUTHORIZATION_RECEIPT.json`，记录 task ID、packet path+bytes+SHA、用户授权文本摘要和时间；没有该 receipt，监督器不得 claim、启动 GPU 或执行 GOV-99。

## 1. 本阶段唯一目标与明确非目标

只解决五类已观察到的问题：

1. Controller、绑带、线缆和附件未可靠选中，或被错误并入任务物体。
2. Clean 残留、误擦、Raw 回灌，以及 `M_flow` 膨胀越权成为最终写入域。
3. donor 来自错误表面，遮挡后的任务物体没有合法外观来源。
4. Robot 真正的数字几何穿透，与 compositor 前后绘制顺序错误没有分开。
5. Robot 总耗时与“合格 Robot 帧/合格 H50 配对窗口”的产率没有分阶段核算。

执行链固定为：

```text
基线/门限/来源冻结
├─ 因果与 pixel-source 测试
├─ 新传感器附件多角色 Mask（独立线）
├─ Clean 删除域/写域
├─ Poker 因果对象 atlas（不依赖 Clean）
├─ ownership compositor 单元测试
└─ Robot 目标/门/耗时审计

CLEAN-10B = role orchestration
CLEAN-10C = CLEAN-10B + adaptive boundary（Clean C）
CLEAN-20  = Clean C + background donor（D1）
CLEAN-21  = D1 + Poker atlas（Poker D2）
任务各自 Clean + Robot + compositor → Chips/Poker Robotized RGB
任务各自严格配对数据                 → 四个 Visual Aux checkpoint
```

下列两项不在当前关键路径：

| 项目 | `phase_scope` | `future_readiness` | 是否阻塞数字/视觉任务 |
|---|---|---|---|
| Physical Robot 部署 | `OUT_OF_SCOPE` | `BLOCKED_EXTERNAL` | 否 |
| 真实 Robot Policy | `OUT_OF_SCOPE` | `BLOCKED_EXTERNAL` | 否 |

用户已明确不提供人工 Gold 或外部标定。因此当前不安排人工标注、外部尺量、CAD/TCP/安装测量；也不允许算法猜测替代它们。Physical Robot 将来恢复至少需要实测 adapter CAD、flange→KaiHand、TCP、camera/world→base 与实机安全验证。真实 Policy 将来恢复至少需要同步 robot RGB/state/action、时钟、反馈和 action schema。这些缺口只记录，不等待、不伪造，也不阻塞 Mask、Clean、Digital Robot、Robotized RGB 和 Visual Aux。

## 2. 所有任务共同的硬合同

### 2.1 开始顺序

每个 AI 在任何读取或修改前执行：

```bash
cd /mnt/workspace/code/chaoyang
python -m tools.governance.validate_governance_state
```

随后先完整读取仓库 `AGENTS.md`，再读取 receipt 绑定的 `CURRENT_PROJECT_STATUS_MIN.json`、`PLAN_REVISION.json`、当前 task packet，以及本任务 packet 的其余 `read_set`。首次读取最多 8 个文件、搜索最多 20 条、日志最多 80 行。README、聊天、mtime 和名字中的 `latest` 都不是 authority。

### 2.2 Task Packet 必填字段

`EXEC-00` 必须按 `docs/governance/schemas/task_packet.schema.json` 生成每个任务的独立 JSON；不得让 worker 从本文自由改写。每个 packet 显式包含：

```text
schema_version = exact78-task-packet-v1
task_id / plan_revision / packet_revision / created_at / stage / objective
workdir / command_argv
read_set（初始最多8项）/ write_set
scope / frozen_inputs / unique_experimental_variable
prerequisites / gates / quality_gates
budgets.cpu_seconds / gpu_seconds / wall_seconds
attempt_max（1..3）
stop_condition / stop_conditions
output_contract / claim_limit
executor_epoch
fencing.pid_startticks_required = true
fencing.immutable_final = true
ai_io_limits
artifact_revision_contract.forbid_in_place_overwrite = true
artifact_revision_contract.required_revision_status = VALID_FOR_PINNED_REVISION
fencing.partial_attempt_is_never_successor_input = true
run_signature_contract
dispatch_policy = EXPLICIT_USER_AUTHORIZATION
authorization_receipt_required = true    # EXEC-00 bootstrap 唯一例外为 false
```

规范默认值固定为：`schema_version=exact78-task-packet-v1`、`packet_revision=1`、`executor_epoch` 取 materialize 时 task ledger 的下一正整数、`plan_revision` 取当时 receipt 绑定值；同时增加 `source_execution_plan={path,bytes,sha256,status:NOT_CURRENT_AUTHORITY}`，不能把本文冒充 current plan。`created_at` 使用 materializer 的带时区 ISO-8601 时间。task ID 一律使用大写连字符，目录 `path_slug` 一律为对应小写下划线；registry 同时冻结二者，依赖只能引用 canonical task ID。

`run_signature` 覆盖 input path+bytes+SHA、code、config、weights、calibration 或明确 `ABSENT`、schema 和 packet revision。未知权重身份不得伪造；历史结果只能标 `UNKNOWN_VERIFICATION_REQUIRED`。

本文任务卡中的“current/对应/receipt/manifest”等是 materializer 的逻辑引用，不得原样进入 packet。`EXEC-00` 必须先写 `LOGICAL_REFERENCE_FREEZE.json`，将每个逻辑引用解析为唯一 `path,bytes,sha256`；最终 packet 的 `read_set` 只能包含不超过 8 条确切路径。解析失败即 fail closed。

### 2.3 实现与运行必须分成两个冻结阶段

多数任务要新增脚本，不能在代码尚不存在时伪造 run signature。每个算法任务固定执行：

```text
已授权 TASK_PACKET.json
→ IMPLEMENT phase 只写 source_write_set，完成单元测试
→ IMPLEMENTATION_RECEIPT.json（base commit、逐文件SHA、测试SHA）
→ 在首次推理前生成不可变 RUN_SIGNATURE.json，绑定 implementation、inputs、config 和 phase argv[]
→ dry-run → canary → regression → 条件 full-batch → finalize/verify
```

runtime retry 必须复用同一个 TASK_PACKET、RUN_SIGNATURE、input/code/config；算法、参数或 source SHA 改变必须新 task revision，不能占用 `attempt_0002`。本文 shell 块是 phase 语义；worker 只执行 materialized TASK_PACKET 中由 RUN_SIGNATURE 绑定的 `cwd/env/argv[]`，不得手抄或改参数。

为避免“先有 packet SHA 才能授权、先授权才物化 packet”的循环，EXEC-00 一次性物化第 3.1 节全部核心 packet，初始均写 `authorization_state=NOT_AUTHORIZED`。前置依赖只决定 eligibility，不决定是否生成 packet。收到用户对精确 task ID 的新授权后，唯一协调者执行：

```bash
python src/chaoyang/ops/freeze_project_optimization_execution_v1.py --record-authorization --task-packet <ABSOLUTE_PACKET_PATH> --authorization-text-sha256 <USER_MESSAGE_SHA256> --output <ABSOLUTE_AUTHORIZATION_RECEIPT>
```

该 receipt 绑定 packet path+bytes+SHA 后，worker 才能 claim。没有精确 argv 的条件任务只保留 `PACKET_TEMPLATE_ONLY`，不可 claim。

### 2.4 目录、双写集、并发和 fencing

每个任务有两个互斥写集：`artifact_write_set` 只写计划根下自己的 `<path_slug>/`；`source_write_set` 只写 FILE_OWNERSHIP 中列出的版本化代码文件。schema 的 `write_set` 是两者精确路径的并集，不能写宽泛目录。布局固定为：

```text
<path_slug>/
├── TASK_PACKET.json
├── CONTEXT_CARD.md
├── attempts/attempt_0001/
├── attempts/attempt_0002/
├── attempts/attempt_0003/
└── final/
```

packet 唯一存储位置是：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/<path_slug>/TASK_PACKET.json
```

artifact 唯一存储位置是：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/artifacts/<path_slug>/
```

artifact 目录不复制、不改写 packet。任务卡中“只写 `<path_slug>/`”均指上述 `artifacts/<path_slug>/`。

所有 worker 由 EXEC-00 新增的统一 wrapper 启动：

```text
src/chaoyang/ops/run_project_optimization_task_v1.py
```

它验证 authorization receipt、packet/RUN_SIGNATURE SHA，以 `O_CREAT|O_EXCL` 取得 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/locks/<path_slug>.claim.json`，生成 fencing token，记录 PID/startticks/epoch 并维持 heartbeat。冲突时先成功创建并匹配 CAS 的 token 为唯一 writer，另一个立即终止。RUN receipt 必须记录这些字段。

代码优先新增带版本号的文件，禁止原地覆盖冻结实现。两个任务若必须修改同一路径则不得并发，由唯一协调者排序。共享 baseline 和他人 run dir 永远只读。

GPU phase 必须由统一 wrapper 嵌套当前已有的：

```text
src/chaoyang/ops/run_gpu_command_with_v71_lease.py
```

materialized argv 必须显式包含 task ID、attempt ID、executor epoch、priority、GPU ID、`--min-free-mib`、`--wait-seconds 1800`、wall seconds、immutable receipt、`--lease archive/baseline-20260917-0aa69e9/content/history/runtime/_run/GPU_LEASE.json`、`--lock archive/baseline-20260917-0aa69e9/content/history/runtime/_run/GPU_LEASE.lock`、claim limit，以及 `--` 后的业务 argv。GPU heartbeat=30 秒、TTL=120 秒；等待态 `WAIT_GPU_RESOURCE` 不计 attempt。回收必须同时证明 heartbeat 过期、进程身份不匹配且 GPU 无对应进程。

规范嵌套形状如下，尖括号全部由 materializer 写死：

```bash
python src/chaoyang/ops/run_project_optimization_task_v1.py --task-packet <PACKET> --phase <PHASE> --authorization-receipt <AUTH> --run-signature <SIGNATURE> -- \
python src/chaoyang/ops/run_gpu_command_with_v71_lease.py --task-id <TASK_ID> --attempt-id <ATTEMPT_ID> --executor-epoch <EPOCH> --priority <PRIORITY> --gpu-id 0 --min-free-mib <FROZEN_MIB> --wait-seconds 1800 --wall-seconds <PHASE_WALL> --receipt <GPU_RECEIPT> --lease archive/baseline-20260917-0aa69e9/content/history/runtime/_run/GPU_LEASE.json --lock archive/baseline-20260917-0aa69e9/content/history/runtime/_run/GPU_LEASE.lock --claim-limit <CLAIM_LIMIT> -- <BUSINESS_ARGV>
```

所有 CPU/GPU/wall 预算均为该 task revision 的总预算，覆盖实现、dry-run、canary、regression、full 和所有 runtime attempts；GPU 等待不计 GPU/attempt，但 30 分钟后以 `BLOCKED_RESOURCE` 终结或重新由用户分派。

### 2.5 尝试和终态

只使用项目状态机：

```text
PENDING / CLAIMED / RUNNING / FAILED_RUNTIME_RETRYABLE
PASSED / FAILED_QUALITY_C / FAILED_RUNTIME_FINAL
BLOCKED_PREREQ / BLOCKED_RESOURCE / BLOCKED_EXTERNAL
BLOCKED_REFERENCE_PROOF / UNKNOWN_VERIFICATION_REQUIRED / CANCELLED
```

- runtime 最多初次加 2 次重试；每次进入新 attempt 目录。
- `FAILED_QUALITY_C` 不自动重试，不得看结果后静默调参。
- partial 永远不能成为 successor 输入。
- 条件未触发用 `CANCELLED`，并写 `terminal_reason=CONDITION_NOT_TRIGGERED`，不创造状态枚举。
- “完成”表示有唯一不可变终态，不等于全部 `PASSED`。

### 2.6 固定交付、发布 DAG 和 claim

每项任务的 `final/` 至少包含：

```text
RESULT.json
ARTIFACT_MANIFEST.json
METRICS.json
DECISION.md
RUN_RECEIPT.json
NEXT_ACTION.json
```

为避免自哈希循环，发布顺序固定为：

```text
业务产物
→ METRICS / NEXT_ACTION / DECISION
→ ARTIFACT_MANIFEST（不含自身、RESULT、RUN_RECEIPT）
→ RESULT（引用前述，不引用自身）
→ RUN_RECEIPT（最后写；引用 RESULT 等，不引用自身）
→ 外部 registry/index 记录 RUN_RECEIPT 的 SHA
```

普通 worker 的 `RESULT.json` 至少包含 task/packet revision、status、terminal_reason、claim_limit、`authority_promoted=false`、failed gates、blocker、resume condition，以及被引用 artifact 的 path+bytes+SHA。GOV-99 是唯一例外，可输出逐 claim promotion decision，但必须有独立用户授权和 governance CAS。Digital Robot sidecar 必须 `control_ground_truth=false`；Visual Aux checkpoint 必须 `policy_checkpoint=false`。

证据 claim 只允许：

```text
SUPPORTED_EXTERNAL_TRUTH
SUPPORTED_INTERNAL_CONSISTENCY
DEVELOPMENT_EVIDENCE
HYPOTHESIS_ONLY
UNSUPPORTED
WITHDRAWN
```

本阶段通常只能使用 `SUPPORTED_INTERNAL_CONSISTENCY`、`DEVELOPMENT_EVIDENCE` 或 `HYPOTHESIS_ONLY`。

复核视频默认必须覆盖完整会话。只有任务已进入明确失败终态且无法生成全片时才允许 partial，并必须水印 `PARTIAL_REVIEW`、记录实际帧区间和 `review_status=PARTIAL`。

### 2.7 新脚本统一 CLI

每个新脚本顶部必须用中文完整说明其适用 modes。处理型脚本实现：

```text
--task-packet <绝对路径>
--dry-run
--canary
--resume
--full-batch     # 仅 canary receipt 过门后可用
--output-root <packet冻结路径>
```

纯单元测试或只读审计脚本可不支持 `--full-batch`，但 packet 必须显式写 `supported_modes`。处理型任务的 TASK_PACKET 必须完整列出 dry-run、canary、regression、full-batch、finalize 和 verify 的 argv，并由 RUN_SIGNATURE 绑定；若某 phase 不适用，写 `NOT_APPLICABLE`。

所有脚本先 dry-run，再 canary；full batch 必须读取 canary receipt 并验证 SHA。不得把旧 `aligned.jsonl`、旧 `dataset.hdf5`、其他 RGB30 产物或 Clean 像素当新的时间/几何权威。

## 3. 主 DAG 与领取顺序

```text
EXEC-00 基线、门限、任务包冻结
├─ EXEC-01 因果/pixel-source 测试框架
├─ MASK-10A Controller geometry prompt
├─ CLEAN-10B role orchestration / write-domain audit
├─ ROBOT-10 target/contact/hard-soft 审计
└─ ROBOT-20 合格产出 profiling

EXEC-01 ─┬─ DONOR-10K 背景 donor kernel
         ├─ ATLAS-10 Poker causal atlas
         └─ COMP-10 ownership synthetic tests
CLEAN-10B → CLEAN-10C adaptive boundary
EXEC-01 + CLEAN-10C + DONOR-10K → CLEAN-20（D1）
CLEAN-20 + ATLAS-10 → CLEAN-21（Poker D2）
ROBOT-10 + immutable Robot receipt → ROBOT-11-SELECT
ROBOT-10 + ROBOT-20 → 条件 ROBOT-12/21/22/23/24
COMP-10 + CLEAN-20 + ROBOT-11-SELECT → COMP-20C（Chips）
COMP-10 + CLEAN-21 + ATLAS-10 + ROBOT-11-SELECT → COMP-20P（Poker）
EXEC-01 → DATA-10-IMPL
DATA-10-IMPL + COMP-20C → DATA-10C → Chips Raw/Robotized TRAIN pair
DATA-10-IMPL + COMP-20P → DATA-10P → Poker Raw/Robotized TRAIN pair
所有任务唯一终态 → GOV-99
```

`MASK-10A` 是 0909/0910 新传感器独立线，不是 exact78 Clean 或四 checkpoint 的前置。Robot Geometry 不以 Clean 为前置；Contact、geometry、compositor 分门。Poker 任一任务失败不得阻塞 Chips，反之亦然。

### 3.1 Canonical task ID、目录与分母

| task_id | path_slug | phase-success 角色 | 资源 |
|---|---|---|---:|
| EXEC-00 | `exec_00` | 全局必需 | CPU |
| EXEC-01 | `exec_01` | Chips/Poker 必需 | CPU |
| MASK-10A | `mask_10a` | sensor H4 独立结果 | GPU |
| CLEAN-10B | `clean_10b` | Chips/Poker 必需 | CPU/GPU |
| CLEAN-10C | `clean_10c` | Chips/Poker 必需 | CPU/GPU |
| DONOR-10K | `donor_10k` | Chips/Poker 必需 | CPU/GPU |
| ATLAS-10 | `atlas_10` | Poker 必需 | CPU |
| CLEAN-20 | `clean_20` | Chips/Poker 必需 | GPU |
| CLEAN-21 | `clean_21` | Poker 必需 | GPU |
| COMP-10 | `comp_10` | Chips/Poker 必需 | CPU |
| ROBOT-10 | `robot_10` | Robot 诊断必需 | CPU |
| ROBOT-20 | `robot_20` | Robot profile 必需 | CPU |
| ROBOT-11-SELECT | `robot_11_select` | Chips/Poker 必需 | CPU |
| COMP-20C | `comp_20c` | Chips 必需 | CPU |
| COMP-20P | `comp_20p` | Poker 必需 | CPU |
| DATA-10-IMPL | `data_10_impl` | Chips/Poker 必需 | CPU |
| DATA-10C | `data_10c` | Chips 训练必需 | CPU |
| DATA-10P | `data_10p` | Poker 训练必需 | CPU |
| TRAIN-CHIPS-RAW | `train_chips_raw` | Chips pair 必需 | GPU |
| TRAIN-CHIPS-ROBOTIZED | `train_chips_robotized` | Chips pair 必需 | GPU |
| TRAIN-POKER-RAW | `train_poker_raw` | Poker pair 必需 | GPU |
| TRAIN-POKER-ROBOTIZED | `train_poker_robotized` | Poker pair 必需 | GPU |
| GOV-99 | `gov_99` | 汇总必需 | CPU |

`TASK_REGISTRY.json` 还必须为每项写 `required_for_phase_success`、`required_for_chips_training`、`required_for_poker_training`、`optional_challenger`。GOV-99 可在任何任务已终结后汇总，但只有相应 required gates 全部 `PASSED` 才能把该分支标成阶段成功；否则阶段摘要字段写 `phase_outcome=COMPLETED_WITH_BLOCKERS`、`promotion_outcome=NOT_PROMOTED`。这两个不是 task status，GOV-99 自身仍使用第 2.5 节允许的终态。

`MASK-10B`、`DONOR-11`、`ROBOT-12/21/22/23/24` 和全部 `CH-*` 在本轮均为 `FUTURE_NOT_MATERIALIZED / PACKET_TEMPLATE_ONLY`，不进入完成分母。前置 receipt 只能提出触发建议；用户再次明确授权后，协调者才生成完整 packet。

### 3.2 Source file ownership

| owner task | 精确 source_write_set |
|---|---|
| EXEC-00 | `src/chaoyang/ops/freeze_project_optimization_execution_v1.py`, `src/chaoyang/ops/run_project_optimization_task_v1.py`, `tests/test_project_optimization_task_fencing_v1.py` |
| EXEC-01 | `src/chaoyang/ops/validate_visual_pipeline_causality_v1.py`, `tests/test_visual_pipeline_causality_v1.py` |
| MASK-10A | `src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py`, `src/chaoyang/pipeline/sensor_role_mask_geometry_v2.py` |
| CLEAN-10B | `src/chaoyang/ops/run_clean_role_write_domain_canary_v2.py`, `src/chaoyang/pipeline/clean_role_write_domain_v2.py` |
| CLEAN-10C | `src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py`, `src/chaoyang/pipeline/clean_adaptive_contact_boundary_v1.py` |
| DONOR-10K | `src/chaoyang/pipeline/causal_surface_donor_v1.py`, `tests/test_causal_surface_donor_v1.py` |
| ATLAS-10 | `src/chaoyang/pipeline/causal_poker_atlas_v1.py`, `src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py` |
| CLEAN-20 | `src/chaoyang/ops/run_clean_c_background_donor_v1.py` |
| CLEAN-21 | `src/chaoyang/ops/run_clean_d_poker_atlas_v1.py` |
| COMP-10 | `src/chaoyang/pipeline/ownership_compositor_v2.py`, `tests/test_ownership_compositor_v2.py`, `src/chaoyang/ops/run_robot_object_ownership_integration_v1.py` |
| COMP-20C / COMP-20P | 无；共同只读 COMP-10 implementation receipt |
| ROBOT-10 | `src/chaoyang/ops/audit_robot_target_contact_v1.py` |
| ROBOT-20 | `src/chaoyang/ops/profile_robot_pipeline_v1.py` |
| ROBOT-11-SELECT | `src/chaoyang/ops/select_digital_robot_candidate_v1.py` |
| DATA-10-IMPL | `src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py`, `tests/test_visual_aux_paired_ledger_v1.py` |
| DATA-10C / DATA-10P | 无；共同只读 DATA-10-IMPL implementation receipt |
| 四个 TRAIN | 无；使用冻结 trainer |
| GOV-99 | `src/chaoyang/ops/aggregate_project_optimization_execution_v1.py` |

任何现有文件若确需修改，必须由协调者新建 packet revision、从并行组移除并取得精确代码写锁；worker 不能扩大本表。

## 4. 可领取任务卡

以下任务卡是 `EXEC-00` 生成机器 packet 的唯一来源。物化 JSON 必须展开所有字段，不能只写“继承本文”。预算是硬上限；若 current authority 更严格，取更严格值。

### EXEC-00：执行事实、阈值和 Task Packet 冻结

- `objective`：把当前 authority、基线、输入、canary、门限、资源和写集冻结成机器任务包；不运行算法。
- `prerequisites`：`[]`；唯一前提是用户对 bootstrap packet 精确 SHA 的授权和 governance `PASS/FRESH`。
- `initial_read_set`：`AGENTS.md`、本文、`CURRENT_STATUS_RECEIPT.json`、`CURRENT_PROJECT_STATUS_MIN.json`、`PLAN_REVISION.json`、task packet schema、`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v74/TASK_PACKET.json`、baseline registry，共 8 项。bootstrap packet 自身不计入 read_set；regression 等通过第二阶段 resolver 定向读取。若 status min 的 `next_task.task_id` 不再是 `robot_geometry_expansion_v74`，则 bootstrap 以 `BLOCKED_REFERENCE_PROOF` 停止并要求发布新 revision，不能自行选择多个 active task 中的一个。
- `artifact_write_set`：本计划根下 `exec_00/`、`task_packets/`、`TASK_REGISTRY.json`。
- `source_write_set`：仅 `src/chaoyang/ops/freeze_project_optimization_execution_v1.py`、`src/chaoyang/ops/run_project_optimization_task_v1.py`、`tests/test_project_optimization_task_fencing_v1.py`。
- `unique_experimental_variable`：无，只冻结事实。
- `budgets`：CPU 10,800 秒，GPU 0，wall 21,600 秒，`attempt_max=1`。
- `command_argv`：

```bash
cd /mnt/workspace/code/chaoyang
python -m tools.governance.validate_governance_state
python src/chaoyang/ops/freeze_project_optimization_execution_v1.py \
  --task-packet docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXEC00_BOOTSTRAP_TASK_PACKET_V1.json \
  --plan docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXECUTION_TASKS_V1_ZH.md \
  --run-root archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1 \
  --emit-task-packets --dry-run
python src/chaoyang/ops/freeze_project_optimization_execution_v1.py \
  --task-packet docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXEC00_BOOTSTRAP_TASK_PACKET_V1.json \
  --plan docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXECUTION_TASKS_V1_ZH.md \
  --run-root archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1 \
  --emit-task-packets
```

- `output_contract`：`EXECUTION_BASELINE_FREEZE.json`、`LOGICAL_REFERENCE_FREEZE.json`、`ASSUMPTION_LEDGER.json`、`GATE_THRESHOLDS.json`、`RESOURCE_BUDGET.json`、`FILE_OWNERSHIP.json`、`FROZEN_CANARY_SESSIONS.json`、`FROZEN_CANARY_FRAMESET.json`、`FROZEN_SENSOR_H4_REGRESSION.json`、`TRIGGER_PREDICATES.json`、`TASK_REGISTRY.json`、全部核心 `NOT_AUTHORIZED` TASK_PACKET、future packet templates、两个 future-blocker receipts、fencing tests 和六个标准文件。
- `gates`：governance `PASS/FRESH`；每个输入有 path+bytes+SHA；当前 v74 路径只读；packet 逐个 schema 校验；Physical/Policy 均写 `OUT_OF_SCOPE + BLOCKED_EXTERNAL`。
- `threshold_rule`：优先引用现有 authority 数值及 SHA；没有时写 `UNRESOLVED_GATE`，依赖任务必须 `BLOCKED_PREREQ`。不得看候选后补门。
- `sensor_regression_rule`：在 0910 同域中排除 `play_cards_0910_053`，按固定排序选同时具备完整 VST、双 Controller、MANUS25 和可解码全片的首条结构合格会话；发布全部排除原因，并命名 `STRUCTURALLY_ELIGIBLE_REGRESSION`。只有存在绑定 SHA 的既有 H4/视频通过证据时才可称 `NORMAL_REGRESSION`；否则不得猜 `001`。
- `canary_freeze_rule`：Poker245、Chips039、Poker243、Chips023、0910_053、“4帧/24帧”和 prospective Chips/Poker 必须展开为绝对路径、session UID、presentation-order frame index、用途和 input SHA；不得只保存简称。
- `robot_prospective_rule`：从 current Robot-ready matrix 中排除所有既有 Robot canary/regression/review manifest 已引用 session，再按 canonical session ID 升序分别选择首条 Chips 与 Poker immutable-input session。若任一任务没有未见样例，写 `UNRESOLVED_GATE`。在每条选定 session 的共同有效帧集合上，4 帧使用 global valid-index quantile `round(i*(N-1)/3), i=0..3`，24 帧使用 `round(i*(N-1)/23), i=0..23`；去重后不足目标数则 `BLOCKED_REFERENCE_PROOF`。选择在任何推理前发布。
- `future_blockers`：生成 `PHYSICAL-ROBOT-FUTURE.json` 与 `REAL-POLICY-FUTURE.json`，均为 `phase_scope=OUT_OF_SCOPE`、`status=BLOCKED_EXTERNAL`、`blocks_current_visual_pipeline=false`，但不生成可领取 task packet。
- `stop_condition`：任何 input SHA、写集、门限来源或 schema 不闭合即 `BLOCKED_REFERENCE_PROOF`。
- `forbidden`：重跑基线、改算法、改 current/registry、伪造权重或 calibration。
- `claim_limit`：执行合同和内部 provenance，不产生算法质量结论。

### EXEC-01：因果性与 pixel-source legality 测试框架

- `prerequisites`：`EXEC-00=PASSED`。
- `objective`：验证 Mask、donor、atlas、ProPainter、Robot placement、Robotized RGB 和 Visual Aux current-state 的 prefix causality，并证明 loader 真正消费 `training_valid_mask`。
- `read_set`：EXEC-00 receipt/freeze、`src/chaoyang/pipeline/causal_robotized_compositor_v1.py`、`src/chaoyang/pipeline/robot_clean_compositor.py`、HumanEgo bundle builder/validator、baseline registry。
- `artifact_write_set`：仅 `exec_01/`。
- `source_write_set`：新增 `src/chaoyang/ops/validate_visual_pipeline_causality_v1.py` 及对应测试文件。
- `unique_experimental_variable`：只增加测试/provenance，不改算法输出。
- `budgets`：CPU 21,600，GPU 0，wall 43,200 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/validate_visual_pipeline_causality_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/exec_01/TASK_PACKET.json --dry-run
python src/chaoyang/ops/validate_visual_pipeline_causality_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/exec_01/TASK_PACKET.json --canary --future-offset-frames 15 --resume
```

- `output_contract`：`CAUSAL_PREFIX_MUTATION_REPORT.json`、`PIXEL_SOURCE_SCHEMA.json`、`TRAINING_VALID_MASK_CONSUMPTION.json`、offline/causal loader 拒绝测试和六个标准文件。
- `gates`：修改 `t` 后的数据不得改变训练分支在 `t` 的任何输入 SHA；每个像素有 source-frame 上界或 UNKNOWN；invalid critical pixel 必须令 H50 Raw/Robotized 成对排除。
- `stage_assertions`：分别测试 Mask anchor/reseed 不能从未来选择、donor/atlas source frame≤t、ProPainter prefix-only 或 invalid、Robot pose 不受全段 trajectory optimization/smoothing 反向改变、placement 不使用未来、loader 不读取 offline sidecar。不能只比较最终 RGB SHA。
- `placement_rule`：完整 session 选择的 fixed base 不算 causal。训练只允许冻结 task preset 或 prefix-only lock；lock 前帧 invalid、不回填。full-session optimized placement 只用于 offline visual。
- `stop_condition`：任一阶段不能提供来源或帧上界，正式训练分支 `BLOCKED_REFERENCE_PROOF`。
- `forbidden`：仅改名为 causal、使用未来样本、只写 mask 而 loader 忽略。
- `claim_limit`：因果/来源内部一致性，不是视觉或物理真值。

### MASK-10A：Controller 几何提示与独立角色 Mask

- `prerequisites`：`EXEC-00=PASSED`；与 exact78 主线独立。
- `objective`：用 Controller 几何提示建立独立 glove/forearm/controller/accessory/object roles，不改变 pinned segmentation model。
- `frozen_canary`：失败压力样例 `play_cards_0910_053`；正常回归只读 `FROZEN_SENSOR_H4_REGRESSION.json`。
- `roles`：left/right glove、left/right forearm、left/right controller、cable/strap/accessory、独立 task-object instances。
- `read_set`：EXEC freeze/regression、现有 sensor H4 packet、H0 ledger、SAM3.1 adapter、Controller/MANUS schema。
- `artifact_write_set`：仅 `mask_10a/`。
- `source_write_set`：新增 `src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py`、`src/chaoyang/pipeline/sensor_role_mask_geometry_v2.py`，不得改 v1。
- `unique_experimental_variable`：只加 Controller 几何 ROI/点/框；高分辨率 crop 不在本任务。
- `budgets`：CPU 14,400，GPU 43,200，wall 172,800 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/mask_10a/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/mask_10a/TASK_PACKET.json --canary --resume
python src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/mask_10a/TASK_PACKET.json --regression --resume
python src/chaoyang/ops/run_sensor_h4_geometry_prompt_mask_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/mask_10a/TASK_PACKET.json --full-batch --resume
```

- `output_contract`：每角色 mask/track、最终 delete mask、全片诊断视频、左右/离屏/重入/对象保护/附件支持报表、六个标准文件。
- `gates`：task object 独立 ID；Controller box 只用于搜索，不能作 mask/矩形擦除；附件不要求与 glove 连通；细结构不被 largest/small-component filter 丢弃；离屏 empty；MANUS25 原样；optical hand `isActive=0` 不得生成 PICO21。
- `runtime_attempt_rule`：attempt 2 只能重跑相同 implementation/input/config signature；不能增加 crop 或改提示。
- `stop_condition`：canary 仍漏附件则本任务 `FAILED_QUALITY_C`，并在 NEXT_ACTION 建议 `MASK-10B`；不得自动启动或用大膨胀包住。
- `forbidden`：因为共运动就删任务物体；把内部支持率称 segmentation accuracy；无 Gold 声称外部精度。
- `claim_limit`：sensor-domain development mask evidence only。

`MASK-10B` 是未来模板：仅在 `MASK-10A=FAILED_QUALITY_C` 且 failure code 明确为 `THIN_ACCESSORY_UNDERSAMPLING` 时，允许用户另行授权“原分辨率局部 crop”单变量任务。它还应先 profile pinned SAM adapter 是否支持共享多对象 memory、共享 backbone 推理是否节省成本；未核实前不能把共享 memory 写成已有能力。本轮不物化该 packet。

### CLEAN-10B：角色编排与删除/流/写入域审计

- `prerequisites`：`EXEC-00=PASSED`。
- `frozen_sessions`：失败 Poker245、Chips039；回归 Poker243、Chips023。
- `objective`：相对冻结 A baseline，只引入分角色编排并显式输出 `M_remove/M_flow/M_write`，定位残留/误擦或 Raw 回填层。
- `read_set`：EXEC freeze、现有 Clean exploration packet/result、Clean wrapper、exact78 Clean preparer、pinned ProPainter、baseline registry。
- `artifact_write_set`：仅 `clean_10b/`。
- `source_write_set`：新增 `src/chaoyang/ops/run_clean_role_write_domain_canary_v2.py`、`src/chaoyang/pipeline/clean_role_write_domain_v2.py`。
- `unique_experimental_variable`：只从 union/旧编排变为分角色编排；接触边界、donor、权重、ProPainter 和编码固定。
- `budgets`：CPU 10,800，GPU 3,600，wall 21,600 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_clean_role_write_domain_canary_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10b/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_clean_role_write_domain_canary_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10b/TASK_PACKET.json --canary --variant B --resume
python src/chaoyang/ops/run_clean_role_write_domain_canary_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10b/TASK_PACKET.json --regression --variant B --resume
```

- `variants`：A=冻结旧 baseline；B=仅分角色编排。本任务不得实现 C、donor 或 atlas。
- `output_contract`：审核帧 Raw、role masks、`M_remove/M_flow/M_write`、Clean；若 EXEC-00 冻结了同帧 immutable Robotized artifact 则并列展示，否则明确 `ROBOTIZED_LAYER=NOT_APPLICABLE`；dilation 调用图；逐层差分；六个标准文件。
- `gates`：当前可见 object retention≥99.9% 且报告 coverage；编码前数组在 `M_write` 外 byte-exact；回归不退化门从 EXEC-00 读取；残留、误擦、UNKNOWN 分开。
- `dilation_rule`：upstream CLI 默认值候选为 4；current wrapper 的实际生效值保持 `UNKNOWN`，必须由 argv/config/call graph 审计。`M_flow` 可扩大，`M_write` 外不得改变。双向 ProPainter 只能用于 offline diagnostic；若非 prefix-only causal，对应训练像素必须 invalid/UNKNOWN。
- `stop_condition`：若根因是上游漏 mask 或下游 Raw 回填，写明确 handoff；不得继续膨胀掩盖。门失败即 `FAILED_QUALITY_C`。
- `forbidden`：只凭删除面积下降宣称修好；Clean 像素反喂 Depth/Object6D/Contact/control。
- `claim_limit`：Clean 写域和可见像素内部一致性，不是隐藏真值。

### CLEAN-10C：自适应接触边界

- `prerequisites`：`CLEAN-10B=PASSED`。
- `objective`：在冻结 B 上只加入自适应接触边界，其他输入、donor、ProPainter、编码完全不变。
- `read_set`：EXEC freeze/canary frames、CLEAN-10B receipt+manifest、现有 Clean exploration packet、pinned wrapper/config、regression manifest。
- `artifact_write_set`：仅 `clean_10c/`。
- `source_write_set`：新增 `src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py`、`src/chaoyang/pipeline/clean_adaptive_contact_boundary_v1.py`。
- `unique_experimental_variable`：adaptive contact boundary。
- `budgets`：CPU 10,800，GPU 3,600，wall 21,600 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10c/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10c/TASK_PACKET.json --canary --resume
python src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10c/TASK_PACKET.json --regression --resume
python src/chaoyang/ops/run_clean_adaptive_contact_boundary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_10c/TASK_PACKET.json --full-batch --resume
```

- `output_contract`：Clean C、三域 sidecars、逐层诊断、完整 review video 和标准六文件。
- `gate_source`：`VISIBLE_OBJECT_PIXEL_RETENTION_GE_99_9_PERCENT`、`CONTACT_BAND_ADDED_REMOVAL_REDUCTION_GE_50_PERCENT`、`OUTSIDE_AUTHORIZED_BAND_BYTE_EXACT` 来自 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/TASK_PACKET.json`；EXEC-00 必须绑定其 bytes+SHA。若该 packet 不具备可沿用的计划权威，则将数值写 `UNRESOLVED_GATE`，不得硬编码。
- `gates`：上述冻结门、两失败 canary 改善、两回归不退化；删除面积减少不能单独过门。
- `stop_condition`：任何 gate C 即 `FAILED_QUALITY_C`；算法变化必须新 revision，runtime retry 只复用同 signature。
- `claim_limit`：adaptive-boundary Clean C development candidate。

### DONOR-10K：背景 surface-aware causal donor kernel

- `prerequisites`：`EXEC-01=PASSED`；不依赖 Clean，可与 CLEAN-10B/C 并行。
- `objective`：在 synthetic/frozen correspondence 上实现背景 donor kernel；不接入 Clean。
- `read_set`：EXEC receipt、pixel-source schema、`src/chaoyang/pipeline/deterministic_table_reprojection_clean.py`、Depth/registration authority receipt、frozen canary frames。
- `artifact_write_set`：仅 `donor_10k/`。
- `source_write_set`：新增 `src/chaoyang/pipeline/causal_surface_donor_v1.py`、`tests/test_causal_surface_donor_v1.py`。
- `unique_experimental_variable`：背景 donor 来源。
- `budgets`：CPU 21,600，GPU 7,200，wall 43,200 秒，`attempt_max=2`。
- `command_argv`：

```bash
python -m pytest -q tests/test_causal_surface_donor_v1.py
python src/chaoyang/pipeline/causal_surface_donor_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/donor_10k/TASK_PACKET.json --dry-run
```

- `allowed_sources`：经 authority 验证的平面单应、可见性闭合的 Depth 重投影、经验证光流。普通背景可有明确标记的 synthetic residual；关键任务物体/接触区不允许。
- `output_contract`：known-answer donor arrays、完整 source maps、causality/visibility report、unit-test receipt 和标准六文件。
- `gates`：source map coverage=100%（含 UNKNOWN）；known-answer wrong-surface donor=0；future mutation invariant；同时报告 fidelity、supported coverage、UNKNOWN ratio。
- `stop_condition`：表面、registration 或 correspondence 无法确认即 UNKNOWN，不得猜。
- `forbidden`：仅因整数像素相同/颜色相近就判同一表面；使用未来帧。
- `claim_limit`：合法因果背景来源的内部一致性。

### DONOR-11：同一时刻右眼 donor（未来条件模板）

- `trigger`：目标 session 的 H3 stereo/sync/registration 全部 authority-valid，且 DONOR-10K/CLEAN-20 仍因左眼不可见产生明确 background 或 same-instance task-object 缺口；两种 surface 必须是独立子实验和 claim。
- `status_when_not_triggered`：`CANCELLED/CONDITION_NOT_TRIGGERED`。
- `unique_experimental_variable`：只加入 same-time right-eye donor。
- `budgets`：CPU 7,200，GPU 0，wall 14,400 秒，`attempt_max=1`。
- `gates`：右→左 registration 同帧、同分辨率、同 optical frame；遮挡一致；所有像素记录 right-eye source index。
- `forbidden`：对 H3 C、缺 sidecar 或借其他会话 calibration 的数据启用。
- `claim_limit`：registered stereo donor development evidence。

本轮 `DONOR-11` 不物化、不可领取；上述内容只用于以后生成独立 packet。

### ATLAS-10：Poker 因果对象 atlas

- `prerequisites`：`EXEC-00=PASSED`、`EXEC-01=PASSED`；明确不依赖 Clean。
- `objective`：由 Raw/Object evidence 为同一张牌、同一面建立截至 `t` 的 causal homography atlas。
- `read_set`：EXEC freeze/causality receipts、Object mask/identity authority、Raw manifest、Depth/registration receipt（若使用）、regression manifest。
- `artifact_write_set`：仅 `atlas_10/`。
- `source_write_set`：新增 `src/chaoyang/pipeline/causal_poker_atlas_v1.py`、`src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py`。
- `unique_experimental_variable`：从无 atlas 变为 same-instance/same-face causal atlas。
- `budgets`：CPU 28,800，GPU 0，wall 57,600 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/atlas_10/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/atlas_10/TASK_PACKET.json --canary --resume
python src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/atlas_10/TASK_PACKET.json --regression --resume
python src/chaoyang/ops/run_causal_poker_atlas_canary_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/atlas_10/TASK_PACKET.json --full-batch --resume
```

- `output_contract`：per-time atlas、per-pixel source frame、instance/face ID、RANSAC inliers、reprojection residual、visibility、UNKNOWN、完整回放视频和标准六文件。
- `gates`：source frame≤t；不混牌、不混正反面；当前 Raw 可见像素优先 byte-exact；重入恢复≤2帧；future mutation invariant；residual 门只读 EXEC-00 freeze，若 unresolved 则 `BLOCKED_PREREQ`。
- `stop_condition`：弯折、身份歧义或 residual 超门时停止传播并 UNKNOWN。
- `forbidden`：把 Chips 可形变包装套 rigid homography/SE(3)；用其他牌补纹理。
- `claim_limit`：Poker 同实例表面因果重建，不是隐藏外观真值。

### CLEAN-20：Clean C 与背景 donor 集成（D1）

- `prerequisites`：`CLEAN-10C=PASSED`、`DONOR-10K=PASSED`、`EXEC-01=PASSED`。
- `objective`：将已通过单元测试的背景 donor kernel 接入冻结 Clean C。
- `read_set`：三个 prerequisite receipts、CLEAN-10C manifest、DONOR-10K manifest、frozen canary frames、regression manifest。
- `unique_experimental_variable`：只把 DONOR-10K 接入冻结 Clean C，其他固定。
- `artifact_write_set`：仅 `clean_20/`。
- `source_write_set`：新增 `src/chaoyang/ops/run_clean_c_background_donor_v1.py`。
- `budgets`：CPU 14,400，GPU 7,200，wall 28,800 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_clean_c_background_donor_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_20/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_clean_c_background_donor_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_20/TASK_PACKET.json --canary --resume
python src/chaoyang/ops/run_clean_c_background_donor_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_20/TASK_PACKET.json --regression --resume
python src/chaoyang/ops/run_clean_c_background_donor_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_20/TASK_PACKET.json --full-batch --resume
```

- `output_contract`：完整 Clean D1、pixel-source map、full-session review video、标准六文件。
- `gates`：CLEAN-10C 全部门继续成立；DONOR-10K source legality 成立；失败 canary 改善且两个回归不退化；全片前通过冻结 24 帧 canary。
- `stop_condition`：任一 prerequisite SHA、causal source 或回归门失败即明确终态，不扩量。
- `claim_limit`：background-aware Clean development candidate，不是 authority。

### CLEAN-21：Clean D 与 Poker atlas 集成

- `prerequisites`：`CLEAN-20=PASSED`、`ATLAS-10=PASSED`、`EXEC-01=PASSED`。
- `objective`：仅对 Poker 将合法 causal object atlas 接入 Clean D1，形成 D2；Chips 不消费本任务。
- `read_set`：三个 prerequisite receipts、CLEAN-20/ATLAS manifests、frozen Poker canary frames、regression manifest。
- `unique_experimental_variable`：只加入 Poker object atlas；Chips 保持 CLEAN-20，不套 rigid atlas。
- `artifact_write_set`：仅 `clean_21/`。
- `source_write_set`：新增 `src/chaoyang/ops/run_clean_d_poker_atlas_v1.py`。
- `budgets`：CPU 14,400，GPU 7,200，wall 28,800 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/run_clean_d_poker_atlas_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_21/TASK_PACKET.json --dry-run
python src/chaoyang/ops/run_clean_d_poker_atlas_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_21/TASK_PACKET.json --canary --resume
python src/chaoyang/ops/run_clean_d_poker_atlas_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_21/TASK_PACKET.json --regression --resume
python src/chaoyang/ops/run_clean_d_poker_atlas_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/clean_21/TASK_PACKET.json --full-batch --resume
```

- `output_contract`：Poker Clean D2、object/background source maps、UNKNOWN/training-valid masks、full-session review、标准六文件。
- `gates`：可见 Raw object 优先 byte-exact；无合法来源为 UNKNOWN；pixel source=100%；future mutation；failure 改善+regression 不退化；invalid critical pixels 能被 loader 排除。
- `stop_condition`：若 loader 不能消费 pixel mask，包含 invalid critical pixel 的 Raw/Robotized H50 窗口成对剔除。
- `claim_limit`：合法来源受控的 Clean D candidate。

### COMP-10：Ownership compositor 已知答案单元测试

- `prerequisites`：`EXEC-00=PASSED`、`EXEC-01=PASSED`。
- `objective`：不改 Robot/Object geometry，只验证 z/ownership 核心规则。
- `read_set`：EXEC receipts、现有 occlusion/causal compositor、Robot RGBA/Z schema、Object optical-Z schema。
- `artifact_write_set`：仅 `comp_10/`。
- `source_write_set`：新增 `src/chaoyang/pipeline/ownership_compositor_v2.py`、`tests/test_ownership_compositor_v2.py`、`src/chaoyang/ops/run_robot_object_ownership_integration_v1.py`。
- `unique_experimental_variable`：ownership kernel。
- `budgets`：CPU 10,800，GPU 0，wall 21,600 秒，`attempt_max=2`。
- `command_argv`：

```bash
python -m pytest -q tests/test_ownership_compositor_v2.py
python src/chaoyang/pipeline/ownership_compositor_v2.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/comp_10/TASK_PACKET.json --dry-run
```

- `output_contract`：逐像素 `BACKGROUND/OBJECT_FRONT/ROBOT_FRONT/TIE_UNKNOWN`、known-answer fixtures、z convention report 和六个标准文件。
- `gates`：只在实际重叠像素比较；Z 同帧/同网格/同 optical-Z；invalid/过近/冲突为 `TIE_UNKNOWN`；非重叠可见 object 不因 depth invalid 被擦；桌沿/盘子可作可信场景遮挡；已删除 human/controller 不能遮挡 Robot；tie 不逐帧翻转。
- `threshold_rule`：tie epsilon 从 EXEC-00 读取；若 unresolved，kernel 性质测试可结束，真实集成 `BLOCKED_PREREQ`。
- `stop_condition`：性质测试失败为 `FAILED_QUALITY_C`；Z frame/grid/convention 不闭合为 `BLOCKED_REFERENCE_PROOF`。
- `forbidden`：永远将 object 置顶；用绘制顺序掩盖数字穿透。
- `claim_limit`：synthetic known-answer software correctness，不是外部几何真值。

### COMP-20C / COMP-20P：按任务独立的真实片段集成

两个 packet 使用同一版本化实现 `src/chaoyang/ops/run_robot_object_ownership_integration_v1.py`，但输入、目录、receipt 和终态完全独立。

- `prerequisites`：COMP-20C=`COMP-10/CLEAN-20/ROBOT-11-SELECT PASSED`；COMP-20P=`COMP-10/CLEAN-21/ATLAS-10/ROBOT-11-SELECT PASSED`。

| 字段 | COMP-20C | COMP-20P |
|---|---|---|
| objective | Chips Robot/Object ownership | Poker Robot/Object ownership |
| prerequisites | COMP-10、CLEAN-20、ROBOT-11-SELECT | COMP-10、CLEAN-21、ATLAS-10、ROBOT-11-SELECT |
| path_slug | `comp_20c` | `comp_20p` |
| Clean input | D1 | Poker D2 |
| atlas | 不适用 | 必需 |

- `read_set`：每个 packet 只能展开本任务 prerequisite receipts/manifests、Robot RGBA/Z/part-id、Object visible Z/instance-id 和 Clean pixel-source map，总数≤8；所有 logical ref 由 EXEC-00 映射为精确 path+bytes+SHA。
- `artifact_write_set`：分别仅 `comp_20c/`、`comp_20p/`。
- `source_write_set`：两个任务均为空；共同只读 COMP-10 的 `IMPLEMENTATION_RECEIPT`，不存在 Chips→Poker 隐式依赖。
- `unique_experimental_variable`：只替换 compositor；Robot solve、Clean、Object evidence 固定。
- `budgets`（两个 packet 各自）：CPU 21,600，GPU 0，wall 43,200 秒，`attempt_max=2`。
- `command_argv`：各 materialized TASK_PACKET 分别冻结以下 modes 的完整 argv，并由 RUN_SIGNATURE 绑定：

```bash
python src/chaoyang/ops/run_robot_object_ownership_integration_v1.py --task-packet <COMP-20C-or-P-absolute-packet> --dry-run
python src/chaoyang/ops/run_robot_object_ownership_integration_v1.py --task-packet <COMP-20C-or-P-absolute-packet> --canary --resume
python src/chaoyang/ops/run_robot_object_ownership_integration_v1.py --task-packet <COMP-20C-or-P-absolute-packet> --regression --resume
python src/chaoyang/ops/run_robot_object_ownership_integration_v1.py --task-packet <COMP-20C-or-P-absolute-packet> --full-batch --resume
```

尖括号只允许由 materializer 写入 TASK_PACKET 时替换，worker 不得手工执行本文占位命令。

- `output_contract`：对应任务完整 Robotized RGB、ownership/Z/part-id sidecars、全片视频和标准六文件。
- `gates`：Clean object 误擦与 Robotized 前后关系分报；`OBJECT_FRONT` 无合法纹理仍 UNKNOWN；禁止 Raw diff 回填真人；Robot pixel 有 part-id/Z/ownership；全片解码且 sidecar 帧一致。
- `stop_condition`：一个任务失败只终止自身，不阻塞另一个任务。
- `claim_limit`：digital visual ownership candidate；`control_ground_truth=false`。

### ROBOT-10：目标、Contact、hard/soft 门和穿透来源审计

- `prerequisites`：`EXEC-00=PASSED`；不依赖 Clean。
- `objective`：重建 human wrist/object→Robot target 链，区分 hard geometry、soft morphology 和 compositor 错误；不换 solver、不调目标。
- `read_set`：EXEC freeze、EXEC-00 冻结的 immutable Robot receipt、baseline registry、Robot result schema、Object/Contact receipt、端到端算法文档。
- `artifact_write_set`：仅 `robot_10/`。
- `source_write_set`：新增 `src/chaoyang/ops/audit_robot_target_contact_v1.py`。
- `unique_experimental_variable`：无，只审计。
- `budgets`：CPU 21,600，GPU 0，wall 43,200 秒，`attempt_max=1`。
- `command_argv`：

```bash
python src/chaoyang/ops/audit_robot_target_contact_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_10/TASK_PACKET.json --dry-run
python src/chaoyang/ops/audit_robot_target_contact_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_10/TASK_PACKET.json --canary
```

- `output_contract`：wrist/object 相对运动、placement 来源/因果性、hard/soft gate 分表、object-centric contact hypothesis、逐帧 `GEOMETRY_PENETRATION/COMPOSITOR_ORDER/UNKNOWN` 和标准六文件。
- `contact_schema`：object instance、finger/tip/pad、phase、relative slip、evidence source、uncertainty、observed/tracked/attachment/unknown；一律 `HYPOTHESIS_ONLY`。
- `gates`：target lineage 可重建；fixed placement 输入上界明确；硬门只用 current registry 数值；sidecar `control_ground_truth=false`。
- `stop_condition`：lineage 不闭合即 `BLOCKED_REFERENCE_PROOF`；attachment 不得反向证明 Contact/Object6D。
- `v74_rule`：若 v74 尚无 immutable terminal，有限等待由 EXEC-00 记录；超出预算后本任务 `BLOCKED_PREREQ`，不得读取其 partial/可变 state。
- `coordinate_regression`：presentation-order frame binding、selected-left/sourceIndex0、无新增 radial warp、REP-103→OpenXR→camera c2w、meter/xyzw/row-major 和 rectified→selected registration 必须绑定 regression manifest SHA。
- `claim_limit`：digital geometry/target diagnostics and contact hypotheses only。

### ROBOT-20：分阶段耗时与合格产率 profiling

- `prerequisites`：`EXEC-00=PASSED`；不依赖 Clean。
- `objective`：只插桩，不改算法；找出合格产出瓶颈。
- `read_set`：EXEC freeze、immutable Robot receipt、Robot runner、Robot validator、regression manifest。
- `artifact_write_set`：仅 `robot_20/`。
- `source_write_set`：新增 `src/chaoyang/ops/profile_robot_pipeline_v1.py`。
- `unique_experimental_variable`：profiling instrumentation。
- `budgets`：CPU 43,200，GPU 0，wall 86,400 秒，`attempt_max=1`。
- `command_argv`：

```bash
python src/chaoyang/ops/profile_robot_pipeline_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_20/TASK_PACKET.json --dry-run
python src/chaoyang/ops/profile_robot_pipeline_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_20/TASK_PACKET.json --canary --resume
```

- `output_contract`：IO/资产、placement、arm IK、hand R1/R2、collision、render、composite/encode、等待/重跑；wall/session、peak VRAM、qualified Robot frames/hour、valid paired H50 windows/hour、cache hit/miss 和标准六文件。
- `profiling_probes`：额外量化资产/渲染场景常驻、前帧解 warm-start、共享初始化的潜在收益，但不在本任务启用优化。warm-start 在 tracking loss、长时无观测或 segment 边界必须失效。
- `gates`：阶段计时与 wall 闭合；质量过滤前后分母明确；不能只报 seconds/frame。
- `stop_condition`：immutable Robot terminal 不存在时有限等待后 `BLOCKED_PREREQ`；计时无法覆盖全 wall 或分母不闭合时 `BLOCKED_REFERENCE_PROOF`。
- `claim_limit`：runtime/internal throughput profile。

### ROBOT-11-SELECT：冻结可用于视觉集成的 Digital Robot receipt

- `prerequisites`：`ROBOT-10=PASSED`，且 EXEC-00 指定的 Robot run 已有 immutable terminal。
- `objective`：从已有结果中按 current hard gates 选择一个可复现 Digital Robot candidate，或诚实给出无合格候选的负面终态；不重新求解。
- `read_set`：EXEC freeze、ROBOT-10 receipt、frozen Robot run receipt/result/manifest、baseline registry、regression manifest。
- `artifact_write_set`：仅 `robot_11_select/`。
- `source_write_set`：新增 `src/chaoyang/ops/select_digital_robot_candidate_v1.py`。
- `unique_experimental_variable`：无，只选择/冻结。
- `budgets`：CPU 7,200，GPU 0，wall 14,400 秒，`attempt_max=1`。
- `command_argv`：

```bash
python src/chaoyang/ops/select_digital_robot_candidate_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_11_select/TASK_PACKET.json --dry-run
python src/chaoyang/ops/select_digital_robot_candidate_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_11_select/TASK_PACKET.json --canary
python src/chaoyang/ops/select_digital_robot_candidate_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/robot_11_select/TASK_PACKET.json --verify --resume
```
- `output_contract`：`SELECTED_DIGITAL_ROBOT_RECEIPT.json`、hard/soft gate table、coordinate/visual regression result 和标准六文件。
- `gates`：finite proper SE(3)、joint limits、fixed world base、current collision/hard gates、完整帧身份；soft morphology 另报，不混入 hard gate；`control_ground_truth=false`。
- `stop_condition`：没有 hard-feasible immutable candidate 即 `FAILED_QUALITY_C`，并由 NEXT_ACTION 指向 `ROBOT-12-TARGET-SUCCESSOR` 或相应条件模板。
- `claim_limit`：Digital Robot visual candidate selection only。

`ROBOT-12-TARGET-SUCCESSOR` 为未来模板：只有 ROBOT-10 以机器可读 failure code 证明“human wrist/object 相对运动未被 target 链保留”，且用户再次授权时，才允许以 target construction 为唯一变量创建 packet。它不得与 base search、solver 或 hand round-2 同时改变。

### ROBOT-21：solve/render/composite 缓存解耦（未来条件模板）

`ROBOT-12/21/22/23/24` 均为 `PACKET_TEMPLATE_ONLY`。EXEC-00 的 `TRIGGER_PREDICATES.json` 必须为“主要”“显著”“可接受”给出在看结果前冻结的公式、分母、阈值、source receipt 和 evaluator；无法引用 authority 时可标 `PLAN_LOCAL_DEVELOPMENT_GATE`，但必须由用户先批准其数值。predicate 为真只产生 `ELIGIBLE_TO_DISPATCH`，不能自动运行。

- `trigger`：ROBOT-20 证明重复 solve/render/composite 或 I/O 是主要可避免成本，并由 `NEXT_ACTION.json` 明确触发。
- `objective`：将 solve、render、composite/encode 分成独立 immutable cache。
- `cache_key`：wrist/hand/object、calibration-or-assumption、solver/config、asset、contact/geometry、code/schema SHA 全覆盖；接触或几何变化必须 invalidate。
- `budgets`：CPU 28,800，GPU 0，wall 57,600 秒，`attempt_max=2`。
- `gates`：cache hit 输出逐字节或允许误差内相同；stale key 拒绝；hard quality 不退化；提升门只读 EXEC-00。
- `forbidden`：仅按 session 名缓存；Clean/compositor 改动触发 IK 重算。

### ROBOT-22：渐进 base search（未来条件模板）

- `trigger`：ROBOT-10 显示 base search 是 hard-feasible/前伸不足主因，且 ROBOT-20 证明预算可接受。
- `unique_experimental_variable`：coarse→local progressive base search。
- `prospective_canary`：EXEC-00 冻结未参与设计的 Chips+Poker 各一条，不得只重放既知六段 Chips。
- `budgets`：CPU 28,800，GPU 0，wall 57,600 秒，`attempt_max=2`。
- `gates`：hard feasibility 不退化；fixed world base；保留人到物体的相对前伸；训练 placement 只能 preset 或 prefix lock。

### ROBOT-23：hand round-2 条件触发（未来条件模板）

- `trigger`：ROBOT-20 显示 round-2 是显著成本，且 ROBOT-10 能用 round-1 指标预测所需帧/段。
- `unique_experimental_variable`：只改变 round-2 调用条件。
- `budgets`：CPU 21,600，GPU 0，wall 43,200 秒，`attempt_max=2`。
- `gates`：hard/soft 质量相对全量 round-2 不退化；触发/未触发均有原因；prospective Chips+Poker 通过。

### ROBOT-24：cuRobo A/B（未来严格条件模板）

- `trigger`：仅当 ROBOT-10 证明当前 IK/collision 形成 hard failure，或 ROBOT-20 证明正确 IK/collision 是主要耗时；同时 third-party code/weight/license/CUDA receipt 闭合。
- `status_when_not_triggered`：`CANCELLED/CONDITION_NOT_TRIGGERED`。
- `unique_experimental_variable`：solver；目标、资产、placement、hand、门和 canary 固定。
- `budgets`：CPU 21,600，GPU 14,400，wall 43,200 秒，`attempt_max=2`。
- `gates`：同一 prospective Chips+Poker；hard feasibility 不退化；合格产率达到冻结提升门；不能称物理可部署。

### DATA-10-IMPL：配对 ledger 共享实现与单元测试

- `prerequisites`：`EXEC-01=PASSED`。
- `objective`：实现统一 paired-ledger builder 和 loader/valid-mask 单元测试，不读取 Chips/Poker full data、不生成训练 ledger。
- `read_set`：EXEC-01 receipt/pixel-source schema、现有 HumanEgo bundle builder/validator/trainer、training policy、task packet schema。
- `artifact_write_set`：仅 `data_10_impl/`。
- `source_write_set`：新增 `src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py`、`tests/test_visual_aux_paired_ledger_v1.py`。
- `unique_experimental_variable`：pairing/valid-mask implementation only。
- `budgets`：CPU 10,800，GPU 0，wall 21,600 秒，`attempt_max=2`。
- `command_argv`：`python -m pytest -q tests/test_visual_aux_paired_ledger_v1.py`，以及 builder 的 `--dry-run` schema smoke；无 full-batch。
- `output_contract`：IMPLEMENTATION_RECEIPT、pairing/causal/invalid-mask tests 和标准六文件。
- `gates`：Raw/Robotized key 一一对应；future-2D/valid/split/seed/config 不可因 RGB domain 改变；invalid critical pixel 成对删除测试通过。
- `stop_condition`：任一性质测试失败为 `FAILED_QUALITY_C`。
- `claim_limit`：paired-ledger software correctness only。

### DATA-10C / DATA-10P：按任务独立的严格配对 H50 ledger

- `prerequisites`：DATA-10C=`DATA-10-IMPL/COMP-20C PASSED`；DATA-10P=`DATA-10-IMPL/COMP-20P PASSED`。

| 字段 | DATA-10C | DATA-10P |
|---|---|---|
| objective | 冻结 Chips Raw/Robotized pair | 冻结 Poker Raw/Robotized pair |
| prerequisites | DATA-10-IMPL、COMP-20C | DATA-10-IMPL、COMP-20P |
| path_slug | `data_10c` | `data_10p` |

- `read_set`：每项只含本分支 prerequisite receipts、Raw authority ledger、对应 Robotized ledger、HumanEgo bundle builder/validator 与冻结训练 policy，总数≤8。
- `artifact_write_set`：分别仅 `data_10c/`、`data_10p/`。
- `source_write_set`：两个任务均为空；共同只读 DATA-10-IMPL implementation receipt。
- `unique_experimental_variable`：无，只选择和冻结。
- `budgets`（两个 packet 各自）：CPU 14,400，GPU 0，wall 28,800 秒，`attempt_max=2`。
- `command_argv`：materialized TASK_PACKET 将 `<TASK>` 和 `<PACKET>` 写为 Chips/`data_10c` 或 Poker/`data_10p` 的绝对值，worker 不得手填：

```bash
python src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py --task-packet <PACKET> --task <TASK> --dry-run
python src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py --task-packet <PACKET> --task <TASK> --canary --resume
python src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py --task-packet <PACKET> --task <TASK> --full-batch --resume
python src/chaoyang/human_ego/tools/build_visual_aux_paired_ledger_v1.py --task-packet <PACKET> --task <TASK> --verify --resume
```
- `frozen_minima_per_task`：计划目标为 16 train sessions、3 validation sessions、256 train H50 windows、48 validation H50 windows。EXEC-00 必须为这些值绑定当前 readiness/task-packet 的 path+bytes+SHA；若找不到来源，则标 `PLAN_LOCAL_DEVELOPMENT_GATE` 并等待用户批准，不能自动当 authority。
- `pairing`：同任务 Raw/Robotized 的 session、frame、future-2D、valid、split、seed、语言、config 全同；唯一差异是 RGB domain。逐侧 valid 保留。
- `invalid_rule`：若 loader 不执行 `training_valid_mask`，包含 invalid critical pixel 的 Raw/Robotized 窗口成对删除。
- `output_contract`：本任务 train/val ledger、pair SHA、drop reasons、本 pair 两条完全展开的 `TRAIN_COMMANDS.json` 和标准六文件。
- `stop_condition`：本任务未达 minima，则仅本任务两条训练 `BLOCKED_PREREQ`；另一任务独立继续。
- `claim_limit`：paired Visual Aux input eligibility，不是 policy data authority。

### 四个 TRAIN 任务

任务 ID：`TRAIN-CHIPS-RAW`、`TRAIN-CHIPS-ROBOTIZED`、`TRAIN-POKER-RAW`、`TRAIN-POKER-ROBOTIZED`。四项分别有独立 packet、run dir 和 GPU lease。

| task_id | path_slug | prerequisite | RGB domain | objective |
|---|---|---|---|---|
| TRAIN-CHIPS-RAW | `train_chips_raw` | DATA-10C | Raw | 训练 Chips Raw future-2D Visual Aux |
| TRAIN-CHIPS-ROBOTIZED | `train_chips_robotized` | DATA-10C | Robotized | 训练 Chips Robotized future-2D Visual Aux |
| TRAIN-POKER-RAW | `train_poker_raw` | DATA-10P | Raw | 训练 Poker Raw future-2D Visual Aux |
| TRAIN-POKER-ROBOTIZED | `train_poker_robotized` | DATA-10P | Robotized | 训练 Poker Robotized future-2D Visual Aux |

- `prerequisites`：对应 DATA-10C 或 DATA-10P ledger 达量且 SHA 冻结。
- `read_set`：对应 DATA receipt、ledger、`TRAIN_COMMANDS.json`、`train_visual_aux_future2d_v53.py`、bundle validator、训练 config。
- `artifact_write_set`：只写上表各自 path_slug 目录。
- `source_write_set`：无，固定使用 receipt 绑定的已有 trainer/validator；若需要改代码，必须先另建 implementation task。
- `unique_experimental_variable`：同一任务 Raw/Robotized 间只有 RGB domain。
- `budgets`（四个 packet 各自）：CPU 43,200，GPU 86,400，wall 172,800 秒，`attempt_max=3`。
- `command_argv`：DATA-10C/P 的 `TRAIN_COMMANDS.json` 必须给每项 exact `cwd/env/epoch0_validate_argv/train_argv/inference_argv/verify_argv` 及 SHA；协调者原样写进 TASK_PACKET 并由 RUN_SIGNATURE 绑定。不得人工改 argv。barrier 只在任务 pair 内。
- `pair_epoch0_protocol`：两支先各自只运行 epoch0 并写 immutable `EPOCH0_RECEIPT.json`。唯一协调者随后执行 `freeze_project_optimization_execution_v1.py --seal-epoch0-pair --raw-receipt <RAW> --robotized-receipt <ROBOTIZED> --output <PAIR_EPOCH0_BARRIER.json>`；只有两份 receipt 均 `PASSED` 且 pair SHA 相同才写 pass barrier。训练 argv 必须验证该 barrier SHA。任一支失败时，另一支有限结束为 `CANCELLED/PAIR_EPOCH0_FAILED`；不能无限等待，也不影响另一任务类别。
- `output_contract`：checkpoint、训练 receipt、sample/config SHA、ADE/FDE/PCK、coverage、identity/temporal/loss、完整配对推理视频和标准六文件。
- `gates`：manifests/legacy_data/label/split/seed/config pair SHA 一致；本 pair epoch0 两项通过；checkpoint SHA 完整；上述指标齐全。
- `stop_condition`：pair 任一 epoch0 失败时该 pair 均不训练；另一个任务 pair 不受影响。runtime 最多两次重试；质量失败不换 seed。
- `claim_limit`：single-seed future-2D `VISUAL_AUX_CHECKPOINT`；`policy_checkpoint=false`，不是 action/Policy。

### 条件论文 challenger（未来模板，本轮不物化）

| Task | 唯一触发条件 | 范围 | 未触发/失败 |
|---|---|---|---|
| CH-CUTIE | task-object 重入有冻结失败 | 一失败+两回归 canary | 保留 baseline |
| CH-PROVE | CLEAN-21 已有候选 | 仅 RC-S/RC-T 辅助排序 | 不作 Gold |
| CH-TAPNEXT | Cutie/现有重入仍失败 | 稀疏长期对应 sidecar | 不替换 Mask/Depth |
| CH-POINT2POSE | rigid+adapter+Depth/registration 有效且 atlas 失败 | 少量 rigid canary | 不用于 Chips |
| CH-FASTSTEREO | profiling 证明 Depth 是瓶颈 | 速度/质量 Pareto | 不作默认精度升级 |
| CH-CUROBO | 等同 ROBOT-24 trigger | 固定资产 solver A/B | 不提供部署 authority |

这些名字不能直接领取。触发只允许前置 `NEXT_ACTION` 产生建议；用户另行授权后，由协调者生成具备完整 read/source/artifact write sets、argv、预算、门和 output contract 的新 packet。每项先生成 repo commit、license、依赖/CUDA、weight path+SHA、入口、I/O schema、VRAM/耗时与退出条件 receipt。一次只启动一个 re-entry challenger；不自动启动 SAM2Long、SVOR、Do as I Do、EgoPHI、Dyn-HaMR、EMPIRE 等整套替换。

### GOV-99：唯一聚合、文档 currentize 和晋升决定

- `prerequisites`：分母内任务全部有唯一 immutable terminal；条件任务明确触发或 `CONDITION_NOT_TRIGGERED`。
- `objective`：只消费 receipts，验证 evidence DAG、写集、状态和 SHA，决定哪些 candidate 可晋升。普通 worker 不执行。
- `read_set`：TASK_REGISTRY、receipts 索引、authority index、baseline registry、plan revision、regression manifest、治理 schema。
- `artifact_write_set`：仅 `gov_99/`。
- `source_write_set`：新增 `src/chaoyang/ops/aggregate_project_optimization_execution_v1.py`；TASK_PACKET 还必须把允许 CAS 的每个 current status/registry/authority/canonical 文档列成精确路径，禁止宽泛目录。
- `budgets`：CPU 14,400，GPU 0，wall 28,800 秒，`attempt_max=2`。
- `command_argv`：

```bash
python src/chaoyang/ops/aggregate_project_optimization_execution_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/gov_99/TASK_PACKET.json --dry-run
python src/chaoyang/ops/aggregate_project_optimization_execution_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/gov_99/TASK_PACKET.json --verify-receipts
python src/chaoyang/ops/aggregate_project_optimization_execution_v1.py --task-packet archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_project_optimization_execution_v1/task_packets/gov_99/TASK_PACKET.json --finalize --expected-governance-revision <FROZEN_REVISION>
```

`<FROZEN_REVISION>` 只能由已授权 TASK_PACKET 物化并由 RUN_SIGNATURE 绑定，worker 不得手填。若用户只授权汇总而未授权 promotion，第三条必须替换为 `--finalize-no-promotion`。
- `output_contract`：`AGGREGATED_RESULT.json`、`EVIDENCE_DAG.json`、`PROMOTION_DECISIONS.json`、`DOCUMENT_UPDATE_MANIFEST.json` 和标准六文件。
- `gates`：证据均 path+bytes+SHA；evidence DAG 无环；不消费 partial；current v74 终态不被覆盖；同字节 final 可幂等、不同字节拒绝；结束 governance `PASS/FRESH`。
- `promotion_rule`：worker `PASSED` 只代表任务门通过。只有 GOV-99 可写 `PROMOTED` 或更新 current registry。
- `documentation`：更新 0909 canonical task card、端到端文档和下一步计划；旧流程只能由本任务标记 superseded，不删除历史 evidence。
- `stop_condition`：任何 receipt/SHA/evidence cycle/CAS mismatch 时不更新 current；GOV-99 以合同内终态结束，并在摘要字段写 `phase_outcome=COMPLETED_WITH_BLOCKERS`、`promotion_outcome=NOT_PROMOTED`。
- `claim_limit`：governance aggregation and bounded authority promotion。

## 5. 必须锁死的算法边界

### 5.1 手、Controller 与任务物体

角色 ID 独立保持至最终 merge：

```text
left/right glove
left/right forearm
left/right controller
cable/strap/accessory
task-object instances
```

共运动不等于同一删除对象。Controller pose 只给 ROI/点/框，不能给像素轮廓。细线缆在原分辨率 crop 内处理，不要求与 glove 连通。

### 5.2 Clean 三个域

```text
M_remove：语义上需要删除的区域
M_flow：允许 flow/inpainting 获得上下文的区域
M_write：最终输出允许被改写的区域
```

`M_flow` 可以大于 `M_write`，但编码前无损数组在 `M_write` 外必须与 Raw byte-exact。每帧同时输出 Raw、role mask、三个域、Clean、Robotized，用于定位错误层。

### 5.3 背景与物体 donor

- 背景只允许 verified homography、visibility-aware Depth reprojection 或 verified flow。
- 任务物体只允许 same-instance、same-face、source-frame≤t 的 causal atlas。
- 任务物体身份纹理、牌面花纹和接触关键区无合法来源时写 UNKNOWN，不能用生成纹理降低 UNKNOWN。普通背景 residual 可以 synthetic completion，但必须记录 source/synthesis type，且不得声称真实恢复。
- 同帧右眼只在 DONOR-11 的 H3 合法条件下使用。

### 5.4 Robot geometry 与 compositor

先判断 Robot 是否在数字空间真实穿透，再判断绘制顺序是否错误。ownership 只在重叠像素比较同定义 optical-Z；不确定即 `TIE_UNKNOWN`。Object 不能永远置顶，已删除的 human/controller 也不能遮挡 Robot。

### 5.5 Contact、Depth 与 claim

Contact sidecar 是 hypothesis。FoundationStereo confidence 不得伪造；Depth/Object6D 数值不得升级成外部毫米真值。证据方向只允许：

```text
direct/tracked object → contact hypothesis → attachment hypothesis
```

attachment 不得反向证明 Contact、Object6D、Gold 或 tactile。

### 5.6 因果训练

训练时刻 `t` 只能消费 source frame≤t。后验 motion correlation 可写 latency diagnostics，但必须 `applied_to_timestamps=false`；不得移动已经生成的时间轴。Clean 只服务视觉，不反喂 Depth、Object6D、Contact 或 control。

## 6. 验收矩阵

### 6.1 Canary 顺序

```text
Clean：Poker245 + Chips039（失败）→ Poker243 + Chips023（回归）→ 全片
Sensor H4：Poker0910_053（压力）→ EXEC-00 冻结同域回归 → 全片诊断
Robot：冻结4帧 → 24帧 → 全片；prospective Chips+Poker 各至少一条
Compositor：synthetic known-answer → 冻结真实片段 → 全片
Training：四项 epoch0 validate-only → 单GPU正式训练 → 成对推理视频
```

### 6.2 不允许混为一个指标

- Clean visible-object retention 与 Robotized occlusion correctness 分开。
- geometry penetration 与 compositor order 分开。
- seconds/frame 与 qualified frames/hour 分开。
- offline nearest 与 causal input 分开。
- supported coverage、fidelity 和 UNKNOWN ratio 分开。
- hard geometry/collision/limits 与 soft morphology 分开。

### 6.3 每项提交前自检

```text
required outputs 均存在且 SHA 匹配
输入/输出 frame count 和 identity 闭合
时间戳/索引严格匹配
视频完整可解码
无软链接和未声明外部消费路径
旧 baseline SHA 未变化
attempt/final 不可变
RESULT 与 METRICS 分母一致
claim_limit 未越界
governance 再次 PASS/FRESH
```

Clean、Compositor、Robot 和 DATA 的 regression packet 还必须绑定并检查：presentation-order frame↔sidecar、selected-left/sourceIndex0、无新增 radial warp、REP-103→OpenXR→camera 的 c2w、translation=meter、quaternion=xyzw、matrix=row-major，以及 rectified→selected registration。任何一项不闭合都不能仅凭视频可播放而过门。

## 7. 执行 AI 回报格式

```text
TASK_ID:
STATUS / TERMINAL_REASON:
RUN_RECEIPT: path, bytes, sha256
RESULT: path, bytes, sha256
PASSED_GATES:
FAILED_GATES:
BLOCKER:
RESUME_CONDITION:
NEXT_ACTION:
AUTHORITY_PROMOTED: false
```

不能用“看起来更好”“应该通过”或聊天记忆替代 artifact。

## 8. 整体完成定义

1. 所有必做任务与已触发条件任务达到唯一 immutable terminal。
2. Mask 附件、Clean 写域、合法 donor/atlas、geometry/compositor 分类和 Robot 合格产率均有可复核产物。
3. 训练像素有合法 source 或显式 UNKNOWN；invalid H50 配对处理被 loader 测试证明。
4. 达量时训练并比较四个 Visual Aux checkpoint；未达量则诚实 `BLOCKED_PREREQ`，不凑数据。
5. Physical Robot 和真实 Policy 保持 `OUT_OF_SCOPE + BLOCKED_EXTERNAL`，且不阻塞数字/视觉结果。
6. GOV-99 完成证据聚合、canonical 文档更新和 governance 复验。

本计划不以“接入多少论文代码”为完成，也不要求全部 156 条都达到最高视觉质量。正确终点是：每个决定都有冻结输入、单一变量、明确门限、有限终态、可追溯产物和不越界的 claim。
