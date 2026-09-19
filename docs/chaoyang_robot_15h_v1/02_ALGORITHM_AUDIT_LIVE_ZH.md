# chaoyang：实时使用算法审计与证据视图

版本：v1.0  
当前文件状态：初始模板，尚未连接本地运行事实；不是已经实时更新的项目看板  
报告基线：用户提供提交 `4228645`、治理 `12075`  
配套任务：[15 小时执行任务书](01_EXECUTION_PLAN_15H_ZH.md)

> 本文只回答“这一刻真正用了什么算法、有什么证据、哪些下游可以用”。计划目标不放进结果列。运行进度不依靠 AI 凭记忆手写，必须由实际 task receipts、进程心跳和固定产物生成。

## 1. 唯一事实来源，不再建立第二个治理系统

权威关系固定为：

```text
既有 CURRENT_STATUS_RECEIPT + ALGORITHM_CONTRACT + 任务/结果收据
                            ↓
                机器聚合的算法审计视图
                            ↓
               本 Markdown + 简短运行汇报
```

本包 `03_ALGORITHM_AUDIT_STATE.template.json` 只是字段模板，不是已被项目接受的 schema，也不是可调度任务包。优先将必要字段映射到既有结果文件；确需一个 live JSON 时，放在已授权运行根，引用现有 receipt，避免多份权威真相。

只有 publisher 写当前视图。Worker 只写其独占 attempt 的不可变结果和心跳。当前文档更新不修改历史终态，也不授予新 authority。

### 更新时机

- 本轮启动后第一次实际核验。
- 某算法候选完成、采用、撤回或回滚时。
- 某任务完成、批次完成、数据域发生变化时。
- 运行中每 10 分钟根据真实信息生成快照；worker 心跳建议每 60 秒。
- 最终封账前一次完整重建。

建议快照过期阈值 15 分钟，标为本轮运维参数，不代表算法质量。当数据过期时显示 `STALE`，不能保持绿色 RUNNING/PASS。生成采用临时文件加原子替换；文件不完整时保留上一完整视图并注明过期。

## 2. 顶部实时状态卡

下面在运行前全部保持待核实；不得把用户报告填成当前 PID 检查。

| 字段 | 本包初始值 | 运行时填充方式 |
|---|---|---|
| 状态来源 | USER_REPORTED_SNAPSHOT | receipt/PID/lease/RESULT 的真实路径与 SHA |
| repo / 实际 HEAD | 待核实 / 用户报告 4228645 | 本地 Git 命令 |
| effective governance revision | 待核实 / 用户报告 12075 | 当前 receipt |
| run_id / parent task_id | 未注册 | 实际任务系统 |
| started_at / deadline_at | 未开始 | 启动时真实时间，另用 monotonic 管时间盒 |
| active children / next authorized tasks | 待核实 | 当前 INDEX 和父子任务状态 |
| GPU owner / PID / lease | 待核实 | 租约与系统进程交叉验证 |
| 当前 batch manifest SHA | 未生成 | 已冻结 manifest |
| 已成功 R0 / R1-E / R1-H / R2 | 本轮尚未运行 | 真实 RESULT 聚合，不来自目标数 |
| 最后算法采用 | 无本轮采用 | adoption receipt |
| heartbeat_at / published_at | 空 | 本地实际事件 |
| 是否 stale | 是 | 最新事实时间与当前时间比较 |

## 3. 算法登记表：一行一个能力与当前版本

“能力”允许分 R0、R1-E、R1-H，因为它们回答不同问题；同一能力不能同时写 v1/v2/latest 都是当前。

| 能力 | 用户报告的起点 | 允许本轮动作 | 核心消费限制 |
|---|---|---|---|
| encoded image adapter | 裁切/resize 已确认 | 复用并对新采集域 preflight | 不复活 equiDis62；K/P 域要一致 |
| FoundationStereo | 当前 150 帧开发级通过 | 新会话固定权重推理；有依据的单候选修复 | 全局通过不等于局部 5 mm 准确 |
| HaWoR bounded v2 | 用户接受的单样本 prior | 同版本运行新会话，源缓存不改 | observed 与推断帧分开 |
| SAM3.1 strict/weak | 主体部分可用，弱角色拒绝 | state/reseed/ID 时序修复 | raw/semantic/inferred 分开 |
| Removal/Clean | V1/V2 拒绝 | 冻结，不是主链依赖 | removal/inpaint 不进几何真值链 |
| Object surface v2 | 独立卡面可观测量可用 | 增量有限表面与可观测性 sidecar | 可见中心不是固定中心；不造完整体积 |
| Human-Stereo alignment | 授权拒绝 | 原因审计与有依据 successor | 不直接接受 0.783730；不以接触自证 |
| Direct stereo local relation | 本轮拟新增 | 同域可见表面关系 | 不依赖所有 MANO absolute-Z 对齐；不是隐藏指尖真值 |
| Interaction observed | 部分局部证据，严格窗不足 | 按字段/实例准入 | 缺证据 unknown，不改 NO_EVIDENCE |
| Occlusion reconstruction | 本轮拟新增 | 非因果短窗重建、回放验证 | 不覆盖 observed，不进在线输入 |
| Contact hypotheses | 本轮拟新增 | 有反证的有限竞争解释 | support_score 非概率；含 no-contact |
| Kai22 R0 | 当前样本已完成 | 必须真实新会话批处理 | 无接触声明，不是控制 GT |
| Kai22 R1-E | BLOCKED_LOCAL_EVIDENCE | 合法证据出现后才执行/采用 | 沿用严格门和观测 scope |
| Kai22 R1-H | 本轮拟新增 | 明确 task hypothesis 的轨迹合成 | 不能顶替 R1-E 或证明真实接触 |
| R2 virtual arm | 未运行 | 可从 R0 独立推进 | 虚拟 placement 固定且可追溯，真机标定仍缺 |
| Batch executor | 待核对当前可复用实现 | 新 manifest、真实运行、resume | 终态数量与 Robot 质量通过数量分开 |

本表的文字只表达报告起点。运行版本必须进一步填写下面每项字段，而不是只写“算法已优化”。

## 4. 每个算法条目的必填字段

### 身份与实现

- `capability_id`：能力名，例如 `kai22_r1_hypothesis`。
- `selected_version`：实际采用的语义版本；未采用为空。
- `entrypoint`、真实 repo-relative 代码路径、代码 SHA、Git commit。
- `config_path/config_sha`、环境标识与锁文件 SHA。
- `weights`：实际模型路径/权重 SHA；纯 CPU 非学习算法才写 ABSENT。
- `asset_shas`：URDF、mesh、joint map、proxy、相机资产。

### 输入与几何

- `input_frame`、`units`、image size、pixel-center 约定、sourceIndex。
- crop/resize/flip/pad 每步的映射以及实际 K/P 来源。
- Stereo disparity 的定义与输出回域规则。
- 会话、source group、手别、object identity、允许 evidence type。
- 与外部尺度有关的字段是否独立验证；未知不能写 verified。

### 质量与声明

- `implementation_status`：未实现/已实现/测试通过/实片运行，不能互相代替。
- `execution_status`：以当前任务系统枚举为准。
- `quality_status`：候选、拒绝、局部 DEV 准入等 sidecar 状态。
- `allowed_consumers` 与 `forbidden_consumers`：具体列明，不只笼统写 PASS。
- `claims`：验证了什么、未验证什么、人工验收是否完成。
- 实际 development/validation/holdout manifest SHA，源组是否独立。
- 测量结果、阈值、阈值来源、单位、分母、覆盖率和失败数量。
- 本轮 candidate/adopt/reject receipt、产物路径/SHA、替代哪一版本、回滚版本。

### 性能与资源

- model loads、累计 GPU 秒、CPU 秒、每帧耗时、峰值内存/显存。
- 真实 processed/expected 帧数、非空输出、有效输出、invalid/unknown。
- 最近进度、停止原因、超时/预算、不相关任务是否受影响。

## 5. 坐标审计矩阵

| 量 | 应声明的 frame/语义 | 必测事项 | 禁止混用 |
|---|---|---|---|
| HaWoR joints/mesh | wrist-local、camera 或 world 中的一种 | transform 方向、左右手、单位、原始/稳定化版本 | 未说明的 world 和 optical camera 直接相减 |
| Stereo XYZ | 实际 encoded physical-left 相机域 | crop/resize/flip 恢复、P/K/B、optical-Z | ray distance 当 Z、错域 K、忽略主点偏移 |
| 可见物体中心 | 可见 surface patch 的统计中心 | 可见范围变化、物体/相机运动 | 当整张牌固定刚体原点 |
| 平面法向/面内轴 | 指定符号约定/对称周期 | ±n、π 对称与物体真实旋转 | 参数翻符号就叫物理跳变 |
| 关联手表面点 | 当前像素的表面观测 | skin/sleeve/unknown、指部关联、边界置信 | 当隐藏的接触点或关节中心 |
| Robot wrist | wrist-local/object-relative/virtual frame | native robot geometry、来源与修正 | 用未对齐 HaWoR Z 减 Stereo object |
| arm base | VIRTUAL_DESIGN_PARAMETER 或校验 proxy | 会话级固定、scene 同步刚体变换 | 当实测 camera/world→base |
| TCP/mount | 实测/开发proxy/unknown 中之一 | 来源、单位、变换方向 | render 看起来对即 measured |

## 6. 数据级准入矩阵

| 数据层 | 严格 observed Contact | R1-E | R1-H | R0/R2 显示 | 在线控制 |
|---|---:|---:|---:|---:|---:|
| 已准入当前帧 Stereo/semantic 表面 | 按局部门允许 | 按局部门允许 | 允许 | 允许 | 本轮禁止 |
| 有独立验证的派生变换/距离 | 按 scope 允许 | 按 scope 允许 | 允许 | 允许 | 本轮禁止 |
| HaWoR prior 未对齐 absolute-Z | 不允许当公制接触 | 不允许直接使用 | 仅姿态先验或明示独立对齐 | 允许并标来源 | 本轮禁止 |
| short_gap / temporal reconstruction | 不允许 | 不改 observed 授权 | 明示离线使用 | 单独图层 | 禁止非因果输入 |
| task contact hypothesis | 不允许回写观测 | 不替代严格证据 | 允许，需反证评估 | 允许 | 禁止 |
| removal / feather / inpaint | 禁止 | 禁止作几何依据 | 禁止作独立观测 | 仅图像展示 | 禁止 |
| 安装 visual proxy | 禁止作实测依据 | 仅数字机器人定义 | 仅数字机器人定义 | 允许 | 禁止 |

本轮所有训练资格默认 false。不是因为这些数据永远不能用于学习，而是不能在没有独立数据合同、split 和来源审计时自动混入正式训练。

## 7. 阈值账本

所有阈值按三类来源记录，不写成无出处的“物理真理”。

| 名称 | 报告/设计值 | 类型 | 本轮约束 |
|---|---|---|---|
| 既有严格 Contact 距离 | 5 mm | INHERITED_CONTRACT | 不能为当前 6.60 mm 结果放宽 |
| 历史 Human scale 搜索区间 | [0.8,1.2] | HISTORICAL_SEARCH_BOUND | 边界命中是诊断；新科学解释须独立证明，不硬改下界 |
| 无约束拟合尺度 | 0.783730 | REPORTED_MEASUREMENT | 不是新默认标定 |
| 有界 P90 残差 | 19.33 mm | REPORTED_MEASUREMENT | 不是可声明的接触精度 |
| W1 真实新会话目标 | 12 | THIS_RUN_TARGET | 不保证完成，不用 BLOCKED 数凑 |
| 新会话上限 | 220 | THIS_RUN_CAP | 实际以冻结 inventory 为准 |
| CPU 心跳/事实刷新 | 60 s / 10 min | OPERATING_PARAMETER | 非算法指标 |
| 推断窗初始范围 | 约 0.5–1 s | PRE_REGISTERED_DESIGN | 首次验证前固定，不用失败后无界增长 |
| candidate 数 | 每窗最多3，含 no-contact | SEARCH_BUDGET | 不是接触概率 |
| Robot 修正/运动限制 | 读取当前合同 | INHERITED_CONTRACT | 时间单位与实际 dt 核对 |

新阈值缺依据时写“用于本轮开发筛选”，并报告敏感性。不得用独立测试集反复调到过门。

## 8. 实验卡：每次算法更新只填一张

```text
experiment_id:
capability_id:
parent_selected_version:
question:
registered_before_run_at:
input_manifest_sha:
source_group_split:
proposed_change:
controlled_variables:
fit_objective:
independent_evaluation:
negative_controls:
thresholds_and_origin:
max_attempts / time_budget:
code/config/asset hashes:
actual_command:
actual_runtime / resources:
measured_results_with_denominators:
coverage_change:
failed_cases:
decision: ADOPT / REJECT / INCONCLUSIVE
allowed_scope_after_decision:
forbidden_claims:
receipt/artifacts:
rollback_target:
```

一个参数改动要能回答“为什么变”“用什么独立证据判断”，而不是只写“优化阈值”。拒绝的实现可以保留审计证据，但不能继续作为默认运行入口。

## 9. 深度/对齐审计专页应显示的内容

- 两侧手分开；skin 与 equipment 分开；近远深度/视角/时段分开。
- K、principal point、baseline、projection matrix 和 encoded view 的实际来源。
- raw/cropped/resized/flipped/unflipped 每阶段尺寸与矩阵。
- optical-Z 与 ray-distance 的数值核对。
- 相同像素可见表面对应的比例；轮廓/遮挡拒绝比例。
- 固定尺度与新候选模型在 development/独立 validation 的比较。
- 参数是否触边、条件数/耦合、按时间块重采样的稳定性。
- 任何 scale/offset 的具体作用对象，不能只输出一个无 frame 的 s。
- 全局、手部、物体、边界四类有效性；不能只列一个 LR 平均值。
- metric 接触是否被准入及 scope，不能从“Depth PASS”隐式继承。

## 10. 遮挡/Contact 审计专页应显示的内容

每个固定 `(hand_id, finger_id, object_id)` 单独列段：

```text
observed frames
metric surface evidence frames
occluded/offscreen/unknown frames
reconstructed frames and support windows
identity certainty / competing identities
strict contact admitted frames
hypothesis states and support_score
no-contact alternative
negative evidence
noncausal flag
```

同时报告人工遮挡回放和真实遮挡重现，两者不能混在“准确率”一个数字里。没有人工真值或独立测量，只列代理指标/内部一致性，不写 contact accuracy 95%。

物体没有运动不等于没有接触；手和物体在图像同动不等于抓稳。第一视角相机运动必须被处理或在相对观测定义中明确抵消。

## 11. Robot 审计专页应显示的内容

| 字段 | 应填内容 |
|---|---|
| 轨迹层 | R0 / R1-E / R1-H / R2，不可只写 Robot PASS |
| 生成情况 | 实际 solver 运行、目标样本数、输出样本数、有效帧 |
| 采用情况 | 比 R0 通过哪些固定对比门；未采用修正的原因 |
| 几何 | Kai URDF/mesh/pad/joint map 与 proxy SHA |
| 坐标 | wrist/frame/object/virtual base，变换链可复现 |
| 修正 | q/wrist 修正范围、窗口/过渡区、未改帧是否保持 |
| 连续性 | 实际 dt 下的速度/加速度、边界接续、缺帧 |
| 碰撞 | 自碰撞、可见表面相交、未建模物体/环境 |
| 任务关系 | 指部patch、切向逃逸、对接触假设的敏感性 |
| 独立性 | 优化目标与评估数据分离，固定样本分母 |
| 物理能力 | calibrated=false / deployable=false / control_eligible=false |
| 用户验收 | pending，除非有实际人工验收事件 |

R1-H 在新输入假设扰动下变化很大时必须显示敏感性；不能只展示最好看的一条。若仅存在 R0，不生成伪 before/after 标作 refined。

## 12. 批量事实表与计数口径

从每会话不可变 RESULT 聚合，禁止从文件夹数量或日志关键词计算成功。

```text
session_id, source_group, task, capture_domain, frame_count
input_eligibility, human_status, depth_status, object_status
r0_exported, r0_quality_admitted, r0_valid_side_frames
strict_contact_valid_frames, inferred_contact_frames
r1_evidence_adopted, r1_hypothesis_adopted
r2_virtual_ik_admitted, r2_valid_frames
first_blocker, terminal_reason, evidence_scope
input_sha, result_sha, review_path
```

最终汇总同时显示：

- 清单总数 / 合格输入 / 实际启动 / 终态覆盖。
- 真正 R0 质量通过会话数与完整帧/有效帧数。
- R1-E、R1-H、R2 的独立数量。
- 未执行、预算终止、失败域、未知几何、运行失败的分布。
- 每任务/采集域/源组的覆盖和吞吐。
- `physical_deployable=0`，不把虚拟机械臂样本写成物理部署通过。

目标未达到时直接写差多少，不能把 terminal=100% 写成 Robot conversion=100%。

## 13. 测试矩阵与质量改进判据

### 不变性/契约

源数据与 sealed SHA 不变；拒绝 wrong-domain；坐标/单位/左右手正确；R1-H 不可被严格 Contact loader 读取；推断帧不增加 observed 计数；joint order 正确。

### 几何测试

合成已知平面/手表面、optical-Z 与 ray-distance 区分、crop/resize/flip 回域、主点差/视差符号、面法向等价、可见中心偏移不误判真实平移、开放面不能声明封闭穿透体积。

### 算法验证

独立源组；遮挡屏蔽不泄漏；identity ambiguity 保留；无接触对照；修改尺度不以旧门通过为唯一目标；Robot 不靠移动物体/缩放mesh/删帧优化指标。

### 运行与批量

resume 不重复、不读半成品；缓存依赖变更失效范围正确；单会话失败其他继续；GPU lease 独占；无幽灵进程；新会话真实视频完整解码。

### 指标改善

至少报告效果和覆盖两轴。只提高有效性门严格度、把难帧删掉取得更小误差，不能叫算法全面改善。已优化目标下降必须与独立支持、约束无退化一起判断。没有真值时不写真实准确率提升。

## 14. 最终可复制摘要模板

```text
本轮窗口：实际开始___，实际结束___，执行___小时。
实际 HEAD/治理：___；用户报告起点___。
本轮任务是否全部终态：___；无本轮后台进程/lease/writer：___。

新数据：inventory___；eligible___；attempted___；terminal___。
R0：实际导出___，质量准入___，有效side-frame___。
R1-E：采用___；R1-H：采用___；R2虚拟机械臂：准入___。
物理部署：0。

尺度原因：发现___ / 未决___；采用修复___。
深度：局部质量___；未经外部验证部分___。
遮挡：回放___；真实重现___；unknown___。
严格Contact：___；hypothesis支持___，不是GT或概率。

当前每个能力选用版本与代码/配置/权重：见审计视图___。
真实批处理命令___；resume命令___。
审阅视频与结果文件___。
目标12个完整新会话：已达到/未达到，原因___。
未完成与首阻塞___；下一轮非active待办___。
用户视觉验收：待定。
```

本模板中空值必须由实际事实填充。未运行保持未运行，不能把本文件当作已经完成的最终报告。
