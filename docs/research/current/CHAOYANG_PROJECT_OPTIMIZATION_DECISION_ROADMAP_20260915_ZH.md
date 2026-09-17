# Chaoyang / HumanEgo 下一阶段整体优化决策与执行路线

状态：`PROJECT_OPTIMIZATION_PROPOSAL / NOT_CURRENT_AUTHORITY`
制定日期：2026-09-15
当前事实快照：2026-09-15 15:25:27+08:00，governance revision `9922`
治理校验：`PASS / FRESH`

执行入口：本文件保留“为什么做/不做”的决策依据；给其他 AI 分派时使用 `CHAOYANG_PROJECT_OPTIMIZATION_EXECUTION_TASKS_V1_ZH.md` 及其 `EXEC-00` bootstrap packet。本文本身不作为可领取 task packet。

## 0. 结论

下一阶段不应整套替换 HaWoR、SAM3.1、FoundationStereo 或 ProPainter。真正值得投入的主线是：

```text
可复现基线与因果门
→ object/contact-aware Clean
→ 同一物体的合法因果外观来源
→ Robot/Object 确定性 ownership compositor
→ hard-feasible Robot Visual
→ Chips/Poker 严格配对的四个 Visual Aux checkpoint
```

与它并行，只推进一条独立的新传感器线：

```text
Controller wrist + MANUS25 hand
→ glove / controller / cable / accessory 独立 Mask
→ 同会话 Stereo 仅作有界表面深度修正
```

论文候选中，当前真正值得做的是少量、有退出条件的 challenger：

- 完成项目已经准备好的 Cutie 物体重入 canary；
- 将 PROVE 接成辅助 QA，不作发布真值；
- 在基础重入仍失败时测试 TAPNext++ 稀疏对应；
- 在当前 Robot 目标和评价合同审计完成后，再决定是否测试 cuRoboV2；
- Poker 先做轻量因果单应/纹理 atlas，验证失败后才考虑 Point2Pose。

Fast-FoundationStereo、SVOR、Do as I Do、Dyn-HaMR、EgoPHI、EMPIRE 等都不应成为当前主线前置。

用户已经明确不提供人工或外部标定。本路线不把“等待人工 Gold、外部距离标定、Robot TCP/安装测量”列为当前执行任务，也不以算法猜测替代它们。相应结果只能保持 `INTERNAL_CONSISTENCY`、`HYPOTHESIS_ONLY`、`VISUAL_CANDIDATE` 或 `SIMULATION_DERIVED`；Physical Robot authority 与真实 Policy 继续明确阻塞，但不阻塞视觉数据研究线。

## 1. 材料与权威边界

本路线分析了：

- `PROJECT_STAGE_RESEARCH_REVIEW_20260915_ZH.md`；
- `PAPER_CODE_WEIGHT_LEDGER_20260915.json`；
- 用户粘贴的论文建议文本；
- 当前 authority index、baseline registry、stage baseline、status、regression manifest 和端到端算法文档；
- 2026-09-15 当日 Clean、Sensor wrist 和 Robot/Occlusion 开发结果。

前三项都属于研究建议，不是项目 authority，也不构成自动执行指令。论文台账中的 19 个候选全部为：

```text
tested_in_user_environment = false
```

所以：

```text
官方代码存在 ≠ 本地依赖闭合
权重链接可下载 ≠ 本地 checkpoint 已验证
论文指标更好 ≠ 当前任务会改善
视觉结果更顺 ≠ 外部物理精度更高
```

任何候选进入实验前必须先生成 artifact receipt，至少固定：repo commit、license、依赖/CUDA、权重路径与 SHA256、推理入口、输入输出 schema、显存/耗时上限和失败退出条件。

## 2. 当前项目事实与真正瓶颈

### 2.1 exact78 当前状态

| 阶段 | 当前状态 | 正确解释 |
|---|---:|---|
| Raw | 156/156 PASSED | exact78 固定分母 |
| HaWoR | 144 PASSED，12 C | 单目 MANO/Z；权重身份仍需闭合 |
| Role Mask | 124 PASSED，32 C | 四角色独立；C 不可下游授权 |
| Object Mask | 121 PASSED，35 C | 物理实例 identity；禁止 union 过门 |
| Depth | 58/58 PASSED | 仅 metric-ready 58 条；内部公制链闭合，不是毫米真值 |
| Object6D | 58/58 PASSED | 仅 observed visible surface；遮挡保持 invalid |
| Clean | 58/58 PASSED | 仅结构/来源/解码闭环，不代表接触边界和语义正确 |
| Contact | 0 条 authority | 两条都只是 blocked/hypothesis |
| Robot Visual | 0/156 authority | 有 hard-feasible 开发候选，但没有 current authority |
| HumanEgo Aux | 0/4 | schema 和容量预估存在，训练输入仍被质量门阻塞 |
| HumanEgo Policy | 0/4 | 缺真实同步 Robot action，保持外部阻塞 |

58 条 Depth/Object6D/Clean 只占 Raw 156 条的约 `37.2%`。不能把 `58/58` 表述成 156 条完整链路已经完成。

### 2.2 新传感器线

0909/0910 Controller+MANUS+PICO 线与 exact78 分母严格分离：

- H0 admission、H1 hand、H2 tactile：已通过当前内部合同；
- H3 same-session Stereo、H4 sensor-role Mask：仍为质量 C；
- `play_cards_0910_001` 的 HaWoR 只有左 3/191、右 2/191 帧，不适合作为手套线主锚；
- 当前合理链路是 Controller 主腕、MANUS25 手形，Stereo 只修正可见表面相对深度。

已完成的腕部修正 canary 使用：15 帧因果滚动中值、±30 mm innovation clip、Stereo 权重 0.15、EMA alpha 0.15；最大实际修正左约 4.29 mm、右约 4.50 mm。它证明的是有界、稳定的相对修正，不证明解剖腕部或外部毫米真值。Stereo 不应提高为主锚，也不应在 Controller 缺失时接管 wrist。

### 2.3 当前最大的三个可行动瓶颈

1. **Clean → Occlusion → Visual Aux 的质量链。**当前 same-pixel donor 会复制错误表面，固定大膨胀破坏接触边界；这是用户已经在 Poker245、Chips039 中确认的真实视觉失败。
2. **严格因果性与像素来源。**离线双向 Mask、未来 temporal donor、整段 inpainting、最终 atlas 回写都可能把未来信息泄漏到当前训练 RGB。
3. **新传感器 H4 角色定义。**手套、Controller、线缆/绑带、前臂不能再混成一个文本分割目标。

当前 Robot 的主要问题不是“缺少一个更新的 IK 库”。已有若干候选通过数字 URDF 硬碰撞/限位门，严格 C 主要由跨形态人手相似度软门触发。因此先修目标、门分类和接触语义，比立即接 cuRoboV2 更值。

## 3. 必须先做的项目优化

### OPT-00：基线、资产、性能和因果门闭合

优先级：`P0`
规模：小到中
是否更换模型：否

#### 工作内容

1. 让当前 `robot_geometry_expansion_v74` 及其 Visual Aux/Occlusion watcher 先到 immutable terminal，不在运行中原地换算法。
2. 为 HaWoR 当前 checkpoint 建立可验证 SHA。若历史权重已无法恢复，则保留旧结果为 `UNKNOWN_VERIFICATION_REQUIRED`，另建一个已知权重 challenger；不得伪造旧权重身份。
3. 生成全链 runtime/VRAM/coverage profile，决定 Depth 等阶段是否真是吞吐瓶颈。
4. 建立端到端 prefix consistency test：同一时刻 `t` 分别由 `0..t` 和 `0..t+k` 输入产生，比较 Mask、donor、atlas、Clean、Robotized RGB 和训练 current-state。
5. 正式拆开：

```text
OFFLINE_BIDIRECTIONAL_QA
CAUSAL_TRAINING_INPUT
```

#### 必需产物

```text
OPTIMIZATION_BASELINE_FREEZE.json
THIRD_PARTY_ARTIFACT_RECEIPTS/
PIPELINE_RUNTIME_PROFILE.csv
CAUSAL_PREFIX_MUTATION_REPORT.json
ASSUMPTION_LEDGER.json
```

`ASSUMPTION_LEDGER.json` 记录所有无外部标定支持的固定数值、来源、适用范围和敏感性测试，禁止通过看视频逐次静默改参数。

#### 通过条件

- 候选的 repo/weight/config/schema 均可重现；
- 修改未来帧不会改变 `t` 时刻正式训练输入；
- offline 产物不能被 loader 当作 causal 训练输入；
- 当前运行任务有唯一不可变终态；
- profiling 后再决定速度模型是否值得测试。

### OPT-01：完成 object/contact-aware Clean successor

优先级：`P0，当前最值得做的算法工作`
规模：中
是否更换模型：否，继续 SAM3.1 + ProPainter

#### 已有证据，不要重复从零开始

2026-09-15 的 T0 开发结果已经显示：

- Poker245 删除区域相对旧方法减少约 21.72%；
- Chips039 减少约 21.46%；
- 当前可见物体像素变化为 0；
- 但结果仍是 visual diagnostic，没有 fresh ProPainter、因果分层 donor 或 Clean authority。

这说明“自适应边界”值得继续，不说明隐藏物体和 donor 已解决。

#### 下一步只完成冻结的 A/B/C/D，不再另起重复方案

```text
A：旧 Mask + 固定膨胀 + same-pixel donor + ProPainter
B：SAM3.1 分角色编排，其他保持 A
C：B + 自适应接触边界
D：因果重播种 + 背景/物体分层 donor + object atlas + UNKNOWN
```

冻结会话：

- 失败：`play_cards_0903_245`、`get_potato_chips_0902_039`；
- 回归：`play_cards_0903_243`、`get_potato_chips_0902_023`；
- 新线压力测试：`play_cards_0910_053`，不进入 exact78 晋升分母。

#### 必须实现

- `human core`、接触不确定带、远离物体的扩张区分层；
- 静态背景 donor 使用经验证的像素对应/深度重投影/平面单应之一；
- 动态任务物体只能使用同一物理实例的因果 atlas；
- 无合法来源时输出 `UNKNOWN + training_valid_mask=0`；
- source map 至少包含：source frame、surface/instance、warp type/error、visibility、causal max frame、synthesis type；
- ProPainter 只填 residual hole，不能冒充真实背景或隐藏物体真外观。

#### 采用当前已冻结的关键门

- 当前可见任务物体 retention ≥99.9%；
- authorized band 外 byte-exact 不变；
- source map coverage 100%；
- 冻结审核帧中 wrong-surface donor 为 0；
- 重入恢复延迟 ≤2 帧；
- future mutation test 通过；
- 两个失败 canary 改善且两个 regression 不退化后，才做全片。

### OPT-02：新传感器 H4 多角色 Mask

优先级：`P0，与 OPT-01 并行`
规模：中
是否更换模型：否

固定角色：

```text
left/right glove
left/right forearm
left/right controller
cable / strap / accessory
task-object instance
```

Controller 6DoF 投影只生成点/框提示，SAM3.1 决定像素边界；文本提示不能作为 Controller 已被分割的充分证据。MANUS25 原样保留，不在 Mask/转换器中静默改成 MANO21。

先跑 `play_cards_0910_053` 压力 canary，再加一条同域正常回归。重点检查左右身份、重入、离屏 empty、附件残留和任务物体误伤。没有人工 Gold 时只报告内部支持率和诊断视频，不报告 segmentation accuracy。

H3 Stereo 只修同会话 rectification、分辨率、左右帧和 registration 合同；没有可信 sidecar 的会话保持 review-only。不得借 exact78 的固定旋转或其他会话标定过门。

### OPT-03：Poker 轻量因果对象 atlas

优先级：`P1，OPT-01/02 后`
规模：中
是否先上 Point2Pose：否

先实现：

```text
同一张目标牌的可靠纹理点
→ RANSAC 单应
→ 正/反面身份
→ 截至 t 的纹理 atlas
→ hidden appearance 或 UNKNOWN
```

这是当前最低成本、最贴合牌面近似平面的方案。每个 atlas 像素必须能追溯到不晚于目标时刻的 Raw frame，且属于同一物理牌和同一面。当前可见牌面永远优先使用 Raw byte-exact 像素。

若牌发生明显弯折、身份歧义或单应残差不满足冻结门，则停止传播并标 UNKNOWN，不把牌强行当刚体。

Point2Pose 只在下列条件同时成立时再做旁路 canary：

- selected-camera adapter 已通过；
- Depth/registration 对该会话有效；
- 对象近似刚体；
- 轻量 atlas 在重入或位姿稳定性上仍失败。

Chips 的可形变包装不能默认套用刚体 SE(3)/TSDF。

### OPT-04：确定性 Robot/Object ownership compositor

优先级：`P1`
规模：中到大
前置：OPT-01 + OPT-03 + hard-feasible Robot candidate

替换 Raw-versus-Robot 差分回填，正式消费：

```text
Robot RGBA + optical-Z + part-id
Object visible optical-Z + instance-id
合法 causal object atlas
Clean background + pixel-source map
```

逐像素输出：

```text
BACKGROUND
OBJECT_FRONT
ROBOT_FRONT
TIE_UNKNOWN
```

已知深度顺序不等于拥有隐藏纹理；OBJECT_FRONT 但没有合法外观时仍输出 UNKNOWN。当前 087 的 99.72% 条件物体保留率、84.12% known coverage 和 15.88% UNKNOWN 只是可见表面内部诊断，不能当作遮挡已解决。下一步目标是保持 registered retention 门，同时用合法来源降低接触区域 UNKNOWN；不能靠猜像素降低 UNKNOWN。

### OPT-05：Contact 合同与当前 Robot 目标审计

优先级：`P1，与 OPT-03/04 部分并行`
规模：中
是否立刻换 solver：否

先固定 object-centric contact sidecar：

```text
object_instance_id
finger / fingertip / pad region
contact phase
relative slip
evidence source
confidence/uncertainty class
observed / tracked / attachment / unknown
```

证据方向保持：

```text
direct/tracked object → contact hypothesis → attachment hypothesis
```

attachment 不能反向证明 Contact 或 formal Object6D。

随后审计当前 Robot：

- human wrist/object 相对运动是否被目标链保留；
- world placement 是否为 session 固定；
- 人到物体时 Robot 腕/臂前伸幅度为什么不足；
- hard geometry/collision/limits 与 soft human-shape similarity 是否被错误混成一票否决；
- 两候选 base-backoff 和 hand round-2 的收益是否能在 prospective Chips+Poker canary 复现。

只有当当前 solver 确实在硬可达、碰撞或整段优化上形成主要失败，才进入 cuRoboV2 A/B。若只是软形态相似度未达标，换 solver 不是根因修复。

### OPT-06：冻结 Visual Aux 数据并训练四个 checkpoint

优先级：`P2，前述门闭合后`
规模：大

四个 checkpoint 固定为：

```text
Chips Raw
Chips Robotized
Poker Raw
Poker Robotized
```

每个任务内 Raw/Robotized 必须共享 session、frame、future-2D label、valid、split、seed、语言和训练配置，唯一变量为 RGB 域。逐侧 valid 保留，不能强制双手同时可见。

训练启动条件：

- causal prefix test 通过；
- contact-aware Clean 与 pixel-source legality 通过；
- Occlusion Silver 所有注册门通过；
- train/validation session 和 H50-window ledger 达到冻结最低量；
- 不把 `visual_robot_trajectory_sidecar` 当真实 action。

输出指标：ADE/FDE/PCK、有效窗口覆盖、左右身份错误、时序稳定、训练/验证 loss，并做严格成对比较。四个都仍是 Visual Aux checkpoint，不是 Robot Policy。

## 4. 论文/代码候选的最终决策

| 候选 | 决策 | 现在真正值得做的范围 | 不允许的解释 |
|---|---|---|---|
| Cutie | `做` | 完成已有 task-object 重入 canary；一条失败+两条回归 | 不替换四角色 Role lane，不自动晋升 SAM3.1 |
| PROVE | `做，辅助` | Clean 局部 RC-S/RC-T 排序诊断 | 不能替代物体身份、接触边界或 Gold |
| TAPNext++ | `条件性做` | Cutie/现有重入仍失败时，做稀疏长期对应 sidecar | 不是 Mask、Depth 或 Object6D 真值 |
| cuRoboV2 | `条件性做` | 当前 Robot 目标审计后，固定数字资产做整段 IK/碰撞 A/B | 不能提供接触目标、物理标定或部署授权 |
| Point2Pose | `后置 canary` | 仅近似刚体、Depth/adapter 有效的少量对象 | 不覆盖 formal observed-only Object6D，不用于所有 Chips |
| Fast-FoundationStereo | `由 profiling 触发` | Depth 是主要吞吐瓶颈时做速度/质量 Pareto | 不是默认精度升级，不能修标定 |
| DexUMI | `借流程，不接模型` | 借采集→映射→replay→同步训练数据合同 | 没有现成 KaiHand policy |
| SAM2Long | `只借思想` | 多 memory 假设/剪枝设计参考 | 不为复刻而降级 SAM3.1 |
| FoundationPose | `当前不做` | 未来拥有可靠 CAD/参考图的刚体再评估 | 不适用于输入条件缺失或明显形变物体 |
| Dyn-HaMR | `暂缓` | HaWoR 权重闭合后，仅裸手 C 簇小样本对照 | 不用于手套/Controller 主线，不全量重跑 |
| SVOR | `暂缓` | 上游 donor/atlas 修好后可做离线视觉对照 | 默认结果不得进入 causal training RGB |
| ContactOpt | `仅历史参考` | MANO+object 接触优化概念对照 | 软组织穿透不能成为刚性 KaiHand 规则 |
| EgoPHI | `观察` | 等权重/本地运行闭合后研究接触先验 | 预测力不是触觉或物理真值 |
| Do as I Do | `后续独立仿真` | 未来数字资产/动力学参数齐备时研究 | Sharpa/UR3 默认资产不可冒充 Tianji/KaiHand |
| EMPIRE | `基线后研究` | 四 checkpoint 完成后再研究显式计划表示 | 不在首次 Raw/Robotized A/B 同时换模型 |
| Ego2Robot | `不做，观察发布` | 只读方法 | 官方完整管线/权重未核实 |
| EgoEngine | `不做，观察发布` | 只读方法 | 不成为主线依赖 |
| CHOIR | `不做，观察发布` | 只读接触耦合思想 | 作者可运行资产未核实 |
| ReForce | `不做，观察发布` | 只读力反馈思想 | 当前五指事件不能冒充标定力 |

## 5. 并行执行方式

```text
Lane A：当前 v74 Robot 批次及 watcher 跑到不可变终态

Lane B：OPT-00 causal/provenance + OPT-01 Clean C/D

Lane C：OPT-02 Sensor H4；H3 仅修同会话软件合同

Lane D：Cutie/PROVE artifact smoke，不修改 authority

              ↓ 共同会合

OPT-03 Poker causal atlas
→ OPT-04 ownership compositor
→ OPT-06 四个 Visual Aux checkpoint

OPT-05 Robot target/contact 审计可与 OPT-03 并行，
cuRobo/Point2Pose/Fast-Stereo 只有满足触发条件才启动。
```

不要同时启动 TAPNext++、Cutie、SAM2Long 和 Point2Pose 四套重入逻辑。每轮只改变一个主要变量，并保留失败退出条件。

## 6. 停止条件与晋升规则

每个 challenger 必须满足：

1. 代码/权重/依赖 receipt 闭合；
2. 固定失败 canary 改善；
3. 固定回归样例不退化；
4. 全片解码、帧身份和 provenance 闭合；
5. 资源成本可接受；
6. 因果训练分支通过 future-mutation test；
7. 新结果写独立 immutable revision，不覆盖当前 baseline；
8. 只有治理 aggregator 可更新 registry/authority。

任一项不满足就保留旧基线并终止扩量。论文新颖性、单张好看截图或内部残差下降都不能替代上述门。

## 7. 明确不投入的工作

当前不做：

- 为获得物理 authority 继续等待用户提供外部标定或人工 Gold；
- 用假设 transform 冒充测量的 CAD/TCP/world→base；
- 把 Stereo 可见表面深度当解剖手腕，并提高成主锚；
- 在 Clean/Occlusion/causal 门未闭合前启动四 checkpoint；
- 把 Visual Aux checkpoint 或 visual trajectory 命名为 Robot Policy/action；
- 用 Clean 像素反喂 Depth、Object6D 或 Contact；
- 全量替换 HaWoR、SAM3.1、FoundationStereo、ProPainter；
- 全量接入 Point2Pose、Dyn-HaMR、SVOR 或 Do as I Do；
- 为降低 UNKNOWN 使用无来源的桌面、盘子或生成纹理冒充隐藏任务物体；
- 根据一次用户观感静默调参而不写 assumption/provenance ledger。

## 8. 本阶段完成定义

本阶段不是以“安装了多少论文模型”为完成，而是以以下结果为完成：

- 当前 exact78 Robot/Visual watcher 全部有不可变终态；
- Clean 的错误表面 donor 和接触边界在冻结难例中被证伪并通过回归；
- 所有训练像素有合法来源或显式 UNKNOWN；
- 新传感器手套、Controller 和附件角色分开，Controller+MANUS 主链不依赖 HaWoR；
- Poker 隐藏外观来自同一张牌的因果 atlas，或诚实保持 UNKNOWN；
- Robot/Object ownership 由统一 z-buffer 和来源合同决定；
- Robot hard/soft 门分离，保留可用于视觉研究的 hard-feasible 候选；
- Chips/Poker 各形成严格配对的 Raw/Robotized 训练 ledger；
- 完成四个 Visual Aux checkpoint 和同配置比较；
- 所有结论仍正确标注为视觉/内部/假设，不越权成为物理 Robot 或 Policy 真值。

## 9. 主要依据

项目当前权威：

- `docs/governance/CURRENT_AUTHORITY_INDEX.json`
- `docs/governance/CURRENT_BASELINE_REGISTRY_V2.json`
- `docs/governance/CURRENT_STAGE_BASELINES_ZH.md`
- `docs/governance/CURRENT_PROJECT_STATUS_ZH.md`
- `docs/reference/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md`
- `docs/governance/CURRENT_REGRESSION_MANIFEST.json`

专项证据：

- `docs/research/current/reports/sensor_pipeline/20260915/NEW_SENSOR_PIPELINE_GO_HOLD_ZH.md`
- `docs/research/current/reports/visualization/20260914/PLAY_CARDS_0910_001_BASELINE_PROBLEM_SUMMARY_ZH.md`
- `docs/research/current/reports/visualization/20260915/CLEAN_CONTACT_AND_WRIST_BASELINE_AUDIT_ZH.md`
- `docs/research/current/CLEAN_LAYERED_SAM31_SUCCESSOR_EXPLORATION_V1_ZH.md`

已核实的候选官方入口：

- TAPNext++：https://github.com/google-deepmind/tapnet
- Point2Pose：https://github.com/tzuyuan/point-to-pose
- PROVE：https://github.com/xiaomi-research/prove
- Fast-FoundationStereo：https://github.com/NVlabs/Fast-FoundationStereo
- cuRoboV2：https://github.com/NVlabs/curobo
- SVOR：https://github.com/xiaomi-research/svor
- Do as I Do：https://github.com/malik-group/do-as-i-do

这些链接只证明公开入口存在；在本地 artifact receipt 和 canary 通过之前，均不是当前项目 baseline。
