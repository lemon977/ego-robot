# 0915 Robot 质量恢复与跨会话验证：第二轮 15 小时执行任务书

版本：v2.1。编制日期：2026-09-19。
状态：EXECUTION_SPEC_AWAITING_CAS_REGISTRATION。当前文件不是已运行结果；只有注册后的机器任务包授予执行权。
基准：用户最新报告治理 `12362 / gov-012362-8d04bae5bef9`；实际 HEAD 启动时读取。
工作项目：机器人 `/mnt/workspace/code/chaoyang`，不得混入同名交易研究项目。
配套：[启动指令](00_START_HERE_ZH.md) · [算法审计文档](02_ALGORITHM_AUDIT_LIVE_ZH.md) · [验收测试](04_ACCEPTANCE_TESTS_ZH.md) · [技术来源](05_TECHNICAL_SOURCES_ZH.md)。

> 本轮不是扩大“已导出但质量为零”的产量。先查清拒绝是否来自真实算法错误、字段语义/实现错误、还是消费者依赖过宽；分别修复并在固定样本上验证。不得把 UNKNOWN 改成 PASS，也不得让错误的全局依赖挡住合法的独立工作。

### 0.1 V2.1 覆盖条款

本节覆盖后文 V2.0 中所有含糊的“W1/剩余六条/最多四条验证”表述；不改变旧 sealed 结果。

| 层级 | 固定会话 | 用途与访问规则 |
|---|---|---|
| W0 开发 | Poker031、Poker119、Chips007、Chips042 | 失败定位与主修复 |
| W1-DIAG | Poker044、Chips097 | 可用于决定候选；不能决定采用 |
| W1-ADOPTION | Poker106、Chips029 | 首次打开前冻结候选；之后只 ADOPT/REJECT/同签名故障重试 |
| H9 FINAL HOLDOUT | Poker054、Poker003、Chips068、Chips056 | 只运行已采用能力；禁止调参、换实例和尺度拟合 |
| EXTRA FINAL | Poker030、Poker085、Chips017、Chips095 | H0封存；H11扩批门未过则不打开、不得替换 |

HaWoR/R0 的共享候选必须同时通过两条 W1-ADOPTION 才采用；SAM/Object 可以按 Poker 与 Chips 能力分别采用。某能力未采用时，其 H9 对应会话保持 `NO_ADOPTED_FIX`。EXTRA FINAL 固定盐为 `chaoyang_robot_recovery_15h_v2_extra_final_v1`，执行器必须从 source group、采集时间和帧数重新计算并核对上述四条，不一致就 fail-closed。

V2.1 只把 `METADATA_VERIFIED_NO_USER_ATTESTATION` 作为 source-group 当前声明。新 sidecar 不修改旧 `SOURCE_GROUP_MANIFEST.json`；旧 `USER_CONFIRMATION_SEPARATELY_RECORDED` 不再作为本轮 authority。

## 1. 起点：事实、未知与本轮成功含义

以下来自用户报告，启动时核验对应机器文件；没有核验前标 USER_REPORTED，不声称已远程检查。

| 项目 | 报告状态 | 本轮动作 |
|---|---|---|
| 调度 | 47 节点终态，无活动任务，next_task=null | 注册新父任务，不复活旧节点 |
| 0915 库存 | 220；已冻结 W0=4、W1=8；其余208未运行 | 先处理现有12，最多另验证4条 |
| 0916 | 240终态，222 CLEANED、18 REJECTED | 只读，不进下游 |
| HaWoR | 001 单样本可视；W0严格0/4；W1推理0/8 | W0错误分解、有限修复、W1诊断运行 |
| SAM | 手/主要牌部分可用；弱角色失败 | 解耦初始化依赖、修时序与实例身份 |
| Depth | 12/12、3,390帧内部质量通过 | 首先复用，不为产量重复跑 |
| Object6D | W0两个Poker会话部分几何可用 | 同12会话增量几何，不套用刚体模型到袋体 |
| 对齐 | s=0.8触边；无约束0.783730；hold-out P90=19.33mm | 既有授权仍拒绝；独立原因对照 |
| Contact | 严格连续窗0；有限patch最近6.60mm | 观测可满足性审计，不改5mm门 |
| R0 | 4导出、严格0/4；W1阻塞 | 独立审查Robot真实失败，修复后再导出 |
| R1-E / R1-H | E无结果；H一条导出被拒绝 | 先读拒绝原因，有新证据才开求解 |
| 虚拟R2 | 4导出、质量0/4；旧合同SHA漂移 | 合同闭包与IK质量分开；小样本恢复 |
| Removal/Clean | V1/V2拒绝 | 全轮暂停，不是本轮主链依赖 |

目前没有 W0 四会话逐帧拒绝码、R0/R2 完整质量判定和真实视频字节的本地复核。本计划不预先认定“算法没问题，只是门太严”。

### 1.1 本轮交付目标，不能预填结果

- 必做：W0四会话形成逐手逐帧拒绝矩阵，定位拒绝来源，并至少实施一个有实片对照的主要原因修复；无可修复原因则给出反证和明确能力上限。
- 运行目标：W1-DIAG两条不再因为W0全局失败而停在推理之前；随后按冻结协议打开两条W1-ADOPTION。其余四条是H9 FINAL HOLDOUT，只运行已采用能力。实际尝试/完成实报。
- 质量目标：争取至少两个W0会话在原适用严格门下恢复、另至少两个W1会话独立通过R0。四条是工程目标，不是允许降低门的理由。
- 如果仅有局部合法窗口，则如实交付局部窗口，整会话仍拒绝；不声称上述整会话目标已完成。
- 几何目标：既有12条Depth尽可能获得独立对象观测；Poker发布有限平面，Chips发布可信局部表面。Unknown尺寸、隐藏面保持unknown。
- 研究目标：查清Contact“零窗口”是否因几何误差、身份/采样失败，或实际接触面不可见；在独立验证成立时才产生新的R1候选。
- 虚拟机械臂目标：对一个合格R0窗口验证真实FK/IK与安装链；通过后最多两个会话完整审阅，不再直接盲批四条失败候选。
- 所有产物仍为开发候选，默认 training_eligible=false、control_eligible=false、physical_deployment_authorized=false。

## 2. 固定范围与不变性

只消费当前合法根 `/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`。按 0.1 节冻结16个会话的任务类别、总帧数、来源、输入SHA和访问级别；不把每条都裁成150帧。

001及已用于开发、失败分析的W0/其他样本一律记为development。W1只能在确未参与相应参数调整的情况下作为该能力的validation；同一源组不跨分组。编号或裁剪后SHA不同不是独立录制证明。缺独立源组时允许运行与工程回归，不声称跨录制泛化。

默认不动其余204条。EXTRA FINAL 四条虽然在H0封存，但只有主要R0链达到H11扩批门、原12条全部终态、H9候选冻结、至少2条W0同口径严格恢复、至少2条W1 R0质量准入、无未闭合运行/实现错误且P90预算允许在H13.5前全部启动时才运行。任何条件失败均写 `NOT_OPENED_EXPANSION_GATE_FAILED`；超出16条需另轮授权。

固定 HaWoR、SAM3.1、FoundationStereo 权重与环境。允许修封装、状态管理、数据变换、有界MANO后处理、现有CPU求解器和对应测试；不新装框架、不训练新模型、不换SAM2.1/Cutie、不引入PICO26替代手部真值、不缩放Kai机器人几何。

encoded图像只做物理眼裁切和resize，沿用已登记的双眼水平镜像视差适配和输出反镜像。禁用重复lens-undistortion/equiDis62 remap；若发现新的域错误，只在新corrective中处理，旧产物不可覆盖。

不请求用户提供牌尺寸、TCP、安装变换、camera/world→base、新录制或人工标注。不用标准牌尺寸或猜测厚度填真值。真机物理链继续BLOCKED_EXTERNAL。

## 3. 本轮关键改动：三种权限与局部范围

### 3.1 运行许可不等于结果准入

| 层 | 检查对象 | 失败后影响 |
|---|---|---|
| 运行/输入许可 | 正确图像域、资产SHA、可解码、运行环境、资源租约 | 阻止受影响的模型运行 |
| 字段/手/实例/窗口许可 | 该消费者真正需要的观测、身份、运动/几何质量 | 只阻止对应字段/窗口消费 |
| 整会话严格质量 | 当前完整合同与覆盖要求 | 保留严格拒绝；不得把局部片段伪装整会话PASS |
| 物理部署许可 | 实测安装、TCP、外参和安全验证 | 全轮为false，不挡离线数字开发 |

例：可见左手某段通过自身姿态/运动门，右手一段缺失，不能自动使左手全部归零。反过来，左手某段真实自碰撞或身份错误，也不能因只做可视化就当作高质量动作。

### 3.2 消费者能力矩阵，先shadow验证后注册

| 消费者 | 必需证据 | 不应成为统一前置项 |
|---|---|---|
| HaWoR诊断推理 | 合法RGB、固定资产、环境/输入测试通过 | W0整会话严格成功数 |
| SAM任务物体 | 合法RGB、任务/实例提示、其自身实例验证 | HaWoR整会话3D严格PASS |
| SAM手部 | 合法RGB + 当前帧可靠2D提示或SAM自身候选 | 全片MANO骨长/公制尺度通过 |
| Poker Object表面 | 合格对象Mask + 同域有效Depth | 手部、前臂、皮套、Clean |
| R0 q22 | 同手合法姿态/方向、机器人映射与限位/FK/碰撞/时序门 | Object6D、Contact、Human-Stereo绝对平移 |
| R0 wrist camera-relative | 合法相机域腕部prior及其时序门 | 真实robot-base外参 |
| R0 wrist world-relative | 上项 + 可信c2w/SLAM | 不可静默从camera级升级 |
| R1-E | 该固定配对的合法局部公制关系和严格接触证据 | 无关对象/另一只手全片通过 |
| R1-H | 经反证验证的有限假设、合法数字几何、身份与不确定性 | E已经PASS；但不能绕过自身几何缺陷 |
| 虚拟R2 | 合格R0窗口 + 冻结数字安装/arm资产 + IK/碰撞门 | R1成功、真实外参 |

旧session_quality原样保留。新增consumer_admission与window_validity作为独立sidecar，先对同一固定数据并行计算旧门与能力门，验证误拒/误收。只允许纠正单位、范围、分母或字段错用；不得以达成目标数为理由降低适用数值阈值。门变化单独版本化并报告历史门、新门、同固定样本三列结果。

## 4. 最小任务组织与资源

一个父协调任务：建议语义ID `robot_quality_recovery_0915_15h_v2`，本地核对重名后注册。9个工作包即可，attempt留在包内，不再为了每个小文档拆出几十个节点。

| 包 | 主责 | 前置与可并行性 | 预算 |
|---|---|---|---|
| P0 证据闭包 | publisher | 启动核验后立即；与合法只读诊断并行 | 首轮最多45min |
| A0 拒绝与依赖审计 | Lane A | W0缓存/门代码可读取 | H0–H1.5 |
| A1 HaWoR质量恢复 | Lane A | A0已定位原因 | H1.5–H4.5主修复 |
| B0/B1 SAM依赖审计、独立提示与时序 | Lane B | H0即启动依赖审计；合法RGB/固定SAM，无需A1 | H0–H4.5 |
| B2 已有Depth到Object | Lane B | 对象实例逐个通过即可，复用Depth | H2–H7，后续批量复用 |
| C0/C1 尺度与Contact可观测性漏斗 | Lane C | H0即启动漏斗；已合法样本缓存/原合同 | H0–H4.5 |
| C2 遮挡与交互验证 | Lane C | C1原因分型、局部输入可用 | H4.5–H8.5 |
| A2 R0恢复/限量R1/R2/批处理 | Lane A + publisher调度 | 分能力/窗口触发，不全链一刀切 | H4.5–H13.5 |
| F0 回归与封账 | publisher | 过程持续验收，最后集中 | H13.5–H15 |

预算为并行墙钟区间，不是单线程可相加的工时承诺。每个主原因最多一个主修复和一个因明确故障产生的successor；连续30min无新证据换独立就绪任务，不无休止试参数。

GPU只有一个模型owner。优先：HaWoR原因明确的小修复/W1-DIAG → SAM对象/手提示修复 → W1-ADOPTION → 已采用能力的H9 HOLDOUT；FoundationStereo重算只在局部失败明确且剩余时间足够时排队。不要为了保持GPU忙给204条未封存库存无谓跑Depth。

CPU并发服从实际内存、核数和IO，预留约25%系统资源，此为运维预算而非质量门。worker独占代码任务/运行目录，共享schema/任务指针只由publisher写；现有调度不支持并行代码写入时，worker产生隔离补丁，publisher串行集成。

## 5. P0：R2合同SHA漂移，闭包而不洗白

读取旧R2 RESULT中的确切SHA、字节数、技术合同内容范围。用户提及4,438字节仅作为定位提示，不以长度相同代替SHA一致。

- 能恢复原始exact bytes：放入新任务的不可变evidence_snapshot，验证SHA，发布resolution sidecar，旧RESULT不改。
- 找不到：为受影响R2资产发布tombstone/lineage撤权；最终引用审计可确认撤权闭包，但旧R2不因此质量通过。
- 若查明只是技术合同夹带gov revision导致每次发布重写，拆开技术snapshot和治理引用；不得删除字段后谎称仍是原SHA。
- 新运行每次绑定完整技术快照SHA。publisher刷新治理不修改该文件；新增版本走新路径/新receipt。

到45min仍未闭合，停止寻找旧字节，记录缺失并隔离R2，使用现有治理允许的撤权/阻塞流程继续独立合法任务。如果治理机制本身不允许合法新任务注册，就明确该真阻塞；禁止绕过任务系统。

P0 与 A0、B0、C0 同时启动。P0 的终态只控制新 R2 lineage；即使 P0 发布 tombstone，HaWoR、SAM、Depth、Object、R0 和 Contact audit 仍继续使用各自合法输入。禁止把旧 R2 的一个 SHA 漂移扩成全局算法阻塞。

重新跑引用审计并加回归：“更新治理元数据后，技术合同SHA不变；更改安装数值则旧R2消费者拒绝”。证据闭包PASS与R2算法质量PASS分开。

## 6. A0：W0四会话失败拆解，先看代码和坏帧

输出 `W0_FAILURE_MATRIX.json` 与四份整片/坏帧对照。每个会话、手别、帧/窗口记录所有拒绝位与第一拒绝原因。保留固定全时间轴分母，不只列剩下的有效帧。

必须回答：

1. 首次失败出在detector/track、raw MANO、bounded处理、质量计算、retarget、Robot FK/碰撞还是聚合调度？
2. 丢失的是整只手、少数帧、一根手指、相机世界轨迹，还是单个超限标量？
3. W0的0/4是否由共同代码错误造成？是否把offscreen、合法缺失或另一只手缺失计入了不适用的门？不得预设答案。
4. 时序门是否正确使用dt，是否跨长gap计算速度，是否把度/弧度、radius/diameter、mm/m、图像宽高混用？
5. “骨长变化”是在MANO固定拓扑上的3D骨段，还是2D投影/深度/机器人不同骨架上测到的变化？
6. bound update饱和、重投影回退、身份翻转、边缘截断分别多少？raw和bounded有无反向退化？
7. R0被拒是继承上游session flag，还是自身存在真实限位、碰撞、映射或时间问题？
8. 47节点中多少实际模型推理、多少算法求解、多少只有验证/治理？仅作时间与依赖诊断，不把节点多寡当绩效。

固定三类评估：原strict gate、修复后的同口径strict gate、另行登记的consumer/window gate。若只改变评价语义而轨迹没变，只能叫评价/依赖修复，不能叫HaWoR精度提升。

坏帧选择为固定时间分桶 + 所有首发错误及其前后帧，显示RGB、raw/bounded骨架、状态、数值和拒绝码。没有人工真值时不声称真实3D误差被测准。

## 7. A1：同权重HaWoR恢复，不做全片过度平滑

HaWoR官方把camera-space手重建与world-space相机轨迹分开，本轮也必须分开检查；官方存在运动补全，不代表本项目可以把补全写成直接观测。[S1]

### 7.1 只实现与失败原因对应的候选

- 检测/裁剪错误：核查xyxy/归一化框、crop padding、resize回映射、left/right mirror、帧索引和track关联。局部detector重检限已有模型，重检帧保留来源。
- 左右身份：利用稳定锚点和时序关联，遇交叉/遮挡不强制最近邻。单帧标签变更不得无证据地重命名整段。
- MANO形状：沿用每只手/可识别录制组的稳定beta；只在当前约束内调整。不同录制者不共享手形；机器人骨长不动。
- 平移/旋转抖动：沿用有界Whittaker/SO(3)思想，分别处理相机运动和手部真实变化；接触前后的快速真实动作不应被抹掉。重投影与raw回退保持。
- 短缺口：已有short_gap规则独立层复用。raw observed状态不增加；时序滤波或未来帧支持明确noncausal。
- 若主要失败来自world轨迹，单独输出camera-relative姿态准入；不能把相机运动造成的全局失败机械传给wrist-local q22。

### 7.2 W1诊断不再以W0整波通过为前提

W1-DIAG固定为Poker044与Chips097，不按后验好坏挑选。模型运行仅要求输入/资产/运行时正确，产物先为diagnostic，不自动进入R0。

先完成两条W1-DIAG以检验W0原因是否普遍。新故障若说明同一输入域/模型运行普遍错误，则停止该分支；若只是样本局部质量差，不能因此把其他层级预先判失败。随后先冻结候选，再打开W1-ADOPTION的106/029；H9四条只由采用决策解封。

每条完整原时间轴保留。已有合法模型输出复用；只有受明确修复影响的阶段重算。报告新推理成本与缓存复用，不把重复跑四会话当“新增四条”。

### 7.3 采用标准

固定W0原分母上，主要错误指标改善且覆盖/其他质量无不合理退化；未参与调参的W1验证同类改进。最小窗口长度、运动上限读取原消费者合同，新增量化指标在验证前冻结，不在看到结果后挑阈值。

整会话原严格门未通过就继续记拒绝。局部窗口可以另外有用途准入，但仅合格窗口使用，保留全部invalid间隙及覆盖率。不以丢弃困难帧使median/P95变好。

## 8. B1：SAM物体、手部独立启动与时序恢复

SAM官方支持text/visual prompts的图像与视频分割，HaWoR严格3D不是该模型的固有输入前提；本地封装应按真实需要改依赖，具体调用以本地pinned SAM3.1代码为准。[S2]

### 8.1 独立提示优先级

- Object：合法RGB + 已有task短类别提示，或同会话已确认的visual box，多个实际实例分开。不同会话不能复用旧像素box。
- Hand：当前帧有可靠2D检测/骨架框时作为optional prompt；否则由SAM自身hand候选初始化。不能把同手3D尺度未通过等同“图像里没有手”。
- 2D候选只证明实例位置候选，不证明3D、左右身份或Contact。左右身份不明时允许几何实例追踪，禁止伪分left/right。
- 分割恢复不得与HaWoR互相自证：SAM帮助crop后，还需固定图像/时间验证；同源预测一致不是独立GT。

### 8.2 时序故障定位与修复

每帧分开保存raw candidate、admitted semantic、reason、model track ID、physical instance ID、prompt来源、传播方向。定位raw没有输出、返回帧索引错、state重复清空、角色之间state误复用、还是quality gate丢弃。

已有每帧model输出不能由“propagate函数yield了150次”替代。处理长度变化、双向传播合并、重播种后旧缓存失效范围、状态是否真正更新。

质量触发重新找可见seed，有限box/point refinement + 相应局部重播。不用固定周期不停重初始化，也不以hold前帧mask伪装成功。相似牌无法区分时记identity_unknown，geometry-only实例可保留，不生成带ID的Contact。

tracking_state与visibility_state保持两轴；遮挡/离屏不能误算为已证实不存在。缺人工maskGT时用固定可见证据帧、重现一致性与审阅对照，不把自设unknown从分母删掉。

### 8.3 本轮不重新做Clean

主力只修hand/object。forearm、sleeve、cable保持现有状态，不为Clean重新调整膨胀/HSV；即使主体改善也不自动重启Removal V2或inpaint。这是任务优先级限制，不是宣称附件已解决。

## 9. B2：先消费12条既有Depth，而不是优先W2扩深度

缓存完整性通过后，固定12条Depth、双眼回域、原权重不重跑。对SAM已准入的对象逐实例产出几何：

- Poker：有限可见平面、法向、可用面内方向与边缘。visible centroid不是物体固定中心，遮挡使其变动不能直接判物体跳跃。
- Chips：包装袋可能变形，发布局部点云/表面片和可信运动线索，不以一张刚性平面或全袋6DoF统一替代。
- 未知完整尺寸不影响可见局部输出。估计尺寸需真实边缘跨帧支持，与同源Stereo不能相互认证外部尺度；本轮不专门花小时追完整矩形。

### 9.1 局部深度质量图

分手掌、指部、物体内部、手物边界、背景、遮挡区记录有效率、LR一致、局部残差、视角/深度范围、拒绝原因。全图PASS不当接触区毫米精度认证。FoundationStereo本身输出视差，公制转换仍依赖适当内参/基线。[S3]

最多开一个已有权重的局部质量候选，例如原图更高有效分辨率或受支持的迭代设置；必须由明确局部失败和预算触发，先少量完整合法stereo帧对照，不能任意单眼crop后直接推理。raw与候选分别存，不跨手/牌边界平滑，不把填补视作观测。

没有独立毫米真值时报告internal/residual与误差来源未知，不能从平面拟合残差小就宣称绝对深度准。

## 10. C1-A：尺度原因审计必须跨现有会话，而非再拟合001

0.783730是某关联/目标下最优参数，并非单独证明MANO或Stereo坏。已有001与参与过审计的holdout转为development，本轮需自动找现有非接触片段和未使用的验证source group；找不到时限制结论，不索取外部资产。

### 10.1 三层核查

1. 变换与单位：optical-Z/ray length、mm/m、wrist/camera/world、crop/resize/pad、K/P、baseline/主点差、镜像回域。
2. 表面对应：MANO可见前表面z-buffer、同像素同物体支持、skin/sleeve分类、边界/遮挡剔除；不拿关节中心或背面最近邻对应可见Stereo表面。
3. 参数可辨识性：尺度作用于局部手形、camera平移还是world轨迹；固定尺度刚体、单尺度、深度offset等最多三种预登记解释，避免同时自由拟合K、beta、s、offset吸收所有误差。

设图像变换为A，投影矩阵按其真实语义变换。水平镜像可写F=[[-1,0,W-1],[0,1,0],[0,0,1]]；在同一3D基下FK会包含负水平焦距，不能只改cx而暗中改变3D手性。优先输出反镜像回物理左域后使用已核验的encoded K/P，不改现有正确像素路径。[S4]

按手别、近远、图像半径、视角、时间块和附件分层，检查s与offset耦合及触边情况。optical-Z与ray length混用是待检验假设，不预定为原因。

### 10.2 采用边界

实现错误明确：修对应变量并跨未用片段验证，原输入保持；表面污染明确：修对应而非缩放机器人；真形状/尺度适配：只允许独立支持的新development alignment sidecar，不修改历史边界门来求Contact通过。

独立性无法证明或公制误差仍不辨识：严格对齐继续拒绝，报告未决原因，停止在001反复搜索。不会阻塞同Stereo域可见表面关系和R0局部手形。

## 11. C1-B：Contact零窗口的可观测性和门可满足性

这是本轮必做的新任务，不是放宽门。

### 11.1 把4500行变成逐门漏斗

读取真实配对清单，逐行记录：HaWoR observed、hand ID、finger关联、SAM可见、Stereo采样、单位/不确定度、object ID、有限patch、表面距离、时间连续性。固定4500分母只适用于001，其他会话按实际帧数和配对数。

同时报告unique surface_sample_id、重复用于多个object的pair行数，不能把同一个采样当三次独立观测。436条合格surface与仅4条patch内的条件语义需要代码核对。

### 11.2 检查是否存在定义性矛盾

如果采样明确要求像素属于 `hand_mask AND NOT object_mask`，而后续又要求**同一个像素**属于可见object_mask，则交集由构造趋于空。必须查代码是否这样写，不能未经检查断言已经存在该bug。

若后续用的是三维点向牌面正交投影，再检查投影足点是否在有限patch，二者不构成上述严格逻辑矛盾；仍可能因为接触区被手遮挡、mask边界侵蚀或错误投影，导致只剩极少候选。

画出四层：采样pixel、可见手表面点、平面正交足点、有限object patch。明确相机z方向与plane法向signed distance是不同量，不能用全局Z大小代替不同射线点的遮挡判断。

### 11.3 三种表面不能混用

- hand在相机可见的背面/皮套表面；
- 解剖学finger joint/tip或机器人pad；
- 与物体真正相接的隐藏内侧表面。

Stereo单个像素回投是可见表面点，不自动是接触面。[S4] 手物交互重建研究也明确把遮挡的接触区域当作需要先验/约束恢复的问题。[S5]

因此“6.60mm大于5mm”只能说明当前观测测试没有通过；不能推出真实无接触，也不能减去假设皮套厚度变为通过。

### 11.4 最小可满足性测试

用现有几何工具生成已知同域简单场景，测试：可见接近但不接触、接触面可观测、真接触面完全被挡、投影落无限平面但不在实体范围、finger关联错误。测试不是训练数据或现场真值。

严格分支对隐藏接触应unknown，而不是要求它无论如何PASS；假设分支可产生解释但不得改写observed。测试应检出坐标/布尔条件/投影错误，而不是以合成PASS证明真实精度。

严格5mm及实际不确定度/连续性合同保持。只有原门实现错误修正且新验证通过，才改变合法消费范围；不得将无限平面、补全Mask或想象厚度放进严格分支。

### 11.5 LOCAL_STEREO_METRIC_DEV 与外部公制权威分离

每个窗口必须分别写 `local_stereo_metric_dev` 与 `external_metric_authority`。前者只有在手关联可见表面和物体表面来自同帧、同一 encoded Stereo depth，均已回到 physical-left 像素域，使用相同 K/P、baseline、镜像回域与 Depth SHA，手表面为直接可见 stereo sample，并且 object identity、finite patch、LR consistency、局部深度、像素注册与不确定度闭合时才可为 true。MANO absolute-Z、短缺口、时序补全、Removal 或 Contact-window 拟合 alignment 均不得进入该分支。

满足上述条件时，5mm只可作为局部 Stereo 坐标内的开发严格门，并允许 `r1_e_development_allowed=true`。无论是否满足，`external_metric_authority=false`、`control_ground_truth=false`、`physical_deployment_authorized=false`。Tactile 只能增加已有关联的支持度，不能创造几何接触点。

## 12. C2：离线遮挡恢复与事件证据，限量重开R1-H

先读上一轮R1-H拒绝原因。若没有新的输入证据、可复现实现错误或新的独立验证设计，不允许只换版本名再导出一次同样假设。

### 12.1 输出三条证据层

`interaction_observed`保持原当前帧观测；`interaction_temporal_reconstruction`记录离线推断；`contact_hypotheses`记录接触/非接触竞争解释。所有推断附支持帧、最大缺口、时钟、身份、noncausal、来源/反证，不能覆盖observed。

先对固定短遮挡窗口使用可靠两端锚点和可见物体几何。刚性物体的历史表面只能在有可信同实例位姿变换时搬到当前帧；不能以visible centroid位移替代刚体运动。遮挡区从历史搬来的点仍是temporal-derived，不是当前直接观测。Chips形变不能套整袋刚性传播。

### 12.2 两种不同Contact用途

- 严格surface evidence：保持既有metric门，只放行真的合格窗口；可以继续为零。
- Event/hypothesis evidence：2D关联、局部几何可行性、相对运动、遮挡重现、可选触觉时序支持，输出未校准support_score。可识别接近/可能夹持事件，但不声称接触点精度或力。

触觉须核实真实时钟映射、单位、通道/手指身份和offline_source_valid。无法定位物体时不能仅凭触觉峰给card00加Contact。没有触觉不阻塞其他证据。相机运动造成的全场同动不当抓取；接触后停顿不因没立即搬运被拒绝。

每窗口最多三种解释，必须含no-contact；不自由移动物体、缩放手和机器人去制造接触。当前观测与拟合目标共享来源时，不当成多个独立证据相乘。

### 12.3 验证与采用

在预先固定的可见段隐藏部分输入观测，重建后与留出观测比较；需要重新屏蔽/计算所有依赖该观测的缓存，尤其不能用已经看过这些帧的时序HaWoR/SLAM/特征缓存冒充无泄漏验证。

预算不足无法证明缓存独立时标“回放诊断，非独立验证”，不得据此采用R1-H。真实遮挡重现检查只验证重现状态，不认证隐藏过程真值。

只有候选比固定R0/无修复对照有新增解释力，且身份、可见几何、碰撞、时序和反证无退化，才进入R1-H。若不可区分多种解释，保留歧义/拒绝，不硬选最低loss叫真实动作。

## 13. A2：Robot恢复必须独立检查自身质量

### 13.1 R0先查映射和求解，不只继承HaWoR拒绝

审计 joint_names顺序、rad/deg、左右手URDF、轴/符号、MANO到Kai局部旋转、FK link来源、限位、非邻接自碰撞、时间戳、初值连续性、边界速度。有限FK不等于姿态合理，也不等于无自碰撞。

使用原生Kai骨长，human方向/闭合度作软目标，关节限位与真实机器人碰撞按固定用途门检查。不要把人手尺寸差强制变成机器人指尖绝对位置差。不可达目标应报告，不缩放mesh、不事后clip关节伪合格。

R0 q22与wrist frame分开准入：合法局部q22可以独立保存；camera wrist不自动升级为world wrist。缺帧/坏帧明确invalid，所有视频仍保留完整原时间轴。离线推断轨迹另层，不改observed。

### 13.2 严格整会话与局部窗口同时报告

优先目标仍是原严格质量恢复。新局部准入用于开发能推进，不用于美化整会话数字。完整视频如果大段invalid，必须在首页显示实际有效时长/覆盖；不能只展示最好一秒称批量成功。

### 13.3 R1只做有新依据的窗口

R1-E须满足该窗口自身公制/身份/Contact门。R1-H需C2新验证通过，且不能以“这是hypothesis”为由忽略未对齐wrist/object或机器手几何。

优化q22与有限wrist修正，固定机器人/合法物体参考，保留human prior。边界允许冻结长度的transition区，区外保持q_init；真实dt下校验速度/加速度，不放慢播放过门。

对同一冻结配对和时间轴报告before/after、切向滑出有限patch、覆盖、碰撞、prior偏离。训练内距离改善不能独立证明任务成功。初始已达容差则不强迫固定百分比改善。

本轮不下载ContactOpt，其研究仅支持“接触先验可以帮助优化”的设计思想；其软组织穿透建模不能复制为刚性KaiHand/薄牌安全容差。[S6]

### 13.4 虚拟R2从合格R0窗口出发，不从零质量整片盲跑

新技术合同先冻结exact bytes。明确T_A_B将B点变到A：

`T_base_handroot = T_base_flange @ T_flange_handroot`

如果求解器目标为flange，应使用正确逆变换，不能拿hand-root目标直接当法兰目标。区分视觉mesh原点、hand-root、tool、TCP、flange。左右安装/手性、SE(3)正交性与det=+1需校验。

先neutral FK与固定合成可达目标，再选一个合格R0窗口；先静态关键帧再完整轨迹，验证真实mesh/link坐标、IK误差、限位、非邻接碰撞、分支切换与速度。虚拟placement最多三个冻结候选，会话内固定；人体目标与物体场景整体同变换，不能逐帧移动base或只挪物体降低误差。

累计预算最多90min，仅在窗口通过后扩到最多两个会话。几何proxy缺失/不可验证则停R2，不拦R0。纯q22可用但腕部位姿不合格时，不生成伪装可达的整臂结果。

碰撞分机器人自碰撞、已观测有限面相交、隐藏物体/环境未知三层。开放面不等于闭合物体，不能把signed plane distance当完整穿透体积。[S7]

## 14. 15小时墙钟与强制里程碑

| 时段 | Lane A：HaWoR / R0 | Lane B：SAM / Object | Lane C：尺度 / 交互 | 必须有的产物 |
|---|---|---|---|---|
| H0–H0.75 | 读取W0坏帧/真实门 | 核查12条RGB/对象提示 | 核查对齐/Contact实现 | 新父任务；P0独立闭包/撤权；冻结manifest |
| H0.75–H2 | W0原因矩阵；定主修复 | 无全片HaWoR依赖的SAM试运行 | 尺度定义、Contact可满足性测试 | 拒绝来源与错误依赖清单，不是泛泛“质量差” |
| H2–H4.5 | 固定原因修复；W1首两条诊断 | 手/object时序修复，复用Depth | 独立非接触对照、surface漏斗 | 首次真实前后视频；W1诊断实际运行或真实资源故障 |
| H4.5–H7 | 冻结候选后只打开 W1-ADOPTION 106/029 | 同签名SAM/Object采用判定 | Contact开发资格判定 | `CANDIDATE_FREEZE` 与 `ADOPTION_DECISION`，结果不得触发返修 |
| H7–H9 | 已采用窗口R0；限量R1/R2 | 冻结主体SAM/Object能力 | 新假设独立验证/拒绝 | H9冻结所有能力；无采用能力标 `NO_ADOPTED_FIX` |
| H9–H11 | 只运行已采用能力对应的四条 FINAL HOLDOUT | 已采用能力的冻结推理 | 不用holdout调参/拟合 | 冻结后泛化结果、主体12条事实矩阵 |
| H11–H13.5 | 同时满足扩批门才打开 EXTRA FINAL 4条 | 只运行已准入消费者 | 结果审计与失败簇 | 未达门写 `NOT_OPENED_EXPANSION_GATE_FAILED` |
| H13.5–H15 | 排空本轮模型/求解 | 完整解码与哈希 | 回归、碰撞/坐标审计 | 单次最终发布、无进程、Git clean、不push |

H4.5若W0没有任何质量改善且W1原因同型，A线切换为“最小窗口根因修复+明确能力边界”，不虚耗批量R2。B线仍可用合法Object/Depth继续，C线仍可做可观测性实验。

H7若没有任何可用R0窗口，暂停R1/R2，资源回到Hawor/retarget定位或已就绪Object任务。H9仍无候选可采用，就冻结“未采用”结果，不强推批量成功；本轮执行目标未达到如实报告。

H9后不新增算法，只用冻结版本处理已授权清单。运行故障修复涉及数值变化时需要新签名/最小回归，否则不进入批次。H13.5后不启动新会话，按实测p90耗时+30%余量判断是否有时间完成，不能只看平均。

## 15. 运行纪律与有限继续

任务开始记录真实T0与monotonic deadline，耗时包括注册、失败和收尾。每60s进程心跳、每10min `_run/current`原子快照；心跳来自真实PID/启动时间/帧进展。受治理文档只在里程碑/最终发布更新。

连续15min无进展先检查GPU/CPU/IO/日志再判断卡死；不可把“没有新日志”自动等同进程已死。仅终止本轮所属子进程树，确认结束后释放租约。

未就绪与未实现不同：接口未接好是本轮工程工作，不写BLOCKED_EXTERNAL逃避实施；原始公制真值缺失则不能承诺算法可以证实。

每条就绪任务必须满足真实数据/资产/租约/预算；某个分支失败不全局提前收工。但合法工作已做完或预算已不够时，允许提前诚实封账，不空转凑15小时。

## 16. 同源资产、缓存与发布规则

- 复用既有 `_run/current/<task_id>/attempts/attempt_NNNN/` 目录规范。
- 原始、processed、sealed、frozen exact78、旧失败记录全部只读；旧错误图像域和旧安装结果不当基线。
- 新模型推理缓存绑定input/domain/code/config/weights/env；后处理与consumer gate变化不能无意义失效无关Depth。
- 结果绑定本次读取技术合同的exact bytes/SHA，不能只绑定一个后来会重写的current路径。
- 统计以真实RESULT/receipt为准，不用文件数量/目录存在判成功。未运行不生成空q轨迹当占位。
- 修改共享接口/当前文档由publisher串行执行。技术内容与gov revision引用分离；变更真实技术参数必须新版本，不规避哈希审计。

## 17. 实时算法审计与最小文件集

使用[算法审计文档](02_ALGORITHM_AUDIT_LIVE_ZH.md)的字段；它是现有receipt/合同的派生视图，不是新权威层。

建议本轮attempt保留以下内容，原目录结构可适配，不能声称尚未创建的路径已经存在：

```text
input_manifest.json
contract_snapshots/
  snapshot_index.json
failure_audit/
  W0_FAILURE_MATRIX.json
  CONSUMER_GATE_AUDIT.json
  CONTACT_OBSERVABILITY_AUDIT.json
algorithm_audit/
  LIVE_STATE.json
  EXPERIMENTS.jsonl
sessions/<actual_session_id>/
  human/raw_reference.json
  human/corrective/
  masks/raw_and_semantic/
  object/observed/
  interaction/observed/
  interaction/hypothesis/
  robot/r0/
  robot/r1_evidence/
  robot/r1_hypothesis/
  robot/r2_virtual/
  review/
  RESULT.json
release/
  BATCH_RESULT.json
  FINAL_AUDIT.json
  README_ZH.md
```

V2.1 机器合同至少包含：`SOURCE_GROUP_PROVENANCE_CORRECTION_V2`、`COHORT_ACCESS_LEDGER_V1`、`W0_FAILURE_MATRIX_V2`、`CONSUMER_ADMISSION_V1`、`WINDOW_VALIDITY_V1`、`CONTACT_OBSERVABILITY_AUDIT_V1`、`LOCAL_STEREO_METRIC_DEV_V1`、`CANDIDATE_FREEZE_V1`、`ADOPTION_DECISION_V1`、`TECHNICAL_CONTRACT_SNAPSHOT_INDEX_V1`、`LINEAGE_TOMBSTONE_V1` 与 `ALGORITHM_AUDIT_LIVE_V2`。所有 sidecar 绑定输入/代码/配置/权重/技术合同 SHA；名称存在不代表状态 PASS。

旧47节点封账状态不改。入口文档引用最新本轮发布，不把较早单样本叙述当新批量状态。历史数值保留上下文，旧36.79/36.91等只作为旧方法结果，不覆盖最新19.33或本轮实际指标。

## 18. 最终验收与报告

### 18.1 真实质量，不止测试数

按[测试矩阵](04_ACCEPTANCE_TESTS_ZH.md)运行相关测试和原回归。单测通过不能替代实片。出现可证伪的共同错误至少需要实片复测；仅改文档、门定义或消费依赖时不得宣称模型精度提高。

新增时序/分母/单位门以相同冻结样本验证，显示覆盖与误差两轴。不能删除失败帧、把unknown改为offscreen、把whole-session改名window后算整会话通过。

### 18.2 必须交付

1. W0四会话逐手逐帧失败矩阵和同片对照视频，说明哪类改动实际采用。
2. W1-DIAG、W1-ADOPTION、H9 HOLDOUT各层会话的真实运行/未运行、原始推理/后处理/R0状态与预算原因。原ID不重命名、不重分母。
3. R0严格整会话成功数、局部可用窗口数/有效时长、真实轨迹与完整时间轴审阅。
4. 同12条Depth的复用清单与Object可观测性，不能将仅Depth通过计为Robot完成。
5. Contact可满足性测试、采样→patch漏斗、尺度跨片段结论；遮挡回放和真实重现分开报告。
6. R1-E/H、R2实际尝试/导出/采用/拒绝分别计数；无新合法尝试则写未运行及真正原因。
7. 固定技术合同快照、复查过的引用闭包、真实批处理与resume命令。
8. 两份文档的里程碑更新、最终简报、本地提交；所有本轮任务终态，无worker/GPU lease/writer，不push。
9. 浅层目录 `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/`，包含 W0 before/after、W1-DIAG/W1-ADOPTION、Contact全漏斗、SAM/Object identity、Kai22 local-window/session-strict、H9 holdout，以及仅在解封时才生成的 EXTRA FINAL 汇总。

### 18.3 最终计数

`inventory=220`仅在实际核实后使用。冻结主体12为固定分母，条件额外最多4另列。至少同时报告：

- inferred_model_sessions / exported_sessions / strict_quality_sessions；
- per-hand usable frames、usable duration、longest valid window、unknown/offscreen/occluded；
- R0严格通过与局部consumer-admitted分开；
- R1-E、R1-H、R2各自成功数；R1-H永远training_eligible=false；
- 结构/引用审计通过与算法质量通过分开；
- physical_deployable=0。

最终算法分级只允许：`TARGET_MET`、`PARTIAL_MATERIAL_PROGRESS`、`REJECTED_NO_RECOVERY`、`INCONCLUSIVE_RUNTIME`。任务终态、测试通过、导出数量增加均不能单独构成算法成功。必须独立报告 `HaWoR session strict`、`R0 local DEV_VALID windows`、`R0 strict quality`、`LOCAL_STEREO_METRIC_DEV`、strict Contact windows、R1-E/R1-H/R2 adopted、holdout generalization 与 `physical_deployable=0`。

**本轮目标是把“为什么全被拒绝”变成可修复的具体问题，恢复至少一条可跨会话复用的高质量Robot基础链，并验证遮挡推断能否提供新增价值。若未做到，应明确目标未达，不用更多终态节点或更漂亮状态表代替。**
