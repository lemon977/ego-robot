# chaoyang：新数据到 Robot 的 15 小时并行执行任务书

版本：v1.0  
文档性质：待本地注册的执行计划，不是运行结果或调度授权凭证  
编制日期：2026-09-18  
基准来源：用户本轮报告，提交 `4228645`，治理 `12075 / gov-012075-a3a8a7cce875`  
运行窗口：从本地 coordinator 实际启动并写入 `started_at` 起 15 小时，不从聊天发送时间起算  
项目：机器人数据工程 `/mnt/workspace/code/chaoyang`；不得误接同名股票研究项目

> 本轮必须同时推进算法证据与真实批处理。不能把“注册任务、增加单测、输出 BLOCKED 表格”当作批量 Robot 已完成；也不能为了产量，把推断写成观测、把数值优化写成真实接触、把虚拟机械臂写成真机可部署。

## 0. 先读结论

本轮有两个并行主目标。

**交付目标：** 建立并实际运行“新会话 → 输入域验证 → HaWoR → Kai22 R0 → Robot 轨迹文件/审阅视频”的批处理入口，并按实例、帧和证据层增量加入 Depth、Object、Interaction、R1、R2。优先完整处理预先选定的 12 个新会话，覆盖 Poker 和 Chips；之后按实测吞吐与剩余时间扩展到冻结清单中的其他合格会话，最多 220 个。12/220 是目标和上限，不是已经具备或保证能完成的数量。

**研究目标：** 查明或缩小 `0.783730` 尺度异常原因，改善指部/物体局部深度质量与遮挡后身份恢复，完成独立的离线交互假设重建和机器人任务约束轨迹求解。严格 Contact 没有证据时继续阻塞严格 R1，但不阻塞 R0、虚拟机械臂验证或明确隔离的 R1-H 假设实验。

不等待用户提供新尺寸、TCP、外参、手套标定或新录制视频。优先从现有采集资产自动寻找验证片段。数据不足时如实限制声明范围，不反复向用户索取外部真值。

## 1. 启动快照与不可误读事项

以下均为用户报告，启动时必须以本地 receipt 核实，不能手工写成已重新验证。

| 模块 | 报告状态 | 本轮处理 |
|---|---|---|
| 调度 | 无活动任务，`next_task=null` | 注册新父任务，不能复活历史任务 |
| 0916 清洗 | 240 终态，222 CLEANED，18 REJECTED | 只读；CLEANED 不自动意味着 Robot 准入 |
| VST 图像 | encoded SBS 按眼裁切与 resize | 不做旧 lens-undistortion/equiDis62 remap |
| HaWoR bounded v2 | 当前单样本视觉已接受，293 direct side-frame | 当前样本缓存冻结；允许新会话按固定版本运行 |
| SAM3.1 | 严格角色部分可用，弱角色失败 | 固定模型，研究时序生命周期/重播种/身份，而非长关键词 |
| Removal V1/V2 | 均 REJECTED_QUALITY | 不调半径，不重启；Clean 不阻塞 Robot 数据与几何链 |
| encoded FoundationStereo | 150 帧内部质量通过 | 当前结果不可覆盖；新会话可运行固定权重 |
| Object6D v2 | 有限可见表面、方向与部分字段可用 | 独立实例；可见中心不当物体固定中心 |
| Human/Stereo 对齐 | 受限 P90 19.33 mm，s=0.8 命中下界；无约束 s=0.783730 | 原授权仍拒绝；开展独立原因审计 |
| Contact | 最近有限牌面距离 6.60 mm，严格 5 mm 门下零连续窗口 | 不改历史门、不声称无物理接触 |
| Kai22 R0 | 已交付 | 本轮首先使其可新数据批处理 |
| R1 | BLOCKED_LOCAL_EVIDENCE | 保留；新假设实验不得冒名顶替 |
| R2 | 未运行 | 明确虚拟安装后可从 R0 独立做 arm IK |

`6.60 mm > 5 mm` 只解释既有算法未准入，不证明真实动作没有接触。`0.783730` 是某目标函数下的最优参数，不独立证明 Stereo 错了或 MANO 错了。

## 2. 本轮授权范围与冻结边界

### 2.1 允许本地注册后的自主操作

- 读取当前权威入口、实际代码、当前资产和原始/processed 的必要元数据；建立新任务包与新运行输出。
- 对当前样本使用只读缓存作审计。新增校正只能在新 successor 中运行，原产物不覆盖。
- 对冻结新数据清单运行现有固定 HaWoR、SAM3.1、FoundationStereo。不能把上一轮“当前单样本不重跑”误读为“所有新会话禁止推理”。
- 增加有限算法修复、CPU 优化器、批处理适配、回归测试、可视化及调度保护。
- 在当前资产目录与权威审计中寻找开发级安装 proxy；保持来源、SHA 和用途限制。
- 允许离线短窗推断、局部机器人轨迹合成，但单独输出并标注推断依赖和非因果性。
- 不需要每个批次再次询问用户。只有范围外数据删除、付费、联网账户操作或真实硬件控制才不在本轮授权内。

### 2.2 禁止

- 修改源数据、processed 发布内容、sealed 结果、frozen exact78、撤权 Depth、历史对齐边界审计和失败 Removal。
- 自动换 SAM2.1/Cutie/其他大模型；不升级环境、驱动或模型权重，不启动训练/RL 或新框架迁移。
- 仅为让旧样本通过而把尺度下界改为 0.75、把 5 mm 改为 7 mm，或统一减去一个残差中位数。
- 机器人 URDF/mesh/骨长/轴/限位缩放；真实物体几何也不能被优化器随意移动来降低接触损失。
- 用 inferred mask、Clean、inpaint、Robot render 当 Depth/Contact 直接观测。
- 把标准扑克牌名义尺寸、假设厚度、安装 proxy 当实测真值。
- 将 `R1-H` 写入既有严格 `R1` 字段、发布假接触概率、以自拟合目标改善自证成功。
- 全批统一抹掉缺失手；将非因果补全用于在线控制或在线评估输入。
- 远端 push、真机连接/下发、无所属任务的后台进程。

### 2.3 运行前核实

按现有入口顺序读取 receipt、最小状态、DOC_AUTHORITY_MAP、ALGORITHM_CONTRACT、任务 INDEX 和本任务 read_set。执行项目已存在的治理与结构校验，不发明新的权威体系。

若当前提交高于 `4228645`，记录差异并使用合法更新，不回滚用户的新工作。若有活动任务，不能擅自终止或抢占。若起始工作树非干净，保护已有改动并隔离本轮；不能 reset/clean 达到假干净。

本轮启动后的第一条运行消息必须包含真实 `run_id`、父任务 ID、子任务/worker 状态、开始时间、截止时间、输入清单和下一项实际命令。不能只生成计划后退出。

## 3. 两条数据链与证据隔离

```text
A. 必交付基础链
新数据 → 合法 encoded 输入 → HaWoR observed/prior → Kai22 R0
                                                     ├→ 轨迹包/独立3D视频
                                                     └→ 可选虚拟机械臂 R2

B. 增强链
Stereo + SAM Object/Hand → observed surface evidence
                         ├→ 严格局部 Contact evidence → R1-E
                         └→ 时序/任务假设重建 → R1-H

R1-E 或 R1-H 均可接 R2，但不是 R2 的唯一入口。
Clean 是旁路展示能力，不是上述任一数据链的前置条件。
```

### 3.1 数据证据类别

| 类别 | 含义 | 可消费范围 |
|---|---|---|
| OBSERVED | 当前输入中的直接观测或明确来源的模型测量 | 按质量门进入对应观测证据链；不等同人工 GT |
| DERIVED | 由合法观测计算的距离、平面、变换等 | 必须保留完整依赖与不确定性 |
| TEMPORAL_INFERRED | 依据前后帧估计遮挡/缺失状态 | 离线 hypothesis/visual；不能改 observed 标志 |
| TASK_HYPOTHESIS | 接触、抓取或任务约束下推断的状态 | 只用于显式 hypothesis 求解和诊断 |
| ASSUMED_PROXY | 虚拟安装、显示坐标、代理形状等假设 | 仅声明的开发/可视化用途 |
| UNKNOWN | 无足够证据或不可区分 | 不填零、不默认没有接触 |

数据质量等级与物理部署权分开。所有层本轮都是 `NON_CONTROL / NON_DEPLOYABLE`，自动通过不等同用户视觉验收。

### 3.2 Robot 交付层

- **R0：** 固定几何的 Kai22 retarget、wrist prior、有效性与可视化，不声称接触成立。
- **R1-E：** 用通过独立局部准入的交互证据进行 refinement。兼容当前严格 R1 范围，不降低门。
- **R1-H：** 在独立假设文件中求“能解释观测并满足任务约束的机器人候选轨迹”。它可以在严格 Contact 未通过时研究，但不能改写该失败，也不能声称恢复了真实隐藏动作。
- **R2：** 使用明确虚拟 base/安装 proxy 计算机械臂关节轨迹和 IK/碰撞审阅，可接 R0、R1-E 或 R1-H。缺实测标定不阻塞虚拟 IK；几何资产或虚拟安装变换不明确则只交付 wrist-local 手。

R0 批量成功不等于 R1 成功；R1-H 成功不等于严格接触通过；R2 成功不等于真机可达安全。最终分别计数。

## 4. 并行架构：三个 worker，一个唯一发布者

| 角色 | 主责 | 可写范围 |
|---|---|---|
| Coordinator / Publisher | DAG、GPU 租约、共享接口、合并、事实视图与最终发布 | 当前合同/任务索引/审计视图的唯一写者 |
| Lane A：Geometry | 尺度原因、encoded K/P、局部深度可靠性、对齐校正 | 分配的几何代码/测试/独立 attempt |
| Lane B：Interaction | SAM 时序、身份、遮挡重建、严格与假设 Contact | 分配的时序与交互代码/测试/独立 attempt |
| Lane C：Robot & Batch | R0 批处理、轨迹优化、虚拟 arm IK、可视化/吞吐 | 分配的 Robot/批处理代码/测试/独立 attempt |

不要同时保留“全项目只能一个 executable”与“三个 worker 并行”两种相矛盾规则。父任务是唯一顶层 executable；其冻结 child packets 在当前调度器支持的命名空间内调度。若现有调度器没有这种能力，CPU 子任务可以先并行只读审计和产生隔离补丁，合并及模型运行仍由唯一 coordinator 串行执行。禁止临时创建绕过租约的私有调度器。

优先复用现有隔离机制。确实需要 worktree 时，只在被批准运行根下创建；记录基准提交、任务所有者和文件所有权，不引入 repo 外散落副本。不同 worker 不编辑同一合同、同一当前文档、同一 manifest 或同一输出路径。

### 4.1 资源与进度

- 一张 GPU 同时只有一个模型任务。HaWoR/SAM/FoundationStereo 按冻结小批次分阶段运行，减少模型反复加载。
- CPU 并发由启动检测的实际核数、内存和 IO 决定，预留至少 25% 给 coordinator/系统；不要把历史 H20 配置当当前资源事实。
- GPU 队列优先保障新数据准入与 R0，不允许 SAM 附件探索占满窗口。
- 在 H8 前尽量将 GPU 时间至少一半留给固定模型的新会话处理；探索超时则先停探索，而非停批处理。
- 实际 worker 进程每 60 秒心跳一次；每 10 分钟更新一次机器状态视图。记录真实 PID、进程启动时间、任务 ID、已处理帧、最近推进时间，不依据聊天猜运行中。
- 连续 15 分钟无进度时先看真实 CPU/GPU/IO/日志；确认停滞再终止所属任务树。心跳缺失不等于 GPU 任务已停止，释放租约前检查实际进程。
- 不得为了跑满 15 小时空转。完成所有准入工作可提前封账。也不得完成一个小任务就整体退回无下一任务；队列中有合法独立任务必须继续。

## 5. 数据选择、留出与分波批处理

### 5.1 建立新清单，而不是续跑被取消的 220 全批

枚举当前合法 0915 数据；0916 中只有实际含所需 RGB/时间戳/相机域且任务兼容的会话才可加入。源根和 processed 均只读。冻结清单上限为 220 个合格新会话，实际数量由 inventory 决定，不能沿用历史 220 作为实数。

每行至少包含：session_id、task、source_recording_group、采集域与设备配置、输入 SHA、fps/逐帧时间戳、总帧数、手套/皮套配置、Stereo 可用性、对象路由、缓存签名、选择阶段、未准入原因。

不得混用不同镜头域/设备域标定。不同设备域独立 preflight。未知域标记 `BLOCKED_IMAGE_DOMAIN`，不把错误旧 remap 作为备用方案。

### 5.2 独立性

- 当前 `play_cards_0915_001` 及所有已被反复检查的样本属于 development，不再称作最终 holdout。
- 尺度审计先从现有数据找清晰、非接触、无明显附件污染、多个视角/距离的片段。
- development/validation/final-holdout 优先按独立 source recording group 分开；同一长录像裁成多个 session 不能当独立重复实验。
- 启动时预先排序候选与替补。不能看哪个 Contact 最容易 PASS 再决定测试集。
- 可复用冻结模型独立推理产生的缓存，但不得用 holdout 的隐藏标签、后验结果调参数。
- 找不到独立数据时限制跨会话/跨任务声明范围，但不阻塞可读新数据的 R0 处理，不要求用户重新采集。

### 5.3 三个波次

| 波次 | 默认规模与用途 | 进入下一波的条件 |
|---|---|---|
| W0 | 4 个新会话，尽量 Poker 2 + Chips 2；完整长度，不全截成 150 帧 | 两类任务分别通过输入/映射/时间戳/轨迹文件/真实渲染检查 |
| W1 | 目标累计 12 个完整新会话，尽量各 6，含未调参留出源组 | 冻结实现/阈值，真实批处理与断点续跑测试通过 |
| W2 | 当前清单剩余合格会话，最多 220 个，总量受预算限制 | 每个采集域通过自己的准入，预计可在封账前完成 |

W0 某采集域失败只阻塞该域扩展，其他域继续。W1 未达到 12 个就不能称达到目标；不得用“12 个终态，其中 12 个 BLOCKED”冒充批量转换成功。

H3 前测到至少一个新会话的真实每阶段耗时；H7 和 H10 更新 ETA。新任务预计耗时加 30% 余量若超过剩余执行预算，不启动。13:30 后不启动新会话，14:00 前排空模型推理，剩余时间做验证与封账。

## 6. 15 小时墙钟计划

| 时间段 | Lane A：Geometry | Lane B：Interaction | Lane C：Robot & Batch | 强制交付 |
|---|---|---|---|---|
| H0–H0.5 | 核查域/K/深度依赖 | 核查证据和语义状态入口 | 清单、资源、R0 调用审计 | 父任务真实启动；两份文档已绑定事实源 |
| H0.5–H3 | G1 尺度原因审计，独立验证选样 | M1 SAM 时序诊断与局部修复 | B1 批处理骨架；W0 预处理/R0 | 尺度原因分型；至少一个新会话真实 R0 或明确运行错误 |
| H3–H6 | G2 唯一有依据修复/局部深度提升 | I1 遮挡推断，Contact E/H 分开 | W0 完整闭环；R2 静态/轨迹验收 | W0 真实机器轨迹与视频；模型资源不闲置 |
| H6–H9 | 独立验证，冻结合格几何版本 | H1 假设筛选，遮挡回放验证 | R1-E/H 有限优化，W1 启动 | H9 冻结本轮 release candidate，拒绝无限调参 |
| H9–H12 | 已冻结算法服务新会话 | 已冻结身份/交互服务批处理 | W1 完成，W2 按预算继续 | 实际成功数量、失败簇、吞吐报告 |
| H12–H13.5 | 不再研究新分支 | 只处理已知可复现故障 | 批处理/渲染/导出、resume 验证 | 批量 manifest、样本对照视频、终态清单 |
| H13.5–H15 | 回归、证据闭包 | 审阅/误判报告 | 收敛进程、集成、发布 | 审计视图更新；干净 Git；无本轮后台任务；不 push |

R0 能力通过 W0 后可以早于 H9 单独冻结，先启动 W1 的基础轨迹处理；后续 R1-E/R1-H 以独立签名增量附加，不覆盖已冻结 R0。

各 lane 的任务预算是并行墙钟预算，不能相加后误以为所有工作都能在单线程 15 小时内完成。H9 后仅允许修复明确运行故障；修复涉及数值算法须新签名并重过最小回归，否则不进入本轮冻结批次。

## 7. G1：Human/Stereo 尺度原因审计

预算：累计 2.5 小时；不要求必须找出唯一原因。

### 7.1 核查顺序

1. **表达与单位：** MANO wrist-local/camera/world；mm/m；左右手轴约定；T_ab 的作用方向；SE(3) 与 Sim(3) 的具体变量。确认 0.783730 缩放的是骨长、相机空间点、平移还是 SLAM 世界尺度，不能混称“人手缩放”。
2. **像素域：** sourceIndex、SBS 裁切偏移、resize 尺度、padding/unpadding、pixel-center 约定、K/P 在每次变换前后是否一致。
3. **水平镜像：** 两眼输入与输出还原的像素映射，恢复后的 disparity 符号与 principal-point 差项，不能只拿正的数组当原域视差。
4. **投影几何：** encoded 相机内参/有效基线是否真的属于编码视点，不能只因为极线近水平就认定旧物理镜头 K/B 在编码域仍精确适用。
5. **深度类型：** optical-Z 与沿射线距离分开。P=(x_n Z,y_n Z,Z)；若输入是射线距离 r，则 Z=r/sqrt(1+x_n²+y_n²)。
6. **Stereo 三角化：** 可用时以编码域 P_L/P_R 三角化并重投影两眼，与当前深度转换交叉核对。一般不能忽略左右主点差；不在未知标定下随意套 fxB/d。
7. **表面对应：** 对 MANO 做可见性/z-buffer，仅比较有同像素支持的同侧可见表面，避开轮廓、皮套、任务物体、遮挡和深度跳边。不能用全 mesh 最近邻把背面配前面。
8. **可辨识性：** 分别比较固定尺度刚体、单尺度、深度偏移等少量预注册解释，检查参数耦合；同时拟合自由焦距、尺度、深度偏移、MANO beta 不允许作为默认解法。

Stereo 的尺度依赖合适的相机几何；多帧一致不自动证明外部毫米精度。HaWoR 的相机空间手与世界轨迹也应分别检查。[S1][S2][S3]

### 7.2 报告与分流

输出按左右手、深度、视角、时间块、skin/equipment、图像边界、mask/深度质量分层的残差与参数稳定性。统计以时间块/源组重采样，不把相邻 20 帧当 20 次独立测量。

- 明确实现错误：新 corrective successor，定位代码行和受影响变量；冻结输入重现故障，再到未使用源组验证。
- 对应关系污染：修可见表面采样；不缩放机器人，也不通过触碰牌面完成手的标定。
- 单目形状/尺度先验不匹配：允许新估计 sidecar，但只在独立证据支持下消费，原数据不改。
- encoded 几何/尺度欠确定：保留内部几何开发能力，metric authority 降级；不能用“估出来的牌尺寸”反向证明同一 Stereo 的尺度准确。
- 原因仍不清：标 `UNRESOLVED_SCALE_CAUSE`；保留严格 Contact 阻塞，转向直接 Stereo 局部证据和独立 hypothesis 实验，R0 继续。

改变尺度允许区间必须有与 Contact 成败无关的证据和新留出验证。不得通过同一旧 holdout 调到通过。

## 8. G2：局部深度可靠性与直接 Stereo 交互路线

预算：最多一个主修复候选及一个有明确故障原因的 successor；不增加新深度模型。

### 8.1 全图通过不替代接触区质量

分别统计：桌面/物体内部、手掌内部、手指表面、轮廓边界、遮挡区的 LR consistency、有效像素比例、极线残差、鲁棒局部表面残差和时序创新量。单位、分母、拒绝比例必须齐全。

全局 `LR median=0.94666` 不能独立保证指尖 5 mm 接触判别。若稳定背景高质量而手指边界无效，优先缩小可信区域，不做跨手/牌边界的均值滤波。

### 8.2 可实施修复

- 像素域/数值错误修正优先。
- 采用固定质量筛选、前后向一致性、局部鲁棒估计，分别保留 raw disparity、filtered observed depth 和填补 hypothesis。
- 必要时对少量困难帧用同一 FoundationStereo 权重提高分辨率/迭代进行新 canary；仅当留出接触区改进、覆盖不被偷换时采用。
- ROI 左右裁切必须保持合法 Stereo 坐标关系并变换 K/P；未经双眼几何单测不引入 crop 优化。
- 静态场景可做相机运动补偿的时序检查；动态手/物体不做全图盲目时序平均。

### 8.3 不把全局 MANO 对齐当所有局部证据的前置条件

当同一 Stereo 域能直接提供手/皮套可见表面点和物体表面点时，可以计算两者的局部关系。HaWoR 在这里提供手别、手指关联和方向 prior，而不是必须提供绝对深度真值。

字段命名为 `finger_associated_visible_surface_point`，保留采样像素、surface_role、来源、关联质量和不确定度。它不自动等于指尖关节或不可见的实际接触面。

严格证据只消费真实可见且关联可靠的表面。表面距离不能区分接触与分离时保持未决；禁止扩大不确定度让更多距离自动过 5 mm 门。

Robot wrist 与 object 的相对平移必须来自同一可用几何域，或有明确独立验证的对齐。不能在前面拒绝 HaWoR absolute-Z，后面又用它减 Stereo object 生成腕部目标。

## 9. M1：SAM 时序与遮挡身份，而非 Clean 扩张

预算：Lane B 前 3 小时主攻；与新数据模型队列共享 GPU，但优先任务物体/手身份。

### 9.1 先定位 raw 无输出与 gate 丢弃

逐 role/instance/frame 记录：raw candidate 是否存在、state/session 是否复用、prompt 是否生效、propagation 是否输出该 ID、方向、admission 拒绝原因、面积/边界/身份变化。不能把 tracker 没输出说成门太严格。

核查当前 pinned SAM3.1 的真实 API 和 state 生命周期。共享记忆是实现能力，不是身份稳定的保证。[S7]

### 9.2 固定模型下的有限修复

- 从清晰帧自动选 anchor；hand 利用 HaWoR 有效 box，物体使用已通过的独立实例 seed。
- 质量触发局部重播种，保留稳定全局 physical_instance_id 与模型临时 track_id 的映射。
- 前后向复核；物体重现时结合外观、局部几何、时间和支持区域，而不是只找最近质心。
- `tracking_state` 与 `visibility_state` 分开；occluded/offscreen/unknown 不当不存在。
- 三张相似牌不可辨别时保留多个候选或 `identity_unknown`，不能强行继承最近 ID。
- 前臂、皮套、线缆只在有真实候选支持时修复。不得画到画面边缘凭空制造前景，不恢复 V1/V2。

mask 质量至少同时报告“系统自己声明的有效覆盖”和“独立固定可见帧集合上的成功/失败”；不能把失败帧改成 unknown 后让成功率虚高。没有人工 mask GT 时只声明代理指标和开发视觉结果。

## 10. I1/H1：遮挡下的离线交互重建与 Contact 双轨

这是本轮新增核心算法，而不是给 UNKNOWN 换个名字。

### 10.1 永不改写 observed

保留三个独立文件域：

- `interaction_observed`：当前帧合法观测及派生量。
- `interaction_reconstructed`：利用前后帧和先验重建的离线状态。
- `contact_hypotheses`：no-contact/接近/接触/持物等解释及分数。

每个推断带支持帧、时间跨度、instance/finger 身份、surface patch、假设类型、反证、非因果标记、适用消费者。没有支持时留 unknown，不平铺 forward-fill。

### 10.2 有限窗口求解

默认先使用约 0.5–1 秒短窗，按真实 fps/时间戳换算，最大窗口与求解次数在首次拟合前冻结。长遮挡或缺少两端身份锚定时不追求连续完整轨迹。

窗口利用：可见轮廓/像素、合法深度、HaWoR 局部手形与方向、物体面/有限表面片、相机补偿后的运动关系、物体重现一致性、可选 tactile。目标是找与观测不矛盾的解释，不是强迫接触。

每窗最多 3 个竞争假设，必须含 no-contact。可包含“拇指/食指局部夹持”和“表面触碰/滑动”等与当前证据相关假设，不能枚举所有手指组合。

先固定可观测相机与物体表面锚点，再估计有限隐藏状态；不联合任意缩放手、物体、相机和机器人去制造零损失。

普通手指套可作为人体外表面的一部分推断，但未知皮套厚度不能拿来把 6.6 mm 自动扣成 0。没有器件几何证据则误差保持未决。

### 10.3 严格 Contact E

沿用历史固定 5 mm 及实际合同中的不确定度、有限表面、身份和连续性要求。只在本轮独立验证后的合法观测范围放行。支持不足维持原 R1 的 `BLOCKED_LOCAL_EVIDENCE`。

不要求接触后 5 帧内必须移动；静态按压、握住后停顿均可能发生。事件关系用作任务相关诊断，不做错误全局门。

### 10.4 Contact H

输出 `support_score`，不是概率。几何近邻、物体共动、纹理/轮廓、触觉等证据分别列明；共享同一模型/标定的信号不得假装独立证据相乘。

触觉使用真实时钟映射与延迟质量；缺 tactile 不阻塞视觉假设。有离线触觉只能支持解释，不能自动定位实际接触点。

只有身份、局部几何和时序反证检查通过的假设，才进入 R1-H。证据不够则输出若干未决假设或拒绝。完全隐藏、不受约束的动作通常存在多个解释；不能以单一平滑解声明恢复真值。

### 10.5 两种独立验证

1. **遮挡回放：** 从冻结的真实可见片段屏蔽连续若干帧的指定观测，重建后用被留出的像素/深度/姿态测恢复误差。缓存、拟合、重播种不得泄漏被屏蔽真值；合法离线窗口可以使用窗口外的前后证据。
2. **真实遮挡重现：** 比较重现后的实例、轮廓、局部表面与预测是否一致；只验证可见部分，不认定隐藏过程唯一正确。

遮挡回放通过不等于真实遮挡全面解决，必须分别报告两类结果。任务假设/机器人求解目标与独立评估数据分开；优化用过的接触距离变小只是训练内指标。

接触约束用于修正手姿态、以及保留手物交互进行 retarget，均有相关研究依据；本轮只借鉴问题分解，不引入整套新模型或承诺复现其结果。[S4][S5]

## 11. R0/R1/R2：真实 Robot 产物，而不是只留接口

### 11.1 R0 必须尽早跑新会话

复用当前 Kai22 真实 retarget 与 FK，不用模拟随机 q 或复制当前单样本到新 session。输出完整时间轴、左右手各自 mask、22 关节命名/顺序/限位、q22、wrist prior 与 frame 定义。

缺帧保持无效；若另有 offline motion completion，写独立层，不覆盖 observed 或原始 q。render 可以显示轨迹间断，不能靠冻结机器人掩盖失效。

正常 RGB 为默认原图输入。无需 inpaint 即可输出原视频旁路加独立机器人场景/透明叠加的审阅视频，明确不是替换完成人类外观的 clean robotized RGB。

### 11.2 R1-E 与 R1-H 共用有限机器人求解内核，输入授权不同

优化变量优先为 `q22(t)` 与小幅 `T_wrist(t)` 修正。human 提供姿态/方向先验，有限 object patch 与任务假设提供相对约束，机器人原生几何提供可行域。

可用目标：

```
E = E_prior + E_surface_relation + E_direction
  + E_self_collision + E_visible_patch_intersection
  + E_velocity + E_acceleration + E_transition
```

- 关节限位、冻结的修正幅度和明确碰撞约束不能仅作为小权重软项忽略。
- 第一版只允许已关联指部，不自动增加第三根手指来制造更低损失；增加接触指是另一个假设，需单独比较。
- 物体位置、隐藏尺寸、真实世界 base 不能随优化器一起自由移动。
- 接触窗两侧允许有限过渡区平滑衔接；过渡不产生 Contact 标签，不改变原观测。
- 时间导数使用真实 dt；不以变慢播放作弊满足速度。
- 机器人关节/mesh 尺寸固定。Human 运动不要求逐毫米复刻，但优化结果不能被称作全局最优或真实执行动作。

before/after 比较固定样本集，不用删难帧提升指标。报告 prior deviation、有限 patch 距离、切向逃逸、覆盖率、自碰撞、可见面相交、关节/速度/加速度、窗口边缘连续性，并列出未建模碰撞区域。

若起始解已达容差，保持不退化，不强迫 20% 改善。若没有候选胜过 R0，诚实保留 R0，记录优化已运行但未采用。

### 11.3 碰撞与薄牌边界

分别验收机器人自碰撞、手/臂与已观测有限表面的相交，以及未验证的隐藏体积/环境。开放平面片不能提供完整物体 inside/outside，不能把有符号平面距离当完整实体穿透量。[S6]

已有毫米级数字几何门只能用于其定义范围，不代表外部深度达到相同精度。未知卡厚度不补成实体；如果只用可见面交叉或距离检查，collision scope 必须写清。

### 11.4 R2 虚拟机械臂

从当前权威安装审计读取允许的开发 proxy，并核查 arm URDF、hand mount frame、joint ordering 和 mesh。优先使用项目已存在的实际求解/渲染引擎，不安装新仿真栈。

- 可从 R0 直接生成虚拟 arm trajectory，R1 未闭合不是其前置阻塞。
- 虚拟 base 是明确的设计变量，不声称 0915 实测外参。先确定一次会话级 placement，整个片段固定；hand/object 场景必须同时经过同一刚体变换。
- 不逐帧移动 base、不单独移动牌、不换算到未标定真实机器人。
- IK 失败、奇异/限位、手臂自碰撞保留局部 invalid，不截断关节或用 tool-origin 代替真实手 pad 验收。
- proxy 不通过时仍交付 wrist-local R0/R1 手，不等待用户补外参。
- R2 不等同通过力学抓取验证；本轮不新建物体质量/摩擦的物理 rollout 成功率。

## 12. B1：新数据批处理入口与每会话数据合同

不能只交付尚未被真实调用的函数或空 CLI。

### 12.1 调用形态

复用现有 `chaoyang.cli run <operation>` 注册机制。operation 名称由本地核对后确定，使用语义化 `<verb>_<subject>_vN`；本文件不冒充某个新命令已经存在。任务完成时必须给出一条真实运行并验证过的批处理命令和 resume 命令。

入口至少支持：冻结 manifest、输出根、stage selection、resume、strict signature、budget/deadline、per-session timeout、失败清单。默认每个 session 完整长度，不统一截断到 150 帧。

缓存键至少绑定输入、采集域/标定/坐标变换、代码、配置、权重、schema 和上游签名。某一会话失败不影响其他会话；上游变更仅失效对应后继，不删除无关有效缓存。

### 12.2 运行产物布局

在现有允许的 `_run/current/<task_id>/attempts/attempt_NNNN/` 内建立，每个 worker 独占子目录。以下是新运行内部建议结构，不改原始/processed 数据：

```text
batch_manifest.json
batch_results.jsonl
batch_summary.json
sessions/<session_id>/
  inputs.json
  timeline.jsonl
  human_prior/
  geometry_observed/
  geometry_hypothesis/
  interaction_observed/
  contact_hypotheses/
  robot/r0/
  robot/r1_evidence/
  robot/r1_hypothesis/
  robot/r2_virtual_arm/
  review/
  RESULT.json
```

未运行能力不用空动作文件占位。可缺目录，但 RESULT 必须有明确状态与原因。

### 12.3 Robot 最小导出字段

推荐复用已用 NPZ/HDF5 格式；metadata 用 JSON，避免新增存储依赖。

```
session_id, source_group, capture_domain
frame_id, timestamp_ns, timestamp_clock
hand_side, joint_names, joint_units, q22
T_frame_wrist, frame_definition, transform_provenance
observed_valid, trajectory_valid, evidence_type
robot_layer, applied_refinement_mask, transition_mask
object_instance_ids, contact_hypothesis_ids
q_arm (仅存在合法 R2 时)
noncausal, control_eligible=false, training_eligible=false
input/config/code/weight/URDF/proxy hashes
```

训练资格默认不授予；本轮交付是可检查的机器人动作候选数据，不训练新 policy、不声称 Robot GT。未来训练数据准入必须按来源、split 和 evidence type 单独审批。

### 12.4 两类任务路由

- Poker：薄刚体/有限平面表面，三张牌独立；未知尺寸与遮挡边界不伪造。尺寸可从清晰多帧估计，但不能与同源 Stereo 构成自证循环。
- Chips：R0 可用相同 retarget。袋体为潜在可变形对象，不强行整袋 6DoF 或全片刚性平面；Interaction 先消费局部表面/任务阶段，证据不足只保留 R0。

跨任务批处理能力与跨任务接触能力分开声明。Poker R1 通过不自动授权 Chips R1。

### 12.5 批量运行验收

- 至少实际 W0 全链调用；目标 W1 12 个完整新会话达到各自 R0 质量要求。
- 启动一次受控中断/恢复测试，已完成缓存 SHA 不变，未完成输出不当成功，恢复不重复计数。
- 注入一个无效输入 fixture，其他任务继续、错误原因可追溯。fixture 不计成功真实会话。
- 不把 150 帧单样本复制 12 次算批处理；不把不同视频帧数统一误标 150。
- 所有计划会话都有终态，但终态覆盖率与 Robot 成功率分别计算。

## 13. 闸门、失败与止损

各模块质量指标必须在本轮 validation 前冻结来源/单位/分母。既有严格阈值保留；新增工程预算和代理阈值必须标 `DESIGN_PARAMETER`，不宣传行业标准。

- 每个原因簇最多 1 个主修复 + 1 次明确原因 successor，累计不得超过所属 lane 预算。
- 同一个问题 30 分钟没有新的可验证证据：先记录未决，转去独立可执行任务，不整个窗口提前停止。
- 未实现求解器是本轮应完成的工程任务，不可反复标 `BLOCKED_EXTERNAL` 躲过实现。
- 只有真缺数据、不可辨识几何、缺真实标定等才标相应原因；预算耗尽单独标 `BUDGET_EXHAUSTED`。
- 运行终态沿用仓库现有 enum。局部状态、scope、first blocker、预算终止原因放 sidecar，不扩一套新顶层状态机。
- 尺度 cause 未闭合：严格 alignment/Contact 保持拒绝；R0/R2 与有合法输入的假设实验继续。
- SAM 弱角色失败：该角色 unknown；物体/手合法部分继续，不进入 Clean。
- R1-H 不胜过 R0：不采用修正；仍保留真实候选比较文件，不写 R1 PASS。
- 整个 capture domain 错：该域不扩批；禁止改门让全量跑。

## 14. 实时算法审计与最少治理

运行时使用 `02_ALGORITHM_AUDIT_LIVE_ZH.md` 的字段/模板。它是既有 ALGORITHM_CONTRACT、任务结果和 receipt 的派生视图，不成为第二套权威账本。

开始前登记本轮 read/write set、冻结数据清单和边界。中间只在算法采用/拒绝、接口变化、任务完成和每 10 分钟状态快照时更新事实视图。worker 只交结果，publisher 原子更新 JSON 和 Markdown。

审计必须回答：当前实际用哪个版本？哪个代码入口？哪些权重/参数？输入是什么坐标？哪些推断？哪次真实运行证明？哪些消费者允许？还有多少真实会话产物？

原总览中的 36.79 mm 文档债务应由 publisher 修正为指向最新审计结论，不新增第三份人工摘要。

## 15. 最終交付与成功定义

### 15.1 必交付文件

1. 冻结任务清单、真实批处理命令与 resume 命令。
2. 新会话逐帧 Kai22 R0 机器文件，及其质量报告；目标 12 个，真实数量实报。
3. 尺度原因矩阵、局部深度质量对比、是否采用修复的结论。
4. observed / reconstructed / contact hypothesis 隔离文件和遮挡验证报告。
5. R1-E、R1-H、R2 的真实候选/采用结果；不能用空 schema 冒充完成。
6. 批量计数、first-blocker 分布、失败样本、吞吐、未完成原因和剩余 ETA。
7. 至少 Poker/Chips 各一份真实新会话 R0 审阅；存在 R1/R2 结果时交相应对照。保留原 RGB，不假装 Clean 已完成。
8. 更新后的执行任务文档和实时算法审计文档，并绑定当前代码/配置/receipt。

视频每个完整解码，帧数与原会话一致，显示 session/frame/time、evidence type、Robot layer 和非部署水印。人工视觉验收仍待用户回来完成；模型自动视觉检查不能冒充用户验收。

### 15.2 最终报告必须分别给出

```
inventoried / eligible / scheduled / attempted / terminal
r0_exported / r0_quality_admitted
r1_evidence_adopted / r1_hypothesis_adopted
r2_virtual_ik_admitted
contact_observed_frames / inferred_frames / unknown_frames
physical_deployable = 0
```

按 session、source group、task 和有效帧数分别列明，不把 side-frame 当会话、不把文件数当动作数量。

### 15.3 封账

H13:30 停新会话，H14 前排空推理与 writer；H15 前完成回归、治理/结构/链接校验、产物哈希和本地提交。保留既有测试并运行新增测试，不以测试数量凑大于 989。

记录被阻塞/预算终止任务的恢复入口，所有已启动任务终态化；尚未执行的下一轮计划放非 active backlog，不伪装继续运行。只终止本轮所属任务树，保护别人的进程。

`git status --porcelain` 干净不能靠删除失败结果或 reset 改动实现；提交有效代码与失败 receipt，运行产物按既有规范保存。最终无本轮 GPU 租约、writer 或后台任务，不 push。

**成功不是“把所有门改成 PASS”。本轮真正的成功，是新数据批量 Robot 基础链已经被真实执行，增强算法有独立证据及明确适用范围，失败也能被限定在局部而不是再拖停全项目。**

## 16. 公开技术依据与使用边界

以下来源用于方法与几何事实核对，不替代本地 pinned 代码或资产审计，不要求引入新模型。

- [S1] FoundationStereo 官方实现：输入 Stereo、输出 disparity；公制转换依赖相机内参与 baseline。https://github.com/NVlabs/FoundationStereo
- [S2] OpenCV calib3d：相机模型、投影/三角化、Stereo 几何。https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- [S3] HaWoR 官方项目：相机空间手与世界相机轨迹分解。https://hawor-project.github.io/
- [S4] ContactOpt 原论文：利用接触约束优化手姿态；该方法的接触学习与软组织建模不能直接当本项目机器人刚体碰撞门。https://arxiv.org/abs/2104.07267
- [S5] TopoRetarget 原论文，2026-06 v2：交互关系、方向、运动学和穿透约束用于 retarget；本轮不是完整复现或新颖性认证。https://arxiv.org/abs/2606.16272v2
- [S6] Open3D Distance Queries：开放面距离与闭合几何有符号距离的区别。https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html
- [S7] SAM3 官方仓库：提示与视频 API，具体使用以本地固定 SAM3.1 版本为准。https://github.com/facebookresearch/sam3

读取日期：2026-09-18。本任务书中的时间、数量、预算和新能力划分是本轮工程设计，不是论文报告结果。
