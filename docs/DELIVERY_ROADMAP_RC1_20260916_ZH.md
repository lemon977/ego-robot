# Chaoyang 批量生产与四个 Visual Aux checkpoint 有限交付路线

状态：PROPOSED_NOT_EXECUTED。本文是基于 revision 10823 审阅包的新执行提案，不是远端当前合同，也不是完成收据。
日期：2026-09-16。
本轮目标：有限修复后冻结一套转换规则，运行完整批次，发布可复算的分域转化率，并训练 Chips Raw / Chips Robotized / Poker Raw / Poker Robotized 四个 H50 future-2D Visual Aux 模型。

## 0. 结论与完成定义

本轮不再以“所有模块获得最高 authority”“所有遮挡恢复物理真值”“每条视频全部通过”为目标。完成必须同时满足：

1. 冻结版本和输入清单，所有清单内会话获得唯一运行终态及独立质量等级。
2. 至少有真实会话走通 Raw → Mask/Clean、Robot → Compositor → 配对训练 bundle；不是仅 schema/preflight 通过。
3. 发布整段可用、部分窗口可用、拒绝三类数量，明确分母、采集域、任务、帧/窗口覆盖率和实际耗时。
4. 四个正式 Visual Aux 训练任务实际执行，best/last 可重新加载，有验证集、指标、loss 和配对推理结果。
5. 没有数据泄漏、伪造缺失、跨会话冒用标定、假 action 或将 UNKNOWN 冒充真实物体纹理。

算法修复最多两轮；达到最低交付条件立即停止加模型。数据不足时按预先冻结的扩量顺序补现有会话，不通过复制数据、改变验证集、放宽安全/身份门或改名 checkpoint 凑数。若固定预算用尽仍不足，产出明确的有限失败报告，不宣称四模型已完成。

## 1. 本次核验范围与事实起点

### 1.1 核验范围

审阅包时间为 2026-09-16 05:17:16+08:00，revision 10823。本地重算 MANIFEST(1).sha256 所列 12 个文件的 SHA256，12/12 一致。这只认证收到的包未改变，不等于重新连接远端执行 current receipt 校验。

两个 MP4 均完整解码 196 帧、30 fps、6.533333 秒。视觉检查为每段抽查 0/30/60/90/120/150/180/195 共 8 帧，不是全片逐帧语义验收。Poker227 的 Clean 栏抽查可见浅色腕部/设备样残留及局部补图纹理异常；该视频本身标注结构性 Clean 与基础几何遮挡，不应当作正式训练底图。

### 1.2 当前可确认的数量

| 项目 | 快照事实 | 可支持的结论 |
|---|---:|---|
| exact78 | 156 条，Chips/Poker 各 78 | 原始固定统计总体 |
| HaWoR 与两条 Mask 共同 A/B | 101 条 | 上游候选，不等于端到端可用 |
| metric-ready | 58 条，CSV 重算 Chips 39 / Poker 19 | 已有 Depth/Object6D 的优先处理池 |
| v77 已处理 | 25 条全部终态 | 运行终态覆盖 100% |
| v77 硬几何 | 13/25 = 52% | 只针对所选 25 条 |
| v77 Chips | 10/10 | 所选 Chips 硬几何通过率 100% |
| v77 Poker | 3/15 | 所选 Poker 硬几何通过率 20% |
| 四个 Visual Aux | 0/4 | 训练未完成 |

12 条 Robot C 中，摘要记录 11 条首要失败为 arm_temporal_and_unknown_contract，1 条为 hawor_temporal_successor；不能据此推断所有 C 的每个局部帧均可用，必须读取逐帧原因。

最新 CLEAN20 CPU RESULT 中，Poker245 和 Chips039 的 accepted_temporal_pixels 都为 0，且 unknown_write_pixels 等于各自 m_write_pixels，即该前置产物的写入域 100% 仍未知。摘要中的 82.36%/69.90% 是另一统计描述，不能套在这个 RESULT 的分母上。fresh_propainter_started=false。

SAM3.1 开发 canary 的三个对象均在有效 seed 后下一帧消失。定位报告排除了帧域、ID 重映射、后归一化过滤和下游 presence 门；具体 predictor 内部机制仍未知。这是该候选路径的事实，不否定旧基线所有 A/B 结果。

新传感器 play_cards_0910_001：sourceIndex0 是物理右眼、sourceIndex1 是物理左眼，selected MP4 使用右眼。rectification P90 为 47.98/35.09/14.31 px，未达现有 ≤5 px 门，原始 factory calibration payload 未物化，FoundationStereo 未执行。

## 2. 本轮范围与不可混淆的资格

### 2.1 主要交付域

生产验收总体保持 exact78 156 条。优先执行 58 条 metric-ready，再对其余上游候选按已冻结路由处理，所有 156 条最终都必须有结论；不具备输入的会话可以明确拒绝，但仍留在原始总体分母内。

0909/0910 新传感器线单独统计。附件 Mask 做一次有界修复/回归；Stereo 只做一次同会话资产与软件域修复。该线失败不得阻塞 exact78 的四模型，也不能被 exact78 的成功掩盖。

这里的“通用”指同一代码和预先声明的任务/采集配置自动处理多个会话，不允许每条会话肉眼调参后再算全自动成功。不能从 Chips/Poker 两类任务推断任意新任务具有同一个成功率。

### 2.2 本轮交付边界

| 资格 | 本轮处理 |
|---|---|
| 数字 Robot 轨迹 | 可生成、验证和消费，control_ground_truth=false |
| 视觉训练输入 | 按本轮视觉/因果/配对质量门授予 consumer eligibility |
| 像素/接触外部真值 | 不宣称，不作为视觉研究必须等待的前置 |
| 安全下发真机 | OUT_OF_SCOPE，外部安装/TCP/实机验证缺口保留 |
| Real-Robot-Supervised Policy | OUT_OF_SCOPE，真实同步动作缺口保留 |
| 原版 HumanEgo-style 6DoF Policy | 本轮不切换任务，另立后续课题 |

所有新批准只限视觉研究消费。不要用一个全局 authority=false 同时阻止所有 consumer，也不要因为允许视觉消费就把它改成物理部署资格。

## 3. 必须明确修改的执行合同

以下是建议修改，必须一次性通过现有文档/算法合同发布机制登记；不能静默改旧 RESULT。

1. 将 R2.2 的“四 checkpoint 只预检、不训练”替换为本轮的真实训练交付任务。
2. Contact-10 不作为基础可见表面 z-buffer 或 future-2D Aux 的全局前置；只有依赖接触假设的 refinement 使用它。
3. Poker 完整 3D Atlas 不作为所有背景 Clean 的前置。背景补全、对象外观恢复分别判定。
4. 不要求 real donor 非零才启动补图。缺少可信 donor 的普通背景残余孔洞，可由因果 ProPainter 生成并标 SYNTHETIC_CAUSAL_BACKGROUND。
5. 不将生成背景当物理真值；未知牌面/对象关键纹理仍 UNKNOWN，不由背景生成冒充。
6. 将“整段 Robot C”与“局部帧/窗口能否消费”分开：旧 C 不改写，新增窗口资格报告，逐帧通过后才允许使用局部结果。
7. 恢复旧输入和权重来源的工作要有界。未知来源不能伪造；需要重算时只用已固定来源的新结果。已有SHA冻结的标签可作为固定输入资产复现实验，并显式记录历史producer来源缺口；这不等于证明从Raw冷启动能逐字节再生这些标签。新生产推理必须固定真实代码和权重；不要因无法追回历史权重，就无限阻塞所有下游固定数据实验。
8. 暂停新论文接入、大规模清理、非必要治理重构。既有日志/收据由程序自动产生，不将再写一份分析文档当作算法交付。

## 4. 冻结后的最小生产 DAG

```text
Raw / 帧身份 / selected-eye / 已有标定与 tracking
      ├── 当前手状态 → Robot 几何 → RGBA / Z / part-id ──┐
      ├── Role + accessory masks → 受控 Clean ────────┤
      └── Object mask + 有效 Depth/registration ──────┤
                                                      ↓
                                      基础 ownership compositor
                                                      ↓
                                当前帧视觉资格与逐侧 future label valid
                                                      ↓
                                    Raw / Robotized paired bundle
                                                      ↓
                               Chips pair 与 Poker pair 独立启动训练
```

对象 Atlas 是可选外观增强；Contact 是可选几何 refinement。缺少它们时只把受影响像素/帧标为未知或拒绝，不使整个 DAG 全局等待。

同一帧的输入不能借未来，但标签可以由未来帧产生。数字机器人当前图像若使用了未来轨迹优化结果，同样需要从正式因果输入中剔除。

## 5. 有限执行任务

下列预算为工程建议，不是已实测时间。以 T0 执行开始计时；算法修改总计最多两轮、累计 24 个有效工程小时，建议在两个工作日内结束。GPU 无可用时段单独报告资源等待，不伪装为算法进展。

### T0：一次性冻结入口、总体、split 与验收合同

预算：2 小时。不启动全库扫描或治理大重构。

- 在真实工作环境验证最新 receipt；若已高于 10823，记录差异，不把本文快照当实时事实。
- 读取 ALGORITHM_CONTRACT、当前 loader/trainer 配置、Robot 逐帧门、现有 split。保留仍有效的隔离关系。
- 形成 156 行 MASTER_LEDGER；标记 task、capture domain、source recording group、input eligibility、metric eligibility 和已验证缓存。
- 优先锁定每任务验证集来源及预先排序的备用组，不能最后发现 validation=0 再重新随机拆帧。
- 新建本轮 visual training consumer 门，历史 authority 和历史 C 不覆盖。
- 修正本轮选择集的过期 null：10823 已有 Poker227/031/032 硬几何候选。只更新新 Task Packet 的冻结选择，不修改 G0 历史。
- 只在这里解决当前数据最小数量与本文建议值差异，记录一次性修订；批量评估后不再追着成功率改门。

交付：一个 RELEASE_SPEC，绑定代码/权重/输入/门/split；一个主清单；一个真实下一任务。NOT_IMPLEMENTED 属于应完成的开发工作，不能反复作为外部阻塞退出。

### T1：把 Mask 从 seed 成功推进到可生产

预算：主路径最多两次原因明确的修复，每次先 3 帧，再 24 帧；该故障累计不超过 8 小时，计入全轮 24 小时。

主路径：同一固定 SAM3.1，检查 state/session 生命周期、prompt 后正式传播入口、实例确认/输出路径。关闭编译仅作为诊断选项，是否保留由固定 canary 决定；不靠 hook 无日志推断模型已正确。

唯一备用路径：复用已经证明可生成 seed 的当前帧提示逻辑，采用当前帧重检测/重播种与身份关联，必要时有限短程 warp。不新增模型。不将原始预测实例 ID 当成永久物理 ID；显式维护 human-left/right、device、object-instance。

备用路径必须实测通过后才能冻结为生产策略。主路径和备用路径都失败则该会话/采集域拒绝，不自动启动第三个新模型。

附件：手套、前臂、Controller/Tracker、线缆/绑带分别保留身份。Controller 位姿仅提供点/框搜索提示；不能把框当轮廓。黄色细附件采用保持原分辨率细节的局部区域处理；检查小连通域过滤、ROI 截断。被握住的任务对象不是佩戴物，不能因运动相同被并入删除。

最小验收：明确可见的 24 帧功能窗中，被初始化目标不能在下一帧系统性消失；固定正/负支持和身份检查通过。先验证输出真实存在，再计算复杂质量分数。

回归覆盖：Poker245、Poker227、Chips039、Chips023；新传感器0910_053另列，不混入 exact78。

交付必须包含真实 mask 图与短片、输出 ID/面积/来源逐帧表，而不只是 adapter 单元测试通过。

### T2：产出新的 Clean 图像，不再只产前置审计

预算：8 小时，计入全轮 24 小时。固定 SAM3.1 + ProPainter。

- 保留 M_remove、M_flow、M_write 分层。检查 ProPainter 实际入口是否再膨胀，保存最终生效 mask。
- M_write 外从目标 Raw 数组逐像素保留；可见任务物体优先使用当前 Raw。
- 背景 donor 只有通过实际几何对应、源可见性/表面排除检查才接收。禁止 same-pixel 双颜色共识作为充分条件。
- 没有可信背景 donor 时，允许因果 ProPainter 补普通背景，不等待 Poker Atlas。
- 独立记录来源：RAW_VISIBLE / REAL_WARP / OBJECT_ATLAS / SYNTHETIC_CAUSAL_BACKGROUND / UNKNOWN。
- 来源真实与视觉可用性是两个字段；SYNTHETIC 不自动等于不可训练，也不自动等于合格。通过视觉门后方能消费。
- 被手遮挡的对象外观，仅从同一实例、同一面的已知像素/验证 Atlas 获取；无法确认则 UNKNOWN。

Poker Atlas 最多一个轻量单应/纹理对应 canary；不依赖错误的每帧 camera-facing 3D 轴。失败则本轮该增强禁用，不能拖住普通背景生成。现有投影 IoU 不佳足以支持不采纳旧模型；不能单凭整张倾斜牌的 optical-Z span 就断言其法向厚度错误。

因果实现：正式训练当前帧 t 只允许处理 [max(0,t-15),t] 的原始 RGB、mask 和状态，内部可以在这个历史窗口内双向处理，但只发布 t，不向过去回写。默认历史长度16是本次建议参数，冻结前用困难窗验证。最初不足两帧且模型不能处理的观测直接不取作训练起点，不复制当前帧伪装时间历史。不能用完整视频的双向结果再贴上 causal 标签。

成本控制：全片离线可视化与训练观测帧分开保存。建议正式观测起点步长5，未来标签仍为原始30 fps下的 t+1...t+50，绝非未来50个降采样帧。只对实际将进入训练的观测 t 生成严格因果补图，可显著减少无用重复工作；实际提速必须计时。显示视频可离线生成，但 loader 不得消费。

验收产物：四个回归会话至少有真实新 Clean 片段；每任务至少一条全片离线对照和一个严格因果训练样本。没有 fresh ProPainter/合法补图像素就不能报 Clean 改善。

### T3：Robot 有效窗口、输入对齐与吞吐修复

预算：4 小时，计入全轮 24 小时。先复用 v77，不立即换 solver。

- 读取 12 条 C 的逐帧失败原因。仅对可定位的局部问题计算新的 per-side / per-frame 资格。
- finite、坐标闭包、限位、数字碰撞等硬门不放宽。全局手性/安装链错误使整个会话无效。
- 单个 IK 失败/出画/短未知不必自动否决整段；可保留其他通过门的连续片段。旧 C 结果不改写。
- 不插值补动作、不 forward-fill、不删掉中间帧后把两段拼成连续 H50。
- 当前画面中一只可见手无法生成合法 Robot 时，不能仅将该侧未来 label 置零，就把缺手/残留图当合格输入。当前帧图像资格与未来逐侧标签资格分开。
- 保持 wrist 位移默认尺度1.0；目标为同一端点语义。可行域裁剪后明显不再匹配共同标签的帧剔除，不偷偷压缩运动。
- 因果 Robot 使用截至 t 的参考、合法的前一时刻初始化、前缀决定的固定 placement；整段未来优化的 q 只作离线候选。

性能改动仅三项：缓存 q/FK，缓存 RGBA/Z/part-id，少候选先试失败再扩展。hand 第二轮仅按冻结的失败触发条件运行。修改 Clean/Atlas/合成只重跑合成；修改求解依赖时才失效 q 缓存。

报告加载、候选搜索、arm IK、hand、碰撞、渲染、合成、编码、等待的实际耗时。候选数下降不是墙钟加速。

### T4：基础 Occlusion 与训练 bundle 小闭环

预算：2 小时集成与 smoke，不继续追加新的研究路线。

- 基础合成独立于 Contact。Robot Z、对象可见 Z 必须在同一帧/相机网格/optical-Z 定义下。
- 基础可见表面排序直接消费当前对象mask和有效、已注册的像素Depth；不以整帧完整Object6D pose可用为必要条件。只有隐藏几何/Atlas重投影才需要对应pose。不得把“整体pose invalid”扩大成“所有可见像素Depth invalid”。
- 没有 Robot 重叠的可见物体，不因缺 Depth 退化成背景。
- 已知前后关系按 z-buffer 决定；接近/证据冲突为 UNKNOWN。先后关系不能从同一个要验证的合成结果反证。
- OBJECT_FRONT 但无合法外观时仍 UNKNOWN，不能取 Clean 桌面冒充牌面。
- 清除域里的真人/Controller 表面不能继续遮挡 Robot；非任务桌沿等可信表面可参与场景排序。
- 区分 Clean 的无条件可见物体保护与最终 Robotized 的条件保护。Robot 真正在前时覆盖物体像素是允许的。
- 对关键未知区按第6节严格决定遮罩或拒绝；两个 RGB 分支处理一致。

先生成 Poker227 与 Chips023 的真正配对 bundle，再用独立 smoke run 训练约200步并重载。这个 smoke checkpoint 不是四个正式 checkpoint。它的意义是验证磁盘文件确实穿过 loader、loss、梯度和保存链。

若角色/图片/来源合同仍有系统性错误，不能进入全量。若只有某个难片段失败，则局部拒绝并在分母中计入。

### T5：冻结生产版本，批量处理与验收

- 先用初始开发会话修复，冻结后不再逐会话改参数。
- 批量路由先处理已有合格上游的会话；不能为了掩饰成功率删除困难会话。
- 优先使已冻结 validation 来源与训练来源同时得到合格产物，不再先攒大量 train、最后发现 val=0。
- 58 条 metric-ready 为主池。其余43条共同A/B可走明确标记的 visual-only 路由：不输出虚假 metric depth，只保留不需要未知几何排序、且满足图像资格的当前帧/窗口。不能因无深度就粗暴把所有 Robot 画最前。
- 剩余上游不合格行按确定原因拒绝。若一次通用修复已修好该簇，可按冻结算法重新处理；不得在批次中另行调参。
- 两种路由在同一配对原则下可以用于本轮 Visual Aux，但必须分路由报告质量/覆盖，不能宣称一样的几何证据。
- 每会话最多首次执行+一次仅运行时恢复。质量 C 不自动重跑；新增任务名不能清零同一故障簇预算。
- Worker 单会话隔离；每3条形成一个调度小批，但其中1条失败不丢失另外2条已完成产物。
- 每阶段输出由 manifest/哈希决定可复用性，禁止依赖 latest 目录名。
- 中断恢复与重复运行需实际测试：不重复计算已完成会话，不改变冻结质量判定，不消费 partial。

冻结后另选未参与算法调参的 holdout 会话执行，报告与开发会话分开的转化率。它们可属于已有训练/验证分组，但转换算法在看到其失败前已冻结。若确实已看过所有候选，就不称全新泛化测试，仅报告固定总体复算。

### T6：四个训练任务必须存在并实际结束

任务名固定：TRAIN_CHIPS_RAW、TRAIN_CHIPS_ROBOTIZED、TRAIN_POKER_RAW、TRAIN_POKER_ROBOTIZED。

每个任务对准备好即可启动，不等另一任务对，也不等全部156条完成。四模型最终 bundle 清单在每个 pair 启动前冻结，后续新数据不在训练中途混入。

两分支必须共享原始会话组、观测t、未来label、label-valid、输入未知区处理、split、seed、初始化、优化器、更新预算与所有非RGB输入。唯一核心对照为RGB域。

默认最低建议：每任务16个train会话组、3个validation会话组，至少256/48个唯一有效H50起点。精简包仅列出了 minimum_sessions / minimum_H50_windows 门名，本次未核实远端具体数值；这些数字作为新交付合同建议，不伪称当前已经冻结的值。T0只允许一次正式确认并登记，禁止结果不好后降低门。

同源长录制的多个裁剪必须同组。不同窗口不等于独立会话。重复抽样、数据增强和重叠窗口不能增加独立会话数。

沿用现有 VisualAuxFuture2DModel 和 trainer，不切换到原版HumanEgo flow policy。优化器/LR优先保留现有可执行配置；建议正式训练上限为100 epochs或10000次更新，先到为止，两个分支共享同一预算。每epoch验证，按冻结的主验证指标保存best，结束时保存last；不要为了让Robotized胜出无限加训练。

每支输出：best、last、可重载验证结果、loss CSV/PNG、ADE/FDE/PCK、有效标签数与覆盖、模型/配置/数据SHA、推理可视化。checkpoint必须有实际权重更新，不能用初始化权重或smoke文件冒充。

将“完成训练”和“有应用收益”分开。保存四模型并不自动意味着Robotized优于Raw；结果持平/更差也完成这次对照，但价值结论必须如实报告。单seed结果不宣称普遍显著优势。

## 6. 建议的质量门：少量、可计算、允许局部利用

### 6.1 门的来源

当前合同已有四类：HARD_STRUCTURAL、DOWNSTREAM_ELIGIBILITY、SOFT_DIAGNOSTIC、EXTERNAL_AUTHORITY，继续沿用。下面新增数值均为建议初值，不是论文标准或实测验证结果。冻结前只在开发集做一次可行性确认，正式批次不改门。

### 6.2 结构硬门

所有被消费数据100%满足：源帧和时间网格真实；采集眼别/坐标链正确；输入/权重/schema版本绑定；没有NaN/非法旋转/越限的有效状态；Raw/Robotized严格配对；source-map完全覆盖；loader不读取未来；训练/验证原始会话组无交集。

已知身份错误或跨域错标不是软门。新传感器rectification在原合同的测量分辨率/像素域下保留现有median≤2px、P90≤5px和其覆盖门，不为通过率改成50px。诊断门通过也不自动建立公制标定资格。

### 6.3 视觉输入消费门

| 门 | 建议判据 | 解释/失败处理 |
|---|---|---|
| 当前删除角色支持 | 对冻结的当前帧高置信支持，mask覆盖≥98%；当前检测到的身份串换0 | 这是支持一致性，不是独立像素真值recall；支持不可得则该项NOT_MEASURABLE，不许伪报100% |
| seed功能回归 | 冻结可见24帧窗目标输出存在≥95%，不存在seed后全灭 | 仅验证功能路径；真实出画/遮挡不作为必须存在 |
| Clean写域 | 编码前M_write外改变像素0 | 外域变化为实现错误；不以有损编码重压后的像素比较 |
| 可见对象保留 | Clean阶段当前可见对象像素保留≥99.9%，关键身份纹理不得发生已检测替换 | 单一预测mask内保留不足以证明边界正确，须同步看边界诊断 |
| 人体/附件残留 | 在已冻结可识别残留证据上的比例≤2%，并且相对原接受基线不退化 | 不是使用自己的删除mask当ground truth；不确定残留进入风险区或拒绝当前帧 |
| Robot | 被消费帧通过现有结构/限位/数字碰撞门；预测端点语义一致，投影与共同参考端点差≤图像对角线1% | 1280×960时为16px；这是视觉对齐门，不是外部公制精度。原更严格结构门保留 |
| 前后排序 | 所有标known的重叠像素符合有效深度排序与来源规则；不存在人为“对象永远在前” | 仅基础内部一致性，不声明Gold accuracy |
| 未知像素预算 | 输入中需屏蔽的UNKNOWN占全图≤1%，且占冻结关键交互ROI≤5% | 超限拒绝当前观测，不丢弃整段其他可用观测；已合理生成的普通背景不算UNKNOWN |
| 因果性 | source_max_frame≤t；固定种子前缀/未来扰动测试通过 | 不仅查donor，还查mask、atlas、Robot当前q、placement及所有current-state |

上述残留/支持指标需要写清证据生产者，不能由“同一个mask里自己已经删光”自证。没有人工Gold时报告工程QA通过率、NOT_MEASURABLE数量和错误例，不宣称人类级像素准确率。可通过原始可见区域构造已知答案遮挡和合成前后场景做自动测试，不需要用户新增人工标注；它们仍不能代表真实遮挡的全部准确率。

UNKNOWN的输入处理：只对预算内的小面积未知区采用固定无语义占位值，例如归一化后的零值，并在Raw和Robotized两分支使用同一张因果mask。不能把未知纹理原样喂模型、只在loss端屏蔽标签。该占位是输入处理，不是“真实外观恢复”；其面积与位置必须报告。大面积缺失或已知假物体/身份错序不能用占位掩盖为整段通过。

关键ROI从当前Raw的任务对象支持、手部/接触邻域和确定性安全边界形成，冻结定义；不能通过扩大ROI稀释未知比例，也不能只用一个漏掉牌角的对象mask当全部任务区域。

### 6.4 帧、窗口、会话三级判定

**当前帧资格**：只管模型实际看到的观测，失败则不作为任何分支的观测起点。

**未来标签资格**：逐侧保留原有valid。对H50中至少40个时间步存在至少一侧有效端点的窗口可保留；loss仅计算实际有效侧/步。未来全不可用的侧不伪造；FDE只计算第50步真正有效的项，不能把最后一个有效点叫成第50步FDE。

**建议新会话分级**：

- HIGH_COVERAGE_VISUAL：全片视觉资格覆盖≥90%，不存在全局身份/坐标错误；同时报告所有失败帧，不把它称100%无缺陷视频。
- PARTIAL_TRAINABLE：没有全局错误，至少10个唯一H50有效起点，且有效起点占计划起点≥20%；允许局部利用。
- REJECT：其余，记录首要原因和全部相关原因。

有HIGH_COVERAGE_VISUAL标记但不足H50窗口数的短会话，只能计视频产出，不能计训练会话数。训练使用的每一个起点均须通过帧/标签门；不能把一个PARTIAL会话的所有帧整体授权。

## 7. 转化率：四种数字必须分别公布

设原始总体N_raw=156；N_upstream=101；N_metric=58。生产开始时保存实际清单，后续不删行。

1. 终态覆盖率 = 有唯一诚实终态的会话数 / 156。目标100%，但单独不能证明算法可用。
2. 全片高覆盖可用率 = HIGH_COVERAGE_VISUAL会话数 / 156。必须同时给Chips、Poker各自/78。
另行报告严格全帧通过率，即全部帧都通过视觉门的会话数/156，不能把90%覆盖叫作全帧通过。
3. 训练会话转化率 = 满足训练会话门的会话数 / 156。另报/101和/58的条件通过率，不混称总体成功率。
4. 窗口保留率 = 合格配对H50起点数 / 冻结采样规则下计划起点数。另报当前帧可用率、每侧有效标签量、起/中/末阶段覆盖，防止只留下容易的静止片段。

所有指标再按采集域和metric-backed / visual-only路由分层。相同处理版本下明确记录全自动、使用固定task profile、任何人工干预；人工单条调参的结果不混入纯自动转化率。

v77的13/25、Chips10/10、Poker3/15作为历史开发基准保存。不得拿13/101当已经完成101条的通过率，也不能把三种过滤百分比直接相乘当最终率，必须按同一session的实际交集计算。

“确定”是冻结版本、总体和判据后可重算；对于未来未见任务只能给有范围的经验估计，不保证固定百分比。小样本/筛选样本的成功率不外推所有任务。

只有全部输入资格明确、所有已准入会话的必要算法步骤均真正执行并有结果，rate_finalized才能为true。预算不足而未执行的准入会话记NOT_EVALUATED_BUDGET，不能伪称质量失败。若有u条未评估，只能报告已确认成功数s及总体率边界[s/N,(s+u)/N]，不能宣称最终完整转化率。输入实质缺失导致的明确拒绝与单纯“没时间跑”必须分开。

### 吞吐报告

同时给出总墙钟时间、纯计算、资源等待、首次通过率、恢复后通过率、缓存命中、合格帧/小时、可训练起点/小时和每个pair的数据缺口。至少各任务一条会话验证从实际Raw输入到bundle的冷启动路径；其余使用合法缓存时标记cache-assisted，不能把缓存读取耗时当作完整冷启动吞吐。

T0后用真实短会话计时，对每个阶段采用同类会话P95估计批次上界并登记。超过冻结上界只允许运行时定位/降并发恢复，不自动开启模型研究。不能把未运行的会话从分母删除。

## 8. Split 与训练资格如何防止再次卡死

- 首先处理已冻结validation来源，而不是全部算完train才安排validation。
- 每任务建议目标20个train/4个validation，最低16/3；不因某个固定难会话失败就禁止已有合格会话进入训练。
- 对现有58池，Poker总计19条意味着仅依靠整段全部通过几乎没有余量；必须验证局部窗口可用性，或按预先规则扩到其他可用输入，不能假定当前3个整段硬通过已够训练。
- 备用validation来源在看训练效果之前排序固定；不能因模型在某段表现不好就换验证集。
- 若原始capture group合并后独立组不足，诚实计数，不把文件夹数当独立会话。
- 一个任务pair先达门就训练并冻结。另一任务pair继续生产，模型训练不等待全部stage authority。
- 首次正式训练结束不启动“为了改善A/B结果再做第N轮Clean”。后续优化是新的版本和新实验。

## 9. 收敛与停止规则

1. 算法修复两轮上限按根因簇计数，不按目录名、attempt名或版本号重置。
2. 每轮必须带来真实像素、合法轨迹、可用窗口或实际速度改进。只有新的报告/探针/schema不算算法完成。
3. 研究候选失败可回到明确验证过的旧基线/唯一备用路径；不允许无验证的fallback伪装成功。
4. 不将“正常出画”“某几帧未知”“与人手不够像”“缺外部contact真值”升格为不必要的全局硬失败。
5. 不为交付下调身份、坐标、时间、数据泄漏和数字硬约束。
6. GPU资源共享不可抢占他人任务；准备好数据后四训练任务优先于新增非阻塞canary，至少预留本轮约30%的可用GPU时段给训练。
7. 算法冻结后，批量和训练预算按实测profile登记；达到预算发布终态，不留下无限RUNNING。
8. 存在训练数据缺口时输出数量与首要原因；不以降低独立验证组、复制窗口或把smoke改名来完成。

## 10. 最终交付目录建议

复用现有run/receipt框架，不要求新建另一个治理体系。建议新增一次性release根：

```text
releases/visual_aux_delivery_rc1/
  RELEASE_SPEC.json
  MASTER_LEDGER.csv
  WINDOW_LEDGER.parquet
  QUALITY_SUMMARY.json
  CONVERSION_REPORT.md
  RUNTIME_PROFILE.csv
  DATASET_PAIR_INDEX.json
  checkpoints/
    chips_raw/{best,last,...}
    chips_robotized/{best,last,...}
    poker_raw/{best,last,...}
    poker_robotized/{best,last,...}
  CHECKPOINT_INDEX.json
  PAIRED_EVALUATION_REPORT.md
  FINAL_ACCEPTANCE.json
```

这些是待实现/映射到现有路径的产物接口，不表示路径和命令现在已经存在。

FINAL_ACCEPTANCE 至少区分：PIPELINE_OPERATIONALLY_CLOSED、DATA_MINIMUM_MET、FOUR_CHECKPOINTS_TRAINED、VISUAL_AB_VALUE、PHYSICAL_DEPLOYMENT_AUTHORIZED=false。A/B效果不佳不能改写成未训练；四模型不足也不能靠流水线封账标记DONE。

## 11. 可直接交给执行AI的任务指令

> 基于最新receipt和本提案建立一次有限交付revision。目标是冻结通用处理规则、完成156条总体的诚实分类/可用子集生产、报告分任务转化率，并实际训练四个H50 future-2D Visual Aux checkpoint。保留现有模型、数字硬约束、因果和配对要求；不要求Physical Robot或Real Policy外部证据。最多两轮原因明确的算法修复，总计24个有效工程小时；随后冻结。优先修复SAM3.1 seed后实例消失，提供同模型当前帧重播种的唯一已验证备用路径；附件独立身份。将背景因果补图与对象Atlas解耦，不以真实donor为零阻止所有ProPainter；未知任务对象纹理不能生成后当真。基础Occlusion不等待Contact10。逐帧/窗口验证Robot局部可用性，不改写历史C、不伪造缺失。先处理验证集来源，任务pair达到冻结门即可训练，不等另一任务或全部156条通过。每个阶段必须有真实产物；纯审计/接口通过不等于阶段交付。通过实际中断恢复、配对loader、训练保存重载验证后发布最终结果，不增加新论文模型、不做大清理、不无限新建诊断任务。

## 12. 主要依据与引用定位

本文事实来自本次包；新增策略、预算与阈值均为提案。

- 01_PROJECT_SUMMARY_ZH.md：第38–70行Robot失败/诊断；79–117行Mask；125–165行Clean；172–199行眼别；267–290行训练缺口。
- 02_CURRENT_AUTHORITY_AND_STATUS.md：CURRENT_STATUS_RECEIPT、ALGORITHM_CONTRACT；第1381–1415行Aux输入与门；第289–293行门分类。
- 03_R22_PLAN_AND_EXECUTION.md：第17行R2.2不训练；第120–154行Clean/Robot/Occlusion/Aux；第224–227行当前边界。
- 04_ROBOT_EVIDENCE.md：终态JSON实际25行；v78只是离线提取，solver_run=false。
- 05_MASK_EVIDENCE.md：下一帧ID消失的定位及instrumentation coverage mismatch。
- 06_CLEAN_EVIDENCE.md：最新CPU RESULT，两个会话accepted_temporal_pixels=0，fresh_propainter_started=false。
- 07_DEPTH_CONTACT_OCCLUSION_EVIDENCE.md：selected-eye失败、58行readiness CSV、Poker227196帧基础遮挡且training_eligible=false。
- 08_EXECUTION_SELECTION_AND_TASK_PACKETS.md：G0的旧Poker选择为null；基础Occlusion objective独立于Contact。
- 官方补充核对（不是本地实测）：SAM仓库README说明视觉提示和视频入口；ProPainter inference_propainter.py含双向处理、参考帧和mask dilation。具体调用必须匹配项目固定SHA，不执行git pull覆盖已冻结代码。
