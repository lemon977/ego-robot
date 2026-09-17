# RC1 解阻实验阶段结果与下一步（2026-09-16）

后续多线实跑结果已单独封存：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1/FINAL_RESULT.json`，视频入口为 `archive/baseline-20260917-0aa69e9/content/history/docs/stale-linked/docs/current/visuals/RC1/20260916_MULTILINE_VIDEO_INDEX_ZH.md`。本文之前的点时研究结论保留历史语义，不据此覆盖新轮结果或正式 current。

性质：研究证据交接，不是 current authority，不修改 RC1-FINAL 的正式质量门、T1/T2/T3 终态或训练资格。

本轮最新 A/B/C 有界交接的机器入口：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/round2_handoff/attempts/attempt_0003/RESULT.json`，人类简卡为同目录 `DECISION.md`。其中绑定最新 SAM 单因素/播种证据、原始 RGB 生产证据审计、111条可扩清单，以及**字节冻结的点时治理 receipt 与 RC1 小状态快照**。`attempt_0001` 曾把会继续变化的本研究文档纳入不可变证据闭包，`attempt_0002` 又绑定活的 current 路径；两者保留作闭包设计修正证据，不再用作最新交接入口。更早的 `handoff_summary/attempt_0002` 是 GPU 资源阻塞时的历史点时交接。研究交接通过不替代各任务 authority 成功。

## 1. 当前机器状态

- 本页最近一次点时核验为 governance revision `11161 / PASS/FRESH`；实时状态仍必须读取 current receipt，不能从本页续写。该 revision 通过 CAS 将 T5 收据中的 `RC1_RELEASE_STATUS=INCOMPLETE` 同步到实时页，并新增 T5/current 错配验证；没有晋升 authority 或更改任务终态。
- Robot 有界续跑已经终结；`play_cards_0902_053` 与 `play_cards_0902_042` 均形成硬几何通过的离线视觉终态。
- 所有现有 Robot 队列结果仍为 `OFFLINE_BIDIRECTIONAL_VISUALIZATION`、`training_eligible=false`、`control_ground_truth=false`。
- 正式 Chips/Poker Visual Aux pair 继续受独立 source group 容量门阻塞；本页不得作为放宽数据门的依据。

## 2. SAM3.1 状态路由诊断

结果：`PASSED_DIAGNOSTIC`，但诊断证明的是失败定位，不是 Mask 质量通过。

冻结 Poker015 提示和相同 SAM3.1 权重下测试了重复图、已知平移和真实短片。三个输入中，目标都只在播种帧存在：

| 输入 | 路由 | presence |
|---|---|---:|
| 重复同一张图 8 帧 | 官方低层直连、无 Priming | 1/8 |
| 重复同一张图 8 帧 | 项目适配器、无 Priming | 1/8 |
| 重复同一张图 8 帧 | 项目适配器、有 Priming | 1/8 |
| 已知平移 8 帧 | 项目适配器、有/无 Priming | 均为 1/8 |
| 真实短片 16 帧 | 项目适配器、有/无 Priming | 均为 1/16 |

已确认：

- 重复图 fixture 上，项目适配器与官方低层直连逐帧结果相同。
- 是否加入 Priming 对目标消失没有影响。
- 因而在**已测 fixture 和共享低层调用路径**上，当前证据不支持“项目适配器末端输出过滤导致消失”或“Priming 对象挤掉目标”这两个解释。
- 这不能排除官方低层直连和项目适配器共同使用了错误配置或错误调用逻辑；public predictor 尚未跑通，不能写成“已经排除适配器/调用问题”。
- 目标在完全重复的图像上仍于下一帧消失。进一步查看同一封账 `RESULT.json` 的 `seed_rows`：冻结检测对象 `FROZEN_DETECTOR_REPLAY` 面积为 `9,647 px`，另起 tracking session 后 `STABLE_POINT_BOX_TRACK_SEED` 仅 `1,036 px`（约为前者 `10.7%`）；这个缩小在 repeated、translation、real 三个输入与有/无 Priming 路径均相同。因此**首个可观测异常已提前到 tracking 播种阶段**，不是只有传播第1帧才出错。尚未确定造成缩小的具体调用或模型机制；不能仅凭面积差断言 point prompt 是唯一原因。
- 官方 public predictor 入口因 `offload_state_to_cpu` 参数与当前模型 `init_state` 签名不兼容而运行失败；这属于公开封装入口兼容问题，不改变低层直连的诊断结论。

不应声称：SAM3.1 模型整体不可用、项目适配器已经正确、或 Mask successor 已经通过。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/sam31_state_route/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/sam31_state_route/attempt_0001/SAM31_repeated_seed_8_路由与Priming对照.mp4`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/sam31_state_route/attempt_0001/SAM31_known_translation_8_路由与Priming对照.mp4`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/sam31_state_route/attempt_0001/SAM31_real_prefix_16_路由与Priming对照.mp4`

建议的下一诊断：先核对权重、代码和配置是否属于同一 SAM3.1 兼容组合；在重复图 fixture 上优先逐步审计**检测 9,647 px → tracking 播种 1,036 px**之间的独立 session、box 与 point 提示、对象注册及原始 mask/score，再检查 memory、detector 关联、confirmed/removed 状态和最终输出。只改变一个已证实因素后重跑重复图、平移和真实短片。不要直接换模型或扩大批量。

补充执行状态：原生 text+box semantic-full 路径的 `attempt_0001` 在推理前因实验脚本读取了不存在的 `selected_box_xyxy` 字段而运行失败。修正后，`attempt_0002` 已实际完成 repeated、translation 和 real 三组 native full propagation，但在结果落盘前错误调用了兼容适配器不存在的 `close()`，因此仍是运行失败收据，而不是 Mask 质量结论。该 cleanup-only wrapper 问题已最小修正；新代码签名的 `attempt_0003` 最终因 GPU 等待预算耗尽而封为资源阻塞，旧失败收据均保留。

资源终态更新：`attempt_0003` 的中央 GPU 等待达到冻结的 1800 秒上限，按 `GPU_WRAPPER_ATTEMPT_0003.json` 封为 `BLOCKED_RESOURCE/GPU_WAIT_BUDGET_EXHAUSTED`；没有执行新推理，不能把它计成 Mask 质量失败或修复成功。后来 GPU 资源恢复，在新的不可变研究任务中完成了下述单因素实验；旧 attempt 的资源终态保持不变。

`attempt_0002` 的三段诊断视频虽因上述 cleanup 错误没有对应完整 `RESULT.json`，仍可作为**非正式、未封账的视觉线索**：重复图与已知平移的后续帧继续显示冻结对象的洋红遮罩；真实短片第14帧的遮罩仍落在冻结提示框所指的右侧牌背上。这只能说明「该路径至少能持续输出一个对象区域」，不能证明整个短片身份无交换、边界质量合格或正式 Mask 修复。

代码对比已确认两个实验不只相差一个参数：旧路径在独立 tracking session 内下 box/point 提示、使用稳定 point ID，随后走 multiplex 的 `propagation_partial`；原生路径保留 text+box 对象并走 `propagation_full`，同时显式传入 `is_last_batch=True`。旧路由未显式传此标志，multiplex 外层默认是 `False`。因提示方式、tracking session 和 batch 标志同时变化，当前证据**不能**把成败归因于 `is_last_batch`、point refinement 或单一适配器 bug。GPU 资源恢复后才可启动已准备的同一语义路由单因素消融；不得在诊断前改正式 SAM3.1 authority。

同时间轴三栏对照已生成：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/sam31_route_comparison/attempt_0001/Poker015_SAM31_旧StableTracking_vs_原生SemanticFull_同帧对照.mp4`（16帧；左：Raw，中：旧 stable tracking，右：native semantic-full）。其 `RESULT.json` 明确标记 `PASSED_VISUAL_DIAGNOSTIC_ONLY`；右栏来源是 cleanup 失败 attempt 的可视化，不是已封账 Mask 结果。第14帧右栏仍覆盖冻结提示框所指的右侧牌背，中栏无 Mask；只能支持这段可视化差异，不能支持分割 accuracy 或唯一根因。

新的 `is_last_batch` 单因素消融已通过中央 GPU lease 完成，封账证据为 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/sam31_batchflag_ablation/attempts/attempt_0002/RESULT.json`，GPU wrapper 为同一研究目录的 `GPU_WRAPPER_ATTEMPT_0002.json`。两组仅改变 batch-final 标志；模型权重、text+box 提示、冻结物理对象 ID、图像和 semantic-full 路由相同。重复图的 `8/8` 帧与真实短片的 `16/16` 帧，两设置的 stream 数、目标存在率和**逐帧二值 Mask SHA 均相同**。因此在这两个 fixture 上，`is_last_batch` 既未补出被缓冲的帧，也未改变目标 Mask；不能把旧路由 `1/16` 与新路由 `16/16` 的差异归因于该标志。逐场景 `*_comparison.json` 已在每个场景完成时单独落盘，两个对照 MP4 均完整解码。下一步仍是逐阶段检查旧独立 tracking 播种（检测 `9,647 px` 到 tracking `1,036 px`）与 semantic-full 保留对象之间的差异；本次没有独立分割标注，不授予 Mask accuracy 或正式 successor。

原生 semantic-full 的正式开发诊断后来也在完整 session `play_cards_0901_015` 上封账：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/sam31_semantic_full_route/attempt_0004/RESULT.json` 绑定冻结 Raw 帧、代码、权重和三场景。重复图 `8/8`、已知平移 `8/8`、真实短片 `16/16` 的冻结 detector ID 都有输出；三段 MP4 完整解码。它证明所测 text+box semantic-full 路由并非播种后必然消失，**不证明**整个会话的物理牌实例身份或像素分割准确率。

播种阶段微探针 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/sam31_seed_stage_probe/attempts/attempt_0002/RESULT.json` 进一步把首个可观测分叉限定在旧路径的独立 tracking session：同一冻结帧的 semantic text+box 目标面积 `9,647 px`，路由 `propagation_full`；tracking box-only 阶段尚无对象 Mask 输出，随后 point+box 提示生成 ID 10001 的 `1,036 px` Mask，并切为 `propagation_partial`。这不是 `is_last_batch` 的作用。当前只能确认**不同提示/对象注册/路由合同导致不同播种状态**；哪一个具体子步骤或模型机制是质量根因仍未由独立单因素实验证实。微探针第一次因诊断脚本把空 action IDs 当可迭代对象而运行失败，失败 wrapper 收据保留；第二次修复后通过，没有覆盖第一次证据。

新的真实时序研究 canary 固定完整会话 `play_cards_0901_015` 的前224帧，覆盖冻结的第79/80帧局部遮挡及第213–223帧翻面/重入片段。原生 SAM3.1 text+box semantic-full 对选定 detector ID 在 `224/224` 帧输出非空 Mask；224帧 MP4 已通过 `ffmpeg -xerror` 全解码。关键帧拼图显示 Mask 视觉上跟随右侧紫背牌移动，翻面后仍覆盖牌面，但**没有独立逐像素标注或物理实例身份真值**，所以只授予开发诊断，不推翻 T1 正式质量 C，也不作为训练输入（全片传播可能使用未来帧）。证据：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_reentry_research_v1/attempts/attempt_0001/RESULT.json`；可视化：同目录 `POKER015_FROZEN_REENTRY_KEYFRAMES.png` 与 `SAM31_Poker015_native_semantic_full_224_路由与Priming对照.mp4`。下一门是两条冻结旧 B 会话回归，并将旧 B 仅作为内部参考、不能称作 Gold accuracy。

关键帧中第221→222帧 Mask 面积由 `1,194` 跳到 `7,584 px`（约6.35倍），同时画面显示牌从边缘翻出为较完整牌面；这既不能单凭面积判为实例跳变，也不能单凭视觉判为正确，需要在旧 B 回归和独立实例参考中检查。冻结旧 B 回归的第一条 `play_cards_0901_001` 已注册运行，但中央租约等待60秒后因外部 GPU 占用封为 `BLOCKED_RESOURCE/GPU_WAIT_BUDGET_EXHAUSTED`；没有启动模型推理，第二条 `play_cards_0901_005` 未启动。收据：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_b_regression_v1/play_cards_0901_001/GPU_WRAPPER_ATTEMPT_0001.json`。不能把资源阻塞写成回归通过或质量失败；下次 GPU 空闲时使用新的不可变 attempt 继续，且不改变正式 T1 状态。

回归输入预检发现旧冻结选择清单把 `play_cards_0901_001`、`play_cards_0901_005` 的 `frame_count` 写为 `0`；这不是 Raw 实际长度。两条旧 B 的 `RESULT.json` 与对象 Mask manifest 分别闭合为 `645`、`520` 帧。后续实验按完整 session ID、旧 B manifest 与 Raw 路径复核帧身份，不能直接使用选择清单里的零值作为视频长度。

两条回归已经改为**一次**带总等待预算的顺序队列，冻结输入与代码见 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_b_regression_queue_v1/QUEUE_START.json`；队列按 `play_cards_0901_001`→`play_cards_0901_005` 执行，中央租约要求至少61,440 MiB空闲、总等待最多1,800秒。当前/最终状态应读取同目录 `QUEUE_HEARTBEAT.json` 或 `QUEUE_RESULT.json`，不能从本静态文档猜测；资源耗尽时仅写 `BLOCKED_RESOURCE`，不重复创建短等待 attempt，也不改正式 T1。

在两条回归出结果之前，已冻结同目录 `REGRESSION_DECISION_CONTRACT.json`：每条固定11个均匀复核帧，再加全部漏选、内部 IoU 最差帧、面积突变与可见重入/翻面段；旧 B 只用于内部差异排序，不能作为像素 Gold。只有完整视频解码、SHA 闭合、冻结旧 B 可见帧无未解释漏选、复核未见明显错牌/混手，并把不确定正反面保留 UNKNOWN，才可进入**开发 successor 候选**；仍不能直接授予正式 Mask 或训练资格。

已从224帧原生输出构建独立 CPU 下游诊断包：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_224_cpu_diagnostic_v1/attempts/attempt_0001/RESULT.json`。它逐帧绑定 Raw 来源、可见 Mask 与边界/突变不确定区，输出224帧30 FPS三栏视频并通过 `ffmpeg -xerror`；第222帧的面积突变标为不确定。`sam_track_id=0` 只是模型 ID，**不等于**已验证的物理牌实例 ID；牌背/牌面的外观提示亦不等于正式 face ID。两者均保持 UNKNOWN，隐藏表面不作恢复，训练资格为 false。

## 3. Poker 因果平面 donor

结果：轻量平面 donor 对“历史中真实可见的同一牌面”有明确开发价值，但未解决真实隐藏表面。该 V1 实验的接纳/拒绝使用了伪遮挡区域真值的 MAE/PSNR，因此只能视为 oracle 评价，不能直接作为真实遮挡时的运行时准入门。

Poker245 使用过去帧到当前帧的单应性，在原本可见区域中程序化遮住一块区域，再与真实当前像素比较：

- 共 7 个因果帧对，6 个通过质量门，1 个注册成功但质量拒绝。
- 通过帧对的中位覆盖率为 `100%`。
- 通过帧对的中位 RGB MAE 为约 `7.37`。
- 通过帧对的中位 PSNR 为约 `27.90 dB`。
- 失败帧对 `28→33`：覆盖率 `40.625%`、MAE `42.63`、PSNR `13.88 dB`；它在 V1 中被 oracle 质量门拒绝，但该拒绝逻辑不能部署到真实隐藏区域。

判断：T2 之前的“无合法 donor”至少部分是 donor 搜索、注册和质量筛选链未实现，而不是历史纹理绝对不存在。下一 revision 必须把运行时门改成只依赖可见历史证据：匹配点覆盖与分布、可见点重投影误差、单应退化、可见区域光度一致性、实例/牌面身份和 donor 来源；伪遮挡真值只作事后评价。通过该验证后才可注册为 Clean challenger。不得用于薯片袋的非刚性表面或从未可见的牌面。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_plane_donor/attempt_0002/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_plane_donor/attempt_0002/Poker245_因果平面Donor_伪遮挡留出对照.mp4`

该实验是 `ORACLE_VISIBLE_MASK_PSEUDO_OCCLUSION_DIAGNOSTIC`，`training_eligible=false`，不能作为真实遮挡恢复 authority。

### 3.1 非 oracle 运行门 V2

V2 已把运行门和伪遮挡评价彻底拆开，并加入“只修改隐藏像素时运行决策和运行证据必须不变”的机器测试。结果是 `0/7` 帧对获运行门接纳：V1 的 6/7 不能解释为可部署准入能力。隐藏区域 MAE/PSNR 不再参与任何运行决策。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_plane_donor_runtime_v2/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_plane_donor_runtime_v2/attempt_0001/Poker245_因果平面Donor_V2运行门与Oracle分离.mp4`
- `tests/test_run_poker_causal_plane_donor_runtime_gate_v2.py`

### 3.2 Poker245 真实遮挡事件 V3/V4

真实事件冻结为：0–33 帧同一右侧牌背直接可见，34–89 帧正式 Object Mask 为 invalid，90–98 帧对象重新出现但已经可能翻到另一牌面。90–98 只用于事后审计，不允许反馈到早期门禁或 atlas。

- V3 只用过去牌背、当前非人体可见像素、单应性和光度证据，数值上错误地接纳了 56/56 帧。
- 全事件视频复核发现：目标牌被拿起/翻面后，V3 会切换到桌上另一张相同紫色牌背。说明“同外观、低重投影、低光度误差”不能证明物理实例身份。
- V4 增加从最后直接观测帧开始的逐帧双向 LK 光流身份连续性，候选单应性必须与该因果运动预测一致。最终只接纳 `2/56` 帧，其余 `54/56` 明确为 UNKNOWN；没有未来帧或隐藏真值进入运行门。这里的 `54/56` 是**保守弃权次数**，没有隐藏区域独立真值，不能称为 54 次“正确拒绝”；`2/56` 也不是恢复 accuracy。

这不是完整 Clean 改善：V4 只证明在这个已观察事件中，加入因果运动一致性后可以让大多数 V3 接纳转为保守弃权；由于隐藏区域没有独立真值，它**没有**证明这 54 次是“正确拒绝”，也没有证明余下 2 次恢复正确。当前纯视觉证据仍不足以恢复大部分真实手指压牌区域。下一步若要提升覆盖率，必须引入独立于 Attachment 的实例轨迹或对象 pose 真值；`HAND_OBJECT_ATTACHMENT` 只能延续已由独立证据建立的 pose，不能反向证明对象身份、Contact、Gold 或触觉事实，更不能放松成“匹配到任意同牌背即可”。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_real_occlusion_donor_v3/attempt_0001/Poker245_真实遮挡_因果Donor_V3_全事件.mp4`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_real_occlusion_donor_v4/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/poker_real_occlusion_donor_v4/attempt_0001/Poker245_真实遮挡_因果Donor_V4_身份连续全事件.mp4`

### 3.1 过去帧静态像素 donor 的有界 CPU challenger

为具体测试 Clean-20 的 `NO_LEGAL_CAUSAL_REAL_DONOR_PIXELS`，新开发 canary **不读取旧 Clean RGB 或双向 donor map**：每个待写像素仅在过去两次 Raw 可见、两次颜色相符、未进入截至当前帧的任务物体保守区域，且当前可见邻域通过静态一致性门时提出同坐标历史像素。未满足条件的像素保持 UNKNOWN。它不构成独立 support-surface 语义或隐藏物体纹理证明。

固定 `play_cards_0903_245` 与 `get_potato_chips_0902_039` 各前64帧：Poker 提出 `2,876,810 / 11,806,294` 待写像素（24.37%），Chips 提出 `1,744,650 / 9,079,941`（19.21%）。稀疏当前可见静态像素留出集的 RGB L∞ P95 分别为 8、10；此留出集**不是隐藏手部区域真值**，不能当作恢复准确率。Poker 同样输入运行32帧与64帧时，前32帧逐帧统计完全一致；这只验证本 challenger 的前缀重放，不证明 SAM3.1、ProPainter、Robot 或完整训练 input payload 因果。剩余大多数待写区域依旧 UNKNOWN，故 T2 正式 `BLOCKED_PREREQ` 与训练资格均不变。

证据与三栏 Raw/候选/UNKNOWN 拼图：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_causal_static_donor_research_v1/Poker245/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_causal_static_donor_research_v1/Chips039/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_causal_static_donor_research_v1/Poker245/PREFIX32_CAUSALITY_RECEIPT.json`

下一步仅在独立冻结语义与 Mask 回归足够时，将候选升级为逐像素 source-map 并在**手物交互局部**做伪遮挡留出；不能因全图稀疏静态 P95 较低就把这些候选发布为正式 Clean。

## 4. HaWoR 上游因果性

结果：已在两条真实轨迹上确认，当前全序列时间平滑会让前缀状态受未来输入影响。Robot 自身前缀测试通过并不能证明整条输入链因果。

| 会话 | 截断实验 | 终点 root translation 差 | pose rotation 差 |
|---|---|---:|---:|
| Chips103 | cutoff 21/48/100/200 | 约 0.0677–0.1189 mm | 最高约 0.0816° |
| Poker227 | cutoff 21/48/100/150 | 约 0.3945–0.9333 mm | 最高约 0.6488° |

全长重算与保存结果逐值一致，排除了读取或重算路径漂移。上述数值只是“真实输入轨迹上的参数级未来依赖”，不是外部误差，也不能直接等同于 Robot 末端误差。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/hawor_causality/chips103/attempt_0002/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/hawor_causality/poker227/attempt_0001/RESULT.json`

正式因果输入下一步必须从原始前缀重新生成 HaWoR 上游状态；不能直接采用全片平滑后的已有轨迹。离线 Robot 复核视频可以继续保留，但不得改写为 causal training input。

进一步代码核验确认：旧 HaWoR 原始推理本身也是 16 帧时序网络，历史实现按不重叠窗口一次发布整窗。因此只替换后处理平滑器仍不够。已经新增 `src/chaoyang/pipeline/hawor_causal_tail_inference.py`：目标帧 `t` 只读取 `max(segment_start,t-15)..t`，短前缀左侧复制首帧，只发布窗口最后一个预测；对应测试全部通过。

当前边界仍需保留：旧 `c2w` 来自离线 world-consistency 结果，尚未证明 suffix-invariant。新的 Chips103/Poker227 candidate 先只授予 `camera-space causal candidate`；在获得因果 camera pose 前，`robot_causal_eligible=false`，不得为了完成两条 Robot 视频而忽略这个依赖。

此前 `camera_pose_source_audit/*/attempt_0001` 对 `tracking_index` 的解释**有错误，已撤回其中“索引样本比帧早约若干毫秒”的结论**：原始 trackingData 文件首行是 merged-session header，不是姿态样本；`tracking_index=i` 指向其后的第 `i` 个姿态行，即物理文件行 `i+1`。旧收据仍保留为历史失败诊断，不覆盖。

修正索引后重新独立运行：Chips103 的 `284/284` 帧、Poker227 的 `196/196` 帧，索引姿态行的 `timeStampNs` 与 `training_data.metadata.ts` **逐帧完全相等**，且 metadata `c2w` 仍与 HaWoR NPZ 逐值一致。`tracking_sync_error_ms` 与「索引 tracker 时间减去以首帧为原点的名义 30 FPS 网格时间」相符：最大差分别约 `0.00000033 ms`、`0.000333 ms`。因此不能再把该字段当作独立的 RGB 曝光时间与 tracker 同步误差。独立 RGB 逐帧采集时间、旧编辑器中的时间选择/插值和旧 `c2w` 生产者的 suffix-invariance **仍未证明**。

质量状态不能只看数值矩阵：Chips103 映射的 `284/284` 帧 `TrackerState=accurate`、`Head.status=3`；Poker227 的 `196/196` 帧 `TrackerState=notAccurate`，虽然 `Head.status=3`。`Head.status=3` 表示对应 pose 字段存在或状态码通过，不能覆盖总体 `notAccurate` 警告；Poker227 保持负向诊断，不能凑成 Robot 正样本。当前 clip manifest 指向 `/home/pc/chaoyang/pico_dataset_editor-1.1.0` 旧生产环境，当前主机无该路径、仓库内亦未找到可重放的精确旧生产者。证明目前止于**元数据↔tracker 行闭合**；不能执行旧生产者级前缀变异，也不能生成有真实 RGB 时间依据的 previous-only 派生姿态。因而两条均仍为 `camera_pose_producer_causality=UNKNOWN_VERIFICATION_REQUIRED`、`robot_causal_eligible=false`，不生成虚假的 HaWoR/Robot 因果通过结果。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/camera_pose_source_audit/chips103/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_research_unblock_v1/camera_pose_source_audit/poker227/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/c2w_index_correction/chips103/attempt_0003/RESULT.json`（修正索引后的当前诊断）
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/c2w_index_correction/poker227/attempt_0003/RESULT.json`（修正索引后的当前诊断）

两条 21 帧并列诊断视频已生成，左栏是原始 RGB，右栏逐帧写出 tracker 状态、时间证据和哪一层被阻塞；**没有伪造 HaWoR/Robot 后继输出**：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/pico_c2w_blocked_visual/chips103/attempt_0001/get_potato_chips_0902_103_PICO_c2w_HaWoR_Robot_因果阻塞诊断.mp4`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/pico_c2w_blocked_visual/poker227/attempt_0001/play_cards_0903_227_PICO_c2w_HaWoR_Robot_因果阻塞诊断.mp4`

按完整 session ID 进一步查找原始 RGB/旧 exporter，而非重做索引：`get_potato_chips_0902_103`（284帧）与 `play_cards_0903_227`（196帧）的 clip RGB、stereo MP4、clip manifest、HumanEgo manifest、首帧 metadata 均已绑定路径/bytes/SHA；审计为 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/rgb_producer_evidence/attempts/attempt_0001/RGB_PRODUCER_EVIDENCE_AUDIT.json`。检查到的 MP4 只有本地 30 FPS 时间基及常规容器标签，没有原始逐帧曝光时间；clip manifest 只给 merged video 的片段时间范围，不给原始 RGB 帧身份。两条 manifest 指向的 `/home/pc/chaoyang/pico_dataset_editor-1.1.0` exporter 根和 merged source video 在当前主机均不存在；两日期的迁移 manifest 亦未找到逐帧 RGB timestamp/source-frame 字段。因此目前仍缺**原始 RGB 逐帧采集时间表、原始 RGB→merged→clip 映射、可重放的旧 exporter 源码**。这只说明已检查资产未提供证明，不断言原设备从未记录；不解除 producer causality 或 Robot training 阻塞。

## 4.1 原始来源容量重审（独立研究收据，不修改 T0）

新审计只消费 T0 固定 156 行的显式 manifest 路径，并对存在的 clip RGB、tracking、SLAM 文件计算 bytes/SHA；不扫描整个数据根。156 条中，154 条的 tracking 与 SLAM `sourceSession` 集合相同，2 条（Chips002/003）缺 T0 可绑定的 clip manifest；31 条 clip 跨多个 `sourceSession`。全部会话中 Chips 有 59、Poker 有 33 个**来源标识**，这仍不是已验证的独立采集数。

冻结 T0 候选集为 Chips 21 条、Poker 24 条；其中来源标识分别为 14、13 个。按共享来源和跨来源 clip 的传递闭包合并后，仅剩 Chips `12`、Poker `9` 个候选来源簇；当前 T0 split 中两任务的候选验证簇均为 `0`。即使暂把这些簇当作独立，也不满足每任务至少 `16 train + 3 validation` 的上限门。原始 RGB 的逐帧 sourceSession/采集 UUID 闭包仍未提供，故已证实独立 RGB 采集数为 `0`（表示**证据未闭合**，不是断言数据只有零组）；T0 质量尚未评价。正式两个 checkpoint pair 保持 `BLOCKED_DATA_VOLUME`，本审计不改写旧 T0 或 current authority。

证据：`archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/original_source_capacity/attempt_0002/RESULT.json`。若要修订分组，下一步必须找到 original RGB source-to-frame 映射与原始采集 manifest/UUID，并按完整来源集合重新冻结 split；不能仅凭不同 `sourceSession` 字符串或 clip SHA 放行训练。

为避免把“当前候选不足”误写成“整个 exact78 不足”，新增 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/expandable_candidates/attempts/attempt_0001/EXPANDABLE_CANDIDATES.json`，逐条列出冻结 shortlist 外的 111 条完整 session ID、首阻塞、对应旧 RESULT、clip RGB/tracking/SLAM 的路径与 SHA，以及研究级可扩路线。当前 shortlist 是 Chips 21、Poker 24 条；外部还有 Chips 57、Poker 54 条。后者按首阻塞分布：Chips 缺标定14、HaWoR C6、Role Mask C19、metric-ready但未入 shortlist18；Poker 缺标定16、HaWoR C6、Role Mask C1、Object Mask C23、metric-ready但未入 shortlist8。缺标定30条可审 Visual Tier，不能据此宣称 metric contact；metric-ready但未入 shortlist 的26条要先审开发/测试角色、split 和已有 Robot 结果，不能把“未选择”当质量 C。按 shared `sourceSession` 连通关系，全部 exact78 出现 Chips 39、Poker 22 个**候选来源簇**（当前 shortlist 仅12/9）。其中仅审“缺标定 Visual Tier 或已 metric-ready 但未入 shortlist”两条较轻路线，短名单外可能新增 Chips 17、Poker 7 个簇；Poker 即使条件性合并也只有 `9+7=16`，仍需 Object Mask 同实例重识别等 successor，后者最多再贡献5个未被轻路线覆盖的候选簇。以上都是基于 `sourceSession` 的条件性算术，不是已验证的独立 RGB 采集组、质量通过数或可用 validation split；原始 RGB 独立采集和配对窗口仍未证实。正式两组 checkpoint 继续 `BLOCKED_DATA_VOLUME`。

这26条现已有逐行处置收据 `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_expand26_disposition_v1/attempts/attempt_0001/RESULT.json` 和同目录 CSV/JSON：26 是会话数，**不是26个新增独立来源**。其中 Chips 两条（041、107）与现 shortlist 共享来源组件；余24条最多对应20个仍未证明独立的来源组件。逐项路由为：4条 Poker 有本次 Robot30 离线硬几何通过结果，优先补原始 RGB 来源及因果配对证据；16条 Chips 有历史 Robot 质量 C，须按当前硬/软门复核，不能直接沿用旧 C；6条未评估属选择或预算原因，不应计作算法失败。Poker189/202 的早期 hard/soft C 与随后 Robot30 离线硬几何通过属于不同 revision，均保留历史，不得混成训练通过。26条全部 `train_eligible=false`，正式 source-group 门不变。

## 5. Robot 队列与正式 RC1 收敛

Robot 队列已经完成固定 Robot30 的离线硬几何有限终态，不会自动获得 Visual Aux 训练资格。最终索引显示：

- Chips：`10/30` 有离线硬几何通过证据，`20/30` 因本 release 无合法因果消费者而封为 `NOT_EVALUATED_BUDGET`。
- Poker：`14/30` 有离线硬几何通过证据，`1/30` 为 `FAILED_RUNTIME_FINAL`，`15/30` 为 `NOT_EVALUATED_BUDGET`。
- `049` 的会话级终态为 `FAILED_RUNTIME_FINAL`，但其 batch task 已由完整收据正常封闭；不能把 task 级终态覆盖完成误写成该会话质量通过。
- Poker039 硬几何通过，但严格姿态相似度未通过；后者是软诊断，不否定硬几何终态。
- Poker039 单会话 wall time 约 `729.9 s`。

队列完成后的收敛步骤已经执行：

1. Robot30 terminal index 已重建，60/60 行均有唯一终态。
2. 未执行的 35 条 causal production 行已按预算封为 `NOT_EVALUATED_BUDGET`，没有伪装为算法 C 或 PASS。
3. T4 bundle/smoke preflight 因无 fresh causal Clean、独立 source group 不足而 `BLOCKED_PREREQ`，没有生成 bundle 或 checkpoint。
4. T5 conversion ledger 已发布，`RATE_FINALIZED=false`、RC1 release 为 `INCOMPLETE`。
5. 两个正式 checkpoint pair 继续为 `BLOCKED_DATA_VOLUME`。

证据：

- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/robot30_causal_budget_closure/attempts/attempt_0001/ROBOT30_FINAL_TERMINAL_INDEX.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/rc1_t4_bundle_smoke/attempts/attempt_0001/RESULT.json`
- `archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/rc1_t5_batch_conversion/attempts/attempt_0001/QUALITY_SUMMARY.json`

## 6. 可直接交给下一 AI 的判断

1. **Mask**：失败不是 `compile=True`；共享低层路径上没有证据支持末端 adapter 过滤或 Priming 冲突。完整 `play_cards_0901_015` 的原生 semantic-full 三场景开发诊断已封账；`is_last_batch` 单因素不改变逐帧 Mask。旧路径在另建 tracking session 后由 box-only 无对象输出、point+box 的 `1,036 px` Mask 和 `propagation_partial` 构成首个可观测分叉；若继续修复，应只改该已定位的提示/注册合同并回归完整同实例视频，不扩批、不晋升正式 Mask。
2. **Clean**：V2 已去掉 oracle；真实事件 V4 仅安全恢复 2/56，54/56 保持 UNKNOWN。纯单应性 donor 不能独立解决同外观实例切换，也不能推广到 Chips 非刚体。
3. **Robot 因果性**：现有离线 v77 结果不能直接进入训练；camera-space HaWoR 因果尾窗已实现，但完整 `get_potato_chips_0902_103`、`play_cards_0903_227` 的旧 RGB/c2w 生产者仍不可重放。`metadata.ts` 等于所索引 tracker 行，不能证明 RGB 曝光同步；后者的总体 tracker 状态全程 `notAccurate`。当前主机缺明确的原始 RGB 逐帧时间表、来源映射和旧 exporter，暂不接入正式 Robot input。
4. **训练容量**：当前 shortlist 仅 12/9 个可能来源簇且验证簇为 0；全 exact78 则有 39/22 个可能来源簇上界，不能据此判定整体永远不足。111 条外部会话已有逐条首阻塞与研究级可扩路线，但原始 RGB 独立采集证明、质量和配对窗口均未闭合；正式四 checkpoint 仍 `BLOCKED_DATA_VOLUME`。
5. **Authority**：本页所有结论均为 development evidence。SAM、Clean donor、HaWoR causality 和离线 Robot 均未晋升正式训练或物理 authority。
