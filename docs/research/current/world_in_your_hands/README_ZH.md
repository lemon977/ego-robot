# World In Your Hands 支线：MANUS＋PICO＋TAC 可视化

> **当前最新结论（V53–V55，2026-09-20）：** 新鲜跨会话 `play_cards_0916_101`
> 已完成 122 帧真盲测。V53 mask、V54 可见性门和 V55 HaWoR 均正常完成，但冻结 V45
> MANUS origin/basis 双侧质量拒绝；当前方法不能作为 ML 监督标签，也未解锁 Depth。完整
> 数值和视频见 [Session 101 真盲测](SESSION101_BLIND_VALIDATION_V53_V55_ZH.md)。下方
> V1–V34 内容保留为算法演进记录，其中把 094/V33/V34 称为“当前”的文字属于历史阶段，
> 不再覆盖本段 V55 终态。

> 当前终态（2026-09-20）：V1–V4 均被人工否决；V6–V9 将问题收敛到相机/图像域与右侧时段性残差。
> V10/V11 仍未通过独立 HOLDOUT；V14 用 RGB-only mask 框恢复了 18/18 HaWoR 观测，但左腕根因画面下缘截断不可验证。
> V15 已质量拒绝；V16 确认原始 MANUS node0 全程恒等，但右侧仅 1 帧满足严格可见掌面门，已按合同失败终止。
> V17 GPU 运行成功但因右手 FIT 仅 12/29 帧可观测而质量拒绝；有效 receipt 保留。
> V18 已在模型推理前因相对 manifest 路径路由失败并终态化。
> V19 路径修复后完成推理，但左手 joint-in-frame P50=0.7619 未过冻结门。
> V20 CPU-only 非腕指骨段审计双侧通过，允许构建独立 rotation-only 后继；V17 HOLDOUT 继续冻结，V5 depth 继续锁定。
> V21/V22 的代码、输入分区和验收阈值已在读取新 HOLDOUT 前一起冻结：V21 仅生产一次性盲测 HaWoR 观测，V22 只用 V19 FIT 拟合每侧一个常量 rotation，再一次性验收 V21/V17 HOLDOUT。
> V21 已成功生成并冻结 19/10 个盲测 HOLDOUT 观测；V22 已按冻结合同运行并质量拒绝。
> V22 证明 PICO 动态腕根已经贴近独立 hand mask，但 MANUS 指骨方向和区域重合仍不合格；不提升基线、不解锁 depth。
> V23 已完成：PICO 腕点不动，只拟合每侧一个固定 MANUS 原点 offset 与基座
> rotation；左右已暴露开发诊断均通过，但这不是新鲜盲测，仍不提升基线、不解锁
> depth。下一步必须冻结 V23 参数并使用全新未见帧验证。
> V24 已完成：46 个全新 HOLDOUT 与 prior-exposed 零交集；独立 hand mask 可观测
> 左 45、右 17，双侧覆盖门通过。它只允许进入 HaWoR 观测任务，尚未评价冻结 V23
> 骨架参数。
> V25/V26 盲测完成但双侧质量门失败；V27 仅在已暴露证据上完成常量 MANUS 原点/基座开发重拟合。
> V28 在新序列 093 上因单锚点 RGB mask 跨序列泛化失败而终止；V29/V30 未运行。
> V31 将 093 作为已暴露开发集验证多锚点方案；右手明显改善但左手区域重合与双侧画内率仍失败，无 authority 提升。
> V32 已在读取 094 RGB 前冻结并完成一次性盲测：HOLDOUT hand mask 左 39、右 55，双侧覆盖门通过；输入未改写。
> V33/V34 已只绑定不可预知的 V32 manifest SHA 和 proposal 数，算法及阈值沿用冻结 V29/V30；当前等待运行。
> 当前设计、入口与解释边界见 [V31–V34 文档](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)。

本目录只属于隔离分支 `research/wiyh-hand-depth-v1`。它不是 Chaoyang 正式基线、训练标签或物理真值。canonical 主工作树目前由另一个 AI 处理 Robot15h 长程任务；本支线不得修改、清理、提交、注册或占用其任务资源。

## 唯一当前入口

| 项目 | 当前值 |
|---|---|
| 当前盲测输入 | 只读 `/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned/playing_cards/play_cards_0916_094`；V31 的 093 仅为已暴露开发集 |
| 当前几何结论 | `V32_MASKS_PASSED / V33_V34_BOUND_READY / DEPTH_STILL_LOCKED` |
| V4 | 历史、不可变、用户否决；见 [结果更正](RESULT_ATTEMPT_V4_0001_ZH.md) |
| V5 深度 | `BLOCKED_BY_USER_REJECTED_HAND_ALIGNMENT / PRODUCER_NOT_RUN` |
| V6 | attempt_0003 已完成并终态：mask producer `PASSED`，但 Controller HOLDOUT 注册失败；C2 未拟合、视频未渲染、数据未改写；见 [V6 结果](RESULT_ATTEMPT_V6_0003_ZH.md) |
| V6 配置 | [world_in_your_hands_alignment_v6.json](../../../../configs/pipeline/world_in_your_hands_alignment_v6.json) |
| V6 设计 | [ALIGNMENT_SUCCESSOR_V6_DRAFT_ZH.md](ALIGNMENT_SUCCESSOR_V6_DRAFT_ZH.md) |
| V7 相机链诊断 | attempt_0001 已完成；P2 为推荐诊断候选但不可晋级；见 [设计与终态](CAMERA_CHAIN_SUCCESSOR_V7_ZH.md) 和 [结果](RESULT_ATTEMPT_V7_0001_ZH.md) |
| V8 全新帧验收 | [FRESH_CAMERA_VALIDATION_V8_ZH.md](FRESH_CAMERA_VALIDATION_V8_ZH.md)：attempt_0001 为软件路由失败；attempt_0002、attempt_0003 均为质量拒绝。v3 在 14/13 个画内 HOLDOUT 上把残差收敛到 21.17/25.95px P95，但仍未通过冻结门；见 [v3 结果](RESULT_ATTEMPT_V8_0003_ZH.md) |
| V9 相机刚体修正 | [设计](CAMERA_RIG_FIT_V9_ZH.md)：v1 因左右相机语义错误被拒绝；v2 精确复现 V8 后仍因右侧 HOLDOUT P95=38.03px 被质量拒绝；见 [v1](RESULT_ATTEMPT_V9_0001_ZH.md)、[v2](RESULT_ATTEMPT_V9_0002_ZH.md) |
| V10 手腕/骨架贴合 | [设计](HAND_ALIGNMENT_V10_ZH.md)：完整 SE(3) FIT 后左侧旧门通过、右侧 HOLDOUT 失败，并发现可通过改变深度缩小骨架的退化解；已质量拒绝；见 [结果](RESULT_ATTEMPT_V10_0001_ZH.md) |
| V11 局部基旋转 | [设计](HAND_BASIS_V11_ZH.md)：固定 P2、记录腕根平移和 MANUS 尺度，只拟合每侧固定 3DoF 局部基；未消费 HOLDOUT 左右均失败，右腕根仍偏离；已质量拒绝；见 [结果](RESULT_ATTEMPT_V11_0001_ZH.md) |
| V12 独立 RGB 关键点 | [设计](HAWOR_LANDMARK_CANARY_V12_ZH.md)：attempt_0001 在模型启动前因冻结环境缺少治理依赖失败；无模型推理、数据未改；见 [结果](RESULT_ATTEMPT_V12_0001_ZH.md) |
| V13 双层 HaWoR canary | [设计](HAWOR_LANDMARK_CANARY_V13_ZH.md)：双层运行架构成功，48 帧左右 detector 均零覆盖，无关键点；质量拒绝；见 [结果](RESULT_ATTEMPT_V13_0001_ZH.md) |
| V14 mask 框 HaWoR canary | [设计](HAWOR_MASK_PROPOSAL_CANARY_V14_ZH.md)：18/18 proposal 成功生成观测；右侧通过、左腕根全部画外，整体质量拒绝；见 [结果](RESULT_ATTEMPT_V14_0001_ZH.md) |
| V15 HaWoR 引导局部基 | [设计](HAWOR_GUIDED_BASIS_V15_ZH.md)：PICO wrist 平移、相机链与尺度冻结；FIT 骨段角误差下降，但独立 HOLDOUT 左右均失败，右 wrist 仍为 41.40 px P95；已质量拒绝；见 [结果](RESULT_ATTEMPT_V15_0001_ZH.md) |
| V16 MANUS root/可见掌面 | [设计](PALM_BASIS_V16_ZH.md)：全量审计确认左右 node0 全程恒等；右侧严格可见掌面仅 1 帧，低于预登记 3 帧门，唯一尝试已运行失败；未拟合、未消费 HOLDOUT；见 [结果](RESULT_ATTEMPT_V16_0001_ZH.md) |
| V17 全新 RGB mask bank | [设计](FRESH_MASK_BANK_V17_ZH.md)：48 个所选帧与既往清单零交集；左手 FIT/HOLDOUT=27/19、右手=12/10，右 FIT 未达到预登记 18 帧门，已质量拒绝；见 [结果](RESULT_ATTEMPT_V17_0001_ZH.md) |
| V18 fresh-FIT HaWoR 关键点 | [设计](HAWOR_FRESH_FIT_V18_ZH.md)：只消费 V17 可观测 FIT hand mask（左 27、右 12），通过中央 GPU 租约生成开发关键点；HOLDOUT 禁止进入本任务，不拟合 mount、不提升 authority；模型推理前路径路由失败，已终态；见 [结果](RESULT_ATTEMPT_V18_0001_ZH.md) |
| V19 路径修正版 HaWoR 关键点 | [设计](HAWOR_FRESH_FIT_V19_ZH.md)：新 task ID，只修复 V18 的相对 manifest 路径锚定；FIT-only 与全部质量门不变；左侧画面裁切导致质量拒绝；见 [结果](RESULT_ATTEMPT_V19_0001_ZH.md) |
| V20 可见非腕骨段审计 | [设计](VISIBLE_BONE_AUDIT_V20_ZH.md)：不拟合参数，只检查 V19 FIT 的双侧指内骨段覆盖和方向非退化性；GPU 禁止、HOLDOUT 禁止；双侧通过，未拟合参数；见 [结果](RESULT_ATTEMPT_V20_0001_ZH.md) |
| V21 盲测 HOLDOUT HaWoR | [设计](BLIND_HOLDOUT_HAWOR_V21_ZH.md)：19/10 个观测完整生成，中央 GPU 租约等待 29.716s、推理 10.33s；不拟合任何参数；见 [结果](RESULT_ATTEMPT_V21_0001_ZH.md) |
| V22 固定腕根基座盲测 | [设计](FIXED_BASIS_BLIND_EVAL_V22_ZH.md)：唯一尝试已完成；PICO 腕根距离 mask 左/右 P95=4.33/0 px，但右手方向 P50=35.60° 超过 35°，且双侧独立 mask 门失败，已质量拒绝；见 [结果](RESULT_ATTEMPT_V22_0001_ZH.md) |
| V23 固定腕点＋MANUS 原点/基座 | [设计与终态](MANUS_ORIGIN_FIT_V23_ZH.md)：仅用 V19 FIT 拟合每侧一个常量 origin offset＋rotation，PICO 腕点逐位不变；左右已暴露开发诊断通过，治理终态 `PASSED_PENDING_SUCCESSOR`；见 [结果](RESULT_ATTEMPT_V23_0001_ZH.md) |
| V24 全新盲测 RGB mask bank | [设计与终态](FRESH_BLIND_MASK_BANK_V24_ZH.md)：46 个全新 HOLDOUT 与 prior-exposed 零交集；hand mask 左 45、右 17，覆盖门通过；见 [结果](RESULT_ATTEMPT_V24_0001_ZH.md) |
| V25/V26 冻结几何盲测 | [设计](FRESH_BLIND_ALIGNMENT_V25_V26_ZH.md)与[结果](RESULT_ATTEMPT_V25_0001_ZH.md)：GPU/CPU 正常完成但双侧质量失败；不提升 authority、不解锁 depth |
| V27 已暴露开发重拟合 | [设计](MANUS_ORIGIN_REFIT_V27_ZH.md)与[结果](RESULT_ATTEMPT_V27_0001_ZH.md)：冻结 PICO wrist 与尺度，只拟合常量 MANUS 原点/基座；开发门通过但不是盲测 |
| V28–V30 首次跨序列盲测 | [设计](CROSS_SESSION_BLIND_ALIGNMENT_V28_V30_ZH.md)与[结果](RESULT_ATTEMPT_V28_0001_ZH.md)：093 mask 左 0、右 3，V28 质量拒绝，V29/V30 未运行 |
| V31 多锚点开发 | [V31–V34 当前文档](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)：093 已暴露开发，排除大时间偏移和错侧关联；固定 V27 几何仍未通过双侧门 |
| V32 多锚点跨序列盲测 | [V31–V34 当前文档](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)：094 在查看 RGB 前冻结，143 个 HOLDOUT；hand mask 左 39、右 55，覆盖门通过 |
| V33/V34 HaWoR＋冻结几何验收 | [V31–V34 当前文档](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)：manifest SHA 与 39/55 数量已绑定，单一入口已冻结，当前 `BOUND_READY` |
| 深度后继 | [DEPTH_SUCCESSOR_V5_DRAFT_ZH.md](DEPTH_SUCCESSOR_V5_DRAFT_ZH.md) |
| 输出边界 | 仅隔离 worktree 的 `_run/current/experiments/` |
| 数据边界 | 不移动、不删除、不改写 `/mnt/data/egodata` |
| 资源边界 | CPU 最多 2 线程；RGB mask/深度模型只能走中央 GPU 租约；不抢占其他 AI |

## 已确认事实

1. PICO 已在同一个 tracking space 中提供头显和左右 Controller 的动态 6DoF；不需要重新标定“头显到手柄的距离”。
2. 同会话有头显到 VST 相机外参、Controller 位姿、wrist 位姿和 MANUS wrist-local 25 节点。
3. HDF5 明确记录：
   - `wrist_pose = compose_pose(controller_pose, controller_to_wrist_calibration)`；
   - `left/right_hand_joints` 是相对 MANUS node0 的 wrist-local 坐标。
4. 采集端的预期重建链是 `T_camera_controller @ T_controller_wrist @ p_MANUS_local`。V4 把“头显 REP103→OpenXR 基矩阵的转置”额外当成 MANUS 手部局部基，HDF5 合同没有证明这个等同关系。
5. 0916 的视频/PICO/MANUS 时间同步在毫秒级，现有大偏移不是 40 ms 同步门造成的。
6. 原始清洗转换生成的 `review/manus_projection_eight_frame.jpg` 已经出现同样的大偏移，问题早于本支线渲染器，也与 FoundationStereo 无关。
7. 现有 Controller→wrist 平移模长约 157.8 mm。删除它的 V2 仍被否决，所以不能仅靠“平移归零”解决。
8. V4 只旋转手指方向，没有修复腕根的大偏移；用户已明确否决。

## 为什么旧机器 PASS 不成立为贴合证明

V4 的自动门只检查：

- 固定矩阵正交、行列式为 +1、角度为 120°；
- 代码中的矩阵与 `camera_params.json` 某个头部坐标基字段一致；
- 变换前后腕根平移不变；
- 关节仍在画面内。

这些检查能证明“变换执行一致”，不能证明“变换语义正确”。画内率也不测量骨架是否落在真实手上。V4 没有独立 hand mask、mocap、实体标记或 Controller 视觉检测，因此不能自证。

## 当前诊断顺序

V6 严格按两层定位，不能直接盲调腕偏移：

1. 用只看 RGB 的左右 Controller mask 检查投影后的 PICO Controller 原点。
2. 如果 Controller 原点持续远离可见 Controller，优先排查 head→camera 外参、相机方向、sourceIndex、像素域或时间链。
3. 只有 Controller 投影通过，才检查固定 `T_controller_wrist`。
4. 用独立左右 hand mask 计算 joint-in-mask、bone-in-mask、wrist-to-mask 和低分帧。
5. 若需要估计新安装变换，只允许每只手一个整段恒定 SE(3)，只在 FIT 帧拟合；禁止逐帧偏移、时间变化偏移或用触觉反证。
6. 完全未参与拟合的 HOLDOUT 帧达到冻结阈值，并且用户检查完整视频后，才允许进入 V5 深度。

## 独立 RGB 证据合同

已实现：

- `src/chaoyang/research/world_in_your_hands/alignment_metrics.py`
- `src/chaoyang/research/world_in_your_hands/alignment_audit_v6.py`（只读消费完整 mask manifest，先验收 Controller，再比较 C0/C1；不拟合、不写产物）
- `src/chaoyang/research/world_in_your_hands/sam31_alignment_v6_contract.py`（RGB-only 提示与时序质量门）
- `src/chaoyang/research/world_in_your_hands/sam31_mask_producer_v6.py`（项目内库；无 CLI、未执行，等待治理 packet 与 GPU 租约）
- `src/chaoyang/research/world_in_your_hands/alignment_fit_v6.py`（C2 固定 SE(3) FIT-only 优化器；Controller 门失败或输入含 HOLDOUT mask 时拒绝）
- `src/chaoyang/research/world_in_your_hands/alignment_review_v6.py`（全片 C0/C2 并排可视化；只显示候选，不自行提升 authority）
- `configs/pipeline/world_in_your_hands_alignment_v6.json`
- `tests/research/test_world_in_your_hands_alignment_metrics.py`
- `tests/research/test_world_in_your_hands_alignment_v6_config.py`
- `tests/research/test_world_in_your_hands_sam31_alignment_v6_contract.py`

Mask 可使用项目内固定 SAM3.1 或 RGB-only HaWoR 图像框，但 receipt 必须声明 `INDEPENDENT_RGB_ONLY`，且 conditioning 不能包含 MANUS、PICO、Controller/wrist 位姿、投影骨架、TAC 或触觉。FIT/HOLDOUT 同一 frame/side 交叉会直接失败。

冻结的 HOLDOUT 门包括：

- 左右手各至少 8 帧；
- joint-in-mask P50 ≥ 0.85；
- bone-in-mask P50 ≥ 0.75；
- wrist-to-mask P95 ≤ 20 px；
- joint-to-mask P95 ≤ 20 px；
- Controller 原点在独立 Controller mask 内的比例 ≥ 0.80，距离 P95 ≤ 20 px；
- 低分帧与完整视频仍需人工复核。

这些门只证明可见 2D RGB 重合，不是毫米级 3D、解剖、接触、力或遮挡表面的真值。

## V5 深度状态

V5 代码可以继续做静态和单元测试，但不得构建新可执行 packet、注册或运行。当前配置包含：

- `current_status=USER_REJECTED_GROSS_IMAGE_MISALIGNMENT`
- `required_status=PASS_INDEPENDENT_RGB_ALIGNMENT_HOLDOUT_AND_USER_ACCEPTED_FULL_VIDEO`

`task_contract_v5.require_alignment_prerequisite(...)` 已接入 builder、registrar 和 runner。此前未注册的 packet 与修改后的代码/config SHA 已漂移，已从 `tasks/current/` 迁到 `tasks/receipts/WIYH_V5_UNREGISTERED_PACKET_FINAL_STALE_ALIGNMENT/` 并写入 `SUPERSESSION.json`；它只能历史追溯，不是执行授权。

磁盘方面，`df` 虽报告 0 可用，但 64 MiB 写入→fsync→读回 SHA→删除 canary 已实际通过且无残留。新策略只允许：

- canary 成功时承认 CPFS 仍可写；
- 输出树硬上限 2 GiB；
- 输出和 canary 只能位于本 worktree 的 `_run/current`；
- 禁止符号链接逃逸；
- 若 `df` 恢复为非零，仍执行 10 GiB 安全余量/下降门。

这只解除“虚假磁盘只读”判断，不解除几何、治理、GPU 或并行任务门。

## 历史结果

- [V1](RESULT_ATTEMPT_0001_ZH.md)：人工否决。
- [V2](RESULT_ATTEMPT_V2_0001_ZH.md)：删除 157.8 mm 平移，人工否决。
- [V3](RESULT_ATTEMPT_V3_0001_ZH.md)：恢复采集端预期固定腕变换并用 encoded-domain pinhole，人工否决。
- [V4](RESULT_ATTEMPT_V4_0001_ZH.md)：额外套用头部坐标基旋转，机器自洽但人工否决。
- [V5](DEPTH_SUCCESSOR_V5_DRAFT_ZH.md)：未运行，等待 V6。
- [V6](RESULT_ATTEMPT_V6_0003_ZH.md)：独立 RGB 证据已终态定位到相机/图像域注册失败；V6 设计见 [设计页](ALIGNMENT_SUCCESSOR_V6_DRAFT_ZH.md)。
- [V7](RESULT_ATTEMPT_V7_0001_ZH.md)：P2 双目链通过已暴露 mask 诊断。
- [V8 v3](RESULT_ATTEMPT_V8_0003_ZH.md)：P2 在全新 HOLDOUT 上仍以 21.17/25.95px P95 未通过，保持诊断初值。
- [V9 v2](RESULT_ATTEMPT_V9_0002_ZH.md)：常量相机修正不能在右侧 HOLDOUT 泛化，质量拒绝。
- [V10 结果](RESULT_ATTEMPT_V10_0001_ZH.md)：全 SE(3) hand-mask 拟合右侧失败，并暴露缩小骨架的退化解。
- [V11 结果](RESULT_ATTEMPT_V11_0001_ZH.md)：固定腕根平移与尺度后，局部基 rotation-only 仍未通过独立 HOLDOUT；相机/腕根与局部基误差并存。
- [V12 结果](RESULT_ATTEMPT_V12_0001_ZH.md)：模型启动前的软件路由失败；后继改为系统治理外壳与 HaWoR inner worker 分离。
- [V13 结果](RESULT_ATTEMPT_V13_0001_ZH.md)：双层入口与 GPU 执行成功；自带 detector
  在 48 帧上零覆盖，未产生 HaWoR 关键点，质量拒绝。
- [V14 结果](RESULT_ATTEMPT_V14_0001_ZH.md)：独立 RGB-only mask 框恢复全部 HaWoR
  观测；左腕根因画面下缘截断不可验证，质量拒绝。
- [V21 结果](RESULT_ATTEMPT_V21_0001_ZH.md)：冻结盲测观测完整生成，不代表姿态质量通过。
- [V22 结果](RESULT_ATTEMPT_V22_0001_ZH.md)：PICO 动态腕根锚点得到独立 mask 支持；
  固定 MANUS 基座的指骨方向与手部区域重合未通过冻结门，质量拒绝。
- [V23 结果](RESULT_ATTEMPT_V23_0001_ZH.md)：固定 PICO 动态腕点后，常量 MANUS
  原点 offset＋rotation 在已暴露开发证据上双侧通过；仍需冻结参数后的全新盲测，
  不提升正式基线、不解锁深度。
- [V24 结果](RESULT_ATTEMPT_V24_0001_ZH.md)：46 个全新 HOLDOUT 的 RGB-only mask
  覆盖门通过；尚未运行冻结 V23 参数的几何盲测，不提升 authority、不解锁深度。
- [V25/V26 冻结设计](FRESH_BLIND_ALIGNMENT_V25_V26_ZH.md)：在 V25 GPU 运行前冻结 V23 参数、V26 验收算法和阈值；随后以 V24 新鲜 HOLDOUT 运行 HaWoR、CPU 盲测并生成 166 帧完整视频。
- [V25/V26 结果](RESULT_ATTEMPT_V25_0001_ZH.md)：GPU 与盲测正常完成但质量失败；PICO wrist 接近手区，失败集中在右手指骨架区域重合和左侧展开度，不提升 authority、不解锁深度。
- [V27 设计](MANUS_ORIGIN_REFIT_V27_ZH.md)：只用已暴露 V24/V25 做开发重拟合，固定 PICO wrist、尺度和所有逐帧参数；结果必须再经新的盲测。
- [V27 结果](RESULT_ATTEMPT_V27_0001_ZH.md)：双侧暴露开发门通过并冻结候选；这不是盲测，后继仍须用从未参与本系列算法的帧做独立 RGB-only/HaWoR 验收。
- [V28–V30 跨序列盲测设计](CROSS_SESSION_BLIND_ALIGNMENT_V28_V30_ZH.md)：092 已无未见帧，因此冻结使用从未被本支线引用的 093；先生成 RGB-only masks，再运行 HaWoR 和不拟合的 V27 跨序列验收。
- [V28 结果](RESULT_ATTEMPT_V28_0001_ZH.md)：GPU 与数据完整性检查均正常，但复用自
  092 的冻结 RGB 提示不能在 093 泛化；HOLDOUT hand mask 左 0、右 3，低于每侧 20
  帧门槛，质量拒绝。V29/V30 未获执行授权；093 从此只可作为开发诊断，不再属于新鲜
  盲测。
- [V31 多锚点开发结果与时间审计](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)：093 上 mask/HaWoR/完整视频已生成；无强全局时间移位证据，左右手关联正确，但冻结几何仍未过门。
- [V32–V34 冻结设计](MULTISEED_BLIND_ALIGNMENT_V31_V34_ZH.md)：094 为未见跨序列输入；先验收独立多锚点 RGB masks，通过后才允许机械绑定并运行冻结 V29/V30 算法。

V1–V4 的不可变产物保留用于追溯，不得覆盖、重跑或作为当前入口。

## 论文关系

World In Your Hands 只提供方法原则：把腕部追踪、手骨架、相机、触觉和深度分别声明，并把骨架投到第一视角图像，用独立手部区域和人工低分样本复核。它使用的硬件不同，不能把论文精度直接迁移到 MANUS＋PICO＋TAC。

本支线运行时不下载外部代码、模型或包。参考只作 provenance：

- 论文：https://arxiv.org/html/2512.24310
- 官方代码：https://github.com/tars-robotics/World-In-Your-Hands
- HaWoR 论文：https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.html
