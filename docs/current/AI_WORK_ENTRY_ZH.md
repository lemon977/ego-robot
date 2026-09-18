# 后续 AI 工作入口与优化边界

状态：CURRENT
基线：`clean-baseline-v1`
发布标签：`clean-baseline-v1-final`

本页是交给后续 AI 的最小执行入口。事实仍以当前状态 receipt、文档权威表、算法合同和任务索引为准；聊天、历史目录及文件名中的 `current/latest/final` 均不授予执行权。

## 必须按顺序读取

1. [`CURRENT_STATUS_RECEIPT.json`](../governance/CURRENT_STATUS_RECEIPT.json)
2. RC1 工作读 [`CURRENT_RC1_STATUS_MIN.json`](../governance/CURRENT_RC1_STATUS_MIN.json)，其他工作读 [`CURRENT_PROJECT_STATUS_MIN.json`](../governance/CURRENT_PROJECT_STATUS_MIN.json)
3. [`DOC_AUTHORITY_MAP.json`](../governance/DOC_AUTHORITY_MAP.json)
4. [`ALGORITHM_CONTRACT.json`](../governance/ALGORITHM_CONTRACT.json)
5. [`tasks/current/INDEX.json`](../../tasks/current/INDEX.json)
6. 与任务直接相关的合同、代码和 fixture；不要默认加载整个归档

开始前执行：

```bash
PYTHONPATH=src python -m chaoyang.cli validate-governance
PYTHONPATH=src python scripts/migration/validate_structure.py --allow-dirty
```

## 当前可执行状态

- 已封账的 bounded 历史路由顺序为
  `0915_stereo_interaction_cpu_canary_v1 → 0915_sam31_weak_role_canary_v1 →
  0915_removal_envelope_single_session_canary_v1 →
  0915_foundationstereo_single_session_canary_v1 →
  0915_planar_object6d_single_session_canary_v1`。当前索引无活动任务；其中
  FoundationStereo 使用了错误的重复 lens-undistortion，Object6D 未启动，不能从这条
  历史路由续跑。
- 只有索引中 `execution_allowed=true` 且与账本 `next_task` 一致的任务包可以调度。
- 空索引内的 revision 11169 是创建 revision；当前生效 revision 读取 `CURRENT_V71_TASK_PACKET_INDEX.json` 和 `CURRENT_STATUS_RECEIPT.json`。冻结 payload 的创建 revision 不要求被原地改写。
- `tasks/receipts/HISTORICAL_TASK_CATALOG.json` 和 `archive/` 只用于查询/恢复，任何历史终态任务都不得直接重启。
- 0911/0914/0915 数据清洗已经完成，见 [`DATA_CLEANING_0911_0915_ZH.md`](DATA_CLEANING_0911_0915_ZH.md)；它不是待办任务。
- 0915 原重复去畸变全链已经按用户要求停止；此后只以独立的新任务运行 resize-only
  bounded canary。用户已确认所有 VST 编码视频本来没有畸变；所有
  视频消费者只能按 `sourceIndex` 裁出物理眼再 resize，禁止把 `camera_params.json` 中的
  `equiDis62` 字段应用到解码像素。现有重复去畸变派生结果不得作为基线。先读
  [`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)，再复核
  [`单会话 A/B`](visuals/0915_VST_IMAGE_DOMAIN_AB_V1/README_ZH.md)。A/B 已证明当前
  remap 是百像素级变换，且 legacy 单目是物理右目。用户已经确认物理左目
  `sourceIndex=1 + resize-only` 为正确单目画面。raw HaWoR canary 的右手画面边界/骨长
  门为 `FAILED_QUALITY_C`；其后 `hawor_bounded_v2` 已通过冻结数值门和用户全片复核。
  上游 detector/track 的
  1–2 帧内部缺口另有短缺口连续性 successor：它保持 `observed` 和已有观测几何不变，
  只新增 `short_gap_inferred` 与 `visual_continuity_valid`。Contact、严格覆盖率及直接
  观测统计只能消费 `observed`；离线可视化/运动消费者只有显式声明后才能消费
  `visual_continuity_valid`，Robot online 不得消费非因果补帧。用户已确认单样本视觉
  无明显问题；运行与可视化产物随后按要求带收据清理，代码、单元测试及结果口径保留，
  未自动扩批。复核入口见
  [`bounded_v2`](visuals/0915_HAWOR_BOUNDED_V2_CANARY_V1/README_ZH.md)，短缺口结论见
  [`结果收据`](../../tasks/receipts/0915_HAWOR_SHORT_GAP_CONTINUITY_V1_RESULT.json)。Depth 仍为
  独立未决问题；后续 Mask 路线固定为 SAM3.1，不运行 SAM2.1/Cutie 选型。
- 0916 独立清洗已经完成；0915/0916 状态见
  [`FULL_FUNNEL_0915_AND_CLEANING_0916_ZH.md`](FULL_FUNNEL_0915_AND_CLEANING_0916_ZH.md)。
- 用户已确认 resize-only 图像域和单样本 HaWoR 视觉结果，并接受现有 SAM3.1 结果作为
  下一轮 bounded 优化起点，不是全角色质量通过。左右手与前两张牌形成回归证据；前臂、
  皮套、线缆和第三张牌仍有大量 `unknown`，旧大框还存在吞并整手的初始化偏差。当前只对
  这些弱角色运行更紧的 `initial_visual_box`、逐个可见 sleeve 实例和质量触发 reseed；不得
  把 `unknown` 解释为角色不在画面，也不得自动进入批量 Clean/Contact。旧结果入口见
  [`0915_SAM31_STRICT_ROLE_CANARY_V1`](visuals/0915_SAM31_STRICT_ROLE_CANARY_V1/README_ZH.md)。
  只能使用已固定的 SAM3.1 权重，不得创建 SAM2.1/Cutie 候选、胜者选择任务或自动扩到
  220 会话。

- SAM3.1 弱角色单样本已经执行完成，浅层入口见
  [`0915_SAM31_WEAK_ROLE_CANARY_V1`](visuals/0915_SAM31_WEAK_ROLE_CANARY_V1/README_ZH.md)。
  v5 左右手及前两张牌通过 SHA/字节回归门且没有重算；新前臂有证据 92/150、90/150，
  `playing_card_02` 为 115/150，右线缆为 62/150，但各皮套与左线缆仅 6–8/150。
  因此任务运行终态为 `PASSED`，但用户视觉验收已将其判为
  `REJECTED_QUALITY_AS_CLEAN_BASELINE`，不得自动扩批或进入 Clean/Contact。皮套/左线缆
  虽有完整方向的 propagation yield，但 primary/fallback 各自仅 4/150 帧 raw mask 非空；
  主阻塞是 raw track 没产出，面积门只是将空结果显式 fail-closed。前臂另有反向 tracker
  未确认实例，后续必须分开优化。

- 下一版 Clean 候选固定为 Raw Candidate / Semantic / Removal / Feather 四层隔离。V1
  结构性失败见 [`REMOVAL_ENVELOPE_V1_ZH.md`](REMOVAL_ENVELOPE_V1_ZH.md)；V2 合同见
  [`REMOVAL_ENVELOPE_V2_ZH.md`](REMOVAL_ENVELOPE_V2_ZH.md)。V2 以 admitted SAM
  foreground 为唯一主体，MANO、sleeve、forearm proposal 和 cable appearance 只能做
  有局部支持的有限 repair，证据不足保持 `UNKNOWN`。V2 核心、schema、配置和合成测试已
  实现，但真实视频 Canary 尚未运行，不能宣称 Clean 质量通过。Removal/Feather 继续禁止
  进入 Depth、Object6D、Contact、Robot geometry/control truth。

- Removal Envelope V1 已完成 150 帧 CPU 单样本执行，浅层入口见
  [`0915_REMOVAL_ENVELOPE_CANARY_V1`](visuals/0915_REMOVAL_ENVELOPE_CANARY_V1/README_ZH.md)。
  自动门确认 293 个直接观测 side-frame、7 个双端验证内部 hold、0 个 geometry unknown，
  direct MANO21 关节覆盖 1.0，source bits 闭合且 protected core 零交集；这些自一致性门
  没有测量背景误擦或 cable 身份精度。用户完整视频复核已将质量终态判为
  `REJECTED_QUALITY`：V1 把 MANO 中心线、wrist→边界前臂猜测及宽松黄色候选直接升级成
  强 removal。V1 作为失败实验封存，禁止调半径/HSV 重试、inpaint 或扩批。V2 只允许
  `SAM admitted foreground + validated local repairs`；当前 16 项 V2 单测通过，但缺真实
  foreground proposal、stable-background hook 与 cable reverse-pass 的实片证据，因此
  状态为 `NOT_EVALUATED`。Depth/Object6D 不依赖 Clean，继续走独立证据门。

首个 CPU coordinator 已完成运行，两条 lane 的浅层证据见
[`0915_STEREO_INTERACTION_CPU_CANARY_V1`](visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1/README_ZH.md)。
Stereo 与 Interaction 是两条不同证据链：Stereo 比较 raw
resize-only 和单会话内参＋图像估计/held-out 验收的 rectified 域；Interaction v0a 只计算
2D 邻接、接近和共动，并允许 `entities.tactile.offline_source_valid` 提供弱支持。Interaction
v0a 明确没有 relative-Z、遮挡顺序、接触真值、Object6D 或 Robot authority。Stereo 的
raw resize-only 匹配统计保留为诊断；消费 `equiDis62` 的 rectified 分支及原
`PASS_GPU_DEPTH_ADMISSION` 已按用户确认撤销，不能授权 FoundationStereo。
当前 0915 processed 挂载实体名为单下划线 `chips_cards_hands_0915`，而冻结收据保留逻辑
发布名 `chips_cards_hands__0915`；只有枚举文件 bytes/SHA 与
[`0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json`](../../tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json)
同时闭合时才允许解析该只读路径搬迁，禁止宽泛字符串替换或修改数据侧文件。

- FoundationStereo 单会话曾完整运行 150 帧，模型加载 1 次、双向推理 300 次且运行
  闭包通过；不可变执行终态为 `REJECTED_QUALITY`。但该运行对已经无畸变的 VST 编码
  像素再次应用 `equiDis62`，当前证据 authority 为 `WITHDRAWN_WRONG_IMAGE_DOMAIN`。
  左右一致性残差跨帧 P90 为
  `17.8653 px`（门限 5 px），时序深度中位数步长 P90 为 `0.4290 m`（门限 0.35 m）。
  `consumption_authorized=false` 且授权 scope 为空，因此 Planar Object6D 没有启动。
  浅层复核见
  [`0915_FOUNDATIONSTEREO_CANARY_V1`](visuals/0915_FOUNDATIONSTEREO_CANARY_V1/README_ZH.md)。

- 全局 VST 编码域确认收据为
  [`VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json`](../../tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json)。
  相机文件中的 distortion 字段只能保留作采集 provenance。未来 Depth 如需极线对齐，
  必须新建编码域、零 lens-undistortion 的标定 successor；不得复活旧任务或旧 remap。
  该边界同样适用于旧 exact78 worker；原 Depth `58/58` 与依赖它的 Object6D `58/58`
  已分别改记为 `0 passed / 58 blocked`。依赖这批几何的 Contact/Occlusion 声明已撤回，
  后续不得把历史文件或旧内部一致性指标当作当前 Depth/Object6D authority。

- 新的 CPU-only encoded-domain Stereo preflight 已在 `play_cards_0915_001` 全 150 帧完成，
  浅层入口见
  [`0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_V1`](visuals/0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_V1/README_ZH.md)。
  它只做 `sourceIndex` 裁切和 resize，未消费 distortion、未做 lens remap、未使用 GPU。
  34,280 个 robust matches 的 `|dy|` median/P90/P95 为
  `1.9432 / 2.8167 / 3.0964 px`，150/150 帧满足最小匹配门，frame 81/94 均无局部异常，
  因此结论为 `PASS_DIRECT_FOUNDATION_INPUT`：本会话不需要额外 encoded-domain 极线 remap。
  但物理左减物理右的视差符号以 `0.9994` 一致率为负；随后 GPU successor 已按用户
  确认固定“两眼同时水平镜像→模型→输出镜像回来”的正视差 model adapter。该适配不等于
  去畸变，左右物理相机没有交换。

- 新的 FoundationStereo encoded-domain canary 已完成 150 帧并通过全部内部质量门，浅层
  入口见
  [`0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1`](visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1/README_ZH.md)。
  输出已反镜像回原物理左目 640×480 并使用原左目 K；RGB/Depth 像素往返最大误差为
  `0`，错位像素为 `0`。几何有效比例中位数 `0.95983`，左右一致比例中位数
  `0.94666`，Depth 中位数帧间变化 P90 为 `0.00445 m`。结果仅授权给同会话新建的
  三牌独立 Object6D observability canary；外部 30/50/70/100 cm 标定缺失，因此
  `external_accuracy=UNVERIFIED`，不授权 Contact、Robot 或批处理。

- 用户所称“旧基线”固定为 156 会话 frozen exact78 批量基线，而不是本轮 SAM/Removal
  单样本实验。旧 Poker 对照为 `play_cards_0902_042`，旧 Chips 对照为
  `get_potato_chips_0902_103`；完整比较边界见
  [`0915_OLD_BATCH_BASELINE_COMPARISON_ZH.md`](0915_OLD_BATCH_BASELINE_COMPARISON_ZH.md)。
  旧 Depth/Object6D 已撤权，不参与新深度数值比较。

- 新三牌 Object6D v2 已在同一 encoded physical-left 域完成。`card00/01/02` 的可见表面
  中心分别为 `143/146/95` 帧，平面法向分别为 `139/113/48` 帧；其余帧按 Mask unknown
  或非平面证据 fail-closed。卡托保持独立 `UNKNOWN`，`card_set` 仅为语义组。牌尺寸
  尚未实测，因此 full extent 为 `0/150`，没有补造完整 6DoF。入口见
  [`0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2`](visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2/README_ZH.md)。
  Interaction/Contact 前仍需逐帧几何叠加审阅和实测牌宽高。

- 单样本 Interaction → Contact → Kai22 开发链已经完成，入口见
  [`INTERACTION_CONTACT_ROBOT_DEV_V1_ZH.md`](INTERACTION_CONTACT_ROBOT_DEV_V1_ZH.md)。
  Object6D 逐帧 QA 通过，牌尺寸因完整边界不可观测保持 `UNKNOWN`。旧 joint-centre
  对齐 hold-out P90 为 `36.91 mm`；冻结 MANO 前表面候选把有界结果改善到
  median/P90 `7.87/19.33 mm`，但尺度正好卡在 `0.8` 下界，无约束最优值为 `0.783730`。
  因此最终对齐状态是 `REJECTED_BOUNDED_FIT_SATURATION`，不授权公制 wrist-object 平移。
  4,500 条固定配对中最近有限 patch 距离仍为 `6.60 mm`，没有五帧 Contact 窗口。
  因而 Kai22 R0 已在 293/293 direct-observed side-frame 交付，R1 正确终态为
  `BLOCKED_LOCAL_EVIDENCE`，R2 未运行。不得统一减去 48 mm、放宽 5 mm Contact 门或把
  未修改的 R0 称为 Contact-aware refinement，也不得仅放宽尺度边界后用同一数据自证。
  三段浅层视频均完整解码 150 帧；新增表面关联和尺度边界浅层诊断图见该入口。

- Removal Envelope V2 真实 150 帧 canary 已完成并终态为 `REJECTED_QUALITY`，浅层入口见
  [`0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1`](visuals/0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1/README_ZH.md)。
  V2 没有重跑 SAM、没有 inpaint、没有 GPU；repair P95 贡献仅 `0.00293`，面积膨胀 P95
  `1.00294`，保护物体核心损伤 `0 px`，且相对 semantic baseline 没有增加闪烁。
  但最终 temporal area derivative P95 仍为 `0.8424`，几乎等于原 SAM base 的 `0.8437`；
  说明 V2 的保守 repair 不会制造 V1 的大片误擦，却也无法修复上游 SAM 闪烁。全片严格稳定
  背景 hook 只覆盖 `0.00163`，同样未过门。不得继续调 repair 半径，也不得进入 inpaint/扩批；
  下一次 Clean 实验必须先解决 semantic temporal admission，并重新设计局部背景 QA。

如果用户提出新目标，应建立新的、有限收敛的任务包并发布新的治理 revision；不要把旧任务包改回 `PENDING`。任务包至少固定输入、代码、配置、权重或 `ABSENT`、标定或 `ABSENT`、输出 schema、质量门、预算、终止条件和回滚路径。

## 脚本和任务命名

- 当前 Python 操作名采用 `<verb>_<subject>[_<scope>]_vN.py`，CLI operation 等于文件 stem。
- 名称必须表达动作和对象，例如 `tactile_quality_gate_v1`、`batch_clean_handle_content_v3`；禁止使用含糊的 `run.py`、`new.py`、`latest.py`、`final.py`。
- 版本升级必须在算法合同中只保留一个当前实现；旧版本移入 `archive/`，不得长期维护双实现。
- 新任务 ID 采用 `<domain>_<bounded_deliverable>_vN`；日期只用于不可变采集批次或运行实例，不代替语义版本。
- 运行产物放在 `_run/current/<task_id>/attempts/attempt_NNNN/`，不得散落到源码、文档或仓库外路径。
- 当前维护操作统一使用 `PYTHONPATH=src python -m chaoyang.cli run <operation> ...`；CLI 会拒绝未登记到算法合同的模块。

## 后续优化顺序

以下是边界清晰的候选工作，不代表已授权执行；每项都需要新的 current 任务包。

1. **P0：解除 RC1 数据量门。** Chips 与 Poker 当前均为 `BLOCKED_DATA_VOLUME`。先满足合同要求的独立 source group 数量与 train/validation 隔离；不得用同一 merged recording 切出的多个 session 重复计数。
2. **P1：RC1 checkpoint 与比率验收。** 只有 P0 通过后，才按 [`CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md`](../governance/CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md) 训练和比较四个 checkpoint；GPU 必须走租约，不与现有实验抢占。
3. **P2：算法 bounded canary。** HaWoR、Mask、Depth、Object6D、Contact、Robot 或 HumanEgo 的改动先固定失败样本、输入签名和质量门，运行小 canary，再决定是否扩批。不得依据视觉观感提升 authority。
4. **P3：数据清洗增强。** V3 已完成；后续可增加标定力/独立接触真值、更多损坏 fixture 和性能 profiling，但不得改变三批已发布终态或把缺失 MANUS 伪造成数据。

H4 自动 glove/Controller Mask-to-Clean 路线已经被当前策略拒绝，除非出现新的独立证据和新合同，不得自动恢复。Attachment 不得反向证明对象身份、Contact 或物理真值。

## 路径和资源边界

- 原始数据根：`/mnt/data/egodata/datasets/ego`
- 处理数据根：`/mnt/data/egodata/datasets/ego/processed`
- 运行根：`<repo>/_run/current`
- 模型根：`<repo>/assets/models`
- FoundationStereo 环境：`<repo>/_run/current/environments/foundationstereo-py311-v1`
- HaWoR 环境：`<repo>/_run/current/environments/hawor-py310-v1`
- HaWoR 算子必须通过 `src/chaoyang/ops/hawor_python.sh <runner.py> ...` 启动；不得用系统
  `python` 直接运行。HaWoR runner 必须在读取输入或通过 preflight 前以
  `WRONG_RUNTIME_ENTRYPOINT` 拒绝错误解释器。

不得移动、删除或改写原始数据；不得把临时结果写到约定路径之外。GPU 工作必须经过租约，治理、哈希、文档、归档和普通测试使用 CPU。

## 完成定义

每个后续任务结束时必须同时满足：

- 有不可变结果/失败/阻塞收据，且计数来自机器文件；
- 代码、合同、文档和任务指针由同一发布 receipt 绑定；冻结 payload 必须显式区分 created/effective/published revision；
- 当前 Markdown 链接、artifact bytes/SHA、导入和配置引用均通过校验；
- `pytest`、治理校验、结构校验和相关 canary 通过；
- `git status --porcelain` 为空；
- 归档或删除动作有 inventory、Merkle、PATH_REDIRECTS 和恢复说明。

若任何门不满足，应明确写 `BLOCKED_*` 或 `FAILED_*`，不得用聊天描述替代终态收据。
