# Clean 分层 SAM3.1 successor 探索测试 V1

状态：`EXPERIMENT_DEFINED / GPU_CANARY_NOT_STARTED`

任务：`clean_layered_sam31_successor_exploration_v1`

产物 revision：`R7_CLEAN_SAM31_LAYERED_EXP_1`

## 1. 结论先行

这条路线值得测试，但不照搬 Masquerade 的旧模型组合，也不修改当前 Clean 基线。

冻结结论如下：

- 借用 Masquerade 的流程编排：最佳锚帧、左右手独立、双向时序复核、分割与补洞解耦。
- 掩码核心使用项目已经固定代码和权重 SHA 的 **SAM3.1 multiplex video predictor**，不退回 SAM2/SAM2.1。
- 残余视频补洞继续优先使用项目已有 **ProPainter**，不因 Masquerade 使用 E2FGVI 就降级替换。
- 不照搬固定大膨胀。接触物体附近采用更小的自适应边界，远离物体才允许扩大。
- Clean 不再只有一张最终 RGB；必须同时输出背景层、物体层、未知区和逐像素来源。
- 双向传播只用于离线掩码复核和展示；正式训练输入必须是因果模式，帧 `t` 不使用未来 RGB donor。
- 当前 58 条 `same_pixel_temporal_donor_then_propainter_v1` 结果保持原路径、原 SHA 和原 authority，本实验不得覆盖或自动触发重跑。

这个方案的主要目标不是“让补洞更像”，而是先解决当前两个已观察到的错误：

1. 固定大膨胀把手指接触边界外的像素也擦除。
2. donor 没有语义/实例约束，可能把盘子或桌面像素复制到错误位置；物体被手挡住后又缺少合法外观来源。

## 2. 当前基线与隔离边界

当前机器注册表中的 Clean 基线是：

```text
same_pixel_temporal_donor_then_propainter_v1
Wave0: 58 / 58 structural PASSED
```

这里的 `PASSED` 只表示结构、来源清单和视频解码闭环，不代表接触边界或语义正确。

本实验只能写入：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/
```

禁止写入或覆盖：

```text
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/
archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/
docs/governance/CURRENT_AUTHORITY_INDEX.json
docs/governance/CURRENT_BASELINE_REGISTRY_V2.json
```

只有 canary、旧 A/B regression 和人工接触帧审核全部通过后，才可由治理 aggregator 建立独立 successor revision；旧 `R7_0` 仍保持可复现。

## 3. 从 Masquerade 借什么、不借什么

### 3.1 已由官方源码核实的做法

Masquerade/Phantom 开源处理器把手/臂分割、视频补洞和 Robot overlay 分开。分割处理器从质量较高的锚帧初始化，分别处理左右手，并向前、向后传播；补洞处理器使用 E2FGVI 的邻帧和参考帧上下文；Robot overlay 可比较真实深度和渲染深度。

核实来源：

- [Masquerade 项目页](https://masquerade-robot.github.io/)
- [Phantom/Masquerade 官方仓库](https://github.com/MarionLepert/phantom)，本次核实 HEAD：`a8bb81c1bbe6ade129a1f6f0906482f510354a5e`
- [segmentation_processor.py 固定提交](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/segmentation_processor.py)
- [handinpaint_processor.py 固定提交](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/handinpaint_processor.py)
- [robotinpaint_processor.py 固定提交](https://github.com/MarionLepert/phantom/blob/a8bb81c1bbe6ade129a1f6f0906482f510354a5e/phantom/processors/robotinpaint_processor.py)

### 3.2 直接采用的流程思想

| 思想 | 本项目实现 |
|---|---|
| 不从固定 frame 0 启动 | 每只手/前臂独立选择质量锚帧；因果模式使用“截至当前最佳锚帧” |
| 左右手分开传播 | 保留 left/right hand、forearm、Controller、accessory 独立层，最后才组合 removal mask |
| 双向传播 | 仅用于 `OFFLINE_BIDIRECTIONAL_QA`，用于发现漂移和重入失败 |
| 检测框与手证据关联 | HaWoR 或 Controller/MANUS 提供身份锚；SAM3.1 文本、点、框共同决定真实像素边界 |
| 视频级补洞 | 真实 donor/atlas 优先，剩余未知区才交给 ProPainter |
| 渲染与 clean 解耦 | Clean 不决定 Robot/Object 谁在前；后续 compositor 使用统一 z-buffer |

### 3.3 明确不照搬的部分

| Masquerade 做法 | 本项目决定 | 原因 |
|---|---|---|
| SAM2 | SAM3.1 | 已有固定权重和视频 predictor，不应降级 |
| E2FGVI-HQ | ProPainter 为主，E2FGVI 仅可做独立研究对照 | 项目已有 ProPainter 代码/权重；换模型不是当前主要错误来源 |
| 固定 3×3 CROSS、4 次 dilation | 接触区/远区分开的自适应膨胀网格 | 固定扩大正是当前物体和接触边界被误删的重要原因 |
| person/arm 检测框直接决定手臂 | 身份传感器作锚、SAM3.1 作像素边界、检测器只作 proposal | 第一视角只有局部前臂；手套、手柄和线缆不是普通 person 框 |
| 单张 inpainted RGB 作为最终结果 | 背景、任务物体、未知区和来源图分层 | 隐藏物体纹理不能由背景补洞模型合法猜出 |
| 全片双向参考用于任何用途 | 展示与训练模式严格分离 | future donor 会向 H50 训练输入泄漏未来信息 |

ProPainter 官方实现本身就是视频级 flow propagation + spatiotemporal completion，而非逐帧 LaMa；其项目与论文分别见 [官方仓库](https://github.com/sczhou/ProPainter) 和 [论文](https://arxiv.org/abs/2309.03897)。因此本实验首先修复 mask、layer 和 provenance，不把“换成 E2FGVI”当成主假设。

## 4. Successor 架构

```text
Raw RGB
  │
  ├─ 身份锚
  │    exact78: HaWoR left/right joints + bbox + valid
  │    新传感器: Controller wrist + MANUS hand + Controller 6DoF
  │
  ├─ SAM3.1 role/object video segmentation
  │    left hand / left forearm
  │    right hand / right forearm
  │    left/right Controller
  │    cable/accessory
  │    task-object instance 0..N
  │    support surface / distractor proposal
  │
  ├─ Temporal QA
  │    offline: global best anchor + forward/backward
  │    causal: best-so-far anchor + forward + bounded re-seed
  │
  ├─ Adaptive removal support
  │    removal_core = observed human/controller/accessory
  │    removal_support = small near object, larger far from object
  │    visible task object always excluded from removal
  │
  ├─ Layer reconstruction
  │    background: semantic-valid causal donor/stereo warp
  │    task object: current raw or pose/flow-validated causal atlas
  │    remaining hole: ProPainter
  │    no legal source: UNKNOWN
  │
  └─ Outputs
       clean_background_rgb
       task_object_rgba per instance
       removal_core / removal_support
       unknown_mask / training_valid_mask
       pixel_source_id
       review composite (not authority)
```

### 4.1 SAM3.1 锚帧规则

每个角色单独选择锚帧，不允许用一只手的锚帧初始化另一只手。

候选帧得分只由冻结输入计算：

```text
anchor_score =
  0.30 * distance_from_image_edge
  + 0.25 * visible_keypoint_ratio
  + 0.20 * mask_seed_separation_from_task_object
  + 0.15 * temporal_sharpness
  + 0.10 * role_identity_consistency
```

剔除：

- 完全离屏或严重运动模糊。
- 左右手交叉且身份不可判。
- 手/Controller 与任务物体几乎完全重叠、无法给负提示。
- 锚点位于其他角色内部。

离线 QA 使用全片最佳锚帧并双向传播；因果模式在 `0..t` 内选最佳锚帧，发生离屏重入或质量代理连续 3 帧失败时重新播种。

### 4.2 SAM3.1 提示合同

裸手 exact78：

```text
文本：left/right human hand and forearm
正点：HaWoR wrist、MCP、有效指节、前臂轴
框：由有效 joints 和已有 role proposal 构造
负点：另一只手、任务物体、Tracker
```

手套新线：

```text
文本：left/right white glove and forearm
正点：MANUS+Controller 投影锚
Controller：由6DoF投影框/点独立分割，不依赖文本单独成功
黄色线缆/附件：独立 accessory role
任务物体：独立实例层
```

当前手套文本提示探针表现不完整，所以本实验不以“文本里写了 glove/controller”为充分条件。只有文本 + 几何正点/框 + 角色负点 + 重入重播种共同通过，才能进入 Clean。

### 4.3 自适应边界，不再固定大膨胀

冻结首轮参数网格：

| 区域 | human/forearm | Controller/accessory |
|---|---:|---:|
| 距可见物体 ≤20 px | 0 / 2 / 4 px | 2 / 4 / 8 px |
| 距可见物体 >20 px | 6 / 8 px | 12 / 20 px |

规则：

- 可见 task-object mask 从 removal 中 byte-exact 排除。
- 对 Poker，牌面和牌边分别审核；不能因为牌薄就把正反面 union 成背景。
- 对 Chips，三个物理实例分别保护，不能 union 后传播身份。
- 没有合法 object mask 的接触帧，不用“大膨胀 + 补洞”掩盖问题，直接将接触窄带标 `UNKNOWN`。

### 4.4 像素来源合同

每个生成像素必须属于以下固定枚举之一：

```text
RAW_CURRENT_UNCHANGED
RAW_CURRENT_VISIBLE_OBJECT
RAW_CAUSAL_TEMPORAL_BACKGROUND
RAW_CAUSAL_STEREO_BACKGROUND
CAUSAL_OBJECT_ATLAS
PROPAINTER_RESIDUAL
UNKNOWN
```

约束：

- 背景 donor 不得跨越 task-object、盘子/支撑面或其他语义层。
- 物体隐藏纹理只能来自同一物理实例的当前可见像素或通过 pose/flow 验证的因果 atlas。
- ProPainter 只填 residual hole；其像素标成生成像素，不冒充真实背景或真实物体。
- `UNKNOWN` 对应 `training_valid_mask=0`，不能为追求画面完整而偷用桌面/盘子像素。
- Clean 结果永远不能反喂 Depth、Object6D、Contact 或真实控制监督。

## 5. 冻结 A/B 实验

### 5.1 会话

| 角色 | 会话 | 主要问题 |
|---|---|---|
| 失败 canary | `play_cards_0903_245` | 手牌接触时物体被 Clean 影响、隐藏牌面来源不足 |
| 失败 canary | `get_potato_chips_0902_039` | donor 复制盘子/背景，边界过度擦除 |
| 旧 A/B regression | `play_cards_0903_243` | Poker 完整时序和重入不退化 |
| 旧 A/B regression | `get_potato_chips_0902_023` | Chips 已有较好 Robot 候选链不退化 |
| 新线压力测试 | `play_cards_0910_053` | 白手套、Controller、黄色附件；只判断路线可行性 |

新线压力测试不参与 exact78 successor 晋升分母。

### 5.2 对照变量

| Variant | Mask | 边界 | donor/layer | 补洞 | 用途 |
|---|---|---|---|---|---|
| A | 当前 frozen role/object mask | 当前固定扩大 | same-pixel donor | ProPainter | 冻结基线 |
| B | SAM3.1 最佳锚 + 左右独立 + 离线双向 | 当前固定扩大 | 与 A 相同 | ProPainter | 只测 mask 编排 |
| C | 与 B 相同 | 自适应、物体排除 | 与 A 相同 | ProPainter | 再测边界策略 |
| D | SAM3.1 因果重播种 | 自适应 | 背景/物体分层、语义 donor、UNKNOWN | ProPainter residual | 完整 successor 候选 |

不运行“退回 SAM2 + E2FGVI”作为晋升候选。若未来希望科研比较，可另开完全隔离的 reference variant，不能成为 D 失败时的静默 fallback。

### 5.3 帧集合

每个标准会话首轮冻结 24 帧：

```text
6 uniform
6 fingertip/object contact
4 arm/image-edge entry or exit
4 occlusion/re-entry
4 known difficult or donor-contamination frames
```

必须先写 `FROZEN_FRAMESET.json` 再运行 B/C/D，禁止看结果后换成容易帧。新线选择 12 帧，覆盖双手、Controller、线缆和物体。

## 6. 指标与 Go/No-Go

### 6.1 可自动计算

| 指标 | 首轮门 |
|---|---:|
| 可见任务物体 protected pixel retention | ≥99.9% |
| 接触窄带新增擦除面积相对 A | 至少减少 50% |
| authorized band 外变化 | byte-exact，0 个像素 |
| 左/右角色交叉重叠 | 审计帧为 0，真实接触允许 `TIE_UNKNOWN` |
| 重入恢复延迟 | ≤2 帧 |
| source map coverage | 100% |
| 盘子/支撑面跨语义 donor | 冻结审核帧为 0 |
| current-visible object 来源 | 必须为 `RAW_CURRENT_VISIBLE_OBJECT` |
| full decode / frame count / fps | 精确一致 |
| causal leak mutation test | 修改未来帧后，当前帧输出 SHA 不变 |

### 6.2 必须人工看 24 帧和视频

- 手、前臂、Controller 和附件是否有残留。
- 指尖与牌/薯片接触轮廓是否被吃掉。
- 牌边、薯片边、盘子边是否被错误重建。
- 左右手交叉和离屏重入是否串身份。
- ProPainter 区域是否出现拖影、复制物和时序闪烁。
- `UNKNOWN` 棋盘是否诚实覆盖无合法外观来源的区域。

没有冻结人工 gold mask 时，不报告 segmentation accuracy；只能报告内部覆盖、保留和时序代理。

### 6.3 晋升规则

```text
T0: 2个失败canary的24帧 B/C/D
  ↓ 两个都优于A且无新硬错误
T1: 2个旧A/B regression的24帧
  ↓ 无退化
T2: Poker245 + Chips039完整全片
  ↓ 全解码、来源门、人工复核通过
R7_CLEAN_SAM31_LAYERED_CANDIDATE
```

候选仍不是 authority。只有治理 aggregator 发布独立 receipt、输入/代码/权重 SHA 闭包并明确 downstream scope 后才可晋升。

任何一项发生时有限终止：

- SAM3.1 权重/代码签名不一致：`BLOCKED_REFERENCE_PROOF`。
- GPU lease 不可获得：`BLOCKED_RESOURCE`，不回退 SAM2。
- 可见物体 retention 不达标或复制盘子：`FAILED_QUALITY_C`。
- regression 退化：冻结候选，不扩批。
- 物体隐藏区没有合法来源：标 `UNKNOWN`，不判成算法崩溃。

## 7. 运行资源与当前就绪度

已核实本地固定资产：

| 资产 | SHA256 | 状态 |
|---|---|---|
| `assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt` | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` | READY |
| `src/chaoyang/pipeline/sam31_compat_adapter_v1.py` | `5c6513c0bdd8fc29bf9248c8ca007b492e0d3d3051baf16f5c1cf870d7c48be2` | READY |
| `vendor/ProPainter/weights/ProPainter.pth` | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` | READY |
| `vendor/ProPainter/weights/raft-things.pth` | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` | READY |

许可证边界：项目内 ProPainter 上游声明为非商业用途许可，实验和任何后续分发/商用前必须继续保留并复核许可证；本实验不复制 Masquerade 源码，只记录算法编排与固定链接。

GPU canary 必须通过中央 lease 执行。当前文档完成不等于 GPU canary 已执行，不能将 `EXPERIMENT_DEFINED` 写成 `PASSED_CLEAN_SUCCESSOR`。

## 8. 预计输出

每个 canary 至少生成：

```text
MASK_LAYERS.npz
REMOVAL_SUPPORT.npz
PIXEL_SOURCE_ID.npz
UNKNOWN_MASK.npz
TRAINING_VALID_MASK.npz
CLEAN_BACKGROUND_MASTER.mp4
TASK_OBJECT_LAYER_MASTER.mp4
COMPOSITE_PREVIEW.mp4
中文四栏/六栏复核视频.mp4
RESULT_SUMMARY.json
SOURCE_MAP_MANIFEST.json
```

便于人工查看的复核视频最终另发布到浅层目录：

```text
docs/current/visuals/Clean_SAM31分层路线_Poker245_24帧.mp4
docs/current/visuals/Clean_SAM31分层路线_Chips039_24帧.mp4
docs/current/visuals/Clean_SAM31分层路线_手套0910_053_12帧.mp4
```

在视频实际生成前不得创建冒充成品的占位 MP4。

## 9. 本轮事实边界

当前可以说：

- Masquerade 的锚帧、左右独立和双向时序编排与本项目问题高度相关。
- 项目已有 SAM3.1 视频 predictor 与 ProPainter 资产，可以按升级路线实现，不需退回 SAM2/E2FGVI。
- 新路线已被定义为隔离、可 A/B、可有限终止的实验。

当前不能说：

- SAM3.1 分层 Clean 已优于现有基线。
- 手牌接触遮挡已解决。
- 隐藏物体纹理已经恢复正确。
- 该路线已获得 Clean 或 Robotized training authority。

这些结论必须等冻结 canary 实际运行和人工复核后更新到独立 RESULT，不能靠方案文本推断。
