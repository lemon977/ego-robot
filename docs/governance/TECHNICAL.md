# 技术合同草案说明

> **历史草案，不是当前执行入口。** 当前任务与授权边界见
> [`CURRENT_TASK.md`](CURRENT_TASK.md)；0909 `rgb30_v1` 状态见
> [`CURRENT_STATUS_ZH.md`](../../tasks/control/runs/20260911_handle0909_rgb30_v1/CURRENT_STATUS_ZH.md)。
> 下方2026-08-28内容只用于追溯早期合同形成过程。

状态：`G1_CONTRACT_VALIDATED_GOVERNANCE_HOLD / ROUTE_A_D1_FULL_002_012_RESULT_INTEGRITY_PASS_VISUAL_HOLD / ROUTE_A_D2_TASK32_P0_3_STOPPED / ROUTE_A_COMBINED_SKIPPED / ROUTE_B_POINT_SCOPE_CPU_QA_PASS_GPU_HOLD / ORACLE_CLEAN_TABLE_REPROJECTION_PENDING_ATLAS_SKIPPED / EEVEE_FULL_ASSET_GPU_THROUGHPUT_PASS_VISUAL_ONLY / NULL_CALIBRATION_N2_HOLD / ROBOT_CHAIN_MOUNT_HOLD / B3_BLIND_HOLD / MASK_FORMAL_CLEAN_TRAINING_PROMOTION_HOLD`

## 2026-08-28 23:38 技术状态增量

本增量覆盖下方较早的 `9/24` 与 `pending/running` 状态描述，但不改写其历史证据。
完整短入口为 `task/CURRENT_EXECUTION_STATUS_20260828.md`，总审计 SHA-256 为
`5d18b73b8722f66b8115460a316f88c8f5e9c549e5563e2a162a041953fe3d03`。

1. task29 在同一 raw inventory 上完成 old-v3/task26 双 selector 的 002+012 全757帧配对。
   D1 structural ACCEPT 最终 left=`542/757`、right=`757/757`，差=`28.40pp`；
   16,912条 ordered candidate-audit 闭合且 result integrity QA P0=0。
2. selector ACCEPT 不是 H-completeness。独立视觉 QA 仍见系统性 wearable gap、
   `012` 左侧215空HOLD、31个accepted hand-only/non-boundary和38次IoU=0/状态翻转，
   因此状态为 `HOLD_VISUAL_H_SCOPE_INCOMPLETE_NO_PROMOTION`。
3. task32 V3 在运行前因三项治理 P0 停止：真实P1 admission合同不兼容、ffmpeg symlink
   覆写面、缺success/failure terminal与post-child release。四条wearable prompt均未执行；
   条件task33跳过。task34等P4结果仅为只读diagnostic，不是gate。
4. Route B task23 V4 的pair-alias与typed HaWoR source-state CPU门已独立复核P0/P1=0；
   但production authority pin、绑定新输入的显式owner GPU release与future input-bound
   独立re-QA（P0/P1=0）均未闭合。补pin本身不解锁GPU，CPU PASS不得写成Route B/MASK PASS。

所有正式 MASK/CLEAN/RobotRGB、blind、training、promotion 与 bucket unlock 继续关闭。

本文件描述 G1 正在固化的语义，不是启动 producer 的授权。v9 已由独立 QA 判 HOLD；启发式路线止于 v9，v10 不授权。用户已授权并完成 T2-B0a/B0b：benchmark freeze SHA256 `09f94084…073b` 绑定 45 palette、180 个新 `{0,255}` mask 与 45/45 反向一致性；原 T0 `{0,1}` 快照不变。`auditor_a1` freeze SHA256 `d2cd400a…fd9c` 固定 `HUMAN_REVIEW_POLICY`，无自动 PASS/晋级。B1 在 0/8 时由用户免除，经验 noise floor 不可用。SAM3.1 adapter 仍仅 compatibility PASS；P2/P3 development-15 均为独立 QA 确认的 0/15 strict HOLD。B3/blind、CLEAN、formal production 和训练继续关闭。

## 1. 单一所有权

| 组件 | 唯一输入 | 唯一正式输出 | 不得做 |
|---|---|---|---|
| Source resolver | 只读 RAW、source manifest | 已验证 frame references | 写 RAW、用副本顶替 |
| Session profiler | 已验证 RAW/metadata | `session_context.json`、canary list | 生成正式像素 |
| MASK benchmark curator | RAW、自动分层选择、人工 H/O/U/B 标注 | 冻结 benchmark | 调 producer/auditor、泄露 blind 标签 |
| H/O/U/B label freezer | 已审 palette PNG、source/benchmark manifest | 四类互斥二值 mask、label manifest、freeze ref | 改人工像素、推断缺失标签、读取候选 |
| MASK | RAW、上下文 | `H_core`、arm/forearm evidence | 修物体、写 CLEAN |
| Object evidence | RAW、Object6D/CAD/解析几何 | `O_visible_core`、geometry/depth/SDF | 删手、改 q_hand |
| Reveal router | H/O/U、几何、donor coverage | reveal labels 与 route decision | 直接补图 |
| CLEAN | RAW、reveal labels、donor | 背景/物体 reveal、coverage/confidence | 渲染机器人 |
| Retarget | 冻结 human motion、robot model | 版本化 q_hand/wrist/validity | 写图像、render-time 改 q |
| Base/IK | wrist、robot model、场景几何 | session-constant base、q_arm、adapter | 移动冻结 wrist |
| Renderer | 冻结几何/pose/camera | robot/object RGBA、per-link ID/depth | 修 CLEAN、隐藏失败 link |
| Compositor | CLEAN、render layers、depth | composite 与诊断图 | 改上游 geometry/q |
| Harmonizer | 几何 PASS composite | 仅 robot RGB/窄边界外观 | 改 alpha/depth/object/background |
| MASK auditor | source、候选、冻结 H/O/U/B benchmark | `mask_independent_qa` | 改 MASK、跨 auditor 排名、覆盖不足时 PASS |
| QA | 全部 manifest/证据 | 归因、建议、PASS/HOLD | 覆盖任何 producer 输出 |

每个正式 artifact type 在 `pipeline_contract_v1.yaml` 中只能有一个 producer 和一个 schema。并行实验只写 `_run/<run_id>/`，不能成为正式 consumer 的候选路径。

MASK producer 和 auditor 必须分别带版本身份。`mask_evidence` 绑定 `producer_pN`、像素语义 SHA 与实现 SHA；`mask_independent_qa` 绑定冻结 `auditor_a1`、阈值/profile SHA、benchmark SHA 与实际覆盖。v3（460 帧）、v8（4 帧）、v9（15 帧）覆盖和历史 auditor 不同，历史数字只能说明各自证据，不能排序。

## 1.1 MASK 冻结比较协议

人工 benchmark 使用互斥且穷尽的 `H / O / U / B` 标签，共 45–60 帧，分为 004 development、004 same-session blind、cross-session blind 三组，各 15–20 帧。U 是接触、运动模糊或混合像素不确定带，硬边界指标排除 U。当前 v2 manifest SHA256 为 `60b9df265169bb2a4d575bfc3a836039c9ba0d618922a6f8b08dbf7b44b9fde1`，固定为 004 development 15、004 same-session blind 15、025 cross-session blind 15；v1 中 149 的 15 帧不迁入 v2 并保留为 final blind。228–232 的 pilot review SHA256 为 `9d218c494dd6bdc67a8249c74ea0364c746247a607dfb4463dce02a383d50960`，五张迁移标签在 UI 中按 SHA 锁定。

标注 UI 的规范输出是原尺寸 P-mode palette PNG，像素索引严格为 `B=0/H=1/O=2/U=3`；颜色只用于显示。T0 freezer 曾展开 45×4 个值域 `{0,1}` 的快照（manifest SHA256 `3038770f…01d33`），但 auditor 合同只接受 lossless `{0,255}`，因此没有直接冻结旧 PNG。T2-B0a 在新不可覆盖 namespace 生成 180 个 `{0,255}` mask，验证互斥、穷尽和 45/45 reverse palette bit-exact；freeze SHA256 `09f94084a84cd0ac0d3e93a3311da2b41c85788742029e89347b18afb6ed073b`。旧 T0 快照保持不变且 `consumer_allowed=false`。

冻结 `auditor_a1` 统一计算：human miss、object corruption、background over-mask、boundary excluding U、temporal flip。实现 SHA256 为 `82d21b01a623093d2c89308ba7d1dd596f6d062d6273b8044db5df12356e9c7c`；B2 的 H/B core erosion 与 Boundary IoU 1/3/5 px 仍只作 diagnostic，零分母显式为 null。T2-B0b 已把实现、公式、schema、runtime、benchmark freeze 与 `HUMAN_REVIEW_POLICY` 绑定到 freeze SHA256 `d2cd400a5ffb7e30f365f1ae3af21fb916b31302e64034ca8ba75c36c001fd9c`。经验 noise floor 和 automatic thresholds 均为 null；任何候选必须等待人工 review。只有另行授权 B3 后，同一 benchmark/auditor identity、三分区完整覆盖的候选才可诊断排名。几何、PCA、Object6D 与时序只可寻找提示或拒绝结果；A2 段③唯一允许的例外是合同显式的 object ownership subtract，仍不得新增人体像素。

B1 已免除，因此阈值不能声称来自实测标注者一致性。用户已选择并冻结 `HUMAN_REVIEW_POLICY`；`CONSERVATIVE_POLICY` 未采用。这不自动授权 B3、不解封 blind，也不授权 full460/CLEAN/训练。

T1 `producer_p1` r3 不是正式 `mask_evidence`：它虽证明 60/60 输出始终是当前帧 SAM 候选子集、段④非接触区 bit-exact、物体保护 overlap=0，但 frame 200/201 吞入大面积桌面，且 run manifest 对 `contracts/mask_evidence.schema.json` 有 614 errors。r3 只能法证封存，不得被 A3、CLEAN、review packager 或 selector 消费。

`004_old/` 的 5 个文件经最终只读审计可稳定解码，但只有 legacy 视觉参考资格：MASK overlay 可辅助理解 ownership，CLEAN/机器人视频可列举失败类别，两张 retarget 图可回顾历史接口。它们没有正式 manifest/producer/逐层深度血缘，formal/training 输入资格为 0，禁止用作阈值 GT、donor 或 fallback。

人工标注工具 `tools/houb_annotation_app.py` 不是 producer。2026-08-27 已修复 RGB 差值 `int16` 平方溢出、Gradio 半透明画笔预乘 alpha/重复混色误拒绝，并增加已审核迁移标签 SHA 锁定。错误输出移入隔离区，不进入 active labels；UI、v1/v2 validator 与 v2 集成门当前 24 项测试通过。正式标签仍必须经过独立 freezer/validator，不能因为 UI 保存成功就自动 PASS。

## 2. 数据流

```text
RAW -> profiler -> session_context + canaries
RAW -> MASK -------------------- H_core
RAW + Object6D/CAD ------------ O_visible_core + geometry + SDF
H_core + O_visible_core ------ U_contact + reveal labels
reveal(object) -> geometry/texture donor ----+
reveal(background) -> temporal donor/inpaint +--> CLEAN
R2 or CONTACT_GOLD sidecar -> retarget/base/IK -> renderer
CLEAN + render depth/ID -> compositor -> geometry gate -> harmonizer
-> QA -> _run evidence -> atomic promotion -> processed/<session>_vN
```

`EXACT78_R2_VISUAL_DOMAIN` 中 retarget 只读原 R2；`004_CONTACT_GOLD` 使用独立 sidecar namespace。两者不能互相读取输出。

## 3. Session context 与归一化

context 至少包含：分辨率、真实 FPS、hand/wrist/forearm 尺度、object bbox/投影尺度、motion/blur、contact ratio、Object6D validity、donor coverage、左右手可见性与 source hashes。

项目 profile 只冻结公式、系数与上下界，例如：

```text
dilation_px = clamp(round(k_dilate * hand_scale_px), image-relative bounds)
contact_band_px = clamp(round(k_contact * object_scale_px), image-relative bounds)
forearm_width_px = k_forearm * measured_wrist_width_px
temporal_window_frames = round(window_seconds * measured_fps)
```

禁止固定像素常数跨分辨率传播、逐帧手调、session ID 分支和隐式缺失值。必要证据缺失必须 HOLD。

## 4. Reveal 与 CLEAN 语义

当前合同锁定的生产实现仍为 SAM2.1。SAM3.1 非官方镜像已完成隔离下载和 strict load；独立 `sam31_compat_adapter_v1` 在不修改官方源码、不 monkeypatch 的前提下，通过固定签名映射解决 wrapper API mismatch。其独立 QA 仍仅为 `PASS_COMPATIBILITY_EVIDENCE_WITH_CLAIM_LIMIT`。在 004:228–242 上，P2 text route 为 0/15 sufficient，P3 official box-only route 为 0/30 eligible、0/15 sufficient 且 final MASK 全空；两条 HOLD 都是完整袖臂证据不足，不是 formal MASK 或 PASS。H/O/U 证据语义为：

详细 prompt probe 随后改变了“所有文本提示均不足”的旧结论：`an arm` 在三帧中稳定给出左右 ego 的分离 raw SAM instance，而 `a hand`、`a forearm`、`a sleeve` 与腕部佩戴物短语都不完整。A′ 只允许从 raw instances 选择，不能改像素。跨session为2/24。6帧raw=0且HaWoR种子在画外，但RAW目视均清楚存在双侧ego手臂，因此不得记为`NO_TARGET_IN_FRAME`，而应记为`VISIBLE_TARGET_AUTHORITY_EVIDENCE_FAILURE`；分母保持24。剩余失败仍呈强左右不对称，11帧属于selector拒绝已召回实例。后续必须使用left/right独立身份和判定，三个既有阈值不变；左右通过率差>10个百分点一律不得晋级。

Route A 与 Route B 的像素链必须分开标注 `route_of_evidence`。Route A 为 text/concept→instance recall→ego selector→MASK；Route B 为 HaWoR point prompt→SAM2.1 mask decoder→MASK，不含文本召回或 selector。A′ 的 8 个 boundary 误拒、3 个 side alias 和 2/24 跨 session 结果只能用于 Route A，不得推断 Route B。

修订 QA 的逐侧结果是 Route A left=`3/24`、right=`16/24`、绝对差=`54.17pp`；005:370 是 source-slot identity exchange，不是物理左成功/右失败。证据见 `audits/INDEPENDENT_QA_SAM31_AN_ARM_APRIME_CROSS_SESSION_V2.md` 与 `_run/rejected_visual_review_v5/`。

路线 B 的模型选择由原 trainability decision rule 完成：官方 SAM3 point/refinement 公开便利入口处于 `torch.inference_mode()`，最终将 logit 阈值化为 bool，调用链不能提供可微 logit，因此规则触发 SAM2.1。该结论只覆盖 point API，不是“SAM3.1不可训练”。官方 SAM3.1 concept/text 训练栈存在但属于 Route A；只有 Route A 完成 per-side identity 修复并重测后仍由召回不足主导才允许触发。监督路线的 U 像素现采用方向性损失：预测 H 不罚、预测背景罚；`weight=0` 与把 U 并入 H 均为合同违例。

Route B 的 HaWoR point prompt 必须先具有独立 left/right identity、lineage、quality 与单侧越界判定；Route A 的 selector 修复不能充当这道门。另必须单独验证单点能否生成完整 `hand + wrist/cuff + sleeve + forearm to image boundary`。范围系统性偏小的错误码为 `SYSTEMATIC_SCOPE_TOO_SMALL`，不得用左右通过率或总覆盖率掩盖。Route-A per-side CPU V3独立QA为`PASS_CPU_ADMISSION_EXACT`、P0=0；冻结GPU候选随后得到004 `15/15`、自由session `9/24`，左右差`8.33pp`通过而泛化差`62.5pp`失败，最终严格HOLD。独立像素重放确认39帧/62个通过侧直接引用唯一raw instance，禁止像素操作均为0。该结果现仅是task18历史panel基线；task25已证明其召回标签大范围错用，当前full-session结论由本文件顶部增量覆盖。它始终不是Route B/MASK准入。NULL calibration N=2 的磁盘 blocker 已解除，但仍因 selector、only-seed config 等价性及025角色冲突HOLD，未启动GPU训练。

Oracle plumbing 使用004 development人工H/U，只验证下游I/O并与正式CLEAN隔离。v1 15帧四个硬门为0但可靠填充为0；H/U residual分开记账。CLEAN v1不建设历史贡献仅0.12%的object atlas，该区保持品红unresolved；唯一恢复主线是历史贡献94.78%的table reprojection。AprilTag局部table support已15/15 LOO PASS；罐底接触15/15仍UNRESOLVED，冻结c2w baseline仅4.927mm且强制plane后的XY scatter为107.784mm，故正式重锚关闭。每帧物理门仍须算腕部到桌面的有符号距离、手在桌下与指尖进圆柱，confidence不得替代物理合规。

EEVEE-Next headless通过项目内GLVND NVIDIA ICD加载H20 OpenGL后，完整名义资产复杂度v5_r3六项矩阵全部出图：每个worker加载3份URDF、63/63 pinned visual mesh。240×320最佳两路`0.493580 s/frame`，1280×960最佳三路`0.684467 s/frame`；32,318帧外推分别为`4.4310 h`与`6.1446 h`。192/192 PNG逐文件复核，无software GL/CPU/Cycles fallback。该场景仍为`VISUAL_ONLY_NOMINAL_ASSET_COMPLEXITY`且`mount_applied=false`，不提供q_arm、正式pose/depth或RobotRGB授权。

- `H_core`：高置信真人手/手腕/前臂。
- `O_visible_core`：当前可见物体核心。
- `U_contact`：接触/遮挡不确定带，不直接决定 donor。

删手后应露出物体的像素必须由 Object6D、CAD、解析几何、跨帧物体纹理 donor 支持；应露出背景的像素才允许真实时序 donor、atlas 或 ProPainter。无法确定 reveal 身份或 donor 覆盖不足时 `HOLD_UNSUPPORTED`，不得用背景纹理覆盖物体。

## 5. 接触、retarget 与几何

Object6D 输出需包含 pose、validity/confidence、metric 或有明确尺度的 geometry；004 gold refinement 另存版本并保留原始证据。物体 SDF 与完整 KaiHand link collision geometry用于接触距离、法向一致性、非穿透和逐指遮挡。

004 每轮只允许修改一个分支：

- `OBJECT_GEOMETRY`：pose/scale/geometry/SDF 或物体纹理重建证据错误。
- `RETARGET_QHAND`：接触目标、关节姿态、时序或非穿透错误。
- `COMPOSITOR`：geometry/q 正确，但 depth/alpha/边界合成错误。

exact78 只忠实消费 R2，不运行 refinement 或 render-time IK。base search 只能寻找一个 session-constant Tianji base，并在固定 wrist 下求 q_arm；不可达即 HOLD。只读 preflight 已确认 Tianji 双 7-DOF 关节/限位、KaiHand joint identity、R2 wrist/q_hand 和 RAW K 可消费；`q_arm + session-constant base` 可在固定装配后统一求解。当前唯一不可推断输入是左右 `Tianji tool → KaiHand root` 固定装配变换：它必须来自 pin 的 CAD/测量/明确批准 visual proxy，不能用 identity、零平移、逐帧 offset 或低 IK residual 猜测。

## 6. 渲染与合成

Renderer 必须输出完整机器人 RGBA、物体层、逐 link ID/depth、object depth、camera/pose digest 与完整 link visibility。Thumb-only 只能是诊断图，不能作为最终合成输入。

Compositor 使用 depth 决定 object/robot 的前后关系，不能靠 link 名单或单帧阈值隐藏穿模。机器人未覆盖区域必须来自已通过 reveal 门的 CLEAN。

Harmonizer 只在几何门通过后运行，允许受限曝光、白平衡、噪声、锐度、阴影和窄边缘融合；其 mask 及变更范围必须可审计。

## 7. Canary、路由与质量门

每条 session 自动选择 12–24 帧，覆盖低/高 arm area、接触前/首次/稳定/释放、最大遮挡、高 motion/blur、Object6D 无效段和 donor 稀疏段。固定统计进入标准 route registry，输出三态：

```text
PASS_FULL_SESSION
RETRY_STANDARD_ROUTE
HOLD_UNSUPPORTED
```

标准重试只能切换 profile 中预先登记的有限路由，不能创建 session-specific route。

质量分 L0–L4：数据完整性、像素/区域、几何/遮挡、时序/session、人工/可选语义。任何更高层处理都不能覆盖较低层失败。004 stop gate 由全部层共同判定。

## 8. 失败归因最小字段

`failure_attribution.json` 至少包含：schema version、session/frame、product line、diagnostic track、failed gate、single owner、evidence refs/SHA、observed/threshold、recommended standard route、claim limit、created_at。owner 必须取自有限集合：

```text
RAW_SOURCE / SESSION_CONTEXT / MASK / OBJECT_GEOMETRY /
CLEAN_BACKGROUND / RETARGET_QHAND / BASE_IK / RENDERER /
COMPOSITOR / HARMONIZER / MANIFEST
```

同一轮只能有一个 primary owner。跨层症状可列 secondary evidence，但不得自动修改多个 producer。

所有 `REJECT/HOLD` 必须由单一 registry/indexer 集中登记，记录失败阶段/帧、RAW ref/SHA、候选与可视化 ref/SHA、QA ref/SHA、单一 owner 和人工 review 状态。v1 registry 为 `_run/rejected_visual_review_v1/MANIFEST.json`（SHA256 `9ae29338…abcf`），HTML index SHA256 `edd14102…888`，含 13 项 MASK v1–v9/A2 代表性证据。可视化是 review-only 入口而非正式数据源；人工 review 不能改像素、失败归因、晋级状态或 T2 授权。未来失败必须写入新的不可变 pack 版本，不得覆盖 v1。

P2 strict HOLD 使用独立 v2 registry：`_run/rejected_visual_review_v2/MANIFEST.json` SHA256 `e0556a84…c88f`、`INDEX.html` SHA256 `92ca7196…065d`，5 项覆盖代表帧与完整 15 帧 review 视频。v2 不覆盖 v1，也不使 P2/P3、B3、full460 或 CLEAN 获得 formal consumer 资格。

P3 box-only strict HOLD 使用独立 v3 registry：`_run/rejected_visual_review_v3/MANIFEST.json` SHA256 `352af39f…f29a`、`INDEX.html` SHA256 `423a3758…630`，4 项覆盖 228/235/242 和完整 development-15 视频。v3 不覆盖 v1/v2，也不改变 P3 的 0/15 HOLD 或任何 formal consumer 资格。

## 9. 公平训练/评测

RAW 与 Robot 主对照必须绑定同一 `PAIRED_KEPT_MANIFEST.json`、split、R2、非图像输入、初始化、seed、batch、epoch、augmentation、validation 选择规则和 checkpoint 选择规则。test8 一次性消费冻结流水线。

速度对齐若执行，必须双域采用相同 frame/window subsampling 与权重，参数只由 train62 和目标机器人统计决定，并单独报告，不替代主对照。

## 10. 当前实现能力与不可替代的 HOLD

- 合同 validator 为 22/22 PASS，但 `pipeline_contract_v1.yaml` 与 profile 仍是 DRAFT、formal production=false。
- 本机唯一合同注册的 MASK backbone 仍是 `SAM2_1_HIERA_LARGE`。image predictor 与 video predictor 共用同一权重，不能冒充 backbone A/B；SAM3.1 的模型、strict load 和显式 adapter 技术 smoke 已通过，但 P2 text 与 P3 box-only 在 development-15 上均为 0/15 strict HOLD，因此只能保留为隔离 T1 法证路线，不能注册 production。
- CLEAN 的 policy/donor/atlas/review schema 与 synthetic fail-closed scaffold 已存在，但没有真实 object texture donor、temporal atlas manifest 或正式 CLEAN 授权。
- robot-chain proxy 的41.000068mm、-101.717561mm、48.713401/7.308745mm与145mm均复算一致，但43.282439mm只是flange→rear-plane gap，不是tool→hand-base平移；完整左右4×4 mount仍不唯一，所以 q_arm/base T1、正式 renderer 和 RobotRGB 仍未启动。
- HumanEgo 已补严格 manifest、source SHA、resume/config 与原子冻结 guard；正式 RAW/Robot paired selector/cohort 尚未实现，因此训练、评测与 freeze 保持 HOLD。
- 三份 checkpoint 与两个 bundle 的 digest 谱系完整。pretrained 只可作共同初始化候选，旧 Kai22 只可法证，R2 checkpoint 只可数值 control；三者都不能直接充当未来 matched RAW/Robot 训练臂。
