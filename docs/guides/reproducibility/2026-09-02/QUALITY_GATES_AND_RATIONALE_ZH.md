# 质量门与理由（2026-09-02）

## 总原则

一个产物只有在其输入、算法、语义、时序、输出完整性和 claim scope 同时通过时才是 PASS。数值连续、视频好看或文件存在都不能替代上游权威。缺输入、版本漂移、许可证不明、非盲数据或局部窗口只能得到对应范围的 HOLD/PASS，不能向外扩张。

## 共用门

| 门 | PASS 条件 | HOLD/FAIL 条件 | 理由 |
|---|---|---|---|
| G-ID 身份 | 绝对路径、bytes、SHA-256、角色、帧数/旋转冻结 | 任一缺失或漂移 | 防止“同名不同数据” |
| G-DECODE | 全文件解码；尺寸、帧数、像素格式符合合同 | 截断、只验 magic bytes、帧数不符 | 文件存在不代表内容可消费 |
| G-SCOPE | formal/dev、blind/nonblind、full/window 明确 | scope 缺失或被扩大 | 局部证据不能证明全局 |
| G-FRESH | 新输出根，禁止覆盖历史证据 | 输出根已存在 | 保留失败谱系和可恢复性 |
| G-INTEGRITY | 输出 bytes/SHA、机器 JSON、完整解码均通过 | digest/解析/解码失败 | 让视觉与机器消费同源 |
| G-LICENSE | 依赖和资产权利可追溯 | 许可证/再分发权未记录 | 技术可复现不等于可发布 |

## Mask 门

1. 输入门：冻结 source、display rotation、runner、adapter、official code commit 和 checkpoint。
2. 初帧门：prompt sweep 只用于 frame 0；候选满足预注册 anchor、面积、bbox 与窗口。
3. Raw 门：只保存同一 SAM3.1 object ID 的未经修改实例。空缺帧必须空并 HOLD。
4. 覆盖门：静态全帧 present + spatial gate；动作全帧 present、12/12 canary present 且无预注册 gross drift。
5. 语义门：object core、hand skin、wristband、tracker 分点审核。高 temporal IoU 只证明实例稳定，不证明语义闭包。
6. Object 保护门：object/human 独立运行和存储；接触 overlap 仅作诊断，不能 union/subtract 或反写 object。
7. 像素门：禁止颜色阈值、morphology、dilation、fill、corridor 和 union 修补 raw Mask。
8. 产物门：raw 目录精确帧数/尺寸，像素仅 `{0,255}`；review MP4 精确帧数且完整解码。

早期 generic object 四链通过手机开发门；早期 human 两链虽然 360/360、480/480 且 IoU 高，仍因 hand/wearable/tracker 语义失败而 HOLD。这正是语义门必须独立存在的理由。后继 assisted v4 没有改写这批历史 producer，而是建立新的独立 lineage。

### Sleeve / tracker rescue 与训练入口门

- 三类必须独立：`human_core`、`sleeve_cuff`、`wrist_wearable` 各自保留 raw object ID；class presence 不能替代目标腕 identity。poker tracker 虽 12/12 有类实例，但 raw ID 在双腕间切换，仍 HOLD。
- 12-canary 任一 frame 或 frozen anchor 失败就禁止 full propagation/union。现状 chips 为 human 11/12、sleeve 0/12、wearable 11/12；poker 为 11/12、最好 5/12、class-only 12/12。
- 禁止用颜色、morphology、dilation、fill、corridor、crop tuning、session-specific flow 补 raw semantic mask；现有 rescue 的 forbidden repairs 为空，object mask read/modified=0。
- 数据 firewall：002/139 只做 frozen nonblind smoke；032/051/sealed 004 reads 必须为 0，不能参与 prompt/model 选择。
- 九类 annotator tool ready 不等于模型 ready。正式训练前必须按 whole session 冻结 split，导出并人工复核多会话标签，绑定 production trainer，并为 base/adapted checkpoint、代码、环境、seed、资源峰值和 blind report 建 SHA authority。
- 只有左右 skin/sleeve/tracker 各类和 side gate 都通过后才允许形成 human union；union 永远不能覆盖独立 card/chip object raw mask。

### Assisted bilateral flow-refresh v4

- 身份门：画面中 upper/lower 两个 `human_core` 和每个可见
  `tracker_wearable/sleeve_cuff` 必须是独立 raw role；12 canary 全 role present、
  upper-before-lower 且所有 flow/refresh/human-hold 门通过。
- Human 门：SAM text raw 保持 `[0.25,4.0]×frame0`。越界时仅允许双向 DIS 的
  inverse-warp 与 forward-splat RAW intersection；相邻面积 `[0.70,1.30]`、cycle
  p90 ≤4 half-resolution px、boundary gradient ≥1，仍不得放宽 4×。
- Wearable 门：frame 0 以 assisted 点注册；每 20 帧自动 refresh，IoU ≥0.10、
  centroid distance ≤36 px、area `[0.30,3.0]×warp`、same-side min distance ≤18 px、
  opposing centroid margin ≥4 px、object raw overlap=0；refresh 接受率必须 100%。
- Priming 门：为绕过已复现的 first-object runtime quirk，每个独立 wearable state
  固定先创建 task-object priming raw ID，并要求 `priming_output_consumed=false`。
- 发布门：只有全门 PASS 才能写 full masks；binary union 必须逐帧严格等于所有
  role raw OR 再扣 protected object。所有 PNG 只能是 `{0,255}`、540×960、完整帧数；
  review MP4 必须完整解码。

chips v4 通过：四 role 360/360 非空，12 canary PASS，双 tracker 各 17/17
refresh 与 359/359 flow，union 等式累计 mismatch=0，published object overlap=0。
poker v4 保持 HOLD：human 双 PASS、lower cuff 23/23 PASS；upper tracker 17/23、
lower tracker 0/23 且 224 flow adjacency failures。角色在 12 张图中“看得到”不能
替代全时序 refresh/identity 门，因此没有发布 poker union。

## Clean 门

### L2 源准备

- 六个输入均冻结并完整解码。
- ORB+RANSAC 只用于注册/漂移诊断，不提升为相机位姿权威。
- 固定机位 donor 必须同 session、同相机、同方向、同曝光/焦点、同家具/对象状态。
- 移动机位必须有逐帧 K、camera-to-world、静态米制桌面和平行的同 session empty sweep。

### L3 同 MOV donor

- target 与 donor 的源 SHA 必须相同，跨视频 donor 次数必须为 0。
- donor 像素必须是真实观测，且 donor human mask 为 0。
- object、木架、碗等 fixture/protected 区不得由平面 donor 擦除。
- 每个替换像素都必须写 `source_map(frame,x,y)`；无合格 donor 的像素写 `unsupported_mask`。
- 输出仅为 `PHONE_DIAGNOSTIC_ONLY`；不能进入正式 PICO 交付。

封存验证覆盖 10 个目标帧；61,202 个替换像素逐字节等于其 source-map 指向的源像素，mismatch=0，跨视频 donor=0，unsupported 中有 source 的像素=0，已知语义漏点被填充数=0。chips registered fill 中位数 77.45%，poker 仅 0.98%，两条仍因 human/wearable 语义未闭包而 HOLD。

理由：相似桌垫和可拟合单应只能说明外观接近，不能证明同 session，也不能恢复非平面物体或被遮挡而未观测的像素。

### Clean visible-removal successor gates

| 门 | 冻结检查 | 失败含义 |
|---|---|---|
| Mask authorization | 只消费顶层 `consumption_authorized=true` 的 root；复核 union inventory、尺寸和 binary values | diagnostic/HOLD mask 不得进入 Clean |
| Real-pixel source | 每个替换像素记录 source video/frame/x/y；same-MOV source mask 必须为 0，距 mask 至少 12 px，PatchMatch/quilting 至少 32 px | donor 可能仍含 human/wearable |
| Generated controls | Telea/NS 必须标 `GENERATED_CLASSICAL`，不得声称 real-source | classical inpaint 不是观测 RGB |
| Support integrity | 窄变体 support 外 byte-identical；whole-table 变体 non-table 区 byte-identical | 禁止隐藏全局 retouch |
| Object protection | bowl、三个 chips、green fixture changed pixels=0 | task semantics 被擦除 |
| Poisson causality | donor-purity PASS 后若手形仍在，禁止 target-interior gradients | solver 可从 target 重新解出手边界/阴影 |
| Whole-table projective safety | 四桌角 target-to-static denominator 同号、`min(abs(denominator))>0.25`、matrix condition `<1e6`；失败直接 HOLD，不回退 affine | 存在 projective pole/radial warp 风险 |
| Independent geometry | green centroid/tip 或外围点必须在两图独立测量，禁止旧 H backprojection | pseudo-correspondence 没有新增几何证据 |
| Frame0 promotion | frame0 不得有 human silhouette、duplicate donor objects、block seam 或不可信 table warp，才可进入 10-frame/full | 数字 fill 或 safety PASS 不能代替目视 Clean |
| Claim boundary | PHONE_DEV/cross-camera static 结果保持 non-formal；unsupported 不叫“已消除” | 防止诊断被升级为正式通用结论 |

最终 bounded frame0 的 affine right-chip residual 为 `85.2412 px`；受约束 H 的四角
denominator 为 `[0.9480,2.0260,1.8862,0.8082]`、condition `170934.21`，六锚点
residual 为 `[11.4510,50.1833,37.1642,53.3942,7.4643,25.7999] px`。它通过
projective safety，但 duplicate-object/warp 目视门失败，所以状态是
`HOLD_CROSS_CAMERA_STATIC_INSUFFICIENT`，且 10 帧/全片均未启动。

## Robot 门

### 数据与初始化

- G0：RGB 可解码、同步 ≤16.7 ms、c2w 有限且为 SE(3)、单 pose segment、TrackerState accurate。
- G1：正式跨会话必须存在声明过的开场 fiducial/中立块。frame-0 人体双腕 SVD fallback 可做诊断，但不能使 G1 PASS。
- 每个 session 的 `T_world_rig` 只能读本段自己的 c2w + 双腕：旧 fallback 用 frame 0，window successor 用预声明的最早 24 个双腕有效帧；均不得复制 004 Tworld/Tcamera。

### 求解与物理代理

- Pose：位置误差 ≤10 mm、旋转误差 ≤5°。
- Limits：精确 pinned URDF 限位。
- Temporal：速度 ≤0.12 rad/frame、加速度 ≤0.06 rad/frame²；004 两窗口实际最大值为 0.055920 与 0.039790。
- Historical window contact proxy：仅当旧 ideal human-root KaiHand 要求 overlap 时，实际必须 overlap 且 whole-hand min-SDF 差 ≤25 mm。它只保留旧 window/cross-session downstream 诊断语义，不是当前 MANO21/KaiHand 指级物理 authority；`overlap_required=0` 的 PASS 更不是抓取。
- Natural bend：肩-肘-腕折线、外向 camera-x 与 elbow in-frame 同时通过。三段现有结果均 HOLD。

跨会话 contact 只能采用后继 audit：002 PASS、068 HOLD（f340/f397，physical side 1）、139 PASS。旧源矩阵的 contact 聚合存在布尔更新缺陷，不能再作为下游权威。

### Correct MANO21 / KaiHand contact gates

| 门 | PASS 条件 | HOLD/撤销条件 |
|---|---|---|
| Array-order authority | final_v3 wrist `0`、tips `[4,8,12,16,20]`、MCP `[2,5,9,13,17]`；palm basis=`0/5/9/17` | final_v3 `joint[5]` 当 wrist，或 `points[:5]` 当五指尖，整条 root/palm/tip/contact/grasp lineage 撤销 |
| Explicit HumanEgo reorder | wrist5 只在 `mano_to_humanego_21` 显式重排后使用 | 把 HumanEgo JSON 顺序直接套到 final_v3 NPZ |
| Shape retarget | palm-frame 中每指四段单位骨向量；thumb 独立六关节；qhand limits PASS | metric fingertip target，或 shape-only 读取 Object6D |
| Root/arm | 初始 translation 精确 wrist0；collision response ≤10 mm、0°；terminal arm ≤10 mm/5° 且 URDF limits PASS | 用错误 root 补偿，逐帧移基座，或把 position/orientation 合并成模糊 pose 指标 |
| Human contact intent | 正确 MANO 每指所有骨段同时满足 finite-Y-cylinder 3D near-surface 与 2D projected adjacency | 单靠 2D 遮挡、单 tip、错误 `points[:5]` 或只挑最好样本 |
| Kai required contact | 正确 finger chain 全部 CAD visual surface 的真正最近 absolute/signed SDF；required contact band 0–3 mm | 从采样中挑接近 1.5 mm 的点冒充最近距离 |
| Object nonpenetration | 所有 visual vertices + deterministic 2 mm triangle samples，dense minimum SDF ≥-1 mm | 只看 sparse vertices、深度合成，或已有网格穿模 |
| Self-collision | 同手不同机械 finger 的 visual-triangle exact SAT；有效 triangles 一律不放宽 | palm/base↔finger 与同 finger link pair 不在部署 pair authority；twice-area ≤1e-9 m² 的 degenerate facet 先排除 |
| Stage/promotion | f345 与 f402 各自 arm/limits/contact/dense/self-collision 全 PASS；先扩小窗口 CPU | 两帧 PASS 不能跳成 10-frame CAD、full460、GPU/render、跨会话或 grasp PASS |

ORDER audit 覆盖 864 个检查点：R2 root→joint0 距离严格为 0，root→joint5 平均
86.0649 mm。当前 f345 required thumb/index、f402 required thumb；root correction 都是
9.9 mm/0°，all-link dense minima 分别 +0.510345/+0.620999 mm，各门 PASS。状态只到
`PASS_F345_F402_CPU_READY_FOR_WINDOW_EXPANSION`。两组 adapter/geometry 测试 30/30 PASS；
GPU/renderer 0/0，10-frame CAD/full460/cross-session correct-MANO gate 均未运行。

旧 full460 的 `contact_required=227`、`hidden=224` 只代表错误指尖标签下的像素遮挡，
明确撤销其 contact/nonpenetration/grasp 含义；文件可解码或视觉遮挡正确不能覆盖物理门。

### 跨会话窗口 initializer successor

- 初始化门：每段独立使用最早 24 个双腕有效帧，冻结唯一 `T_world_rig`；禁止 004 外参、per-frame/per-side base 和 session-specific threshold。
- 数据门先于算法门：002 数据与双臂 downstream PASS；035/065 虽未调参且双臂 downstream PASS，但 `TrackerState=notAccurate`，session 必须 HOLD。
- Temporal 门不放宽：068 side 0/1 的最低可达速度 0.121575/0.123222 > 0.12 rad/frame；139 side 0 的最低可达加速度 0.072740 > 0.06 rad/frame²，side 1 PASS。总体保持 HOLD。
- 接触门必须报告分母：所有通过的 opening window 都有 `contact_required_frames=0`；此时 proxy PASS 是 vacuous，明确禁止抓取 claim。
- Natural-bend v2 只把可见性定义改为肘腕线段与图像相交且腕正深度，同时保留 offset ≥0.08 m、turn ≥35° 和 outward sign；这不是降阈值。

### Event rescue 与独立 QA

- rescue 只扩展有限分支格：首帧、原故障帧、末帧各 31 个 frozen seed，双向延拓后做同一二阶 DP；没有改 session base、initializer、mount、pose/contact/limits/temporal 门或逐帧 offset。
- 三个 12-frame 窗口重算通过：068 side 0 `v=0.076701/a=0.017366`，side 1 `0.093358/0.032418`；139 side 0 `0.067900/0.011624`，都分别不超过 0.12/0.06。
- 独立 QA 从 predecessor、rescue JSON/NPZ 和 pinned Robot assets 重算 88 项：五段 `T_world_rig` 唯一，068/139 transform bytes 不变；36 行 pose ≤10 mm/5°、q 在精确 pinned limits；002/035/065 六条保护 q 的 float64 SHA 与数值记录不变。
- 算法门通过不覆盖数据门：035/065/068/139 继续 `HOLD_TRACKER_NOT_ACCURATE`。所有 session `contact_required_frames=0` 且 `grasp_claim_allowed=false`；禁止把空分母 PASS 写成接触/抓取。
- producer 与独立 QA 均为 CPU-only，GPU/renderer 计数 0/0。

### 视频门

- 渲染前 CPU preflight 固定 42 个 frame、explicit cut、输入 SHA、engine/device/samples/resolution。
- Cycles/OPTIX、4 spp、640×480；overlay/opaque 均 H.264/yuv420p、30 fps、42 帧、完整 decode rc=0。
- PASS 只覆盖 f155..175 与 f205..225；FULL460 PENDING。

## 新任务合同与几何门

- FORMAL_CAPTURE 必须在 SHA-bound bytes 上用 Pillow 11.3.0 先 `verify()` 再完整 `load()`；拒绝截断图、扩展名不匹配和非有限像素。
- 机器解码不能完成“尺子/卡尺与正确对象同帧且刻度可读”的人工门；必须保留 `manual_scene_review_completed_by_machine=false`。
- 对象 geometry type 与 ID 精确匹配；扑克/薯片不能回退到旧 CYLINDER。
- CARD_THIN_BOX、RACK_BOX、CHIP_SADDLE、BOWL_REVOLVE 分别走已声明的解析/网格路径；visual 和 collision 分开。
- depth 必须为正、有序的 2× optical-Z；protect mask 是 amodal 几何建议，不是 segmentation ground truth。
- illustrative measurement、手机 2D anchor、synthetic pose 均不能生成正式米制/PICO 权威。

## 终态判定

- `PASS`：只在本门声明的作用域内通过。
- `HOLD_MISSING_*`：缺权威输入/许可证/人工审核；不得猜测。
- `HOLD_DRIFT`：bytes/SHA/版本/路径漂移；必须建立新 lineage。
- `FAIL`：合同明确不满足或产物损坏。
- `PENDING`：尚未运行的范围，例如 Robot FULL460；不能写成失败，也不能写成通过。
