# 失败谱系与修复（2026-09-02）

本页只记录已经有证据的故障、修复及仍未关闭的 HOLD。禁止把建议写成已完成事实。

## Mask

### CPFS PNG I/O 长时间无 stdout

- 现象：传播正常，但数百帧载入/落盘长时间没有日志。
- 判断：同时检查 PID、输出文件计数和 GPU lease；不能只看 stdout。
- 恢复：保留单个 one-shot，不重复启动第二个 GPU 进程；等待计数前进或明确退出。

### Robot 优先任务触发安全中断

- 现象：原动作 one-shot 在 poker-human 已传播 480/480、写出 379/480 时被安全 SIGINT。
- 处理：partial 原样保留，释放 GPU，未覆盖也未伪装为完成。
- 恢复：仅对未完成链创建 fresh continuation root。

### poker-human continuation v1 import 失败

- 根因：wrapper 没把 pinned SAM3.1 code root 插入 `sys.path`，在模型构建前触发 `ModuleNotFoundError`。
- 修复：先做 CPU import preflight，再以 fresh v2 root/lease 运行；失败目录保留。

### human/wearable 高 IoU 但语义失败

- 现象：human 链全帧存在且 temporal IoU 高，仍漏 hand/wrist/tracker；旧 nonblind 10-session 在 10/10 固定左腕帧排除了可见 tracker。
- 根因：generic `an arm` / `a person's hand and forearm` 的语义范围不足，不是单纯时序漂移。
- 未关闭：`HOLD_MISSING_WEARABLE_PROVIDER`。
- 正确方向：拆出独立 `human_core` 与 `wearable/tracker_raw` provider，做点提示/标注和跨 session 审计。禁止颜色阈值、morph、膨胀或 union 伪补。

### Neutral prompt wearable rescue 仍未闭包

- v3 canary：chips human 11/12、sleeve 0/12、wearable 11/12；poker human 11/12、sleeve 最好 5/12、tracker class 12/12 但 raw identity 在两腕间切换。002 tracker 0/2 prompt，139 仅 1/2 命中 anchor。
- 处理：canary gate fail 后停止，不运行全片传播或 union；颜色、形态学、膨胀、flow 等 forbidden repair 均未使用，object masks 完全未读未改；032/051/sealed 004 reads=0。
- 运行谱系：v1 adapter path pre-inference 失败；v2 暴露 one-frame 后不支持 `reset_session`；fresh v3 改为每 canary 独立 one-frame session。最终 aggregate 的 Python `false/False` typo 只影响峰值持久化，live runner 已做 one-token repair，历史结果不改。
- 资源边界：另做同形状 replay 得到 6.08 GB allocated / 6.61 GB reserved，不能写成 v3 完整 process peak。
- 下一步：九类 offline annotation tool 已通过 synthetic/E2E/HTTP smoke，但真实 multi-session labels、production trainer、class-conditioned weight 均不存在，继续 HOLD；不能再把调 prompt 写成通用修复路线。

### Assisted bilateral v1→v4：从“有类”到可审计 identity

- 首轮现象：multiplex point object 在 temporal API 中只有 frame 0 非空；同一 state
  的第一个 wearable raw ID 还会被 first-object quirk 压空。
- 修复：每个 wearable 使用独立 state；固定先在 protected task object 上创建并
  丢弃 priming ID，再创建实际 wearable ID。后续不依赖 SAM point temporal，而用
  RAW DIS flow；每 20 帧在独立 single-frame state 以 flow centroid 自动 refresh。
- v2 的 upper tracker 在 chips f80 被提成整只手，且旧 min-set identity 在 human
  raw 重叠时两侧距离都为 0。v3 加入确定性负点（同侧 human 内离正点最远像素），
  identity 改为 wearable centroid 到两 human centroid 的距离差；chips 两 tracker
  随即达到 17/17 refresh、359/359 flow。
- v3 仍在 chips upper human f294 附近 HOLD：SAM raw 达 4.104×，单向 inverse warp
  从 3.994×长到 4.014×，不能通过未变的 4×门。
- v4 正确修复为真正双向 raw consensus：`current→previous` inverse warp 与
  `previous→current` nearest forward splat 做 Boolean intersection；不做 morphology、
  dilation 或 fill。chips 因此正式 360/360 PASS，union 对 role OR-minus-object 的
  全片 mismatch=0，object overlap=0。

### Poker v4 为什么仍 HOLD

- 相同 v4 参数没有做视频特调。human 双侧 PASS，lower sleeve cuff 23/23 refresh
  PASS；但 upper tracker 只有 17/23，拒绝帧为 220/280/300/420/440/460。
- lower tracker 的 frame-0 raw 只有 640 px，23 次 refresh 全拒绝；SAM proposal
  经常变成约 4.3k 或更大的邻近 wrist/cuff 区域，被冻结 area/centroid/IoU 门拒绝。
  flow 在 204--351、400--405、409、411--479 共 224 帧与同侧 human 距离超过 18 px。
- 正确处理：保留 HOLD、禁止 full artifact promotion，也不再松门。最低离线标注
  8 帧 `0,200,280,352,400,420,460,479` 的双 tracker（16 实例）；稳妥版再加
  `220,300,440`（11 帧/22 实例）。只补完整 tracker housing+strap，human、已通过
  cuff 和 object 不重标。
- 恢复验收：新标签必须进入 fresh lineage；重新冻结 anchor manifest/producer SHA，
  先过 12-canary 和全部 refresh，再允许 full。禁止用现有 poker HOLD mask 给 Clean。

### Clean 首帧“白 tracker 仍白”的误判

- 初看 downstream A/B 以为 chips union 没有 OR 入 tracker。逐像素审计证明 f0
  upper tracker 1927/1927、lower 3843/3843 eligible pixels 全部进入 union；四个
  anchor 坐标在 role raw 与 union 都为 true，12 canary tracker coverage 都是 100%。
- 结论：Mask union 没漏；白色残留属于 Clean donor/unsupported/edge 问题。不要为
  下游可见残留修改已通过的上游 union 或增加颜色/形态学补丁。

## Clean

### L2 没有输出 CLEAN_RGB

- 原因：当时两个动作 MOV 的 Mask 未到位；跨视频像素复制被合同禁止。
- 处理：只冻结源身份、角色、解码、注册/光度/覆盖诊断，状态保持 L2。
- 恢复：Mask 到位后只做同一 MOV 的 L3 temporal donor 小样；正式 PICO 需同 session 正式 donor。

### 移动相机无法由单张底图恢复

- 证据：`yidongzhuomian.mp4` 相对中帧 177 时，f0/f44 无单应模型，f88 仅 10 内点，L 均值范围 14.23。
- 根因：视场变化、低纹理和曝光变化；桌外背景/木架/碗/手上物体也不共面。
- 未关闭：`HOLD_MISSING_INTRINSICS_C2W_TABLE_PLANE`。
- 恢复：取得逐帧 K、camera-to-world、session-static metric plane 和同 session empty sweep；ORB 仍只作诊断。

### Donor 能注册但无权使用

- 现象：空桌→扑克/薯片中点有可用内点和低残差。
- 原因：没有同 session/相机/曝光锁证明，且空桌会擦除木架、碗和操作物。
- 处理：跨视频 donor 使用次数保持 0；无直接观测区域保持 unsupported。

### Clean L3 首版 validator 未逐字节复核 source map

- 风险：只有“source map 存在”不足以证明 clean 像素来自所声明的 donor 坐标。
- 修复：封存前补上逐像素 byte-equality 检查；61,202 个 copied pixel mismatch=0，同时复核 unsupported 无 source、语义漏点未被填充。
- 终态：产物完整性 PASS，但策略状态仍为 `HOLD_HUMAN_SEMANTIC_CLOSURE_UNRESOLVED`。

### Clean：为什么 bilateral mask 正确后仍不能消掉手影

- 表象：窄区域 A/B/C 仍像一只手，最初看起来像 donor 被人体污染。
- 排查：强制 donor 对应像素 bilateral-mask=0 且距 mask 至少 12 px，并输出
  donor-purity/source map；donor 已干净，手形仍在。chips f0 tracker 也已逐像素确认进入
  union，因此不能把 Clean 残留归因到上游 Mask。
- 根因 1：Poisson 使用 target-interior gradients/边界时，会把原图手边界和阴影重新
  解回 donor。改为 donor core 100% source-driven、禁止 target-interior gradients 后，
  手才真正消失。
- 根因 2：局部 PatchMatch、单块平移和 quilting 无法可靠延拓大面积非平稳皮纹桌面，
  形成竖缝、重复纹理和矩形块；这些只能做诊断。
- 根因 3：不同机位/物体布局的 static whole-table plate 无法由中心对象点充分约束。
  旧四点 H 出现 projective pole；增加独立绿色夹子点及分母/条件数约束后虽数值稳定，
  仍会重复碗/筹码并产生可见透视畸变。
- 修复边界：本素材停止新增算法变体。最小可靠输入是同机位、同姿态、尽量同物体布局的
  2--3 秒无人手 clean plate；若无法补录，至少提供四个覆盖 replacement polygon 外围、
  在两图独立观测的同名点。集中于桌面中心的对象点不能替代外围几何。
- 防复发：frame0 可视门必须在 10-frame/full 之前；projective safety PASS 不能覆盖
  duplicate-object/seam/warp HOLD。终态为 `HOLD_CROSS_CAMERA_STATIC_INSUFFICIENT`，
  `ten_frame_started=false`、`full_video_started=false`。

## Robot

### MANO 数组顺序误读：joint5/root 与 points[:5] lineage 全部撤销

- 误读 1：把 final_v3 NPZ `joint[5]` 当 wrist/root，再反过来判定旧 R2 `joint[0]`
  root 错误。ORDER authority 的 864 个检查点证明 R2 root→final_v3 joint0 严格 0，
  root→joint5 平均 86.0649 mm；final_v3 实际是 MANO21 wrist0。
- 误读 2：把 final_v3 `points[:5]` 当五个 fingertips。它其实是 wrist + 完整 thumb
  chain，因此由此产生的 middle/ring 等 contact 标签和 grasp 语义无效。
- 唯一当前顺序：wrist `0`；tips `[4,8,12,16,20]`；MCP `[2,5,9,13,17]`；
  palm basis 用 `0/5/9/17`。HumanEgo wrist5 只在显式
  `mano_to_humanego_21=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19,-1]`
  后使用。
- 以下 10 个 run 均有撤销标记，不得作为 root、palm、tip、contact 或 grasp authority：
  `archive/legacy_runs/robot/robot_004_full460_collision_constrained_cpu_20260902_v1`、
  `_run/robot_004_full460_visual_soft_depth_cycles_20260902_v1`、
  `archive/legacy_runs/robot/robot_004_full460_visual_soft_depth_keyframes_cycles_20260902_v1`、
  `archive/legacy_runs/robot/robot_004_kaihand_contact_response_cpu_20260902_v1`、
  `archive/legacy_runs/robot/robot_004_kaihand_contact_rootcause_readonly_20260902_v1`、
  `archive/legacy_runs/robot/robot_anatomical_wrist_frame_adapter_cpu_20260902_v1`、
  `archive/legacy_runs/robot/robot_anatomical_wrist_kai_contact_gate_cpu_20260902_v1`、
  `archive/legacy_runs/robot/robot_global_wrist_kai_adapter_cpu_20260902_v1`、
  `archive/legacy_runs/robot/robot_global_wrist_kai_session_initializer_cpu_20260902_v1`、
  `archive/legacy_runs/robot/robot_global_wrist_kai_session_initializer_cpu_20260902_v2`。
- 防复发：在任何手部优化前先校验 ORDER authority SHA；array order 必须写入 protocol，
  禁止靠画面或“更像 wrist”反推索引。

### 旧 227/224 遮挡统计被误写成抓取

- 旧结论：`contact_required=227`、`hidden=224` 被解释为物体挡住接触手指，进而证明抓取。
- 根因：这两个数字来自错误 `points[:5]` 标签下的渲染像素遮挡；2D depth composite
  还能遮住已经在 3D 中穿入物体的网格。
- 撤销边界：旧 full460 仍可作为历史视觉/解码诊断，但不能作为 hand contact、object
  nonpenetration、grasp 或训练数据 authority。
- 正确 successor：人手 intent 同时要求正确 MANO 全骨段 3D 邻近 + 2D 邻接；KaiHand
  使用正确指链全部 CAD visual surfaces 的真实最近 SDF、dense triangle nonpenetration 和
  distinct-finger exact SAT。f345/f402 有界 CPU PASS，GPU/renderer 0/0。
- 未完成：尚无 10-frame CAD/物理连续性、新 full460 或正确门下跨会话结果；timeline 的
  251 个 required frames 是 human intent 标签，不是 251 帧 Robot contact PASS。

### EEVEE 预览 0 帧

- 现象：`robot_004_event_windows_eevee_preview_20260902_v3/TERMINAL_FAILURE.json` 记录 EGL/GLX context unavailable，0 帧。
- 处理：失败根保留，不创建视频短链。
- 修复：改用真实 Blender Cycles/OPTIX one-shot；输出 42 帧 overlay/opaque 并通过完整解码。

### 004 事件窗口 full-pose IK 分支跳变

- 现象：f162、f215 有分支跳变。
- 根因：不是目标轨迹内在不连续，而是局部 IK 分支选择。
- 修复：固定有限 branch bank、backward continuation 和二阶 DP；两窗口双臂最大速度 0.055920、最大加速度 0.039790，均在门内。
- 边界：窗口外和窗口间没有证明；FULL460 仍 PENDING。

### 旧跨会话 contact 聚合布尔缺陷

- 根因：source runner 在 pose gate 之前把 `contact_pass=true`；pose 失败后虽补算接触，却没有回写 aggregate。
- 处理：历史 `RESULTS.json` / `PASS_HOLD_MATRIX.json` 不改；创建 immutable successor audit。
- 修复权威：002 PASS；068 在 f340/f397 physical side 1 失败而 HOLD；139 PASS。总体仍 `HOLD_NOT_FORMAL_CROSS_SESSION_PASS`。

### Human-origin fallback 的结构性失败

- 证据：home root 间距 0.551 m，而 002/068/139 为 0.497/0.363/0.504 m；068 差 0.188 m。所有 54 arm-side row 的 elbow 均投影到画外，natural-bend HOLD。
- 根因：frame-0 双腕刚体拟合不能吸收身体/臂展差异，也不能把肩肘自然放进相机视野。
- 未关闭：三段均缺声明的 fiducial/opening-neutral block；068/139 另有 `TrackerState=notAccurate`。
- 正确方向：正式采集开场标定块并建立可审计的 session initializer。禁止 per-frame offset 或复制 004 外参。

### Window initializer 改善 downstream，但未关闭 session HOLD

- successor：`robot_cross_session_window_initializer_dp_cpu_20260902_v1` 用每段最早 24 个双腕有效帧拟合唯一 session base；002 数据与双臂 downstream PASS，未调参 035/065 双臂 downstream 也 PASS。
- 未关闭：035/065/068/139 的 `TrackerState=notAccurate` 仍是上游数据 HOLD，数值 downstream PASS 不能覆盖它。
- 结论：这是 generic initializer 的非盲窗口诊断进展，不是 sealed/blind/full-video 跨会话 PASS。

### Temporal blocker audit 保留原阈值

- 068 side 0/1 首个失败窗口最低可达速度分别为 0.12157525069142139、0.12322151130925929 rad/frame，均超过 0.12。
- 139 side 0 最低可达速度 0.045774 通过，但最低可达加速度 0.07274000132444713 rad/frame² 超过 0.06；side 1 PASS。
- audit 没有改 threshold、seed、base、004 外参或 renderer，状态为 `COMPLETE_SOURCE_HOLD_PRESERVED`。正确恢复方向是查输入目标跳变/跟踪质量并建立新 lineage，不是豁免 temporal gate。

### Event rescue 关闭三个 downstream temporal blocker

- 修复：仅在同一 12-frame 窗口增加首帧/故障帧/末帧 frozen 多分支双向延拓，再以原 0.12/0.06 硬门做二阶 DP；没有改 base、initializer、target、threshold 或逐帧 offset。
- 结果：068 两侧和 139 side 0 均找到 pose/limits/temporal/natural-bend-v2 合格路径；独立 QA 从 q bytes 重算为 88/88 PASS。
- 保护：002/035/065 原 q 序列 byte-identical，六条 float64 SHA 与完整 numeric record 不变。
- 未关闭：035/065/068/139 的 TrackerState 数据门仍 HOLD；所有 rescue/protection window 的 `contact_required_frames=0`，因此 downstream 修复不产生抓取或正式跨会话 claim。

### 数值 pose 与有效性冲突

- 现象：139 两侧数值 9/9 通过，但一个 target 被标 invalid。
- 处理：数据有效性优先于数值误差，pose gate 仍 HOLD。

### Contact proxy 被误读为抓取

- 风险：`overlap_required=0` 的行会按协议通过。
- 修复：报告 required overlap 计数，并明确 Object6D cylinder/CAD 顶点代理不等于真实抓取。最新通过 opening window 的 `contact_required_frames=0`，因此 contact PASS 是 vacuous。

## 新任务合同与几何

### v2.1 四字节 JPEG 可穿过 signature-only 检查

- 根因：只看文件签名字节，没有完整解码。
- 修复：v2.2 在精确 SHA-bound bytes 上固定 Pillow 11.3.0，执行 `verify()` + full `load()`，禁止截断图，并核对扩展名/格式、正尺寸、通道和有限像素。
- 仍需人工：尺子/卡尺和正确对象是否同帧、刻度是否可读，机器不得宣称完成。

### 旧 CYLINDER 几何被错误复用

- 风险：把扑克或薯片简化成旧圆柱会通过错误的形状合同。
- 修复：firewall 直接拒绝；扑克走 CARD_THIN_BOX + RACK_BOX，薯片走 CHIP_SADDLE + BOWL_REVOLVE；不存在 fallback。

### Bowl cavity 因 collision inflation 闭合

- 根因：visual/collision 混用或输入尺寸自相矛盾。
- 恢复：修正有证据的测量维度，visual 与 collision 保持分离；不得按每帧调几何。

### 手机 anchor 漂移

- 处理：只修订/复核 normalized 2D development anchor；禁止把像素调整换算成米制几何。

## 通用恢复纪律

1. 先保留失败根和 terminal evidence，再建立 fresh successor。
2. 先验证输入 SHA/协议/环境，后修算法；不从输出反推“正确输入”。
3. 修复只触碰最小故障层，禁止顺便改 threshold、session、keyframe 或 claim。
4. 缺权威就 HOLD；缺权重就写 `weights=[]` 或 `HOLD_MISSING_WEIGHT`，不能编造名称/版本。
5. 复跑后重新计算 bytes/SHA、JSON parse、图像/视频完整解码，并在 artifact index 标 successor 关系。
