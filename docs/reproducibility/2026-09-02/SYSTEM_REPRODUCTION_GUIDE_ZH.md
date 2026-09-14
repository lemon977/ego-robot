# 系统复现指南（2026-09-02）

状态：`SEALED_WITH_EXPLICIT_HOLDS`。本指南把 Mask、Clean、Robot、扑克/薯片新任务合同与几何准备串成一个可审计的恢复顺序。它不把开发素材升级为正式 PICO 数据，也不把局部或非盲结果扩成正式通用性结论。

项目根：`/mnt/workspace/code/chaoyang`。所有复跑必须写入新的 `_run/<name>_repro_vN`；历史根是证据，不得覆盖。

## 当前可复现结论

| 子系统 | 当前状态 | 可以声称 | 不能声称 |
|---|---|---|---|
| Mask | `CHIPS_ASSISTED_FULL_PASS / POKER_TRACKER_HOLD` | 同一冻结 v4 在 chips 输出 360/360 双 human+双 tracker union；poker 双 human 与 cuff 通过 | tracker 已跨任务通用、poker 可消费、零人工或正式 PICO Mask |
| Clean L2 | `SOURCE_READINESS_COMPLETE` | 六段手机素材的身份、解码、角色、注册/光度/覆盖路线已冻结 | 正式 `CLEAN_RGB`、跨视频 donor 合法、移动相机已具备 K/c2w |
| Clean L3 | `HOLD_HUMAN_SEMANTIC_CLOSURE_UNRESOLVED` | 10 个目标帧、61,202 个替换像素均与源字节一致、跨视频 donor=0 的手机诊断 | 正式 donor、跨视频迁移、ORB=K/c2w 或无证据补洞 |
| Clean chips frame0 successor | `HOLD_CROSS_CAMERA_STATIC_INSUFFICIENT` | 已消费授权的 chips bilateral union；frame0 能消除手且 object/non-table byte-identical | 现有跨机位 static 可作真实 clean plate、10 帧/全片已启动、数值安全门等于可视 Clean PASS |
| Robot 004 | `PASS_F345_F402_CPU_READY_FOR_WINDOW_EXPANSION / FULL460_HOLD` | 正确 MANO21/KaiHand authority 下 f345、f402 的有界单侧 CPU arm/limits/contact/nonpenetration/self-collision PASS；旧 42 帧仅保留姿态视觉 | 10-frame CAD/物理连续性、新 full460 scene/video、GPU/render、双侧或跨会话正确接触、真实抓取 |
| Robot 跨会话 | `PASS_DOWNSTREAM_EVENT_RESCUE_TRACKER_HOLDS_PRESERVED` | 002 与 035/065 原窗口 downstream PASS；068 双侧和 139 side0 event rescue 后三窗口也通过冻结硬门 | 总体/正式跨会话 PASS；TrackerState HOLD 可忽略；contact_required=0 可证明抓取 |
| 新任务合同 | `COMPLETE_FULL` | v2.2 文件解码权威、防错 schema 与 fail-closed firewall | 机器已完成尺子/对象同帧人工审核 |
| 新任务几何 | `COMPLETE_APPROX_GEOMETRY` | 解析几何、2× optical-Z depth/protect 的开发准备 | 正式米制尺寸、PICO 位姿或正式 session |

## 配置层级：必须分开记录

### Generic（跨 session 固定）

- Mask：SAM3.1 固定模型/代码/adapter；v4 固定 human text raw、wearable point raw、20 帧 refresh、双向 DIS/identity/edge/area/object 门；禁止颜色与形态学像素修补。
- Wearable 标签接口：左右 `SKIN/SLEEVE/TRACKER`、任务对象、uncertain、background 九类互斥；按 whole session 冻结 train/dev/blind，object 是 protected negative。
- Clean：同源 donor、真实观测像素、human/object/fixture 保护、`source_map` 与 `unsupported_mask`、无 inpaint/生成式补洞；跨机位 whole-table static 只允许作 PHONE_DEV 诊断，不能成为正式 donor。
- Robot：固定资产/URDF 限位、审计 mount、home pair、`robot_to_human_side=(1,0)`、六 seed IK、10 mm/5 deg、0.12 rad/frame、0.06 rad/frame²、自然弯臂门；手部 authority 固定为 MANO21 wrist `0`、tips `[4,8,12,16,20]`、MCP `[2,5,9,13,17]`，KaiHand 接触/非穿透/自碰门必须分别审计。
- 新任务：合同 v2.2、对象类型 firewall、几何 adapter v2.0、正 optical-Z 和状态驱动保护。

### Per-session / per-source（每段独立冻结）

- 输入路径、bytes、SHA-256、帧数、旋转、K/c2w/时间戳（若存在）、Mask 结果和数据角色。
- Mask 的 frame-0 upper/lower identity anchor、可见 wearable 实例数和任务 object raw 路径；它们属于源输入，不是 generic 参数。v4 自动 refresh 点由上一 accepted flow mask 决定，不是人工 session patch。
- Clean 的 target frame 与 donor bank；donor 只能来自同一源 SHA。若做跨机位 frame0 诊断，还必须冻结 clean plate 身份、独立对应点、table polygon 与 projective safety protocol，并保持正式 claim 为 HOLD。
- Robot 的 manifest、SLAM、R2 sidecar、Object6D、RGB/keyframe metadata/c2w、final_v3 数组顺序和显式 reorder；旧 human-origin 诊断使用本段 frame-0 c2w + 双腕 SVD，window successor 使用本段最早 24 个双腕有效帧，但两者都只冻结该 session 唯一的 `T_world_rig`。正确 MANO 接触标签时间线属于 human intent，不自动成为 Robot contact 结果。

### Session patch

当前封存结果中没有 session-specific threshold、prompt、crop、外参、关节 offset 或 per-frame patch。若新数据需要 patch，必须另建 fresh root、预注册理由和作用域，并在正式 claim 前先标 `HOLD_SESSION_PATCH_INTRODUCED`。

## 从零复现顺序

### 0. 只读完整性预检

```bash
cd /mnt/workspace/code/chaoyang
sha256sum -c _run/newtask_baseline_clean_20260901_233238/SHA256SUMS
sha256sum -c _run/newtask_geometry_prep_v2/SHA256SUMS
python -m json.tool docs/reproducibility/2026-09-02/ALGORITHM_AND_WEIGHT_MANIFEST.json >/dev/null
python -m json.tool docs/reproducibility/2026-09-02/ARTIFACT_INDEX.json >/dev/null
```

任何 path、bytes、SHA 或依赖版本漂移都先 `HOLD_INPUT_DRIFT`，不要继续渲染或推理。

### 1. 新任务对象合同 v2.2（CPU）

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_newtask_object_firewall.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python _run/newtask_contract_v2_2/dry_run_contract.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_object_occlusion_geometry.py tests/test_sam31_ego_human_producer_p3.py tests/test_newtask_object_firewall.py
```

预期：53 个任务测试与 81 个邻接回归通过；四字节 JPEG、截断 PNG、错扩展名、F1 illustrative→formal 和两条 CYLINDER canary 均按预期拒绝。

### 2. 新任务参数化几何（CPU）

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONHASHSEED=0 python tools/run_newtask_geometry_prep_v2.py --output-root _run/newtask_geometry_prep_v2_repro_v1
PYTHONPATH=. PYTHONHASHSEED=0 python -m pytest -q tests/test_newtask_parametric_geometry.py tests/test_newtask_object_firewall.py tests/test_object_occlusion_geometry.py tests/test_depth_occlusion_v3.py
```

输出根必须不存在。这里 `CUDA_VISIBLE_DEVICES=0` 只保留命名空间，实际路线为 NumPy/OpenCV CPU；不得把 illustrative dimensions 转成正式米制权威。

### 3. Mask（单次 GPU lease，串行）

先冻结六个源文件、runner、adapter、官方 code commit 和 checkpoint SHA。以下是历史实际命令；三条推理都使用同一 run root，且动作 runner 预期该 root 已有 360/480 张、按 display rotation 解码的 `action_input_frames/{chips,poker}`：

```bash
python3 tools/run_newtask_baseline_sam31_mask_probe.py --root /mnt/workspace/code/chaoyang/_run/newtask_baseline_mask_probe_20260901_v1 --cases puke shupian
python3 tools/run_newtask_baseline_sam31_action_probe.py --root /mnt/workspace/code/chaoyang/_run/newtask_baseline_mask_probe_20260901_v1 --cases chips poker
python3 tools/run_newtask_baseline_sam31_poker_human_continuation.py --source-root /mnt/workspace/code/chaoyang/_run/newtask_baseline_mask_probe_20260901_v1 --output-root /mnt/workspace/code/chaoyang/_run/newtask_baseline_mask_probe_20260901_v1/action_continuation_poker_human_20260902_v2
```

不要直接执行上述路径：case/action/continuation 都拒绝覆盖或会与历史证据冲突。当前代码集没有封存“源 MOV → action_input_frames”的参数化 materializer；所以真正从源视频归零复现必须先实现并冻结该 CPU 解码步骤，在确认每帧 SHA 与显示方向后才运行 fresh root。若只做冻结帧 replay，可在 fresh root 只读链接历史 `action_input_frames`，但必须把 scope 写成 `FROZEN_FRAME_REPLAY_NOT_FROM_SOURCE_DECODE`。模型 checkpoint 约 3.50 GB，历史峰值约 28.48 GB allocated / 30.65 GB reserved；无 stdout 不能视为死锁，须同时看 PID、输出计数和 central lease。

下面两条整合/验证脚本同样硬编码历史 root 且拒绝覆盖，只是历史命令，不是 safe replay：

```bash
python3 tools/build_newtask_mask_action_integration.py
python3 tools/validate_newtask_mask_artifacts.py
```

在 fresh lineage 使用它们前，必须先增加显式 `--root/--continuation-root`，冻结修改后 SHA，并写入新 manifest；否则标 `HOLD_MISSING_PARAMETERIZED_MASK_INTEGRATOR`。

上段是早期 generic action producer 的历史边界。它没有被修改；最新 assisted v4 successor 的独立结论如下。

固定 neutral prompt 的 wearable semantic rescue 历史命令如下；v3 已存在，不得原地重跑：

```bash
python3 tools/run_mask_wearable_semantic_rescue.py \
  --root /mnt/workspace/code/chaoyang/_run/gpt_mask_wearable_semantic_rescue_20260902_v3 \
  --protocol /mnt/workspace/code/chaoyang/_run/gpt_mask_wearable_semantic_rescue_20260902_v1/PROTOCOL.json
```

12-canary 结果：chips human 11/12、sleeve 0/12、wearable 11/12；poker human 11/12、sleeve 最好 5/12、tracker class 12/12 但 raw identity 在双腕间切换。002 两个 tracker prompt 均 0，139 仅一个 prompt 命中 anchor；032/051/sealed 004 pixel reads 均为 0。因为 canary 门失败，没有 full propagation、union、颜色/形态学/flow 修补，object masks read/modified=0。

同形状 one-frame resource replay 为 6,083,911,168 B allocated / 6,610,223,104 B reserved；它不能冒充丢失的 v3 process peak。

#### Assisted bilateral wearable v4（当前权威）

固定协议与 producer：

- `tools/run_assisted_bilateral_flow_refresh.py`，SHA-256
  `65cb00ffaa0d6c36fd8fea6a7c5448abcf3083f738fe5c1d5d3ca37bb969a774`；
- `_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1/FLOW_REFRESH_PROTOCOL_V4.json`，SHA-256
  `ab52ed204cb5221a72df62db8b0cbf5759f71399c6d9191efac9639a734d57bd`；
- frame-0 source anchors SHA-256
  `028b6b2e1d0e309d4688addf43fc150bbf42e8d7237d9bf814b539860d4762e6`；
- checkpoint `sam3.1_multiplex.pt` 3,502,755,717 B，SHA-256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；
- adapter 使用 official code commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`。

固定算法不是对某帧做颜色补丁：human 由 `a person's hand and forearm` text raw
stream 产生，frame 0 仅按 12 px 内最近 raw contour 分离 upper/lower；raw 面积越过
冻结 `[0.25,4.0]×frame0` 时，只能使用 `current→previous` inverse warp 与
`previous→current` nearest forward splat 的 RAW Boolean 交集，并继续过 cycle、
相邻面积、edge 和 object 门。每个 tracker/cuff 只在 frame 0 人工注册；后续使用
270×480 DIS 双向 flow，每 20 帧在独立 single-frame SAM state 自动 refresh。每次
refresh 先以 task-object centroid 建一个明确丢弃的 priming raw ID，再以 flow
预测 centroid 为正点、同侧 human 内最远像素为确定性负点；proposal 必须同时过
IoU、centroid、area、same-side adjacency、opposing-side identity、edge、cycle 和
object-overlap=0。禁止 color threshold、morphology、dilate、erode、fill、GrabCut、
人工 refresh 点和 session-specific ROI。

历史实际命令如下。它们要求 central lease 为本任务 `ACQUIRED`、GPU 无其他 compute
进程且输出根不存在；复跑必须改成新的 `_repro_vN` 根，绝不能覆盖下列历史根：

```bash
cd /mnt/workspace/code/chaoyang
/usr/local/bin/python3.11 tools/run_assisted_bilateral_flow_refresh.py \
  --case chips \
  --frame-root _run/newtask_baseline_mask_probe_20260901_v1/action_input_frames/chips \
  --output-root _run/gpt_mask_assisted_bilateral_flow_refresh_chips_20260902_v4 \
  --frame-count 360 --fps 30

/usr/local/bin/python3.11 tools/run_assisted_bilateral_flow_refresh.py \
  --case poker \
  --frame-root _run/newtask_baseline_mask_probe_20260901_v1/action_input_frames/poker \
  --output-root _run/gpt_mask_assisted_bilateral_flow_refresh_poker_20260902_v4 \
  --frame-count 480 --fps 30
```

chips 结果为 `PASS_CANARY12_FULL_ARTIFACTS_EMITTED`：四个可见 role 均
360/360 非空，union 是四 role OR 后减 protected object，逐像素重算差异 0，
发布 union 的 object overlap 最大 0；class/union 视频都完整解码 360 帧。
`consumption_authorized=true`。poker 使用 byte-identical runner/protocol/阈值，
human 双侧 PASS、lower cuff 23/23 refresh PASS，但 upper tracker 17/23、lower
tracker 0/23；后者 frame-0 raw 仅 640 px，并有 224 个 same-side adjacency flow
失败。因此 poker 为 `HOLD_CANARY12_NO_FULL_ARTIFACT_PROMOTION`，没有 full union，
`consumption_authorized=false`。不得用 chips PASS 扩成跨任务 tracker 通用性声明。

poker 恢复先离线标注 tracker，不再松门。最低标注 8 帧
`0,200,280,352,400,420,460,479`，每帧 upper/lower tracker，共 16 个实例；稳妥版
再加 `220,300,440`，共 11 帧/22 实例。mask 必须覆盖完整白色 housing 与随动
灰/绿 strap，排除 skin、fabric cuff、hand、table 和 task object。human、已通过的
lower cuff 与 task object 不需重标。详见
`_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1/POKER_OFFLINE_LABEL_PLAN.md`。

### 4. Clean L2/L3（CPU）

L2 源检查：

```bash
python tools/probe_baseline_clean_sources.py
python -m pytest -q tests/test_probe_baseline_clean_sources.py
```

L3 封存命令如下；安全复跑必须换成 fresh output root：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/envs/humanego/bin/python tools/run_newtask_baseline_clean_l3.py --output _run/newtask_baseline_clean_l3_repro_v1
```

通用不变量是：同一 MOV、同一源 SHA、donor 非 human、object/fixture 不被覆盖、每像素 source map、无 donor 则 unsupported。不能用 `puke.mp4`、`shupian.mp4` 或空桌视频给动作 MOV 贴像素。封存结果 chips 注册 60/60、registered fill median 77.45%；poker 34/61、0.98%。两条都因上游 human/wearable 语义未闭包而 HOLD。

#### Clean PHONE_DEV bilateral-mask frame0 successor

chips bilateral union `_run/gpt_mask_assisted_bilateral_flow_refresh_chips_20260902_v4`
已正式允许本次 PHONE_DEV 消费：`consumption_authorized=true`，360/360 张 binary
union 完整。Clean 仍为 `HOLD_CROSS_CAMERA_STATIC_INSUFFICIENT`，因为正确 removal
mask 不会凭空产生干净、同视角的真实 donor。

frame0 的冻结失败链如下；每一步都停留在 PHONE_DEV、没有提升为正式 Clean：

1. 窄 same-MOV donor 的 multiband/Poisson 留下手形，最初怀疑 donor 污染。
2. pure-donor 审计强制每个源像素 bilateral-mask=0 且距离 mask 至少 12 px。donor
   已干净，但 Poisson 仍通过 target-interior gradients 重建手边界/阴影，证明污染不是
   唯一根因。
3. 100% source-driven donor core + graph-cut 能消手；下方跨 session static 留竖缝和
   纹理块，上方仍有灰影。
4. target-frame PatchMatch 与双尺度 real-pixel quilting 虽保留 source coordinates，
   并距 human/object mask 至少 32 px，仍在非平稳桌面纹理上产生平移/重复块。
   Telea/NS 只作为 `GENERATED_CLASSICAL` 对照，不得声称真实观测 RGB。
5. whole-table plate 把 seam 移到真实桌边；旧四点 H 有 projective pole、ECC 失败。
   最终加入独立测量的绿色夹子 centroid/tip 后，affine 的 right-chip residual 仍为
   85.2412 px；受约束 H 虽过分母/条件数门，六锚点 residual 仍为
   `[11.4510,50.1833,37.1642,53.3942,7.4643,25.7999]` px，画面有重复碗/筹码和畸变。

安全 replay 仅允许 frame0：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/build_clean_frame0_whole_table_bounded_final.py \
  --output _run/newtask_clean_chips_frame0_whole_table_bounded_final_repro_v1
```

输出根必须不存在；若 protocol 要预置，只能放 byte-pinned `PROTOCOL.json`。runner
SHA-256 为 `4a4382cc2e9833755b46f579500d7e8db1294381d10bafebca12f8193fa9247e`，
不加载学习权重。新 frame0 未同时通过 source-authenticity、object protection、geometry
和 visible-residual 门前，禁止启动 10 帧或全片。

最小补录首选：动作视频相机完全不动、姿态/曝光及物体布局尽量不变，录 2--3 秒没有
手、袖口和 tracker 的 clean plate。若无法补录，至少在 static/action 两图中独立标出
四个分布于 table replacement polygon 外围的同名点；不能集中在桌面中心，也不能由旧
H 反投影生成。当前短入口为 `NOW/clean/cross_camera_final_hold.md`。

### 5. Robot 跨会话 CPU 诊断与接触复核

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /usr/local/bin/python3.11 _run/robot_cross_session_human_origin_keyframes_cpu_20260902_v1/run_cross_session.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /usr/local/bin/python3.11 _run/robot_cross_session_human_origin_contact_audit_cpu_20260902_v1/audit_contact.py
```

历史脚本路径内嵌了历史输出根，直接运行会覆盖证据，因此这些是“历史实际命令”，不是安全 replay 命令。安全 replay 必须复制 producer 到新 root、只改输出根、重新冻结 producer SHA，并保留相同协议/输入；不得改 session、keyframe、seed 或阈值。下游 contact 必须读取 successor audit，不能读旧 `PASS_HOLD_MATRIX.json` 的 contact 汇总。

### 6. Robot 004 事件窗口与视频

CPU solver authority 是 `_run/robot_004_bilateral_bidirectional_event_dp_cpu_20260901_v1`；渲染 authority 是 `_run/robot_004_event_windows_cycles_preview_20260902_v1`。安全复跑先运行同根中的 `build_cpu_preflight.py` 到新根，再在取得 GPU/renderer lease 后以 Blender 4.5 background + Cycles/OPTIX、4 spp、640×480 渲染 42 帧。输出必须是 H.264/yuv420p、30 fps、42 帧，完整解码 exit 0。该 lineage 现在只保留为 arm trajectory/姿态视觉证据，不再作为手指接触、非穿透或抓取 authority。

这一步只复现 f155..175、f205..225；f176..204 是显式 cut，FULL460 仍 PENDING。

#### Correct MANO21 / KaiHand bounded CPU successor（当前手部 authority）

先执行关节顺序 authority。`final_v3` NPZ 是 MANO21：wrist=`0`、tips=
`[4,8,12,16,20]`、MCP=`[2,5,9,13,17]`；palm basis 用 wrist0、index MCP5、
middle MCP9、pinky MCP17。HumanEgo JSON 的 wrist5 只在显式
`mano_to_humanego_21=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19,-1]`
重排后有效。ORDER audit 在 864 个检查点上得到 R2 root→joint0 严格 0，
root→joint5 平均 86.0649 mm；authority SHA-256 为
`ab6a280589a26c34e3c1c3e495911e00e26c66055b8199fb47fa1e128a7c29c1`。

因此明确撤销两类旧语义：

- `_run/robot_004_full460_visual_soft_depth_cycles_20260902_v1` 的
  `contact_required=227` / `hidden=224` 只是错误标签下的渲染像素遮挡，不能证明正确
  手指接触、非穿透或抓取；旧 full460 只可作历史视觉诊断和文件解码证据。
- final_v3 `joint[5]` 不是 wrist/root，`points[:5]` 也不是五个 fingertips，而是 wrist
  加 thumb chain。所有基于这两个假设的 root/palm/tip/contact/grasp 报告均已撤销；完整
  10-root 清单见 `FAILURES_AND_FIXES_ZH.md` 与
  `archive/legacy_runs/robot/robot_004_mano21_kai_contact_gate_cpu_20260902_v1/ARTIFACT_PROCESS_AUDIT.json`。

当前 shape retarget 只比较 palm-frame 内每指四段归一化骨向量，thumb 独立六关节；
shape-only 阶段禁止 metric fingertip target 和 Object6D。wrist translation 初值严格取
MANO wrist0，碰撞响应最多平移 10 mm、旋转固定 0°。人手 required finger 必须同时
满足正确 MANO 每指全骨段对有限 Y 轴圆柱的 3D 邻近和 2D 投影邻接。KaiHand 则从正确
finger chain 的全部 CAD visual surfaces 求最近 signed/absolute SDF；全 visual vertices
加确定性 2 mm triangle samples 的 dense minimum 必须 ≥-1 mm。自碰只检查同手不同机械
手指的 visual-triangle exact SAT；palm/base↔finger、同 finger 内 link pair 按部署语义
排除，twice-area `<=1e-9 m²` 的零面积 facet 在 broadphase 前排除。

f345/f402 均通过 arm 10 mm/5°、qhand limits、required contact、dense nonpenetration 和
distinct-finger SAT；root correction 均 9.9 mm/0°。f345 required thumb+index，最近 contact
为 +1.499990/+1.499998 mm，all-link dense minimum +0.510345 mm；f402 required thumb，
最近 contact 与 dense minimum 均 +0.620999 mm。状态仅为
`PASS_F345_F402_CPU_READY_FOR_WINDOW_EXPANSION`；GPU/renderer 调用 0/0，未运行 10-frame
CAD/物理连续性、新 full460 或正确门下的 002/035/065 cross-session。

正确 human-intent timeline 仅是标签：460 帧中 251 帧任一 finger required；thumb
`90–152,186–192,232–254,283–300,317–438`，index
`63–76,125–142,288,329–335,340–345,356–362,370–375`，middle `59–69`，
ring/pinky 无 required。10 张原始证据帧是
`[0,59,108,143,251,331,367,402,438,459]`；这些不是 460 帧 Robot contact 解。

安全回归只运行测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_robot_wrist_kai_adapter.py tests/test_robot_contact_geometry.py
```

预期 `30 passed`。历史 `run_gate.py` 直接把结果写回自己的 root，禁止原地重跑；若要
replay，必须把 byte-pinned `run_gate.py` 与 `PROTOCOL.json` 复制到不存在的新 root，
保持输入/资产/阈值不变并冻结新路径。当前 pins 见 manifest 和证据索引。

### 7. Robot 跨会话窗口 initializer successor 与 temporal audit

最新 successor 是 `_run/robot_cross_session_window_initializer_dp_cpu_20260902_v1`。其 generic initializer 在每段最早 24 个双腕有效帧上，用逐坐标位置中位数和 proper chordal mean orientation 得到人腕目标，再以固定 natural-bend q prior 拟合该 session 唯一的 `T_world_rig`；下游继续用固定六 seed、最多四个 anchor/backward track 和二阶 DP。它不复制 004 外参，也没有 per-frame/per-side base 或 session patch。

历史实际命令：

```bash
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_window_initializer_dp_cpu_20260902_v1/run_window_init_dp.py
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_temporal_blocker_audit_cpu_20260902_v1/audit_temporal.py
```

两条命令均为 CPU-only（`CUDA_VISIBLE_DEVICES=''`），Python 3.11.11、NumPy 1.26.4、OpenCV 4.11.0。历史脚本绑定历史 root，不得原地重跑；safe replay 必须复制 producer 到不存在的新 root，只改输出目标并冻结新的 producer SHA。

原 successor 中：002 数据门和双臂 downstream 都 PASS；未调参 smoke 035/065 双臂 downstream PASS，但 `TrackerState=notAccurate`，故 session HOLD；068 两侧及 139 side 0 有 temporal blocker。后继 event rescue 保持同一 session `T_world_rig`、门限与 12-frame 窗口，以首帧/故障帧/末帧多起点双向延拓和二阶 DP 找到冻结硬门内分支：068 side 0/1 的 `(v,a)` 为 `(0.076701,0.017366)`、`(0.093358,0.032418)`；139 side 0 为 `(0.067900,0.011624)`。独立 QA 88/88 PASS，并用 pinned URDF limits 重算。

历史实际命令：

```bash
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_event_rescue_cpu_20260902_v1/run_event_rescue.py
/usr/local/bin/python3.11 /mnt/workspace/code/chaoyang/_run/robot_cross_session_event_rescue_independent_qa_20260902_v1/run_independent_qa.py
```

这些脚本绑定历史输出根，不是 safe replay 命令。035/065/068/139 的 `TrackerState=notAccurate` 与 terminal HOLD 全部保留；所有五段 `contact_required_frames=0`，所以只有 downstream window PASS，没有接触/抓取 claim，也没有 sealed/full-video/formal generalization claim。

## 环境基线

- 通用 CPU：Linux 5.10.134，Python 3.11.11，NumPy 1.26.4，OpenCV 4.11.0。
- 新任务合同/几何：Pillow 11.3.0，jsonschema 4.26.0，pytest 9.1.1；几何另用 OpenCV 4.11.0.86 标识。
- Mask：`/usr/local/bin/python3.11` 3.11.11，PyTorch 2.9.1+cu128，CUDA runtime 12.8，NumPy 1.26.4，OpenCV 4.11.0，Pillow 11.3.0，FFmpeg 4.4.2；GPU 为 NVIDIA H20、driver 570.133.20。v4 chips 用时 210.151 s、峰值 26,962,155,008 B allocated / 30,247,223,296 B reserved；poker 用时 234.969 s、峰值 28,479,868,416 B / 31,608,274,944 B。早期 wearable 同形状 replay 的 6.08/6.61 GB 只属于自己的进程，不能替代这些 v4 authority。
- Clean L3：`humanego` Python 3.11.15，NumPy 2.4.6，OpenCV 5.0.0，Pillow 11.3.0，CPU-only；该环境没有 pytest，封存证据使用直接调用 3 个测试函数、py_compile、Ruff、artifact validator 和 ffmpeg 全解码。chips frame0 bounded successor 同样为 CPU-only、无学习权重，只允许 fresh output root 的 frame0 replay。
- Robot CPU successor：Linux 5.10.134，Python 3.11.11，NumPy 1.26.4，OpenCV 4.11.0，GPU/renderer 0/0；正确 MANO/Kai adapter+geometry 回归为 30/30 PASS。Robot preview：Blender 4.5.0 build `8cb6b388974a`，Cycles/OPTIX，4 spp；视频 QA 依赖 ffprobe/ffmpeg，但当前 f345/f402 successor 尚未调用 renderer。

版本变化不自动判失败，但必须写入新 manifest 并重新跑所有相关门；未记录的变化一律先 HOLD。

## 恢复优先级

1. 冻结原始输入和协议；先验证 SHA，不修输出。
2. 恢复合同/schema，再恢复几何；缺字段就 fail closed。
3. 恢复 raw Mask provider；不允许用后处理伪造语义闭包。
4. 恢复同源 Clean donor；缺 K/c2w 或直接观测就保持 unsupported。跨机位 static 失败时优先补录同机位 2--3 秒 clean plate，不用继续堆 Poisson/PatchMatch/quilting/projective 变体。
5. 恢复 Robot CPU 求解和 successor audits；先验证 MANO21 order authority，再过姿态、限位、正确指链接触、dense nonpenetration、自碰、时间和弯臂门。两个 keyframe PASS 后先扩小窗口，不能直接跳到 full460。
6. 最后才做 GPU 推理和 Cycles 视频；可视化永远不能替代数值 authority。
7. 复算 bytes/SHA、完整解码、JSON 解析和本目录索引；只有全部一致才 seal。

## 权利与发布边界

SAM3.1 code/checkpoint 随本地 `SAM License`，分发需连同该许可并满足其条款。Robot 的 Tianji/KaiHand/法兰/URDF/CAD/mesh 再分发权尚未确认。项目自有脚本和用户素材没有在本证据集中发现统一发布许可证。因而本指南授权本工作区内复现，不授权对外打包；具体状态见 `ALGORITHM_AND_WEIGHT_MANIFEST.json`。
