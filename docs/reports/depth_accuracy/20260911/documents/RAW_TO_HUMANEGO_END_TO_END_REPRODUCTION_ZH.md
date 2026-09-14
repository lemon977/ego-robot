# Raw → HaWoR → Depth → Mask → Object6D → Clean → Robot → HumanEgo：端到端复现与质量门

<!-- BEGIN TWO_TASK_78_NINE_HOUR_EVIDENCE -->

## 两任务78流程九小时证据状态

> 证据刷新：`2026-09-11T00:25:04+08:00`；总状态：`IN_PROGRESS_EVIDENCE_GATED`。这里不使用模型记忆；仅消费已落盘、SHA-256复算通过的`RESULT.json`、`AGENT_REVIEW.json`或`STATE.json`。九小时directive SHA：`a2640f5df02d362937246e5e11b6114b0d99652cd8dd5f167f396dd471a2bdc7`。

| 必交付项 | Chips | Poker |
|---|---|---|
| Mask 78交付 | `COMPLETE_TERMINAL_MIXED`：Chips 78条Mask双lane均有诚实终态：role B7/C71，task-object B72/C6；C不晋升，fresh successor继续提高A/B覆盖。；当前活动`HOLD`：PAUSED_BY_USER at safe boundary: no Poker8 full run, role metadata resume, or Clean78x2 closure launched; role parent PID1617658 and completed-child launcher PID1622219 are SIGSTOP; GPU Clean watcher/workers exited and lease released after authorized Clean203 terminal. | `COMPLETE_TERMINAL_MIXED`：Poker 78条Mask双lane均有诚实终态：role B47/C31，task-object B49/C29；C不晋升，fresh successor继续提高A/B覆盖。；当前活动`HOLD`：PAUSED_BY_USER at safe boundary: no Poker8 full run, role metadata resume, or Clean78x2 closure launched; role parent PID1617658 and completed-child launcher PID1622219 are SIGSTOP; GPU Clean watcher/workers exited and lease released after authorized Clean203 terminal. |
| Clean 78交付 | `HOLD`：PAUSED_BY_USER after in-flight Clean203 reached terminal B. Authorized Clean subbatch is 4/4 B and explicitly synthetic; no further Clean work or 78x2 terminal-mixed index started. | `HOLD`：PAUSED_BY_USER after in-flight Clean203 reached terminal B. Authorized Clean subbatch is 4/4 B and explicitly synthetic; no further Clean work or 78x2 terminal-mixed index started. |
| Robot 78交付 | `HOLD`：PAUSED_BY_USER at safe boundary: two-task numeric preparation and four hand A/B PNGs preserved; zero Robot authority/sidecars, no batch launched, no matching process remains; resume requires new user instruction and hand visual approval. | `HOLD`：PAUSED_BY_USER at safe boundary: two-task numeric preparation and four hand A/B PNGs preserved; zero Robot authority/sidecars, no batch launched, no matching process remains; resume requires new user instruction and hand visual approval. |
| HUMAN_RAW_RGB checkpoint | `HOLD`：PAUSED_BY_USER: HumanEgo watcher stopped at WAIT_ROBOT_ACTION_SIDECAR; zero training/checkpoints; resume requires a new instruction. | `HOLD`：PAUSED_BY_USER: HumanEgo watcher stopped at WAIT_ROBOT_ACTION_SIDECAR; zero training/checkpoints; resume requires a new instruction. |
| ROBOT_VIEW_RGB checkpoint | `HOLD`：PAUSED_BY_USER: four-checkpoint visual A/B branch stopped before bundle/training; exact watcher count is zero. | `HOLD`：PAUSED_BY_USER: four-checkpoint visual A/B branch stopped before bundle/training; exact watcher count is zero. |
| 同帧同action推理对比 | `HOLD`：PAUSED_BY_USER: real comparison remains ungenerated; tested CPU tools are preserved as preparation only. | `HOLD`：PAUSED_BY_USER: real comparison remains ungenerated; tested CPU tools are preserved as preparation only. |

机读状态：[`CURRENT_9H_STATUS.json`](../../tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/evidence_governance/CURRENT_9H_STATUS.json)；追加式事件账本：[`9H_EVENT_LEDGER.jsonl`](../../tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/evidence_governance/9H_EVENT_LEDGER.jsonl)。

完成口径：Mask、Clean、Robot均需两任务各78条诚实终态；含C时记`COMPLETE_TERMINAL_MIXED`且C不得晋升，真正78条全过才记`COMPLETE_PASS`。训练需四个checkpoint（两任务×两RGB域），只消费明确A/B并满足16/3会话与256/48配对H50门；随后需逐任务同帧、同action推理可视化对比。

<!-- END TWO_TASK_78_NINE_HOUR_EVIDENCE -->
更新时间：2026-09-10 22:48 +08:00

## 2026-09-11 六会话全程复核后继（current）

六会话 current 入口已切换到 [`20260911_six_session_full_pipeline_video_review_v2`](../../tasks/control/runs/20260911_six_session_full_pipeline_video_review_v2/INDEX_ZH.md)。旧 `20260911_six_session_pipeline_video_review_v1` 经审计同时使用 15 fps、`trim=12s`、`-stream_loop -1` 和 48 帧 Robot 输入，只能称为 `SUPERSEDED_REVIEW_ONLY` 的循环快速预览，禁止再作为完整任务或 current 输入。

### HaWoR V3 时序后继

输入是同会话 V2 `HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz` 与原始 1280×960/30 fps 视频；输出为 `HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz`、真正 bounded-only 的全片、V2/V3 48 帧 A/B、严格 `frame_id=0..N-1` 的 `FRAME_MANIFEST.jsonl` 和逐会话 `RESULT.json`。唯一机读指针是 [`CURRENT_SUCCESSOR_MANIFEST.json`](../../tasks/control/runs/20260911_six_session_hawor_temporal_successor_v1/CURRENT_SUCCESSOR_MANIFEST.json)，消费者必须同时绑定 NPZ 路径与 SHA，不能 fallback 到旧 RAW-vs-bounded 叠加视频。

V3 只在每个连续 observed segment 内，对 root translation、root SO(3) 和 15 个 pose SO(3) 做置信加权三阶差分拟合；missing gap 两侧完全分开。统一门限是：2D 重投影 P95≤2 px、max≤10 px；动作空间跨度保留≥90%；root translation 更新≤6 mm；root/pose 旋转更新≤4°/6°；jerk P95 不回归。050/203/224 的 jerk P95 下降约 26%–44%，140/182 轻度下降约 3%–13%，243 因 2D 守卫选择 `alpha=.03` 而基本不变。六条均为数值通过、待用户视觉确认，不自动构成 Robot authority。

旧六会话 `01_HAWOR_CURRENT.mp4` 的真实内容是 RAW 细暗线和 bounded 亮线叠加；视觉上的部分“抖动”来自仍被画出的 RAW。该文件不得再命名为 bounded-only/current。完整根因、输入输出、六条指标和运行环境见 [`HaWoR temporal successor README`](../../tasks/control/runs/20260911_six_session_hawor_temporal_successor_v1/README_ZH.md)。

### Robot motion-transfer 后继

旧六会话开发 renderer 硬编码了 `motion_scale=0.35`：

`target_root_translation(t) = root(0) + 0.35 * (human_wrist_world(t) - human_wrist_world(0))`

因此人手已经到达物体时，机器人腕部只执行约 35% 的位移；这不是 IK 正则项造成的主要压幅。current 后继改为：

1. 从 HaWoR V3 读取 `joints_3d_world(t)`，保持 world-first 去头戴相机运动；`motion_gain=1.0`。
2. 使用通用 deterministic full-trajectory placement selector，为每条采集轨迹选择一个整段固定的 base backoff；选择器不读取 session ID、门限一致，且整段内 `T_world_base` 严格不动。当前输出依次为 Chips050 `0.2595 m`、Chips140/182 `0.200 m`、Poker203/224 `0.260 m`、Poker243 `0.258 m`。这只是固定摆放机器人底座，hand target 不变；禁止逐帧移动整机追手。
3. Arm 对 observed contiguous segment 做正/反向 lookahead 和时域可行路径选择；missing gap 不插值、不 hold。六条 arm 均通过 10 mm、5°、branch、0.12 rad/帧速度和 0.06 rad/帧²加速度门。
4. KaiHand 保持拇指 `q[0:6]` 独立求解；index/middle/ring/pinky 分别按 `MCP→PIP→DIP→TIP` 逐段对应，不再用腕到指尖弧长重采样。5/6 会话通过；Chips182 右小指在物理 joint lower bound 上仍有 18.756466° 最大 tip-direction error，115 个 observed rows 保持 `HOLD_HAND_NUMERIC...`，视频明确水印，不伪装成通过。
5. 输出包括每条 arm/hand states、逐帧 manifest、30 fps 全片 Robot review、RESULT 与 SHA。唯一入口见 [`Robot CURRENT_MANIFEST.json`](../../tasks/control/runs/20260911_six_session_robot_motion_transfer_successor_v2/CURRENT_MANIFEST.json)。全部仍为 `authority=false`、`action_sidecar_published=false`。

### 完整视频时间轴合同

六条四阶段复核片严格使用原视频解码顺序 `frame_id=0..N-1`，HaWoR/Mask/Clean/Robot 四路均需 manifest 闭包并解码为相同 N 和原始 30 fps。合成器禁止 `stream_loop`、`loop`、`tpad`、尾帧保持、重复补帧和按时长裁切；真实单侧缺测只能显示 `UNKNOWN_HAWOR_SIDE` 空白。最终帧数为 Chips 302/415/351，Poker 151/144/106，共 1469 帧。当前总状态是 `HOLD_NO_AUTHORITY_FULL_SESSION_REVIEW_DELIVERED`：六条视频可完整播放，但 Chips182 Robot hand 数值 HOLD，且所有 Robot 都仍待用户视觉确认。

## 0. 文档定位

本文是扑克与薯片任务的“从零搭建、复现、审核、扩批”技术说明。它回答四个问题：每个阶段读什么、用什么技术、产出什么、什么条件下才能交给下一阶段。

本文不是运行状态数据库。执行前必须以当前机器权威文件为准：

- 当前阶段 authority：[`BASELINE_AUTHORITY.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_AUTHORITY.json)
- 两条基线复用表：[`BASELINE_REUSE_GUIDE_ZH.md`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_REUSE_GUIDE_ZH.md)
- 人工视频入口：[`CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md`](CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md)
- exact78 批量状态：[`CURRENT_BASELINE_BATCH_STATUS_ZH.md`](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/CURRENT_BASELINE_BATCH_STATUS_ZH.md)
- 中断与自动续跑：[`AUTOMATION_CONTINUATION_HANDOFF_ZH.md`](AUTOMATION_CONTINUATION_HANDOFF_ZH.md)
- 第三方源码、版本与许可证：[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)

任何目录名、文件修改时间、聊天记录或历史 `current` 都不能代替 `RESULT.json + AGENT_REVIEW.json + 精确 SHA256`。

### 0.1 2026-09-10 已确认且必须通用的 Robot 合同

下面是当前唯一允许继续派生的算法与数据处理路径。旧 camera-first 单帧脚本、交叉左右手映射和全片 V1 只保留为失败证据，不能作为 current 输入：

1. HaWoR 逐帧输出 `joints_3d_camera(t)`、`c2w(t)`，并以 `joints_3d_world(t)=c2w(t)@joints_3d_camera(t)` 消除头戴相机运动。
2. 由 world 关节构建 `T_world_hand(t)`；任务级只使用一个固定 `T_world_base`，机器人 IK 满足 `T_world_base@FK(q_arm)@T_tool_hand≈T_world_hand(t)`。
3. `physical-left→human-left`、`physical-right→human-right` 恒为同侧映射。第一视角与整机观察都同向朝外；左臂抬起、右臂靠桌，禁止穿身换边。
4. 覆盖渲染才逐帧计算 `T_camera_base(t)=inv(c2w(t))@T_world_base`。相机运动只改变观察，不改变固定机器人底座或 world wrist trajectory。
5. KaiHand 拇指 `q[0:6]` 独立使用真实 CAD 近节链、端点和 thumb-index 虎口联合优化；四条非拇指链必须按 MANO 的 `MCP→PIP→DIP→TIP` 语义与机器人骨段逐段对应，禁止把 `[wrist, link1..link4]` 整条弧长重采样后错配掌段。
6. 显示合同为双手暖白、机械臂与 NaturalV2 法兰暖象牙白；蓝/红只作为左/右身份标记，不改变运动学。

Poker042 与 Chips034 的连续 0–47 帧 V3 已由用户在 2026-09-10 目视确认，20/20 数值门均通过，因而只授权继续做 fresh 全片开发审核。全片 V1 随后诚实终止为 `FULLSESSION_HOLD_NUMERIC_REVIEW`：Poker 14/18 门、Chips 16/18 门；两者都不是 Robot authority、action sidecar 或训练输入。Poker fullsession V2 保持阈值不变并尝试非拇指 direct/hybrid 骨段审核与局部 branch buffer，但仍失败5门且机械臂位姿退化，因此同样只能作HOLD证据。

并行状态以 [`PARALLEL_CONTINUATION_STATE.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/PARALLEL_CONTINUATION_STATE.json) 为准：Exact78扩批已完成22/22个Depth Grade B和36/36个独立Object6D child Grade B，中央GPU lease已释放；Clean 已原子发布4条/552帧 expanded-role V3 prepare、real-donor与ProPainter specs，但状态为`NOT_RUNNABLE_WAIT_REAL_DONOR_AND_AUTHORITY`且GPU未启动；Poker Robot V1/V2均HOLD；HumanEgo 唯一 watcher 仍为 `WAIT_ROBOT_ACTION_SIDECAR`、`training_started=false`。

### 0.2 0909 handle 新格式转 tracker 合同（当前仅 010）

0909批次已只读枚举001..104共104会话，当前唯一已验证兼容性canary为 [`20260911_handle0909_010_pipeline_compatibility_v1`](../../tasks/control/runs/20260911_handle0909_010_pipeline_compatibility_v1/README_ZH.md)。旧`USER_SWITCH_001_TO_010`只取消了001单条转换请求，不排除001进入未来104会话inventory。010 HDF5 有403个 complete rows，`video_frame_idx=0..402`严格1:1；裸流多解码的第404帧不进入目标时间轴。用户已确认 sourceIndex0 无径向 warp 直通画面；所有 equiDis62/FOV90 二次处理版均为 `NONCANONICAL_INVALIDATED`。K 仅是本会话 factory/source K 按像素缩放，未独立证明为编码域 metric ground truth；D 在 passthrough 编码域不应用。

HDF5 pose 为 REP-103（X前/Y左/Z上），camera extrinsic 输入是 OpenXR 头系（X右/Y上/Z后）；构造 c2w 前必须显式应用 `[[0,-1,0],[0,0,1],[-1,0,0]]`轴基变换。只使用同会话 `raw/camera_params.json`；HDF5 内嵌的 2160×810 stale `video_cam` 不得覆盖实际每眼2048×1536几何。

源提供三类不可混称的手部数据：PICO controller 6DoF（raw左右1377/1377行present，HDF5各403×7全finite）；controller经同会话`controller_to_wrist_calibration`组合的wrist；wrist-local MANUS25（两侧`hand_valid=403/403`）。PICO optical Hand 虽`count=26`，但左右1377/1377行均`isActive=0`且为dummy，不构成PICO21 authority。转换器通过`controller_poses_<session>.jsonl`和逐帧`controller6d`元数据保留controller/wrist/MANUS25，并明确`pico21_authority=false`。

因此 tracker-style 格式成功只授权 visual/development：CPU intake v4 的 G0 媒体和 G1 数值 camera timeline PASS，G2 因无有效PICO21 fail closed，G3 HaWoR 必须等中央 GPU lease 安全后单独实测。在独立标定/算法门之前，不得授权 metric depth、Object6D 绝对尺度、robot-base placement、contact/collision 或训练准入。

## 1. 目标、非目标与当前结论

目标是把同一段第一视角双目采集转换成以下可追溯数据：

1. 人手三维状态：HaWoR/MANO21。
2. 场景可见表面深度：FoundationStereo 公制 optical-Z。
3. 两套不同语义的像素掩码：人体/腕带删除域，以及任务物体身份保护域。
4. 任务物体的可见帧 6D 位姿：Object6D。
5. 可用于视觉输入的去手背景：真实 donor 优先，缺失区由 ProPainter 合成。
6. Tianji 双臂 + 双 KaiHand 的动作、接触与渲染 sidecar。
7. HumanEgo 严格 A/B：`HUMAN_RAW_RGB` 与 `ROBOT_VIEW_RGB` 只改变视觉 RGB，动作、时序切分、ICT、种子和训练配方保持完全相同。

不在当前证据范围内的结论：

- HaWoR 不是毫米级手部真值。
- 双目深度不是接触真值。
- Object6D 的 B 级结果不是触觉或成功标签。
- ProPainter 像素只属于视觉合成背景，不能反喂 Depth、Object6D、接触、IK、标定或真机部署。
- 当前没有可下游使用的任务 Robot authority，也没有本轮 HumanEgo checkpoint。

当前两条基线会话为：

- Chips：`get_potato_chips_0902_034`，293 帧。
- Poker：`play_cards_0902_042`，171 帧。

两条会话的 Raw、HaWoR、Depth、两套 Mask、Object6D、Clean 已形成 A/B authority；Robot 尚在装配、相机到基座 placement、手性与可达性整改阶段。

## 2. 理想整改管线

![理想端到端整改管线](assets/raw_to_humanego_ideal_pipeline_zh.svg)

图中的关键设计不是简单串行，而是三处并行与三处汇合：

- Raw QA 通过后，HaWoR 与双目 Depth 可并行。
- Mask 必须拆成 `role-removal` 与 `task-object identity` 两条独立语义 lane；两者 A/B 后才能下游。
- Object6D 使用原始 RGB、Depth 与物体身份 Mask；Clean 使用 role-removal 减去物体保护 union。两者都不能消费对方的合成像素。
- Robot 同时读取 HaWoR、Object6D、两套 Mask、原始相机与可选 Clean；Robot 的动作求解不依赖 ProPainter 像素。
- HumanEgo A/B 共享完全相同的 Robot action sidecar；只有视觉输入不同。

## 3. 全局数据与坐标合同

### 3.1 原始图像与帧身份

当前采集的原始双目视频是 4096×1536 side-by-side，每眼 2048×1536、30 fps；送入主流程的 selected-left RGB 为 1280×960。典型单会话目录包含：

```text
session_root/
├── camera_params.json
├── source_stereo/*_stereo.mp4
├── slam_trajectory_*.jsonl
├── trackingData_*.txt
├── clip_manifest.json
└── preprocess/all_data/
    └── 00000/
        ├── rgb.png
        └── training_data.json
```

所有阶段都必须绑定同一个 `session_id`、精确帧集合 `0..N-1`、逐帧 RGB SHA、时间戳、相机内参 `K` 与 `c2w`。不允许通过“视频长度差不多”或文件名相近进行对齐。

### 3.2 为什么流程里会同时出现四种分辨率

这些分辨率不是同一张图被随意改成不同大小，而是对应四个不同的数据域。必须同时记录“像素尺寸、相机模型、内参和坐标变换”，只写宽高是不够的。

| 数据域 | 分辨率 | 实际含义 | 为什么使用这个尺寸 | 是否是最终视觉网格 |
|---|---:|---|---|---|
| 原始传输帧 | 4096×1536 | 左右眼横向拼接的原始 PICO 双目视频 | 保留采集器输出，不做推理前信息丢失 | 否 |
| 单眼原始镜头域 | 2048×1536 | 从拼接帧切出的单只鱼眼；镜头模型为 `equiDis62` | 畸变校正和双目几何必须从原始镜头域出发 | 否 |
| selected-left 主流程域 | 1280×960 | 左眼去畸变后的 90° 水平视场虚拟针孔相机 | HaWoR、Mask、Object6D 可视化、Clean、Robot 合成和 HumanEgo 共用一个 4:3 主网格 | 是 |
| stereo-rectified 全尺寸临时域 | 1280×960×2 | 左右眼分别去畸变，并旋转到共同极线坐标系 | 保证同一三维点主要只在水平方向有视差 | 否 |
| FoundationStereo/depth-grid | 640×480 | 上一行的左右图各缩小 0.5 后送模型；输出同尺寸视差与 optical-Z | 像素数是 1280×960 的 1/4，显著降低显存和批量耗时；深度只承担可见表面几何，不承担最终 RGB 细节 | 否 |
| ProPainter 工作域 | 960×720 | Clean 中仅对 UNKNOWN 删除区做视频补全的内部网格 | 兼顾时序模型开销和细节；推理后回到 1280×960 并重新套物体保护与 source-map | 否 |

`selected-left` 不是把 2048×1536 左眼直接 `resize` 成 1280×960。预处理先为一个 1280×960、水平视场 90° 的虚拟针孔相机构造射线，再通过 `camera_params.json` 中的 `equiDis62` 内参和畸变系数反投到原始左眼，用 `cv2.remap` 采样。当前主相机内参为：

```text
K_selected = [[640,   0, 639.5],
              [  0, 640, 479.5],
              [  0,   0,   1  ]]
```

这也是为什么“1280×960”和“原始单眼2048×1536”不能只按宽高比例互换：前者已经换成无畸变的虚拟针孔相机，视场和像素射线均由新 `K` 定义。

深度使用的 640×480 是 stereo-rectified 1280×960 的严格半尺度。OpenCV 的像素中心约定对应：

```text
u_full = 2 × u_depth + 0.5
v_full = 2 × v_depth + 0.5

K_depth = [[320,   0, 319.5],
           [  0, 320, 239.5],
           [  0,   0,   1  ]]
```

因此深度公式必须使用 `fx=320`，不能使用主流程的 `fx=640`。若错误使用 640，同一视差会得到恰好两倍的错误深度。反过来，把 640×480 深度简单复制为 2×2 像素也不正确，因为 depth-grid 与 selected-left 不仅尺度不同，相机朝向还相差双目极线校正旋转；必须使用第 6 节记录的 registration 变换。

中文审阅视频的尺寸只代表排版，不代表数据分辨率。例如深度审阅视频是 1280×480：左侧是缩到 640×480 的 RGB，右侧是 640×480 深度色图；Robot 的 1920×480 三栏视频同理。任何下游都不得从审阅拼图反推数据，也不得把审阅视频当 master RGB。

### 3.3 坐标命名

- `camera`：selected-left 相机坐标，+Z 为 optical-Z。
- `world`：由每帧 `c2w` 定义；`p_world = R_c2w p_camera + t_c2w`。
- `depth-grid`：校正后的 640×480 左目深度网格。
- `selected-left`：1280×960 的主视觉网格。
- `robot-base`：Tianji URDF 的公共基座坐标。
- `tool/flange`、`hand-root`：必须由显式 4×4 刚体变换连接，不能靠渲染时“看起来接上”。

刚体矩阵必须有限、末行为 `[0,0,0,1]`、旋转满足 `RᵀR≈I` 且 `det(R)>0`。长度统一使用 metre，关节角统一使用 radian。

### 3.4 不可变输入与 no-clobber

正式执行先生成 `RUN_SPEC/INPUT_SNAPSHOT`，其中记录所有输入、代码、模型、schema 的路径、字节数和 SHA256。随后：

1. `--validate-only` 重算 SHA、帧集合与 schema。
2. 写入 fresh staging 目录。
3. 全部质量门完成后原子发布 final。
4. final 已存在时拒绝覆盖；修代码或算法必须创建新版本目录。
5. Grade C、用户否决和 runtime failure 都保留收据，但不得成为 current authority。

### 3.5 全阶段输入、机器输出、审阅产物与 authority 一览

这里把“数据输出”和“审阅产物”分开：NPZ/PNG/JSON sidecar 是下游程序读取的数据；中文 MP4、contact sheet 和说明文档只用于人看，绝不能反向当作训练或几何输入。`RESULT.json` 说明一次运行做了什么，`AGENT_REVIEW.json` 说明人工/代理审阅结论，`BASELINE_AUTHORITY.json` 才决定哪一个版本是 current。

| 阶段 | 必需输入 | 下游机器数据 | 审阅/证明产物 | 本阶段可授予的最大范围 |
|---|---|---|---|---|
| S0 Raw QA | 原始双目视频、相机参数、tracking/SLAM、clip manifest | `rgb.png`、逐帧 `training_data.json`、固定帧集合与 SHA 闭包 | 解码/帧数/相机检查收据 | 只证明帧、RGB、时间和相机身份可复现 |
| S1 HaWoR | selected-left RGB、K、c2w、HaWoR/MANO 模型 | 双手 `joints_3d_camera/world`、2D、MANO 参数、observed/confidence | raw-vs-bounded 全片视频、RESULT、REVIEW | 人手视觉/运动候选；不是毫米级真值 |
| S2 Depth | 原始左右眼、双目标定、FoundationStereo checkpoint | 每帧 disparity、optical-Z、valid；depth→selected 注册矩阵 | 深度中文视频、registration 复核、RESULT | `VISUAL_OBJECT6D_CANDIDATE_INPUT`；不授权接触 |
| S3 Mask A | RGB、HaWoR、腕/掌提示、SAM3.1 | 四类 role mask 与 `clean_removal` | 全片 role-mask 视频、manifest、RESULT、REVIEW | Clean 删除域/Robot 人体遮挡输入 |
| S3 Mask B | RGB、任务动作身份锚、SAM3.1 | 各 physical object 的 observed/valid/identity mask | 身份全片视频、manifest、RESULT、REVIEW | 物体保护与 Object6D 输入；不等于 6D pose |
| S4 Object6D | 原始 RGB、object mask、Depth、registration、K/c2w | `T_object_to_camera/world`、valid、near/far、size | 轴/box/深度叠加视频、RESULT、REVIEW | 可见帧视觉 6D；遮挡帧保持 invalid |
| S5 Clean | RGB、role removal、object protection、真实 donor、ProPainter | 1280×960 clean frames/master、逐像素 source map | Clean 全片视频、困难窗/contact sheet、RESULT、REVIEW | 视觉背景；synthetic 像素严禁反喂几何 |
| S6 Robot | HaWoR、Object6D、两套 Mask、K/c2w、URDF/CAD、可选 Clean | `arm_q`、`hand_q`、SE(3)、contact/collision、Robot RGB/depth、action sidecar | 静态装配图、关键帧/全片视频、RESULT、REVIEW | 只有 current Grade A/B 且 downstream=true 才可交 HumanEgo |
| S7 HumanEgo | 成对 RGB 分支、同一 Robot action/object/HaWoR sidecar、split/recipe | bundle、H50 windows、checkpoint、训练日志与指标 | epoch-0、配对一致性和评估报告 | 只回答严格 A/B 视觉输入差异 |

每个 handoff 引用文件时统一使用：

```json
{"path": "/absolute/path", "bytes": 123, "sha256": "64 lowercase hex"}
```

只给相对路径、只给文件名或只给目录 mtime 均不算闭包。逐帧数据还必须带 `frame_id/source_frame`，使数组第 `i` 项不会被误当作原始第 `i` 帧。

## 4. S0：Raw QA 与输入闭包

### 输入

原始 stereo 视频、selected-left 解码帧、相机参数、SLAM/c2w、采集清单和会话身份。

### 方法

- 完整解码视频，核对分辨率、fps、精确帧数。
- 对 `preprocess/all_data/00000..N-1/rgb.png` 建精确集合与聚合 SHA；缺帧、多帧、重复帧均拒绝。
- 把每帧 `training_data.json` 中的 `K/c2w` 与 frame id 绑定。
- 检查左右眼顺序、双目同步、相机参数和 baseline。
- 对任务、日期、会话目录做 cohort identity 检查，禁止跨 clip 混用。

### 质量门

| 门 | 要求 |
|---|---|
| 帧闭包 | 恰好 `0..N-1`，无缺失、重复或额外帧 |
| 视频 | 可完整解码；分辨率、fps、帧数与 manifest 一致 |
| 相机 | 每帧 `K/c2w` 有限，矩阵形状与坐标合同一致 |
| 双目 | 左右眼顺序、校正参数、metric baseline 有证据 |
| 身份 | session/task/date 与 cohort 一致 |

缺少公制双目标定的会话仍可继续 Raw、HaWoR 与 Mask，但 Depth、Object6D、Robot geometry 必须终止为 `C_CALIBRATION_MISSING`，不能借用另一会话标定。

### 4.1 Raw 输入字段与落盘产物

以 Poker042 为例，原始会话根目录是 `/mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_042/`。Raw QA 不生成新的视觉内容，它把采集数据规范化成所有后继都能精确引用的帧级接口。

| 输入/来源 | 关键字段 | 用途 |
|---|---|---|
| `CameraRecord_<session>.mp4` / `source_stereo/*.mp4` | 4096×1536、fps、codec、frame count | 原始 SBS 双目字节与完整解码证据 |
| `camera_params.json` | `left/right` 内参和畸变、`extrinsics`、reference resolution | 去畸变、校正、公制 baseline 与左右眼身份 |
| `slam_trajectory_<session>.jsonl` | 帧/时间对应的相机位姿 | 构造或核验逐帧 `c2w` |
| `trackingData_<session>.txt` | tracking index、时间、PICO 手/腕数据 | 时间同步、旧 tracking 兼容信息；不是 HaWoR authority |
| `clip_manifest.json` | `selection/pieces/files/video/coordinate_system/calibration_sha256_by_session` | clip 身份、裁剪来源、坐标与标定闭包 |

逐帧目录 `preprocess/all_data/{frame:05d}/` 的机器接口为：

| 文件 | 数据/schema | 下游消费者 |
|---|---|---|
| `rgb.png` | 1280×960、8-bit BGR/RGB 文件字节；manifest 另存 decoded-pixel SHA | HaWoR、Mask、Object6D 可视化、Clean、HumanEgo raw 分支 |
| `training_data.json.metadata` | `idx/ts/video_time_s/tracking_index/tracking_sync_error_ms/w/h/fps/k/d/c2w/camera_eye/camera_model/world_coordinate_system` | 所有时序与 camera/world 变换 |
| `training_data.json.obs` | `rgb_path` 等源引用 | 输入定位；消费前需重绑定迁移后的绝对路径并核 SHA |
| `training_data.json.entities.hands.{left,right}` | 21 点名称、3D camera/world、2D、valid/in-image、wrist/palm SE(3)、confidence | Raw tracking 对照与兼容，不替代 S1 HaWoR |

Raw 阶段的“产物”是闭包而不是一条新视频：精确帧集合、每帧 RGB 引用、聚合 SHA、相机/时间身份以及可解析的 `training_data.json`。后续每个 RESULT 都应能沿这些引用回到同一份 raw 字节。

## 5. S1：HaWoR 手部三维与 bounded-v2

### 5.1 使用的技术

HaWoR vendor 源码固定在 [`third_party/HaWoR`](../../third_party/HaWoR)，来源提交与许可证见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。当前项目不修改 vendor 源码；适配与后处理位于 [`tools/run_hawor_bounded_parameter_successor.py`](../../tools/run_hawor_bounded_parameter_successor.py)。

HaWoR 从单目第一视角 RGB 获得左右手检测、跟踪置信度与 MANO 参数，随后通过 MANO forward 得到 21 个关节和网格。当前 MANO21 索引：

- wrist：0。
- thumb/index/middle/ring/pinky tip：4/8/12/16/20。
- 对应 MCP：2/5/9/13/17。

HumanEgo 的 wrist 索引不能直接假定为 MANO wrist；任何重排必须在 adapter 中显式记录。

### 5.2 bounded-v2 做了什么

原始单帧手轨迹会抖动，但对输出关节直接滑动平均会破坏手掌刚体结构。因此当前方法平滑的是 MANO 参数：

- 对每个连续 `observed=true` 区间独立处理，绝不跨不可见 gap 补轨迹。
- 置信度权重为 `clip(confidence, 0.05, 1.0)^2`。
- root translation、root orientation、15 个 pose rotation 与 10 维 beta 分别做二阶差分 Whittaker 拟合。
- 旋转先转为连续符号 quaternion，再平滑、单位化并投回 SO(3)。
- 每只手在每个 observed segment 内重建 MANO21；`camera→world` 通过同帧 `c2w` 重算。

单帧更新硬上限：

- root translation：30 mm。
- root rotation：12°。
- 每个手部 pose rotation：15°。
- beta 更新 L2：0.75。

拟合后以 `α∈{1,.75,.5,.35,.25,.15,.1,.06,.03,.015}` 回溯，直到全分辨率 2D 重投影相对 raw 的 P95 不超过 12 px。未观测帧保持 NaN/invalid。

### 5.3 为什么这样改

历史问题包括窗口间 gauge 漂移、跨遮挡段过度平滑、直接关节点平滑造成骨长变化。当前输入只有单轨迹而没有可核验的重叠窗口输出，因此明确采用 `NO_WINDOW_GAUGE`，不伪称做了窗口融合。bounded-v2 保留单帧证据附近的解，只抑制置信度允许范围内的时间抖动。

### 5.4 输出与质量门

输出 NPZ 至少包含 observed、confidence、MANO 参数、`joints_3d_camera/world`、`joints_2d`、`K/c2w`、frame names；同时生成 RESULT、AGENT_REVIEW 和中文全片 raw-vs-bounded 视频。

| 门 | 要求 |
|---|---|
| 缺失语义 | 不可见帧仍 invalid/NaN，不跨 gap 插值 |
| 几何 | MANO 骨长/手性一致，旋转均在 SO(3) |
| 更新界 | 30 mm / 12° / 15° / beta L2 0.75 |
| 2D 保护 | 相对 raw 重投影误差 P95 ≤ 12 px |
| 世界闭包 | `joints_world == c2w @ joints_camera` |
| 人工可视化 | 左右手、腕点、指尖、掉手和抖动可在全片检查 |

复现入口：

```bash
python tools/run_hawor_bounded_parameter_successor.py \
  --contract /absolute/fresh_contract.json \
  --output-root /absolute/fresh_output \
  --preflight-only

python tools/run_hawor_bounded_parameter_successor.py \
  --contract /absolute/fresh_contract.json \
  --output-root /absolute/fresh_output
```

### 5.5 HaWoR 的精确输入/输出 schema

| 输入 | 最低内容 | 为什么需要 |
|---|---|---|
| Raw frame manifest | 有序 `frame_id → rgb path/bytes/SHA` | 防止推理帧和审阅帧错位 |
| selected-left RGB | `N×1280×960×3 uint8`（按文件序列读取） | HaWoR 图像输入 |
| 相机序列 | `intrinsics (N,3,3)`、`c2w (N,4,4)` | 2D 投影与 camera→world |
| HaWoR raw 输出 | MANO root/pose/beta、左右检测/跟踪置信度及 observed | bounded-v2 的唯一可修改对象 |
| 模型闭包 | HaWoR 代码、checkpoint、MANO 资产的 path/bytes/SHA | 保证同输入能重放相同模型 |

当前 `HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz` 使用左右手顺序 `(left,right)`。以帧数 `N` 表示，其字段是：

| 字段 | shape / dtype | 语义 |
|---|---|---|
| `joints_3d_camera` | `(2,N,21,3) float32` | MANO21，selected-left camera metre |
| `joints_3d_world` | `(2,N,21,3) float32` | 同一点经逐帧 c2w 转到 world |
| `joints_2d` | `(2,N,21,2) float32` | selected-left 像素坐标 |
| `betas` | `(2,N,10) float32` | MANO shape 参数 |
| `root_orient_camera` | `(2,N,3,3) float32` | 根旋转矩阵 |
| `hand_pose_rotmat` | `(2,N,15,3,3) float32` | 15 个 MANO 关节旋转 |
| `root_translation_camera` | `(2,N,3) float32` | 腕/根平移，metre |
| `observed` | `(2,N) bool` | 该手是否由当前帧观测支持 |
| `visibility/provenance` | `(2,N) string` | 可见性与 `OBSERVED` 等来源语义 |
| `detector_confidence/boxes_xyxy` | `(2,N)` / `(2,N,4)` | 检测置信度与框 |
| `intrinsics/c2w` | `(N,3,3)` / `(N,4,4) float64` | 与 Raw 完全同帧的相机数据 |
| `original_frame_indices` | `(N,) int32` | 数组槽位到 raw frame id 的显式映射 |
| `mano_joint_names`、tip/MCP/wrist index | identity-bearing arrays/scalars | 禁止只凭21这个 shape 猜关节顺序 |
| `fps/method/backtracking_alpha/window_gauge_status` | scalar | 时序单位与算法决策 |

一个完整 S1 目录必须同时含：主 NPZ、`RESULT.json`、`AGENT_REVIEW.json`、`HAWOR_RAW_VS_BOUNDED_PARAMETER_中文全片固定尺度.mp4`。NPZ 才是下游输入；MP4 只审查左右身份、重投影、掉手与抖动；RESULT 中的 `outputs` 必须为二者记录 path/bytes/SHA。

## 6. S2：FoundationStereo 公制深度

### 6.1 使用的技术

当前流程使用 FoundationStereo 预测校正双目视差。执行入口为 [`tools/run_exact78_foundationstereo_corrected_depth_worker.py`](../../tools/run_exact78_foundationstereo_corrected_depth_worker.py)，环境检查入口为 [`tools/verify_foundationstereo_environment.py`](../../tools/verify_foundationstereo_environment.py)。

当前两个基线会话的深度不是单目网络“猜尺度”，而是由同步双目视差和相机基线换算得到的公制 optical-Z。模型 checkpoint 固定在 `assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth`；worker 内固定 checkpoint SHA，运行时不匹配就拒绝加载。双目去畸变/极线校正实现来自 [`pico_stereo_depth.py`](../../third_party/FoundationStereo/scripts/pico_stereo_depth.py)，固定几何在 [`calibration.json`](../../tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json)。

当前字节身份如下；来源和许可证边界见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)：

| 组件 | 当前路径/身份 | SHA256 |
|---|---|---|
| FoundationStereo官方来源 | `NVlabs/FoundationStereo`；本地vendor缺少可信独立commit标记，因此不凭记忆填写revision | 以本表文件SHA闭包 |
| PICO去畸变/校正适配 | `third_party/FoundationStereo/scripts/pico_stereo_depth.py` | `a3dd48f22508c603d2ae689ed17af25ff49393d5d440f00de8ae87fbaa4d9284` |
| 当前exact78 depth worker | `tools/run_exact78_foundationstereo_corrected_depth_worker.py` | `dc1c09e847afa88f4dba8bd7d39656aa1afac143bec07e12b7382335ba21fc31` |
| ViT-Large checkpoint | `assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth` | `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1` |
| 模型配置 | `assets/models/checkpoints/foundationstereo/23-51-11/cfg.yaml` | `a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f` |
| 当前双目几何 | `tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json` | `5671aa81cf50d6c53dcbb3e54652e62749b58d4f782fe11d42665de085942768` |

上表是本文更新时间的复现证据，不是允许未来代码静默漂移的“永远正确”常量。正式run仍必须把当次runner、checkpoint、config、calibration和输入manifest重新写入immutable snapshot。

### 6.2 当前深度是怎样逐帧得到的

完整处理顺序如下：

1. 从 4096×1536 side-by-side 拆出两眼 2048×1536。
2. 从 `camera_params.json` 读取左右眼 `equiDis62` 内参、8个畸变参数和各自 4×4 外参。每个相机中心按 `C=-Rᵀt` 计算，左右相机中心距离给出公制基线；当前 calibrated cohort 固定为 `B=0.0637716504026918 m`。
3. 在 1280×960、90° 水平视场的虚拟针孔网格上生成射线；用左右 `rectification_rotation` 把 rectified ray 变回各自原相机射线，再通过 `equiDis62` 投到2048×1536原图，以 `cv2.remap` 得到一对 1280×960 stereo-rectified 图。这个步骤同时完成鱼眼去畸变和极线对齐，不是裁切。
4. 左右 rectified 图各缩小到 640×480，保持顺序 `left, right`。转为 `[1,3,480,640]` RGB tensor，padding 到32的倍数；FoundationStereo 在 `FP16 autocast + inference_mode` 下执行16次迭代，输出 float32 左视图 disparity `d=u_left-u_right`。
5. 采用和 disparity 同一网格的 `fx=fy=320 px`、`cx=319.5`、`cy=239.5` 与公制基线计算：

```text
Z = fx × B / disparity
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

其中 `Z` 是沿 rectified 左相机光轴的深度，不是相机到点的欧氏距离。举例：`d=40 px` 时，`Z=320×0.0637716504/40≈0.5102 m`；同一场景若视差减半到20 px，深度约变为1.0203 m。

6. 建立逐像素有效域：disparity 必须 finite 且 `d>0.25 px`，右对应点 `u-d≥0`，并要求 `0.10m≤Z≤3.0m`。失败像素 `valid=false`、`depth_m=NaN`，不得补成0或邻域中值。
7. 每帧原子写一个 640×480 NPZ，至少含 `frame_id/disparity_px/depth_m/valid/scaled_intrinsics/input_closure_sha256`。恢复运行时先重算该帧公式、shape和closure；一致才跳过，否则隔离旧帧并 fresh 重算。
8. 为了让 Object6D 使用1280×960物体 Mask，另外发布 `REGISTRATION_AUTHORITY.npz`。半分辨率 depth pixel 先按

```text
H_half_to_full = [[2,0,0.5], [0,2,0.5], [0,0,1]]
```

回到全尺寸 stereo-rectified 像素中心，再使用

```text
H_full_rectified_to_selected = K_selected × R_leftᵀ × K_selected⁻¹
H_depth_to_selected = H_full_rectified_to_selected × H_half_to_full
```

映射到 selected-left。3D 向量则使用 `T_stereo_rectified_camera_to_selected_camera` 中同一个 `R_leftᵀ`。这里的 `H` 只负责图像射线/像素注册，公制深度仍由 `Z=fxB/d` 决定；二者不能混为一个缩放操作。

### 6.3 标定从哪里来、如何限制复用

当前固定 [`calibration.json`](../../tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json) 记录左右 rectification rotation、投影矩阵和 `Q`。其来源相机参数为同一硬件配置的 `play_cards_0902_017/camera_params.json`；标定阶段跨8个采样帧获得2661个匹配、1894个内点，内点率约0.7118。正式会话仍必须逐条核对 `camera_params.json`、相机身份、左右顺序、baseline 和精确 SHA，不能仅因任务/日期相近就借用该标定。

当前 034/042 authority 输出分别位于：

- [`Chips corrected metric depth RESULT`](../../data/processed/chips/get_potato_chips_0902_034/depth/20260908_foundationstereo_corrected_metric_batch_v1/RESULT.json)
- [`Poker corrected metric depth RESULT`](../../data/processed/poker/play_cards_0902_042/depth/20260908_foundationstereo_corrected_metric_batch_v1/RESULT.json)

两条都发布完整逐帧 `FRAME_MANIFEST`、registration authority/result 与中文深度视频。其授权范围仅为 `VISUAL_OBJECT6D_CANDIDATE_INPUT`，明确不授权 Robot contact 或毫米级真值。

### 6.4 解决过的问题

旧流程曾把 selected-left 的全分辨率焦距错误代入半分辨率 disparity 公式，使公制深度整体缩放错误。修复要点是：深度公式必须使用“产生 disparity 的那张图”的内参。注册矩阵与深度公式是两个不同合同，不能因改公式而静默改 registration。

另一个容易混淆的问题是：selected-left 和 stereo-rectified-left 虽然都是1280×960，却不是同一个相机朝向。如果直接把640×480深度乘2贴到 selected-left，物体边界会系统偏移。当前使用固定 `R_leftᵀ`、显式 pixel-center 变换和SIFT registration复核来阻断这种错位。

### 6.5 输出、复核方式与质量门

每帧 NPZ 包含 `frame_id/disparity_px/depth_m/valid/scaled_intrinsics/input_closure_sha256`，另有 FRAME_MANIFEST、REGISTRATION_AUTHORITY、REGISTRATION_RESULT、RESULT 和中文视频。

| 门 | 要求 |
|---|---|
| 公式 | 逐像素重算 `Z=320×B/d` 与发布值一致 |
| 数值 | finite、`d>.25`、右对应点在界内、`0.1≤Z≤3.0` |
| 注册 | 采样帧 matches ≥100、median ≤0.5 px、P90 ≤1.5 px |
| 输入 | stereo、camera params、frame identity、代码和 checkpoint SHA 闭包 |
| 范围 | 仅可见表面 optical-Z；不授权接触或毫米级物理真值 |

registration 门的实现不是只检查矩阵可逆：worker在起始、中间、末尾三个采样帧上，从原始左眼重新生成 stereo-rectified 图，再用 `H_full_rectified_to_selected` 预测主流程 selected-left；两张图做SIFT匹配，要求每帧 matches≥100、median≤0.5 px、P90≤1.5 px。该检查验证的是“深度像素能否落到主RGB正确位置”，不是验证FoundationStereo本身在无纹理、反光、遮挡边缘上的深度误差。

中文深度视频右栏是把 `0.10–3.0m` 的 optical-Z映射为颜色，仅供发现错帧、全局尺度和边缘异常。颜色不是原始深度值；机器消费必须读取逐帧NPZ及其SHA闭包。

### 6.6 Depth 的文件级 handoff

| 输入 | 数据要求 |
|---|---|
| 原始 stereo | 完整 SBS 帧，左右眼顺序固定，逐源文件 SHA |
| camera params / calibration | equiDis62 参数、左右外参、公制 baseline、rectification rotations、精确 SHA |
| checkpoint/config | `model_best_bp2.pth` 与 `cfg.yaml` 字节身份 |
| Raw identity | session、`0..N-1`、fps、源视频和 selected-left SHA |

每帧 `frames/{frame:06d}.npz` 的固定接口为：`frame_id () int64`、`disparity_px (480,640) float32`、`depth_m (480,640) float32`、`valid (480,640) bool`、`scaled_intrinsics (3,3) float64`、`input_closure_sha256 () string`。无效深度必须同时满足 `valid=false` 且 `depth_m=NaN`，不能写成0。

整段目录的产物职责如下：

| 产物 | 内容 | 谁读取 |
|---|---|---|
| `FRAME_MANIFEST.json` | 每帧 NPZ 的相对路径、bytes/SHA、valid pixels、depth min/max、disparity-array SHA | Object6D、恢复/完整性检查 |
| `REGISTRATION_AUTHORITY.npz` | dense `depth_pixel_to_selected_rgb_xy`、in-bounds mask、正反单应、rectified→selected SE(3)、两套 K | Mask-depth 对齐与3D反投影 |
| `REGISTRATION_RESULT.json` | 三个采样帧的 SIFT matches/median/P90 与 PASS | authority 审核，不是像素数据 |
| `RESULT.json` | input closure、授权 scopes、manifest/registration/video 引用 | 编排与 current 判定 |
| `*_校正公制深度_逐帧.mp4` | RGB/深度色图审阅 | 仅人看 |

### 6.7 当前方法的可信边界

- FoundationStereo输出是学习式视差估计；低纹理桌面、反光物体、手指细边、遮挡边界和运动模糊可能出现局部错误。
- 当前每帧独立推断，没有把上一帧深度作为当前帧真值；因此不会积累时序漂移，但也不保证时序完全平滑。
- 640×480只提供每个depth pixel的可见表面；映射到1280×960时会有采样空洞，不能靠2×2膨胀伪造高分辨率几何。
- `depth_m` 是左相机可见表面 optical-Z；被手遮住的桌面深度本来就不可见，Clean合成像素也没有对应真实深度。
- Object6D可把物体Mask内的有效深度点做稳健中心/PCA候选，但接触仍必须在Robot阶段结合KaiHand pad、Object6D几何和signed distance单独计算。

## 7. S3：SAM3.1 Mask——必须拆成两条语义 lane

### 7.1 SAM3.1 使用的能力

SAM3 源码固定提交见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)，兼容层在 [`pipeline/sam31_compat_adapter_v1.py`](../../pipeline/sam31_compat_adapter_v1.py)。当前没有把 SAM3 当作“输入一句话就自动全片正确”的黑盒，而是组合使用四种能力：

1. **正/负点提示**：HaWoR 的 wrist、palm、MCP 或动作锚点提供正点；相邻手、背景、非目标同类物体提供负点。
2. **当前帧独立分割**：每次重入、身份变化或周期 refresh 都重新读取当帧 RGB，避免把很久以前的错误 memory 继续传播。
3. **短程 video memory/propagation**：仅用于连续、短时、身份稳定的 masklet，不允许跨长遮挡或跨身份歧义段硬传播。
4. **光流承接**：refresh 之间可用双向 DIS optical flow warp，但只能留在当前腕 ROI/身份 ROI 内；越界、远漂或不可见时必须 empty。

这四者解决了 SAM 视频分割的典型问题：单锚长传播会变薄、漂到桌面、出画后无法重入；相似薯片会被合并；卡牌翻面时颜色和纹理发生变化。核心不是调低阈值，而是让当前帧几何与动作身份不断重新约束模型。

### 7.2 lane A：role-removal Mask

语义固定为四类：`left_human/right_human/left_tracker/right_tracker`。它只回答“Clean/Robot 画面中哪些人手、手臂边缘和腕带像素应删除”，不回答任务物体是谁。

exact78 当前 bounded-v2.1 做法：

- 人手由 HaWoR 骨架范围和 SAM3.1 mask 约束。
- Tracker/腕带以当帧 wrist 与 palm 外推点周期重锚，而不是永远从上一帧质心 refresh。
- 当前 HaWoR 判定出画/不可见时，Tracker mask 必须为空；重入后用当帧 RGB 重新捕获。
- refresh 拒绝时，只允许当前腕 ROI 内的 DIS warp 或 empty，禁止漂到袖口、桌面远背景。
- expected-visible coverage 与全帧 coverage 分开报告，避免把真实离屏帧错误计为漏检。

### 7.3 lane B：task-object identity/protection Mask

这条 lane 只回答“被操作的物理实例是谁、哪些像素必须保护”。身份必须由任务动作和实例历史定义，不能按颜色、最近区域或 mask union 猜。

- Chips：三个薯片是三个独立 `physical_object_0/1/2`。当前 034 定义为初始左上/第一次拿起、初始右上/第二次拿起、初始下方/第三次拿起。相接、遮挡或无法区分时，各实例独立 `observed=false + empty`；union 只用于 Clean 保护，不代表一个物体。
- Poker：动作前锚定被翻开的最右侧牌背，翻牌遮挡段 invalid，动作后锚定同一张 A♦。不允许从桌面任意牌重新选择。

### 7.4 删除域公式

```text
role_union = left_human ∪ right_human ∪ left_tracker ∪ right_tracker
object_union = union(all observed physical task objects)
clean_removal = role_union - object_union
```

Object mask 对删除域有否决权：即使手和物体重叠，物体保护像素也必须保持 raw byte-exact。

### 7.5 质量门

| 门 | role-removal | task-object identity |
|---|---|---|
| 帧/SHA | 精确 N 帧、每帧 raw SHA 一致 | 同左 |
| 二值性 | 每类 mask 必须二值、shape 正确 | 同左 |
| 身份 | 左右物理侧不交换 | ID 不切换、不把同类 union 当实例 |
| 覆盖 | human visible ≥0.95；tracker expected-visible ≥0.80 | observed 帧必须是目标实例；歧义帧允许 empty |
| 漂移 | not-visible 远背景像素=0 | 非目标实例像素=0；实例间互斥 |
| 组合 | removal 等式逐像素成立 | protected union 可复算 |
| 视频 | 全片解码并检查重入、遮挡、边界 | 检查动作前后同一物体和三实例 |

### 7.6 两条 Mask lane 的输入与产物

| lane | 输入数据 | 每帧机器输出 |
|---|---|---|
| role-removal | `rgb.png`、HaWoR `observed/joints_2d/boxes/confidence`、左右 wrist/palm 提示、SAM3.1 checkpoint、上一 accepted masklet（仅短程） | `left_human/right_human/left_tracker/right_tracker` 四张 1280×960 二值 PNG；四类 union；扣除保护域后的 `clean_removal` |
| task-object identity | `rgb.png`、任务/会话身份、动作阶段、physical instance 初始锚点、正负提示、SAM3.1 | 每个 `physical_object_i` 的二值 PNG，以及 `valid/observed/instance_id/identity_state/visibility/confidence/area/centroid` |

role `FRAME_MANIFEST.json` 的每帧至少记录 `source_frame`、源 RGB path/bytes/SHA、decoded-pixel SHA、四个 `role_masks` 引用、role 来源、物体保护引用、最终 removal 引用、`object_pixels_removed_from_deletion_union` 和 overlap 检查。object `OBJECT_MASK_MANIFEST.json` 的每帧至少记录：

```text
frame_id, valid, observed, physical_instance_id, identity_state,
observed_identity, visibility, confidence, area_px, centroid_xy,
source_rgb_decoded_sha256, mask{path,bytes,sha256}, producer_record
```

`observed=false` 时必须输出同尺寸 empty PNG、`physical_instance_id=-1`、无 centroid，而不是省略帧；这样下游能区分“明确不可见”与“文件缺失”。

完整目录的职责分工：

| 产物 | 职责 |
|---|---|
| `INPUT_SNAPSHOT.json` / pins | 输入 RGB、HaWoR、checkpoint、代码的不可变身份 |
| `FRAME_MANIFEST.json` | 源帧到所有 mask 文件的逐帧闭包 |
| `OBJECT_MASK_MANIFEST.json` | physical instance 的 observed/identity 语义 |
| `REUSED_HAND_TRACKER_MANIFEST.json` | 仅在复用已接受 role lane 时说明来源，不可伪装成新推理 |
| `clean_removal_masks_object_protected/*.png` | Clean 实际消费的删除域 |
| `RESULT.json` | 数值门、coverage、soft defects、artifact 引用 |
| `AGENT_REVIEW.json` + `*_MASK_中文全片复核.mp4` | 人工检查出画/重入、错手、漂移、实例切换和物体损伤 |

## 8. S4：Object6D——可见帧几何，不猜遮挡帧

### 8.1 使用的技术

当前实现入口为 [`tools/run_visual_fixed_instance_object6d.py`](../../tools/run_visual_fixed_instance_object6d.py)。它读取原始 RGB 对应的物体身份 Mask、校正 Depth、registration、K 和 c2w；不读取 Clean 合成像素，也不把手指夹持关系当作位姿真值。

单个可见实例的计算：

1. 将 selected-left object mask 映射到 640×480 depth-grid。
2. 取 mask 内有限深度点并反投影到相机三维。
3. 以鲁棒中位数估计中心。
4. 通过 SVD/PCA 得到局部平面法向与长/短轴。
5. 统一法向朝向相机并构造右手系 `T_object_camera`。
6. 用同帧 `c2w` 得到 `T_object_world`。
7. 报告观测点 optical-Z 分位数、固定尺寸 box corner 的 analytic near/far，以及中位 point-to-plane residual。

当前几何先验：

- 薯片：薄椭圆近似，尺寸 `[0.050, 0.038, 0.003] m`，最少 20 个有效深度像素。
- 扑克牌：薄盒近似，尺寸 `[0.088, 0.063, 0.001] m`，最少 50 个有效深度像素。

### 8.2 KEEP_INVALID 策略

当前 authority 明确采用 observed-only：

- object mask 有效且深度支持充分时才发布 pose。
- 遮挡、实例相接、身份歧义或深度不足时写 `valid=false`、`instance_id=-1`，pose 与 near/far 字段不存在。
- `propagated=0`；不通过插值或手部运动推测遮挡帧物体姿态。

这是为了避免“轨迹看起来连续”却在遮挡后换成另一片薯片或另一张牌。

### 8.3 质量门

| 门 | 要求 |
|---|---|
| 输入闭包 | RGB、Mask、Depth、registration、K/c2w 精确 SHA |
| 深度支持 | 有效点数达到实例阈值，点集 finite |
| SE(3) | 旋转正交、`det(R)>0`、单位 metre |
| 坐标闭包 | `T_world_object == c2w @ T_camera_object` |
| 尺寸 | 固定实例几何尺度，不随单帧 mask 面积缩放 |
| 身份 | 全片不换实例；Chips 三实例独立 |
| invalid | `instance=-1`，pose/near-far 缺省，不插值 |
| 可视化 | 中心、轴、box、near/far 与 mask/深度叠加全片审查 |

### 8.4 Object6D 输入/输出与产物

| 输入 | 精确消费内容 |
|---|---|
| Raw | selected-left `rgb.png`、`K (N,3,3)`、`c2w (N,4,4)`、frame id/SHA |
| object Mask lane | 各 physical instance 的 1280×960 mask、`observed/valid/identity`；不读取 role union 猜物体 |
| Depth | 每帧 640×480 `depth_m/valid` 及其 manifest/SHA |
| registration | depth-pixel→selected dense map/单应和 rectified-camera→selected-camera SE(3) |
| geometry contract | task 对应 `object_size_m`、最少深度点数、法向/轴符号规则 |

`VISUAL_FIXED_INSTANCE_OBJECT6D.npz` 以 `N` 帧为轴，至少包含：

| 字段 | shape / dtype | 语义 |
|---|---|---|
| `frame_indices` | `(N,) int64` | 明确对应 raw frame |
| `valid/observed` | `(N,) bool` | 本帧是否发布直接可见位姿 |
| `visibility` | `(N,) float32` | 可见支持度 |
| `physical_instance_id` | `(N,) int32` | 固定物理实例；invalid 为 -1 |
| `T_object_to_camera` | `(N,4,4) float64` | object→selected-left camera |
| `T_object_to_world` | `(N,4,4) float64` | object→world |
| `observed_near_far_optical_z_m` | `(N,2) float32` | mask 内观测点深度范围 |
| `analytic_near_far_optical_z_m` | `(N,2) float32` | 固定尺寸 box 在当前 pose 的解析范围 |
| `object_size_m` | `(3,) float64` | 任务固定物理尺寸，不随 mask 缩放 |
| `provenance` | scalar string | direct/observed-only 方法声明 |

逐帧 `numeric/FRAME_MANIFEST.json` 还记录 `identity_transition/label`、`pose_provenance`、有效 depth pixels、平面残差、RGB/mask/depth bundle SHA 和 frame artifact SHA。目录级 `numeric/RESULT.json` 负责数值门；外层 `RESULT.json` 负责强制审阅事务；`BASELINE_RESULT.json` 才记录 Grade 和 authorized scopes；`OBJECT6D_中文逐帧审查.mp4` 只用于看轴、box、深度支持和身份是否跳变。

## 9. S5：Clean——删除容易，补背景才是难点

### 9.1 像素来源优先级

Clean 不是简单将 mask 置黑。当前 expanded-role V3 的来源优先级为：

1. removal 外原始像素：byte-exact 保持。
2. task-object 保护像素：byte-exact 恢复原始 RGB。
3. 同一 session 的左目时间 donor / 同步右目真实 donor：来源坐标、帧号、眼别逐像素记录。
4. 仍不可观测的 removal 区域：送 ProPainter，标记 `SYNTHETIC_PROPAINTER`。

ProPainter 固定提交与依赖见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。主入口是 [`tools/run_clean_synthetic_propainter_baseline.py`](../../tools/run_clean_synthetic_propainter_baseline.py)，扩域 handoff 是 [`tools/build_expanded_role_clean_handoff.py`](../../tools/build_expanded_role_clean_handoff.py)。

### 9.2 为什么 Mask 正确后仍会有残手

SAM mask 通常覆盖实体手掌，但 Clean 还需消除半透明运动边缘、手臂轮廓、阴影和腕带。V1 曾出现 Poker 左下手、Chips 3 秒后右腕带残留。修复不是放宽 Clean 供体门，而是扩展删除语义：

- Chips：left/right/tracker 扩展约 24/20/60 px。
- Poker：left/right/tracker 扩展约 24/18/60 px。
- Tracker 独立扩展，做对称邻帧稳定交集。
- 扩域后再次减去 task-object protection union，保证三片薯片或 A♦ 不被涂掉。

ProPainter 内部使用 960×720 工作分辨率，最终与原始/真实 donor/source map 合成并发布 1280×960 master。它利用光流传播和视频时序补全比逐帧 Telea 更稳定；但生成结果仍只属于视觉像素。

### 9.3 source map

每个输出像素必须可归类：

| code | 语义 | 几何可用性 |
|---|---|---|
| 0 | `TARGET_RAW_UNCHANGED` | 可追溯原始视觉，不自动等于深度真值 |
| 1 | `REAL_DONOR` | 同会话真实像素，可核 source eye/frame/x/y |
| 2 | `PROTECTED_OBJECT_RAW` | 原始任务物体像素 |
| 3 | `SYNTHETIC_PROPAINTER` | 仅视觉，严禁反喂几何/接触/标定 |

### 9.4 质量门

| 门 | 要求 |
|---|---|
| 修改域 | removal 外改动=0 |
| 物体保护 | protected object 改动=0，逐字节等于 raw |
| 来源 | source map shape/帧数正确，real donor 坐标可回采原像素 |
| master | 1280×960、30 fps、精确 N 帧；不能把 2×2 中文审阅拼图当主背景 |
| 审阅 | 全片与困难窗检查手、手臂、腕带、阴影、闪烁和物体损伤 |
| 合成边界 | synthetic 只供 visual RGB，不授权 Depth/Object6D/contact/IK |

当前删除域中 real donor 占比约 Chips 47.6311%、Poker 22.5920%；其余由 ProPainter 合成。比例低不自动判失败，只要来源诚实、物体保护和视觉质量通过。

### 9.5 Clean 输入、逐像素输出与发布目录

| 输入 | 数据要求 | 禁止事项 |
|---|---|---|
| Raw RGB | `N×1280×960` 原始 selected-left，逐帧 SHA | 不从中文拼图或重编码视频解帧 |
| role removal | 四类 role union 经扩域后的二值删除域 | 不把 object mask 并入删除语义 |
| object protection | observed physical objects 的 union | 保护区最终必须 raw byte-exact |
| real donor manifest | 同 session temporal/stereo 来源帧、eye、source xy、置信/遮挡门 | 不使用跨 session donor |
| ProPainter | checkout commit、三个 weight 文件 SHA、960×720 输入帧/holes | 生成像素不能标作 observed |

prepare 阶段先输出 `prepared_full_resolution/`、`propainter_input/`、逐帧 `source_kind`/removal/protected maps 与 `PREPARE_RESULT.json`。其中每帧要给出 raw/base/source-kind/removal/protected 路径，以及 removal、real donor、synthetic、protected 像素计数。ProPainter 完成后再合成和发布：

| 产物 | 格式/内容 | 下游用途 |
|---|---|---|
| `clean_frames/{frame:06d}.png` | 1280×960 uint8 clean master 帧 | Robot visual composite、HumanEgo 视觉分支（若合同选择） |
| `pixel_sources/{frame:06d}.png` | 与 RGB 同尺寸的 source-kind code 0/1/2/3 | 逐像素来源审计；3禁止进入几何 |
| `SOURCE_MAP_MANIFEST.json` | 每帧 clean/source-map path/bytes/SHA 和四类像素计数 | 下游 provenance 与完整性校验 |
| `CLEAN_SYNTHETIC_MASTER.mp4` | 精确 N 帧主视觉视频 | 便于播放；机器训练仍优先读无损帧序列 |
| `CLEAN_SYNTHETIC_中文全片复核.mp4` | 对照/标签审阅视频 | 仅人看 |
| `PROPAINTER_RUN.log` | vendor 执行、错误/旁路说明 | 运行复现；不是质量结论 |
| `RESULT.json` / `AGENT_REVIEW.json` | 输入闭包、比例、硬门、Grade 与视觉结论 | current/下游判定 |

当前 Poker042 RESULT 的方法字段为 `REAL_TEMPORAL_STEREO_DONOR_THEN_PROPAINTER_FOR_REMAINDER`，171 帧；删除域内 real donor 约 22.592%，synthetic 约 77.408%。这些比例描述来源，不是几何置信度。

## 10. S6：Robot——动作、装配、接触与渲染必须同时闭环

### 10.1 目标输出

Robot 阶段不是只画一层 CAD。正式输出至少包括：

- Tianji 双 7-DoF 手臂 `arm_q`。
- 双 KaiHand 22-DoF `hand_q`。
- wrist/tool/hand-root 的 4×4 轨迹与 validity/confidence。
- 每个手指 pad 到物体的接触距离、接触开关和碰撞指标。
- Robot RGB/alpha/depth 或 range、Object6D 遮挡与最终 Robot-view RGB。
- action sidecar、RESULT、AGENT_REVIEW、全片中文视频和逐文件 SHA。

正式 runner 是 [`tools/run_tianji_kai_robot_baseline.py`](../../tools/run_tianji_kai_robot_baseline.py)，后处理在 [`pipeline/tianji_kai_robot_baseline_post.py`](../../pipeline/tianji_kai_robot_baseline_post.py)。

### 10.2 动作链

```text
HaWoR MANO21
  ├─ wrist/palm orientation → Tianji tool pose target → bounded arm IK
  └─ finger bone directions → KaiHand 22-DoF bounded retarget

Object6D + Kai fingertip pads
  └─ contact / non-contact signed distance

Robot FK + camera + Object6D near/far + raw/Clean background
  └─ z-buffer composite → ROBOT_VIEW_RGB + action sidecar
```

正式时序 Robot IK 必须先消除人头/随身相机运动。本文统一采用 `T_A_B`“把B坐标表示到A坐标”的记法：

```text
T_world_hand_target(t) = c2w(t) @ T_camera_hand_HaWoR(t)
T_world_base            = 一个task/session级固定Robot placement
T_world_base @ FK_base_tool(q(t)) @ T_tool_hand ≈ T_world_hand_target(t)
T_camera_base(t)        = inv(c2w(t)) @ T_world_base       # 只用于每帧渲染
```

等价地，若实现保存的是固定 `T_base_world=inv(T_world_base)`，则 IK 目标可写为 `T_base_hand_target(t)=T_base_world @ T_world_hand_target(t)`。关键不是变量叫 `world→base` 还是 `base→world`，而是它必须在整个task/session内固定，且逐帧 `c2w(t)` 只负责把 HaWoR camera 量转入稳定 world 以及把固定Robot场景变回camera做渲染。

当前用户指定的目标语义是同侧映射：`human-left → physical-left`、`human-right → physical-right`，颜色固定蓝=左、红=右。机器人相机必须从机器人身后与人的第一视角同向朝外；若相机在机器人正面却仍强行对齐同侧腕像素，双臂会穿过机身换边。左右身份、相机朝向、掌面法向和拇指/虎口必须分别检查，不能靠交换数组索引或屏幕左右名称糊在一起。

时序求解使用 previous-accepted-only warm start，防止每帧跳到不同 IK branch。目标不是硬截断关节，而是在 URDF limit、速度和二阶变化的可行域内求最接近目标的姿态。

### 10.3 装配与相机 placement

必须分别证明：

1. 公共 Tianji base、左右 Base/Link1..7、flange/tool 的 URDF FK。
2. 真实腕法兰 CAD 身份。
3. `T_flange_hand` 的位置、旋转、左右手性与实体不穿插。
4. 固定 `T_world_base`（或其逆 `T_base_world`）对当前task/session既保持已确认朝向，也让world手根目标位于双臂可达域；`T_camera_base(t)`必须由`inv(c2w(t)) @ T_world_base`每帧派生，不是时序IK的固定placement。

当前已知边界：NaturalV2 STL 是历史确认的真实腕部法兰，但原本连接另一种机械手；仓库中仍缺少已验证的 Tianji→KaiHand adapter CAD。当前静态平面闭包可以作为 development evidence，不能冒充真机安装标定。

历史 V2C03 相机/基座朝向在视觉上曾被接受，但其原 translation 对 Poker042 公制腕点不可达。正确整改是固定已接受的旋转/双臂 P3 朝向，在同一个 task-level 刚体 placement 内做有界平移/小角求解；不能为左右手分别移动相机，也不能缩放 Robot 来“覆盖人手”。

### 10.4 接触与遮挡

- Object6D 提供物体表面/box 的可见几何，不提供接触标签。
- 接触由 KaiHand 命名 fingertip pad 几何与物体计算：需要接触时目标约 0–3 mm；不接触时 signed distance 不得明显穿透。
- 必须对整手/手臂做 self-collision、Robot-object collision 与 branch jump 检查。
- 渲染时 RobotDepth 与 Object6D near/far 做 z-buffer；被物体遮住的 Robot 像素不能盖在物体前面。
- Clean 只提供视觉背景。即使使用 ProPainter，IK、接触和深度仍从 HaWoR/Depth/Object6D/URDF 计算。

### 10.5 质量门

| 门 | 当前目标 |
|---|---|
| 装配闭包 | flange/tool→hand-root gap ≤1 mm、旋转闭包 ≤1°；无未声明假 adapter |
| 手性 | proper rotation；thumb side、palm normal、screen winding 全部一致 |
| 可达性 | 目标在两臂 URDF 可达域，端点残差不能靠改相机尺度掩盖 |
| 关节 | 所有关节在 URDF limit 内 |
| 时序 | arm step ≤0.12 rad/frame；hand ≤0.08；二阶 ≤0.06；previous-only |
| 分支 | branch jump=0；一步制动/viability 可持续 |
| 碰撞 | 全片固定分母 self-collision/Robot-object 审核 |
| 接触 | 五个命名 pad；contact 与 non-contact signed distance 分开 |
| 遮挡 | RobotDepth/ObjectDepth z-buffer 顺序正确 |
| 视觉 | 基座、臂、法兰、双手、拇指方向、覆盖关系在关键帧和全片可辨 |

### 10.6 当前 Robot 状态

截至本文更新时间，`BASELINE_AUTHORITY.stage_authorities.robot.status` 仍为 `NO_CURRENT_TASK_ROBOT_AUTHORITY`。已撤回的视频不能用于汇报、训练或部署。静态接口审核图：

[`TIANJI_NATURALV2_KAIHAND_静态接口闭包_双侧四视图.png`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/static_interface_closure_audit_v1/TIANJI_NATURALV2_KAIHAND_静态接口闭包_双侧四视图.png)

它证明端口约为水平向前外展、数学接口 gap/rotation 闭合；不证明 adapter 实体、当前会话相机 placement、动作 IK 或接触正确。

两条旧 Poker 24帧视频均已被用户否决并撤回。第一版同侧第0帧虽然左右身份/颜色基本正确，但仍暴露两个独立问题：相机在机器人正面，迫使双臂穿过机身换边，右肘弯到背后；简化 KaiHand 优化器未收敛，手指形态与人手不一致。后续只有显式用户接受的静帧可以作为新短片的首帧来源。

用户认为V3整机姿态基本正常，但要求美化双手/法兰配色，并指出左拇指虎口仍不够张开。V4虽然改变了拇指端点角度数值，但用户复核确认肉眼差异不足，因此V4已撤出current。V5修正近节外展后，用户已明确确认姿态没有问题。当前审核 [Poker042 第0帧白手 V8](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_outward_frame0_pose_v8/POKER_042_第0帧_同侧左右手_同向整机姿态确认.png)与 [Chips034 world-first 第0帧](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_outward_frame0_world_v2/CHIPS_034_第0帧_同侧左右手_同向整机姿态确认.png)。V3/V5/V7/V8 做了以下修复：

- 将 front-facing Robot 相机绕 robot-base 竖直轴转180°，相机位于机器人身后、与人类第一视角同向朝外；
- 用几何 branch barrier 约束左右 `Link3/Link5` 留在各自物理侧，且两个 `Link5.x` 保持在机身前方；
- Poker静态审核仍使用一个共享 `T_camera_base` 和真实 Robot 尺度，不允许 per-hand camera 或 scale；这只是单帧可视化placement，不是正式时序坐标链；
- KaiHand 22-DoF 拆成五条独立指链：拇指 `q[0:6]` 独立6-DoF求解，index/middle/ring/pinky 各4-DoF；
- 旧 thumb adapter 明确记录的是 cross-side 标定，不能套到同侧映射。V3 在 chirality-normalized same-side palm basis 中逐指求解，避免二次旋转拇指；
- V5先锁定V3的机械臂、相机、手根、法兰安装状态、右拇指和全部非拇指关节，只对左侧`q_hand[0:6]`做真实CAD拇指近节链、端点与可见间隙联合优化；
- V7参考用户指定的460帧Robot004视频使用钢蓝灰手、暖象牙白机械臂/法兰；用户随后要求手改为白色，V8只把双手改为暖白，蓝/红只保留为左右身份文字。

V5与V3的`q_arm/T_camera_base/T_tool_hand_root/T_actual_hand_root_camera/robot_hand_roots_base`、右拇指和全部非拇指链逐字节相同，只有左拇指6-DoF发生变化。左侧归一化thumb-index可见间隙由0.1401增至0.2122，人体参考0.2306；左拇指真实CAD端点的投影方向误差由7.426°降至4.604°，同侧解剖基底中的tip方向误差为8.504°。V7和V8的全部状态数组都与已接受姿态的V5 bit-exact，唯一变化是显示颜色。颜色参考视频为`/home/lemon/音乐/current_clean_full_robot_visual_override_4spp_full460 (2).mp4`，12,682,152 bytes，SHA `efc98e79a71b2379793152688346e369c48af1260018e5e3d3ab5a1205ec1888`；V8双手为用户指定白色。用户已接受V8姿态和颜色，其收据只授权一段短world-first视频，`downstream/training/deployment=false`仍保持。

Poker V3/V5/V7/V8代码的实际链是camera-first：手形、手根和手腕target直接读`joints_3d_camera`，IK使用`inv(T_camera_base) @ T_camera_hand_target @ inv(T_tool_hand)`，该路径没有读`joints_3d_world`或`c2w`。对单帧视觉审核这是可接受的局部placement，但如果把同一`T_camera_base`直接延伸到全片，人头/相机运动会混入Robot wrist trajectory。旧`run_newtask_robot_kinematic_canary.py`虽使用了world-first的`T_world_rig`，却仍硬编码已被否决的交叉左右映射，不是current authority。

Chips034新单帧是第一个同时显式保留“world-first + 同侧映射”的静态候选：它从Chips自身`joints_3d_world`重算，保存固定`T_world_base/T_base_world`、第0帧`c2w/T_camera_base`、world/camera目标与实际手根。其`c2w @ T_camera_base == T_world_base`闭包通过，两手world位置IK最大残差0.0175 mm；但Chips第0帧`c2w[0]`与单位阵的最大绝对差只有3.47e-18，因此静帧本身不能证明整段轨迹已去头动。用户已经接受该静帧，并只授权生成一段短视频。

当前多帧证明取两个任务各自连续源帧0–47，源时间基准30 fps，输出12 fps慢放为4秒、1920×480三联画。第0帧`q_arm/q_hand/T_camera_base`必须与各自已接受静帧bit-exact；之后从每帧HaWoR world关节构建手根目标，world base保持常量，整机栏使用固定观察相机，因此相机运动只改变人类视角和覆盖渲染，不改变机器人场景placement。当前结果如下：

| 任务 | 视频 / RESULT | `c2w`相对第0帧最大运动 | `robot_world_base_motion_max` | 末端位置 / 旋转最大误差 | arm速度 / 加速度 | 结果 |
|---|---|---:|---:|---:|---:|---|
| Poker042 | [48帧慢放V3](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/POKER_042_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4) / [RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/RESULT.json) | 8.317 mm / 1.152° | 0 | 0.01295 mm / 0.00340° | 0.05455 / 0.01577 rad·帧⁻¹/² | 20/20，用户已确认；仅授权全片开发审核 |
| Chips034 | [48帧慢放V3](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/CHIPS_034_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4) / [RESULT](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/RESULT.json) | 18.471 mm / 5.535° | 0 | 0.4203 mm / 0.1877° | 0.10813 / 0.05898 rad·帧⁻¹/² | 20/20，用户已确认；仅授权全片开发审核 |

两条`camera_world_base_closure_max_abs`均为4.44e-16；HaWoR `joints_camera → c2w → joints_world`回代误差分别为2.01e-8 m与2.87e-8 m，属于float32关节与float64位姿组合的20–29 nm舍入量。它们证明当前短片实际去除了相机/头部运动，而不是把固定`T_camera_base`直接套到全片。

失败版本不得进入current：Poker temporal V1误把逐帧拇指细化重新算进手根朝向，首帧根旋转偏离已接受状态9.109°；V2改为“已接受首帧根朝向 + human-world掌基相对旋转”。Chips temporal V1把机械臂求解单步误设为0.06 rad，快速右手区间逐帧落后，峰值44.21 mm / 9.21°；V2在不改变0.12 rad/帧审核速度门与0.06 rad/帧²加速度门的前提下，让IK使用完整0.12 rad速度包络，误差恢复到门内。V3让两任务使用同一最终runner，并把误导性的`frame0_camera_base_bit_exact`门改为诚实的绝对误差≤1e-12；Poker实测6.94e-18，Chips为0，`q_arm/q_hand`仍要求真正bit-exact。用户已确认两段 V3；该接受只授权 fresh 全片开发审核，不是 Robot authority、action sidecar 或训练授权。

全片 V1 不能晋升。Poker 171 帧失败峰集中在 frame 90–102：arm world 位置 11.0162 mm（frame101 physical-right）、hand bone 104.886°（frame96 right pinky）、tip 29.827°（frame94 right ring），右臂 branch 从 frame98 到170失败。Chips 293 帧的 tip 峰 15.6319°（frame66 right pinky），右臂最后4帧 branch 失败。后继修复必须使用正确的非拇指骨段语义，并把branch可行性作为跨时域路径问题处理；禁止通过放宽10 mm、60°、15°或 Link5.x 0.04 m 门限获得通过。

Poker fullsession V2 也不能晋升。它把右臂首次branch失败从frame98推迟到103，却在frame102为局部保分支把arm位置/旋转误差扩大到49.0604 mm / 17.9956°；hand bone/tip虽降至92.9914° / 25.2499°，仍未通过60° / 15°门。该结果证明逐帧局部branch penalty和单纯direct/hybrid指骨残差不足以解决问题；下一版必须重新设计跨时域分支可行路径与解剖/机器人指链对应，而不是继续调松阈值。

### 10.7 Robot 的输入与正式输出 schema

| 输入 | 关键内容 |
|---|---|
| HaWoR | 同侧 MANO21、observed/confidence、wrist/palm frame、finger bone directions |
| Object6D | physical instance、visible-frame SE(3)、size、near/far；invalid 帧不得伪补 pose |
| Mask | role removal、object protection、observed/identity；分别消费、权限不合并 |
| camera/world placement | selected-left K/逐帧c2w；一个task/session-level固定`T_world_base`（或逆变换`T_base_world`）；逐帧渲染`T_camera_base(t)=inv(c2w(t))@T_world_base` |
| Robot assets | Tianji URDF、KaiHand 左右 URDF/mesh、NaturalV2 flange、mount/adapter contract，全部 SHA |
| optional visual background | Raw 或 Clean 1280×960；Clean code=3 仅参与 RGB 合成 |

当前 Poker V8 单帧 `FRAME0_STATES.npz` 的实际字段是：`source_frame ()`、`physical_to_human (2,)`、`q_arm (2,7)`、`q_hand (2,22)`、`T_camera_base (4,4)`、`T_tool_hand_root (2,4,4)`、`T_actual_hand_root_camera (2,4,4)`、`robot_hand_roots_base (2,3)`；全部字段与V5/V7 bit-exact。这些字段证明它是camera-first单帧审核，不是正式 action sidecar。

Chips034 world-first 单帧的`FRAME0_STATES.npz`另外保存`c2w (4,4)`、`T_world_base/T_base_world (4,4)`、`T_target_hand_root_world/T_actual_hand_root_world (2,4,4)`和`T_actual_hand_root_camera (2,4,4)`。它用于审核正确坐标语义，仍不是全片 trajectory/action sidecar。

当前48帧开发复核的`WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz`输入与输出关系为：

| 字段 | shape / 语义 | 来源或用途 |
|---|---|---|
| `source_frames/local_frames` | `(48,)` | 原始连续帧0–47及局部输出序号 |
| `physical_to_human` | `(2,)` | 恒为`left→left/right→right` |
| `c2w` | `(48,4,4)` | HaWoR逐帧camera→world |
| `T_world_base/T_base_world` | `(4,4)` | 一个固定task-level placement及其逆 |
| `T_camera_base` | `(48,4,4)` | 只用于覆盖渲染，逐帧等于`inv(c2w)@T_world_base` |
| `T_target_hand_root_world` | `(48,2,4,4)` | world-first双手目标；首帧朝向锚定已接受静帧，后续叠加human-world掌基相对旋转 |
| `q_arm/q_hand` | `(48,2,7)` / `(48,2,22)` | previous-accepted-only有界IK与五条独立指链解 |
| `T_actual_hand_root_base/world` | `(48,2,4,4)` | FK实际根位姿，用于world闭包与残差审计 |
| `T_tool_hand_root` | `(2,4,4)` | 已接受法兰/tool→KaiHand root固定安装变换 |

配套`RESULT.json`保存输入HaWoR、Mask manifest、已接受首帧state/decision、renderer，以及trajectory/video的绝对path、bytes、SHA256；同时保存逐帧arm/hand audit、速度/加速度、world误差、camera/world闭包和20项门禁。视频是人类关节、Robot覆盖、固定整机三个同步栏的审阅产物，不是训练action sidecar。

正式全片 Robot handoff 至少要发布：

| 产物 | 最低字段/内容 |
|---|---|
| trajectory NPZ | `frame_indices/valid/q_arm(N,2,7)/q_hand(N,2,22)`、base/tool/hand-root SE(3)、fps、单位、joint names |
| action sidecar | HumanEgo 所需双臂/双手 action、valid/confidence、source frame、单位和 schema；每帧有限且关节限位通过 |
| contact sidecar | 每个 physical object、每侧五个命名 pad 的 signed distance/contact state 与 Object6D validity |
| collision/temporal audit | self/Robot-object 固定分母、branch jump、velocity/acceleration/viability |
| render bundle | Robot RGB/alpha/depth、object near/far、z-buffer composite、背景来源 |
| `RESULT.json` | 所有输入与输出 path/bytes/SHA、逐门结论、Grade、`downstream_authorized` |
| `AGENT_REVIEW.json` + review video | 关键帧和全片的身份、手形、肘分支、接触、碰撞、遮挡审阅 |

任何缺少 `robot_action_sidecar` 或 `downstream_authorized=true` 的静态图/视频都不能被 S7 bundle builder 接受。

## 11. S7：HumanEgo 视觉 A/B 训练

### 11.1 实验问题

用户要比较的是：同一个任务、同一套 Robot action target 下，将原始人类第一视角 RGB 与 Robot 处理后的 RGB 分别输入 HumanEgo，训练效果有什么差异。

两支定义：

- `HUMAN_RAW_RGB`：同会话原始 selected-left `rgb.png`。
- `ROBOT_VIEW_RGB`：同会话背景 + 任务物体保护像素 + Robot RGBA/depth 合成结果。

以下内容必须 byte-identical：Robot action sidecar、HaWoR/Object ICT、帧集合、H50 window、60/8/5/5 split、预训练权重、seed=7、训练 recipe、优化器和评估帧。唯一计划差异是视觉 RGB。

权威合同：

- [`HUMANEGO_VISUAL_INPUT_AB_COMPARISON_CONTRACT_V1.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/HUMANEGO_VISUAL_INPUT_AB_COMPARISON_CONTRACT_V1.json)
- [`TRAINING_RECIPE_REBIND_CONTRACT.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/TRAINING_RECIPE_REBIND_CONTRACT.json)

bundle builder：[`HumanEgo/tools/build_exact78_visual_ab_bundles.py`](../../HumanEgo/tools/build_exact78_visual_ab_bundles.py)。

### 11.2 数据切分与窗口

每个任务固定 78 sessions：

- train 60。
- validation 8。
- test 5。
- heldout 5，永不进入 optimizer。

HumanEgo 预测 horizon 为 50；可用样本必须有连续 51 帧 metadata、RGB 和双手有效 Robot action。A/B 两支的 window start 集合必须完全相同。

### 11.3 启动门

- Robot RESULT 为 current Grade A/B 且 `downstream_authorized=true`。
- action sidecar finite、单位正确、URDF limits 与时序门通过。
- 每任务至少 train≥16、validation≥3；H50 windows 至少 256/48 才允许早期启动。
- 两支 action/frame/window SHA 完全一致。
- 普通文件 nlink=1，禁止硬链接导致训练时互相污染。
- epoch-0 CPU preflight 先跑；中央 GPU lease 后才启动 optimizer。
- 禁止 `archive/legacy/fallback/Robot-C` 路径。

复现顺序：

```bash
python HumanEgo/tools/build_exact78_visual_ab_bundles.py \
  --task chips --validate-only

python HumanEgo/tools/build_exact78_visual_ab_bundles.py \
  --task chips

python HumanEgo/tools/train_newtask_robot.py \
  --bundle /absolute/chips_human_or_robot_bundle \
  --run-root /absolute/fresh_run_root \
  --tag chips_human_raw_epoch0 \
  --preflight-only

python tools/launch_exact78_visual_ab_training.py --watch
```

### 11.4 HumanEgo bundle、训练输出与当前等待产物

每个可接纳 session 的 batch handoff 必须同时给出以下引用：`session/split/status/current_authority/robot_grade/downstream_authorized`、`robot_result`、`robot_action_sidecar`、`hawor_sidecar`、`object_state_npz/json`、metadata adapter root，以及 `human_raw_rgb_adapter_root` 和 `robot_view_rgb_adapter_root`。每个文件引用仍是 path/bytes/SHA。

两条 bundle 只有 RGB 路径不同：

| 项 | `HUMAN_RAW_RGB` | `ROBOT_VIEW_RGB` | 一致性要求 |
|---|---|---|---|
| RGB | raw adapter `*/rgb.png` | Robot-view adapter `*/rgb.png` | 允许不同，且这是唯一实验变量 |
| metadata | 同一 adapter/同一 frame id | 同左 | byte-identical |
| action | 同一 Robot action sidecar | 同左 | byte-identical |
| hand/object state | 同一 HaWoR/Object ICT | 同左 | byte-identical |
| windows | 同一连续51帧起点集合 | 同左 | window SHA 相同 |
| split/recipe/seed | 60/8/5/5、同 checkpoint/optimizer、seed=7 | 同左 | 全部相同 |

materialize 后预期产物包括：两支 session adapter roots、成对 H50 window index、bundle manifest、pairing proof、训练配置快照、epoch-0 preflight、checkpoint、optimizer state、train/validation/test 指标和最终 A/B 比较报告。所有 checkpoint 必须绑定 bundle/recipe/code SHA；不能只靠 run name 判断是哪一支。

当前准备目录 `training_exact78_visual_ab_prepare_v1/` 已有但尚未训练的产物包括：

- `BATCH_TRAINING_HANDOFF_SCHEMA.json`：约束 READY session 的必要字段和 A/B 两个 RGB adapter root；
- `SPLIT_AND_PAIRING_PLAN.json`：固定两任务各 60/8/5/5、每 session 的 raw SHA 和 shared/branch-only 字段；
- `HUMAN_RAW_RGB_SELECTOR_PLAN.json` / `ROBOT_VIEW_RGB_SELECTOR_PLAN.json`：两支 RGB 选择计划；
- `READINESS.json`、`WAIT_ROBOT_ACTION_SIDECAR.json`、`LAUNCHER_STATE.json`：当前阻塞原因与计数；
- `CPU_TEST_PROOF.json`、`EPOCH0_VALIDATE_ONLY_STATE.json`：builder/epoch-0 的无训练验证证据。

当前状态明确为 `WAIT_ROBOT_ACTION_SIDECAR`、`training_started=false`。这些准备文件不是 bundle、checkpoint 或训练结果；Poker V8 和 Chips world-first 单帧 Robot 图也不能解除等待。

实际 CLI 参数以各脚本 `--help` 与当前 launcher README 为准，不能复制历史 run command 后直接覆盖旧目录。

## 12. A/B/C、人工视频与失败语义

### A/B/C

- Grade A：所有硬门通过，视觉无明显问题，误差优于 A 阈值。
- Grade B：所有硬门通过，可下游；允许轻微不影响身份/结构/用途的问题。
- Grade C：任一硬门失败、视觉发现明显错物体/漂移/残手/装配错误，或输入谱系不闭合。C 必须 `downstream_authorized=false`。

### Agent review 与用户视觉反馈

每阶段必须有 `AGENT_REVIEW.json` 和完整中文视频。数值先过、视觉后发现明显错误时，视觉审核有权把结果降为 C。用户明确否决后要追加不可变 withdrawal receipt，并从 current 视频索引与 authority 移除；不能覆盖旧证据，也不能继续把它描述为“当前基线”。

### runtime failure

导入、CUDA 初始化、字段名、shape 或发布 wrapper 错误属于代码失败，不等同算法质量 C。必须保存 failure receipt、释放 GPU lease、修复后使用 fresh output 重新运行；不得把半成品目录晋升。

### fail-forward 批量

单会话 C 不阻塞其他会话。guardian 应写终态后继续下一条；下游只消费同 session 所需 lanes 全部 A/B 的 join-ready 集合。批量 “RUNNING” 不能写成 “COMPLETE”。

## 13. exact78 批量编排

当前 cohort 共 156 sessions：Chips 78 + Poker 78。manifest 中 97 条具备 calibrated metric stereo，59 条缺标定。批量策略：

1. HaWoR bounded-v2 CPU 全量。
2. role-removal Mask GPU 串行 guardian；task-object identity CPU/GPU 独立 guardian。
3. 同一 session 的 HaWoR 与两条 Mask lane 都 A/B 后，才进入 Depth/Object6D/Clean。
4. 无标定会话在 Mask 后终止 `C_CALIBRATION_MISSING`。
5. Join-ready 会话可流式进入下游，不必等 156 条全部结束。
6. Robot sidecar A/B 达到训练最低数量后，HumanEgo A/B launcher 才从 WAIT 转为 materialize/epoch0/train。

截至 `2026-09-10T22:36:30+08:00`：HaWoR 156/156（A1/B143/C12），role-removal 156/156（B54/C102），task-object identity 156/156（B121/C35）；双Mask同会话A/B交集36，其中calibrated-ready 26、calibration-missing 10。Exact78 expansion 的Chips050 canary和successor21均已`COMPLETE_PHASE`，合计22个Depth结果全为B、36个独立Object6D child全为B，GPU lease已释放。Clean四条/552帧CPU prepare已原子发布，但因等待fresh real-donor producer与authority不可运行GPU；Poker Robot fullsession V1/V2均HOLD；HumanEgo仍等待action sidecar。该快照用于解释数据如何流动，恢复时仍须读取下列状态文件。

分任务必须区分“终态”和“通过”：HaWoR为Chips 78终态=72 A/B+6 C、Poker 78终态=72 A/B+6 C；role-removal Mask为Chips 7 B+71 C、Poker 47 B+31 C；task-object Mask为Chips 72 B+6 C、Poker 49 B+29 C。两套Mask同时A/B为36条，其中只有Chips 7条、Poker 19条具备公制标定并进入已完成的Depth/Object6D。Clean当前只有034/042两条用户接受成片；新增exact78四条Poker只是552帧prepare/spec，不是Clean成片。

实时状态只读以下文件：

- [`STATE.json`](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/STATE.json)
- [`MASK_ROLE_STATE.json`](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_ROLE_STATE.json)
- [`MASK_TASK_OBJECT_STATE.json`](../../tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_STATE.json)
- [`LAUNCHER_STATE.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/LAUNCHER_STATE.json)

## 14. 两条 current 基线凭证

下表只做导航，精确 SHA 以 [`BASELINE_AUTHORITY.json`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_AUTHORITY.json) 为准。

| 阶段 | Chips 034 | Poker 042 |
|---|---|---|
| HaWoR | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/get_potato_chips_0902_034/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/get_potato_chips_0902_034/HAWOR_RAW_VS_BOUNDED_PARAMETER_中文全片固定尺度.mp4) | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/play_cards_0902_042/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/play_cards_0902_042/HAWOR_RAW_VS_BOUNDED_PARAMETER_中文全片固定尺度.mp4) |
| Depth | [`RESULT`](../../data/processed/chips/get_potato_chips_0902_034/depth/20260908_foundationstereo_corrected_metric_batch_v1/RESULT.json) / [`视频`](../../data/processed/chips/get_potato_chips_0902_034/depth/20260908_foundationstereo_corrected_metric_batch_v1/get_potato_chips_0902_034_校正公制深度_逐帧.mp4) | [`RESULT`](../../data/processed/poker/play_cards_0902_042/depth/20260908_foundationstereo_corrected_metric_batch_v1/RESULT.json) / [`视频`](../../data/processed/poker/play_cards_0902_042/depth/20260908_foundationstereo_corrected_metric_batch_v1/play_cards_0902_042_校正公制深度_逐帧.mp4) |
| role Mask | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/mask_fresh_fullsession_v1/get_potato_chips_0902_034/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/mask_fresh_fullsession_v1/get_potato_chips_0902_034/get_potato_chips_0902_034_实例语义MASK_BASELINE_中文全片复核.mp4) | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/mask_fresh_fullsession_v1/play_cards_0902_042_direct_frame_component_v4/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/mask_fresh_fullsession_v1/play_cards_0902_042_direct_frame_component_v4/play_cards_0902_042_对象身份起点SAM31_MASK_中文全片复核.mp4) |
| object Mask | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/chips_multi_object_mask_v2/get_potato_chips_0902_034/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/chips_multi_object_mask_v2/get_potato_chips_0902_034/get_potato_chips_0902_034_三物理实例_MASK_中文全片复核.mp4) | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/action_conditioned_object_identity_v1/play_cards_0902_042/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/action_conditioned_object_identity_v1/play_cards_0902_042/play_cards_0902_042_动作条件同一张牌_MASK_中文全片复核.mp4) |
| Object6D | [`聚合 RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/chips_multi_object_object6d_v1/RESULT.json) / `instances/physical_object_{0,1,2}/OBJECT6D_中文逐帧审查.mp4` | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/object6d_action_conditioned_v2/play_cards_0902_042/BASELINE_RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/object6d_action_conditioned_v2/play_cards_0902_042/OBJECT6D_中文逐帧审查.mp4) |
| Clean | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/get_potato_chips_0902_034_expanded_role_v3/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/get_potato_chips_0902_034_expanded_role_v3/CLEAN_SYNTHETIC_中文全片复核.mp4) | [`RESULT`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/play_cards_0902_042_expanded_role_v3/RESULT.json) / [`视频`](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1/play_cards_0902_042_expanded_role_v3/CLEAN_SYNTHETIC_中文全片复核.mp4) |
| Robot | 无 current authority；[Chips034第0帧](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_outward_frame0_world_v2/CHIPS_034_第0帧_同侧左右手_同向整机姿态确认.png)及[48帧world-first V3已确认](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/CHIPS_034_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4)；293帧V1为HOLD | 无 current authority；[Poker042第0帧V8](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_outward_frame0_pose_v8/POKER_042_第0帧_同侧左右手_同向整机姿态确认.png)及[48帧world-first V3已确认](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/POKER_042_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4)；171帧V1为HOLD |

## 15. 从零复现清单

### 环境与资产

- 核对 Python/CUDA/ffmpeg/ffprobe。
- 核对 `third_party/HaWoR`、`third_party/SAM3`、FoundationStereo 环境、ProPainter checkout。
- 核对所有 checkpoint SHA，不接受同名不同字节。
- 核对 MANO、Tianji、KaiHand、NaturalV2 资产与许可证边界。

### 单会话顺序

1. 建 Raw manifest，完整视频解码并做 K/c2w/frame SHA 闭包。
2. HaWoR raw inference 后运行 bounded-v2；看完整中文视频。
3. 有双目标定时运行 corrected FoundationStereo；验证公式与 registration。
4. 分别运行 role-removal 与 task-object identity Mask；禁止合并权限。
5. 用原始 RGB + object mask + depth 运行 observed-only Object6D。
6. 构建 expanded-role Clean handoff；real donor first，再对 UNKNOWN 运行 ProPainter。
7. 对 Robot 先做静态装配与当前 session 可达性；再做 4 关键帧、24 帧 canary、最后全片。
8. Robot A/B 后原子构建 HumanEgo 两支 bundle；先 epoch-0，再训练。

### 发布检查

- RESULT/REVIEW schema 可解析。
- 所有引用 path 存在，bytes/SHA 重算一致。
- full video 帧数/fps/分辨率正确且 `ffmpeg -xerror` 全解码。
- Grade A/B 才设 downstream true。
- 用户否决、C、runtime failure 均未进入 current 索引。
- staging=0、中央 GPU lease RELEASED 或由当前 holder 正确占用。

## 16. 常见错误与对应修复

| 现象 | 根因 | 正确修复 |
|---|---|---|
| HaWoR 平滑后手指漂离图像 | 直接平滑关节或无界平滑 MANO | 参数空间 SO(3) 平滑 + 更新上限 + 2D 回溯 |
| 深度整体尺度错误 | 用 1280×960 的 fx 配 640×480 disparity | 使用 disparity grid 的 fx=320，再独立注册 |
| Tracker 出画后漂到桌面 | 长程 memory/上一质心 refresh | 当前帧腕/掌重锚 + expected-visible + ROI-bounded warp |
| Chips 把两片薯片并成一个 | 按颜色/连通域 union 追踪 | 三个 action-conditioned physical ID，歧义时 empty |
| Poker 开始没选中或翻面后换牌 | 只从正面 A♦ 锚定，未绑定动作前牌背 | 动作前最右牌背 ↔ 翻转遮挡 invalid ↔ 动作后 A♦ |
| Clean 明明有 mask 仍残手/腕带 | 删除域只含实体手掌，没有阴影/腕带扩域 | 独立 role expansion + temporal symmetry + object subtract |
| Clean 把物体涂掉 | 物体 mask 与删除 mask 混为一体 | `role_union - object_union`，protected byte-exact |
| Robot 手悬空或拇指朝下 | flange→hand 变换/手性共轭错误 | 静态接口闭包 + proper rotation + thumb/palm/winding 门 |
| Robot 覆盖人手但两臂不可达 | 用每手 camera/scale 拟合视觉 | 一个 task-level rigid placement + URDF reachability 预检 |
| Robot 右肘绕到背后/双臂穿身 | 正面相机与 egocentric 同侧像素约束冲突，IK 又没有肘 branch barrier | 相机从机器人身后同向朝外；Link3/Link5 本侧与前向几何门 |
| KaiHand 手指不像人手 | 只拟合掌根/五指尖或22关节联合优化未收敛 | 五条指链独立求解；拇指6-DoF单独算，常规手指各4-DoF，并审 tip/bone/web |
| Robot 接触看似合理 | 从 human/object mask 猜接触 | 用 Kai fingertip pad 与 Object6D 几何计算 signed distance |
| A/B 结果不可解释 | 两支动作或 window 不同 | 原子双 bundle，action/frame/window SHA 完全相同 |

## 17. 文档治理与清理

当前 authority、复用指南、视频索引和本页属于“现行入口”。历史 C、失败 canary 和用户撤回结果可移动到 `archive/`，但必须满足：

- 无 current 文档直链。
- 无运行进程 open FD/CWD。
- 无 current 代码、测试、schema 或 launcher import。
- 移动前后 tree SHA 一致，并提供恢复命令。
- 归档不等于释放磁盘；同盘 move 只减少 active runs 噪声。

临时脚本只有在“无 current import、无测试引用、无文档引用、无活跃进程”同时成立时才能归档。不要为了目录看起来干净而删除仍承担回归测试或谱系证明的代码。

## 18. 进一步阅读

- HaWoR/Depth 数学细节：本页第6–7节；旧拆分文档已移出活动路径。
- 现行完整处理顺序：本页第2–15节；旧重复流程文档已移出活动路径。
- 当前视频导航：[`CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md`](CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md)
- 环境迁移：[`HAWOR_LOCAL_ENVIRONMENT_MIGRATION.md`](../reproducibility/HAWOR_LOCAL_ENVIRONMENT_MIGRATION.md)
- 第三方来源与许可证：[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)

## 19. 2026-09-11 六会话四阶段视频审阅包替代关系

current 入口已经替换为 [`20260911_six_session_full_pipeline_video_review_v2`](../../tasks/control/runs/20260911_six_session_full_pipeline_video_review_v2/INDEX_ZH.md)。旧 `20260911_six_session_pipeline_video_review_v1` 的 `05_FOUR_STAGE_CURRENT_REVIEW_12S.mp4` 经代码审计确认是固定 12 秒、15 fps、循环输入的快速预览，其中 Robot 只有 48 个真实帧；它只保留为失败根因与谱系证据，状态必须是 `SUPERSEDED_REVIEW_ONLY`，不得再称 current 或完整任务。

V2 六条四宫格均按原始 30 fps 和 `frame_id=0..N-1` 播放完整任务，并绑定新的 HaWoR temporal successor 与 Robot motion-transfer successor。Robot arm 为 6/6 数值通过；KaiHand 为 5/6 通过，Chips182 的 115 个右小指 rows 保持数值 HOLD。因此总包状态为 `HOLD_NO_AUTHORITY_FULL_SESSION_REVIEW_DELIVERED`，不是全阶段 PASS。任何用户视觉接受、阶段晋级或 action sidecar 发布都必须回写新的不可变 RESULT/AGENT_REVIEW，不能只依赖聊天记录。

## 20. 量化深度、位姿与 Robot 接触精度的证据边界（2026-09-11）

当前项目的实际外部真值边界为：HaWoR wrist/fingertip optical-Z MAE、P95 与 drift **NOT MEASURED**；FoundationStereo 在 0.3/0.5/0.7 m 的绝对深度误差 **NOT MEASURED**；depth→selected-left 对外部控制点的像素误差 **NOT MEASURED**；Object6D 平移/旋转真误差 **NOT MEASURED**；Robot 物理接触误差预算 **NOT AVAILABLE**。

现有的 HaWoR–PICO 差异、深度公式重算闭合、SIFT 注册残差、Object6D 时序限速、IK/FK 对自身 target 的残差与 mount 视觉 proxy 均为代理比较或内部一致性，不是外部物理真值。这些值跨单位、彼此相关，且没有标定不确定性分布与 Jacobian，严禁直接相加或 RSS 伪造接触精度。

完整的数值、版本/会话区分、指标定义和证据 SHA 见：

- [`REPORT_ZH.md`](../../tasks/control/runs/20260911_quantitative_accuracy_evidence_audit_v1/REPORT_ZH.md)
- [`RESULT.json`](../../tasks/control/runs/20260911_quantitative_accuracy_evidence_audit_v1/RESULT.json)

任何新的“毫米级手/深度/位姿/接触精度”声明，都必须绑定独立标定面/控制点、MoCap/测量治具、实测 flange→KaiHand 变换或力/触觉接触标签之一，并发布独立不可变 RESULT/SHA；否则继续标为 `NOT_MEASURED`。

## 21. HaWoR 表面–Stereo 深度固定回归 QA（2026-09-11）

已新增同 session、同 selected-left camera 域的固定代理 QA：从 bounded MANO 参数重建三角形网格并 z-buffer 得到可见表面 optical-Z，将 FoundationStereo 按当前 `REGISTRATION_AUTHORITY.npz` 注册到同一像素/相机域，主指标为 `delta-Z = Z_mano_surface - Z_stereo_registered`。关节中心仅做闭合诊断，不替代表面指标。

当前 Chips034/Poker042 canary 已通过“有足够有效支持”的 admission 门，并产出左右曲线、差异热图和 RGB overlay MP4。该 PASS **不是精度门**；两路都非外部 GT，残余物体/桌面遮挡仍为 UNKNOWN，Depth schema 无 confidence 时明确记为 absent 而不伪造。因此不能将它用作 Depth/Contact/Object6D/Robot authority，也不改变第20节“外部真值未测”结论。

权威入口：

- [`REPORT_ZH.md`](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/REPORT_ZH.md)
- [`RESULT.json`](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/RESULT.json)
- [`INVENTORY.json`](../../tasks/control/runs/20260911_hawor_stereo_surface_consistency_qa_v1/INVENTORY.json)
