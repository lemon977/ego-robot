# 技术合同草案说明

状态：`DESIGN_ONLY_WHILE_PROJECT_STOPPED`

本文件描述 G1 应固化的语义，不是启动流水线的授权。

## 1. 单一所有权

| 组件 | 唯一输入 | 唯一正式输出 | 不得做 |
|---|---|---|---|
| Source resolver | 只读 RAW、source manifest | 已验证 frame references | 写 RAW、用副本顶替 |
| Session profiler | 已验证 RAW/metadata | `session_context.json`、canary list | 生成正式像素 |
| MASK | RAW、上下文 | `H_core`、arm/forearm evidence | 修物体、写 CLEAN |
| Object evidence | RAW、Object6D/CAD/解析几何 | `O_visible_core`、geometry/depth/SDF | 删手、改 q_hand |
| Reveal router | H/O/U、几何、donor coverage | reveal labels 与 route decision | 直接补图 |
| CLEAN | RAW、reveal labels、donor | 背景/物体 reveal、coverage/confidence | 渲染机器人 |
| Retarget | 冻结 human motion、robot model | 版本化 q_hand/wrist/validity | 写图像、render-time 改 q |
| Base/IK | wrist、robot model、场景几何 | session-constant base、q_arm、adapter | 移动冻结 wrist |
| Renderer | 冻结几何/pose/camera | robot/object RGBA、per-link ID/depth | 修 CLEAN、隐藏失败 link |
| Compositor | CLEAN、render layers、depth | composite 与诊断图 | 改上游 geometry/q |
| Harmonizer | 几何 PASS composite | 仅 robot RGB/窄边界外观 | 改 alpha/depth/object/background |
| QA | 全部 manifest/证据 | 归因、建议、PASS/HOLD | 覆盖任何 producer 输出 |

每个正式 artifact type 在 `pipeline_contract_v1.yaml` 中只能有一个 producer 和一个 schema。并行实验只写 `_run/<run_id>/`，不能成为正式 consumer 的候选路径。

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

SAM3 仅作为证据源：

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

exact78 只忠实消费 R2，不运行 refinement 或 render-time IK。base search 只能寻找一个 session-constant Tianji base，并在固定 wrist 下求 q_arm；不可达即 HOLD。

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

## 9. 公平训练/评测

RAW 与 Robot 主对照必须绑定同一 `PAIRED_KEPT_MANIFEST.json`、split、R2、非图像输入、初始化、seed、batch、epoch、augmentation、validation 选择规则和 checkpoint 选择规则。test8 一次性消费冻结流水线。

速度对齐若执行，必须双域采用相同 frame/window subsampling 与权重，参数只由 train62 和目标机器人统计决定，并单独报告，不替代主对照。
