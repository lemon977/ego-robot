# Camera / World → Tianji Base、TCP 与安装标定历史审计

更新时间：2026-09-13（Asia/Shanghai）  
性质：只读历史证据与代码路径审计  
结论状态：`AUDIT_COMPLETE_NO_EXECUTION_GRADE_EXTRINSIC_MOUNT_OR_TCP_CALIBRATION_FOUND`

## 1. 一句话结论

仓库中能找到一条数学闭环的 **development-only world-first** 链路，以及多组视觉放置、任务拟合和几何代理矩阵；但没有找到可支持真实 Robot 或接触精度声明的 ego camera/world→Tianji base 外参、Tianji→KaiHand 安装标定或 TCP 实测标定。

因此：现有结果可以继续用于带水印的开发可视化和内部几何检查，不能写成真实机器人标定，不能据此给出“最终接触精度 X mm”，也不能提升为 Robot authority。

## 2. 本审计做了什么、没有做什么

本审计系统检查了：

- 现行与历史 Robot 生产脚本中的 camera/world/base/tool/hand 变换链；
- Tianji URDF 的固定关节和 nominal FK；
- 历史 `T_camera_base`、`T_world_base`、`T_tool_hand_root` 候选；
- NaturalV2、法兰环和 KaiHand 安装代理合同；
- 单帧与 48 帧用户视觉裁决的授权边界；
- 2026-09-13 contact canary 的外部阻塞条件；
- 候选矩阵的有限性、齐次末行、旋转正交性、行列式，以及统一方向后可比候选间的旋转/平移差。

本审计没有：

- 修改中央 authority；
- 运行 Robot、IK、渲染或接触流水线；
- 选择一个历史视觉矩阵作为物理标定；
- 从图片或旧视频反推缺失的真实安装参数。

机读证据见同目录 `ROBOT_COORDINATE_INSTALL_CALIBRATION_AUDIT.json`。

## 3. 坐标约定和当前代码链

统一约定：

```text
p_A = T_A_B @ p_B
```

当前开发时序链的代码证据支持以下 world-first 关系：

```text
HaWoR 相机系手位姿
T_camera_hand(t)
        │
        │ c2w(t)
        ▼
T_world_hand(t) = c2w(t) @ T_camera_hand(t)
        │
        │ 固定的开发用 T_world_base
        ▼
T_world_base @ FK_base_tool(q(t)) @ T_tool_hand_root
        │
        └─ 逼近 T_world_hand(t)

仅为逐帧叠加渲染：
T_camera_base(t) = inv(c2w(t)) @ T_world_base
```

直接代码证据：

- `tools/render_same_side_world_temporal_review.py:594-602,633,747-750`
- `tools/run_same_side_world_fullsession_successor.py:315-322,335,432-435`
- `tools/render_development_robot_review_v1.py:37-40,49,114,126`
- `tools/run_robot_motion_transfer_arm_canary_v2.py:112-118,132-137,166,200`

需要特别防混淆：

- `tools/render_poker_same_side_outward_frame0.py` 是旧的单帧 camera-first 开发路径；
- `tools/run_robot_renderer_eevee_fullchain_t1.py:930,954` 记录的是 `T_camera_base` session constant 的旧 renderer 语义；
- `tools/audit_robot_contact_visual_two_session.py:297` 会读取 `T_camera_robot_base`，但目前没有一条获授权的正式 Robot trajectory 可以把它变成真实安装外参；
- `render_development_robot_review_v1.py` 会对新 session 用 `c2w_new(0) @ accepted_camera_base` 重建 `T_world_base`，然后施加 session-specific backoff。这是视觉运动迁移策略，不是跨 session 的物理 world→base 标定。

## 4. Camera / World → Tianji Base 候选

所有下表矩阵自身都通过基本 SE(3) 数值检查。这个事实仅说明矩阵形式合法，不说明来源真实。

| 候选 | 方向 | 来源/适用范围 | 数值状态 | 裁决 |
|---|---|---|---|---|
| V2C03 | `p_camera=T_camera_base p_base` | 历史开发视觉放置 | det=1，正交误差约 `2.1e-17` | 仅保留 base 朝向和 P3 臂姿态；旧手安装、尺寸和覆盖不获接受；非标定 |
| P2H | 同上 | 旧 `grap_a_cap_002` home diagnostic | det=1 | 明确为 `UNCALIBRATED_EXTRINSIC_DIAGNOSTIC_NOT_CANDIDATE` |
| frame0 PnP | 同上 | 单帧拟合 | SE(3) 合法；重投影 RMSE `88.803 px` | 高误差且被拒绝 |
| Poker task translation V2 | 同上 | `play_cards_0902_042` 任务平移 | 与 V2C03 同旋转 | 用户拒绝；仅任务视觉拟合 |
| Poker frame0 V8 | 同上 | Poker042 frame0 → 48 帧 lineage | SE(3) 合法 | 视觉接受只授权短开发视频；`current_robot_authority=false` |
| Chips frame0 world V2 | `p_world=T_world_base p_base` | Chips034 frame0 → 48 帧 lineage | SE(3) 合法 | 同上；且 `c2w(0)` 近单位阵，单帧不能验证头动抵消 |

### 4.1 关键候选数值

V2C03 历史视觉放置：

```text
[[ 0.000000,  1.000000,  0.000000,  0.000000],
 [-0.866025,  0.000000, -0.500000,  1.500000],
 [-0.500000,  0.000000,  0.866025,  0.250000],
 [ 0.000000,  0.000000,  0.000000,  1.000000]]
```

Poker042 frame0 V8 的视觉接受矩阵：

```text
[[ 0.000000, -1.000000,  0.000000, -0.052118],
 [ 0.866025,  0.000000, -0.500000,  0.755818],
 [ 0.500000,  0.000000,  0.866025, -1.142485],
 [ 0.000000,  0.000000,  0.000000,  1.000000]]
```

Chips034 frame0 world V2 的视觉接受矩阵：

```text
[[ 0.000000, -1.000000,  0.000000, -0.000200],
 [ 0.866025,  0.000000, -0.500000,  0.794181],
 [ 0.500000,  0.000000,  0.866025, -1.187824],
 [ 0.000000,  0.000000,  0.000000,  1.000000]]
```

### 4.2 候选差异

| 对比 | 相对旋转 | 平移向量差 | 解释边界 |
|---|---:|---:|---|
| V2C03 vs P2H | 5.000° | 0.403 m | 两组旧开发放置已明显不同 |
| V2C03 vs rejected frame0 PnP | 97.262° | 0.979 m | PnP 同时有 88.803 px 重投影误差，不可用 |
| V2C03 vs Poker task translation V2 | 0.000° | 0.863 m | 后者保留朝向但按任务拟合平移，且已拒绝 |
| V2C03 vs Poker frame0 V8 | 180.000° | 1.580 m | outward/same-side 视觉约定发生大改变 |
| P2H vs rejected frame0 PnP | 94.402° | 1.277 m | 与历史审计数值一致 |
| Poker frame0 V8 vs Chips frame0 world V2 | 0.000° | 0.078886 m | 数值旋转相同、平移差 78.89 mm；二者属于不同 session/world 范围 |

最后一行不能解释成两条 session 的“标定误差 78.89 mm”。它们的 world 原点、任务放置和视觉拟合范围不同，只能说明保存的开发矩阵数值不同。

## 5. Tianji nominal FK、法兰与 TCP

`assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf` 的 SHA256 为：

```text
3c3bdfa9aa397c55dea2b3bc94d42c3081d4292d041b1ff43573d175d5faf309
```

URDF 中可核验的固定关节包括：

| 关系 | xyz (m) | rpy (rad) | 解释 |
|---|---|---|---|
| base→onboard camera | `[0.0369887, 0, 0.48752]` | `[0, 0.785, 0]` | 机器人自带相机，不是人类 ego 相机，禁止充当 ego `T_camera_base` |
| base→left arm base | `[0, +0.026, 1.121]` | `[-1.57, 0, 0]` | nominal URDF geometry |
| base→right arm base | `[0, -0.026, 1.121]` | `[+1.57, 0, 0]` | nominal URDF geometry |
| link7→flange | `[0, -0.088, 0]` | `[1.5708, -1.5708, 0]` | nominal URDF geometry |
| flange_L→left_tool | `[0, 0, 0.145]` | `[3.14, -1.57, 0]` | nominal tool link，不是实测 TCP |
| flange_R→right_tool | `[0, 0, 0.145]` | `[-3.14, -1.57, 0]` | nominal tool link，不是实测 TCP |

`assets/robot/tianji/calibration/` 当前文件数为 0。`ROBOT_ASSET_PIN.json` 也把 calibration 标记为需要时补齐，而不是已经完成。

所以必须区分：

```text
URDF nominal FK / fixed joint
        ≠
实体 Robot 的 TCP、零偏、安装和负载后的几何标定
```

本审计没有找到 TCP 球心/尖点标定、flange→TCP 实测矩阵、关节零偏辨识、base/world 外参或实体误差统计。

## 6. Tianji Tool / Flange → KaiHand 候选

### 6.1 当前开发几何代理

历史 C1 的方向是：

```text
p_flange = T_flange_hand_root @ p_hand_root
```

其核心假设为同侧根平面代理：

```text
T_flange_hand_root = diag(1,-1,-1,1), z = 75.199987 mm
```

把 C1 通过 Tianji URDF 的 `T_flange_tool` 统一换算为：

```text
p_tool = T_tool_hand_root @ p_hand_root
```

得到的左右矩阵与 `ASSEMBLY_CLOSURE_AUDIT.json` / 后续 accepted development state 中的 corrected 矩阵逐元素完全一致：

```text
max_abs_difference(left)  = 0
max_abs_difference(right) = 0
```

这证明了**代码/几何链内部来源一致**，不证明实体安装准确。

统一后的 translation norm 为约 `69.8000 mm`。小的 `0.00159265` / `0.00079633` 旋转项来自 URDF 中 `3.14` 和 `1.57` 的截断角，不是实测安装偏差。

### 6.2 静态闭包能证明什么

内部静态几何审计得到：

- 双侧 interface position gap：`0.0 mm`；
- 双侧 interface rotation closure：`0.0°`；
- 左/右 tool forward azimuth：约 `-36.050° / +36.050°`；
- 左/右 elevation：约 `-0.027° / -0.027°`。

这些值只证明“选定的数学平面和 nominal FK 可闭合”。该证据明确缺少：

- KaiHand 真实 adapter CAD；
- 法兰/手端螺孔匹配；
- keyed tangent clocking；
- 左右实体安装变换；
- 碰撞与线缆路径证明；
- 可部署装配证明。

NaturalV2 STL 已被用户识别为“另一只手的腕部法兰”，不能拉伸、改名或宣称为 Tianji→KaiHand adapter。黄色法兰环也是程序化圆环代理，不是实物 CAD 或标定件。

### 6.3 与历史 mount 候选的数值差异

将可比较候选统一到 `p_tool=T_tool_hand_root p_hand_root` 后：

| 候选 vs corrected proxy | 左旋转 / 平移差 | 右旋转 / 平移差 | 状态 |
|---|---:|---:|---|
| C2 NAS 20260823 零平移 | 119.921° / 69.800 mm | 179.954° / 69.800 mm | 旧任务视觉 mount；非测量，未选 |
| C3 `-43.282 mm` | 89.954° / 82.160 mm | 89.954° / 82.160 mm | 43.282 mm 是 flange→mesh rear-plane envelope，不是安装变换 |
| V2C03 legacy mount | 179.954° / 101.718 mm | 179.954° / 101.718 mm | 旧 mount 后续已被手朝向/接口审查替代 |

历史候选间出现接近 90°、120°、180° 的旋转差，进一步说明它们不是围绕一个已知物理标定的小扰动，而是不同视觉约定/代理定义。

## 7. 用户裁决的精确边界

### Poker042 frame0 V8

- 状态：`SINGLE_FRAME_ACCEPTED_SHORT_VIDEO_AUTHORIZED`
- 接受：Poker pose、白色 KaiHand 和 same-side outward camera 的单帧开发视觉；
- 授权：一条短 world-first review video；
- 不授权：full session、training、deployment、Robot authority。

### Chips034 frame0 world V2

- 状态同上；
- 接受：该 session 的 frame0 pose/color/world-chain 开发视觉；
- 授权边界同样仅为短 review video。

### Poker042 / Chips034 temporal48 V3

- 状态：`SHORT_TEMPORAL_VIDEO_ACCEPTED_PARALLEL_CONTINUATION_AUTHORIZED`
- 接受：前 48 帧 world-first same-side 的视觉表现；
- 授权：有依赖门的开发续作；
- 明确字段：`current_robot_authority=false`、`downstream_authorized_now=false`、`deployment_authorized=false`。

因此，“用户看起来接受”不能被改写为“外参/安装/TCP 已标定”。

## 8. 真实与仿真安装现状

| 项目 | 仿真/内部是否有模型 | 外部实测是否完成 | 当前允许的表述 |
|---|---:|---:|---|
| Tianji arm FK | 有 URDF nominal model | 未见关节零偏/实体 FK 误差标定 | 内部 FK |
| ego camera/world→base | 有多组视觉 placement | 否 | development placement |
| flange→tool | URDF 有 nominal 145 mm fixed joint | 未见 TCP 实测 | nominal tool frame |
| tool/flange→KaiHand | 有 root-plane geometry proxy | 否 | geometry proxy |
| NaturalV2→KaiHand assembly | 有 STL 和数学平面闭包 | 缺 adapter CAD/安装测量 | interface concept only |
| contact signed distance | 数字模型可计算 | 无真实 TCP/安装/物体真值闭环 | hypothesis/internal metric only |

## 9. 当前硬阻塞与下一步最小证据

`20260913_contact_robot_v1_canary_v3/BLOCKER_ROBOT_EXECUTION_CLOSURE.json` 已给出正确的硬门：

1. 提供经过核验的 NaturalV2 flange→KaiHand 装配变换，或真实 adapter CAD + 实测安装变换；
2. 完成 world→Tianji-base 标定，并明确该 world 是否就是 HaWoR session world；
3. 完成 Robot TCP/安装标定；
4. 获得以上证据后，只重跑冻结的四帧 retarget gate；通过前不要启动 24 帧或全片 Robot；
5. 不得从历史 visual proxy 猜测缺失变换。

建议外部标定交付最少包含：

- 坐标系图和每个矩阵方向；
- 左右臂分别的 `T_base_flange` / `T_flange_tcp` / `T_tool_hand_root`；
- adapter CAD 的版本、SHA、装配孔位和 keyed clocking；
- world/标定板→ego camera 与 world/标定板→Robot base 的共同观测；
- 残差、重复性、测量次数和单位；
- 标定适用的设备序列号、装配版本、session 范围和日期；
- 独立验证帧，而不只报告拟合帧残差。

## 10. 最终裁决

```text
当前 development coordinate chain：代码和内部闭环支持
物理 camera/world→base 外参：缺失
物理 flange/tool→KaiHand 安装标定：缺失
物理 TCP / 实体 FK 精度标定：缺失
Robot execution：BLOCKED_EXTERNAL
current_robot_authority：false
```

本报告不得作为 Robot 执行许可、真实安装图纸或接触精度证明。
