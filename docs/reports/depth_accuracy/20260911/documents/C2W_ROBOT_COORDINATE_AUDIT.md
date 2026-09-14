# c2w → Robot 坐标链只读审计

审计日期：2026-09-11  
状态：`READ_ONLY_CODE_PATH_AUDIT_COMPLETE`  
代码修改：无  
Robot authority：无；当前产物仍是development review，不是action sidecar

## 1. 直接回答待确认问题

当前六会话Robot开发预览的实际链是：

```text
joints_3d_camera(t)
  └─ 用c2w(t)变换并核验
     joints_3d_world(t)
       └─ 固定的task-level T_world_base下做world-first IK
          q_arm(t), q_hand(t)
            └─ 渲染时 T_camera_base(t)=inv(c2w(t)) @ T_world_base
```

也就是说，它不是“整段固定`T_camera_base`并直接把`joints_3d_camera`送入IK”的旧camera-first路径。当前生产器明确读取`joints_3d_world`，Robot基座在本会话world中固定，只有渲染回相机画面时`T_camera_base(t)`随`c2w(t)`变化。

但这个结论只说明代码路径。它不能证明c2w的真实物理精度，也不能证明当前Robot轨迹已经适合训练或接触。当前`BASELINE_AUTHORITY.stage_authorities.robot.status`仍是`NO_CURRENT_TASK_ROBOT_AUTHORITY`，六会话结果均为`authority=false`、`action_sidecar_published=false`。

## 2. 审计对象与SHA

| 对象 | SHA256 | 角色 |
|---|---|---|
| [`render_development_robot_review_v1.py`](../../tools/render_development_robot_review_v1.py) | `e6b9ef58c9f6117bb7d9bff17ab95fd354035c9c5c70c8fb5e0e9c3005a22eb7` | 当前六会话48帧development producer |
| [`render_same_side_world_temporal_review.py`](../../tools/render_same_side_world_temporal_review.py) | `6228a8d724075a7771d8c751ad4b1c0c7c70a9aa672159ed7316912966c83b6f` | 034/042 world-first时序基线与坐标合同 |
| [`render_poker_same_side_outward_frame0.py`](../../tools/render_poker_same_side_outward_frame0.py) | `cb019366b9b2c036e4e461041834310342f95aa8792193d0d67dd10f8f1d51ec` | 历史camera-first单帧placement/手型路径 |
| [`derive_hawor_world_consistent.py`](../../tools/derive_hawor_world_consistent.py) | `ed6d339e6021c6b30e4e2097c5f6bbca2579221905c7a66f75502b45d615fd30` | `joints_world=c2w@joints_camera`显式派生工具 |
| [六会话Robot开发索引](../../tasks/control/runs/20260911_six_session_pipeline_video_review_v1/robot_development/RESULT.json) | `02a703827ee8ac215363aedd38fe4f41e51385ab9d699da9441b7002405beada` | 六条实际结果及算法合同 |
| [Poker243单会话RESULT](../../tasks/control/runs/20260911_six_session_pipeline_video_review_v1/robot_development/play_cards_0903_243/RESULT.json) | `84ba2f9399b68e4b251da108969d2a83468433b8678c97a7db9a3074e1a78fd5` | 单会话lineage与authority边界示例 |
| [中央BASELINE_AUTHORITY](../../tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_AUTHORITY.json) | `b7835b367126bed6748b79f4d44d426c631268dba096e4938a3b6c77eddb8da6` | Robot仍无current authority |

SHA是本次审计时的文件字节身份；代码后续变化时必须重做审计，不能只复用本文结论。

## 3. HaWoR camera/world数据关系

HaWoR sidecar同时保存：

- `joints_3d_camera[side,frame,joint,xyz]`
- `joints_3d_world[side,frame,joint,xyz]`
- `c2w[frame,4,4]`

显式派生工具的操作是：

```text
p_world = R_c2w @ p_camera + t_c2w
```

即`joints_3d_world = c2w @ joints_3d_camera`。这不是两个独立3D传感器结果：world字段继承了单目HaWoR camera-space的手深度/手形估计，也继承了c2w的尺度、旋转、平移和漂移。

当前world-first runner在使用前再次重建world points，并要求二者最大差异`<=1e-7 m`。注释明确float32 joints与float64 c2w会产生约20–30 nm的舍入。这一道门只证明文件字段和矩阵乘法一致，不证明任一量相对真实世界准确到亚微米。

## 4. 034/042 world-first时序链的逐步证据

[`render_same_side_world_temporal_review.py`](../../tools/render_same_side_world_temporal_review.py)中的关键路径：

1. 第595–601行：从已接受首帧读取`T_world_base`；若旧state没有该字段，才使用`c2w[0] @ T_camera_base`构造一次固定world base。
2. 第604–616行：读取`joints_3d_camera`和`c2w`，重建world points并与`joints_3d_world`做`1e-7 m`内部闭环。
3. 第245–268行：掌面basis和wrist target都直接来自`joints_3d_world`。
4. 第285–307行：IK目标为`inv(T_world_base) @ T_target_hand_world @ inv(T_tool_hand)`。
5. 第312–321行：Robot实际手根通过`T_world_base @ FK @ T_tool_hand`回到world，与world target计算残差。
6. 第633–639行：逐帧计算`T_camera_base(t)=inv(c2w(t)) @ T_world_base`，并验证`c2w(t) @ T_camera_base(t)`保持同一个world base。
7. 第701–716行：输出state同时保存`c2w`、`T_world_base`、逐帧`T_camera_base`、world target和world actual，便于离线复核。
8. 第746–751行：RESULT把上述target/IK/render语义作为显式坐标合同写出。

因此，若头部相机在固定世界中移动且c2w正确，camera-space手坐标的相机运动分量应在world变换后被抵消；Robot base本身不会随头运动。

## 5. 当前六会话development producer的实际链

[`render_development_robot_review_v1.py`](../../tools/render_development_robot_review_v1.py)没有直接把旧042 world坐标绝对值套给所有session，而是采用以下策略：

1. 第38–40行：读取已接受042 state的首帧`T_camera_base`和Robot q/mount；对当前新session建立：

   ```text
   T_world_base_new = c2w_new(0) @ T_camera_base_accepted
   ```

   这把同一个camera-relative首帧placement重锚到每个session自己的SLAM world。不同session的world原点不被假定相同。

2. 第41–49行：手根相对初始位置使用当前session的`joints_3d_world(t)-joints_3d_world(0)`，并乘固定`motion_scale=0.35`；手根方向使用world palm basis相对首帧的旋转。
3. 第49–66行：在固定`T_world_base_new`下做有界Robot IK。
4. 第70–99行：KaiHand手形target读取当前session的`joints_3d_world`。
5. 第126行：叠加回RGB时使用`inv(c2w(t)) @ T_world_base_new`。
6. 第114、132行：输出只发布development states/result；明确不发布action sidecar或authority。

六会话聚合RESULT也明确记录：

```text
joints_3d_camera → c2w(t) → joints_3d_world
→ fixed task placement → same-side Robot IK
```

## 6. 与旧camera-first路径的区别

历史Poker单帧姿态生成器[`render_poker_same_side_outward_frame0.py`](../../tools/render_poker_same_side_outward_frame0.py)确实直接读取第0帧`joints_3d_camera`：例如第396–401行构造手型target，第596–608行用camera wrist和单个`T_camera_base`构造Robot目标。它没有在单帧中使用c2w。

单帧审核本身不存在“随时间把头动带入Robot”的问题，因为没有时间序列；但把同一个固定`T_camera_base`和camera-space joints直接扩展到全片，会让相机/头部运动进入Robot wrist trajectory。当前world-first时序代码没有沿用这种扩展方式。

旧camera-first文件仍可能作为首帧姿态、颜色或camera-relative placement的输入证据；这不等于后续时序IK仍在camera frame中运行。必须区分“首帧seed从哪里来”和“每帧target在哪个坐标系求解”。

## 7. 头部运动问题：已经确认与尚未确认

### 已由代码确认

- 当前时序target是world-space，不是逐帧camera-space。
- `T_world_base`在一个session内固定。
- `T_camera_base(t)`只用于把固定world Robot重投影回移动相机。
- camera/world字段在数值上有严格矩阵闭环。
- 六会话重新锚定各自session的world，不假设不同SLAM world共享原点。

### 尚未由外部证据确认

- c2w的轴定义、metric scale和时间同步相对真实相机运动是否准确。
- c2w在快速头动、低纹理或长时段是否存在漂移/跳变。
- 单目HaWoR absolute-Z变化与c2w组合后，world wrist是否真的稳定。
- `motion_scale=0.35`是开发可视化缩放，不是人体到Robot的物理运动等价映射。
- 每session以首帧camera-relative accepted placement重锚world base，是否符合真实Robot/相机固定安装关系。
- Robot相机到真实Robot base的外参、NaturalV2/KaiHand安装、TCP都没有外部标定。

所以当前不能写“头动问题已经物理解决”。准确表述应是：**代码已经采用world-first语义避免直接把camera frame当固定世界；物理效果仍依赖c2w与camera→Robot安装标定，尚未由外部真值验证。**

## 8. 重要限制：不是训练sidecar

Poker243示例RESULT明确：

- `grade=DEVELOPMENT_ONLY_NOT_AUTHORITY`
- `authority=false`
- `action_sidecar_published=false`
- `frame_count=48`
- `motion_scale=0.35`

六会话聚合结果同样是六条48帧可视化审核，不是全会话Robot action。HumanEgo训练等待器不能把这些开发视频或NPZ自动视为训练authority。

## 9. 下一次代码/数据审核建议

不修改代码前，先补三类证据：

1. 在静态桌面/静态手段中画`joints_camera`、`c2w@joints_camera`和world wrist时间曲线，检查头动是否被抵消。
2. 对c2w做独立相机运动验证，例如AprilTag静态世界或外部SLAM/轨迹参考，报告真实平移与旋转误差。
3. 获得真实`T_camera_robot_base`和hand mount/TCP标定后，判断首帧placement是否还应按session重锚，而不是继续使用视觉对齐参数。

在上述证据前，不将任何内部`world_closure_max`、IK residual或视频视觉重叠写成真实Robot位置精度。

## 10. 审计结论

- **当前代码路径：world-first。**
- **旧单帧seed：存在camera-first来源，但没有被当作全片固定camera target。**
- **头动直接混入风险：代码结构上已通过c2w/world target规避；物理上尚未由外部c2w和安装标定证实。**
- **Robot状态：development review only，无current authority、无action sidecar、不可启动HumanEgo Robot视觉分支训练。**
- **本审计没有修改任何Robot代码或placement。**

