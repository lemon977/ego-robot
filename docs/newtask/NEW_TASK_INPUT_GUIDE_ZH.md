# 新任务输入与采集指南（Robot / Mask / Clean）

这份文档面向以后所有新任务。目标不是让采集流程变复杂，而是一次把真正影响结果的资料收齐，减少“某个视频能手调得很好，换一个会话又失效”的情况。

如果时间很紧，只做“最少必需”即可；如果希望算法容易优化、结果能够迁移，尽量达到“强烈建议”；如果要做正式跨会话评测或长期数据集，按“理想配置”执行。

## 0. 60 秒回答：Clean 不是只拍一张空桌和一张物体

**一张空桌照片 + 一张单独物体照片，只在相机完全固定、任务期间视角不变、曝光和灯光不变时，勉强够做单场景静态底图实验。**它不能告诉算法移动相机后桌面如何对应，也不能提供物体真实 3D 位姿、前后深度或“应该遮住哪根手指”。单独物体照片可以帮助确认外观，不能代替尺寸/CAD 和 Object6D。

### 最低可跑：不要过度采集

如果只想先把一个新任务跑起来，最短拍下面四段：

1. **空桌**：
   - 固定相机：同一灯光和曝光下录 2–3 秒，约 60–90 帧；一张照片只作为不得已的降级方案。
   - 头戴/移动相机：缓慢覆盖任务会出现的视角，录 5–10 秒，并提供逐帧相机位姿 `T_world_camera(t)`；只有空桌照片不够。
2. **物体放在桌上不动**：录 2–3 秒，标为 `OBJECT_CONTACT_FIT`；同时记录物体尺寸/几何和 Object6D。
3. **把物体抬离桌面**：抬高约 5 cm 或更多，停 2 秒，标为 `OBJECT_LIFT_NEGATIVE`。它用于证明算法没有把悬空物体误当成桌面接触。
4. **正常任务**：完整录一次，不需要为了 Clean 故意放慢；保留 RGB、时间戳、相机位姿、Object6D 和对应 MASK。

这套最低配置可以生成当前场景候选，但没有独立 eval，不能声称已经跨会话迁移。若现有 MASK 尚未证明覆盖袖口/腕带，再额外录 5 秒双手和袖口进入/离开画面；若 MASK 已经可靠，则不要求重复录这段。

### 推荐可迁移：只比最低版多三小段

在上面四段基础上增加：

1. **独立接触验证**：稍后把物体重新放到桌面另一位置，静止 2–3 秒，标为 `OBJECT_CONTACT_EVAL`，不能和 contact-fit 使用同一批帧。
2. **第二次正常任务**：独立重复一次，第一次用于 fit，第二次只用于 eval，不看第二次结果调参数。
3. **结束空桌**：录 2–3 秒，用来检查桌子、灯光或曝光是否中途变化。

移动 ego 相机时，桌面边缘放四个固定标志点会明显提高可迁移性；固定相机且绝不移动时，它们是强烈建议而不是最低硬要求。曝光、白平衡和对焦应全程锁定。

### 不同物体还需要什么

- **圆柱**：半径、高度、轴方向、哪一端是底面。
- **扑克牌/薄卡片**：长、宽、实际厚度、正反面、四角编号顺序；不要按圆柱处理。
- **任意形状**：优先给米制 CAD/mesh、单位、原点和接触面。没有 CAD 时，至少给同步深度 + 逐帧 object mask；否则可以做外观候选，但不能可靠验证物体遮挡手指的深度关系。

最短推荐拍摄顺序就是：`空桌 → 接触 fit → 抬离 negative → 正常任务 fit → 接触 eval → 正常任务 eval → 结束空桌`。固定相机约多录十几秒；移动 ego 相机约多录二十秒左右，不需要为每一种片段采很长视频。

---

## 1. 三档交付标准

| 档位 | 适合情况 | 必须交付什么 | 能得到什么 |
|---|---|---|---|
| 最少必需 | 当天快速试做、单任务验证 | 一个完整任务会话；原始 RGB；统一时间戳；相机内参；Robot 的 URDF/mesh/单位、base 和关节数据；左右手与掌心定义；法兰/连接件信息；物体类型与尺寸/Object6D；包含手、前臂、袖口、佩戴物的 MASK 标签或可核验标签子集；同场景空桌、物体接触桌面和抬离桌面片段 | 可以开始做，但只能称当前场景候选，跨会话能力有限 |
| 强烈建议 | 希望减少反复、可以迁移到同类会话 | 两次独立任务重复；四个以上桌面标志点；锁定曝光/白平衡/对焦；相机逐帧位姿；固定世界坐标且无重置；物体接触 fit/eval 分离；抬离 negative；分侧语义标签；fit/eval split 预先冻结 | 可以优化统一参数，并判断是否真的能跨会话 |
| 理想配置 | 正式数据集、跨人/跨衣服/跨物体/盲测 | 独立 fit/eval/test；test 像素封存；同步深度；IMU 重力；物理接触传感器；完整法兰和物体 CAD；相机、Robot、物体在同一世界坐标；标注质检与双人复核；自动 validator 和 SHA256 清单 | 可以做可审计、可复现、可迁移的正式评测 |

最关键的原则：**同一条数据必须能回答“什么时候、哪台设备、哪个坐标系、哪只手、哪个物体、什么尺寸、用了哪份文件”**。如果这些信息只能靠看画面猜，后面很容易出现左右手接反、法兰方向错、物体遮挡深度错、Clean 光影不一致等问题。

---

## 2. 推荐目录：短、固定、可复制

不要把每一次实验参数都塞进目录名。任务目录保持短，版本和参数写进 manifest。

```text
tasks/
└── cap_20260901/                  # task_id：短、稳定
    ├── README_ZH.md               # 任务用白话说明
    ├── task_request.yaml          # 用户目标、优先级、验收条件
    ├── task_manifest.json         # 全任务资产、session 与版本总索引
    ├── split_manifest.json        # fit/eval/test，采集前或看像素前冻结
    ├── checksums.sha256
    ├── assets/
    │   ├── robot/
    │   │   ├── robot.urdf
    │   │   ├── meshes/
    │   │   ├── left_hand.urdf
    │   │   ├── right_hand.urdf
    │   │   ├── flange_left.step
    │   │   ├── flange_right.step
    │   │   └── robot_asset_manifest.json
    │   ├── objects/
    │   │   ├── object_manifest.json
    │   │   └── object_mesh_or_cad/
    │   └── calibration/
    │       ├── camera_intrinsics.json
    │       ├── camera_extrinsics.json
    │       ├── table_landmarks.json
    │       └── gravity_and_world.json
    ├── sessions/
    │   ├── cap_001/
    │   │   ├── session_manifest.json
    │   │   ├── rgb.mp4
    │   │   ├── timestamps.csv
    │   │   ├── camera_pose.npz
    │   │   ├── robot_q.npz
    │   │   ├── object6d.npz
    │   │   ├── events.csv
    │   │   ├── masks/
    │   │   │   ├── class_id/
    │   │   │   └── label_schema.json
    │   │   └── clips.json
    │   ├── cap_002/
    │   └── cap_003/
    ├── labels/
    │   ├── fit/
    │   └── eval/
    └── review/
        ├── 01_robot_arm_position.png
        ├── 02_hand_flange.png
        ├── 03_object_finger_occlusion.mp4
        ├── 04_mask_overlay.mp4
        └── 05_clean_raw_mask_clean.mp4
```

目录规则：

- `task_id` 和 `session_id` 一旦发布就不要改变。
- session 建议用 `cap_001`、`cap_002` 这种短编号；日期、人员、地点、物体写在 manifest，不重复堆在目录名里。
- 不要把 collection token、日期片段或目录中的数字当作 session 身份。例如 `0812` 只能是采集批次，不能偷偷解释成 session 081。
- 原始数据只读；所有处理结果写到新版本目录，不能覆盖原始帧、原始标签或旧结果。
- 给用户看的文件统一放在 `review/` 或仓库 `REVIEW_NOW/`，按 `01_、02_、03_` 排序。

---

## 3. 每个 session 的共同输入

### 3.1 RGB 和视频

最少必需：

- 原始 RGB 视频或逐帧无损/高质量图像。
- 原始分辨率、帧率、编码格式、总帧数。
- 明确第一帧编号，推荐从 0 开始。
- 不允许丢帧后重新连续编号却不留下映射。

强烈建议：

- 保留原始逐帧图片和一个便于浏览的 MP4；MP4 只是预览，逐帧图片是权威输入。
- 锁定曝光、白平衡和对焦，并在 manifest 写明是否锁定。
- 记录是否裁剪、缩放、旋转、去畸变；若处理过，给出从原始像素到当前像素的变换。

理想配置：

- RGB、深度、相机位姿、Robot、Object6D 使用同一硬件时间源。
- 每个流都保留原始设备时间戳和换算到统一时间轴的结果。

### 3.2 时间戳

每一帧至少需要：

```csv
frame_index,timestamp_ns,source_timestamp_ns,dropped,duplicate
0,0,728190013244001,false,false
1,33333333,728190046577334,false,false
```

要求：

- 时间严格单调；重复帧、丢帧必须显式标记。
- 相机、Robot q、Object6D、接触事件之间要报告最大同步误差。
- 不要用“第 100 帧大约对应这个姿态”代替时间同步。

### 3.3 相机内外参

最少必需：

- `fx, fy, cx, cy`、图像宽高、畸变模型和参数。
- 参数对应的分辨率必须与实际帧一致。
- 明确相机坐标轴方向，例如 `+X 向右、+Y 向下、+Z 向前`。

强烈建议：

- 每帧 `T_world_camera(t)`，并明确变换方向：推荐统一写成 `T_A_B 表示把 B 坐标变换到 A`。
- 记录 tracking world 的 epoch UUID、reset/relocalization 日志；一次任务内 `reset_count=0`。
- 用桌面固定标志点做独立重投影检查，不要拿同一批点既拟合又评分。

---

## 4. Robot 必须提供什么

Robot 优化应分两步：**先把双臂位置和自然姿态调对，再调手、法兰环和连接件**。不要在手臂位置还错的时候，用手腕或手指参数补偿。

### Robot 通用性三层

- **全局固定，可跨会话复用**：Robot URDF/mesh/单位/限位、左右装配、法兰和 hand mount、rig 内左右 base 关系、home 构型分支、自然弯曲规则与求解器。
- **每会话初始化一次**：本次 world epoch、桌面/人体录制原点/相机与 Robot rig 的共同关系 `T_world_rig`。不得从 004 或上一 session 直接继承数值。
- **逐帧计算**：`T_world_camera(t)`、手腕/手指目标、`q_arm(t)`、`q_hand(t)`、valid/confidence。

固定物理基座不等于 ego 画面中的 `T_camera_base` 固定。头戴相机下必须使用：

```text
T_camera(t)_base(side)
  = inverse(T_world_camera(t))
  @ T_world_rig
  @ T_rig_base(side)
```

每个新会话建议先录 8–12 秒初始化块；完成后 `T_world_rig` 在同一 world epoch 内冻结，禁止为了贴腕逐帧移动 base。

### 4.1 Robot 资产

最少必需：

- 完整 URDF，以及 URDF 引用的全部 visual/collision mesh。
- 每个 mesh 的单位和缩放，统一推荐米；明确 `1 mesh unit = ? m`。
- joint 名称、顺序、旋转轴、零位、上下限。
- 左臂、右臂、左手、右手的资产身份，不能只靠文件名中的 `L/R` 猜。
- 渲染坐标、Robot base 坐标、法兰坐标、手掌坐标的定义。
- 固定 base 位姿和初始关节 `q0`。

强烈建议：

- visual 与 collision mesh 都提供，并给一张实物 Robot 的正面/侧面参考照片。
- 提供一段已确认自然、安全的双臂参考姿态，记录对应 q；图片只能作视觉参考，q 才是数值参考。
- 提供资产 manifest：文件路径、字节数、SHA256、单位、版本、左右身份、许可证或来源。

### 4.2 左右手、掌心和拇指方向

必须同时写清四件事：

1. 人体数据 axis0/axis1 分别是哪只解剖学手。
2. Robot 左臂/右臂分别安装哪只 KaiHand。
3. 每只手模型的掌心法向、手背法向、指尖方向、拇指方向在手掌坐标系中是什么。
4. 人体手到 Robot 手是同侧映射还是交叉映射；如果是交叉映射，手指 q 不能直接跨左右手复制，必须有经验证的镜像/关节语义变换。

推荐放进 manifest：

```json
{
  "human_axis0": "ANATOMICAL_LEFT",
  "human_axis1": "ANATOMICAL_RIGHT",
  "robot_left_hand_asset": "kaihand_left_v3",
  "robot_right_hand_asset": "kaihand_right_v3",
  "palm_normal_axis": "+Z_HAND",
  "finger_forward_axis": "+X_HAND",
  "thumb_side_axis_left": "+Y_HAND",
  "thumb_side_axis_right": "-Y_HAND",
  "retarget_side_map": {
    "robot_left": "human_left",
    "robot_right": "human_right"
  }
}
```

这段内容必须配一张自动生成的“左右手、掌心、拇指、坐标轴”示意图。只写一个旋转矩阵而没有图，很容易读反。

### 4.3 法兰、连接件和 KaiHand 安装

最少必需：

- 左右法兰和连接件的 CAD/mesh；没有 CAD 时至少提供外径、内径、厚度、孔位、轴向长度。
- `T_flange_connector` 与 `T_connector_hand_root`，包含平移和旋转。
- 明确手腕应连接在哪个实体面、法兰环位于手腕哪一侧。
- 左右件是否镜像，不能默认使用同一个固定 roll。

强烈建议：

- 拍摄装配近照：法兰、连接件、手腕接口、掌心方向同时入镜。
- 给每个接口坐标系画 RGB 三轴，自动检查 hand root 是否落在连接面而不是 mesh 中心或错误端面。
- 提供实物测量尺寸，用于核对 CAD 单位，禁止为了覆盖人手随意缩放真实 Robot CAD。

### 4.4 Robot 时序数据

- `robot_q.npz` 至少包含 `timestamp_ns`、`q`、`joint_names`、`valid`。
- q 的列顺序必须由 `joint_names` 确定，不按“通常是 7 个关节”猜。
- 若有 base/torso/hand q，分组写明，避免 7 轴手臂 q 与 22 轴手 q 混用。
- 固定 base 表示 `T_world_rig` 在一个 world epoch 内固定，不表示 ego 视频中的 `T_camera_robot_base` 逐帧不变。
- 必须写明 `T_world_camera(t)`、`T_world_rig` 和 `T_rig_base_left/right` 的变换方向。
- 任一 world reset 必须建立新 epoch，并用 fiducial 重新绑定 rig；无法重绑就从 reset 帧起 HOLD。

---

## 5. 物体与遮挡输入

物体类型必须显式 dispatch。未知形状应当报错或 HOLD，不能静默按圆柱处理。

### 5.1 圆柱

提供：

- 半径、高度、单位。
- 物体坐标轴和正方向。
- signed bottom、signed top；哪一端是桌面接触底面。
- `T_camera_object(t)` 或 `T_world_object(t)`、时间戳、valid/confidence。

### 5.2 扑克牌或薄卡片

提供：

- 长、宽、厚度，不能把厚度写成 0。
- 正面、背面、厚度轴。
- 四角稳定编号顺序，例如从正面看左上开始顺时针。
- 如果有弯曲，说明是按刚性薄盒近似，还是提供可变形模型。
- 物体 6D 位姿；仅有中心点不够判断遮挡哪根手指。

### 5.3 任意形状

优先提供米制 CAD/mesh：

- metric scale、原点、坐标轴、接触面 ID、封闭性和法线方向。
- 如果没有 CAD，提供同步深度和逐帧 object mask；同时明确它只是观测几何，不冒充精确 CAD。

### 5.4 Object6D 通用字段

```text
timestamp_ns
T_world_object 或 T_camera_object     # 明确方向
geometry_type                          # CYLINDER / CARD_THIN_BOX / METRIC_MESH
valid
confidence
pose_covariance
tracking_epoch_uuid
```

物体遮挡验收至少要有两类动作：

- 物体真实从手指前方经过，应当遮住对应手指。
- 物体与手分开，不能无缘无故遮住手指。

输出不仅要给 2D overlap，还要报告前后深度关系；“画面上重叠”不等于物体真的在手指前面。

---

## 6. MASK 标签：必须把袖口和佩戴物真正纳入链路

标签建议使用整数 class-id PNG；调色板只用于显示，不能把颜色本身当类别权威。

最少必需的分侧类别：

```text
0  BACKGROUND_TABLE
1  LEFT_HAND_SKIN
2  RIGHT_HAND_SKIN
3  LEFT_FOREARM_SKIN
4  RIGHT_FOREARM_SKIN
5  LEFT_SLEEVE_CUFF
6  RIGHT_SLEEVE_CUFF
7  LEFT_TRACKER_STRAP_WORN_ITEM
8  RIGHT_TRACKER_STRAP_WORN_ITEM
9  HELD_OBJECT
10 UNKNOWN_OR_IGNORE
```

规则：

- `hand + forearm + sleeve/cuff + tracker/strap` 合并成最终 human foreground `H`，但原始类别必须保留，便于分别测召回。
- 操作物体必须单独成类，不能并入 H；否则 Clean 会把物体一起删掉。
- 物体合法遮挡手指时，只标画面真正可见的像素，不在物体后面“脑补”可见 MASK。
- 画面边缘的袖子/前臂必须标到可见边界，不能只标到手腕附近。
- 左右手按解剖身份标注，不按画面左/右临时改名。
- tracker、腕带、手套袖口、衣服袖口的边界规则必须写进 `label_schema.json`。

如果以前已经标过袖口，仍需提供“标签消费证明”：

- 当前 producer 实际读取了哪份 label 文件和 SHA256。
- 每个类别读取了多少帧、多少像素。
- `SLEEVE_CUFF -> H` 的映射是否执行。
- 分侧 sleeve/tracker recall 是否在独立 eval 标签上计算。

仅仅“数据集里有袖口标签”不代表当前生成的 MASK 用到了它。存在袖口标签但消费计数为 0、映射缺失或召回未测，应直接 HOLD。

强烈建议每个 session 固定标注：

- 均匀时间位置，而不是只挑最好或最差帧。
- 手刚进入画面、袖口贴桌面、物体贴手指、快速运动、腕带遮断手腕、手离开画面等 stress case。
- fit 和 eval 的标注帧在采集或看结果前冻结，不能看完结果再换帧。

---

## 7. Clean 为了自然和可迁移，需要额外录什么

Clean 不是简单“把 MASK 里面涂白”。它需要知道真实桌面背景、桌面在三维空间的位置、不同帧之间怎样对应，以及哪些 donor 可信。

### 7.1 同一场景必须锁定

- 相机安装、桌子、桌面标志点和主灯从校准到任务结束不能移动。
- 锁定曝光、白平衡、对焦。
- 如果中途移动或 tracking world 重置，必须开始新的 scene/world epoch，不能继续共用旧 atlas。

### 7.2 空桌

最少必需：

- 固定相机且任务视角不变：任务前录 2–3 秒空桌，画面中没有手、袖子、操作物体和人体阴影。
- 头戴/移动相机：任务前缓慢覆盖任务视角范围，录 5–10 秒空桌，并保留逐帧相机位姿。
- 任务后再录 2–3 秒空桌，检查灯光和桌面是否发生变化。

强烈建议：

- 若正式任务视角范围很大或遮挡很多，可把移动空桌延长到 10–20 秒以增加 atlas 覆盖。
- 桌面操作区外围固定至少四个不共线的已知尺寸标志点。
- 开头拍 2–3 秒灰卡或色卡，用来检查光度漂移；灰卡不能替代空桌。

### 7.3 桌面接触、独立验证和抬离 negative

同一个物体至少录三类片段：

1. `OBJECT_CONTACT_FIT`：物体在桌上静止，拟合桌面 offset。
2. `OBJECT_CONTACT_EVAL`：时间和动作都与 fit 分开，只做验证。
3. `OBJECT_LIFT_NEGATIVE`：把物体抬离桌面至少约 5 cm 并静止，防止算法把悬空物体误判为接触。

接触标签最好来自力、压力、开关或预先记录的明确动作。不能先看算法算出的距离，再把距离小的帧标为“接触”，这会形成自证。

### 7.4 只展示人体前景

不拿操作物体，双手和前臂从正式进入方向缓慢进入/离开：

- 掌心、手背、腕部、袖口、腕带/追踪器都至少完整出现一次。
- 穿正式任务当天同一件衣服、同一佩戴物。
- 双掌相对的任务，要录到双掌相对、手背相反的真实方向，供 Robot 和 MASK 同时核对。

### 7.5 Confidence-aware atlas

未来 Clean 的背景 atlas 应当为每个像素保存：

- 世界/桌面坐标和相机可见性。
- donor 时间与来源 session。
- 与 H/O 的距离和遮挡状态。
- 几何、时间、光度置信度。

只有通过全部门的 donor 才能填入 H。没有可信 donor 的区域标为 `UNKNOWN/HOLD`，不要为了画面看起来平滑而偷偷填入错误纹理。条纹手臂、残轮廓和跨会话光影不一致，通常正是低置信 donor 或错误桌面对齐被当成真背景。

---

## 8. 一次采集的推荐顺序

下面的顺序可以放在一次连续录制中，用事件时间段区分，不要求拆成很多视频。

### 最少必需版

1. 固定相机、桌子、标志点和灯光；锁定曝光/白平衡/对焦。
2. 拍一张 Robot 全景和左右法兰/连接件近照。
3. **Robot 每会话初始化块（总计约 8–12 秒）**：桌面多 tag 标志板完整可见 2–3 秒；人位于正式录制原点，双臂中立 3 秒（大臂外展、肘外弯、小臂内收、双掌按任务关系）；双腕再做近端/远端小范围校验动作 3–6 秒。只允许用这一开场块求 session init。
4. 录静止空桌。
5. 物体放桌面静止：contact fit。
6. 抬起物体静止：lift negative。
7. 重新放桌面静止：独立 contact eval。
8. 不拿物体展示双手、掌心/手背、袖口和腕带。
9. 正常任务；若是本任务，双掌保持相对。
10. 做一次物体遮挡手指、一次物体与手分开的动作。
11. 录结束空桌。

### 强烈建议追加

12. 再录一次独立任务重复，作为 eval，不参与参数选择。
13. 空桌缓慢覆盖正常头部运动范围。
14. 录手贴桌面、袖口贴桌面、快速运动、画面边缘进出等困难片段。
15. 完成后立即运行 manifest、时间戳、文件数、视频解码和 checksum 校验。

### 理想追加

16. 换操作者、袖子颜色或同类物体再采 fit/eval session。
17. 单独封存 test session；优化人员只能看到 manifest 和通过/失败结果，不能提前看 test RGB/MASK。
18. 同步录制深度、IMU 重力和物理接触信号。

---

## 9. Fit / Eval / Test 与盲测防泄漏

### 最少规则

- `fit`：允许选择统一参数。
- `eval`：参数冻结后只运行一次；不能看结果再改参数并继续叫 eval。
- `test`：正式盲测；没有独立管理条件时可以暂时不设，但不能拿已经看过的会话冒充 test。

### 防泄漏要求

- 在打开 eval/test 像素前冻结 `split_manifest.json` 和 SHA256。
- split 按 session 或完整采集动作划分，不能把相邻帧随机拆到 fit/test；相邻帧几乎相同，会虚高迁移效果。
- 同一次 task repetition、同一个连续视频切片、同一 donor atlas 来源不能跨 fit/test。
- test 的 RGB、MASK、人工分数和预览图统一封存；代码只能得到允许的最终指标或 PASS/HOLD。
- 如果某人已经看过 test 视频或元数据，必须在审计里披露，由独立负责人决定是否失格。
- 缺失 session 必须显示 HOLD，不能换一个容易的 session 补数。

推荐 split：

```json
{
  "schema_version": "task-split-v1",
  "fit_sessions": ["cap_001"],
  "eval_sessions": ["cap_002"],
  "test_sessions": ["cap_003"],
  "frozen_before_eval_pixel_access": true,
  "selection_rule": "ACQUISITION_ORDER_NO_RESULT_SELECTION"
}
```

### Robot 跨会话声称门

- 单一 004 通过只能称 `single-session candidate`，不能称通用。
- 正式同 rig、同采集配置的跨会话验证，至少使用 `3 fit + 2 validation + 3 sealed test sessions`。
- validation/test 只允许读取每会话事前登记的 8–12 秒开场校准块来求 `T_world_rig`；禁止用正式任务帧调 base、左右映射、权重或阈值。
- 每个 test session 必须单独通过位置、自然弯曲、左右身份、关节连续性、碰撞和时序门；禁止用平均分掩盖某一会话的严重失败。
- 全部门通过后，只能声称 `PASS_CROSS_SESSION_SAME_RIG_SAME_CAPTURE_PROFILE`。更换相机、Robot 安装或 rig 后必须建立新 profile 并重新验收。

---

## 10. Manifest 和 checksum

每个 session 的 `session_manifest.json` 至少包含：

```json
{
  "schema_version": "new-task-session-v1",
  "task_id": "cap_20260901",
  "session_id": "cap_001",
  "collection_id": "lab_a_20260901_am",
  "role": "FIT",
  "scene_lock_id": "scene_01",
  "world_epoch_uuid": "...",
  "tracking_reset_count": 0,
  "camera_configuration_id": "ego_cam_01_locked",
  "lighting_configuration_id": "lights_01",
  "frame_count": 460,
  "fps": 30,
  "resolution": [1280, 960],
  "timestamp_unit": "ns",
  "robot_asset_revision": "robot_v3",
  "object_id": "can_01",
  "object_geometry_type": "CYLINDER",
  "palms_relationship": "FACING_EACH_OTHER",
  "files": []
}
```

每个文件记录：

```json
{
  "path": "sessions/cap_001/rgb.mp4",
  "bytes": 123456789,
  "sha256": "64位小写十六进制",
  "ordinary_file_not_symlink": true
}
```

`checksums.sha256` 应覆盖：原始 RGB、时间戳、相机参数/位姿、Robot q、URDF/mesh、法兰/连接件、Object6D、MASK 标签、事件和 split manifest。

---

## 11. 自动校验应该检查什么

建议提供一个只读校验器，例如：

```bash
python tools/validate_new_task_bundle.py \
  --task-root tasks/cap_20260901 \
  --manifest tasks/cap_20260901/task_manifest.json
```

如果仓库暂时没有这个脚本，也应按下面清单人工/脚本核验：

- 所有 JSON/YAML 可解析，未知字段 fail-closed。
- 路径存在、是普通文件、bytes 和 SHA256 一致。
- 视频能完整解码，帧数、分辨率、fps 与 manifest 相同。
- `frame_index` 与时间戳一一对应；时间单调；丢帧/重复帧已标记。
- 相机、Robot、Object6D 的时间范围有交集且同步误差不超预算。
- 旋转矩阵/SE(3) 有效，单位一致，没有 cm/m 混用。
- Robot joint 名称、q 列数和 URDF 一致，q 在物理限位内。
- 左右手、掌心、拇指、法兰和连接件示意图与数值变换一致。
- 物体 shape dispatch 合法；CARD 不进入 cylinder consumer。
- MASK class-id 只使用 schema 中的类别；左右身份、袖口、tracker 和物体类完整。
- 空桌、contact fit、contact eval、lift negative、human foreground、normal task 片段齐全。
- fit/eval/test 不交叠，test 没有像素访问记录。
- 缺失/低置信输入产生 HOLD，而不是默认值或隐藏 fallback。

建议同时准备合成反例：左右手交换、掌心法向翻转、法兰装到错误端面、cm 当 m、圆柱上下端互换、扑克牌走圆柱分支、袖口标签存在但消费为 0、world 中途 reset、fit/eval 重复帧。校验器必须把这些反例全部拒绝。

---

## 12. 用户期望输出和优先级也要作为输入

技术输入齐全仍不代表优化目标明确。请在 `task_request.yaml` 写清：

```yaml
priority:
  - P0: robot_arm_position_and_natural_posture
  - P1: hand_handedness_palm_and_flange
  - P2: object_finger_depth_occlusion
  - P3: mask_complete_sleeve_tracker_no_table_blob
  - P4: clean_natural_background_no_outline_no_flicker

robot:
  arm_position_first: true
  hand_and_flange_after_arm: true
  palms_relationship: FACING_EACH_OTHER
  motion_priority: TRAJECTORY_APPROXIMATION_ALLOWED
  natural_posture_reference: assets/robot/natural_pose_reference.json

review_outputs:
  - static_contact_sheet
  - full_or_fixed_sample_video
  - raw_mask_clean_difference_panel
  - object_finger_depth_occlusion_panel
  - quantitative_summary_and_hold_reasons
```

还应说明：

- 哪些是硬要求，哪些可以用误差换自然姿态或可解性。
- 是否允许轨迹近似；允许多大腕部位置/旋转误差。
- 最先看什么：通常先看 Robot 双臂位置，再看手和法兰，最后看动态和遮挡。
- 输出图片/视频希望放在哪里，文件名是否要短。
- 哪个旧任务仅作视觉参考，哪个文件可以作为数值权威；视觉参考不能自动升级为 ground truth。

---

## 13. 交付前一分钟自检

### 最少必需

- [ ] 原始 RGB、帧数、分辨率、fps、时间戳齐全。
- [ ] 相机内参和坐标约定齐全。
- [ ] Robot URDF/mesh/单位/base/q/joint names 齐全。
- [ ] 左右手、掌心、拇指、同侧/交叉映射写清。
- [ ] 法兰/连接件 CAD 或实测尺寸与安装变换齐全。
- [ ] 物体明确为圆柱、扑克牌/薄盒或米制 mesh；Object6D 齐全。
- [ ] MASK 包含分侧 hand/forearm/sleeve/tracker，物体单独成类。
- [ ] 有空桌、contact fit、contact eval、lift negative、人体展示、正常任务。
- [ ] 用户优先级和希望查看的图片/视频写清。

### 强烈建议

- [ ] 曝光、白平衡、对焦锁定。
- [ ] 四个以上固定桌面标志点和逐帧相机位姿。
- [ ] world epoch 无 reset，设备同步误差已报告。
- [ ] 两次独立任务重复，可分 fit/eval。
- [ ] 所有文件有 bytes/SHA256，视频完整解码。
- [ ] 左右手/掌心/法兰自动示意图已人工确认。

### 理想

- [ ] fit/eval/test 三段独立，test 已封存。
- [ ] 同步深度、IMU 重力和物理接触信号齐全。
- [ ] 标签双人复核，逐类消费账本与召回报告齐全。
- [ ] 自动 validator 的正例和反例全部通过。

如果任何关键项缺失，不要猜。把缺失项写成明确 HOLD，通常比生成一段看似完整但姿态、遮挡或背景错误的视频更节省总时间。
