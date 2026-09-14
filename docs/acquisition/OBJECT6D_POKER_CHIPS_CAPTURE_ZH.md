# 扑克牌、薯片与碗的 Object6D 采集规范

## 2026-09-07 当前执行决策：不要求用户补采

当前薯片与扑克牌任务不再把用户补拍物体视频、照片或尺寸作为执行前置。系统将使用已有资料自动估计：

- 已保留的左右眼双目 RGB 原视频、相机内外参和时间戳；
- 单目 RGB 物体检测/分割、双目深度、跨帧实例关联与刚体/平面位姿拟合；
- HaWoR/PICO 手指和接触时序，用于判断当前操作的物体；
- 扑克牌薄盒、碗/盘、木架以及薯片弯曲薄片的任务级几何先验。

自动结果必须标记 `AUTO_ESTIMATED_OBJECT6D`，逐帧保存 `confidence` 和 `provenance`。遮挡、身份不明或双目失效的帧写为 `UNKNOWN`，不允许用静态槽位、相邻帧或手部位姿伪造物体位姿。下文的额外拍摄内容仅作为可选的高精度增强和误差标定，不是用户必须交付的资料。

## 当前已有深度与拟采用的估计路线

- 现有 0902 原始数据没有预生成的 sensor depth/disparity 文件，但保留了左右眼拼接双目视频和完整标定；实测双目基线为 `63.7716504 mm`。
- 当前 HaWoR 的手部 Z 是单目网络预测相机尺度后恢复的估计值，不是双目实测深度；PICO 手部 Z 来自头显追踪，两者都不能直接代替物体深度。
- 当前 Mask 是 2D 分割/光流，Clean 是平面单应补画，Robot 只有机器人 CAD 自身的 z-buffer；三者目前都没有可靠的场景/物体 dense depth。
- 正式改进路线沿用历史罐子流程：双目拆分与校正 → FoundationStereo/视差公制深度 → 物体轮廓与实例身份 → 几何/位姿拟合 → 逐帧物体 near/far Z → 与机器人 z-buffer 做遮挡和接触检验。

完整审计见 [20260907 深度来源审计](../../tasks/control/runs/20260907_depth_source_audit_v1/REPORT_ZH.md)。

## 一句话结论

每件物品拍一段全方位视频可以用来重建**形状和外观**，但不能单独提供任务过程中的
Object6D。完整输入由两部分组成：

1. 米制物体模型：物体究竟有多大、坐标原点和轴在哪里；
2. 任务逐帧位姿：每一帧 `T_object_to_camera` 或 `T_object_to_world`、时间戳、有效性和来源。

缺第 1 项时只能用近似形状；缺第 2 项时只能做视觉演示，不能可靠验证物体遮挡手指、真实接触
或三维非穿透。

## 可选：如需要实测精度，额外拍摄什么

### A. 每种物体的模型素材

相机固定、曝光/白平衡/对焦锁定，物体放在带已知尺寸 AprilTag/ArUco 或标尺的转台上。

- 水平一圈：缓慢 360°，20–30 秒；
- 俯视一圈：相机约高 30°–45°，20–30 秒；
- 仰视/底面一圈：碗和架子必须拍，20–30 秒；
- 近距离静态照片：正、反、左、右、上、下各一张，不能裁掉标尺或 tag；
- 卡尺照片：长、宽、厚/高度/直径均需数值可读，物体和卡尺同一画面；
- 若设备可输出深度，RGB、depth、K、畸变、时间戳一起保留。

不要手持物体转圈；手会遮住边缘并使尺度和相机位姿不可恢复。用转台、细透明支架或分两次翻面。

### B. 正常任务中的逐帧位姿素材

- 相机原始 RGB、K、畸变、c2w、硬件时间戳完整保存；
- 桌面外围固定至少四个不共线的已知尺寸 tag，整个任务不移动；
- 每个物体从开始到结束保持唯一 ID，不能在遮挡后交换；
- 任务前拍 3–5 秒静态初始化：三张牌/三片薯片、碗和木架都清楚可见；
- 每次拿起、首次接触、完全在手中、放下/入碗前后各停约 0.5 秒；
- 另拍一次物体与手明确分离、一次物体真实遮住指腹的验证动作；
- 若有光学 tracker/AprilTag，可贴在牌背非接触区、碗底/外侧或独立刚性夹具上，并提供
  `T_tag_to_object`。薯片本体不适合贴重 tag，可用模型跟踪加人工冻结关键帧复核。

## 各物体的额外要求

### 扑克牌

- 一副牌可共享一个米制 `CARD_THIN_BOX` 几何，但 `card_0/1/2` 必须有独立外观与逐帧 pose；
- 测量长、宽、真实厚度，记录四角顺序和正反面；
- 拍牌边和抽离木架的过程，避免只有正面；
- 木架不是操作物体，但需要米制尺寸、模型、任务中的固定 pose 和架顶 support plane。

### 薯片

- 不能用薄盒或平面代替。至少扫描实际使用的三片；若只给一片模板，必须明确
  `APPROX_SHARED_CHIP_TEMPLATE`；
- 测量长、宽、中心厚度和两个主曲率方向；最好提供 depth/摄影测量 mesh；
- 易碎或变形时，每次采集前后各拍一张，确认几何未改变；
- `chip_0/1/2` 始终独立，入碗后也不能交换身份。

### 碗/盘

- 必须拍内表面、外表面、碗沿和底面；只拍俯视无法恢复外侧与厚度；
- 测量口径、底径、总高、碗沿厚度、内深；
- 输出 visual mesh 和 collision mesh，坐标原点建议在底面中心，+Z 指向碗口；
- 碗通常是 session-static：每个任务会话只需一个固定 pose，但必须由桌面 tag/模型配准验证。

## 可选补采材料的参考结构（当前不要求用户交付）

```text
object6d_capture/
├── measurement.json
├── calibration/
│   ├── camera_intrinsics.json
│   ├── tag_board.json
│   └── T_tag_to_object.json
├── models/
│   ├── card_template.obj
│   ├── chip_0.ply
│   ├── chip_1.ply
│   ├── chip_2.ply
│   ├── bowl_visual.ply
│   ├── bowl_collision.ply
│   └── rack.obj
├── turntable_raw/
├── measurement_photos/
└── sessions/<session_id>/
    ├── object6d.npz
    ├── object_state.json
    ├── contact_canary.json
    └── object_masks/
```

`object6d.npz` 最低字段：

- `frame_id [N]`、`timestamp_ns [N]`；
- `object_ids [M]`；
- `T_object_to_camera [N,M,4,4]` 或明确命名的 world 版本；
- `valid [N,M]`、`confidence [N,M]`；
- `provenance [N,M]`：`DIRECT_TRACKED / MODEL_FIT / INTERPOLATED / MISSING`；
- `geometry_sha256 [M]`、`coordinate_convention`、单位固定为 meter。

可从 [OBJECT6D_CAPTURE_MANIFEST.template.json](../../tasks/templates/OBJECT6D_CAPTURE_MANIFEST.template.json)
复制清单填写。

## 如收到可选实测材料时才执行的硬门

- mesh/尺寸/照片 SHA 全匹配，单位和坐标轴明确；
- 旋转正交且 `det(R)=+1`，位姿有限、时间戳严格递增；
- RGB/Object6D 最大同步误差在采集预算内；
- 物体 ID 全程不交换，插值帧与直接观测帧分开统计；
- 预注册接触关键帧必须有直接观测，不允许靠插值证明接触；
- 投影轮廓与独立 object mask 的边界误差过门；
- 物体前后深度、具名指腹接触、非接触网格穿透分别验收；
- 扑克牌不得进入 cylinder consumer，薯片不得静默降级为平板，碗内外表面不得合并。
