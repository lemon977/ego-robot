# 0915 局部 Interaction → Contact → Kai22 开发级结果

状态：CURRENT

范围固定为 `play_cards_0915_001` 的 150 帧。运行只消费冻结的 resize-only HaWoR、SAM3.1、encoded-domain FoundationStereo、Object6D v2 和 processed 内触觉/时间戳；没有重跑模型，没有读取 Removal/Clean，没有消费 `short_gap_inferred` 或 PICO26 手部结果。

浅层视频和机器指标入口：

- [`visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/README_ZH.md`](visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/README_ZH.md)
- [`OBJECT6D_GEOMETRY_REVIEW.mp4`](visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/OBJECT6D_GEOMETRY_REVIEW.mp4)
- [`INTERACTION_CONTACT_REVIEW.mp4`](visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/INTERACTION_CONTACT_REVIEW.mp4)
- [`KAI22_R0_VS_R1_REVIEW.mp4`](visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/KAI22_R0_VS_R1_REVIEW.mp4)
- [`METRICS.json`](visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/METRICS.json)

## 终态

- Object6D QA：`PASS`。三张牌逐帧字段闭合，法向先统一符号；visible-surface centroid 明确不是物体固定中心。跨帧中心/法向变化只作诊断，不作 15 mm 固定中心硬门。
- 自动尺寸：`UNKNOWN`。现有 Object6D v2 没有任何完整边界观测，因此未使用标准牌尺寸，也未把遮挡边界补成完整矩形。
- Human/Stereo alignment：`REJECTED_HELDOUT_ALIGNMENT`。非接触可见手表面有 318 条训练记录和 78 条冻结 hold-out；拟合为 `scale=0.90175`、`offset=-49.97 mm`。hold-out 绝对残差 median 为 `13.56 mm`，但 P90 为 `36.79 mm > 30 mm`，所以不授权公制 wrist—object translation。该 offset 由非接触手表面拟合，不是对所有 hand/object 配对统一减去 48 mm。
- Interaction：共 4,500 条固定 hand/finger/object 记录；1,034 条具备 finger-associated visible surface 与有限 object patch。最近有限 patch 距离为 `6.60 mm`，仍高于冻结的 `5 mm` Contact 门；只有 4 条投影落在直接可见 patch 内。
- Contact：`UNKNOWN=3466`、`NO_EVIDENCE=1032`、`APPROACH=2`，没有 `CONTACT_CANDIDATE`，也没有同一 hand/finger/object 的五帧连续合格窗口。大不确定度没有扩大 5 mm 门，缺失不确定度保持 unknown；无限平面命中不能越过有限 patch 门。
- Kai22 R0：`COMPLETED_DEVELOPMENT_BASELINE`。293/293 个 direct-observed side-frame 完成 q22 retarget；左右各 22 个关节顺序与 pinned URDF 一致，中间代表帧各 23 个 link FK 均有限。R0 不依赖 Contact。
- Kai22 R1：`BLOCKED_LOCAL_EVIDENCE`。首阻塞为 alignment hold-out 未通过，第二阻塞为没有五帧局部 Contact 窗口；没有修改 q22/wrist，也没有把 R0 冒充 refinement。
- Kai22 R2：`NOT_RUN_OPTIONAL_R1_NOT_CLOSED`，不反向阻塞 R0/R1 的诚实终态。

三个视频均为 1280×480、30 FPS、完整解码 150 帧。画面来自正确的 physical-left encoded resize-only 域，未做 lens undistortion。

## 权限与碰撞边界

所有输出固定为 `DEVELOPMENT_RELATIVE / NON_CONTROL / NON_DEPLOYABLE`：

- `control_ground_truth=false`
- `physical_deployment_authorized=false`
- `calibration_authority=DEVELOPMENT_ONLY`
- `collision_scope=ROBOT_SELF_PLUS_OBSERVED_OBJECT_PATCH`
- `full_object_collision=UNVERIFIED`
- `full_object_environment=UNVERIFIED`

由于 R1 未执行，robot self-collision 和 visible-patch crossing 均记为 `NOT_EVALUATED_R1_BLOCKED`；没有补造牌厚度、隐藏背面、卡托或完整环境。

## 失败收据保留

正式成功运行是 corrective V3。V1 在 R0 前因 HaWoR provenance 枚举名假设错误早停；V2 在 R0 完成后因尺寸 schema 漏列预算字段早停。两次都没有重跑模型或修改源数据，失败收据保留在 `tasks/receipts/`，没有通过删除失败证据换取干净状态。

下一步若继续 R1，不能放宽 5 mm 门或删除困难帧。应先独立解决 hold-out 尾部残差，并建立不依赖 Contact 自证的 KaiHand pad ↔ Human wrist 局部坐标映射；在此之前正确状态仍是 R1 阻塞。
