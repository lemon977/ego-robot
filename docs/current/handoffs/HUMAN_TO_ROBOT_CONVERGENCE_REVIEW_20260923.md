# Human→Robot Baseline v1：当前结论、证据修正与外部AI问题

核查日期：2026-09-23。项目：`/mnt/workspace/code/chaoyang`。

本文为只读复核后的交接说明，不是新算法执行回执，不更改历史RESULT、质量门或任务权限。可直接把本文交给其他AI。本文没有重新观看全部视频；数值来自实际收据、已保存数组及对应生产代码，视觉质量仍须按明确范围复核。

## 1. 当前状态与必须先纠正的结论

- 当前治理revision：13967，本次用 `/usr/local/bin/python` 运行正式 `validate-governance`，PASS；当前无活动任务。
- 最新任务：`human_to_robot_baseline_v1_convergence_20260923`，机器终态 `REJECTED_QUALITY`，release仍为INCOMPLETE。
- 四产品结构4/4，质量0/4，采用0/4。历史结果不能因新文件出现而升级。
- 起点为16:59:24，终态为17:31:33，实际约32分09秒；17:41:03追加验证发布。十二小时是预算上限，本次没有运行满十二小时。
- 提前终态不等于所有授权实现都已完成。持久表面对应、连接件碰撞接入、031牌边修复及完整新旧运动审阅等工作没有完成实作；“缺派生产物”不能自动证明这些工作都不可执行。
- 当前HEAD为`2d2306d61b2c38daa847d7468bab48a6ddd909b1`，存在工作树修改，HEAD不代表本次所有运行代码。

**本次复核发现上次汇报有明确错误：007角色Mask并非全空。**同一线缆RESULT记录且本次读取PNG复核一致：前臂16/16帧非空、capture_device 7/16、task_object 11/16；但RESULT的`propainter_admission.reason`硬编码成“角色Mask为空”。该阻塞理由不成立。线缆身份和关联尚未被证明仍是问题，但必须重新追溯已有支持，不可沿用“完全没有支持”作为前提。

此外，以下声明需收窄：

1. `102/102 direction_qualified`只证明当前程序从模型21点成功构造非退化掌坐标系；程序同时将独立RGB支持记为UNKNOWN。不能解释为102帧方向均经独立视觉确认。
2. 15/15表示15个路径存在并解码到预期帧数，不表示全部可视化内容验收通过。Motion两个“全链审阅”槽实际指向R2的`robot_candidate.mp4`，没有证明包含本次要求的Raw/旧新运动/全时间轴曲线；应视为槽位内容待核验。
3. 左手16帧视频已生成，但其生成脚本直接写入统一的“无可靠第二只手”结论，没有逐帧独立judge表或相应审阅理由。它可用于复核，不能代替已完成的逐帧独立可观测性证据。
4. Contact有158次模型条件采样，但尚未证明每个采样点满足严格有限物体patch要求，不能写成158个真实接触。
5. “没有独立外部真值”限制绝对精度和实际效果的胜负结论，不妨碍在共同冻结目标下比较跟踪残差及适用质量门。

## 2. 已证实的进展及各线未完成项

### Motion / Product

原031：总149帧，右手102帧输入（47–148），左手0。维持原分母。

新候选在66–81固定16帧运行SciPy SLSQP，位置欧氏距离≤20mm为硬约束，方向损失为软目标；方向验收≤15°，完整SO(3)≤15°独立计算。相机、base、安装和手指q固定；以同帧position-only候选初始化，无跨帧热启动。

| 指标 | P50 | P95 | 最大值 | 结论 |
|---|---:|---:|---:|---|
| 位置残差 | 20.000mm | 20.000mm | 20.000000009mm | 按实现数值容差16/16通过，几乎全部触边 |
| 掌法线夹角 | 2.777° | 5.930° | 6.576° | 数值均低于15° |
| 掌纵向夹角 | 25.539° | 48.819° | 50.218° | 联合方向门未通过 |
| 完整旋转 | 25.686° | 49.164° | 50.594° | 0/16通过 |

联合适用门0/16，小窗拒绝，没有扩到102帧。该结果表明一次冻结局部求解未满足全部门，不构成不可达证明，也不授权事后删除失败方向。

**目标语义疑问：**程序构造每帧`direction_root = R_target.T @ direction_camera`，因此当Robot采用原`R_target`时自然重建同一方向。两个方向都获准时，期望旋转仍被原full-pose目标锚定；有限角度容差和损失形式虽不同，却未证明获得了独立的机器人掌轴定义。应审查这是合法模型到机器人坐标映射，还是绕回原始腕旋转目标。

新候选碰撞检查仅覆盖固定KaiHand内部自碰撞；adapter、arm-hand、cross-arm和连续扫掠均未评价。

连接件的真实视觉消费、68.4mm完整开发安装合同、正式S2入口及旧同签名resume已有历史证据。本轮无新合格运动/背景进入正式产品；四条产品继续引用旧拒绝候选。

### Scene / Clean

007固定181–196窗得到16/16局部top-1候选。当前实现是在固定走廊中做亮度/低色差阈值、形态闭运算和最高连通分量选择。它没有完整实现跨帧实例匹配、端点到设备的关联或逐帧中心线证据，因此16/16“单帧唯一”不自动等于“同一物理线缆持续身份”。

角色支持实际非空情况为16/7/11；未验证它们是否覆盖投诉线缆及reference实际输入。新ProPainter调用0。四会话Clean质量仍拒绝。031牌边问题本轮没有新的可证伪修复；0902也没有新的同配方生产，只复用旧结果。

### Depth / Occlusion / Contact

031已有149帧开发级Depth及可见物体几何；007旧固定匹配统计1021，低于1500门；0902无合格对应Depth。未改变门。

031旧16帧遮挡known decision coverage约82.87%，不是准确率。旧736次切换/3269次可比较项属于屏幕像素比较。本轮同表面对应尝试0，未实现新的link/Scene持久表面对应；没有据此评价真实翻转或A→B→A。

Contact单位是“帧×侧×指尖”：102右手帧×5指尖=510条资格检查，158条实际模型条件邻近采样，严格资格0，R1执行0窗口。352条没有进入采样。

采样实现：在物体mask内寻找距指尖2D最近点，距离≤40px；映射到半分辨率Depth，取5×5邻域内`valid & lr_consistent`深度中位数并反投影，再计算到模型指尖3D的距离。

需审查：5×5有效深度未进一步限制为同物体/同侧/同连续patch；目标像素自己无效也可能借邻域产生数值；采样点身份、边界及逐点不确定性没有闭合。故158是执行量，不是严格几何合格量。收据还把inferred、offline noncausal与公制资格揉在一个拒绝码中：非因果性本身不排除离线接触诊断，应明确真正缺的是哪种几何与来源证据。

连接件碰撞：真实STEP派生STL及固定装配存在，本轮查询0。审计脚本因缺专门碰撞语义而记录未授权，但没有完成真实mesh拓扑、既有检查器可消费性和接触对范围验证；这不是“该网格不能用于碰撞”的证据。

### Sensor

三会话165+179+122=466帧，新生成完整回放；复用原MANUS/HandMotion/q/FK，没有新求解或标定。新增可见内容包含合法线段裁切、固定原点等比例世界视图、朝向、局部手形/夹合及保存的FK。

已证实的是绘制实现及回放产物；像素贴合、腕定义与完整动态质量仍待复核。不能由“用户待审”推断算法只差验收，不再有几何问题。

### Local / HuRo

共同frame/time/target/mount/映射做了数值相等检查，复用两侧数组，没有新求解。统计读取已保存残差字段，并非本轮独立从q重算完整FK。下表均为共同模型目标残差，不是外部精度。

| 会话 | 方法 | 位置P50/P95（mm） | 完整旋转P50/P95（°） |
|---|---|---|---|
| 007，756侧帧 | Local | 约0 / 14.63 | 约0 / 5.09 |
| 007，756侧帧 | HuRo | 8.57 / 11.56 | 31.06 / 49.21 |
| 031，102侧帧 | Local | 77.50 / 104.39 | 30.04 / 39.83 |
| 031，102侧帧 | HuRo | 79.53 / 115.66 | 28.50 / 40.70 |

**必须单独追查的异常：031第47帧右侧，Local位置残差1397.82mm、HuRo1399.05mm。**本次已从两份NPZ确认异常帧号。共同出现首有效帧异常，提示检查共同输入/初始化/保存FK，尚不能确定根因。

## 3. 请外部AI集中回答的七个问题

请按“证据→判断→最小修改或诊断→实际消费者→停止条件”回答。先指出需要补验的假设，不把未知写成已定位；不要重新设计四线架构。

1. **031方向合同是否真正正确？** 在独立RGB支持UNKNOWN时，两方向为何全部qualified？`R_target.T @ direction_camera`的每帧局部轴是否使所谓部分方向实质继承旧full-pose目标？应如何用固定机器人掌轴和可验证来源建立合同，而不根据求解结果挑方向？请区分单次局部优化失败、约束冲突、错误目标与不可达证明。
2. **第47帧约1.4米残差如何定位？** 两方法同帧异常时，最短检查链应覆盖哪些时间/坐标/初始化/invalid及独立FK字段？能否先建立有限差分或可行姿态对照，不进行多初始化择优或算法调参？
3. **007现有非空支持怎样用于持续线缆证据？** 已有16/7/11帧角色支持，什么最小关联/传播/端点检查能验证投诉线缆，而不是因错误的“全空”阻塞？当前高亮连通分量是否可能选错线？何种证据足以进入一次固定小窗Clean？031牌边应独立如何推进？
4. **如何补完同表面遮挡与Contact采样？** 在既有camera、Depth、mesh/FK及物体mask上，link局部反投影和可见Scene对应能做到何种范围？边界、显露和未知怎样记账？Contact的5×5邻域怎样约束同物体/连续patch并保存不确定性？哪些离线模型条件诊断可做，哪些严格接触仍应拒绝？
5. **连接件真实mesh能否直接做有限碰撞诊断？** 已有CAD和安装，哪些缺的是实现/网格检查，哪些确需新的装配语义？能否在不新增忽略对、不造包络的条件下查询并按接触类别报告，避免将缺收据误作缺资产？
6. **Sensor与比较怎样完成数值—画面验收？** Sensor应补哪些固定帧/连续窗误差与缺失原因？Local/HuRo能否先从q独立复算FK、逐帧门及极值原因，给出明确的合同内结论？什么证据才能称更贴近人手？
7. **本轮应当如何继续未完成实现？** 32分钟提前终态是否把可实现的表面对应、碰撞接入和全链可视化当成外部阻塞？下一执行应明确哪些工作已有授权、哪些需要新证据；禁止原样重试已拒绝数学候选，但不要把所有剩余工程都要求用户重新批准。

请给出优先级：先修“结论/来源错误”，再补真实消费者；每项只提出最小可验收动作，不新增大规模训练、模型下载或平台。

## 4. 证据路径与视频入口

以下`R`为：

`/mnt/workspace/code/chaoyang/_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001`

| 内容 | 相对R的路径 |
|---|---|
| 终态记录 | `RESULT.json` |
| 验证记录 | `FINAL_VALIDATION.json` |
| 15槽位路径/SHA/帧数 | `delivery/DELIVERY_MANIFEST.json` |
| 031 IK结果和方向资格/候选NPZ引用 | `lanes/motion_product/partial_direction_031/wave0/RESULT.json` |
| 031新旧骨架小窗视频 | `lanes/motion_product/partial_direction_031/review_v1/031_IK_WINDOW_OLD_NEW_REVIEW.mp4` |
| 007角色计数及矛盾拒绝码 | `lanes/scene/cable_007/wave0/RESULT.json` |
| 007线缆小窗 | `lanes/scene/cable_007/wave0/CABLE_TOP1_EVIDENCE_REVIEW.mp4` |
| Contact计数/逐行NPZ | `lanes/scene/contact_screen_031/wave0/RESULT.json` |
| 同表面未实现说明 | `lanes/scene/occlusion_same_surface_031/wave0/RESULT.json` |
| 连接件碰撞未执行说明 | `lanes/motion_product/adapter_collision/wave0/RESULT.json` |
| 三Sensor视频及源数组签名 | `lanes/sensor/review_v1/RESULT.json` |
| Local/HuRo数值及输入签名 | `lanes/compare/numeric_v1/RESULT.json` |

统一可点击视频导航：[15槽位索引](../visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md)。外部AI无法访问服务器时，本文表格可用于推理；视频链接本身不等于已传输文件。

源代码位于`/mnt/workspace/code/chaoyang/src/chaoyang/`：

- `ops/run_human_to_robot_convergence_motion_wave0.py`：资格、初始化、硬位置、方向及门。
- `pipeline/huro_hand_frame_v2.py`：`palm_basis`定义，列为掌宽、正交纵向、叉乘法线。
- `ops/run_human_to_robot_convergence_cable_evidence.py`：固定走廊、阈值、组件选择及硬编码阻塞理由。
- `ops/run_human_to_robot_convergence_contact_screen.py`：邻近物体像素、Depth采样与资格。
- `ops/run_human_to_robot_convergence_remaining_audits.py`：左手统一结论、连接件/遮挡未评价路径。
- `ops/build_human_to_robot_convergence_delivery.py`：15槽位实际绑定和解码检查范围。
- `ops/run_human_to_robot_convergence_compare_numeric.py`：共同合同检查、直接读取保存残差。

## 5. 测试和权限边界

变更相关33项通过；碰撞/当前状态14项在补齐环境后通过。项目内TMP全量运行原始结果1513通过、14失败、1跳过；其中13项为CPFS hardlink transaction identity drift，另一环境变量缺失项在定向重跑中恢复。不得把这些不同运行拼成“全量全绿”，也不得用测试数量证明动作质量。

默认`python`本次缺少jsonschema，`/usr/local/bin/python`可完成治理检查；不需要因此安装或升级模型环境。

全部维持OFFLINE_VISUAL；training_eligible/control_ground_truth/physical_deployable/external_metric_authority均false。所有新增文件必须在项目内。当前任务已终态；本次仅提供交接及错误说明，历史收据保留原字节，当前机器结论尚未因本复核自动修正。
