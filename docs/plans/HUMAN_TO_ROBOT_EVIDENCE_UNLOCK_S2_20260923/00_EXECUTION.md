# Human→Robot S2：装配、动作、遮挡与四支线稳定基线收敛

状态：`AUTHORIZED_FOR_REGISTRATION`。本计划是 `HUMAN_TO_ROBOT_BASELINE_V1`
的有限后继，不复活或覆盖 V5、R1、R2、S1。

## 目标与预算

- task：`human_to_robot_evidence_unlock_s2_20260923`
- 总预算：12小时；H3/H6/H9检查，H10后只做验收和收敛。
- 主产品：`get_potato_chips_0915_007`、`play_cards_0915_031`。
- 同配方回归：`get_potato_chips_0902_103`、`play_cards_0902_042`。
- Sensor：`play_cards_0916_097/098/101`。
- 单GPU租约；CPU总软上限8；前三小时GPU不超过90分钟。
- 不训练、不下载新模型、不调用真机；所有新写入均位于项目根内。

## 四条隔离支线

1. `scene`：007线缆/污迹和031牌面边界，场景深度资格、遮挡像素归属、Contact/R1筛选。
2. `sensor`：复用三会话共同后端，修实际绘制/坐标消费并给出独立视觉结论。
3. `motion_product`：031来源及腕目标残差、完整安装合同、连接件、renderer和正式入口。
4. `compare`：冻结公共输入，增量复核Local/HuRo；本地独享R1不进入公平R0比较。

历史目录中的 `lane3_sensor` 对应支线2，`lane2_motion` 对应支线3。当前任务只按
上述功能名和完整路径分配所有权。

## H3并列主任务

- 在007帧181–196、031帧66–81冻结q、相机、安装、背景，输出：A旧产品、B只接入真实
  连接件、C在B上接入通过资格检查的遮挡。若实片不可比较，C保持UNKNOWN并交受控测试。
- 求解、独立FK和renderer绑定同一完整68.4mm开发安装合同；目标腕只作诊断标记，正式
  手根只能来自真实q/FK。连接件必须进入RGB、部件ID和Robot光学轴深度。
- 031腕目标残差与装配并列：分别评价装配一致、显示一致和目标跟踪。追溯右侧102帧
  observed/inferred最早写入者，左侧独立输入缺口单列，不翻标志、不镜像、不填默认姿态。
- 007固定窗先区分支持缺失、传递丢失、reference污染或修复能力问题，只选择一个有失败
  证据的修复包；031同窗回归不能退化。

## 遮挡和碰撞边界

- 复用031现有149帧encoded-domain Depth/Object几何，禁止因来源说明变化重跑Depth。
- 排序前闭合相机、像素、尺度和时间；3mm只是冻结排序容差，不是精度证明。
- 排除原人手/设备及未知边界深度；Clean不得提供几何。分别统计
  `SCENE_FRONT/ROBOT_FRONT/UNKNOWN/NOT_APPLICABLE`及连续窗切换。
- 连接件视觉和碰撞覆盖分别验收；没有批准的连接件碰撞模型时写`UNVERIFIED`，不得继承
  旧机械臂+双手碰撞PASS或临时增加忽略对。
- Contact与遮挡分开；Robot R1冻结腕、arm、camera、object、mount，只修许可手指。

## 正式入口、缓存和采用

- 沿用 `run_human_to_robot_baseline_v1`。S2配置必须显式绑定route/domain/Clean/R0、完整
  mount、CAD、Depth、renderer和compositor SHA；缺新接口即失败，禁止静默退回RGB覆盖。
- 左右映射由producer/consumer语义和非对称测试闭合，不硬编码某一排列为通用答案。
- 安装变化失效FK/残差/碰撞/渲染；遮挡变化只失效归属/合成；Clean变化不重跑Motion；
  文档变化不失效算法缓存。同签名resume不得重跑基础模型。
- 分开发布模块技术冻结、会话候选、会话技术质量/待用户审阅、用户已验收采用。
  短窗不是全片，复用不是新增，完整解码不是动作完整。

## 测试与停止条件

必须覆盖：目标腕偏离不能移动正式手根；非零visual/inertial origin只应用一次；连接件未被
最终产品消费或未进入depth/ID时失败；错域/单位/时间或原手深度参与遮挡时失败；全UNKNOWN
不算排序成功；正式CLI绕过新renderer/compositor时失败；Local/HuRo不同target/mount/分母
时拒绝；publisher不能回退旧状态。

H9冻结组件组合，H10后不启动无法验收的新计算。H12未达整体目标时保留冻结模块、失败实物
和具体解除动作，不自动创建换名后继。所有结果保持`OFFLINE_VISUAL`，并固定：

```text
training_eligible=false
control_ground_truth=false
physical_deployable=false
external_metric_authority=false
```
