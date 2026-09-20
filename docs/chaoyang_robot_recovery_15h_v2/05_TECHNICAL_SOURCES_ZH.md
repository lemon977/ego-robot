# 技术来源、事实范围与不确定性

检索日期：2026-09-19。只使用原作者论文/项目与官方文档。公开主分支信息不代替本地固定commit，不要求更新依赖或下载新模型。

## 输入事实

本轮实时状态仅来自用户贴出的治理12362汇报与Interaction结果。ChatGPT未连接用户主机，没有读取远端W0坏帧、最终SHA审计原文件或执行模型。本包要求本地AI重新核验；没有预先认定任何具体bug已存在。

上一版任务书与算法审计文档已读取，用于修订依赖、范围和交付口径，不作为当前运行状态。旧文件涉及0916的潜在扩展表述被本轮“仅0915”明确排除。

## S1 HaWoR

官方项目：`https://hawor-project.github.io/`
官方代码：`https://github.com/ThunderVVV/HaWoR`

支持的事实：方法分开处理camera-space手运动与world-space相机轨迹，并含运动补全模块。工程引申：q22局部姿态、camera wrist、world wrist应分层审查；任何补全必须标来源。本轮不启用新模型或把补全写成直接观测。

## S2 SAM3 / SAM3.1

官方仓库：`https://github.com/facebookresearch/sam3`

支持的事实：支持text/visual prompt的检测、分割和视频跟踪；3.1更新包含多目标共享记忆机制。工程引申：本地Object入口不应无条件依赖HaWoR全片严格3D通过；提示质量、左右身份和时序QA仍需本地验证。共享记忆不保证本地身份一定稳定，pinned API才是实际调用依据。

## S3 FoundationStereo

官方说明：`https://raw.githubusercontent.com/NVlabs/FoundationStereo/master/readme.md`
官方仓库：`https://github.com/NVlabs/FoundationStereo`

支持的事实：输入双目图、输出dense disparity；点云/公制转换需要合适内参和以米为单位的baseline；说明涉及输入极线对齐和分辨率/迭代设置。工程引申：先复用既有合法Depth；内部一致率不独立认证接触区的绝对毫米误差。本轮不把官方通用去畸变要求当作对已处理VST再次去畸变的授权。

## S4 OpenCV calib3d

官方文档：`https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html`

支持的事实：相机投影/三角化、图像缩放下的内参变换、disparity到三维表面的回投。工程引申：surface sample不是隐藏接触点；crop/resize/flip必须显式处理坐标。文档不是本项目编码视点K/B正确的证明。

## S5 遮挡的手物表面重建

原论文：In-Hand 3D Object Reconstruction from a Monocular RGB Video。
`https://arxiv.org/abs/2312.16425`

支持的事实：研究将手物接触区域的遮挡作为重建难点，并使用遮挡先验与接触约束。工程引申：隐藏面需要独立重建/假设层，不应强迫可见证据门恢复不可见真值。本项目场景、传感器和实现不同，不移植该论文实验成绩。

## S6 ContactOpt

原论文：`https://arxiv.org/abs/2104.07267`

支持的事实：从手物mesh预测/优化接触以改善姿态，其软组织模型允许一定互穿。工程引申：接触可作优化先验，但不能把人手软组织策略直接当Kai刚性碰撞阈值。本轮不声称复现、不下载新模型。

## S7 Open3D距离查询

官方文档：`https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html`

支持的事实：区分无符号距离、signed distance和occupancy；内外判定需要适当闭合几何假设。工程引申：开放牌面patch不能支持完整物体体积穿透的声明。本轮实际碰撞引擎继续用项目既有实现，不迁移Open3D。

## 不应误读的本轮判断

- “Contact门可能同像素条件冲突”是待代码验证的假设；若使用3D正交足点，则不能宣称同样的布尔矛盾。
- “W0全部失败可能有scope/聚合问题”是审计问题，不是门过严的既定结论。
- 0.783730、19.33mm、6.60mm以及4/436均是用户报告的特定统计，不证明唯一误差原因或真实无接触。
- 本包预算、波次数量、时间盒与里程碑是工程设计，不是论文证实的最佳参数。
- 本轮目标是实片恢复和有限跨会话验证；不能保证15小时内一定得到严格Contact或合格整臂轨迹。
