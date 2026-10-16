# V1全链路和实际算法

本页只描述199条原版生产链。最终选用与全部SHA以[MANIFEST](MANIFEST.json)为准，历史名称只标明依赖来源。

| 阶段 | 实际实现 | 固定输入／输出与边界 |
|---|---|---|
| 原始读取与时间 | `pick_place_1008/source.py`、`camera.py` | 原始`vst.h264`直接解码、冻结眼别裁切／resize到640×480、QPC／PTS映射；不把历史转码视频替代原始输入 |
| 动作读取与安装 | 同目录`source.py`；冻结Camera／installation配置 | 同会话Controller/Head腕运动＋MANUS骨架；未验证公制标定和视觉拟合安装先验仍明确保留 |
| Robot动作 | `pick_place_1008/pipeline.py::motion_worker`；共享`exact78_baseline/retarget.py`与`robot_model.py` | 完整Tianji双臂、法兰、KaiHand22掌／指；原动作数组、有效性和诊断标记不改成实机真值 |
| 分割与跟踪 | `exact78_baseline/mask.py::Sam3Video`、`pick_place_1008/sam_point_adapter.py`及`mask_seeds.py` | SAM3.1唯一语义分割核心；199个原逐会话可见点种子及反向传播边界修复，不偷偷改成其他候选 |
| 去除／保护遮罩 | `pick_place_1008/visual.py::build_visual_masks` | 人手、手套／设备去除支持与可信可见物体保护分开；隐藏物体区域保持UNKNOWN，不因补出像素而获得真实支持 |
| Clean | `exact78_baseline/clean.py::ProPainter`、`pick_place_1008/visual.py::ProtectedProPainter` | 同一RAFT＋flow completion＋ProPainter路径和冻结SETTINGS；保护像素恢复／校验、有效去除域外原像素不变。非因果离线补图，不作为因果训练RGB |
| 渲染与合成 | `pick_place_1008/render_stage.py`、共享Robot渲染／z-buffer | Robot内部完整几何深度与原物体支持合成；场景公制Depth／Object6D／真实Contact未作为可信生产输入。现有遮挡策略不是物理遮挡准确性的证明 |
| 发布 | `pick_place_1008/pipeline.py`与既有执行器 | 独立source/motion/gpu/render阶段收据、逐帧状态、`final.mp4`、`review.mp4`、`SESSION.json`及批次INDEX；partial不算完整交付 |

## 权重、资产和配置

SAM3.1使用`assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt`；ProPainter三权重位于`assets/models/vendor/propainter/`，其`vendor/ProPainter/weights/`逻辑路径必须继续可用。第三方源码、许可证、Tianji／KaiHand URDF和STL、相机固定法兰及安装合同都从原阶段闭包引用，不另猜资产版本。

`MANIFEST`分别保存19算法核心、93项历史运行包装、当前新包装闭包和三阶段合并依赖。治理包装变化单独签名，不冒称历史93文件逐字相同。历史阶段包含421个唯一路径；大权重身份沿原已签名收据和大小核对，未重新读取时明确记录，没有把stat校验冒充新全量SHA验证。

原始manifest、拟合Camera、all199种子索引和种子leaf均有SHA。原来每批种子子集及source复用索引仍保存；重跑不得临时选择C3/QP/P9修正结果或更换相机先验。

## 资格不是一列PASS

采用：用户指定原版199作为V1。执行：原199已完成，计数沿冻结交付。视觉：保留原FAIL／缺陷／人工待审。几何、真实Robot动作、物理部署和训练资格：本基线不授权。后续研究ICT训练另按已有合同，不把本轮采用当作新的训练前置，也不修改其已封存标签。
