# Human→Robot R2：根因优先、按会话推进、十二小时闭环

旧 V5 与 R1 只读封存；R2 保留四支线，但按真实依赖推进，不使用 `all(031,007)` 全局闸门。

## 固定边界

- task：`human_to_robot_root_cause_gated_r2_20260923`
- route：`HUMAN_TO_ROBOT_BASELINE_V1_R2`
- 12 小时总预算；前 6 小时为根因与首候选检查点，后 6 小时完成条件下游、回归、比较和收尾。
- raw、processed、archive、sealed 只读；不训练、不下载模型、不调用真机。
- `training_eligible=false`、`control_ground_truth=false`、`physical_deployable=false`、`external_metric_authority=false`。

## 执行顺序

Lane 1 负责 031 闪烁/角色链、007 绑带和黄色线、四会话 Scene/Clean、遮挡和 Contact 筛选；Clean 运行前只做输入准入，调用、完整输出和解码属于运行后结构验收。

Lane 3 从启动即修 031 ROI/模型输入、HaWoR/raw/bounded/R0、装配和唯一 Product 入口；不等待 Clean。031 与 007 按各自输入就绪独立合成，0902_103/042 做真实同配方回归。

Lane 2 先让 0916_097 真实通过 Controller/MANUS→HandMotion→共同 target-builder→solver→FK，再按同一后端扩展 098/101；不使用 HaWoR 补传感器缺失。

Lane 4 前半程做 Local R0/HuRo 的关节、限位、FK、腕目标和四种渲染确定性诊断；后半程才做同输入、同评估器的数值和条件同背景比较，不宣布未经验证的赢家。

## 状态与验收

结果分开记录 `execution / structure / quality / adoption`。每个阻塞必须写明缺失产物、消费者、owner、解除动作和不受影响的工作。候选可执行但不能绕过质量门；Raw overlay、历史视频、空 Motion 或改名文件不能充当产品。

最终分别报告 Scene/Clean、Motion/R0、Sensor 后端、Product、遮挡/Contact、Local/HuRo、入口/resume 的执行、结构、质量、采用和改善证据。终态允许 `TERMINAL_WITH_GAPS`，但不得将其写成全部成功。
