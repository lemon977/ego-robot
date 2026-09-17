# 数据入口

本目录保存工程输入、历史参考、隔离实验素材，以及按任务/会话/阶段组织的处理数据。
权威 RAW 始终留在 `/mnt/data/egodata/datasets/ego/`，本项目只记录只读路径与哈希，
禁止复制或改写 RAW。

- `inputs/`：可被任务 manifest 引用的输入素材。当前手机开发素材统一在
  `inputs/baseline_background/`。
- `references/`：只读历史参考。当前 Robot 004 历史媒体在
  `references/robot_004_old/`，不等同正式几何或轨迹 authority。
- `experiments/`：不进入正式任务结果的隔离试验素材。
- `processed/<task>/<session>/<stage>/<version>/`：项目处理数据。固定包含来源 manifest、
  阶段输出、QA 和复现信息；例如
  `processed/chips/get_potato_chips_0901_001/hawor/20260902_v2/`。一个阶段不得混写
  Mask、Clean 或 Robot 产物。

扑克和薯片通过 `archive/baseline-20260917-0aa69e9/content/history/tasks/<task>/inputs/` 的相对软链引用这些文件，不复制大视频。

`archive/baseline-20260917-0aa69e9/content/history/data/processed/` 是数据产物层，`archive/baseline-20260917-0aa69e9/content/history/tasks/<task>/runs/` 是一次算法执行及门控证据层。
后者只能通过 manifest 引用前者，不能复制一份同名结果。
