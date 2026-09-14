# 新任务工作区（canonical）

从 2026-09-02 起，新任务放在 `tasks/<task_id>/`；旧单数目录已归档到
`archive/legacy/task_cards/`，其中是 2026-08 的历史任务卡和审计语境，不再用于创建
新任务或新运行。

## 固定布局

```text
tasks/<task_id>/
├── README.md
├── task.manifest.json       # 任务配置、输入只读引用、系统 SHA、当前证据
├── inputs/                  # 可选：只读 symlink/view；原视频不复制
├── annotations/             # 原始人工标注和独立 QA；永不被运行覆盖
├── runs/
│   ├── mask/<run_id>/       # Mask fresh run
│   ├── clean/<run_id>/      # Clean fresh run
│   ├── robot/<run_id>/      # Robot fresh run
│   └── pipeline/<id>/       # 仅聚合三个 stage run 的 manifest 引用
├── reports/                 # 被接纳的任务级报告
└── exports/                 # 对外查看包；不作为训练/算法输入
```

空目录无需提前制造；第一次写入时按上述名字创建。新运行默认写
`tasks/<task_id>/runs/<system>/<run_id>`，不再往 `_run/` 根堆长目录名。一次端到端链路
可在 `runs/pipeline/<pipeline_run_id>/manifest.json` 引用三个 stage run，但禁止复制
stage 产物。现有 `_run/` 是历史证据，本次没有移动或删除。

## 哪些共享，哪些按任务分开

- `systems/`：算法、基础 checkpoint、许可、代码 SHA 和跨任务硬门，共享。
- `tasks/<task>/task.manifest.json`：物体类别、几何、frame-0 anchor、标注计划、需保留
  对象、clean plate/donor、当前 PASS/HOLD，各任务分开。
- `tasks/<task>/runs/<system>/<run_id>`：任务和系统双重隔离，不允许混写。
- 任务目录不得直接登记 `checkpoint_path` 或复制权重；只能引用 system manifest SHA 和
  system 内的 `weight_id`。

当前扑克和薯片使用同一 Mask checkpoint；Clean/Robot 没有学习权重。详见
[`systems/README.md`](../systems/README.md)。完整中文规范见
[`TASK_WORKSPACE_AND_WEIGHT_POLICY_ZH.md`](../docs/newtask/TASK_WORKSPACE_AND_WEIGHT_POLICY_ZH.md)。

新 PICO/HaWoR 数据先执行
[`HAWOR_MASK_CLEAN_ROBOT_QUALITY_GATES_V1.md`](../docs/newtask/HAWOR_MASK_CLEAN_ROBOT_QUALITY_GATES_V1.md)
中的分层准入；机器入口为 `tools/evaluate_session_stage_admission.py`。HaWoR 的 Mask 种子门
与 Robot 三维门是两个独立结论，缺少下游摘要只能是 `NOT_EVALUATED`。

扑克/薯片现行端到端顺序、各阶段输入输出、双目深度、Object6D 与人工视频合同见
[`POKER_CHIPS_DATA_PROCESSING_ZH.md`](../docs/pipeline/POKER_CHIPS_DATA_PROCESSING_ZH.md)。旧15小时计划已归档。

明日补采物体几何与任务逐帧位姿时，按
[`OBJECT6D_POKER_CHIPS_CAPTURE_ZH.md`](../docs/acquisition/OBJECT6D_POKER_CHIPS_CAPTURE_ZH.md)
执行；全方位物体视频只解决模型形状/外观，不能替代任务视频中的逐帧 Object6D。

## 新增第三个任务

1. 复制 [`templates/task.manifest.template.json`](templates/task.manifest.template.json) 和
   [`templates/TASK_README_TEMPLATE_ZH.md`](templates/TASK_README_TEMPLATE_ZH.md) 到新的短
   slug 目录；不要复制扑克/薯片的 anchor 或物体配置。
2. 把输入作为只读 manifest 记录：相对路径、bytes、SHA-256、允许/禁止用途；不要复制
   原视频。
3. 引用 `systems/registry.json` 中的系统 SHA。仅当真的训练出新 checkpoint 时才先更新
   system manifest。
4. 运行：

   ```bash
   python tools/validate_task_registry.py
   ```

5. 用分配器原子创建一次性 run 根，例如：

   ```bash
   python tools/allocate_task_run.py \
     --task poker --system mask --run-id 20260902_mask_v1
   ```

分配器会冻结 task/system manifest SHA，拒绝覆盖、路径穿越和跨任务/系统混写；它只
创建 `RUN_MANIFEST.json`，不会自动执行算法。验证通过后再启动 producer。模板和 schema
不代表算法已经对新任务通过；新任务仍需自己的 canary、Clean donor 门和 Robot
几何/碰撞门。
