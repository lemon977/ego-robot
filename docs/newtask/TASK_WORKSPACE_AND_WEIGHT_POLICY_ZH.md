# 多任务目录、算法与权重管理规范

日期：2026-09-02  
状态：已落地；机器校验入口为 `tools/validate_task_registry.py`

## 1. 直接结论

扑克和薯片的 Mask/Clean **不是各自一套学习权重**。

- Mask：当前两任务引用同一份 SAM3.1 checkpoint
  `sam3.1_multiplex.pt`，精确 bytes/SHA/来源/许可证只在
  `systems/mask/system.manifest.json` 登记一次。
- Clean：当前是确定性真实像素 donor 管线，没有学习权重。
- Robot：当前 actual-FK、几何、深度和碰撞链没有学习权重。
- 每任务确实不同的是配置：输入 SHA、左右身份、frame-0 instance anchor、可见
  tracker/cuff、物体 ID、保护 mask、几何类型、需保留对象、clean plate/donor 和 QA
  状态。这些必须隔离，但不能叫“权重”。

共享一份 checkpoint 也不等于已经证明通用。相同 v4 算法/权重/阈值在薯片手机视频
通过，在扑克视频仍因 tracker 时序失败而 HOLD。这正是“共享系统 + 任务独立证据”的
原因：复用代码和模型，同时不把一个任务的 PASS 偷换成另一个任务的 PASS。

## 2. 目录职责

```text
chaoyang/
├── systems/
│   ├── registry.json
│   ├── mask/system.manifest.json
│   ├── clean/system.manifest.json
│   └── robot/system.manifest.json
├── tasks/
│   ├── registry.json
│   ├── schema/
│   ├── templates/
│   ├── poker/
│   │   ├── README.md
│   │   └── task.manifest.json
│   └── chips/
│       ├── README.md
│       └── task.manifest.json
├── tools/validate_task_registry.py
├── archive/legacy/task_cards/    # legacy：2026-08 历史任务卡，不新增 run
└── _run/                         # legacy evidence：保留旧证据，不再堆新任务
```

未来任务固定使用：

```text
tasks/<task_id>/
  task.manifest.json
  inputs/                 # 只读 view/symlink，可选；原数据不复制
  annotations/<session>/
  runs/mask/<run_id>/
  runs/clean/<run_id>/
  runs/robot/<run_id>/
  runs/pipeline/<pipeline_run_id>/manifest.json
  reports/
  exports/
```

任务名用短、稳定的英文 slug；日期、参数和版本写 manifest/run manifest，不塞进几十字
目录名。pipeline manifest 只聚合引用三个 stage run，不复制它们的产物。`puke`、
`shupian` 只作为 manifest 中的 legacy alias，canonical 分别是 `poker`、`chips`。

## 3. 输入不复制

当前 `data/inputs/baseline_background/` 手机素材只在两个 task manifest 中保存相对路径、bytes、
SHA-256 和用途边界，没有复制视频：

- poker action → `*204*.mov`，object reference → `puke.mp4`；
- chips action → `*0ad*.mov`，object reference → `shupian.mp4`；
- 两者共享 `jingzhizhuomian.mp4` / `yidongzhuomian.mp4` 作为 PHONE_DEV 配准诊断和
  采集规划参考。

这两条空桌视频与动作视频机位/布局不同，manifest 明确禁止把它们当正式 Clean donor、
相机权威或正式 PICO 输入。正式数据到位后新增 `FORMAL_CAPTURE` 输入记录，不覆盖这些
PHONE_DEV 记录。

## 4. 每任务该分开的内容

### Poker

- Mask 对象：`card_0..2`；fixture：`rack`；当前还需 upper/lower tracker 标注。
- Clean 必须保留：三张牌、木架、`mat_plane`、`rack_top_plane`。
- Robot 几何：`CARD_THIN_BOX`、`RACK_BOX`、两层 support plane。
- 当前 Mask：`HOLD_CANARY12_NO_FULL_ARTIFACT_PROMOTION`，不能给 Clean。

### Chips

- Mask 对象：`chip_0..2`；fixture：`bowl`。
- Clean 必须保留：三片薯片、碗和 `mat_plane`。
- Robot 几何：`CHIP_SADDLE`、`BOWL_REVOLVE`、support plane。
- 当前 Mask：手机基线 v4 可消费；当前 Clean 仍因跨机位 static 不足而 HOLD。

两个任务各自在自己的 manifest 保存这些差异；一个 task 的 anchor、物体 ID 或 donor
不能复制给另一个 task。

## 5. Mask 标注如何管理

通用离线工具仍是 `tools/offline_mask_annotator/`。正常 CFR MP4 只需改输入 MP4、输出
目录和 session ID，默认均匀冻结 24 帧；输出建议写：

```text
tasks/<task_id>/annotations/<session_id>/mask_v1/
```

默认 24 帧是新 session 的覆盖性起点，不是“所有任务永远只标 24 张”的保证。已有失败
证据时，按失败区间冻结额外帧：当前扑克只需恢复双 tracker，最低 8 帧 × 2 tracker =
16 实例，稳妥版 11 帧 × 2 = 22 实例；已经通过的人体和 lower cuff 不重复标。正式
fit/eval/test 要按完整 session 划分，禁止把同视频相邻帧随机拆成训练和测试。

## 6. Clean 如何管理

Clean 没有“扑克权重”和“薯片权重”。每个 task/session 需要自己的：

- 已通过消费门的完整人体/sleeve/tracker Mask；
- 需保护的 task object/fixture 清单及逐帧状态；
- 同一 session、同一相机、同姿态/曝光的 2–3 秒无手 clean plate；
- 相机移动时的 view sweep、K/c2w/table support；
- 每像素 source map、unsupported mask 和可见残留复核。

如果空桌来自不同相机或布局，不应通过“换权重”解决；它是 donor authority 不成立。

## 7. 什么时候才新增权重

下列变化不产生新权重：改输入路径、物体类别、anchor、标注帧、object geometry、Clean
保留对象、donor bank、质量门状态。

只有训练/微调真正输出了新 checkpoint，才在对应 system manifest 新增条目。项目微调
权重必须同时具备：

- checkpoint bytes + SHA-256；
- base weight ID；
- 训练数据 manifest SHA 和 session 级 split manifest SHA；
- 训练代码 SHA、完整命令、环境与完成时间；
- eval/test manifest SHA、逐任务指标和 claim limit；
- 适用任务范围；
- 许可证 ID、文件 SHA 和可分发状态。

缺任一项先 HOLD。即使只服务 poker，也仍登记在 `systems/mask/`，设
`scope=task_scoped`、`applicable_tasks=["poker"]`；任务 manifest 只引用 ID 和系统
manifest SHA，不出现散落的绝对 checkpoint 路径。未来 checkpoint 建议落在被 Git
忽略的大文件区 `assets/models/<system>/<weight_id>/`，但 registry 必须保留精确身份。

## 8. 新任务复制流程

1. 从 `tasks/templates/` 复制 manifest/README 到 `tasks/<new_slug>/`。
2. 重新登记输入路径、bytes、SHA；不复制 RAW。
3. 重新定义 object IDs、几何、左右身份、可见 role 和 Clean preserve 清单。
4. 引用 `systems/registry.json` 的当前系统 SHA；默认先复用共享算法/权重。
5. 先冻结 canary/标注计划，再跑新任务；不能看失败结果后偷换评测帧。
6. 每次写 fresh `tasks/<task>/runs/<system>/<run_id>`，禁止覆盖旧 run；端到端索引只写
   `runs/pipeline/<pipeline_run_id>/manifest.json` 引用，不复制 stage 结果。
7. 执行校验：

   ```bash
   python tools/validate_task_registry.py
   ```

   默认会验证 schema、系统/task SHA、代码/协议/证据和小型输入文件；不会读取 3.5 GB
   checkpoint 全字节。需要重验 checkpoint 时加 `--deep-weight-hash`。

校验器还会拒绝 task-local checkpoint path、未注册 weight ID、系统 SHA 漂移、PHONE_DEV
素材未声明 formal 禁用、物体合同与 Mask/Clean/Robot 对象集合不一致等混乱来源。

8. 通过后由分配器创建不可覆盖的 stage run 根：

   ```bash
   python tools/allocate_task_run.py \
     --task poker --system mask --run-id 20260902_mask_v1
   ```

   它会在新目录写 `RUN_MANIFEST.json`，冻结 task/system manifest SHA、algorithm ID 与
   weight IDs，但不会运行 producer。重复 run ID、路径穿越和跨 system 写入会直接失败。
