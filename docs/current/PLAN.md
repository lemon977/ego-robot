# 四线稳定基线执行计划 V3.2

状态：CURRENT EXECUTION PLAN  
任务：`four_stream_pretraining_baseline_v32`

本轮从提交 `f43357d37ed9bfd089832744402f1880699dd347` 起步，只在当前功能分支工作，不切换
`main`、不 reset、不 push。Exact78、AI1、AI2、AI4/HuRo 使用隔离 writer 和产物根；唯一 publisher
串行更新共享任务索引、合同和当前文档。源数据、processed、archive 与 sealed 资产只读。

## 四线目标

- Exact78：恢复固定 156 成员及 source-group split，生产真实 Raw/Robotized pair，完成一次真实参数更新、
  checkpoint 重载，并按共同 update 推进四个 H50 Visual Aux 模型。Chips023 Stereo 是独立有界实验，
  不阻塞训练。
- AI1：冻结 097/098/101 原生 MANUS25、PICO/controller 与 M0；复现并诊断巨大 M1 偏移。
  `selected=M1` 不等于 adopted。只有明确修复或非拟合门通过才可首次打开 102/103。
- AI2：先为 `play_cards_0915_031` 和 `get_potato_chips_0915_007` 生产与被评 HaWoR 解耦的
  可观测证据，再在同一帧集比较 raw/bounded，并生成完整 Kai22 q/FK sidecar 与分层 R0。
- AI4：固定 HuRo 官方提交 `033197778fcc30edc3631dddf3343a967683da09`，只做共同 MANO21 输入的
  fixed-wrist、root-relative Kai22 hand-only 外部对照。它不是官方完整 HuRo 复现；Wuji20 因资产缺失阻塞。

## 隔离与调度

Canonical 根固定为：

```text
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/
├── exact78/
├── ai1/
├── ai2/
└── ai4_huro/
```

CPU 可并行，初始总软上限 8 线程。GPU 始终单租约；第一资源周期上限为 Exact 12h、必要观测 4h、
HuRo 4h、Stereo 0.25h。T+24h 是复核点，不是成功期限。预算耗尽须保存可恢复状态，不能把暂停写成完成。

诊断只约束直接消费者，一线失败不得清零其他线。任何缺失值不得以零、单位旋转或 forward-fill
冒充观测。评价口径修复、数值改善、训练完成、视觉审阅、训练准入、控制真值与真机资格分别报告。

## 固定权限

默认且不得由内部一致性自动升级：

```text
control_ground_truth = false
physical_deployable = false
external_metric_authority = false
```

历史产物不可覆盖。若新证据证明旧结果只在特定用途上错误，应追加局部限权收据，而不是修改旧字节或
全局清零。实时进度仅看 [STATUS.json](STATUS.json) 与各 lane 的不可变结果；本计划不证明算法已经运行。
