# 后续 AI 工作入口

状态：CURRENT；当前执行范围是[四支线算法基线 V2](PLAN.md)，不是历史 V3.2 checkpoint 训练。

## 启动顺序

1. [当前 receipt](../governance/CURRENT_STATUS_RECEIPT.json) 与 [权威登记](../governance/DOC_AUTHORITY_MAP.json)。
2. [当前任务索引](../../tasks/current/INDEX.json)、[算法合同](../governance/ALGORITHM_CONTRACT.json)。
3. [PLAN](PLAN.md)、[机器生成状态](STATUS.json)，随后只读本支线：[Exact78](EXACT78.md)、[PICO/MANUS](AI1.md)、[HaWoR/Kai22](AI2.md)、[HuRo](AI4_HURO.md)。

```bash
PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli validate-governance
```

治理PASS说明登记和引用在检查时一致，不认证真实精度。FRESH只用于活动任务心跳。当前浅层STATUS是同一publisher发布的时间点快照，不通过mtime找“最新结果”。

任务包绑定实际base、writer PID/startticks/epoch/fence、读写范围和入口。单一publisher负责共享文档与合同；worker只写自己的worktree和lane。接续前取得明确交接，不覆盖仍存活的writer、不伪造心跳、不停止别人。

源数据、processed、archive与封存结果只读；全部新文件位于chaoyang。不依赖外部兄弟项目、不下载模型、不改共享环境。只有用户额外批准的HuRo独立环境可联网补必要依赖。

## 不可跨越的语义

- 不以工程测试PASS替代算法质量、人工审阅、训练资格或真机资格。
- 不用 Attachment 证明对象身份/Contact；不恢复已撤权Depth/Object6D。
- 图像按已核验sourceIndex裁眼/resize；不得把RAW fisheye参数重复用于encoded VST像素。
- fixed-prior、虚拟安装和物理标定分开；不以好看的叠加证明标定。
- 观测、补帧、缺失、左右手及各模态有效性独立保留。
- 历史R0已clip；旧手部初始化的用途仍保留，不改写历史。

## 已完成数据清洗与历史查询

0911/0914/0915清洗见[数据清洗基线](DATA_CLEANING_0911_0915_ZH.md)；0916见[0915/0916交接](FULL_FUNNEL_0915_AND_CLEANING_0916_ZH.md)。本轮不重新清洗921条、不改数据布局。

历史V3.2及其hand-only结果在原任务attempt保留；本轮[产物根](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/)中的前驱文档快照只供追溯，不能直接续跑旧计划。当前采用与下一步任务以本轮最终发布清单为准，未发布期间不要把中间输出当完成。
