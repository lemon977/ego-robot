# 支线4：官方HuRo核心适配

官方核心已实际接入并完成007/031共527帧，不再停在依赖预检。两条完整对照视频来自真实58DOF求解，不是历史derived hand-only方法改名。不是完整Stage8复现。

- [核心交接与精确环境](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/HANDOFF_HURO_ZH.md)
- [核心不可变结果](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/HURO_CORE_HANDOFF_RESULT.json)
- [共同渲染与公平评价](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/common_review_v2_run1/RESULT.json)
- [九片时序/幅度审计](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/exact78/temporal_audit_v2/README_ZH.md)

工程接入可复现：项目内独立env、固定官方源码与依赖、CPU真实求解，官方与独立FK最大差低于8e-7，固定placement目标误差为0。未下载模型、未修改活跃环境、未使用GPU。

算法候选 **REJECTED_QUALITY**：476个有效侧帧全部超过既有限位容差，最大约0.046rad；包装器不暴露收敛状态。007部分手指误差下降但腕部误差增大；031仍有米级目标偏差。不能以loss下降或只看手指指标宣称真实贴合。

诊断渲染直接消费原q，明确显示RAW LIMIT VIOLATIONS；数学FK只为显示失败输出创建无限位检查的局部不可变视图，原生产FK和资产限位不变，绝不clip或删除越界帧。官方索引时间正则、32帧分段及未知姿态先验均在交接披露。

入口：`run_huro_fixed_placement_core_v2`、`run_huro_sessions_core_v2`、`run_huro_common_review_v2`，均通过已登记`chaoyang run`调用；求解使用自身env，渲染使用维护解释器。新候选需要新签名与登记，先解决硬限位、姿态约束和真实dt问题，不扩大为placement搜索或训练。
