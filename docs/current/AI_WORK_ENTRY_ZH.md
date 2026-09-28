# AI工作入口：先确认本轮已结束

1. 阅读AGENTS与本目录README，核验生成STATUS与当前receipt。
2. 本轮completion任务全部终态，当前任务包索引不提供其执行路由；不得沿历史next_action续跑。
3. 结果事实以最终结束清单及不可变前驱为准，不把任务关闭等同质量通过。
4. 其他AI任务、维护目录和数据原件保持不动。新任务必须有新授权、范围、writer与登记。

## 当前证据与观看

- [最终结束清单](../../_run/current/four_stream_completion_20260928/attempts/attempt_0001/final_closeout_20260928/RESULT.json)：任务关闭与质量采用分开；质量采用0，人工审阅未完成。
- [逐会话交付、取消与文件校验](../../_run/current/four_stream_completion_20260928/attempts/attempt_0001/final_closeout_20260928/DELIVERY_INVENTORY.json)
- [观看说明](../../_run/current/four_stream_completion_20260928/attempts/attempt_0001/final_closeout_20260928/先看这里.md)
- [工程回归](../../_run/current/four_stream_completion_20260928/attempts/attempt_0001/final_closeout_20260928/ENGINEERING_CHECKS.json)
- [历史检查点8](../../_run/current/four_stream_completion_20260928/attempts/attempt_0001/checkpoints/PROGRESS_0008.json)：历史事实保留，其后续研究建议已停止。
- [生成状态](STATUS.json)、[支线1/Clean](V5_SCENE.md)、[支线2/PICO＋MANUS](V5_SENSOR.md)、[支线3/HaWoR](V5_MOTION.md)、[HuRo终态](V5_HURO.md)。
