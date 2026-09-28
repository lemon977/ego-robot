# Human→Robot 当前AI入口

先运行 `PYTHONPATH=src python -m chaoyang.cli validate-governance`，再读取：

1. [当前终态](PLAN.md)
2. [唯一结果导航](SHARED_HAND_DELIVERY_RESULT_ZH.md)
3. `docs/current/STATUS.json`
4. 仅在核查具体失败时读取相应lane的 `STATE.json` 与结果收据

当前无活动任务。不得恢复本轮Robot、Data或Clean候选，不得把局部数值改善、视频存在、有限loss或结构PASS改写成产品/标签质量通过。新工作必须经用户明确授权并登记；所有写入、TMP、缓存继续限制在 `/mnt/workspace/code/chaoyang/` 内。
