# 后续AI接手步骤

1. 在项目内设置TMP、缓存与输出；运行 `PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli validate-governance`。
2. 读取[总入口](README_ZH.md)、[机器状态](STATUS.json)、[当前任务索引](../../tasks/current/INDEX.json)和receipt绑定的[权威登记](../governance/DOC_AUTHORITY_MAP.json)。
3. 仅阅读本线短卡和其中明确的最新RESULT/STATE。需要追溯再查[历史索引](../plans/INDEX_ZH.md)，不要从旧计划恢复命令。
4. 当前无活动算法任务。[后续建议](NEXT_ACTIONS_ZH.md)不等于执行授权；获得授权后登记有限任务、冻结输入/代码/环境/门槛并取得独立写入范围。
5. 发布前核对PID/startticks/epoch/fencing；不得热改其他执行者、复活历史包或覆盖封存结果。
6. 报告执行、结构、质量、改善、审阅与采用六个独立状态；视频存在与CPU测试PASS不是产品PASS。

旧AI1=Sensor，AI2=HaWoR/Motion，AI4=HuRo；支线1现为共享Scene/Clean。CPU维护B与原始数据清洗不是额外算法支线。

仅在原H20上能直接解析历史绝对路径；GitHub克隆不包含模型、视频、processed、环境和运行证据实体。见[Git交付说明](GIT_DELIVERY_ZH.md)。
