# Exact78 V3.2 执行入口

总计划：[PLAN.md](PLAN.md)；机器状态：[STATUS.json](STATUS.json)。

固定 cohort 为 156 个历史成员，不补位：train=0901 两个 source group（59），validation=0902（77），
development final=0903（20）。0903 已暴露，只是冻结后的 development final，不是首次盲测。

当前顺序：身份与 lineage 重绑定 → 真实 Raw/Robotized 像素和 H50 标签 → 两次隔离生产的
suffix-invariance → loader/loss/backward/参数变化/save/reload → 四模型共同 update 训练。Robotized 四小时
仍被可复现错误阻塞时，可单独建立 `RAW_ONLY_DEV`，但不得冒充成对 A/B 完成。

四模型是 Chips/Poker × Raw/Robotized 的 H50 future-2D Visual Aux，不是 Robot policy。第一资源周期
四模型合计 12 GPU 小时；任何临时 A/B 只比较共同 checkpoint，并同时报告 stay-put 与 constant-velocity。
Stereo canary 独立终态化，不阻塞 pair 或训练。
