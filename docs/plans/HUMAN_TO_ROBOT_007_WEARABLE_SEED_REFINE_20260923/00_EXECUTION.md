# Human→Robot 007 指端设备：单帧 text+box 建 ID、point 细化

这是上一轮 point-only 0/6 的**不同、有限**候选，不修改已封存结果。只在 `get_potato_chips_0915_007` 原始 physical-left resize-only 帧 184 上调用固定 SAM3.1 权重。六个设备的 box、正负点沿用上一轮冻结值；每个设备独立 state，以固定文本 `finger-mounted sensor` 和 box 建立原始实例 ID，再对该 ID 施加原有点提示。不得根据输出换文本、点、box、阈值或帧。先保存 seed 和 refined 原始 ID/mask，再按既有 20–15000 px、正点覆盖、负点排除、至少半数面积落在原 box 中的门检查。

总墙钟最多两小时，GPU最多30分钟，单租约。只允许一次正常调用和一次同签名运行时故障恢复；质量失败不重试。本任务不做传播、ProPainter、Clean 或产品采用。若六个均失败，保留原始掩码及失败原因，后续工作转到其他可证伪的来源或场景方法，不在 SAM 提示词上无界搜索。所有新写入与缓存位于项目内。原始、历史 attempt 和收据只读。
