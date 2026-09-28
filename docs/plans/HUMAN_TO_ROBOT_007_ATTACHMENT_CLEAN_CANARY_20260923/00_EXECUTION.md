# 007 已封存附件跟踪掩码 → 真实 ProPainter 固定窗 canary

固定会话 `get_potato_chips_0915_007`、原帧181–196、physical-left resize-only域。上游只读：已封存 rebound 人手/线缆模型输入与write/protect、质量未采用但可追溯的附件跟踪16帧候选掩码、原始RGB、旧拒绝Clean。新的输入按每帧 `new_write=old_write|attachment_candidate`，模型域用既有 `merge_support` 规则闭合；旧protect不得被覆盖或删除。旧任务物体mask与附件候选的交集逐帧记录为**未解决语义冲突**，不当作可信对象保护或零风险证据。帧184六个投诉设备点必须全部进入新write；所有16帧附件掩码实际存在、非空且域正确，否则G0失败，不运行模型。固定参考点不当像素GT。

满足结构准入后，只运行一次当前固定ProPainter权重和旧同配置16帧推理。原始/旧拒绝Clean/新候选Clean/删除支持做同帧审阅；输出全帧无损PNG与可全解码MP4。编码前write外及旧可信protect区域逐像素不变。最终独立审阅181、184、190、196与连续窗的残手、设备、盘/薯片、污迹和闪烁。即使局部改善，源附件与任务物体重叠未消除前也不授予Clean质量；质量C不原样重试、不扩378帧、不改四产品采用。

墙钟最多2小时、GPU最多30分钟、单租约；一次正常attempt和一次同签名运行时故障恢复，质量不重试。无新模型下载、训练、真机。旧attempt、raw、processed、archive、sealed只读。新增产物、TMP、缓存只在项目内。
