# 新传感器 H4 有界决策

当前“白色手套 + Controller 的 SAM3.1 自动 Mask → Clean → Robotized RGB”路线封为 `FAILED_QUALITY_C`，不再对同一输入、代码、权重和提示合同无限重试。

两条冻结 canary 的手套左右实例门均为 0%；Controller 角色门均只有 39.58%；Chips 的任务物体保护门为 0%。因此没有启动 fresh Clean。

此决定不等于新传感器数据不可用。Controller + MANUS 手部观测、触觉时序、修复标定合同后的 Stereo 和 Raw 视觉研究仍可独立推进。正式 H4 机器状态保持 `BLOCKED_RESOURCE / NOT_EVALUATED / POLICY_DEFERRED`，像素 Mask authority 仍为 false；开发 canary 的质量 C 不覆盖正式状态。
