# Retained HumanEgo policy core

本目录只保留当前项目需要的 HumanEgo 模型、dataloader、R2 schema、H50/rolling evaluation、最小推理逻辑和测试。旧采集、预处理、硬件控制、下载器、session 特调和视觉替换 producer 已清除。

正式训练必须通过 `tools/train_embodiment.py`，并显式提供 reviewed `--config`、`--split` 和对应 sidecar contract。RobotRGB 虚拟 selector 的历史实现已删除；新的 manifest-backed selector 只能在 G1 合同审核后实现。

上游/历史许可证见 `LICENSE`，冻结数值数据和 checkpoint 的位置与 SHA 见项目根 `DATA_SPEC.md`。
