# Frozen local evidence

本目录只保留 exact78 的紧凑冻结数值证据，不再包含 producer、环境、旧视频、mask、CLEAN、overlay 或 RGB symlink。

它是 checkpoint/R2 谱系审计的本地证据，不是新流水线的正式图像输入，也不会提交 Git。新代码不得写入这里；所有 staging 写 `_run/`，正式结果经验证后写 `processed/`。
