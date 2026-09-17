# HaWoR 本地运行快照

此目录是上游 HaWoR 提交
`66c7d4108d58a716deccd192cb7645170cdc7bd7` 的未修改源码快照，并包含本项目运行所需的
HaWoR、infiller 与 detector 权重，以及本地运行所需的 MANO 左右手模型。HaWoR
许可证见 `license.txt`；MANO 模型仅限已获授权的本地研究环境使用，不随公开发布包分发。

本项目的适配、质量门和数据组织逻辑不写入此目录，而在：

- `../../tools/run_hawor_pico_canary.py`
- `../../data/processed/<task>/<session>/hawor/<version>/`

不得把原始数据复制到这里，不得直接修改 vendor 源码。若需修复接口，应在本项目适配层
实现并记录源码、权重和输入 SHA-256。
