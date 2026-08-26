# Third-party provenance and redistribution notes

本仓库保留的 HumanEgo 源码来自内部历史提交 `2101d562b9c4ed4dcc95f16981f9b2bb5cbd7dd3`，其许可证见 `HumanEgo/LICENSE`（PolyForm Noncommercial 1.0.0）。提交或发布前应继续保留许可证与变更说明。

HaWoR 原来源为 <https://github.com/ThunderVVV/HaWoR.git>，历史锁定提交 `66c7d4108d58a716deccd192cb7645170cdc7bd7`，许可证为 CC BY-NC-ND 4.0。由于本地副本曾含修改且 NoDerivatives 条款存在风险，本轮已删除内嵌副本；后续只按审核后的固定版本作为外部依赖获取，不把修改副本上传本仓库。

ProPainter 原来源为 <https://github.com/sczhou/ProPainter.git>，历史锁定提交 `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`。本轮已删除内嵌副本；需要背景 inpainting 时再按合同锁定外部依赖。

Tianji、KaiHand、法兰及其 URDF/package/CAD/mesh 的再分发权尚未确认，因此 `.gitignore` 默认排除运行资产，只提交 README 与 hash provenance。确认权利和远端可见性后再决定是否纳入源码仓库。
