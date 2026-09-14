# Third-party provenance and redistribution notes

本仓库保留的 HumanEgo 源码来自内部历史提交 `2101d562b9c4ed4dcc95f16981f9b2bb5cbd7dd3`，其许可证见 `HumanEgo/LICENSE`（PolyForm Noncommercial 1.0.0）。提交或发布前应继续保留许可证与变更说明。

HaWoR 原来源为 <https://github.com/ThunderVVV/HaWoR.git>，锁定提交 `66c7d4108d58a716deccd192cb7645170cdc7bd7`，许可证为 CC BY-NC-ND 4.0。历史上含本地修改的平行副本已经移除；当前按用户要求在 `third_party/HaWoR/` 保存该提交的未修改源码快照、原许可证和运行权重。项目适配代码只能放在本仓库 `tools/`/`pipeline/`，不得直接改 vendor 源码。

ProPainter 原来源为 <https://github.com/sczhou/ProPainter.git>，历史锁定提交 `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`。本轮已删除内嵌副本；需要背景 inpainting 时再按合同锁定外部依赖。

Tianji、KaiHand、法兰及其 URDF/package/CAD/mesh 的再分发权尚未确认，因此 `.gitignore` 默认排除运行资产，只提交 README 与 hash provenance。确认权利和远端可见性后再决定是否纳入源码仓库。

SAM3 源码来自 <https://github.com/facebookresearch/sam3.git>，锁定提交 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`，canonical 未修改源码位于 `third_party/SAM3/`，许可证原文见 `third_party/SAM3/LICENSE`。共享 SAM3.1 权重以官方模型 revision 与经核验的镜像 transport revision 双重固定；镜像仅作为字节传输来源，不改变官方代码、模型身份或许可证约束。

FoundationStereo 原来源为 <https://github.com/NVlabs/FoundationStereo>，本地代码位于 `third_party/FoundationStereo/`，许可证原文见 `third_party/FoundationStereo/LICENSE`（研究/非商业使用限制）。当前本地 vendor 目录没有独立 `.git` 元数据或可信的 upstream commit 标记，因此文档**不得凭记忆声称某个 FoundationStereo commit**。本轮可执行身份改用逐文件闭包：PICO适配脚本 SHA `a3dd48f22508c603d2ae689ed17af25ff49393d5d440f00de8ae87fbaa4d9284`、ViT-Large `23-51-11` checkpoint SHA `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`、配置 SHA `a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f`。若未来替换 vendor，必须先新增可信 upstream revision 记录和tree manifest，不能沿用本段SHA。
