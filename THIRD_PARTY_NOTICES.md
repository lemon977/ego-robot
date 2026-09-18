# Third-party provenance and redistribution notes

本仓库保留的 HumanEgo 源码来自内部历史提交 `2101d562b9c4ed4dcc95f16981f9b2bb5cbd7dd3`，其许可证见 `src/chaoyang/human_ego/LICENSE`（PolyForm Noncommercial 1.0.0）。提交或发布前应继续保留许可证与变更说明。

HaWoR 原来源为 <https://github.com/ThunderVVV/HaWoR.git>，锁定提交 `66c7d4108d58a716deccd192cb7645170cdc7bd7`，许可证为 CC BY-NC-ND 4.0。历史上含本地修改的平行副本已经移除；当前按用户要求在 `vendor/HaWoR/` 保存该提交的最小运行源码与原许可证，权重本体位于 `assets/models/vendor/hawor/`。项目适配代码只能放在本仓库 `src/chaoyang/ops/`/`src/chaoyang/pipeline/`，不得直接改 vendor 源码。

ProPainter 原来源为 <https://github.com/sczhou/ProPainter.git>，历史锁定提交 `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`。当前仅保留 `vendor/ProPainter/` 的运行闭包与许可证；嵌套 `.git`、示例和媒体已归档，权重位于 `assets/models/vendor/propainter/`。

Tianji、KaiHand、法兰及其 URDF/package/CAD/mesh 的再分发权尚未确认，因此 `.gitignore` 默认排除运行资产，只提交 README 与 hash provenance。确认权利和远端可见性后再决定是否纳入源码仓库。

SAM3 源码来自 <https://github.com/facebookresearch/sam3.git>，上游身份锁定到提交 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`（完整上游 tree `6bc2384dbbfb370fe28096d23955df9b7b0bcdd9`）。`vendor/SAM3/` 仅保留 226 个文件的运行时闭包，不再宣称是完整上游树；闭包 tree、字节数和文件数登记在 `manifests/vendor.json`，示例、测试等非运行内容已归档。许可证原文见 `vendor/SAM3/LICENSE`。共享 SAM3.1 权重以官方模型 revision 与经核验的镜像 transport revision 双重固定；镜像仅作为字节传输来源，不改变官方代码、模型身份或许可证约束。

FoundationStereo 原来源为 <https://github.com/NVlabs/FoundationStereo>，本地代码位于 `vendor/FoundationStereo/`，许可证原文见 `vendor/FoundationStereo/LICENSE`（研究/非商业使用限制）。当前本地 vendor 目录没有独立 `.git` 元数据或可信的 upstream commit 标记，因此文档**不得凭记忆声称某个 FoundationStereo commit**。当前 successor 可执行身份改用逐文件闭包：PICO适配脚本 SHA `dc6a68d200542932aa014d3cd656ba52d84a4430cddbf8658edb990f372d33c6`、FoundationStereo runtime tree SHA `cdd17c958140e132263b2dfc39e2381fd45efcb13d8f0e3a1b82e702c90a9302`（198 文件、2,305,320 bytes）、ViT-Large `23-51-11` checkpoint SHA `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`、配置 SHA `a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f`。旧 environment authority/lock 中的 PICO/tree SHA 仅保留为历史环境证据，不是当前 GPU canary admission；当前 bounded admission 由 `tasks/receipts/FOUNDATIONSTEREO_RUNTIME_CLOSURE_V1.json` 约束。若未来替换 vendor，必须先新增可信 upstream revision 记录和 tree manifest，不能沿用本段 SHA。
