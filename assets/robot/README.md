# Canonical local robot assets

本目录是 Tianji、KaiHand 与相机法兰的唯一项目内资产根。`HumanEgo/vendor/kaihand` 仅为指向这里的 symlink，禁止复制第二份资产树。

历史资产清单记录：

- Tianji `marvin_CCS_m6.urdf` SHA256：`3c3bdfa312f41663bcf1522953a39e31618a7e92c991caa51f36535a88513b1f`
- Tianji `package.xml` SHA256：`9b1ca307429bf3d6c7535f84d58fd23be73c30543395fa19005187618554002f`
- Tianji m6 mesh tree：16 files / 16,020,944 bytes / tree SHA256 `ceab272bc195b57aad2a8c7b52c4e0b6a5a5a4f2a116c34254596892bcd63c24`
- Tianji base mesh tree：4 files / 16,477,286 bytes / tree SHA256 `23ac59358ef79711eead93f6838afcb1271f74d373dad7053914f56a6450e860`
- KaiHand left tree：31 files / 13,860,857 bytes / tree SHA256 `d73cc8cb58994ab8711391a53d5ea06cfe428090eb26137210114e50da8efb5d`
- KaiHand right tree：31 files / 13,881,910 bytes / tree SHA256 `f1ce730c26f0897d055f1dc6d28b991dcf49e53ed8860f9d735ca4d51a2d9ce7`
- 法兰 STL：337,084 bytes / SHA256 `d47925c2d6841675782ee0666c8c803d5fd6577571230071ef40078d954a3965`

URDF/package/mesh/CAD 均为本地运行依赖，默认不进入 Git；再分发权确认前不得强制添加。Git 仅保留本说明和 `kaihand/asset_manifest.json` 的 hash provenance。
