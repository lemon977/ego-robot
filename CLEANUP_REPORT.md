# Repository Cleanup Report

日期：2026-08-26

## 结果

项目目录由约 172 GB 清理至约 2.8 GiB，回收约 169 GB。删除为不可恢复操作；冻结资产在删除前后均做了引用与 SHA 校验。

保留：

- exact78 的 78 份 R2 sidecar 与 62/8/8 split。
- 78 sessions 的紧凑 HaWoR/Object6D/adapter 数值证据和帧级 JSON。
- pretrained、旧 frozen Kai22、R2 control 三个 checkpoint 及 manifest。
- HumanEgo 主干源码、测试和当前 R2 配置。
- Tianji/KaiHand 本地唯一资产树。

删除：

- `_run`、历史 worker/watcher/guardian、logs、venv、cache、`__pycache__`。
- 非 exact78 的 79 sessions、旧 stage 01/03/04/05/06/07、失败尝试和重复视频/图片。
- 64,636 个已断裂的 RGB symlink，以及旧 model masks、overlay、smoke/ready 标记。
- 旧 EgoRobotVisual 与内嵌 ProPainter、修改过的 HaWoR 副本、HumanEgo 内嵌 Git 历史。
- v1/v2/v3/Wuji/旧 candidate producer、断裂的 RobotRGB/VS78 入口和 session 特调启动器。
- 旧采集、预处理、下载、硬件控制入口和 Wuji 资产；HumanEgo 只保留训练/评测所需最小核心。

## 校验结果

```text
FROZEN_RETENTION_VALIDATION_OK
sessions=78
frames=32318
sidecars_sha_ok=78
checkpoints_sha_ok=3
nested_git_repositories=0
unintended_symlinks=0
unit_tests=9_passed
```

唯一 symlink 为 `HumanEgo/vendor/kaihand -> ../../assets/robot/kaihand/packages`，用于避免复制第二份机器人资产。

## 边界

本轮没有恢复 RAW、没有生成新 MASK/CLEAN/RobotRGB、没有修改任何 R2 sidecar、没有启动训练或评测。全局磁盘同时可能有其他作业变化，因此这里记录的是项目目录体积，不声称整个文件系统的 free space 精确增加值。
