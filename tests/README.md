# 测试分类与检查入口

`tests/` 包含合同、单元、候选实现及维护测试。默认入口 `make check` 仅运行有限的合成 CPU 维护测试；测试通过的范围取决于实际选中的文件，不能据此宣布算法质量、训练资格或物理部署通过。

## 测试目录

| 位置 | 主要用途 |
| --- | --- |
| 根目录与 `unit/` | 项目合同及单元测试；具体范围查对应测试文件。 |
| `governance/`、`pipeline/`、`ops/` | 治理绑定、流水线和操作相关测试。 |
| `humanego_ict/`、`human_ego/` | HumanEgo ICT 候选接口与相关实现测试。 |
| `pick_place_1008/` | pick_place 候选实现测试及其历史 fixture。 |
| `maintenance/`、`helpers/` | 有限 CPU 检查与合成交接 fixture。 |
| `fixtures/` | 测试输入材料。 |
| `research/`、`storage_relocation/` | 研究与存储迁移相关测试。 |

2026-10-16 清理后，本地 `tests/` 实测有 886 个非缓存普通 `.py` 文件（不沿目录或文件符号链接计数）；其中剩余 7 个 `_tmp_*` 目录含 291 个历史 fixture／运行副本，其余 595 个为测试或辅助源码。目录名、文件数量和忽略规则均不证明可删除或测试已通过。

本次仅移出获独立证明的生成测试树和缓存，不合并仍有引用的 fixture／版本实现。已移出项、保留理由、恢复包及实际回归范围见[最终清理报告](../_run/maintenance/project_files_cleanup_20261016/attempt_a06/REPORT_ZH.md)与[恢复索引](../archive/project_files_cleanup_20261016/INDEX.json)。保留的 `_tmp_*` 不是算法入口，后续清理仍须新的有限任务和即时引用、进程、Git、许可证及归档校验。

## 有限 CPU 检查

从项目根运行：

```bash
make check
make check-server
```

| 命令 | 实际范围与前置条件 |
| --- | --- |
| `make check` 或 `make check-portable` | 合成输入的可移植 CPU 检查，选中两个维护测试文件；不需要服务器历史产物。 |
| `make check-server` | 选中四个服务器绑定测试文件；先校验 20261009 历史证据 SHA 及当前只读交接视图，干净克隆可能没有所需产物。 |
| `make handoff` | 只读检查当前收据、冻结源码及选择绑定；不执行 pytest。 |

上述测试由 [run_project_checks_v1.py](../scripts/maintenance/run_project_checks_v1.py)选择，默认限时 180 秒、单 CPU，不使用 GPU、模型或训练。它禁用 pytest 插件自动加载，每次新建 `_run/maintenance/project_checks_v1/` 下的 attempt，保存运行包、stdout、stderr 与 `RESULT.json`；测试临时目录及缓存也放在该 attempt 内。

2026-10-16 本轮两个可移植文件共 74 项通过，含新增 16 项 Inpainting 原版／R02／R03 合成回归：合法闭包保持只读，未授权包装、核心或 helper 改写、父绑定篡改、额外副作用及发布任务越权必须拒绝。实际项目的 19 核心、95 运行源码另由只读交接检查核验；这些检查不使用生产帧，也不替代画质、人工审阅或训练资格验收。

## 扩大测试范围

完整 `pytest` 收集范围大于上述维护检查。开展某条算法或候选路线的回归时，先核对当前 Task Packet 的测试范围、read_set、write_set 和运行预算，再执行指定测试。测试文件、fixture 或目录中的版本号不决定当前采用状态。

`pyproject.toml` 的 pytest 默认缓存位于 `_run/caches/pytest`。其他测试运行也必须显式把 TMPDIR、缓存及产物设置在项目内，核对真实路径；不要向 `/tmp` 或外部数据目录写入。算法源码以 `src/chaoyang/` 为开发树，`baseline_v1/` 保持冻结参考角色。
