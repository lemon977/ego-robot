# 0911 / 0914 / 0915 数据清洗验收基线

状态：CURRENT / COMPLETE
适用基线：clean-baseline-v1
机器收据：[`tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json`](../../tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json)

本页只说明三批数据的清洗终态、复用边界和脚本职责。它不创建活动任务，也不授权覆盖已发布数据。

## 验收结果

| 批次 | 原始会话 | 清洗通过 | 拒绝 | 失败 | 终态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0911 | 241 | 139 | 102 | 0 | COMMITTED |
| 0914 | 220 | 202 | 18 | 0 | COMMITTED |
| 0915 | 220 | 220 | 0 | 0 | COMMITTED |
| 合计 | 681 | 561 | 120 | 0 | 681/681 terminal |

0911 的 102 个拒绝均为触觉质量失败。0914 有 17 个触觉质量失败，另有 `playing_cards/088` 因 `VISUAL_SENSOR_PROJECTION_MISMATCH` 被拒绝。0915 未采集 MANUS，按 `ABSENT_NOT_CAPTURED` 明确记录，不把真实缺失模态伪造成错误；PICO26 数据保留。

## 路径与证据

原始数据只读：

- `/mnt/data/egodata/datasets/ego/chips_cards_handle_0911`
- `/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0914`
- `/mnt/data/egodata/datasets/ego/chips_cards_hands_0915`

已发布数据：

- `/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0911`
- `/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0914`
- `/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`

0915 当前已发布 processed 根使用单下划线路径。旧合同和历史收据中的
`chips_cards_hands__0915` 仅是逻辑发布标识，不是可直接读取的目录；不得据此拼接
或回源。0915 发布根中的 `SOURCE_PATH_REPAIR_20260917.json` 只修复元数据来源路径；未改原始数据，也未改已发布 payload。

0915 processed 单目像素域按会话元数据保持 `passthrough_scaled_source_domain`，
`video_rectification.rectified=false`。用户已进一步确认所有 VST 编码视频像素本来没有
畸变；该字段表示清洗过程没有新增 rectification，不表示下游应读取工厂 `equiDis62`
再次去畸变。所有视频消费者只允许物理眼裁切加 resize，见
[`VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。

每个发布根以以下文件为事实来源：

- `DATASET_RESULT.json`：数据集终态与计数；
- `STATE.json`：可恢复状态；
- `PREFLIGHT_AUDIT.json`：逐会话接纳或拒绝原因；
- `QUALITY_POLICY.json`：质量策略；
- `cleaned/` 与 `rejected/`：通过数据和拒绝收据。

0911 发布根中的 `QUEUE_STATE.json` 已为 `COMMITTED`。仓库内机器收据固定上述终态文件的 bytes/SHA-256，后续 AI 不得只根据目录数量推断完成状态。

## 当前算法和脚本职责

| CLI operation / 文件名 | 唯一职责 | 是否直接发布数据 |
| --- | --- | --- |
| `tactile_quality_gate_v1` | 对单会话原始触觉流做 fail-closed 完整性和活动度审计 | 否 |
| `convert_handle_egodex_v3` | 保留触觉并按模态合同转换一个会话；0915 不伪造 MANUS | 仅写目标 staging，验证后原子发布 |
| `batch_clean_handle_content_v3` | 串行审计/转换一个数据集，持有目标锁并写终态收据 | 是 |
| `run_handle_cleaning_v3_queue` | 按 0911 → 0914 → 0915 顺序编排三个数据集 | 是；当前不得重启 |

算法设计、阈值和输出格式见 [`HANDLE_DATA_CLEANING_V3.md`](../guides/data/HANDLE_DATA_CLEANING_V3.md)。统一调用格式为：

```bash
python -m pip install -e '.[data-cleaning]'  # 新环境只需执行一次
PYTHONPATH=src python -m chaoyang.cli run <operation> --help
```

`data-cleaning` extra 固定了 `numpy`、`h5py` 和无界面 OpenCV 依赖；不得假设任意 Conda 环境已预装这些库。本次服务器上的只读 CLI smoke 也已在具有这些依赖的 `egoforce` 环境通过。

生产运行必须继续使用 CPU、`nice -n 15` 与 `ionice -c 3`；本清洗不需要 GPU。

## 复用和优化边界

- 三批任务已完成。除非用户明确要求重跑且先产生新的任务包，不得重启队列、覆盖发布根或删除拒绝收据。
- 缺少未采集模态不是自动拒绝条件；必须由采集合同声明为 `ABSENT_NOT_CAPTURED`。应采集却缺失、损坏、冻结、饱和或时序异常仍 fail closed。
- 触觉门证明原始整数信号活动度和完整性，不证明标定力、接触真值或物理精度。
- 单侧全零在允许单手任务中是 warning；双侧无有效活动才拒绝。
- 后续算法优化先新增固定 fixture 和 bounded canary，再修改阈值；不得为了提高通过率降低门槛或回填伪数据。
- 项目目录重构不得移动、删除或改写 `/mnt/data/egodata`；验证只读收据和哈希即可。

## 只读复核

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path('/mnt/data/egodata/datasets/ego/processed')
for name in ('chips_cards_handle_0911',
             'chips_cards_handle_highview_0914',
             'chips_cards_hands_0915'):
    value = json.loads((base / name / 'DATASET_RESULT.json').read_text())
    print(name, {key: value.get(key) for key in
          ('state', 'session_count', 'completed', 'cleaned', 'rejected', 'failed')})
PY
```
