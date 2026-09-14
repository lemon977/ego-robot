# chips_cards_handle RGB30 v1 数据处理与脚本使用手册

> 2026-09-11 准入勘误：0909/0910 的 rgb30_v1 虽已完整封账，但其高拒绝率主要来自
> 本项目对 raw 的二次 QA 使用了“任一流短缺口修复比例≤1%”硬门，不能解释成采集数据
> 大面积损坏。采集端 `dataset.hdf5` 已按 40 ms gate 删除不完整行并保证≥95%完整覆盖；
> 104/104 与 228/228 均声明导出行完整。按用户最终决策，旧结果已退出正式入口，
> acquisition-aligned v2 从零重建后仍发布到原有两个 processed 路径，不新增并列数据根。详细复核见
> `tasks/control/runs/20260911_handle_acquisition_aligned_v2/ADMISSION_REAUDIT_ZH.md`。

## 1. 文档目的与适用范围

本文是 `chips_cards_handle_0909 / rgb30_v1` 的用户操作手册，总结从新采集
EGO-DEX 原始会话到 tracker-style 训练数据的完整处理方式。

适用代码根：

```text
/mnt/workspace/code/chaoyang
```

当前数据源已于 2026-09-11 按任务整理：

```text
/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/
├── potato_chips/001..102
├── playing_cards/103..104
└── SOURCE_LAYOUT_V2_RECEIPT.json
```

旧 v1 曾按历史策略排除 `potato_chips/001`；当前 acquisition-aligned v2 不再继承该
策略排除，而是让它通过 acquisition HDF5 与转换合同独立判定。源目录迁移只改变位置，
104 个会话、1,350 个文件和 4,225,492,941 bytes 均已逐文件 SHA256 前后复核一致。

当前正式输出：

```text
/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909
```

以下是已经被准入复核取代的历史 v1 时间链，仅用于理解旧结果，不能作为当前批次入口：

```text
时钟域识别/映射一次
→ 映射后原始样本域 QA 与清洗
→ 从原始样本直接重采样到 RGB 30Hz 一次
→ tracker converter 只消费 canonical clean bundle
```

严禁使用旧 `aligned.jsonl`、旧 `dataset.hdf5` 或其他 RGB30 产物作为新的
时间轴数据源。它们只可用于读取静态 calibration/schema 和交叉核验。

## 2. 两个用户入口

用户只需直接使用两个脚本：

```text
tools/qa_clean_handle_egodex_session.py
tools/batch_convert_handle_egodex_to_tracker.py
```

### 2.1 单会话 QA/Clean 脚本

`qa_clean_handle_egodex_session.py` 负责：

1. 扫描原始会话并生成文件清单、大小、mtime 和 SHA256。
2. 将原始会话完整复制到同盘 staging 的 `source_archive/`。
3. 以“相对路径 + 字节数 + SHA256”重新验证归档副本。mtime 只是 provenance，
   不参与可迁移内容身份判定。
4. 验证 VST、PICO、MANUS 和 Tactile 时钟域，映射到 VST reference。
5. 在原始样本域做缺失、时间戳、跳变、四元数、轨迹和触觉 QA/修复。
6. 从原始样本一次性构建严格 30Hz 时间轴。
7. 生成 canonical clean bundle、原始/30Hz flags、QA 结果和全片复核视频。

该脚本不生成最终 tracker-style 目录，也不写 `PUBLISHED.json`。正式发布应使用
批处理脚本。

### 2.2 批处理与发布脚本

`batch_convert_handle_egodex_to_tracker.py` 负责：

1. 枚举会话和任务映射。
2. 调用单会话 QA/Clean 脚本的实现。
3. 仅对 QA PASS/ELIGIBLE 会话调用 tracker converter。
4. 验证帧数、时间轴、bundle UUID、手腕投影、相对路径和软链接。
5. 将会话分流到 `cleaned/` 或 `rejected/`。
6. 最后写单会话 commit marker，全批结束后再写数据集级结果。

## 3. 原始会话输入合同

一条正常会话至少应包含：

```text
<session>/
├── manifest.json
├── dataset.hdf5
├── aligned.jsonl
└── raw/
    ├── camera_params.json
    ├── camera_params.meta.json
    ├── pico.jsonl
    ├── manus.jsonl
    ├── manus.meta.json
    ├── tactile.jsonl
    ├── tactile.meta.json
    ├── vst.h264
    ├── vst.ts.jsonl
    └── vst.qpc.ts.jsonl
```

发布必需流：

```text
vst_rgb
pico_head
pico_controller_left
pico_controller_right
manus_left
manus_right
tactile_left
tactile_right
```

可选流：

```text
pico_optical_hand_left
pico_optical_hand_right
```

PICO optical Hand `isActive=0` 时必须当作无效，不得生成、填充或冒充
PICO21。MANUS25 必须原样保存；MANUS25 到 MANO21 必须由独立、版本化的
provider/mapping 完成，不允许 converter 静默重排或丢节点。

## 4. 时间链和坐标链

### 4.1 Canonical 时间轴

Canonical 时间单位是 `int64 ns`，零点是第一张合法 VST presentation-order RGB 帧：

```text
canonical_time_origin = FIRST_VALID_VST_PRESENTATION_FRAME
grid_anchor_ns = 0
timeline_timestamp_ns[n] = round(n * 1_000_000_000 / 30)
```

QPC 原始 tick、frequency、clock ID、epoch、unit 和 timestamp semantics 必须保留。只有 clock ID、
epoch 和 unit 全部一致时才允许 identity mapping。其他情况使用成对 anchor 估计
scale/offset，绝对大计数器在转为 float64 前先中心化。

Wall/QPC anchor 只用于判断时钟单位、频率和映射，不用来推断 transport latency，
也不宣称是传感器硬件采样时刻。

后验运动相关性只写入 `LATENCY_DIAGNOSTICS.json`：

```json
{
  "applied_to_timestamps": false
}
```

它不得改写时间戳或触发第二次重采样。

### 4.2 PICO、Wrist 和 c2w

0909 原始 `pico.jsonl` 的 Head/Controller 位姿使用 OpenXR 轴。在运动 QA、插值、
Wrist 派生和 HDF5 写入前，显式转到 REP-103：

```text
C = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
T_rep103 = C @ T_openxr @ inverse(C)
```

Wrist 是派生量：

```text
wrist_pose = controller_pose @ controller_to_wrist_calibration
```

Wrist validity/imputation 继承 Controller 和 calibration 状态。下游 c2w 仍使用已确认的
REP-103 → OpenXR → camera 链。

图像使用已人工确认的 sourceIndex0 无径向 warp 直通缩放，不再对原始 VST 视频
做不正确的去畸变/再畸变。

### 4.3 2048×1536 与 1280×960 的关系

这两个尺寸都正确，但属于不同数据层：

| 层级 | 分辨率 | 保存位置/用途 |
| --- | ---: | --- |
| 原始SBS | 4096×1536 | `source_archive/raw/vst.h264` 与发布后的 `source_stereo/` |
| 原始单眼半幅 | 2048×1536 | 从SBS按`sourceIndex`切出；不得丢失，是未来Stereo几何入口 |
| tracker-compatible mono | 1280×960 | `CameraRecord_*.mp4` 与 `preprocess/all_data/*/rgb.png`，供视觉主流程消费 |

历史0901正式目录也采用相同的文件尺寸分层：主`CameraRecord_*.mp4`是1280×960，
`source_stereo/*_stereo.mp4`是4096×1536；“0901是2048×1536”通常指原始SBS拆出的
单眼，而不是正式主流程MP4。

handle0909/0910当前选择`legacy_tracker_left`：消费SBS的`sourceIndex0`以保持旧tracker
槽位兼容。0910相机文件按物理外参重新命名后，物理左眼其实是`sourceIndex1`，因此文档、
manifest和下游代码不得把`legacy_tracker_left`再表述成“物理左相机”。当前1280×960是
2048×1536按0.625等比缩放的视觉兼容输出；这不等同于Exact78 Stereo链的90°虚拟针孔
remap，也不授权从该mono直接推导公制Depth。未来handle Stereo必须读取保留的原始SBS和
同会话相机标定，单独完成双目校正、视差尺度和registration审核。

## 5. QA 与清洗要点

QA policy 版本：

```text
chips_cards_handle_0909_rgb30_v1
```

主要门限：

| 项目 | 门限 |
| --- | --- |
| 原始时间缺口 | `delta > 1.5 × median period` |
| 四元数合法 norm | `[0.98, 1.02]` |
| Head 平移/旋转硬限 | `3 m/s`, `720 deg/s` |
| Controller/Wrist 平移/旋转硬限 | `8 m/s`, `1440 deg/s` |
| MANUS node 平移硬限 | `4 m/s` |
| 时钟 anchor 数 | 每段至少 100 |
| anchor 覆盖比 | 至少 80% |
| clock residual P95/max | `5 ms / 15 ms` |
| 时钟漂移 | 绝对值不超过 500 ppm |
| RGB 正常/硬 nearest | `16.67 ms / 30 ms` |
| Pose 正常 bracket | 不超过 30 ms |
| 短缺口修复 | 最多连续 2 个 RGB30 点 |
| Pose repair bracket 硬限 | 105 ms |
| Tactile causal/offline 年龄 | `25 ms / 25 ms` |
| 单流 imputed ratio | 不超过 1% |
| 公共支持时长比 | 不低于 99% |

正常插值和短缺口修复分开记录。禁止跨越 tracking loss、active 变化、glove ID、
标定状态、设备重连、clock reset 或 segment 边界插值。

四元数只有在 norm 合法时才允许归一化。SLERP 前强制四元数同半球，防止绕长弧。

触觉跳变规则：

```text
threshold = max(20 ADC, median_abs_diff + 10 × MAD)
```

只有两侧邻居闭合的 A-B-A 瞬态跳变才自动修复，并在 `RAW_SAMPLE_FLAGS.jsonl`
记录原始 sample/channel。int16 rail 只标记为代理饱和，不宣称是经物理标定的
真饱和。没有足够触觉活动时，坏点检查状态为 `UNTESTABLE`。

### 5.1 RGB闪烁与固定30 Hz cadence

曝光/颜色闪烁和时间重采样顿挫必须分开审核。2026-09-11对当时已发布的18条0910主
`CameraRecord`做了6044帧完整解码：没有≥3灰度级的单帧全局亮度脉冲，也没有肉眼可见
整帧闪白/闪黑；但固定30 Hz nearest映射共出现125/6026（2.07%）相邻对复用同一
`rgb_source_index`。052、101、111分别为6.34%、10.89%、15.60%，应标记
`TEMPORAL_CADENCE_REVIEW`。这属于采集时间间隔抖动造成的停顿/跳步，不是MP4曝光闪烁，
也不得靠静默生成光流帧掩盖。快照报告见：

```text
tasks/control/runs/20260911_chips_cards_handle_0910_rgb30_v1/
  flicker_audit_snapshot_1805/REPORT_ZH.md
```

该数字只覆盖当时已经发布的18条；整批`COMMITTED`后必须对最终主MP4清单重跑同一审计。

## 6. 单会话命令

所有命令先进入项目根：

```bash
cd /mnt/workspace/code/chaoyang
```

### 6.1 Dry-run

只检查输入、文件数、大小和必需文件，不复制、不生成数据：

```bash
python3 tools/qa_clean_handle_egodex_session.py \
  --source /mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips/010 \
  --output /mnt/data/egodata/datasets/ego/processed_canary/session_010 \
  --session-id 010 \
  --dry-run
```

Dry-run 中 `--output` 只用于显示目标，不会创建。

### 6.2 生成 QA/Clean canary

```bash
python3 tools/qa_clean_handle_egodex_session.py \
  --source /mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips/010 \
  --output /mnt/data/egodata/datasets/ego/processed_canary/session_010 \
  --session-id 010 \
  --render-review
```

`--output` 必须不存在。脚本不会覆盖旧目录。

### 6.3 策略排除会话

```bash
python3 tools/qa_clean_handle_egodex_session.py \
  --source /path/to/session \
  --output /path/to/output \
  --session-id 001 \
  --policy-excluded \
  --render-review
```

这仍会做完整 QA，但 `training_eligibility` 为 `POLICY_EXCLUDED`。

## 7. 历史 v1 批处理命令（只用于解释旧结果）

本节命令属于已被 acquisition-aligned v2 取代的 raw 二次 QA 路线，不得用于继续生成
当前正式 0909/0910。当前执行入口是：

```text
tools/batch_convert_handle_acquisition_aligned_v2.py
tools/guard_handle_acquisition_batches_v2.py
```

### 7.1 只查看映射，不写数据

```bash
python3 tools/batch_convert_handle_egodex_to_tracker.py \
  --task-source potato_chips=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips \
  --task-source playing_cards=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/playing_cards \
  --policy-exclude potato_chips=001 \
  --date-tag 0909 \
  --dry-run
```

显式任务源可用 `--task-sessions TASK=RANGE` 选择子集。范围支持逗号和闭区间，例如：

```text
001-104
010,038,104,001
002-020,025,030
```

### 7.2 四会话 canary

```bash
python3 tools/batch_convert_handle_egodex_to_tracker.py \
  --task-source potato_chips=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips \
  --task-source playing_cards=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/playing_cards \
  --task-sessions potato_chips=010,038,001 \
  --task-sessions playing_cards=104 \
  --policy-exclude potato_chips=001 \
  --date-tag 0909 \
  --dataset-id chips_cards_handle_0909_canary_layout_v2 \
  --target-root /mnt/data/egodata/datasets/ego/processed_canary/chips_cards_handle_0909 \
  --workers 1 \
  --render-review \
  --canary
```

Canary 顺序和目的：

1. `010`：正常 cleaned 发布、坐标/视觉/时间基准。
2. `038`：长缺口、segment 和 rejected 分流。
3. `104`：缺 VST QPC 时只允许 review-only。
4. `001`：数据 QA 和策略排除分离。

### 7.3 已分任务源目录的全批

下面命令只用于复现旧 v1 的任务分层试验，不能作为当前正式重跑入口：

```bash
python3 tools/batch_convert_handle_egodex_to_tracker.py \
  --task-source potato_chips=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips \
  --task-source playing_cards=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/playing_cards \
  --policy-exclude potato_chips=001 \
  --date-tag 0909 \
  --dataset-id chips_cards_handle_0909_task_layout_v2 \
  --dataset-version rgb30_v1_source_layout_v2 \
  --target-root /mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909_task_layout_v2 \
  --workers 2 \
  --render-review
```

### 7.4 中断后续跑

只有在数据集根还没有最终 `COMMITTED` 时使用：

```bash
python3 tools/batch_convert_handle_egodex_to_tracker.py \
  --task-source potato_chips=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/potato_chips \
  --task-source playing_cards=/mnt/data/egodata/datasets/ego/chips_cards_handle_0909/playing_cards \
  --policy-exclude potato_chips=001 \
  --date-tag 0909 \
  --dataset-id chips_cards_handle_0909_task_layout_v2 \
  --dataset-version rgb30_v1_source_layout_v2 \
  --target-root /mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909_task_layout_v2 \
  --workers 2 \
  --render-review \
  --resume
```

Resume 不是“看见目录就跳过”。脚本会验证：

- source freeze SHA；
- QA policy SHA；
- mapping SHA；
- 代码 SHA 与依赖版本；
- camera/MANUS/tactile calibration SHA；
- `CONTENT_MANIFEST.json`、`RESULT.json` 和 commit marker SHA；
- `source_archive` 内容身份。

任一字段不一致都不会静默跳过。已经 `DATASET_RESULT.json.state=COMMITTED` 的版本根
是只读的，不得在原根上 resume。policy、mapping、代码或 calibration 改变时应创建
新 dataset version/新目标根。

## 8. 0909 任务映射与未来分目录限制

旧 legacy preset 的 0909 mapping 是：

```text
001       旧 v1 策略排除；当前 v2 已取消该规则
002–102   potato_chips
103–104   playing_cards
```

对应目标名：

```text
potato_chips/get_potato_chips_0909_<session_id>
playing_cards/play_cards_0909_<session_id>
```

### 重要限制

`qa_clean_handle_egodex_session.py` 已可处理任意单会话路径。

`batch_convert_handle_egodex_to_tracker.py` 保留上述 0909 legacy preset，以便解释历史结果；
新数据必须用可重复的 `--task-source TASK=PATH` 显式指定任务。显式任务模式完全不根据
编号猜任务，因此 chips 和 cards 都可以从 `001` 重新编号，也不会把新 `001` 当成历史
策略排除项。

### 8.1 0910 已分任务源数据

当前磁盘实测：

```text
/mnt/data/egodata/datasets/ego/chips_cards_handle_0910/
├── chips_119_0910/001..119       # 119 条
└── cards_109_0910/001..109       # 109 条
```

两边的 `manifest.json`、`dataset.hdf5`、`aligned.jsonl` 和 `raw/` 均无缺失。
正式 dry-run：

```bash
python3 tools/batch_convert_handle_egodex_to_tracker.py \
  --task-source potato_chips=/mnt/data/egodata/datasets/ego/chips_cards_handle_0910/chips_119_0910 \
  --task-source playing_cards=/mnt/data/egodata/datasets/ego/chips_cards_handle_0910/cards_109_0910 \
  --date-tag 0910 \
  --dataset-id chips_cards_handle_0910 \
  --target-root /mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0910 \
  --dry-run
```

当前 dry-run 结果应为 228 条：Chips 119 + Poker 109。

## 9. 正式目录结构

```text
/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909/
├── DATASET_RESULT.json
├── BATCH_STATUS.csv
├── cleaned/
│   ├── potato_chips/
│   │   └── get_potato_chips_0909_<id>/
│   └── playing_cards/
│       └── play_cards_0909_<id>/
└── rejected/
    ├── potato_chips/<source_session_id>/
    └── playing_cards/<source_session_id>/
```

新显式任务模式下，`cleaned/` 和 `rejected/` 都按任务分开，防止两个任务从 `001`
重新编号时发生碰撞。历史 0909 已提交 v1 经用户确认后也已执行一次可恢复结构迁移；
会话目录内容不变，只更新数据集级路径索引，并保留迁移前索引。

### 9.1 Cleaned 会话

```text
<cleaned-session>/
├── source_archive/
├── clean_bundle/
│   ├── BUNDLE_MANIFEST.json
│   ├── QA_POLICY.json
│   ├── CLEAN_TIMELINE.npz
│   └── clean_dataset.hdf5
├── qa/
│   ├── SOURCE_FREEZE.json
│   ├── CLOCK_ALIGNMENT.json
│   ├── RAW_STREAM_QA.json
│   ├── RAW_SAMPLE_FLAGS.jsonl
│   ├── FRAME_FLAGS.jsonl
│   ├── LATENCY_DIAGNOSTICS.json
│   ├── QA_POLICY.json
│   └── qa_review_full.mp4
├── preprocess/all_data/<frame>/
│   ├── rgb.png
│   ├── rgb_WoArm_WArmObjKpts.png
│   └── training_data.json
├── source_stereo/
├── CameraRecord_<session>.mp4
├── camera_params.json
├── clip_manifest.json
├── controller_poses_<session>.jsonl
├── slam_trajectory_<session>.jsonl
├── trackingData_<session>.txt
├── CONVERSION_RESULT.json
├── RESULT.json
├── CONTENT_MANIFEST.json
└── PUBLISHED.json
```

### 9.2 Rejected 会话

```text
rejected/<task>/<id>/
├── source_archive/
├── qa/
├── clean_bundle/       # 若可建权威时间轴则保留
├── RESULT.json
├── CONTENT_MANIFEST.json
└── REJECTED.json
```

104 缺 VST QPC sidecar，只包含非权威 `CLOCK_ALIGNMENT.json(status=UNAVAILABLE)`、
review timeline/frame map 和带 `REVIEW_ONLY/TIMEBASE_UNTRUSTED/NOT_FOR_TRAINING` 水印的
复核视频，不生成正式 clean bundle。

## 10. Clean bundle 和触觉数据

`clean_dataset.hdf5` 是 canonical payload。`CLEAN_TIMELINE.npz` 只保存轻量时间轴、index 和
flags cache。两者共享 bundle UUID 和 timeline SHA。

`BUNDLE_MANIFEST.json` 至少锁定：

```json
{
  "schema_version": "chips_cards_handle_rgb30_v1",
  "authoritative": true,
  "source_level": "RAW_SOURCE_ARCHIVE",
  "clock_mapping_count": 1,
  "resample_count": 1,
  "timeline_policy": "VST_QPC_RGB30",
  "timeline_rate_hz": 30
}
```

Converter 对任一字段不匹配都 fail closed。

每只手的触觉合同：

```text
wire_values_369              int16
wire_active_mask_369         static bool
wire_valid_mask_369          per-sample bool
finger_grid_5x4x8            float32，padding=NaN
active_mask_5x4x8            static bool
valid_mask_5x4x8             per-sample bool
offline_source_index/valid/offset_ms
causal_source_index/valid/age_ms
```

每手物理有效 taxel 为 144 个，`5×4×8` 容器中另有 16 个 padding。HDF5/NPZ 中 padding 是
NaN；JSON 中写为 `null`，loader 必须转回 float NaN 并结合 active/valid mask。

每帧 `training_data.json` 直接嵌入 causal source 的 369 值和 finger grid。Offline 仅保存
source index/valid/offset，从同会话 HDF5 获取 payload，避免复制第二份 369 数组。

## 11. 状态与 commit marker

会话结果分开保存：

```text
pipeline_status
qa_status
training_eligibility
publish_status
reason_codes
archive_status
review_status
```

Cleaned 会话的最后文件是 `PUBLISHED.json`；Rejected 会话的最后文件是
`REJECTED.json`。目录存在但 marker 缺失、receipt 不匹配或 SHA 错误，都是
`INCOMPLETE`，不可消费。

数据集级只以：

```text
DATASET_RESULT.json.state == COMMITTED
```

作为 001–104 整批完成的权威证据。

## 12. 快速验证命令

查看整批状态：

```bash
python3 - <<'PY'
import json
p = "/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909/DATASET_RESULT.json"
d = json.load(open(p))
print({k: d.get(k) for k in (
    "state", "session_count", "cleaned_count", "rejected_count",
    "failed_uncommitted_sessions"
)})
PY
```

查看所有 cleaned 会话：

```bash
find /mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909/cleaned \
  -name PUBLISHED.json -print
```

查看淘汰原因分布：

```bash
python3 - <<'PY'
import collections, csv
p = "/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909/BATCH_STATUS.csv"
rows = list(csv.DictReader(open(p)))
counts = collections.Counter()
for row in rows:
    counts.update(filter(None, row["reason_codes"].split(";")))
print(counts)
PY
```

检查一条复核视频：

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames,duration \
  -of json \
  /path/to/session/qa/qa_review_full.mp4
```

检查软链接：

```bash
find /path/to/session -type l -print
```

结果必须为空。

## 13. 当前 0909 正式结果

2026-09-11 22:34:21，acquisition-aligned v2 已在原固定路径完成并通过实盘复核：

```text
state: COMMITTED
session_count: 104
cleaned_count: 103
rejected_count: 1
failed_uncommitted_sessions: []
```

任务分布：

- `potato_chips`：102 个正式会话目录、102 份 `CONVERSION_RESULT.json`；
- `playing_cards`：103 为正式 cleaned；104 为唯一 rejected；
- 两个任务均无遗留 staging；
- 唯一 rejected 原因是源会话 104 同时缺少 `raw/vst.ts.jsonl` 与
  `raw/vst.qpc.ts.jsonl`，不得伪造时间 sidecar。

权威结果：

```text
/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0909/DATASET_RESULT.json
SHA256 9897fcec346cb13c14d2ac03e71f0a6659b77144783c6ba6bfc214c7e2d3b0a5
```

旧 v1 的 `13 cleaned / 91 rejected`、旧 reason-code 统计和旧 SHA 只属于被取代的历史
执行，不再是正式数据入口或当前质量结论。

## 14. processed 根整理后的结构

```text
/mnt/data/egodata/datasets/ego/processed/
├── chips_cards_handle_0909/
└── chips_cards_handle_0910/
```

两个正式根都按任务分层：

```text
<dataset-root>/
├── cleaned/potato_chips/<task_session>/
├── cleaned/playing_cards/<task_session>/
├── rejected/potato_chips/<source_id>/
├── rejected/playing_cards/<source_id>/
├── STATE.json
└── DATASET_RESULT.json
```

当前两批均无 staging。旧错误结果已按用户授权退出并清理；原因、旧策略与修正证据保留在
`tasks/control/runs/20260911_handle_acquisition_aligned_v2/ADMISSION_REAUDIT_ZH.md`。

## 15. 常见问题

### 为什么旧结果 rejected 很多，而当前结果几乎没有？

旧 v1 在 acquisition 已完成 40 ms 同步清理后，又对 raw 做一次插值并用“任一流修复
比例≤1%”拒绝整会话；这属于重复且过严的二次准入。当前 v2 直接验证 acquisition
`dataset.hdf5` 的完整行、覆盖率、finite 数组和视频索引；cadence 重复作为报告指标，
不再冒充传感器缺失。因此 0909 当前只拒绝真正缺时间 sidecar 的 104，0910 零拒绝。

### 可以手工把 rejected 移到 cleaned 吗？

不可以。这会破坏 marker、manifest、SHA 和数据集级统计。准入合同变化必须先重跑
canary，再从源数据 no-clobber 重建并重新生成数据集级终态；本次用户明确要求继续使用
原固定路径，因此旧结果已退出后由 acquisition-aligned v2 原子重建，而不是手工搬目录。

### 为什么目录存在但不能用？

只有有效 `PUBLISHED.json` 的 cleaned 会话可消费。Staging、无 marker 目录、review-only 和
rejected 都不得作为训练数据。

### 处理中断时会不会丢原始数据？

不会。源根始终只读。只有 source archive 完整复制并逐文件 SHA 验证后，后续才开始。
复制失败时记录 `SOURCE_ARCHIVE_COPY_FAILED`，会话保持 uncommitted，数据集不会冒充
`COMMITTED`。

## 16. 最终使用原则

1. 训练只读 `cleaned/<task>/<session>/`。
2. 处理结果只信 commit marker 和 SHA，不信目录是否“看起来完整”。
3. 同一会话时钟映射一次、RGB30 重采样一次。
4. 不得从无效 optical hand 伪造 PICO21。
5. 不得静默将 MANUS25 冒充 MANO21。
6. 不得把 internal consistency 指标写成毫米级外部真值精度。
7. 任何 policy、mapping、代码、坐标或 calibration 变更都必须形成新的合同身份和完整
   终态收据。默认应使用新数据根；只有用户明确要求复用固定路径、旧结果已安全退出且
   从源数据原子重建时，才允许沿用物理目录名。

## 17. 0910 正式终态

2026-09-11 23:57:31，0910 acquisition-aligned v2 已完成：

```text
state: COMMITTED
session_count: 228
cleaned_count: 228
rejected_count: 0
potato_chips: 119/119
playing_cards: 109/109
```

正式输出与权威 SHA：

```text
/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0910/DATASET_RESULT.json
SHA256 a7d4d1887d46931546ac50954178d0d3841e847e8206b0d11c95199f4ed062ce
```

228 份会话收据全部为 `PASS_FORMAT_COMPATIBLE`，均包含 acquisition preflight 与
admission policy；两个任务均无 rejected、无 staging。后半程在完整会话提交边界从2个
worker安全切换到4个worker；严格 adoption/resume 与 flock 防双写证据位于：

```text
/mnt/workspace/code/chaoyang/tasks/control/runs/20260911_handle_acquisition_aligned_v2/
WORKER_EXPANSION_0910_2_TO_4_RECEIPT.json
SHA256 65116d8202c3e125f89b4b73f953743f8dc9011519aafcadeb7e9f4bbd23131a
```
