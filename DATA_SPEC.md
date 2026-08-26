# 数据与 I/O 规范

状态：`STOPPED / CLEAN_BASELINE_VALIDATED`

日期：2026-08-26

## 1. 根目录

- 项目根：`/mnt/workspace/code/chaoyang`
- 权威外部 RAW：`/mnt/data/egodata`（只读；当前不可读，正式出图硬阻塞）
- 临时运行根：`/mnt/workspace/code/chaoyang/_run`
- 正式产物根：`/mnt/workspace/code/chaoyang/processed`

除只读 RAW 外，后续所有代码、配置、context、日志、staging、QA、预览和正式输出必须位于项目根内。

## 2. 当前保留的冻结数据

### exact78 R2

- split/sidecar manifest：`HumanEgo/data_manifests/retarget_ab/kai22_r2.json`
- manifest SHA256：`4378c6011e5af39ed0e6bc0fd3e8a054ce1d1ceb5279cb513b6be17375984e6c`
- sidecar root：`HumanEgo/data_manifests/robot_sidecars_v4_candidate/r2_robust24_confidence_temporal/kai22`
- sessions：78（train 62 / validation 8 / test 8）
- source frames：32,318
- 校验：78/78 `sidecar.npz` 与 manifest 中 `sidecar_sha256` 一致。

历史 source split `HumanEgo/data_manifests/shared_split.json` 的 SHA256 为 `3716ce97961ce42cb8d15b44d84f73dfe2ab2901e707580c6ecc40d990056978`，为旧 checkpoint 谱系保留。其旧 `sidecar_root` 字段指向已删除 v1，不得作为新执行默认值。

### checkpoints

- pretrained：`HumanEgo/artifacts/pretrained_humanego/latest.pt`

  SHA256 `663f21b4350333fb852d0f19d042613d1e7bc5ead2577f9862d113ccea04b183`
- frozen Kai22：`HumanEgo/artifacts/frozen_checkpoints/kai22/best.pt`

  SHA256 `5d8862d400b2e8eb2bdb5dfe1fdff389b7e5f39cf53ddab3600a8e4c4df08bc3`
- R2 control：`HumanEgo/artifacts/retarget_ab_checkpoints/r2/kai22/best.pt`

  SHA256 `2a9b5c54bf38ce84512821772ba6cf9e4dcae7c25944981451637bbc616e5a37`

这些大文件仅本地保留并由 Git 忽略；对应 JSON manifest 可提交。

### compact production evidence

`hand_benchmark/production_runs/final_v3_grap_a_cap_0812/<session>/` 仅保留 exact78 的：

```text
02_raw_hawor/{hawor_projection.npz, hawor_joints_baseline.json}
08_final_v3/{joint_optimized_3d.npz,json, per_frame_quality.csv, reconstruction_quality_gate.json}
09_humanego_adapter/{adapter manifests, hand_object_interaction_v1.npz, training_data.json}
10_object6d_v2/v2_cylinder_center_rotation_gated/<compact evidence>
status.json
```

旧 adapter manifest 中可能仍描述已删除的 RGB symlink；所有这些帧级 RGB/overlay link 已清除，不得据此声称 RAW/CLEAN 可读。G0 必须用新 source manifest 解析权威 RAW。

### robot assets

本地唯一资产根为 `assets/robot/`。首次 Git 只提交资产 README 与 hash manifest；URDF、package、mesh、CAD 和 vendor symlink 均保留本地，权利确认前不得上传。`HumanEgo/vendor/kaihand` 是指向该唯一根的有意 symlink，不是第二份 producer。

## 3. 计划中的 session context

`session_context.json` 必须通过 `session_context.schema.json`，至少包含：

```text
schema_version, session_id, source_manifest_ref/sha256
width, height, fps, frame_count, frame/timestamp policy
hand_scale_px, wrist_width_px, forearm_width_px, object_scale_px
arm_area_ratio, contact_ratio, motion, blur
object6d_valid_ratio, donor_coverage
canary_frames, evidence_refs, created_at
```

所有 measured field 必须说明统计口径与缺失状态；缺失不得用 004 常数填充。

## 4. `_run` 与正式目录

一次运行只能写：

```text
_run/<run_id>/
  TASK_CARD.yaml
  run_manifest.json
  sessions/<session>/...
  qa/failure_attribution.json
```

通过全部门后，按同一文件系统原子晋级：

```text
processed/<session>_vN/
  SESSION_MANIFEST.json
  context/session_context.json
  masks/
  clean/
  object/
  retarget/
  render/
  composite/
  qa/quality_report.json
  PREVIEW.mp4
```

manifest 中列出的每个正式文件必须是普通文件、具有 bytes/SHA256、producer、schema、输入 refs 和 product line。目录版本不可覆盖；`N` 由已存在正式 manifest 分配。

晋级顺序：写 staging -> 关闭文件 -> 校验 schema/count/SHA -> fsync 文件和目录 -> 同文件系统 rename -> 写 cohort manifest。任一步失败保留 staging/HOLD，不得发布半成品。

## 5. 正式 cohort 与 selector

exact78 完成后唯一 cohort manifest 记录 source78、passed、held、paired-kept frame/window、失败分布与 session manifest SHA。训练 selector 只允许：

- RAW：由冻结 source manifest 解析的普通文件。
- RobotRGB：由正式 cohort manifest 解析的 `processed/.../FINAL` 普通文件。

禁止 selector 回退 legacy、`_run`、缺失 RAW 或其他版本。RAW 与 Robot 必须对同一 `(session, frame/window)` 做对称筛选。

## 6. 数据保留与 Git

Git 只管理源码、小型配置、schema、文档、hash manifest 和 provenance。以下不得提交：RAW、production data、R2 二进制 sidecar、checkpoint、运行目录、正式图像/视频、mesh/CAD、缓存、环境、日志、密钥和 token。

已清除的数据不可恢复；需要重建时只能从权威 RAW 和冻结 manifest 经新主干产生，不能恢复旧 producer。
