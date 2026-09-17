# Depth / Mask / Clean / Contact / Occlusion 开发合同 R3

状态：`DEVELOPMENT_CONTRACT / NOT_CURRENT_AUTHORITY`
适用计划：`Chaoyang V7.1-R3`
实现目录：`src/chaoyang/pipeline/*_contracts_r3.py`、`contracts/*_r3.schema.json`

## 当前边界

- Mask 正式视觉基线是 **SAM3.1**；SAM2/SAM2.1/Cutie 只能作为 challenger。
- 正式 H4 仍是 `BLOCKED_RESOURCE / NOT_EVALUATED / POLICY_DEFERRED`，没有像素 Mask authority。
- 手套 Mask 的既有 `FAILED_QUALITY_C` 只属于单独的开发 canary。
- FoundationStereo 输出是可见表面 optical-Z；`depth_confidence_present=false`。
- 所有 Depth 数值只支持内部一致性，`external_metric_accuracy=UNKNOWN`。
- Controller 是新传感器手腕主锚点；MANUS25 提供手指；Stereo 只能施加因果、有界 Z 修正。
- Contact 输出是 `HYPOTHESIS_ONLY`；Attachment 不得反向证明 Contact/Object6D。
- Occlusion Silver 不报告 accuracy；UNKNOWN 像素必须 `training_valid=false`。
- 本合同不更新 current governance、不晋升任何 authority。

## 合同链路

```text
DEPTH-00 → DEPTH-10 → DEPTH-20
MASK-ROLE + MASK-OBJECT → Clean M_remove/M_flow/M_write
Raw/Object → Atlas
Causal history → Donor
Depth/Object/Mask → CONTACT-10
Robot/Clean/Contact → OCCLUSION-SILVER
```

Clean 三域固定为：

```text
M_write ⊆ M_remove ⊆ M_flow
M_write ∩ visible_object = ∅
M_write 外 byte-exact
```

训练 donor 必须满足 `donor_frame_id <= target_frame_id`。对象 Atlas 还必须同时通过实例身份和 pose 来源验证。

## 运行

所有模式均为 CPU dry-run，不运行 SAM3.1、FoundationStereo 或 ProPainter：

```bash
python src/chaoyang/ops/validate_pipeline_contracts_r3.py --mode depth-00 --dry-run
python src/chaoyang/ops/validate_pipeline_contracts_r3.py --mode mask-role --dry-run
python src/chaoyang/ops/validate_pipeline_contracts_r3.py --mode clean --dry-run
python src/chaoyang/ops/validate_pipeline_contracts_r3.py --mode contact-10 --dry-run
python src/chaoyang/ops/validate_pipeline_contracts_r3.py --mode occlusion-silver --dry-run
```

真实运行必须等待治理恢复，通过相应 Task Packet 新建 immutable attempt；不得把本次 dry-run 结果作为真实数据质量或算法 authority。
