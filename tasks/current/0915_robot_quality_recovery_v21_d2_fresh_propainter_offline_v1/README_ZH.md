# D2 fresh-fill successor 交接

本目录冻结了唯一的 D2 候选：对 D1 `PREPARED_V1` 进行 fresh ProPainter 离线视觉补全。本次只完成代码、配置、CPU 测试和远端只读 preflight，**没有执行模型**。

关键文件：

- `D2_FRESH_PROPAINTER_OFFLINE_V1.json`：单一冻结配置。
- `run_d2_fresh_propainter_offline_v1.py`：候选 runner。
- `publish_d2_shallow_visuals_v1.py`：深层结果通过后的原子浅层 MP4 发布器。
- `test_d2_fresh_propainter_offline_v1.py`：CPU 合同测试。
- `PREFLIGHT_RESULT.json`：远端 D1/模型/权重只读校验。
- `DECISION.md`：权威边界和硬拒绝条件。

部署后的 CPU preflight 形式：

```bash
python -m chaoyang.ops.run_d2_fresh_propainter_offline_v1 \
  --config contracts/robot_recovery/D2_FRESH_PROPAINTER_OFFLINE_V1.json \
  --preflight-only --full-sha --receipt /NEW/NO_CLOBBER/PREFLIGHT.json
```

真实模型运行必须在登记候选 runner 和浅层 publisher 后，用下列唯一 V7.1 GPU lease 命令：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m chaoyang.cli run \
  run_gpu_command_with_v71_lease \
  --task-id 0915_robot_quality_recovery_v21_d2_fresh_propainter_offline_v1 \
  --attempt-id attempt_0001 \
  --executor-epoch 1 \
  --priority CANARY \
  --gpu-id 0 \
  --min-free-mib 61440 \
  --wait-seconds 1800 \
  --wall-seconds 7200 \
  --receipt /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/D2_FRESH_FILL_GPU_RECEIPT.json \
  --claim-limit 'OFFLINE_BIDIRECTIONAL_VISUAL_ONLY; SYNTHETIC pixels remain UNKNOWN; NOT_FOR_TRAINING' \
  -- \
  python -m chaoyang.cli run run_d2_fresh_propainter_offline_v1 \
  --config /mnt/workspace/code/chaoyang/contracts/robot_recovery/D2_FRESH_PROPAINTER_OFFLINE_V1.json \
  --output-root /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/D2_FRESH_FILL_OFFLINE_V1
```

runner 不直接向最终路径写数据：它在最终目录的同盘 sibling staging 中生成全部内容，异常时删除 staging 且保持 final 不存在，成功时只执行一次 `os.replace(staging, final)`。所有 JSON 内部引用在提交前已映射为 final 路径，出现 staging 路径将硬失败并清理。

Mask 语义已明确冻结：传给 ProPainter 的输入 mask 是缩放到 `960×720` 的 D1 `M_flow`；固定参数 `mask_dilation=4` 会在模型内部再得到 `DILATE(RESIZED_M_flow,4)`。这个扩展只是模型推理上下文，**不是可发布写域**；最终候选仍只能在原分辨率 `M_write` 内写入，其他像素必须与 Raw byte-exact。

权重 provenance 同时冻结两层路径：`vendor/ProPainter/weights/<name>` 是模型实际打开的词法路径，当前其 `weights` 目录是指向 `assets/models/vendor/propainter` 的软链接；配置因此同时固定 `lexical_path` 和 `resolved_path`。校验器先要求词法路径精确等于 vendor 入口，再要求解析目标精确等于 assets 实体，最后对解析实体验证 bytes/SHA。不得因 `resolve()` 将词法路径折叠后误判为“权重不在 vendor 中”。

固定 vendor 原生保存四位序列 `0000.png..0393.png`。D2 先校验该序列的文件名集合完全相等（不是只数数量），再 byte-exact 复制为唯一允许消费的六位序列 `000000.png..000393.png`，并再次校验完整文件名序列。

深层 GPU 收据为 `PASSED` 且 D2 `RESULT.json` 状态正确后，再原子发布浅层视频：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m chaoyang.cli run \
  publish_d2_shallow_visuals_v1 \
  --deep-result /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/D2_FRESH_FILL_OFFLINE_V1/RESULT.json \
  --gpu-receipt /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/D2_FRESH_FILL_GPU_RECEIPT.json \
  --shallow-root /mnt/workspace/code/chaoyang/docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/D2_FRESH_FILL
```

发布器在复制视频前会读取并 SHA 绑定 GPU receipt，严格要求 `schema=v71-gpu-command-receipt-v1`、`status=PASSED`、精确 task/attempt，以及 receipt 中的 command 必须逐项等于上述 D2 命令且 `--output-root` 精确指向 deep `RESULT.json` 的父目录。任何不匹配都在 shallow staging 可见前硬拒绝。然后它复制两个实体 MP4，拒绝软链接，重算 SHA 和 166/394 帧解码数，在同盘 sibling staging 写完 `INDEX.json`/`README_ZH.md` 后仅用一次 `os.replace` 提交。全局浅层 README 只能在该原子发布成功后通过现有文档聚合流程更新。

Poker 不调用模型；唯一 GPU 任务是 Chips097 394 帧。
