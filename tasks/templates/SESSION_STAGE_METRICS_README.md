# Session stage metrics input

新数据经过 HaWoR 后，将每条 session 的摘要写为 JSON，然后运行：

```bash
python tools/evaluate_session_stage_admission.py \
  tasks/<task>/reports/<session_id>_stage_metrics.json \
  --output tasks/<task>/reports/<session_id>_stage_gate.json
```

字段和阈值见 `docs/newtask/HAWOR_MASK_CLEAN_ROBOT_QUALITY_GATES_V1.md`。输出会给出最早
失败 stage 和责任 owner。缺少阶段摘要时是 `NOT_EVALUATED`，不会静默 PASS。

可复制 `SESSION_STAGE_METRICS.example.json` 作为完整字段模板；其中的数值只是通过接口测试
的示例，不是新数据的实测结果。
