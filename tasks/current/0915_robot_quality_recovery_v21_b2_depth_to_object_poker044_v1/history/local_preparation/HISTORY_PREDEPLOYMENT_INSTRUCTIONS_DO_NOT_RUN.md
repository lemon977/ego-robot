# 远端注册与执行（由主执行者完成）

本准备阶段没有写远端。主执行者应使用 `apply_patch` 部署代码和合同，并在两份生成器源中加入完全相同的算法条目：

```text
src/chaoyang/ops/current_r3_contracts.py
src/chaoyang/governance/current_r3_contracts.py
```

算法名建议固定为：

```text
b2_depth_to_visible_object_poker044_v1
```

算法条目内容：

```python
"b2_depth_to_visible_object_poker044_v1": {
    "execution_scope": "POKER044_PINNED_DEPTH_AND_VISIBLE_MASK_CANDIDATE_CPU_REVIEW_ONLY",
    "weights": "ABSENT",
    "gpu_required": False,
    "code_closure": [
        artifact_ref(repo_root / "src/chaoyang/ops/run_b2_depth_to_visible_object_poker044_v1.py"),
        artifact_ref(repo_root / "contracts/robot_recovery/B2_DEPTH_TO_VISIBLE_OBJECT_POKER044_V1.json"),
        artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_object6d_wave0_v1.py"),
        artifact_ref(repo_root / "src/chaoyang/pipeline/object6d_planar_observability_v2.py"),
        artifact_ref(repo_root / "src/chaoyang/pipeline/object6d_planar_observability_v1.py"),
    ],
    "parent_task_id": "0915_robot_quality_recovery_15h_v2",
    "candidate_id": "B2_DEPTH_TO_VISIBLE_OBJECT_GEOMETRY_SUCCESSOR_V1",
    "session_id": "play_cards_0915_044",
    "depth_execution": "REUSE_PINNED_CACHE_NO_RERUN",
    "physical_card_identity": "UNKNOWN_UNBOUND",
    "face_identity": "UNKNOWN_UNBOUND",
    "review_only": True,
    "consumer_allowed": False,
    "authority_promoted": False,
    "claim_limit": "CPU-only Poker044 direct-visible finite-surface review canary from pinned cached Depth and visible-mask candidate data; physical-card/face identity remains UNKNOWN and no Contact, hidden geometry, external metric, Clean, Robot, training, control or deployment authority is granted.",
},
```

部署后先等待现有聚合器 heartbeat 生成新的 `ALGORITHM_CONTRACT.json`，然后执行：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m chaoyang.cli validate-governance
```

验证为 `PASS/FRESH` 后，唯一执行命令为：

```bash
cd /mnt/workspace/code/chaoyang
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m chaoyang.cli run run_b2_depth_to_visible_object_poker044_v1 \
  --config /mnt/workspace/code/chaoyang/contracts/robot_recovery/B2_DEPTH_TO_VISIBLE_OBJECT_POKER044_V1.json
```

执行后验收：

```bash
cd /mnt/workspace/code/chaoyang
python - <<'PY'
import json
from pathlib import Path
p = Path("_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/B2_DEPTH_TO_OBJECT_POKER044/RESULT.json")
d = json.loads(p.read_text())
assert d["review_only"] is True
assert d["consumer_allowed"] is False
assert d["physical_card_identity"] == "UNKNOWN_UNBOUND"
assert d["face_identity"] == "UNKNOWN_UNBOUND"
assert d["gpu_used"] is False
assert d["foundationstereo_rerun_performed"] is False
assert d["contact_authority"] is False
assert d["hidden_geometry_inferred"] is False
assert d["training_eligible"] is False
assert d["control_ground_truth"] is False
assert d["immutable_regressions_mutated"] is False
assert d["review"]["full_decode"] is True
assert d["review"]["frame_count"] == 166
print(d["status"], d["visible_patch_counts"])
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m chaoyang.cli validate-governance
```

该运行是 fresh-output 合同；输出目录存在时会拒绝覆盖。不得为“重跑”而删除旧输出；如需新尝试，必须创建新版本合同与新输出路径。
> **SUPERSEDED_DO_NOT_EXECUTE**
>
> 本文档是部署前历史指令。B2 已登记、执行并通过独立收据审计；
> 当前唯一入口为上级目录 `DEPLOYED_CLOSURE.json`。不得重放下文命令。
