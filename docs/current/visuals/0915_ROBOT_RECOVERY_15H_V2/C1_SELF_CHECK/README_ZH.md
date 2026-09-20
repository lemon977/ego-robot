# C1 CPU-only internal self-check

结论：实现值得保留，真实冻结缓存已通过内部实现自检；但 Contact successor 仍是明确 **NO-GO**。92–98 七帧同时满足 packed card02 overlap 与对齐指尖触觉（均 7/7），却没有任何独立可见 finger surface（0/7），且物理 RGB-depth registration 仍为 `UNBOUND`。因此输出只能是 `Contact=UNKNOWN / authority=NONE / R1-E=CLOSED`。

## 文件

- `c1_cpu_internal_selfcheck.py`：纯 CPU、只读输入审计；不含模型推理、拟合、publisher 或 authority 提升路径。
- `test_c1_cpu_internal_selfcheck.py` 与 `fixtures/unit_fixture.json`：有限 patch、packed overlap、阈值漂移和 fail-closed 单元回归。
- `REMOTE_RUN_RECEIPT.json`：对远端冻结缓存的真实实跑统计。
- `TARGET_CONTROL_STATS.csv`：92–98 与七个诊断对照的逐帧有限统计。
- `C1_SELF_CHECK_SUMMARY.svg`：一页关系图。
- `DESIGN_RECEIPT.json`：输入/生产代码 SHA、冻结策略和 claim boundary。
- `TEST_RESULT.json`：本地 CPU 单测收据。
- `SHA256SUMS.txt`：本目录交付文件哈希。

## 真实数据只读运行

脚本经 stdin 送到远端解释器，不在远端创建脚本或结果文件；`CUDA_VISIBLE_DEVICES` 置空：

```bash
ssh -o BatchMode=yes aliyun \
  'CUDA_VISIBLE_DEVICES="" python - \
  --input-index /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/C1/C1_EXECUTABILITY_INPUT_INDEX_V1.json \
  --design /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/C1/C1_MINIMAL_DIAGNOSTIC_DESIGN_V1.json' \
  < c1_cpu_internal_selfcheck.py
```

成功终态是 `PASS_INTERNAL_SELF_CHECK_FAIL_CLOSED_CONTACT_UNKNOWN`，退出码 0。任何 SHA、坐标、identity、timestamp、5 mm gate、packed support 或 authority 漂移都会以具名 `STOP_*` 终态退出码 2，Contact 仍保持 UNKNOWN。

## 单测

```bash
python3 -m unittest -v test_c1_cpu_internal_selfcheck.py
```

依赖只有 Python 3 与 NumPy；不导入 torch/CUDA，也不调用任何模型。

## 实跑摘要

- 27 个顶层输入和 168 个嵌套深度帧全部按 byte count + SHA-256 验证；嵌套深度共 652,959,266 bytes。
- 504 个 packed support rows 全覆盖：455 个可观测有限 patch，49 个不可观测且严格全零；23,202 个边界点 100% 位于对应 support，最大重投影残差 `6.3553e-14 px`，边界 Z/depth 最大误差 `0 m`。
- 251 个 metric rows 全部重放；inside finite patch 为 0，固定 5 mm 内为 0；最小有限 patch 距离为 `24.304 mm`。
- 92–98 的 object overlap=7/7、tactile=7/7、独立 finger surface=0/7、metric row=0。
- 七个 negative rows 仅是诊断对照，绝不作为 physical no-contact ground truth；其中 frame 163 的独立可见 surface 也被如实保留。

内部 packed/depth/K roundtrip 只能证明实现自洽，不能建立外部 registration 或 Contact 真值。要重开 Contact successor，至少还需要同 session、带校准 lineage 的独立物理 registration bound，以及 contact-adjacent window 内独立观测的 finger surface（或另一种独立几何测量）。
