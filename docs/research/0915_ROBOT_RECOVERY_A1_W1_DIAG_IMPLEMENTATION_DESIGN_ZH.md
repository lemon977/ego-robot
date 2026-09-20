# A1 W1-DIAG 最小维护实现草案

状态：`LOCAL_DRAFT_PREFLIGHT_PASS_NOT_DEPLOYED`。未运行 GPU，未写远端或治理。

## 结论

旧 Kai22 R0 的依赖实现确实错误，不是数值门失败本身：

- `run_0915_hawor_persistent_worker_v1.py:225-227` 用两侧
  `numeric_mask_gate_pass` 的 `all(...)` 生成 whole-session HaWoR strict gate。
- `run_0915_robot15h_kai22_r0_wave0_v1.py:489-492` 直接把
  `hawor_by_id[session_id]["status"] == "PASS_DEVELOPMENT_HAWOR"` 与全覆盖绑定为
  `r0_quality_admitted`。它没有在这里执行/聚合 Kai22 自身逐侧的限位、FK、
  非邻接碰撞和时间门。因此另一侧缺失或任一整片 strict 失败会把已有合法侧的
  全部 R0 窗口归零。
- `run_hawor_bounded_parameter_successor.py:421-459` 将任一侧
  `NO_OBSERVATION`/质量失败放进同一全局 failure list；Kai22 R0
  `:461-466` 再按整会话在 bounded 与 raw 间切换，两侧不能独立选择。
- 已封存 W0 事实证明该传播真实发生：Chips007 有 378 个直接观察 side-frame，
  Chips042 有 710 个，R0 四条都导出，但 `r0_success=0`；这不能证明这些 frame
  已通过 R0 自身门，只证明 whole-session HaWoR flag 把它们统一拒绝。

修复原则：保留原 whole-session strict 结果；运行许可、逐侧 structural window、
R0 自身质量准入分别输出。不得把 structural eligibility 直接改名成
`r0_quality_admitted=true`。

## 最小维护文件

部署时新增，不覆盖任何旧文件：

1. `src/chaoyang/ops/run_0915_robot_recovery_hawor_w1_diag_v1.py`
   - 本地草案 SHA256：`534416707138c42d65d844f321b579a3b351462d749a48c43573182bc0784dc0`
   - 只从冻结 access ledger 选择固定 044/097；不读 W0 RESULT。
   - 两条一次 persistent load，并通过现有 V7.1 GPU 单租约运行。
   - 输出逐侧 strict、structural windows、consumer admission、candidate freeze。
2. `src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py`
   - 本地草案 SHA256：`1288883c661d0fafb100f31345c7697724a8589328611e9fdfcadff7af9c2fe4`
   - 新 schema；固定 044/097、2 sessions、560 frames；一个 model load、一个 detector load、每会话 tracker reset。
   - 原 strict 阈值和原 session strict 聚合保持不变，额外输出 `per_side_strict`。
3. `contracts/robot_recovery/A1_HAWOR_W1_DIAG_CONFIG_V1.json`
   - 本地草案 SHA256：`625e0add5d6580e35e44318fd41bde83520834527a460b0649ebf56da3b9ee6d`
4. `contracts/robot_recovery/A1_PREPARED_MANIFEST_V2.schema.json`
   - 本地草案 SHA256：`4bdbe826b4caddb9a372d80f641ead83c886bc114066feb6f642e4e6493aae10`

旧 `run_0915_hawor_resize_only_persistent_worker_v2.py` 保持原 9231 bytes / SHA256
`1288bf26bb335564552fbb2fc73475fbc460fe68154f8486d1d6edfe45445240`；新 worker
只 import 它的稳定 helper/legacy binding，不改变旧证据绑定。

## 固定输入

Access ledger：23834 bytes / `f08c34638b22fbfa4a66a3413915c90e77d667310e5341e71cbc860aa640f17f`。

| session | frames | source stereo bytes / SHA256 | camera params bytes / SHA256 | source snapshot SHA256 |
|---|---:|---|---|---|
| play_cards_0915_044 | 166 | 14556213 / `b48b40bb52a6c469e72e2c542658fe889e4baa8a909de5c52fd285a0b1990a23` | 3454 / `7f19d5ce92b5188a0013bd7ae85e3c031bc81303e54b5343605151bbdb777b0e` | `a0f5a6c672ed809facf65b16b4362875c7b188ce47a7dd131c404097ed161108` |
| get_potato_chips_0915_097 | 394 | 31347559 / `642d0b6450d8eca29971a8724fd1382c9042ff0f1a6d7e310134aad11084b314` | 3454 / `a40d4d0a56c229ac24be9558e88ad3c0f47dc146fec7068b89cf1fb24aad4f41` | `891786766dd447206ecb38f035a83f6bf3ae4cfe85c09374992220ed173eb94c` |

Prepared manifest 还必须绑定源 `training_data.json` 的有序逐文件 manifest，以及生成后的
adapter metadata 有序逐文件 manifest；只绑定视频 SHA 不足以冻结 c2w/frame axis。

## 固定代码与权重

现有依赖（部署前再次 exact-ref 断言）：

- W0 preparation helper：`901e38db7e46a70effbce6b3f36ad02690710a94287ee046c0a7381f621b570d`
- 原 v2 worker（只读依赖）：`1288bf26bb335564552fbb2fc73475fbc460fe68154f8486d1d6edfe45445240`
- legacy persistent/quality：`aa9cedc28b99c20c7cd122b5880ba205b535624fb5bec371e279328103fa50a3`
- upstream raw：`1ca9e682eab7d9d176e6ff7b7c848d2bc647658ce23aef0b3f2288feecc0604b`
- resize/domain helper：`e08740ba3b070f749968e569d8dd90532783a4a53568965152b70b18f7ad3845`
- `hawor_python.sh`：`e3db614e9f59ac68fac09d84c57455cebd35953d4c6ff4c410e4fe05e1a17a8a`
- V7.1 lease wrapper：`de1bc50327f012bf662c24dc4650902ee2802aea5848225a258ad9f6156aa7e3`
- bounded successor（首轮仅审计、不消费）：`91faf80669b3fb971db5f77362d50a80539a24483d1c959136520e4ce47c6df8`
- Kai22 R0 audit target：`5d389faef24255be58cf45036977059c9df5c52c43c619cc05e43cc767df612c`

权重：

- logical bundle：887 bytes / `45b130f19369d7a7184f21fac27cfaa53c5eea3716ab2fa8b69d86a56ca87d79`
- HaWoR checkpoint：3267481572 bytes / `4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2`
- detector：53582271 bytes / `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033`

Candidate signature 是上述输入、代码、config、weights、input domain 的 canonical JSON SHA；
运行结果不能改变 signature。

## 阈值与 admission

原 whole-session strict 阈值全部不变：observed fraction 0.95、最长缺口 8、confidence
median 0.65、p05 0.45、positive depth 0.995、bone CV 0.08、rotation orthogonality
0.0001、determinant >0、in-frame 0.90，且原 session strict 仍要求两侧都通过。

新增输出仅拆权限：

- `HAWOR_DIAGNOSTIC_REVIEW`：合法输入、签名和 runtime artifact 通过即可消费。
- `KAI22_R0_Q22_EVALUATION` 与 camera-relative wrist：逐 anatomical side 输出 maximal
  contiguous structural windows，允许进入 R0 自身求解/门检查。
- `r0_quality_admitted` 固定写 `PENDING_R0_OWN_GATES`，直到限位、FK、非邻接碰撞和
  时间门完成；A1 不发质量 PASS。
- world-relative wrist 默认 false，未绑定可信 c2w/SLAM 合同不得升级。
- 当前 bounded successor 不进入首轮 W1-DIAG 消费，因为它仍是两侧/整会话耦合。
  后续必须实现 per-side candidate selection 与 per-side raw fallback，且不改变既有数值阈值。

## GPU 租约与终态

调用现有 `run_gpu_command_with_v71_lease.py`：`priority=CANARY`、`gpu_id=0`、
`min_free_mib=61440`、`wait=1800s`、`wall=3600s`，全局最多一个 owner。

终态：

- `BLOCKED_INPUT_FINAL`：ledger/access/session/input/config/SHA/frame axis 不符。
- `BLOCKED_RESOURCE`：GPU 等待预算耗尽；不计质量失败。
- `FAILED_RUNTIME_FINAL`：worker runtime 失败或任一会话无 runtime terminal。
- `FAILED_EVIDENCE_FINAL`：结果/冻结顺序不闭合。
- `COMPLETED_DIAGNOSTIC_ALL_TERMINAL`：两条均 runtime terminal；strict 可为 0/2，进程仍成功完成诊断。

Freeze 必填：candidate id/signature、完整 code/config/weight/input-domain refs、
`numeric_thresholds_changed=false`、W1-DIAG evidence、106/029 在 freeze 时仍为
`SEALED_UNTIL_CANDIDATE_FREEZE`、`result_based_candidate_change_allowed=false`、
同签名 runtime retry policy 与 attempt limit。

## 建议部署命令（由父任务 publisher 执行）

```bash
python -m chaoyang.ops.run_0915_robot_recovery_hawor_w1_diag_v1 \
  --config /mnt/workspace/code/chaoyang/contracts/robot_recovery/A1_HAWOR_W1_DIAG_CONFIG_V1.json \
  --output-root /mnt/workspace/code/chaoyang/_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A1 \
  --gpu-id 0 --min-free-mib 61440 --gpu-wait-seconds 1800 --gpu-wall-seconds 3600 \
  --executor-epoch 1
```

部署前还应将四个新文件的实际远端 SHA 写入父任务 technical snapshot；本草案未做该写入。

## 已做 preflight

- 三个 Python 文件 AST/compile PASS。
- 新 schema Draft 2020-12 metaschema PASS，560-frame synthetic 044/097 manifest validation PASS。
- 新 worker/orchestrator 的远端只读 import + `--help` PASS。
- orchestrator worker command 只指向新 worker；旧 v2 仅作为 code/helper dependency。
- 旧 v2 本地副本 SHA 回归等于远端原 SHA。
- 未运行 GPU，未创建远端产物。
