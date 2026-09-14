# 数据与 I/O 规范

状态：`G0_PASS / T2_B0A_BENCHMARK_FROZEN_0_255 / T2_B0B_AUDITOR_HUMAN_REVIEW_POLICY_FROZEN / ROUTE_A_D1_FULL_002_012_RESULT_INTEGRITY_PASS_VISUAL_HOLD / ROUTE_A_D2_TASK32_P0_3_STOPPED_NO_RUN / ROUTE_A_COMBINED_SKIPPED / ROUTE_B_POINT_SCOPE_CPU_QA_PASS_GPU_HOLD / ORACLE_CLEAN_TABLE_REPROJECTION_PENDING_ATLAS_SKIPPED / EEVEE_FULL_ASSET_GPU_THROUGHPUT_PASS_VISUAL_ONLY / NULL_CALIBRATION_N2_HOLD / ROBOT_CHAIN_MOUNT_HOLD / B3_BLIND_HOLD / FORMAL_CLEAN_TRAINING_PROMOTION_HOLD`

日期：2026-08-28

## 0. 终态数据口径增量（23:38）

当前执行入口为 `task/CURRENT_EXECUTION_STATUS_20260828.md`；总审计为
`audits/OVERNIGHT_SUMMARY_20260828.md`，SHA-256
`5d18b73b8722f66b8115460a316f88c8f5e9c549e5563e2a162a041953fe3d03`。

- task18 的 `9/24、32/48` 只保留为历史 panel baseline。后续 task26 的固定39帧为
  `18/24、42/48` 但仍HOLD；task29的当前full-session数据为002+012共757帧，
  task26 selector left=`542/757`、right=`757/757`，差=`28.40pp`。
- task29 保存4,228个shared raw instances ×2 selectors ×2 sides=`16,912`条有序审计记录。
  formal ACCEPT mask只能绑定被接受的唯一raw instance；HOLD mask为空。该局部runner证据闭合
  不等于task31 common-trunk observability已通过，也不等于视觉H完整。
- P1视觉QA仍发现wearable gap、`012`左侧215个空HOLD、31个accepted hand-only/
  non-boundary和38次zero-IoU/status flip。两侧wearable semantic coverage均为
  `UNMEASURED`；topology/coverage proxy不得作为语义缺陷率或正式gate。
- task32四条中性wearable prompt未进入推理；P2在run前P0=3，0 GPU/0 candidate。
  因此不存在004 wearable修复视频、赢家prompt或可被consumer读取的D2 MASK。
- Route B V4只通过point identity/scope的CPU QA；真实production authority pin、绑定新输入的
  显式owner release与future input-bound独立re-QA（P0/P1=0）均未闭合。补pin本身不解锁
  GPU，任何Route-B prompt/mask路径保持不可消费。

上述候选/诊断仍只位于`_run/`或`audits/`，不得登记进`DATA_CATALOG.json`或
`processed/`。本次没有新增正式artifact type或production数据身份。

## 1. 根目录

- 项目根：`/mnt/workspace/code/chaoyang`
- 权威外部 RAW：`/mnt/data/egodata`（项目只读；G0 当前验证可读）
- 临时运行根：`/mnt/workspace/code/chaoyang/_run`
- 正式产物根：`/mnt/workspace/code/chaoyang/processed`

除只读 RAW 外，后续所有代码、配置、context、日志、staging、QA、预览和正式输出必须位于项目根内。

G0 已验证 004/025/149 共 1,365 帧和三条原始视频均可读且全量解码通过，exact78 的 78/78 RAW 视频均为普通非空文件并可打开。OSS 系统挂载选项实际为 `rw`，这不扩大项目权限：source resolver 必须做根目录约束、拒绝 symlink/fallback 并仅以只读方式打开；挂载失效时由平台恢复，禁止本地副本替代。

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

E0 已严格按 manifest 为 78 sessions / 32,318 frames 重建 adapter 内的绝对权威 RAW `rgb.png` 只读 symlink，并逐 session 复核 count、target 与随机/顺序读取；错误/缺失为 0，`rgb_WoArm` 仍为 0。它们只提供 RAW selector 基础，不是 CLEAN/RobotRGB，也不解除 cohort/训练 HOLD。

### robot assets

本地唯一资产根为 `assets/robot/`。D1 T0 pin SHA256 为 `4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330`：84 files、63/63 mesh refs 可解析/加载；calibration file count=0。robot-chain preflight JSON SHA256 `f6dd417d20b1f235cb949fcf6bb27740e920731da79d6c199332bc171bf10099` 进一步确认 Tianji 左右各 7 个有限位关节、KaiHand 22+22 joint identity、004 R2 460 帧与 RAW K 460/460 一致。proxy复算的四个几何数值成立，但43.282439mm只是flange到Kai rear-plane的gap，不能定义`T_tool→hand_base`；左右4×4 mount的旋转/roll/平移仍不唯一。`q_arm + session-constant base` 只能在该mount另行pin后形成候选。空 calibration 目录只能允许 nominal-URDF visual candidate，不能声明实机标定。首次 Git 只提交资产 README/hash manifest/小型验证代码；URDF、package、mesh、CAD 和 vendor symlink 均保留本地，权利确认前不得上传。`HumanEgo/vendor/kaihand` 是指向该唯一根的有意 symlink，不是第二份 producer。

Oracle管路数据严格位于`_run/oracle_clean_plumbing_t1_v1/oracle_mask_plumbing/`，字段固定`mask_source=HUMAN_ORACLE_DEVELOPMENT_LABEL`、`plumbing_validation_only=true`、`is_clean_candidate=false`、`formal_consumer_allowed=false`。只含004:228–242；H/U residual分别2,776,105/28,069。CLEAN v1把object-atlas路由固定为0并保留品红unresolved；未来table reprojection只允许使用digest-bound、按session估计一次并冻结的静态table plane写像素。逐帧reanchor仅在该帧存在独立罐底接触证据时允许；当前contact 15/15均为`UNRESOLVED`，故逐帧分支保持HOLD。每个写入像素仍须source+confidence，U填充/残留单独统计。该目录禁止成为benchmark、训练输入、正式CLEAN或RobotRGB。

### SAM3.1 admission evidence

独立准入复核见 `audits/SAM31_MIRROR_ADMISSION_REVIEW.md`：非官方镜像 `Supbatomic/sam3-1-model@99cb53e…` 与官方固定 revision 的 12/12 metadata/blob identity 和两项 LFS SHA256 一致；权重为 3,502,755,717 bytes、SHA256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。用户接受镜像与 SAM License 后，隔离下载、strict load 和真实 004 四帧均完成。固定官方 wrapper 的 API mismatch 已由独立 `sam31_compat_adapter_v1` 解决，adapter 报告 SHA256 `9b32159bbd20d5e408fbb9d1b0e0f34abb242108e9b5d961ad7ed6c592da09c4`，独立 QA SHA256 `3a039495627113826a75beeb180488d538bc27b2f8acc19175bb1d50ff076089`。结论仍为 `PASS_COMPATIBILITY_EVIDENCE_WITH_CLAIM_LIMIT`，只证明 adapter/API 可运行。

完整 ego-human development 证据已继续 fail-closed：P2 text route 独立 QA SHA256 `bdd610e54d1de254f6f3e26dab27e4d8a8346a1c224b876c8bea32bd315df729`，004:228–242 为 0/15 sufficient、15/15 HOLD；P3 box-only 独立 QA SHA256 `efbd57d52f41302a11a1c8ec3652b2d47e473798c5c665ed295971dc2594143c`，为 0/30 eligible、0/15 sufficient、0/15 nonempty final MASK。两条 run 均要求 human review、阻塞下一 bucket，且 formal consumer=false；镜像不是官方来源，SAM3.1 未注册为 production input，P2/P3 不得作为 CLEAN 输入。

详细提示 probe 的 run manifest SHA256 为 `b950060a8915625eeb418ea4ba8a4af02e0de324cab5a4df181f8ddd3ad55fa3`。A′ dev15为9/15，固定自由session为2/24。6个`AUTHORITY_EVIDENCE_OUTSIDE_IMAGE`帧经RAW目视均存在双侧ego手臂，正式归因不得使用`NO_TARGET_IN_FRAME`，分母保持24。每帧证据必须显式包含`target_present_left/right`、`pass_left/right`、左右失败原因和分母资格；raw=0或种子在画外只能标记authority evidence failure。所有selector拒绝已召回实例必须保存独立overlay；双侧组合仍仅`VISUALIZATION_ONLY`，025全session禁止读取。

上述 A′ 数据的 `route_of_evidence=A`。task18历史per-side panel口径为004=`15/15`、free 002/005/012=`6/8、0/8、3/8`，合计`9/24`；left/right=`15/24、17/24`，左右差`8.33pp`，development→free泛化差`62.5pp`。它不是当前full-session口径。39帧的通过侧输出只能引用唯一raw instance，失败侧路径必须为null，禁止union/fill/crop/morph/Object6D subtract。Route A 的 selector/recall 事实不得用于证明或否定 Route B；Route B 数据链为 label-independent HaWoR left/right point prompts→SAM2.1 decoder→MASK，必须独立保存 prompt identity、source lineage、quality、越界状态和每侧结果。单侧证据失败不得使另一侧全局 abort。Route B 还必须单列 `scope_complete_left/right`，检查点提示是否覆盖完整 hand、wrist/cuff、sleeve 与前臂到图像边界；系统性偏短记录 `SYSTEMATIC_SCOPE_TOO_SMALL`。

Route A 修订 QA 的逐侧 denominator 固定为24：left=`3/24`、right=`16/24`，差=`54.17pp`；6帧 visible-target global abort、8帧 boundary-only misreject、3帧 side identity alias 分开记账。005:370 记录为 source-slot identity exchange。所有23项反例入口固定在 `_run/rejected_visual_review_v5/`，不得作为 Route B 训练输入。

`audits/SAM31_TRAINABILITY_T0.md` 的范围仅为 Route B point 接口，不能引用为“SAM3.1不可训练”。官方 concept/text 训练栈属于 Route A；其触发条件见 `task/17_OFFICIAL_SAM31_TRAINING_STACK_SCOPE_T0.md` 与旁挂 scope errata。监督 U 像素采用 `ASYMMETRIC_FALSE_NEGATIVE_ONLY`，必须同时报告 `*_u_as_h` 与 `*_u_excluded`，禁止零权重或硬并入 H。

benchmark development 是 004:228–242 连续 15 帧、跨度 0.467 s，相邻 SSIM 均值 0.930739。修法 B fold 固定为 `[228,231,234,237,240]`、`[229,232,235,238,241]`、`[230,233,236,239,242]`；它仍不能替代第三方 session 视觉检查。15:00负责人指令已解除 NULL 的磁盘 blocker；N=2 当前只因 formal selector 未实现、only-seed config不等价与025封存冲突而不得启动。

### 004_old legacy reference

`004_old/` 最终只读审计稳定为 5 个普通文件：3/3 视频全量解码 PASS、2/2 PNG 解码 PASS。机器清单 `audits/004_OLD_READ_ONLY_INVENTORY_V2_FINAL.json` SHA256 `6e90941a…53ae`，报告 SHA256 `5ce85088…90bf`。这些文件只可用于 MASK ownership、CLEAN 负例、机器人构图和 retarget 的定性参考；formal artifact、阈值 GT、producer/training 输入均为 0，禁止 legacy fallback。

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

## 3.1 MASK benchmark、人工标签与实验身份

benchmark 必须通过合同 schema，总计 45–60 帧，并绑定权威 source manifest、每帧 RAW SHA 和人工 H/O/U/B 标签 SHA。三分区为：

```text
development_004:    15–20，允许 producer 开发查看
same_session_blind: 15–20，评测前封存
cross_session_blind:15–20，来自未调参 session，评测前封存
```

每个 MASK 候选还必须绑定 `producer_pN`、像素语义 SHA 和实现 SHA；每份 MASK 独立 QA 必须绑定冻结 `auditor_a1`、协议/阈值 SHA、benchmark SHA 和实际评测覆盖。v3 的 460 帧、v8 的 4 帧和 v9 的 15 帧不构成同覆盖数据，统一重评前禁止排名。

### 当前 v1 staging

- 根：`_run/g2_mask_benchmark_v1/`
- manifest：`BENCHMARK_MANIFEST.json`
- manifest SHA256：`85f09ef99f3445350619a2f202f70c9182fc5ffd5f0297e32dfd32038f212a36`
- 结构：60 帧，004 development 15、004 same-session blind 15、025/149 cross-session blind 各 15。
- 状态：`AWAITING_HUMAN_HOUB_ANNOTATION`，不是 frozen benchmark。
- 2026-08-27 15:55:43 +08:00 快照：active label 6/60（`00228–00233`）。首个连续窗口 `00228–00232` 的五张均为普通 P-mode 1280×960 文件且只含 0/1/2/3，RAW/标签 overlay 显示双臂 H、任务物体 O、窄 U 接触带和背景排除在五帧间一致；`00233` 同样通过格式门，但尚未进入视觉 pilot review。pilot review 为 `human_labels/PILOT_5_REVIEW.json`；它只批准迁移五张已审核标签到新 v2，不冻结 benchmark、不解封 blind、不授权候选。manifest/run manifest 中的 0/60 是创建时快照，不得当实时进度。

标注 UI 只写 `human_labels/palette_png/<session>/<frame>.png`：原尺寸、P mode、索引 `B=0/H=1/O=2/U=3`。前景 H 只指需要替换的 ego 双手/前臂/衣袖，背景人物为 B。早期因颜色溢出产生的 4 张错误标签已移动到 `rejected_human_labels/`，不能被 validator、freezer 或 auditor 读取；失败提交的可恢复证据也不等于 active label。

### 当前 v2=45 staging

v2 已原子建立于 `_run/g2_mask_benchmark_v2/`，直接采用 v1 冻结顺序中的前九个五帧窗口：004 development 15、004 same-session blind 15、025 cross-session blind 15；v1 中 149 的 15 帧全部排除并保留为后续 final blind。v2 manifest SHA256 为 `60b9df265169bb2a4d575bfc3a836039c9ba0d618922a6f8b08dbf7b44b9fde1`，run manifest SHA256 为 `0692d73123c8ad710367da3277d0a907620e8c76a9b8f21e1ae809de6d6c5ce4`。独立 builder/validator SHA256 分别为 `e2f3a432571cbab2e5eff85057d5c1258402e503ed2c2cb73862976ee6507a4a` 与 `f87150516df6dc5d6fc5933922650b231d4f18299493f2941cece9f72f12523c`。

五张已审 development 标签在 source SHA、尺寸、palette 语义和 pilot review SHA 完全匹配后迁移并锁定；migration provenance SHA256 为 `7d57cfcf4c7741ad43068acb693607b016d3d9dcd3df31190e1c4639d6be7b92`。2026-08-27T20:41+08 人工 45/45 已齐，`validate_houb_labels_v2.py` PASS，锁定 5 张 SHA 仍 5/5 一致。v1 的 233 未通过完整窗口 review，故没有静默迁移。442–446 连续 5 帧 U=0：标注者确认该窗无手物接触，保持 U=0，不补画。

UI palette → 四张二值 H/O/U/B 的 T0 快照位于 `_run/g2_mask_benchmark_v2/derived_labels_t0/3fa396.../label_manifest.json`，SHA256 `3038770f…01d33`。独立 QA 发现其像素值域是 `{0,1}`，而 auditor 明确要求 `{0,255}`，所以 T2-B0a 没有直接冻结这些 PNG。新冻结 namespace `_run/g2_mask_benchmark_freeze_v1/8bdd4c…/` 包含 180 个 `{0,255}` mask、正式 benchmark/label manifests 与批准记录；`BENCHMARK_FREEZE.json` SHA256 `09f94084a84cd0ac0d3e93a3311da2b41c85788742029e89347b18afb6ed073b`。45/45 reverse palette bit-exact，原 T0 不变且禁止 consumer 读取。B1 独立复标包 manifest SHA256 `7f346743…f565`，用户已在 0/8 时免除复标；noise floor=`UNAVAILABLE_USER_WAIVED`。

`auditor_a1` 实现 SHA256 为 `82d21b01a623093d2c89308ba7d1dd596f6d062d6273b8044db5df12356e9c7c`。T2-B0b 已绑定实现、公式、schema、runtime 和 benchmark freeze，以 `HUMAN_REVIEW_POLICY` 冻结：`_run/g2_auditor_a1_freeze_v1/8e084d…/AUDITOR_FREEZE.json`，SHA256 `d2cd400a5ffb7e30f365f1ae3af21fb916b31302e64034ca8ba75c36c001fd9c`。经验 noise floor/自动阈值均为 null；它只支持诊断排名，不能输出 production PASS。B3 与 blind 评测仍须独立授权；缺少完整覆盖或统一 identity 时只能 HOLD/`FORENSIC_NOT_COMPARABLE`。

## 4. `_run` 与正式目录

一次运行只能写：

```text
_run/<run_id>/
  TASK_CARD.yaml
  run_manifest.json
  sessions/<session>/...
  qa/failure_attribution.json
```

v9 仅是 `G2_DIAGNOSTIC_ONLY_NOT_MASK_CANDIDATE`。A2 r3 位于 `_run/a2_004_window60_producer_p1_r3/`，manifest SHA256 `d70afdf99fe45647cc8e42d2cb5434453f45c795c993dec277bbc2df859055b2`；它是 T1 法证 run，不是合法 `mask-evidence-v1`，且存在两帧严重桌面误选。SAM3.1 P2/P3 development-15 也分别是严格 0/15 HOLD：P2 不能完整覆盖袖臂，P3 fail-closed 输出 15/15 空 final MASK。D2/D3/D4 的 `_run`/audit 也分别只是失败尝试、synthetic compositor 证据与输入任务 manifest。以上任何路径均不得被 selector、CLEAN 或训练读取；只有集中 registry 可以只读引用 review 证据，benchmark/人工标签也不因 45/45 自动成为 `processed/`。

所有 `REJECT/HOLD` 产物必须由单一证据 registry/indexer 引用，不移动或覆盖原法证文件。当前不可变 v1 为 `_run/rejected_visual_review_v1/MANIFEST.json`，SHA256 `9ae293385f4f0f5cafec8f3d66d8a2ad5403759abc0f749f6d32413c3da5abcf`；其 `INDEX.html` SHA256 为 `edd14102eb994023fb1b0f54afa383bbea413d82748ea089218dbd3aa1abe888`，13 项覆盖 MASK v1–v9 与 A2 的代表性失败证据。每条记录绑定 RAW、候选、QA 与可视化 SHA；整包只供人工 review，不是 formal artifact，禁止 selector/producer 消费。未来失败新增不可变版本，不覆盖 v1，也不因登记或 review 解封 T2。

P2 HOLD 的不可变 v2 为 `_run/rejected_visual_review_v2/MANIFEST.json`，SHA256 `e0556a84d2d24dc841d8c084d0ecad30a7db497b1926f37ad45d492dc91cc88f`；`INDEX.html` SHA256 `92ca7196b256fe92f3bef4a1d5b13cf172aa2710b32b51b8c7063a9a65bc065d`，5 assets 覆盖四张代表帧和完整 15 帧 review 视频。v2 不覆盖 v1，不是 formal artifact，也不解封 P2/P3、B3、full460 或 CLEAN。

P3 HOLD 的不可变 v3 为 `_run/rejected_visual_review_v3/MANIFEST.json`，SHA256 `352af39f9ce0d7e849a91f4acf1acade735f9486a9ba067a6317f15cf093f29a`；`INDEX.html` SHA256 `423a37585567698ab17c0b5026aa6676a081018c7f89ef58c26897aaff215630`，4 assets 覆盖 228/235/242 代表帧和完整 development-15 视频。v3 不覆盖 v1/v2，不是 formal artifact，也不解封任何下游。

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

## 7. checkpoint 与 matched 数据前置

三份 checkpoint 与两个 bundle 的 bytes/SHA、split/run/stats/config 引用以及 R2 的 78/78 sidecar 谱系已通过只读审计。这只证明保留完整：pretrained 是共同初始化候选，旧 Kai22 是 legacy forensic，R2 best 是 numerical control；都不是未来 matched RAW/Robot 任一训练臂。

E1 已统一 `humanego-formal-run-v2` 静态字段与 runtime validator；E2 已把 RAW/Robot 配置收敛为仅 `img_name` 一项差异、固定 batch64、`persistent_workers=false` 并禁止 autotune。Robot `img_name` 仍为 null，不能猜测 selector。正式 matched 数据在以下七类证据闭合前不存在：manifest-backed RAW selector、manifest-backed Robot/cohort selector、唯一 `PAIRED_KEPT_MANIFEST`、两臂共同初始化 transfer report、canonical resolved config、validation-only checkpoint selection ledger、test8 one-shot ledger。训练/评测/freeze guard 保持 `HOLD_MANIFEST_SELECTOR_UNIMPLEMENTED`。
