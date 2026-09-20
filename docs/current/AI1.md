# AI1 当前执行入口

计划：[PLAN.md](PLAN.md)  
机器状态：[STATUS.json](STATUS.json)

## 数据角色

- 097/098：development。
- 101：已暴露失败 regression，只验证不拟合。
- 102/103：候选冻结后的 adoption。

## 当前任务

1. 核对 MANUS 原生 25 节点、local/global、node0、左右、单位和 root。
2. 按 `T_A_B` 闭合 wrist/controller/world/camera 链。
3. 在 camera/controller-local 两个域比较 M0/M1/M2 静态标定。
4. 输出 anatomical wrist 与 visible wrist surface 双表示；无可靠对应则 region-only。
5. 输出真实 NPZ/JSON 和完整回放；融合失败不撤销合格的原生或静态结果。

当前结果最多为 development sensor fusion，不是控制、训练真值或外部毫米标定。

## V3.1 实际终态

- CPFS 的 `renameat2(RENAME_NOREPLACE)` 兼容故障已由固定锁、目标复核和原子 rename 的有限 fallback
  修复；没有改旧失败 attempt。
- 097/098 development 与 101 regression 共完成 466 帧 position-only 对照。M0 仅是 legacy prior；
  M1 是未采用的 development candidate，不能由内部残差下降升级为真实标定。
- M2 因 orientation evidence 为 0 而阻塞；102/103 因缺少 pinned anatomical-wrist observation 阻塞。
- visible wrist surface 仍为 region-only，没有伪造三维 surface point。
- 后续审计发现旧 successor 内有 13 个 staging 引用、对应 8 个唯一文件；独立 rebind successor 已
  验证目标 bytes/SHA 并生成 corrected evidence tree，旧 canonical 树逐字节未变。

下一步只应补 102/103 的独立腕部观测并冻结后验证；不得填补缺失观测、逐帧拟合 wrist，或用
PICO/HaWoR 融合输入自证 M1 精度。
