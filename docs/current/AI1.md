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

