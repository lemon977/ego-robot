# exact78 转换率首阻塞归因 V2

> 本页由 156 行机器账本生成；C 不自动等于模型视觉失败。

## 首个互斥阻塞

| 首阻塞 | 数量 |
|---|---:|
| `HAWOR_C` | 12 |
| `ROLE_MASK_C` | 20 |
| `OBJECT_MASK_C` | 23 |
| `CALIBRATION_MISSING` | 43 |
| `METRIC_GEOMETRY_READY` | 58 |

## 条件转换率

| 条件 | 通过/分母 | 转换率 |
|---|---:|---:|
| `hawor_given_raw` | 144/156 | 92.3% |
| `role_given_hawor` | 124/144 | 86.1% |
| `object_given_hawor_role` | 101/124 | 81.5% |
| `metric_given_triple_ab` | 58/101 | 57.4% |
| `metric_given_raw` | 58/156 | 37.2% |

## 原因性质

| 性质 | 数量 |
|---|---:|
| `ALGORITHM_QUALITY` | 54 |
| `INFRASTRUCTURE_FAILURE` | 1 |
| `MISSING_EVIDENCE` | 43 |
| `NONE` | 58 |

## 当前直接结论

- 58/156 是达到公制几何层的总转换率，不是 Robot 转换率。
- 缺标定的 43 条已经通过三路视觉输入，只能进入 Visual Tier；它们不是算法失败。
- Role Mask 首阻塞 20 条中包含运行时/输入合同问题，必须与真正的 mask 质量失败分开整改。
- Object Mask 的 23 条首阻塞均为 Poker 身份锚点缺失或锚点后快速丢失，适合独立的因果重入 successor。
- 所有明细、证据路径和 successor route 见同目录 JSON/CSV。
