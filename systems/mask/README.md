# Mask

当前有两条独立SAM3.1 lane：`sam31_role_successor_v3`与`sam31_task_object_identity_v1`。入口、权重身份、证据和质量门见 [当前注册表](../../docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)。

固定边界：四角色独立；Chips三个物理实例禁止union；歧义/不可见时fail-closed empty；任一lane为C都不可绕过。
