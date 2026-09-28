# Git快照、运行证据与复现边界

本次整理保存截至2026-09-28的既有工程改动和文档，不代表算法采用，不训练模型。

## 提交范围

源码、测试、配置/合同、文档、任务包及小型治理收据受Git管理。模型权重、环境、缓存、原始/processed数据、运行视频及完整attempt实体不提交；现有大型第三方资产仍按既有规则排除。当前分支不强推，不自动合并main。

GitHub上的代码与计划可读；历史收据引用的绝对路径指向H20，不能据此声称任意新克隆能离线重放所有实验。远端已有资产由清单/收据确认，不能因缺文件自动联网补模型。

## 状态解释

HEAD/提交记录用于代码版本；治理revision用于发布事实；Task Packet revision用于冻结任务。它们不要求数值相等。提交后的HEAD变化不会改写旧收据中的执行时commit。

当前算法事实从[STATUS](STATUS.json)和[最新结果](SHARED_HAND_DELIVERY_RESULT_ZH.md)读取。只备份Git不等于备份运行数据；忽略的archive和_run需要单独受控保管。

## 本次验证与已知问题（2026-09-28）

- 文档治理与相关工程回归：72 passed，含7项新增文档整理检查；旧正文13份SHA一致，85项历史任务没有新增执行授权。
- 全仓探索性回归在发现重复失败后主动停止：151 passed / 34 failed，未完成全套，不能标为全项目PASS。
- 其中strict_io的测试夹具放在_run路径时触发“禁止staging来源”门。改用项目内.cache/documentation_handoff_20260928独立夹具后，66 passed；生产安全门未修改。
- checkpoint发布身份检查在新夹具路径仍复现：test_runtime_checkpoint_authority_is_exclusive_and_complete，ValueError: runtime checkpoint authority transaction identity drift（frozen_contract.py:727）。本次未修改该实现，需后续独立调查文件身份/硬链接发布语义，不能据此宣布训练checkpoint发布可靠。
- 首次独立夹具试跑因父目录缺失出现2项setup errors，补建本任务目录后重跑；失败记录保留，不覆盖。
- 原始回归XML保留于服务器的_run/current/documentation_handoff_20260928/attempts/attempt_0001，不随Git上传。未重跑模型、未训练checkpoint。
- 已有历史Markdown中的硬换行空格和末尾空行原样保留；不为消除格式警告改写冻结文本。

GitHub目标为公开仓库lemon977/ego-robot；新增4张采集画面问题截图、模型/环境/缓存/数据及运行视频均不提交。GitHub文档中的这些媒体链接只能在H20对应项目树内访问。本提交是待继续优化的工程快照，不是产品质量发布。

## 恢复文档

[归档说明](../archive/navigation_20260928/README_ZH.md)与MANIFEST记录整理前原字节及SHA。恢复正文后必须重新经过publisher发布新revision，不能倒写旧receipt。历史Task Packet/RESULT不搬迁、不改写。
