# Poker 当前推进状态

- PHONE_DEV Mask：双手/前臂与 lower cuff 已通过；双 tracker 时序未闭合，禁止给 Clean。
- 当前最小 assisted 路线：最低 8 帧×双 tracker=16实例；稳妥 11 帧×双 tracker=22实例。
- PHONE_DEV Clean：尚未运行正式全片；现有空桌与动作视角/布局不同，只能诊断。
- 新 PICO 到位后：先跑分层 admission，再在预注册24帧 canary 上执行同一共享 Mask；若
  tracker 不过直接走 assisted keyframe，不反复放宽阈值。
- 正式 Clean 必须消费同一 PICO session 的空桌/视角扫描段。

