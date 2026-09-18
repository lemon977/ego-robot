# 0915 Removal Envelope V2 真实视频 Canary

本目录比较封存 SAM semantic base 与 V2 的局部 repair。V2 没有重跑 SAM，没有 inpaint，也不允许进入 Depth、Object6D、Contact 或 Robot 几何。

- 自动终态：`REJECTED_QUALITY`
- Removal admission：`REJECTED_QUALITY`
- 150 帧完整解码：是
- repair-majority frames：`0`
- protected object core damage：`0` px
- 即使自动门通过，也必须完成人工全片复核后才能成为 Clean authority。
