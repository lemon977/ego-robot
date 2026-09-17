"""Frozen label schema shared by the UI, rasterizer and validator."""

SCHEMA_VERSION = "offline-mask-9class-v1"

CLASSES = (
    {"id": 0, "symbol": "B_BACKGROUND", "name_zh": "背景", "color": "#000000", "hotkey": "0"},
    {"id": 1, "symbol": "L_SKIN", "name_zh": "左手/裸露前臂", "color": "#21d4d4", "hotkey": "1"},
    {"id": 2, "symbol": "R_SKIN", "name_zh": "右手/裸露前臂", "color": "#ff922b", "hotkey": "2"},
    {"id": 3, "symbol": "L_SLEEVE", "name_zh": "左袖口/衣袖", "color": "#845ef7", "hotkey": "3"},
    {"id": 4, "symbol": "R_SLEEVE", "name_zh": "右袖口/衣袖", "color": "#f06595", "hotkey": "4"},
    {"id": 5, "symbol": "L_TRACKER", "name_zh": "左tracker/腕带", "color": "#51cf66", "hotkey": "5"},
    {"id": 6, "symbol": "R_TRACKER", "name_zh": "右tracker/腕带", "color": "#94d82d", "hotkey": "6"},
    {"id": 7, "symbol": "O_TASK_OBJECT", "name_zh": "任务物体", "color": "#ffd43b", "hotkey": "7"},
    {"id": 8, "symbol": "U_UNCERTAIN", "name_zh": "不确定边界", "color": "#ffffff", "hotkey": "8"},
)

BY_ID = {item["id"]: item for item in CLASSES}
BY_SYMBOL = {item["symbol"]: item for item in CLASSES}
KNOWN_IDS = frozenset(BY_ID)
