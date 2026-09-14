import os
import cv2
import numpy as np
import json
from typing import Any


def read_json(path: str) -> dict:
    """Read required JSON without fabricating a missing/invalid frame."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid required JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"required JSON is not an object: {path}")
    return value


def safe_imread_gray(path: str, h: int, w: int) -> np.ndarray:
    """Read a required grayscale image and resize; never return a blank fallback."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"OpenCV cannot decode required grayscale image: {path}")
    if im.shape[0] != h or im.shape[1] != w:
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
    return im


def safe_imread_rgb(path: str, h: int, w: int) -> np.ndarray:
    """Read a required color image and resize; never return a blank fallback."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"OpenCV cannot decode required color image: {path}")
    if im.shape[0] != h or im.shape[1] != w:
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    return im


def make_json_serializable(obj: Any) -> Any:
    """Recursive helper to convert types/numpy to serializable JSON."""
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (list, tuple)): return [make_json_serializable(i) for i in obj]
    if isinstance(obj, dict): return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, type): return str(obj)
    if hasattr(obj, '__dict__'): return make_json_serializable(vars(obj))
    return str(obj)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
