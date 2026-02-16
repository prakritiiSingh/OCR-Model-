# utils.py
import os
import json
from typing import List, Tuple, Dict
import numpy as np

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def bbox_from_xywh(x,y,w,h):
    # convert to [x0,y0,x1,y1]
    return [int(x), int(y), int(x + w), int(y + h)]

def bbox_iou(a: List[int], b: List[int]):
    # a, b = [x0,y0,x1,y1]
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0, inter_x1 - inter_x0)
    ih = max(0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0, ax1-ax0) * max(0, ay1-ay0)
    area_b = max(0, bx1-bx0) * max(0, by1-by0)
    union = area_a + area_b - inter + 1e-8
    return inter / union

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
