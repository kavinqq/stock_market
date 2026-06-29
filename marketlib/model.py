from __future__ import annotations
import json, math
from typing import Any, Optional

def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def predict_proba(model: dict, features: dict) -> float:
    mean = model["standardization"]["mean"]
    std = model["standardization"]["std"]
    median = model.get("imputation_median", {})
    coef = model["coefficients"]
    z = float(model.get("intercept", 0.0))
    for name in model["feature_names"]:
        x = features.get(name)
        if x is None:
            x = median.get(name, 0.0)
        s = std.get(name, 1.0) or 1.0
        x_std = (x - mean.get(name, 0.0)) / s
        z += coef.get(name, 0.0) * x_std
    return 1.0 / (1.0 + math.exp(-z))

def proba_to_direction(model: dict, p: float) -> str:
    t = model["thresholds"]
    if p >= t["bull"]:
        return "偏多"
    if p <= t["bear"]:
        return "偏空"
    return "震盪"

def proba_to_confidence(model: dict, p: float) -> str:
    bands = model.get("confidence_bands", {"mid": 0.05, "high": 0.15})
    dist = abs(p - 0.5)
    if dist >= bands["high"]:
        return "高"
    if dist >= bands["mid"]:
        return "中"
    return "低"
