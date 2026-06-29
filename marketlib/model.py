from __future__ import annotations
import json, math

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
    p = 1.0 / (1.0 + math.exp(-z))
    # Platt recalibration (spec §8.1/§8.2): if the model stores a Platt calibrator
    # (fit on trainval OOS because balanced class_weight distorts probabilities),
    # apply p_cal = 1/(1+exp(-(w*z + b))). Otherwise return the raw sigmoid.
    cal = model.get("calibration") or {}
    if cal.get("method") == "platt":
        params = cal.get("params") or {}
        w = params.get("w")
        b = params.get("b")
        if w is not None and b is not None:
            return 1.0 / (1.0 + math.exp(-(w * z + b)))
    return p

def proba_to_direction(model: dict, p: float) -> str:
    t = model["thresholds"]
    if p >= t["bull"]:
        return "偏多"
    if p <= t["bear"]:
        return "偏空"
    return "震盪"

def proba_to_confidence(model: dict, p: float) -> str:
    bands = model.get("confidence_bands", {"mid": 0.04, "high": 0.10})
    dist = abs(p - 0.5)
    if dist >= bands["high"]:
        return "高"
    if dist >= bands["mid"]:
        return "中"
    return "低"
