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


def passes_deploy_gate(metrics: dict) -> tuple[bool, str]:
    """Decide whether an OOS metrics artifact is fit to promote to production.

    Two AND conditions (see spec Part 3):
      1. Non-degenerate: every direction class is predicted at least once. This
         is what caught the US model, which collapsed to always-偏多.
      2. Beats naive guessing: hit_rate >= random-prior expected hit rate. Both
         are hit-rate scale, so the comparison is apples-to-apples (macro-F1 is
         a different scale and is NOT compared against a hit-rate baseline).
    Returns (passes, human-readable Chinese reason).
    """
    m = metrics.get("metrics", {})
    hit_rate = m.get("hit_rate")
    per_class = m.get("classification", {}).get("per_class", {})
    random_prior = (metrics.get("baselines", {}).get("random_prior") or {}).get(
        "expected_hit_rate"
    )

    missing = [cls for cls, v in per_class.items() if not v.get("predicted_count")]
    if missing:
        return False, f"退化：{'/'.join(missing)} 從未被預測"

    if random_prior is not None and hit_rate is not None and hit_rate < random_prior:
        return False, f"命中率 {hit_rate:.3f} 未達隨機先驗 {random_prior:.3f}"

    if random_prior is None:
        return True, f"通過：命中率 {hit_rate:.3f}，三類皆有預測（無隨機先驗可比）"
    return True, (
        f"通過：命中率 {hit_rate:.3f} ≥ 隨機先驗 {random_prior:.3f}，三類皆有預測"
    )
