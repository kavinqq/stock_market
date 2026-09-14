from __future__ import annotations
import json, math
from typing import Optional

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


DIRECTION_LABELS = ("偏多", "震盪", "偏空")


def _stage_z(stage: dict, features: dict, feature_names: list, median: dict) -> float:
    """Raw linear score for one stage: intercept + Σ coef · standardized_x."""
    mean = stage["standardization"]["mean"]
    std = stage["standardization"]["std"]
    coef = stage["coefficients"]
    z = float(stage.get("intercept", 0.0))
    for name in feature_names:
        x = features.get(name)
        # NaN must be imputed exactly as training's `_imp` does. Testing only for
        # None let a NaN through, which made z NaN; every comparison in the
        # isotonic lookup then evaluated False and the lookup fell through to its
        # last breakpoint — the HIGHEST calibrated probability. Missing data came
        # out as maximum confidence.
        if x is None or (isinstance(x, float) and math.isnan(x)):
            x = median.get(name, 0.0)
        s = std.get(name, 1.0) or 1.0
        z += coef.get(name, 0.0) * ((x - mean.get(name, 0.0)) / s)
    return z


def _apply_isotonic(cal: dict, z: float) -> float:
    """Piecewise-linear interpolation over the stored isotonic breakpoints.

    Mirrors sklearn's IsotonicRegression(out_of_bounds="clip"): inside the table
    interpolate linearly between neighbouring knots, outside it clamp to the
    endpoint. We deliberately never extrapolate — an unseen extreme score means
    we have no evidence about its hit rate, so claiming a more extreme
    probability than anything we measured would be inventing confidence.
    """
    xs, ys = cal.get("x") or [], cal.get("y") or []
    if math.isnan(z):
        # Never let an unusable score silently pick a breakpoint. The midpoint of
        # the table is the honest "no information" answer.
        return float(ys[len(ys) // 2]) if ys else 0.5
    if not xs:
        return 1.0 / (1.0 + math.exp(-z))
    if z <= xs[0]:
        return float(ys[0])
    if z >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if z <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return float(y1)
            return float(y0 + (y1 - y0) * (z - x0) / (x1 - x0))
    return float(ys[-1])


def _stage_probability(stage: dict, features: dict, feature_names: list,
                       median: dict) -> float:
    z = _stage_z(stage, features, feature_names, median)
    cal = stage.get("calibration") or {}
    if cal.get("method") == "isotonic":
        p = _apply_isotonic(cal, z)
    else:
        p = 1.0 / (1.0 + math.exp(-z))
    # Shrink toward the stage's prior when the fit ranks better than it measures.
    # The direction stage orders days informatively (AUC ≈ 0.59) but its isotonic
    # output overshoots enough to make the published probabilities worse than a
    # constant; λ, chosen on pooled OOS, keeps the ordering and returns the
    # magnitudes to something a reader can act on. λ=1 is a no-op, λ=0 discards
    # the stage entirely. Absent λ means no shrinkage was applied.
    lam = stage.get("shrinkage")
    if lam is not None:
        prior = stage.get("prior")
        if prior is not None:
            p = prior + lam * (p - prior)
    return min(1.0, max(0.0, p))


def is_two_stage(model: dict) -> bool:
    return model.get("model_type") == "two_stage_logistic"


def predict_three_class(model: dict, features: dict) -> dict:
    """Calibrated P(偏多) / P(震盪) / P(偏空) from the two-stage model.

    The split exists because the two questions have different answers. "How big
    is tomorrow's move?" carries real signal (volatility clusters — measured OOS
    AUC ≈ 0.61); "which way?" carries much less (AUC ≈ 0.59 on big-move days
    only). Forcing both through a single P(up) — as the previous binary model
    did — throws the magnitude information away, because a P(up) near 0.5 cannot
    distinguish "quiet day" from "big move, direction unknown".

        P(震盪) = magnitude stage
        P(偏多) = (1 − P(震盪)) · P(偏多 | 大動)
        P(偏空) = (1 − P(震盪)) · (1 − P(偏多 | 大動))

    which sums to 1 by construction.
    """
    names = model["feature_names"]
    median = model.get("imputation_median", {})
    stages = model["stages"]
    p_neutral = _stage_probability(stages["magnitude"], features, names, median)
    p_up_given_big = _stage_probability(stages["direction"], features, names, median)
    p_move = 1.0 - p_neutral
    return {
        "偏多": p_move * p_up_given_big,
        "震盪": p_neutral,
        "偏空": p_move * (1.0 - p_up_given_big),
    }


def direction_from_probs(probs: dict) -> str:
    """Most likely class. Ties break in DIRECTION_LABELS order (偏多/震盪/偏空)."""
    return max(DIRECTION_LABELS, key=lambda lbl: probs.get(lbl, 0.0))


def confidence_from_probs(model: dict, probs: dict) -> str:
    """Confidence tier for the winning class.

    Tiers are cut points on the winning probability whose *measured* OOS hit
    rate is recorded next to them at training time, so "高" means a rate we
    actually observed rather than a feeling about the spread of the sigmoid.
    """
    top = max(probs.values()) if probs else 0.0
    for tier in model.get("confidence_tiers") or []:
        if top >= tier.get("min_prob", 0.0):
            return tier.get("label", "低")
    return "低"


def tier_evidence(model: dict, probs: dict) -> Optional[dict]:
    """The measured OOS track record behind this prediction's confidence tier."""
    top = max(probs.values()) if probs else 0.0
    for tier in model.get("confidence_tiers") or []:
        if top >= tier.get("min_prob", 0.0):
            if tier.get("oos_hit_rate") is None:
                return None
            return {"label": tier.get("label"), "min_prob": tier.get("min_prob"),
                    "oos_hit_rate": tier.get("oos_hit_rate"),
                    "oos_sample_size": tier.get("oos_sample_size"),
                    "oos_baseline_hit_rate": tier.get("oos_baseline_hit_rate")}
    return None


def edge_demonstrated(metrics: dict) -> bool:
    """Whether the model's advantage over the best constant survives its own
    error bars — a strictly stronger claim than passing the deploy gate.

    A model can be worth deploying (calibrated probabilities, no worse than a
    constant) while having no proven edge. TW is exactly that case: 0.4405 vs
    0.4325 across eight rolling windows, bootstrap CI [-0.016, +0.032]. Keeping
    the two claims separate is what stops the report from asserting skill the
    data does not support.
    """
    rolling = metrics.get("rolling_validation") or {}
    boot = ((rolling.get("significance") or {})
            .get("vs_always_neutral_bootstrap") or {})
    ci_low = boot.get("ci_low")
    return ci_low is not None and ci_low > 0


def passes_deploy_gate(metrics: dict) -> tuple[bool, str]:
    """Decide whether an OOS metrics artifact is fit to promote to production.

    Three AND conditions. The bar was raised on 2026-09-14 after the previous
    gate passed a model that was measurably worse than guessing:

      1. Non-degenerate — every direction class is predicted at least once.
         This is what caught the US model, which collapsed to always-偏多.
      2. Beats every constant guess, not just the random prior. The old gate
         only compared against random_prior (0.352), which "always 偏多" (0.362)
         and "always 震盪" (0.438) both clear — so it waved through a model
         scoring 0.365, i.e. 7pp WORSE than a one-line constant. In TW 震盪 is
         the plurality class, so always_neutral is normally the real bar.
      3. Probabilities carry skill — Brier must beat the constant-rate Brier.
         Ranking skill (AUC) alone is not enough here: these probabilities are
         shown to a human as decision support, so a model whose numbers are
         less accurate than a fixed base rate is not fit to publish even if its
         ordering is informative. Skipped when the artifact reports no Brier.

    Returns (passes, human-readable Chinese reason).
    """
    m = metrics.get("metrics", {})
    hit_rate = m.get("hit_rate")
    per_class = m.get("classification", {}).get("per_class", {})
    baselines = metrics.get("baselines", {}) or {}

    missing = [cls for cls, v in per_class.items() if not v.get("predicted_count")]
    if missing:
        return False, f"退化：{'/'.join(missing)} 從未被預測"

    # --- strongest naive competitor: any constant guess, plus the random prior ---
    # Fallback names keep the rejection reason readable when an artifact stores a
    # baseline without its own label.
    default_labels = {
        "always_bull": "永遠偏多", "always_neutral": "永遠震盪",
        "always_bear": "永遠偏空", "momentum": "今日方向延續",
        "random_prior": "依實際先驗隨機",
    }
    competitors: list[tuple[str, float]] = []
    for key, entry in baselines.items():
        if not isinstance(entry, dict):
            continue
        rate = entry.get("hit_rate")
        if rate is None:
            rate = entry.get("expected_hit_rate")
        if rate is not None:
            competitors.append((entry.get("label") or default_labels.get(key, key),
                                float(rate)))

    if competitors and hit_rate is not None:
        best_label, best_rate = max(competitors, key=lambda t: t[1])
        if hit_rate < best_rate:
            return False, (
                f"命中率 {hit_rate:.3f} 未達最強基準「{best_label}」{best_rate:.3f}"
            )
    else:
        best_label, best_rate = None, None

    # --- probabilities must be better than a constant base rate ---
    brier = m.get("brier")
    brier_baseline = m.get("brier_baseline")
    if brier is not None and brier_baseline is not None and brier >= brier_baseline:
        return False, (
            f"機率未校準：Brier {brier:.4f} 未優於常數基準 {brier_baseline:.4f}"
        )

    # --- rolling-origin estimate outranks the single held-out window ---
    # Where the fixed split lands moves the number by several points, so when a
    # rolling estimate is available it is the one that decides. TW showed 0.468
    # on the fixed split (p=0.008 vs always-震盪) but 0.441 across eight rolling
    # windows (p=0.57): the fixed split was period luck, not skill.
    if metrics.get("rolling_validation_error"):
        # Fail closed. Falling back to the single split when validation crashes
        # would promote on exactly the evidence this gate exists to distrust.
        return False, f"滾動驗證未完成：{metrics['rolling_validation_error']}"

    rolling = metrics.get("rolling_validation") or {}
    r_hit = rolling.get("hit_rate")
    r_base = (rolling.get("baselines") or {}).values()
    r_best = max([v for v in r_base if v is not None], default=None)
    if r_hit is not None and r_best is not None and r_hit < r_best:
        return False, (
            f"滾動驗證命中率 {r_hit:.3f}（n={rolling.get('pooled_sample_size')}）"
            f"未達最強常數基準 {r_best:.3f}"
        )
    r_brier = rolling.get("brier")
    r_brier_base = rolling.get("brier_constant_prior")
    if r_brier is not None and r_brier_base is not None and r_brier >= r_brier_base:
        return False, (
            f"滾動驗證機率無 skill：Brier {r_brier:.4f} 未優於常數 {r_brier_base:.4f}"
        )

    if best_label is None:
        return True, f"通過：命中率 {hit_rate:.3f}，三類皆有預測（無基準可比）"
    detail = f"通過：命中率 {hit_rate:.3f} ≥ 最強基準「{best_label}」{best_rate:.3f}，三類皆有預測"
    if brier is not None and brier_baseline is not None:
        detail += f"，Brier {brier:.4f} < 常數 {brier_baseline:.4f}"
    if r_hit is not None:
        detail += f"，滾動驗證 {r_hit:.3f}（n={rolling.get('pooled_sample_size')}）"
        detail += ("；優勢通過信賴區間檢定" if edge_demonstrated(metrics)
                   else "；但優勢未通過信賴區間檢定（與最強常數無顯著差異）")
    return True, detail
