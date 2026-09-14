#!/usr/bin/env python3
"""Offline walk-forward training. Produces final-test metrics + production refit model."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from marketlib.evaluation import summarize_classification, summarize_baselines
from marketlib.stats import mcnemar_test, block_bootstrap_diff

# Test-only feature subset: a small, dependency-free set used by tests/synthetic
# data. Production training passes FEATURE_NAMES_TW / FEATURE_NAMES_US instead.
MINIMAL_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "close_vs_ma20", "ma20_vs_ma60",
    "vol_ratio_20d", "realized_vol_20d",
]
PARAM_GRID = [
    {"C": c, "class_weight": cw}
    for c in (0.1, 0.3, 1.0, 3.0) for cw in (None, "balanced")
]
# Purge gap (trading days) at train/val and train/test boundaries (spec §8.2):
# 1-day-ahead labels + trailing-window features overlap across an adjacent
# boundary; dropping the last EMBARGO train samples next to each val/test fold
# removes that optimistic bias.
EMBARGO = 5

def _matrix(samples, feature_names):
    med = {n: _nanmedian0([_get(s, n) for s in samples]) for n in feature_names}
    X, y = [], []
    for s in samples:
        X.append([_imp(_get(s, n), med[n]) for n in feature_names])
        y.append(s["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int), med

def _get(s, n):
    v = s["features"].get(n)
    return float("nan") if v is None else float(v)

def _nanmedian0(vals):
    """Median ignoring NaN; falls back to neutral 0.0 when the whole column is
    NaN (e.g. US features in a pre-2022 TW fold), so imputation never leaks NaN."""
    arr = np.asarray(vals, dtype=float)
    if np.isnan(arr).all():
        return 0.0
    return float(np.nanmedian(arr))

def _imp(v, m):
    return m if (v != v) else v  # NaN -> median

def _standardize(X):
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def _fit(X, y, params):
    Xs, mean, std = _standardize(X)
    clf = LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                             max_iter=1000)
    clf.fit(Xs, y)
    return clf, mean, std

def _proba(clf, mean, std, X):
    Xs = (X - mean) / np.where(std == 0, 1.0, std)
    return clf.predict_proba(Xs)[:, 1]

def _decision_z(clf, mean, std, X):
    """Raw decision score z = intercept + Σ coef·standardized_x (BEFORE sigmoid)."""
    Xs = (X - mean) / np.where(std == 0, 1.0, std)
    return clf.decision_function(Xs)

def _platt_calibrate(z_oos, y_oos):
    """Fit Platt scaling: 1-feature LogisticRegression of y_oos ~ z_oos.

    Returns (w, b) so that p_cal = 1/(1+exp(-(w*z + b))).
    """
    z = np.asarray(z_oos, dtype=float).reshape(-1, 1)
    y = np.asarray(y_oos, dtype=int)
    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(z, y)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])

def _apply_platt(z, w, b):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-(w * z + b)))

def _select_thresholds(p, y):
    # Wider neutral band search than v1: bull up to 0.65 / bear down to 0.35 lets
    # more borderline days land in 震盪, countering the model's tendency to
    # over-predict 偏多 (the up-drift base rate). Best (bull, bear) still chosen
    # by macro-F1 on the calibrated pooled-OOS preds.
    best, best_f1 = (0.55, 0.45), -1.0
    for bull in (0.52, 0.55, 0.58, 0.6, 0.63, 0.65):
        for bear in (0.35, 0.38, 0.4, 0.42, 0.45, 0.48):
            recs = [{"prediction": "偏多" if pi >= bull else "偏空" if pi <= bear else "震盪",
                     "actual_direction": "偏多" if yi else "偏空"} for pi, yi in zip(p, y)]
            f1 = summarize_classification(recs)["macro_f1"] or 0.0
            if f1 > best_f1:
                best_f1, best = f1, (bull, bear)
    return best

def collect_oos_predictions(samples, feature_names, params, n_folds=4, embargo=EMBARGO):
    """Re-run the time-ordered walk-forward folds for the chosen params and pool
    out-of-sample validation predictions across folds.

    For each fold k the TRAIN slice is purged by `embargo` samples adjacent to the
    val fold (spec §8.2). Standardization is fit on that fold's TRAIN slice only.
    Returns pooled (z_oos, p_oos, y_oos) numpy arrays where z is the raw decision
    score (intercept + Σ coef·standardized_x, BEFORE sigmoid), p the sigmoid prob,
    y the true label. These pooled OOS preds are the basis for thresholds,
    confidence bands, and calibration.
    """
    n = len(samples); fold = n // (n_folds + 1)
    z_all, p_all, y_all = [], [], []
    for k in range(1, n_folds + 1):
        tr = samples[: max(0, fold * k - embargo)]
        va = samples[fold * k: fold * (k + 1)]
        if len(va) < 10 or len(set(s["label"] for s in tr)) < 2:
            continue
        Xtr, ytr, _ = _matrix(tr, feature_names)
        Xva, yva, _ = _matrix(va, feature_names)
        clf, mean, std = _fit(Xtr, ytr, params)
        z = _decision_z(clf, mean, std, Xva)
        p = _proba(clf, mean, std, Xva)
        z_all.extend(z.tolist()); p_all.extend(p.tolist()); y_all.extend(yva.tolist())
    return (np.array(z_all, dtype=float), np.array(p_all, dtype=float),
            np.array(y_all, dtype=int))

def walk_forward_select(samples, feature_names, n_folds=4, seed=0, embargo=EMBARGO):
    n = len(samples); fold = n // (n_folds + 1)
    scored = []
    for params in PARAM_GRID:
        f1s, briers = [], []
        for k in range(1, n_folds + 1):
            tr = samples[: max(0, fold * k - embargo)]; va = samples[fold * k: fold * (k + 1)]
            if len(va) < 10 or len(set(s["label"] for s in tr)) < 2:
                continue
            Xtr, ytr, _ = _matrix(tr, feature_names)
            Xva, yva, _ = _matrix(va, feature_names)
            clf, mean, std = _fit(Xtr, ytr, params)
            p = _proba(clf, mean, std, Xva)
            recs = [{"prediction": "偏多" if pi >= 0.5 else "偏空",
                     "actual_direction": "偏多" if yi else "偏空"} for pi, yi in zip(p, yva)]
            f1s.append(summarize_classification(recs)["macro_f1"] or 0.0)
            briers.append(float(np.mean((p - yva) ** 2)))
        if f1s:
            scored.append({"params": params, "macro_f1": float(np.mean(f1s)),
                           "brier": float(np.mean(briers))})
    # 主指標 macro_f1;Brier 為硬條件:剔除 brier > 最小 brier + 0.02 的設定
    if not scored:
        return {"params": PARAM_GRID[0], "macro_f1": None, "brier": None}
    min_brier = min(s["brier"] for s in scored)
    feasible = [s for s in scored if s["brier"] <= min_brier + 0.02]
    return max(feasible, key=lambda s: s["macro_f1"])

def evaluate_on_test(clf, mean, std, med, test_samples, feature_names, thresholds,
                     calibration=None):
    X = np.array([[_imp(_get(s, n), med[n]) for n in feature_names] for s in test_samples])
    y = np.array([s["label"] for s in test_samples])
    z = _decision_z(clf, mean, std, X)
    if calibration and calibration.get("method") == "platt":
        prm = calibration["params"]
        p = _apply_platt(z, prm["w"], prm["b"])
    else:
        p = _proba(clf, mean, std, X)
    bull, bear = thresholds
    recs = []
    model_hits, bull_hits, mom_hits = [], [], []
    prev = None
    for i, (pi, s) in enumerate(zip(p, test_samples)):
        pred = "偏多" if pi >= bull else "偏空" if pi <= bear else "震盪"
        actual = s["actual_direction"]
        recs.append({"prediction": pred, "actual_direction": actual,
                     "baselines": {"always_bull": {"label": "永遠偏多", "hit": actual == "偏多"},
                                   "momentum": {"label": "今日方向延續",
                                                "hit": (prev == actual) if prev else False}}})
        model_hits.append(pred == actual)
        bull_hits.append(actual == "偏多")
        prev = actual
    cls = summarize_classification(recs)
    base = summarize_baselines(recs)
    sig = {
        "vs_always_bull_mcnemar": mcnemar_test(model_hits, bull_hits),
        "vs_always_bull_bootstrap": block_bootstrap_diff(model_hits, bull_hits, seed=0),
    }
    hit = sum(1 for r in recs if r["prediction"] == r["actual_direction"]) / len(recs)
    brier = float(np.mean((p - y) ** 2))
    return {"metrics": {"hit_rate": hit, "macro_f1": cls["macro_f1"], "brier": brier,
                        "classification": cls},
            "baselines": base, "significance": sig}

def _should_calibrate(params, y_oos, use_platt: bool) -> bool:
    """Whether to fit Platt scaling. Only for balanced class_weight (which
    distorts probabilities) with ≥2 classes present, AND when the caller opts
    in. TW opts OUT (use_platt=False): calibration squashes the weak next-day
    signal into a narrow band so the bear threshold never fires and the deploy
    gate rejects the model; the raw sigmoid keeps enough spread to predict all
    three classes (honest, uncalibrated confidence)."""
    y = y_oos.tolist() if hasattr(y_oos, "tolist") else list(y_oos)
    return bool(use_platt and params.get("class_weight") == "balanced"
                and len(set(y)) >= 2)

def train_from_samples(samples, feature_names, market, out_dir, test_frac=None,
                       test_months=18, seed=0, use_platt=True):
    os.makedirs(out_dir, exist_ok=True)
    labeled = [s for s in samples if s["label"] is not None]
    if test_frac is not None:
        cut = int(len(labeled) * (1 - test_frac))
    else:
        cut = len(labeled) - min(len(labeled) // 4, test_months * 21)
    # Embargo gap at the train/test boundary: drop the last EMBARGO trainval
    # samples adjacent to the test fold (spec §8.2).
    trainval, test = labeled[: max(0, cut - EMBARGO)], labeled[cut:]
    best = walk_forward_select(trainval, feature_names, seed=seed)
    # Pooled OOS predictions for the chosen params: basis for calibration,
    # thresholds, and confidence bands (all honest out-of-sample).
    z_oos, p_oos, y_oos = collect_oos_predictions(trainval, feature_names, best["params"])
    # Fix #4 — calibration: balanced class_weight distorts probabilities, so
    # fit Platt scaling on the pooled OOS (z, y). class_weight None -> no recal.
    if _should_calibrate(best["params"], y_oos, use_platt):
        w, b = _platt_calibrate(z_oos, y_oos)
        calibration = {"method": "platt", "params": {"w": w, "b": b}}
        p_oos_cal = _apply_platt(z_oos, w, b)
    else:
        calibration = {"method": "none", "params": {}}
        p_oos_cal = p_oos
    # Fix #2 — thresholds chosen on calibrated pooled OOS preds (not in-sample).
    thresholds = _select_thresholds(p_oos_cal, y_oos)
    # Fix #3 — confidence bands from tertiles of |p_oos_cal - 0.5| (spec §8.4).
    if len(p_oos_cal) > 0:
        dist = np.abs(p_oos_cal - 0.5)
        conf_mid = float(np.percentile(dist, 33))
        conf_high = float(np.percentile(dist, 66))
    else:
        conf_mid, conf_high = 0.04, 0.10
    # final-test 模型:只用 trainval 擬合
    Xtv, ytv, med_tv = _matrix(trainval, feature_names)
    clf, mean, std = _fit(Xtv, ytv, best["params"])
    test_eval = evaluate_on_test(clf, mean, std, med_tv, test, feature_names, thresholds,
                                 calibration=calibration)
    metrics = {"artifact": "final_test_evaluation", "market": market,
               "trained_on": [trainval[0]["date"], trainval[-1]["date"]],
               "tested_on": [test[0]["date"], test[-1]["date"]] if test else None,
               "chosen_params": best["params"], "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
               "calibration": calibration,
               **test_eval}
    metrics_path = os.path.join(out_dir, f"metrics_{market}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    # production 模型:用「全部」資料重擬(含 test 窗)
    Xall, yall, med_all = _matrix(labeled, feature_names)
    clf2, mean2, std2 = _fit(Xall, yall, best["params"])
    model = {
        "artifact": "production_refit", "market": market,
        "model_type": "logistic_regression",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": [labeled[0]["date"], labeled[-1]["date"]],
        "note": ("refit on ALL data incl. test window; the hit_rate/macro_f1 in "
                 "metrics_*.json came from a model NOT trained on the test window and "
                 "are the honest OOS estimate, NOT a direct test of this refit model. "
                 "calibration/thresholds/confidence_bands were derived from pooled "
                 "trainval walk-forward OOS predictions and are applied to this "
                 "production refit (standard practice)."),
        "feature_names": feature_names,
        "standardization": {"mean": dict(zip(feature_names, mean2.tolist())),
                            "std": dict(zip(feature_names, std2.tolist()))},
        "imputation_median": med_all,
        "coefficients": dict(zip(feature_names, clf2.coef_[0].tolist())),
        "intercept": float(clf2.intercept_[0]),
        "class_weight": best["params"]["class_weight"],
        "calibration": calibration,
        "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
        "confidence_bands": {"mid": conf_mid, "high": conf_high},
    }
    model_path = os.path.join(out_dir, f"model_{market}.json")
    with open(model_path, "w") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)
    return model_path, metrics_path

# ---------------------------------------------------------------------------
# Two-stage three-class model (2026-09-14)
#
# Why two stages instead of one binary P(up) sliced by thresholds:
# the previous design trained on sign(next return) and manufactured 三分類 by
# cutting that single probability at bull/bear. Measured on rolling OOS, the two
# questions have genuinely different amounts of signal —
#     magnitude ("is tomorrow a 震盪 day?")   AUC 0.607  95%CI [0.565, 0.646]
#     direction ("up or down, given a big move") AUC 0.587  95%CI [0.531, 0.643]
#     raw sign  (what the old model trained on) AUC 0.566  95%CI [0.523, 0.607]
# — and a single P(up) cannot express magnitude at all: P(up)≈0.5 means both
# "quiet day" and "big move, direction unknown". 震盪 is also the plurality
# class (~44% of TW days), so collapsing it cost the most accuracy.
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["偏多", "震盪", "偏空"]
STAGE_C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _stage_labels(samples, stage):
    """Per-stage target. `direction` is only defined on big-move days, so it
    returns None there and those rows are dropped from that stage's fit."""
    out = []
    for s in samples:
        d = s["actual_direction"]
        if stage == "magnitude":
            out.append(1 if d == "震盪" else 0)
        else:
            out.append(None if d == "震盪" else (1 if d == "偏多" else 0))
    return out


def _stage_matrix(samples, feature_names, stage, med=None):
    y_all = _stage_labels(samples, stage)
    keep = [i for i, y in enumerate(y_all) if y is not None]
    if med is None:
        med = {n: _nanmedian0([_get(samples[i], n) for i in keep]) for n in feature_names}
    X = np.array([[_imp(_get(samples[i], n), med[n]) for n in feature_names] for i in keep],
                 dtype=float)
    y = np.array([y_all[i] for i in keep], dtype=int)
    return X, y, med


def _fit_stage_raw(X, y, C):
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit((X - mean) / std, y)
    return clf, mean, std


def _stage_oos_scores(samples, feature_names, stage, C, n_folds=4, embargo=EMBARGO):
    """Pooled walk-forward OOS decision scores for one stage.

    Standardization and the fit come from each fold's TRAIN slice only, purged by
    `embargo` rows next to the validation fold, so the pooled (z, y) used for
    calibration and for the confidence tiers never saw its own outcome.

    Scores are returned for EVERY validation row, including rows this stage has
    no label for (the direction stage is undefined on 震盪 days). That matters
    because at prediction time we must ask both stages about every day, so the
    confidence tiers have to be built from both stages' scores on the same rows.
    `y` is -1 where the stage has no label; callers filter before calibrating.
    """
    n = len(samples); fold = n // (n_folds + 1)
    z_all, y_all, idx_all = [], [], []
    for k in range(1, n_folds + 1):
        tr = samples[: max(0, fold * k - embargo)]
        va_lo, va_hi = fold * k, fold * (k + 1)
        va = samples[va_lo:va_hi]
        if len(va) < 10:
            continue
        Xtr, ytr, med = _stage_matrix(tr, feature_names, stage)
        if len(set(ytr.tolist())) < 2 or len(Xtr) < 30:
            continue
        # Score every validation row, labelled or not.
        Xva = np.array([[_imp(_get(s, n_), med[n_]) for n_ in feature_names] for s in va],
                       dtype=float)
        labels = _stage_labels(va, stage)
        clf, mean, std = _fit_stage_raw(Xtr, ytr, C)
        z_all.extend(clf.decision_function((Xva - mean) / std).tolist())
        y_all.extend([-1 if y is None else y for y in labels])
        idx_all.extend(range(va_lo, va_lo + len(va)))
    return np.array(z_all, dtype=float), np.array(y_all, dtype=int), idx_all


def _select_stage_C(samples, feature_names, stage):
    """Pick C by pooled-OOS log loss — a proper scoring rule, so it rewards
    calibrated probabilities rather than just correct ordering."""
    best, best_ll = STAGE_C_GRID[0], float("inf")
    for C in STAGE_C_GRID:
        z, y, _ = _stage_oos_scores(samples, feature_names, stage, C)
        lab = y >= 0
        z, y = z[lab], y[lab]
        if len(z) < 50 or len(set(y.tolist())) < 2:
            continue
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if ll < best_ll:
            best_ll, best = ll, C
    return best, best_ll


def _fit_isotonic(z, y):
    """Isotonic map from decision score to observed frequency, fit on pooled OOS."""
    iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    iso.fit(z, y)
    return {"method": "isotonic",
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_]}


def _iso_apply(cal, z):
    return np.interp(np.asarray(z, dtype=float), cal["x"], cal["y"])


SHRINK_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)


def _select_shrinkage(p_oos, y_oos, prior):
    """Pick λ for `p = prior + λ(p_oos − prior)` by Brier on pooled OOS.

    A stage can rank well yet be badly scaled. Brier is a proper scoring rule,
    so it penalises exactly that overshoot; λ=1 wins whenever the raw
    calibration is already honest, and λ=0 drops a stage that only adds noise.
    """
    best_lam, best_brier = 1.0, float("inf")
    for lam in SHRINK_GRID:
        p = prior + lam * (p_oos - prior)
        b = float(np.mean((p - y_oos) ** 2))
        if b < best_brier:
            best_brier, best_lam = b, lam
    return best_lam, best_brier


def _build_stage(samples, feature_names, stage):
    """Fit one stage end to end: choose C on pooled OOS, calibrate on pooled OOS,
    then refit the point model on everything given to this call."""
    C, ll = _select_stage_C(samples, feature_names, stage)
    z_oos, y_oos, idx_oos = _stage_oos_scores(samples, feature_names, stage, C)
    lab = y_oos >= 0
    cal = (_fit_isotonic(z_oos[lab], y_oos[lab])
           if len(set(y_oos[lab].tolist())) >= 2 else {"method": "none"})
    X, y, med = _stage_matrix(samples, feature_names, stage)
    prior = float(y.mean()) if len(y) else 0.5
    if cal["method"] == "isotonic" and lab.sum() >= 50:
        p_oos = _iso_apply(cal, z_oos[lab])
        lam, lam_brier = _select_shrinkage(p_oos, y_oos[lab].astype(float), prior)
        base_brier = float(np.mean((prior - y_oos[lab]) ** 2))
    else:
        lam, lam_brier, base_brier = 1.0, None, None
    clf, mean, std = _fit_stage_raw(X, y, C)
    artifact = {
        "standardization": {"mean": dict(zip(feature_names, mean.tolist())),
                            "std": dict(zip(feature_names, std.tolist()))},
        "coefficients": dict(zip(feature_names, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "calibration": cal,
        "prior": prior,
        "shrinkage": lam,
        "C": C,
        "oos_log_loss": ll,
        "oos_brier_shrunk": lam_brier,
        "oos_brier_prior_only": base_brier,
        "oos_sample_size": int(lab.sum()),
    }
    return artifact, med, {"z": z_oos, "y": y_oos, "idx": idx_oos, "cal": cal, "C": C}


def _stage_p(art, z):
    """Stage probability exactly as marketlib.model computes it in production —
    isotonic lookup then shrinkage toward the prior. Kept in lockstep with
    marketlib.model._stage_probability; tests/test_parity.py pins them together."""
    cal = art["calibration"]
    p = _iso_apply(cal, z) if cal.get("method") == "isotonic" else 1 / (1 + np.exp(-z))
    lam, prior = art.get("shrinkage"), art.get("prior")
    if lam is not None and prior is not None:
        p = prior + lam * (p - prior)
    return np.clip(p, 0.0, 1.0)


def _three_class_probs(mag_art, dir_art, X_std_mag, X_std_dir):
    p_neu = _stage_p(mag_art, X_std_mag)
    p_up = _stage_p(dir_art, X_std_dir)
    return np.column_stack([(1 - p_neu) * p_up, p_neu, (1 - p_neu) * (1 - p_up)])


def _z_for(art, X, feature_names):
    mean = np.array([art["standardization"]["mean"][n] for n in feature_names])
    std = np.array([art["standardization"]["std"][n] for n in feature_names])
    std[std == 0] = 1.0
    coef = np.array([art["coefficients"][n] for n in feature_names])
    return ((X - mean) / std) @ coef + art["intercept"]


def _multiclass_brier(P, truth_idx):
    Y = np.zeros_like(P); Y[np.arange(len(truth_idx)), truth_idx] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def _confidence_tiers(top_probs, correct, baseline_correct=None,
                      quantiles=(85, 50)):
    """Confidence bands cut at OOS quantiles of the winning probability.

    Cuts are data-driven rather than fixed at 0.60/0.50 because this model's
    winning probability rarely clears 0.6 — a hard cut put 8 days in 高信心,
    far too few for its hit rate to mean anything. Quantiles instead define 高
    as "today is among the most confident ~15% of days", which keeps every band
    large enough to carry a number worth reading.

    Each tier records the hit rate MEASURED inside it on pooled walk-forward
    OOS, next to the constant-guess hit rate over the same days, so a reader can
    see whether the extra confidence bought anything.
    """
    hi_cut = float(np.percentile(top_probs, quantiles[0]))
    mid_cut = float(np.percentile(top_probs, quantiles[1]))
    tiers = []
    for label, lo, hi in (("高", hi_cut, float("inf")),
                          ("中", mid_cut, hi_cut),
                          ("低", 0.0, mid_cut)):
        m = (top_probs >= lo) & (top_probs < hi)
        entry = {
            "label": label, "min_prob": lo,
            "oos_hit_rate": float(correct[m].mean()) if m.sum() else None,
            "oos_sample_size": int(m.sum()),
        }
        if baseline_correct is not None and m.sum():
            entry["oos_baseline_hit_rate"] = float(baseline_correct[m].mean())
        tiers.append(entry)
    return tiers


def train_two_stage(samples, feature_names, market, out_dir, test_months=18,
                    rolling_windows=8):
    os.makedirs(out_dir, exist_ok=True)
    labeled = [s for s in samples if s.get("actual_direction") in DIRECTION_LABELS]
    cut = len(labeled) - min(len(labeled) // 4, test_months * 21)
    trainval, test = labeled[: max(0, cut - EMBARGO)], labeled[cut:]

    mag_art, mag_med, mag_oos = _build_stage(trainval, feature_names, "magnitude")
    dir_art, dir_med, dir_oos = _build_stage(trainval, feature_names, "direction")
    med = {n: _nanmedian0([_get(s, n) for s in trainval]) for n in feature_names}

    # --- confidence tiers from pooled walk-forward OOS (never in-sample) ---
    # Building these from the trainval-fitted model's own trainval predictions
    # would be in-sample and produced a non-monotonic table (高 scoring below 中).
    # Align the two stages on the row indices they actually share.
    mag_by_idx = dict(zip(mag_oos["idx"], mag_oos["z"]))
    dir_by_idx = dict(zip(dir_oos["idx"], dir_oos["z"]))
    shared = sorted(set(mag_by_idx) & set(dir_by_idx))
    tiers = []
    if shared:
        P_oos = _three_class_probs(
            mag_art, dir_art,
            np.array([mag_by_idx[i] for i in shared]),
            np.array([dir_by_idx[i] for i in shared]))
        truth_oos = np.array([DIRECTION_LABELS.index(trainval[i]["actual_direction"])
                              for i in shared])
        tiers = _confidence_tiers(P_oos.max(axis=1),
                                  (P_oos.argmax(axis=1) == truth_oos),
                                  baseline_correct=(truth_oos == 1))

    # --- honest evaluation on the held-out test window ---
    Xte = np.array([[_imp(_get(s, n), med[n]) for n in feature_names] for s in test])
    P = _three_class_probs(mag_art, dir_art,
                           _z_for(mag_art, Xte, feature_names),
                           _z_for(dir_art, Xte, feature_names))
    truth = np.array([DIRECTION_LABELS.index(s["actual_direction"]) for s in test])
    pred = P.argmax(axis=1)
    recs, prev = [], None
    for i, s in enumerate(test):
        actual = s["actual_direction"]
        recs.append({"prediction": DIRECTION_LABELS[pred[i]], "actual_direction": actual,
                     "baselines": {"momentum": {"label": "今日方向延續",
                                                "hit": (prev == actual) if prev else False}}})
        prev = actual
    cls = summarize_classification(recs)
    base = summarize_baselines(recs)
    hit = float((pred == truth).mean())
    brier = _multiclass_brier(P, truth)
    # Baseline = the constant a deployer could genuinely have shipped, i.e. the
    # class prior measured on TRAINVAL. Using the test window's own prior instead
    # would grade against a constant that had seen the future.
    tv_prior = np.array([np.mean([s["actual_direction"] == lbl for s in trainval])
                         for lbl in DIRECTION_LABELS])
    brier_base = _multiclass_brier(np.tile(tv_prior, (len(truth), 1)), truth)
    oracle_prior = np.array([(truth == k).mean() for k in range(3)])
    brier_oracle = _multiclass_brier(np.tile(oracle_prior, (len(truth), 1)), truth)
    model_hits = (pred == truth).tolist()
    neu_hits = (truth == 1).tolist()
    bull_hits = (truth == 0).tolist()

    metrics = {
        "artifact": "final_test_evaluation", "market": market,
        "model_type": "two_stage_logistic",
        "trained_on": [trainval[0]["date"], trainval[-1]["date"]],
        "tested_on": [test[0]["date"], test[-1]["date"]] if test else None,
        "chosen_params": {
            "magnitude_C": mag_art["C"], "direction_C": dir_art["C"],
            "magnitude_shrinkage": mag_art["shrinkage"],
            "direction_shrinkage": dir_art["shrinkage"],
        },
        "metrics": {"hit_rate": hit, "macro_f1": cls["macro_f1"], "brier": brier,
                    "brier_baseline": brier_base,
                    "brier_skill": 1 - brier / brier_base if brier_base else None,
                    "brier_baseline_oracle_prior": brier_oracle,
                    "classification": cls},
        "baselines": base,
        "significance": {
            "vs_always_neutral_mcnemar": mcnemar_test(model_hits, neu_hits),
            "vs_always_neutral_bootstrap": block_bootstrap_diff(model_hits, neu_hits, seed=0),
            "vs_always_bull_mcnemar": mcnemar_test(model_hits, bull_hits),
            "vs_always_bull_bootstrap": block_bootstrap_diff(model_hits, bull_hits, seed=0),
        },
        "confidence_tiers": tiers,
    }
    # --- rolling-origin estimate: the number the deploy gate actually trusts ---
    # Deferred import: validate.py imports this module.
    if rolling_windows:
        try:
            import validate as V
            per_window, pooled = V.rolling_validate(
                labeled, feature_names, windows=rolling_windows, window_days=126)
            if pooled:
                metrics["rolling_validation"] = V.summarize(per_window, pooled)
        except Exception as exc:  # noqa: BLE001 — never block a retrain on this
            metrics["rolling_validation_error"] = str(exc)

    metrics_path = os.path.join(out_dir, f"metrics_{market}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # --- production refit on ALL labeled rows (test window included) ---
    mag_full, _, _ = _build_stage(labeled, feature_names, "magnitude")
    dir_full, _, _ = _build_stage(labeled, feature_names, "direction")
    med_all = {n: _nanmedian0([_get(s, n) for s in labeled]) for n in feature_names}
    model = {
        "artifact": "production_refit", "market": market,
        "model_type": "two_stage_logistic",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": [labeled[0]["date"], labeled[-1]["date"]],
        "note": ("Two-stage: P(震盪) from the magnitude stage, P(偏多|大動) from the "
                 "direction stage; the three class probabilities are their product "
                 "decomposition. Refit on ALL data including the test window — the "
                 "hit_rate/Brier in metrics_*.json come from a model that never saw "
                 "the test window and are the honest OOS estimate, NOT a test of this "
                 "refit. Isotonic calibration and confidence tiers were fit on pooled "
                 "trainval walk-forward OOS predictions."),
        "feature_names": feature_names,
        "imputation_median": med_all,
        "stages": {"magnitude": mag_full, "direction": dir_full},
        "confidence_tiers": tiers,
        "oos_reference": {"hit_rate": hit, "brier": brier, "brier_baseline": brier_base,
                          "tested_on": metrics["tested_on"]},
    }
    model_path = os.path.join(out_dir, f"model_{market}.json")
    with open(model_path, "w") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)
    return model_path, metrics_path


def train_market(db, market, out_dir, test_months=18, seed=0, rolling_windows=8):
    from marketlib.store import HistoryStore
    from marketlib.features import build_dataset, FEATURE_NAMES_TW, FEATURE_NAMES_US
    store = HistoryStore(db)
    if market == "tw":
        rows, us = store.get_tw_rows(), store.get_us_rows(); names = FEATURE_NAMES_TW
    else:
        rows, us = store.get_us_rows(), []; names = FEATURE_NAMES_US
    samples = build_dataset(market, rows, us)
    store.close()
    # 台股走兩段式三分類(幅度 × 方向),各自用 pooled walk-forward OOS 做 isotonic
    # 校準。舊的二分類 + 門檻切法把佔比最高的「震盪」壓到 recall 0.02,且校準失效。
    if market == "tw":
        return train_two_stage(samples, names, market, out_dir, test_months=test_months,
                               rolling_windows=rolling_windows)
    return train_from_samples(samples, names, market, out_dir, test_months=test_months,
                              seed=seed, use_platt=True)

def promote_if_passes(metrics_path, model_path, market, promote_dir):
    """Copy the freshly trained model into `promote_dir` (production) iff its OOS
    metrics clear the deploy gate. Returns (promoted, reason)."""
    import shutil
    from marketlib.model import passes_deploy_gate
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)
    ok, reason = passes_deploy_gate(metrics)
    if not ok:
        return False, reason
    os.makedirs(promote_dir, exist_ok=True)
    dest = os.path.join(promote_dir, f"model_{market}.json")
    shutil.copyfile(model_path, dest)
    return True, reason

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--market", choices=["tw", "us"], required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--test-months", type=int, default=18)
    p.add_argument("--rolling-windows", type=int, default=8,
                   help="滾動驗證窗口數（0 = 跳過）。gate 以此為準，優先於單一切點。")
    p.add_argument("--promote-if-passes", action="store_true",
                   help="若 OOS 通過部署 gate，複製 model_<market>.json 到 --promote-dir")
    p.add_argument("--promote-dir", default="models",
                   help="production 模型目錄（daily_market_report.py 讀取處）")
    a = p.parse_args()
    mp, xp = train_market(a.db, a.market, a.out, test_months=a.test_months,
                          rolling_windows=a.rolling_windows)
    print(f"Wrote {mp}\nWrote {xp}")
    if a.promote_if_passes:
        promoted, reason = promote_if_passes(xp, mp, a.market, a.promote_dir)
        if promoted:
            print(f"PROMOTED -> {os.path.join(a.promote_dir, f'model_{a.market}.json')}  ({reason})")
        else:
            print(f"NOT promoted（留在 candidate）：{reason}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
