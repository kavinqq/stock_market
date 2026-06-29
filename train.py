#!/usr/bin/env python3
"""Offline walk-forward training. Produces final-test metrics + production refit model."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
import numpy as np
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
    med = {n: float(np.nanmedian([_get(s, n) for s in samples])) for n in feature_names}
    X, y = [], []
    for s in samples:
        X.append([_imp(_get(s, n), med[n]) for n in feature_names])
        y.append(s["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int), med

def _get(s, n):
    v = s["features"].get(n)
    return float("nan") if v is None else float(v)

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
    best, best_f1 = (0.55, 0.45), -1.0
    for bull in (0.52, 0.55, 0.58, 0.6):
        for bear in (0.4, 0.42, 0.45, 0.48):
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

def train_from_samples(samples, feature_names, market, out_dir, test_frac=None,
                       test_months=18, seed=0):
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
    if best["params"]["class_weight"] == "balanced" and len(set(y_oos.tolist())) >= 2:
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

def train_market(db, market, out_dir, test_months=18, seed=0):
    from marketlib.store import HistoryStore
    from marketlib.features import build_dataset, FEATURE_NAMES_TW, FEATURE_NAMES_US
    store = HistoryStore(db)
    if market == "tw":
        rows, us = store.get_tw_rows(), store.get_us_rows(); names = FEATURE_NAMES_TW
    else:
        rows, us = store.get_us_rows(), []; names = FEATURE_NAMES_US
    samples = build_dataset(market, rows, us)
    store.close()
    return train_from_samples(samples, names, market, out_dir, test_months=test_months, seed=seed)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--market", choices=["tw", "us"], required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--test-months", type=int, default=18)
    a = p.parse_args()
    mp, xp = train_market(a.db, a.market, a.out, test_months=a.test_months)
    print(f"Wrote {mp}\nWrote {xp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
