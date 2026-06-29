#!/usr/bin/env python3
"""Offline walk-forward training. Produces final-test metrics + production refit model."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from marketlib.evaluation import summarize_classification, summarize_baselines
from marketlib.stats import mcnemar_test, block_bootstrap_diff

MINIMAL_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "close_vs_ma20", "ma20_vs_ma60",
    "vol_ratio_20d", "realized_vol_20d",
]
PARAM_GRID = [
    {"C": c, "class_weight": cw}
    for c in (0.1, 0.3, 1.0, 3.0) for cw in (None, "balanced")
]

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

def walk_forward_select(samples, feature_names, n_folds=4, seed=0):
    n = len(samples); fold = n // (n_folds + 1)
    scored = []
    for params in PARAM_GRID:
        f1s, briers = [], []
        for k in range(1, n_folds + 1):
            tr = samples[: fold * k]; va = samples[fold * k: fold * (k + 1)]
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

def evaluate_on_test(clf, mean, std, med, test_samples, feature_names, thresholds):
    X = np.array([[_imp(_get(s, n), med[n]) for n in feature_names] for s in test_samples])
    y = np.array([s["label"] for s in test_samples])
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
    trainval, test = labeled[:cut], labeled[cut:]
    best = walk_forward_select(trainval, feature_names, seed=seed)
    # final-test 模型:只用 trainval 擬合
    Xtv, ytv, med_tv = _matrix(trainval, feature_names)
    clf, mean, std = _fit(Xtv, ytv, best["params"])
    p_tv = _proba(clf, mean, std, Xtv)
    thresholds = _select_thresholds(p_tv, ytv)
    test_eval = evaluate_on_test(clf, mean, std, med_tv, test, feature_names, thresholds)
    metrics = {"artifact": "final_test_evaluation", "market": market,
               "trained_on": [trainval[0]["date"], trainval[-1]["date"]],
               "tested_on": [test[0]["date"], test[-1]["date"]] if test else None,
               "chosen_params": best["params"], "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
               **test_eval}
    metrics_path = os.path.join(out_dir, f"metrics_{market}.json")
    json.dump(metrics, open(metrics_path, "w"), ensure_ascii=False, indent=2)
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
                 "are the honest OOS estimate, NOT a direct test of this refit model."),
        "feature_names": feature_names,
        "standardization": {"mean": dict(zip(feature_names, mean2.tolist())),
                            "std": dict(zip(feature_names, std2.tolist()))},
        "imputation_median": med_all,
        "coefficients": dict(zip(feature_names, clf2.coef_[0].tolist())),
        "intercept": float(clf2.intercept_[0]),
        "class_weight": best["params"]["class_weight"],
        "calibration": {"method": "none", "params": {}},
        "thresholds": {"bull": thresholds[0], "bear": thresholds[1]},
        "confidence_bands": {"mid": 0.04, "high": 0.10},
    }
    model_path = os.path.join(out_dir, f"model_{market}.json")
    json.dump(model, open(model_path, "w"), ensure_ascii=False, indent=2)
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
