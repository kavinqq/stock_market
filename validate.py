#!/usr/bin/env python3
"""Rolling-origin validation: how much should we trust the TW model?

`train.py` reports one held-out window. That single number moves a lot with
where the split happens to fall, so it is not enough to decide whether an edge
is real. This walks the split forward instead — retraining before each window
and scoring the days after it — and pools the results, which is the estimate
worth quoting.

Every comparison here is against the constant guesses a one-line script could
ship (always 震盪 / always 偏多), because those are what the model has to beat
to be worth running at all.

    python validate.py --market tw --windows 8 --window-days 126
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

import train as T
from marketlib.evaluation import summarize_classification
from marketlib.stats import block_bootstrap_diff, mcnemar_test

L = T.DIRECTION_LABELS


def _load(market: str, db: str):
    from marketlib.features import (FEATURE_NAMES_TW, FEATURE_NAMES_US,
                                    build_dataset)
    from marketlib.store import HistoryStore

    store = HistoryStore(db)
    if market == "tw":
        rows, us, names = store.get_tw_rows(), store.get_us_rows(), FEATURE_NAMES_TW
    else:
        rows, us, names = store.get_us_rows(), [], FEATURE_NAMES_US
    samples = build_dataset(market, rows, us)
    store.close()
    return [s for s in samples if s.get("actual_direction") in L], names


def rolling_validate(samples, names, windows=6, window_days=126, min_train=500):
    """Retrain before each window; score the days inside it. Nothing in a
    window's training set comes from that window or the EMBARGO days before it."""
    total = len(samples)
    starts = [total - k * window_days for k in range(windows, 0, -1)]
    starts = [s for s in starts if s - T.EMBARGO >= min_train]
    per_window, pooled = [], []

    for start in starts:
        train = samples[: start - T.EMBARGO]
        test = samples[start: start + window_days]
        if len(test) < 30:
            continue
        mag, _, _ = T._build_stage(train, names, "magnitude")
        dirn, _, _ = T._build_stage(train, names, "direction")
        med = {n: T._nanmedian0([T._get(s, n) for s in train]) for n in names}
        X = np.array([[T._imp(T._get(s, n), med[n]) for n in names] for s in test])
        probs = T._three_class_probs(mag, dirn,
                                     T._z_for(mag, X, names), T._z_for(dirn, X, names))
        truth = np.array([L.index(s["actual_direction"]) for s in test])
        pred = probs.argmax(axis=1)

        per_window.append({
            "from": test[0]["date"], "to": test[-1]["date"], "n": len(test),
            "hit_rate": float((pred == truth).mean()),
            "always_neutral": float((truth == 1).mean()),
            "always_bull": float((truth == 0).mean()),
            "brier": T._multiclass_brier(probs, truth),
        })
        for i, s in enumerate(test):
            pooled.append({"date": s["date"], "pred": L[pred[i]],
                           "actual": s["actual_direction"],
                           "top_prob": float(probs[i].max()),
                           "probs": {lbl: float(probs[i][j]) for j, lbl in enumerate(L)}})
    return per_window, pooled


def summarize(per_window, pooled, train_prior=None):
    model = [r["pred"] == r["actual"] for r in pooled]
    neutral = [r["actual"] == "震盪" for r in pooled]
    bull = [r["actual"] == "偏多" for r in pooled]
    bear = [r["actual"] == "偏空" for r in pooled]
    cls = summarize_classification([{"prediction": r["pred"],
                                     "actual_direction": r["actual"]} for r in pooled])
    P = np.array([[r["probs"][lbl] for lbl in L] for r in pooled])
    truth = np.array([L.index(r["actual"]) for r in pooled])
    # With no train_prior supplied this falls back to the evaluation period's own
    # class frequencies — an oracle constant that saw the future. That is a
    # STRICTER bar than a deployable constant, never a easier one, and the flag
    # below records which of the two was used.
    prior = (np.array(train_prior) if train_prior is not None
             else np.array([(truth == k).mean() for k in range(3)]))
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pooled_sample_size": len(pooled),
        "windows_won": sum(1 for w in per_window
                           if w["hit_rate"] > max(w["always_neutral"], w["always_bull"])),
        "windows_total": len(per_window),
        "hit_rate": float(np.mean(model)),
        "macro_f1": cls["macro_f1"],
        "brier": T._multiclass_brier(P, truth),
        "brier_constant_prior": T._multiclass_brier(np.tile(prior, (len(truth), 1)), truth),
        "brier_constant_prior_is_oracle": train_prior is None,
        # All three constants: the gate takes the max, so omitting always_bear
        # would let a model through in a bearish period by measuring it against
        # a weaker competitor than the one available.
        "baselines": {"always_neutral": float(np.mean(neutral)),
                      "always_bull": float(np.mean(bull)),
                      "always_bear": float(np.mean(bear))},
        "significance": {
            "vs_always_neutral_mcnemar": mcnemar_test(model, neutral),
            "vs_always_neutral_bootstrap": block_bootstrap_diff(model, neutral, seed=0),
        },
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"],
                          "predicted_count": v["predicted_count"], "support": v["support"]}
                      for k, v in cls["per_class"].items()},
        "per_window": per_window,
    }
    # Selective accuracy: the model is allowed to stay silent. This is the
    # number that matters for decision support — a call made on 20% of days at
    # 55% is more useful than a call made every day at 45%.
    tiers = []
    for q in (0, 50, 75, 85, 90):
        cutoff = float(np.percentile([r["top_prob"] for r in pooled], q))
        sel = [r for r in pooled if r["top_prob"] >= cutoff]
        if len(sel) < 30:
            continue
        tiers.append({
            "top_prob_cutoff": cutoff,
            "coverage": len(sel) / len(pooled),
            "n": len(sel),
            "hit_rate": float(np.mean([r["pred"] == r["actual"] for r in sel])),
            "always_neutral_same_days": float(np.mean([r["actual"] == "震盪" for r in sel])),
        })
    out["selective"] = tiers
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/history.db")
    ap.add_argument("--market", choices=["tw", "us"], default="tw")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-days", type=int, default=126)
    ap.add_argument("--out", default="data/validation_tw.json")
    a = ap.parse_args()

    samples, names = _load(a.market, a.db)
    per_window, pooled = rolling_validate(samples, names, a.windows, a.window_days)
    if not pooled:
        print("not enough history to validate")
        return 1
    report = summarize(per_window, pooled)

    print(f"=== rolling-origin validation ({a.market}) — {report['windows_total']} windows, "
          f"{report['pooled_sample_size']} days ===\n")
    print(f"{'window':<26}{'n':>5}{'model':>9}{'always震盪':>12}{'always偏多':>12}   verdict")
    for w in per_window:
        best = max(w["always_neutral"], w["always_bull"])
        print(f"{w['from']}..{w['to']}{w['n']:>5}{w['hit_rate']:>9.4f}"
              f"{w['always_neutral']:>12.4f}{w['always_bull']:>12.4f}   "
              f"{'WIN' if w['hit_rate'] > best else 'lose'}")
    print(f"\nPOOLED  hit={report['hit_rate']:.4f}  "
          f"always震盪={report['baselines']['always_neutral']:.4f}  "
          f"always偏多={report['baselines']['always_bull']:.4f}  "
          f"(won {report['windows_won']}/{report['windows_total']} windows)")
    print(f"        macro_f1={report['macro_f1']:.4f}  brier={report['brier']:.4f} "
          f"vs constant {report['brier_constant_prior']:.4f}")
    sig = report["significance"]
    print(f"\nvs always-震盪: McNemar p={sig['vs_always_neutral_mcnemar']['p_value']:.4f}   "
          f"bootstrap diff={sig['vs_always_neutral_bootstrap']['mean_diff']:+.4f} "
          f"CI[{sig['vs_always_neutral_bootstrap']['ci_low']:+.4f}, "
          f"{sig['vs_always_neutral_bootstrap']['ci_high']:+.4f}]")
    print("\n--- selective accuracy (model may stay silent) ---")
    print(f"{'cutoff':>8}{'coverage':>11}{'n':>6}{'hit':>9}{'always震盪(同期)':>18}")
    for t in report["selective"]:
        print(f"{t['top_prob_cutoff']:>8.4f}{t['coverage']:>10.0%}{t['n']:>6}"
              f"{t['hit_rate']:>9.4f}{t['always_neutral_same_days']:>16.4f}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
