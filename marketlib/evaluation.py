from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Optional

from marketlib.core import MarketBar

DIRECTION_LABELS: list[str] = ["偏多", "震盪", "偏空"]

DEFAULT_NEUTRAL_THRESHOLD: float = 0.005
MIN_NEUTRAL_THRESHOLD: float = 0.003
MAX_NEUTRAL_THRESHOLD: float = 0.009
NEUTRAL_VOL_MULTIPLIER: float = 0.5


def bar_close(bar: Any) -> Optional[float]:
    if isinstance(bar, MarketBar):
        return bar.close
    if isinstance(bar, dict):
        return bar.get("close")
    return None


def close_return_series(bars: list[Any]) -> list[float]:
    returns: list[float] = []
    for prev, current in zip(bars, bars[1:]):
        prev_close = bar_close(prev)
        current_close = bar_close(current)
        if prev_close in {None, 0} or current_close is None:
            continue
        returns.append(current_close / prev_close - 1)
    return returns


def neutral_threshold_from_bars(bars: list[Any], lookback: int = 60) -> float:
    returns = close_return_series(bars)[-lookback:]
    if len(returns) < 20:
        return DEFAULT_NEUTRAL_THRESHOLD
    threshold = pstdev(returns) * NEUTRAL_VOL_MULTIPLIER
    threshold = max(MIN_NEUTRAL_THRESHOLD, min(MAX_NEUTRAL_THRESHOLD, threshold))
    return round(threshold, 4)


def direction_from_return(ret: Optional[float], threshold: Optional[float] = None) -> str:
    threshold = DEFAULT_NEUTRAL_THRESHOLD if threshold is None else threshold
    if ret is None:
        return "未知"
    if ret >= threshold:
        return "偏多"
    if ret <= -threshold:
        return "偏空"
    return "震盪"


def summarize_classification(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {
        actual: {prediction: 0 for prediction in DIRECTION_LABELS}
        for actual in DIRECTION_LABELS
    }
    actual_counts = {label: 0 for label in DIRECTION_LABELS}
    prediction_counts = {label: 0 for label in DIRECTION_LABELS}
    for row in records:
        actual = row.get("actual_direction")
        prediction = row.get("prediction")
        if actual not in DIRECTION_LABELS or prediction not in DIRECTION_LABELS:
            continue
        matrix[actual][prediction] += 1
        actual_counts[actual] += 1
        prediction_counts[prediction] += 1

    per_class: dict[str, dict[str, Optional[float]]] = {}
    f1_scores: list[float] = []
    for label in DIRECTION_LABELS:
        tp = matrix[label][label]
        predicted = prediction_counts[label]
        actual = actual_counts[label]
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        # A class counts toward macro-F1 whenever it is "in play" — it either
        # really occurs (support > 0) or the model claims it (predicted > 0).
        # Such a class with no true positives scores 0, it is NOT dropped.
        # Dropping it used to let a degenerate always-震盪 predictor post a
        # higher macro-F1 than a genuine three-class model, so the deploy gate
        # actively rewarded collapse. Only a class that neither occurs nor is
        # ever predicted is excluded (it carries no information either way).
        if actual or predicted:
            f1_scores.append(f1)
        else:
            f1 = None
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
            "predicted_count": predicted,
        }

    total = sum(actual_counts.values())
    actual_distribution = {
        label: (actual_counts[label] / total if total else None)
        for label in DIRECTION_LABELS
    }
    return {
        "macro_f1": mean(f1_scores) if f1_scores else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "actual_counts": actual_counts,
        "prediction_counts": prediction_counts,
        "actual_distribution": actual_distribution,
    }


def summarize_baselines(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    baseline_keys = sorted(
        {
            key
            for row in records
            for key in (row.get("baselines") or {}).keys()
        }
    )
    baselines: dict[str, dict[str, Any]] = {}
    for key in baseline_keys:
        rows = [row for row in records if key in (row.get("baselines") or {})]
        hits = sum(1 for row in rows if (row.get("baselines") or {}).get(key, {}).get("hit"))
        sample_size = len(rows)
        label = (
            (rows[0].get("baselines") or {}).get(key, {}).get("label", key)
            if rows
            else key
        )
        baselines[key] = {
            "label": label,
            "sample_size": sample_size,
            "hit_count": hits,
            "hit_rate": hits / sample_size if sample_size else None,
        }

    actual_counts = {label: 0 for label in DIRECTION_LABELS}
    for row in records:
        actual = row.get("actual_direction")
        if actual in actual_counts:
            actual_counts[actual] += 1

    # Constant-guess baselines, derived straight from the outcomes so they are
    # always present even when the caller injected no per-row baselines. In TW
    # 震盪 is the plurality class (~44%), which makes always_neutral — not
    # always_bull and not random_prior — the bar a model must actually clear.
    scored = sum(actual_counts.values())
    for key, label_zh, cls in (
        ("always_bull", "永遠偏多", "偏多"),
        ("always_neutral", "永遠震盪", "震盪"),
        ("always_bear", "永遠偏空", "偏空"),
    ):
        baselines[key] = {
            "label": label_zh,
            "sample_size": scored,
            "hit_count": actual_counts[cls],
            "hit_rate": actual_counts[cls] / scored if scored else None,
        }

    if total:
        random_prior_rate = sum((count / total) ** 2 for count in actual_counts.values())
    else:
        random_prior_rate = None
    baselines["random_prior"] = {
        "label": "依實際先驗隨機",
        "sample_size": total,
        "expected_hit_rate": random_prior_rate,
    }
    return baselines
