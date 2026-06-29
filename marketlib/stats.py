from __future__ import annotations
import math, random
from typing import Any

def mcnemar_test(model_hits: list[bool], baseline_hits: list[bool]) -> dict[str, Any]:
    if len(model_hits) != len(baseline_hits):
        raise ValueError("model_hits and baseline_hits must align by sample")
    b = sum(1 for m, x in zip(model_hits, baseline_hits) if m and not x)
    c = sum(1 for m, x in zip(model_hits, baseline_hits) if x and not m)
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected chi-square, df=1
    # df=1 chi-square survival = erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": stat, "p_value": p}

def block_bootstrap_diff(
    model_hits: list[bool], baseline_hits: list[bool],
    block: int = 5, n_resamples: int = 2000, seed: int = 0,
) -> dict[str, float]:
    if len(model_hits) != len(baseline_hits):
        raise ValueError("inputs must align by sample")
    n = len(model_hits)
    diffs = [(1.0 if m else 0.0) - (1.0 if x else 0.0)
             for m, x in zip(model_hits, baseline_hits)]
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    point = sum(diffs) / n
    rng = random.Random(seed)
    n_blocks = math.ceil(n / block)
    means = []
    for _ in range(n_resamples):
        sample = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - 1)
            for k in range(block):
                sample.append(diffs[(start + k) % n])
        sample = sample[:n]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {"mean_diff": round(point, 6), "ci_low": round(lo, 6), "ci_high": round(hi, 6)}
