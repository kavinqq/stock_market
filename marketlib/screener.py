"""
Cross-sectional screener: ranks a set of stocks against each other on
multiple facets, then combines facet scores into a weighted composite.

All ranking is cross-sectional — each stock is ranked *relative to the
other stocks in the current call*, not against its own history.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

SCREENER_WEIGHTS: dict[str, float] = {
    "valuation": 0.30,
    "revenue":   0.25,
    "technical": 0.20,
    "momentum":  0.15,
    "chip":      0.10,
}


# ---------------------------------------------------------------------------
# Core ranking primitive
# ---------------------------------------------------------------------------

def cross_sectional_rank(
    values: list[Optional[float]],
    higher_is_better: bool = True,
) -> list[Optional[float]]:
    """Return percentile ranks in [0, 1] for each element of *values*.

    Rules
    -----
    * ``None`` inputs produce ``None`` outputs (they are excluded from ranking).
    * Among the non-None values:
      - The *best* value (per direction) receives 1.0.
      - The *worst* receives 0.0.
      - With n == 1 the single value receives 0.5.
      - Ties share the **average** of the percentile positions they would
        otherwise occupy (standard "average rank" / "fractional rank").
    * ``higher_is_better=True``  → larger numeric value = better score.
    * ``higher_is_better=False`` → smaller numeric value = better score
      (e.g. lower P/E = cheaper = better).
    """
    n = len(values)
    if n == 0:
        return []

    # Separate non-None entries (keep original indices so we can scatter back)
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    m = len(indexed)

    result: list[Optional[float]] = [None] * n

    if m == 0:
        return result

    if m == 1:
        result[indexed[0][0]] = 0.5
        return result

    # Sort by value; for higher_is_better=False invert the sort key so that
    # the lowest value ends up last (= best position) after the generic logic.
    # We use a single convention: after sorting ascending, position 0 is
    # "worst ascending" and position m-1 is "best ascending" for higher_is_better.
    sorted_indexed = sorted(indexed, key=lambda t: t[1])  # ascending

    # Assign raw positions (0-based) then average ties.
    # For higher_is_better=True: position 0 = worst (percentile 0), m-1 = best (percentile 1).
    # For higher_is_better=False: invert — position 0 = best (percentile 1), m-1 = worst (percentile 0).

    # Build value → list-of-positions mapping to detect ties.
    from collections import defaultdict
    pos_by_value: dict[float, list[int]] = defaultdict(list)
    for pos, (_, v) in enumerate(sorted_indexed):
        pos_by_value[v].append(pos)

    # Map original index → averaged percentile
    orig_idx_to_pct: dict[int, float] = {}
    for v, positions in pos_by_value.items():
        avg_pos = sum(positions) / len(positions)
        if higher_is_better:
            pct = avg_pos / (m - 1)
        else:
            # Invert: position 0 (lowest value) → 1.0, position m-1 → 0.0
            pct = 1.0 - avg_pos / (m - 1)
        for pos in positions:
            orig_idx = sorted_indexed[pos][0]
            orig_idx_to_pct[orig_idx] = pct

    for orig_idx, pct in orig_idx_to_pct.items():
        result[orig_idx] = pct

    return result


# ---------------------------------------------------------------------------
# Screener builder
# ---------------------------------------------------------------------------

# Mapping of facet name → (raw field, higher_is_better)
_FACET_CONFIG: dict[str, tuple[str, bool]] = {
    "valuation": ("pe",           False),  # lower P/E = better
    "revenue":   ("rev_yoy",      True),
    "momentum":  ("ret20",        True),
    "chip":      ("chip_net_lots", True),
    "technical": ("tech_raw",     True),
}

_RAW_FIELDS = ("pe", "pb", "dividend_yield", "rev_yoy", "ret20", "chip_net_lots", "tech_raw")


def build_screener(
    records: list[dict],
    weights: dict[str, float] = SCREENER_WEIGHTS,
) -> list[dict]:
    """Score and rank a list of stock records cross-sectionally.

    Parameters
    ----------
    records:
        Each dict must have ``code`` plus the raw facet fields
        ``pe, pb, dividend_yield, rev_yoy, ret20, ret60,
        chip_net_lots, tech_raw``.  Any value may be ``None``.
    weights:
        Facet weights (will be renormalized per-stock over present facets).

    Returns
    -------
    List of result dicts sorted by ``composite`` DESC (``None`` composite
    last), each containing::

        {
            "code":      str,
            "composite": float | None,
            "rank":      int,               # 1-based
            "scores":    {facet: float | None, ...},
            "raw":       {field: value, ...},
        }
    """
    if not records:
        return []

    # --- 1. Compute cross-sectional facet scores ---
    facet_scores: dict[str, list[Optional[float]]] = {}
    for facet, (field, higher) in _FACET_CONFIG.items():
        raw_values = [r.get(field) for r in records]
        facet_scores[facet] = cross_sectional_rank(raw_values, higher_is_better=higher)

    # --- 2. Build per-record results with composite ---
    results = []
    for i, rec in enumerate(records):
        scores_for_rec: dict[str, Optional[float]] = {
            facet: facet_scores[facet][i]
            for facet in _FACET_CONFIG
        }

        # Renormalize weights over present (non-None) facets
        present = {f: s for f, s in scores_for_rec.items() if s is not None}
        if present:
            total_w = sum(weights.get(f, 0.0) for f in present)
            if total_w > 0:
                composite = sum(
                    weights.get(f, 0.0) * s for f, s in present.items()
                ) / total_w
            else:
                composite = None
        else:
            composite = None

        raw = {field: rec.get(field) for field in _RAW_FIELDS}

        results.append(
            {
                "code":      rec["code"],
                "composite": composite,
                "rank":      None,  # filled in after sorting
                "scores":    scores_for_rec,
                "raw":       raw,
            }
        )

    # --- 3. Sort: non-None composite DESC, then None composites last ---
    results.sort(
        key=lambda r: (r["composite"] is None, -(r["composite"] or 0.0))
    )

    # --- 4. Assign 1-based ranks ---
    for rank_idx, res in enumerate(results, start=1):
        res["rank"] = rank_idx

    return results


# ---------------------------------------------------------------------------
# Technical intersection — 0050 潛力股
#
# The composite score above ranks every stock against every other, so something
# always comes out on top even when nothing looks good. This layer answers a
# different question: does this stock satisfy ALL of a fixed set of technical
# conditions at once? An empty result is a legitimate — and informative —
# answer, which a ranking can never give.
#
# Three kinds of condition, deliberately separated:
#   core   — the intersection proper. Every one must hold.
#   veto   — disqualifies regardless of the rest (already overheated).
#   bonus  — reported as supporting evidence, never gates.
# ---------------------------------------------------------------------------

RSI_OVERHEATED = 80.0
VOLUME_EXPANSION = 1.0
KD_OVERBOUGHT = 80.0
MA20_GAP_STRETCHED = 0.15


def _macd_bullish(macd: Optional[dict]) -> Optional[bool]:
    if not macd:
        return None
    hist = macd.get("hist")
    if macd.get("cross") == "golden":
        return True
    if macd.get("cross") == "dead":
        return False
    return None if hist is None else hist > 0


def _core_conditions(m: dict) -> list[tuple[str, Optional[bool]]]:
    """(readable condition, holds?) — None where the data cannot answer it."""
    close, ma5, ma20, ma60 = m.get("close"), m.get("ma5"), m.get("ma20"), m.get("ma60")
    kd = m.get("kd") or {}
    k, d = kd.get("k"), kd.get("d")
    return [
        ("收盤站上月線（20日均）",
         None if close is None or ma20 is None else close >= ma20),
        ("均線多頭排列（5日 ≥ 20日 ≥ 60日）",
         None if None in (ma5, ma20, ma60) else ma5 >= ma20 >= ma60),
        ("日 MACD 翻多（柱狀轉正或黃金交叉）", _macd_bullish(m.get("macd_daily"))),
        ("週 MACD 翻多（中期趨勢同向）", _macd_bullish(m.get("macd_weekly"))),
        ("KD 金叉（K ≥ D）", None if k is None or d is None else k >= d),
    ]


def _veto_conditions(m: dict) -> list[tuple[str, bool]]:
    rsi_val = m.get("rsi")
    out = []
    if rsi_val is not None and rsi_val >= RSI_OVERHEATED:
        out.append((f"RSI {rsi_val:.0f} ≥ {RSI_OVERHEATED:.0f}，短線過熱", True))
    return out


def _bonus_conditions(m: dict) -> list[tuple[str, Optional[bool]]]:
    close = m.get("close")
    boll = m.get("boll") or {}
    pctb = boll.get("pctb")
    cost60 = m.get("cost_vwap60")
    vol = m.get("volume_ratio_20d")
    return [
        ("布林通道位置偏上（%b > 0.5）", None if pctb is None else pctb > 0.5),
        ("站上 60 日成本均價（大戶成本之上）",
         None if close is None or cost60 is None else close >= cost60),
        ("量能高於 20 日均量（買盤有跟上）",
         None if vol is None else vol > VOLUME_EXPANSION),
    ]


def core_condition_texts() -> list[str]:
    """The core conditions, in order, independent of any stock.

    Every stock that passes satisfies all of them, so a report states them once
    rather than repeating the identical list under each name.
    """
    return [text for text, _ in _core_conditions({})]


def _metrics_snapshot(m: dict) -> dict:
    """The numbers behind the verdict. Passing stocks are indistinguishable on
    the conditions themselves — these values are what lets a reader rank them."""
    close, ma20 = m.get("close"), m.get("ma20")
    kd = m.get("kd") or {}
    macd_d = m.get("macd_daily") or {}
    macd_w = m.get("macd_weekly") or {}
    boll = m.get("boll") or {}
    return {
        "close": close,
        "rsi": m.get("rsi"),
        "kd_k": kd.get("k"),
        "kd_d": kd.get("d"),
        "macd_hist": macd_d.get("hist"),
        "macd_weekly_hist": macd_w.get("hist"),
        "pctb": boll.get("pctb"),
        "volume_ratio_20d": m.get("volume_ratio_20d"),
        "close_vs_ma20": (close / ma20 - 1) if close is not None and ma20 else None,
        "price_vs_cost60": m.get("price_vs_cost60"),
    }


def _warnings(m: dict) -> list[str]:
    """Stretch signals: true but worth knowing, and never disqualifying.

    A condition can be satisfied in a way that argues against buying. %b > 1
    means the close is outside the upper band, not merely in the upper half of
    it; a K above 80 is a golden cross made in overbought territory. Left
    unflagged, the shortlist silently ranks the most extended names highest,
    which is the opposite of what a screen is for. These stay warnings rather
    than vetoes because whether an extended trend is an entry or an exit is a
    judgement call, not something the data settles.
    """
    out: list[str] = []
    boll = m.get("boll") or {}
    pctb = boll.get("pctb")
    if pctb is not None and pctb > 1.0:
        out.append(f"收盤在布林上軌之外（%b {pctb:.2f}），短線延伸過度")
    k = (m.get("kd") or {}).get("k")
    if k is not None and k >= KD_OVERBOUGHT:
        out.append(f"K 值 {k:.0f} 已在超買區（≥ {KD_OVERBOUGHT:.0f}）")
    close, ma20 = m.get("close"), m.get("ma20")
    if close is not None and ma20:
        gap = close / ma20 - 1
        if gap > MA20_GAP_STRETCHED:
            out.append(f"距 20 日均線 +{gap:.0%}，正乖離偏大")
    return out


def technical_intersection(metrics: dict) -> dict:
    """Evaluate one stock against the fixed technical intersection.

    `metrics` is the dict from ``marketlib.stocks.technical_metrics``.

    A condition that cannot be computed counts as NOT met and sets
    ``insufficient_data``. Treating unknown as a pass would quietly promote
    exactly the newly-listed or thinly-traded stocks we have least evidence
    about, which is the opposite of what an intersection is for.

    Returns a dict carrying, for every condition, whether it held — so the
    report can show the reasoning rather than just a verdict.
    """
    core = _core_conditions(metrics)
    vetoes = _veto_conditions(metrics)
    bonus = _bonus_conditions(metrics)

    core_passed = [text for text, ok in core if ok is True]
    core_failed = [text for text, ok in core if ok is False]
    unknown = [text for text, ok in core if ok is None]
    core_failed += [f"{text}（資料不足，無法判定）" for text in unknown]

    bonus_passed = [text for text, ok in bonus if ok is True]
    bonus_failed = [text for text, ok in bonus if ok is False]
    bonus_unknown = [text for text, ok in bonus if ok is None]

    veto_texts = [text for text, hit in vetoes if hit]
    is_candidate = not core_failed and not veto_texts

    total = len(core) + len(bonus)
    met = len(core_passed) + len(bonus_passed)

    return {
        "is_candidate": is_candidate,
        "reasons": core_passed + bonus_passed,
        "core_passed": len(core_passed),
        "core_total": len(core),
        "core_failed": core_failed,
        "bonus_passed": len(bonus_passed),
        "bonus_total": len(bonus),
        "bonus_failed": bonus_failed + [f"{t}（資料不足）" for t in bonus_unknown],
        "vetoes": veto_texts,
        "insufficient_data": bool(unknown),
        "score": met / total if total else 0.0,
        "metrics_snapshot": _metrics_snapshot(metrics),
        "warnings": _warnings(metrics),
    }
