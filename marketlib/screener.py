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
