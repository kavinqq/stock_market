# Data-Quality Adjustments Report

## Adjustment 1 — Misses table shows real 預測/實際 + de-dupe (daily_market_report.py)

**Problem**: `build_improvement_log` appended rows with only `market/date/reason/action`.
The HTML renderer `html_miss_rows` reads `prediction` and `actual_direction`, rendering them as "N/A". Upstream miss records carry those keys. Duplicate entries were also emitted.

**Fix** (lines ~885–920 in `daily_market_report.py`):
- `recent_misses` rows: added `"prediction": miss.get("prediction")` and `"actual_direction": miss.get("actual_direction")`.
- `archive_eval.records` rows: same, from `miss.get("prediction")` / `miss.get("actual_direction")`.
- Added de-duplication: after building the raw list, filter by a `seen` set keyed on `(market, date, prediction, actual_direction, reason)`. Cap of `rows[-12:]` applied after dedup.

**Tests added** (`tests/test_daily.py` — `TestBuildImprovementLog`):
- `test_rows_contain_prediction_and_actual_direction`: rolling-bt miss → row has correct prediction/actual_direction values.
- `test_archive_rows_contain_prediction_and_actual_direction`: archive miss → same.
- `test_duplicate_misses_collapse_to_one_row`: same miss appearing in rolling-bt twice and archive once → collapses to 1 row.
- `test_cap_at_12_after_dedup`: many unique misses from both rolling bt sources → capped at ≤12.

---

## Adjustment 2 — Store-date misdating in build_stock_2330 (daily_market_report.py)

**Problem**: `_pick_query_date` returns `(date_iso, result)` — the actual date for which data was found. However:
- Valuation upsert used `end_iso` (today's date) instead of `val_date`.
- T86 chip data was attached to `daily_bars[-1].date` (always the latest bar) instead of `t86_date` (the bar for which data was actually retrieved).

**Fix** (lines ~2013–2065 in `daily_market_report.py`):
- Replaced `_t86_date` (throwaway) with `t86_date`, defaulting to `end_iso`.
- Replaced `_val_date` (throwaway) with `val_date`, defaulting to `end_iso`.
- Chip upsert: `bar.date == t86_date` (was `daily_bars[-1].date`).
- Valuation upsert: `"date": val_date` (was `end_iso`).

**Unit-test note**: This path requires live network (TWSE). The fix is obviously correct by inspection: variables that were previously discarded (`_t86_date`, `_val_date`) are now kept and used. Existing `TestPickQueryDate` tests already verify `_pick_query_date` returns the correct date. No additional unit test was added for the upsert path (network dependency); correctness is verified by the live run.

---

## Adjustment 3 — Valuation thin-history guard (marketlib/stocks.py)

**Problem**: `valuation_metrics` always computed `pe_pct` using `percentile_rank`, even with only ~2 history points. This produced a −1.00 "偏貴" score that was statistically meaningless. `valuation_facet` then emitted a strong misleading signal.

**Fix** (`marketlib/stocks.py`):
- Added module constant `MIN_VALUATION_HISTORY = 60` (after `MONTHLY_REVENUE_URL`).
- In `valuation_metrics`: `pe_pct` is set only when `len(pe_history) >= MIN_VALUATION_HISTORY`; same for `pb_pct`. Raw `pe`, `pb`, `dividend_yield` are still passed through.
- In `valuation_facet`: when `pe_pct is None and pb_pct is None`, distinguishes "no raw data" (returns `_MISSING_FACET`) from "has raw PE but thin history" (returns `{"score": None, "reason": "估值歷史不足(需累積更多PE)", "metrics": {pe/pb/dividend_yield}}`).
- Updated existing `test_valuation_pct` test: now uses 60 history points (12 × [10,15,20,25,30]) to satisfy the guard — still asserts `pe_pct ≈ 0.4`.

**New tests** (`tests/test_stocks.py` — `TestScorecard`):
- `test_valuation_metrics_short_history_yields_none_pct`: 59-point history → `pe_pct is None`.
- `test_valuation_metrics_long_history_yields_pct`: 60-point history → `pe_pct` computed.
- `test_valuation_facet_thin_history_score_none_metrics_present`: `pe_pct=None, pb_pct=None` with raw pe/pb → `score=None`, `reason` contains "估值歷史不足", `metrics` has `pe` and `pb`.
- `test_valuation_facet_cheap_is_bullish` still passes (uses explicit `pe_pct=0.1`).

---

## Test Summary

```
Ran 41 tests in 0.006s — OK
```

7 new tests added (4 in test_daily.py, 3 in test_stocks.py) + 1 existing test updated.

---

## Live-Run Output

```
miss row sample: {'market': '台股', 'date': '2026-07-01', 'prediction': '震盪', 'actual_direction': '偏多'}
valuation facet: None 估值歷史不足(需累積更多PE)
```

- Miss row: real `prediction` ("震盪") and `actual_direction` ("偏多") now present (not None).
- Valuation facet: `score=None` with reason "估值歷史不足(需累積更多PE)" — history is still accumulating; bogus score correctly suppressed.

---

## Files Changed

- `/Users/kuanlin/stock_market/daily_market_report.py` — Adjustments 1 and 2
- `/Users/kuanlin/stock_market/marketlib/stocks.py` — Adjustment 3
- `/Users/kuanlin/stock_market/tests/test_daily.py` — Tests for Adjustment 1
- `/Users/kuanlin/stock_market/tests/test_stocks.py` — Tests for Adjustment 3

---

## Concerns / Notes

- **Adjustment 2 network path**: The misdating fix is structurally obvious (renamed discarded variables to used ones). Fully automated testing would require mocking `fetch_t86_for_stock` / `fetch_valuation_for_stock` at the store-upsert level, which is feasible but adds complexity. The existing `TestPickQueryDate` suite validates the date-resolution logic itself.
- **MIN_VALUATION_HISTORY = 60**: This threshold means the valuation facet score will be suppressed until 60 trading days (~3 months) of PE data have accumulated in the store. `combine_lean` already handles `score=None` by re-normalising weights over available facets, so the suppression propagates cleanly to the lean scores.
